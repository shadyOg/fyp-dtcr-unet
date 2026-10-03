"""Main Training Script for DTCR-U-Net with Mixed Precision (AMP).

Runs Dual-Task Semi-Supervised Training with AdamW, Cosine Annealing learning rate schedule,
Automatic Mixed Precision (AMP) for peak memory efficiency, dynamic consistency loss ramp-up,
and automated checkpointing matching Section 4 of the paper.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import DTCRConfig
from data.datasets import get_dataloaders
from losses.dtc_loss import DTCRTotalLoss
from models.dtcr_unet import DTCR_UNet
from utils import compute_binary_metrics, save_checkpoint


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    criterion: DTCRTotalLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> dict:
    """Run one epoch of training with AMP and gradient accumulation."""
    model.train()
    running_losses = {"total": 0.0, "seg": 0.0, "lsf": 0.0, "dtc": 0.0}
    total_batches = len(loader)
    actual_trained_batches = 0  # Count batches that actually contributed to training
    skipped_batches = 0  # Count batches skipped by GradScaler or NaN

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        level_sets = batch["level_set"].to(device, non_blocking=True)
        is_labeled = batch["is_labeled"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            # Forward pass
            f1_logits, f2_lsf, f2_trans = model(images)

            # Compute dual-task loss
            loss, loss_dict = criterion(
                f1_logits=f1_logits,
                f2_lsf=f2_lsf,
                f2_trans=f2_trans,
                target_mask=masks,
                target_lsf=level_sets,
                is_labeled_mask=is_labeled,
                epoch=epoch,
                max_epochs=max_epochs,
            )
            scaled_loss = loss / grad_accum_steps

        # Check if loss is finite before backward
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        if use_amp and device.type == "cuda":
            scaler.scale(scaled_loss).backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check if unscaled gradients are finite
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    actual_trained_batches += 1
                else:
                    skipped_batches += 1

                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                actual_trained_batches += 1

        running_losses["total"] += loss_dict["loss_total"]
        running_losses["seg"] += loss_dict["loss_seg"]
        running_losses["lsf"] += loss_dict["loss_lsf"]
        running_losses["dtc"] += loss_dict["loss_dtc"]

        pbar.set_postfix({
            "Loss": f"{loss_dict['loss_total']:.4f}",
            "Seg": f"{loss_dict['loss_seg']:.4f}",
            "LSF": f"{loss_dict['loss_lsf']:.4f}",
            "DTC": f"{loss_dict['loss_dtc']:.4f}",
        })

    # GradScaler recovery: if too many batches were skipped, reset the scaler
    # to prevent the death spiral where scale keeps shrinking until all batches underflow.
    if use_amp and device.type == "cuda":
        current_scale = scaler.get_scale()
        skip_ratio = skipped_batches / max(total_batches, 1)
        if skip_ratio > 0.5 or current_scale < 1.0:
            print(f"  [GradScaler Recovery] scale={current_scale:.1f}, skipped={skipped_batches}/{total_batches} "
                  f"({skip_ratio:.0%}) — resetting scale to 2^16")
            scaler.update(new_scale=2**16)

    # Average over actual contributing batches (not total) to avoid artificially low loss
    divisor = max(total_batches, 1)  # Use total for consistent loss magnitude reporting
    avg_losses = {k: v / divisor for k, v in running_losses.items()}
    avg_losses["_actual_batches"] = actual_trained_batches
    avg_losses["_skipped_batches"] = skipped_batches
    return avg_losses


def evaluate_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
) -> dict:
    """Evaluate model on validation or test set, computing overall, lesion-only, and validation loss metrics."""
    model.eval()
    all_metrics = []
    lesion_metrics = []
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            level_sets = batch["level_set"].to(device, non_blocking=True)
            is_labeled = batch["is_labeled"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
                f1_logits, f2_lsf, f2_trans = model(images)
                if criterion is not None:
                    batch_loss, _ = criterion(
                        f1_logits=f1_logits,
                        f2_lsf=f2_lsf,
                        f2_trans=f2_trans,
                        target_mask=masks,
                        target_lsf=level_sets,
                        is_labeled_mask=is_labeled,
                        epoch=80,
                        max_epochs=80,
                    )
                    val_losses.append(batch_loss.item())

                probs = torch.sigmoid(f1_logits).float().cpu().numpy()

            masks_np = masks.cpu().numpy()
            for i in range(len(probs)):
                m = compute_binary_metrics(probs[i, 0], masks_np[i, 0])
                all_metrics.append(m)
                if masks_np[i, 0].sum() > 0:
                    lesion_metrics.append(m)

    mean_metrics = {}
    if all_metrics:
        for k in all_metrics[0].keys():
            mean_metrics[k] = float(np.mean([m[k] for m in all_metrics]))

    if lesion_metrics:
        mean_metrics["lesion_dice"] = float(np.mean([m["dice"] for m in lesion_metrics]))
        mean_metrics["lesion_count"] = len(lesion_metrics)
    else:
        mean_metrics["lesion_dice"] = mean_metrics.get("dice", 0.0)
        mean_metrics["lesion_count"] = 0

    if val_losses:
        mean_metrics["val_loss"] = float(np.mean(val_losses))
    else:
        mean_metrics["val_loss"] = 0.0

    return mean_metrics


def train(
    cfg: DTCRConfig,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    filter_empty_val: bool = True,
):
    """Main training execution function."""
    device = torch.device(cfg.device)
    print(f"Using device: {device} | AMP Mixed Precision: {use_amp} | Filter Empty Val Slices: {filter_empty_val}")

    # 1. Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        filter_empty_val=filter_empty_val,
    )
    print(f"Data ready: {len(train_loader.dataset)} train samples, {len(val_loader.dataset)} val samples.")

    # 2. Build Model & Loss
    model = DTCR_UNet(
        in_channels=cfg.in_channels,
        n_classes=cfg.n_classes,
        bilinear=cfg.bilinear,
        base_c=cfg.base_channels,
    ).to(device)

    criterion = DTCRTotalLoss(
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        beta=cfg.beta,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup_epochs = 5
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs - warmup_epochs, 1),
        eta_min=cfg.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_dice = 0.0
    best_lesion_dice = 0.0
    best_val_loss = float("inf")
    history = []

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"\nStarting DTCR-U-Net training for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epochs=cfg.epochs,
            grad_accum_steps=grad_accum_steps,
            use_amp=use_amp,
        )

        val_metrics = evaluate_dataset(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            use_amp=use_amp,
        )
        scheduler.step()

        elapsed = time.time() - t0
        val_dice = val_metrics["dice"]
        lesion_dice = val_metrics.get("lesion_dice", val_dice)
        val_loss = val_metrics.get("val_loss", 0.0)

        # Select best model based on lesion_dice so empty background slices don't distort checkpointing
        is_best = lesion_dice > best_lesion_dice
        if is_best:
            best_lesion_dice = lesion_dice
            best_val_dice = val_dice
            best_val_loss = val_loss

        epoch_log = {
            "epoch": epoch + 1,
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        history.append(epoch_log)

        skipped_info = ""
        if train_losses.get("_skipped_batches", 0) > 0:
            skipped_info = f" | Skipped: {train_losses['_skipped_batches']} batches"

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} ({elapsed:.1f}s) | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.2f}% (Lesion: {lesion_dice:.2f}%) | "
            f"Val SE: {val_metrics['sensitivity']:.2f}% | "
            f"Val SP: {val_metrics['specificity']:.2f}% | "
            f"Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"Val F1: {val_metrics['f1']:.2f}% | "
            f"Val HD: {val_metrics['hd']:.2f}px"
            f"{' | (Best Lesion!)' if is_best else ''}"
            f"{skipped_info}"
        )

        # Save Checkpoint
        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "best_lesion_dice": best_lesion_dice,
                "best_val_loss": best_val_loss,
                "config": vars(cfg),
            },
            is_best=is_best,
            checkpoint_dir=cfg.checkpoint_dir,
            filename="last_model.pth",
        )

        # Clean GPU memory cache
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save final training history
    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete! Best Lesion Dice: {best_lesion_dice:.2f}% (Overall Val Dice: {best_val_dice:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Train DTCR-U-Net")
    parser.add_argument("--data", default="data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs (default: 80 per paper)")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-step batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4 -> effective batch 32 per paper)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 0.001 per paper)")
    parser.add_argument("--labeled-ratio", type=float, default=0.4, help="Semi-supervised labeled ratio (default: 0.4)")
    parser.add_argument("--no-filter-val", action="store_true", help="Include empty background slices in validation")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP Mixed Precision")
    parser.add_argument("--checkpoints", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--outputs", default="outputs", help="Directory to save logs and results")
    args = parser.parse_args()

    cfg = DTCRConfig(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        labeled_ratio=args.labeled_ratio,
        checkpoint_dir=args.checkpoints,
        output_dir=args.outputs,
    )

    train(
        cfg,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        filter_empty_val=not args.no_filter_val,
    )


if __name__ == "__main__":
    main()