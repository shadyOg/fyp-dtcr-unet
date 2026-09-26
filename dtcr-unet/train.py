"""Main Training Script for DTCR-U-Net.

Runs Dual-Task Semi-Supervised Training with AdamW, Cosine Annealing learning rate schedule,
dynamic consistency loss ramp-up, and automated checkpointing matching Section 4 of the paper.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    criterion: DTCRTotalLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
) -> dict:
    """Run one epoch of training."""
    model.train()
    running_losses = {"total": 0.0, "seg": 0.0, "lsf": 0.0, "dtc": 0.0}
    total_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]")
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        level_sets = batch["level_set"].to(device, non_blocking=True)
        is_labeled = batch["is_labeled"].to(device, non_blocking=True)

        optimizer.zero_grad()

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

        loss.backward()
        optimizer.step()

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

    avg_losses = {k: v / max(total_batches, 1) for k, v in running_losses.items()}
    return avg_losses


def evaluate_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on validation or test set."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Evaluating]", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].cpu().numpy()

            f1_logits, _, _ = model(images)
            probs = torch.sigmoid(f1_logits).cpu().numpy()

            for i in range(len(probs)):
                m = compute_binary_metrics(probs[i, 0], masks[i, 0])
                all_metrics.append(m)

    mean_metrics = {}
    for k in all_metrics[0].keys():
        mean_metrics[k] = float(np.mean([m[k] for m in all_metrics]))

    return mean_metrics


def train(cfg: DTCRConfig):
    """Main training execution function."""
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 1. Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
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
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )

    best_val_dice = 0.0
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
            criterion=criterion,
            device=device,
            epoch=epoch,
            max_epochs=cfg.epochs,
        )

        val_metrics = evaluate_dataset(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        val_dice = val_metrics["dice"]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice

        epoch_log = {
            "epoch": epoch + 1,
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Dice: {val_dice:.2f}% | "
            f"Val F1: {val_metrics['f1']:.2f}% | "
            f"Val HD: {val_metrics['hd']:.2f}px | "
            f"{'(Best!)' if is_best else ''}"
        )

        # Save Checkpoint
        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "config": vars(cfg),
            },
            is_best=is_best,
            checkpoint_dir=cfg.checkpoint_dir,
            filename="last_model.pth",
        )

    # Save final training history
    with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining Complete! Best Validation Dice: {best_val_dice:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train DTCR-U-Net")
    parser.add_argument("--data", default="data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--labeled-ratio", type=float, default=0.4, help="Semi-supervised labeled ratio")
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

    train(cfg)


if __name__ == "__main__":
    main()