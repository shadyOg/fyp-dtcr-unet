"""Main Training Script for DTCR-U-Net with Mixed Precision (AMP).

Runs Dual-Task Semi-Supervised Training with AdamW, Cosine Annealing learning rate schedule,
Automatic Mixed Precision (AMP), dynamic consistency loss ramp-up, and checkpointing
matching Section 4 of the paper.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from config import DTCRConfig
from data.datasets import get_dataloaders
from losses.dtc_loss import DTCRTotalLoss
from models.dtcr_unet import DTCR_UNet
from utils import save_checkpoint

LOSS_KEYS = ("loss_total", "loss_seg", "loss_lsf", "loss_dtc", "l_main", "l_grad", "l_interact")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: "torch.amp.GradScaler",
    criterion: DTCRTotalLoss,
    device: torch.device,
    epoch: int,
    max_epochs: int,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
) -> dict:
    """Run one epoch of training with AMP and gradient accumulation."""
    model.train()
    running = {k: 0.0 for k in LOSS_KEYS}
    total_batches = len(loader)
    counted_batches = 0
    optimizer_steps = 0
    skipped_batches = 0
    lambda_d = 0.0
    amp_on = use_amp and device.type == "cuda"

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{max_epochs} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        level_sets = batch["level_set"].to(device, non_blocking=True)
        is_labeled = batch["is_labeled"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
            f1_logits, f2_lsf, f2_trans = model(images)
            loss, loss_dict = criterion(
                f1_logits=f1_logits,
                f2_lsf=f2_lsf,
                f2_trans=f2_trans,
                target_mask=masks,
                target_lsf=level_sets,
                is_labeled_mask=is_labeled,
                epoch=epoch + step / max(total_batches, 1),  # ramp lambda_d(t) every iteration
                max_epochs=max_epochs,
            )
            scaled_loss = loss / grad_accum_steps

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            skipped_batches += 1
            continue

        is_update_step = (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches
        if amp_on:
            scaler.scale(scaled_loss).backward()
            if is_update_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    optimizer_steps += 1
                else:
                    skipped_batches += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            scaled_loss.backward()
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        for k in LOSS_KEYS:
            running[k] += loss_dict[k]
        counted_batches += 1
        lambda_d = loss_dict["lambda_d"]

        pbar.set_postfix({
            "Loss": f"{loss_dict['loss_total']:.4f}",
            "Seg": f"{loss_dict['loss_seg']:.4f}",
            "LSF": f"{loss_dict['loss_lsf']:.4f}",
            "DTC": f"{loss_dict['loss_dtc']:.4f}",
        })

    # GradScaler recovery: if too many batches were skipped, reset the scaler
    # to prevent the death spiral where scale keeps shrinking until all batches underflow.
    if amp_on:
        current_scale = scaler.get_scale()
        skip_ratio = skipped_batches / max(total_batches, 1)
        if skip_ratio > 0.5 or current_scale < 1.0:
            print(f"  [GradScaler Recovery] scale={current_scale:.1f}, skipped={skipped_batches}/{total_batches} "
                  f"({skip_ratio:.0%}) - resetting scale to 2^16")
            try:
                scaler._scale = torch.tensor(65536.0, dtype=torch.float32, device=device)
            except Exception:
                scaler.update()

    avg = {k.replace("loss_", ""): v / max(counted_batches, 1) for k, v in running.items()}
    avg["lambda_d"] = lambda_d
    avg["_optimizer_steps"] = optimizer_steps
    avg["_skipped_batches"] = skipped_batches
    return avg


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    use_amp: bool = True,
    fuse_val: bool = False,
    threshold: float = 0.50,
) -> dict:
    """Evaluate on a validation / test loader.

    The headline `dice` is the PATIENT-LEVEL Dice: predictions of all slices of a
    patient are pooled into one volume before computing Dice, then averaged over
    patients. This is the quantity the paper reports (Table 2, mean +/- std over
    patients). `lesion_dice` is the older per-slice mean over lesion slices, which
    is systematically lower because small lesions on single slices dominate it.
    """
    model.eval()
    amp_on = use_amp and device.type == "cuda"

    patient_counts: Dict[str, np.ndarray] = {}   # pid -> [tp, fp, fn, tn]
    patient_dataset: Dict[str, str] = {}
    slice_dices = []
    loss_sums = {k: 0.0 for k in ("loss_total", "loss_seg", "loss_lsf", "loss_dtc")}
    n_samples = 0
    fg_prob_sum, fg_pixels = 0.0, 0.0
    bg_prob_sum, bg_pixels = 0.0, 0.0
    lsf_agree_inter, lsf_agree_card = 0.0, 0.0

    for batch in tqdm(loader, desc="[Evaluating]", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        level_sets = batch["level_set"].to(device, non_blocking=True)
        is_labeled = batch["is_labeled"].to(device, non_blocking=True)
        bsz = images.size(0)

        with torch.amp.autocast(device_type=device.type, enabled=amp_on):
            f1_logits, f2_lsf, f2_trans = model(images)
            if criterion is not None:
                _, loss_dict = criterion(
                    f1_logits=f1_logits,
                    f2_lsf=f2_lsf,
                    f2_trans=f2_trans,
                    target_mask=masks,
                    target_lsf=level_sets,
                    is_labeled_mask=is_labeled,
                    epoch=1,
                    max_epochs=1,  # full consistency weight: comparable across epochs
                )
                for k in loss_sums:
                    loss_sums[k] += loss_dict[k] * bsz
        n_samples += bsz

        f1_probs = torch.sigmoid(f1_logits.float())
        f2_trans = f2_trans.float()
        probs = 0.5 * f1_probs + 0.5 * f2_trans if fuse_val else f1_probs

        pred = probs > threshold
        target = masks > 0.5
        dims = (1, 2, 3)
        tp = (pred & target).sum(dims).double()
        fp = (pred & ~target).sum(dims).double()
        fn = (~pred & target).sum(dims).double()
        tn = (~pred & ~target).sum(dims).double()
        counts = torch.stack([tp, fp, fn, tn], dim=1).cpu().numpy()

        has_lesion = (tp + fn) > 0
        slice_dice = (2 * tp / (2 * tp + fp + fn).clamp(min=1.0))[has_lesion]
        slice_dices.extend(slice_dice.cpu().tolist())

        fg_prob_sum += f1_probs[target].sum().item()
        fg_pixels += target.sum().item()
        bg_prob_sum += f1_probs[~target].sum().item()
        bg_pixels += (~target).sum().item()

        # Agreement between the two heads (binarized), a direct read-out of task consistency
        lsf_pred = f2_trans > 0.5
        lsf_agree_inter += (pred & lsf_pred).sum().item()
        lsf_agree_card += pred.sum().item() + lsf_pred.sum().item()

        for i in range(bsz):
            pid = batch["patient_id"][i]
            if pid not in patient_counts:
                patient_counts[pid] = np.zeros(4, dtype=np.float64)
                patient_dataset[pid] = batch["dataset"][i]
            patient_counts[pid] += counts[i]

    def _patient_metrics(c: np.ndarray) -> Dict[str, float]:
        tp, fp, fn, tn = c
        return {
            "dice": 100.0 * 2 * tp / max(2 * tp + fp + fn, 1.0),
            "sensitivity": 100.0 * tp / max(tp + fn, 1.0),
            "specificity": 100.0 * tn / max(tn + fp, 1.0),
            "accuracy": 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1.0),
        }

    per_patient = {pid: _patient_metrics(c) for pid, c in patient_counts.items()}
    metrics: Dict[str, float] = {}
    for k in ("dice", "sensitivity", "specificity", "accuracy"):
        vals = [m[k] for m in per_patient.values()]
        metrics[k] = float(np.mean(vals)) if vals else 0.0
    metrics["dice_std"] = float(np.std([m["dice"] for m in per_patient.values()])) if per_patient else 0.0
    metrics["num_patients"] = len(per_patient)

    # Per source dataset (e.g. mosmed vs lidc): shows which one limits the hybrid score
    for ds in sorted(set(patient_dataset.values())):
        vals = [per_patient[pid]["dice"] for pid, d in patient_dataset.items() if d == ds]
        metrics[f"dice_{ds}"] = float(np.mean(vals))
        metrics[f"patients_{ds}"] = len(vals)

    # Dice over all pixels pooled together (a single global confusion matrix)
    total = np.sum(list(patient_counts.values()), axis=0) if patient_counts else np.zeros(4)
    metrics["global_dice"] = _patient_metrics(total)["dice"]

    metrics["lesion_dice"] = 100.0 * float(np.mean(slice_dices)) if slice_dices else 0.0
    metrics["lesion_count"] = len(slice_dices)
    metrics["val_loss"] = loss_sums["loss_total"] / max(n_samples, 1)
    metrics["val_seg"] = loss_sums["loss_seg"] / max(n_samples, 1)
    metrics["val_lsf"] = loss_sums["loss_lsf"] / max(n_samples, 1)
    metrics["val_dtc"] = loss_sums["loss_dtc"] / max(n_samples, 1)
    metrics["mean_fg_prob"] = fg_prob_sum / max(fg_pixels, 1.0)
    metrics["mean_bg_prob"] = bg_prob_sum / max(bg_pixels, 1.0)
    metrics["task_agreement"] = 100.0 * 2 * lsf_agree_inter / max(lsf_agree_card, 1.0)
    return metrics


def train(
    cfg: DTCRConfig,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    filter_empty_val: bool = False,
):
    """Main training execution function."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device} | AMP Mixed Precision: {use_amp}")

    # 1. Prepare DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        labeled_ratio=cfg.labeled_ratio,
        seed=cfg.seed,
        filter_empty_val=filter_empty_val,
    )
    print(f"Data ready: {len(train_loader.dataset)} train slices, {len(val_loader.dataset)} val slices, "
          f"{len(train_loader)} iterations/epoch (batch {cfg.batch_size} x accum {grad_accum_steps}).")
    print(f"Semi-supervised split (labeled ratio {cfg.labeled_ratio}):")
    for ds, s in sorted(train_loader.dataset.summary().items()):
        print(f"  {ds:<10} labeled: {s['labeled_patients']} patients / {s['labeled_slices']} slices | "
              f"unlabeled: {s['unlabeled_patients']} patients / {s['unlabeled_slices']} slices")

    # 2. Build Model & Loss
    model = DTCR_UNet(
        in_channels=cfg.in_channels,
        n_classes=cfg.n_classes,
        bilinear=cfg.bilinear,
        base_c=cfg.base_channels,
        num_stages=cfg.num_stages,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters (paper Table 1: 18.65M)")

    criterion = DTCRTotalLoss(
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        beta=cfg.beta,
        bce_weight=cfg.bce_weight,
        dice_weight=cfg.dice_weight,
        pos_weight=cfg.pos_weight,
        lambda_context=cfg.lambda_context,
        use_focal=cfg.use_focal,
        consistency_weight=cfg.consistency_weight,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_epochs = min(5, max(cfg.epochs - 1, 0))
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup_epochs, 1), eta_min=cfg.min_lr)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = cosine_scheduler

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_loss = float("inf")
    best_val_dice = -1.0
    dice_at_best_loss = 0.0
    history = []

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    history_path = os.path.join(cfg.output_dir, "training_history.json")

    print(f"\nStarting DTCR-U-Net training for {cfg.epochs} epochs...\n")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

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
            fuse_val=cfg.fuse_val,
            threshold=cfg.val_threshold,
        )
        scheduler.step()
        elapsed = time.time() - t0

        val_dice = val_metrics["dice"]
        val_loss = val_metrics["val_loss"]

        # Paper Section 4.1: "the model checkpoint with the lowest validation loss is selected".
        is_best_loss = val_loss < best_val_loss
        if is_best_loss:
            best_val_loss = val_loss
            dice_at_best_loss = val_dice
        # Also keep the checkpoint with the highest patient-level Dice for comparison.
        is_best_dice = val_dice > best_val_dice
        if is_best_dice:
            best_val_dice = val_dice

        history.append({
            "epoch": epoch + 1,
            "train_losses": train_losses,
            "val_metrics": val_metrics,
            "lr": lr_now,
            "time_sec": elapsed,
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        per_ds = " ".join(
            f"{k[5:]}={v:.1f}" for k, v in sorted(val_metrics.items())
            if k.startswith("dice_") and k != "dice_std"
        )
        flags = ("  [best loss]" if is_best_loss else "") + ("  [best dice]" if is_best_dice else "")
        skipped = f" | skipped {train_losses['_skipped_batches']}" if train_losses["_skipped_batches"] else ""
        print(
            f"Epoch {epoch + 1:02d}/{cfg.epochs} ({elapsed:.0f}s) lr {lr_now:.1e} lam_d {train_losses['lambda_d']:.3f} | "
            f"train {train_losses['total']:.4f} (seg {train_losses['seg']:.3f} lsf {train_losses['lsf']:.4f} "
            f"dtc {train_losses['dtc']:.4f}: main {train_losses['l_main']:.4f} grad {train_losses['l_grad']:.4f} "
            f"int {train_losses['l_interact']:.4f}) | "
            f"val loss {val_loss:.4f} (seg {val_metrics['val_seg']:.3f} lsf {val_metrics['val_lsf']:.4f} "
            f"dtc {val_metrics['val_dtc']:.4f}) | "
            f"PATIENT DICE {val_dice:.2f}+-{val_metrics['dice_std']:.1f} [{per_ds}] | "
            f"global {val_metrics['global_dice']:.2f} slice {val_metrics['lesion_dice']:.2f} | "
            f"SE {val_metrics['sensitivity']:.1f} SP {val_metrics['specificity']:.2f} "
            f"Acc {val_metrics['accuracy']:.2f} | agree {val_metrics['task_agreement']:.1f} "
            f"p_fg {val_metrics['mean_fg_prob']:.2f} p_bg {val_metrics['mean_bg_prob']:.3f}"
            f"{skipped}{flags}"
        )

        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "best_val_loss": best_val_loss,
            "val_metrics": val_metrics,
            "config": vars(cfg),
        }
        save_checkpoint(state=state, is_best=is_best_loss, checkpoint_dir=cfg.checkpoint_dir, filename="last_model.pth")
        if is_best_dice:
            torch.save(state, os.path.join(cfg.checkpoint_dir, "best_dice_model.pth"))

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(
        f"\nTraining complete. Lowest val loss {best_val_loss:.4f} (patient Dice {dice_at_best_loss:.2f}%, "
        f"saved as best_model.pth) | highest patient Dice {best_val_dice:.2f}% (best_dice_model.pth)"
    )


def main():
    parser = argparse.ArgumentParser(description="Train DTCR-U-Net")
    parser.add_argument("--data", default="data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--epochs", type=int, default=80, help="Total training epochs (default: 80 per paper Section 4.1)")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-step batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4 -> effective batch 32 per paper Section 4.1)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001 per paper Section 4.1)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--labeled-ratio", type=float, default=0.4, help="Fraction of training patients that keep labels (default: 0.4; use 1.0 for fully-supervised)")
    parser.add_argument("--beta", type=float, default=0.5, help="Gradient constraint weight for supervised LSF loss (default: 0.5 per Section 4.6)")
    parser.add_argument("--lambda1", type=float, default=0.3, help="Gradient consistency weight for L_DTC (default: 0.3 per paper Fig. 7 optimum)")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Interaction enhancement weight for L_DTC (default: 1.0 per paper Section 4.6)")
    parser.add_argument("--lambda-context", type=float, default=0.0, help="Context-guided loss weight (default: 0.0 matching Eq. 4; optional per Eq. 3)")
    parser.add_argument("--consistency-weight", type=float, default=1.0, help="Maximum of the consistency ramp lambda_d(t) (default: 1.0 per Eq. 10)")
    parser.add_argument("--bce-weight", type=float, default=0.0, help="BCE loss weight (default: 0.0 for pure Soft Dice per Eq. 8; >0 for compound loss)")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight (default: 1.0 matching Eq. 8 & 10)")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class weight if BCE is enabled")
    parser.add_argument("--use-focal", action="store_true", default=False, help="Use Focal Loss instead of BCE if bce-weight > 0")
    parser.add_argument("--val-threshold", type=float, default=0.50, help="Validation binarization threshold (default: 0.50)")
    parser.add_argument("--num-stages", type=int, default=4, help="Encoder stages: 4 (18.65M params matching Table 1) or 5 (legacy)")
    parser.add_argument("--fuse-val", action="store_true", default=False, help="Average both heads during validation (default: segmentation head only)")
    parser.add_argument("--no-fuse-val", action="store_true", help="Explicitly disable dual-task prediction fusion during validation")
    parser.add_argument("--filter-val", action="store_true", help="Validate on lesion slices only (drops the lesion-free context slices)")
    parser.add_argument("--no-filter-val", action="store_true", help="Deprecated no-op: all validation slices are used by default")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP Mixed Precision")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (also fixes the labeled/unlabeled patient split)")
    parser.add_argument("--checkpoints", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--outputs", default="outputs", help="Directory to save logs and results")
    args = parser.parse_args()

    cfg = DTCRConfig(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        labeled_ratio=args.labeled_ratio,
        beta=args.beta,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda_context=args.lambda_context,
        consistency_weight=args.consistency_weight,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
        use_focal=args.use_focal,
        num_stages=args.num_stages,
        fuse_val=args.fuse_val and not args.no_fuse_val,
        val_threshold=args.val_threshold,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoints,
        output_dir=args.outputs,
    )

    train(
        cfg,
        grad_accum_steps=args.grad_accum,
        use_amp=not args.no_amp,
        filter_empty_val=args.filter_val,
    )


if __name__ == "__main__":
    main()
