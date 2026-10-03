"""Evaluation and Testing Script for DTCR-U-Net.

Loads trained checkpoints and computes all metrics (Dice, Sensitivity, Specificity, Accuracy, F1, HD)
on the test set, saving predictions and visualization overlays.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import DTCRConfig
from data.datasets import get_dataloaders
from models.dtcr_unet import DTCR_UNet
from utils import compute_binary_metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str = "outputs/evaluation",
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_visualizations: int = 10,
    threshold: float = 0.5,
    fuse_dual_task: bool = False,
):
    """Load model checkpoint and run comprehensive test set evaluation."""
    dev = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)
    comparisons_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Evaluation settings: Threshold={threshold} | Dual-Task Fusion={fuse_dual_task}")
    checkpoint = torch.load(checkpoint_path, map_location=dev)

    # Initialize model
    model = DTCR_UNet(in_channels=1, n_classes=1).to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")

    metrics_list = []
    lesion_slice_metrics = []
    patient_slices = {}  # pid -> list of (prob_2d, mask_2d)
    saved_vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch["image"].to(dev)
            masks = batch["mask"].cpu().numpy()
            patient_ids = batch.get("patient_id", [f"unknown_{batch_idx}_{i}" for i in range(len(images))])

            f1_logits, f2_lsf, f2_trans = model(images)
            f1_probs = torch.sigmoid(f1_logits).cpu().numpy()
            trans_probs = f2_trans.cpu().numpy()
            images_np = images.cpu().numpy()

            if fuse_dual_task:
                probs = 0.5 * f1_probs + 0.5 * trans_probs
            else:
                probs = f1_probs

            for i in range(len(probs)):
                m = compute_binary_metrics(probs[i, 0], masks[i, 0], threshold=threshold)
                metrics_list.append(m)

                if masks[i, 0].sum() > 0:
                    lesion_slice_metrics.append(m)

                pid = patient_ids[i] if isinstance(patient_ids, (list, tuple)) else str(patient_ids)
                patient_slices.setdefault(pid, []).append((probs[i, 0], masks[i, 0]))

                # Save qualitative visual comparisons
                if saved_vis_count < save_visualizations and masks[i, 0].sum() > 0:
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    axes[0].imshow(images_np[i, 0], cmap="gray")
                    axes[0].set_title("CT Slice")
                    axes[0].axis("off")

                    axes[1].imshow(images_np[i, 0], cmap="gray")
                    axes[1].imshow(masks[i, 0], cmap="Reds", alpha=0.4)
                    axes[1].set_title("Ground Truth Mask")
                    axes[1].axis("off")

                    axes[2].imshow(images_np[i, 0], cmap="gray")
                    axes[2].imshow(probs[i, 0] > 0.5, cmap="Greens", alpha=0.4)
                    axes[2].set_title(f"Task 1 Pred (Dice: {m['dice']:.1f}%)")
                    axes[2].axis("off")

                    axes[3].imshow(trans_probs[i, 0], cmap="jet")
                    axes[3].set_title("Task 2 LSF -> Prob T^-1(z)")
                    axes[3].axis("off")

                    plt.tight_layout()
                    plt.savefig(os.path.join(comparisons_dir, f"test_sample_{saved_vis_count + 1}.png"), dpi=120, bbox_inches="tight")
                    plt.close()
                    saved_vis_count += 1

    # 1. Slice-Level Metrics (All slices)
    slice_summary = {}
    for k in metrics_list[0].keys():
        values = [m[k] for m in metrics_list]
        slice_summary[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    # 2. Lesion-Only Slices Metrics
    lesion_summary = {}
    if lesion_slice_metrics:
        for k in lesion_slice_metrics[0].keys():
            values = [m[k] for m in lesion_slice_metrics]
            lesion_summary[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }

    # 3. Patient-Level 3D Volume Metrics (Section 4.4.1 Table 2 of the paper)
    patient_metrics = []
    eps = 1e-8
    for pid, pairs in patient_slices.items():
        p_vol = np.stack([p[0] for p in pairs], axis=0) > threshold
        t_vol = np.stack([p[1] for p in pairs], axis=0) > 0.5

        tp = float(np.logical_and(p_vol, t_vol).sum())
        tn = float(np.logical_and(~p_vol, ~t_vol).sum())
        fp = float(np.logical_and(p_vol, ~t_vol).sum())
        fn = float(np.logical_and(~p_vol, t_vol).sum())

        p_dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps) * 100.0
        p_se = tp / (tp + fn + eps) * 100.0
        p_sp = tn / (tn + fp + eps) * 100.0
        p_acc = (tp + tn) / (tp + tn + fp + fn + eps) * 100.0
        p_f1 = p_dice

        patient_metrics.append({
            "patient_id": pid,
            "dice": p_dice,
            "sensitivity": p_se,
            "specificity": p_sp,
            "accuracy": p_acc,
            "f1": p_f1,
            "num_slices": len(pairs),
        })

    vol_summary = {}
    for k in ["dice", "sensitivity", "specificity", "accuracy", "f1"]:
        vals = [pm[k] for pm in patient_metrics]
        vol_summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }

    full_results = {
        "slice_level_all": slice_summary,
        "slice_level_lesion_only": lesion_summary,
        "patient_level_3d": vol_summary,
        "patient_breakdown": patient_metrics,
        "total_test_slices": len(metrics_list),
        "total_lesion_slices": len(lesion_slice_metrics),
        "total_patients": len(patient_metrics),
    }

    print("\n" + "=" * 60)
    print("      PATIENT-LEVEL 3D EVALUATION (Paper Protocol Table 2)")
    print("=" * 60)
    for k, v in vol_summary.items():
        print(f"  {k.upper():<15}: {v['mean']:.2f}% ± {v['std']:.2f}%")
    print("=" * 60)

    if lesion_summary:
        print("\n" + "=" * 60)
        print("      LESION-ONLY SLICE EVALUATION (Excl. Empty Slices)")
        print("=" * 60)
        for k, v in lesion_summary.items():
            print(f"  {k.upper():<15}: {v['mean']:.2f} ± {v['std']:.2f}")
        print("=" * 60)

    print("\n" + "=" * 60)
    print("      RAW 2D SLICE EVALUATION (All Slices)")
    print("=" * 60)
    for k, v in slice_summary.items():
        print(f"  {k.upper():<15}: {v['mean']:.2f} ± {v['std']:.2f}")
    print("=" * 60)

    # Save to JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DTCR-U-Net")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data", default="data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--out", default="outputs/evaluation", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    parser.add_argument("--fuse-dual-task", action="store_true", help="Ensemble Task 1 (Dice head) and Task 2 (Level Set inverse mapping)")
    parser.add_argument("--save-vis", type=int, default=20, help="Number of qualitative comparisons to save (0 to disable)")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        output_dir=args.out,
        batch_size=args.batch_size,
        save_visualizations=args.save_vis,
        threshold=args.threshold,
        fuse_dual_task=args.fuse_dual_task,
    )


if __name__ == "__main__":
    main()