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
):
    """Load model checkpoint and run comprehensive test set evaluation."""
    dev = torch.device(device)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=dev)

    # Initialize model
    model = DTCR_UNet(in_channels=1, n_classes=1).to(dev)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size, transform=False)
    print(f"Test samples: {len(test_loader.dataset)}")

    metrics_list = []
    saved_vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch["image"].to(dev)
            masks = batch["mask"].cpu().numpy()

            f1_logits, f2_lsf, f2_trans = model(images)
            probs = torch.sigmoid(f1_logits).cpu().numpy()
            trans_probs = f2_trans.cpu().numpy()
            images_np = images.cpu().numpy()

            for i in range(len(probs)):
                m = compute_binary_metrics(probs[i, 0], masks[i, 0])
                metrics_list.append(m)

                # Save qualitative visual comparisons
                if saved_vis_count < save_visualizations:
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
                    plt.savefig(os.path.join(output_dir, f"test_sample_{saved_vis_count + 1}.png"))
                    plt.close()
                    saved_vis_count += 1

    # Aggregate metric summary
    summary = {}
    for k in metrics_list[0].keys():
        values = [m[k] for m in metrics_list]
        summary[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
        }

    print("\n" + "=" * 50)
    print("           TEST SET EVALUATION RESULTS           ")
    print("=" * 50)
    for k, v in summary.items():
        print(f"{k.upper():<15}: {v['mean']:.2f} ± {v['std']:.2f}")
    print("=" * 50)

    # Save to JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate DTCR-U-Net")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data", default="data/processed", help="Path to preprocessed dataset")
    parser.add_argument("--out", default="outputs/evaluation", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data,
        output_dir=args.out,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()