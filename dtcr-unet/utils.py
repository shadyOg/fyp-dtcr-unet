"""Utility and Evaluation Metric functions for DTCR-U-Net.

Implements Dice, Sensitivity (SE), Specificity (SP), Accuracy, F1-Score,
and Hausdorff Distance (HD) exactly as defined in Section 4.3 (Equations 31-38)
of the DTCR-U-Net paper.
"""

from typing import Dict, Tuple
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def compute_binary_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute all evaluation metrics defined in Section 4.3 of the paper.

    Args:
        pred_mask: Probability map or binary mask (H, W) in [0, 1].
        true_mask: Ground-truth binary mask (H, W) in {0, 1}.
        threshold: Binarization threshold.

    Returns:
        Dictionary with Dice, Sensitivity, Specificity, Accuracy, F1, and Hausdorff Distance.
    """
    p = (pred_mask > threshold).astype(bool)
    t = (true_mask > 0.5).astype(bool)

    tp = np.logical_and(p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()

    eps = 1e-8

    # Eq. (31): Sensitivity (SE)
    se = float(tp / (tp + fn + eps))

    # Eq. (32): Specificity (SP)
    sp = float(tn / (tn + fp + eps))

    # Eq. (33): Dice coefficient
    dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))

    # Eq. (35): Accuracy
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))

    # Eq. (36-38): Precision, Recall, F1
    precision = float(tp / (tp + fp + eps))
    recall = se
    f1 = float(2.0 * precision * recall / (precision + recall + eps))

    # Eq. (34): Hausdorff Distance (HD) in pixels
    p_pts = np.argwhere(p)
    t_pts = np.argwhere(t)
    if len(p_pts) == 0 or len(t_pts) == 0:
        hd = 0.0 if (len(p_pts) == 0 and len(t_pts) == 0) else float(np.sqrt(pred_mask.shape[0]**2 + pred_mask.shape[1]**2))
    else:
        d_p_t = directed_hausdorff(p_pts, t_pts)[0]
        d_t_p = directed_hausdorff(t_pts, p_pts)[0]
        hd = float(max(d_p_t, d_t_p))

    return {
        "dice": dice * 100.0,
        "sensitivity": se * 100.0,
        "specificity": sp * 100.0,
        "accuracy": accuracy * 100.0,
        "f1": f1 * 100.0,
        "hd": hd,
    }


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: str = "checkpoints",
    filename: str = "last_model.pth",
):
    """Save training checkpoint to disk."""
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)