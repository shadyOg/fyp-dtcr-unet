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
    compute_hd: bool = False,
) -> Dict[str, float]:
    """Compute all evaluation metrics defined in Section 4.3 of the paper.

    Args:
        pred_mask: Probability map or binary mask (H, W) in [0, 1].
        true_mask: Ground-truth binary mask (H, W) in {0, 1}.
        threshold: Binarization threshold.
        compute_hd: Whether to compute CPU-intensive Hausdorff distance (skip during fast per-epoch validation).

    Returns:
        Dictionary with Dice, Sensitivity, Specificity, Accuracy, F1, and Hausdorff Distance.
    """
    pred_mask = np.nan_to_num(pred_mask, nan=0.0, posinf=1.0, neginf=0.0)
    p = (pred_mask > threshold).astype(bool)
    t = (true_mask > 0.5).astype(bool)

    tp = np.logical_and(p, t).sum()
    tn = np.logical_and(~p, ~t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()

    eps = 1e-8
    has_target = np.any(t)
    has_pred = np.any(p)

    if not has_target and not has_pred:
        # Both empty: perfect prediction
        dice = 1.0
        se = 1.0
        sp = 1.0
        accuracy = 1.0
        f1 = 1.0
        hd = 0.0
    elif not has_target and has_pred:
        # False alarm on empty slice
        dice = 0.0
        se = 1.0
        sp = float(tn / (tn + fp + eps))
        accuracy = float(tn / (tn + fp + eps))
        f1 = 0.0
        hd = float(np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)) if compute_hd else 0.0
    elif has_target and not has_pred:
        # Missed lesion completely
        dice = 0.0
        se = 0.0
        sp = 1.0
        accuracy = float(tn / (tn + fn + eps))
        f1 = 0.0
        hd = float(np.sqrt(pred_mask.shape[0] ** 2 + pred_mask.shape[1] ** 2)) if compute_hd else 0.0
    else:
        # Both contain foreground pixels
        se = float(tp / (tp + fn + eps))
        sp = float(tn / (tn + fp + eps))
        dice = float((2.0 * tp) / (2.0 * tp + fp + fn + eps))
        accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))
        precision = float(tp / (tp + fp + eps))
        recall = se
        f1 = float(2.0 * precision * recall / (precision + recall + eps))

        if compute_hd:
            p_pts = np.argwhere(p)
            t_pts = np.argwhere(t)
            d_p_t = directed_hausdorff(p_pts, t_pts)[0]
            d_t_p = directed_hausdorff(t_pts, p_pts)[0]
            hd = float(max(d_p_t, d_t_p))
        else:
            hd = 0.0

    return {
        "dice": dice * 100.0,
        "sensitivity": se * 100.0,
        "specificity": sp * 100.0,
        "accuracy": accuracy * 100.0,
        "f1": f1 * 100.0,
        "hd": hd,
    }


def predict_probs(model: torch.nn.Module, images: torch.Tensor, tta: bool = False, fuse: bool = False):
    """Run the model and return (raw outputs, lesion probability map).

    Args:
        tta: Average with the prediction on the horizontally flipped image.
        fuse: Average the segmentation head with the inverse-transformed level set head.

    Returns:
        ((f1_logits, f2_lsf, f2_trans) of the plain forward pass, probabilities (B, 1, H, W) float32).
    """
    outputs = model(images)
    f1_logits, _, f2_trans = outputs
    seg = torch.sigmoid(f1_logits.float())
    lsf = f2_trans.float()
    if tta:
        fl_logits, _, fl_trans = model(torch.flip(images, dims=[3]))
        seg = 0.5 * (seg + torch.flip(torch.sigmoid(fl_logits.float()), dims=[3]))
        lsf = 0.5 * (lsf + torch.flip(fl_trans.float(), dims=[3]))
    probs = 0.5 * seg + 0.5 * lsf if fuse else seg
    return outputs, probs


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, num_stages=None):
    """Rebuild the model a checkpoint was trained with and load its weights.

    Returns:
        (model in eval mode, checkpoint dict).
    """
    from models.dtcr_resnet import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}
    arch = cfg.get("arch", "dtcr")
    if num_stages is None:
        num_stages = cfg.get("num_stages", 5 if any("down4" in k for k in state_dict) else 4)
    model = build_model(
        arch,
        in_channels=cfg.get("in_channels", 1),
        n_classes=cfg.get("n_classes", 1),
        pretrained=False,  # weights come from the checkpoint
        base_c=cfg.get("base_channels", 64),
        num_stages=num_stages,
        bilinear=cfg.get("bilinear", False),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


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