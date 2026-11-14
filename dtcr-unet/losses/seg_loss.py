"""Supervised Segmentation Loss for DTCR-U-Net.

Implements Dice Loss and Hybrid BCE + Dice Loss as described in Section 3.1.2
(Equation 8) of the DTCR-U-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation (Eq. 8 of the paper)."""

    MODES = ("batch", "slice", "hybrid")

    def __init__(self, smooth: float = 1.0, mode: str = "hybrid"):
        """
        Args:
            smooth: Additive constant (epsilon in Eq. 8).
            mode: How Dice is aggregated over a batch.
                "batch": pool all pixels of the batch, then one ratio. Robust on
                    lesion-free slices, but weights each lesion by its pixel count,
                    so a 20 px nodule next to a 3000 px consolidation barely counts.
                "slice": one Dice per slice, averaged. Every lesion counts equally,
                    but on a lesion-free slice a few false-positive pixels cost ~1.0.
                "hybrid" (default): mean of "batch" over all slices and "slice" over
                    the slices that contain lesion. Small lesions get their own term,
                    while false positives on empty slices are still priced by pixel mass.
        """
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.smooth = smooth
        self.mode = mode

    def _dice(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inter = (p * t).sum(dim=1)
        card = p.sum(dim=1) + t.sum(dim=1)
        return (2.0 * inter + self.smooth) / (card + self.smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities of shape (B, 1, H, W).
            target: Ground truth binary mask of shape (B, 1, H, W).
        """
        p = torch.clamp(pred.float(), 0.0, 1.0).reshape(pred.size(0), -1)
        t = target.float().reshape(target.size(0), -1)

        batch_loss = 1.0 - self._dice(p.reshape(1, -1), t.reshape(1, -1)).mean()
        if self.mode == "batch":
            return batch_loss

        if self.mode == "slice":
            return (1.0 - self._dice(p, t)).mean()

        has_lesion = t.sum(dim=1) > 0
        if not has_lesion.any():
            return batch_loss
        slice_loss = (1.0 - self._dice(p[has_lesion], t[has_lesion])).mean()
        return 0.5 * (batch_loss + slice_loss)


class FocalLoss(nn.Module):
    """Focal Loss for binary segmentation handling severe class imbalance."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_factor * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


class SupervisedSegLoss(nn.Module):
    """Supervised Segmentation Loss matching Eq. 8 of the DTCR-U-Net paper.

    Primary objective is Soft Dice Loss per Eq. 8 with weight 1.0.
    Optionally allows combining with BCE/Focal loss if requested.
    """

    def __init__(
        self,
        bce_weight: float = 0.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
        pos_weight: float = 1.0,
        use_focal: bool = False,
        dice_mode: str = "hybrid",
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_focal = use_focal
        self.register_buffer("pw", torch.tensor([pos_weight]))
        self.dice_loss = DiceLoss(smooth=smooth, mode=dice_mode)
        if use_focal:
            self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits from segmentation head f1(x) of shape (B, 1, H, W).
            target: Ground truth binary mask of shape (B, 1, H, W).
        """
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, target)

        if self.bce_weight > 0:
            if self.use_focal:
                bce = self.focal_loss(logits, target)
            else:
                if self.pw.item() != 1.0:
                    pw = self.pw.to(device=logits.device, dtype=logits.dtype)
                    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
                else:
                    bce = F.binary_cross_entropy_with_logits(logits, target)
            return self.bce_weight * bce + self.dice_weight * dice

        return self.dice_weight * dice