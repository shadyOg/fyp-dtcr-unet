"""Supervised Segmentation Loss for DTCR-U-Net.

Implements Dice Loss and Hybrid BCE + Dice Loss as described in Section 3.1.2
(Equation 8) of the DTCR-U-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities or sigmoid outputs of shape (B, 1, H, W).
            target: Ground truth binary mask of shape (B, 1, H, W).
        """
        pred_safe = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        pred_flat = pred_safe.contiguous().view(pred_safe.size(0), -1)
        target_flat = target.contiguous().view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class SupervisedSegLoss(nn.Module):
    """Hybrid BCE + Dice Loss for Task 1 (Pixel-level Segmentation)."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits from segmentation head f1(x) of shape (B, 1, H, W).
            target: Ground truth binary mask of shape (B, 1, H, W).
        """
        bce = self.bce_loss(logits, target)
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, target)

        return self.bce_weight * bce + self.dice_weight * dice