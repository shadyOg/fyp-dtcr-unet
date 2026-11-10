"""Dual-Task Consistency Regularization (DTCR) Loss.

Implements Main Consistency (L_main), Gradient Consistency (L_grad), and
Interaction Enhancement (L_interact) losses for semi-supervised co-learning
as described in Section 3.1.2 (Equations 4-7, 10) of the DTCR-U-Net paper.
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_spatial_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradients (grad_x, grad_y) using Sobel filters matching tensor dtype."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0

    pad_x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(pad_x, sobel_x)
    grad_y = F.conv2d(pad_x, sobel_y)

    return grad_x, grad_y


def get_dynamic_consistency_weight(current_epoch: int, max_epochs: int, max_weight: float = 1.0) -> float:
    """Calculate the Gaussian ramp-up consistency weight lambda_d(t).

    Per Eq. (10) of the paper:
        lambda_d(t) = max_weight * exp(-5 * (1 - t / t_max)^2)

    max_weight=1.0 matches the paper: consistency becomes equal to supervised
    loss strength at the end of training, fully exploiting unlabeled samples.
    """
    if max_epochs <= 0:
        return max_weight
    t = min(current_epoch, max_epochs)
    ramp = max_weight * math.exp(-5.0 * ((1.0 - t / max_epochs) ** 2))
    return float(ramp)


class DualTaskConsistencyLoss(nn.Module):
    """Dual-Task Consistency Regularization Loss L_DTC.

    Implements:
    - Main Consistency Loss L_main: ||f1(x) - T^{-1}(f2(x))||^2 (Eq. 5)
    - Gradient Consistency Loss L_grad: ||nabla f1(x) - nabla T^{-1}(f2(x))||^2 (Eq. 6)
    - Interaction Enhancement Loss L_interact: ||f1(x) * f2(x)|| (Eq. 7)
    - Context-guided Loss L_context: ||nabla f2(x) - nabla(f1(x) * W(x))||^2 (Eq. 3, Fig. 1)
    """

    def __init__(
        self,
        lambda1: float = 0.3,
        lambda2: float = 1.0,
        lambda_context: float = 0.0,
    ):
        """
        Args:
            lambda1: Weight for gradient consistency term L_grad (optimal: 0.3 per paper Fig. 7).
            lambda2: Weight for interaction enhancement term L_interact (default: 1.0 per Section 4.6).
            lambda_context: Weight for context-guided loss L_context (default: 0.0 matching Eq. 4 & Section 4.6).
        """
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_context = lambda_context

    def forward(
        self,
        f1_prob: torch.Tensor,
        f2_lsf: torch.Tensor,
        f2_trans: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            f1_prob: Segmentation probability map f1(x) = sigmoid(logits) of shape (B, 1, H, W).
            f2_lsf: Raw predicted Level Set Function map f2(x) of shape (B, 1, H, W).
            f2_trans: Inversely transformed map T^{-1}(f2(x)) of shape (B, 1, H, W).

        Returns:
            Tuple of (L_DTC scalar loss, dictionary of individual consistency losses).
        """
        # 1. Main Consistency Loss L_main: ||f1(x) - T^{-1}(f2(x))||^2 (Eq. 5)
        l_main = F.mse_loss(f1_prob, f2_trans)

        # 2. Gradient Consistency Loss L_grad: ||nabla f1(x) - nabla T^{-1}(f2(x))||^2 (Eq. 6)
        f1_gx, f1_gy = compute_spatial_gradients(f1_prob)
        t_gx, t_gy = compute_spatial_gradients(f2_trans)
        l_grad = F.mse_loss(f1_gx, t_gx) + F.mse_loss(f1_gy, t_gy)

        # 3. Interaction Enhancement Loss L_interact: ||f1(x) * f2(x)|| (Eq. 7)
        # Symmetrically penalizes task discrepancy:
        # - False positives where Task 1 predicts lesion but Level Set indicates background (f2 > 0)
        # - False negatives where Task 1 misses lesion but Level Set indicates lesion (f2 < 0)
        # Correctly aligned predictions receive zero penalty.
        f2_clamped = f2_lsf.clamp(-1.0, 1.0)
        interaction_fp = f1_prob * torch.clamp(f2_clamped, min=0.0)
        interaction_fn = (1.0 - f1_prob) * torch.clamp(-f2_clamped, min=0.0)
        l_interact = (interaction_fp + interaction_fn).mean()

        # 4. Context-guided Loss L_context per Eq. 3 & Fig. 1 (optional, 0.0 default matching Eq. 4):
        # Aligns target-weighted segmentation gradients with transformed level-set gradient field
        if self.lambda_context > 0:
            w_x = torch.sigmoid(f1_prob * 5.0)
            target_feat = f1_prob * w_x
            tf_gx, tf_gy = compute_spatial_gradients(target_feat)
            l_context = F.mse_loss(t_gx, tf_gx) + F.mse_loss(t_gy, tf_gy)
        else:
            l_context = torch.tensor(0.0, device=f1_prob.device)

        # Total L_DTC per Eq. 4 (+ optional context guidance)
        l_dtc = l_main + self.lambda1 * l_grad + self.lambda2 * l_interact + self.lambda_context * l_context

        loss_breakdown = {
            "l_main": l_main.item(),
            "l_grad": l_grad.item(),
            "l_interact": l_interact.item(),
            "l_context": l_context.item(),
        }
        return l_dtc, loss_breakdown


class DTCRTotalLoss(nn.Module):
    """Unified Total Loss Function combining Supervised & Dual-Task Consistency losses."""

    def __init__(
        self,
        lambda1: float = 0.3,
        lambda2: float = 1.0,
        beta: float = 0.5,
        bce_weight: float = 0.0,
        dice_weight: float = 1.0,
        pos_weight: float = 1.0,
        lambda_context: float = 0.0,
        use_focal: bool = False,
    ):
        super().__init__()
        from .seg_loss import SupervisedSegLoss
        from .lsf_loss import SupervisedLSFLoss

        self.seg_criterion = SupervisedSegLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            use_focal=use_focal,
        )
        self.lsf_criterion = SupervisedLSFLoss(beta=beta)
        self.dtc_criterion = DualTaskConsistencyLoss(
            lambda1=lambda1,
            lambda2=lambda2,
            lambda_context=lambda_context,
        )

    def forward(
        self,
        f1_logits: torch.Tensor,
        f2_lsf: torch.Tensor,
        f2_trans: torch.Tensor,
        target_mask: torch.Tensor,
        target_lsf: torch.Tensor,
        is_labeled_mask: torch.Tensor,
        epoch: int,
        max_epochs: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculates L_total = L_seg + L_LSF + lambda_d(t) * L_DTC (Eq. 10).
        Supervised terms apply only to labeled samples (is_labeled == True).
        Consistency term L_DTC applies to ALL samples (labeled + unlabeled).
        """
        f1_prob = torch.sigmoid(f1_logits)

        # 1. Supervised Losses on Labeled Subset (Eq. 8 & 9)
        if torch.any(is_labeled_mask):
            labeled_f1_logits = f1_logits[is_labeled_mask]
            labeled_target_mask = target_mask[is_labeled_mask]
            labeled_f2_lsf = f2_lsf[is_labeled_mask]
            labeled_target_lsf = target_lsf[is_labeled_mask]

            l_seg = self.seg_criterion(labeled_f1_logits, labeled_target_mask)
            l_lsf = self.lsf_criterion(labeled_f2_lsf, labeled_target_lsf)
        else:
            l_seg = torch.tensor(0.0, device=f1_logits.device)
            l_lsf = torch.tensor(0.0, device=f1_logits.device)

        # 2. Dual-Task Consistency Loss on ALL samples (Eq. 4-7)
        l_dtc, dtc_dict = self.dtc_criterion(f1_prob, f2_lsf, f2_trans)

        # 3. Dynamic Weight lambda_d(t) per Eq. 10
        lambda_d = get_dynamic_consistency_weight(epoch, max_epochs)

        # 4. Total Loss L_total = L_seg + L_LSF + lambda_d * L_DTC (Eq. 10)
        total_loss = l_seg + l_lsf + lambda_d * l_dtc

        metrics = {
            "loss_total": total_loss.item(),
            "loss_seg": l_seg.item(),
            "loss_lsf": l_lsf.item(),
            "loss_dtc": l_dtc.item(),
            "lambda_d": lambda_d,
            **dtc_dict,
        }

        return total_loss, metrics