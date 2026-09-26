"""Supervised Level Set Function (LSF) Loss for Task 2 Boundary Regression.

Implements Mean Squared Error + Gradient Difference constraint against
the ground-truth Level Set Map T(y) as described in Section 3.1.2 (Equation 9)
of the DTCR-U-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def compute_spatial_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradients (grad_x, grad_y) using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3) / 8.0

    pad_x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(pad_x, sobel_x)
    grad_y = F.conv2d(pad_x, sobel_y)

    return grad_x, grad_y


class SupervisedLSFLoss(nn.Module):
    """Supervised Boundary Loss L_LSF for Task 2."""

    def __init__(self, beta: float = 0.5):
        """
        Args:
            beta: Hyperparameter weighting the gradient constraint term (default: 0.5 per Section 4.6).
        """
        super().__init__()
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_lsf: torch.Tensor, target_lsf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_lsf: Predicted Level Set Function map f2(x) of shape (B, 1, H, W).
            target_lsf: Ground truth Level Set map T(y) of shape (B, 1, H, W).
        """
        # 1. Main MSE loss ||f2(x) - T(y)||^2
        loss_mse = self.mse_loss(pred_lsf, target_lsf)

        # 2. Gradient constraint ||nabla f2(x) - nabla T(y)||^2
        pred_gx, pred_gy = compute_spatial_gradients(pred_lsf)
        targ_gx, targ_gy = compute_spatial_gradients(target_lsf)

        loss_grad = F.mse_loss(pred_gx, targ_gx) + F.mse_loss(pred_gy, targ_gy)

        return loss_mse + self.beta * loss_grad