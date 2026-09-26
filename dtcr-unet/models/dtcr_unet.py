"""DTCR-U-Net: Dual-Task Consistency Regularized U-Net with Multi-Scale Adaptive Attention.

Assembles the full dual-task architecture for pneumonia lesion segmentation:
- Shared Encoder & Decoder backbone with Multi-Scale Skip Connections (MsSC).
- Multi-Scale Adaptive Attention (MSAA) mechanism combining CCE, MSGS, and AEF.
- Dual Output Heads: Task 1 (Pixel-level Segmentation) and Task 2 (Boundary Level Set Regression).
- Differentiable Inverse Mapping T^{-1}(z) for task-level consistency regularization.

Paper Reference:
"DTCR-U-Net: a dual-task consistency and multi-scale adaptive attention framework
for pneumonia lesion segmentation" (MBEC, 2026).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aef import AttentionEmbeddingFusion
from .cce import ChannelContextualEnhancement
from .decoder import DecoderUpBlock
from .encoder import ConvBlock, EncoderBlock
from .msgs import MultiScaleGlobalSpatial
from .mssc import MultiScaleSkipConnection


def compute_gradient_norm(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradient norms ||nabla z|| and ||nabla z||^2 using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=z.dtype, device=z.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=z.dtype, device=z.device).view(1, 1, 3, 3) / 8.0

    pad_z = F.pad(z, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(pad_z, sobel_x)
    grad_y = F.conv2d(pad_z, sobel_y)

    grad_norm_sq = grad_x**2 + grad_y**2
    grad_norm = torch.sqrt(grad_norm_sq + 1e-6)

    return grad_norm, grad_norm_sq


def inverse_level_set_transform(
    z: torch.Tensor,
    k: float = 1.0,
    lambda1: float = 0.01,
    lambda2: float = 0.01,
) -> torch.Tensor:
    """Differentiable inverse mapping T^{-1}(z) converting Level Set output back to segmentation space.

    Per Eq. (2) in the DTCR-U-Net paper:
        T^{-1}(z) = sigma(k * z) + lambda1 * ||nabla z||^2 + lambda2 * ||nabla z||

    Args:
        z: Predicted Level Set map from Task 2 (shape: B, 1, H, W).
        k: Sigmoid scaling constant.
        lambda1: Gradient smoothing coefficient.
        lambda2: Laplacian / first-order gradient coefficient.

    Returns:
        Transformed segmentation probability map (shape: B, 1, H, W) bounded in [0, 1].
    """
    z_clamped = torch.clamp(z, -5.0, 5.0)
    sig_z = torch.sigmoid(-k * z_clamped)

    grad_norm, grad_norm_sq = compute_gradient_norm(z_clamped)
    transformed = sig_z + lambda1 * grad_norm_sq + lambda2 * grad_norm

    return torch.clamp(transformed, 0.0, 1.0)


class DTCR_UNet(nn.Module):
    """Complete DTCR-U-Net Dual-Task Architecture."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        base_c: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # ---------------- 1. Encoder Stages ----------------
        self.inc = ConvBlock(in_channels, base_c)                   # E1: (B, 64, H, W)
        self.down1 = EncoderBlock(base_c, base_c * 2)               # E2: (B, 128, H/2, W/2)
        self.down2 = EncoderBlock(base_c * 2, base_c * 4)           # E3: (B, 256, H/4, W/4)
        self.down3 = EncoderBlock(base_c * 4, base_c * 8)           # E4: (B, 512, H/8, W/8)
        self.down4 = EncoderBlock(base_c * 8, base_c * 8)           # Bottleneck: (B, 512, H/16, W/16)

        # ---------------- 2. MSAA Mechanism ----------------
        # CCE: Channel Contextual Enhancement
        self.cce = ChannelContextualEnhancement(
            in_channels_list=[base_c, base_c * 2, base_c * 4, base_c * 8],
            proj_dim=128,
            token_size=(16, 16),
        )
        # MSGS: Multi-Scale Global Spatial
        self.msgs = MultiScaleGlobalSpatial(in_channels=512, out_channels=128)

        # AEF: Attention Embedding Fusion
        self.aef = AttentionEmbeddingFusion(cce_channels=512, msgs_channels=128, out_channels=512)

        # Bottleneck fusion conv
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(base_c * 8 + 512, base_c * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c * 8),
            nn.ReLU(inplace=True),
        )

        # ---------------- 3. MsSC Skip Connections & Decoder ----------------
        # Multi-scale skip aggregation across encoder levels
        self.mssc4 = MultiScaleSkipConnection(
            in_channels_list=[base_c * 4, base_c * 8],
            out_channels=base_c * 8,
        )
        self.mssc3 = MultiScaleSkipConnection(
            in_channels_list=[base_c * 2, base_c * 4, base_c * 8],
            out_channels=base_c * 4,
        )
        self.mssc2 = MultiScaleSkipConnection(
            in_channels_list=[base_c, base_c * 2, base_c * 4],
            out_channels=base_c * 2,
        )

        # Decoder Upsampling Stages
        self.up1 = DecoderUpBlock(in_channels=base_c * 8, skip_channels=base_c * 8, out_channels=base_c * 8, bilinear=bilinear)
        self.up2 = DecoderUpBlock(in_channels=base_c * 8, skip_channels=base_c * 4, out_channels=base_c * 4, bilinear=bilinear)
        self.up3 = DecoderUpBlock(in_channels=base_c * 4, skip_channels=base_c * 2, out_channels=base_c * 2, bilinear=bilinear)
        self.up4 = DecoderUpBlock(in_channels=base_c * 2, skip_channels=base_c, out_channels=base_c, bilinear=bilinear)

        # ---------------- 4. Dual Output Heads ----------------
        # Task 1: Pixel-level Segmentation Head (f1)
        self.seg_head = nn.Conv2d(base_c, n_classes, kernel_size=1)

        # Task 2: Boundary / Level Set Function Regression Head (f2)
        self.lsf_head = nn.Sequential(
            nn.Conv2d(base_c, base_c // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_c // 2, 1, kernel_size=1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through DTCR-U-Net.

        Args:
            x: Input CT slice tensor of shape (B, 1, H, W).

        Returns:
            f1_logits: Segmentation logits for Task 1 (shape: B, 1, H, W).
            f2_lsf: Predicted Level Set Function map for Task 2 (shape: B, 1, H, W).
            f2_trans: Inversely transformed segmentation probability map T^{-1}(f2) (shape: B, 1, H, W).
        """
        # 1. Encoder extraction
        e1 = self.inc(x)         # (B, 64, H, W)
        e2 = self.down1(e1)      # (B, 128, H/2, W/2)
        e3 = self.down2(e2)      # (B, 256, H/4, W/4)
        e4 = self.down3(e3)      # (B, 512, H/8, W/8)
        bottle = self.down4(e4)  # (B, 512, H/16, W/16)

        # 2. MSAA: Multi-Scale Adaptive Attention at bottleneck
        f_cce = self.cce([e1, e2, e3, e4])                    # (B, 512, 16, 16)
        f_msgs = self.msgs(f_cce)                             # (B, 128, 16, 16)
        f_aef = self.aef(f_cce, f_msgs)                       # (B, 512, 16, 16)

        # Align spatial dimensions with bottleneck feature if needed
        if f_aef.shape[2:] != bottle.shape[2:]:
            f_aef = F.interpolate(f_aef, size=bottle.shape[2:], mode="bilinear", align_corners=True)

        bottle_fused = self.bottleneck_fuse(torch.cat([bottle, f_aef], dim=1))

        # 3. Decoder with MsSC Skip Connections
        skip4 = self.mssc4([e3, e4])
        d1 = self.up1(bottle_fused, skip4)  # (B, 512, H/8, W/8)

        skip3 = self.mssc3([e2, e3, e4])
        d2 = self.up2(d1, skip3)            # (B, 256, H/4, W/4)

        skip2 = self.mssc2([e1, e2, e3])
        d3 = self.up3(d2, skip2)            # (B, 128, H/2, W/2)

        d4 = self.up4(d3, e1)               # (B, 64, H, W)

        # 4. Dual Task Predictions
        f1_logits = torch.clamp(self.seg_head(d4), min=-20.0, max=20.0)       # Task 1: Segmentation logits (B, 1, H, W)
        f2_lsf = torch.clamp(self.lsf_head(d4), min=-5.0, max=5.0)            # Task 2: Level Set Function (B, 1, H, W)

        # 5. Differentiable Inverse Mapping for Consistency Regularization
        f2_trans = inverse_level_set_transform(f2_lsf)

        return f1_logits, f2_lsf, f2_trans