"""DTCR-U-Net: Dual-Task Consistency Regularized U-Net with Multi-Scale Adaptive Attention.

Assembles the full dual-task architecture for pneumonia lesion segmentation:
- 4-Stage Encoder backbone with Multi-Scale Skip Connections (MsSC) matching Section 3.1 & 3.2.1.
- Multi-Scale Adaptive Attention (MSAA) mechanism combining CCE, MSGS, and AEF (Section 3.2).
- Dual Output Heads: Task 1 (Pixel-level Segmentation) and Task 2 (Boundary Level Set Regression).
- Differentiable Inverse Mapping T^{-1}(z) for task-level consistency regularization (Eq. 2).
- Native 4-stage architecture matching the 18.65M parameter budget from Table 1, with optional
  5-stage compatibility mode for legacy checkpoints.

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


def compute_tensor_gradients(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradients (grad_x, grad_y) using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8.0

    pad_x = F.pad(x, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(pad_x, sobel_x)
    grad_y = F.conv2d(pad_x, sobel_y)
    return grad_x, grad_y


def inverse_level_set_transform(
    z: torch.Tensor,
    k: float = 50.0,
    include_smoothing_terms: bool = False,
    lambda1: float = 0.01,
    lambda2: float = 0.01,
) -> torch.Tensor:
    """Differentiable inverse mapping T^{-1}(z) converting Level Set output back to segmentation space.

    Per Eq. (2) in the DTCR-U-Net paper:
        T^{-1}(z) = sigma(-k * z) + R(z) + S(z)
        where sigma(-k * z) maps z < 0 (inside lesion) -> 1.0, z > 0 (background) -> 0.0.
        R(z) = ||nabla z||^2 is gradient smoothing regularization.
        S(z) = ||nabla z|| is geometric consistency regularization.

    For normalized distance maps z in [-1, 1], k=50.0 yields a proper ~3-5px transition band
    around lesion boundaries, ensuring high probability (0.80-0.98) inside lesions.
    For primary probability space alignment, sigma(-k * z) is clamped to [0, 1].
    """
    z_clamped = torch.clamp(z, min=-10.0, max=10.0)
    prob_map = torch.sigmoid(-k * z_clamped)

    if include_smoothing_terms:
        gx, gy = compute_tensor_gradients(z_clamped)
        grad_norm_sq = gx**2 + gy**2
        grad_norm = torch.sqrt(grad_norm_sq + 1e-8)
        transformed = prob_map + lambda1 * grad_norm_sq + lambda2 * grad_norm
        return torch.clamp(transformed, 0.0, 1.0)

    return prob_map


class DTCR_UNet(nn.Module):
    """Complete DTCR-U-Net Dual-Task Architecture.

    By default implements the 4-stage encoder / 3-stage decoder architecture
    (18.65M parameters per Table 1 of the paper).
    Supports num_stages=5 for legacy checkpoint compatibility.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 1,
        bilinear: bool = False,
        base_c: int = 64,
        num_stages: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_c = base_c
        self.num_stages = num_stages

        # ---------------- 1. Encoder Stages ----------------
        # 4 stages: E1 (H), E2 (H/2), E3 (H/4), E4 (H/8) per Eq. 11-12
        self.inc = ConvBlock(in_channels, base_c)                    # E1: (B, 64, H, W)
        self.down1 = EncoderBlock(base_c, base_c * 2)                # E2: (B, 128, H/2, W/2)
        self.down2 = EncoderBlock(base_c * 2, base_c * 4)            # E3: (B, 256, H/4, W/4)
        self.down3 = EncoderBlock(base_c * 4, base_c * 8)            # E4: (B, 512, H/8, W/8)

        # ---------------- 2. MSAA Mechanism ----------------
        # CCE: Channel Contextual Enhancement on [E1, E2, E3, E4]
        self.cce = ChannelContextualEnhancement(
            in_channels_list=[base_c, base_c * 2, base_c * 4, base_c * 8],
            proj_dim=128,
            token_size=(16, 16),
        )
        # MSGS: Multi-Scale Global Spatial
        self.msgs = MultiScaleGlobalSpatial(in_channels=512, out_channels=128)

        # AEF: Attention Embedding Fusion (projects to C = 128 per Eq. 22-26; 512 for legacy)
        aef_out = 512 if num_stages == 5 else 128
        self.aef = AttentionEmbeddingFusion(cce_channels=512, msgs_channels=128, out_channels=aef_out)

        if num_stages == 4:
            # Paper standard 4-stage architecture (18.65M parameters)
            # Bottleneck fusion conv at E4: combines E4 (512) and AEF (128) -> 512
            self.bottleneck_fuse = nn.Sequential(
                nn.Conv2d(base_c * 8 + 128, base_c * 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_c * 8),
                nn.ReLU(inplace=True),
            )

            # MsSC Skip Connections per Eq. 27-30 (dense + multi-scale conv + residual)
            # D1 at H/4 <- target E3 (256), context E2 (128), E4 (512)
            self.mssc3 = MultiScaleSkipConnection(
                in_channels_list=[base_c * 4, base_c * 2, base_c * 8],
                out_channels=base_c * 4,
            )
            # D2 at H/2 <- target E2 (128), context E1 (64), E3 (256)
            self.mssc2 = MultiScaleSkipConnection(
                in_channels_list=[base_c * 2, base_c, base_c * 4],
                out_channels=base_c * 2,
            )
            # D3 at H <- target E1 (64), context E2 (128), E3 (256)
            self.mssc1 = MultiScaleSkipConnection(
                in_channels_list=[base_c, base_c * 2, base_c * 4],
                out_channels=base_c,
            )

            # 3 Decoder Upsampling Stages: H/8 -> H/4 -> H/2 -> H
            self.up1 = DecoderUpBlock(in_channels=base_c * 8, skip_channels=base_c * 4, out_channels=base_c * 4, bilinear=bilinear)
            self.up2 = DecoderUpBlock(in_channels=base_c * 4, skip_channels=base_c * 2, out_channels=base_c * 2, bilinear=bilinear)
            self.up3 = DecoderUpBlock(in_channels=base_c * 2, skip_channels=base_c, out_channels=base_c, bilinear=bilinear)

        else:
            # Legacy 5-stage architecture mode (for loading checkpoints with down4)
            self.down4 = EncoderBlock(base_c * 8, base_c * 8)
            self.bottleneck_fuse = nn.Sequential(
                nn.Conv2d(base_c * 8 + 512, base_c * 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_c * 8),
                nn.ReLU(inplace=True),
            )
            self.mssc4 = MultiScaleSkipConnection(
                in_channels_list=[base_c * 8, base_c * 4],
                out_channels=base_c * 8,
            )
            self.mssc3 = MultiScaleSkipConnection(
                in_channels_list=[base_c * 4, base_c * 2, base_c * 8],
                out_channels=base_c * 4,
            )
            self.mssc2 = MultiScaleSkipConnection(
                in_channels_list=[base_c * 2, base_c, base_c * 4],
                out_channels=base_c * 2,
            )
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
        # 1. Encoder extraction (4 stages: E1..E4)
        e1 = self.inc(x)         # (B, 64, H, W)
        e2 = self.down1(e1)      # (B, 128, H/2, W/2)
        e3 = self.down2(e2)      # (B, 256, H/4, W/4)
        e4 = self.down3(e3)      # (B, 512, H/8, W/8)

        # 2. MSAA: Multi-Scale Adaptive Attention
        f_cce = self.cce([e1, e2, e3, e4])                    # (B, 512, 16, 16)
        f_msgs = self.msgs(f_cce)                             # (B, 128, 16, 16)
        f_aef = self.aef(f_cce, f_msgs)                       # (B, 128, 16, 16)

        if self.num_stages == 4:
            # 3a. Bottleneck fusion at E4 resolution (H/8, W/8)
            if f_aef.shape[2:] != e4.shape[2:]:
                f_aef = F.interpolate(f_aef, size=e4.shape[2:], mode="bilinear", align_corners=True)
            bottle_fused = self.bottleneck_fuse(torch.cat([e4, f_aef], dim=1))  # (B, 512, H/8, W/8)

            # 3b. Decoder with 3 MsSC Skip Connections
            skip3 = self.mssc3([e3, e2, e4])        # target H/4 from e3
            d1 = self.up1(bottle_fused, skip3)      # (B, 256, H/4, W/4)

            skip2 = self.mssc2([e2, e1, e3])        # target H/2 from e2
            d2 = self.up2(d1, skip2)                # (B, 128, H/2, W/2)

            skip1 = self.mssc1([e1, e2, e3])        # target H from e1
            d_out = self.up3(d2, skip1)             # (B, 64, H, W)

        else:
            # Legacy 5-stage forward pass
            bottle = self.down4(e4)
            if f_aef.shape[2:] != bottle.shape[2:]:
                f_aef = F.interpolate(f_aef, size=bottle.shape[2:], mode="bilinear", align_corners=True)
            bottle_fused = self.bottleneck_fuse(torch.cat([bottle, f_aef], dim=1))

            skip4 = self.mssc4([e4, e3])
            d1 = self.up1(bottle_fused, skip4)
            skip3 = self.mssc3([e3, e2, e4])
            d2 = self.up2(d1, skip3)
            skip2 = self.mssc2([e2, e1, e3])
            d3 = self.up3(d2, skip2)
            d_out = self.up4(d3, e1)

        # 4. Dual Task Predictions
        f1_logits = torch.clamp(self.seg_head(d_out), min=-20.0, max=20.0)
        f2_lsf = torch.clamp(self.lsf_head(d_out), min=-5.0, max=5.0)

        # 5. Differentiable Inverse Mapping T^{-1}(z) per Eq. 2
        f2_trans = inverse_level_set_transform(f2_lsf)

        return f1_logits, f2_lsf, f2_trans
