"""Multi-Scale Global Spatial (MSGS) Module for DTCR-U-Net.

Implements Spatial Scaled Dot-Product Self-Attention and Multi-Scale Convolutional
Filter Fusion (3x3, 5x5, 7x7) as described in Section 3.2.3 (Equations 17-21)
of the DTCR-U-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGlobalSpatial(nn.Module):
    """MSGS Module: captures cross-scale spatial dependencies and multi-scale spatial features."""

    def __init__(
        self,
        in_channels: int = 512,  # C_sigma from CCE
        out_channels: int = 128, # Target projection dimension C
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1x1 convolutions for Qs, Ks, Vs generation (Eq. 17)
        self.q_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Residual shortcut projection
        self.res_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Multi-scale convolutional filter kernels: 3x3, 5x5, 7x7 (Eq. 21)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Learnable multi-scale fusion weights alpha_s (Eq. 21)
        self.scale_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, f_cce: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_cce: Channel-enhanced feature map from CCE of shape (B, C_sigma, H, W).

        Returns:
            F_multi_MSGS: Multi-scale spatial-enhanced feature map of shape (B, out_channels, H, W).
        """
        # Force entire spatial attention in FP32 to prevent GradScaler
        # overflow/underflow death spiral in both forward AND backward passes.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            f_cce = f_cce.float()
            b, c, h, w = f_cce.shape
            n = h * w

            # 1. 1x1 conv transformations (Eq. 17)
            q_s = self.q_conv(f_cce).view(b, self.out_channels, n)  # (B, C, N)
            k_s = self.k_conv(f_cce).view(b, self.out_channels, n)  # (B, C, N)
            v_s = self.v_conv(f_cce).view(b, self.out_channels, n)  # (B, C, N)

            # 2. Scaled Dot-Product Spatial Similarity matrix M_s (Eq. 19)
            scale = float(self.out_channels) ** 0.5
            sim_s = torch.bmm(q_s.transpose(1, 2), k_s) / scale
            sim_s_max = torch.max(sim_s, dim=-1, keepdim=True)[0]
            m_s = F.softmax(sim_s - sim_s_max, dim=-1)  # (B, N, N)

            # 3. Spatial attention weighting and residual connection (Eq. 20)
            attended_s = torch.bmm(v_s, m_s.transpose(1, 2)).view(b, self.out_channels, h, w)
            residual = self.res_proj(f_cce)
            f_msgs = attended_s + residual  # (B, C, H, W)

            # 4. Multi-scale convolutional filter fusion (3x3, 5x5, 7x7) (Eq. 21)
            feat_3 = self.conv3x3(f_msgs)
            feat_5 = self.conv5x5(f_msgs)
            feat_7 = self.conv7x7(f_msgs)

            norm_weights = F.softmax(self.scale_weights, dim=0)
            f_multi_msgs = (
                norm_weights[0] * feat_3 +
                norm_weights[1] * feat_5 +
                norm_weights[2] * feat_7
            )

        return f_multi_msgs