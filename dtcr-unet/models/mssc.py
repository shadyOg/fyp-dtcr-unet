"""Multi-Scale Skip Connection (MsSC) Scheme for DTCR-U-Net.

Implements Cascade, Residual, and Dense skip connectivity across hierarchical
encoder-decoder stages as described in Section 3.3 (Equations 27-30) of the paper,
optimized with 1x1 channel projection before upsampling to preserve GPU memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiScaleSkipConnection(nn.Module):
    """MsSC Block: bridges semantic gap by fusing hierarchical encoder features with decoder."""

    def __init__(
        self,
        in_channels_list: List[int],  # e.g., [F_low, F_mid, F_high]
        out_channels: int,
    ):
        """
        Args:
            in_channels_list: Channel counts of the multi-scale features being routed.
            out_channels: Output channel dimension after skip fusion.
        """
        super().__init__()
        self.out_channels = out_channels

        # 1x1 projections to reduce channels before upsampling (saves huge VRAM)
        self.channel_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_channels // len(in_channels_list), kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // len(in_channels_list)),
                nn.ReLU(inplace=True),
            ) if in_c > out_channels // len(in_channels_list) else nn.Identity()
            for in_c in in_channels_list
        ])

        # Calculate effective concatenated channels after projections
        proj_channels = [
            out_channels // len(in_channels_list) if in_c > out_channels // len(in_channels_list) else in_c
            for in_c in in_channels_list
        ]
        total_proj_channels = sum(proj_channels)

        # Dense connection fusion conv F(Concat([...])) (Eq. 29 & 30)
        self.dense_conv = nn.Sequential(
            nn.Conv2d(total_proj_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # High-level residual transform H(F_high) (Eq. 28 & 30)
        self.high_transform = nn.Sequential(
            nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Low-level shortcut alignment
        self.low_proj = nn.Sequential(
            nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels_list[0] != out_channels else nn.Identity()

        # Multi-scale refinement convs with adaptive weights alpha_i (Eq. 30)
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

        self.alpha = nn.Parameter(torch.ones(2) / 2.0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors [F_low, ..., F_high] from encoder/decoder.
                      Target spatial size is taken from features[0] (F_low).

        Returns:
            F_out: Skip-connected feature map of shape (B, out_channels, H_low, W_low).
        """
        target_size = (features[0].shape[2], features[0].shape[3])

        # 1. Project channels first, then resample to match F_low spatial resolution
        aligned_features = []
        for proj, feat in zip(self.channel_projs, features):
            projected = proj(feat)
            if (projected.shape[2], projected.shape[3]) != target_size:
                aligned_features.append(F.interpolate(projected, size=target_size, mode="bilinear", align_corners=True))
            else:
                aligned_features.append(projected)

        f_low = features[0]
        f_high = features[-1]
        if (f_high.shape[2], f_high.shape[3]) != target_size:
            f_high = F.interpolate(f_high, size=target_size, mode="bilinear", align_corners=True)

        # 2. Dense connection (Eq. 29)
        f_concat = torch.cat(aligned_features, dim=1)  # Concat([F_low, F_mid, F_high])
        f_dense = self.dense_conv(f_concat)

        # 3. Residual connections (Eq. 28)
        f_res = self.low_proj(f_low) + self.high_transform(f_high)

        # 4. Multi-scale aggregation with adaptive weighting (Eq. 30)
        c3 = self.conv3x3(f_dense)
        c5 = self.conv5x5(f_dense)
        alpha_norm = F.softmax(self.alpha, dim=0)

        f_out = alpha_norm[0] * c3 + alpha_norm[1] * c5 + f_res

        return f_out