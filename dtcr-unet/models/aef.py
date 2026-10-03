"""Attention Embedding Fusion (AEF) Module for DTCR-U-Net.

Implements complementary optimization between channel and spatial attention
representations as described in Section 3.2.4 (Equations 22-26) of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionEmbeddingFusion(nn.Module):
    """AEF Module: fuses CCE (channel) and MSGS (spatial) representations via dynamic attention."""

    def __init__(self, cce_channels: int = 512, msgs_channels: int = 128, out_channels: int = 512):
        super().__init__()
        self.out_channels = out_channels

        # 1x1 Convolutions to project both representations to common dimension (Eq. 22-23)
        self.cce_align = nn.Sequential(
            nn.Conv2d(cce_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.msgs_align = nn.Sequential(
            nn.Conv2d(msgs_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Spatial-level attention convolution for As (Eq. 25)
        self.spatial_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

        # Paper Section 3.2.4: initialize attention weights from N(0, sigma^2) with small sigma
        # so weights start near-uniform and adapt gradually during training
        self._init_gaussian_weights(sigma=0.01)

    def _init_gaussian_weights(self, sigma: float = 0.01) -> None:
        """Initialize attention conv weights from N(0, sigma^2) per paper Section 3.2.4."""
        nn.init.normal_(self.spatial_conv.weight, mean=0.0, std=sigma)

    def forward(self, f_cce: torch.Tensor, f_msgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_cce: Feature map from CCE of shape (B, cce_channels, H, W).
            f_msgs: Feature map from MSGS of shape (B, msgs_channels, H, W).

        Returns:
            F_AEF: Fused attention feature map of shape (B, out_channels, H, W).
        """
        # 1. Align channel dimensions (Eq. 22 & 23)
        f_cce_prime = self.cce_align(f_cce)    # (B, C, H, W)
        f_msgs_prime = self.msgs_align(f_msgs)  # (B, C, H, W)

        # 2. Channel-level attention weights Ac via GAP + Softmax (Eq. 24)
        # Paper: Ac = Softmax(GAP(F'_CCE)) — normalizes across C channels so weights
        # sum to 1, enforcing competitive (not independent) channel selection.
        gap_cce = f_cce_prime.mean(dim=(2, 3))          # (B, C)
        a_c = F.softmax(gap_cce, dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)

        # 3. Spatial-level attention weights As via 1x1 Conv + Softmax (Eq. 25)
        # Paper: As = Softmax(Conv1×1(F'_MSGS)) — normalizes spatial map across channels
        # so each spatial position competitively selects which channel to amplify.
        spatial_logits = self.spatial_conv(f_msgs_prime)  # (B, C, H, W)
        a_s = F.softmax(spatial_logits, dim=1)             # (B, C, H, W)

        # 4. Element-wise weighted fusion (Eq. 26)
        f_aef = a_c * f_cce_prime + a_s * f_msgs_prime

        return f_aef