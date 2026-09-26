"""Channel Contextual Enhancement (CCE) Module for DTCR-U-Net.

Implements Multi-Scale Feature Embedding (MSFE) and Channel Self-Attention with
Global Channel Pooling (GCP) calibration as described in Section 3.2.1 and 3.2.2
(Equations 11-16) of the DTCR-U-Net paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ChannelContextualEnhancement(nn.Module):
    """CCE Module: models inter-channel dependencies across multi-scale encoder features."""

    def __init__(
        self,
        in_channels_list: List[int] = [64, 128, 256, 512],
        proj_dim: int = 128,
        token_size: Tuple[int, int] = (16, 16),
    ):
        """
        Args:
            in_channels_list: List of channel dimensions from 4 encoder stages [E1, E2, E3, E4].
            proj_dim: Unified projection dimension C (default: 128 per paper Eq. 11).
            token_size: Common spatial resolution (h0, w0) to align token lengths d = h0 * w0.
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.token_size = token_size
        self.full_dim = len(in_channels_list) * proj_dim  # C_sigma = 4 * C = 512

        # 1x1 convolutions for channel alignment to C = 128 (Eq. 11)
        self.align_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, proj_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            for in_c in in_channels_list
        ])

        # Linear projection matrices for Channel Self-Attention (Eq. 14)
        self.q_proj = nn.Linear(self.full_dim, self.full_dim, bias=False)
        self.k_proj = nn.Linear(self.full_dim, self.full_dim, bias=False)
        self.v_proj = nn.Linear(self.full_dim, self.full_dim, bias=False)

        # Instance Normalization for stabilizing channel similarity matrix
        self.inst_norm = nn.InstanceNorm1d(self.full_dim)

        # Global Channel Pooling (GCP) calibration (Eq. 16)
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.full_dim, self.full_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.full_dim // 4, self.full_dim),
            nn.Sigmoid(),
        )
        self.fc_beta = nn.Sequential(
            nn.Linear(self.full_dim, self.full_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.full_dim // 4, self.full_dim),
        )

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoder_features: List of 4 feature tensors [E1, E2, E3, E4] from encoder stages.

        Returns:
            F_CCE: Channel-enhanced feature map of shape (B, C_sigma, token_h, token_w).
        """
        b = encoder_features[0].size(0)
        tokens = []

        # 1. Channel alignment and tokenization (Eq. 11 & 12)
        for i, (conv, feat) in enumerate(zip(self.align_convs, encoder_features)):
            aligned = conv(feat)  # (B, C, H_i, W_i)
            # Pool to common spatial token size
            pooled = F.adaptive_avg_pool2d(aligned, self.token_size)  # (B, C, h0, w0)
            tokens.append(pooled)

        # 2. Multi-scale feature concatenation along channel dimension (Eq. 13)
        # T_sigma shape: (B, C_sigma, h0, w0) -> flattened to (B, C_sigma, d) where d = h0 * w0
        t_sigma_spatial = torch.cat(tokens, dim=1)  # (B, C_sigma, h0, w0)
        h0, w0 = t_sigma_spatial.shape[2], t_sigma_spatial.shape[3]
        d = h0 * w0
        t_sigma = t_sigma_spatial.flatten(2)  # (B, C_sigma, d)

        # Transpose for linear projection across channel vectors: (B, d, C_sigma)
        t_trans = t_sigma.transpose(1, 2)
        q_c = self.q_proj(t_trans).transpose(1, 2)  # (B, C_sigma, d)
        k_c = self.k_proj(t_trans).transpose(1, 2)  # (B, C_sigma, d)
        v_c = self.v_proj(t_trans).transpose(1, 2)  # (B, C_sigma, d)

        # 3. Channel dot-product attention matrix M_c (Eq. 15)
        # Similarity: (B, C_sigma, d) @ (B, d, C_sigma) -> (B, C_sigma, C_sigma)
        scale = (self.full_dim) ** 0.5
        sim = torch.bmm(q_c, k_c.transpose(1, 2)) / scale
        m_c = F.softmax(sim, dim=-1)  # (B, C_sigma, C_sigma)

        # 4. Weight value matrix and residual connection: (B, C_sigma, d)
        attended = torch.bmm(m_c, v_c) + t_sigma

        # 5. Global Channel Pooling (GCP) calibration (Eq. 16)
        # GAP across sequence dimension d
        gap = attended.mean(dim=-1)  # (B, C_sigma)
        gamma = self.fc_gamma(gap).unsqueeze(-1)  # (B, C_sigma, 1)
        beta = self.fc_beta(gap).unsqueeze(-1)    # (B, C_sigma, 1)

        f_cce_flat = gamma * attended + beta  # (B, C_sigma, d)
        f_cce = f_cce_flat.view(b, self.full_dim, h0, w0)

        return f_cce