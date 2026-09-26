"""Decoder module and upsampling blocks for DTCR-U-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ConvBlock


class DecoderUpBlock(nn.Module):
    """Upsampling block supporting both Bilinear Interpolation and ConvTranspose2d."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Deep feature map to upsample (B, in_channels, H, W).
            skip: Skip connection feature map (B, skip_channels, 2H, 2W).
        """
        x = self.up(x)

        # Match spatial dimensions if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        merged = torch.cat([skip, x], dim=1)
        return self.conv(merged)