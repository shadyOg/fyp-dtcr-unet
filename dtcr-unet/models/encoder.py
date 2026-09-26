"""Encoder module and building blocks for DTCR-U-Net."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv2D 3x3 => BatchNorm => ReLU) * 2 with padding=1."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """Downscaling with MaxPool2d then ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)