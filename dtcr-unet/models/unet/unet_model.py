""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # checkpointing flag — when True, forward() will wrap selected
        # module calls with torch.utils.checkpoint.checkpoint for memory savings.
        self._use_checkpoint = False

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # Use checkpointing wrappers if enabled. checkpoint.checkpoint requires
        # functions that take tensors and return tensors, so we wrap modules.
        if self._use_checkpoint:
            x1 = torch.utils.checkpoint.checkpoint(self.inc, x)
            x2 = torch.utils.checkpoint.checkpoint(self.down1, x1)
            x3 = torch.utils.checkpoint.checkpoint(self.down2, x2)
            x4 = torch.utils.checkpoint.checkpoint(self.down3, x3)
            x5 = torch.utils.checkpoint.checkpoint(self.down4, x4)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
        # For up blocks, checkpointing is less common but supported similarly
        if self._use_checkpoint:
            x = torch.utils.checkpoint.checkpoint(self.up1, x5, x4)
            x = torch.utils.checkpoint.checkpoint(self.up2, x, x3)
            x = torch.utils.checkpoint.checkpoint(self.up3, x, x2)
            x = torch.utils.checkpoint.checkpoint(self.up4, x, x1)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable checkpointing for the forward pass.

        Call this before training to save memory (at the cost of extra
        compute during the backward pass).
        """
        self._use_checkpoint = True