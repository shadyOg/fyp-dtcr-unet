""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from .unet_parts import *
from torch.utils.checkpoint import checkpoint


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
        # Preserve input spatial size by padding input so final logits can be
        # center-cropped back to the original HxW. We first compute how the
        # network maps an input size to an output size (integer math) and find
        # the minimal symmetric padding required.
        def _sim_out_size(h: int) -> int:
            # encoder
            h = h - 4  # inc: two valid 3x3 convs
            h = (h // 2) - 4  # down1
            h = (h // 2) - 4  # down2
            h = (h // 2) - 4  # down3
            h = (h // 2) - 4  # down4 (bottom)
            # decoder
            h = (h * 2) - 4  # up1
            h = (h * 2) - 4  # up2
            h = (h * 2) - 4  # up3
            h = (h * 2) - 4  # up4
            return h

        orig_h, orig_w = x.size(2), x.size(3)

        # find minimal total padding (added to dimension) so output >= original
        def _find_pad(orig_dim: int) -> int:
            max_pad = 1024
            for pad in range(0, max_pad + 1):
                dim = orig_dim + pad
                out = _sim_out_size(dim)
                if out >= orig_dim:
                    return pad
            return max_pad

        pad_h_total = _find_pad(orig_h)
        pad_w_total = _find_pad(orig_w)

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        if pad_h_total or pad_w_total:
            x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        # Use checkpointing wrappers if enabled. checkpoint.checkpoint requires
        # functions that take tensors and return tensors, so we wrap modules.
        if self._use_checkpoint:
            x1 = checkpoint(self.inc, x)
            x2 = checkpoint(self.down1, x1)
            x3 = checkpoint(self.down2, x2)
            x4 = checkpoint(self.down3, x3)
            x5 = checkpoint(self.down4, x4)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

        # For up blocks, checkpointing is less common but supported similarly
        if self._use_checkpoint:
            x = checkpoint(self.up1, x5, x4)
            x = checkpoint(self.up2, x, x3)
            x = checkpoint(self.up3, x, x2)
            x = checkpoint(self.up4, x, x1)
        else:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        logits = self.outc(x)

        # If we padded the input, center-crop the logits back to original size
        if pad_h_total or pad_w_total:
            out_h, out_w = logits.size(2), logits.size(3)
            start_h = (out_h - orig_h) // 2
            start_w = (out_w - orig_w) // 2
            logits = logits[:, :, start_h:start_h + orig_h, start_w:start_w + orig_w]

        return logits

    def use_checkpointing(self):
        """Enable checkpointing for the forward pass.

        Call this before training to save memory (at the cost of extra
        compute during the backward pass).
        """
        self._use_checkpoint = True