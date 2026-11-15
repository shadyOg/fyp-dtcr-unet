"""DTCR-U-Net with an ImageNet-pretrained ResNet-34 encoder.

Same design as `DTCR_UNet` (paper Sections 3.1-3.3): MSAA attention (CCE -> MSGS ->
AEF) on the multi-scale encoder features, MsSC skip connections, and the dual
heads (segmentation f1, level set f2, inverse transform T^{-1}(f2)). Only the
encoder changes: a ResNet-34 pretrained on ImageNet instead of a randomly
initialised one. With ~90 annotated training patients, transfer learning is the
standard way to get a stronger, better-generalising encoder.

Input is 1 or 3 channels in [0, 1]; 3 channels = 2.5D (slice and its two
neighbours), which maps naturally onto the pretrained RGB filters.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aef import AttentionEmbeddingFusion
from .cce import ChannelContextualEnhancement
from .decoder import DecoderUpBlock
from .dtcr_unet import inverse_level_set_transform
from .encoder import ConvBlock
from .msgs import MultiScaleGlobalSpatial
from .mssc import MultiScaleSkipConnection

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DTCR_ResNetUNet(nn.Module):
    """DTCR-U-Net on a ResNet-34 backbone.

    Encoder features (input H x W):
        e0  32 @ H      full-resolution conv stem (keeps detail for tiny lesions)
        e1  64 @ H/2    ResNet stem
        e2  64 @ H/4    layer1
        e3 128 @ H/8    layer2
        e4 256 @ H/16   layer3
        e5 512 @ H/32   layer4
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1, pretrained: bool = True, bilinear: bool = False):
        super().__init__()
        from torchvision.models import ResNet34_Weights, resnet34

        self.in_channels = in_channels
        self.n_classes = n_classes
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        self.stem0 = ConvBlock(in_channels, 32)
        self.stem1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4

        # MSAA on the four deepest scales (64, 128, 256, 512 channels), as in DTCR_UNet
        self.cce = ChannelContextualEnhancement(in_channels_list=[64, 128, 256, 512], proj_dim=128, token_size=(16, 16))
        self.msgs = MultiScaleGlobalSpatial(in_channels=512, out_channels=128)
        self.aef = AttentionEmbeddingFusion(cce_channels=512, msgs_channels=128, out_channels=128)
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(512 + 128, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # MsSC skips: [target scale, finer neighbour, coarser neighbour]
        self.mssc4 = MultiScaleSkipConnection([256, 128, 512], out_channels=256)
        self.mssc3 = MultiScaleSkipConnection([128, 64, 256], out_channels=128)
        self.mssc2 = MultiScaleSkipConnection([64, 64, 128], out_channels=64)
        self.mssc1 = MultiScaleSkipConnection([64, 32, 64], out_channels=64)

        self.up4 = DecoderUpBlock(512, 256, 256, bilinear=bilinear)   # -> H/16
        self.up3 = DecoderUpBlock(256, 128, 128, bilinear=bilinear)   # -> H/8
        self.up2 = DecoderUpBlock(128, 64, 64, bilinear=bilinear)     # -> H/4
        self.up1 = DecoderUpBlock(64, 64, 64, bilinear=bilinear)      # -> H/2
        self.up0 = DecoderUpBlock(64, 32, 32, bilinear=bilinear)      # -> H

        self.seg_head = nn.Conv2d(32, n_classes, kernel_size=1)
        self.lsf_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.size(1) != 3:
            raise ValueError(f"expected 1 or 3 input channels, got {x.size(1)}")
        return (x - self.img_mean) / self.img_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.stem0(x)
        e1 = self.stem1(self._to_rgb(x))
        e2 = self.layer1(self.pool(e1))
        e3 = self.layer2(e2)
        e4 = self.layer3(e3)
        e5 = self.layer4(e4)

        f_cce = self.cce([e2, e3, e4, e5])
        f_aef = self.aef(f_cce, self.msgs(f_cce))
        if f_aef.shape[2:] != e5.shape[2:]:
            f_aef = F.interpolate(f_aef.to(e5.dtype), size=e5.shape[2:], mode="bilinear", align_corners=False)
        b = self.bottleneck_fuse(torch.cat([e5, f_aef.to(e5.dtype)], dim=1))

        d = self.up4(b, self.mssc4([e4, e3, e5]))
        d = self.up3(d, self.mssc3([e3, e2, e4]))
        d = self.up2(d, self.mssc2([e2, e1, e3]))
        d = self.up1(d, self.mssc1([e1, e0, e2]))
        d = self.up0(d, e0)

        f1_logits = torch.clamp(self.seg_head(d), min=-20.0, max=20.0)
        f2_lsf = torch.tanh(self.lsf_head(d).float())
        return f1_logits, f2_lsf, inverse_level_set_transform(f2_lsf)


def build_model(arch: str, in_channels: int = 1, n_classes: int = 1, pretrained: bool = True,
                base_c: int = 64, num_stages: int = 4, bilinear: bool = False) -> nn.Module:
    """Construct a model by name: 'dtcr' (paper architecture) or 'dtcr_resnet34'."""
    if arch == "dtcr_resnet34":
        return DTCR_ResNetUNet(in_channels=in_channels, n_classes=n_classes, pretrained=pretrained, bilinear=bilinear)
    if arch == "dtcr":
        from .dtcr_unet import DTCR_UNet
        return DTCR_UNet(in_channels=in_channels, n_classes=n_classes, bilinear=bilinear, base_c=base_c,
                         num_stages=num_stages)
    raise ValueError(f"unknown architecture {arch!r}")
