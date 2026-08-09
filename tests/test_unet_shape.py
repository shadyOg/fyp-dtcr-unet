import sys
import torch

# Make local package importable
sys.path.insert(0, "dtcr-unet")

from models.unet.unet_model import UNet


def run_test():
    for bilinear in (False, True):
        model = UNet(1, 1, bilinear=bilinear)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        print(f"bilinear={bilinear} -> out.shape={out.shape}")


if __name__ == '__main__':
    run_test()
