import sys
import torch

sys.path.insert(0, "dtcr-unet")

from models.dtcr_unet import DTCR_UNet, inverse_level_set_transform
from losses.dtc_loss import DTCRTotalLoss


def test_dtcr_unet_forward_backward():
    model = DTCR_UNet(in_channels=1, n_classes=1, bilinear=False)
    x = torch.randn(2, 1, 256, 256)
    
    seg_logits, lsf_pred, lsf_trans = model(x)
    assert seg_logits.shape == (2, 1, 256, 256), f"Unexpected seg_logits shape: {seg_logits.shape}"
    assert lsf_pred.shape == (2, 1, 256, 256), f"Unexpected lsf_pred shape: {lsf_pred.shape}"
    assert lsf_trans.shape == (2, 1, 256, 256), f"Unexpected lsf_trans shape: {lsf_trans.shape}"

    # Verify inverse mapping calibration:
    # z = -1 (lesion) -> sigma(5) >= 0.99
    # z = +1 (background) -> sigma(-5) <= 0.01
    z_lesion = torch.tensor([-1.0])
    z_bg = torch.tensor([1.0])
    assert inverse_level_set_transform(z_lesion).item() > 0.99, "Inverse mapping under-calibrated for lesion"
    assert inverse_level_set_transform(z_bg).item() < 0.01, "Inverse mapping under-calibrated for background"

    # Loss computation & backward pass test
    criterion = DTCRTotalLoss()
    target_mask = (torch.rand(2, 1, 256, 256) > 0.5).float()
    target_lsf = torch.randn(2, 1, 256, 256).clamp(-1.0, 1.0)
    is_labeled = torch.tensor([True, False])

    loss, loss_dict = criterion(
        seg_logits, lsf_pred, lsf_trans,
        target_mask, target_lsf, is_labeled,
        epoch=1, max_epochs=80
    )
    assert not torch.isnan(loss), "Loss computed is NaN"
    loss.backward()
    print("All DTCR-U-Net tests passed successfully!")


if __name__ == "__main__":
    test_dtcr_unet_forward_backward()
