import sys
import torch

sys.path.insert(0, "dtcr-unet")

from models.dtcr_unet import DTCR_UNet, inverse_level_set_transform
from models.aef import AttentionEmbeddingFusion
from losses.dtc_loss import DTCRTotalLoss, DualTaskConsistencyLoss
from losses.seg_loss import DiceLoss


def test_dtcr_unet_forward_backward():
    # 1. Test 4-stage paper architecture (18.65M parameter budget)
    model4 = DTCR_UNet(in_channels=1, n_classes=1, bilinear=False, num_stages=4)
    param_count = sum(p.numel() for p in model4.parameters()) / 1e6
    print(f"4-stage model parameter count: {param_count:.2f}M (Paper Table 1: 18.65M)")
    assert 17.0 <= param_count <= 20.0, f"Expected ~18.65M params, got {param_count:.2f}M"

    x = torch.randn(2, 1, 256, 256)
    seg_logits, lsf_pred, lsf_trans = model4(x)
    assert seg_logits.shape == (2, 1, 256, 256), f"Unexpected seg_logits shape: {seg_logits.shape}"
    assert lsf_pred.shape == (2, 1, 256, 256), f"Unexpected lsf_pred shape: {lsf_pred.shape}"
    assert lsf_trans.shape == (2, 1, 256, 256), f"Unexpected lsf_trans shape: {lsf_trans.shape}"

    # 2. Test 5-stage legacy architecture
    model5 = DTCR_UNet(in_channels=1, n_classes=1, bilinear=False, num_stages=5)
    s5, l5, t5 = model5(x)
    assert s5.shape == (2, 1, 256, 256)
    assert l5.shape == (2, 1, 256, 256)
    assert t5.shape == (2, 1, 256, 256)

    # 3. Verify inverse mapping calibration:
    # z = -1 (lesion) -> sigma(5) >= 0.99
    # z = +1 (background) -> sigma(-5) <= 0.01
    z_lesion = torch.tensor([-1.0])
    z_bg = torch.tensor([1.0])
    assert inverse_level_set_transform(z_lesion).item() > 0.99, "Inverse mapping under-calibrated for lesion"
    assert inverse_level_set_transform(z_bg).item() < 0.01, "Inverse mapping under-calibrated for background"

    # 4. Verify AEF Channel & Spatial Attention Energy Balance
    aef = AttentionEmbeddingFusion(cce_channels=512, msgs_channels=128, out_channels=128)
    dummy_cce = torch.randn(2, 512, 16, 16)
    dummy_msgs = torch.randn(2, 128, 16, 16)
    f_aef = aef(dummy_cce, dummy_msgs)
    assert f_aef.shape == (2, 128, 16, 16)
    # Check that channel and spatial branches have comparable magnitude (~1:1 ratio)
    gap_cce = aef.cce_align(dummy_cce).mean(dim=(2, 3))
    a_c = torch.softmax(aef.channel_proj(gap_cce), dim=1).unsqueeze(-1).unsqueeze(-1) * 128
    spatial_logits = aef.spatial_conv(aef.msgs_align(dummy_msgs)).view(2, 1, -1)
    a_s = torch.softmax(spatial_logits, dim=-1).view(2, 1, 16, 16) * 256
    assert 0.5 <= (a_c.mean() / a_s.mean()).item() <= 2.0, "AEF attention weights are severely unbalanced"

    # 5. Verify Interaction Loss does NOT penalize correct lesion predictions
    dtc_loss = DualTaskConsistencyLoss(lambda1=0.3, lambda2=1.0, lambda_context=0.0)
    # Correct prediction: lesion (prob=1.0, lsf=-1.0) and bg (prob=0.0, lsf=+1.0)
    f1_correct = torch.tensor([[[[1.0, 0.0]]]])
    f2_correct = torch.tensor([[[[-1.0, 1.0]]]])
    f2_t = inverse_level_set_transform(f2_correct)
    _, breakdown = dtc_loss(f1_correct, f2_correct, f2_t)
    assert breakdown["l_interact"] < 0.01, f"Interaction loss penalized correct predictions: {breakdown['l_interact']}"

    # 6. Verify Soft Dice Loss on empty target with zero prediction is 0.0 (no penalty)
    dice_fn = DiceLoss(smooth=1.0)
    empty_pred = torch.zeros(1, 1, 256, 256)
    empty_targ = torch.zeros(1, 1, 256, 256)
    assert dice_fn(empty_pred, empty_targ).item() == 0.0, "Empty target with zero prediction got non-zero Dice loss"

    # 7. Loss computation & backward pass test with paper default components (Eq. 4 & 10)
    criterion = DTCRTotalLoss(lambda1=0.3, lambda2=1.0, beta=0.5, lambda_context=0.0)
    target_mask = (torch.rand(2, 1, 256, 256) > 0.5).float()
    target_lsf = torch.randn(2, 1, 256, 256).clamp(-1.0, 1.0)
    is_labeled = torch.tensor([True, False])

    loss, loss_dict = criterion(
        seg_logits, lsf_pred, lsf_trans,
        target_mask, target_lsf, is_labeled,
        epoch=1, max_epochs=80
    )
    assert not torch.isnan(loss), "Loss computed is NaN"
    assert "loss_total" in loss_dict
    assert "loss_seg" in loss_dict
    assert "loss_lsf" in loss_dict
    assert "loss_dtc" in loss_dict
    assert "l_main" in loss_dict
    assert "l_grad" in loss_dict
    assert "l_interact" in loss_dict

    loss.backward()

    # 8. Test edge cases: completely empty targets and all-foreground targets
    empty_mask = torch.zeros(2, 1, 256, 256)
    empty_lsf = torch.ones(2, 1, 256, 256)  # background distance = +1.0
    loss_empty, _ = criterion(
        seg_logits, lsf_pred, lsf_trans,
        empty_mask, empty_lsf, is_labeled,
        epoch=40, max_epochs=80
    )
    assert not torch.isnan(loss_empty) and torch.isfinite(loss_empty), "Empty target produced non-finite loss"

    full_mask = torch.ones(2, 1, 256, 256)
    full_lsf = -torch.ones(2, 1, 256, 256)  # lesion distance = -1.0
    loss_full, _ = criterion(
        seg_logits, lsf_pred, lsf_trans,
        full_mask, full_lsf, is_labeled,
        epoch=80, max_epochs=80
    )
    assert not torch.isnan(loss_full) and torch.isfinite(loss_full), "Full target produced non-finite loss"

    print("All DTCR-U-Net tests passed successfully!")


if __name__ == "__main__":
    test_dtcr_unet_forward_backward()
