"""Level Set Function (LSF) target T(y) for the boundary-regression head (Eq. 1).

Kept in its own light module (numpy + scipy only) so the training Dataset can
recompute T(y) on the fly from the (augmented) mask without importing the heavy
preprocessing dependencies.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

# Distance (in pixels at the training resolution) at which the signed distance
# saturates to +/-1. Must stay in sync with the steepness `k` of the inverse
# transform T^{-1}(z) = sigmoid(-k z): one pixel is 1/LSF_TRUNC_PX in LSF units.
LSF_TRUNC_PX = 8.0


def compute_level_set(mask_2d: np.ndarray, trunc_px: float = LSF_TRUNC_PX) -> np.ndarray:
    """Signed distance map of a binary mask: negative inside, positive outside (Eq. 1).

    The distance is truncated at `trunc_px` pixels and scaled to [-1, 1].

    Why truncated instead of dividing by the image size: with distance / 256 a
    10-pixel-wide lesion only reaches about -0.02 inside, while the background
    ramps up to +0.5. The regression loss is then dominated by the background
    ramp and small lesions are numerically invisible to Task 2, so T^{-1}(f2)
    says "no lesion" and the consistency loss drags the segmentation head with it.
    Truncation gives every lesion, whatever its size, a strong and local signal.

    Args:
        mask_2d: Binary mask (H, W); non-zero means lesion.
        trunc_px: Saturation distance in pixels.

    Returns:
        float32 array (H, W) in [-1, 1].
    """
    mask_bool = np.asarray(mask_2d) > 0
    h, w = mask_bool.shape

    if not mask_bool.any():
        return np.ones((h, w), dtype=np.float32)
    if mask_bool.all():
        return -np.ones((h, w), dtype=np.float32)

    dist_out = distance_transform_edt(~mask_bool)  # >= 1 outside, 0 inside
    dist_in = distance_transform_edt(mask_bool)    # >= 1 inside, 0 outside

    # Shift by half a pixel so the zero level set sits between the last inside
    # pixel and the first outside pixel (the true contour dS).
    sdf = np.where(mask_bool, -(dist_in - 0.5), dist_out - 0.5)
    return np.clip(sdf / float(trunc_px), -1.0, 1.0).astype(np.float32)
