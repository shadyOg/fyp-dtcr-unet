"""
preprocessing.py

Preprocesses raw MosMedData CT volumes into 2D slices
ready for the MosMedDataset class.

Raw MosMedData structure expected:
    raw_data_dir/
    ├── COVID19_1110/
    │   ├── studies/
    │   │   └── CT-0/   ← CT volumes (.nii.gz)
    │   │   └── CT-1/
    │   │   └── CT-2/
    │   │   └── CT-3/
    │   │   └── CT-4/
    │   └── masks/
    │       └── COVID19_1110_1_CT-1/ ← mask volumes (.nii.gz)

Output structure (what MosMedDataset expects):
    output_dir/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── masks/
        ├── train/
        ├── val/
        └── test/

Usage (in Colab):
    python preprocessing.py \
        --raw_dir /content/mosmeddata \
        --out_dir /content/processed \
        --augment
"""

import os
import argparse
import random
import numpy as np
import nibabel as nib
from glob import glob
from pathlib import Path
from skimage.transform import resize
from skimage import exposure
from tqdm import tqdm


# ─────────────────────────────────────────────
#  SEED — for reproducible train/val/test split
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────
#  STEP 1 — LOAD A NIFTI VOLUME
# ─────────────────────────────────────────────

def load_nifti(path):
    """Load a .nii or .nii.gz file and return as numpy array."""
    vol = nib.load(str(path))
    return np.asarray(vol.get_fdata(), dtype=np.float32)


# ─────────────────────────────────────────────
#  STEP 2 — NORMALIZE PIXEL VALUES TO [0, 1]
# ─────────────────────────────────────────────

def normalize(slice_2d, window_min=-1000.0, window_max=400.0):
    """Apply a fixed CT window and normalize to [0, 1]."""
    slice_2d = np.clip(slice_2d, window_min, window_max)
    return (slice_2d - window_min) / (window_max - window_min + 1e-8)


# ─────────────────────────────────────────────
#  STEP 3 — HISTOGRAM EQUALIZATION
#  Enhances lesion contrast by redistributing
#  pixel intensities more evenly
# ─────────────────────────────────────────────

def histogram_equalize(slice_2d):
    """Apply histogram equalization to enhance lesion contrast."""
    # equalize_hist expects values in [0, 1]
    return exposure.equalize_hist(slice_2d)


# ─────────────────────────────────────────────
#  STEP 4 — CONTRAST ADJUSTMENT
#  Clips extreme intensity values (top/bottom 2%)
#  to improve separability between lesion and background
# ─────────────────────────────────────────────

def adjust_contrast(slice_2d, low_pct=2, high_pct=98):
    """Clip intensities to [low_pct, high_pct] percentile range and rescale."""
    p_low, p_high = np.percentile(slice_2d, (low_pct, high_pct))
    clipped = np.clip(slice_2d, p_low, p_high)
    if p_high > p_low:
        return (clipped - p_low) / (p_high - p_low + 1e-8)
    return clipped


# ─────────────────────────────────────────────
#  STEP 5 — DOWNSAMPLE TO 256x256
# ─────────────────────────────────────────────

def downsample(slice_2d, target_size=(256, 256)):
    """Resize a 2D slice to target_size using bilinear interpolation."""
    return resize(
        slice_2d,
        target_size,
        order=1,            # bilinear interpolation for images
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True
    ).astype(np.float32)


def downsample_mask(mask_2d, target_size=(256, 256)):
    """Resize a 2D mask using nearest neighbour (preserves binary values)."""
    return resize(
        mask_2d,
        target_size,
        order=0,            # nearest neighbour for masks — no interpolation
        mode='reflect',
        anti_aliasing=False,
        preserve_range=True
    ).astype(np.float32)


# ─────────────────────────────────────────────
#  STEP 6 — AUGMENTATION (labeled data only)
#  Returns a list of augmented (image, mask) pairs
#  including the original
# ─────────────────────────────────────────────

def augment(image, mask):
    """
    Apply flipping, rotation, and random crop augmentations.
    Returns list of (image, mask) tuples including the original.
    """
    augmented = [(image, mask)]  # always include original

    # Horizontal flip
    aug_img = np.fliplr(image).copy()
    aug_mask = np.fliplr(mask).copy()
    augmented.append((aug_img, aug_mask))

    # Vertical flip
    aug_img = np.flipud(image).copy()
    aug_mask = np.flipud(mask).copy()
    augmented.append((aug_img, aug_mask))

    # 90 degree rotation
    aug_img = np.rot90(image, k=1).copy()
    aug_mask = np.rot90(mask, k=1).copy()
    augmented.append((aug_img, aug_mask))

    # 180 degree rotation
    aug_img = np.rot90(image, k=2).copy()
    aug_mask = np.rot90(mask, k=2).copy()
    augmented.append((aug_img, aug_mask))

    # 270 degree rotation
    aug_img = np.rot90(image, k=3).copy()
    aug_mask = np.rot90(mask, k=3).copy()
    augmented.append((aug_img, aug_mask))

    # Random crop then resize back to 256x256
    h, w = image.shape
    crop_size = int(h * 0.85)       # crop 85% of the image
    top  = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    aug_img  = image[top:top+crop_size, left:left+crop_size]
    aug_mask = mask[top:top+crop_size, left:left+crop_size]
    # resize back to 256x256
    aug_img  = downsample(aug_img, (256, 256))
    aug_mask = downsample_mask(aug_mask, (256, 256))
    augmented.append((aug_img, aug_mask))

    return augmented


# ─────────────────────────────────────────────
#  STEP 7 — PROCESS ONE VOLUME PAIR
#  Extracts annotated slices + adjacent slices,
#  applies all preprocessing steps,
#  optionally augments
# ─────────────────────────────────────────────

def process_volume(img_vol, mask_vol, vol_id, out_img_dir,
                   out_mask_dir, apply_augment=False):
    """
    Process one CT volume + its mask.

    Args:
        img_vol:       3D numpy array (H, W, D)
        mask_vol:      3D numpy array (H, W, D) — binary mask
        vol_id:        unique string identifier for this volume
        out_img_dir:   where to save processed image slices
        out_mask_dir:  where to save processed mask slices
        apply_augment: whether to apply augmentation

    Returns:
        Number of slices saved
    """
    saved = 0
    num_slices = img_vol.shape[2]

    # Find which slices contain annotated lesions
    annotated_slices = [
        z for z in range(num_slices)
        if mask_vol[:, :, z].sum() > 0
    ]

    if len(annotated_slices) == 0:
        # No annotations in this volume — skip
        # (this volume will be used as unlabeled data)
        return 0

    # For each annotated slice, take it + adjacent slices
    slices_to_extract = set()
    for z in annotated_slices:
        slices_to_extract.add(max(0, z - 1))        # slice above
        slices_to_extract.add(z)                     # annotated slice
        slices_to_extract.add(min(num_slices-1, z+1))# slice below

    for z in sorted(slices_to_extract):
        # Extract 2D slice
        img_slice  = img_vol[:, :, z].astype(np.float32)
        mask_slice = mask_vol[:, :, z].astype(np.float32)

        # Apply preprocessing pipeline
        img_slice = normalize(img_slice)
        img_slice = histogram_equalize(img_slice)
        img_slice = adjust_contrast(img_slice)
        img_slice = downsample(img_slice, (256, 256))

        mask_slice = downsample_mask(mask_slice, (256, 256))
        # Binarize mask after resize (nearest neighbour can introduce
        # fractional values at boundaries — snap back to 0/1)
        mask_slice = (mask_slice > 0.5).astype(np.float32)

        if apply_augment:
            pairs = augment(img_slice, mask_slice)
        else:
            pairs = [(img_slice, mask_slice)]

        for aug_idx, (img_out, mask_out) in enumerate(pairs):
            filename = f"{vol_id}_slice{z:03d}_aug{aug_idx}.npy"
            np.save(os.path.join(out_img_dir,  filename), img_out)
            np.save(os.path.join(out_mask_dir, filename), mask_out)
            saved += 1

    return saved


# ─────────────────────────────────────────────
#  STEP 8 — FIND VOLUME PAIRS IN RAW DATA
#  MosMedData stores CT volumes in studies/CT-X/
#  and masks in masks/ with matching names
# ─────────────────────────────────────────────

def find_volume_pairs(raw_dir):
    """
    Find all (image_path, mask_path) pairs in MosMedData.
    Returns:
        labeled:   list of (img_path, mask_path) — has a mask
        unlabeled: list of img_path — no mask available
    """
    # Find all NIfTI volumes under raw_dir
    all_paths = sorted(glob(os.path.join(raw_dir, '**', '*.nii.gz'), recursive=True))

    # Masks and images are both .nii.gz, so classify by filename hints
    mask_paths = [p for p in all_paths if 'mask' in p.lower() or 'segm' in p.lower() or 'annotation' in p.lower()]
    img_paths  = [p for p in all_paths if p not in mask_paths]

    labeled   = []
    unlabeled = []

    # Build a lookup by mask basename for robust matching
    mask_lookup = {Path(m).stem: m for m in mask_paths}

    for img_path in img_paths:
        img_stem = Path(img_path).stem

        # Exact stem match first
        if img_stem in mask_lookup:
            labeled.append((img_path, mask_lookup[img_stem]))
            continue

        # Fallback: match using partial stem overlap
        matches = [m for stem, m in mask_lookup.items() if img_stem in stem or stem in img_stem]
        if matches:
            labeled.append((img_path, matches[0]))
        else:
            unlabeled.append(img_path)

    return labeled, unlabeled


def validate_sample_pair(labeled_pairs, raw_dir):
    """Print one example labeled volume/mask pair and its shape."""
    if not labeled_pairs:
        print('No labeled pairs found to validate.')
        return

    sample_img, sample_mask = labeled_pairs[0]
    img_vol = load_nifti(sample_img)
    mask_vol = load_nifti(sample_mask)

    print('\nValidation check:')
    print(f'  sample image: {sample_img}')
    print(f'  sample mask:  {sample_mask}')
    print(f'  image shape: {img_vol.shape}')
    print(f'  mask shape:  {mask_vol.shape}')
    print(f'  matching slices with annotation: {int((mask_vol.sum(axis=(0,1)) > 0).sum())}')
    print(f'  first annotated slice index: {np.where(mask_vol.sum(axis=(0,1)) > 0)[0][0] if (mask_vol.sum(axis=(0,1)) > 0).any() else None}')


# ─────────────────────────────────────────────
#  STEP 9 — TRAIN / VAL / TEST SPLIT
#  Split at volume level (6:2:2)
#  Slice-level split would cause data leakage
# ─────────────────────────────────────────────

def split_volumes(volume_list, train_ratio=0.6, val_ratio=0.2):
    """
    Split a list of volumes into train/val/test at 6:2:2.
    Shuffled with fixed seed for reproducibility.
    """
    indices = list(range(len(volume_list)))
    random.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train = [volume_list[i] for i in train_idx]
    val   = [volume_list[i] for i in val_idx]
    test  = [volume_list[i] for i in test_idx]

    return train, val, test


# ─────────────────────────────────────────────
#  MAIN — ORCHESTRATES EVERYTHING
# ─────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("DTCR-U-Net MosMedData Preprocessing")
    print("=" * 60)

    # ── Create output folder structure ───────
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(args.out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, 'masks',  split), exist_ok=True)
    # Also create unlabeled folder (for semi-supervised training later)
    os.makedirs(os.path.join(args.out_dir, 'images', 'unlabeled'), exist_ok=True)

    # ── Find volume pairs ─────────────────────
    print("\nScanning for volume pairs...")
    labeled, unlabeled = find_volume_pairs(args.raw_dir)
    print(f"  Labeled volumes found:   {len(labeled)}")
    print(f"  Unlabeled volumes found: {len(unlabeled)}")

    validate_sample_pair(labeled, args.raw_dir)

    # ── Split labeled volumes ─────────────────
    train_vols, val_vols, test_vols = split_volumes(labeled)
    print(f"\nSplit (labeled volumes):")
    print(f"  Train: {len(train_vols)}")
    print(f"  Val:   {len(val_vols)}")
    print(f"  Test:  {len(test_vols)}")

    # ── Process each split ────────────────────
    split_map = {
        'train': train_vols,
        'val':   val_vols,
        'test':  test_vols,
    }

    total_slices = 0

    for split_name, vol_list in split_map.items():
        print(f"\nProcessing {split_name} split...")

        out_img_dir  = os.path.join(args.out_dir, 'images', split_name)
        out_mask_dir = os.path.join(args.out_dir, 'masks',  split_name)

        # Only augment training data
        apply_aug = args.augment and (split_name == 'train')

        for img_path, mask_path in tqdm(vol_list, desc=split_name):
            vol_id = Path(img_path).stem.replace('.nii', '')

            img_vol  = load_nifti(img_path)
            mask_vol = load_nifti(mask_path)

            n_saved = process_volume(
                img_vol, mask_vol,
                vol_id=vol_id,
                out_img_dir=out_img_dir,
                out_mask_dir=out_mask_dir,
                apply_augment=apply_aug
            )
            total_slices += n_saved

    # ── Process unlabeled volumes ─────────────
    print(f"\nProcessing unlabeled volumes...")
    out_img_dir = os.path.join(args.out_dir, 'images', 'unlabeled')

    for img_path in tqdm(unlabeled, desc='unlabeled'):
        vol_id   = Path(img_path).stem.replace('.nii', '')
        img_vol  = load_nifti(img_path)
        num_slices = img_vol.shape[2]

        # For unlabeled data: save ALL slices (no mask needed)
        # Just normalize — no augmentation, no histogram eq needed
        for z in range(num_slices):
            img_slice = img_vol[:, :, z].astype(np.float32)
            img_slice = normalize(img_slice)
            img_slice = downsample(img_slice, (256, 256))

            filename = f"{vol_id}_slice{z:03d}.npy"
            np.save(os.path.join(out_img_dir, filename), img_slice)

    # ── Final summary ─────────────────────────
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Total labeled slices saved: {total_slices}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 60)

    # Print final counts per split
    for split in splits:
        n = len(glob(os.path.join(args.out_dir, 'images', split, '*.npy')))
        print(f"  {split}: {n} image slices")
    n_unlab = len(glob(os.path.join(args.out_dir, 'images', 'unlabeled', '*.npy')))
    print(f"  unlabeled: {n_unlab} image slices")


# ─────────────────────────────────────────────
#  ARGUMENT PARSER
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess MosMedData for DTCR-U-Net'
    )
    parser.add_argument(
        '--raw_dir',
        required=True,
        help='Path to raw MosMedData root directory'
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help='Path to output directory for processed slices'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply augmentation to training slices (recommended)'
    )
    args = parser.parse_args()
    main(args)