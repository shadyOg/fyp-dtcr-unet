"""Data Preprocessing Pipeline for DTCR-U-Net.

Implements dataset extraction, HU windowing, CLAHE contrast enhancement,
Level Set Function (LSF) / Signed Distance Map computation, and patient-level
splitting for MosMedData and LIDC-IDRI datasets as specified in the paper.
"""

import argparse
import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def compute_level_set(mask_2d: np.ndarray, lambda_smooth: float = 0.3) -> np.ndarray:
    """Compute the continuous Level Set Function (LSF) T(y) from a binary mask.

    Per Eq. (1) in the DTCR-U-Net paper:
        T(x) = -inf_{y in dS} ||x - y||_2 (inside lesion, S_in)
        T(x) = 0                         (on boundary, dS)
        T(x) = +inf_{y in dS} ||x - y||_2 (outside lesion, S_out)

    Args:
        mask_2d: Binary mask of shape (H, W) where 1 indicates lesion, 0 is background.
        lambda_smooth: Smoothing coefficient for gradient regularization.

    Returns:
        Signed distance / Level Set map of shape (H, W), dtype float32.
    """
    mask_bool = mask_2d > 0

    if not np.any(mask_bool):
        # Empty mask (no lesion in this slice) -> entire slice is outside
        # Distance to nearest boundary is max image distance
        h, w = mask_2d.shape
        max_dist = np.sqrt(h**2 + w**2)
        return np.full((h, w), max_dist, dtype=np.float32)

    if np.all(mask_bool):
        # Entire slice is lesion
        h, w = mask_2d.shape
        max_dist = np.sqrt(h**2 + w**2)
        return np.full((h, w), -max_dist, dtype=np.float32)

    # Euclidean distance transform
    dist_out = distance_transform_edt(~mask_bool)
    dist_in = distance_transform_edt(mask_bool)

    # Level set: negative inside, positive outside
    lsf = (dist_out - dist_in).astype(np.float32)

    # Optional gradient smoothing adjustment if lambda_smooth > 0
    if lambda_smooth > 0:
        grad_y, grad_x = np.gradient(lsf)
        grad_norm_sq = grad_x**2 + grad_y**2
        lsf = lsf + lambda_smooth * grad_norm_sq

    return lsf.astype(np.float32)


def normalize_ct_slice(
    slice_2d: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    apply_clahe: bool = True,
) -> np.ndarray:
    """Clip Hounsfield Units (HU) to lung window, normalize to [0, 1], and apply CLAHE.

    Args:
        slice_2d: 2D array of CT slice in HU.
        hu_min: Lower bound of lung window (default -1000 HU).
        hu_max: Upper bound of lung window (default +400 HU).
        apply_clahe: Whether to apply CLAHE for contrast enhancement.

    Returns:
        Normalized 2D array of shape (H, W), dtype float32 in range [0, 1].
    """
    clipped = np.clip(slice_2d, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min + 1e-8)

    if apply_clahe:
        # Convert to 8-bit for cv2 CLAHE
        img_uint8 = (normalized * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_uint8)
        normalized = (img_clahe.astype(np.float32) / 255.0)

    return normalized.astype(np.float32)


def resize_slice(
    arr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    is_mask: bool = False,
) -> np.ndarray:
    """Resize slice to target dimensions.

    Uses INTER_NEAREST for binary masks, INTER_LINEAR for images and continuous maps.
    """
    if arr.shape[0] == target_size[0] and arr.shape[1] == target_size[1]:
        return arr

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(arr, target_size, interpolation=interp)
    return resized


def find_mosmed_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """Recursively discover CT volumes and paired mask files in MosMed directory.

    Supports:
    - studies/CT-* and masks/ layout
    - flat / nested folder structures with _mask suffix
    """
    root_path = Path(root_dir)
    
    # 1. Find all mask files
    mask_files = list(root_path.glob("**/masks/*_mask.nii*")) or list(root_path.glob("**/*_mask.nii*"))
    mask_files = sorted(set(str(f) for f in mask_files))

    paired_studies = []
    paired_masks = []

    for mask_path in mask_files:
        stem = Path(mask_path).stem
        if stem.endswith(".nii"):
            stem = Path(stem).stem
        study_id = stem.replace("_mask", "")

        # Search for corresponding CT scan across studies/ subfolders
        candidates = list(root_path.glob(f"**/{study_id}.nii*"))
        # Filter out the mask file itself from candidates
        ct_candidates = [c for c in candidates if "_mask" not in str(c)]

        if ct_candidates:
            paired_studies.append(str(ct_candidates[0]))
            paired_masks.append(mask_path)

    return paired_studies, paired_masks


def process_mosmed_dataset(
    mosmed_root: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    context_slices: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    """Extract, preprocess, and save MosMed slices with patient-level split.

    Args:
        mosmed_root: Root folder containing MosMedData (studies and masks).
        output_dir: Destination folder for preprocessed .npy files.
        target_size: (H, W) resolution (default: (256, 256) per DTCR-U-Net paper).
        context_slices: Neighboring slices to extract alongside each lesion slice (default: 1 for [-1, 0, +1]).
        test_ratio: Fraction for test set (default: 0.2).
        val_ratio: Fraction of train+val for validation (default: 0.2).
        labeled_ratio: Fraction of training set that retains labels for semi-supervised training.
        seed: Random seed for reproducibility.

    Returns:
        Manifest dictionary with dataset statistics and saved counts.
    """
    out_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(out_path / split / "images", exist_ok=True)
        os.makedirs(out_path / split / "masks", exist_ok=True)
        os.makedirs(out_path / split / "level_sets", exist_ok=True)

    ct_files, mask_files = find_mosmed_files(mosmed_root)
    print(f"Found {len(ct_files)} paired MosMed volumes.")

    if not ct_files:
        raise FileNotFoundError(f"No paired MosMed studies/masks found in {mosmed_root}")

    # Build unique (study_id, slice_idx) slice registry with 3-slice context
    all_study_pairs: Dict[str, List[int]] = {}

    for ct_path, mask_path in zip(ct_files, mask_files):
        study_id = Path(ct_path).stem.replace(".nii", "")
        mask_vol = nib.load(mask_path).get_fdata()
        num_slices = mask_vol.shape[2]

        lesion_slices = np.where(mask_vol.sum(axis=(0, 1)) > 0)[0]
        if len(lesion_slices) == 0:
            continue

        selected_indices = set()
        for s in lesion_slices:
            for offset in range(-context_slices, context_slices + 1):
                idx = s + offset
                if 0 <= idx < num_slices:
                    selected_indices.add(int(idx))

        all_study_pairs[study_id] = sorted(selected_indices)

    unique_studies = sorted(all_study_pairs.keys())
    print(f"Total annotated studies with lesions: {len(unique_studies)}")

    # Patient/Study-level split: 6:2:2 ratio
    train_val_studies, test_studies = train_test_split(
        unique_studies, test_size=test_ratio, random_state=seed
    )
    val_rel_ratio = val_ratio / (1.0 - test_ratio)
    train_studies, val_studies = train_test_split(
        train_val_studies, test_size=val_rel_ratio, random_state=seed
    )

    splits_map = {
        "train": train_studies,
        "val": val_studies,
        "test": test_studies,
    }

    print(
        f"Split breakdown: Train={len(train_studies)} studies, "
        f"Val={len(val_studies)} studies, Test={len(test_studies)} studies."
    )

    # Semi-supervised assignment on training set: pick labeled subset
    random.seed(seed)
    shuffled_train = list(train_studies)
    random.shuffle(shuffled_train)
    num_labeled = int(round(len(shuffled_train) * labeled_ratio))
    labeled_study_set = set(shuffled_train[:num_labeled])

    counts = {"train": 0, "val": 0, "test": 0}
    labeled_counts = {"train_labeled": 0, "train_unlabeled": 0}

    study_to_paths = {Path(c).stem.replace(".nii", ""): (c, m) for c, m in zip(ct_files, mask_files)}

    for split_name, study_list in splits_map.items():
        pbar = tqdm(study_list, desc=f"Processing {split_name} split")
        for study_id in pbar:
            ct_path, mask_path = study_to_paths[study_id]
            ct_vol = nib.load(ct_path).get_fdata()
            mask_vol = nib.load(mask_path).get_fdata()

            is_labeled = (split_name != "train") or (study_id in labeled_study_set)

            for idx in all_study_pairs[study_id]:
                ct_slice = ct_vol[:, :, idx]
                mask_slice = mask_vol[:, :, idx]

                # Preprocess CT image
                ct_norm = normalize_ct_slice(ct_slice)
                ct_resized = resize_slice(ct_norm, target_size=target_size, is_mask=False)

                # Preprocess binary mask
                mask_bin = (mask_slice > 0).astype(np.uint8)
                mask_resized = resize_slice(mask_bin, target_size=target_size, is_mask=True)

                # Compute Level Set map T(y)
                lsf_map = compute_level_set(mask_resized)

                stem = f"mosmed_{study_id}_s{idx}"

                # Save arrays
                np.save(out_path / split_name / "images" / f"{stem}.npy", ct_resized)
                np.save(out_path / split_name / "masks" / f"{stem}.npy", mask_resized)
                np.save(out_path / split_name / "level_sets" / f"{stem}.npy", lsf_map)

                counts[split_name] += 1
                if split_name == "train":
                    if is_labeled:
                        labeled_counts["train_labeled"] += 1
                    else:
                        labeled_counts["train_unlabeled"] += 1

    manifest = {
        "dataset": "MosMedData",
        "target_size": list(target_size),
        "total_extracted_slices": sum(counts.values()),
        "splits": counts,
        "semi_supervised_breakdown": labeled_counts,
        "labeled_ratio": labeled_ratio,
        "seed": seed,
    }

    with open(out_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nPreprocessing complete!")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT datasets for DTCR-U-Net")
    parser.add_argument("--mosmed", required=True, help="Path to MosMed dataset directory")
    parser.add_argument("--out", default="data/processed", help="Output directory for processed .npy files")
    parser.add_argument("--size", type=int, default=256, help="Target image size (e.g. 256 for 256x256)")
    parser.add_argument("--context", type=int, default=1, help="Number of adjacent context slices (+/-)")
    parser.add_argument("--labeled-ratio", type=float, default=1.0, help="Fraction of labeled samples (0.4 for semi-supervised)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    process_mosmed_dataset(
        mosmed_root=args.mosmed,
        output_dir=args.out,
        target_size=(args.size, args.size),
        context_slices=args.context,
        labeled_ratio=args.labeled_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()