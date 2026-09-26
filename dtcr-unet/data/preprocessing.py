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

    # Level set: negative inside, positive outside, normalized to [-1, 1]
    h, w = mask_2d.shape
    scale = float(max(h, w))
    lsf = ((dist_out - dist_in) / scale).astype(np.float32)

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
    """Recursively discover CT volumes and paired mask files in any directory layout."""
    import re
    root_path = Path(root_dir)

    all_nii_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for f in filenames:
            if f.endswith((".nii", ".nii.gz")):
                all_nii_files.append(os.path.join(dirpath, f))

    mask_files = [f for f in all_nii_files if "_mask" in os.path.basename(f).lower()]
    ct_files = [f for f in all_nii_files if "_mask" not in os.path.basename(f).lower()]

    paired_studies = []
    paired_masks = []

    # Map study ID -> ct file path
    ct_map = {}
    for c in ct_files:
        basename = os.path.basename(c)
        match = re.search(r"(study_\d+)", basename, re.IGNORECASE)
        if match:
            ct_map[match.group(1).lower()] = c

    for m in mask_files:
        basename = os.path.basename(m)
        match = re.search(r"(study_\d+)", basename, re.IGNORECASE)
        if match:
            study_key = match.group(1).lower()
            if study_key in ct_map:
                paired_studies.append(ct_map[study_key])
                paired_masks.append(m)

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


def find_lidc_pairs(lidc_root: str) -> List[Tuple[Path, Path, str]]:
    """Discover LIDC-IDRI image/mask pairs from the washingtongold/lidcidri30 dataset.

    Expected structure (Kaggle: washingtongold/lidcidri30):
        seg_LIDC-IDRI/
          LIDC-IDRI-0001/
            nodule_01/
              slice_001/
                slice.png      <- CT image
                MV05.png       <- majority vote mask (>=50% annotators)
    
    Also supports the shuyueg/lidc-idri-byslices variant with same structure.
    Falls back to generic *_mask*/*_seg* pattern matching if LIDC structure not found.
    """
    import re
    root = Path(lidc_root)
    
    # Strategy 1: Look for the specific seg_LIDC-IDRI directory structure
    seg_dirs = list(root.rglob("seg_LIDC-IDRI"))
    if not seg_dirs:
        # Also try without the seg_ prefix
        seg_dirs = list(root.rglob("LIDC-IDRI"))
    
    pairs = []
    
    if seg_dirs:
        seg_dir = seg_dirs[0]
        print(f"Found LIDC-IDRI structure at: {seg_dir}")
        
        for patient_dir in sorted(seg_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name  # e.g., "LIDC-IDRI-0001"
            
            for nodule_dir in sorted(patient_dir.iterdir()):
                if not nodule_dir.is_dir():
                    continue
                    
                for slice_dir in sorted(nodule_dir.iterdir()):
                    if not slice_dir.is_dir():
                        continue
                    
                    # Look for image file (slice.png or similar)
                    img_path = slice_dir / "slice.png"
                    if not img_path.exists():
                        # Try other common names
                        for candidate in ["image.png", "img.png", "slice.jpg"]:
                            img_path = slice_dir / candidate
                            if img_path.exists():
                                break
                    
                    # Look for mask file (MV05.png = majority vote >= 50%)
                    mask_path = slice_dir / "MV05.png"
                    if not mask_path.exists():
                        # Try other mask names
                        for candidate in ["mask.png", "seg.png", "MV50.png", "annotation.png"]:
                            mask_path = slice_dir / candidate
                            if mask_path.exists():
                                break
                    
                    if img_path.exists() and mask_path.exists():
                        pairs.append((img_path, mask_path, patient_id))
        
        if pairs:
            return pairs
    
    # Strategy 2: Fallback - generic filename pattern matching
    print("LIDC-IDRI seg directory not found, trying generic pattern matching...")
    all_files = list(root.rglob("*.*"))
    img_exts = {".png", ".jpg", ".bmp", ".npy", ".tif", ".tiff"}
    
    img_files = [f for f in all_files 
                 if f.suffix.lower() in img_exts 
                 and "mask" not in f.name.lower() 
                 and "seg" not in f.name.lower()
                 and "MV" not in f.name]
    mask_files = [f for f in all_files 
                  if f.suffix.lower() in img_exts 
                  and ("mask" in f.name.lower() or "seg" in f.name.lower() or "MV" in f.name)]
    
    mask_lookup = {}
    for f in mask_files:
        stem = f.stem.replace("_mask", "").replace("mask_", "").replace("_seg", "")
        mask_lookup[stem] = f
    
    for img_p in img_files:
        if img_p.stem in mask_lookup:
            patient_match = re.search(r"(LIDC-IDRI-\d+|patient_\d+|\d{4})", str(img_p), re.IGNORECASE)
            patient_id = patient_match.group(1) if patient_match else img_p.parent.name
            pairs.append((img_p, mask_lookup[img_p.stem], patient_id))
    
    return pairs


def process_lidc_dataset(
    lidc_root: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    max_samples: int = 1000,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    """Extract and preprocess LIDC-IDRI slices with patient-level split."""
    out_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(out_path / split / "images", exist_ok=True)
        os.makedirs(out_path / split / "masks", exist_ok=True)
        os.makedirs(out_path / split / "level_sets", exist_ok=True)

    # Find image and mask pairs under lidc_root
    paired = find_lidc_pairs(lidc_root)

    if not paired:
        print(f"No LIDC-IDRI image/mask pairs found in {lidc_root}.")
        print(f"  Searched for: seg_LIDC-IDRI/<patient>/<nodule>/<slice>/slice.png + MV05.png")
        print(f"  Also tried generic *_mask*/*_seg* patterns.")
        print(f"  Contents of {lidc_root}:")
        try:
            for item in sorted(Path(lidc_root).iterdir())[:10]:
                print(f"    {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        except Exception:
            print(f"    (could not list directory)")
        return {}

    print(f"Found {len(paired)} LIDC-IDRI pairs.")
    
    # Subsample at patient level if too many pairs
    unique_patients = sorted(list(set(p[2] for p in paired)))
    print(f"  Across {len(unique_patients)} patients.")
    
    if len(paired) > max_samples:
        # Subsample patients, not individual slices, to maintain patient-level integrity
        random.seed(seed)
        pairs_by_patient = {}
        for img_p, mask_p, pid in paired:
            pairs_by_patient.setdefault(pid, []).append((img_p, mask_p, pid))
        
        shuffled_patients = list(pairs_by_patient.keys())
        random.shuffle(shuffled_patients)
        
        paired = []
        for pid in shuffled_patients:
            paired.extend(pairs_by_patient[pid])
            if len(paired) >= max_samples:
                break
        
        unique_patients = sorted(list(set(p[2] for p in paired)))
        print(f"  Subsampled to {len(paired)} pairs across {len(unique_patients)} patients.")

    train_val_patients, test_patients = train_test_split(unique_patients, test_size=test_ratio, random_state=seed)
    val_rel_ratio = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=val_rel_ratio, random_state=seed)

    splits_map = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }

    print(f"  Split: Train={len(train_patients)} patients, Val={len(val_patients)} patients, Test={len(test_patients)} patients.")

    # Semi-supervised assignment on training set
    shuffled_train_pts = list(train_patients)
    random.shuffle(shuffled_train_pts)
    num_labeled = int(round(len(shuffled_train_pts) * labeled_ratio))
    labeled_patients_set = set(shuffled_train_pts[:num_labeled])

    counts = {"train": 0, "val": 0, "test": 0}
    labeled_counts = {"train_labeled": 0, "train_unlabeled": 0}

    for idx, (img_p, mask_p, patient_id) in enumerate(tqdm(paired, desc="Processing LIDC-IDRI")):
        if patient_id in splits_map["test"]:
            split_name = "test"
            is_labeled = True
        elif patient_id in splits_map["val"]:
            split_name = "val"
            is_labeled = True
        else:
            split_name = "train"
            is_labeled = (patient_id in labeled_patients_set)

        if img_p.suffix == ".npy":
            img_arr = np.load(img_p).astype(np.float32)
        else:
            img_arr = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                continue
            img_arr = img_arr.astype(np.float32) / 255.0

        if mask_p.suffix == ".npy":
            mask_arr = np.load(mask_p).astype(np.float32)
        else:
            mask_arr = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if mask_arr is None:
                continue
            mask_arr = mask_arr.astype(np.float32) / 255.0

        img_resized = resize_slice(img_arr, target_size=target_size, is_mask=False)
        mask_bin = (mask_arr > 0.5).astype(np.uint8)
        mask_resized = resize_slice(mask_bin, target_size=target_size, is_mask=True)
        lsf_map = compute_level_set(mask_resized)

        stem = f"lidc_{patient_id}_{idx}"
        np.save(out_path / split_name / "images" / f"{stem}.npy", img_resized)
        np.save(out_path / split_name / "masks" / f"{stem}.npy", mask_resized)
        np.save(out_path / split_name / "level_sets" / f"{stem}.npy", lsf_map)

        counts[split_name] += 1
        if split_name == "train":
            if is_labeled:
                labeled_counts["train_labeled"] += 1
            else:
                labeled_counts["train_unlabeled"] += 1

    manifest = {
        "dataset": "LIDC-IDRI",
        "splits": counts,
        "semi_supervised_breakdown": labeled_counts,
    }
    print(f"\nLIDC-IDRI preprocessing complete!")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT datasets for DTCR-U-Net")
    parser.add_argument("--mosmed", default=None, help="Path to MosMed dataset directory")
    parser.add_argument("--lidc", default=None, help="Path to LIDC-IDRI dataset directory (optional)")
    parser.add_argument("--out", default="data/processed", help="Output directory for processed .npy files")
    parser.add_argument("--size", type=int, default=256, help="Target image size (e.g. 256 for 256x256)")
    parser.add_argument("--context", type=int, default=1, help="Number of adjacent context slices (+/-)")
    parser.add_argument("--labeled-ratio", type=float, default=1.0, help="Fraction of labeled samples (0.4 for semi-supervised)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.mosmed is None and args.lidc is None:
        parser.error("At least one of --mosmed or --lidc must be provided.")

    if args.mosmed:
        process_mosmed_dataset(
            mosmed_root=args.mosmed,
            output_dir=args.out,
            target_size=(args.size, args.size),
            context_slices=args.context,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )

    if args.lidc:
        process_lidc_dataset(
            lidc_root=args.lidc,
            output_dir=args.out,
            target_size=(args.size, args.size),
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()