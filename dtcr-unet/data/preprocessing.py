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


def _find_ct_series_folder(patient_path: Path) -> Optional[str]:
    """Find the DICOM CT series folder under a patient directory."""
    import SimpleITK as sitk
    for root, dirs, files in os.walk(patient_path):
        dcm_files = [f for f in files if f.endswith(".dcm")]
        if len(dcm_files) > 10:  # CT series typically has many slices (>10)
            try:
                series_ids = sitk.ImageSeriesReader().GetGDCMSeriesIDs(root)
                if series_ids:
                    return root
            except Exception:
                continue
    return None


def _build_xml_uid_index(xml_root: str) -> Dict[str, str]:
    """Build a SeriesInstanceUID -> XML file path lookup from LIDC XML annotations.
    
    Namespace-agnostic to handle various XML schema versions.
    """
    import xml.etree.ElementTree as ET
    uid_map = {}

    xml_dir = Path(xml_root)
    xml_files = list(xml_dir.rglob("*.xml"))
    print(f"  Indexing {len(xml_files)} XML annotation files...")

    for xml_path in xml_files:
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag.lower() == "seriesinstanceuid" and elem.text:
                    uid_map[elem.text.strip()] = str(xml_path)
                    break
        except Exception:
            continue

    print(f"  Indexed {len(uid_map)} SeriesInstanceUIDs.")
    return uid_map


def _extract_nodule_masks_from_xml(
    xml_path: str,
    ct_origin_z: float,
    ct_spacing_z: float,
    num_slices: int,
    volume_shape: Tuple[int, int],
) -> np.ndarray:
    """Parse LIDC XML and convert radiologist nodule contours to a 3D binary mask volume.

    Uses majority voting: a voxel is positive if >= 2 radiologists marked it,
    or >= 1 if only 1 radiologist marked any nodule.
    """
    import xml.etree.ElementTree as ET
    from skimage.draw import polygon

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Accumulate votes from all reading sessions
    vote_volume = np.zeros((num_slices, volume_shape[0], volume_shape[1]), dtype=np.int32)

    for session in root.iter():
        session_tag = session.tag.split("}")[-1] if "}" in session.tag else session.tag
        if session_tag != "readingSession":
            continue

        for nodule in session.iter():
            nodule_tag = nodule.tag.split("}")[-1] if "}" in nodule.tag else nodule.tag
            if nodule_tag != "unblindedReadNodule":
                continue

            for roi in nodule.iter():
                roi_tag = roi.tag.split("}")[-1] if "}" in roi.tag else roi.tag
                if roi_tag != "roi":
                    continue

                z_pos = None
                edges = []
                for child in roi.iter():
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag == "imageZposition" and child.text:
                        try:
                            z_pos = float(child.text)
                        except ValueError:
                            pass
                    elif child_tag == "edgeMap":
                        x, y = None, None
                        for coord in child.iter():
                            coord_tag = coord.tag.split("}")[-1] if "}" in coord.tag else coord.tag
                            if coord_tag == "xCoord" and coord.text:
                                try:
                                    x = float(coord.text)
                                except ValueError:
                                    pass
                            elif coord_tag == "yCoord" and coord.text:
                                try:
                                    y = float(coord.text)
                                except ValueError:
                                    pass
                        if x is not None and y is not None:
                            edges.append((x, y))

                if z_pos is None or len(edges) < 3:
                    continue

                # Convert physical Z to slice index
                slice_idx = int(round((z_pos - ct_origin_z) / ct_spacing_z))
                if slice_idx < 0 or slice_idx >= num_slices:
                    continue

                contour = np.array(edges)
                try:
                    rr, cc = polygon(
                        contour[:, 1], contour[:, 0],
                        shape=volume_shape,
                    )
                    vote_volume[slice_idx, rr, cc] += 1
                except Exception:
                    continue

    max_votes = vote_volume.max()
    threshold = 2 if max_votes >= 2 else 1
    mask_volume = (vote_volume >= threshold).astype(np.uint8)
    return mask_volume


def process_lidc_dataset(
    lidc_root: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    max_patients: int = 100,
    context_slices: int = 1,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 1.0,
    seed: int = 42,
) -> Dict:
    """Extract and preprocess LIDC-IDRI from raw DICOM + XML annotations.

    Handles the washingtongold/lidcidri30 Kaggle dataset structure:
        lidcidri30/
          LIDC-IDRI-0001-0200/   (batch dirs with patient DICOM folders)
          LIDC-IDRI-0201-0400/
          LIDC-IDRI-0401-0600/
          LIDC-XML-only/          (XML nodule annotations)

    Pipeline per patient:
      1. Load 3D CT volume from DICOM series (SimpleITK)
      2. Match to XML annotations via SeriesInstanceUID
      3. Parse XML contours -> 3D binary mask (majority vote)
      4. Extract 2D slices with nodules (+context)
      5. HU windowing + CLAHE + resize + LSF computation
      6. Save as .npy files
    """
    import SimpleITK as sitk
    import pydicom

    lidc_path = Path(lidc_root)
    out_path = Path(output_dir)
    for split in ["train", "val", "test"]:
        os.makedirs(out_path / split / "images", exist_ok=True)
        os.makedirs(out_path / split / "masks", exist_ok=True)
        os.makedirs(out_path / split / "level_sets", exist_ok=True)

    # 1. Find batch directories and XML annotations
    candidate_batches = list(lidc_path.rglob("LIDC-IDRI-0001-0200"))
    if candidate_batches:
        actual_root = candidate_batches[0].parent
    else:
        actual_root = lidc_path

    batch_dirs = sorted([
        d for d in actual_root.iterdir()
        if d.is_dir() and d.name.startswith("LIDC-IDRI-") and d.name != "LIDC-XML-only"
    ])

    xml_dir = actual_root / "LIDC-XML-only"
    if not xml_dir.exists():
        xml_candidates = list(lidc_path.rglob("LIDC-XML-only"))
        if xml_candidates:
            xml_dir = xml_candidates[0]

    print(f"LIDC-IDRI Dataset Discovery:")
    print(f"  Search root: {lidc_path}")
    print(f"  Resolved root: {actual_root}")
    print(f"  Batch directories: {[d.name for d in batch_dirs]}")
    print(f"  XML annotations: {xml_dir}")

    if not batch_dirs:
        print(f"  ERROR: No LIDC-IDRI batch directories found!")
        print(f"  Contents of {lidc_root}:")
        for item in sorted(lidc_path.iterdir())[:15]:
            print(f"    {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        return {}

    # 2. Build XML UID index
    uid_to_xml = _build_xml_uid_index(str(xml_dir)) if xml_dir.exists() else {}

    # 3. Collect all patient directories
    all_patients = []
    for batch_dir in batch_dirs:
        for patient_dir in sorted(batch_dir.iterdir()):
            if patient_dir.is_dir() and patient_dir.name.startswith("LIDC-IDRI-"):
                all_patients.append(patient_dir)

    print(f"  Total patients found: {len(all_patients)}")

    # Subsample patients if requested
    if max_patients > 0 and len(all_patients) > max_patients:
        random.seed(seed)
        random.shuffle(all_patients)
        all_patients = sorted(all_patients[:max_patients], key=lambda p: p.name)
        print(f"  Subsampled to {len(all_patients)} patients.")

    # 4. Patient-level split
    patient_ids = [p.name for p in all_patients]
    train_val_ids, test_ids = train_test_split(patient_ids, test_size=test_ratio, random_state=seed)
    val_rel = val_ratio / (1.0 - test_ratio)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_rel, random_state=seed)

    split_lookup = {}
    for pid in train_ids:
        split_lookup[pid] = "train"
    for pid in val_ids:
        split_lookup[pid] = "val"
    for pid in test_ids:
        split_lookup[pid] = "test"

    print(f"  Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Semi-supervised labeling
    random.seed(seed)
    shuffled_train = list(train_ids)
    random.shuffle(shuffled_train)
    num_labeled = int(round(len(shuffled_train) * labeled_ratio))
    labeled_set = set(shuffled_train[:num_labeled])

    counts = {"train": 0, "val": 0, "test": 0}
    labeled_counts = {"train_labeled": 0, "train_unlabeled": 0}
    skipped = {"no_ct": 0, "no_xml": 0, "no_nodules": 0, "error": 0}

    # 5. Process each patient
    patient_to_dir = {p.name: p for p in all_patients}

    for patient_id in tqdm(patient_ids, desc="Processing LIDC-IDRI patients"):
        patient_dir = patient_to_dir[patient_id]
        split_name = split_lookup[patient_id]
        is_labeled = (split_name != "train") or (patient_id in labeled_set)

        try:
            # 5a. Find and load CT volume
            ct_folder = _find_ct_series_folder(patient_dir)
            if ct_folder is None:
                skipped["no_ct"] += 1
                continue

            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(ct_folder)
            if not series_ids:
                skipped["no_ct"] += 1
                continue

            series_files = reader.GetGDCMSeriesFileNames(ct_folder, series_ids[0])
            reader.SetFileNames(series_files)
            ct_image = reader.Execute()
            ct_array = sitk.GetArrayFromImage(ct_image)  # (Z, H, W) in HU

            origin_z = ct_image.GetOrigin()[2]
            spacing_z = ct_image.GetSpacing()[2]
            num_slices = ct_array.shape[0]
            h, w = ct_array.shape[1], ct_array.shape[2]

            # 5b. Get SeriesInstanceUID and find matching XML
            dcm_sample = pydicom.dcmread(series_files[0], stop_before_pixels=True)
            series_uid = str(dcm_sample.SeriesInstanceUID)

            xml_path = uid_to_xml.get(series_uid)
            if xml_path is None:
                skipped["no_xml"] += 1
                continue

            # 5c. Extract nodule masks from XML
            mask_volume = _extract_nodule_masks_from_xml(
                xml_path, origin_z, spacing_z, num_slices, (h, w)
            )

            # 5d. Find slices with nodules and extract with context
            lesion_slices = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
            if len(lesion_slices) == 0:
                skipped["no_nodules"] += 1
                continue

            selected_indices = set()
            for s in lesion_slices:
                for offset in range(-context_slices, context_slices + 1):
                    idx = s + offset
                    if 0 <= idx < num_slices:
                        selected_indices.add(int(idx))

            # 5e. Preprocess and save each selected slice
            for idx in sorted(selected_indices):
                ct_slice = ct_array[idx].astype(np.float32)
                mask_slice = mask_volume[idx]

                # HU windowing + CLAHE
                ct_norm = normalize_ct_slice(ct_slice)
                ct_resized = resize_slice(ct_norm, target_size=target_size, is_mask=False)

                mask_resized = resize_slice(mask_slice, target_size=target_size, is_mask=True)
                lsf_map = compute_level_set(mask_resized)

                stem = f"lidc_{patient_id}_s{idx}"
                np.save(out_path / split_name / "images" / f"{stem}.npy", ct_resized)
                np.save(out_path / split_name / "masks" / f"{stem}.npy", mask_resized)
                np.save(out_path / split_name / "level_sets" / f"{stem}.npy", lsf_map)

                counts[split_name] += 1
                if split_name == "train":
                    if is_labeled:
                        labeled_counts["train_labeled"] += 1
                    else:
                        labeled_counts["train_unlabeled"] += 1

        except Exception as e:
            skipped["error"] += 1
            continue

    manifest = {
        "dataset": "LIDC-IDRI",
        "target_size": list(target_size),
        "total_extracted_slices": sum(counts.values()),
        "splits": counts,
        "semi_supervised_breakdown": labeled_counts,
        "skipped": skipped,
        "labeled_ratio": labeled_ratio,
        "seed": seed,
    }

    # Save LIDC manifest
    with open(out_path / "lidc_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nLIDC-IDRI preprocessing complete!")
    print(json.dumps(manifest, indent=2))
    if sum(skipped.values()) > 0:
        print(f"Skipped patients: {skipped}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT datasets for DTCR-U-Net")
    parser.add_argument("--mosmed", default=None, help="Path to MosMed dataset directory")
    parser.add_argument("--lidc", default=None, help="Path to LIDC-IDRI dataset directory (raw DICOM + XML)")
    parser.add_argument("--out", default="data/processed", help="Output directory for processed .npy files")
    parser.add_argument("--size", type=int, default=256, help="Target image size (e.g. 256 for 256x256)")
    parser.add_argument("--context", type=int, default=1, help="Number of adjacent context slices (+/-)")
    parser.add_argument("--labeled-ratio", type=float, default=1.0, help="Fraction of labeled samples (0.4 for semi-supervised)")
    parser.add_argument("--max-patients", type=int, default=100, help="Max LIDC patients to process (default: 100)")
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
            max_patients=args.max_patients,
            context_slices=args.context,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()