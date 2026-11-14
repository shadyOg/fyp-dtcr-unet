"""Data Preprocessing Pipeline for DTCR-U-Net (paper Section 4.1).

For every annotated CT volume:
  1. Load it and bring it to one canonical orientation (LPS), so MosMedData,
     LIDC-IDRI and COVID-19-CT-Seg slices all look the same way up.
  2. Keep each slice that contains annotated lesion plus its `context` adjacent
     slices above and below.
  3. Window to [0, 1], contrast-enhance (CLAHE), resize to 256 x 256.
  4. Split at patient level 6:2:2 into train / val / test.

Which training patients are treated as unlabeled is decided at training time by
`data.datasets.DualTaskDataset` (per dataset, per patient, seeded), not here.
"""

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:  # imported as data.preprocessing
    from .level_set import compute_level_set
except ImportError:  # run as a script: python data/preprocessing.py
    from level_set import compute_level_set


# --------------------------------------------------------------------------- #
# Slice-level helpers
# --------------------------------------------------------------------------- #

def normalize_ct_slice(
    slice_2d: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    apply_clahe: bool = True,
    is_hu: bool = True,
) -> np.ndarray:
    """Scale a CT slice to [0, 1] and apply CLAHE contrast enhancement.

    Args:
        slice_2d: 2D CT slice, in Hounsfield Units unless `is_hu` is False.
        hu_min, hu_max: Intensity window used when the slice is in HU.
        apply_clahe: Apply contrast-limited adaptive histogram equalization.
        is_hu: False for volumes already exported as 8-bit [0, 255] images
            (the radiopaedia cases of COVID-19-CT-Seg); those are only divided by 255.
    """
    if is_hu:
        clipped = np.clip(slice_2d, hu_min, hu_max)
        normalized = (clipped - hu_min) / (hu_max - hu_min)
    else:
        normalized = np.clip(slice_2d, 0.0, 255.0) / 255.0

    if apply_clahe:
        img_uint8 = np.round(normalized * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(img_uint8).astype(np.float32) / 255.0

    return normalized.astype(np.float32)


def resize_slice(
    arr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    is_mask: bool = False,
) -> np.ndarray:
    """Resize a slice to `target_size` = (H, W).

    Downsampling uses area averaging. For masks the averaged coverage is
    thresholded at 0.5, which keeps lesion area unbiased (nearest-neighbour
    sampling randomly drops or keeps thin structures).
    """
    h, w = target_size
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr.astype(np.uint8) if is_mask else arr

    shrinking = arr.shape[0] > h or arr.shape[1] > w
    if is_mask:
        if shrinking:
            cover = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            return (cover >= 0.5).astype(np.uint8)
        return cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=interp)


def volume_is_hu(ct_zyx: np.ndarray) -> bool:
    """False when a volume was exported as an 8-bit image instead of Hounsfield Units."""
    return not (float(ct_zyx.min()) >= 0.0 and float(ct_zyx.max()) <= 255.0)


def select_slices(mask_zyx: np.ndarray, context_slices: int) -> List[int]:
    """Indices of slices with annotated lesion plus `context_slices` neighbours each side."""
    num_slices = mask_zyx.shape[0]
    lesion_slices = np.where(mask_zyx.reshape(num_slices, -1).any(axis=1))[0]
    selected = set()
    for s in lesion_slices:
        for offset in range(-context_slices, context_slices + 1):
            idx = int(s) + offset
            if 0 <= idx < num_slices:
                selected.add(idx)
    return sorted(selected)


# --------------------------------------------------------------------------- #
# Volume loading in one canonical orientation
# --------------------------------------------------------------------------- #

def _oriented_arrays(ct_img, mask_img) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ct, mask) as (Z, Y, X) arrays in LPS orientation.

    In LPS, rows run anterior -> posterior and columns run patient-right ->
    patient-left: the standard radiological view. Without this, MosMedData slices
    were fed transposed relative to LIDC-IDRI ones.
    """
    import SimpleITK as sitk

    if mask_img.GetSize() == ct_img.GetSize():
        # Masks occasionally carry a slightly different header than their CT.
        mask_img.CopyInformation(ct_img)
    try:
        ct_img = sitk.DICOMOrient(ct_img, "LPS")
        mask_img = sitk.DICOMOrient(mask_img, "LPS")
    except Exception as e:  # keep the native layout rather than lose the case
        print(f"  Warning: could not reorient volume ({e}); using native orientation.")

    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    return ct, mask


def load_nifti_pair(ct_path: str, mask_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a CT volume and its lesion mask from NIfTI as LPS-oriented (Z, Y, X) arrays."""
    import SimpleITK as sitk

    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))
    if ct_img.GetDimension() != 3 or mask_img.GetDimension() != 3:
        print(f"  Skipping {Path(ct_path).name}: not a 3D volume.")
        return None
    if ct_img.GetSize() != mask_img.GetSize():
        print(f"  Skipping {Path(ct_path).name}: CT {ct_img.GetSize()} and mask {mask_img.GetSize()} differ in size.")
        return None
    return _oriented_arrays(ct_img, mask_img)


# --------------------------------------------------------------------------- #
# Splitting and saving
# --------------------------------------------------------------------------- #

def split_cases(
    case_ids: List[str], test_ratio: float = 0.2, val_ratio: float = 0.2, seed: int = 42
) -> Dict[str, str]:
    """Patient-level 6:2:2 split. Returns {case_id: 'train' | 'val' | 'test'}."""
    case_ids = sorted(case_ids)
    train_val, test = train_test_split(case_ids, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio / (1.0 - test_ratio), random_state=seed)
    lookup = {c: "train" for c in train}
    lookup.update({c: "val" for c in val})
    lookup.update({c: "test" for c in test})
    return lookup


def make_output_dirs(out_path: Path) -> None:
    for split in ("train", "val", "test"):
        for sub in ("images", "masks", "level_sets"):
            os.makedirs(out_path / split / sub, exist_ok=True)


def save_case_slices(
    ct_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    slice_indices: List[int],
    stem_prefix: str,
    split_dir: Path,
    target_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Preprocess and save the selected slices of one case.

    Returns:
        (number of slices saved, number of those that contain lesion pixels).
    """
    is_hu = volume_is_hu(ct_zyx)
    saved, with_lesion = 0, 0
    for idx in slice_indices:
        ct_resized = resize_slice(normalize_ct_slice(ct_zyx[idx], is_hu=is_hu), target_size, is_mask=False)
        mask_resized = resize_slice(mask_zyx[idx], target_size, is_mask=True)
        lsf_map = compute_level_set(mask_resized)

        stem = f"{stem_prefix}_s{idx}"
        np.save(split_dir / "images" / f"{stem}.npy", ct_resized.astype(np.float32))
        np.save(split_dir / "masks" / f"{stem}.npy", mask_resized.astype(np.uint8))
        np.save(split_dir / "level_sets" / f"{stem}.npy", lsf_map)
        saved += 1
        with_lesion += int(mask_resized.any())
    return saved, with_lesion


def _process_nifti_dataset(
    name: str,
    stem_prefix: str,
    pairs: Dict[str, Tuple[str, str]],
    output_dir: str,
    target_size: Tuple[int, int],
    context_slices: int,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    manifest_name: str,
) -> Dict:
    """Shared driver for datasets stored as paired NIfTI CT / mask volumes."""
    out_path = Path(output_dir)
    make_output_dirs(out_path)

    split_lookup = split_cases(list(pairs.keys()), test_ratio, val_ratio, seed)
    counts = {"train": 0, "val": 0, "test": 0}
    lesion_counts = {"train": 0, "val": 0, "test": 0}
    case_counts = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for case_id in tqdm(sorted(pairs.keys()), desc=f"Processing {name}"):
        ct_path, mask_path = pairs[case_id]
        loaded = load_nifti_pair(ct_path, mask_path)
        if loaded is None:
            skipped += 1
            continue
        ct_zyx, mask_zyx = loaded
        indices = select_slices(mask_zyx, context_slices)
        if not indices:
            skipped += 1
            continue

        split_name = split_lookup[case_id]
        saved, with_lesion = save_case_slices(
            ct_zyx, mask_zyx, indices, f"{stem_prefix}_{case_id}", out_path / split_name, target_size
        )
        counts[split_name] += saved
        lesion_counts[split_name] += with_lesion
        case_counts[split_name] += 1

    manifest = {
        "dataset": name,
        "target_size": list(target_size),
        "context_slices": context_slices,
        "cases": case_counts,
        "slices": counts,
        "slices_with_lesion": lesion_counts,
        "skipped_cases": skipped,
        "seed": seed,
    }
    with open(out_path / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{name} preprocessing complete!")
    print(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# MosMedData
# --------------------------------------------------------------------------- #

def find_mosmed_files(root_dir: str) -> Tuple[List[str], List[str]]:
    """Recursively discover CT volumes and paired mask files in any directory layout."""
    all_nii_files = []
    for dirpath, _, filenames in os.walk(Path(root_dir)):
        for f in filenames:
            if f.endswith((".nii", ".nii.gz")):
                all_nii_files.append(os.path.join(dirpath, f))

    mask_files = [f for f in all_nii_files if "_mask" in os.path.basename(f).lower()]
    ct_files = [f for f in all_nii_files if "_mask" not in os.path.basename(f).lower()]

    ct_map = {}
    for c in ct_files:
        match = re.search(r"(study_\d+)", os.path.basename(c), re.IGNORECASE)
        if match:
            ct_map[match.group(1).lower()] = c

    paired_studies, paired_masks = [], []
    for m in sorted(mask_files):
        match = re.search(r"(study_\d+)", os.path.basename(m), re.IGNORECASE)
        if match and match.group(1).lower() in ct_map:
            paired_studies.append(ct_map[match.group(1).lower()])
            paired_masks.append(m)

    return paired_studies, paired_masks


def process_mosmed_dataset(
    mosmed_root: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    context_slices: int = 2,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 1.0,  # unused: the labeled subset is chosen at training time
    seed: int = 42,
) -> Dict:
    """Preprocess the 50 annotated MosMedData studies with a patient-level 6:2:2 split."""
    ct_files, mask_files = find_mosmed_files(mosmed_root)
    print(f"Found {len(ct_files)} paired MosMed volumes.")
    if not ct_files:
        raise FileNotFoundError(f"No paired MosMed studies/masks found in {mosmed_root}")

    pairs = {}
    for c, m in zip(ct_files, mask_files):
        study_id = re.search(r"(study_\d+)", os.path.basename(c), re.IGNORECASE).group(1).lower()
        pairs[study_id] = (c, m)

    return _process_nifti_dataset(
        "MosMedData", "mosmed", pairs, output_dir, target_size, context_slices,
        test_ratio, val_ratio, seed, "manifest.json",
    )


# --------------------------------------------------------------------------- #
# COVID-19-CT-Seg
# --------------------------------------------------------------------------- #

def _resolve_listed_file(root: Path, raw) -> Optional[Path]:
    """Locate a file listed in metadata.csv under `root`.

    COVID-19-CT-Seg stores a file with the SAME name in infection_mask/,
    lung_mask/ and lung_and_infection_mask/, so the match must also agree on the
    parent folder; taking the first file with that name can silently return a
    lung mask instead of the infection mask.
    """
    raw_path = Path(str(raw).replace("\\", "/"))
    matches = [c for c in root.rglob(raw_path.name) if c.parent.name == raw_path.parent.name]
    return sorted(matches)[0] if matches else None


def find_covid19_files(covid_root: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Pair CT volumes with INFECTION masks in COVID-19-CT-Seg (metadata.csv or folder scan)."""
    covid_path = Path(covid_root)

    csv_candidates = sorted(covid_path.rglob("metadata.csv"))
    if csv_candidates:
        try:
            import pandas as pd
            df = pd.read_csv(csv_candidates[0])
            paired_ct, paired_masks = [], []
            for _, row in df.iterrows():
                ct_p, mask_p = row.get("ct_scan"), row.get("infection_mask")
                if pd.isna(ct_p) or pd.isna(mask_p):
                    continue
                p_ct = _resolve_listed_file(covid_path, ct_p)
                p_mask = _resolve_listed_file(covid_path, mask_p)
                if p_ct is not None and p_mask is not None:
                    paired_ct.append(str(p_ct))
                    paired_masks.append(str(p_mask))
            if paired_ct:
                return paired_ct, paired_masks
        except Exception as e:
            print(f"Warning: Failed reading metadata.csv ({e}), falling back to folder scan.")

    # Folder scan: CT and infection mask share a file name in sibling folders.
    all_niis = sorted(covid_path.rglob("*.nii*"))
    infection = {p.name: p for p in all_niis if p.parent.name.lower() == "infection_mask"}
    ct_files, mask_files = [], []
    for p in all_niis:
        if "mask" in p.parent.name.lower():
            continue
        if p.name in infection:
            ct_files.append(str(p))
            mask_files.append(str(infection[p.name]))
    return ct_files, mask_files


def process_covid19_dataset(
    covid_root: str,
    output_dir: str,
    target_size: Tuple[int, int] = (256, 256),
    context_slices: int = 2,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 0.4,  # unused: the labeled subset is chosen at training time
    seed: int = 42,
) -> Dict:
    """Preprocess COVID-19-CT-Seg (infection masks) with a patient-level split."""
    ct_files, mask_files = find_covid19_files(covid_root)
    print(f"Found {len(ct_files)} paired COVID-19-CT-Seg volumes.")
    if not ct_files:
        raise FileNotFoundError(f"No paired COVID-19 CT scans and infection masks found in {covid_root}")

    pairs = {}
    for c, m in zip(ct_files, mask_files):
        case_id = Path(c).name.replace(".nii.gz", "").replace(".nii", "")
        pairs[case_id] = (c, m)

    return _process_nifti_dataset(
        "COVID-19-CT-Seg", "covid19", pairs, output_dir, target_size, context_slices,
        test_ratio, val_ratio, seed, "covid19_manifest.json",
    )


# --------------------------------------------------------------------------- #
# LIDC-IDRI (raw DICOM + XML annotations)
# --------------------------------------------------------------------------- #

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
    context_slices: int = 2,
    test_ratio: float = 0.2,
    val_ratio: float = 0.2,
    labeled_ratio: float = 1.0,  # unused: the labeled subset is chosen at training time
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
      1. Load the 3D CT volume from its DICOM series (SimpleITK)
      2. Match it to its XML annotation via SeriesInstanceUID
      3. Rasterize radiologist contours into a 3D mask (majority vote)
      4. Reorient to LPS, keep nodule slices (+context), window, CLAHE, resize, save

    `max_patients` counts patients that actually yield annotated nodules, so the
    requested number of usable cases is reached whenever the data allows it.
    """
    import SimpleITK as sitk
    import pydicom

    lidc_path = Path(lidc_root)
    out_path = Path(output_dir)
    make_output_dirs(out_path)

    # Prefer the given folder when it already holds the LIDC-IDRI-* folders; a recursive
    # search has to walk every DICOM file of the dataset.
    if any(d.is_dir() and d.name.startswith("LIDC-IDRI-") for d in lidc_path.iterdir()):
        actual_root = lidc_path
    else:
        candidate_batches = list(lidc_path.rglob("LIDC-IDRI-0001-0200"))
        actual_root = candidate_batches[0].parent if candidate_batches else lidc_path

    batch_dirs = sorted([
        d for d in actual_root.iterdir()
        if d.is_dir() and d.name.startswith("LIDC-IDRI-") and d.name != "LIDC-XML-only"
    ])

    xml_dir = actual_root / "LIDC-XML-only"
    if not xml_dir.exists():
        xml_candidates = list(lidc_path.rglob("LIDC-XML-only"))
        if xml_candidates:
            xml_dir = xml_candidates[0]

    print("LIDC-IDRI Dataset Discovery:")
    print(f"  Search root: {lidc_path}")
    print(f"  Resolved root: {actual_root}")
    print(f"  Batch directories: {[d.name for d in batch_dirs]}")
    print(f"  XML annotations: {xml_dir}")

    if not batch_dirs:
        print("  ERROR: No LIDC-IDRI batch directories found!")
        print(f"  Contents of {lidc_root}:")
        for item in sorted(lidc_path.iterdir())[:15]:
            print(f"    {'[DIR]' if item.is_dir() else '[FILE]'} {item.name}")
        return {}

    uid_to_xml = _build_xml_uid_index(str(xml_dir)) if xml_dir.exists() else {}

    all_patients = []
    for batch_dir in batch_dirs:
        # Batch folders either hold patient folders directly or are patient folders themselves.
        children = [p for p in sorted(batch_dir.iterdir()) if p.is_dir() and p.name.startswith("LIDC-IDRI-")]
        all_patients.extend(children if children else [batch_dir])
    print(f"  Total patients found: {len(all_patients)}")

    # Visit patients in a seeded random order; stop once enough usable ones are collected.
    rng = random.Random(seed)
    order = sorted(all_patients, key=lambda p: p.name)
    rng.shuffle(order)

    skipped = {"no_ct": 0, "no_xml": 0, "no_nodules": 0, "error": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    lesion_counts = {"train": 0, "val": 0, "test": 0}
    case_counts = {"train": 0, "val": 0, "test": 0}
    kept_ids: List[str] = []
    saved_by_patient: Dict[str, Tuple[int, int]] = {}
    staging_root = out_path / "_lidc_staging"
    shutil.rmtree(staging_root, ignore_errors=True)
    reader = sitk.ImageSeriesReader()

    pbar = tqdm(order, desc="Processing LIDC-IDRI patients")
    for patient_dir in pbar:
        if max_patients > 0 and len(kept_ids) >= max_patients:
            break
        patient_id = patient_dir.name
        try:
            ct_folder = _find_ct_series_folder(patient_dir)
            if ct_folder is None:
                skipped["no_ct"] += 1
                continue
            series_ids = reader.GetGDCMSeriesIDs(ct_folder)
            if not series_ids:
                skipped["no_ct"] += 1
                continue
            series_files = reader.GetGDCMSeriesFileNames(ct_folder, series_ids[0])
            series_uid = str(pydicom.dcmread(series_files[0], stop_before_pixels=True).SeriesInstanceUID)
            xml_path = uid_to_xml.get(series_uid)
            if xml_path is None:
                skipped["no_xml"] += 1
                continue

            reader.SetFileNames(series_files)
            ct_image = reader.Execute()
            size_x, size_y, num_slices = ct_image.GetSize()

            # Physical z -> slice index, following the stacking direction of the series.
            z_step = ct_image.GetSpacing()[2] * ct_image.GetDirection()[8]
            if abs(z_step) < 1e-6:
                skipped["error"] += 1
                continue
            mask_native = _extract_nodule_masks_from_xml(
                xml_path, ct_image.GetOrigin()[2], z_step, num_slices, (size_y, size_x)
            )
            if not mask_native.any():
                skipped["no_nodules"] += 1
                continue

            ct_zyx, mask_zyx = _oriented_arrays(ct_image, sitk.GetImageFromArray(mask_native))
            indices = select_slices(mask_zyx, context_slices)
            if not indices:
                skipped["no_nodules"] += 1
                continue

            # Stage first: the 6:2:2 split is drawn over the patients that turned out usable.
            staging = staging_root / patient_id
            for sub in ("images", "masks", "level_sets"):
                os.makedirs(staging / sub, exist_ok=True)
            saved_by_patient[patient_id] = save_case_slices(
                ct_zyx, mask_zyx, indices, f"lidc_{patient_id}", staging, target_size
            )
            kept_ids.append(patient_id)
            pbar.set_postfix({"usable": len(kept_ids)})
        except Exception as e:
            print(f"  Error on {patient_id}: {e}")
            skipped["error"] += 1

    if not kept_ids:
        print("  ERROR: no LIDC-IDRI patient produced annotated nodule slices.")
        return {}

    split_lookup = split_cases(kept_ids, test_ratio, val_ratio, seed)
    for patient_id in kept_ids:
        split_name = split_lookup[patient_id]
        staging = staging_root / patient_id
        for sub in ("images", "masks", "level_sets"):
            for f in (staging / sub).iterdir():
                shutil.move(str(f), str(out_path / split_name / sub / f.name))
        saved, with_lesion = saved_by_patient[patient_id]
        counts[split_name] += saved
        lesion_counts[split_name] += with_lesion
        case_counts[split_name] += 1
    shutil.rmtree(staging_root, ignore_errors=True)

    manifest = {
        "dataset": "LIDC-IDRI",
        "target_size": list(target_size),
        "context_slices": context_slices,
        "cases": case_counts,
        "slices": counts,
        "slices_with_lesion": lesion_counts,
        "skipped": skipped,
        "seed": seed,
    }
    with open(out_path / "lidc_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nLIDC-IDRI preprocessing complete!")
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT datasets for DTCR-U-Net")
    parser.add_argument("--mosmed", default=None, help="Path to MosMed dataset directory")
    parser.add_argument("--lidc", default=None, help="Path to LIDC-IDRI dataset directory (raw DICOM + XML)")
    parser.add_argument("--covid19", default=None, help="Path to COVID-19-CT-Seg dataset directory")
    parser.add_argument("--out", default="data/processed", help="Output directory for processed .npy files")
    parser.add_argument("--size", type=int, default=256, help="Target image size (256 per Section 4.1)")
    parser.add_argument("--context", type=int, default=2, help="Adjacent slices kept above and below each lesion slice (Section 4.1)")
    parser.add_argument("--labeled-ratio", type=float, default=None, help="Ignored (kept for old commands): pass --labeled-ratio to train.py instead")
    parser.add_argument("--max-patients", type=int, default=100, help="Number of usable LIDC patients to collect (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.mosmed is None and args.lidc is None and args.covid19 is None:
        parser.error("At least one of --mosmed, --lidc, or --covid19 must be provided.")

    size = (args.size, args.size)
    if args.mosmed:
        process_mosmed_dataset(args.mosmed, args.out, target_size=size, context_slices=args.context, seed=args.seed)
    if args.lidc:
        process_lidc_dataset(args.lidc, args.out, target_size=size, max_patients=args.max_patients,
                             context_slices=args.context, seed=args.seed)
    if args.covid19:
        process_covid19_dataset(args.covid19, args.out, target_size=size, context_slices=args.context, seed=args.seed)


if __name__ == "__main__":
    main()
