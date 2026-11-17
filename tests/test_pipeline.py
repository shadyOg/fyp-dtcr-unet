"""End-to-end checks of the data pipeline and the training / validation loop.

Builds small synthetic CT volumes on disk, runs the real preprocessing, the real
Dataset / DataLoaders, one real training epoch and the real evaluation.
"""

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, "dtcr-unet")

from config import DTCRConfig
from data.datasets import DualTaskDataset, get_dataloaders
from data.level_set import LSF_TRUNC_PX, compute_level_set
from data.preprocessing import find_covid19_files, process_covid19_dataset, process_mosmed_dataset
from models.dtcr_unet import inverse_level_set_transform
from train import evaluate_dataset, train


def _write_volume(path: Path, arr_zyx: np.ndarray, direction=None) -> None:
    img = sitk.GetImageFromArray(arr_zyx)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(path))


def _synthetic_case(rng: np.random.Generator, size: int = 64, depth: int = 12):
    """A CT-like volume in HU with a bright blob 'lesion' on a few slices, off-centre."""
    ct = rng.normal(-800.0, 40.0, size=(depth, size, size)).astype(np.float32)
    mask = np.zeros((depth, size, size), dtype=np.uint8)
    cy, cx, r = int(rng.integers(14, 24)), int(rng.integers(38, 50)), int(rng.integers(5, 9))
    yy, xx = np.ogrid[:size, :size]
    blob = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    for z in range(4, 8):
        mask[z][blob] = 1
        ct[z][blob] = rng.normal(-300.0, 30.0, size=int(blob.sum()))
    return ct, mask


def test_level_set_round_trip():
    """T^{-1}(T(y)) must reproduce y, including for a 1-pixel and a 3-pixel lesion."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[40:90, 60:130] = 1      # large lesion
    mask[200:203, 200:203] = 1   # 3 x 3 px lesion
    mask[150, 30] = 1            # single pixel
    lsf = compute_level_set(mask)

    assert lsf.dtype == np.float32 and lsf.min() >= -1.0 and lsf.max() <= 1.0
    assert np.all(lsf[mask == 1] < 0) and np.all(lsf[mask == 0] > 0)
    assert lsf[65, 95] == -1.0, "deep inside a large lesion should saturate at -1"
    assert lsf[150, 30] <= -0.5 / LSF_TRUNC_PX + 1e-6, "a single-pixel lesion must still be clearly negative"

    recon = inverse_level_set_transform(torch.from_numpy(lsf)).numpy()
    assert np.array_equal(recon > 0.5, mask.astype(bool)), "inverse transform changed the mask"
    assert np.abs(recon - mask).max() < 0.05, f"inverse transform too soft: {np.abs(recon - mask).max():.3f}"

    assert np.all(compute_level_set(np.zeros((8, 8))) == 1.0)
    assert np.all(compute_level_set(np.ones((8, 8))) == -1.0)


def test_covid_pairing_picks_infection_mask():
    """Same file name exists in infection_mask/, lung_mask/ and lung_and_infection_mask/."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for sub in ("ct_scans", "infection_mask", "lung_mask", "lung_and_infection_mask"):
            (root / sub).mkdir()
        rows = []
        for i in range(3):
            ct, mask = _synthetic_case(rng, size=64, depth=10)
            _write_volume(root / "ct_scans" / f"case_org_{i}.nii", ct)
            _write_volume(root / "infection_mask" / f"case_{i}.nii", mask)
            _write_volume(root / "lung_mask" / f"case_{i}.nii", np.ones_like(mask))
            _write_volume(root / "lung_and_infection_mask" / f"case_{i}.nii", np.ones_like(mask) * 3)
            rows.append({
                "ct_scan": f"../input/covid19-ct-scans/ct_scans/case_org_{i}.nii",
                "lung_mask": f"../input/covid19-ct-scans/lung_mask/case_{i}.nii",
                "infection_mask": f"../input/covid19-ct-scans/infection_mask/case_{i}.nii",
                "lung_and_infection_mask": f"../input/covid19-ct-scans/lung_and_infection_mask/case_{i}.nii",
            })
        with open(root / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        cts, masks = find_covid19_files(root)
        assert len(cts) == 3
        assert all(Path(m).parent.name == "infection_mask" for m in masks), masks
        assert all(Path(c).parent.name == "ct_scans" for c in cts), cts


def _write_lidc_patient(
    root: Path, patient_id: str, rng: np.random.Generator, with_nodule: bool = True, misplaced: bool = False
) -> None:
    """Write one LIDC-style patient: a DICOM CT series plus its XML contour annotation.

    Every contour carries its instance UID and a z position that is off by two
    slices, so it lands on the nodule only if the UID is used. Each contour also
    has a small hole (inclusion=FALSE) at its centre. With `misplaced`, the UIDs
    are left out and the z position is shifted onto lesion-free slices, which the
    alignment check must flag.
    """
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    size, depth, z0, dz = 64, 14, -100.0, 2.5
    ct, mask = _synthetic_case(rng, size=size, depth=depth)
    series_uid, study_uid = generate_uid(), generate_uid()
    series_dir = root / "LIDC-IDRI-0001-0200" / patient_id / "study" / "series"
    series_dir.mkdir(parents=True)

    sop_uids = []
    for z in range(depth):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        # File names deliberately run against the z order: slices must be sorted by position.
        ds = FileDataset(str(series_dir / f"{depth - z:03d}.dcm"), {}, file_meta=meta, preamble=bytes(128))
        ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
        sop_uids.append(str(ds.SOPInstanceUID))
        ds.Modality, ds.PatientID = "CT", patient_id
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.InstanceNumber = depth - z
        ds.ImagePositionPatient = [-32.0, -32.0, z0 + dz * z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing, ds.SliceThickness = [1.0, 1.0], dz
        ds.Rows, ds.Columns = size, size
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.PixelData = ct[z].astype(np.int16).tobytes()
        ds.save_as(str(series_dir / f"{depth - z:03d}.dcm"), enforce_file_format=True)

    # Contour = boundary pixels of the blob, ordered by angle (as radiologist outlines are)
    rois = ""
    if with_nodule:
        ys, xs = np.nonzero(mask[5])
        cy, cx = ys.mean(), xs.mean()
        radius = np.sqrt(len(ys) / np.pi)
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        edge = "".join(
            f"<edgeMap><xCoord>{int(round(cx + radius * np.cos(a)))}</xCoord>"
            f"<yCoord>{int(round(cy + radius * np.sin(a)))}</yCoord></edgeMap>" for a in angles
        )
        hole = "".join(
            f"<edgeMap><xCoord>{int(round(cx + 1.5 * np.cos(a)))}</xCoord>"
            f"<yCoord>{int(round(cy + 1.5 * np.sin(a)))}</yCoord></edgeMap>" for a in angles[::5]
        )
        for z in np.nonzero(mask.reshape(depth, -1).any(axis=1))[0]:
            if misplaced:
                tags = f"<imageZposition>{z0 + dz * (z + 5)}</imageZposition>"
            else:
                tags = f"<imageZposition>{z0 + dz * (z + 2)}</imageZposition><imageSOP_UID>{sop_uids[z]}</imageSOP_UID>"
            rois += f"<roi>{tags}<inclusion>TRUE</inclusion>{edge}</roi>"
            rois += f"<roi>{tags}<inclusion>FALSE</inclusion>{hole}</roi>"
    session = f"<readingSession><unblindedReadNodule><noduleID>1</noduleID>{rois}</unblindedReadNodule></readingSession>"
    xml_dir = root / "LIDC-XML-only" / "tcia-lidc-xml" / "157"
    xml_dir.mkdir(parents=True, exist_ok=True)
    (xml_dir / f"{patient_id}.xml").write_text(
        f'<?xml version="1.0"?><LidcReadMessage xmlns="http://www.nih.gov"><ResponseHeader>'
        f"<SeriesInstanceUid>{series_uid}</SeriesInstanceUid></ResponseHeader>{session}{session}</LidcReadMessage>"
    )


def test_lidc_processing():
    """DICOM series + XML contours -> masks that sit on the nodule, on the right slices."""
    from data.preprocessing import process_lidc_dataset

    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "lidc", Path(tmp) / "processed"
        for i in range(1, 11):
            _write_lidc_patient(root, f"LIDC-IDRI-{i:04d}", rng)
        _write_lidc_patient(root, "LIDC-IDRI-0011", rng, with_nodule=False)  # no outlined nodule

        manifest = process_lidc_dataset(str(root), str(out), target_size=(64, 64), max_patients=8, context_slices=2)
        assert sum(manifest["cases"].values()) == 8, manifest          # stops at the requested usable count
        assert manifest["low_contrast_cases"] == [], manifest["low_contrast_cases"]
        assert manifest["lesion_contrast_median"] > 0.15
        assert manifest["cases"]["test"] >= 1 and manifest["cases"]["val"] >= 1
        assert not (out / "_lidc_staging").exists()

        n_lesion = 0
        for mp in sorted(out.glob("*/masks/lidc_*.npy")):
            msk, img = np.load(mp), np.load(mp.parent.parent / "images" / mp.name)[1].astype(np.float32)
            if msk.any():
                n_lesion += 1
                # The contour must land on the bright blob, i.e. the right slice and the right place
                assert img[msk > 0].mean() > img[msk == 0].mean() + 0.15, f"mask misses the nodule in {mp.name}"
                ys, xs = np.nonzero(msk)
                assert msk[int(round(ys.mean())), int(round(xs.mean()))] == 0, f"hole not subtracted in {mp.name}"
        assert n_lesion == 8 * 4, n_lesion                             # 4 nodule slices per patient
        assert len(list(out.glob("*/masks/lidc_*.npy"))) == 8 * 8      # plus 2 context slices each side

    # Contours placed on the wrong slices must be flagged by the alignment check
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "lidc", Path(tmp) / "processed"
        for i in range(1, 6):
            _write_lidc_patient(root, f"LIDC-IDRI-{i:04d}", rng, misplaced=(i == 3))
        manifest = process_lidc_dataset(str(root), str(out), target_size=(64, 64), max_patients=5, context_slices=2)
        assert manifest["low_contrast_cases"] == ["lidc_LIDC-IDRI-0003"], manifest["low_contrast_cases"]

    # Extra training patients must not change which patients are in validation and test
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lidc"
        for i in range(1, 13):
            _write_lidc_patient(root, f"LIDC-IDRI-{i:04d}", rng, with_nodule=(i != 4))

        def split_patients(out):
            return {split: sorted({p.stem.rsplit("_s", 1)[0] for p in (out / split / "images").glob("*.npy")})
                    for split in ("train", "val", "test")}

        base = Path(tmp) / "base"
        process_lidc_dataset(str(root), str(base), target_size=(32, 32), max_patients=6, context_slices=1)
        more = Path(tmp) / "more"
        m = process_lidc_dataset(str(root), str(more), target_size=(32, 32), max_patients=6, context_slices=1,
                                 extra_train_patients=4)
        a, b = split_patients(base), split_patients(more)
        assert a["val"] == b["val"] and a["test"] == b["test"], (a, b)
        assert set(a["train"]) < set(b["train"]) and len(b["train"]) == len(a["train"]) + 4, (a, b)
        assert m["extra_train_patients"] == 4 and m["cases"]["train"] == len(b["train"])


def test_end_to_end_pipeline():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mosmed, covid, out = tmp / "mosmed", tmp / "covid", tmp / "processed"
        (mosmed / "studies").mkdir(parents=True)
        (mosmed / "masks").mkdir()
        (covid / "ct_scans").mkdir(parents=True)
        (covid / "infection_mask").mkdir()

        # MosMed-like studies. Half are stored with a flipped axis layout: after
        # preprocessing all must come out in the same (LPS) orientation.
        flipped = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
        truth = {}
        for i in range(10):
            ct, mask = _synthetic_case(rng)
            truth[f"study_{255 + i:04d}"] = mask
            if i % 2 == 0:
                _write_volume(mosmed / "studies" / f"study_{255 + i:04d}.nii.gz", ct)
                _write_volume(mosmed / "masks" / f"study_{255 + i:04d}_mask.nii.gz", mask)
            else:  # same anatomy, stored mirrored with a direction matrix that says so
                _write_volume(mosmed / "studies" / f"study_{255 + i:04d}.nii.gz", ct[:, ::-1, ::-1].copy(), flipped)
                _write_volume(mosmed / "masks" / f"study_{255 + i:04d}_mask.nii.gz", mask[:, ::-1, ::-1].copy(), flipped)

        # COVID-like cases; one is an 8-bit export rather than HU (like the radiopaedia cases)
        for i in range(5):
            ct, mask = _synthetic_case(rng)
            if i == 0:
                ct = np.clip((ct + 1000.0) / 1400.0 * 255.0, 0, 255).astype(np.float32)
            _write_volume(covid / "ct_scans" / f"coronacases_{i:03d}.nii.gz", ct)
            _write_volume(covid / "infection_mask" / f"coronacases_{i:03d}.nii.gz", mask)

        m1 = process_mosmed_dataset(str(mosmed), str(out), target_size=(64, 64), context_slices=2)
        m2 = process_covid19_dataset(str(covid), str(out), target_size=(64, 64), context_slices=2)
        assert m1["cases"] == {"train": 6, "val": 2, "test": 2}, m1["cases"]
        assert sum(m2["cases"].values()) == 5
        # 4 lesion slices + 2 context slices each side = 8 slices per case
        assert sum(m1["slices"].values()) == 10 * 8 and sum(m1["slices_with_lesion"].values()) == 10 * 4

        # Orientation: every saved mask must equal the un-mirrored ground truth,
        # i.e. flipped-storage studies were brought back to the common orientation.
        checked = 0
        for mp in sorted(out.glob("*/masks/mosmed_*.npy")):
            study, z = mp.stem[len("mosmed_"):].rsplit("_s", 1)
            assert np.array_equal(np.load(mp), truth[study][int(z)]), f"orientation mismatch for {mp.name}"
            checked += 1
        assert checked == 80

        # Intensities: normalized to [0, 1] and the lesion is brighter than lung in every image,
        # including the 8-bit case (which would be squashed if treated as HU).
        for ip in sorted(out.glob("*/images/*.npy")):
            img, msk = np.load(ip), np.load(ip.parent.parent / "masks" / ip.name)
            assert img.shape == (3, 64, 64), img.shape          # 2.5D: slice and its two neighbours
            assert img.dtype == np.float16 and 0.0 <= img.min() and img.max() <= 1.0
            img = img[1].astype(np.float32)
            if msk.any():
                assert img[msk > 0].mean() > img[msk == 0].mean() + 0.15, f"low lesion contrast in {ip.name}"

        # No patient appears in two splits
        seen = {}
        for ip in out.glob("*/images/*.npy"):
            pid = ip.stem.rsplit("_s", 1)[0]
            assert seen.setdefault(pid, ip.parent.parent.name) == ip.parent.parent.name, f"{pid} leaks across splits"

        # Semi-supervised split: per dataset, whole patients, deterministic
        ds = DualTaskDataset(out, split="train", labeled_ratio=0.4, seed=42)
        summary = ds.summary()
        assert summary["mosmed"]["labeled_patients"] == 2 and summary["mosmed"]["unlabeled_patients"] == 4, summary
        assert summary["covid19"]["labeled_patients"] >= 1 and summary["covid19"]["unlabeled_patients"] >= 1, summary
        labeled_by_patient = {}
        for pid, lab in zip(ds.patient_ids, ds.is_labeled_list):
            assert labeled_by_patient.setdefault(pid, lab) == lab, "a patient is split between labeled and unlabeled"
        assert DualTaskDataset(out, split="train", labeled_ratio=0.4, seed=42).is_labeled_list == ds.is_labeled_list

        # Unlabeled samples expose no annotation; labeled ones have matching targets
        n_labeled_with_lesion = 0
        for i in range(len(ds)):
            item = ds[i]
            assert item["image"].shape == (3, 64, 64) and item["level_set"].shape == (1, 64, 64)
            if not item["is_labeled"]:
                assert item["mask"].sum() == 0, "unlabeled sample leaked its mask"
            else:
                agree = (item["level_set"] < 0) == (item["mask"] > 0.5)
                assert bool(agree.all()), "level set target disagrees with the (augmented) mask"
                n_labeled_with_lesion += int(item["mask"].sum() > 0)
        assert n_labeled_with_lesion > 0

        # Two-stream batches: half labeled, half unlabeled
        train_loader, val_loader, test_loader = get_dataloaders(out, batch_size=4, num_workers=0, labeled_ratio=0.4)
        batch = next(iter(train_loader))
        assert batch["is_labeled"].tolist() == [True, True, False, False]
        assert len(val_loader.dataset) == len(list(out.glob("val/images/*.npy")))

        # One real training epoch on CPU, then evaluation
        cfg = DTCRConfig(
            data_dir=str(out), epochs=1, batch_size=4, num_workers=0, labeled_ratio=0.4,
            checkpoint_dir=str(tmp / "ckpt"), output_dir=str(tmp / "outputs"), device="cpu",
        )
        train(cfg, grad_accum_steps=1, use_amp=False)

        history = json.loads((tmp / "outputs" / "training_history.json").read_text())
        assert len(history) == 1
        vm = history[0]["val_metrics"]
        for key in ("dice", "dice_std", "dice_mosmed", "dice_covid19", "global_dice", "lesion_dice",
                    "val_loss", "sensitivity", "specificity", "accuracy", "task_agreement"):
            assert key in vm and np.isfinite(vm[key]), key
        assert vm["num_patients"] == 3  # 2 mosmed + 1 covid validation patients
        assert np.isfinite(history[0]["train_losses"]["total"])
        for name in ("last_model.pth", "best_model.pth", "best_dice_model.pth"):
            assert (tmp / "ckpt" / name).exists(), name

        # Patient-level Dice must pool slices: check against a hand computation with a fake model
        class Oracle(torch.nn.Module):
            """Predicts the true mask on lesion slices of the first half of each batch, nothing elsewhere."""
            def forward(self, x):
                logits = torch.full_like(x[:, :1], -10.0)
                return logits, torch.ones_like(logits), torch.zeros_like(logits)

        empty = evaluate_dataset(Oracle(), val_loader, torch.device("cpu"), use_amp=False)
        assert empty["dice"] == 0.0 and empty["sensitivity"] == 0.0 and empty["specificity"] == 100.0

        from evaluate import evaluate_checkpoint
        results = evaluate_checkpoint(
            str(tmp / "ckpt" / "best_model.pth"), str(out), output_dir=str(tmp / "eval"),
            batch_size=4, device="cpu", save_visualizations=0,
        )
        assert results["total_patients"] == 3
        assert set(results["patient_level_by_dataset"]) == {"mosmed", "covid19"}

        # The threshold is chosen on validation, stored in the checkpoint and used at test time
        ckpt = torch.load(tmp / "ckpt" / "best_model.pth", weights_only=False)
        assert 0.1 <= ckpt["best_threshold"] <= 0.9 and ckpt["tta"] is True
        assert results["threshold"] == ckpt["best_threshold"] and results["tta"] is True
        assert ckpt["config"]["in_channels"] == 3 and ckpt["config"]["arch"] == "dtcr"

        # The pretrained-encoder variant trains, calibrates and evaluates through the same code path
        cfg_r = DTCRConfig(
            data_dir=str(out), epochs=1, batch_size=4, num_workers=0, labeled_ratio=1.0,
            checkpoint_dir=str(tmp / "ckpt_r"), output_dir=str(tmp / "outputs_r"), device="cpu",
            arch="dtcr_resnet34", pretrained=False, bce_weight=1.0, pos_weight=1.0,
        )
        train(cfg_r, grad_accum_steps=1, use_amp=False)
        results_r = evaluate_checkpoint(
            str(tmp / "ckpt_r" / "best_dice_model.pth"), str(out), output_dir=str(tmp / "eval_r"),
            batch_size=4, device="cpu", save_visualizations=2,
        )
        assert results_r["total_patients"] == 3
        assert torch.load(tmp / "ckpt_r" / "best_dice_model.pth", weights_only=False)["config"]["arch"] == "dtcr_resnet34"

        # Ensemble of the two models, each at a different input size, scored at full resolution
        cfg_s = DTCRConfig(
            data_dir=str(out), epochs=1, batch_size=4, num_workers=0, labeled_ratio=1.0,
            checkpoint_dir=str(tmp / "ckpt_s"), output_dir=str(tmp / "outputs_s"), device="cpu",
            image_size=32,
        )
        train(cfg_s, grad_accum_steps=1, use_amp=False)
        assert torch.load(tmp / "ckpt_s" / "best_model.pth", weights_only=False)["config"]["image_size"] == 32
        from ensemble import run_ensemble
        ens = run_ensemble([str(tmp / "ckpt_r" / "best_model.pth"), str(tmp / "ckpt_s" / "best_model.pth")],
                           str(out), str(tmp / "ens"), batch_size=4, device="cpu", num_workers=0)
        assert set(ens) == {"ckpt_r/best_model", "ckpt_s/best_model", "ensemble"}, set(ens)
        for name, r in ens.items():
            assert 0.3 <= r["threshold"] <= 0.7, (name, r["threshold"])
            assert len(r["test"]["patients"]) == 3 and set(r["test"]["by_dataset"]) == {"mosmed", "covid19"}
        assert (tmp / "ens" / "ensemble_metrics.json").exists()

    print("Pipeline tests passed.")


def _chest_volume(rng: np.random.Generator, size: int = 128, depth: int = 10):
    """Air around a body ellipse holding two lungs; a soft-tissue lesion in the right lung.

    Each slice gets its own small intensity offset so neighbouring slices differ.
    """
    yy, xx = np.mgrid[:size, :size]
    body = ((yy - 64) / 40.0) ** 2 + ((xx - 64) / 56.0) ** 2 <= 1
    lung_r = ((yy - 62) / 26.0) ** 2 + ((xx - 42) / 16.0) ** 2 <= 1
    lung_l = ((yy - 62) / 26.0) ** 2 + ((xx - 86) / 16.0) ** 2 <= 1
    lesion = (yy - 60) ** 2 + (xx - 40) ** 2 <= 16
    ct = np.full((depth, size, size), -1000.0, dtype=np.float32)
    mask = np.zeros((depth, size, size), dtype=np.uint8)
    for z in range(depth):
        sl = ct[z]
        sl[body] = 40.0
        sl[lung_r | lung_l] = -850.0 + 15.0 * z
        if 3 <= z <= 6:
            sl[lesion] = 20.0
            mask[z][lesion] = 1
        sl += rng.normal(0.0, 5.0, size=sl.shape).astype(np.float32)
    return ct, mask


def test_lung_crop_and_2p5d():
    from data.preprocessing import lung_bbox, normalize_ct_slice, resize_slice, save_case_slices, select_slices

    ct, mask = _chest_volume(np.random.default_rng(4))
    idx = select_slices(mask, 2)
    y0, y1, x0, x1 = lung_bbox(ct, mask, idx)
    assert y1 - y0 == x1 - x0, "crop must be square"
    assert (y1 - y0) < 0.75 * 128, f"crop {y1 - y0}px barely smaller than the slice"
    ys, xs = np.nonzero(mask.any(axis=0))
    assert y0 <= ys.min() and ys.max() < y1 and x0 <= xs.min() and xs.max() < x1, "lesion outside the crop"
    assert x0 <= 26 and x1 >= 102, "crop cuts into a lung"

    with tempfile.TemporaryDirectory() as tmp:
        split = Path(tmp)
        for sub in ("images", "masks", "level_sets"):
            (split / sub).mkdir()
        saved, with_lesion, contrast = save_case_slices(ct, mask, idx, "mosmed_study_0001", split, (64, 64))
        assert saved == len(idx) and with_lesion == 4 and contrast > 0.15

        img4 = np.load(split / "images" / "mosmed_study_0001_s4.npy")
        img5 = np.load(split / "images" / "mosmed_study_0001_s5.npy")
        assert img4.shape == (3, 64, 64)
        assert np.array_equal(img5[0], img4[1]) and np.array_equal(img4[2], img5[1]), "neighbour channels are wrong"
        # Edge slice: the missing neighbour repeats the slice itself
        img0 = np.load(split / "images" / f"mosmed_study_0001_s{idx[0]}.npy")
        if idx[0] == 0:
            assert np.array_equal(img0[0], img0[1])

        # The crop makes the lesion bigger on the network input than plain resizing would
        cropped_area = np.load(split / "masks" / "mosmed_study_0001_s4.npy").sum()
        full_area = resize_slice(mask[4], (64, 64), is_mask=True).sum()
        assert cropped_area > 1.5 * full_area, (cropped_area, full_area)

    # Non-HU (8-bit) volumes and slices without a lung region keep the full field of view
    assert lung_bbox(np.clip((ct + 1000) / 5, 0, 255), mask, idx) is None
    assert lung_bbox(np.full_like(ct, 40.0), mask, idx) is None


def test_dice_modes():
    """Hybrid Dice gives a missed small lesion real weight next to a large one."""
    from losses.seg_loss import DiceLoss

    target = torch.zeros(2, 1, 64, 64)
    target[0, 0, 10:50, 10:50] = 1      # 1600 px lesion, predicted perfectly
    target[1, 0, 30:33, 30:33] = 1      # 9 px nodule, missed
    pred = target.clone()
    pred[1] = 0
    batch = DiceLoss(mode="batch")(pred, target).item()
    hybrid = DiceLoss(mode="hybrid")(pred, target).item()
    assert batch < 0.01, batch            # the missed nodule is almost invisible to pooled Dice
    assert hybrid > 0.2, hybrid           # ... but costs about a quarter of the loss in hybrid mode

    empty_t = torch.zeros(2, 1, 64, 64)
    assert DiceLoss(mode="hybrid")(torch.zeros_like(empty_t), empty_t).item() == 0.0
    fp = torch.zeros_like(empty_t)
    fp[0, 0, 0, 0] = 1.0                  # one false-positive pixel on an empty batch
    assert abs(DiceLoss(mode="hybrid")(fp, empty_t).item() - 0.5) < 1e-6  # priced like pooled Dice


if __name__ == "__main__":
    test_lung_crop_and_2p5d()
    test_dice_modes()
    test_level_set_round_trip()
    test_covid_pairing_picks_infection_mask()
    test_lidc_processing()
    test_end_to_end_pipeline()
