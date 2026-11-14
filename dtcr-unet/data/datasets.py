"""PyTorch Dataset & DataLoader for DTCR-U-Net."""

import math
import random
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .level_set import compute_level_set


def parse_stem(stem: str) -> Tuple[str, str]:
    """Split a slice file stem '<dataset>_<case>_s<idx>' into (dataset, patient_id)."""
    stem = stem[:-4] if stem.endswith(".npy") else stem
    patient_id = stem.rsplit("_s", 1)[0] if "_s" in stem else stem
    return patient_id.split("_", 1)[0], patient_id


class DualTaskDataset(Dataset):
    """PyTorch Dataset loading preprocessed CT slices and building dual-task targets.

    Returns:
        image: FloatTensor (1, H, W) in [0, 1].
        mask: FloatTensor (1, H, W) binary segmentation target. All zeros for
            unlabeled training samples: their annotation is never read (Section 4.2).
        level_set: FloatTensor (1, H, W) level set target T(y), computed from the
            (augmented) mask so both targets always agree.
        is_labeled: BoolTensor, whether the ground truth may be used.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: bool = True,
        labeled_ratio: float = 1.0,
        seed: int = 42,
        only_lesion_slices: bool = False,
    ):
        """
        Args:
            data_dir: Path to processed data folder containing train/val/test directories.
            split: 'train', 'val', or 'test'.
            transform: Whether to apply data augmentation (only if split == 'train').
            labeled_ratio: Fraction of training PATIENTS that keep their labels (0.0 to 1.0),
                drawn separately inside each source dataset.
            seed: Random seed for selecting labeled vs unlabeled patients.
            only_lesion_slices: If True, keep only slices with positive ground truth masks.
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform and (split == "train")

        self.img_paths = sorted(glob(str(self.data_dir / "images" / "*.npy")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No .npy images found in {self.data_dir / 'images'}")

        if only_lesion_slices:
            lesion_paths = [
                p for p in self.img_paths
                if (self.data_dir / "masks" / Path(p).name).exists()
                and np.load(self.data_dir / "masks" / Path(p).name).any()
            ]
            if lesion_paths:
                self.img_paths = lesion_paths

        parsed = [parse_stem(Path(p).stem) for p in self.img_paths]
        self.datasets = [d for d, _ in parsed]
        self.patient_ids = [pid for _, pid in parsed]

        # Semi-supervised split: whole patients, stratified by source dataset so that
        # every dataset contributes labeled cases (a global draw can leave one with none).
        self.is_labeled_list = [True] * len(self.img_paths)
        if split == "train" and labeled_ratio < 1.0:
            patients_by_dataset: Dict[str, set] = {}
            for d, pid in parsed:
                patients_by_dataset.setdefault(d, set()).add(pid)

            labeled_patients = set()
            for d in sorted(patients_by_dataset):
                patients = sorted(patients_by_dataset[d])
                random.Random(f"{seed}-{d}").shuffle(patients)
                n_labeled = min(len(patients), max(1, int(round(len(patients) * labeled_ratio))))
                labeled_patients.update(patients[:n_labeled])

            self.is_labeled_list = [pid in labeled_patients for pid in self.patient_ids]

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Per-dataset counts of patients and slices, split by labeled / unlabeled."""
        acc: Dict[str, dict] = {}
        for d, pid, lab in zip(self.datasets, self.patient_ids, self.is_labeled_list):
            e = acc.setdefault(d, {"labeled_patients": set(), "unlabeled_patients": set(),
                                   "labeled_slices": 0, "unlabeled_slices": 0})
            e["labeled_patients" if lab else "unlabeled_patients"].add(pid)
            e["labeled_slices" if lab else "unlabeled_slices"] += 1
        return {
            d: {k: (len(v) if isinstance(v, set) else v) for k, v in e.items()}
            for d, e in acc.items()
        }

    def __len__(self) -> int:
        return len(self.img_paths)

    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronized flipping, rotation and scale-cropping (Section 4.1) plus mild gamma jitter."""
        # 1. Random horizontal flip (left/right lung symmetry)
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # 2. Small rotation (+/- 10 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            h, w = img.shape
            rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale=1.0)
            img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            mask = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # 3. Random scale, then centre crop / pad back to size (zoom 0.9x to 1.1x)
        if random.random() > 0.5:
            scale = random.uniform(0.90, 1.10)
            h, w = img.shape
            new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            img_s = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if scale >= 1.0:
                sh, sw = (new_h - h) // 2, (new_w - w) // 2
                img = img_s[sh:sh + h, sw:sw + w]
                mask = mask_s[sh:sh + h, sw:sw + w]
            else:
                ph, pw = (h - new_h) // 2, (w - new_w) // 2
                pad = ((ph, h - new_h - ph), (pw, w - new_w - pw))
                img = np.pad(img_s, pad, mode="constant", constant_values=0)
                mask = np.pad(mask_s, pad, mode="constant", constant_values=0)

        # 4. Mild gamma jitter (scanner / reconstruction variability)
        if random.random() > 0.5:
            img = np.clip(img, 0.0, 1.0) ** random.uniform(0.85, 1.15)

        return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_paths[idx]
        name = Path(img_path).name
        is_labeled = self.is_labeled_list[idx]

        img = np.load(img_path).astype(np.float32)

        # Unlabeled samples never touch their annotation file.
        mask_path = self.data_dir / "masks" / name
        if is_labeled and mask_path.exists():
            mask = (np.load(mask_path) > 0).astype(np.float32)
        else:
            mask = np.zeros_like(img)

        if self.transform:
            img, mask = self._augment(img, mask)

        # T(y) from the final mask, so augmentation can never desynchronize the two targets.
        lsf = compute_level_set(mask)

        return {
            "image": torch.from_numpy(img[None]).float(),
            "mask": torch.from_numpy(mask[None]).float(),
            "level_set": torch.from_numpy(lsf[None]).float(),
            "is_labeled": torch.tensor(is_labeled, dtype=torch.bool),
            "stem": name,
            "patient_id": self.patient_ids[idx],
            "dataset": self.datasets[idx],
        }


class TwoStreamBatchSampler(torch.utils.data.Sampler):
    """Two-stream batch sampler for semi-supervised training.

    Every batch holds a fixed number of labeled and unlabeled slices, so the
    supervised losses always anchor the consistency term.
    """

    def __init__(
        self,
        primary_indices: List[int],
        secondary_indices: List[int],
        batch_size: int,
        secondary_batch_size: int,
    ):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        assert self.primary_batch_size > 0, "Batch size must be greater than secondary batch size"
        assert len(self.primary_indices) > 0, "Primary indices cannot be empty"
        assert len(self.secondary_indices) > 0, "Secondary indices cannot be empty"

    def __iter__(self):
        def stream(indices: List[int]):
            while True:
                for i in np.random.permutation(indices).tolist():
                    yield i

        primary, secondary = stream(self.primary_indices), stream(self.secondary_indices)
        for _ in range(len(self)):
            yield (
                [next(primary) for _ in range(self.primary_batch_size)]
                + [next(secondary) for _ in range(self.secondary_batch_size)]
            )

    def __len__(self) -> int:
        # One epoch visits every sample of the longer stream once; the shorter one cycles.
        return max(
            math.ceil(len(self.primary_indices) / self.primary_batch_size),
            math.ceil(len(self.secondary_indices) / self.secondary_batch_size),
        )


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own augmentation random stream."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 2,
    labeled_ratio: float = 1.0,
    seed: int = 42,
    filter_empty_val: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train, Val, and Test DataLoaders.

    Validation and test keep every extracted slice (lesion slices and their
    context slices) so that patient-level metrics also count false positives
    on lesion-free slices.
    """
    train_ds = DualTaskDataset(
        data_dir=data_dir,
        split="train",
        transform=True,
        labeled_ratio=labeled_ratio,
        seed=seed,
    )
    val_ds = DualTaskDataset(
        data_dir=data_dir,
        split="val",
        transform=False,
        only_lesion_slices=filter_empty_val,
    )
    test_ds = DualTaskDataset(
        data_dir=data_dir,
        split="test",
        transform=False,
        only_lesion_slices=False,
    )

    labeled_idxs = [i for i, lab in enumerate(train_ds.is_labeled_list) if lab]
    unlabeled_idxs = [i for i, lab in enumerate(train_ds.is_labeled_list) if not lab]

    if labeled_ratio < 1.0 and batch_size >= 2 and labeled_idxs and unlabeled_idxs:
        batch_sampler = TwoStreamBatchSampler(
            primary_indices=labeled_idxs,
            secondary_indices=unlabeled_idxs,
            batch_size=batch_size,
            secondary_batch_size=max(1, batch_size // 2),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=_seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=len(train_ds) > batch_size,
            worker_init_fn=_seed_worker,
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
