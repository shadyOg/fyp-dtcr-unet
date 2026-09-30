"""PyTorch Dataset & DataLoader for DTCR-U-Net."""

import os
import random
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DualTaskDataset(Dataset):
    """PyTorch Dataset loading preprocessed CT slices, masks, and Level Set Maps.

    Returns:
        image: torch.FloatTensor of shape (1, H, W) normalized to [0, 1].
        mask: torch.FloatTensor of shape (1, H, W) binary segmentation target (0 or 1).
        level_set: torch.FloatTensor of shape (1, H, W) continuous signed distance map T(y).
        is_labeled: torch.BoolTensor indicating whether ground-truth mask is used (for semi-supervised learning).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        transform: bool = True,
        labeled_ratio: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to processed data folder containing train/val/test directories.
            split: 'train', 'val', or 'test'.
            transform: Whether to apply data augmentation (only if split == 'train').
            labeled_ratio: Fraction of training samples that are labeled (0.0 to 1.0).
            seed: Random seed for selecting labeled vs unlabeled samples.
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform and (split == "train")

        self.img_paths = sorted(glob(str(self.data_dir / "images" / "*.npy")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No .npy images found in {self.data_dir / 'images'}")

        # Determine which samples are labeled in semi-supervised training while keeping patient-level integrity
        self.is_labeled_list = [True] * len(self.img_paths)
        if split == "train" and labeled_ratio < 1.0:
            stems_file = self.data_dir / "labeled_stems.txt"
            if stems_file.exists():
                with open(stems_file, "r") as f:
                    labeled_stems = set(line.strip() for line in f if line.strip())
                self.is_labeled_list = [(Path(p).stem in labeled_stems) for p in self.img_paths]
            else:
                rng = random.Random(seed)
                patient_map: Dict[str, List[int]] = {}
                for i, p in enumerate(self.img_paths):
                    stem = Path(p).stem
                    pid = stem.rsplit("_s", 1)[0] if "_s" in stem else stem
                    patient_map.setdefault(pid, []).append(i)

                unique_patients = sorted(patient_map.keys())
                rng.shuffle(unique_patients)
                num_labeled_patients = int(round(len(unique_patients) * labeled_ratio))
                labeled_patients = set(unique_patients[:num_labeled_patients])

                self.is_labeled_list = [False] * len(self.img_paths)
                for pid in labeled_patients:
                    for idx in patient_map[pid]:
                        self.is_labeled_list[idx] = True

    def __len__(self) -> int:
        return len(self.img_paths)

    def _augment(
        self, img: np.ndarray, mask: np.ndarray, lsf: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply synchronized geometric and intensity data augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            lsf = np.fliplr(lsf).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            lsf = np.flipud(lsf).copy()

        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()
            lsf = np.rot90(lsf, k).copy()

        # Random scale and crop / pad (zoom 0.85x to 1.15x)
        if random.random() > 0.5:
            scale = random.uniform(0.85, 1.15)
            h, w = img.shape
            new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            img_s = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_s = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            lsf_s = cv2.resize(lsf, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            if scale >= 1.0:
                sh = (new_h - h) // 2
                sw = (new_w - w) // 2
                img = img_s[sh:sh+h, sw:sw+w]
                mask = mask_s[sh:sh+h, sw:sw+w]
                lsf = lsf_s[sh:sh+h, sw:sw+w]
            else:
                ph = (h - new_h) // 2
                pw = (w - new_w) // 2
                img = np.pad(img_s, ((ph, h - new_h - ph), (pw, w - new_w - pw)), mode="constant", constant_values=0)
                mask = np.pad(mask_s, ((ph, h - new_h - ph), (pw, w - new_w - pw)), mode="constant", constant_values=0)
                lsf = np.pad(lsf_s, ((ph, h - new_h - ph), (pw, w - new_w - pw)), mode="edge")

        # Subtle intensity jitter (gamma)
        if random.random() > 0.5:
            gamma = random.uniform(0.85, 1.15)
            img = np.clip(img ** gamma, 0.0, 1.0)

        # Intensity shift
        if random.random() > 0.3:
            shift = random.uniform(-0.05, 0.05)
            img = np.clip(img + shift, 0.0, 1.0)

        # Gaussian noise
        if random.random() > 0.3:
            noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        # Gaussian blur
        if random.random() > 0.3:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.3, 0.8))

        return img, mask, lsf

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.img_paths[idx]
        stem = Path(img_path).name

        mask_path = self.data_dir / "masks" / stem
        lsf_path = self.data_dir / "level_sets" / stem

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32) if mask_path.exists() else np.zeros_like(img)
        lsf = np.load(lsf_path).astype(np.float32) if lsf_path.exists() else np.zeros_like(img)

        # Ensure LSF values are strictly normalized to [-1.0, 1.0], fixing any unnormalized empty slices from disk
        lsf = np.clip(lsf, -1.0, 1.0)

        if self.transform:
            img, mask, lsf = self._augment(img, mask, lsf)

        # Add channel dimension: (H, W) -> (1, H, W)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        if lsf.ndim == 2:
            lsf = np.expand_dims(lsf, axis=0)

        is_labeled = self.is_labeled_list[idx]
        patient_id = stem.rsplit("_s", 1)[0] if "_s" in stem else stem

        return {
            "image": torch.from_numpy(img).float(),
            "mask": torch.from_numpy(mask).float(),
            "level_set": torch.from_numpy(lsf).float(),
            "is_labeled": torch.tensor(is_labeled, dtype=torch.bool),
            "stem": stem,
            "patient_id": patient_id,
        }


class TwoStreamBatchSampler(torch.utils.data.Sampler):
    """Balanced two-stream batch sampler for semi-supervised training.

    Ensures that every single training batch contains an exact balance of labeled
    and unlabeled slices, preventing un-anchored consistency drift.
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
        primary_perm = np.random.permutation(self.primary_indices).tolist()
        secondary_perm = np.random.permutation(self.secondary_indices).tolist()

        # Infinite generator for secondary (unlabeled) stream
        sec_idx = 0
        for i in range(0, len(primary_perm) - self.primary_batch_size + 1, self.primary_batch_size):
            p_batch = primary_perm[i : i + self.primary_batch_size]
            s_batch = []
            while len(s_batch) < self.secondary_batch_size:
                if sec_idx >= len(secondary_perm):
                    secondary_perm = np.random.permutation(self.secondary_indices).tolist()
                    sec_idx = 0
                s_batch.append(secondary_perm[sec_idx])
                sec_idx += 1
            yield p_batch + s_batch

    def __len__(self) -> int:
        return len(self.primary_indices) // self.primary_batch_size


def get_dataloaders(
    data_dir: Union[str, Path],
    batch_size: int = 16,
    num_workers: int = 2,
    labeled_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create Train, Val, and Test DataLoaders."""
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
    )
    test_ds = DualTaskDataset(
        data_dir=data_dir,
        split="test",
        transform=False,
    )

    if labeled_ratio < 1.0 and batch_size >= 2:
        labeled_idxs = [i for i, lab in enumerate(train_ds.is_labeled_list) if lab]
        unlabeled_idxs = [i for i, lab in enumerate(train_ds.is_labeled_list) if not lab]
        sec_batch = max(1, batch_size // 2)

        batch_sampler = TwoStreamBatchSampler(
            primary_indices=labeled_idxs,
            secondary_indices=unlabeled_idxs,
            batch_size=batch_size,
            secondary_batch_size=sec_batch,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if len(train_ds) > batch_size else False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader