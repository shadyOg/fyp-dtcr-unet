import os
from glob import glob
from torch.utils.data import Dataset
import torch
import numpy as np
import nibabel as nib


class MosMedDataset(Dataset):
    """Simple Dataset wrapper for MosMed-style data.

    Expects either:
    - root/images/{split} and root/masks/{split} directories with paired files, or
    - a flat folder where masks are named with a `_mask` suffix matching images.
    """

    def __init__(self, root_dir, split='train', transform=None, exts=('.nii', '.nii.gz', '.png', '.jpg', '.npy')):
        self.root = root_dir
        self.transform = transform

        img_dir = os.path.join(root_dir, 'images', split)
        mask_dir = os.path.join(root_dir, 'masks', split)

        if os.path.isdir(img_dir) and os.path.isdir(mask_dir):
            self.images = sorted(glob(os.path.join(img_dir, '*')))
            self.masks = sorted(glob(os.path.join(mask_dir, '*')))
        else:
            # fallback: find files in root_dir and pair by `_mask` suffix
            files = sorted(glob(os.path.join(root_dir, '*')))
            images = [p for p in files if p.lower().endswith(exts) and not p.lower().endswith('_mask' + os.path.splitext(p)[1])]
            self.images = sorted(images)
            self.masks = [self._find_mask_for_image(p) for p in self.images]

        if len(self.images) != len(self.masks):
            raise RuntimeError(f"Images and masks count mismatch: {len(self.images)} vs {len(self.masks)}")

    def _find_mask_for_image(self, img_path):
        base, ext = os.path.splitext(img_path)
        candidates = [base + '_mask' + ext, base + '_mask' + ext + '.gz']
        for c in candidates:
            if os.path.exists(c):
                return c
        # try common extensions
        for ext in ('.nii', '.nii.gz', '.png', '.npy'):
            c = base + '_mask' + ext
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"Mask not found for image {img_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = self._load_volume(img_path)
        mask = self._load_volume(mask_path)

        # normalize image
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Ensure channel-first
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            img = np.expand_dims(img, 0)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, 0)
        elif mask.ndim == 3 and mask.shape[0] not in (1, 3):
            mask = np.expand_dims(mask, 0)

        img = torch.from_numpy(img.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        if self.transform:
            img, mask = self.transform(img, mask)

        return img, mask

    def _load_volume(self, path):
        path = str(path)
        if path.lower().endswith(('.nii', '.nii.gz')):
            return np.asarray(nib.load(path).get_fdata())
        if path.lower().endswith('.npy'):
            return np.load(path)
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                from imageio import imread

                data = imread(path)
                # if grayscale, return 2D
                if data.ndim == 3 and data.shape[2] in (3, 4):
                    # convert to grayscale by averaging channels
                    data = data[..., :3].mean(axis=2)
                return np.asarray(data)
            except Exception as e:
                raise RuntimeError(f"Failed loading image {path}: {e}")

        raise RuntimeError(f"Unsupported file type for {path}")
