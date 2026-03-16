"""GL261 ultrasound segmentation dataset for PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor (works around broken np<->torch in 2.7+)."""
    return torch.tensor(np.ascontiguousarray(arr))


def _elastic_deform(img: np.ndarray, msk: np.ndarray,
                    alpha: float = 80.0, sigma: float = 8.0) -> tuple:
    """Apply elastic deformation to image and mask jointly."""
    shape = img.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    coords = [np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)]
    img_out = map_coordinates(img, coords, order=1, mode="reflect")
    msk_out = map_coordinates(msk, coords, order=0, mode="reflect")
    return img_out, msk_out


class GL261SegDataset(Dataset):
    """GL261 ultrasound segmentation dataset.

    Images: grayscale PNGs named GL261_NNNN_0000.png
    Masks:  binary 0/1 PNGs named GL261_NNNN.png
    """

    def __init__(self, img_dir: Path, msk_dir: Path, img_size: int = 512,
                 augment: bool = False, case_ids: set = None,
                 aug_preset: str = "base"):
        self.img_dir = Path(img_dir)
        self.msk_dir = Path(msk_dir)
        self.img_size = img_size
        self.augment = augment
        self.aug_preset = aug_preset

        self.samples = []
        for img_path in sorted(self.img_dir.glob("GL261_*_0000.png")):
            case_id = img_path.stem.replace("_0000", "")
            if case_ids is not None and case_id not in case_ids:
                continue
            msk_path = self.msk_dir / f"{case_id}.png"
            if msk_path.exists():
                self.samples.append((img_path, msk_path, case_id))
            else:
                print(f"[WARN] No mask for {img_path.name}, skipping")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gray(self, path: Path) -> np.ndarray:
        """Load and resize a single grayscale image to img_size."""
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        img_pil = Image.fromarray(img.astype(np.uint8), mode="L")
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(img_pil, dtype=np.float32) / 255.0

    def __getitem__(self, idx: int):
        img_path, msk_path, case_id = self.samples[idx]

        img = self._load_gray(img_path)
        img = img[np.newaxis]  # (1, H, W)

        # Load and resize mask
        msk = np.array(Image.open(msk_path).convert("L"), dtype=np.float32)
        if msk.max() > 1:
            msk = msk / 255.0
        msk_pil = Image.fromarray((msk * 255).astype(np.uint8), mode="L")
        msk_pil = msk_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        msk = np.array(msk_pil, dtype=np.float32) / 255.0

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=2).copy()
                msk = np.flip(msk, axis=1).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
                msk = np.flip(msk, axis=0).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                msk = np.rot90(msk, k).copy()
            if self.aug_preset == "medical_v1" and np.random.rand() < 0.3:
                alpha = np.random.uniform(50, 100)
                sigma = np.random.uniform(5, 10)
                deformed_img, msk = _elastic_deform(img[0], msk, alpha=alpha, sigma=sigma)
                img = deformed_img[np.newaxis]

        img_t = _to_tensor(img)
        msk_t = _to_tensor(msk).unsqueeze(0)

        return img_t, msk_t, case_id
