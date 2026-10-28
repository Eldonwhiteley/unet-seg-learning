"""
Synthetic Colored Squares Dataset for U-Net Segmentation.

This module generates synthetic images containing random colored squares
along with their ground-truth segmentation masks. Synthetic datasets are
ideal for learning and debugging U-Net models because they:
1. Require zero external data downloads.
2. Train to high accuracy within minutes.
3. Clearly reveal the pixel-level precision of skip connections.
"""

from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_single_sample(
    img_size: int = 128,
    min_squares: int = 1,
    max_squares: int = 4,
    min_sq_size: int = 15,
    max_sq_size: int = 35,
    num_classes: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single synthetic image with colored squares and its mask.

    Parameters
    ----------
    img_size : int
        Width and height of the square image (default: 128).
    min_squares : int
        Minimum number of squares per image.
    max_squares : int
        Maximum number of squares per image.
    min_sq_size : int
        Minimum side length of each square.
    max_sq_size : int
        Maximum side length of each square.
    num_classes : int
        2 for binary segmentation (0: background, 1: square),
        > 2 for multi-class color-coded segmentation.
    rng : np.random.Generator, optional
        NumPy random generator for reproducible generation.

    Returns
    -------
    image : np.ndarray
        RGB image of shape (img_size, img_size, 3) in range [0.0, 1.0].
    mask : np.ndarray
        Ground-truth mask of shape (img_size, img_size).
        Values are integers in {0, 1} for binary or {0, ..., num_classes-1}.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Black background (0.0)
    image = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.int64)

    num_squares = rng.integers(min_squares, max_squares + 1)

    for _ in range(num_squares):
        size = rng.integers(min_sq_size, max_sq_size + 1)
        x = rng.integers(0, img_size - size)
        y = rng.integers(0, img_size - size)

        if num_classes == 2:
            # Binary mode: random bright RGB color for square, mask value = 1
            color = rng.uniform(0.3, 1.0, size=(3,)).astype(np.float32)
            class_id = 1
        else:
            # Multi-class mode: class 1 = Red-dominant, class 2 = Green-dominant, class 3 = Blue-dominant
            class_id = int(rng.integers(1, num_classes))
            palette = {
                1: np.array([rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3)]), # Red
                2: np.array([rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0), rng.uniform(0.0, 0.3)]), # Green
                3: np.array([rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3), rng.uniform(0.7, 1.0)]), # Blue
            }
            color = palette.get(class_id, rng.uniform(0.4, 1.0, size=(3,))).astype(np.float32)

        # Draw square onto image and ground truth mask
        image[y : y + size, x : x + size] = color
        mask[y : y + size, x : x + size] = class_id

    return image, mask


class ColoredSquaresDataset(Dataset):
    """
    PyTorch Dataset wrapper for synthetic colored squares segmentation.

    Important PyTorch vs TensorFlow note:
    - TensorFlow/Keras expects images as (Height, Width, Channels) [HWC].
    - PyTorch expects image tensors as (Channels, Height, Width) [CHW].
    This Dataset automatically converts HWC -> CHW.
    """

    def __init__(
        self,
        num_samples: int = 500,
        img_size: int = 128,
        num_classes: int = 2,
        seed: Optional[int] = None,
        dynamic: bool = False,
    ):
        """
        Parameters
        ----------
        num_samples : int
            Number of dataset samples (size of epoch).
        img_size : int
            Height and width of generated square images.
        num_classes : int
            Number of segmentation classes (default 2: background vs square).
        seed : int, optional
            Random seed for pre-generating deterministic datasets.
        dynamic : bool
            If True, generates brand-new samples on every getitem call.
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes
        self.dynamic = dynamic
        self.rng = np.random.default_rng(seed)

        # If not dynamic, pre-generate dataset in memory for fast, deterministic epochs
        self.images = []
        self.masks = []
        if not self.dynamic:
            for _ in range(num_samples):
                img, msk = generate_single_sample(
                    img_size=self.img_size,
                    num_classes=self.num_classes,
                    rng=self.rng,
                )
                self.images.append(img)
                self.masks.append(msk)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dynamic:
            img, msk = generate_single_sample(
                img_size=self.img_size,
                num_classes=self.num_classes,
            )
        else:
            img, msk = self.images[idx], self.masks[idx]

        # Convert HWC numpy array -> CHW PyTorch FloatTensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        if self.num_classes == 2:
            # For binary segmentation, mask shape is (1, H, W) float32 for BCE / Dice loss
            msk_tensor = torch.from_numpy(msk).unsqueeze(0).float()
        else:
            # For multi-class segmentation, mask shape is (H, W) int64 for CrossEntropyLoss
            msk_tensor = torch.from_numpy(msk).long()

        return img_tensor, msk_tensor
