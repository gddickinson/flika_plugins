"""
Dataset classes for loading calcium imaging data.

UPDATED VERSION:
- D4 augmentation (rotations + reflections) from paper
- Better organized code
- Support for different augmentation modes
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import warnings


def apply_d4_augmentation(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random D4 dihedral group transformation (8 transformations total).

    This matches the paper's augmentation strategy and provides:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 2 reflections per rotation
    = 8 total transformations
    """
    # Random rotation
    k = np.random.randint(0, 4)
    if k > 0:
        image = np.rot90(image, k=k, axes=(-2, -1))
        mask = np.rot90(mask, k=k, axes=(-2, -1))

    # Random horizontal flip
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-1)
        mask = np.flip(mask, axis=-1)

    # Random vertical flip
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-2)
        mask = np.flip(mask, axis=-2)

    return image.copy(), mask.copy()


def apply_weak_augmentation(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply only horizontal and vertical flips (simpler, faster)."""
    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-1)
        mask = np.flip(mask, axis=-1)

    if np.random.rand() > 0.5:
        image = np.flip(image, axis=-2)
        mask = np.flip(mask, axis=-2)

    return image.copy(), mask.copy()


class CalciumEventDataset(Dataset):
    """
    Dataset for calcium event detection from full-frame confocal imaging.

    UPDATED with D4 augmentation from the original paper.

    Args:
        data_dir: Directory containing image files and annotations
        segment_length: Number of frames per segment
        step_size: Step size for sliding window
        temporal_context: Frames to ignore at start/end
        augment: Whether to apply data augmentation
        augmentation_mode: "d4" for rotations+flips, "weak" for flips only
        normalize_min: Minimum value for normalization
        normalize_max: Maximum value for normalization
        transform: Optional transform to apply
    """

    def __init__(
        self,
        data_dir: str,
        segment_length: int = 256,
        step_size: int = 32,
        temporal_context: int = 6,
        augment: bool = True,
        augmentation_mode: str = "d4",
        normalize_min: float = 0.0,
        normalize_max: float = 65535.0,
        transform: Optional[callable] = None
    ):
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.step_size = step_size
        self.temporal_context = temporal_context
        self.augment = augment
        self.augmentation_mode = augmentation_mode
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.transform = transform

        # Select augmentation function
        if augmentation_mode == "d4":
            self.augment_fn = apply_d4_augmentation
        elif augmentation_mode == "weak":
            self.augment_fn = apply_weak_augmentation
        else:
            raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")

        # Load dataset index
        self.samples = self._load_dataset_index()

    def _load_dataset_index(self) -> List[dict]:
        """
        Load dataset index from directory structure.
        Expected structure:
        - data_dir/
          - images/
            - 01_video.tif
            - 02_video.tif
          - masks/
            - 01_class.tif
            - 02_class.tif
        """
        samples = []

        image_dir = self.data_dir / 'images'
        mask_dir = self.data_dir / 'masks'

        if not image_dir.exists():
            warnings.warn(f"Image directory not found: {image_dir}")
            return samples

        # Find all image files
        image_files = sorted(list(image_dir.glob('*.tif')) +
                           list(image_dir.glob('*.tiff')))

        for img_file in image_files:
            sample = {'image_path': str(img_file)}

            # Look for corresponding mask files
            base_name = img_file.stem.replace('_video', '')
            class_mask_path = mask_dir / f"{base_name}_class.tif"

            if class_mask_path.exists():
                sample['class_mask_path'] = str(class_mask_path)

            # Only add samples that have class masks (for training)
            if 'class_mask_path' in sample or len(samples) == 0:
                samples.append(sample)

        if len(samples) == 0:
            warnings.warn(f"No valid samples found in {self.data_dir}")

        return samples

    def _load_tiff(self, path: str) -> np.ndarray:
        """Load TIFF file (supports both PIL and tifffile)."""
        try:
            from tifffile import imread
            return imread(path)
        except ImportError:
            from PIL import Image
            img = Image.open(path)
            frames = []
            try:
                while True:
                    frames.append(np.array(img))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
            return np.array(frames)

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return (image - self.normalize_min) / (self.normalize_max - self.normalize_min)

    def __len__(self) -> int:
        """Return total number of segments across all recordings."""
        total_segments = 0
        for sample in self.samples:
            img = self._load_tiff(sample['image_path'])
            T = img.shape[0]
            num_segments = (T - self.segment_length) // self.step_size + 1
            total_segments += max(1, num_segments)
        return total_segments

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Get a single segment.

        Returns:
            image: (C, T, H, W) tensor
            mask: (T, H, W) tensor with class labels
            metadata: dict with additional info
        """
        # Map global index to (recording_idx, segment_idx)
        recording_idx = 0
        segment_idx = idx

        for i, sample in enumerate(self.samples):
            img = self._load_tiff(sample['image_path'])
            T = img.shape[0]
            num_segments = (T - self.segment_length) // self.step_size + 1
            num_segments = max(1, num_segments)

            if segment_idx < num_segments:
                recording_idx = i
                break
            segment_idx -= num_segments

        # Load the recording
        sample = self.samples[recording_idx]
        image = self._load_tiff(sample['image_path'])

        # Load mask if available
        if 'class_mask_path' in sample:
            mask = self._load_tiff(sample['class_mask_path'])
        else:
            mask = np.zeros_like(image, dtype=np.int64)

        # Extract segment
        start = segment_idx * self.step_size
        end = start + self.segment_length

        if end > image.shape[0]:
            start = max(0, image.shape[0] - self.segment_length)
            end = image.shape[0]

        image_segment = image[start:end]
        mask_segment = mask[start:end]

        # Normalize image
        image_segment = self._normalize(image_segment.astype(np.float32))

        # Add channel dimension: (T, H, W) -> (C, T, H, W)
        if image_segment.ndim == 3:
            image_segment = image_segment[np.newaxis, ...]

        # Apply augmentation (UPDATED - uses D4 or weak augmentation)
        if self.augment and self.transform is None:
            image_segment, mask_segment = self.augment_fn(image_segment, mask_segment)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_segment).float()
        mask_tensor = torch.from_numpy(mask_segment).long()

        # Apply custom transform if provided
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        # Create metadata
        metadata = {
            'recording_idx': recording_idx,
            'segment_idx': segment_idx,
            'start_frame': start,
            'end_frame': end,
            'image_path': sample['image_path']
        }

        return image_tensor, mask_tensor, metadata


def create_dataloaders(config, data_dir: Optional[str] = None):
    """
    Create training and validation dataloaders.

    Args:
        config: Configuration object
        data_dir: Optional override for data directory

    Returns:
        train_loader, val_loader
    """
    if data_dir is None:
        data_dir = config.data.data_dir

    # Create dataset
    full_dataset = CalciumEventDataset(
        data_dir=data_dir,
        segment_length=config.data.segment_length,
        step_size=config.data.step_size,
        temporal_context=config.data.temporal_context,
        augment=config.data.augment,
        augmentation_mode=config.data.augmentation_mode,  # NEW
        normalize_min=config.data.normalize_min,
        normalize_max=config.data.normalize_max
    )

    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(config.data.train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.data.random_seed)
    )

    # Create dataloaders
    # Force num_workers=0 on CPU to avoid multiprocessing memory issues
    num_workers = config.training.num_workers if config.training.device == 'cuda' else 0

    if num_workers == 0 and config.training.num_workers > 0:
        print(f"   Note: Setting num_workers=0 for CPU training (was {config.training.num_workers})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.training.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.training.device == 'cuda' else False
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    print("Dataset class with D4 augmentation defined successfully")
    print("\nAugmentation modes:")
    print("  - 'd4': 8 transformations (4 rotations × 2 reflections)")
    print("  - 'weak': 4 transformations (flips only)")
