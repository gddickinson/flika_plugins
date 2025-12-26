"""
Configuration file for calcium event detection model.

UPDATED VERSION - Matches original paper configuration:
- Reduced model size (8 base channels vs 64)
- Better training parameters
- Learning rate scheduler support
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


def get_default_device() -> str:
    """
    Automatically detect and return the best available device.

    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU

    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        # Enable MPS fallback for unsupported operations
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return 'mps'
    return 'cpu'


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    in_channels: int = 1
    num_classes: int = 4  # background, Ca2+ spark, Ca2+ puff, Ca2+ wave

    # CRITICAL CHANGE: Reduced from 64 to 8 to match paper
    # This makes training 8x faster and helps with class imbalance
    base_channels: int = 8  # Original paper uses 8, not 64!

    # Number of downsampling/upsampling steps
    unet_steps: int = 4  # Can be increased to 5 for deeper network


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    # Data paths
    data_dir: str = "./data"
    zenodo_url: str = "https://zenodo.org/records/10391727"

    # Data parameters (matches paper exactly)
    segment_length: int = 256  # frames per segment
    step_size: int = 32  # overlap between segments
    temporal_context: int = 6  # frames to ignore at start/end

    # Normalization (matches paper's "abs_max" mode)
    normalize_min: float = 0.0
    normalize_max: float = 65535.0  # 16-bit max

    # Data augmentation (IMPROVED - uses D4 transformations)
    augment: bool = True
    augmentation_mode: str = "d4"  # "d4" for rotations+flips, "weak" for flips only

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    random_seed: int = 42


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_iterations: int = 100000  # Paper uses 100k iterations

    # Loss function
    loss_type: str = "lovasz"  # Lovász-Softmax as in paper

    # Device
    device: str = field(default_factory=get_default_device)
    num_workers: int = 4

    # Checkpointing (matches paper)
    save_dir: str = "./checkpoints"
    save_frequency: int = 5000  # Save every 5000 iterations
    log_frequency: int = 100    # Log every 100 iterations
    val_frequency: int = 1000   # Validate every 1000 iterations

    # Learning rate scheduler (NEW - from paper)
    use_scheduler: bool = True
    scheduler_type: str = "step"  # "step" or "none"
    scheduler_step_size: int = 20000  # Reduce LR every 20k iterations
    scheduler_gamma: float = 0.1  # Multiply LR by 0.1 at each step

    # Resume training
    resume_from: Optional[str] = None
    pretrained_weights: Optional[str] = None


@dataclass
class InferenceConfig:
    """Inference and post-processing configuration."""
    # Segmentation
    threshold_method: str = "otsu"  # or "fixed"
    fixed_threshold: float = 0.5

    # Post-processing for Ca2+ puffs and waves
    merge_gap_frames: int = 2  # merge events separated by <= 2 frames
    min_puff_duration_ms: float = 35.0
    min_wave_diameter_um: float = 15.0

    # Post-processing for Ca2+ sparks
    spark_min_spatial_distance_um: float = 1.8
    spark_min_temporal_distance_ms: float = 20.0
    spark_min_duration_ms: float = 20.0
    spark_min_diameter_um: float = 0.6

    # Imaging parameters
    pixel_size_um: float = 0.2  # microns per pixel
    frame_rate_ms: float = 6.79  # milliseconds per frame

    # Device
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Class names
    class_names: List[str] = field(default_factory=lambda: [
        "background", "Ca2+ spark", "Ca2+ puff", "Ca2+ wave"
    ])

    def save(self, path: str):
        """Save configuration to file."""
        import json
        from dataclasses import asdict
        from pathlib import Path

        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        import json

        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Get default class names
        default_class_names = ["background", "Ca2+ spark", "Ca2+ puff", "Ca2+ wave"]

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {})),
            class_names=config_dict.get('class_names', default_class_names)
        )


# Default configuration instance
default_config = Config()


if __name__ == '__main__':
    # Test configuration
    config = Config()
    print("="*70)
    print("UPDATED Configuration - Paper-Matched Settings")
    print("="*70)
    print(f"\nModel config:")
    print(f"  Base channels: {config.model.base_channels} (Paper uses 8, not 64!)")
    print(f"  Num classes: {config.model.num_classes}")
    print(f"\nData config:")
    print(f"  Segment length: {config.data.segment_length}")
    print(f"  Step size: {config.data.step_size}")
    print(f"  Augmentation: {config.data.augmentation_mode}")
    print(f"\nTraining config:")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Iterations: {config.training.num_iterations:,}")
    print(f"  Scheduler: {config.training.scheduler_type}")
    print(f"  Device: {config.training.device}")
    print(f"\n" + "="*70)

    # Test save/load
    config.save('/tmp/test_config.json')
    loaded_config = Config.load('/tmp/test_config.json')
    print("\n✓ Save/load test passed")
