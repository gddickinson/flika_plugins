"""
Calcium Event Detector
Deep learning-based detection and classification of local Ca2+ release events.
"""

__version__ = "1.0.0"

from .models.unet3d import UNet3D
from .configs.config import Config
from .inference.detect import CalciumEventDetector

__all__ = ['UNet3D', 'Config', 'CalciumEventDetector']
