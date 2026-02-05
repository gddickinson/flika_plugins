#!/usr/bin/env python3
"""
Unified parameter class definitions for u-track Python port.

This module provides consistent parameter structures used across different
components of the detection pipeline.

Copyright (C) 2025, Danuser Lab - UTSouthwestern
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List
import numpy as np


@dataclass
class DetectionParam:
    """
    Detection parameters for u-track feature detection.
    
    This class matches the expected interface for detect_sub_res_features_2d_standalone
    and other detection functions.
    """
    # Core detection parameters
    psf_sigma: float = 1.0
    calc_method: str = 'g'  # 'g' for Gaussian, 'gv' for variable sigma, 'c' for centroid
    
    # Statistical test parameters
    test_alpha: Dict[str, float] = field(default_factory=lambda: {
        'alphaR': 0.05,    # F-test for model improvement
        'alphaA': 0.05,    # Amplitude significance test
        'alphaD': 0.05,    # Distance significance test
        'alphaF': 0.0      # Final test (usually 0)
    })
    
    # Local maxima detection
    alpha_loc_max: float = 0.05
    
    # Processing options
    visual: bool = False
    do_mmf: bool = True
    bit_depth: int = 16
    num_sigma_iter: int = 0
    integ_window: Union[int, List[int]] = 0
    
    # Background and ROI
    background: Optional['BackgroundParam'] = None
    roi_mask: Optional[np.ndarray] = None
    mask_loc: Optional[str] = None
    
    def __post_init__(self):
        """Ensure test_alpha is properly initialized."""
        if self.test_alpha is None:
            self.test_alpha = {
                'alphaR': 0.05,
                'alphaA': 0.05, 
                'alphaD': 0.05,
                'alphaF': 0.0
            }
        
        # Ensure all required keys exist
        required_keys = ['alphaR', 'alphaA', 'alphaD', 'alphaF']
        for key in required_keys:
            if key not in self.test_alpha:
                self.test_alpha[key] = 0.05 if key != 'alphaF' else 0.0


@dataclass
class MovieParam:
    """
    Movie parameters for image sequence processing.
    
    This class defines how to access the image sequence, either from
    individual files or from a channel object.
    """
    # File-based image access
    image_dir: Optional[str] = None
    filename_base: str = ""
    digits_4_enum: int = 5
    
    # Frame range
    first_image_num: int = 1
    last_image_num: int = 1
    
    # Channel-based access (alternative to file-based)
    channel: Optional[object] = None
    
    # Computed properties
    image_exists: Optional[List[bool]] = None
    image_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Initialize computed properties."""
        if self.image_indices is None:
            self.image_indices = list(range(self.first_image_num, self.last_image_num + 1))
        
        if self.image_exists is None:
            num_images = self.last_image_num - self.first_image_num + 1
            self.image_exists = [True] * num_images


@dataclass 
class BackgroundParam:
    """
    Background estimation parameters.
    """
    image_dir: str = ""
    filename_base: str = ""
    alpha_loc_max_abs: float = 0.05


@dataclass
class SaveResults:
    """
    Parameters for saving detection results.
    """
    dir: str = "."
    filename: str = "detectedFeatures.mat"


# Convenience functions for creating parameter objects

def create_default_detection_params(
    psf_sigma: float = 1.0,
    lenient: bool = False
) -> DetectionParam:
    """
    Create detection parameters with sensible defaults.
    
    Args:
        psf_sigma: PSF sigma in pixels
        lenient: If True, use more lenient detection criteria
        
    Returns:
        DetectionParam object with appropriate settings
    """
    if lenient:
        alpha_values = {
            'alphaR': 0.1,
            'alphaA': 0.1, 
            'alphaD': 0.1,
            'alphaF': 0.0
        }
        alpha_loc_max = 0.1
    else:
        alpha_values = {
            'alphaR': 0.05,
            'alphaA': 0.05,
            'alphaD': 0.05, 
            'alphaF': 0.0
        }
        alpha_loc_max = 0.05
    
    return DetectionParam(
        psf_sigma=psf_sigma,
        test_alpha=alpha_values,
        alpha_loc_max=alpha_loc_max,
        do_mmf=True,
        bit_depth=16
    )


def create_movie_params_from_directory(
    image_dir: str,
    filename_base: str,
    first_frame: int = 1,
    last_frame: int = 1,
    digits: int = 5
) -> MovieParam:
    """
    Create movie parameters for file-based image access.
    
    Args:
        image_dir: Directory containing images
        filename_base: Base filename (without frame number and extension)
        first_frame: First frame number
        last_frame: Last frame number  
        digits: Number of digits in frame numbering
        
    Returns:
        MovieParam object for file-based access
    """
    return MovieParam(
        image_dir=image_dir,
        filename_base=filename_base,
        first_image_num=first_frame,
        last_image_num=last_frame,
        digits_4_enum=digits
    )


def create_movie_params_from_channel(
    channel: object,
    first_frame: int = 1,
    last_frame: int = 1
) -> MovieParam:
    """
    Create movie parameters for channel-based image access.
    
    Args:
        channel: Channel object that provides loadImage method
        first_frame: First frame number
        last_frame: Last frame number
        
    Returns:
        MovieParam object for channel-based access
    """
    return MovieParam(
        channel=channel,
        first_image_num=first_frame,
        last_image_num=last_frame
    )


# Legacy compatibility - aliases for old names
DetectParam = DetectionParam  # Alias for backward compatibility

# Example usage and testing
if __name__ == "__main__":
    print("Testing parameter classes...")
    
    # Test default detection parameters
    detect_params = create_default_detection_params(psf_sigma=1.2, lenient=True)
    print(f"Detection params: {detect_params}")
    
    # Test movie parameters
    movie_params = create_movie_params_from_directory(
        image_dir="/path/to/images",
        filename_base="frame_",
        first_frame=1,
        last_frame=10
    )
    print(f"Movie params: {movie_params}")
    
    print("Parameter classes test completed!")
