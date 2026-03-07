#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThunderSTORM Integration Module for SPT Batch Analysis
Provides thunderSTORM-based particle detection as an alternative to U-Track
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os, sys


# Add the current directory to Python path to ensure import thunderstorm_python works acrosss differnt platforms
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # Try to import thunderSTORM Python package
    from thunderstorm_python import ThunderSTORM, create_default_pipeline, filters, detection, fitting
    THUNDERSTORM_AVAILABLE = True
except ImportError:
    THUNDERSTORM_AVAILABLE = False
    print("Warning: thunderstorm_python package not found. ThunderSTORM detection will be disabled.")


class ThunderSTORMDetector:
    """Wrapper for thunderSTORM detection compatible with SPT Batch Analysis pipeline"""

    def __init__(self, parameters=None):
        """Initialize thunderSTORM detector with parameters

        Parameters
        ----------
        parameters : dict, optional
            ThunderSTORM parameters including:
            - filter_type: 'wavelet', 'gaussian', 'dog', etc.
            - filter_scale: scale for wavelet filter
            - detector_type: 'local_maximum', 'non_maximum_suppression', etc.
            - detector_threshold: detection threshold expression
            - fitter_type: 'gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle', etc.
            - fit_radius: fitting radius in pixels
            - pixel_size: camera pixel size in nm
            - initial_sigma: initial PSF sigma estimate
        """
        if not THUNDERSTORM_AVAILABLE:
            raise ImportError("thunderstorm_python package is required for thunderSTORM detection")

        # Default parameters
        self.params = {
            'filter_type': 'wavelet',
            'filter_scale': 2,
            'detector_type': 'local_maximum',
            'detector_threshold': 'std(Wave.F1)',
            'fitter_type': 'gaussian_lsq',
            'fit_radius': 3,
            'pixel_size': 108.0,  # nm
            'initial_sigma': 1.3,
            'photons_per_adu': 1.0,
            'baseline': 100.0,
            'em_gain': 1.0
        }

        # Update with provided parameters
        if parameters:
            self.params.update(parameters)

        # Create thunderSTORM pipeline
        self.pipeline = None
        self._create_pipeline()

    def _create_pipeline(self):
        """Create thunderSTORM analysis pipeline"""
        # Create filter parameters
        filter_params = {}
        if self.params['filter_type'] == 'wavelet':
            filter_params = {'scale': self.params['filter_scale']}

        # Create detector parameters
        detector_params = {}

        # Create fitter parameters
        fitter_params = {'initial_sigma': self.params['initial_sigma']}

        # Create pipeline
        self.pipeline = ThunderSTORM(
            filter_type=self.params['filter_type'],
            filter_params=filter_params,
            detector_type=self.params['detector_type'],
            detector_params=detector_params,
            fitter_type=self.params['fitter_type'],
            fitter_params=fitter_params,
            threshold_expression=self.params['detector_threshold'],
            pixel_size=self.params['pixel_size'],
            photons_per_adu=self.params['photons_per_adu'],
            baseline=self.params['baseline'],
            em_gain=self.params['em_gain']
        )

    def detect_and_fit(self, image_stack, show_progress=True):
        """Detect and fit particles in an image stack

        Parameters
        ----------
        image_stack : ndarray
            3D array (frames, height, width) or 2D array for single frame
        show_progress : bool
            Show progress bar during analysis

        Returns
        -------
        localizations : dict
            Dictionary with keys:
            - 'x': x positions in nm
            - 'y': y positions in nm
            - 'frame': frame numbers (0-indexed)
            - 'intensity': fitted intensities
            - 'background': local backgrounds
            - 'sigma_x', 'sigma_y': PSF widths in nm
            - 'uncertainty': localization precision in nm
            - 'chi_squared': goodness of fit
        """
        # Ensure 3D stack
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        # Run thunderSTORM analysis
        localizations = self.pipeline.analyze_stack(
            image_stack,
            fit_radius=self.params['fit_radius'],
            show_progress=show_progress
        )

        return localizations

    def save_localizations(self, localizations, output_path):
        """Save localizations in SPT-compatible format

        Parameters
        ----------
        localizations : dict
            Localization results from detect_and_fit
        output_path : str or Path
            Output CSV file path (will add _locsID.csv suffix if not present)
        """
        # Prepare output path
        output_path = Path(output_path)
        if not str(output_path).endswith('_locsID.csv'):
            output_path = output_path.parent / (output_path.stem + '_locsID.csv')

        # Create DataFrame in SPT-compatible format
        # SPT format: frame, x [nm], y [nm], intensity [photon], id
        df = pd.DataFrame({
            'frame': localizations['frame'] + 1,  # Convert to 1-indexed for compatibility
            'x [nm]': localizations['x'],
            'y [nm]': localizations['y'],
            'intensity [photon]': localizations['intensity'],
            'id': np.arange(len(localizations['x']))  # Sequential IDs
        })

        # Save to CSV
        df.to_csv(output_path, index=False)

        return output_path

    @staticmethod
    def create_from_gui_parameters(gui_params):
        """Create detector from GUI parameters dictionary

        Parameters
        ----------
        gui_params : dict
            Parameters from the GUI including all thunderSTORM settings

        Returns
        -------
        detector : ThunderSTORMDetector
            Configured detector instance
        """
        params = {
            'filter_type': gui_params.get('ts_filter_type', 'wavelet'),
            'filter_scale': gui_params.get('ts_filter_scale', 2),
            'detector_type': gui_params.get('ts_detector_type', 'local_maximum'),
            'detector_threshold': gui_params.get('ts_detector_threshold', 'std(Wave.F1)'),
            'fitter_type': gui_params.get('ts_fitter_type', 'gaussian_lsq'),
            'fit_radius': gui_params.get('ts_fit_radius', 3),
            'pixel_size': gui_params.get('pixel_size', 108.0),
            'initial_sigma': gui_params.get('ts_initial_sigma', 1.3),
            'photons_per_adu': gui_params.get('ts_photons_per_adu', 1.0),
            'baseline': gui_params.get('ts_baseline', 100.0),
            'em_gain': gui_params.get('ts_em_gain', 1.0)
        }

        return ThunderSTORMDetector(parameters=params)


def is_thunderstorm_available():
    """Check if thunderSTORM is available"""
    return THUNDERSTORM_AVAILABLE


def get_available_filters():
    """Get list of available filter types"""
    return ['wavelet', 'gaussian', 'dog', 'lowered_gaussian', 'difference_of_gaussians']


def get_available_detectors():
    """Get list of available detector types"""
    return ['local_maximum', 'non_maximum_suppression', 'centroid']


def get_available_fitters():
    """Get list of available PSF fitters"""
    return ['gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle', 'radial_symmetry', 'centroid']


def get_default_threshold_expressions():
    """Get list of common threshold expressions"""
    return [
        'std(Wave.F1)',
        '2*std(Wave.F1)',
        '3*std(Wave.F1)',
        'mean(Wave.F1) + 3*std(Wave.F1)',
        '100'  # Fixed threshold
    ]
