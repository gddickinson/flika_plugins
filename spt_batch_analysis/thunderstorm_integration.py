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
            - photons_per_adu: photoelectrons per A/D count
            - baseline: camera baseline offset in ADU
            - is_em_gain: whether EM gain is enabled
            - em_gain: EM multiplication gain
            - quantum_efficiency: sensor quantum efficiency (0-1)
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
            'photons_per_adu': 3.6,
            'baseline': 100.0,
            'is_em_gain': True,
            'em_gain': 100.0,
            'quantum_efficiency': 1.0,
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

        # Note: we pass baseline=0 and em_gain=1 to the pipeline because
        # we handle offset subtraction and photon conversion ourselves
        # in detect_and_fit() and save_localizations() respectively.
        is_emccd = self.params.get('is_em_gain', False)
        try:
            self.pipeline = ThunderSTORM(
                filter_type=self.params['filter_type'],
                filter_params=filter_params,
                detector_type=self.params['detector_type'],
                detector_params=detector_params,
                fitter_type=self.params['fitter_type'],
                fitter_params=fitter_params,
                threshold_expression=self.params['detector_threshold'],
                pixel_size=self.params['pixel_size'],
                photons_per_adu=1.0,
                baseline=0.0,
                em_gain=1.0,
                is_emccd=is_emccd
            )
        except TypeError:
            # Fallback if ThunderSTORM hasn't been reloaded with is_emccd param
            self.pipeline = ThunderSTORM(
                filter_type=self.params['filter_type'],
                filter_params=filter_params,
                detector_type=self.params['detector_type'],
                detector_params=detector_params,
                fitter_type=self.params['fitter_type'],
                fitter_params=fitter_params,
                threshold_expression=self.params['detector_threshold'],
                pixel_size=self.params['pixel_size'],
                photons_per_adu=1.0,
                baseline=0.0,
                em_gain=1.0
            )
        # Set EMCCD flag directly on the fitter (works even if module cached)
        self.pipeline.fitter.is_emccd = is_emccd

    def _adu_to_photons(self, digital_counts):
        """Convert digital counts (offset-subtracted ADU) to photons.

        Uses the same formula as thunderSTORM's CameraSetupPlugIn:
            photons = digital_counts * photons_per_adu / quantum_efficiency / em_gain

        Parameters
        ----------
        digital_counts : float or ndarray
            Intensity values in offset-subtracted ADU

        Returns
        -------
        photons : float or ndarray
        """
        photons_per_adu = self.params['photons_per_adu']
        qe = self.params['quantum_efficiency']
        if self.params.get('is_em_gain', False):
            em_gain = self.params['em_gain']
        else:
            em_gain = 1.0
        return digital_counts * photons_per_adu / qe / em_gain

    def detect_and_fit(self, image_stack, show_progress=True):
        """Detect and fit particles in an image stack

        The camera baseline offset is subtracted from the image before
        filtering and fitting, matching the behaviour of thunderSTORM
        in ImageJ.

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
            - 'intensity': fitted Gaussian amplitude (offset-subtracted ADU)
            - 'background': local background level (offset-subtracted ADU)
            - 'sigma_x', 'sigma_y': PSF widths in nm
            - 'uncertainty': localization precision in nm
            - 'chi_squared': goodness of fit
        """
        # Ensure 3D stack
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        # Subtract camera baseline offset before fitting
        # (matches real thunderSTORM which does this in AnalysisPlugIn.java)
        baseline = self.params['baseline']
        if baseline != 0:
            image_stack = image_stack.astype(np.float32) - float(baseline)

        # Run thunderSTORM analysis
        localizations = self.pipeline.analyze_stack(
            image_stack,
            fit_radius=self.params['fit_radius'],
            show_progress=show_progress
        )

        return localizations

    def _compute_bkgstd(self, image_stack, localizations):
        """Compute background standard deviation in the fitting region.

        For each localization, compute the std of pixels in the local
        ROI of the frame where it was detected. This matches
        thunderSTORM's bkgstd column.

        Parameters
        ----------
        image_stack : ndarray
            3D image stack (already offset-subtracted)
        localizations : dict
            Localization results with 'x', 'y', 'frame' keys (x,y in nm)

        Returns
        -------
        bkgstd : ndarray
            Background standard deviation for each localization
        """
        pixel_size = self.params['pixel_size']
        fit_radius = self.params['fit_radius']
        n_locs = len(localizations['x'])
        bkgstd = np.zeros(n_locs)

        for i in range(n_locs):
            frame_idx = int(localizations['frame'][i])
            # Convert nm back to pixels for ROI extraction
            col = localizations['x'][i] / pixel_size
            row = localizations['y'][i] / pixel_size

            if frame_idx < 0 or frame_idx >= image_stack.shape[0]:
                continue

            frame = image_stack[frame_idx]
            h, w = frame.shape

            r0 = max(0, int(row) - fit_radius)
            r1 = min(h, int(row) + fit_radius + 1)
            c0 = max(0, int(col) - fit_radius)
            c1 = min(w, int(col) + fit_radius + 1)

            if r1 > r0 and c1 > c0:
                roi = frame[r0:r1, c0:c1]
                bkgstd[i] = float(np.std(roi))

        return bkgstd

    def save_localizations(self, localizations, output_path, image_stack=None):
        """Save localizations in SPT-compatible format with all thunderSTORM columns.

        Output columns match real thunderSTORM format:
        - id, frame, x [nm], y [nm], sigma [nm], intensity [AU],
          intensity [photon], offset [photon], bkgstd [photon],
          chi2, uncertainty [nm]

        The intensity column uses [AU] (arbitrary units) since these are
        the raw Gaussian amplitude values from fitting (in offset-subtracted
        ADU), not calibrated photon counts.

        Parameters
        ----------
        localizations : dict
            Localization results from detect_and_fit
        output_path : str or Path
            Output CSV file path (will add _locsID.csv suffix if not present)
        image_stack : ndarray, optional
            Original image stack for computing bkgstd. If None, bkgstd
            will be estimated from the background values.
        """
        # Prepare output path
        output_path = Path(output_path)
        if not str(output_path).endswith('_locsID.csv'):
            output_path = output_path.parent / (output_path.stem + '_locsID.csv')

        n_locs = len(localizations['x'])

        # Compute sigma [nm] — average of sigma_x and sigma_y
        sigma_x = localizations.get('sigma_x', np.zeros(n_locs))
        sigma_y = localizations.get('sigma_y', np.zeros(n_locs))
        if sigma_x is not None and sigma_y is not None:
            sigma_nm = (sigma_x + sigma_y) / 2.0
        elif sigma_x is not None:
            sigma_nm = sigma_x
        else:
            sigma_nm = np.zeros(n_locs)

        # intensity [AU] — raw fitted Gaussian amplitude (offset-subtracted ADU)
        intensity_au = localizations.get('intensity', np.zeros(n_locs))

        # intensity [photon] — fitted amplitude converted to photons
        intensity_photon = self._adu_to_photons(intensity_au)

        # offset [photon] — local background converted to photons
        background_adu = localizations.get('background', np.zeros(n_locs))
        offset_photon = self._adu_to_photons(background_adu)

        # bkgstd [photon] — background standard deviation converted to photons
        if image_stack is not None:
            # Subtract baseline for consistency (detect_and_fit already did this)
            baseline = self.params['baseline']
            if baseline != 0:
                stack_for_bkgstd = image_stack.astype(np.float32) - float(baseline)
            else:
                stack_for_bkgstd = image_stack
            bkgstd_adu = self._compute_bkgstd(stack_for_bkgstd, localizations)
        else:
            # Estimate from background values if image stack not available
            bkgstd_adu = np.zeros(n_locs)
        bkgstd_photon = self._adu_to_photons(bkgstd_adu)

        # chi2 — goodness of fit
        chi2 = localizations.get('chi_squared', np.zeros(n_locs))

        # uncertainty [nm] — localization precision
        uncertainty_nm = localizations.get('uncertainty', np.zeros(n_locs))

        # Create DataFrame matching thunderSTORM column format
        df = pd.DataFrame({
            'id': np.arange(1, n_locs + 1, dtype=float),
            'frame': localizations['frame'] + 1,  # Convert to 1-indexed
            'x [nm]': localizations['x'],
            'y [nm]': localizations['y'],
            'sigma [nm]': sigma_nm,
            'intensity [AU]': intensity_au,
            'intensity [photon]': intensity_photon,
            'offset [photon]': offset_photon,
            'bkgstd [photon]': bkgstd_photon,
            'chi2': chi2,
            'uncertainty [nm]': uncertainty_nm,
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
            'photons_per_adu': gui_params.get('ts_photons_per_adu', 3.6),
            'baseline': gui_params.get('ts_baseline', 100.0),
            'is_em_gain': gui_params.get('ts_is_em_gain', True),
            'em_gain': gui_params.get('ts_em_gain', 100.0),
            'quantum_efficiency': gui_params.get('ts_quantum_efficiency', 1.0),
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
