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
    # Force reload of submodules to pick up code changes without restarting FLIKA
    import importlib
    import thunderstorm_python
    from thunderstorm_python import filters, detection, fitting
    from thunderstorm_python import pipeline as _pipeline_mod
    for _mod in [filters, detection, fitting, _pipeline_mod, thunderstorm_python]:
        importlib.reload(_mod)
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
            'filter_order': 3,
            'filter_sigma': 1.6,
            'filter_sigma1': 1.0,
            'filter_sigma2': 1.6,
            'filter_size1': 3,
            'filter_size2': 5,
            'filter_pattern': 'box',
            'detector_type': 'local_maximum',
            'detector_threshold': 'std(Wave.F1)',
            'detector_connectivity': '8-neighbourhood',
            'detector_radius': 1,
            'fitter_type': 'gaussian_lsq',
            'fit_radius': 3,
            'pixel_size': 108.0,  # nm
            'initial_sigma': 1.3,
            'photons_per_adu': 3.6,
            'baseline': 100.0,
            'is_em_gain': True,
            'em_gain': 100.0,
            'quantum_efficiency': 1.0,
            'use_watershed': True,
            'multi_emitter_enabled': False,
            'multi_emitter_max': 5,
            'multi_emitter_pvalue': 1e-6,
            'multi_emitter_keep_same_intensity': True,
            'multi_emitter_fixed_intensity': False,
            'multi_emitter_intensity_min': 500,
            'multi_emitter_intensity_max': 2500,
            'mfa_fitting_method': 'wlsq',  # 'wlsq' (matches ImageJ) or 'mle'
            'mfa_model_selection_iterations': 50,  # iterations during model comparison
            'mfa_enable_final_refit': True,  # re-fit winning model with no iter limit
            'sigma_min': None,        # min sigma in pixels (None = no limit)
            'sigma_max': None,        # max sigma in pixels (None = no limit)
            'fwhm_min_nm': None,      # min FWHM in nm (alternative to sigma_min)
            'fwhm_max_nm': None,      # max FWHM in nm (alternative to sigma_max)
        }

        # Update with provided parameters
        if parameters:
            self.params.update(parameters)

        # Create thunderSTORM pipeline
        self.pipeline = None
        self._create_pipeline()

    def _create_pipeline(self):
        """Create thunderSTORM analysis pipeline"""
        # Create filter parameters based on filter type
        filter_type = self.params['filter_type']
        filter_params = {}
        if filter_type == 'wavelet':
            filter_params = {
                'scale': self.params['filter_scale'],
                'order': self.params.get('filter_order', 3),
            }
        elif filter_type == 'gaussian':
            filter_params = {'sigma': self.params.get('filter_sigma', 1.6)}
        elif filter_type == 'dog':
            filter_params = {
                'sigma1': self.params.get('filter_sigma1', 1.0),
                'sigma2': self.params.get('filter_sigma2', 1.6),
            }
        elif filter_type == 'lowered_gaussian':
            filter_params = {'sigma': self.params.get('filter_sigma', 1.6)}
        elif filter_type == 'diff_avg':
            filter_params = {
                'size1': self.params.get('filter_size1', 3),
                'size2': self.params.get('filter_size2', 5),
            }
        elif filter_type == 'median':
            filter_params = {
                'size': self.params.get('filter_size1', 3),
                'pattern': self.params.get('filter_pattern', 'box'),
            }
        elif filter_type == 'box':
            filter_params = {'size': self.params.get('filter_size1', 3)}

        # Create detector parameters
        detector_type = self.params['detector_type']
        detector_params = {}
        if detector_type == 'centroid':
            detector_params['use_watershed'] = self.params.get('use_watershed', True)
        if detector_type == 'non_maximum_suppression':
            detector_params['radius'] = self.params.get('detector_radius', 1)
        # Pass connectivity for all detector types
        conn = self.params.get('detector_connectivity', '8-neighbourhood')
        if detector_type == 'local_maximum':
            detector_params['connectivity'] = conn
        elif detector_type in ('non_maximum_suppression', 'centroid'):
            # These use integer connectivity (1 or 2)
            detector_params['connectivity'] = 2 if '8' in str(conn) else 1

        # Determine effective fitter type
        fitter_type = self.params['fitter_type']
        if self.params.get('multi_emitter_enabled', False):
            fitter_type = 'multi_emitter'

        # Create fitter parameters
        fitter_params = {'initial_sigma': self.params['initial_sigma']}
        if fitter_type == 'multi_emitter':
            fitter_params['max_emitters'] = self.params.get('multi_emitter_max', 5)
            fitter_params['p_value_threshold'] = self.params.get('multi_emitter_pvalue', 1e-6)
            fitter_params['keep_same_intensity'] = self.params.get('multi_emitter_keep_same_intensity', True)
            fitter_params['fixed_intensity'] = self.params.get('multi_emitter_fixed_intensity', False)
            fitter_params['intensity_range'] = (
                self.params.get('multi_emitter_intensity_min', 500),
                self.params.get('multi_emitter_intensity_max', 2500),
            )
            fitter_params['mfa_fitting_method'] = self.params.get('mfa_fitting_method', 'wlsq')
            fitter_params['mfa_model_selection_iterations'] = self.params.get('mfa_model_selection_iterations', 50)
            fitter_params['mfa_enable_final_refit'] = self.params.get('mfa_enable_final_refit', True)

        # Build sigma/FWHM range constraint
        sigma_range = None
        fwhm_range_nm = None
        s_min = self.params.get('sigma_min')
        s_max = self.params.get('sigma_max')
        if s_min is not None or s_max is not None:
            sigma_range = (s_min, s_max)
        else:
            f_min = self.params.get('fwhm_min_nm')
            f_max = self.params.get('fwhm_max_nm')
            if f_min is not None or f_max is not None:
                fwhm_range_nm = (f_min, f_max)

        fit_radius = self.params['fit_radius']

        # Note: we pass baseline=0 and em_gain=1 to the pipeline because
        # we handle offset subtraction and photon conversion ourselves
        # in detect_and_fit() and save_localizations() respectively.
        is_emccd = self.params.get('is_em_gain', False)
        self.pipeline = ThunderSTORM(
            filter_type=filter_type,
            filter_params=filter_params,
            detector_type=detector_type,
            detector_params=detector_params,
            fitter_type=fitter_type,
            fitter_params=fitter_params,
            threshold_expression=self.params['detector_threshold'],
            pixel_size=self.params['pixel_size'],
            photons_per_adu=1.0,
            baseline=0.0,
            em_gain=1.0,
            is_emccd=is_emccd,
            sigma_range=sigma_range,
            fwhm_range_nm=fwhm_range_nm,
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
        """Compute background standard deviation from fit residuals.

        For each localization, compute the RMS of (data - model) in the
        fitting ROI, where the model is the integrated Gaussian PSF with
        the fitted parameters.  This matches ThunderSTORM's bkgstd column
        (noise estimated from fit residuals, not raw ROI statistics).

        Parameters
        ----------
        image_stack : ndarray
            3D image stack (already offset-subtracted)
        localizations : dict
            Localization results with 'x', 'y', 'frame', 'intensity',
            'background', 'sigma_x', 'sigma_y' keys (x,y in nm)

        Returns
        -------
        bkgstd : ndarray
            Background standard deviation for each localization (in ADU)
        """
        from thunderstorm_python.fitting import integrated_gaussian_value

        pixel_size = self.params['pixel_size']
        fit_radius = self.params['fit_radius']
        n_locs = len(localizations['x'])
        bkgstd = np.zeros(n_locs)

        # Get fit parameters for residual computation
        intensities = localizations.get('intensity', np.zeros(n_locs))
        backgrounds = localizations.get('background', np.zeros(n_locs))
        sigma_x = localizations.get('sigma_x', np.ones(n_locs) * 1.3 * pixel_size)
        sigma_y = localizations.get('sigma_y', np.ones(n_locs) * 1.3 * pixel_size)

        for i in range(n_locs):
            frame_idx = int(localizations['frame'][i])
            # Convert nm back to pixels for ROI extraction
            # Undo the +0.5 offset applied during pipeline conversion
            col_px = localizations['x'][i] / pixel_size - 0.5
            row_px = localizations['y'][i] / pixel_size - 0.5

            if frame_idx < 0 or frame_idx >= image_stack.shape[0]:
                continue

            frame = image_stack[frame_idx]
            h, w = frame.shape

            col_int = int(round(col_px))
            row_int = int(round(row_px))
            r0 = max(0, row_int - fit_radius)
            r1 = min(h, row_int + fit_radius + 1)
            c0 = max(0, col_int - fit_radius)
            c1 = min(w, col_int + fit_radius + 1)

            if r1 <= r0 or c1 <= c0:
                continue

            roi = frame[r0:r1, c0:c1].astype(float)

            # Compute model using fitted parameters
            # Position relative to ROI origin
            x0_local = col_px - c0
            y0_local = row_px - r0
            sigma_px = (sigma_x[i] + sigma_y[i]) / (2.0 * pixel_size)
            intensity_val = intensities[i]
            bg_val = backgrounds[i]

            ny, nx = roi.shape
            residuals = np.zeros(ny * nx)
            k = 0
            for ii in range(ny):
                for jj in range(nx):
                    model_val = integrated_gaussian_value(
                        float(jj), float(ii),
                        x0_local, y0_local,
                        sigma_px, intensity_val, bg_val
                    )
                    residuals[k] = roi[ii, jj] - model_val
                    k += 1

            bkgstd[i] = float(np.std(residuals[:k]))

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

        # uncertainty [nm] — recompute using actual bkgstd (matching ThunderSTORM)
        # ThunderSTORM uses bkgstd (not offset) as the background noise parameter
        from thunderstorm_python.fitting import compute_localization_precision
        is_emccd = self.params.get('is_em_gain', False)
        # Determine fitting method
        fitter_type = self.params.get('fitter_type', 'gaussian_wlsq')
        if 'lsq' in fitter_type and 'wlsq' not in fitter_type:
            fit_method = 'lsq'
        elif 'mle' in fitter_type:
            fit_method = 'mle'
        else:
            fit_method = 'wlsq'
        pixel_size = self.params['pixel_size']
        uncertainty_nm = np.zeros(n_locs)
        for i in range(n_locs):
            uncertainty_nm[i] = compute_localization_precision(
                intensity_photon[i],
                bkgstd_photon[i],
                sigma_nm[i] / pixel_size,  # sigma in pixels
                1.0,  # pixel_size=1 for pixel-unit computation
                fitting_method=fit_method,
                is_emccd=is_emccd
            ) * pixel_size  # convert back to nm

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
            # Filter parameters
            'filter_type': gui_params.get('ts_filter_type', 'wavelet'),
            'filter_scale': gui_params.get('ts_filter_scale', 2),
            'filter_order': gui_params.get('ts_filter_order', 3),
            'filter_sigma': gui_params.get('ts_filter_sigma', 1.6),
            'filter_sigma1': gui_params.get('ts_filter_sigma1', 1.0),
            'filter_sigma2': gui_params.get('ts_filter_sigma2', 1.6),
            'filter_size1': gui_params.get('ts_filter_size1', 3),
            'filter_size2': gui_params.get('ts_filter_size2', 5),
            'filter_pattern': gui_params.get('ts_filter_pattern', 'box'),
            # Detector parameters
            'detector_type': gui_params.get('ts_detector_type', 'local_maximum'),
            'detector_threshold': gui_params.get('ts_detector_threshold', 'std(Wave.F1)'),
            'detector_connectivity': gui_params.get('ts_detector_connectivity', '8-neighbourhood'),
            'detector_radius': gui_params.get('ts_detector_radius', 1),
            # Fitter parameters
            'fitter_type': gui_params.get('ts_fitter_type', 'gaussian_lsq'),
            'fit_radius': gui_params.get('ts_fit_radius', 3),
            'pixel_size': gui_params.get('pixel_size', 108.0),
            'initial_sigma': gui_params.get('ts_initial_sigma', 1.3),
            # Camera parameters
            'photons_per_adu': gui_params.get('ts_photons_per_adu', 3.6),
            'baseline': gui_params.get('ts_baseline', 100.0),
            'is_em_gain': gui_params.get('ts_is_em_gain', True),
            'em_gain': gui_params.get('ts_em_gain', 100.0),
            'quantum_efficiency': gui_params.get('ts_quantum_efficiency', 1.0),
            # Detection options
            'use_watershed': gui_params.get('ts_use_watershed', True),
            # Multi-emitter
            'multi_emitter_enabled': gui_params.get('ts_multi_emitter_enabled', False),
            'multi_emitter_max': gui_params.get('ts_multi_emitter_max', 5),
            'multi_emitter_pvalue': gui_params.get('ts_multi_emitter_pvalue', 1e-6),
            'multi_emitter_keep_same_intensity': gui_params.get('ts_multi_emitter_keep_same_intensity', True),
            'multi_emitter_fixed_intensity': gui_params.get('ts_multi_emitter_fixed_intensity', False),
            'multi_emitter_intensity_min': gui_params.get('ts_multi_emitter_intensity_min', 500),
            'multi_emitter_intensity_max': gui_params.get('ts_multi_emitter_intensity_max', 2500),
            'mfa_fitting_method': gui_params.get('ts_mfa_fitting_method', 'wlsq'),
            'mfa_model_selection_iterations': gui_params.get('ts_mfa_model_selection_iterations', 50),
            'mfa_enable_final_refit': gui_params.get('ts_mfa_enable_final_refit', True),
            # Sigma / FWHM range constraint
            'sigma_min': gui_params.get('ts_sigma_min', None),
            'sigma_max': gui_params.get('ts_sigma_max', None),
            'fwhm_min_nm': gui_params.get('ts_fwhm_min_nm', None),
            'fwhm_max_nm': gui_params.get('ts_fwhm_max_nm', None),
        }

        return ThunderSTORMDetector(parameters=params)


def is_thunderstorm_available():
    """Check if thunderSTORM is available"""
    return THUNDERSTORM_AVAILABLE


def get_available_filters():
    """Get list of available filter types"""
    return ['wavelet', 'gaussian', 'dog', 'lowered_gaussian',
            'diff_avg', 'median', 'box', 'none']


def get_available_detectors():
    """Get list of available detector types"""
    return ['local_maximum', 'non_maximum_suppression', 'centroid']


def get_available_fitters():
    """Get list of available PSF fitters"""
    return ['gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
            'elliptical_gaussian_mle', 'multi_emitter', 'phasor',
            'radial_symmetry', 'centroid']


def get_default_threshold_expressions():
    """Get list of common threshold expressions"""
    return [
        'std(Wave.F1)',
        '2*std(Wave.F1)',
        '3*std(Wave.F1)',
        'mean(Wave.F1) + 3*std(Wave.F1)',
        '100'  # Fixed threshold
    ]
