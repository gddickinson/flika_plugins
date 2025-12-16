#!/usr/bin/env python3
"""
Volume Processor Module
======================

Advanced image processing algorithms for 3D/4D volume data.
Includes motion correction, deconvolution, filtering, and analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage as ndi
from scipy import optimize, interpolate
from scipy.fft import fftn, ifftn, fftshift, ifftshift
from skimage import filters, restoration, segmentation, measure
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
import warnings
from qtpy.QtCore import QObject, Signal


class ProcessingMethod(Enum):
    """Available processing methods."""
    GAUSSIAN_FILTER = "gaussian_filter"
    MEDIAN_FILTER = "median_filter"
    BACKGROUND_SUBTRACTION = "background_subtraction"
    MOTION_CORRECTION = "motion_correction"
    DECONVOLUTION = "deconvolution"
    SHEAR_CORRECTION = "shear_correction"
    INTENSITY_NORMALIZATION = "intensity_normalization"
    EDGE_ENHANCEMENT = "edge_enhancement"
    NOISE_REDUCTION = "noise_reduction"


@dataclass
class ProcessingParameters:
    """Parameters for volume processing operations."""

    # Gaussian filtering
    gaussian_sigma: Union[float, Tuple[float, float, float]] = 1.0

    # Median filtering
    median_size: Union[int, Tuple[int, int, int]] = 3

    # Background subtraction
    background_method: str = "constant"  # 'constant', 'rolling_ball', 'top_hat'
    background_value: float = 100.0
    rolling_ball_radius: float = 10.0

    # Motion correction
    reference_frame: int = 0
    max_shift: int = 50
    upsample_factor: int = 1

    # Deconvolution
    deconv_method: str = "richardson_lucy"  # 'richardson_lucy', 'wiener'
    deconv_iterations: int = 10
    psf_sigma: Union[float, Tuple[float, float, float]] = 1.0
    noise_variance: float = 0.01

    # Shear correction
    shear_angle: float = 45.0
    shear_factor: float = 1.0
    interpolation_order: int = 1

    # Intensity processing
    normalization_method: str = "minmax"  # 'minmax', 'zscore', 'percentile'
    percentile_range: Tuple[float, float] = (1.0, 99.0)

    # Edge enhancement
    edge_method: str = "sobel"  # 'sobel', 'canny', 'laplacian'
    edge_sigma: float = 1.0

    # Noise reduction
    noise_method: str = "bilateral"  # 'bilateral', 'non_local_means', 'tv_denoise'
    bilateral_sigma_color: float = 0.1
    bilateral_sigma_spatial: float = 1.0


class VolumeProcessor(QObject):
    """
    Comprehensive volume processing engine for microscopy data.

    Features:
    - Motion correction with sub-pixel accuracy
    - 3D deconvolution algorithms
    - Advanced filtering and noise reduction
    - Geometric transformations
    - Intensity normalization
    - Batch processing capabilities
    - Progress reporting via signals
    """

    # Signals for progress reporting
    processing_started = Signal(str)  # method_name
    processing_progress = Signal(int, str)  # percentage, status
    processing_completed = Signal(str, np.ndarray)  # method_name, result
    processing_failed = Signal(str, str)  # method_name, error_message

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Processing state
        self._is_processing = False
        self._current_method = None

        # Cached results for efficiency
        self._cache = {}
        self._max_cache_size = 5

        # Default parameters
        self.default_params = ProcessingParameters()

    def process_volume(self, data: np.ndarray, method: ProcessingMethod,
                      params: Optional[ProcessingParameters] = None) -> np.ndarray:
        """
        Process volume data with specified method.

        Args:
            data: Input volume data (3D or 4D)
            method: Processing method to apply
            params: Processing parameters (uses defaults if None)

        Returns:
            Processed volume data
        """
        try:
            if params is None:
                params = self.default_params

            self._is_processing = True
            self._current_method = method.value

            self.logger.info(f"Starting {method.value} processing on data shape {data.shape}")
            self.processing_started.emit(method.value)

            # Validate input data
            self._validate_input_data(data)

            # Route to specific processing method
            processor_map = {
                ProcessingMethod.GAUSSIAN_FILTER: self._apply_gaussian_filter,
                ProcessingMethod.MEDIAN_FILTER: self._apply_median_filter,
                ProcessingMethod.BACKGROUND_SUBTRACTION: self._subtract_background,
                ProcessingMethod.MOTION_CORRECTION: self._correct_motion,
                ProcessingMethod.DECONVOLUTION: self._apply_deconvolution,
                ProcessingMethod.SHEAR_CORRECTION: self._correct_shear,
                ProcessingMethod.INTENSITY_NORMALIZATION: self._normalize_intensity,
                ProcessingMethod.EDGE_ENHANCEMENT: self._enhance_edges,
                ProcessingMethod.NOISE_REDUCTION: self._reduce_noise
            }

            processor = processor_map.get(method)
            if processor is None:
                raise ValueError(f"Unknown processing method: {method}")

            # Apply processing
            result = processor(data, params)

            self.logger.info(f"Processing completed successfully: {method.value}")
            self.processing_completed.emit(method.value, result)

            # Update cache
            self._update_cache(method.value, result)

            self._is_processing = False
            return result

        except Exception as e:
            self._is_processing = False
            error_msg = f"Processing failed: {str(e)}"
            self.logger.error(error_msg)
            self.processing_failed.emit(method.value, error_msg)
            raise

    def batch_process(self, data: np.ndarray, methods: List[Tuple[ProcessingMethod, ProcessingParameters]],
                     ) -> np.ndarray:
        """
        Apply multiple processing methods in sequence.

        Args:
            data: Input volume data
            methods: List of (method, parameters) tuples

        Returns:
            Final processed data
        """
        try:
            self.logger.info(f"Starting batch processing with {len(methods)} methods")

            result = data.copy()

            for i, (method, params) in enumerate(methods):
                self.processing_progress.emit(
                    int((i / len(methods)) * 100),
                    f"Applying {method.value} ({i+1}/{len(methods)})"
                )

                result = self.process_volume(result, method, params)

            self.processing_progress.emit(100, "Batch processing completed")
            self.logger.info("Batch processing completed successfully")

            return result

        except Exception as e:
            error_msg = f"Batch processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _validate_input_data(self, data: np.ndarray):
        """Validate input data."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be numpy array")

        if data.ndim < 3 or data.ndim > 4:
            raise ValueError(f"Data must be 3D or 4D, got {data.ndim}D")

        if data.size == 0:
            raise ValueError("Data array is empty")

        if not np.issubdtype(data.dtype, np.number):
            raise ValueError(f"Data must be numeric, got {data.dtype}")

        # Check for invalid values
        if np.any(np.isnan(data)):
            warnings.warn("Data contains NaN values, processing may fail")

        if np.any(np.isinf(data)):
            warnings.warn("Data contains infinite values, processing may fail")

    def _apply_gaussian_filter(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply 3D Gaussian filtering."""
        self.processing_progress.emit(10, "Applying Gaussian filter...")

        if data.ndim == 4:
            # Process each timepoint
            result = np.zeros_like(data)
            n_timepoints = data.shape[1]

            for t in range(n_timepoints):
                self.processing_progress.emit(
                    10 + int((t / n_timepoints) * 80),
                    f"Filtering timepoint {t+1}/{n_timepoints}"
                )
                result[:, t, :, :] = ndi.gaussian_filter(
                    data[:, t, :, :],
                    sigma=params.gaussian_sigma
                )
        else:
            # Single 3D volume
            result = ndi.gaussian_filter(data, sigma=params.gaussian_sigma)

        self.processing_progress.emit(100, "Gaussian filtering completed")
        return result

    def _apply_median_filter(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply 3D median filtering."""
        self.processing_progress.emit(10, "Applying median filter...")

        if data.ndim == 4:
            result = np.zeros_like(data)
            n_timepoints = data.shape[1]

            for t in range(n_timepoints):
                self.processing_progress.emit(
                    10 + int((t / n_timepoints) * 80),
                    f"Filtering timepoint {t+1}/{n_timepoints}"
                )
                result[:, t, :, :] = ndi.median_filter(
                    data[:, t, :, :],
                    size=params.median_size
                )
        else:
            result = ndi.median_filter(data, size=params.median_size)

        self.processing_progress.emit(100, "Median filtering completed")
        return result

    def _subtract_background(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Subtract background using various methods."""
        self.processing_progress.emit(10, "Subtracting background...")

        if params.background_method == "constant":
            result = np.maximum(data - params.background_value, 0)

        elif params.background_method == "rolling_ball":
            result = self._rolling_ball_background_subtraction(data, params.rolling_ball_radius)

        elif params.background_method == "top_hat":
            result = self._top_hat_background_subtraction(data, params.rolling_ball_radius)

        else:
            raise ValueError(f"Unknown background method: {params.background_method}")

        self.processing_progress.emit(100, "Background subtraction completed")
        return result

    def _correct_motion(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Correct motion artifacts using phase correlation."""
        if data.ndim != 4:
            raise ValueError("Motion correction requires 4D data (Z, T, Y, X)")

        self.processing_progress.emit(10, "Starting motion correction...")

        z_size, t_size, y_size, x_size = data.shape
        result = np.zeros_like(data)

        # Get reference frame
        if params.reference_frame >= t_size:
            params.reference_frame = 0

        reference = data[:, params.reference_frame, :, :]
        result[:, params.reference_frame, :, :] = reference

        shifts = []

        for t in range(t_size):
            if t == params.reference_frame:
                shifts.append((0, 0, 0))
                continue

            self.processing_progress.emit(
                10 + int((t / t_size) * 80),
                f"Correcting timepoint {t+1}/{t_size}"
            )

            current_volume = data[:, t, :, :]

            # Calculate shift for each Z slice and take median
            z_shifts = []
            for z in range(z_size):
                try:
                    shift, _, _ = phase_cross_correlation(
                        reference[z, :, :],
                        current_volume[z, :, :],
                        upsample_factor=params.upsample_factor
                    )

                    # Limit shifts to reasonable range
                    shift = np.clip(shift, -params.max_shift, params.max_shift)
                    z_shifts.append(shift)

                except Exception as e:
                    self.logger.warning(f"Phase correlation failed for t={t}, z={z}: {str(e)}")
                    z_shifts.append((0, 0))

            # Use median shift across Z slices
            if z_shifts:
                median_shift = np.median(z_shifts, axis=0)
                shifts.append(tuple(median_shift) + (0,))  # Add 0 for Z shift

                # Apply shift to entire volume
                for z in range(z_size):
                    shifted_slice = ndi.shift(
                        current_volume[z, :, :],
                        shift=median_shift,
                        order=1,
                        mode='constant',
                        cval=0
                    )
                    result[z, t, :, :] = shifted_slice
            else:
                shifts.append((0, 0, 0))
                result[:, t, :, :] = current_volume

        self.processing_progress.emit(100, "Motion correction completed")

        # Store shifts for later analysis
        self._cache['motion_shifts'] = shifts

        return result

    def _apply_deconvolution(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply 3D deconvolution."""
        self.processing_progress.emit(10, "Starting deconvolution...")

        # Generate PSF
        psf = self._generate_3d_psf(data.shape[-3:], params.psf_sigma)

        if params.deconv_method == "richardson_lucy":
            result = self._richardson_lucy_deconvolution(data, psf, params)
        elif params.deconv_method == "wiener":
            result = self._wiener_deconvolution(data, psf, params)
        else:
            raise ValueError(f"Unknown deconvolution method: {params.deconv_method}")

        self.processing_progress.emit(100, "Deconvolution completed")
        return result

    def _correct_shear(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Correct lightsheet shear artifacts."""
        self.processing_progress.emit(10, "Correcting shear artifacts...")

        # Convert angle to radians
        angle_rad = np.radians(params.shear_angle)

        if data.ndim == 4:
            result = np.zeros_like(data)
            z_size, t_size, y_size, x_size = data.shape

            for t in range(t_size):
                self.processing_progress.emit(
                    10 + int((t / t_size) * 80),
                    f"Correcting shear for timepoint {t+1}/{t_size}"
                )

                result[:, t, :, :] = self._apply_shear_transform_3d(
                    data[:, t, :, :], angle_rad, params.shear_factor, params.interpolation_order
                )
        else:
            result = self._apply_shear_transform_3d(
                data, angle_rad, params.shear_factor, params.interpolation_order
            )

        self.processing_progress.emit(100, "Shear correction completed")
        return result

    def _normalize_intensity(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Normalize image intensities."""
        self.processing_progress.emit(10, "Normalizing intensities...")

        if params.normalization_method == "minmax":
            data_min, data_max = np.min(data), np.max(data)
            result = (data - data_min) / (data_max - data_min) if data_max > data_min else data

        elif params.normalization_method == "zscore":
            data_mean, data_std = np.mean(data), np.std(data)
            result = (data - data_mean) / data_std if data_std > 0 else data

        elif params.normalization_method == "percentile":
            p_low, p_high = params.percentile_range
            v_low, v_high = np.percentile(data, [p_low, p_high])
            result = np.clip((data - v_low) / (v_high - v_low), 0, 1) if v_high > v_low else data

        else:
            raise ValueError(f"Unknown normalization method: {params.normalization_method}")

        self.processing_progress.emit(100, "Intensity normalization completed")
        return result

    def _enhance_edges(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Enhance edges in the data."""
        self.processing_progress.emit(10, "Enhancing edges...")

        if data.ndim == 4:
            result = np.zeros_like(data)
            n_timepoints = data.shape[1]

            for t in range(n_timepoints):
                self.processing_progress.emit(
                    10 + int((t / n_timepoints) * 80),
                    f"Processing timepoint {t+1}/{n_timepoints}"
                )
                result[:, t, :, :] = self._apply_edge_filter_3d(
                    data[:, t, :, :], params.edge_method, params.edge_sigma
                )
        else:
            result = self._apply_edge_filter_3d(data, params.edge_method, params.edge_sigma)

        self.processing_progress.emit(100, "Edge enhancement completed")
        return result

    def _reduce_noise(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Reduce noise using advanced filtering."""
        self.processing_progress.emit(10, "Reducing noise...")

        if params.noise_method == "bilateral":
            result = self._apply_bilateral_filter(data, params)
        elif params.noise_method == "non_local_means":
            result = self._apply_non_local_means(data, params)
        elif params.noise_method == "tv_denoise":
            result = self._apply_tv_denoising(data, params)
        else:
            raise ValueError(f"Unknown noise reduction method: {params.noise_method}")

        self.processing_progress.emit(100, "Noise reduction completed")
        return result

    # Helper methods for specific algorithms

    def _rolling_ball_background_subtraction(self, data: np.ndarray, radius: float) -> np.ndarray:
        """Apply rolling ball background subtraction."""
        # Simplified implementation - for production, use more sophisticated algorithm
        from skimage.morphology import white_tophat, disk

        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                for z in range(data.shape[0]):
                    selem = disk(int(radius))
                    result[z, t, :, :] = white_tophat(data[z, t, :, :], selem)
        else:
            result = np.zeros_like(data)
            selem = disk(int(radius))
            for z in range(data.shape[0]):
                result[z, :, :] = white_tophat(data[z, :, :], selem)

        return result

    def _top_hat_background_subtraction(self, data: np.ndarray, size: float) -> np.ndarray:
        """Apply morphological top-hat background subtraction."""
        from skimage.morphology import white_tophat, ball

        selem = ball(int(size))

        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                result[:, t, :, :] = white_tophat(data[:, t, :, :], selem)
        else:
            result = white_tophat(data, selem)

        return result

    def _generate_3d_psf(self, shape: Tuple[int, int, int], sigma: Union[float, Tuple]) -> np.ndarray:
        """Generate 3D Gaussian PSF."""
        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma, sigma)

        z_size, y_size, x_size = shape
        z, y, x = np.mgrid[0:z_size, 0:y_size, 0:x_size]

        z_center, y_center, x_center = np.array(shape) // 2

        psf = np.exp(-((z - z_center)**2 / (2 * sigma[0]**2) +
                      (y - y_center)**2 / (2 * sigma[1]**2) +
                      (x - x_center)**2 / (2 * sigma[2]**2)))

        return psf / np.sum(psf)  # Normalize

    def _richardson_lucy_deconvolution(self, data: np.ndarray, psf: np.ndarray,
                                     params: ProcessingParameters) -> np.ndarray:
        """Apply Richardson-Lucy deconvolution."""
        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                self.processing_progress.emit(
                    10 + int((t / data.shape[1]) * 80),
                    f"Deconvolving timepoint {t+1}/{data.shape[1]}"
                )
                result[:, t, :, :] = restoration.richardson_lucy(
                    data[:, t, :, :], psf, num_iter=params.deconv_iterations
                )
        else:
            result = restoration.richardson_lucy(data, psf, num_iter=params.deconv_iterations)

        return result

    def _wiener_deconvolution(self, data: np.ndarray, psf: np.ndarray,
                            params: ProcessingParameters) -> np.ndarray:
        """Apply Wiener deconvolution."""
        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                self.processing_progress.emit(
                    10 + int((t / data.shape[1]) * 80),
                    f"Deconvolving timepoint {t+1}/{data.shape[1]}"
                )
                result[:, t, :, :] = restoration.wiener(
                    data[:, t, :, :], psf, balance=params.noise_variance
                )
        else:
            result = restoration.wiener(data, psf, balance=params.noise_variance)

        return result

    def _apply_shear_transform_3d(self, data: np.ndarray, angle: float, factor: float,
                                order: int) -> np.ndarray:
        """Apply 3D shear transformation."""
        z_size, y_size, x_size = data.shape
        result = np.zeros_like(data)

        # Apply shear slice by slice in XY plane
        for z in range(z_size):
            # Calculate shear offset for this Z slice
            z_offset = (z - z_size // 2) * factor

            # Create affine transform matrix
            shear_matrix = np.array([
                [1, 0, z_offset * np.cos(angle)],
                [0, 1, z_offset * np.sin(angle)],
                [0, 0, 1]
            ])

            transform = AffineTransform(matrix=shear_matrix)

            # Apply transform
            result[z, :, :] = warp(
                data[z, :, :],
                transform,
                order=order,
                mode='constant',
                cval=0
            )

        return result

    def _apply_edge_filter_3d(self, data: np.ndarray, method: str, sigma: float) -> np.ndarray:
        """Apply 3D edge detection filter."""
        if method == "sobel":
            # Apply Sobel filter to each axis
            grad_z = ndi.sobel(data, axis=0)
            grad_y = ndi.sobel(data, axis=1)
            grad_x = ndi.sobel(data, axis=2)
            magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

        elif method == "laplacian":
            magnitude = ndi.laplace(data)

        elif method == "canny":
            # Apply Canny edge detection slice by slice
            magnitude = np.zeros_like(data)
            for z in range(data.shape[0]):
                magnitude[z, :, :] = filters.canny(data[z, :, :], sigma=sigma)

        else:
            raise ValueError(f"Unknown edge detection method: {method}")

        return magnitude

    def _apply_bilateral_filter(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply bilateral filtering for noise reduction."""
        from skimage.restoration import denoise_bilateral

        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                for z in range(data.shape[0]):
                    result[z, t, :, :] = denoise_bilateral(
                        data[z, t, :, :],
                        sigma_color=params.bilateral_sigma_color,
                        sigma_spatial=params.bilateral_sigma_spatial
                    )
        else:
            result = np.zeros_like(data)
            for z in range(data.shape[0]):
                result[z, :, :] = denoise_bilateral(
                    data[z, :, :],
                    sigma_color=params.bilateral_sigma_color,
                    sigma_spatial=params.bilateral_sigma_spatial
                )

        return result

    def _apply_non_local_means(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply non-local means denoising."""
        from skimage.restoration import denoise_nl_means

        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                for z in range(data.shape[0]):
                    result[z, t, :, :] = denoise_nl_means(data[z, t, :, :])
        else:
            result = np.zeros_like(data)
            for z in range(data.shape[0]):
                result[z, :, :] = denoise_nl_means(data[z, :, :])

        return result

    def _apply_tv_denoising(self, data: np.ndarray, params: ProcessingParameters) -> np.ndarray:
        """Apply total variation denoising."""
        from skimage.restoration import denoise_tv_chambolle

        if data.ndim == 4:
            result = np.zeros_like(data)
            for t in range(data.shape[1]):
                result[:, t, :, :] = denoise_tv_chambolle(data[:, t, :, :])
        else:
            result = denoise_tv_chambolle(data)

        return result

    def _update_cache(self, key: str, data: np.ndarray):
        """Update processing cache."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = list(self._cache.keys())[0]
            del self._cache[oldest_key]

        self._cache[key] = data.copy()

    def clear_cache(self):
        """Clear processing cache."""
        self._cache.clear()
        self.logger.info("Processing cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            'cached_results': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'max_cache_size': self._max_cache_size
        }

    def is_processing(self) -> bool:
        """Check if processing is currently running."""
        return self._is_processing

    def get_current_method(self) -> Optional[str]:
        """Get currently running processing method."""
        return self._current_method


class ProcessingPresets:
    """Predefined processing parameter presets for common use cases."""

    @staticmethod
    def calcium_imaging_cleanup() -> List[Tuple[ProcessingMethod, ProcessingParameters]]:
        """Preset for cleaning calcium imaging data."""
        return [
            (ProcessingMethod.BACKGROUND_SUBTRACTION, ProcessingParameters(
                background_method="rolling_ball",
                rolling_ball_radius=15.0
            )),
            (ProcessingMethod.GAUSSIAN_FILTER, ProcessingParameters(
                gaussian_sigma=0.8
            )),
            (ProcessingMethod.INTENSITY_NORMALIZATION, ProcessingParameters(
                normalization_method="percentile",
                percentile_range=(1.0, 99.5)
            ))
        ]

    @staticmethod
    def lightsheet_correction() -> List[Tuple[ProcessingMethod, ProcessingParameters]]:
        """Preset for lightsheet microscopy artifact correction."""
        return [
            (ProcessingMethod.SHEAR_CORRECTION, ProcessingParameters(
                shear_angle=45.0,
                shear_factor=1.0
            )),
            (ProcessingMethod.MOTION_CORRECTION, ProcessingParameters(
                max_shift=25,
                upsample_factor=2
            )),
            (ProcessingMethod.NOISE_REDUCTION, ProcessingParameters(
                noise_method="bilateral",
                bilateral_sigma_color=0.05,
                bilateral_sigma_spatial=1.0
            ))
        ]

    @staticmethod
    def deconvolution_enhancement() -> List[Tuple[ProcessingMethod, ProcessingParameters]]:
        """Preset for deconvolution-based enhancement."""
        return [
            (ProcessingMethod.NOISE_REDUCTION, ProcessingParameters(
                noise_method="tv_denoise"
            )),
            (ProcessingMethod.DECONVOLUTION, ProcessingParameters(
                deconv_method="richardson_lucy",
                deconv_iterations=15,
                psf_sigma=(1.5, 0.8, 0.8)
            )),
            (ProcessingMethod.INTENSITY_NORMALIZATION, ProcessingParameters(
                normalization_method="minmax"
            ))
        ]


# Utility functions for common operations

def calculate_df_f0(data: np.ndarray, f0_frames: Tuple[int, int],
                   baseline_frames: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Calculate ΔF/F₀ for fluorescence data.

    Args:
        data: 4D data array (Z, T, Y, X)
        f0_frames: Tuple of (start, end) frame indices for F₀ calculation
        baseline_frames: Optional baseline subtraction frames

    Returns:
        ΔF/F₀ data array
    """
    if data.ndim != 4:
        raise ValueError("Data must be 4D (Z, T, Y, X)")

    f0_start, f0_end = f0_frames

    # Calculate F₀ (baseline fluorescence)
    f0 = np.mean(data[:, f0_start:f0_end, :, :], axis=1, keepdims=True)

    # Avoid division by zero
    f0_safe = np.where(f0 > 0, f0, 1)

    # Calculate ΔF/F₀
    df_f0 = (data - f0_safe) / f0_safe

    return df_f0


def estimate_noise_level(data: np.ndarray, method: str = "mad") -> float:
    """
    Estimate noise level in image data.

    Args:
        data: Input data array
        method: Estimation method ('mad', 'std', 'percentile')

    Returns:
        Estimated noise level
    """
    if method == "mad":
        # Median Absolute Deviation
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return 1.4826 * mad  # Scale factor for Gaussian noise

    elif method == "std":
        return np.std(data)

    elif method == "percentile":
        # Use inter-quartile range
        q25, q75 = np.percentile(data, [25, 75])
        return (q75 - q25) / 1.349  # Scale factor for Gaussian noise

    else:
        raise ValueError(f"Unknown noise estimation method: {method}")


def create_processing_report(original_data: np.ndarray, processed_data: np.ndarray,
                           methods_used: List[str]) -> Dict[str, Any]:
    """
    Create a comprehensive processing report.

    Args:
        original_data: Original data array
        processed_data: Processed data array
        methods_used: List of processing methods applied

    Returns:
        Dictionary containing processing report
    """
    report = {
        'processing_methods': methods_used,
        'original_statistics': {
            'shape': original_data.shape,
            'dtype': str(original_data.dtype),
            'min': float(np.min(original_data)),
            'max': float(np.max(original_data)),
            'mean': float(np.mean(original_data)),
            'std': float(np.std(original_data)),
            'noise_estimate': estimate_noise_level(original_data)
        },
        'processed_statistics': {
            'shape': processed_data.shape,
            'dtype': str(processed_data.dtype),
            'min': float(np.min(processed_data)),
            'max': float(np.max(processed_data)),
            'mean': float(np.mean(processed_data)),
            'std': float(np.std(processed_data)),
            'noise_estimate': estimate_noise_level(processed_data)
        },
        'improvement_metrics': {
            'snr_improvement': float(
                (np.mean(processed_data) / estimate_noise_level(processed_data)) /
                (np.mean(original_data) / estimate_noise_level(original_data))
            ),
            'dynamic_range_ratio': float(
                (np.max(processed_data) - np.min(processed_data)) /
                (np.max(original_data) - np.min(original_data))
            )
        }
    }

    return report
