#!/usr/bin/env python3
"""
Analysis Algorithms Module
==========================

Comprehensive analysis algorithms for fluorescence microscopy data.
Specialized for calcium imaging, lightsheet microscopy, and general fluorescence analysis.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import ndimage, signal, optimize, stats
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, minimum_filter
from scipy.spatial.distance import cdist
from skimage import measure, morphology, filters, feature, segmentation
from sklearn.cluster import DBSCAN
import pandas as pd
from qtpy.QtCore import QObject, Signal


@dataclass
class DetectionParameters:
    """Parameters for event detection algorithms."""
    threshold_method: str = 'adaptive'  # 'fixed', 'adaptive', 'otsu'
    threshold_value: float = 3.0  # Standard deviations above baseline
    min_event_duration: int = 3  # Minimum frames for valid event
    max_event_duration: int = 50  # Maximum frames for valid event
    min_event_size: float = 2.0  # Minimum spatial extent (pixels)
    max_event_size: float = 20.0  # Maximum spatial extent (pixels)
    temporal_smoothing: float = 1.0  # Gaussian smoothing sigma in time
    spatial_smoothing: float = 1.0  # Gaussian smoothing sigma in space
    baseline_method: str = 'percentile'  # 'mean', 'median', 'percentile'
    baseline_percentile: float = 10.0  # For percentile baseline method
    merge_overlapping: bool = True  # Merge spatially overlapping events
    merge_distance: float = 5.0  # Distance threshold for merging (pixels)


@dataclass
class EventProperties:
    """Properties of detected fluorescence events."""
    centroid: Tuple[float, float, float]  # Z, Y, X coordinates
    amplitude: float  # Peak amplitude above baseline
    baseline: float  # Baseline fluorescence
    onset_time: int  # Frame of event onset
    peak_time: int  # Frame of peak amplitude
    duration: int  # Event duration in frames
    rise_time: float  # 20%-80% rise time
    decay_time: float  # 80%-20% decay time
    spatial_extent: float  # Spatial size (pixels)
    integrated_intensity: float  # Total integrated intensity
    signal_to_noise: float  # Signal-to-noise ratio
    volume_index: int  # Volume/timepoint index
    roi_id: Optional[int] = None  # Associated ROI ID if applicable


@dataclass
class ROIProperties:
    """Properties of regions of interest."""
    roi_id: int
    centroid: Tuple[float, float, float]  # Z, Y, X coordinates
    area: float  # Area in pixels
    perimeter: float  # Perimeter length
    circularity: float  # 4π*area/perimeter²
    mean_intensity: float  # Mean fluorescence intensity
    max_intensity: float  # Maximum fluorescence intensity
    integrated_intensity: float  # Total integrated intensity
    background_level: float  # Local background estimate
    signal_to_background: float  # Signal-to-background ratio
    temporal_profile: np.ndarray  # Intensity over time


class AnalysisEngine(QObject):
    """
    Comprehensive analysis engine for fluorescence microscopy data.

    Features:
    - Calcium event detection (puffs, waves, sparks)
    - ROI-based analysis
    - Statistical analysis and characterization
    - Motion correction
    - Background estimation and subtraction
    - Temporal filtering and smoothing
    - Spatial clustering and segmentation
    """

    # Signals for progress reporting
    analysis_progress = Signal(int, str)  # progress %, status message
    analysis_completed = Signal(str, object)  # analysis_type, results
    analysis_failed = Signal(str, str)  # analysis_type, error_message

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Analysis state
        self.current_data: Optional[np.ndarray] = None
        self.baseline_image: Optional[np.ndarray] = None
        self.noise_estimate: Optional[float] = None

        # Results storage
        self.detected_events: List[EventProperties] = []
        self.roi_list: List[ROIProperties] = []
        self.analysis_metadata: Dict[str, Any] = {}

    def set_data(self, data: np.ndarray) -> bool:
        """
        Set data for analysis.

        Args:
            data: 4D array (Z, T, Y, X) or 3D array (T, Y, X)

        Returns:
            bool: Success status
        """
        try:
            if data.ndim == 3:
                data = data[np.newaxis, :, :, :]  # Add Z dimension
            elif data.ndim != 4:
                raise ValueError(f"Data must be 3D or 4D, got {data.ndim}D")

            self.current_data = data.astype(np.float32)
            self.logger.info(f"Analysis data set: shape={data.shape}")

            # Reset analysis state
            self.baseline_image = None
            self.noise_estimate = None
            self.detected_events.clear()
            self.roi_list.clear()

            return True

        except Exception as e:
            self.logger.error(f"Failed to set analysis data: {str(e)}")
            return False

    def detect_calcium_events(self, params: DetectionParameters) -> List[EventProperties]:
        """
        Detect calcium events (puffs, sparks, waves) in fluorescence data.

        Args:
            params: Detection parameters

        Returns:
            List of detected events with properties
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info("Starting calcium event detection")
            self.analysis_progress.emit(0, "Initializing event detection...")

            # Prepare data
            data = self.current_data.copy()
            z_size, t_size, y_size, x_size = data.shape

            # Apply temporal smoothing
            if params.temporal_smoothing > 0:
                self.analysis_progress.emit(10, "Applying temporal smoothing...")
                for z in range(z_size):
                    for y in range(y_size):
                        for x in range(x_size):
                            data[z, :, y, x] = gaussian_filter(
                                data[z, :, y, x], params.temporal_smoothing
                            )

            # Apply spatial smoothing
            if params.spatial_smoothing > 0:
                self.analysis_progress.emit(20, "Applying spatial smoothing...")
                for t in range(t_size):
                    data[:, t, :, :] = gaussian_filter(
                        data[:, t, :, :],
                        sigma=(0, params.spatial_smoothing, params.spatial_smoothing)
                    )

            # Calculate baseline
            self.analysis_progress.emit(30, "Calculating baseline...")
            baseline = self._calculate_baseline(data, params.baseline_method,
                                              params.baseline_percentile)

            # Calculate noise estimate
            noise_std = self._estimate_noise(data, baseline)
            self.noise_estimate = noise_std

            # Detect candidate events
            self.analysis_progress.emit(40, "Detecting candidate events...")
            candidates = self._detect_event_candidates(data, baseline, noise_std, params)

            # Analyze event properties
            self.analysis_progress.emit(60, "Analyzing event properties...")
            events = []
            for i, candidate in enumerate(candidates):
                if i % 10 == 0:  # Update progress
                    progress = 60 + (30 * i / len(candidates))
                    self.analysis_progress.emit(int(progress),
                                              f"Analyzing event {i+1}/{len(candidates)}...")

                event_props = self._analyze_event_properties(
                    data, candidate, baseline, noise_std, params
                )
                if event_props is not None:
                    events.append(event_props)

            # Merge overlapping events
            if params.merge_overlapping:
                self.analysis_progress.emit(90, "Merging overlapping events...")
                events = self._merge_overlapping_events(events, params.merge_distance)

            self.detected_events = events
            self.analysis_progress.emit(100, f"Detection complete: {len(events)} events found")

            self.logger.info(f"Event detection completed: {len(events)} events detected")
            self.analysis_completed.emit("calcium_events", events)

            return events

        except Exception as e:
            error_msg = f"Event detection failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("calcium_events", error_msg)
            return []

    def analyze_rois(self, roi_masks: List[np.ndarray],
                    roi_positions: List[Tuple[int, int, int]] = None) -> List[ROIProperties]:
        """
        Analyze regions of interest in the data.

        Args:
            roi_masks: List of 3D binary masks for each ROI
            roi_positions: Optional list of (z, y, x) positions for ROIs

        Returns:
            List of ROI properties
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info(f"Starting ROI analysis for {len(roi_masks)} ROIs")
            self.analysis_progress.emit(0, "Initializing ROI analysis...")

            roi_properties = []

            for i, mask in enumerate(roi_masks):
                progress = int(100 * i / len(roi_masks))
                self.analysis_progress.emit(progress, f"Analyzing ROI {i+1}/{len(roi_masks)}...")

                props = self._analyze_single_roi(mask, i, roi_positions[i] if roi_positions else None)
                if props is not None:
                    roi_properties.append(props)

            self.roi_list = roi_properties
            self.analysis_progress.emit(100, f"ROI analysis complete: {len(roi_properties)} ROIs")

            self.logger.info(f"ROI analysis completed: {len(roi_properties)} ROIs analyzed")
            self.analysis_completed.emit("roi_analysis", roi_properties)

            return roi_properties

        except Exception as e:
            error_msg = f"ROI analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("roi_analysis", error_msg)
            return []

    def calculate_df_f0(self, f0_start: int, f0_end: int,
                       volume_start: int = 0, volume_end: int = -1) -> np.ndarray:
        """
        Calculate ΔF/F₀ for the data.

        Args:
            f0_start: Start frame for F₀ calculation
            f0_end: End frame for F₀ calculation
            volume_start: Start volume index
            volume_end: End volume index (-1 for all)

        Returns:
            ΔF/F₀ data array
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info("Calculating ΔF/F₀")
            self.analysis_progress.emit(0, "Calculating baseline F₀...")

            data = self.current_data.copy()
            if volume_end == -1:
                volume_end = data.shape[1]

            # Calculate F₀ (baseline)
            f0_data = data[:, f0_start:f0_end, :, :]
            f0_mean = np.mean(f0_data, axis=1, keepdims=True)

            # Avoid division by zero
            f0_mean = np.where(f0_mean == 0, 1e-10, f0_mean)

            self.analysis_progress.emit(50, "Computing ΔF/F₀...")

            # Calculate ΔF/F₀
            data_subset = data[:, volume_start:volume_end, :, :]
            df_f0 = (data_subset - f0_mean) / f0_mean

            self.analysis_progress.emit(100, "ΔF/F₀ calculation complete")

            self.logger.info("ΔF/F₀ calculation completed")
            self.analysis_completed.emit("df_f0", df_f0)

            return df_f0

        except Exception as e:
            error_msg = f"ΔF/F₀ calculation failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("df_f0", error_msg)
            return np.array([])

    def correct_motion(self, reference_volume: int = 0) -> np.ndarray:
        """
        Perform motion correction using cross-correlation.

        Args:
            reference_volume: Volume index to use as reference

        Returns:
            Motion-corrected data
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info("Starting motion correction")
            self.analysis_progress.emit(0, "Initializing motion correction...")

            data = self.current_data.copy()
            z_size, t_size, y_size, x_size = data.shape

            # Use reference volume
            reference = data[:, reference_volume, :, :]
            corrected_data = data.copy()

            for t in range(t_size):
                if t == reference_volume:
                    continue

                progress = int(100 * t / t_size)
                self.analysis_progress.emit(progress, f"Correcting volume {t+1}/{t_size}...")

                current_volume = data[:, t, :, :]

                # Calculate shifts for each Z plane
                for z in range(z_size):
                    shifts = self._calculate_2d_shift(reference[z], current_volume[z])
                    corrected_data[z, t, :, :] = self._apply_shift(current_volume[z], shifts)

            self.analysis_progress.emit(100, "Motion correction complete")

            self.logger.info("Motion correction completed")
            self.analysis_completed.emit("motion_correction", corrected_data)

            return corrected_data

        except Exception as e:
            error_msg = f"Motion correction failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("motion_correction", error_msg)
            return self.current_data

    def subtract_background(self, method: str = 'rolling_ball',
                          radius: float = 50.0) -> np.ndarray:
        """
        Subtract background from the data.

        Args:
            method: Background subtraction method ('rolling_ball', 'gaussian', 'median')
            radius: Characteristic size for background estimation

        Returns:
            Background-subtracted data
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info(f"Starting background subtraction: {method}")
            self.analysis_progress.emit(0, "Calculating background...")

            data = self.current_data.copy()
            z_size, t_size, y_size, x_size = data.shape

            if method == 'rolling_ball':
                background = self._rolling_ball_background(data, radius)
            elif method == 'gaussian':
                background = self._gaussian_background(data, radius)
            elif method == 'median':
                background = self._median_background(data, radius)
            else:
                raise ValueError(f"Unknown background method: {method}")

            self.analysis_progress.emit(80, "Subtracting background...")

            # Subtract background
            corrected_data = data - background

            # Ensure non-negative values
            corrected_data = np.maximum(corrected_data, 0)

            self.baseline_image = background
            self.analysis_progress.emit(100, "Background subtraction complete")

            self.logger.info("Background subtraction completed")
            self.analysis_completed.emit("background_subtraction", corrected_data)

            return corrected_data

        except Exception as e:
            error_msg = f"Background subtraction failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("background_subtraction", error_msg)
            return self.current_data

    def create_automatic_rois(self, method: str = 'watershed',
                             min_size: int = 10, max_size: int = 1000) -> List[np.ndarray]:
        """
        Automatically create ROIs based on image segmentation.

        Args:
            method: Segmentation method ('watershed', 'threshold', 'blob')
            min_size: Minimum ROI size in pixels
            max_size: Maximum ROI size in pixels

        Returns:
            List of ROI masks
        """
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info(f"Creating automatic ROIs: {method}")
            self.analysis_progress.emit(0, "Analyzing image for segmentation...")

            # Use maximum intensity projection for segmentation
            max_proj = np.max(self.current_data, axis=(0, 1))  # Max over Z and T

            if method == 'watershed':
                roi_masks = self._watershed_segmentation(max_proj, min_size, max_size)
            elif method == 'threshold':
                roi_masks = self._threshold_segmentation(max_proj, min_size, max_size)
            elif method == 'blob':
                roi_masks = self._blob_segmentation(max_proj, min_size, max_size)
            else:
                raise ValueError(f"Unknown segmentation method: {method}")

            self.analysis_progress.emit(100, f"ROI creation complete: {len(roi_masks)} ROIs")

            self.logger.info(f"Automatic ROI creation completed: {len(roi_masks)} ROIs")
            self.analysis_completed.emit("automatic_rois", roi_masks)

            return roi_masks

        except Exception as e:
            error_msg = f"Automatic ROI creation failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("automatic_rois", error_msg)
            return []

    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the current data."""
        try:
            if self.current_data is None:
                raise ValueError("No data loaded for analysis")

            self.logger.info("Calculating comprehensive statistics")
            self.analysis_progress.emit(0, "Computing basic statistics...")

            data = self.current_data
            stats_dict = {}

            # Basic statistics
            stats_dict['shape'] = data.shape
            stats_dict['dtype'] = str(data.dtype)
            stats_dict['size_mb'] = data.nbytes / (1024 * 1024)

            self.analysis_progress.emit(25, "Computing intensity statistics...")

            # Intensity statistics
            stats_dict['mean_intensity'] = float(np.mean(data))
            stats_dict['std_intensity'] = float(np.std(data))
            stats_dict['min_intensity'] = float(np.min(data))
            stats_dict['max_intensity'] = float(np.max(data))
            stats_dict['median_intensity'] = float(np.median(data))

            # Percentiles
            percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
            for p in percentiles:
                stats_dict[f'percentile_{p}'] = float(np.percentile(data, p))

            self.analysis_progress.emit(50, "Computing temporal statistics...")

            # Temporal statistics
            temporal_mean = np.mean(data, axis=(0, 2, 3))  # Mean over Z, Y, X
            stats_dict['temporal_mean'] = temporal_mean.tolist()
            stats_dict['temporal_std'] = np.std(temporal_mean)
            stats_dict['temporal_drift'] = float(temporal_mean[-1] - temporal_mean[0])

            self.analysis_progress.emit(75, "Computing spatial statistics...")

            # Spatial statistics
            spatial_mean = np.mean(data, axis=(0, 1))  # Mean over Z, T
            stats_dict['spatial_uniformity'] = float(np.std(spatial_mean) / np.mean(spatial_mean))

            # Signal-to-noise estimate
            if self.noise_estimate is not None:
                stats_dict['snr_estimate'] = float(np.mean(data) / self.noise_estimate)

            # Event statistics
            if self.detected_events:
                event_amplitudes = [e.amplitude for e in self.detected_events]
                stats_dict['n_events'] = len(self.detected_events)
                stats_dict['mean_event_amplitude'] = float(np.mean(event_amplitudes))
                stats_dict['event_rate'] = len(self.detected_events) / (data.shape[1] * data.shape[0])

            # ROI statistics
            if self.roi_list:
                roi_intensities = [r.mean_intensity for r in self.roi_list]
                stats_dict['n_rois'] = len(self.roi_list)
                stats_dict['mean_roi_intensity'] = float(np.mean(roi_intensities))

            self.analysis_progress.emit(100, "Statistics calculation complete")

            self.analysis_metadata['statistics'] = stats_dict
            self.logger.info("Statistics calculation completed")
            self.analysis_completed.emit("statistics", stats_dict)

            return stats_dict

        except Exception as e:
            error_msg = f"Statistics calculation failed: {str(e)}"
            self.logger.error(error_msg)
            self.analysis_failed.emit("statistics", error_msg)
            return {}

    # Private methods for algorithm implementation

    def _calculate_baseline(self, data: np.ndarray, method: str,
                          percentile: float) -> np.ndarray:
        """Calculate baseline fluorescence."""
        if method == 'mean':
            return np.mean(data, axis=1, keepdims=True)
        elif method == 'median':
            return np.median(data, axis=1, keepdims=True)
        elif method == 'percentile':
            return np.percentile(data, percentile, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

    def _estimate_noise(self, data: np.ndarray, baseline: np.ndarray) -> float:
        """Estimate noise standard deviation."""
        # Use residuals from baseline for noise estimation
        residuals = data - baseline

        # Use robust noise estimation (median absolute deviation)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        noise_std = mad * 1.4826  # Convert MAD to std for Gaussian noise

        return noise_std

    def _detect_event_candidates(self, data: np.ndarray, baseline: np.ndarray,
                               noise_std: float, params: DetectionParameters) -> List[Dict]:
        """Detect candidate events using threshold-based detection."""
        candidates = []

        # Calculate significance map
        significance = (data - baseline) / noise_std

        # Apply threshold
        if params.threshold_method == 'fixed':
            threshold = params.threshold_value
        elif params.threshold_method == 'adaptive':
            threshold = params.threshold_value * np.std(significance)
        elif params.threshold_method == 'otsu':
            threshold = filters.threshold_otsu(significance.flatten())
        else:
            threshold = params.threshold_value

        # Find connected components above threshold
        binary_mask = significance > threshold

        for t in range(data.shape[1]):
            # Label connected components in this timepoint
            labels = measure.label(binary_mask[:, t, :, :])
            props = measure.regionprops(labels, intensity_image=data[:, t, :, :])

            for prop in props:
                if (prop.area >= params.min_event_size and
                    prop.area <= params.max_event_size):

                    candidate = {
                        'timepoint': t,
                        'centroid': prop.centroid,
                        'area': prop.area,
                        'max_intensity': prop.max_intensity,
                        'mean_intensity': prop.mean_intensity,
                        'bbox': prop.bbox,
                        'coords': prop.coords
                    }
                    candidates.append(candidate)

        return candidates

    def _analyze_event_properties(self, data: np.ndarray, candidate: Dict,
                                baseline: np.ndarray, noise_std: float,
                                params: DetectionParameters) -> Optional[EventProperties]:
        """Analyze detailed properties of a detected event."""
        try:
            t = candidate['timepoint']
            centroid = candidate['centroid']
            coords = candidate['coords']

            # Extract temporal profile at event location
            z_center = int(centroid[0])
            y_center = int(centroid[1])
            x_center = int(centroid[2])

            # Get spatial region around event
            z_min, y_min, x_min, z_max, y_max, x_max = candidate['bbox']

            # Extract temporal traces
            temporal_profile = data[z_min:z_max+1, :, y_min:y_max+1, x_min:x_max+1]
            baseline_profile = baseline[z_min:z_max+1, :, y_min:y_max+1, x_min:x_max+1]

            # Calculate event properties
            event_trace = np.mean(temporal_profile, axis=(0, 2, 3))
            baseline_trace = np.mean(baseline_profile, axis=(0, 2, 3))

            # Find event boundaries
            onset_time, peak_time, duration = self._find_event_boundaries(
                event_trace, baseline_trace, t, params
            )

            if duration < params.min_event_duration or duration > params.max_event_duration:
                return None

            # Calculate amplitude and kinetics
            amplitude = event_trace[peak_time] - baseline_trace[peak_time]
            baseline_level = baseline_trace[peak_time]

            rise_time, decay_time = self._calculate_kinetics(
                event_trace, baseline_trace, onset_time, peak_time, duration
            )

            # Calculate spatial properties
            spatial_extent = np.sqrt(candidate['area'])

            # Calculate integrated intensity
            integrated_intensity = np.sum(
                (event_trace[onset_time:onset_time+duration] -
                 baseline_trace[onset_time:onset_time+duration])
            )

            # Signal-to-noise ratio
            snr = amplitude / noise_std

            return EventProperties(
                centroid=(z_center, y_center, x_center),
                amplitude=amplitude,
                baseline=baseline_level,
                onset_time=onset_time,
                peak_time=peak_time,
                duration=duration,
                rise_time=rise_time,
                decay_time=decay_time,
                spatial_extent=spatial_extent,
                integrated_intensity=integrated_intensity,
                signal_to_noise=snr,
                volume_index=t
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze event properties: {str(e)}")
            return None

    def _find_event_boundaries(self, signal: np.ndarray, baseline: np.ndarray,
                             peak_time: int, params: DetectionParameters) -> Tuple[int, int, int]:
        """Find event onset, peak, and duration."""
        # Find peak
        diff_signal = signal - baseline
        peak_idx = np.argmax(diff_signal)

        # Find onset (go backwards from peak)
        onset_idx = peak_time
        threshold = 0.2 * diff_signal[peak_idx]  # 20% of peak

        for i in range(peak_idx, max(0, peak_idx - 50), -1):
            if diff_signal[i] < threshold:
                onset_idx = i
                break

        # Find end (go forwards from peak)
        end_idx = peak_time
        for i in range(peak_idx, min(len(signal), peak_idx + 50)):
            if diff_signal[i] < threshold:
                end_idx = i
                break

        duration = end_idx - onset_idx

        return onset_idx, peak_idx, duration

    def _calculate_kinetics(self, signal: np.ndarray, baseline: np.ndarray,
                          onset: int, peak: int, duration: int) -> Tuple[float, float]:
        """Calculate rise and decay kinetics."""
        diff_signal = signal - baseline
        peak_amplitude = diff_signal[peak]

        # Rise time (20% to 80% of peak)
        rise_20 = 0.2 * peak_amplitude
        rise_80 = 0.8 * peak_amplitude

        rise_start_idx = onset
        rise_end_idx = peak

        for i in range(onset, peak):
            if diff_signal[i] >= rise_20:
                rise_start_idx = i
                break

        for i in range(onset, peak):
            if diff_signal[i] >= rise_80:
                rise_end_idx = i
                break

        rise_time = rise_end_idx - rise_start_idx

        # Decay time (80% to 20% of peak)
        decay_start_idx = peak
        decay_end_idx = onset + duration

        for i in range(peak, onset + duration):
            if diff_signal[i] <= rise_80:
                decay_start_idx = i
                break

        for i in range(peak, onset + duration):
            if diff_signal[i] <= rise_20:
                decay_end_idx = i
                break

        decay_time = decay_end_idx - decay_start_idx

        return float(rise_time), float(decay_time)

    def _merge_overlapping_events(self, events: List[EventProperties],
                                distance_threshold: float) -> List[EventProperties]:
        """Merge spatially overlapping events."""
        if not events:
            return events

        # Extract centroids for distance calculation
        centroids = np.array([e.centroid for e in events])

        # Calculate pairwise distances
        distances = cdist(centroids, centroids)

        # Find overlapping events
        overlapping = distances < distance_threshold
        np.fill_diagonal(overlapping, False)

        # Group overlapping events
        merged_events = []
        used_indices = set()

        for i, event in enumerate(events):
            if i in used_indices:
                continue

            # Find all events that overlap with this one
            overlapping_indices = np.where(overlapping[i])[0]
            overlapping_indices = [idx for idx in overlapping_indices if idx not in used_indices]

            if len(overlapping_indices) == 0:
                merged_events.append(event)
            else:
                # Merge overlapping events
                group_events = [events[i]] + [events[idx] for idx in overlapping_indices]
                merged_event = self._merge_event_group(group_events)
                merged_events.append(merged_event)

                used_indices.add(i)
                used_indices.update(overlapping_indices)

        return merged_events

    def _merge_event_group(self, events: List[EventProperties]) -> EventProperties:
        """Merge a group of overlapping events."""
        # Weighted average based on amplitude
        weights = [e.amplitude for e in events]
        total_weight = sum(weights)

        # Calculate weighted centroid
        weighted_centroid = np.average(
            [e.centroid for e in events], weights=weights, axis=0
        )

        # Take properties from strongest event
        strongest_event = max(events, key=lambda e: e.amplitude)

        # Merge some properties
        merged_amplitude = max(e.amplitude for e in events)
        merged_duration = int(np.mean([e.duration for e in events]))
        merged_spatial_extent = np.mean([e.spatial_extent for e in events])

        return EventProperties(
            centroid=tuple(weighted_centroid),
            amplitude=merged_amplitude,
            baseline=strongest_event.baseline,
            onset_time=strongest_event.onset_time,
            peak_time=strongest_event.peak_time,
            duration=merged_duration,
            rise_time=strongest_event.rise_time,
            decay_time=strongest_event.decay_time,
            spatial_extent=merged_spatial_extent,
            integrated_intensity=sum(e.integrated_intensity for e in events),
            signal_to_noise=strongest_event.signal_to_noise,
            volume_index=strongest_event.volume_index
        )

    def _analyze_single_roi(self, mask: np.ndarray, roi_id: int,
                          position: Optional[Tuple[int, int, int]]) -> Optional[ROIProperties]:
        """Analyze properties of a single ROI."""
        try:
            if self.current_data is None:
                return None

            # Apply mask to data
            if mask.ndim == 2:
                # 2D mask - apply to all Z planes
                mask_3d = np.tile(mask[np.newaxis, :, :], (self.current_data.shape[0], 1, 1))
            else:
                mask_3d = mask

            # Extract ROI data
            roi_data = self.current_data * mask_3d[:, np.newaxis, :, :]

            # Calculate properties
            area = np.sum(mask_3d)
            if area == 0:
                return None

            # Centroid
            coords = np.where(mask_3d)
            centroid = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))

            # Perimeter (approximate for 3D)
            perimeter = measure.perimeter(mask_3d[mask_3d.shape[0]//2])  # Use middle Z plane

            # Circularity
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

            # Intensity measurements
            nonzero_data = roi_data[roi_data > 0]
            if len(nonzero_data) == 0:
                return None

            mean_intensity = np.mean(nonzero_data)
            max_intensity = np.max(nonzero_data)
            integrated_intensity = np.sum(nonzero_data)

            # Background estimation (surrounding region)
            dilated_mask = ndimage.binary_dilation(mask_3d, iterations=5)
            background_mask = dilated_mask & ~mask_3d
            background_data = self.current_data * background_mask[:, np.newaxis, :, :]
            background_data = background_data[background_data > 0]

            background_level = np.median(background_data) if len(background_data) > 0 else 0
            signal_to_background = mean_intensity / background_level if background_level > 0 else np.inf

            # Temporal profile
            temporal_profile = np.mean(roi_data, axis=(0, 2, 3))

            return ROIProperties(
                roi_id=roi_id,
                centroid=centroid,
                area=area,
                perimeter=perimeter,
                circularity=circularity,
                mean_intensity=mean_intensity,
                max_intensity=max_intensity,
                integrated_intensity=integrated_intensity,
                background_level=background_level,
                signal_to_background=signal_to_background,
                temporal_profile=temporal_profile
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze ROI {roi_id}: {str(e)}")
            return None

    # Motion correction helpers

    def _calculate_2d_shift(self, reference: np.ndarray,
                           image: np.ndarray) -> Tuple[float, float]:
        """Calculate 2D shift between images using cross-correlation."""
        correlation = signal.correlate2d(image, reference, mode='same')
        peak_coords = np.unravel_index(np.argmax(correlation), correlation.shape)

        shift_y = peak_coords[0] - correlation.shape[0] // 2
        shift_x = peak_coords[1] - correlation.shape[1] // 2

        return shift_y, shift_x

    def _apply_shift(self, image: np.ndarray, shifts: Tuple[float, float]) -> np.ndarray:
        """Apply 2D shift to image."""
        shift_y, shift_x = shifts
        return ndimage.shift(image, [shift_y, shift_x], mode='constant', cval=0)

    # Background subtraction helpers

    def _rolling_ball_background(self, data: np.ndarray, radius: float) -> np.ndarray:
        """Rolling ball background subtraction."""
        # Simplified implementation - use morphological opening
        structure = morphology.ball(int(radius))
        background = np.zeros_like(data)

        for t in range(data.shape[1]):
            background[:, t, :, :] = morphology.opening(data[:, t, :, :], structure)

        return background

    def _gaussian_background(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian background estimation."""
        background = np.zeros_like(data)

        for t in range(data.shape[1]):
            background[:, t, :, :] = gaussian_filter(data[:, t, :, :], sigma)

        return background

    def _median_background(self, data: np.ndarray, size: float) -> np.ndarray:
        """Median filter background estimation."""
        background = np.zeros_like(data)
        kernel_size = int(2 * size + 1)

        for t in range(data.shape[1]):
            background[:, t, :, :] = median_filter(data[:, t, :, :], size=kernel_size)

        return background

    # Segmentation helpers

    def _watershed_segmentation(self, image: np.ndarray, min_size: int,
                              max_size: int) -> List[np.ndarray]:
        """Watershed-based segmentation."""
        # Preprocessing
        smoothed = gaussian_filter(image, sigma=1.0)

        # Find local maxima as seeds
        local_maxima = feature.peak_local_maxima(smoothed, threshold_abs=np.percentile(smoothed, 90))
        markers = np.zeros_like(image, dtype=int)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1

        # Watershed
        labels = segmentation.watershed(-smoothed, markers, mask=smoothed > np.percentile(smoothed, 50))

        # Extract ROI masks
        roi_masks = []
        for region_id in np.unique(labels):
            if region_id == 0:  # Background
                continue

            mask = (labels == region_id)
            if min_size <= np.sum(mask) <= max_size:
                roi_masks.append(mask.astype(bool))

        return roi_masks

    def _threshold_segmentation(self, image: np.ndarray, min_size: int,
                              max_size: int) -> List[np.ndarray]:
        """Threshold-based segmentation."""
        # Apply threshold
        threshold = filters.threshold_otsu(image)
        binary = image > threshold

        # Label connected components
        labels = measure.label(binary)

        # Extract ROI masks
        roi_masks = []
        for region in measure.regionprops(labels):
            if min_size <= region.area <= max_size:
                mask = (labels == region.label)
                roi_masks.append(mask.astype(bool))

        return roi_masks

    def _blob_segmentation(self, image: np.ndarray, min_size: int,
                         max_size: int) -> List[np.ndarray]:
        """Blob detection-based segmentation."""
        # Detect blobs
        min_sigma = np.sqrt(min_size / np.pi)
        max_sigma = np.sqrt(max_size / np.pi)

        blobs = feature.blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                               threshold=np.percentile(image, 80))

        # Create ROI masks from blobs
        roi_masks = []
        for blob in blobs:
            y, x, sigma = blob
            radius = int(sigma * np.sqrt(2))

            # Create circular mask
            yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2

            if min_size <= np.sum(mask) <= max_size:
                roi_masks.append(mask.astype(bool))

        return roi_masks

    # Export methods

    def export_events_to_dataframe(self) -> pd.DataFrame:
        """Export detected events to pandas DataFrame."""
        if not self.detected_events:
            return pd.DataFrame()

        data_list = []
        for event in self.detected_events:
            data_list.append({
                'centroid_z': event.centroid[0],
                'centroid_y': event.centroid[1],
                'centroid_x': event.centroid[2],
                'amplitude': event.amplitude,
                'baseline': event.baseline,
                'onset_time': event.onset_time,
                'peak_time': event.peak_time,
                'duration': event.duration,
                'rise_time': event.rise_time,
                'decay_time': event.decay_time,
                'spatial_extent': event.spatial_extent,
                'integrated_intensity': event.integrated_intensity,
                'signal_to_noise': event.signal_to_noise,
                'volume_index': event.volume_index
            })

        return pd.DataFrame(data_list)

    def export_rois_to_dataframe(self) -> pd.DataFrame:
        """Export ROI properties to pandas DataFrame."""
        if not self.roi_list:
            return pd.DataFrame()

        data_list = []
        for roi in self.roi_list:
            data_list.append({
                'roi_id': roi.roi_id,
                'centroid_z': roi.centroid[0],
                'centroid_y': roi.centroid[1],
                'centroid_x': roi.centroid[2],
                'area': roi.area,
                'perimeter': roi.perimeter,
                'circularity': roi.circularity,
                'mean_intensity': roi.mean_intensity,
                'max_intensity': roi.max_intensity,
                'integrated_intensity': roi.integrated_intensity,
                'background_level': roi.background_level,
                'signal_to_background': roi.signal_to_background
            })

        return pd.DataFrame(data_list)


# Utility functions
def create_default_detection_params() -> DetectionParameters:
    """Create default detection parameters for calcium events."""
    return DetectionParameters()


def validate_detection_params(params: DetectionParameters) -> List[str]:
    """Validate detection parameters and return list of warnings."""
    warnings = []

    if params.threshold_value <= 0:
        warnings.append("Threshold value should be positive")

    if params.min_event_duration >= params.max_event_duration:
        warnings.append("Minimum event duration should be less than maximum")

    if params.min_event_size >= params.max_event_size:
        warnings.append("Minimum event size should be less than maximum")

    if params.baseline_percentile < 0 or params.baseline_percentile > 100:
        warnings.append("Baseline percentile should be between 0 and 100")

    return warnings
