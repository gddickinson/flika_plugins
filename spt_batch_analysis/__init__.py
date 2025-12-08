#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIKA SPT Batch Analysis Plugin - Enhanced Version with Dual Linking Methods and File Logging
Comprehensive single particle tracking analysis with fully integrated detection and autocorrelation

ENHANCEMENTS:
1. Iterative track linking as default (prevents recursion errors)
2. Optional recursive linking with configurable recursion limit
3. Comprehensive file logging system
4. Enhanced error handling and debugging
5. All original functionality preserved
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
from tqdm import tqdm
import os, glob, sys
import subprocess  # Added for thunderSTORM execution
import json
import math
from pathlib import Path
import time
import random
from collections import defaultdict, deque
import traceback
import logging
from datetime import datetime

# FLIKA imports
import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import open_file
from flika.utils.misc import open_file_gui, save_file_gui
from flika.roi import open_rois

# Scientific computing
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats, spatial, ndimage, optimize
import skimage.io as skio
from skimage import filters, morphology, measure

# Qt imports
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QComboBox, QTextEdit, QProgressBar,
                           QGroupBox, QGridLayout, QFormLayout, QFileDialog,
                           QListWidget, QFrame, QApplication, QSplitter,
                           QScrollArea, QSlider, QButtonGroup, QRadioButton,
                           QLineEdit, QMessageBox)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont

# Matplotlib for autocorrelation plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from scipy.optimize import linear_sum_assignment


# ==================== FILE LOGGING SYSTEM ====================

class FileLogger:
    """Comprehensive file logging system for SPT analysis"""

    def __init__(self):
        self.current_log_file = None
        self.file_handler = None
        self.logger = None
        self.console_handler = None

    def setup_file_logging(self, analysis_dir, file_name=None):
        """Setup file logging for the current analysis"""
        try:
            if file_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"spt_analysis_log_{timestamp}.log"

            self.current_log_file = os.path.join(analysis_dir, file_name)

            # Create logger
            self.logger = logging.getLogger('spt_analysis')
            self.logger.setLevel(logging.DEBUG)

            # Remove existing handlers
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)

            # Create file handler
            self.file_handler = logging.FileHandler(self.current_log_file, mode='w', encoding='utf-8')
            self.file_handler.setLevel(logging.DEBUG)

            # Create console handler for critical errors
            self.console_handler = logging.StreamHandler()
            self.console_handler.setLevel(logging.ERROR)

            # Create detailed formatter
            detailed_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            simple_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            self.file_handler.setFormatter(detailed_formatter)
            self.console_handler.setFormatter(simple_formatter)

            # Add handlers to logger
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)

            # Log initialization
            self.logger.info("="*80)
            self.logger.info("SPT Analysis Session Started")
            self.logger.info("="*80)
            self.logger.info(f"Log file: {self.current_log_file}")
            self.logger.info(f"Python version: {sys.version}")
            self.logger.info(f"NumPy version: {np.__version__}")
            self.logger.info(f"Pandas version: {pd.__version__}")

            return self.current_log_file

        except Exception as e:
            print(f"Error setting up file logging: {e}")
            return None

    def log(self, level, message, extra_data=None):
        """Log a message with specified level and optional extra data"""
        if self.logger:
            log_message = str(message)
            if extra_data:
                log_message += f" | Extra data: {extra_data}"
            getattr(self.logger, level.lower())(log_message)

    def log_error(self, message, exception=None, context=None):
        """Log an error with optional exception details and context"""
        if self.logger:
            self.logger.error(message)
            if exception:
                self.logger.error(f"Exception type: {type(exception).__name__}")
                self.logger.error(f"Exception details: {str(exception)}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            if context:
                self.logger.error(f"Context: {context}")

    def log_performance(self, operation, duration, details=None):
        """Log performance metrics"""
        if self.logger:
            message = f"Performance: {operation} completed in {duration:.4f} seconds"
            if details:
                message += f" | Details: {details}"
            self.logger.info(message)

    def log_data_summary(self, data_name, data, summary_func=None):
        """Log summary of data structures"""
        if self.logger:
            try:
                if hasattr(data, 'shape'):
                    self.logger.debug(f"Data summary - {data_name}: shape={data.shape}, dtype={data.dtype}")
                elif hasattr(data, '__len__'):
                    self.logger.debug(f"Data summary - {data_name}: length={len(data)}, type={type(data).__name__}")
                else:
                    self.logger.debug(f"Data summary - {data_name}: value={data}, type={type(data).__name__}")

                if summary_func and callable(summary_func):
                    summary = summary_func(data)
                    self.logger.debug(f"Custom summary - {data_name}: {summary}")

            except Exception as e:
                self.logger.warning(f"Could not log data summary for {data_name}: {e}")

    def close(self):
        """Close the file logger"""
        if self.logger:
            self.logger.info("="*80)
            self.logger.info("SPT Analysis Session Ended")
            self.logger.info("="*80)

        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None

        if self.console_handler:
            self.logger.removeHandler(self.console_handler)
            self.console_handler = None

# ==================== U-TRACK DETECTION CLASSES ====================

class UTrackDetector:
    """
    Simplified U-Track detection implementation for FLIKA integration
    """

    def __init__(self, psf_sigma=1.5, alpha_threshold=0.05, min_intensity=0.0):
        self.psf_sigma = psf_sigma
        self.alpha_threshold = alpha_threshold
        self.min_intensity = min_intensity

    def detect_particles_single_frame(self, image, frame_number=0):
        """
        Detect particles in a single frame using U-Track methodology

        Args:
            image: 2D numpy array
            frame_number: Frame index

        Returns:
            pandas DataFrame with columns: x, y, intensity, frame
        """
        try:
            # Ensure 2D array
            if image.ndim > 2:
                if image.shape[0] == 1:
                    image = image[0]
                else:
                    raise ValueError(f"Cannot handle 3D image with shape {image.shape}")

            # Ensure we have a valid image
            if image.size == 0:
                return self._empty_detection_result(frame_number)

            # Step 1: Background estimation
            bg_mean, bg_std = self._estimate_background(image)

            # Step 2: Apply Gaussian pre-filtering
            try:
                filtered_img = filters.gaussian(image, sigma=self.psf_sigma)
            except:
                # Fallback: use scipy gaussian filter
                from scipy.ndimage import gaussian_filter
                filtered_img = gaussian_filter(image.astype(np.float32), sigma=self.psf_sigma)

            # Step 3: Find local maxima using scipy approach (more robust)
            local_maxima = self._find_local_maxima_scipy(filtered_img)

            # Get coordinates and intensities
            max_coords = np.where(local_maxima)
            if len(max_coords[0]) == 0:
                return self._empty_detection_result(frame_number)

            y_coords, x_coords = max_coords
            intensities = filtered_img[max_coords]

            # Step 4: Statistical significance testing
            bg_mean_vals = bg_mean[max_coords] if isinstance(bg_mean, np.ndarray) else bg_mean
            bg_std_vals = bg_std[max_coords] if isinstance(bg_std, np.ndarray) else bg_std

            # Test H0: intensity comes from background N(μ_bg, σ_bg)
            p_values = 1 - stats.norm.cdf(intensities, bg_mean_vals, bg_std_vals)
            significant_mask = p_values < self.alpha_threshold

            # Filter by significance and minimum intensity
            intensity_mask = intensities >= self.min_intensity
            final_mask = significant_mask & intensity_mask

            if not np.any(final_mask):
                return self._empty_detection_result(frame_number)

            # Step 5: Sub-pixel localization (centroid-based)
            x_refined, y_refined, intensities_refined = self._subpixel_localization(
                image, x_coords[final_mask], y_coords[final_mask]
            )

            # Create result DataFrame
            detections = pd.DataFrame({
                'x': x_refined,
                'y': y_refined,
                'intensity': intensities_refined,
                'frame': frame_number
            })

            return detections

        except Exception as e:
            print(f"Detection error in frame {frame_number}: {e}")
            return self._empty_detection_result(frame_number)

    def _estimate_background(self, image):
        """Estimate background using robust statistics"""
        # Use percentile-based background estimation
        bg_percentile = 25  # Use 25th percentile as background estimate
        bg_mean = np.percentile(image, bg_percentile)

        # Estimate noise using MAD (Median Absolute Deviation)
        median_val = np.median(image)
        mad = np.median(np.abs(image - median_val))
        bg_std = mad * 1.4826  # Convert MAD to std

        return bg_mean, max(bg_std, 1.0)  # Ensure minimum std

    def _subpixel_localization(self, image, x_coords, y_coords):
        """Sub-pixel localization using intensity-weighted centroid"""
        window_size = int(2 * self.psf_sigma)
        x_refined = []
        y_refined = []
        intensities_refined = []

        for x, y in zip(x_coords, y_coords):
            # Extract local region
            x_min = max(0, x - window_size)
            x_max = min(image.shape[1], x + window_size + 1)
            y_min = max(0, y - window_size)
            y_max = min(image.shape[0], y + window_size + 1)

            local_region = image[y_min:y_max, x_min:x_max]

            if local_region.size > 0:
                # Calculate intensity-weighted centroid
                y_indices, x_indices = np.mgrid[0:local_region.shape[0], 0:local_region.shape[1]]
                total_intensity = np.sum(local_region)

                if total_intensity > 0:
                    centroid_x = np.sum(x_indices * local_region) / total_intensity + x_min
                    centroid_y = np.sum(y_indices * local_region) / total_intensity + y_min
                    intensity = total_intensity / local_region.size  # Average intensity
                else:
                    centroid_x, centroid_y = float(x), float(y)
                    intensity = float(image[y, x])

                x_refined.append(centroid_x)
                y_refined.append(centroid_y)
                intensities_refined.append(intensity)

        return np.array(x_refined), np.array(y_refined), np.array(intensities_refined)

    def _find_local_maxima_manual(self, image, window_size):
        """Manual local maxima detection as fallback"""
        from scipy.ndimage import maximum_filter

        # Use maximum filter to find local maxima
        local_max_mask = (image == maximum_filter(image, size=window_size))

        # Remove edge pixels to avoid boundary effects
        border = window_size // 2
        local_max_mask[:border, :] = False
        local_max_mask[-border:, :] = False
        local_max_mask[:, :border] = False
        local_max_mask[:, -border:] = False

        return local_max_mask

    def _find_local_maxima_scipy(self, image):
        """Find local maxima using scipy maximum_filter - most robust approach"""
        from scipy.ndimage import maximum_filter

        # Calculate window size
        window_size = max(3, int(2 * self.psf_sigma + 1))
        if window_size % 2 == 0:
            window_size += 1

        # Find local maxima
        local_max_filtered = maximum_filter(image, size=window_size)
        local_maxima = (image == local_max_filtered)

        # Remove edge pixels to avoid boundary effects
        border = window_size // 2
        if border > 0:
            local_maxima[:border, :] = False
            local_maxima[-border:, :] = False
            local_maxima[:, :border] = False
            local_maxima[:, -border:] = False

        # Additional filtering: remove maxima that are too close to image edges
        edge_buffer = max(3, int(self.psf_sigma))
        local_maxima[:edge_buffer, :] = False
        local_maxima[-edge_buffer:, :] = False
        local_maxima[:, :edge_buffer] = False
        local_maxima[:, -edge_buffer:] = False

        return local_maxima

    def _empty_detection_result(self, frame_number):
        """Return empty detection DataFrame"""
        return pd.DataFrame({
            'x': pd.Series([], dtype=np.float64),
            'y': pd.Series([], dtype=np.float64),
            'intensity': pd.Series([], dtype=np.float64),
            'frame': pd.Series([], dtype=np.int64)
        })


class DetectionWorker(QThread):
    """Worker thread for running detection on image sequences"""

    progress_update = Signal(str)
    frame_progress = Signal(int)
    detection_complete = Signal(str)  # Output file path
    detection_error = Signal(str)
    file_processed = Signal(str, str)  # (image_file_path, detection_file_path) for visualization

    def __init__(self, file_paths, detector_params, output_dir, pixel_size, show_results=False):
        super().__init__()
        self.file_paths = file_paths
        self.detector_params = detector_params
        self.output_dir = output_dir
        self.pixel_size = pixel_size
        self.show_results = show_results

    def run(self):
        """Run detection on all files"""
        try:
            total_files = len(self.file_paths)

            for file_idx, file_path in enumerate(self.file_paths):
                self.progress_update.emit(f"Processing file {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")

                # Load image sequence
                try:
                    images = skio.imread(file_path, plugin='tifffile')
                    if images.ndim == 2:
                        images = images[np.newaxis, ...]  # Add time dimension

                    # Apply same transformations as in main plugin
                    images = np.rot90(images, axes=(1,2))
                    images = np.fliplr(images)

                except Exception as e:
                    self.detection_error.emit(f"Error loading {file_path}: {e}")
                    continue

                # Create detector
                detector = UTrackDetector(
                    psf_sigma=self.detector_params['psf_sigma'],
                    alpha_threshold=self.detector_params['alpha_threshold'],
                    min_intensity=self.detector_params['min_intensity']
                )

                # Detect particles in each frame
                all_detections = []
                n_frames = images.shape[0]

                for frame_idx in range(n_frames):
                    frame_detections = detector.detect_particles_single_frame(
                        images[frame_idx], frame_idx
                    )

                    if len(frame_detections) > 0:
                        all_detections.append(frame_detections)

                    # Update progress
                    progress = int((frame_idx + 1) / n_frames * 100)
                    self.frame_progress.emit(progress)

                # Combine all detections
                if all_detections:
                    combined_detections = pd.concat(all_detections, ignore_index=True)

                    # Convert coordinates to nanometers and add required columns
                    combined_detections['x [nm]'] = combined_detections['x'] * self.pixel_size
                    combined_detections['y [nm]'] = combined_detections['y'] * self.pixel_size
                    combined_detections['frame'] = combined_detections['frame'] + 1  # 1-based indexing
                    combined_detections['intensity [photon]'] = combined_detections['intensity']
                    combined_detections['id'] = range(len(combined_detections))

                    # Save in format expected by tracking pipeline
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    output_file = os.path.join(self.output_dir, f"{base_name}_locsID.csv")

                    # Select columns in expected order
                    output_columns = ['frame', 'x [nm]', 'y [nm]', 'intensity [photon]', 'id']
                    combined_detections[output_columns].to_csv(output_file, index=False)

                    self.progress_update.emit(f"Saved {len(combined_detections)} detections to {os.path.basename(output_file)}")

                    # Emit signal for visualization if requested
                    if self.show_results:
                        self.file_processed.emit(file_path, output_file)

                else:
                    self.progress_update.emit(f"No detections found in {os.path.basename(file_path)}")

            self.detection_complete.emit("Detection completed successfully!")

        except Exception as e:
            self.detection_error.emit(f"Detection failed: {str(e)}")



# ==================== U-TRACK WITH MIXED MOTION MODELS INTEGRATION ====================

@dataclass
class TrackingConfig:
    """Enhanced config class for U-Track with mixed motion model compatibility"""
    max_linking_distance: float = 10.0
    max_gap_frames: int = 5
    min_track_length: int = 3
    motion_model: str = 'mixed'  # 'brownian', 'linear', or 'mixed' (PMMS)
    linking_distance_auto: bool = False
    enable_merging: bool = False
    enable_splitting: bool = False

    # PMMS-specific parameters
    enable_iterative_smoothing: bool = True
    num_tracking_rounds: int = 3  # Forward-Reverse-Forward
    motion_regime_detection_sensitivity: float = 0.8
    adaptive_search_radius: bool = True
    min_regime_length: int = 3  # Minimum frames in a motion regime

    # Mixed motion transition parameters
    transition_probability_brownian_to_linear: float = 0.1
    transition_probability_linear_to_brownian: float = 0.1
    brownian_noise_multiplier: float = 3.0
    linear_velocity_persistence: float = 0.8

def get_config():
    """Simple config provider for U-Track compatibility"""
    class SimpleConfig:
        def __init__(self):
            self.tracking = TrackingConfig()
    return SimpleConfig()

def get_logger(name):
    """Simple logger for U-Track compatibility"""
    import logging
    return logging.getLogger(name)

def log_function_call(log_timing=False):
    """Simple decorator for U-Track compatibility"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

class MotionRegimeDetector:
    """
    Detects changes in motion regimes (Brownian -> Linear -> Confined, etc.)
    Based on PMMS algorithm principles
    """

    def __init__(self, sensitivity=0.8, min_regime_length=3):
        self.sensitivity = sensitivity
        self.min_regime_length = min_regime_length
        self.logger = get_logger('motion_regime_detector')

    def detect_motion_regimes(self, trajectory: np.ndarray) -> List[Dict]:
        """
        Detect motion regime changes in a trajectory

        Args:
            trajectory: Nx3 array [frame, x, y]

        Returns:
            List of regime dictionaries with start, end, and type
        """
        if len(trajectory) < self.min_regime_length * 2:
            return [{'start': 0, 'end': len(trajectory)-1, 'type': 'brownian'}]

        regimes = []
        current_regime_start = 0
        current_regime_type = self._classify_motion_segment(trajectory[0:self.min_regime_length])

        # Sliding window analysis
        window_size = self.min_regime_length
        for i in range(window_size, len(trajectory) - window_size + 1):
            window_trajectory = trajectory[i-window_size+1:i+window_size]
            window_type = self._classify_motion_segment(window_trajectory)

            # Check for regime change
            if window_type != current_regime_type:
                # End current regime
                regimes.append({
                    'start': current_regime_start,
                    'end': i,
                    'type': current_regime_type
                })

                # Start new regime
                current_regime_start = i
                current_regime_type = window_type

        # Add final regime
        regimes.append({
            'start': current_regime_start,
            'end': len(trajectory)-1,
            'type': current_regime_type
        })

        return self._merge_short_regimes(regimes)

    def _classify_motion_segment(self, segment: np.ndarray) -> str:
        """Classify motion type for a trajectory segment"""
        if len(segment) < 3:
            return 'brownian'

        # Calculate displacement statistics
        displacements = np.diff(segment[:, 1:], axis=0)  # [dx, dy]
        displacement_magnitudes = np.linalg.norm(displacements, axis=1)

        if len(displacement_magnitudes) < 2:
            return 'brownian'

        # Directional persistence
        if len(displacements) >= 2:
            dot_products = []
            for i in range(len(displacements)-1):
                if np.linalg.norm(displacements[i]) > 0 and np.linalg.norm(displacements[i+1]) > 0:
                    dot_product = np.dot(displacements[i], displacements[i+1])
                    dot_product /= (np.linalg.norm(displacements[i]) * np.linalg.norm(displacements[i+1]))
                    dot_products.append(dot_product)

            if dot_products:
                mean_directional_persistence = np.mean(dot_products)

                # Classification based on directional persistence
                if mean_directional_persistence > 0.5:
                    return 'linear'
                elif mean_directional_persistence < -0.1:
                    return 'confined'
                else:
                    return 'brownian'

        return 'brownian'

    def _merge_short_regimes(self, regimes: List[Dict]) -> List[Dict]:
        """Merge regimes shorter than minimum length with neighbors"""
        if len(regimes) <= 1:
            return regimes

        merged_regimes = []
        i = 0
        while i < len(regimes):
            current_regime = regimes[i].copy()
            regime_length = current_regime['end'] - current_regime['start'] + 1

            if regime_length < self.min_regime_length and i < len(regimes) - 1:
                # Merge with next regime
                next_regime = regimes[i + 1]
                current_regime['end'] = next_regime['end']
                current_regime['type'] = next_regime['type']  # Take type of longer regime
                i += 2  # Skip next regime since we merged it
            else:
                i += 1

            merged_regimes.append(current_regime)

        return merged_regimes

class MixedMotionPredictor:
    """
    Multi-model motion predictor that maintains multiple motion hypotheses
    Inspired by PMMS and Interacting Multiple Models (IMM)
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.logger = get_logger('mixed_motion_predictor')

        # Motion model weights
        self.model_weights = {
            'brownian': 0.4,
            'linear': 0.4,
            'confined': 0.2
        }

        # Transition probabilities matrix
        self.transition_probs = {
            ('brownian', 'brownian'): 0.8,
            ('brownian', 'linear'): config.transition_probability_brownian_to_linear,
            ('brownian', 'confined'): 0.1,
            ('linear', 'linear'): 0.8,
            ('linear', 'brownian'): config.transition_probability_linear_to_brownian,
            ('linear', 'confined'): 0.1,
            ('confined', 'confined'): 0.8,
            ('confined', 'brownian'): 0.15,
            ('confined', 'linear'): 0.05,
        }

    def predict_multiple_models(self, state_history: List[np.ndarray],
                              dt: float = 1.0) -> Dict[str, Dict]:
        """
        Generate predictions from multiple motion models

        Args:
            state_history: List of previous states [x, y, vx, vy]
            dt: Time step

        Returns:
            Dict of predictions for each motion model
        """
        predictions = {}

        if len(state_history) < 1:
            # Initialize with zero velocity
            state = np.array([0, 0, 0, 0])
            predictions['brownian'] = self._predict_brownian(state, dt)
            predictions['linear'] = self._predict_linear(state, dt)
            predictions['confined'] = self._predict_confined(state, dt)
        else:
            last_state = state_history[-1]
            predictions['brownian'] = self._predict_brownian(last_state, dt)
            predictions['linear'] = self._predict_linear(last_state, dt)
            predictions['confined'] = self._predict_confined(last_state, dt)

            # Update model weights based on recent trajectory
            if len(state_history) >= 3:
                self._update_model_weights(state_history)

        return predictions

    def _predict_brownian(self, state: np.ndarray, dt: float) -> Dict:
        """Brownian motion prediction"""
        x, y, vx, vy = state

        # Brownian motion: position changes with some noise, velocity is random
        noise_scale = self.config.brownian_noise_multiplier

        predicted_state = np.array([
            x + vx * dt + np.random.normal(0, noise_scale * dt),
            y + vy * dt + np.random.normal(0, noise_scale * dt),
            np.random.normal(0, noise_scale),  # Random velocity
            np.random.normal(0, noise_scale)
        ])

        # Covariance matrix (higher uncertainty)
        covariance = np.diag([noise_scale**2 * dt, noise_scale**2 * dt,
                             noise_scale**2, noise_scale**2])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['brownian']
        }

    def _predict_linear(self, state: np.ndarray, dt: float) -> Dict:
        """Linear/directed motion prediction"""
        x, y, vx, vy = state

        # Linear motion: velocity persists with some noise
        velocity_persistence = self.config.linear_velocity_persistence
        noise_scale = 1.0  # Lower noise for directed motion

        predicted_state = np.array([
            x + vx * dt,
            y + vy * dt,
            vx * velocity_persistence + np.random.normal(0, noise_scale * 0.1),
            vy * velocity_persistence + np.random.normal(0, noise_scale * 0.1)
        ])

        # Covariance matrix (lower uncertainty in velocity direction)
        covariance = np.diag([noise_scale * dt, noise_scale * dt,
                             noise_scale * 0.1, noise_scale * 0.1])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['linear']
        }

    def _predict_confined(self, state: np.ndarray, dt: float) -> Dict:
        """Confined motion prediction"""
        x, y, vx, vy = state

        # Confined motion: tends to return to center, velocity decays
        confinement_strength = 0.1
        noise_scale = 2.0

        # Assume confinement center is at (0,0) - could be estimated from track history
        predicted_state = np.array([
            x + vx * dt - confinement_strength * x * dt,
            y + vy * dt - confinement_strength * y * dt,
            vx * 0.8 - confinement_strength * x,  # Velocity decays and pulls toward center
            vy * 0.8 - confinement_strength * y
        ])

        # Covariance matrix
        covariance = np.diag([noise_scale * dt, noise_scale * dt,
                             noise_scale * 0.5, noise_scale * 0.5])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['confined']
        }

    def _update_model_weights(self, state_history: List[np.ndarray]):
        """Update model weights based on recent motion patterns"""
        if len(state_history) < 3:
            return

        # Analyze recent motion to update model probabilities
        recent_states = state_history[-3:]

        # Calculate motion characteristics
        velocities = []
        for i in range(len(recent_states)-1):
            dx = recent_states[i+1][0] - recent_states[i][0]
            dy = recent_states[i+1][1] - recent_states[i][1]
            velocities.append([dx, dy])

        velocities = np.array(velocities)

        if len(velocities) >= 2:
            # Velocity consistency (for linear motion)
            velocity_consistency = np.mean([
                np.dot(velocities[i], velocities[i+1]) /
                (np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[i+1]) + 1e-6)
                for i in range(len(velocities)-1)
            ])

            # Speed variability (for Brownian vs confined)
            speeds = np.linalg.norm(velocities, axis=1)
            speed_variability = np.std(speeds) / (np.mean(speeds) + 1e-6)

            # Update weights based on observations
            if velocity_consistency > 0.5:
                self.model_weights['linear'] *= 1.2
            else:
                self.model_weights['linear'] *= 0.8

            if speed_variability > 1.0:
                self.model_weights['brownian'] *= 1.2
            else:
                self.model_weights['brownian'] *= 0.9

            # Normalize weights
            total_weight = sum(self.model_weights.values())
            for key in self.model_weights:
                self.model_weights[key] /= total_weight

class UTrackLinkerWithMixedMotion:
    """
    Enhanced U-Track linker with mixed motion model support (PMMS-style)
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.logger = get_logger('utrack_mixed_motion')

        # Initialize motion predictor and regime detector
        self.motion_predictor = MixedMotionPredictor(self.config)
        self.regime_detector = MotionRegimeDetector(
            sensitivity=self.config.motion_regime_detection_sensitivity,
            min_regime_length=self.config.min_regime_length
        )

        # Track state histories for mixed motion prediction
        self.track_histories: Dict[int, List[np.ndarray]] = {}

        self.logger.info(f"U-Track with mixed motion initialized - Model: {self.config.motion_model}")

    def track_particles(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Main tracking function with mixed motion model support

        Args:
            detections: List of detection DataFrames, one per frame

        Returns:
            DataFrame with complete tracks
        """
        if len(detections) < 1:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        total_particles = sum(len(det) for det in detections)
        self.logger.info(f"Processing {len(detections)} frames with {total_particles} total particles using {self.config.motion_model} motion model")

        try:
            if self.config.motion_model == 'mixed' and self.config.enable_iterative_smoothing:
                return self._track_with_iterative_smoothing(detections)
            else:
                return self._track_with_single_model(detections)

        except Exception as e:
            self.logger.error(f"Mixed motion tracking failed: {e}")
            # Fallback to simple tracking
            return self._fallback_simple_tracking(detections)

    def _track_with_iterative_smoothing(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """
        PMMS-style tracking with iterative smoothing (Forward-Reverse-Forward)
        """
        self.logger.info("Running PMMS-style tracking with iterative smoothing")

        # Initialize with empty tracks
        best_tracks = pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        for round_num in range(self.config.num_tracking_rounds):
            self.logger.debug(f"Tracking round {round_num + 1}/{self.config.num_tracking_rounds}")

            if round_num == 0:
                # Forward pass
                tracks = self._forward_tracking_pass(detections)
            elif round_num == 1:
                # Reverse pass
                tracks = self._reverse_tracking_pass(detections, best_tracks)
            else:
                # Final forward pass with regime information
                tracks = self._final_forward_pass(detections, best_tracks)

            if len(tracks) > len(best_tracks):
                best_tracks = tracks
                self.logger.debug(f"Improved tracks: {len(tracks)} points")

        return best_tracks

    def _forward_tracking_pass(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Forward tracking pass with mixed motion prediction"""
        tracks = []
        active_tracks: Dict[int, Dict] = {}
        next_track_id = 1

        for frame_idx, frame_detections in enumerate(detections):
            if len(frame_detections) == 0:
                continue

            current_detections = frame_detections.copy().reset_index(drop=True)
            frame_assignments = {}

            # Predict existing track positions using mixed motion models
            predicted_positions = {}
            for track_id, track_info in active_tracks.items():
                if self.config.motion_model == 'mixed':
                    predictions = self.motion_predictor.predict_multiple_models(
                        track_info['state_history'])

                    # Use weighted combination of predictions
                    combined_prediction = self._combine_motion_predictions(predictions)
                    predicted_positions[track_id] = combined_prediction
                else:
                    # Single model prediction
                    last_state = track_info['state_history'][-1] if track_info['state_history'] else np.array([0, 0, 0, 0])
                    predicted_positions[track_id] = self._predict_single_model(last_state)

            # Assignment using LAP
            if active_tracks and len(current_detections) > 0:
                assignments = self._solve_assignment_problem(
                    predicted_positions, current_detections, frame_idx)
                frame_assignments.update(assignments)

            # Update existing tracks and create new ones
            unassigned_detections = set(range(len(current_detections)))

            for track_id, detection_idx in frame_assignments.items():
                if detection_idx in unassigned_detections:
                    detection = current_detections.iloc[detection_idx]

                    # Update track
                    tracks.append({
                        'particle_id': detection.get('particle_id', detection_idx),
                        'track_id': track_id,
                        'frame': frame_idx,
                        'x': detection['x'],
                        'y': detection['y'],
                        'intensity': detection.get('intensity', 100.0)
                    })

                    # Update state history
                    new_state = np.array([detection['x'], detection['y'], 0, 0])
                    if track_id in active_tracks:
                        if len(active_tracks[track_id]['state_history']) > 0:
                            prev_state = active_tracks[track_id]['state_history'][-1]
                            new_state[2] = detection['x'] - prev_state[0]  # vx
                            new_state[3] = detection['y'] - prev_state[1]  # vy

                        active_tracks[track_id]['state_history'].append(new_state)
                        active_tracks[track_id]['last_seen'] = frame_idx

                    unassigned_detections.remove(detection_idx)

            # Create new tracks for unassigned detections
            for detection_idx in unassigned_detections:
                detection = current_detections.iloc[detection_idx]

                tracks.append({
                    'particle_id': detection.get('particle_id', detection_idx),
                    'track_id': next_track_id,
                    'frame': frame_idx,
                    'x': detection['x'],
                    'y': detection['y'],
                    'intensity': detection.get('intensity', 100.0)
                })

                # Initialize new track
                active_tracks[next_track_id] = {
                    'state_history': [np.array([detection['x'], detection['y'], 0, 0])],
                    'last_seen': frame_idx
                }

                next_track_id += 1

            # Remove old tracks
            active_tracks = {
                track_id: info for track_id, info in active_tracks.items()
                if frame_idx - info['last_seen'] <= self.config.max_gap_frames
            }

        return pd.DataFrame(tracks) if tracks else pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

    def _reverse_tracking_pass(self, detections: List[pd.DataFrame],
                             forward_tracks: pd.DataFrame) -> pd.DataFrame:
        """Reverse tracking pass to improve track continuity"""
        if len(forward_tracks) == 0:
            return forward_tracks

        # Process frames in reverse order
        reversed_detections = list(reversed(detections))
        reversed_frame_indices = list(reversed(range(len(detections))))

        # Similar to forward pass but in reverse
        tracks = []
        active_tracks = {}

        # Initialize with tracks from forward pass
        for track_id in forward_tracks['track_id'].unique():
            track_data = forward_tracks[forward_tracks['track_id'] == track_id].sort_values('frame', ascending=False)
            if len(track_data) > 0:
                first_point = track_data.iloc[0]
                active_tracks[track_id] = {
                    'state_history': [np.array([first_point['x'], first_point['y'], 0, 0])],
                    'last_seen': first_point['frame']
                }

        # Process in reverse (implementation similar to forward pass)
        # ... (detailed implementation would be similar to forward pass but processing frames backward)

        # For now, return forward tracks (simplified implementation)
        return forward_tracks

    def _final_forward_pass(self, detections: List[pd.DataFrame],
                           previous_tracks: pd.DataFrame) -> pd.DataFrame:
        """Final forward pass incorporating motion regime information"""
        # This would incorporate regime detection results to improve tracking
        # For now, return previous tracks with potential improvements
        return previous_tracks

    def _track_with_single_model(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Standard tracking with single motion model"""
        self.logger.info(f"Running single motion model tracking: {self.config.motion_model}")
        return self._forward_tracking_pass(detections)

    def _combine_motion_predictions(self, predictions: Dict[str, Dict]) -> np.ndarray:
        """Combine predictions from multiple motion models"""
        combined_state = np.zeros(4)  # [x, y, vx, vy]
        total_weight = 0

        for model_name, prediction in predictions.items():
            weight = prediction['likelihood']
            combined_state += weight * prediction['state']
            total_weight += weight

        if total_weight > 0:
            combined_state /= total_weight

        return combined_state

    def _predict_single_model(self, state: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Single model prediction based on config"""
        if self.config.motion_model == 'brownian':
            return self.motion_predictor._predict_brownian(state, dt)['state']
        elif self.config.motion_model == 'linear':
            return self.motion_predictor._predict_linear(state, dt)['state']
        else:
            return self.motion_predictor._predict_confined(state, dt)['state']

    def _solve_assignment_problem(self, predicted_positions: Dict[int, np.ndarray],
                                detections: pd.DataFrame, frame_idx: int) -> Dict[int, int]:
        """Solve assignment problem between predictions and detections"""
        if not predicted_positions or len(detections) == 0:
            return {}

        track_ids = list(predicted_positions.keys())

        # Create cost matrix
        cost_matrix = np.full((len(track_ids), len(detections)), np.inf)

        for i, track_id in enumerate(track_ids):
            pred_pos = predicted_positions[track_id][:2]  # [x, y]

            for j, (_, detection) in enumerate(detections.iterrows()):
                det_pos = np.array([detection['x'], detection['y']])
                distance = np.linalg.norm(pred_pos - det_pos)

                if distance <= self.config.max_linking_distance:
                    cost_matrix[i, j] = distance ** 2

        # Solve using LAP
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            assignments = {}
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < np.inf:
                    assignments[track_ids[row]] = col

            return assignments

        except Exception as e:
            self.logger.warning(f"LAP solving failed: {e}")
            return {}

    def _fallback_simple_tracking(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Simple fallback tracking method"""
        tracks = []
        track_id = 1

        for frame_idx, frame_detections in enumerate(detections):
            for _, detection in frame_detections.iterrows():
                tracks.append({
                    'particle_id': track_id,
                    'track_id': track_id,
                    'frame': frame_idx,
                    'x': detection['x'],
                    'y': detection['y'],
                    'intensity': detection.get('intensity', 100.0)
                })
                track_id += 1

        return pd.DataFrame(tracks) if tracks else pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

class UTrackLinkerAdapter:
    """
    Enhanced adapter class to integrate U-Track with mixed motion models
    """

    def __init__(self, parameters):
        self.parameters = parameters
        self.logger = get_logger('utrack_mixed_adapter')

    def link_particles(self, txy_pts, file_path):
        """
        Main linking method with mixed motion model support
        """
        try:
            self.logger.debug(f"Starting U-Track linking with mixed motion models")

            # Extract data and preserve IDs
            if txy_pts.shape[1] == 4:
                point_ids = txy_pts[:, 3]
                linking_data = txy_pts[:, :3]
            else:
                linking_data = txy_pts
                point_ids = np.arange(len(txy_pts))

            # Convert to U-Track format
            detections = self._convert_to_utrack_format(linking_data, point_ids)

            # Create enhanced U-Track config
            config = self._create_mixed_motion_config()

            # Import and use enhanced UTrackLinker
            try:
                # Try to use the enhanced linker with mixed motion support
                linker = UTrackLinkerWithMixedMotion(config)
                tracks_df = linker.track_particles(detections)

                self.logger.debug(f"Mixed motion U-Track generated {len(tracks_df)} track points")

            except Exception as e:
                self.logger.error(f"Mixed motion U-Track linking failed: {e}")
                # Fallback to original implementation
                from utrack_linking import UTrackLinker
                linker = UTrackLinker(config)
                tracks_df = linker.track_particles(detections)

            # Convert results back to Points-like structure
            points_adapter = self._convert_from_utrack_format(tracks_df, file_path)

            self.logger.debug(f"Mixed motion U-Track linking successful: {len(points_adapter.tracks)} tracks")
            return points_adapter

        except Exception as e:
            self.logger.error(f"Error in mixed motion U-Track adapter: {e}")
            return None

    def _convert_to_utrack_format(self, linking_data, point_ids):
        """Convert FLIKA txy_pts to U-Track format"""
        detections = []

        if len(linking_data) == 0:
            return detections

        frames = np.unique(linking_data[:, 0]).astype(int)
        max_frame = int(np.max(frames))

        for frame_idx in range(max_frame + 1):
            frame_mask = linking_data[:, 0] == frame_idx
            frame_particles = linking_data[frame_mask]
            frame_point_ids = point_ids[frame_mask]

            if len(frame_particles) > 0:
                frame_df = pd.DataFrame({
                    'x': frame_particles[:, 1],
                    'y': frame_particles[:, 2],
                    'intensity': np.full(len(frame_particles), 100.0),
                    'particle_id': frame_point_ids
                })
            else:
                frame_df = pd.DataFrame(columns=['x', 'y', 'intensity', 'particle_id'])

            detections.append(frame_df)

        return detections

    def _create_mixed_motion_config(self):
        """Create enhanced U-Track config with mixed motion parameters"""
        config = TrackingConfig()

        # Basic parameters
        config.max_linking_distance = getattr(self.parameters, 'utrack_max_linking_distance', 10.0)
        config.max_gap_frames = getattr(self.parameters, 'utrack_max_gap_frames', 5)
        config.min_track_length = getattr(self.parameters, 'min_track_segments', 3)
        config.motion_model = getattr(self.parameters, 'utrack_motion_model', 'mixed')
        config.linking_distance_auto = getattr(self.parameters, 'utrack_auto_linking_distance', False)
        config.enable_merging = getattr(self.parameters, 'utrack_enable_merging', False)
        config.enable_splitting = getattr(self.parameters, 'utrack_enable_splitting', False)

        # Mixed motion parameters
        config.enable_iterative_smoothing = getattr(self.parameters, 'utrack_enable_iterative_smoothing', True)
        config.num_tracking_rounds = getattr(self.parameters, 'utrack_num_tracking_rounds', 3)
        config.motion_regime_detection_sensitivity = getattr(self.parameters, 'utrack_regime_sensitivity', 0.8)
        config.adaptive_search_radius = getattr(self.parameters, 'utrack_adaptive_search_radius', True)
        config.min_regime_length = getattr(self.parameters, 'utrack_min_regime_length', 3)

        # Transition probabilities
        config.transition_probability_brownian_to_linear = getattr(self.parameters, 'utrack_trans_prob_b2l', 0.1)
        config.transition_probability_linear_to_brownian = getattr(self.parameters, 'utrack_trans_prob_l2b', 0.1)
        config.brownian_noise_multiplier = getattr(self.parameters, 'utrack_brownian_noise_mult', 3.0)
        config.linear_velocity_persistence = getattr(self.parameters, 'utrack_linear_velocity_persist', 0.8)

        return config

    def _convert_from_utrack_format(self, tracks_df, file_path):
        """Convert U-Track results back to Points-like structure"""
        class UTrackMixedPointsAdapter:
            def __init__(self, tracks_df, file_path, logger):
                self.tracks_df = tracks_df.copy().reset_index(drop=True)
                self.recursiveFailure = False
                self.logger = logger

                # Create tracks list
                self.tracks = []
                track_ids = sorted(tracks_df['track_id'].unique()) if len(tracks_df) > 0 else []

                for track_id in track_ids:
                    track_indices = tracks_df[tracks_df['track_id'] == track_id].index.tolist()
                    if len(track_indices) > 0:
                        self.tracks.append(track_indices)

                # Create txy_pts array
                if len(tracks_df) > 0:
                    self.txy_pts = tracks_df[['frame', 'x', 'y']].values
                    self.point_ids = tracks_df['particle_id'].values if 'particle_id' in tracks_df.columns else np.arange(len(tracks_df))
                else:
                    self.txy_pts = np.array([]).reshape(0, 3)
                    self.point_ids = np.array([])

                # Calculate intensities
                self.intensities = []
                self._calculate_intensities(file_path)

            def _calculate_intensities(self, file_path):
                """Calculate intensities from image data"""
                try:
                    if len(self.tracks_df) == 0:
                        return

                    if os.path.exists(file_path):
                        import skimage.io as skio
                        A = skio.imread(file_path, plugin='tifffile')
                        A = np.rot90(A, axes=(1,2))
                        A = np.fliplr(A)

                        n, w, h = A.shape

                        for _, row in self.tracks_df.iterrows():
                            frame = int(round(row['frame']))
                            x = int(round(row['x']))
                            y = int(round(row['y']))

                            xMin = max(0, x - 1)
                            xMax = min(w, x + 2)
                            yMin = max(0, y - 1)
                            yMax = min(h, y + 2)

                            if frame >= n:
                                frame = n - 1
                            if frame < 0:
                                frame = 0

                            intensity = np.mean(A[frame][yMin:yMax, xMin:xMax])
                            self.intensities.append(intensity)
                    else:
                        self.intensities = [100.0] * len(self.tracks_df)

                except Exception as e:
                    self.logger.error(f"Error calculating intensities: {e}")
                    self.intensities = [100.0] * len(self.tracks_df) if len(self.tracks_df) > 0 else []

        return UTrackMixedPointsAdapter(tracks_df, file_path, self.logger)






# ==================== TRACKING CLASSES  ====================


def get_plugin_directory():
    """Get the directory where this plugin is located"""
    current_file = os.path.abspath(__file__)
    plugin_dir = os.path.dirname(current_file)
    return plugin_dir


def get_default_training_data_path():
    """Get the default path to the training data file"""
    plugin_dir = get_plugin_directory()
    training_file = os.path.join(plugin_dir, 'training_data', 'tdTomato_37Degree_CytoD_training_feats.csv')

    if os.path.exists(training_file):
        return training_file
    else:
        training_dir = os.path.join(plugin_dir, 'training_data')
        os.makedirs(training_dir, exist_ok=True)
        return training_dir


class ROIBackgroundSubtractor:
    """ROI-based background subtraction with proper coordinate mapping and frame-specific options"""

    @staticmethod
    def parse_roi_file(roi_file_path):
        """Parse ROI file to extract rectangle coordinates
        Expected format:
        rectangle
        y1 x1
        width height
        Where (y1,x1) is the top-left corner
        """
        try:
            if not os.path.exists(roi_file_path):
                return None

            with open(roi_file_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) < 3:
                return None

            roi_type = lines[0].lower()
            if roi_type != 'rectangle':
                return None

            # Parse coordinates (format: y, x)
            try:
                y1, x1 = map(float, lines[1].split())
                len1, len2 = map(float, lines[2].split())
            except ValueError:
                return None

            # Convert to integer coordinates
            x_min = int(x1)
            y_min = int(y1)
            width = int(len1)
            height = int(len2)

            # Calculate bottom-right corner
            x_max = x_min + width
            y_max = y_min + height

            roi_coords = {
                'type': 'rectangle',
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'width': width,
                'height': height
            }

            return roi_coords

        except Exception:
            return None

    @staticmethod
    def calculate_roi_intensity(image_array, roi_coords, frame_specific=False):
        """Calculate mean intensity within ROI using same methodology as Points class

        Args:
            image_array: 3D numpy array (frames, height, width) - already transformed like in Points.getIntensities
            roi_coords: Dictionary with ROI coordinates from parse_roi_file
            frame_specific: If True, returns per-frame intensities; if False, returns single mean across all frames

        Returns:
            float or np.array:
                - If frame_specific=False: Single mean intensity across all frames
                - If frame_specific=True: Array of mean intensities for each frame
        """
        try:
            if roi_coords is None:
                return 0.0 if not frame_specific else np.array([])

            n_frames, height, width = image_array.shape

            # Extract coordinates
            x_min = roi_coords['x_min']
            x_max = roi_coords['x_max']
            y_min = roi_coords['y_min']
            y_max = roi_coords['y_max']

            # Handle edge cases - ensure coordinates are within image bounds
            x_min = max(0, x_min)
            x_max = min(width, x_max)
            y_min = max(0, y_min)
            y_max = min(height, y_max)

            if x_min >= x_max or y_min >= y_max:
                return 0.0 if not frame_specific else np.zeros(n_frames)

            # Extract ROI region from all frames
            # Using [frame, y_min:y_max, x_min:x_max] indexing like Points class
            roi_region = image_array[:, y_min:y_max, x_min:x_max]

            if frame_specific:
                # Calculate mean intensity for each frame separately
                frame_intensities = np.array([np.mean(roi_region[frame]) for frame in range(n_frames)])
                return frame_intensities
            else:
                # Calculate mean intensity across all pixels and frames (original behavior)
                mean_intensity = np.mean(roi_region)
                return float(mean_intensity)

        except Exception:
            return 0.0 if not frame_specific else np.zeros(n_frames)

    @staticmethod
    def apply_background_subtraction(tracks_df, roi_intensity, camera_black, frame_specific=False):
        """Apply background subtraction to particle intensities

        Args:
            tracks_df: DataFrame with particle tracks
            roi_intensity: Single value (if frame_specific=False) or array of values per frame (if frame_specific=True)
            camera_black: Estimated camera black level (for reference, not used in subtraction)
            frame_specific: Whether to use frame-specific background subtraction

        Returns:
            tuple: (background_subtracted_intensities, frame_specific_backgrounds_used)
                - background_subtracted_intensities: pd.Series with corrected intensities
                - frame_specific_backgrounds_used: pd.Series with background values used (or None for single mode)
        """
        try:
            if frame_specific and isinstance(roi_intensity, np.ndarray):
                # Frame-specific background subtraction
                # Create a mapping from frame number to ROI background intensity
                frame_to_background = {}
                for frame_idx, roi_val in enumerate(roi_intensity):
                    # Use the ROI intensity directly as the background signal
                    frame_to_background[frame_idx] = roi_val

                # Apply frame-specific background subtraction
                intensity_bg_subtracted = tracks_df.apply(
                    lambda row: row['intensity'] - frame_to_background.get(int(row['frame']), 0),
                    axis=1
                )

                # Create series with the actual background values used for each point
                frame_specific_backgrounds = tracks_df.apply(
                    lambda row: frame_to_background.get(int(row['frame']), 0),
                    axis=1
                )

                return intensity_bg_subtracted, frame_specific_backgrounds

            else:
                # Single background value for all frames (original behavior)
                # Use the ROI intensity directly as the background signal
                background_signal = roi_intensity
                intensity_bg_subtracted = tracks_df['intensity'] - background_signal

                # For single mode, all points get the same background value
                return intensity_bg_subtracted, None

        except Exception:
            return tracks_df['intensity'], None  # Return original intensities on error

    @staticmethod
    def estimate_camera_black_level(image_array, percentile=1.0):
        """Estimate camera black level from darkest pixels"""
        try:
            black_level = np.percentile(image_array, percentile)
            return float(black_level)
        except Exception:
            return 0.0

class AutocorrelationAnalyzer:
    """Directional autocorrelation analysis based on Gorelik & Gautreau (2014)"""

    @staticmethod
    def calculate_normed_vectors(df):
        """Calculate normalized vectors for each step in the trajectory."""
        print("  Calculating normalized vectors...")
        start_time = time.time()

        # Create a copy with just the needed columns
        result_df = df.copy()

        # Add columns for normalized vectors
        result_df['x_vector'] = np.nan
        result_df['y_vector'] = np.nan

        # Use pandas diff to find frame discontinuities
        frame_diff = df['frame'].diff()
        new_traj_mask = (frame_diff <= 0).copy()
        new_traj_mask.iloc[0] = True

        # Find trajectory start indices
        traj_starts = list(new_traj_mask[new_traj_mask].index)
        traj_starts.append(len(df))

        print(f"  Found {len(traj_starts)-1} trajectory segments")

        # Process each trajectory
        for i in range(len(traj_starts) - 1):
            if i % 50 == 0 or i == len(traj_starts) - 2:
                print(f"  Processing trajectory segment {i+1}/{len(traj_starts)-1}...")

            start_idx = traj_starts[i]
            end_idx = traj_starts[i+1]

            if end_idx - start_idx < 2:
                continue

            # Get the trajectory segment
            traj = result_df.iloc[start_idx:end_idx].copy()

            # Calculate differences between consecutive points
            dx = traj['x'].diff(-1).iloc[:-1]
            dy = traj['y'].diff(-1).iloc[:-1]

            # Calculate magnitudes
            magnitudes = np.sqrt(dx**2 + dy**2)

            # Find valid movements (non-zero magnitude)
            valid_moves = magnitudes > 0

            # Normalize vectors where magnitude > 0
            if len(dx) > 0:
                result_df.iloc[start_idx+1:end_idx, result_df.columns.get_loc('x_vector')] = \
                    np.where(valid_moves, dx / magnitudes, np.nan)
                result_df.iloc[start_idx+1:end_idx, result_df.columns.get_loc('y_vector')] = \
                    np.where(valid_moves, dy / magnitudes, np.nan)

        elapsed = time.time() - start_time
        print(f"  Normalized vectors calculated in {elapsed:.2f} seconds")

        return result_df, traj_starts

    @staticmethod
    def calculate_scalar_products(df, traj_starts, time_interval, num_intervals):
        """Calculate scalar products of vectors for different time intervals."""
        print("  Calculating scalar products...")
        start_time = time.time()

        # Initialize results dictionary
        combined_scalar_results = {time_interval * step: [] for step in range(1, num_intervals + 1)}
        individual_track_results = {}

        # Process each trajectory
        for i in range(len(traj_starts) - 1):
            if i % 50 == 0 or i == len(traj_starts) - 2:
                print(f"  Processing trajectory {i+1}/{len(traj_starts)-1}...")

            start_idx = traj_starts[i]
            end_idx = traj_starts[i+1]
            traj_length = end_idx - start_idx

            if traj_length < 2:
                continue

            track_id = f"track_{i+1}"
            individual_track_results[track_id] = {}

            max_intervals = min(num_intervals, traj_length)
            traj_vectors = df.iloc[start_idx:end_idx]

            # For each step size
            for step in range(1, max_intervals):
                time_point = time_interval * step

                # Get vectors
                x_vecs1 = traj_vectors['x_vector'].values[:-step]
                y_vecs1 = traj_vectors['y_vector'].values[:-step]
                x_vecs2 = traj_vectors['x_vector'].values[step:]
                y_vecs2 = traj_vectors['y_vector'].values[step:]

                # Calculate dot products
                dot_products = x_vecs1 * x_vecs2 + y_vecs1 * y_vecs2

                # Filter valid values
                valid_mask = ~np.isnan(dot_products)
                valid_dots = dot_products[valid_mask]

                # Add to combined results
                combined_scalar_results[time_point].extend(valid_dots.tolist())

                # Store for this individual track
                if len(valid_dots) > 0:
                    track_avg_corr = np.mean(valid_dots)
                    individual_track_results[track_id][time_point] = track_avg_corr

        # Convert results to DataFrame
        combined_df = pd.DataFrame({k: pd.Series(v) for k, v in combined_scalar_results.items()})

        # Convert individual track results
        track_data = []
        for track_id, time_points in individual_track_results.items():
            for time_point, corr in time_points.items():
                track_data.append({
                    'track_id': track_id,
                    'time_interval': time_point,
                    'correlation': corr
                })

        tracks_df = pd.DataFrame(track_data)

        elapsed = time.time() - start_time
        print(f"  Scalar products calculated in {elapsed:.2f} seconds")

        return combined_df, tracks_df

    @staticmethod
    def calculate_averages(scalar_products):
        """Calculate averages and standard errors for each time interval."""
        results = pd.DataFrame(index=['AVG', 'SEM'])
        results[0] = [1, 0]  # Perfect correlation at time=0

        for col in scalar_products.columns:
            values = scalar_products[col].dropna()

            avg = values.mean()
            n = len(values)
            sem = values.std() / np.sqrt(n) if n > 0 else 0

            results[col] = [avg, sem]

        return results

    @staticmethod
    def identify_columns(df):
        """Identify required columns in the dataframe."""
        cols_lower = {col.lower(): col for col in df.columns}

        # Look for frame column
        frame_col = None
        for opt in ['frame', 'frames', 'frame_number', 'frameno', 'time', 'f', '#frame']:
            if opt in cols_lower:
                frame_col = cols_lower[opt]
                break

        # Look for x column
        x_col = None
        for opt in ['x', 'x_coord', 'x_coordinate', 'xpos', 'x_position']:
            if opt in cols_lower:
                x_col = cols_lower[opt]
                break

        # Look for y column
        y_col = None
        for opt in ['y', 'y_coord', 'y_coordinate', 'ypos', 'y_position']:
            if opt in cols_lower:
                y_col = cols_lower[opt]
                break

        if frame_col is None or x_col is None or y_col is None:
            return None

        result_df = pd.DataFrame({
            'frame': df[frame_col],
            'x': df[x_col],
            'y': df[y_col]
        })

        return result_df

    @staticmethod
    def process_track_data(tracks_df, time_interval, num_intervals):
        """Process track data for autocorrelation analysis."""
        # Group by track number and sort by frame
        grouped_tracks = []

        for track_num in tracks_df['track_number'].unique():
            track_data = tracks_df[tracks_df['track_number'] == track_num].sort_values('frame')

            if len(track_data) >= 3:  # Need minimum track length
                # Add track data with consistent frame numbering
                track_subset = track_data[['frame', 'x', 'y']].copy()
                grouped_tracks.append(track_subset)

        if not grouped_tracks:
            return None, None, None

        # Combine all tracks into single dataframe for vector analysis
        combined_df = pd.concat(grouped_tracks, ignore_index=True)

        # Calculate normalized vectors
        vectors_df, traj_starts = AutocorrelationAnalyzer.calculate_normed_vectors(combined_df)

        # Calculate scalar products
        scalar_products, individual_tracks = AutocorrelationAnalyzer.calculate_scalar_products(
            vectors_df, traj_starts, time_interval, num_intervals
        )

        # Calculate averages
        averages = AutocorrelationAnalyzer.calculate_averages(scalar_products)

        return scalar_products, averages, individual_tracks


class AutocorrelationWorker(QThread):
    """Worker thread for autocorrelation analysis"""
    progress_update = Signal(str)
    analysis_complete = Signal(object, object, object)
    analysis_error = Signal(str)

    def __init__(self, tracks_df, time_interval, num_intervals):
        super().__init__()
        self.tracks_df = tracks_df
        self.time_interval = time_interval
        self.num_intervals = num_intervals

    def run(self):
        try:
            self.progress_update.emit("Starting autocorrelation analysis...")

            scalar_products, averages, individual_tracks = AutocorrelationAnalyzer.process_track_data(
                self.tracks_df, self.time_interval, self.num_intervals
            )

            if averages is not None:
                self.progress_update.emit("Autocorrelation analysis complete!")
                self.analysis_complete.emit(scalar_products, averages, individual_tracks)
            else:
                self.analysis_error.emit("No valid tracks found for autocorrelation analysis")

        except Exception as e:
            self.analysis_error.emit(f"Error in autocorrelation analysis: {str(e)}")


class AutocorrelationPlotWidget(QWidget):
    """Custom widget for displaying autocorrelation plots"""

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.averages = None
        self.individual_tracks = None

    def plot_autocorrelation(self, averages, individual_tracks=None, show_individual=False, max_tracks=50):
        """Plot autocorrelation results"""
        self.averages = averages
        self.individual_tracks = individual_tracks

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Plot individual tracks if requested and available
        if show_individual and individual_tracks is not None:
            # Pivot individual tracks data
            pivot_df = individual_tracks.pivot(index='track_id', columns='time_interval', values='correlation')
            pivot_df[0] = 1.0  # Add time=0 point
            pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

            # Sample tracks if too many
            tracks_to_plot = pivot_df.index
            if len(tracks_to_plot) > max_tracks:
                tracks_to_plot = random.sample(list(tracks_to_plot), max_tracks)

            # Plot individual tracks
            for track_id in tracks_to_plot:
                if track_id in pivot_df.index:
                    track_data = pivot_df.loc[track_id]
                    ax.plot(track_data.index, track_data.values, '-',
                           color='gray', alpha=0.1, linewidth=0.5)

        # Plot average with error bars
        x = np.array([float(col) for col in averages.columns])
        y = averages.loc['AVG'].values
        yerr = averages.loc['SEM'].values

        ax.errorbar(x, y, yerr=yerr, fmt='o-', color='blue', linewidth=2,
                   label='Average ± SEM', markersize=6, capsize=5)

        # Formatting
        ax.set_xlabel('Time Interval')
        ax.set_ylabel('Direction Autocorrelation')
        ax.set_title('Directional Autocorrelation Analysis')
        ax.set_ylim(-0.2, 1.0)
        ax.set_xlim(0, max(x) * 1.05)
        ax.grid(True, alpha=0.3)

        if show_individual and individual_tracks is not None:
            ax.legend(['Individual Tracks', 'Average ± SEM'])
        else:
            ax.legend()

        self.canvas.draw()

    def save_plot(self, filename):
        """Save the current plot"""
        if self.averages is not None:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')


class GeometricAnalyzer:
    """Geometric analysis methods from trajectory_analyzer.py"""

    @staticmethod
    def get_radius_of_gyration_simple(xy):
        """Calculate radius of gyration using simple geometric formula.

        Args:
            xy: Nx2 matrix of coordinates (can contain NaN values)

        Returns:
            Radius of gyration value
        """
        # Remove NaN values
        xy = xy[~np.isnan(xy).any(axis=1)]

        if len(xy) < 2:
            return np.nan

        # Calculate average position
        avg = np.nanmean(xy, axis=0)

        # Calculate average squared position
        avg2 = np.nanmean(xy**2, axis=0)

        # Calculate radius of gyration
        rg = np.sqrt(np.sum(avg2 - avg**2))

        return rg

    @staticmethod
    def get_mean_step_length(xy):
        """Calculate mean step length of trajectory.

        Args:
            xy: Nx2 matrix of coordinates (can contain NaN values)

        Returns:
            Mean step length
        """
        # Remove NaN values
        xy = xy[~np.isnan(xy).any(axis=1)]

        if len(xy) < 2:
            return np.nan

        # Calculate steps
        steps = np.diff(xy, axis=0)

        # Calculate step lengths
        step_lengths = np.sqrt(np.sum(steps**2, axis=1))

        # Return mean step length
        return np.mean(step_lengths)

    @staticmethod
    def get_scaled_rg(rg, mean_step_length):
        """Calculate scaled radius of gyration as in Golan and Sherman Nat Comm 2017.

        Args:
            rg: Radius of gyration
            mean_step_length: Mean step length

        Returns:
            Scaled radius of gyration value
        """
        if np.isnan(rg) or np.isnan(mean_step_length) or mean_step_length == 0:
            return np.nan

        s_rg = np.sqrt(np.pi/2) * rg / mean_step_length
        return s_rg

    @staticmethod
    def classify_linear_motion_simple(xy, directionality_threshold=0.8, perpendicular_threshold=0.15):
        """Classify trajectory as linear or non-linear using geometric methods.

        This method uses:
        1. Directionality ratio: net displacement / total path length
        2. Mean perpendicular distance: average distance of points from straight line

        Args:
            xy: Nx2 matrix of coordinates (can contain NaN values)
            directionality_threshold: Minimum directionality ratio for linear classification
            perpendicular_threshold: Maximum normalized perpendicular distance for linear classification

        Returns:
            Dictionary with classification and metrics
        """
        # Remove NaN values
        xy = xy[~np.isnan(xy).any(axis=1)]

        if len(xy) < 3:  # Need at least 3 points for meaningful analysis
            return {
                'classification': 'unclassified',
                'directionality_ratio': np.nan,
                'mean_perpendicular_distance': np.nan,
                'normalized_perpendicular_distance': np.nan
            }

        # Calculate directionality ratio
        start_point = xy[0]
        end_point = xy[-1]
        net_displacement = np.linalg.norm(end_point - start_point)

        # Calculate total path length
        steps = np.diff(xy, axis=0)
        step_lengths = np.linalg.norm(steps, axis=1)
        total_path_length = np.sum(step_lengths)

        if total_path_length == 0:
            return {
                'classification': 'unclassified',
                'directionality_ratio': np.nan,
                'mean_perpendicular_distance': np.nan,
                'normalized_perpendicular_distance': np.nan
            }

        directionality_ratio = net_displacement / total_path_length

        # Calculate perpendicular distances from straight line
        if net_displacement > 0:
            # Direction vector from start to end
            direction = (end_point - start_point) / net_displacement

            # Calculate perpendicular distance for each point
            perpendicular_distances = []
            for point in xy:
                # Vector from start to point
                vec_to_point = point - start_point

                # Project onto direction vector
                projection_length = np.dot(vec_to_point, direction)
                projection = projection_length * direction

                # Perpendicular component
                perpendicular = vec_to_point - projection
                perpendicular_dist = np.linalg.norm(perpendicular)
                perpendicular_distances.append(perpendicular_dist)

            mean_perpendicular_distance = np.mean(perpendicular_distances)
            # Normalize by net displacement for scale-invariant measure
            normalized_perpendicular_distance = mean_perpendicular_distance / net_displacement if net_displacement > 0 else np.inf
        else:
            # If start and end are the same point
            mean_perpendicular_distance = np.nan
            normalized_perpendicular_distance = np.nan

        # Classify based on metrics
        is_directional = directionality_ratio >= directionality_threshold
        is_straight = normalized_perpendicular_distance <= perpendicular_threshold if not np.isnan(normalized_perpendicular_distance) else False

        if is_directional and is_straight:
            classification = 'linear_unidirectional'
        elif not is_directional and is_straight:
            # Trajectory is straight but goes back and forth
            classification = 'linear_bidirectional'
        else:
            classification = 'non_linear'

        return {
            'classification': classification,
            'directionality_ratio': directionality_ratio,
            'mean_perpendicular_distance': mean_perpendicular_distance,
            'normalized_perpendicular_distance': normalized_perpendicular_distance
        }

class TrackpyLinker:
    """Trackpy-based particle linking with fallback for missing trackpy"""

    @staticmethod
    def check_trackpy_available():
        """Check if trackpy is available"""
        try:
            import trackpy as tp
            return True
        except ImportError:
            return False

    @staticmethod
    def link_particles_trackpy(locs_df, parameters, log_func=None, log_callback=None):
        """
        Link particles using trackpy

        Args:
            locs_df: DataFrame with columns ['frame', 'x', 'y', 'id'] (coordinates in pixels)
            parameters: SPTAnalysisParameters instance
            log_callback: Function to call for logging messages

        Returns:
            DataFrame with linked tracks or None if linking fails
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)

        try:
            import trackpy as tp

            # Suppress trackpy verbose output
            tp.quiet()

            log("    Using trackpy for particle linking...")
            log(f"    Linking method: {parameters.trackpy_linking_type}")
            log(f"    Search distance: {parameters.trackpy_link_distance} pixels")
            log(f"    Memory: {parameters.trackpy_memory} frames")

            # Prepare data for trackpy (needs specific column names)
            trackpy_df = locs_df.copy()
            trackpy_df = trackpy_df.rename(columns={'frame': 'frame', 'x': 'x', 'y': 'y'})

            # Perform linking based on selected method
            if parameters.trackpy_linking_type == 'standard':
                linked_df = tp.link(trackpy_df,
                                  parameters.trackpy_link_distance,
                                  memory=parameters.trackpy_memory)

            elif parameters.trackpy_linking_type == 'adaptive':
                linked_df = tp.link(trackpy_df,
                                  parameters.trackpy_max_search_distance,
                                  adaptive_stop=parameters.trackpy_adaptive_stop,
                                  adaptive_step=parameters.trackpy_adaptive_step,
                                  memory=parameters.trackpy_memory)

            elif parameters.trackpy_linking_type == 'velocityPredict':
                pred = tp.predict.NearestVelocityPredict()
                linked_df = pred.link_df(trackpy_df,
                                       parameters.trackpy_link_distance,
                                       memory=parameters.trackpy_memory)

            elif parameters.trackpy_linking_type == 'adaptive + velocityPredict':
                pred = tp.predict.NearestVelocityPredict()
                linked_df = pred.link_df(trackpy_df,
                                       parameters.trackpy_max_search_distance,
                                       memory=parameters.trackpy_memory,
                                       adaptive_stop=parameters.trackpy_adaptive_stop,
                                       adaptive_step=parameters.trackpy_adaptive_step)
            else:
                raise ValueError(f"Unknown trackpy linking type: {parameters.trackpy_linking_type}")

            # Rename trackpy's 'particle' column to 'track_number'
            linked_df = linked_df.rename(columns={'particle': 'track_number'})

            # Sort by track and frame
            linked_df = linked_df.sort_values(['track_number', 'frame'])


            # Filter out single-point tracks if min_segments > 1
            if parameters.min_track_segments > 1:
                track_counts = linked_df.groupby('track_number').size()
                valid_tracks = track_counts[track_counts >= parameters.min_track_segments].index
                linked_df = linked_df[linked_df['track_number'].isin(valid_tracks)]

                if len(linked_df) == 0:
                    if log_func:
                        log_func("    No tracks meet minimum segment requirement")
                    return None

                # CRITICAL FIX: Reset index after filtering
                linked_df = linked_df.reset_index(drop=True)

                # Renumber tracks to be sequential starting from 0
                unique_tracks = sorted(linked_df['track_number'].unique())
                track_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracks)}
                linked_df['track_number'] = linked_df['track_number'].map(track_mapping)

            # Sort by track and frame and reset index again
            linked_df = linked_df.sort_values(['track_number', 'frame']).reset_index(drop=True)

            log(f"    Trackpy linking complete: {len(linked_df['track_number'].unique())} tracks, {len(linked_df)} points")
            return linked_df

        except ImportError:
            log("    ERROR: trackpy not available. Install with: pip install trackpy")
            return None
        except Exception as e:
            log(f"    ERROR in trackpy linking: {e}")
            return None

class SPTAnalysisParameters:
    """Enhanced parameter management with dual linking methods"""

    def __init__(self):
        # File processing parameters
        self.pixel_size = 108  # nm per pixel
        self.frame_length = 1  # seconds per frame
        self.min_track_segments = 4

        # Linking parameters - Built-in method with dual options
        self.linking_method = 'builtin'
        self.builtin_linking_algorithm = 'iterative'  # NEW: 'iterative' or 'recursive'
        self.recursive_depth_limit = 1000  # NEW: configurable recursion limit
        self.max_gap_frames = 36
        self.max_link_distance = 3  # pixels

        # Trackpy linking parameters
        self.trackpy_link_distance = 3.0
        self.trackpy_memory = 4
        self.trackpy_linking_type = 'standard'
        self.trackpy_adaptive_stop = 0.1
        self.trackpy_adaptive_step = 0.95
        self.trackpy_max_search_distance = 6.0

        # Feature calculation parameters
        self.nn_radii = [3, 5, 10, 20, 30]
        self.rg_mobility_threshold = 2.11

        # Classification parameters
        self.training_data_path = get_default_training_data_path()
        self.experiment_name = ""
        self.auto_detect_experiment_names = False

        # Analysis step flags (all existing flags preserved)
        self.enable_nearest_neighbors = True
        self.enable_svm_classification = False
        self.enable_velocity_analysis = True
        self.enable_diffusion_analysis = True
        self.enable_background_subtraction = False
        self.enable_localization_error = True
        self.enable_straightness_analysis = True

        # Enhanced analysis options (all existing options preserved)
        self.enable_direction_analysis = True
        self.enable_missing_points_integration = True
        self.enable_enhanced_interpolation = False
        self.enable_distance_differential = True
        self.enable_full_track_interpolation = False
        self.save_separate_interpolated_file = False
        self.extend_interpolation_to_full_recording = False

        # Geometric analysis parameters (all existing preserved)
        self.enable_geometric_analysis = True
        self.geometric_rg_method = 'simple'
        self.geometric_srg_cutoff = 2.22236433588659
        self.enable_geometric_linear_classification = True
        self.geometric_directionality_threshold = 0.8
        self.geometric_perpendicular_threshold = 0.15
        self.geometric_cutoff_length = 3

        # Autocorrelation analysis parameters (all existing preserved)
        self.enable_autocorrelation_analysis = True
        self.autocorr_time_interval = 1.0
        self.autocorr_num_intervals = 25
        self.autocorr_min_track_length = 5
        self.autocorr_show_individual_tracks = True
        self.autocorr_max_tracks_plot = 100
        self.autocorr_save_plots = True
        self.autocorr_save_data = True

        # ROI Background Subtraction parameters (existing preserved)
        self.roi_frame_specific_background = False

        # Output options (existing preserved)
        self.save_intermediate = True

        # Detection parameters (all existing preserved)
        self.enable_detection = False
        self.detection_psf_sigma = 1.5
        self.detection_alpha_threshold = 0.05
        self.detection_min_intensity = 100.0
        self.detection_output_directory = ""
        self.detection_skip_existing = True
        self.detection_show_results = False

        # Enhanced U-Track linking parameters (all existing preserved)
        self.utrack_max_linking_distance = 10.0
        self.utrack_max_gap_frames = 5
        self.utrack_motion_model = 'mixed'
        self.utrack_auto_linking_distance = False
        self.utrack_enable_merging = False
        self.utrack_enable_splitting = False
        self.utrack_min_search_radius = 2.0
        self.utrack_brown_std_mult = 3.0
        self.utrack_lin_std_mult = 1.0

        # Mixed motion model parameters (all existing preserved)
        self.utrack_enable_iterative_smoothing = True
        self.utrack_num_tracking_rounds = 3
        self.utrack_regime_sensitivity = 0.8
        self.utrack_adaptive_search_radius = True
        self.utrack_min_regime_length = 3

        # Motion transition probabilities (all existing preserved)
        self.utrack_trans_prob_b2l = 0.1
        self.utrack_trans_prob_l2b = 0.1
        self.utrack_brownian_noise_mult = 3.0
        self.utrack_linear_velocity_persist = 0.8

    def to_dict(self):
        """Convert parameters to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, param_dict):
        """Load parameters from dictionary"""
        for key, value in param_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Points:
    """Enhanced Points class with both iterative and recursive linking options"""

    def __init__(self, txy_pts):
        self.frames = np.unique(txy_pts[:, 0]).astype(int)
        self.txy_pts = txy_pts
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []
        self.intensities = []
        self.tracks = []
        self.recursiveFailure = False
        self.linking_stats = {}
        self.linking_method = 'iterative'  # Default to iterative
        self.recursive_depth_limit = 1000

    def link_pts(self, maxFramesSkipped, maxDistance, method='iterative', depth_limit=1000):
        """Link points across frames with choice of method"""
        try:
            self.linking_method = method
            self.recursive_depth_limit = depth_limit

            start_time = time.time()
            print(f'Linking points using {method} method...')

            self._prepare_frame_data()

            tracks = []
            total_points = len(self.txy_pts)
            processed_points = 0

            for frame in self.frames:
                frame_points = np.where(self.pts_remaining[frame])[0]

                for pt_idx in frame_points:
                    if not self.pts_remaining[frame][pt_idx]:
                        continue

                    self.pts_remaining[frame][pt_idx] = False
                    abs_pt_idx = self.pts_idx_by_frame[frame][pt_idx]

                    # Choose linking method
                    if method == 'iterative':
                        track = self._extend_track_iterative([abs_pt_idx], maxFramesSkipped, maxDistance)
                    else:  # recursive
                        track = self._extend_track_recursive([abs_pt_idx], maxFramesSkipped, maxDistance, depth=0)

                    tracks.append(track)
                    processed_points += len(track)

            self.tracks = tracks
            elapsed_time = time.time() - start_time

            # Store comprehensive linking statistics
            track_lengths = [len(track) for track in tracks]
            self.linking_stats = {
                'method_used': method,
                'recursive_depth_limit': depth_limit if method == 'recursive' else None,
                'total_input_points': total_points,
                'total_linked_points': processed_points,
                'num_tracks': len(tracks),
                'track_lengths': track_lengths,
                'mean_track_length': np.mean(track_lengths) if track_lengths else 0,
                'max_track_length': max(track_lengths) if track_lengths else 0,
                'min_track_length': min(track_lengths) if track_lengths else 0,
                'linking_efficiency': processed_points / total_points if total_points > 0 else 0,
                'processing_time': elapsed_time,
                'linking_success': True,
                'recursive_failure': self.recursiveFailure
            }

            print(f'Linking complete. Created {len(tracks)} tracks from {total_points} points.')
            print(f'Method: {method}, Time: {elapsed_time:.2f}s, Efficiency: {self.linking_stats["linking_efficiency"]:.1%}')
            if method == 'recursive':
                print(f'Recursive depth limit: {depth_limit}, Recursive failure: {self.recursiveFailure}')

        except Exception as e:
            self.recursiveFailure = True
            self.linking_stats = {
                'method_used': method,
                'linking_success': False,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'recursive_failure': True
            }
            print(f"Error in linking: {e}")
            traceback.print_exc()

    def _prepare_frame_data(self):
        """Prepare frame-by-frame data structures with error handling"""
        try:
            max_frame = int(np.max(self.frames))
            self.pts_by_frame = []
            self.pts_remaining = []
            self.pts_idx_by_frame = []

            for frame in np.arange(0, max_frame + 1):
                indices = np.where(self.txy_pts[:, 0] == frame)[0]
                pos = self.txy_pts[indices, 1:]
                self.pts_by_frame.append(pos)
                self.pts_remaining.append(np.ones(pos.shape[0], dtype=bool))
                self.pts_idx_by_frame.append(indices)

        except Exception as e:
            print(f"Error preparing frame data: {e}")
            raise

    def _extend_track_iterative(self, initial_track, maxFramesSkipped, maxDistance):
        """
        Iterative track extension - prevents recursion errors
        This is the recommended method for large datasets
        """
        track = initial_track.copy()

        # Use a queue to process track extensions iteratively
        extension_queue = deque([track[-1]])
        processed_points = set()  # Prevent infinite loops

        while extension_queue:
            current_pt_idx = extension_queue.popleft()

            # Prevent processing the same point multiple times
            if current_pt_idx in processed_points:
                continue
            processed_points.add(current_pt_idx)

            pt = self.txy_pts[current_pt_idx]
            current_frame = int(pt[0])

            # Look for next point in subsequent frames
            for dt in np.arange(1, maxFramesSkipped + 2):
                next_frame = current_frame + dt

                if next_frame >= len(self.pts_remaining):
                    break

                candidates = self.pts_remaining[next_frame]
                nCandidates = np.count_nonzero(candidates)

                if nCandidates == 0:
                    continue

                # Calculate distances to all candidates
                candidate_indices = np.where(candidates)[0]
                candidate_positions = self.pts_by_frame[next_frame][candidates]

                distances = np.sqrt(np.sum(
                    (candidate_positions - pt[1:]) ** 2, axis=1))

                # Find best candidate within distance threshold
                valid_distances = distances < maxDistance
                if np.any(valid_distances):
                    best_candidate_idx = np.argmin(distances)
                    best_idx = candidate_indices[best_candidate_idx]
                    abs_next_pt_idx = self.pts_idx_by_frame[next_frame][best_idx]

                    # Add to track and mark as used
                    track.append(abs_next_pt_idx)
                    self.pts_remaining[next_frame][best_idx] = False

                    # Add to queue for further extension
                    extension_queue.append(abs_next_pt_idx)
                    break  # Found connection, move to next iteration

        return track

    def _extend_track_recursive(self, track, maxFramesSkipped, maxDistance, depth=0):
        """
        Original recursive track extension with configurable depth limit
        Use with caution for large datasets
        """
        if depth >= self.recursive_depth_limit:
            self.recursiveFailure = True
            print(f"Warning: Recursive depth limit ({self.recursive_depth_limit}) reached")
            return track

        pt = self.txy_pts[track[-1]]

        for dt in np.arange(1, maxFramesSkipped + 2):
            frame = int(pt[0]) + dt
            if frame >= len(self.pts_remaining):
                return track

            candidates = self.pts_remaining[frame]
            nCandidates = np.count_nonzero(candidates)

            if nCandidates == 0:
                continue

            distances = np.sqrt(np.sum(
                (self.pts_by_frame[frame][candidates] - pt[1:]) ** 2, 1))

            if any(distances < maxDistance):
                next_pt_idx = np.where(candidates)[0][np.argmin(distances)]
                abs_next_pt_idx = self.pts_idx_by_frame[frame][next_pt_idx]
                track.append(abs_next_pt_idx)
                self.pts_remaining[frame][next_pt_idx] = False
                track = self._extend_track_recursive(track, maxFramesSkipped, maxDistance, depth + 1)
                return track

        return track

    def getIntensities(self, dataArray):
        """Extract intensities from image data with enhanced error handling"""
        print("Extracting intensities...")

        try:
            if dataArray.ndim != 3:
                raise ValueError(f"Expected 3D array, got {dataArray.ndim}D array")

            n, h, w = dataArray.shape
            self.intensities = []

            for i, point in enumerate(tqdm(self.txy_pts, desc="Processing intensities")):
                try:
                    frame = int(round(point[0]))
                    x = int(round(point[1]))
                    y = int(round(point[2]))

                    # Enhanced bounds checking
                    frame = max(0, min(frame, n - 1))

                    # 3x3 pixel region with bounds checking
                    xMin = max(0, x - 1)
                    xMax = min(w, x + 2)
                    yMin = max(0, y - 1)
                    yMax = min(h, y + 2)

                    # Calculate intensity with additional validation
                    if xMax > xMin and yMax > yMin and frame < n:
                        intensity = np.mean(dataArray[frame][yMin:yMax, xMin:xMax])
                        # Check for NaN or infinite values
                        if not np.isfinite(intensity):
                            intensity = 0.0
                    else:
                        intensity = 0.0

                    self.intensities.append(intensity)

                except Exception as e:
                    print(f"Error processing intensity for point {i}: {e}")
                    self.intensities.append(0.0)

            print(f"Intensity extraction complete: {len(self.intensities)} values extracted")

        except Exception as e:
            print(f"Error in getIntensities: {e}")
            self.intensities = [0.0] * len(self.txy_pts)
            raise


class FeatureCalculator:
    """Enhanced feature calculation methods"""

    @staticmethod
    def radius_gyration_asymmetry(trackDF):
        """Calculate radius of gyration and asymmetry features (tensor method)"""
        points_array = np.array(trackDF[['x', 'y']].dropna())
        center = points_array.mean(0)
        normed_points = points_array - center[None, :]

        rg_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(points_array)
        eig_values, eig_vectors = np.linalg.eig(rg_tensor)

        radius_gyration = np.sqrt(np.sum(eig_values))

        asymmetry_num = (eig_values[0] - eig_values[1]) ** 2
        asymmetry_den = 2 * (eig_values[0] + eig_values[1]) ** 2
        asymmetry = -math.log(1 - (asymmetry_num / asymmetry_den)) if asymmetry_den > 0 else 0

        # Projection for skewness/kurtosis
        maxcol = list(eig_values).index(max(eig_values))
        dom_eig_vect = eig_vectors[:, maxcol]

        points_a = points_array[:-1]
        points_b = points_array[1:]
        ba = points_b - points_a
        proj = np.dot(ba, dom_eig_vect) / np.power(np.linalg.norm(dom_eig_vect), 2)

        skewness = stats.skew(proj) if len(proj) > 0 else 0
        kurtosis = stats.kurtosis(proj) if len(proj) > 0 else 0

        return radius_gyration, asymmetry, skewness, kurtosis

    @staticmethod
    def fractal_dimension(points_array):
        """Calculate fractal dimension"""
        if len(points_array) < 3:
            return 1.0

        try:
            # Check for collinear points
            x0, y0 = points_array[0]
            points = [(x, y) for x, y in points_array if (x != x0) or (y != y0)]
            if len(points) < 2:
                return 1.0

            slopes = [((y - y0) / (x - x0)) if (x != x0) else None for x, y in points]
            if all(s == slopes[0] for s in slopes if s is not None):
                return 1.0  # Collinear points

            total_path_length = np.sum(np.sqrt(np.sum((points_array[1:, :] - points_array[:-1, :]) ** 2, axis=1)))
            stepCount = len(points_array)

            if len(points_array) < 3:
                return 1.0

            candidates = points_array[spatial.ConvexHull(points_array).vertices]
            dist_mat = spatial.distance_matrix(candidates, candidates)
            largestDistance = np.max(dist_mat)

            if total_path_length > 0 and largestDistance > 0:
                fractal_dim = math.log(stepCount) / math.log(stepCount * largestDistance / total_path_length)
                return fractal_dim
            else:
                return 1.0
        except:
            return 1.0

    @staticmethod
    def net_displacement_efficiency(points_array):
        """Calculate net displacement and efficiency"""
        if len(points_array) < 2:
            return 0, 0

        net_displacement = np.linalg.norm(points_array[0] - points_array[-1])

        if len(points_array) < 3:
            return net_displacement, 1.0

        points_a = points_array[1:, :]
        points_b = points_array[:-1, :]
        dist_ab_SumSquared = sum((np.linalg.norm(points_a - points_b, axis=1)) ** 2)

        efficiency = (net_displacement ** 2) / ((len(points_array) - 1) * dist_ab_SumSquared) if dist_ab_SumSquared > 0 else 0
        return net_displacement, efficiency

    @staticmethod
    def summed_sines_cosines(points_array):
        """Calculate straightness features"""
        if len(points_array) < 3:
            return 0, [], 0, []

        # Remove consecutive duplicates
        compare_against = points_array[:-1]
        duplicates_table = points_array[1:] == compare_against
        duplicates_table = duplicates_table.sum(axis=1)
        duplicate_indices = np.where(duplicates_table == 2)
        points_array = np.delete(points_array, duplicate_indices, axis=0)

        if len(points_array) < 3:
            return 0, [], 0, []

        # Generate vector sets
        points_a = points_array[:-2]
        points_b = points_array[1:-1]
        points_c = points_array[2:]

        ab = points_b - points_a
        bc = points_c - points_b

        cross_products = np.cross(ab, bc)
        dot_products = np.einsum('ij,ij->i', ab, bc)
        magnitudes = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)

        # Avoid division by zero
        valid_magnitudes = magnitudes > 0
        cos_vals = np.zeros_like(magnitudes)
        sin_vals = np.zeros_like(magnitudes)

        cos_vals[valid_magnitudes] = dot_products[valid_magnitudes] / magnitudes[valid_magnitudes]
        sin_vals[valid_magnitudes] = cross_products[valid_magnitudes] / magnitudes[valid_magnitudes]

        return np.mean(sin_vals), sin_vals, np.mean(cos_vals), cos_vals


class NearestNeighborAnalyzer:
    """Nearest neighbor analysis functions"""

    @staticmethod
    def get_nearest_neighbors(train, test, k=2):
        """Get k nearest neighbors"""
        tree = KDTree(train, leaf_size=5)
        if k > len(train):
            return np.nan, np.nan
        return tree.query(test, k=k)

    @staticmethod
    def count_neighbors_in_radius(train, test, r=1):
        """Count neighbors within radius"""
        tree = KDTree(train, leaf_size=5)
        return tree.query_radius(test, r=r, count_only=True)

    @staticmethod
    def analyze_frame_neighbors(tracksDF, radii_list=[3, 5, 10, 20, 30]):
        """Analyze neighbors by frame"""
        print("Performing nearest neighbor analysis...")
        tracksDF = tracksDF.sort_values(by=['frame'])

        # First get nearest neighbor distance
        nn_dist_list = []
        frames = tracksDF['frame'].unique()

        for frame in tqdm(frames, desc="NN distance analysis"):
            frame_xy = tracksDF[tracksDF['frame'] == frame][['x', 'y']].to_numpy()

            if len(frame_xy) > 1:
                distances, _ = NearestNeighborAnalyzer.get_nearest_neighbors(frame_xy, frame_xy, k=2)
                if not np.isnan(distances).any():
                    nn_dist_list.extend(distances[:, 1])  # Second column is nearest neighbor
                else:
                    nn_dist_list.extend([np.nan] * len(frame_xy))
            else:
                nn_dist_list.append(np.nan)

        tracksDF['nnDist'] = nn_dist_list

        # Then count neighbors within radii
        for r in radii_list:
            count_list = []

            for frame in frames:
                frame_xy = tracksDF[tracksDF['frame'] == frame][['x', 'y']].to_numpy()

                if len(frame_xy) > 0:
                    counts = NearestNeighborAnalyzer.count_neighbors_in_radius(
                        frame_xy, frame_xy, r=r)
                    counts = counts - 1  # Exclude self
                    count_list.extend(counts)
                else:
                    count_list.append(0)

            tracksDF[f'nnCountInFrame_within_{r}_pixels'] = count_list

        return tracksDF.sort_values(['track_number', 'frame'])


class DirectionAnalyzer:
    """Direction of travel analysis functions"""

    @staticmethod
    def calculate_direction(x1, y1, x2, y2):
        """Calculate direction between two points with inverted Y-axis for microscopy"""
        dx = x2 - x1
        dy = -(y2 - y1)  # Invert Y for microscopy display

        if dx == 0 and dy == 0:
            return np.nan

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    @staticmethod
    def calculate_directional_persistence(angles):
        """Calculate directional persistence (0=random, 1=straight)"""
        if len(angles) < 2 or np.isnan(angles).all():
            return np.nan

        valid_angles = angles[~np.isnan(angles)]
        if len(valid_angles) < 2:
            return np.nan

        angle_diffs = []
        for i in range(1, len(valid_angles)):
            diff = abs(valid_angles[i] - valid_angles[i-1])
            if diff > 180:
                diff = 360 - diff
            angle_diffs.append(diff)

        avg_diff = np.mean(angle_diffs)
        persistence = 1 - (avg_diff / 180)
        return persistence

    @staticmethod
    def calculate_directional_autocorrelation(x_coords, y_coords):
        """Calculate directional autocorrelation"""
        dx = np.diff(x_coords)
        dy = -np.diff(y_coords)  # Invert Y

        angles = np.arctan2(dy, dx)
        N = len(angles)

        if N < 2:
            return [np.nan] * len(x_coords)

        max_interval = N
        autocorr_values = []

        for i in range(N):
            if i < max_interval:
                sum_cos = 0
                count = 0

                for j in range(N - i):
                    angle_diff = angles[j+i] - angles[j]
                    sum_cos += np.cos(angle_diff)
                    count += 1

                if count > 0:
                    autocorr_values.append(sum_cos / count)
                else:
                    autocorr_values.append(np.nan)
            else:
                autocorr_values.append(np.nan)

        # Pad to match original length
        while len(autocorr_values) < len(x_coords):
            autocorr_values.append(np.nan)

        return autocorr_values

    @staticmethod
    def add_direction_analysis(tracks_df):
        """Add direction analysis to tracks dataframe"""
        # Initialize new columns
        tracks_df['direction_degrees'] = np.nan
        tracks_df['direction_radians'] = np.nan
        tracks_df['direction_x'] = np.nan
        tracks_df['direction_y'] = np.nan
        tracks_df['directional_autocorrelation'] = np.nan
        tracks_df['directional_persistence'] = np.nan

        track_numbers = tracks_df['track_number'].unique()

        for track_num in track_numbers:
            track_data = tracks_df[tracks_df['track_number'] == track_num].sort_values('frame')

            if len(track_data) < 2:
                continue

            x_vals = track_data['x'].values
            y_vals = track_data['y'].values
            track_indices = track_data.index.values

            # Calculate directions
            angles = []
            for i in range(len(x_vals) - 1):
                angle = DirectionAnalyzer.calculate_direction(
                    x_vals[i], y_vals[i], x_vals[i+1], y_vals[i+1])
                angles.append(angle)

                idx = track_indices[i]
                tracks_df.at[idx, 'direction_degrees'] = angle
                if not np.isnan(angle):
                    tracks_df.at[idx, 'direction_radians'] = math.radians(angle)
                    tracks_df.at[idx, 'direction_x'] = math.cos(math.radians(angle))
                    tracks_df.at[idx, 'direction_y'] = -math.sin(math.radians(angle))

            # Calculate persistence
            persistence = DirectionAnalyzer.calculate_directional_persistence(np.array(angles))
            tracks_df.loc[tracks_df['track_number'] == track_num, 'directional_persistence'] = persistence

            # Calculate autocorrelation
            if len(track_data) >= 3:
                autocorr_values = DirectionAnalyzer.calculate_directional_autocorrelation(x_vals, y_vals)
                for i, idx in enumerate(track_indices):
                    if i < len(autocorr_values):
                        tracks_df.at[idx, 'directional_autocorrelation'] = autocorr_values[i]

        return tracks_df


class MissingPointsIntegrator:
    """Missing points integration functions - FIXED to calculate intensities"""

    @staticmethod
    def add_missing_localizations(tracks_df, file_path, pixel_size=108):
        """Add back unlinked localizations to the dataset - FIXED to calculate intensities"""
        try:
            # Load original localizations
            locs_file = file_path.replace('.tif', '_locsID.csv')
            if not os.path.exists(locs_file):
                locs_file = file_path.replace('.tif', '_locs.csv')

            if not os.path.exists(locs_file):
                print("    No localization file found for missing points integration")
                return tracks_df

            locs_df = pd.read_csv(locs_file)

            # Check if tracks_df has 'id' column
            if 'id' not in tracks_df.columns:
                print("    Warning: tracks_df missing 'id' column, creating sequential IDs for comparison")
                return tracks_df

            # Check if locs_df has 'id' column
            if 'id' not in locs_df.columns:
                print("    Warning: localization file missing 'id' column, skipping missing points integration")
                return tracks_df

            # Get IDs already in tracks
            linked_ids = set(tracks_df['id'].dropna())
            print(f"    Found {len(linked_ids)} linked localizations")

            # Filter for missing localizations
            missing_locs = locs_df[~locs_df['id'].isin(linked_ids)].copy()
            print(f"    Found {len(missing_locs)} unlinked localizations")

            if len(missing_locs) == 0:
                print("    No missing localizations to add")
                return tracks_df

            # Convert coordinates
            missing_locs['frame'] = missing_locs['frame'].astype(int) - 1
            missing_locs['x'] = missing_locs['x [nm]'] / pixel_size
            missing_locs['y'] = missing_locs['y [nm]'] / pixel_size

            # Calculate intensities from image data
            print(f"    Calculating intensities for {len(missing_locs)} unlinked points...")
            intensities = MissingPointsIntegrator._calculate_intensities_for_missing_points(
                file_path, missing_locs[['frame', 'x', 'y']].values)

            # Use calculated intensities or try to get from original file
            if intensities is not None and len(intensities) == len(missing_locs):
                missing_locs['intensity'] = intensities
                print(f"    Successfully calculated intensities from image data")
            elif 'intensity [photon]' in missing_locs.columns:
                missing_locs['intensity'] = missing_locs['intensity [photon]']
                print(f"    Using intensity values from localization file")
            elif 'intensity' in locs_df.columns:
                # If locs_df already has intensity column, use it
                missing_locs['intensity'] = missing_locs['intensity']
                print(f"    Using existing intensity values from localization file")
            else:
                # Fallback: set to NaN
                missing_locs['intensity'] = np.nan
                print(f"    Warning: Could not calculate or find intensity values, setting to NaN")

            # Add required columns with appropriate values
            for col in tracks_df.columns:
                if col not in missing_locs.columns:
                    if col == 'track_number':
                        missing_locs[col] = np.nan  # Unlinked points have no track
                    elif col == 'n_segments':
                        missing_locs[col] = 1  # Single points
                    elif col in ['lag', 'velocity', 'meanLag', 'meanVelocity', 'track_length']:
                        missing_locs[col] = np.nan  # No movement data for single points
                    elif col == 'SVM':
                        missing_locs[col] = 0  # Unclassified
                    elif col in ['zeroed_X', 'zeroed_Y', 'distanceFromOrigin']:
                        missing_locs[col] = 0  # No movement reference
                    elif col == 'mean_X':
                        missing_locs[col] = missing_locs['x']  # Point is its own mean X
                    elif col == 'mean_Y':
                        missing_locs[col] = missing_locs['y']  # Point is its own mean Y
                    elif col == 'distanceFromMean':
                        missing_locs[col] = 0  # Point is at its own mean
                    elif col == 'lagNumber':
                        missing_locs[col] = 0  # Single time point
                    else:
                        missing_locs[col] = np.nan

            # Select only columns that exist in tracks_df
            missing_locs = missing_locs[tracks_df.columns]

            # Combine with tracks
            combined_df = pd.concat([tracks_df, missing_locs], ignore_index=True)
            print(f"    Added {len(missing_locs)} missing localizations with calculated intensities to dataset")

            return combined_df

        except Exception as e:
            print(f"    Error adding missing points: {e}")
            import traceback
            traceback.print_exc()
            return tracks_df

    @staticmethod
    def _calculate_intensities_for_missing_points(file_path, points_array):
        """Calculate intensities for missing points from image data"""
        try:
            # Load and transform image data (same as other intensity calculations)
            import skimage.io as skio
            if not os.path.exists(file_path):
                print(f"    Warning: Image file {file_path} not found for intensity calculation")
                return None

            A = skio.imread(file_path, plugin='tifffile')
            # Apply same transformations as in other intensity calculations
            #A = np.rot90(A, axes=(1,2))
            #A = np.fliplr(A)

            n, h, w = A.shape
            intensities = []

            for point in points_array:
                frame = int(round(point[0]))
                x = int(round(point[1]))
                y = int(round(point[2]))

                # 3x3 pixel region - same method as Points.getIntensities
                xMin = x - 1
                xMax = x + 2
                yMin = y - 1
                yMax = y + 2

                # Edge case handling
                if xMin < 0:
                    xMin = 0
                if xMax > w:
                    xMax = w
                if yMin < 0:
                    yMin = 0
                if yMax > h:
                    yMax = h

                # Ensure frame is within bounds
                if frame >= n:
                    frame = n - 1
                if frame < 0:
                    frame = 0

                # Calculate intensity using same method as linked points
                intensity = np.mean(A[frame][yMin:yMax, xMin:xMax])
                intensities.append(intensity)

            return intensities

        except Exception as e:
            print(f"    Error calculating intensities for missing points: {e}")
            import traceback
            traceback.print_exc()
            return None


class EnhancedInterpolator:
    """Enhanced interpolation for trapped sites - CORRECTED VERSION"""

    @staticmethod
    def interpolate_trapped_sites(tracks_df, file_path, svm_class=3):
        """Add interpolated points for trapped sites (SVM class 3) - UPDATED to mark interpolated points"""
        try:
            # Initialize is_interpolated column if it doesn't exist
            if 'is_interpolated' not in tracks_df.columns:
                tracks_df['is_interpolated'] = 0  # 0 = actual, 1 = interpolated

            # Load image to get total frame count and for intensity calculation
            import skimage.io as skio
            if not os.path.exists(file_path):
                return tracks_df

            A = skio.imread(file_path, plugin='tifffile')
            # Apply same transformations as in intensity calculation
            #A = np.rot90(A, axes=(1,2))
            #A = np.fliplr(A)
            n_frames, h, w = A.shape

            # Filter for specified SVM class and other tracks
            mobile_tracks = tracks_df[tracks_df['SVM'] != svm_class]
            trapped_tracks = tracks_df[tracks_df['SVM'] == svm_class]

            if len(trapped_tracks) == 0:
                return tracks_df

            interpolated_tracks = []

            for track_num in trapped_tracks['track_number'].unique():
                track_data = trapped_tracks[trapped_tracks['track_number'] == track_num].sort_values('frame')

                if len(track_data) < 1:
                    continue

                # Get track-level properties that should be consistent across all points
                track_level_props = {
                    'track_number': track_num,
                    'SVM': svm_class,
                    'Experiment': track_data['Experiment'].iloc[0] if 'Experiment' in track_data.columns else '',
                }

                # Add other track-level features if they exist
                track_level_features = [
                    'radius_gyration', 'asymmetry', 'skewness', 'kurtosis',
                    'fracDimension', 'netDispl', 'efficiency', 'Straight',
                    'track_intensity_mean', 'track_intensity_std',
                    'meanLag', 'track_length', 'radius_gyration_scaled',
                    'radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength',
                    'radius_gyration_ratio_to_mean_step_size', 'radius_gyration_mobility_threshold'
                ]

                for feature in track_level_features:
                    if feature in track_data.columns:
                        track_level_props[feature] = track_data[feature].iloc[0]

                # Extract actual detected positions
                actual_frames = track_data['frame'].to_numpy()
                actual_x = track_data['x'].to_numpy()
                actual_y = track_data['y'].to_numpy()

                # Get track frame range
                min_track_frame = int(actual_frames.min())
                max_track_frame = int(actual_frames.max())

                # Step 1: Interpolate within the track frame range
                track_frame_range = np.arange(min_track_frame, max_track_frame + 1)

                # Only interpolate if there are missing frames within the range
                if len(track_frame_range) > len(actual_frames):
                    x_interp = np.interp(track_frame_range, actual_frames, actual_x)
                    y_interp = np.interp(track_frame_range, actual_frames, actual_y)
                else:
                    # No missing frames within range, use actual positions
                    x_interp = actual_x
                    y_interp = actual_y

                # Step 2: Extend to full recording length using mean position for padding
                all_frames = np.arange(0, n_frames)
                mean_x = actual_x.mean()
                mean_y = actual_y.mean()

                # Create full arrays with mean position
                full_x = np.full(n_frames, mean_x)
                full_y = np.full(n_frames, mean_y)

                # Replace the track range with interpolated values
                full_x[min_track_frame:max_track_frame+1] = x_interp
                full_y[min_track_frame:max_track_frame+1] = y_interp

                # Create interpolated points for all frames
                interpolated_points = []

                for frame_idx in range(n_frames):
                    # Determine if this point is interpolated
                    is_actual_detection = frame_idx in actual_frames
                    is_interpolated_point = not is_actual_detection

                    # Calculate frame-specific properties
                    point_data = {
                        'frame': frame_idx,
                        'x': full_x[frame_idx],
                        'y': full_y[frame_idx],
                        'is_interpolated': 1 if is_interpolated_point else 0,  # NEW: Mark interpolated points
                    }

                    # Calculate intensity for this specific frame and position
                    intensity = EnhancedInterpolator._calculate_intensity_for_position(
                        A, frame_idx, full_x[frame_idx], full_y[frame_idx])
                    point_data['intensity'] = intensity

                    # Add frame-specific features
                    point_data['lagNumber'] = frame_idx - min_track_frame

                    # Distance from origin (first detected frame position)
                    origin_x = actual_x[0]
                    origin_y = actual_y[0]
                    point_data['zeroed_X'] = full_x[frame_idx] - origin_x
                    point_data['zeroed_Y'] = full_y[frame_idx] - origin_y
                    point_data['distanceFromOrigin'] = np.sqrt(
                        (full_x[frame_idx] - origin_x)**2 + (full_y[frame_idx] - origin_y)**2)

                    # Calculate lag displacement (distance between consecutive positions)
                    if frame_idx > 0:
                        lag_distance = np.sqrt(
                            (full_x[frame_idx] - full_x[frame_idx-1])**2 +
                            (full_y[frame_idx] - full_y[frame_idx-1])**2)
                        point_data['lag'] = lag_distance
                        point_data['velocity'] = lag_distance  # assuming dt=1
                    else:
                        point_data['lag'] = np.nan
                        point_data['velocity'] = np.nan

                    # Distance from track mean position
                    point_data['mean_X'] = mean_x
                    point_data['mean_Y'] = mean_y
                    point_data['distanceFromMean'] = np.sqrt(
                        (full_x[frame_idx] - mean_x)**2 + (full_y[frame_idx] - mean_y)**2)

                    # Direction features
                    point_data['direction_Relative_To_Origin'] = np.degrees(
                        np.arctan2(point_data['zeroed_Y'], point_data['zeroed_X'])) % 360

                    # Add track-level properties
                    point_data.update(track_level_props)

                    interpolated_points.append(point_data)

                # Create DataFrame for this track
                new_track = pd.DataFrame(interpolated_points)

                # Update n_segments to reflect actual interpolated length
                new_track['n_segments'] = len(new_track)

                # Recalculate track-level features for the interpolated track
                valid_lags = new_track['lag'].dropna()
                if len(valid_lags) > 0:
                    new_track['meanLag'] = valid_lags.mean()
                    new_track['track_length'] = valid_lags.sum()
                    new_track['meanVelocity'] = new_track['velocity'].mean()
                else:
                    new_track['meanLag'] = 0.001  # Small value for trapped sites
                    new_track['track_length'] = 0.001
                    new_track['meanVelocity'] = 0.001

                new_track['meanLocDistanceFromCenter'] = new_track['distanceFromMean'].mean()

                # Add geometric analysis features if enabled
                if 'Rg_geometric' in track_data.columns:
                    new_track['Rg_geometric'] = track_data['Rg_geometric'].iloc[0]
                    new_track['sRg_geometric'] = track_data['sRg_geometric'].iloc[0]
                    new_track['mean_step_length_geometric'] = new_track['meanLag']
                    new_track['geometric_mobility_classification'] = 'immobile'

                # Add differential for distance from origin
                distance_diff = np.diff(new_track['distanceFromOrigin'].to_numpy()) / np.diff(new_track['lagNumber'].to_numpy())
                distance_diff = np.insert(distance_diff, 0, 0)
                new_track['dy-dt_distance'] = distance_diff

                interpolated_tracks.append(new_track)

            # Combine all tracks
            if interpolated_tracks:
                interpolated_df = pd.concat(interpolated_tracks, ignore_index=True)

                # Add missing columns from original tracks_df
                for col in tracks_df.columns:
                    if col not in interpolated_df.columns:
                        interpolated_df[col] = np.nan

                # Ensure column order matches
                interpolated_df = interpolated_df[tracks_df.columns]

                # Combine with mobile tracks
                result_df = pd.concat([mobile_tracks, interpolated_df], ignore_index=True)

                # Count interpolated points
                interpolated_count = (interpolated_df['is_interpolated'] == 1).sum()
                print(f"    Enhanced interpolation: {interpolated_count} interpolated points for {len(set(interpolated_df['track_number']))} trapped tracks")

                return result_df
            else:
                return tracks_df

        except Exception as e:
            print(f"Error in enhanced interpolation: {e}")
            import traceback
            traceback.print_exc()
            return tracks_df

    @staticmethod
    def _calculate_intensity_for_position(image_array, frame, x, y):
        """Calculate intensity for a specific position in a specific frame"""
        try:
            n, h, w = image_array.shape

            frame = int(round(frame))
            x = int(round(x))
            y = int(round(y))

            # 3x3 pixel region
            xMin = x - 1
            xMax = x + 2
            yMin = y - 1
            yMax = y + 2

            # Edge case handling
            if xMin < 0:
                xMin = 0
            if xMax > w:
                xMax = w
            if yMin < 0:
                yMin = 0
            if yMax > h:
                yMax = h

            if frame >= n:
                frame = n - 1
            if frame < 0:
                frame = 0

            # Calculate intensity using same method as original
            intensity = np.mean(image_array[frame][yMin:yMax, xMin:xMax])
            return intensity

        except Exception as e:
            print(f"Error calculating intensity: {e}")
            return 0.0

class FullTrackInterpolator:
    """Full track interpolation for all tracks to fill missing frames"""

    @staticmethod
    def interpolate_all_tracks(tracks_df, file_path, extend_to_full_recording=False):
        """Add interpolated points for all tracks to fill missing frames

        Args:
            tracks_df: DataFrame with track data
            file_path: Path to image file
            extend_to_full_recording: If True, extend tracks from frame 0 to final frame
        """
        try:
            # Load image to get total frame count
            import skimage.io as skio
            if not os.path.exists(file_path):
                return tracks_df

            A = skio.imread(file_path, plugin='tifffile')
            n_frames = A.shape[0]

            # Add column to mark actual vs interpolated points
            tracks_df['is_interpolated'] = 0  # 0 = actual, 1 = interpolated

            all_interpolated_tracks = []
            all_interpolated_tracks.append(tracks_df)  # Include original tracks

            track_numbers = tracks_df['track_number'].unique()

            for track_num in track_numbers:
                track_data = tracks_df[tracks_df['track_number'] == track_num].sort_values('frame')

                if len(track_data) < 1:
                    continue

                # Get track-level properties
                track_level_props = {}
                for col in tracks_df.columns:
                    if col not in ['frame', 'x', 'y', 'intensity', 'id', 'lagNumber', 'velocity', 'lag',
                                 'zeroed_X', 'zeroed_Y', 'distanceFromOrigin', 'direction_Relative_To_Origin',
                                 'distanceFromMean', 'dy-dt_distance', 'is_interpolated']:
                        track_level_props[col] = track_data[col].iloc[0]

                # Get actual detected frames and positions
                actual_frames = set(track_data['frame'].astype(int))
                actual_x = track_data['x'].to_numpy()
                actual_y = track_data['y'].to_numpy()
                actual_frame_nums = track_data['frame'].astype(int).to_numpy()

                # Determine frame range for interpolation
                min_track_frame = int(track_data['frame'].min())
                max_track_frame = int(track_data['frame'].max())

                if extend_to_full_recording:
                    # NEW: Extend to entire recording
                    interpolation_start = 0
                    interpolation_end = n_frames - 1
                else:
                    # Original behavior: only within track range
                    interpolation_start = min_track_frame
                    interpolation_end = max_track_frame

                # Find missing frames within interpolation range
                all_interpolation_frames = set(range(interpolation_start, interpolation_end + 1))
                missing_frames = all_interpolation_frames - actual_frames

                if not missing_frames:
                    continue  # No missing frames to interpolate

                # Interpolate positions for missing frames
                interpolated_points = []

                for missing_frame in sorted(missing_frames):
                    # Determine interpolation method based on position relative to detections
                    if missing_frame < min_track_frame:
                        # Before first detection: use first detected position (constant extrapolation)
                        x_interp = actual_x[0]
                        y_interp = actual_y[0]
                    elif missing_frame > max_track_frame:
                        # After last detection: use last detected position (constant extrapolation)
                        x_interp = actual_x[-1]
                        y_interp = actual_y[-1]
                    else:
                        # Within detection range: linear interpolation
                        x_interp, y_interp = FullTrackInterpolator._interpolate_position(
                            missing_frame, actual_frame_nums, actual_x, actual_y)

                    # Calculate intensity for interpolated position
                    intensity = FullTrackInterpolator._calculate_intensity_for_position(
                        A, missing_frame, x_interp, y_interp)

                    # Create interpolated point data
                    point_data = {
                        'frame': missing_frame,
                        'x': x_interp,
                        'y': y_interp,
                        'intensity': intensity,
                        'is_interpolated': 1,  # Mark as interpolated
                    }

                    # Add track-level properties
                    point_data.update(track_level_props)

                    # Calculate frame-specific features
                    origin_x = actual_x[0] if len(actual_x) > 0 else x_interp
                    origin_y = actual_y[0] if len(actual_y) > 0 else y_interp

                    point_data['zeroed_X'] = x_interp - origin_x
                    point_data['zeroed_Y'] = y_interp - origin_y
                    point_data['distanceFromOrigin'] = np.sqrt(
                        (x_interp - origin_x)**2 + (y_interp - origin_y)**2)

                    # Set lagNumber relative to first detection
                    point_data['lagNumber'] = missing_frame - min_track_frame

                    # Direction relative to origin
                    if point_data['distanceFromOrigin'] > 0:
                        point_data['direction_Relative_To_Origin'] = np.degrees(
                            np.arctan2(point_data['zeroed_Y'], point_data['zeroed_X'])) % 360
                    else:
                        point_data['direction_Relative_To_Origin'] = 0

                    # Distance from track mean
                    if 'mean_X' in track_level_props and 'mean_Y' in track_level_props:
                        point_data['distanceFromMean'] = np.sqrt(
                            (x_interp - track_level_props['mean_X'])**2 +
                            (y_interp - track_level_props['mean_Y'])**2)

                    # Set velocity and lag to NaN for interpolated points
                    point_data['velocity'] = np.nan
                    point_data['lag'] = np.nan
                    point_data['dy-dt_distance'] = 0

                    interpolated_points.append(point_data)

                # Add interpolated points if any were created
                if interpolated_points:
                    interpolated_df = pd.DataFrame(interpolated_points)

                    # Add missing columns
                    for col in tracks_df.columns:
                        if col not in interpolated_df.columns:
                            interpolated_df[col] = np.nan

                    # Ensure column order matches
                    interpolated_df = interpolated_df[tracks_df.columns]
                    all_interpolated_tracks.append(interpolated_df)

            # Combine all tracks
            if len(all_interpolated_tracks) > 1:
                result_df = pd.concat(all_interpolated_tracks, ignore_index=True)
                result_df = result_df.sort_values(['track_number', 'frame']).reset_index(drop=True)

                # Recalculate lag and velocity for complete tracks
                result_df = FullTrackInterpolator._recalculate_motion_features(result_df)

                interpolated_count = (result_df['is_interpolated'] == 1).sum()
                if extend_to_full_recording:
                    print(f"    Added {interpolated_count} interpolated points across all tracks (extended to full recording)")
                else:
                    print(f"    Added {interpolated_count} interpolated points across all tracks")
                return result_df
            else:
                return tracks_df

        except Exception as e:
            print(f"Error in full track interpolation: {e}")
            import traceback
            traceback.print_exc()
            if 'is_interpolated' not in tracks_df.columns:
                tracks_df['is_interpolated'] = 0
            return tracks_df

    @staticmethod
    def _interpolate_position(target_frame, frame_nums, x_vals, y_vals):
        """Interpolate x,y position for target frame using linear interpolation"""
        # Find frames before and after target
        before_mask = frame_nums <= target_frame
        after_mask = frame_nums >= target_frame

        if not np.any(before_mask):
            # Target is before all actual frames, use first position
            return x_vals[0], y_vals[0]
        elif not np.any(after_mask):
            # Target is after all actual frames, use last position
            return x_vals[-1], y_vals[-1]
        else:
            # Interpolate between nearest frames
            before_frames = frame_nums[before_mask]
            after_frames = frame_nums[after_mask]

            closest_before = np.max(before_frames)
            closest_after = np.min(after_frames)

            if closest_before == closest_after:
                # Exact frame match (shouldn't happen for missing frames)
                idx = np.where(frame_nums == closest_before)[0][0]
                return x_vals[idx], y_vals[idx]

            # Linear interpolation
            before_idx = np.where(frame_nums == closest_before)[0][0]
            after_idx = np.where(frame_nums == closest_after)[0][0]

            fraction = (target_frame - closest_before) / (closest_after - closest_before)

            x_interp = x_vals[before_idx] + fraction * (x_vals[after_idx] - x_vals[before_idx])
            y_interp = y_vals[before_idx] + fraction * (y_vals[after_idx] - y_vals[before_idx])

            return x_interp, y_interp

    @staticmethod
    def _calculate_intensity_for_position(image_array, frame, x, y):
        """Calculate intensity for a specific position in a specific frame"""
        try:
            n, h, w = image_array.shape

            frame = int(round(frame))
            x = int(round(x))
            y = int(round(y))

            # 3x3 pixel region
            xMin = max(0, x - 1)
            xMax = min(w, x + 2)
            yMin = max(0, y - 1)
            yMax = min(h, y + 2)

            if frame >= n:
                frame = n - 1
            if frame < 0:
                frame = 0

            intensity = np.mean(image_array[frame][yMin:yMax, xMin:xMax])
            return intensity

        except Exception as e:
            print(f"Error calculating intensity: {e}")
            return 0.0

    @staticmethod
    def _recalculate_motion_features(tracks_df):
        """Recalculate motion features for complete tracks including interpolated points"""
        try:
            tracks_df = tracks_df.sort_values(['track_number', 'frame'])

            for track_num in tracks_df['track_number'].unique():
                track_mask = tracks_df['track_number'] == track_num
                track_data = tracks_df[track_mask]

                if len(track_data) < 2:
                    continue

                # Calculate lag displacements between consecutive points
                x_vals = track_data['x'].values
                y_vals = track_data['y'].values

                # Calculate distances between consecutive points
                distances = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)

                # Create lag array (last point has NaN)
                lag_array = np.append(distances, np.nan)

                # Calculate velocities (assuming dt=1 between frames)
                velocity_array = lag_array.copy()  # Same as lag if dt=1

                # Update the dataframe
                track_indices = track_data.index
                tracks_df.loc[track_indices, 'lag'] = lag_array
                tracks_df.loc[track_indices, 'velocity'] = velocity_array

                # Recalculate distance differential
                if 'distanceFromOrigin' in track_data.columns:
                    distances_from_origin = track_data['distanceFromOrigin'].values
                    lag_numbers = track_data['lagNumber'].values

                    if len(distances_from_origin) > 1:
                        distance_diff = np.diff(distances_from_origin) / np.diff(lag_numbers)
                        distance_diff = np.insert(distance_diff, 0, 0)  # First point gets 0
                        tracks_df.loc[track_indices, 'dy-dt_distance'] = distance_diff

            return tracks_df

        except Exception as e:
            print(f"Error recalculating motion features: {e}")
            return tracks_df

class DistanceDifferentialAnalyzer:
    """Distance differential analysis"""

    @staticmethod
    def add_distance_differential(tracks_df):
        """Add dy-dt: distance (rate of change of distance from origin)"""
        tracks_df = tracks_df.sort_values(['track_number', 'frame'])

        # Initialize the column
        tracks_df['dy-dt_distance'] = np.nan

        for track_num in tracks_df['track_number'].unique():
            track_mask = tracks_df['track_number'] == track_num
            track_data = tracks_df[track_mask]

            if len(track_data) < 2 or 'distanceFromOrigin' not in track_data.columns:
                continue

            if 'lagNumber' not in track_data.columns:
                # Create lagNumber if it doesn't exist
                min_frame = track_data['frame'].min()
                lag_numbers = track_data['frame'] - min_frame
            else:
                lag_numbers = track_data['lagNumber']

            distances = track_data['distanceFromOrigin'].values
            lag_vals = lag_numbers.values

            # Calculate differential
            if len(distances) > 1 and len(lag_vals) > 1:
                diff = np.diff(distances) / np.diff(lag_vals)
                diff = np.insert(diff, 0, 0)  # First point gets 0

                # Update the dataframe
                track_indices = track_data.index
                for i, idx in enumerate(track_indices):
                    if i < len(diff):
                        tracks_df.at[idx, 'dy-dt_distance'] = diff[i]

        return tracks_df


class SVMClassifier:
    """SVM classification for track mobility"""

    @staticmethod
    def classify_tracks(tracks_df, training_data_path, experiment_name):
        """Classify tracks using SVM"""
        try:
            if not training_data_path or not os.path.exists(training_data_path):
                print("No training data available for classification")
                return tracks_df

            print("Performing SVM classification...")

            # Load and prepare training data
            train_feats = pd.read_csv(training_data_path)

            # Filter training data if needed
            if 'Experiment' in train_feats.columns:
                train_feats = train_feats.loc[train_feats['Experiment'] == 'tdTomato_37Degree']

            # Required columns for classification
            required_cols = ['NetDispl', 'Straight', 'Asymmetry', 'radiusGyration', 'Kurtosis', 'fracDimension']

            if not all(col in train_feats.columns for col in required_cols):
                print(f"Training data missing required columns: {required_cols}")
                return tracks_df

            # Map column names in tracks data
            tracks_df_mapped = tracks_df.rename(columns={
                'netDispl': 'NetDispl',
                'Straight': 'Straight',
                'asymmetry': 'Asymmetry',
                'radius_gyration': 'radiusGyration',
                'kurtosis': 'Kurtosis',
                'fracDimension': 'fracDimension'
            })

            # Check if tracks data has required columns
            missing_cols = [col for col in required_cols if col not in tracks_df_mapped.columns]
            if missing_cols:
                print(f"Tracks data missing columns: {missing_cols}")
                return tracks_df

            # Prepare training data
            X_train = train_feats[required_cols].dropna()
            y_train = train_feats['Elected_Label'].dropna()

            # Replace labels with numbers
            label_map = {"mobile": 1, "confined": 2, "trapped": 3}
            y_train = y_train.replace(label_map)

            # Prepare test data (get unique tracks only)
            unique_tracks = tracks_df_mapped.groupby('track_number').first().reset_index()
            X_test = unique_tracks[required_cols].dropna()

            if len(X_test) == 0:
                print("No valid tracks for classification")
                return tracks_df

            # Box-Cox transformation
            def prepare_box_cox_data(data):
                data = data.copy()
                for col in data.columns:
                    minVal = data[col].min()
                    if minVal <= 0:
                        data[col] += (np.abs(minVal) + 1e-15)
                return data

            X_train_bc = prepare_box_cox_data(X_train)
            X_test_bc = prepare_box_cox_data(X_test)

            # Transform data
            transformer = PowerTransformer(method='box-cox')
            X_train_transformed = transformer.fit_transform(X_train_bc)
            X_test_transformed = transformer.transform(X_test_bc)

            # Create and train pipeline
            pipeline = Pipeline([
                ("pca", PCA(n_components=3)),
                ("scaler", StandardScaler()),
                ("SVC", SVC(kernel="rbf"))
            ])

            # Grid search for best parameters
            param_grid = {
                "SVC__C": [0.1, 1, 10, 100],
                "SVC__gamma": [0.001, 0.01, 0.1, 1.0]
            }

            grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train_transformed, y_train)

            # Predict on test data
            predictions = grid_search.predict(X_test_transformed)

            # Create mapping from track_number to prediction
            track_predictions = dict(zip(X_test.index, predictions))
            unique_tracks_mapping = dict(zip(unique_tracks.index, unique_tracks['track_number']))

            # Add predictions to all rows in DataFrame
            tracks_df['SVM'] = 0  # Default
            for idx, track_num in unique_tracks_mapping.items():
                if idx in track_predictions:
                    tracks_df.loc[tracks_df['track_number'] == track_num, 'SVM'] = int(track_predictions[idx])

            # Only add/update experiment name if it doesn't already exist or is empty
            if 'Experiment' not in tracks_df.columns or tracks_df['Experiment'].isna().all():
                tracks_df['Experiment'] = experiment_name
                print(f"Added experiment name '{experiment_name}' to tracks dataframe")
            else:
                print(f"Experiment column already exists, preserving existing values")

            print(f"Classification complete. {len(predictions)} tracks classified.")

            return tracks_df

        except Exception as e:
            print(f"Error in SVM classification: {e}")
            return tracks_df


class SPTBatchAnalysis(QWidget):
    """Main FLIKA plugin class for SPT batch analysis with detection capability"""

    def __init__(self):
        super().__init__()
        self.parameters = SPTAnalysisParameters()
        self.autocorr_worker = None
        self.detection_worker = None
        self.current_tracks_df = None
        self.file_logger = FileLogger()
        self.current_analysis_dir = None
        self.setupUI()

    def setupUI(self):
        """Create the main user interface with enhanced linking options"""
        self.setWindowTitle("SPT Batch Analysis - Enhanced with Dual Linking Methods and File Logging")
        self.setMinimumSize(1200, 800)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create tabbed interface
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs (all existing tabs preserved)
        self.create_file_tab()
        self.create_detection_tab()
        self.create_enhanced_parameters_tab()  # Enhanced with linking options
        self.create_analysis_steps_tab()
        self.create_geometric_analysis_tab()
        self.create_autocorrelation_tab()
        self.create_progress_tab()
        self.create_export_control_tab()
        self.create_thunderstorm_tab()  # NEW: ThunderSTORM macro generation

        # Add this connection to update availability when tab is selected
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Control buttons
        self.create_control_buttons(main_layout)

        # Initialize thunderSTORM attributes
        self.add_thunderstorm_attributes_to_init()

    def on_tab_changed(self, index):
        """Handle tab changes to update export control availability"""
        # Check if we switched to the Export Control tab
        if self.tab_widget.tabText(index) == "Export Control":
            self.update_export_control_availability()
            self.update_column_summary()

    def create_detection_tab(self):
        """Create detection configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Enable/Disable Detection
        enable_group = QGroupBox("Enable Particle Detection")
        enable_layout = QVBoxLayout(enable_group)

        self.detection_enable_checkbox = QCheckBox("Enable U-Track Based Particle Detection")
        self.detection_enable_checkbox.setChecked(self.parameters.enable_detection)
        self.detection_enable_checkbox.toggled.connect(self.on_detection_enable_toggled)
        enable_layout.addWidget(self.detection_enable_checkbox)

        info_label = QLabel("Detects particles in images using U-Track methodology and saves as localization CSV files")
        info_label.setStyleSheet("color: #666; font-style: italic; margin-left: 20px;")
        enable_layout.addWidget(info_label)

        scroll_layout.addWidget(enable_group)

        # Detection Parameters
        self.detection_params_group = QGroupBox("Detection Parameters")
        params_layout = QFormLayout(self.detection_params_group)

        # PSF Sigma
        self.detection_psf_sigma_spin = QDoubleSpinBox()
        self.detection_psf_sigma_spin.setRange(0.5, 10.0)
        self.detection_psf_sigma_spin.setDecimals(2)
        self.detection_psf_sigma_spin.setValue(self.parameters.detection_psf_sigma)
        self.detection_psf_sigma_spin.setSuffix(" pixels")
        params_layout.addRow("PSF Sigma:", self.detection_psf_sigma_spin)

        # Alpha threshold
        self.detection_alpha_spin = QDoubleSpinBox()
        self.detection_alpha_spin.setRange(0.001, 0.5)
        self.detection_alpha_spin.setDecimals(3)
        self.detection_alpha_spin.setValue(self.parameters.detection_alpha_threshold)
        params_layout.addRow("Significance Threshold (α):", self.detection_alpha_spin)

        # Minimum intensity
        self.detection_min_intensity_spin = QDoubleSpinBox()
        self.detection_min_intensity_spin.setRange(0, 10000)
        self.detection_min_intensity_spin.setValue(self.parameters.detection_min_intensity)
        params_layout.addRow("Minimum Intensity:", self.detection_min_intensity_spin)

        scroll_layout.addWidget(self.detection_params_group)

        # Output Options
        self.detection_output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(self.detection_output_group)

        # Output directory
        dir_layout = QHBoxLayout()
        self.detection_output_label = QLabel("Same as input directory")
        self.detection_output_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        dir_layout.addWidget(self.detection_output_label)

        self.select_detection_output_btn = QPushButton("Choose Directory")
        self.select_detection_output_btn.clicked.connect(self.select_detection_output_directory)
        dir_layout.addWidget(self.select_detection_output_btn)

        output_layout.addLayout(dir_layout)

        # Skip existing files option
        self.detection_skip_existing_checkbox = QCheckBox("Skip files that already have detection results")
        self.detection_skip_existing_checkbox.setChecked(self.parameters.detection_skip_existing)
        output_layout.addWidget(self.detection_skip_existing_checkbox)

        # Show results option
        self.detection_show_results_checkbox = QCheckBox("Open image and display detection results after detection")
        self.detection_show_results_checkbox.setChecked(self.parameters.detection_show_results)
        output_layout.addWidget(self.detection_show_results_checkbox)

        scroll_layout.addWidget(self.detection_output_group)

        # Detection Controls
        self.detection_controls_group = QGroupBox("Detection Controls")
        controls_layout = QVBoxLayout(self.detection_controls_group)

        # Start detection button
        self.start_detection_btn = QPushButton("Start Detection on Selected Files")
        self.start_detection_btn.clicked.connect(self.start_detection)
        self.start_detection_btn.setEnabled(False)
        controls_layout.addWidget(self.start_detection_btn)

        # Detection progress
        self.detection_progress_bar = QProgressBar()
        self.detection_progress_bar.setVisible(False)
        controls_layout.addWidget(self.detection_progress_bar)

        # Detection status
        self.detection_status_label = QLabel("Ready for detection")
        self.detection_status_label.setStyleSheet("color: #666; font-style: italic;")
        controls_layout.addWidget(self.detection_status_label)

        scroll_layout.addWidget(self.detection_controls_group)

        # Method Description
        desc_group = QGroupBox("Method Description")
        desc_layout = QVBoxLayout(desc_group)

        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(200)
        desc_text.setText("""
U-Track Based Particle Detection:

This detection method implements key components of the U-Track detection pipeline:

1. Background Estimation: Robust background estimation using percentile-based methods
2. Pre-filtering: Gaussian filtering to match expected PSF size
3. Local Maxima Detection: Morphological detection of intensity peaks
4. Statistical Testing: Significance testing against background noise
5. Sub-pixel Localization: Intensity-weighted centroid refinement

Parameters:
• PSF Sigma: Expected point spread function width (typically 1.0-2.0 pixels)
• Significance Threshold: Statistical threshold for detection (0.05 = 95% confidence)
• Minimum Intensity: Absolute minimum intensity threshold

Integration with Main Analysis:
When detection is enabled, particle detection will automatically run before tracking
when you click "Start Analysis". You don't need to manually run detection first.

Output Options:
• Skip existing: Avoid re-detecting files that already have results
• Display results: Open images in FLIKA with detection points overlaid

Output Format:
Results are saved as CSV files with suffix "_locsID.csv" containing:
• frame: Frame number (1-based)
• x [nm], y [nm]: Coordinates in nanometers
• intensity [photon]: Particle intensity
• id: Unique particle identifier

These files are automatically used by the tracking pipeline.
        """)
        desc_layout.addWidget(desc_text)

        scroll_layout.addWidget(desc_group)

        # Enable/disable based on initial state
        self.on_detection_enable_toggled()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Detection")

    def on_detection_enable_toggled(self):
        """Handle detection enable/disable"""
        enabled = self.detection_enable_checkbox.isChecked()
        self.detection_params_group.setEnabled(enabled)
        self.detection_output_group.setEnabled(enabled)
        self.detection_controls_group.setEnabled(enabled)

        # Update start button availability
        if enabled and hasattr(self, 'file_paths') and self.file_paths:
            self.start_detection_btn.setEnabled(True)
        else:
            self.start_detection_btn.setEnabled(False)

    def select_detection_output_directory(self):
        """Select output directory for detection results"""
        directory = QFileDialog.getExistingDirectory(self, "Select Detection Output Directory")
        if directory:
            self.detection_output_label.setText(directory)
            self.parameters.detection_output_directory = directory

    def start_detection(self):
        """Start detection on selected files"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            g.alert("No files selected for detection")
            return

        # Update parameters
        self.update_parameters()

        # Determine output directory
        output_dir = (self.parameters.detection_output_directory
                     if self.parameters.detection_output_directory
                     else os.path.dirname(self.file_paths[0]))

        # Filter files if skip existing is enabled
        files_to_process = []
        if self.parameters.detection_skip_existing:
            for file_path in self.file_paths:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                detection_file = os.path.join(output_dir, f"{base_name}_locsID.csv")
                if not os.path.exists(detection_file):
                    files_to_process.append(file_path)

            if len(files_to_process) != len(self.file_paths):
                skipped = len(self.file_paths) - len(files_to_process)
                self.detection_status_label.setText(f"Skipping {skipped} files with existing results")
        else:
            files_to_process = self.file_paths

        if not files_to_process:
            g.alert("No files to process (all have existing detection results)")
            return

        # Prepare detector parameters
        detector_params = {
            'psf_sigma': self.detection_psf_sigma_spin.value(),
            'alpha_threshold': self.detection_alpha_spin.value(),
            'min_intensity': self.detection_min_intensity_spin.value()
        }

        # Start detection worker
        self.detection_worker = DetectionWorker(
            files_to_process, detector_params, output_dir, self.parameters.pixel_size,
            show_results=self.parameters.detection_show_results
        )

        self.detection_worker.progress_update.connect(self.update_detection_status)
        self.detection_worker.frame_progress.connect(self.update_detection_progress)
        self.detection_worker.detection_complete.connect(self.on_detection_complete)
        self.detection_worker.detection_error.connect(self.on_detection_error)
        self.detection_worker.file_processed.connect(self.on_detection_file_processed)

        # Update UI
        self.detection_progress_bar.setVisible(True)
        self.detection_progress_bar.setValue(0)
        self.start_detection_btn.setEnabled(False)
        self.detection_status_label.setText("Starting detection...")

        self.detection_worker.start()

    def update_detection_status(self, message):
        """Update detection status"""
        self.detection_status_label.setText(message)

    def update_detection_progress(self, progress):
        """Update detection progress bar"""
        self.detection_progress_bar.setValue(progress)

    def on_detection_complete(self, message):
        """Handle detection completion"""
        self.detection_progress_bar.setVisible(False)
        self.start_detection_btn.setEnabled(True)
        self.detection_status_label.setText(message)
        self.detection_status_label.setStyleSheet("color: green; font-weight: bold;")
        g.alert("Detection completed successfully!")

    def on_detection_error(self, error_msg):
        """Handle detection error"""
        self.detection_progress_bar.setVisible(False)
        self.start_detection_btn.setEnabled(True)
        self.detection_status_label.setText(f"Error: {error_msg}")
        self.detection_status_label.setStyleSheet("color: red;")
        g.alert(f"Detection failed: {error_msg}")

    def on_detection_file_processed(self, image_file_path, detection_file_path):
        """Handle visualization of detection results"""
        try:
            self.visualize_detection_results(image_file_path, detection_file_path)
        except Exception as e:
            print(f"Error visualizing detection results: {e}")

    def visualize_detection_results(self, image_file_path, detection_file_path):
        """Open image in FLIKA and overlay detection results as scatter points"""
        try:
            # Open the image in FLIKA
            from flika.process.file_ import open_file
            window = open_file(image_file_path)

            if window is None:
                print(f"Could not open image file: {image_file_path}")
                return

            # Load detection results
            detections_df = pd.read_csv(detection_file_path)

            if len(detections_df) == 0:
                print("No detections to display")
                return

            # Convert coordinates back to pixels
            x_nm = detections_df['x [nm]'].values
            y_nm = detections_df['y [nm]'].values
            frames = detections_df['frame'].values - 1  # Convert to 0-based

            # Convert from nanometers to pixels
            x_pixels = x_nm / self.parameters.pixel_size
            y_pixels = y_nm / self.parameters.pixel_size

            # NO COORDINATE TRANSFORMATION NEEDED!
            # The test shows that detection coordinates are already correct for FLIKA
            x_flika = x_pixels
            y_flika = y_pixels

            # Get image dimensions for bounds checking
            img_height, img_width = window.imageDimensions()

            # Set up detection point properties
            detection_point_size = 6

            # Group detections by frame and add to scatter plot
            points_added = 0
            for frame in np.unique(frames):
                frame_mask = frames == frame
                if np.any(frame_mask):
                    frame_x = x_flika[frame_mask]
                    frame_y = y_flika[frame_mask]

                    # Add points to this frame's scatter plot
                    for x, y in zip(frame_x, frame_y):
                        try:
                            # Ensure coordinates are within image bounds
                            if 0 <= x < img_width and 0 <= y < img_height:
                                # FLIKA addPoint format: [frame, y, x]
                                window.addPoint([int(frame), float(y), float(x)])
                                points_added += 1
                            else:
                                print(f"Point ({x:.1f}, {y:.1f}) outside image bounds ({img_width}, {img_height})")
                        except Exception as e:
                            print(f"Error adding point at ({x}, {y}) frame {frame}: {e}")

            # Switch to the first detection frame if there are detections
            if len(detections_df) > 0:
                first_frame = int(detections_df['frame'].iloc[0]) - 1
                window.setIndex(max(0, first_frame))

            # Set point color and size for better visibility
            try:
                # Update scatter plot appearance for detections
                import flika.global_vars as g

                # Change point settings for better visibility of detections
                g.settings["point_color"] = "#00FFFF"  # Cyan
                g.settings["point_size"] = detection_point_size

                print(f"Successfully displayed {points_added}/{len(detections_df)} detections as cyan points")
                print(f"Image dimensions: {img_height} x {img_width}")
                print("Detection points are shown in cyan. Navigate between frames to see all detections.")
                print("✅ Using direct coordinate mapping (no transformation needed)")

                if points_added < len(detections_df):
                    print(f"Note: {len(detections_df) - points_added} points were outside image bounds")

            except Exception as e:
                print(f"Error updating point appearance: {e}")

        except Exception as e:
            print(f"Error in visualize_detection_results: {e}")
            import traceback
            traceback.print_exc()

    def run_detection_for_file(self, file_path):
        """Run detection for a single file (synchronous for integration with main analysis)"""
        try:
            # Determine output directory
            output_dir = (self.parameters.detection_output_directory
                         if self.parameters.detection_output_directory
                         else os.path.dirname(file_path))

            # Check if detection file already exists
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            detection_file = os.path.join(output_dir, f"{base_name}_locsID.csv")

            if self.parameters.detection_skip_existing and os.path.exists(detection_file):
                self.log_message(f"    Detection file already exists, skipping: {os.path.basename(detection_file)}")
                return True

            self.log_message(f"    Running detection on {os.path.basename(file_path)}")

            # Load image
            images = skio.imread(file_path, plugin='tifffile')
            if images.ndim == 2:
                images = images[np.newaxis, ...]

            # Apply same transformations as tracking pipeline
            images = np.rot90(images, axes=(1,2))
            images = np.fliplr(images)

            # Create detector
            detector = UTrackDetector(
                psf_sigma=self.parameters.detection_psf_sigma,
                alpha_threshold=self.parameters.detection_alpha_threshold,
                min_intensity=self.parameters.detection_min_intensity
            )

            # Detect particles in each frame
            all_detections = []
            n_frames = images.shape[0]

            for frame_idx in range(n_frames):
                frame_detections = detector.detect_particles_single_frame(
                    images[frame_idx], frame_idx
                )
                if len(frame_detections) > 0:
                    all_detections.append(frame_detections)

            # Save results
            if all_detections:
                combined_detections = pd.concat(all_detections, ignore_index=True)

                # Format for tracking pipeline
                combined_detections['x [nm]'] = combined_detections['x'] * self.parameters.pixel_size
                combined_detections['y [nm]'] = combined_detections['y'] * self.parameters.pixel_size
                combined_detections['frame'] = combined_detections['frame'] + 1  # 1-based
                combined_detections['intensity [photon]'] = combined_detections['intensity']
                combined_detections['id'] = range(len(combined_detections))

                # Save file
                output_columns = ['frame', 'x [nm]', 'y [nm]', 'intensity [photon]', 'id']
                combined_detections[output_columns].to_csv(detection_file, index=False)

                self.log_message(f"    Saved {len(combined_detections)} detections to {os.path.basename(detection_file)}")

                # Visualize if requested
                if self.parameters.detection_show_results:
                    self.visualize_detection_results(file_path, detection_file)

                return True
            else:
                self.log_message(f"    No detections found in {os.path.basename(file_path)}")
                return True

        except Exception as e:
            self.log_message(f"    Detection failed for {os.path.basename(file_path)}: {e}")
            return False


    def create_file_tab(self):
        """Create file selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)

        # Directory selection
        file_layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.dir_path_edit = QLabel("No directory selected")
        self.dir_path_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        file_layout.addWidget(self.dir_path_edit, 0, 1)

        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.clicked.connect(self.select_directory)
        file_layout.addWidget(self.select_dir_btn, 0, 2)

        # File pattern
        file_layout.addWidget(QLabel("File Pattern:"), 1, 0)
        self.file_pattern_edit = QComboBox()
        self.file_pattern_edit.setEditable(True)
        self.file_pattern_edit.addItems(['**/*.tif', '**/*_bin*.tif', '**/*_crop*.tif'])
        file_layout.addWidget(self.file_pattern_edit, 1, 1)

        self.refresh_files_btn = QPushButton("Refresh")
        self.refresh_files_btn.clicked.connect(self.refresh_file_list)
        file_layout.addWidget(self.refresh_files_btn, 1, 2)

        layout.addWidget(file_group)

        # File list
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(150)
        layout.addWidget(self.file_list_widget)

        self.file_count_label = QLabel("0 files selected")
        layout.addWidget(self.file_count_label)

        self.tab_widget.addTab(tab, "Files")


    def create_enhanced_parameters_tab(self):
        """Enhanced parameters tab with dual linking algorithm selection - preserves all existing functionality"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)

        # Create main scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()

        # Two-column layout for the scrollable content
        scroll_main_layout = QHBoxLayout(scroll_widget)

        # Left Column
        left_scroll = QScrollArea()
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Right Column
        right_scroll = QScrollArea()
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # === LEFT COLUMN CONTENT ===

        # Basic parameters (unchanged)
        basic_group = QGroupBox("Basic Parameters")
        basic_layout = QFormLayout(basic_group)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.001, 1000.0)
        self.pixel_size_spin.setValue(self.parameters.pixel_size)
        self.pixel_size_spin.setSuffix(" nm")
        basic_layout.addRow("Pixel Size:", self.pixel_size_spin)

        self.frame_length_spin = QDoubleSpinBox()
        self.frame_length_spin.setRange(0.0001, 1000.0)
        self.frame_length_spin.setValue(self.parameters.frame_length)
        self.frame_length_spin.setDecimals(4)
        self.frame_length_spin.setSuffix(" s")
        basic_layout.addRow("Frame Length:", self.frame_length_spin)

        left_layout.addWidget(basic_group)

        # Linking Method Selection (unchanged)
        linking_group = QGroupBox("Particle Linking Method")
        linking_layout = QVBoxLayout(linking_group)

        # Method selection
        self.linking_method_group = QButtonGroup()

        self.builtin_linking_radio = QRadioButton("Built-in Linking (Original Method)")
        self.builtin_linking_radio.setChecked(self.parameters.linking_method == 'builtin')
        self.linking_method_group.addButton(self.builtin_linking_radio, 0)
        linking_layout.addWidget(self.builtin_linking_radio)

        self.trackpy_linking_radio = QRadioButton("Trackpy Linking")
        self.trackpy_linking_radio.setChecked(self.parameters.linking_method == 'trackpy')
        self.linking_method_group.addButton(self.trackpy_linking_radio, 1)
        linking_layout.addWidget(self.trackpy_linking_radio)

        self.utrack_linking_radio = QRadioButton("U-Track Linking with Mixed Motion Models")
        self.utrack_linking_radio.setChecked(self.parameters.linking_method == 'utrack')
        self.linking_method_group.addButton(self.utrack_linking_radio, 2)
        linking_layout.addWidget(self.utrack_linking_radio)

        # Check availability (unchanged)
        if not TrackpyLinker.check_trackpy_available():
            self.trackpy_linking_radio.setEnabled(False)
            trackpy_status = QLabel("⚠️ Trackpy not available. Install with: pip install trackpy")
            trackpy_status.setStyleSheet("color: orange; font-style: italic;")
            linking_layout.addWidget(trackpy_status)
        else:
            trackpy_status = QLabel("✅ Trackpy available")
            trackpy_status.setStyleSheet("color: green;")
            linking_layout.addWidget(trackpy_status)

        # Check U-Track availability (unchanged)
        try:
            plugin_dir = get_plugin_directory()
            utrack_file = os.path.join(plugin_dir, 'utrack_linking.py')
            if os.path.exists(utrack_file):
                utrack_status = QLabel("✅ U-Track with mixed motion models available")
                utrack_status.setStyleSheet("color: green;")
            else:
                utrack_status = QLabel("⚠️ utrack_linking.py not found in plugin directory")
                utrack_status.setStyleSheet("color: orange; font-style: italic;")
                self.utrack_linking_radio.setEnabled(False)
        except Exception:
            utrack_status = QLabel("❌ U-Track linking not available")
            utrack_status.setStyleSheet("color: red; font-style: italic;")
            self.utrack_linking_radio.setEnabled(False)

        linking_layout.addWidget(utrack_status)

        # Connect radio button changes
        self.linking_method_group.buttonToggled.connect(self.on_linking_method_changed)

        left_layout.addWidget(linking_group)

        # Common parameters (unchanged)
        common_group = QGroupBox("Common Linking Parameters")
        common_layout = QFormLayout(common_group)

        self.min_segments_spin = QSpinBox()
        self.min_segments_spin.setRange(1, 1000)
        self.min_segments_spin.setValue(self.parameters.min_track_segments)
        common_layout.addRow("Min Track Segments:", self.min_segments_spin)

        left_layout.addWidget(common_group)

        # Built-in linking parameters (ENHANCED with algorithm selection)
        self.builtin_linking_group = QGroupBox("Built-in Linking Parameters")
        builtin_main_layout = QVBoxLayout(self.builtin_linking_group)

        # NEW: Algorithm Selection
        algorithm_group = QGroupBox("Algorithm Selection")
        algorithm_layout = QVBoxLayout(algorithm_group)

        # Algorithm selection radio buttons
        self.algorithm_group = QButtonGroup()

        self.iterative_radio = QRadioButton("Iterative Algorithm (Recommended)")
        self.iterative_radio.setChecked(getattr(self.parameters, 'builtin_linking_algorithm', 'iterative') == 'iterative')
        self.iterative_radio.setToolTip("Uses iterative algorithm that prevents recursion errors. Recommended for all datasets.")
        self.algorithm_group.addButton(self.iterative_radio, 0)
        algorithm_layout.addWidget(self.iterative_radio)

        iterative_desc = QLabel("• Handles arbitrarily long tracks without errors\n• Memory efficient and scalable\n• Recommended for all new analyses")
        iterative_desc.setStyleSheet("color: #666; font-style: italic; font-size: 10px; margin-left: 20px;")
        iterative_desc.setWordWrap(True)
        algorithm_layout.addWidget(iterative_desc)

        self.recursive_radio = QRadioButton("Recursive Algorithm (Legacy)")
        self.recursive_radio.setChecked(getattr(self.parameters, 'builtin_linking_algorithm', 'iterative') == 'recursive')
        self.recursive_radio.setToolTip("Original recursive algorithm. May fail on complex datasets with 'recursion limit exceeded' errors.")
        self.algorithm_group.addButton(self.recursive_radio, 1)
        algorithm_layout.addWidget(self.recursive_radio)

        recursive_desc = QLabel("• Original algorithm from previous versions\n• May fail with recursion errors on complex data\n• Use only for compatibility with previous analyses")
        recursive_desc.setStyleSheet("color: #666; font-style: italic; font-size: 10px; margin-left: 20px;")
        recursive_desc.setWordWrap(True)
        algorithm_layout.addWidget(recursive_desc)

        # NEW: Recursion limit parameter
        recursion_layout = QFormLayout()
        self.recursion_limit_spin = QSpinBox()
        self.recursion_limit_spin.setRange(100, 10000)
        self.recursion_limit_spin.setValue(getattr(self.parameters, 'recursive_depth_limit', 1000))
        self.recursion_limit_spin.setEnabled(getattr(self.parameters, 'builtin_linking_algorithm', 'iterative') == 'recursive')
        self.recursion_limit_spin.setToolTip("Maximum recursion depth before stopping. Higher values allow longer tracks but may cause stack overflow.")
        self.recursion_limit_spin.setSuffix(" levels")
        recursion_layout.addRow("Recursion Limit:", self.recursion_limit_spin)

        algorithm_layout.addLayout(recursion_layout)
        builtin_main_layout.addWidget(algorithm_group)

        # Connect algorithm selection to enable/disable recursion limit
        self.algorithm_group.buttonToggled.connect(self.on_algorithm_changed)

        # Standard built-in linking parameters (unchanged)
        builtin_params_layout = QFormLayout()

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(self.parameters.max_gap_frames)
        builtin_params_layout.addRow("Max Gap Frames:", self.max_gap_spin)

        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(0.1, 100.0)
        self.max_dist_spin.setValue(self.parameters.max_link_distance)
        self.max_dist_spin.setSuffix(" pixels")
        builtin_params_layout.addRow("Max Link Distance:", self.max_dist_spin)

        builtin_main_layout.addLayout(builtin_params_layout)
        left_layout.addWidget(self.builtin_linking_group)

        # Trackpy linking parameters (unchanged)
        self.trackpy_linking_group = QGroupBox("Trackpy Linking Parameters")
        trackpy_layout = QFormLayout(self.trackpy_linking_group)

        # Link distance
        self.trackpy_distance_spin = QDoubleSpinBox()
        self.trackpy_distance_spin.setRange(0.1, 100.0)
        self.trackpy_distance_spin.setValue(self.parameters.trackpy_link_distance)
        self.trackpy_distance_spin.setSuffix(" pixels")
        trackpy_layout.addRow("Search Distance:", self.trackpy_distance_spin)

        # Memory
        self.trackpy_memory_spin = QSpinBox()
        self.trackpy_memory_spin.setRange(0, 100)
        self.trackpy_memory_spin.setValue(self.parameters.trackpy_memory)
        self.trackpy_memory_spin.setSuffix(" frames")
        trackpy_layout.addRow("Memory:", self.trackpy_memory_spin)

        # Linking type
        self.trackpy_type_combo = QComboBox()
        self.trackpy_type_combo.addItems(['standard', 'adaptive', 'velocityPredict', 'adaptive + velocityPredict'])
        self.trackpy_type_combo.setCurrentText(self.parameters.trackpy_linking_type)
        self.trackpy_type_combo.currentTextChanged.connect(self.on_trackpy_type_changed)
        trackpy_layout.addRow("Linking Type:", self.trackpy_type_combo)

        # Advanced trackpy parameters
        self.trackpy_advanced_group = QGroupBox("Advanced Trackpy Parameters")
        advanced_layout = QFormLayout(self.trackpy_advanced_group)

        self.trackpy_max_search_spin = QDoubleSpinBox()
        self.trackpy_max_search_spin.setRange(0.1, 100.0)
        self.trackpy_max_search_spin.setValue(self.parameters.trackpy_max_search_distance)
        self.trackpy_max_search_spin.setSuffix(" pixels")
        advanced_layout.addRow("Max Search Distance:", self.trackpy_max_search_spin)

        self.trackpy_adaptive_stop_spin = QDoubleSpinBox()
        self.trackpy_adaptive_stop_spin.setRange(0.01, 1.0)
        self.trackpy_adaptive_stop_spin.setDecimals(3)
        self.trackpy_adaptive_stop_spin.setValue(self.parameters.trackpy_adaptive_stop)
        advanced_layout.addRow("Adaptive Stop:", self.trackpy_adaptive_stop_spin)

        self.trackpy_adaptive_step_spin = QDoubleSpinBox()
        self.trackpy_adaptive_step_spin.setRange(0.01, 1.0)
        self.trackpy_adaptive_step_spin.setDecimals(3)
        self.trackpy_adaptive_step_spin.setValue(self.parameters.trackpy_adaptive_step)
        advanced_layout.addRow("Adaptive Step:", self.trackpy_adaptive_step_spin)

        trackpy_layout.addRow(self.trackpy_advanced_group)
        left_layout.addWidget(self.trackpy_linking_group)

        # U-Track with Mixed Motion Parameters (unchanged)
        self.utrack_linking_group = QGroupBox("U-Track Mixed Motion Parameters")
        utrack_layout = QVBoxLayout(self.utrack_linking_group)

        # Basic U-Track parameters
        basic_utrack_layout = QFormLayout()

        # Max linking distance
        self.utrack_max_distance_spin = QDoubleSpinBox()
        self.utrack_max_distance_spin.setRange(0.1, 100.0)
        self.utrack_max_distance_spin.setValue(self.parameters.utrack_max_linking_distance)
        self.utrack_max_distance_spin.setSuffix(" pixels")
        basic_utrack_layout.addRow("Max Link Distance:", self.utrack_max_distance_spin)

        # Max gap frames
        self.utrack_max_gap_spin = QSpinBox()
        self.utrack_max_gap_spin.setRange(0, 50)
        self.utrack_max_gap_spin.setValue(self.parameters.utrack_max_gap_frames)
        self.utrack_max_gap_spin.setSuffix(" frames")
        basic_utrack_layout.addRow("Max Gap Frames:", self.utrack_max_gap_spin)

        utrack_layout.addLayout(basic_utrack_layout)

        # Motion model selection
        motion_model_group = QGroupBox("Motion Model Selection")
        motion_model_layout = QVBoxLayout(motion_model_group)

        self.motion_model_combo = QComboBox()
        self.motion_model_combo.addItems(['brownian', 'linear', 'confined', 'mixed'])
        self.motion_model_combo.setCurrentText(self.parameters.utrack_motion_model)
        self.motion_model_combo.currentTextChanged.connect(self.on_motion_model_changed)
        motion_model_layout.addWidget(self.motion_model_combo)

        # Motion model descriptions
        model_desc = QLabel("""Motion Models:
    • Brownian: Random diffusive motion
    • Linear: Directed motion with persistent velocity
    • Confined: Motion restricted to a region
    • Mixed: PMMS-style heterogeneous motion (switches between types)""")
        model_desc.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        model_desc.setWordWrap(True)
        motion_model_layout.addWidget(model_desc)

        utrack_layout.addWidget(motion_model_group)

        # Mixed motion specific parameters (shown when mixed model is selected)
        self.mixed_motion_group = QGroupBox("Mixed Motion (PMMS) Parameters")
        mixed_layout = QFormLayout(self.mixed_motion_group)

        # Enable iterative smoothing
        self.utrack_iterative_smoothing_checkbox = QCheckBox("Enable Iterative Smoothing (Forward-Reverse-Forward)")
        self.utrack_iterative_smoothing_checkbox.setChecked(self.parameters.utrack_enable_iterative_smoothing)
        mixed_layout.addRow(self.utrack_iterative_smoothing_checkbox)

        # Number of tracking rounds
        self.utrack_tracking_rounds_spin = QSpinBox()
        self.utrack_tracking_rounds_spin.setRange(1, 10)
        self.utrack_tracking_rounds_spin.setValue(self.parameters.utrack_num_tracking_rounds)
        mixed_layout.addRow("Tracking Rounds:", self.utrack_tracking_rounds_spin)

        # Motion regime detection sensitivity
        self.utrack_regime_sensitivity_spin = QDoubleSpinBox()
        self.utrack_regime_sensitivity_spin.setRange(0.1, 1.0)
        self.utrack_regime_sensitivity_spin.setDecimals(2)
        self.utrack_regime_sensitivity_spin.setValue(self.parameters.utrack_regime_sensitivity)
        mixed_layout.addRow("Regime Detection Sensitivity:", self.utrack_regime_sensitivity_spin)

        # Adaptive search radius
        self.utrack_adaptive_search_checkbox = QCheckBox("Adaptive Search Radius")
        self.utrack_adaptive_search_checkbox.setChecked(self.parameters.utrack_adaptive_search_radius)
        mixed_layout.addRow(self.utrack_adaptive_search_checkbox)

        # Minimum regime length
        self.utrack_min_regime_spin = QSpinBox()
        self.utrack_min_regime_spin.setRange(2, 20)
        self.utrack_min_regime_spin.setValue(self.parameters.utrack_min_regime_length)
        self.utrack_min_regime_spin.setSuffix(" frames")
        mixed_layout.addRow("Min Regime Length:", self.utrack_min_regime_spin)

        utrack_layout.addWidget(self.mixed_motion_group)

        # Motion transition probabilities
        self.transition_group = QGroupBox("Motion Transition Probabilities")
        transition_layout = QFormLayout(self.transition_group)

        self.brownian_to_linear_spin = QDoubleSpinBox()
        self.brownian_to_linear_spin.setRange(0.01, 0.5)
        self.brownian_to_linear_spin.setDecimals(3)
        self.brownian_to_linear_spin.setValue(self.parameters.utrack_trans_prob_b2l)
        transition_layout.addRow("Brownian → Linear:", self.brownian_to_linear_spin)

        self.linear_to_brownian_spin = QDoubleSpinBox()
        self.linear_to_brownian_spin.setRange(0.01, 0.5)
        self.linear_to_brownian_spin.setDecimals(3)
        self.linear_to_brownian_spin.setValue(self.parameters.utrack_trans_prob_l2b)
        transition_layout.addRow("Linear → Brownian:", self.linear_to_brownian_spin)

        # Advanced motion parameters
        self.brownian_noise_spin = QDoubleSpinBox()
        self.brownian_noise_spin.setRange(0.5, 10.0)
        self.brownian_noise_spin.setDecimals(2)
        self.brownian_noise_spin.setValue(self.parameters.utrack_brownian_noise_mult)
        transition_layout.addRow("Brownian Noise Multiplier:", self.brownian_noise_spin)

        self.velocity_persistence_spin = QDoubleSpinBox()
        self.velocity_persistence_spin.setRange(0.1, 1.0)
        self.velocity_persistence_spin.setDecimals(2)
        self.velocity_persistence_spin.setValue(self.parameters.utrack_linear_velocity_persist)
        transition_layout.addRow("Linear Velocity Persistence:", self.velocity_persistence_spin)

        utrack_layout.addWidget(self.transition_group)

        # Advanced U-Track options
        self.utrack_advanced_group = QGroupBox("Advanced U-Track Options")
        utrack_advanced_layout = QVBoxLayout(self.utrack_advanced_group)

        self.utrack_auto_distance_checkbox = QCheckBox("Auto-adapt linking distance")
        self.utrack_auto_distance_checkbox.setChecked(self.parameters.utrack_auto_linking_distance)
        utrack_advanced_layout.addWidget(self.utrack_auto_distance_checkbox)

        self.utrack_enable_merging_checkbox = QCheckBox("Enable merge detection")
        self.utrack_enable_merging_checkbox.setChecked(self.parameters.utrack_enable_merging)
        utrack_advanced_layout.addWidget(self.utrack_enable_merging_checkbox)

        self.utrack_enable_splitting_checkbox = QCheckBox("Enable split detection")
        self.utrack_enable_splitting_checkbox.setChecked(self.parameters.utrack_enable_splitting)
        utrack_advanced_layout.addWidget(self.utrack_enable_splitting_checkbox)

        utrack_layout.addWidget(self.utrack_advanced_group)
        left_layout.addWidget(self.utrack_linking_group)

        # Method comparison and descriptions (ENHANCED with algorithm information)
        desc_group = QGroupBox("Linking Method and Algorithm Comparison")
        desc_layout = QVBoxLayout(desc_group)

        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(250)
        desc_text.setText("""
    Linking Method Features:

    Built-in Algorithms:
    • Iterative (Recommended): Handles arbitrarily long tracks without recursion errors
      - Memory efficient and scalable to large datasets
      - Produces identical results to recursive method
      - Recommended for all new analyses
    • Recursive (Legacy): Original algorithm from previous versions
      - May fail with "recursion limit exceeded" on complex data
      - Configurable recursion limit (default: 1000)
      - Use only for compatibility with previous analyses

    Trackpy: Python implementation of Crocker & Grier
    • Good for: Dense particles, adaptive search
    • Limitations: Memory intensive for large datasets

    U-Track Mixed Motion: LAP-based with heterogeneous motion
    • Good for: Dense fields, complex motion patterns, rapid motion switches
    • Features:
      - PMMS algorithm for mixed motion (Brownian ↔ Linear ↔ Confined)
      - Iterative smoothing (Forward-Reverse-Forward tracking)
      - Adaptive search radius based on motion type
      - Motion regime detection and classification
      - Handles particle appearance/disappearance
    • Best for: Biological systems with switching motion behaviors

    Performance: Iterative and recursive algorithms have similar speed, but iterative is more reliable.
            """)
        desc_layout.addWidget(desc_text)

        left_layout.addWidget(desc_group)

        # === RIGHT COLUMN CONTENT (unchanged) ===

        # Feature parameters (unchanged)
        feature_group = QGroupBox("Feature Parameters")
        feature_layout = QFormLayout(feature_group)

        self.rg_threshold_spin = QDoubleSpinBox()
        self.rg_threshold_spin.setRange(0.1, 10.0)
        self.rg_threshold_spin.setValue(self.parameters.rg_mobility_threshold)
        feature_layout.addRow("RG Mobility Threshold:", self.rg_threshold_spin)

        right_layout.addWidget(feature_group)

        # Classification parameters (unchanged)
        class_group = QGroupBox("Classification Parameters")
        class_layout = QVBoxLayout(class_group)

        # Training data path
        train_layout = QHBoxLayout()
        self.train_path_edit = QLabel(self.parameters.training_data_path)
        self.train_path_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.train_path_edit.setWordWrap(True)
        train_layout.addWidget(self.train_path_edit)

        self.select_train_btn = QPushButton("Browse...")
        self.select_train_btn.clicked.connect(self.select_training_data)
        train_layout.addWidget(self.select_train_btn)

        self.reset_train_btn = QPushButton("Reset to Default")
        self.reset_train_btn.clicked.connect(self.reset_training_data_path)
        train_layout.addWidget(self.reset_train_btn)

        class_layout.addLayout(train_layout)

        # Training data status
        self.train_status_label = QLabel()
        self.update_training_data_status()
        class_layout.addWidget(self.train_status_label)

        # Experiment name section (unchanged)
        exp_group = QGroupBox("Experiment Name Configuration")
        exp_layout = QVBoxLayout(exp_group)

        # Auto-detect option
        self.auto_detect_checkbox = QCheckBox("Auto-detect experiment names from subfolder names")
        self.auto_detect_checkbox.setChecked(self.parameters.auto_detect_experiment_names)
        self.auto_detect_checkbox.toggled.connect(self.on_auto_detect_toggled)
        exp_layout.addWidget(self.auto_detect_checkbox)

        # Manual experiment name
        manual_layout = QHBoxLayout()
        self.manual_exp_label = QLabel("Manual experiment name:")
        manual_layout.addWidget(self.manual_exp_label)

        self.experiment_name_edit = QComboBox()
        self.experiment_name_edit.setEditable(True)
        self.experiment_name_edit.addItems(['Control', 'Treatment', 'Drug_A', 'Drug_B', 'DMSO', 'Compound_X'])
        self.experiment_name_edit.setCurrentText(self.parameters.experiment_name)
        manual_layout.addWidget(self.experiment_name_edit)

        exp_layout.addLayout(manual_layout)

        # Status label for auto-detection
        self.exp_status_label = QLabel()
        exp_layout.addWidget(self.exp_status_label)

        self.on_auto_detect_toggled()

        class_layout.addWidget(exp_group)
        right_layout.addWidget(class_group)

        # Add stretch to both columns
        left_layout.addStretch()
        right_layout.addStretch()

        # Configure scroll areas
        left_scroll.setWidget(left_widget)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_scroll.setWidget(right_widget)
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Add columns to main layout
        scroll_main_layout.addWidget(left_scroll, 1)
        scroll_main_layout.addWidget(right_scroll, 1)

        # Configure main scroll area
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        # Update UI state based on current selection
        self.on_linking_method_changed()
        self.on_trackpy_type_changed()
        self.on_motion_model_changed()
        self.on_algorithm_changed()  # NEW: Initialize algorithm-specific UI state

        self.tab_widget.addTab(tab, "Parameters")

    def on_algorithm_changed(self):
        """NEW: Handle algorithm selection changes"""
        if hasattr(self, 'recursive_radio') and hasattr(self, 'recursion_limit_spin'):
            is_recursive = self.recursive_radio.isChecked()
            self.recursion_limit_spin.setEnabled(is_recursive)


    def setup_logging_for_analysis(self, file_path):
        """Setup comprehensive file logging for the current analysis"""
        try:
            # Get directory of the file being analyzed
            analysis_dir = os.path.dirname(file_path)
            self.current_analysis_dir = analysis_dir

            # Create log filename with file-specific identifier
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{base_name}_analysis_log_{timestamp}.log"

            # Setup file logging
            log_file = self.file_logger.setup_file_logging(analysis_dir, log_filename)

            # Log the setup and analysis parameters
            self.log_message(f"File logging initialized: {os.path.basename(log_file)}")
            self.file_logger.log('info', f"Starting analysis for file: {os.path.basename(file_path)}")
            self.file_logger.log('info', f"Analysis directory: {analysis_dir}")

            # Log all parameters
            params_dict = self.parameters.to_dict()
            self.file_logger.log('info', f"Analysis parameters: {json.dumps(params_dict, indent=2)}")

            return log_file

        except Exception as e:
            print(f"Error setting up file logging: {e}")
            return None

    def log_message(self, message):
        """Enhanced log message that logs to both GUI and file"""
        # Original GUI logging
        self.log_text.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        QApplication.processEvents()

        # File logging with enhanced processing
        if self.file_logger and self.file_logger.logger:
            # Clean message for file (remove formatting characters)
            clean_message = message
            for char in ['🚀', '📁', '🏷️', '📈', '🔄', '📊', '🔗', '✅', '❌', '🎉', '⚠️']:
                clean_message = clean_message.replace(char, '').strip()

            # Remove timestamp from beginning if present
            if clean_message.startswith('[') and ']' in clean_message:
                clean_message = clean_message.split(']', 1)[1].strip()

            # Determine log level based on message content
            if any(indicator in message.lower() for indicator in ['error', '❌', 'failed', 'exception']):
                self.file_logger.log('error', clean_message)
            elif any(indicator in message.lower() for indicator in ['warning', '⚠️', 'warn']):
                self.file_logger.log('warning', clean_message)
            elif any(indicator in message.lower() for indicator in ['✅', 'success', 'complete', 'finished']):
                self.file_logger.log('info', clean_message)
            elif any(indicator in message.lower() for indicator in ['debug', 'details']):
                self.file_logger.log('debug', clean_message)
            else:
                self.file_logger.log('info', clean_message)

    def link_particles_builtin(self, txy_pts, file_path):
        """Enhanced built-in linking method with algorithm selection"""
        try:
            # Extract just frame, x, y for linking
            if txy_pts.shape[1] == 4:
                linking_data = txy_pts[:, :3]
                point_ids = txy_pts[:, 3]
            else:
                linking_data = txy_pts
                point_ids = np.arange(len(txy_pts))

            # Log linking details
            algorithm = self.parameters.builtin_linking_algorithm
            self.file_logger.log('info', f"Using built-in linking with {algorithm} algorithm")
            self.file_logger.log_data_summary("linking_data", linking_data)
            self.file_logger.log('debug', f"Linking parameters: max_gap={self.parameters.max_gap_frames}, "
                               f"max_dist={self.parameters.max_link_distance}")

            if algorithm == 'recursive':
                self.file_logger.log('debug', f"Recursion limit: {self.parameters.recursive_depth_limit}")

            points = Points(linking_data)

            # Call link_pts with the selected method
            points.link_pts(
                self.parameters.max_gap_frames,
                self.parameters.max_link_distance,
                method=algorithm,
                depth_limit=self.parameters.recursive_depth_limit
            )

            # Store the IDs for later use
            points.point_ids = point_ids

            # Log linking statistics
            if hasattr(points, 'linking_stats'):
                self.file_logger.log('info', f"Linking statistics: {json.dumps(points.linking_stats, indent=2)}")

            # Load image for intensity extraction
            if os.path.exists(file_path):
                try:
                    self.file_logger.log('debug', f"Loading image for intensity extraction: {file_path}")
                    A = skio.imread(file_path, plugin='tifffile')
                    self.file_logger.log_data_summary("image_array", A)

                    start_time = time.time()
                    points.getIntensities(A)
                    intensity_time = time.time() - start_time

                    self.file_logger.log_performance("intensity_extraction", intensity_time,
                                                   f"extracted {len(points.intensities)} values")

                except Exception as e:
                    self.file_logger.log_error(f"Error during intensity extraction", e)
                    points.intensities = [0.0] * len(points.txy_pts)

            self.log_message(f"    Built-in linking complete: {len(points.tracks)} tracks ({algorithm} method)")
            return points

        except Exception as e:
            self.file_logger.log_error(f"Error in built-in linking", e,
                                     context={"algorithm": self.parameters.builtin_linking_algorithm})
            return None


    def on_motion_model_changed(self):
        """Handle motion model selection changes"""
        current_model = self.motion_model_combo.currentText()

        # Show/hide mixed motion parameters based on selection
        if current_model == 'mixed':
            self.mixed_motion_group.setVisible(True)
            self.transition_group.setVisible(True)
        else:
            self.mixed_motion_group.setVisible(False)
            self.transition_group.setVisible(False)

    def on_linking_method_changed(self):
        """Handle linking method change (unchanged but updated for new algorithm options)"""
        if self.builtin_linking_radio.isChecked():
            self.builtin_linking_group.setEnabled(True)
            self.trackpy_linking_group.setEnabled(False)
            if hasattr(self, 'utrack_linking_group'):
                self.utrack_linking_group.setEnabled(False)
        elif self.trackpy_linking_radio.isChecked():
            self.builtin_linking_group.setEnabled(False)
            self.trackpy_linking_group.setEnabled(True)
            if hasattr(self, 'utrack_linking_group'):
                self.utrack_linking_group.setEnabled(False)
        else:  # U-Track selected
            self.builtin_linking_group.setEnabled(False)
            self.trackpy_linking_group.setEnabled(False)
            if hasattr(self, 'utrack_linking_group'):
                self.utrack_linking_group.setEnabled(True)
                # Update mixed motion visibility based on current model
                self.on_motion_model_changed()

    def on_full_interpolation_toggled(self):
        """Handle full interpolation checkbox toggle"""
        enabled = self.full_interpolation_checkbox.isChecked()
        self.separate_interpolated_file_checkbox.setEnabled(enabled)
        self.extend_to_full_recording_checkbox.setEnabled(enabled)
        if not enabled:
            self.separate_interpolated_file_checkbox.setChecked(False)
            self.extend_to_full_recording_checkbox.setChecked(False)

    def create_analysis_steps_tab(self):
        """Create analysis steps tab with scrollable content"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Create scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        steps_group = QGroupBox("Core Analysis Steps")
        steps_layout = QVBoxLayout(steps_group)

        # Create checkboxes for analysis steps (more compact)
        checkboxes = [
            (self, 'nn_checkbox', "Nearest Neighbor Analysis", 'enable_nearest_neighbors'),
            (self, 'svm_checkbox', "SVM Classification", 'enable_svm_classification'),
            (self, 'velocity_checkbox', "Velocity Analysis", 'enable_velocity_analysis'),
            (self, 'diffusion_checkbox', "Diffusion Analysis", 'enable_diffusion_analysis'),
            (self, 'loc_error_checkbox', "Localization Error Analysis", 'enable_localization_error'),
            (self, 'straightness_checkbox', "Straightness Analysis", 'enable_straightness_analysis'),
            (self, 'missing_points_checkbox', "Missing Points Integration", 'enable_missing_points_integration'),
        ]

        for obj, attr_name, label, param_name in checkboxes:
            checkbox = QCheckBox(label)
            checkbox.setChecked(getattr(self.parameters, param_name))
            setattr(obj, attr_name, checkbox)
            steps_layout.addWidget(checkbox)

        # Background subtraction with frame-specific option (compact)
        self.bg_checkbox = QCheckBox("Background Subtraction (ROI-based)")
        self.bg_checkbox.setChecked(self.parameters.enable_background_subtraction)
        self.bg_checkbox.toggled.connect(self.on_bg_subtraction_toggled)
        steps_layout.addWidget(self.bg_checkbox)

        # Frame-specific background option (indented, compact)
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("    "))
        self.bg_frame_specific_checkbox = QCheckBox("Use frame-specific background")
        self.bg_frame_specific_checkbox.setChecked(self.parameters.roi_frame_specific_background)
        self.bg_frame_specific_checkbox.setEnabled(self.parameters.enable_background_subtraction)
        frame_layout.addWidget(self.bg_frame_specific_checkbox)
        frame_layout.addStretch()
        steps_layout.addLayout(frame_layout)

        scroll_layout.addWidget(steps_group)

        # Enhanced analysis steps (compact)
        enhanced_group = QGroupBox("Enhanced Analysis Steps")
        enhanced_layout = QVBoxLayout(enhanced_group)

        enhanced_checkboxes = [
            (self, 'direction_checkbox', "Direction of Travel Analysis", 'enable_direction_analysis'),
            (self, 'distance_diff_checkbox', "Distance Differential Analysis", 'enable_distance_differential'),
            (self, 'interpolation_checkbox', "Enhanced Interpolation (Advanced)", 'enable_enhanced_interpolation'),
            (self, 'full_interpolation_checkbox', "Full Track Interpolation (All Tracks)", 'enable_full_track_interpolation'),  # NEW: Add this line
        ]

        for obj, attr_name, label, param_name in enhanced_checkboxes:
            checkbox = QCheckBox(label)
            checkbox.setChecked(getattr(self.parameters, param_name))
            setattr(obj, attr_name, checkbox)
            enhanced_layout.addWidget(checkbox)

        # Add separate interpolated file option (indented, depends on full interpolation)
        separate_file_layout = QHBoxLayout()
        separate_file_layout.addWidget(QLabel("    "))
        self.separate_interpolated_file_checkbox = QCheckBox("Save separate file with interpolated results")
        self.separate_interpolated_file_checkbox.setChecked(self.parameters.save_separate_interpolated_file)
        self.separate_interpolated_file_checkbox.setEnabled(self.parameters.enable_full_track_interpolation)
        separate_file_layout.addWidget(self.separate_interpolated_file_checkbox)
        separate_file_layout.addStretch()
        enhanced_layout.addLayout(separate_file_layout)

        # Add extend to full recording option
        extend_full_layout = QHBoxLayout()
        extend_full_layout.addWidget(QLabel("    "))
        self.extend_to_full_recording_checkbox = QCheckBox("Extend interpolation to entire recording (frame 0 to final frame)")
        self.extend_to_full_recording_checkbox.setChecked(self.parameters.extend_interpolation_to_full_recording)
        self.extend_to_full_recording_checkbox.setEnabled(self.parameters.enable_full_track_interpolation)
        extend_full_layout.addWidget(self.extend_to_full_recording_checkbox)
        extend_full_layout.addStretch()
        enhanced_layout.addLayout(extend_full_layout)

        # Connect the full interpolation checkbox to enable/disable the separate file option
        self.full_interpolation_checkbox.toggled.connect(self.on_full_interpolation_toggled)


        scroll_layout.addWidget(enhanced_group)

        # Compact description
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(100)  # Reduced from 150
        desc_text.setText("""Core features (radius of gyration, fractal dimension, net displacement) are always calculated.
        Background Subtraction: Frame-specific calculates separate ROI background per frame; Single mean uses one average.
        Full Track Interpolation: Fills missing frames for all tracks with interpolated positions.
          - When "Save separate file" is enabled, creates both original and interpolated result files.
        Configure Geometric analysis and Autocorrelation in their respective tabs.""")
        scroll_layout.addWidget(desc_text)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Analysis Steps")

    def create_geometric_analysis_tab(self):
        """Create geometric analysis configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Geometric Analysis Enable/Disable
        enable_group = QGroupBox("Enable Geometric Analysis")
        enable_layout = QVBoxLayout(enable_group)

        self.geometric_enable_checkbox = QCheckBox("Enable Geometric Rg and sRg Calculation")
        self.geometric_enable_checkbox.setChecked(self.parameters.enable_geometric_analysis)
        self.geometric_enable_checkbox.toggled.connect(self.on_geometric_enable_toggled)
        enable_layout.addWidget(self.geometric_enable_checkbox)

        info_label = QLabel("Adds columns: Rg_geometric, sRg_geometric, mean_step_length_geometric, geometric_mobility_classification")
        info_label.setStyleSheet("color: #666; font-style: italic; margin-left: 20px;")
        enable_layout.addWidget(info_label)

        scroll_layout.addWidget(enable_group)

        # Geometric Analysis Parameters
        self.geometric_params_group = QGroupBox("Geometric Analysis Parameters")
        geometric_layout = QFormLayout(self.geometric_params_group)

        # Method selection
        method_layout = QHBoxLayout()
        self.geometric_method_simple = QCheckBox("Simple Geometric Method")
        self.geometric_method_simple.setChecked(self.parameters.geometric_rg_method == 'simple')
        method_layout.addWidget(self.geometric_method_simple)

        method_info = QLabel("(Uses direct geometric calculation - faster, less noise-sensitive)")
        method_info.setStyleSheet("color: #666; font-style: italic;")
        method_layout.addWidget(method_info)

        geometric_layout.addRow("Calculation Method:", method_layout)

        # sRg cutoff
        self.geometric_srg_cutoff_spin = QDoubleSpinBox()
        self.geometric_srg_cutoff_spin.setRange(0.1, 10.0)
        self.geometric_srg_cutoff_spin.setDecimals(6)
        self.geometric_srg_cutoff_spin.setValue(self.parameters.geometric_srg_cutoff)
        geometric_layout.addRow("sRg Mobility Threshold:", self.geometric_srg_cutoff_spin)

        # Minimum trajectory length
        self.geometric_cutoff_length_spin = QSpinBox()
        self.geometric_cutoff_length_spin.setRange(2, 100)
        self.geometric_cutoff_length_spin.setValue(self.parameters.geometric_cutoff_length)
        geometric_layout.addRow("Min Trajectory Length:", self.geometric_cutoff_length_spin)

        scroll_layout.addWidget(self.geometric_params_group)

        # Linear Classification Parameters
        self.linear_class_group = QGroupBox("Simple Linear Classification")
        linear_layout = QVBoxLayout(self.linear_class_group)

        self.geometric_linear_enable_checkbox = QCheckBox("Enable Simple Linear Classification")
        self.geometric_linear_enable_checkbox.setChecked(self.parameters.enable_geometric_linear_classification)
        linear_layout.addWidget(self.geometric_linear_enable_checkbox)

        linear_info = QLabel("Adds columns: geometric_linear_classification, geometric_directionality_ratio, geometric_perpendicular_distance")
        linear_info.setStyleSheet("color: #666; font-style: italic; margin-left: 20px;")
        linear_layout.addWidget(linear_info)

        # Linear classification parameters
        linear_params_layout = QFormLayout()

        self.geometric_directionality_spin = QDoubleSpinBox()
        self.geometric_directionality_spin.setRange(0.0, 1.0)
        self.geometric_directionality_spin.setDecimals(3)
        self.geometric_directionality_spin.setValue(self.parameters.geometric_directionality_threshold)
        linear_params_layout.addRow("Directionality Threshold:", self.geometric_directionality_spin)

        self.geometric_perpendicular_spin = QDoubleSpinBox()
        self.geometric_perpendicular_spin.setRange(0.0, 1.0)
        self.geometric_perpendicular_spin.setDecimals(3)
        self.geometric_perpendicular_spin.setValue(self.parameters.geometric_perpendicular_threshold)
        linear_params_layout.addRow("Perpendicular Distance Threshold:", self.geometric_perpendicular_spin)

        linear_layout.addLayout(linear_params_layout)

        scroll_layout.addWidget(self.linear_class_group)

        # Description
        desc_group = QGroupBox("Method Description")
        desc_layout = QVBoxLayout(desc_group)

        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(200)
        desc_text.setText("""
Geometric Analysis Methods:

Geometric Rg Calculation:
• Simple method uses direct geometric formula: Rg = sqrt(sum(avg_pos² - avg²))
• Faster computation, less sensitive to noise compared to tensor method
• Based on methods from Golan & Sherman Nature Communications 2017

Scaled Rg (sRg):
• sRg = sqrt(π/2) × Rg / mean_step_length
• Values > threshold indicate mobile particles
• Scale-invariant measure for comparing different experiments

Simple Linear Classification:
• Directionality Ratio: net displacement / total path length
• Perpendicular Distance: average distance from straight line connecting start to end
• Classifications: linear_unidirectional, linear_bidirectional, non_linear
• Geometric approach requires no eigenvalue decomposition

These methods provide an alternative to the tensor-based calculations and can be useful for:
- Quick analysis with reduced computational requirements
- Comparison with published geometric methods
- Validation of tensor-based results
        """)
        desc_layout.addWidget(desc_text)

        scroll_layout.addWidget(desc_group)

        # Enable/disable based on initial state
        self.on_geometric_enable_toggled()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Geometric Analysis")

    def create_autocorrelation_tab(self):
        """Create autocorrelation analysis configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Enable/Disable Autocorrelation
        enable_group = QGroupBox("Enable Autocorrelation Analysis")
        enable_layout = QVBoxLayout(enable_group)

        self.autocorr_enable_checkbox = QCheckBox("Enable Directional Autocorrelation Analysis")
        self.autocorr_enable_checkbox.setChecked(self.parameters.enable_autocorrelation_analysis)
        self.autocorr_enable_checkbox.toggled.connect(self.on_autocorr_enable_toggled)
        enable_layout.addWidget(self.autocorr_enable_checkbox)

        info_label = QLabel("Performs directional persistence analysis following Gorelik & Gautreau (2014)")
        info_label.setStyleSheet("color: #666; font-style: italic; margin-left: 20px;")
        enable_layout.addWidget(info_label)

        scroll_layout.addWidget(enable_group)

        # Autocorrelation Parameters
        self.autocorr_params_group = QGroupBox("Autocorrelation Parameters")
        params_layout = QFormLayout(self.autocorr_params_group)

        # Time interval
        self.autocorr_time_interval_spin = QDoubleSpinBox()
        self.autocorr_time_interval_spin.setRange(0.001, 1000.0)
        self.autocorr_time_interval_spin.setDecimals(3)
        self.autocorr_time_interval_spin.setValue(self.parameters.autocorr_time_interval)
        self.autocorr_time_interval_spin.setSuffix(" units")
        params_layout.addRow("Time Interval Between Frames:", self.autocorr_time_interval_spin)

        # Number of intervals
        self.autocorr_num_intervals_spin = QSpinBox()
        self.autocorr_num_intervals_spin.setRange(5, 100000)
        self.autocorr_num_intervals_spin.setValue(self.parameters.autocorr_num_intervals)
        params_layout.addRow("Number of Time Intervals:", self.autocorr_num_intervals_spin)

        # Minimum track length
        self.autocorr_min_length_spin = QSpinBox()
        self.autocorr_min_length_spin.setRange(3, 10000)
        self.autocorr_min_length_spin.setValue(self.parameters.autocorr_min_track_length)
        params_layout.addRow("Minimum Track Length:", self.autocorr_min_length_spin)

        scroll_layout.addWidget(self.autocorr_params_group)

        # Visualization Options
        self.autocorr_viz_group = QGroupBox("Visualization Options")
        viz_layout = QVBoxLayout(self.autocorr_viz_group)

        self.autocorr_show_individual_checkbox = QCheckBox("Show Individual Tracks in Plots")
        self.autocorr_show_individual_checkbox.setChecked(self.parameters.autocorr_show_individual_tracks)
        viz_layout.addWidget(self.autocorr_show_individual_checkbox)

        # Max tracks to plot
        max_tracks_layout = QFormLayout()
        self.autocorr_max_tracks_spin = QSpinBox()
        self.autocorr_max_tracks_spin.setRange(10, 1000)
        self.autocorr_max_tracks_spin.setValue(self.parameters.autocorr_max_tracks_plot)
        max_tracks_layout.addRow("Max Tracks to Plot:", self.autocorr_max_tracks_spin)
        viz_layout.addLayout(max_tracks_layout)

        scroll_layout.addWidget(self.autocorr_viz_group)

        # Output Options
        self.autocorr_output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(self.autocorr_output_group)

        self.autocorr_save_plots_checkbox = QCheckBox("Save Autocorrelation Plots")
        self.autocorr_save_plots_checkbox.setChecked(self.parameters.autocorr_save_plots)
        output_layout.addWidget(self.autocorr_save_plots_checkbox)

        self.autocorr_save_data_checkbox = QCheckBox("Save Autocorrelation Data Files")
        self.autocorr_save_data_checkbox.setChecked(self.parameters.autocorr_save_data)
        output_layout.addWidget(self.autocorr_save_data_checkbox)

        scroll_layout.addWidget(self.autocorr_output_group)

        # Live Analysis Section
        self.autocorr_live_group = QGroupBox("Live Autocorrelation Analysis")
        live_layout = QVBoxLayout(self.autocorr_live_group)

        # Info label
        live_info = QLabel("Analyze current tracks data loaded from previous analysis:")
        live_info.setStyleSheet("color: #333; font-weight: bold;")
        live_layout.addWidget(live_info)

        # Control buttons
        buttons_layout = QHBoxLayout()

        self.load_tracks_btn = QPushButton("Load Tracks Data")
        self.load_tracks_btn.clicked.connect(self.load_tracks_data)
        buttons_layout.addWidget(self.load_tracks_btn)

        self.run_autocorr_btn = QPushButton("Run Autocorrelation Analysis")
        self.run_autocorr_btn.clicked.connect(self.run_autocorrelation_analysis)
        self.run_autocorr_btn.setEnabled(False)
        buttons_layout.addWidget(self.run_autocorr_btn)

        live_layout.addLayout(buttons_layout)

        # Status label
        self.autocorr_status_label = QLabel("No tracks data loaded")
        self.autocorr_status_label.setStyleSheet("color: #666; font-style: italic;")
        live_layout.addWidget(self.autocorr_status_label)

        # Progress bar for autocorrelation
        self.autocorr_progress_bar = QProgressBar()
        self.autocorr_progress_bar.setVisible(False)
        live_layout.addWidget(self.autocorr_progress_bar)

        scroll_layout.addWidget(self.autocorr_live_group)

        # Plot Widget
        self.autocorr_plot_group = QGroupBox("Autocorrelation Results")
        plot_layout = QVBoxLayout(self.autocorr_plot_group)

        self.autocorr_plot_widget = AutocorrelationPlotWidget()
        plot_layout.addWidget(self.autocorr_plot_widget)

        # Plot controls
        plot_controls_layout = QHBoxLayout()

        self.show_individual_btn = QCheckBox("Show Individual Tracks")
        self.show_individual_btn.setChecked(True)
        self.show_individual_btn.toggled.connect(self.update_autocorr_plot)
        plot_controls_layout.addWidget(self.show_individual_btn)

        self.save_plot_btn = QPushButton("Save Current Plot")
        self.save_plot_btn.clicked.connect(self.save_autocorr_plot)
        self.save_plot_btn.setEnabled(False)
        plot_controls_layout.addWidget(self.save_plot_btn)

        plot_controls_layout.addStretch()
        plot_layout.addLayout(plot_controls_layout)

        scroll_layout.addWidget(self.autocorr_plot_group)

        # Description
        desc_group = QGroupBox("Method Description")
        desc_layout = QVBoxLayout(desc_group)

        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(150)
        desc_text.setText("""
Directional Autocorrelation Analysis (Gorelik & Gautreau, 2014):

This analysis quantifies directional persistence in cell migration by calculating the autocorrelation
of movement direction vectors over time. The method:

1. Calculates normalized direction vectors between consecutive positions
2. Computes autocorrelation C(τ) = ⟨v̂(t) · v̂(t+τ)⟩ for different time lags τ
3. Generates decay curves showing how directional memory is lost over time
4. Enables quantitative comparison of persistence between conditions

Key outputs:
- Autocorrelation decay curves with statistical error bars
- Individual track correlations for detailed analysis
- Persistence time estimates from exponential decay fitting
- Quantitative metrics for comparing migration behaviors

Applications: Ideal for studying cell migration, particle tracking, and any system where
directional persistence is important for understanding underlying mechanisms.
        """)
        desc_layout.addWidget(desc_text)

        scroll_layout.addWidget(desc_group)

        # Enable/disable based on initial state
        self.on_autocorr_enable_toggled()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Autocorrelation")

    def create_progress_tab(self):
        """Create progress tab with compact layout"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Progress section (more compact)
        progress_group = QGroupBox("Analysis Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start analysis")
        progress_layout.addWidget(self.status_label)

        layout.addWidget(progress_group)

        # Log section (takes remaining space)
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))  # Smaller font
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_group)

        self.tab_widget.addTab(tab, "Progress")

    def create_export_control_tab(self):
        """Create export control tab for column selection and metadata export"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Export Control Header
        header_group = QGroupBox("Export Column Selection")
        header_layout = QVBoxLayout(header_group)

        info_label = QLabel("Select which columns to include in exported CSV files. Unavailable options are grayed out based on your analysis settings.")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        info_label.setWordWrap(True)
        header_layout.addWidget(info_label)

        # Control buttons
        control_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("Select All Available")
        self.select_all_btn.clicked.connect(self.select_all_export_columns)
        control_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("Select None")
        self.select_none_btn.clicked.connect(self.select_no_export_columns)
        control_layout.addWidget(self.select_none_btn)

        self.select_minimal_btn = QPushButton("Select Minimal Set")
        self.select_minimal_btn.clicked.connect(self.select_minimal_export_columns)
        control_layout.addWidget(self.select_minimal_btn)

        control_layout.addStretch()
        header_layout.addLayout(control_layout)

        scroll_layout.addWidget(header_group)

        # Column selection organized by category
        self.export_column_checkboxes = {}

        # Define column categories and their columns
        self.column_categories = {
            'Basic Tracking Data': {
                'columns': ['track_number', 'frame', 'x', 'y', 'intensity', 'id', 'n_segments', 'is_interpolated'],
                'always_available': True,
                'description': 'Essential tracking information always included'
            },
            'Core Features': {
                'columns': ['radius_gyration', 'asymmetry', 'skewness', 'kurtosis', 'fracDimension',
                           'netDispl', 'efficiency', 'Straight', 'track_intensity_mean', 'track_intensity_std'],
                'always_available': True,
                'description': 'Core geometric and statistical features'
            },
            'Lag/Displacement Features': {
                'columns': ['lag', 'meanLag', 'track_length', 'radius_gyration_scaled',
                           'radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength',
                           'radius_gyration_ratio_to_mean_step_size', 'radius_gyration_mobility_threshold'],
                'always_available': True,
                'description': 'Step-by-step displacement analysis'
            },
            'Velocity Analysis': {
                'columns': ['zeroed_X', 'zeroed_Y', 'distanceFromOrigin', 'lagNumber', 'velocity',
                           'meanVelocity', 'direction_Relative_To_Origin', 'dt'],
                'depends_on': 'enable_velocity_analysis',
                'description': 'Movement velocity and direction relative to origin'
            },
            'Diffusion Analysis': {
                'columns': ['d_squared', 'lag_squared'],
                'depends_on': 'enable_diffusion_analysis',
                'description': 'Squared displacement measurements for diffusion analysis'
            },
            'Localization Error': {
                'columns': ['mean_X', 'mean_Y', 'distanceFromMean', 'meanLocDistanceFromCenter'],
                'depends_on': 'enable_localization_error',
                'description': 'Localization precision and track center analysis'
            },
            'Direction Analysis': {
                'columns': ['direction_degrees', 'direction_radians', 'direction_x', 'direction_y',
                           'directional_autocorrelation', 'directional_persistence'],
                'depends_on': 'enable_direction_analysis',
                'description': 'Directional movement analysis and persistence'
            },
            'Distance Differential': {
                'columns': ['dy-dt_distance'],
                'depends_on': 'enable_distance_differential',
                'description': 'Rate of change of distance from origin'
            },
            'SVM Classification': {
                'columns': ['SVM', 'Experiment'],
                'depends_on': 'enable_svm_classification',
                'description': 'Machine learning mobility classification results'
            },
            'Background Subtraction': {
                'columns': ['roi_intensity', 'camera_black_estimate', 'intensity_bg_subtracted',
                           'background_subtracted', 'background_method', 'background_signal_used'],
                'depends_on': 'enable_background_subtraction',
                'description': 'ROI-based background correction data'
            },
            'Geometric Analysis': {
                'columns': ['Rg_geometric', 'sRg_geometric', 'mean_step_length_geometric',
                           'geometric_mobility_classification', 'geometric_linear_classification',
                           'geometric_directionality_ratio', 'geometric_mean_perpendicular_distance',
                           'geometric_normalized_perpendicular_distance'],
                'depends_on': 'enable_geometric_analysis',
                'description': 'Alternative geometric calculation methods'
            },
            'Nearest Neighbors': {
                'columns': ['nnDist'] + [f'nnCountInFrame_within_{r}_pixels' for r in [3, 5, 10, 20, 30]],
                'depends_on': 'enable_nearest_neighbors',
                'description': 'Spatial neighbor analysis within frames'
            }
        }

        # Create checkboxes for each category
        for category, info in self.column_categories.items():
            category_group = QGroupBox(category)
            category_layout = QVBoxLayout(category_group)

            # Description
            desc_label = QLabel(info['description'])
            desc_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
            desc_label.setWordWrap(True)
            category_layout.addWidget(desc_label)

            # Column checkboxes in a grid
            columns_widget = QWidget()
            columns_layout = QGridLayout(columns_widget)
            columns_layout.setSpacing(5)

            row, col = 0, 0
            for column in info['columns']:
                checkbox = QCheckBox(column)
                checkbox.setChecked(True)  # Default to selected
                self.export_column_checkboxes[column] = checkbox
                columns_layout.addWidget(checkbox, row, col)

                col += 1
                if col >= 3:  # 3 columns per row
                    col = 0
                    row += 1

            category_layout.addWidget(columns_widget)
            scroll_layout.addWidget(category_group)

        # File Export Options
        file_options_group = QGroupBox("File Export Options")
        file_options_layout = QVBoxLayout(file_options_group)

        self.export_enhanced_checkbox = QCheckBox("Export Enhanced Analysis File (all selected columns)")
        self.export_enhanced_checkbox.setChecked(True)
        file_options_layout.addWidget(self.export_enhanced_checkbox)

        self.export_tracks_checkbox = QCheckBox("Export Basic Tracks File (essential columns only)")
        self.export_tracks_checkbox.setChecked(True)
        file_options_layout.addWidget(self.export_tracks_checkbox)

        self.export_features_checkbox = QCheckBox("Export Features File (feature columns only)")
        self.export_features_checkbox.setChecked(True)
        file_options_layout.addWidget(self.export_features_checkbox)

        scroll_layout.addWidget(file_options_group)

        # Metadata Export
        metadata_group = QGroupBox("Metadata Export")
        metadata_layout = QVBoxLayout(metadata_group)

        self.export_metadata_checkbox = QCheckBox("Export analysis parameters as metadata file")
        self.export_metadata_checkbox.setChecked(True)
        metadata_layout.addWidget(self.export_metadata_checkbox)

        metadata_info = QLabel("Saves all analysis parameters, column selections, and settings used for this analysis as a JSON file.")
        metadata_info.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        metadata_info.setWordWrap(True)
        metadata_layout.addWidget(metadata_info)

        # Metadata preview
        self.metadata_preview_btn = QPushButton("Preview Metadata")
        self.metadata_preview_btn.clicked.connect(self.preview_metadata)
        metadata_layout.addWidget(self.metadata_preview_btn)

        scroll_layout.addWidget(metadata_group)

        # Column Count Summary
        self.column_summary_group = QGroupBox("Selection Summary")
        summary_layout = QVBoxLayout(self.column_summary_group)

        self.column_count_label = QLabel("Columns selected: 0")
        self.column_count_label.setStyleSheet("font-weight: bold;")
        summary_layout.addWidget(self.column_count_label)

        self.update_summary_btn = QPushButton("Update Summary")
        self.update_summary_btn.clicked.connect(self.update_column_summary)
        summary_layout.addWidget(self.update_summary_btn)

        scroll_layout.addWidget(self.column_summary_group)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Export Control")

        # Store reference to scroll widget for updates
        self.export_control_scroll_widget = scroll_widget

    def update_export_control_availability(self):
        """Update availability of export columns based on current analysis settings"""
        self.update_parameters()  # Get current parameters

        for category, info in self.column_categories.items():
            category_available = True

            # Check if category depends on a setting
            if 'depends_on' in info:
                depends_on = info['depends_on']
                if hasattr(self.parameters, depends_on):
                    category_available = getattr(self.parameters, depends_on)
                else:
                    category_available = False
            elif 'always_available' not in info:
                category_available = True

            # Update checkbox states
            for column in info['columns']:
                if column in self.export_column_checkboxes:
                    checkbox = self.export_column_checkboxes[column]
                    checkbox.setEnabled(category_available)
                    if not category_available:
                        checkbox.setStyleSheet("color: #888;")
                        checkbox.setToolTip(f"Not available - requires {info.get('depends_on', 'unknown setting')}")
                    else:
                        checkbox.setStyleSheet("")
                        checkbox.setToolTip("")

    def select_all_export_columns(self):
        """Select all available export columns"""
        for column, checkbox in self.export_column_checkboxes.items():
            if checkbox.isEnabled():
                checkbox.setChecked(True)
        self.update_column_summary()

    def select_no_export_columns(self):
        """Deselect all export columns"""
        for checkbox in self.export_column_checkboxes.values():
            checkbox.setChecked(False)
        self.update_column_summary()

    def select_minimal_export_columns(self):
        """Select only essential columns"""
        # First deselect all
        self.select_no_export_columns()

        # Select minimal essential set
        minimal_columns = ['track_number', 'frame', 'x', 'y', 'intensity', 'n_segments',
                          'radius_gyration', 'netDispl', 'efficiency', 'SVM', 'Experiment']

        for column in minimal_columns:
            if column in self.export_column_checkboxes:
                checkbox = self.export_column_checkboxes[column]
                if checkbox.isEnabled():
                    checkbox.setChecked(True)

        self.update_column_summary()

    def update_column_summary(self):
        """Update the column selection summary"""
        selected_count = sum(1 for cb in self.export_column_checkboxes.values() if cb.isChecked())
        available_count = sum(1 for cb in self.export_column_checkboxes.values() if cb.isEnabled())
        total_count = len(self.export_column_checkboxes)

        summary_text = f"Columns selected: {selected_count} / {available_count} available ({total_count} total)"
        self.column_count_label.setText(summary_text)

    def get_selected_export_columns(self):
        """Get list of selected column names for export"""
        return [column for column, checkbox in self.export_column_checkboxes.items()
                if checkbox.isChecked() and checkbox.isEnabled()]

    def preview_metadata(self):
        """Preview the metadata that would be exported"""
        self.update_parameters()

        metadata = self.generate_export_metadata()

        # Create preview dialog
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

        dialog = QDialog(self)
        dialog.setWindowTitle("Metadata Preview")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 9))
        text_edit.setText(json.dumps(metadata, indent=2))
        layout.addWidget(text_edit)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec_()

    def generate_export_metadata(self):
        """Generate comprehensive metadata for export"""
        import time
        from datetime import datetime

        selected_columns = self.get_selected_export_columns()

        metadata = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'flika_plugin': 'SPT_Batch_Analysis_Enhanced',
                'version': '2.0',  # Update as needed
                'total_columns_selected': len(selected_columns),
                'selected_columns': selected_columns
            },
            'analysis_parameters': self.parameters.to_dict(),
            'file_export_options': {
                'export_enhanced_analysis': self.export_enhanced_checkbox.isChecked(),
                'export_basic_tracks': self.export_tracks_checkbox.isChecked(),
                'export_features_only': self.export_features_checkbox.isChecked()
            },
            'column_categories': {}
        }

        # Add category information
        for category, info in self.column_categories.items():
            selected_in_category = [col for col in info['columns'] if col in selected_columns]
            metadata['column_categories'][category] = {
                'description': info['description'],
                'total_columns': len(info['columns']),
                'selected_columns': selected_in_category,
                'selected_count': len(selected_in_category),
                'depends_on': info.get('depends_on', 'always_available')
            }

        return metadata

    def export_metadata_file(self, base_path):
        """Export metadata file for an analysis"""
        try:
            metadata = self.generate_export_metadata()
            metadata_path = base_path.replace('.tif', '_analysis_metadata.json')

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.log_message(f"    Metadata exported: {os.path.basename(metadata_path)}")
            return True

        except Exception as e:
            self.log_message(f"    Error exporting metadata: {e}")
            return False


    def create_control_buttons(self, main_layout):
        """Create control buttons"""
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.start_btn)

        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.clicked.connect(self.save_parameters)
        button_layout.addWidget(self.save_params_btn)

        self.load_params_btn = QPushButton("Load Parameters")
        self.load_params_btn.clicked.connect(self.load_parameters)
        button_layout.addWidget(self.load_params_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        main_layout.addLayout(button_layout)

    def on_bg_subtraction_toggled(self):
        """Handle background subtraction checkbox toggle"""
        enabled = self.bg_checkbox.isChecked()
        self.bg_frame_specific_checkbox.setEnabled(enabled)

    def on_geometric_enable_toggled(self):
        """Handle geometric analysis enable/disable"""
        enabled = self.geometric_enable_checkbox.isChecked()
        self.geometric_params_group.setEnabled(enabled)
        self.linear_class_group.setEnabled(enabled)

    def on_autocorr_enable_toggled(self):
        """Handle autocorrelation analysis enable/disable"""
        enabled = self.autocorr_enable_checkbox.isChecked()
        self.autocorr_params_group.setEnabled(enabled)
        self.autocorr_viz_group.setEnabled(enabled)
        self.autocorr_output_group.setEnabled(enabled)
        self.autocorr_live_group.setEnabled(enabled)
        self.autocorr_plot_group.setEnabled(enabled)

    def select_directory(self):
        """Select input directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.dir_path_edit.setText(directory)
            self.refresh_file_list()

    def select_training_data(self):
        """Select training data file"""
        default_dir = os.path.join(get_plugin_directory(), 'training_data')
        if not os.path.exists(default_dir):
            default_dir = get_plugin_directory()

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Data", default_dir, "CSV Files (*.csv)")
        if file_path:
            self.train_path_edit.setText(file_path)
            self.parameters.training_data_path = file_path
            self.update_training_data_status()

    def reset_training_data_path(self):
        """Reset training data path to default"""
        default_path = get_default_training_data_path()
        self.train_path_edit.setText(default_path)
        self.parameters.training_data_path = default_path
        self.update_training_data_status()

    def update_training_data_status(self):
        """Update the training data status label"""
        if os.path.isfile(self.parameters.training_data_path):
            self.train_status_label.setText("✓ Training data file found")
            self.train_status_label.setStyleSheet("color: green;")
        elif os.path.isdir(self.parameters.training_data_path):
            self.train_status_label.setText("⚠ Directory selected - choose a CSV file")
            self.train_status_label.setStyleSheet("color: orange;")
        else:
            self.train_status_label.setText("⚠ Training data file not found")
            self.train_status_label.setStyleSheet("color: red;")

    def refresh_file_list(self):
        """Refresh the file list"""
        directory = self.dir_path_edit.text()
        if directory == "No directory selected":
            return

        pattern = self.file_pattern_edit.currentText()
        file_list = glob.glob(os.path.join(directory, pattern), recursive=True)

        self.file_list_widget.clear()
        for file_path in sorted(file_list):
            self.file_list_widget.addItem(os.path.basename(file_path))

        self.file_count_label.setText(f"{len(file_list)} files selected")
        self.file_paths = file_list

        # Show preview of experiment names if auto-detection is enabled
        if hasattr(self, 'auto_detect_checkbox') and self.auto_detect_checkbox.isChecked():
            self.preview_experiment_names()

        # Update detection tab availability
        if hasattr(self, 'detection_enable_checkbox') and self.detection_enable_checkbox.isChecked():
            self.start_detection_btn.setEnabled(len(file_list) > 0)

    def preview_experiment_names(self):
        """Preview experiment names that will be detected from subfolders"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            return

        experiment_names = set()
        for file_path in self.file_paths[:10]:
            exp_name = self.get_experiment_name_for_file(file_path)
            experiment_names.add(exp_name)

        if experiment_names:
            preview_text = f"{len(self.file_paths)} files selected\nDetected experiment names: {', '.join(sorted(experiment_names))}"
            if len(self.file_paths) > 10:
                preview_text += " (preview from first 10 files)"
        else:
            preview_text = f"{len(self.file_paths)} files selected"

        self.file_count_label.setText(preview_text)

    def on_auto_detect_toggled(self):
        """Handle auto-detect checkbox toggle"""
        auto_detect = self.auto_detect_checkbox.isChecked()

        # Enable/disable manual experiment name input
        self.manual_exp_label.setEnabled(not auto_detect)
        self.experiment_name_edit.setEnabled(not auto_detect)

        if auto_detect:
            self.exp_status_label.setText("ℹ️ Experiment names will be auto-detected from subfolder names")
            self.exp_status_label.setStyleSheet("color: blue;")
            if hasattr(self, 'file_paths') and self.file_paths:
                self.preview_experiment_names()
        else:
            self.exp_status_label.setText("✏️ Using manual experiment name")
            self.exp_status_label.setStyleSheet("color: green;")
            if hasattr(self, 'file_paths') and self.file_paths:
                self.file_count_label.setText(f"{len(self.file_paths)} files selected")

    def get_experiment_name_for_file(self, file_path):
        """Get experiment name for a file based on settings"""
        if self.parameters.auto_detect_experiment_names:
            directory = os.path.dirname(file_path)
            input_dir = self.dir_path_edit.text()

            # Debug logging
            self.log_message(f"    Auto-detect enabled. File directory: {directory}")
            self.log_message(f"    Input directory: {input_dir}")

            if directory == "No directory selected" or input_dir == "No directory selected":
                self.log_message("    Warning: No directory selected, using 'Unknown'")
                return "Unknown"

            try:
                rel_path = os.path.relpath(directory, input_dir)
                self.log_message(f"    Relative path: {rel_path}")

                if rel_path == "." or rel_path == "":
                    # File is directly in the input directory
                    experiment_name = os.path.basename(input_dir)
                    self.log_message(f"    File in root directory, using directory name: {experiment_name}")
                    return experiment_name

                # File is in a subdirectory
                path_parts = rel_path.split(os.sep)
                self.log_message(f"    Path parts: {path_parts}")

                experiment_name = path_parts[0]
                self.log_message(f"    Raw experiment name: {experiment_name}")

                # Clean up the experiment name (remove special characters)
                cleaned_name = "".join(c for c in experiment_name if c.isalnum() or c in "._-")
                self.log_message(f"    Cleaned experiment name: {cleaned_name}")

                final_name = cleaned_name if cleaned_name else "Unknown"
                self.log_message(f"    Final experiment name: {final_name}")
                return final_name

            except Exception as e:
                self.log_message(f"    Error extracting experiment name from path: {e}")
                print(f"Error extracting experiment name from path: {e}")
                return "Unknown"
        else:
            manual_name = self.parameters.experiment_name if self.parameters.experiment_name else "Unknown"
            self.log_message(f"    Using manual experiment name: {manual_name}")
            return manual_name

    def load_tracks_data(self):
        """Load tracks data for autocorrelation analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Tracks Data", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*.*)")

        if file_path:
            try:
                # Try to load the file
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                # Validate required columns
                required_cols = ['track_number', 'frame', 'x', 'y']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    g.alert(f"Missing required columns: {', '.join(missing_cols)}")
                    return

                # Filter for valid tracks
                valid_tracks = df.groupby('track_number').size()
                min_length = self.autocorr_min_length_spin.value()
                valid_track_numbers = valid_tracks[valid_tracks >= min_length].index

                self.current_tracks_df = df[df['track_number'].isin(valid_track_numbers)]

                # Update status
                n_tracks = len(valid_track_numbers)
                n_points = len(self.current_tracks_df)
                self.autocorr_status_label.setText(
                    f"Loaded {n_tracks} tracks with {n_points} points (min length: {min_length})")
                self.autocorr_status_label.setStyleSheet("color: green;")

                # Enable analysis button
                self.run_autocorr_btn.setEnabled(True)

            except Exception as e:
                g.alert(f"Error loading tracks data: {str(e)}")
                self.autocorr_status_label.setText("Error loading data")
                self.autocorr_status_label.setStyleSheet("color: red;")

    def run_autocorrelation_analysis(self):
        """Run autocorrelation analysis on loaded tracks"""
        if self.current_tracks_df is None:
            g.alert("No tracks data loaded")
            return

        # Update parameters
        time_interval = self.autocorr_time_interval_spin.value()
        num_intervals = self.autocorr_num_intervals_spin.value()

        # Show progress
        self.autocorr_progress_bar.setVisible(True)
        self.autocorr_progress_bar.setRange(0, 0)  # Indeterminate progress
        self.run_autocorr_btn.setEnabled(False)

        # Create and start worker thread
        self.autocorr_worker = AutocorrelationWorker(
            self.current_tracks_df, time_interval, num_intervals)

        self.autocorr_worker.progress_update.connect(self.update_autocorr_status)
        self.autocorr_worker.analysis_complete.connect(self.on_autocorr_complete)
        self.autocorr_worker.analysis_error.connect(self.on_autocorr_error)

        self.autocorr_worker.start()

    def update_autocorr_status(self, message):
        """Update autocorrelation status"""
        self.autocorr_status_label.setText(message)

    def on_autocorr_complete(self, scalar_products, averages, individual_tracks):
        """Handle completion of autocorrelation analysis"""
        self.autocorr_progress_bar.setVisible(False)
        self.run_autocorr_btn.setEnabled(True)

        # Update plot
        show_individual = self.show_individual_btn.isChecked()
        max_tracks = self.autocorr_max_tracks_spin.value()

        self.autocorr_plot_widget.plot_autocorrelation(
            averages, individual_tracks, show_individual, max_tracks)

        # Enable save button
        self.save_plot_btn.setEnabled(True)

        # Store results for potential saving
        self.current_autocorr_results = {
            'scalar_products': scalar_products,
            'averages': averages,
            'individual_tracks': individual_tracks
        }

        self.autocorr_status_label.setText("Autocorrelation analysis complete!")
        self.autocorr_status_label.setStyleSheet("color: green; font-weight: bold;")

    def on_autocorr_error(self, error_msg):
        """Handle autocorrelation analysis error"""
        self.autocorr_progress_bar.setVisible(False)
        self.run_autocorr_btn.setEnabled(True)

        self.autocorr_status_label.setText(f"Error: {error_msg}")
        self.autocorr_status_label.setStyleSheet("color: red;")

        g.alert(f"Autocorrelation analysis failed: {error_msg}")

    def update_autocorr_plot(self):
        """Update the autocorrelation plot with current settings"""
        if hasattr(self, 'current_autocorr_results'):
            show_individual = self.show_individual_btn.isChecked()
            max_tracks = self.autocorr_max_tracks_spin.value()

            self.autocorr_plot_widget.plot_autocorrelation(
                self.current_autocorr_results['averages'],
                self.current_autocorr_results['individual_tracks'],
                show_individual, max_tracks)

    def save_autocorr_plot(self):
        """Save the current autocorrelation plot"""
        if hasattr(self, 'current_autocorr_results'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Autocorrelation Plot", "autocorrelation_plot.png",
                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*.*)")

            if file_path:
                try:
                    self.autocorr_plot_widget.save_plot(file_path)
                    g.alert(f"Plot saved to {file_path}")
                except Exception as e:
                    g.alert(f"Error saving plot: {str(e)}")

    def update_parameters(self):
        """Update parameters from GUI"""
        self.parameters.pixel_size = self.pixel_size_spin.value()
        self.parameters.frame_length = self.frame_length_spin.value()
        self.parameters.max_gap_frames = self.max_gap_spin.value()
        self.parameters.max_link_distance = self.max_dist_spin.value()
        self.parameters.min_track_segments = self.min_segments_spin.value()
        self.parameters.rg_mobility_threshold = self.rg_threshold_spin.value()
        self.parameters.experiment_name = self.experiment_name_edit.currentText()
        self.parameters.auto_detect_experiment_names = self.auto_detect_checkbox.isChecked()

        # Update analysis step flags
        self.parameters.enable_nearest_neighbors = self.nn_checkbox.isChecked()
        self.parameters.enable_svm_classification = self.svm_checkbox.isChecked()
        self.parameters.enable_velocity_analysis = self.velocity_checkbox.isChecked()
        self.parameters.enable_diffusion_analysis = self.diffusion_checkbox.isChecked()
        self.parameters.enable_background_subtraction = self.bg_checkbox.isChecked()
        self.parameters.enable_localization_error = self.loc_error_checkbox.isChecked()
        self.parameters.enable_straightness_analysis = self.straightness_checkbox.isChecked()


        # Enhanced analysis step flags
        self.parameters.enable_direction_analysis = self.direction_checkbox.isChecked()
        self.parameters.enable_distance_differential = self.distance_diff_checkbox.isChecked()
        self.parameters.enable_missing_points_integration = self.missing_points_checkbox.isChecked()
        self.parameters.enable_enhanced_interpolation = self.interpolation_checkbox.isChecked()
        self.parameters.enable_full_track_interpolation = self.full_interpolation_checkbox.isChecked()
        self.parameters.save_separate_interpolated_file = self.separate_interpolated_file_checkbox.isChecked()
        self.parameters.extend_interpolation_to_full_recording = self.extend_to_full_recording_checkbox.isChecked()

        # ROI background subtraction options
        self.parameters.roi_frame_specific_background = self.bg_frame_specific_checkbox.isChecked()

        # Geometric analysis parameters
        self.parameters.enable_geometric_analysis = self.geometric_enable_checkbox.isChecked()
        self.parameters.geometric_rg_method = 'simple' if self.geometric_method_simple.isChecked() else 'tensor'
        self.parameters.geometric_srg_cutoff = self.geometric_srg_cutoff_spin.value()
        self.parameters.enable_geometric_linear_classification = self.geometric_linear_enable_checkbox.isChecked()
        self.parameters.geometric_directionality_threshold = self.geometric_directionality_spin.value()
        self.parameters.geometric_perpendicular_threshold = self.geometric_perpendicular_spin.value()
        self.parameters.geometric_cutoff_length = self.geometric_cutoff_length_spin.value()

        # Autocorrelation analysis parameters
        self.parameters.enable_autocorrelation_analysis = self.autocorr_enable_checkbox.isChecked()
        self.parameters.autocorr_time_interval = self.autocorr_time_interval_spin.value()
        self.parameters.autocorr_num_intervals = self.autocorr_num_intervals_spin.value()
        self.parameters.autocorr_min_track_length = self.autocorr_min_length_spin.value()
        self.parameters.autocorr_show_individual_tracks = self.autocorr_show_individual_checkbox.isChecked()
        self.parameters.autocorr_max_tracks_plot = self.autocorr_max_tracks_spin.value()
        self.parameters.autocorr_save_plots = self.autocorr_save_plots_checkbox.isChecked()
        self.parameters.autocorr_save_data = self.autocorr_save_data_checkbox.isChecked()

        # NEW: Detection parameters
        self.parameters.enable_detection = self.detection_enable_checkbox.isChecked()
        self.parameters.detection_psf_sigma = self.detection_psf_sigma_spin.value()
        self.parameters.detection_alpha_threshold = self.detection_alpha_spin.value()
        self.parameters.detection_min_intensity = self.detection_min_intensity_spin.value()
        self.parameters.detection_skip_existing = self.detection_skip_existing_checkbox.isChecked()
        self.parameters.detection_show_results = self.detection_show_results_checkbox.isChecked()

        # Call the enhanced version with trackpy support
        self.update_parameters_with_mixed_motion()



    def start_analysis(self):
        """Start the integrated batch analysis pipeline"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            g.alert("No files selected for analysis")
            return

        self.update_parameters()

        # Switch to progress tab
        self.tab_widget.setCurrentIndex(5)  # Progress tab

        # Disable start button
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.log_message("🚀 Starting integrated SPT batch analysis pipeline...")

        # Log what will be processed
        self.log_message(f"📁 Processing {len(self.file_paths)} files")

        if self.parameters.enable_detection:
            self.log_message("🔍 Detection enabled - will run U-Track detection first")

        if self.parameters.enable_autocorrelation_analysis:
            self.log_message("📊 Autocorrelation analysis enabled - will analyze tracks after main pipeline")

        if self.parameters.auto_detect_experiment_names:
            self.log_message("🏷️ Using auto-detection: experiment names will be extracted from subfolder names")
        else:
            self.log_message(f"🏷️ Using manual experiment name: '{self.parameters.experiment_name}'")

        try:
            total_files = len(self.file_paths)
            successful_files = []

            # Phase 1: Detection + Main Analysis Pipeline
            self.log_message("\n" + "="*60)
            self.log_message("📈 PHASE 1: DETECTION & MAIN ANALYSIS PIPELINE")
            self.log_message("="*60)

            for i, file_path in enumerate(self.file_paths):
                file_name = os.path.basename(file_path)
                self.status_label.setText(f"Phase 1: Processing {file_name}")
                self.log_message(f"\n🔄 Processing file {i+1}/{total_files}: {file_name}")
                QApplication.processEvents()

                file_success = True

                # Step 1: Run detection if enabled
                if self.parameters.enable_detection:
                    self.log_message("  🔍 Running U-Track particle detection...")
                    detection_success = self.run_detection_for_file(file_path)
                    if not detection_success:
                        self.log_message(f"  ❌ Detection failed for {file_name}")
                        file_success = False
                    else:
                        self.log_message(f"  ✅ Detection completed successfully")

                # Step 2: Run main analysis pipeline
                if file_success:
                    self.log_message("  📊 Running main SPT analysis pipeline...")
                    analysis_success = self.process_file(file_path)
                    if not analysis_success:
                        self.log_message(f"  ❌ Main analysis failed for {file_name}")
                        file_success = False
                    else:
                        self.log_message(f"  ✅ Main analysis completed successfully")
                        successful_files.append(file_path)

                if file_success:
                    self.log_message(f"✅ Successfully completed Phase 1 for {file_name}")
                else:
                    self.log_message(f"❌ Phase 1 failed for {file_name}")

                # Update progress for Phase 1 (0-70% of total progress)
                phase1_progress = int((i + 1) / total_files * 70)
                self.progress_bar.setValue(phase1_progress)
                QApplication.processEvents()

            # Phase 2: Autocorrelation Analysis (if enabled)
            if self.parameters.enable_autocorrelation_analysis and successful_files:
                self.log_message("\n" + "="*60)
                self.log_message("📈 PHASE 2: AUTOCORRELATION ANALYSIS")
                self.log_message("="*60)

                self.run_batch_autocorrelation_analysis(successful_files)

            # Final summary
            self.log_message("\n" + "="*60)
            self.log_message("🎉 INTEGRATED PIPELINE COMPLETE")
            self.log_message("="*60)
            self.log_message(f"✅ Successfully processed: {len(successful_files)}/{total_files} files")

            if len(successful_files) < total_files:
                failed_count = total_files - len(successful_files)
                self.log_message(f"❌ Failed to process: {failed_count} files")

            self.status_label.setText("🎉 Integrated analysis pipeline completed successfully!")
            self.progress_bar.setValue(100)

            # Show completion dialog
            if len(successful_files) == total_files:
                g.alert("🎉 Integrated batch analysis completed successfully!\n\nAll files processed with detection, main analysis, and autocorrelation (if enabled).")
            else:
                g.alert(f"⚠️ Batch analysis completed with some issues.\n\nSuccessful: {len(successful_files)}/{total_files} files\n\nCheck the log for details on failed files.")

        except Exception as e:
            error_msg = f"❌ Integrated pipeline failed: {str(e)}"
            self.status_label.setText(error_msg)
            self.log_message(f"ERROR: {error_msg}")
            import traceback
            self.log_message(f"Full error traceback:\n{traceback.format_exc()}")
            g.alert(error_msg)

        finally:
            self.start_btn.setEnabled(True)


    def run_batch_autocorrelation_analysis(self, successful_files):
        """Run autocorrelation analysis on all successfully processed files"""
        self.log_message("🔄 Starting batch autocorrelation analysis...")

        autocorr_successful = 0
        autocorr_failed = 0

        for i, file_path in enumerate(successful_files):
            file_name = os.path.basename(file_path)
            self.status_label.setText(f"Phase 2: Autocorrelation for {file_name}")
            self.log_message(f"  📊 Running autocorrelation analysis for {file_name}")
            QApplication.processEvents()

            try:
                # Look for the tracks file created by main analysis
                tracks_file_candidates = [
                    file_path.replace('.tif', '_tracks.csv'),
                    file_path.replace('.tif', '_enhanced_analysis.csv'),  # Fallback to full analysis file
                ]

                tracks_file = None
                for candidate in tracks_file_candidates:
                    if os.path.exists(candidate):
                        tracks_file = candidate
                        break

                if not tracks_file:
                    self.log_message(f"    ⚠️ No tracks file found for {file_name}, skipping autocorrelation")
                    autocorr_failed += 1
                    continue

                self.log_message(f"    📂 Loading tracks from: {os.path.basename(tracks_file)}")

                # Load tracks data
                tracks_df = pd.read_csv(tracks_file)

                # Validate required columns
                required_cols = ['track_number', 'frame', 'x', 'y']
                missing_cols = [col for col in required_cols if col not in tracks_df.columns]

                if missing_cols:
                    self.log_message(f"    ❌ Missing required columns {missing_cols}, skipping autocorrelation")
                    autocorr_failed += 1
                    continue

                # Filter for valid tracks (minimum length)
                valid_tracks = tracks_df.groupby('track_number').size()
                min_length = self.parameters.autocorr_min_track_length
                valid_track_numbers = valid_tracks[valid_tracks >= min_length].index

                if len(valid_track_numbers) == 0:
                    self.log_message(f"    ⚠️ No tracks meet minimum length requirement ({min_length}), skipping")
                    autocorr_failed += 1
                    continue

                autocorr_tracks = tracks_df[tracks_df['track_number'].isin(valid_track_numbers)]
                self.log_message(f"    ✅ Found {len(valid_track_numbers)} valid tracks for autocorrelation")

                # Perform autocorrelation analysis
                scalar_products, averages, individual_tracks = AutocorrelationAnalyzer.process_track_data(
                    autocorr_tracks,
                    self.parameters.autocorr_time_interval,
                    self.parameters.autocorr_num_intervals
                )

                if averages is not None:
                    # Save autocorrelation results
                    base_name = os.path.splitext(file_path)[0]

                    if self.parameters.autocorr_save_data:
                        # Save averages
                        avg_path = f"{base_name}_autocorr_averages.csv"
                        averages.to_csv(avg_path)
                        self.log_message(f"    💾 Saved averages: {os.path.basename(avg_path)}")

                        # Save individual tracks
                        if individual_tracks is not None:
                            tracks_path = f"{base_name}_autocorr_individual_tracks.csv"
                            individual_tracks.to_csv(tracks_path, index=False)
                            self.log_message(f"    💾 Saved individual tracks: {os.path.basename(tracks_path)}")

                    if self.parameters.autocorr_save_plots:
                        # Create and save plot
                        plot_path = f"{base_name}_autocorr_plot.png"
                        self.save_autocorr_plot_for_file(averages, individual_tracks, plot_path)
                        self.log_message(f"    🖼️ Saved plot: {os.path.basename(plot_path)}")

                    self.log_message(f"    ✅ Autocorrelation analysis complete for {len(valid_track_numbers)} tracks")
                    autocorr_successful += 1

                else:
                    self.log_message(f"    ❌ No valid autocorrelation results generated")
                    autocorr_failed += 1

            except Exception as e:
                self.log_message(f"    ❌ Autocorrelation analysis failed: {str(e)}")
                autocorr_failed += 1

            # Update progress for Phase 2 (70-100% of total progress)
            phase2_progress = 70 + int((i + 1) / len(successful_files) * 30)
            self.progress_bar.setValue(phase2_progress)
            QApplication.processEvents()

        # Summary of autocorrelation results
        self.log_message(f"\n📊 Autocorrelation Analysis Summary:")
        self.log_message(f"  ✅ Successful: {autocorr_successful} files")
        self.log_message(f"  ❌ Failed: {autocorr_failed} files")


    def save_autocorr_plot_for_file(self, averages, individual_tracks, plot_path):
        """Save autocorrelation plot for a specific file"""
        try:
            plt.figure(figsize=(10, 6))

            # Plot individual tracks if requested
            if self.parameters.autocorr_show_individual_tracks and individual_tracks is not None:
                pivot_df = individual_tracks.pivot(index='track_id', columns='time_interval', values='correlation')
                pivot_df[0] = 1.0
                pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)

                # Sample tracks
                tracks_to_plot = pivot_df.index
                if len(tracks_to_plot) > self.parameters.autocorr_max_tracks_plot:
                    tracks_to_plot = random.sample(list(tracks_to_plot), self.parameters.autocorr_max_tracks_plot)

                for track_id in tracks_to_plot:
                    if track_id in pivot_df.index:
                        track_data = pivot_df.loc[track_id]
                        plt.plot(track_data.index, track_data.values, '-',
                               color='gray', alpha=0.1, linewidth=0.5)

            # Plot average
            x = np.array([float(col) for col in averages.columns])
            y = averages.loc['AVG'].values
            yerr = averages.loc['SEM'].values

            plt.errorbar(x, y, yerr=yerr, fmt='o-', color='blue', linewidth=2,
                       label='Average ± SEM', markersize=6, capsize=5)

            plt.xlabel('Time Interval')
            plt.ylabel('Direction Autocorrelation')
            plt.title(f'Directional Autocorrelation - {os.path.basename(plot_path).replace("_autocorr_plot.png", "")}')
            plt.ylim(-0.2, 1.0)
            plt.xlim(0, max(x) * 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.log_message(f"      Error saving autocorr plot: {e}")


    def filter_dataframe_columns(self, df, file_type='enhanced'):
        """Filter dataframe columns based on user selections"""
        if not hasattr(self, 'export_column_checkboxes'):
            return df  # No filtering if export control not available

        selected_columns = self.get_selected_export_columns()

        # Always include essential columns regardless of selection for data integrity
        essential_columns = ['track_number', 'frame', 'x', 'y']

        # Determine which columns to keep based on file type
        if file_type == 'tracks':
            # Basic tracks file - only essential + intensity + experiment
            keep_columns = essential_columns + ['intensity', 'Experiment']
            if 'id' in df.columns:
                keep_columns.append('id')
        elif file_type == 'features':
            # Features file - essential + selected feature columns
            keep_columns = essential_columns + ['intensity', 'n_segments', 'Experiment']
            keep_columns.extend([col for col in selected_columns if col not in essential_columns])
        else:  # 'enhanced' or default
            # Full analysis file - all selected columns
            keep_columns = selected_columns.copy()
            # Ensure essential columns are included
            for col in essential_columns:
                if col not in keep_columns:
                    keep_columns.append(col)

        # Filter to only include columns that exist in the dataframe
        available_columns = [col for col in keep_columns if col in df.columns]

        return df[available_columns]

    # Enhanced process_file method that removes the built-in autocorrelation
    # (since it will be handled in Phase 2)
    def process_file(self, file_path):
        """Enhanced process_file with comprehensive logging and error handling"""
        try:
            # Setup comprehensive file logging
            log_file = self.setup_logging_for_analysis(file_path)

            # Log analysis start
            self.file_logger.log('info', "="*50)
            self.file_logger.log('info', f"STARTING ANALYSIS FOR: {os.path.basename(file_path)}")
            self.file_logger.log('info', "="*50)

            # Get experiment name
            experiment_name = self.get_experiment_name_for_file(file_path)
            self.log_message(f"  Using experiment name: '{experiment_name}'")

            # Load localization data
            self.log_message("  Loading localization data...")
            start_time = time.time()
            data = self.load_localization_data(file_path)
            if data is None:
                self.file_logger.log_error("Failed to load localization data")
                return False

            load_time = time.time() - start_time
            self.file_logger.log_performance("data_loading", load_time, f"loaded {len(data)} localizations")

            # Validate and log linking method
            self.validate_linking_method()
            self.file_logger.log('info', f"Using linking method: {self.parameters.linking_method}")
            if self.parameters.linking_method == 'builtin':
                self.file_logger.log('info', f"Built-in algorithm: {self.parameters.builtin_linking_algorithm}")

            # Link particles with enhanced logging
            self.log_message("  Linking particles...")
            start_time = time.time()
            points = self.link_particles_enhanced_with_mixed_motion(data, file_path)
            linking_time = time.time() - start_time

            if points is None:
                self.file_logger.log_error("Particle linking returned None")
                return False

            if points.recursiveFailure:
                self.file_logger.log_error("Particle linking failed due to recursive failure")
                return False

            self.file_logger.log_performance("particle_linking", linking_time,
                                           f"{len(points.tracks)} tracks created")


            # Calculate basic features
            self.log_message("  Calculating features...")
            start_time = time.time()
            tracks_df = self.calculate_features(points)
            if tracks_df is None:
                self.file_logger.log_error("Feature calculation failed")
                return False
            feature_time = time.time() - start_time
            self.file_logger.log_performance("feature_calculation", feature_time,
                                           f"{len(tracks_df)} track points processed")

            # Add experiment name to tracks EARLY in the pipeline
            tracks_df['Experiment'] = experiment_name
            self.log_message(f"  🏷️ Added experiment name '{experiment_name}' to {len(tracks_df)} rows")

            # Add lag displacement features
            self.log_message("  📏 Adding lag displacement features...")
            tracks_df = self.add_lag_features(tracks_df)

            # Geometric analysis if enabled
            if self.parameters.enable_geometric_analysis:
                self.log_message("  📐 Performing geometric analysis...")
                tracks_df = self.add_geometric_analysis(tracks_df)

            # Nearest neighbor analysis
            if self.parameters.enable_nearest_neighbors:
                self.log_message("  🎯 Performing nearest neighbor analysis...")
                tracks_df = NearestNeighborAnalyzer.analyze_frame_neighbors(
                    tracks_df, self.parameters.nn_radii)

            # Velocity analysis
            if self.parameters.enable_velocity_analysis:
                self.log_message("  🚀 Calculating velocities...")
                tracks_df = self.add_velocity_analysis(tracks_df)

            # Diffusion analysis
            if self.parameters.enable_diffusion_analysis:
                self.log_message("  🌊 Performing diffusion analysis...")
                tracks_df = self.add_diffusion_analysis(tracks_df)

            # Localization error analysis
            if self.parameters.enable_localization_error:
                self.log_message("  🎯 Calculating localization errors...")
                tracks_df = self.add_localization_error(tracks_df)

            # Missing points integration (Core Analysis Step)
            if self.parameters.enable_missing_points_integration:
                self.log_message("  🔄 Integrating missing points...")
                tracks_df = MissingPointsIntegrator.add_missing_localizations(
                    tracks_df, file_path, self.parameters.pixel_size)

            # Direction of travel analysis
            if self.parameters.enable_direction_analysis:
                self.log_message("  🧭 Performing direction analysis...")
                tracks_df = DirectionAnalyzer.add_direction_analysis(tracks_df)

            # Distance differential analysis
            if self.parameters.enable_distance_differential:
                self.log_message("  📊 Calculating distance differentials...")
                tracks_df = DistanceDifferentialAnalyzer.add_distance_differential(tracks_df)

            # SVM classification
            if self.parameters.enable_svm_classification:
                self.log_message("  🤖 Performing SVM classification...")
                tracks_df = SVMClassifier.classify_tracks(
                    tracks_df, self.parameters.training_data_path, experiment_name)

            # Background subtraction
            if self.parameters.enable_background_subtraction:
                self.log_message("  🔍 Performing background subtraction...")
                tracks_df = self.add_background_subtraction(tracks_df, file_path)

            # Enhanced interpolation (must come after SVM classification)
            if self.parameters.enable_enhanced_interpolation and self.parameters.enable_svm_classification:
                self.log_message("  🔄 Performing enhanced interpolation...")
                # Initialize is_interpolated column if not already present
                if 'is_interpolated' not in tracks_df.columns:
                    tracks_df['is_interpolated'] = 0
                tracks_df = EnhancedInterpolator.interpolate_trapped_sites(tracks_df, file_path)

            # Add motion model information to tracks if using U-Track
            if self.parameters.linking_method == 'utrack':
                tracks_df['linking_method'] = 'utrack'
                tracks_df['motion_model_used'] = self.parameters.utrack_motion_model
                if self.parameters.utrack_motion_model == 'mixed':
                    tracks_df['pmms_enabled'] = self.parameters.utrack_enable_iterative_smoothing
                    tracks_df['tracking_rounds_used'] = self.parameters.utrack_num_tracking_rounds
            else:
                tracks_df['linking_method'] = self.parameters.linking_method
                tracks_df['motion_model_used'] = 'single_model'

            # Verify experiment name is still present
            if 'Experiment' in tracks_df.columns:
                unique_experiments = tracks_df['Experiment'].unique()
                self.log_message(f"  ✅ Final check: Experiment column contains {len(unique_experiments)} unique values: {list(unique_experiments)}")
            else:
                self.log_message("  ⚠️ WARNING: Experiment column missing from final results!")

            # Log final track statistics
            self.log_final_track_statistics(tracks_df)

            # Save results with export control filtering
            self.log_message("  💾 Saving results...")

            # Check if export control is available, otherwise use legacy behavior
            use_export_control = hasattr(self, 'export_column_checkboxes') and hasattr(self, 'export_enhanced_checkbox')

            # OPTION 1: Save separate files for original and interpolated data
            if self.parameters.enable_full_track_interpolation and self.parameters.save_separate_interpolated_file:
                self.log_message("  📁 Saving separate original and interpolated result files...")

                # Save original results (before interpolation)
                tracks_df_original = tracks_df.copy()
                if 'is_interpolated' not in tracks_df_original.columns:
                    tracks_df_original['is_interpolated'] = 0

                self.save_analysis_files(tracks_df_original, file_path, use_export_control, suffix="_original")

                # Apply full track interpolation
                self.log_message("  🔄 Applying full track interpolation for interpolated results file...")
                tracks_df_interpolated = FullTrackInterpolator.interpolate_all_tracks(
                    tracks_df, file_path,
                    extend_to_full_recording=self.parameters.extend_interpolation_to_full_recording)

                # Save interpolated results
                self.save_analysis_files(tracks_df_interpolated, file_path, use_export_control, suffix="_interpolated")

            # OPTION 2: Standard processing (single file output)
            else:
                # Apply full track interpolation if enabled (but saving single file)
                if self.parameters.enable_full_track_interpolation:
                    self.log_message("  🔄 Performing full track interpolation...")
                    tracks_df = FullTrackInterpolator.interpolate_all_tracks(
                        tracks_df, file_path,
                        extend_to_full_recording=self.parameters.extend_interpolation_to_full_recording)  # NEW parameter
                elif 'is_interpolated' not in tracks_df.columns:
                    tracks_df['is_interpolated'] = 0

                # Save single file
                self.save_analysis_files(tracks_df, file_path, use_export_control)

            # Final logging
            self.file_logger.log('info', "="*50)
            self.file_logger.log('info', f"ANALYSIS COMPLETED SUCCESSFULLY FOR: {os.path.basename(file_path)}")
            self.file_logger.log('info', f"Final dataset: {len(tracks_df)} points in {len(tracks_df['track_number'].unique())} tracks")
            self.file_logger.log('info', "="*50)

            return True

        except Exception as e:
            error_msg = f"ERROR in main analysis: {e}"
            self.log_message(f"  {error_msg}")
            self.file_logger.log_error(error_msg, e, context={"file_path": file_path})
            return False


    def save_analysis_files(self, tracks_df, file_path, use_export_control, suffix=""):
        """Save analysis files with optional suffix for separate original/interpolated files"""
        try:
            # Create file paths with suffix
            if suffix:
                enhanced_path = file_path.replace('.tif', f'_{suffix.lstrip("_")}_enhanced_analysis.csv')
                tracks_path = file_path.replace('.tif', f'_{suffix.lstrip("_")}_tracks.csv')
                features_path = file_path.replace('.tif', f'_{suffix.lstrip("_")}_features.csv')
                metadata_path = file_path.replace('.tif', f'_{suffix.lstrip("_")}_analysis_metadata.json')
            else:
                enhanced_path = file_path.replace('.tif', '_enhanced_analysis.csv')
                tracks_path = file_path.replace('.tif', '_tracks.csv')
                features_path = file_path.replace('.tif', '_features.csv')
                metadata_path = file_path.replace('.tif', '_analysis_metadata.json')

            suffix_label = f" ({suffix.lstrip('_')})" if suffix else ""

            if use_export_control:
                # New export control system
                if self.export_enhanced_checkbox.isChecked():
                    filtered_df = self.filter_dataframe_columns(tracks_df, 'enhanced')
                    filtered_df.to_csv(enhanced_path, index=False)
                    self.log_message(f"    💾 Saved enhanced analysis{suffix_label}: {os.path.basename(enhanced_path)} ({len(filtered_df.columns)} columns)")
                else:
                    self.log_message(f"    ⭕ Skipped enhanced analysis file{suffix_label} (export disabled)")

                # Export metadata if requested (only for main files, not duplicated for suffix files)
                if self.export_metadata_checkbox.isChecked() and not suffix:
                    try:
                        # Update metadata path for single file
                        metadata_path = file_path.replace('.tif', '_analysis_metadata.json')
                        self.export_metadata_file(file_path)
                    except Exception as e:
                        self.log_message(f"    ⚠️ Warning: Could not export metadata: {e}")
            else:
                # Legacy behavior - save full enhanced analysis file
                tracks_df.to_csv(enhanced_path, index=False)
                self.log_message(f"    💾 Saved enhanced analysis{suffix_label}: {os.path.basename(enhanced_path)} ({len(tracks_df.columns)} columns)")

            # Save intermediate files if requested
            if self.parameters.save_intermediate:
                if use_export_control:
                    # New export control system for intermediate files
                    if self.export_tracks_checkbox.isChecked():
                        tracks_filtered = self.filter_dataframe_columns(tracks_df, 'tracks')
                        tracks_filtered.to_csv(tracks_path, index=False)
                        self.log_message(f"    💾 Saved tracks file{suffix_label}: {os.path.basename(tracks_path)} ({len(tracks_filtered.columns)} columns)")
                    else:
                        self.log_message(f"    ⭕ Skipped tracks file{suffix_label} (export disabled)")

                    if self.export_features_checkbox.isChecked():
                        features_filtered = self.filter_dataframe_columns(tracks_df, 'features')
                        features_filtered.to_csv(features_path, index=False)
                        self.log_message(f"    💾 Saved features file{suffix_label}: {os.path.basename(features_path)} ({len(features_filtered.columns)} columns)")
                    else:
                        self.log_message(f"    ⭕ Skipped features file{suffix_label} (export disabled)")

                else:
                    # Legacy behavior for intermediate files
                    # Save tracks only (basic track data)
                    basic_columns = ['track_number', 'frame', 'x', 'y', 'intensity', 'is_interpolated', 'Experiment',
                                    'linking_method', 'motion_model_used']
                    if 'id' in tracks_df.columns:
                        basic_columns.insert(-3, 'id')  # insert before 'Experiment'

                    tracks_only = tracks_df[[col for col in basic_columns if col in tracks_df.columns]].copy()
                    tracks_only.to_csv(tracks_path, index=False)
                    self.log_message(f"    💾 Saved tracks file{suffix_label}: {os.path.basename(tracks_path)} ({len(tracks_only.columns)} columns)")

                    # Save features file with core features
                    core_columns = [
                        'track_number', 'frame', 'x', 'y', 'intensity', 'n_segments', 'is_interpolated',
                        'radius_gyration', 'asymmetry', 'skewness', 'kurtosis',
                        'fracDimension', 'netDispl', 'efficiency', 'Straight',
                        'track_intensity_mean', 'track_intensity_std',
                        'lag', 'meanLag', 'track_length',
                        'radius_gyration_scaled', 'radius_gyration_scaled_nSegments',
                        'radius_gyration_scaled_trackLength',
                        'radius_gyration_ratio_to_mean_step_size',
                        'radius_gyration_mobility_threshold',
                        'Experiment', 'linking_method', 'motion_model_used'
                    ]

                    # Add other columns based on enabled features (same logic as before but include is_interpolated)
                    if 'id' in tracks_df.columns:
                        core_columns.insert(-3, 'id')

                    # Add geometric columns if enabled
                    if self.parameters.enable_geometric_analysis:
                        core_columns.extend([
                            'Rg_geometric', 'sRg_geometric', 'mean_step_length_geometric',
                            'geometric_mobility_classification'
                        ])
                        if self.parameters.enable_geometric_linear_classification:
                            core_columns.extend([
                                'geometric_linear_classification', 'geometric_directionality_ratio',
                                'geometric_mean_perpendicular_distance', 'geometric_normalized_perpendicular_distance'
                            ])

                    # Add enhanced features if enabled
                    if self.parameters.enable_direction_analysis:
                        core_columns.extend([
                            'direction_degrees', 'direction_radians', 'direction_x', 'direction_y',
                            'directional_autocorrelation', 'directional_persistence'
                        ])

                    if self.parameters.enable_distance_differential:
                        core_columns.append('dy-dt_distance')

                    # Add U-Track specific columns if applicable
                    if self.parameters.linking_method == 'utrack' and self.parameters.utrack_motion_model == 'mixed':
                        core_columns.extend(['pmms_enabled', 'tracking_rounds_used'])

                    # Only include columns that exist in the dataframe
                    features_columns = [col for col in core_columns if col in tracks_df.columns]
                    features_only = tracks_df[features_columns].copy()

                    features_only.to_csv(features_path, index=False)
                    self.log_message(f"    💾 Saved features file{suffix_label}: {os.path.basename(features_path)} ({len(features_columns)} columns)")

        except Exception as e:
            self.log_message(f"    ❌ Error saving analysis files{suffix_label}: {e}")
            import traceback
            self.log_message(f"    Full traceback:\n{traceback.format_exc()}")

    def add_geometric_analysis(self, tracks_df):
        """Add geometric Rg, sRg calculation and linear classification"""
        try:
            # Initialize new columns
            tracks_df['Rg_geometric'] = np.nan
            tracks_df['sRg_geometric'] = np.nan
            tracks_df['mean_step_length_geometric'] = np.nan
            tracks_df['geometric_mobility_classification'] = 'unclassified'

            if self.parameters.enable_geometric_linear_classification:
                tracks_df['geometric_linear_classification'] = 'unclassified'
                tracks_df['geometric_directionality_ratio'] = np.nan
                tracks_df['geometric_mean_perpendicular_distance'] = np.nan
                tracks_df['geometric_normalized_perpendicular_distance'] = np.nan

            track_numbers = tracks_df['track_number'].unique()

            for track_num in track_numbers:
                track_data = tracks_df[tracks_df['track_number'] == track_num].sort_values('frame')

                # Skip tracks that are too short
                if len(track_data) < self.parameters.geometric_cutoff_length:
                    continue

                # Get trajectory coordinates
                xy = track_data[['x', 'y']].to_numpy()

                # Calculate geometric Rg
                rg_geometric = GeometricAnalyzer.get_radius_of_gyration_simple(xy)

                # Calculate mean step length
                mean_step_length = GeometricAnalyzer.get_mean_step_length(xy)

                # Calculate scaled Rg
                srg_geometric = GeometricAnalyzer.get_scaled_rg(rg_geometric, mean_step_length)

                # Determine mobility classification
                mobility_classification = 'mobile' if srg_geometric >= self.parameters.geometric_srg_cutoff else 'immobile'

                # Update the dataframe for this track
                track_mask = tracks_df['track_number'] == track_num
                tracks_df.loc[track_mask, 'Rg_geometric'] = rg_geometric
                tracks_df.loc[track_mask, 'sRg_geometric'] = srg_geometric
                tracks_df.loc[track_mask, 'mean_step_length_geometric'] = mean_step_length
                tracks_df.loc[track_mask, 'geometric_mobility_classification'] = mobility_classification

                # Add linear classification if enabled
                if self.parameters.enable_geometric_linear_classification:
                    linear_result = GeometricAnalyzer.classify_linear_motion_simple(
                        xy,
                        self.parameters.geometric_directionality_threshold,
                        self.parameters.geometric_perpendicular_threshold
                    )

                    tracks_df.loc[track_mask, 'geometric_linear_classification'] = linear_result['classification']
                    tracks_df.loc[track_mask, 'geometric_directionality_ratio'] = linear_result['directionality_ratio']
                    tracks_df.loc[track_mask, 'geometric_mean_perpendicular_distance'] = linear_result['mean_perpendicular_distance']
                    tracks_df.loc[track_mask, 'geometric_normalized_perpendicular_distance'] = linear_result['normalized_perpendicular_distance']

            self.log_message(f"    Calculated geometric features for {len(track_numbers)} tracks")
            return tracks_df

        except Exception as e:
            self.log_message(f"    Error in geometric analysis: {e}")
            return tracks_df

    # [Continue with remaining methods - load_localization_data, link_particles, calculate_features, etc.]
    # [Include all the remaining methods from the original code]

    def load_localization_data(self, file_path):
        """Load localization data from CSV - UPDATED to preserve ID column"""
        try:
            locs_file = file_path.replace('.tif', '_locsID.csv')
            if not os.path.exists(locs_file):
                locs_file = file_path.replace('.tif', '_locs.csv')

            if not os.path.exists(locs_file):
                self.log_message(f"    No localization file found for {os.path.basename(file_path)}")
                return None

            df = pd.read_csv(locs_file)
            df['frame'] = df['frame'].astype(int) - 1  # Zero-based indexing
            df['x'] = df['x [nm]'] / self.parameters.pixel_size
            df['y'] = df['y [nm]'] / self.parameters.pixel_size

            # Store the full dataframe for later use in missing points integration
            self.original_localizations_df = df.copy()

            self.log_message(f"    Loaded {len(df)} localizations")

            # Return data including ID if available, otherwise create sequential IDs
            if 'id' in df.columns:
                return df[['frame', 'x', 'y', 'id']].to_numpy()
            else:
                # Create sequential IDs if not present
                df['id'] = range(len(df))
                return df[['frame', 'x', 'y', 'id']].to_numpy()

        except Exception as e:
            self.log_message(f"    Error loading localization data: {e}")
            return None

    def link_particles(self, txy_pts, file_path):
        """Link particles across frames - UPDATED to preserve IDs"""
        try:
            # Extract just frame, x, y for linking (Points class expects 3 columns)
            if txy_pts.shape[1] == 4:  # includes ID column
                linking_data = txy_pts[:, :3]  # frame, x, y only
                point_ids = txy_pts[:, 3]      # preserve IDs separately
            else:
                linking_data = txy_pts
                point_ids = np.arange(len(txy_pts))  # create sequential IDs

            points = Points(linking_data)
            points.link_pts(self.parameters.max_gap_frames, self.parameters.max_link_distance)

            # Store the IDs for later use
            points.point_ids = point_ids

            # Load image for intensity extraction
            if os.path.exists(file_path):
                A = skio.imread(file_path, plugin='tifffile')
                points.getIntensities(A)

            self.log_message(f"    Created {len(points.tracks)} tracks")
            return points

        except Exception as e:
            self.log_message(f"    Error linking particles: {e}")
            return None

    def calculate_features(self, points):
        """Calculate track features - UPDATED to preserve IDs"""
        try:
            # Convert tracks to DataFrame
            tracks_data = []

            for track_idx, track in enumerate(points.tracks):
                if len(track) < self.parameters.min_track_segments:
                    continue

                for pt_idx in track:
                    pt = points.txy_pts[pt_idx]
                    intensity = points.intensities[pt_idx] if pt_idx < len(points.intensities) else 0

                    # Get original ID if available
                    original_id = points.point_ids[pt_idx] if hasattr(points, 'point_ids') and pt_idx < len(points.point_ids) else pt_idx

                    tracks_data.append({
                        'track_number': track_idx,
                        'frame': int(pt[0]),
                        'x': pt[1],
                        'y': pt[2],
                        'intensity': intensity,
                        'id': original_id  # preserve original ID
                    })

            if not tracks_data:
                self.log_message("    No valid tracks found")
                return None

            tracks_df = pd.DataFrame(tracks_data)
            tracks_df['n_segments'] = tracks_df.groupby('track_number')['track_number'].transform('count')

            # Calculate features for each track (rest of method unchanged)
            feature_data = []
            track_numbers = tracks_df['track_number'].unique()

            for track_num in track_numbers:
                track_data = tracks_df[tracks_df['track_number'] == track_num]
                points_array = track_data[['x', 'y']].to_numpy()

                try:
                    # Calculate features
                    rg, asymmetry, skewness, kurtosis = FeatureCalculator.radius_gyration_asymmetry(track_data)
                    fractal_dim = FeatureCalculator.fractal_dimension(points_array)
                    net_disp, efficiency = FeatureCalculator.net_displacement_efficiency(points_array)

                    # Calculate straightness if enabled
                    straight_mean = 0
                    if self.parameters.enable_straightness_analysis:
                        sin_mean, _, cos_mean, _ = FeatureCalculator.summed_sines_cosines(points_array)
                        straight_mean = cos_mean

                    # Intensity statistics
                    intensity_mean = track_data['intensity'].mean()
                    intensity_std = track_data['intensity'].std()

                    feature_data.append({
                        'track_number': track_num,
                        'radius_gyration': rg,
                        'asymmetry': asymmetry,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'fracDimension': fractal_dim,
                        'netDispl': net_disp,
                        'efficiency': efficiency,
                        'Straight': straight_mean,
                        'track_intensity_mean': intensity_mean,
                        'track_intensity_std': intensity_std
                    })

                except Exception as e:
                    self.log_message(f"    Error calculating features for track {track_num}: {e}")
                    continue

            # Merge features
            if feature_data:
                features_df = pd.DataFrame(feature_data)
                tracks_df = tracks_df.merge(features_df, on='track_number', how='left')

            self.log_message(f"    Calculated features for {len(feature_data)} tracks")
            return tracks_df

        except Exception as e:
            self.log_message(f"    Error calculating features: {e}")
            return None

    # [Add all remaining helper methods like add_lag_features, add_velocity_analysis, etc.]
    # [Include save_parameters, load_parameters, update_gui_from_parameters methods]

    def save_parameters(self):
        """Save parameters to file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json)")
        if file_path:
            self.update_parameters()
            with open(file_path, 'w') as f:
                json.dump(self.parameters.to_dict(), f, indent=2)
            g.alert("Parameters saved successfully")

    def load_parameters(self):
        """Load parameters from file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    param_dict = json.load(f)
                self.parameters.from_dict(param_dict)
                self.update_gui_from_parameters()
                g.alert("Parameters loaded successfully")
            except Exception as e:
                g.alert(f"Error loading parameters: {e}")

    def update_gui_from_parameters(self):
        """Update GUI from parameters"""
        self.pixel_size_spin.setValue(self.parameters.pixel_size)
        self.frame_length_spin.setValue(self.parameters.frame_length)
        self.max_gap_spin.setValue(self.parameters.max_gap_frames)
        self.max_dist_spin.setValue(self.parameters.max_link_distance)
        self.min_segments_spin.setValue(self.parameters.min_track_segments)
        self.rg_threshold_spin.setValue(self.parameters.rg_mobility_threshold)

        if self.parameters.experiment_name:
            self.experiment_name_edit.setCurrentText(self.parameters.experiment_name)
        if self.parameters.training_data_path:
            self.train_path_edit.setText(self.parameters.training_data_path)
            self.update_training_data_status()

        # Update auto-detect setting
        self.auto_detect_checkbox.setChecked(self.parameters.auto_detect_experiment_names)
        self.on_auto_detect_toggled()  # Update UI state

        # Update checkboxes
        self.nn_checkbox.setChecked(self.parameters.enable_nearest_neighbors)
        self.svm_checkbox.setChecked(self.parameters.enable_svm_classification)
        self.velocity_checkbox.setChecked(self.parameters.enable_velocity_analysis)
        self.diffusion_checkbox.setChecked(self.parameters.enable_diffusion_analysis)
        self.bg_checkbox.setChecked(self.parameters.enable_background_subtraction)
        self.loc_error_checkbox.setChecked(self.parameters.enable_localization_error)
        self.straightness_checkbox.setChecked(self.parameters.enable_straightness_analysis)

        # Enhanced analysis checkboxes
        self.direction_checkbox.setChecked(self.parameters.enable_direction_analysis)
        self.distance_diff_checkbox.setChecked(self.parameters.enable_distance_differential)
        self.missing_points_checkbox.setChecked(self.parameters.enable_missing_points_integration)
        self.interpolation_checkbox.setChecked(self.parameters.enable_enhanced_interpolation)
        self.full_interpolation_checkbox.setChecked(self.parameters.enable_full_track_interpolation)
        self.separate_interpolated_file_checkbox.setChecked(self.parameters.save_separate_interpolated_file)
        self.extend_to_full_recording_checkbox.setChecked(self.parameters.extend_interpolation_to_full_recording)

        # ROI background subtraction options
        self.bg_frame_specific_checkbox.setChecked(self.parameters.roi_frame_specific_background)
        self.on_bg_subtraction_toggled()  # Update enable/disable state

        # Geometric analysis checkboxes
        self.geometric_enable_checkbox.setChecked(self.parameters.enable_geometric_analysis)
        self.geometric_method_simple.setChecked(self.parameters.geometric_rg_method == 'simple')
        self.geometric_srg_cutoff_spin.setValue(self.parameters.geometric_srg_cutoff)
        self.geometric_linear_enable_checkbox.setChecked(self.parameters.enable_geometric_linear_classification)
        self.geometric_directionality_spin.setValue(self.parameters.geometric_directionality_threshold)
        self.geometric_perpendicular_spin.setValue(self.parameters.geometric_perpendicular_threshold)
        self.geometric_cutoff_length_spin.setValue(self.parameters.geometric_cutoff_length)

        # Autocorrelation analysis checkboxes
        self.autocorr_enable_checkbox.setChecked(self.parameters.enable_autocorrelation_analysis)
        self.autocorr_time_interval_spin.setValue(self.parameters.autocorr_time_interval)
        self.autocorr_num_intervals_spin.setValue(self.parameters.autocorr_num_intervals)
        self.autocorr_min_length_spin.setValue(self.parameters.autocorr_min_track_length)
        self.autocorr_show_individual_checkbox.setChecked(self.parameters.autocorr_show_individual_tracks)
        self.autocorr_max_tracks_spin.setValue(self.parameters.autocorr_max_tracks_plot)
        self.autocorr_save_plots_checkbox.setChecked(self.parameters.autocorr_save_plots)
        self.autocorr_save_data_checkbox.setChecked(self.parameters.autocorr_save_data)

        # Update enable/disable states
        self.on_geometric_enable_toggled()
        self.on_autocorr_enable_toggled()
        self.on_full_interpolation_toggled()

        # Call the enhanced version with trackpy support
        self.update_gui_from_parameters_with_mixed_motion()

    def add_lag_features(self, tracks_df):
        """Add lag displacement features"""
        try:
            # Sort by track and frame
            tracks_df = tracks_df.sort_values(['track_number', 'frame'])

            # Calculate lag displacements
            tracks_df = tracks_df.assign(x2=tracks_df.groupby('track_number')['x'].shift(-1))
            tracks_df = tracks_df.assign(y2=tracks_df.groupby('track_number')['y'].shift(-1))

            tracks_df['distance'] = np.sqrt(
                (tracks_df['x2'] - tracks_df['x'])**2 +
                (tracks_df['y2'] - tracks_df['y'])**2
            )

            # Mask final positions
            tracks_df['mask'] = True
            tracks_df.loc[tracks_df.groupby('track_number').tail(1).index, 'mask'] = False

            tracks_df['lag'] = tracks_df['distance'].where(tracks_df['mask'])

            # Add track statistics
            tracks_df['meanLag'] = tracks_df.groupby('track_number')['lag'].transform('mean')
            tracks_df['track_length'] = tracks_df.groupby('track_number')['lag'].transform('sum')

            # Scaled radius of gyration features
            tracks_df['radius_gyration_scaled'] = tracks_df['radius_gyration'] / tracks_df['meanLag']
            tracks_df['radius_gyration_scaled_nSegments'] = tracks_df['radius_gyration'] / tracks_df['n_segments']
            tracks_df['radius_gyration_scaled_trackLength'] = tracks_df['radius_gyration'] / tracks_df['track_length']

            # Mobility classification based on RG ratio
            tracks_df['radius_gyration_ratio_to_mean_step_size'] = np.sqrt(
                math.pi / 2 * tracks_df['radius_gyration'] / tracks_df['meanLag']
            )

            threshold = self.parameters.rg_mobility_threshold
            tracks_df['radius_gyration_mobility_threshold'] = (
                tracks_df['radius_gyration_ratio_to_mean_step_size'] > threshold
            ).astype(int)

            # Clean up temporary columns
            tracks_df = tracks_df.drop(columns=['x2', 'y2', 'distance', 'mask'], errors='ignore')

            return tracks_df

        except Exception as e:
            self.log_message(f"    Error adding lag features: {e}")
            return tracks_df

    def add_velocity_analysis(self, tracks_df):
        """Add velocity analysis"""
        try:
            tracks_df = tracks_df.sort_values(['track_number', 'frame'])

            # Calculate positions relative to track origin
            for track_num in tracks_df['track_number'].unique():
                track_mask = tracks_df['track_number'] == track_num
                track_data = tracks_df[track_mask]

                if len(track_data) == 0:
                    continue

                # Origin at first position
                origin_x = track_data['x'].iloc[0]
                origin_y = track_data['y'].iloc[0]

                tracks_df.loc[track_mask, 'zeroed_X'] = track_data['x'] - origin_x
                tracks_df.loc[track_mask, 'zeroed_Y'] = track_data['y'] - origin_y

                # Distance from origin
                tracks_df.loc[track_mask, 'distanceFromOrigin'] = np.sqrt(
                    tracks_df.loc[track_mask, 'zeroed_X']**2 +
                    tracks_df.loc[track_mask, 'zeroed_Y']**2
                )

                # Lag number
                tracks_df.loc[track_mask, 'lagNumber'] = track_data['frame'] - track_data['frame'].iloc[0]

            # Time differences and velocity
            tracks_df['dt'] = tracks_df.groupby('track_number')['frame'].diff() * self.parameters.frame_length
            tracks_df['velocity'] = tracks_df['lag'] / tracks_df['dt']
            tracks_df['meanVelocity'] = tracks_df.groupby('track_number')['velocity'].transform('mean')

            # Direction analysis
            tracks_df['direction_Relative_To_Origin'] = np.degrees(
                np.arctan2(tracks_df['zeroed_Y'], tracks_df['zeroed_X'])
            ) % 360

            return tracks_df

        except Exception as e:
            self.log_message(f"    Error in velocity analysis: {e}")
            return tracks_df

    def add_diffusion_analysis(self, tracks_df):
        """Add diffusion analysis"""
        try:
            # Squared displacement
            tracks_df['d_squared'] = tracks_df['distanceFromOrigin']**2
            tracks_df['lag_squared'] = tracks_df['lag']**2

            return tracks_df

        except Exception as e:
            self.log_message(f"    Error in diffusion analysis: {e}")
            return tracks_df

    def add_localization_error(self, tracks_df):
        """Add localization error analysis"""
        try:
            # Calculate track mean positions
            tracks_df['mean_X'] = tracks_df.groupby('track_number')['x'].transform('mean')
            tracks_df['mean_Y'] = tracks_df.groupby('track_number')['y'].transform('mean')

            # Distance from mean position
            tracks_df['distanceFromMean'] = np.sqrt(
                (tracks_df['mean_X'] - tracks_df['x'])**2 +
                (tracks_df['mean_Y'] - tracks_df['y'])**2
            )

            # Mean localization distance for each track
            tracks_df['meanLocDistanceFromCenter'] = tracks_df.groupby('track_number')['distanceFromMean'].transform('mean')

            return tracks_df

        except Exception as e:
            self.log_message(f"    Error in localization error analysis: {e}")
            return tracks_df

    def add_background_subtraction(self, tracks_df, file_path):
        """Add background subtraction using ROI with proper coordinate mapping and frame-specific options"""
        try:
            # Look for ROI file with multiple naming conventions
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            roi_file_candidates = [
                file_path.replace('.tif', '_ROI.txt'),
                os.path.join(os.path.dirname(file_path), f'ROI_{base_name}.txt'),
                file_path.replace('.tif', '.txt'),
                os.path.join(os.path.dirname(file_path), f'{base_name}_ROI.txt')
            ]

            roi_file_path = None
            for candidate in roi_file_candidates:
                if os.path.exists(candidate):
                    roi_file_path = candidate
                    break

            self.log_message(f"    Looking for ROI file...")
            if roi_file_path:
                self.log_message(f"    Found ROI file: {os.path.basename(roi_file_path)}")

            # Parse ROI file
            roi_coords = ROIBackgroundSubtractor.parse_roi_file(roi_file_path)

            if roi_coords is None:
                self.log_message("    No valid ROI file found, skipping background subtraction")
                # Add placeholder columns
                tracks_df['roi_intensity'] = 0
                tracks_df['camera_black_estimate'] = 0
                tracks_df['intensity_bg_subtracted'] = tracks_df['intensity']
                tracks_df['background_subtracted'] = False
                tracks_df['background_method'] = 'none'
                tracks_df['background_signal_used'] = 0.0
                tracks_df['background_signal_used'] = 0.0
                return tracks_df

            # Load and transform image data using same method as Points class
            self.log_message("    Loading image data for ROI intensity calculation...")

            if not os.path.exists(file_path):
                self.log_message(f"    Warning: Image file not found: {file_path}")
                tracks_df['roi_intensity'] = 0
                tracks_df['camera_black_estimate'] = 0
                tracks_df['intensity_bg_subtracted'] = tracks_df['intensity']
                tracks_df['background_subtracted'] = False
                tracks_df['background_method'] = 'none'
                return tracks_df

            # Load image with same transformations as Points.getIntensities
            A = skio.imread(file_path, plugin='tifffile')

            # Apply same transformations as in Points.getIntensities
            A = np.rot90(A, axes=(1,2))
            A = np.fliplr(A)

            self.log_message(f"    Loaded and transformed image: {A.shape}")
            self.log_message(f"    ROI coordinates: ({roi_coords['x_min']},{roi_coords['y_min']}) to ({roi_coords['x_max']},{roi_coords['y_max']})")

            # Get frame-specific setting from parameters
            frame_specific = getattr(self.parameters, 'roi_frame_specific_background', False)
            self.log_message(f"    Background subtraction mode: {'frame-specific' if frame_specific else 'single mean across all frames'}")

            # Calculate ROI intensity (frame-specific or overall mean)
            roi_intensity = ROIBackgroundSubtractor.calculate_roi_intensity(A, roi_coords, frame_specific=frame_specific)

            # Estimate camera black level
            camera_black = ROIBackgroundSubtractor.estimate_camera_black_level(A)

            # Perform background subtraction
            if (frame_specific and isinstance(roi_intensity, np.ndarray) and len(roi_intensity) > 0) or \
               (not frame_specific and roi_intensity > 0):

                # Apply background subtraction and get frame-specific values if applicable
                intensity_bg_subtracted, frame_specific_backgrounds = ROIBackgroundSubtractor.apply_background_subtraction(
                    tracks_df, roi_intensity, camera_black, frame_specific=frame_specific)

                # Logging
                if frame_specific:
                    self.log_message(f"    Frame-specific background subtraction completed:")
                    self.log_message(f"      ROI intensity range: {roi_intensity.min():.3f} - {roi_intensity.max():.3f}")
                    self.log_message(f"      ROI intensity mean: {roi_intensity.mean():.3f}")
                    self.log_message(f"      Camera black: {camera_black:.3f}")
                    self.log_message(f"      Background signal range: {roi_intensity.min():.3f} - {roi_intensity.max():.3f} (ROI intensities used directly)")

                    # For frame-specific mode, store the actual frame-specific background used for each point
                    roi_intensity_column = tracks_df.apply(
                        lambda row: roi_intensity[int(row['frame'])] if int(row['frame']) < len(roi_intensity) else roi_intensity.mean(),
                        axis=1
                    )
                else:
                    self.log_message(f"    Single-value background subtraction completed:")
                    self.log_message(f"      ROI intensity: {roi_intensity:.3f}")
                    self.log_message(f"      Camera black: {camera_black:.3f}")
                    self.log_message(f"      Background signal: {roi_intensity:.3f} (ROI intensity used directly)")
                    roi_intensity_column = float(roi_intensity)

                self.log_message(f"      Original intensity range: {tracks_df['intensity'].min():.3f} - {tracks_df['intensity'].max():.3f}")
                self.log_message(f"      Background-subtracted range: {intensity_bg_subtracted.min():.3f} - {intensity_bg_subtracted.max():.3f}")

                background_subtracted = True
                background_method = 'frame_specific' if frame_specific else 'single_mean'
            else:
                self.log_message("    Warning: ROI intensity is 0 or invalid, no background subtraction applied")
                intensity_bg_subtracted = tracks_df['intensity']
                background_subtracted = False
                background_method = 'failed'
                roi_intensity_column = 0.0
                frame_specific_backgrounds = None

            # Add new columns
            tracks_df['roi_intensity'] = roi_intensity_column  # Frame-specific values when applicable
            tracks_df['camera_black_estimate'] = camera_black
            tracks_df['intensity_bg_subtracted'] = intensity_bg_subtracted
            tracks_df['background_subtracted'] = background_subtracted
            tracks_df['background_method'] = background_method

            # Add frame-specific background signal column when frame-specific mode is used
            if frame_specific and frame_specific_backgrounds is not None:
                tracks_df['background_signal_used'] = frame_specific_backgrounds
            else:
                # For single mean mode, calculate the background signal used
                if not frame_specific and background_subtracted:
                    background_signal = float(roi_intensity) if isinstance(roi_intensity, (int, float)) else roi_intensity_column
                    tracks_df['background_signal_used'] = background_signal
                else:
                    tracks_df['background_signal_used'] = 0.0

            return tracks_df

        except Exception as e:
            self.log_message(f"    Error in background subtraction: {e}")
            import traceback
            traceback.print_exc()

            # Add placeholder columns on error
            tracks_df['roi_intensity'] = 0
            tracks_df['camera_black_estimate'] = 0
            tracks_df['intensity_bg_subtracted'] = tracks_df['intensity']
            tracks_df['background_subtracted'] = False
            tracks_df['background_method'] = 'error'
            tracks_df['background_signal_used'] = 0.0

            return tracks_df


    # Add this method to the SPTBatchAnalysis class for ROI validation

    def validate_roi_file(self, file_path):
        """Validate ROI file and optionally create preview"""
        try:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            roi_file_candidates = [
                file_path.replace('.tif', '_ROI.txt'),
                os.path.join(os.path.dirname(file_path), f'ROI_{base_name}.txt'),
                file_path.replace('.tif', '.txt'),
                os.path.join(os.path.dirname(file_path), f'{base_name}_ROI.txt')
            ]

            for roi_file_path in roi_file_candidates:
                if os.path.exists(roi_file_path):
                    roi_coords = ROIBackgroundSubtractor.parse_roi_file(roi_file_path)
                    if roi_coords:
                        self.log_message(f"    Valid ROI file: {os.path.basename(roi_file_path)}")
                        self.log_message(f"    ROI: ({roi_coords['x_min']},{roi_coords['y_min']}) to ({roi_coords['x_max']},{roi_coords['y_max']})")
                        self.log_message(f"    Size: {roi_coords['width']} x {roi_coords['height']} pixels")
                        return True

            self.log_message("    No valid ROI file found")
            return False

        except Exception as e:
            self.log_message(f"    Error validating ROI file: {e}")
            return False


    def on_trackpy_type_changed(self):
        """Handle trackpy type change"""
        current_type = self.trackpy_type_combo.currentText()
        show_advanced = 'adaptive' in current_type
        self.trackpy_advanced_group.setVisible(show_advanced)

    # def update_parameters_with_trackpy(self):
    #     """Update parameters from GUI including trackpy options"""
    #     # Basic parameters
    #     self.parameters.pixel_size = self.pixel_size_spin.value()
    #     self.parameters.frame_length = self.frame_length_spin.value()
    #     self.parameters.min_track_segments = self.min_segments_spin.value()
    #     self.parameters.rg_mobility_threshold = self.rg_threshold_spin.value()
    #     self.parameters.experiment_name = self.experiment_name_edit.currentText()
    #     self.parameters.auto_detect_experiment_names = self.auto_detect_checkbox.isChecked()

    #     # Linking method selection and parameters
    #     if self.builtin_linking_radio.isChecked():
    #         self.parameters.linking_method = 'builtin'
    #         self.parameters.max_gap_frames = self.max_gap_spin.value()
    #         self.parameters.max_link_distance = self.max_dist_spin.value()
    #     else:
    #         self.parameters.linking_method = 'trackpy'
    #         self.parameters.trackpy_link_distance = self.trackpy_distance_spin.value()
    #         self.parameters.trackpy_memory = self.trackpy_memory_spin.value()
    #         self.parameters.trackpy_linking_type = self.trackpy_type_combo.currentText()
    #         self.parameters.trackpy_max_search_distance = self.trackpy_max_search_spin.value()
    #         self.parameters.trackpy_adaptive_stop = self.trackpy_adaptive_stop_spin.value()
    #         self.parameters.trackpy_adaptive_step = self.trackpy_adaptive_step_spin.value()

    #     # Update analysis step flags
    #     self.parameters.enable_nearest_neighbors = self.nn_checkbox.isChecked()
    #     self.parameters.enable_svm_classification = self.svm_checkbox.isChecked()
    #     self.parameters.enable_velocity_analysis = self.velocity_checkbox.isChecked()
    #     self.parameters.enable_diffusion_analysis = self.diffusion_checkbox.isChecked()
    #     self.parameters.enable_background_subtraction = self.bg_checkbox.isChecked()
    #     self.parameters.enable_localization_error = self.loc_error_checkbox.isChecked()
    #     self.parameters.enable_straightness_analysis = self.straightness_checkbox.isChecked()

    #     # Enhanced analysis step flags
    #     self.parameters.enable_direction_analysis = self.direction_checkbox.isChecked()
    #     self.parameters.enable_distance_differential = self.distance_diff_checkbox.isChecked()
    #     self.parameters.enable_missing_points_integration = self.missing_points_checkbox.isChecked()
    #     self.parameters.enable_enhanced_interpolation = self.interpolation_checkbox.isChecked()

    #     # ROI background subtraction options
    #     self.parameters.roi_frame_specific_background = self.bg_frame_specific_checkbox.isChecked()

    #     # Geometric analysis parameters
    #     self.parameters.enable_geometric_analysis = self.geometric_enable_checkbox.isChecked()
    #     self.parameters.geometric_rg_method = 'simple' if self.geometric_method_simple.isChecked() else 'tensor'
    #     self.parameters.geometric_srg_cutoff = self.geometric_srg_cutoff_spin.value()
    #     self.parameters.enable_geometric_linear_classification = self.geometric_linear_enable_checkbox.isChecked()
    #     self.parameters.geometric_directionality_threshold = self.geometric_directionality_spin.value()
    #     self.parameters.geometric_perpendicular_threshold = self.geometric_perpendicular_spin.value()
    #     self.parameters.geometric_cutoff_length = self.geometric_cutoff_length_spin.value()

    #     # Autocorrelation analysis parameters
    #     self.parameters.enable_autocorrelation_analysis = self.autocorr_enable_checkbox.isChecked()
    #     self.parameters.autocorr_time_interval = self.autocorr_time_interval_spin.value()
    #     self.parameters.autocorr_num_intervals = self.autocorr_num_intervals_spin.value()
    #     self.parameters.autocorr_min_track_length = self.autocorr_min_length_spin.value()
    #     self.parameters.autocorr_show_individual_tracks = self.autocorr_show_individual_checkbox.isChecked()
    #     self.parameters.autocorr_max_tracks_plot = self.autocorr_max_tracks_spin.value()
    #     self.parameters.autocorr_save_plots = self.autocorr_save_plots_checkbox.isChecked()
    #     self.parameters.autocorr_save_data = self.autocorr_save_data_checkbox.isChecked()

    #     # Detection parameters
    #     self.parameters.enable_detection = self.detection_enable_checkbox.isChecked()
    #     self.parameters.detection_psf_sigma = self.detection_psf_sigma_spin.value()
    #     self.parameters.detection_alpha_threshold = self.detection_alpha_spin.value()
    #     self.parameters.detection_min_intensity = self.detection_min_intensity_spin.value()
    #     self.parameters.detection_skip_existing = self.detection_skip_existing_checkbox.isChecked()
    #     self.parameters.detection_show_results = self.detection_show_results_checkbox.isChecked()

    # def update_gui_from_parameters_with_trackpy(self):
    #     """Update GUI from parameters including trackpy options"""
    #     # Basic parameters
    #     self.pixel_size_spin.setValue(self.parameters.pixel_size)
    #     self.frame_length_spin.setValue(self.parameters.frame_length)
    #     self.min_segments_spin.setValue(self.parameters.min_track_segments)
    #     self.rg_threshold_spin.setValue(self.parameters.rg_mobility_threshold)

    #     if self.parameters.experiment_name:
    #         self.experiment_name_edit.setCurrentText(self.parameters.experiment_name)
    #     if self.parameters.training_data_path:
    #         self.train_path_edit.setText(self.parameters.training_data_path)
    #         self.update_training_data_status()

    #     # Update auto-detect setting
    #     self.auto_detect_checkbox.setChecked(self.parameters.auto_detect_experiment_names)
    #     self.on_auto_detect_toggled()  # Update UI state

    #     # Linking method and parameters
    #     if self.parameters.linking_method == 'builtin':
    #         self.builtin_linking_radio.setChecked(True)
    #         self.max_gap_spin.setValue(self.parameters.max_gap_frames)
    #         self.max_dist_spin.setValue(self.parameters.max_link_distance)
    #     else:
    #         self.trackpy_linking_radio.setChecked(True)
    #         self.trackpy_distance_spin.setValue(self.parameters.trackpy_link_distance)
    #         self.trackpy_memory_spin.setValue(self.parameters.trackpy_memory)
    #         self.trackpy_type_combo.setCurrentText(self.parameters.trackpy_linking_type)
    #         self.trackpy_max_search_spin.setValue(self.parameters.trackpy_max_search_distance)
    #         self.trackpy_adaptive_stop_spin.setValue(self.parameters.trackpy_adaptive_stop)
    #         self.trackpy_adaptive_step_spin.setValue(self.parameters.trackpy_adaptive_step)

    #     # Update linking method UI state
    #     self.on_linking_method_changed()
    #     self.on_trackpy_type_changed()

    #     # Update checkboxes
    #     self.nn_checkbox.setChecked(self.parameters.enable_nearest_neighbors)
    #     self.svm_checkbox.setChecked(self.parameters.enable_svm_classification)
    #     self.velocity_checkbox.setChecked(self.parameters.enable_velocity_analysis)
    #     self.diffusion_checkbox.setChecked(self.parameters.enable_diffusion_analysis)
    #     self.bg_checkbox.setChecked(self.parameters.enable_background_subtraction)
    #     self.loc_error_checkbox.setChecked(self.parameters.enable_localization_error)
    #     self.straightness_checkbox.setChecked(self.parameters.enable_straightness_analysis)

    #     # Enhanced analysis checkboxes
    #     self.direction_checkbox.setChecked(self.parameters.enable_direction_analysis)
    #     self.distance_diff_checkbox.setChecked(self.parameters.enable_distance_differential)
    #     self.missing_points_checkbox.setChecked(self.parameters.enable_missing_points_integration)
    #     self.interpolation_checkbox.setChecked(self.parameters.enable_enhanced_interpolation)

    #     # ROI background subtraction options
    #     self.bg_frame_specific_checkbox.setChecked(self.parameters.roi_frame_specific_background)
    #     self.on_bg_subtraction_toggled()  # Update enable/disable state

    #     # Geometric analysis checkboxes
    #     self.geometric_enable_checkbox.setChecked(self.parameters.enable_geometric_analysis)
    #     self.geometric_method_simple.setChecked(self.parameters.geometric_rg_method == 'simple')
    #     self.geometric_srg_cutoff_spin.setValue(self.parameters.geometric_srg_cutoff)
    #     self.geometric_linear_enable_checkbox.setChecked(self.parameters.enable_geometric_linear_classification)
    #     self.geometric_directionality_spin.setValue(self.parameters.geometric_directionality_threshold)
    #     self.geometric_perpendicular_spin.setValue(self.parameters.geometric_perpendicular_threshold)
    #     self.geometric_cutoff_length_spin.setValue(self.parameters.geometric_cutoff_length)

    #     # Autocorrelation analysis checkboxes
    #     self.autocorr_enable_checkbox.setChecked(self.parameters.enable_autocorrelation_analysis)
    #     self.autocorr_time_interval_spin.setValue(self.parameters.autocorr_time_interval)
    #     self.autocorr_num_intervals_spin.setValue(self.parameters.autocorr_num_intervals)
    #     self.autocorr_min_length_spin.setValue(self.parameters.autocorr_min_track_length)
    #     self.autocorr_show_individual_checkbox.setChecked(self.parameters.autocorr_show_individual_tracks)
    #     self.autocorr_max_tracks_spin.setValue(self.parameters.autocorr_max_tracks_plot)
    #     self.autocorr_save_plots_checkbox.setChecked(self.parameters.autocorr_save_plots)
    #     self.autocorr_save_data_checkbox.setChecked(self.parameters.autocorr_save_data)

    #     # Detection parameters
    #     self.detection_enable_checkbox.setChecked(self.parameters.enable_detection)
    #     self.detection_psf_sigma_spin.setValue(self.parameters.detection_psf_sigma)
    #     self.detection_alpha_spin.setValue(self.parameters.detection_alpha_threshold)
    #     self.detection_min_intensity_spin.setValue(self.parameters.detection_min_intensity)
    #     self.detection_skip_existing_checkbox.setChecked(self.parameters.detection_skip_existing)
    #     self.detection_show_results_checkbox.setChecked(self.parameters.detection_show_results)

    #     # Update enable/disable states
    #     self.on_geometric_enable_toggled()
    #     self.on_autocorr_enable_toggled()
    #     self.on_detection_enable_toggled()

    def update_parameters_with_mixed_motion(self):
        """Update parameters from GUI including mixed motion and new algorithm options"""
        # Basic parameters
        self.parameters.pixel_size = self.pixel_size_spin.value()
        self.parameters.frame_length = self.frame_length_spin.value()
        self.parameters.min_track_segments = self.min_segments_spin.value()
        self.parameters.rg_mobility_threshold = self.rg_threshold_spin.value()
        self.parameters.experiment_name = self.experiment_name_edit.currentText()
        self.parameters.auto_detect_experiment_names = self.auto_detect_checkbox.isChecked()

        # Linking method selection and parameters
        if self.builtin_linking_radio.isChecked():
            self.parameters.linking_method = 'builtin'
            # NEW: Algorithm selection
            if hasattr(self, 'iterative_radio'):
                self.parameters.builtin_linking_algorithm = 'iterative' if self.iterative_radio.isChecked() else 'recursive'
            if hasattr(self, 'recursion_limit_spin'):
                self.parameters.recursive_depth_limit = self.recursion_limit_spin.value()
            # Standard parameters
            self.parameters.max_gap_frames = self.max_gap_spin.value()
            self.parameters.max_link_distance = self.max_dist_spin.value()
        elif self.trackpy_linking_radio.isChecked():
            self.parameters.linking_method = 'trackpy'
            self.parameters.trackpy_link_distance = self.trackpy_distance_spin.value()
            self.parameters.trackpy_memory = self.trackpy_memory_spin.value()
            self.parameters.trackpy_linking_type = self.trackpy_type_combo.currentText()
            self.parameters.trackpy_max_search_distance = self.trackpy_max_search_spin.value()
            self.parameters.trackpy_adaptive_stop = self.trackpy_adaptive_stop_spin.value()
            self.parameters.trackpy_adaptive_step = self.trackpy_adaptive_step_spin.value()
        else:  # U-Track
            self.parameters.linking_method = 'utrack'
            self.parameters.utrack_max_linking_distance = self.utrack_max_distance_spin.value()
            self.parameters.utrack_max_gap_frames = self.utrack_max_gap_spin.value()
            self.parameters.utrack_motion_model = self.motion_model_combo.currentText()
            self.parameters.utrack_auto_linking_distance = self.utrack_auto_distance_checkbox.isChecked()
            self.parameters.utrack_enable_merging = self.utrack_enable_merging_checkbox.isChecked()
            self.parameters.utrack_enable_splitting = self.utrack_enable_splitting_checkbox.isChecked()

            # Mixed motion parameters
            self.parameters.utrack_enable_iterative_smoothing = self.utrack_iterative_smoothing_checkbox.isChecked()
            self.parameters.utrack_num_tracking_rounds = self.utrack_tracking_rounds_spin.value()
            self.parameters.utrack_regime_sensitivity = self.utrack_regime_sensitivity_spin.value()
            self.parameters.utrack_adaptive_search_radius = self.utrack_adaptive_search_checkbox.isChecked()
            self.parameters.utrack_min_regime_length = self.utrack_min_regime_spin.value()

            # Transition probabilities
            self.parameters.utrack_trans_prob_b2l = self.brownian_to_linear_spin.value()
            self.parameters.utrack_trans_prob_l2b = self.linear_to_brownian_spin.value()
            self.parameters.utrack_brownian_noise_mult = self.brownian_noise_spin.value()
            self.parameters.utrack_linear_velocity_persist = self.velocity_persistence_spin.value()

    def update_gui_from_parameters_with_mixed_motion(self):
        """Update GUI from parameters including mixed motion and new algorithm options"""
        # Basic parameters
        self.pixel_size_spin.setValue(self.parameters.pixel_size)
        self.frame_length_spin.setValue(self.parameters.frame_length)
        self.min_segments_spin.setValue(self.parameters.min_track_segments)
        self.rg_threshold_spin.setValue(self.parameters.rg_mobility_threshold)

        if self.parameters.experiment_name:
            self.experiment_name_edit.setCurrentText(self.parameters.experiment_name)
        if self.parameters.training_data_path:
            self.train_path_edit.setText(self.parameters.training_data_path)
            self.update_training_data_status()

        # Update auto-detect setting
        self.auto_detect_checkbox.setChecked(self.parameters.auto_detect_experiment_names)
        self.on_auto_detect_toggled()

        # Linking method and parameters
        if self.parameters.linking_method == 'builtin':
            self.builtin_linking_radio.setChecked(True)
            self.max_gap_spin.setValue(self.parameters.max_gap_frames)
            self.max_dist_spin.setValue(self.parameters.max_link_distance)
            # NEW: Algorithm selection
            if hasattr(self, 'iterative_radio'):
                algorithm = getattr(self.parameters, 'builtin_linking_algorithm', 'iterative')
                if algorithm == 'iterative':
                    self.iterative_radio.setChecked(True)
                else:
                    self.recursive_radio.setChecked(True)
            if hasattr(self, 'recursion_limit_spin'):
                limit = getattr(self.parameters, 'recursive_depth_limit', 1000)
                self.recursion_limit_spin.setValue(limit)
        elif self.parameters.linking_method == 'trackpy':
            self.trackpy_linking_radio.setChecked(True)
            self.trackpy_distance_spin.setValue(self.parameters.trackpy_link_distance)
            self.trackpy_memory_spin.setValue(self.parameters.trackpy_memory)
            self.trackpy_type_combo.setCurrentText(self.parameters.trackpy_linking_type)
            self.trackpy_max_search_spin.setValue(self.parameters.trackpy_max_search_distance)
            self.trackpy_adaptive_stop_spin.setValue(self.parameters.trackpy_adaptive_stop)
            self.trackpy_adaptive_step_spin.setValue(self.parameters.trackpy_adaptive_step)
        else:  # U-Track
            self.utrack_linking_radio.setChecked(True)
            self.utrack_max_distance_spin.setValue(self.parameters.utrack_max_linking_distance)
            self.utrack_max_gap_spin.setValue(self.parameters.utrack_max_gap_frames)
            self.motion_model_combo.setCurrentText(self.parameters.utrack_motion_model)
            self.utrack_auto_distance_checkbox.setChecked(self.parameters.utrack_auto_linking_distance)
            self.utrack_enable_merging_checkbox.setChecked(self.parameters.utrack_enable_merging)
            self.utrack_enable_splitting_checkbox.setChecked(self.parameters.utrack_enable_splitting)

            # Mixed motion parameters
            self.utrack_iterative_smoothing_checkbox.setChecked(self.parameters.utrack_enable_iterative_smoothing)
            self.utrack_tracking_rounds_spin.setValue(self.parameters.utrack_num_tracking_rounds)
            self.utrack_regime_sensitivity_spin.setValue(self.parameters.utrack_regime_sensitivity)
            self.utrack_adaptive_search_checkbox.setChecked(self.parameters.utrack_adaptive_search_radius)
            self.utrack_min_regime_spin.setValue(self.parameters.utrack_min_regime_length)

            # Transition probabilities
            self.brownian_to_linear_spin.setValue(self.parameters.utrack_trans_prob_b2l)
            self.linear_to_brownian_spin.setValue(self.parameters.utrack_trans_prob_l2b)
            self.brownian_noise_spin.setValue(self.parameters.utrack_brownian_noise_mult)
            self.velocity_persistence_spin.setValue(self.parameters.utrack_linear_velocity_persist)

        # Update UI state
        self.on_linking_method_changed()
        self.on_trackpy_type_changed()
        self.on_motion_model_changed()
        self.on_algorithm_changed()  # NEW: Initialize algorithm-specific UI state


    def link_particles_utrack_mixed_motion(self, txy_pts, file_path):
        """Link particles using U-Track with mixed motion models"""
        try:
            self.log_message("    Using U-Track with mixed motion models for particle linking...")
            self.log_message(f"    Motion model: {self.parameters.utrack_motion_model}")
            self.log_message(f"    Max linking distance: {self.parameters.utrack_max_linking_distance} pixels")
            self.log_message(f"    Max gap frames: {self.parameters.utrack_max_gap_frames}")

            if self.parameters.utrack_motion_model == 'mixed':
                self.log_message(f"    Mixed motion enabled with {self.parameters.utrack_num_tracking_rounds} tracking rounds")
                self.log_message(f"    Regime sensitivity: {self.parameters.utrack_regime_sensitivity}")
                self.log_message(f"    Transition probabilities: B→L={self.parameters.utrack_trans_prob_b2l}, L→B={self.parameters.utrack_trans_prob_l2b}")

            # Create U-Track adapter with mixed motion support
            adapter = UTrackLinkerAdapter(self.parameters)

            # Perform linking
            points_adapter = adapter.link_particles(txy_pts, file_path)

            if points_adapter is None or points_adapter.recursiveFailure:
                self.log_message("    Mixed motion U-Track linking failed, falling back to built-in method")
                return self.link_particles_builtin(txy_pts, file_path)

            self.log_message(f"    Mixed motion U-Track linking successful: {len(points_adapter.tracks)} tracks")
            return points_adapter

        except Exception as e:
            self.log_message(f"    Error in mixed motion U-Track linking: {e}")
            self.log_message("    Falling back to built-in linking method")
            return self.link_particles_builtin(txy_pts, file_path)

    def link_particles_enhanced_with_mixed_motion(self, txy_pts, file_path):
        """Enhanced particle linking with mixed motion support"""
        try:
            if self.parameters.linking_method == 'trackpy':
                return self.link_particles_trackpy(txy_pts, file_path)
            elif self.parameters.linking_method == 'utrack':
                return self.link_particles_utrack_mixed_motion(txy_pts, file_path)
            else:
                return self.link_particles_builtin(txy_pts, file_path)
        except Exception as e:
            self.log_message(f"    Error in particle linking: {e}")
            return None

        def link_particles_enhanced(self, txy_pts, file_path):
            """Enhanced particle linking with trackpy option"""
            try:
                if self.parameters.linking_method == 'trackpy':
                    return self.link_particles_trackpy(txy_pts, file_path)
                else:
                    return self.link_particles_builtin(txy_pts, file_path)
            except Exception as e:
                self.log_message(f"    Error in particle linking: {e}")
                return None

    def link_particles_trackpy(self, txy_pts, file_path):
        """Link particles using trackpy"""
        try:
            # Extract just frame, x, y for trackpy, preserve IDs separately
            if txy_pts.shape[1] == 4:  # includes ID column
                point_ids = txy_pts[:, 3]      # preserve IDs separately
                locs_data = txy_pts[:, :3]     # frame, x, y only
            else:
                locs_data = txy_pts
                point_ids = np.arange(len(txy_pts))  # create sequential IDs

            # Create DataFrame for trackpy
            locs_df = pd.DataFrame({
                'frame': locs_data[:, 0].astype(int),
                'x': locs_data[:, 1],
                'y': locs_data[:, 2],
                'id': point_ids
            })

            # Use trackpy linking
            linked_df = TrackpyLinker.link_particles_trackpy(locs_df, self.parameters, log_func=self.log_message)

            if linked_df is None:
                self.log_message("    Trackpy linking failed, falling back to built-in method")
                return self.link_particles_builtin(txy_pts, file_path)

            # Convert trackpy results back to Points-like structure for compatibility
            points_like = self.create_points_from_trackpy(linked_df, file_path)

            self.log_message(f"    Trackpy linking successful: {len(points_like.tracks)} tracks")
            return points_like

        except Exception as e:
            self.log_message(f"    Error in trackpy linking: {e}")
            self.log_message("    Falling back to built-in linking method")
            return self.link_particles_builtin(txy_pts, file_path)



    def create_points_from_trackpy(self, linked_df, file_path):
        """Convert trackpy results to Points-like structure for compatibility"""
        class TrackpyPointsAdapter:
            def __init__(self, linked_df, file_path):
                self.linked_df = linked_df.copy().reset_index(drop=True)  # Reset index to ensure sequential numbering
                self.recursiveFailure = False

                # Create tracks list (list of lists of indices)
                self.tracks = []
                for track_id in sorted(linked_df['track_number'].unique()):
                    track_indices = linked_df[linked_df['track_number'] == track_id].index.tolist()
                    self.tracks.append(track_indices)

                # Create txy_pts array
                self.txy_pts = linked_df[['frame', 'x', 'y']].values

                # Store point IDs
                self.point_ids = linked_df['id'].values if 'id' in linked_df.columns else np.arange(len(linked_df))

                # Calculate intensities
                self.intensities = []
                self.calculate_intensities(file_path)

            def calculate_intensities(self, file_path):
                """Calculate intensities from image data"""
                try:
                    if os.path.exists(file_path):
                        A = skio.imread(file_path, plugin='tifffile')
                        # Apply same transformations as built-in method
                        A = np.rot90(A, axes=(1,2))
                        A = np.fliplr(A)

                        n, w, h = A.shape

                        for _, row in self.linked_df.iterrows():
                            frame = int(round(row['frame']))
                            x = int(round(row['x']))
                            y = int(round(row['y']))

                            # 3x3 pixel region - same as built-in method
                            xMin = max(0, x - 1)
                            xMax = min(w, x + 2)
                            yMin = max(0, y - 1)
                            yMax = min(h, y + 2)

                            intensity = np.mean(A[frame][yMin:yMax, xMin:xMax])
                            self.intensities.append(intensity)
                    else:
                        self.intensities = [0.0] * len(self.linked_df)

                except Exception as e:
                    print(f"Error calculating intensities: {e}")
                    self.intensities = [0.0] * len(self.linked_df)

        return TrackpyPointsAdapter(linked_df, file_path)

    def validate_mixed_motion_parameters(self):
        """Validate mixed motion parameters for U-Track"""
        if self.parameters.linking_method == 'utrack' and self.parameters.utrack_motion_model == 'mixed':
            # Validate transition probabilities
            total_transition_prob = (self.parameters.utrack_trans_prob_b2l +
                                   self.parameters.utrack_trans_prob_l2b)
            if total_transition_prob > 0.8:
                self.log_message("    ⚠️ Warning: High transition probabilities may cause unstable tracking")

            # Validate tracking rounds
            if self.parameters.utrack_num_tracking_rounds < 3:
                self.log_message("    ⚠️ Warning: Less than 3 tracking rounds may reduce mixed motion accuracy")
                self.log_message("    💡 Tip: Use 3+ rounds for optimal PMMS performance")

            # Validate regime parameters
            if self.parameters.utrack_min_regime_length < 3:
                self.log_message("    ⚠️ Warning: Very short minimum regime length may cause over-segmentation")

            # Validate sensitivity
            if self.parameters.utrack_regime_sensitivity > 0.9:
                self.log_message("    ⚠️ Warning: Very high regime sensitivity may miss rapid motion switches")
            elif self.parameters.utrack_regime_sensitivity < 0.3:
                self.log_message("    ⚠️ Warning: Very low regime sensitivity may create false regime changes")

    def log_mixed_motion_parameters(self):
        """Log mixed motion specific parameters for debugging"""
        if self.parameters.linking_method == 'utrack':
            self.log_message("  📊 U-Track Mixed Motion Parameters:")
            self.log_message(f"    Motion model: {self.parameters.utrack_motion_model}")
            self.log_message(f"    Max linking distance: {self.parameters.utrack_max_linking_distance} pixels")
            self.log_message(f"    Max gap frames: {self.parameters.utrack_max_gap_frames}")

            if self.parameters.utrack_motion_model == 'mixed':
                self.log_message(f"    🔄 PMMS Configuration:")
                self.log_message(f"      Iterative smoothing: {self.parameters.utrack_enable_iterative_smoothing}")
                self.log_message(f"      Tracking rounds: {self.parameters.utrack_num_tracking_rounds}")
                self.log_message(f"      Regime sensitivity: {self.parameters.utrack_regime_sensitivity}")
                self.log_message(f"      Min regime length: {self.parameters.utrack_min_regime_length}")
                self.log_message(f"      Adaptive search radius: {self.parameters.utrack_adaptive_search_radius}")

                self.log_message(f"    🔀 Motion Transitions:")
                self.log_message(f"      Brownian → Linear: {self.parameters.utrack_trans_prob_b2l}")
                self.log_message(f"      Linear → Brownian: {self.parameters.utrack_trans_prob_l2b}")
                self.log_message(f"      Brownian noise multiplier: {self.parameters.utrack_brownian_noise_mult}")
                self.log_message(f"      Linear velocity persistence: {self.parameters.utrack_linear_velocity_persist}")

            self.log_message(f"    🔧 Advanced Options:")
            self.log_message(f"      Auto-adapt distance: {self.parameters.utrack_auto_linking_distance}")
            self.log_message(f"      Enable merging: {self.parameters.utrack_enable_merging}")
            self.log_message(f"      Enable splitting: {self.parameters.utrack_enable_splitting}")

    def log_final_track_statistics(self, tracks_df):
        """Log comprehensive statistics about the final tracks"""
        if len(tracks_df) == 0:
            self.log_message("  📈 Final Statistics: No tracks generated")
            return

        n_tracks = len(tracks_df['track_number'].unique())
        n_points = len(tracks_df)

        # Track length statistics
        track_lengths = tracks_df.groupby('track_number').size()
        mean_length = track_lengths.mean()
        median_length = track_lengths.median()
        max_length = track_lengths.max()
        min_length = track_lengths.min()

        self.log_message(f"  📈 Final Track Statistics:")
        self.log_message(f"    Total tracks: {n_tracks}")
        self.log_message(f"    Total points: {n_points}")
        self.log_message(f"    Track lengths - Mean: {mean_length:.1f}, Median: {median_length:.1f}")
        self.log_message(f"    Track lengths - Range: {min_length} to {max_length} points")

        # Linking method specific statistics
        if 'linking_method' in tracks_df.columns:
            linking_methods = tracks_df['linking_method'].unique()
            self.log_message(f"    Linking method(s) used: {', '.join(linking_methods)}")

        if 'motion_model_used' in tracks_df.columns:
            motion_models = tracks_df['motion_model_used'].unique()
            self.log_message(f"    Motion model(s) used: {', '.join(motion_models)}")

        # Mixed motion specific statistics
        if (self.parameters.linking_method == 'utrack' and
            self.parameters.utrack_motion_model == 'mixed' and
            'pmms_enabled' in tracks_df.columns):

            pmms_tracks = tracks_df['pmms_enabled'].sum() if tracks_df['pmms_enabled'].dtype == bool else len(tracks_df[tracks_df['pmms_enabled'] == True])
            self.log_message(f"    🔄 PMMS processing: Applied to {pmms_tracks}/{n_points} points")

            if 'tracking_rounds_used' in tracks_df.columns:
                rounds_used = tracks_df['tracking_rounds_used'].iloc[0] if len(tracks_df) > 0 else 'N/A'
                self.log_message(f"    🔄 Tracking rounds completed: {rounds_used}")

    def validate_linking_method(self):
        """Validate selected linking method is available and fallback if needed"""
        original_method = self.parameters.linking_method

        if self.parameters.linking_method == 'trackpy' and not TrackpyLinker.check_trackpy_available():
            self.log_message("    ⚠️ Trackpy not available, falling back to built-in linking")
            self.parameters.linking_method = 'builtin'

        elif self.parameters.linking_method == 'utrack':
            available, message = check_utrack_availability()
            if not available:
                self.log_message(f"    ⚠️ U-Track not available: {message}")
                self.log_message("    ⚠️ Falling back to built-in linking")
                self.parameters.linking_method = 'builtin'
            else:
                # Additional validation for mixed motion requirements
                if self.parameters.utrack_motion_model == 'mixed':
                    # Check if we have sufficient dependencies for mixed motion
                    try:
                        # Test that mixed motion components are available (they're in the same file)
                        # Just check if the class exists
                        UTrackLinkerAdapter
                        self.log_message("    ✅ Mixed motion U-Track components available")
                    except NameError as e:
                        self.log_message(f"    ⚠️ Mixed motion components not available: {e}")
                        self.log_message("    ⚠️ Falling back to standard U-Track or built-in linking")
                        # Could fallback to standard U-Track with single motion model
                        self.parameters.utrack_motion_model = 'brownian'

        if original_method != self.parameters.linking_method:
            self.log_message(f"    🔄 Linking method changed: {original_method} → {self.parameters.linking_method}")

    def log_mixed_motion_linking_stats(self, points_adapter):
        """Log statistics specific to mixed motion linking results"""
        try:
            if hasattr(points_adapter, 'tracks_df') and len(points_adapter.tracks_df) > 0:
                tracks_df = points_adapter.tracks_df

                # Track length distribution
                track_lengths = tracks_df.groupby('track_id').size()
                self.log_message(f"    📊 Mixed motion linking stats:")
                self.log_message(f"      Track count: {len(track_lengths)}")
                self.log_message(f"      Points per track - Mean: {track_lengths.mean():.1f}, Std: {track_lengths.std():.1f}")

                # Motion regime statistics (if available)
                if hasattr(points_adapter, 'motion_regimes'):
                    regime_counts = {}
                    for regime_list in points_adapter.motion_regimes.values():
                        for regime in regime_list:
                            regime_type = regime.get('type', 'unknown')
                            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1

                    if regime_counts:
                        self.log_message(f"      Motion regimes detected:")
                        for regime_type, count in regime_counts.items():
                            self.log_message(f"        {regime_type}: {count}")

        except Exception as e:
            self.log_message(f"    ⚠️ Could not log mixed motion stats: {e}")

    # ==================== THUNDERSTORM TAB IMPLEMENTATION ====================

    def create_thunderstorm_tab(self):
        """Create thunderSTORM macro generation and execution tab"""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Header (stays at top, not scrollable)
        header = QLabel("ThunderSTORM Macro Generator for ImageJ/Fiji")
        header.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2c5aa0;")
        main_layout.addWidget(header)

        desc = QLabel(
            "Generate ImageJ macros for batch processing STORM/PALM data with ThunderSTORM plugin.\n"
            "ThunderSTORM is a comprehensive ImageJ plugin for single-molecule localization microscopy."
        )
        desc.setWordWrap(True)
        main_layout.addWidget(desc)

        # Create scrollable area for all parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Scrollable content widget
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(15)  # Space between groups

        # === FILE SELECTION ===
        file_group = QGroupBox("File Selection for ThunderSTORM")
        file_layout = QGridLayout(file_group)
        file_layout.setSpacing(8)
        file_layout.setContentsMargins(10, 15, 10, 10)

        # Input directory
        file_layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.ts_input_dir_label = QLabel("No directory selected")
        self.ts_input_dir_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        file_layout.addWidget(self.ts_input_dir_label, 0, 1)

        self.ts_select_input_btn = QPushButton("Browse...")
        self.ts_select_input_btn.clicked.connect(self.select_thunderstorm_input_dir)
        file_layout.addWidget(self.ts_select_input_btn, 0, 2)

        # File pattern
        file_layout.addWidget(QLabel("File Pattern:"), 1, 0)
        self.ts_file_pattern = QComboBox()
        self.ts_file_pattern.setEditable(True)
        self.ts_file_pattern.addItems(['**/*_crop.tif', '**/*_bin10.tif', '**/*_piezo1.tif', '**/*.tif'])
        file_layout.addWidget(self.ts_file_pattern, 1, 1, 1, 2)

        # Output directory
        file_layout.addWidget(QLabel("Macro Save Path:"), 2, 0)
        self.ts_output_macro_label = QLabel("Will be saved in input directory")
        self.ts_output_macro_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        file_layout.addWidget(self.ts_output_macro_label, 2, 1, 1, 2)

        scroll_layout.addWidget(file_group)

        # === FILTER SETTINGS ===
        filter_group = QGroupBox("Image Filtering")
        filter_layout = QFormLayout(filter_group)
        filter_layout.setSpacing(8)
        filter_layout.setContentsMargins(10, 15, 10, 10)

        # Filter type
        self.ts_filter_type = QComboBox()
        self.ts_filter_type.addItems([
            'Wavelet filter (B-Spline)',
            'Gaussian filter',
            'Difference of Gaussians',
            'Lowpass filter',
            'No filter'
        ])
        filter_layout.addRow("Filter Type:", self.ts_filter_type)

        # Wavelet scale
        self.ts_wavelet_scale = QDoubleSpinBox()
        self.ts_wavelet_scale.setRange(1.0, 5.0)
        self.ts_wavelet_scale.setValue(2.0)
        self.ts_wavelet_scale.setSingleStep(0.5)
        filter_layout.addRow("Wavelet Scale:", self.ts_wavelet_scale)

        # Wavelet order
        self.ts_wavelet_order = QSpinBox()
        self.ts_wavelet_order.setRange(1, 5)
        self.ts_wavelet_order.setValue(3)
        filter_layout.addRow("Wavelet Order:", self.ts_wavelet_order)

        scroll_layout.addWidget(filter_group)

        # === DETECTION SETTINGS ===
        detection_group = QGroupBox("Molecular Detection")
        detection_layout = QFormLayout(detection_group)
        detection_layout.setSpacing(8)
        detection_layout.setContentsMargins(10, 15, 10, 10)

        # Detector
        self.ts_detector = QComboBox()
        self.ts_detector.addItems(['Local maximum', 'Centroid of local neighborhood', 'Non-maximum suppression'])
        detection_layout.addRow("Detector:", self.ts_detector)

        # Connectivity
        self.ts_connectivity = QComboBox()
        self.ts_connectivity.addItems(['4-neighbourhood', '8-neighbourhood'])
        detection_layout.addRow("Connectivity:", self.ts_connectivity)

        # Threshold
        self.ts_threshold = QLineEdit("std(Wave.F1)")
        detection_layout.addRow("Threshold Formula:", self.ts_threshold)

        scroll_layout.addWidget(detection_group)

        # === LOCALIZATION SETTINGS ===
        localization_group = QGroupBox("Sub-pixel Localization")
        localization_layout = QFormLayout(localization_group)
        localization_layout.setSpacing(8)
        localization_layout.setContentsMargins(10, 15, 10, 10)

        # Estimator (PSF model)
        self.ts_estimator = QComboBox()
        self.ts_estimator.addItems([
            'PSF: Integrated Gaussian',
            'PSF: Gaussian',
            'PSF: Elliptical Gaussian',
            'PSF: Elliptical Gaussian (3D astigmatism)',
            'Centroid of local neighborhood',
            'Radial symmetry'
        ])
        localization_layout.addRow("Estimator (PSF):", self.ts_estimator)

        # Sigma (PSF width)
        self.ts_sigma = QDoubleSpinBox()
        self.ts_sigma.setRange(0.5, 5.0)
        self.ts_sigma.setValue(1.6)
        self.ts_sigma.setSingleStep(0.1)
        self.ts_sigma.setDecimals(2)
        localization_layout.addRow("Sigma (PSF width):", self.ts_sigma)

        # Fit radius
        self.ts_fit_radius = QSpinBox()
        self.ts_fit_radius.setRange(1, 10)
        self.ts_fit_radius.setValue(3)
        localization_layout.addRow("Fit Radius (pixels):", self.ts_fit_radius)

        # Fitting method
        self.ts_fitting_method = QComboBox()
        self.ts_fitting_method.addItems([
            'Weighted Least squares',
            'Least squares',
            'Maximum likelihood'
        ])
        localization_layout.addRow("Fitting Method:", self.ts_fitting_method)

        # Full image fitting
        self.ts_full_image_fitting = QCheckBox("Full image fitting")
        self.ts_full_image_fitting.setChecked(False)
        localization_layout.addRow("", self.ts_full_image_fitting)

        scroll_layout.addWidget(localization_group)

        # === MULTI-EMITTER ANALYSIS ===
        mfa_group = QGroupBox("Multi-emitter Analysis (MFA)")
        mfa_layout = QFormLayout(mfa_group)
        mfa_layout.setSpacing(8)
        mfa_layout.setContentsMargins(10, 15, 10, 10)

        # Enable MFA
        self.ts_mfa_enabled = QCheckBox("Enable multi-emitter fitting")
        self.ts_mfa_enabled.setChecked(False)
        self.ts_mfa_enabled.toggled.connect(self.on_ts_mfa_toggled)
        mfa_layout.addRow("", self.ts_mfa_enabled)

        # MFA parameters
        self.ts_mfa_keep_same_intensity = QCheckBox("Keep same intensity")
        self.ts_mfa_keep_same_intensity.setChecked(False)
        mfa_layout.addRow("", self.ts_mfa_keep_same_intensity)

        self.ts_mfa_nmax = QSpinBox()
        self.ts_mfa_nmax.setRange(2, 10)
        self.ts_mfa_nmax.setValue(5)
        mfa_layout.addRow("Max emitters (nmax):", self.ts_mfa_nmax)

        self.ts_mfa_fixed_intensity = QCheckBox("Fixed intensity")
        self.ts_mfa_fixed_intensity.setChecked(True)
        mfa_layout.addRow("", self.ts_mfa_fixed_intensity)

        # Expected intensity range
        intensity_layout = QHBoxLayout()
        self.ts_mfa_intensity_min = QSpinBox()
        self.ts_mfa_intensity_min.setRange(10, 10000)
        self.ts_mfa_intensity_min.setValue(100)
        self.ts_mfa_intensity_max = QSpinBox()
        self.ts_mfa_intensity_max.setRange(10, 10000)
        self.ts_mfa_intensity_max.setValue(500)
        intensity_layout.addWidget(self.ts_mfa_intensity_min)
        intensity_layout.addWidget(QLabel(" to "))
        intensity_layout.addWidget(self.ts_mfa_intensity_max)
        mfa_layout.addRow("Expected Intensity:", intensity_layout)

        # P-value
        self.ts_mfa_pvalue = QComboBox()
        self.ts_mfa_pvalue.setEditable(True)
        self.ts_mfa_pvalue.addItems(['1.0E-6', '1.0E-5', '1.0E-4', '1.0E-3'])
        mfa_layout.addRow("P-value:", self.ts_mfa_pvalue)

        scroll_layout.addWidget(mfa_group)

        # === RENDERING/EXPORT SETTINGS ===
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)
        export_layout.setSpacing(8)
        export_layout.setContentsMargins(10, 15, 10, 10)

        # Renderer
        self.ts_renderer = QComboBox()
        self.ts_renderer.addItems([
            'No Renderer',
            'Averaged shifted histograms',
            'Normalized Gaussian',
            'Histogram'
        ])
        export_layout.addRow("Renderer:", self.ts_renderer)

        # Magnification
        self.ts_magnification = QDoubleSpinBox()
        self.ts_magnification.setRange(1.0, 20.0)
        self.ts_magnification.setValue(5.0)
        self.ts_magnification.setSingleStep(1.0)
        export_layout.addRow("Magnification:", self.ts_magnification)

        # Colorize z-stack
        self.ts_colorize_z = QCheckBox("Colorize z-stack")
        self.ts_colorize_z.setChecked(False)
        export_layout.addRow("", self.ts_colorize_z)

        # 3D mode
        self.ts_3d_mode = QCheckBox("3D imaging mode")
        self.ts_3d_mode.setChecked(False)
        export_layout.addRow("", self.ts_3d_mode)

        # Export columns
        export_cols_label = QLabel("Export columns (all selected by default):")
        export_layout.addRow(export_cols_label)

        export_cols_layout = QGridLayout()
        export_cols_layout.setSpacing(5)

        self.ts_export_sigma = QCheckBox("sigma")
        self.ts_export_sigma.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_sigma, 0, 0)

        self.ts_export_intensity = QCheckBox("intensity")
        self.ts_export_intensity.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_intensity, 0, 1)

        self.ts_export_chi2 = QCheckBox("chi2")
        self.ts_export_chi2.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_chi2, 0, 2)

        self.ts_export_offset = QCheckBox("offset")
        self.ts_export_offset.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_offset, 1, 0)

        self.ts_export_x = QCheckBox("x")
        self.ts_export_x.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_x, 1, 1)

        self.ts_export_y = QCheckBox("y")
        self.ts_export_y.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_y, 1, 2)

        self.ts_export_bkgstd = QCheckBox("bkgstd")
        self.ts_export_bkgstd.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_bkgstd, 2, 0)

        self.ts_export_uncertainty = QCheckBox("uncertainty")
        self.ts_export_uncertainty.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_uncertainty, 2, 1)

        self.ts_export_frame = QCheckBox("frame")
        self.ts_export_frame.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_frame, 2, 2)

        self.ts_export_protocol = QCheckBox("save protocol")
        self.ts_export_protocol.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_protocol, 3, 0)

        export_layout.addRow(export_cols_layout)

        scroll_layout.addWidget(export_group)

        # === PYIMAGEJ SETTINGS ===
        pyimagej_group = QGroupBox("Automatic Execution (Optional)")
        pyimagej_layout = QVBoxLayout(pyimagej_group)
        pyimagej_layout.setSpacing(8)
        pyimagej_layout.setContentsMargins(10, 15, 10, 10)

        self.ts_auto_run = QCheckBox("Automatically run macro in ImageJ after generation")
        self.ts_auto_run.setChecked(False)
        self.ts_auto_run.toggled.connect(self.on_ts_auto_run_toggled)
        pyimagej_layout.addWidget(self.ts_auto_run)

        # ImageJ/Fiji path
        fiji_layout = QHBoxLayout()
        fiji_layout.addWidget(QLabel("Fiji/ImageJ Path:"))
        self.ts_fiji_path = QLabel("Not set (required for auto-run)")
        self.ts_fiji_path.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        fiji_layout.addWidget(self.ts_fiji_path, 1)

        self.ts_fiji_browse_btn = QPushButton("Browse...")
        self.ts_fiji_browse_btn.clicked.connect(self.select_fiji_path)
        fiji_layout.addWidget(self.ts_fiji_browse_btn)
        pyimagej_layout.addLayout(fiji_layout)

        # Execution method
        exec_method_layout = QHBoxLayout()
        exec_method_layout.addWidget(QLabel("Execution Method:"))

        self.ts_exec_method_group = QButtonGroup()
        self.ts_exec_pyimagej_radio = QRadioButton("PyImageJ (Python)")
        self.ts_exec_subprocess_radio = QRadioButton("Subprocess (direct ImageJ)")
        self.ts_exec_method_group.addButton(self.ts_exec_pyimagej_radio)
        self.ts_exec_method_group.addButton(self.ts_exec_subprocess_radio)
        self.ts_exec_subprocess_radio.setChecked(True)

        exec_method_layout.addWidget(self.ts_exec_pyimagej_radio)
        exec_method_layout.addWidget(self.ts_exec_subprocess_radio)
        exec_method_layout.addStretch()
        pyimagej_layout.addLayout(exec_method_layout)

        # Info label
        info_label = QLabel(
            "Note: PyImageJ requires 'imagej' package (pip install pyimagej).\n"
            "Subprocess method calls ImageJ directly and is more reliable."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        pyimagej_layout.addWidget(info_label)

        scroll_layout.addWidget(pyimagej_group)

        # === ACTION BUTTONS ===
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        self.ts_generate_macro_btn = QPushButton("Generate Macro")
        self.ts_generate_macro_btn.clicked.connect(self.generate_thunderstorm_macro)
        self.ts_generate_macro_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        self.ts_generate_macro_btn.setMinimumHeight(35)
        button_layout.addWidget(self.ts_generate_macro_btn)

        self.ts_view_macro_btn = QPushButton("View Generated Macro")
        self.ts_view_macro_btn.clicked.connect(self.view_thunderstorm_macro)
        self.ts_view_macro_btn.setEnabled(False)
        self.ts_view_macro_btn.setMinimumHeight(35)
        button_layout.addWidget(self.ts_view_macro_btn)

        scroll_layout.addLayout(button_layout)

        # === MACRO PREVIEW ===
        preview_group = QGroupBox("Macro Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 15, 10, 10)

        self.ts_macro_preview = QTextEdit()
        self.ts_macro_preview.setReadOnly(True)
        self.ts_macro_preview.setMinimumHeight(150)
        self.ts_macro_preview.setMaximumHeight(250)
        self.ts_macro_preview.setPlaceholderText("Generated macro will appear here...")
        preview_layout.addWidget(self.ts_macro_preview)

        scroll_layout.addWidget(preview_group)

        # === STATUS ===
        self.ts_status_label = QLabel("Ready to generate macro")
        self.ts_status_label.setStyleSheet("color: #666; padding: 5px;")
        self.ts_status_label.setWordWrap(True)
        scroll_layout.addWidget(self.ts_status_label)

        # Add stretch at the end of scrollable content
        scroll_layout.addStretch()

        # Set the scroll widget and add to main layout
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Initialize UI state
        self.on_ts_mfa_toggled()
        self.on_ts_auto_run_toggled()

        # Add tab to widget
        self.tab_widget.addTab(tab, "ThunderSTORM")

    def on_ts_mfa_toggled(self):
        """Enable/disable MFA parameters based on checkbox"""
        enabled = self.ts_mfa_enabled.isChecked()
        self.ts_mfa_keep_same_intensity.setEnabled(enabled)
        self.ts_mfa_nmax.setEnabled(enabled)
        self.ts_mfa_fixed_intensity.setEnabled(enabled)
        self.ts_mfa_intensity_min.setEnabled(enabled)
        self.ts_mfa_intensity_max.setEnabled(enabled)
        self.ts_mfa_pvalue.setEnabled(enabled)

    def on_ts_auto_run_toggled(self):
        """Enable/disable auto-run parameters"""
        enabled = self.ts_auto_run.isChecked()
        self.ts_fiji_browse_btn.setEnabled(enabled)
        self.ts_exec_pyimagej_radio.setEnabled(enabled)
        self.ts_exec_subprocess_radio.setEnabled(enabled)

    def select_thunderstorm_input_dir(self):
        """Select input directory for thunderSTORM processing"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory for ThunderSTORM",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.ts_input_dir = directory
            self.ts_input_dir_label.setText(directory)
            self.ts_output_macro_label.setText(os.path.join(directory, "thunderstorm_macro_auto.ijm"))
            self.ts_status_label.setText(f"Input directory selected: {directory}")

    def select_fiji_path(self):
        """Select Fiji/ImageJ installation path"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Fiji/ImageJ Installation Directory (e.g., Fiji.app)",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )

        if directory:
            self.ts_fiji_installation = directory
            self.ts_fiji_path.setText(directory)
            self.ts_status_label.setText(f"Fiji path set: {directory}")

    def get_thunderstorm_file_list(self):
        """Get list of files matching the pattern"""
        if not hasattr(self, 'ts_input_dir') or not self.ts_input_dir:
            return [], []

        pattern = self.ts_file_pattern.currentText()
        files = glob.glob(os.path.join(self.ts_input_dir, pattern), recursive=True)

        # Create result paths
        result_paths = [f.replace('.tif', '_locs.csv') for f in files]

        return files, result_paths

    def generate_thunderstorm_macro_command(self):
        """Generate the thunderSTORM macro command string"""

        # Build export columns string
        export_cols = []
        if self.ts_export_sigma.isChecked():
            export_cols.append("sigma=true")
        if self.ts_export_intensity.isChecked():
            export_cols.append("intensity=true")
        if self.ts_export_chi2.isChecked():
            export_cols.append("chi2=true")
        if self.ts_export_offset.isChecked():
            export_cols.append("offset=true")
        if self.ts_export_protocol.isChecked():
            export_cols.append("saveprotocol=true")
        if self.ts_export_x.isChecked():
            export_cols.append("x=true")
        if self.ts_export_y.isChecked():
            export_cols.append("y=true")
        if self.ts_export_bkgstd.isChecked():
            export_cols.append("bkgstd=true")
        if self.ts_export_uncertainty.isChecked():
            export_cols.append("uncertainty=true")
        if self.ts_export_frame.isChecked():
            export_cols.append("frame=true")

        export_string = " ".join(export_cols)

        # Build Run analysis command
        run_analysis_parts = []

        # Filter
        filter_type = self.ts_filter_type.currentText()
        run_analysis_parts.append(f'filter=[{filter_type}]')

        if 'Wavelet' in filter_type:
            run_analysis_parts.append(f'scale={self.ts_wavelet_scale.value()}')
            run_analysis_parts.append(f'order={self.ts_wavelet_order.value()}')

        # Detector
        run_analysis_parts.append(f'detector=[{self.ts_detector.currentText()}]')
        run_analysis_parts.append(f'connectivity={self.ts_connectivity.currentText()}')
        run_analysis_parts.append(f'threshold={self.ts_threshold.text()}')

        # Estimator
        run_analysis_parts.append(f'estimator=[{self.ts_estimator.currentText()}]')
        run_analysis_parts.append(f'sigma={self.ts_sigma.value()}')
        run_analysis_parts.append(f'fitradius={self.ts_fit_radius.value()}')
        run_analysis_parts.append(f'method=[{self.ts_fitting_method.currentText()}]')

        # Full image fitting
        run_analysis_parts.append(f'full_image_fitting={"true" if self.ts_full_image_fitting.isChecked() else "false"}')

        # MFA
        run_analysis_parts.append(f'mfaenabled={"true" if self.ts_mfa_enabled.isChecked() else "false"}')
        if self.ts_mfa_enabled.isChecked():
            run_analysis_parts.append(f'keep_same_intensity={"true" if self.ts_mfa_keep_same_intensity.isChecked() else "false"}')
            run_analysis_parts.append(f'nmax={self.ts_mfa_nmax.value()}')
            run_analysis_parts.append(f'fixed_intensity={"true" if self.ts_mfa_fixed_intensity.isChecked() else "false"}')
            run_analysis_parts.append(f'expected_intensity={self.ts_mfa_intensity_min.value()}:{self.ts_mfa_intensity_max.value()}')
            run_analysis_parts.append(f'pvalue={self.ts_mfa_pvalue.currentText()}')

        # Renderer
        run_analysis_parts.append(f'renderer=[{self.ts_renderer.currentText()}]')
        run_analysis_parts.append(f'magnification={self.ts_magnification.value()}')
        run_analysis_parts.append(f'colorizez={"true" if self.ts_colorize_z.isChecked() else "false"}')
        run_analysis_parts.append(f'threed={"true" if self.ts_3d_mode.isChecked() else "false"}')

        # Additional parameters
        run_analysis_parts.append('shifts=2')
        run_analysis_parts.append('repaint=50')

        run_analysis_cmd = " ".join(run_analysis_parts)

        # Build complete macro
        macro_template = """for (i=0; i  < datapaths.length; i++) {
    run("Bio-Formats Importer", "open=" + datapaths[i] +" color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
    run("Run analysis", "%s");
    run("Export results", "filepath=["+respaths[i]+"] fileformat=[CSV (comma separated)] %s");
    while (nImages>0) {
        selectImage(nImages);
        close();
    }
}"""

        macro_command = macro_template % (run_analysis_cmd, export_string)

        return macro_command

    def generate_thunderstorm_macro(self):
        """Generate the complete thunderSTORM macro file"""
        try:
            # Check if input directory is set
            if not hasattr(self, 'ts_input_dir') or not self.ts_input_dir:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Directory", "Please select an input directory first.")
                return

            # Get file list
            data_files, result_files = self.get_thunderstorm_file_list()

            if not data_files:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "No Files Found",
                    f"No files found matching pattern: {self.ts_file_pattern.currentText()}"
                )
                return

            # Convert file lists to ImageJ macro arrays
            data_files_str = str(data_files).replace('[', '').replace(']', '')
            result_files_str = str(result_files).replace('[', '').replace(']', '')

            datapaths_line = f'datapaths = newArray({data_files_str});'
            respaths_line = f'respaths = newArray({result_files_str});'

            # Generate macro command
            macro_command = self.generate_thunderstorm_macro_command()

            # Combine into full macro
            full_macro = f"{datapaths_line}\n{respaths_line}\n{macro_command}"

            # Save macro
            save_path = os.path.join(self.ts_input_dir, 'thunderstorm_macro_auto.ijm')
            with open(save_path, 'w') as f:
                f.write(full_macro)

            # Update preview
            self.ts_macro_preview.setPlainText(full_macro)
            self.ts_view_macro_btn.setEnabled(True)

            # Update status
            self.ts_status_label.setText(
                f"✓ Macro generated successfully: {save_path}\n"
                f"  Found {len(data_files)} files to process"
            )
            self.ts_status_label.setStyleSheet("color: green; padding: 5px;")

            # Show success message
            from qtpy.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Macro Generated")
            msg.setText(f"ThunderSTORM macro generated successfully!")
            msg.setInformativeText(
                f"Macro saved to: {save_path}\n\n"
                f"Files to process: {len(data_files)}\n\n"
                "You can now:\n"
                "1. Run the macro manually in ImageJ/Fiji\n"
                "2. Enable 'Auto-run' and click 'Generate' again to run automatically"
            )
            msg.setStandardButtons(QMessageBox.Ok)

            # Add buttons for additional actions
            if self.ts_auto_run.isChecked():
                msg.addButton("Run Now", QMessageBox.ActionRole)

            result = msg.exec_()

            # If auto-run is enabled and user clicked "Run Now", execute the macro
            if self.ts_auto_run.isChecked() and result == 0:
                self.run_thunderstorm_macro(save_path)

        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate macro:\n\n{str(e)}"
            )
            self.ts_status_label.setText(f"✗ Error: {str(e)}")
            self.ts_status_label.setStyleSheet("color: red; padding: 5px;")

    def view_thunderstorm_macro(self):
        """Open the generated macro in a viewer"""
        macro_path = os.path.join(self.ts_input_dir, 'thunderstorm_macro_auto.ijm')

        if not os.path.exists(macro_path):
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "File Not Found",
                "Macro file not found. Please generate the macro first."
            )
            return

        # Try to open with system default editor
        try:
            if os.name == 'nt':  # Windows
                os.startfile(macro_path)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.call(['open' if sys.platform == 'darwin' else 'xdg-open', macro_path])
        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Cannot Open File",
                f"Could not open macro file with system editor:\n{str(e)}\n\n"
                f"File location: {macro_path}"
            )

    def run_thunderstorm_macro(self, macro_path):
        """Run the generated macro in ImageJ"""
        try:
            if not hasattr(self, 'ts_fiji_installation') or not self.ts_fiji_installation:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Fiji Path Not Set",
                    "Please set the Fiji/ImageJ installation path first."
                )
                return

            # Check which execution method to use
            use_pyimagej = self.ts_exec_pyimagej_radio.isChecked()

            if use_pyimagej:
                self.run_with_pyimagej(macro_path)
            else:
                self.run_with_subprocess(macro_path)

        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Execution Error",
                f"Failed to run macro:\n\n{str(e)}"
            )

    def run_with_pyimagej(self, macro_path):
        """Run macro using PyImageJ"""
        try:
            import imagej

            self.ts_status_label.setText("Initializing ImageJ via PyImageJ...")
            QApplication.processEvents()

            # Initialize ImageJ
            ij = imagej.init(self.ts_fiji_installation, headless=False)

            self.ts_status_label.setText("Loading macro...")
            QApplication.processEvents()

            # Read macro content
            with open(macro_path, 'r') as f:
                macro_content = f.read()

            self.ts_status_label.setText("Running macro in ImageJ...")
            QApplication.processEvents()

            # Run macro
            ij.py.run_macro(macro_content)

            self.ts_status_label.setText("✓ Macro execution completed via PyImageJ")
            self.ts_status_label.setStyleSheet("color: green; padding: 5px;")

            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Success",
                "Macro executed successfully via PyImageJ!\n\n"
                "Check the output directory for results."
            )

        except ImportError:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "PyImageJ Not Found",
                "PyImageJ is not installed.\n\n"
                "Install it using: pip install pyimagej\n\n"
                "Alternatively, use the 'Subprocess' execution method."
            )
        except Exception as e:
            raise

    def run_with_subprocess(self, macro_path):
        """Run macro by calling ImageJ as subprocess"""
        try:
            # Find ImageJ executable
            if sys.platform == 'darwin':  # macOS
                imagej_exe = os.path.join(self.ts_fiji_installation, 'Contents', 'MacOS', 'ImageJ-macosx')
            elif sys.platform.startswith('win'):  # Windows
                imagej_exe = os.path.join(self.ts_fiji_installation, 'ImageJ-win64.exe')
            else:  # Linux
                imagej_exe = os.path.join(self.ts_fiji_installation, 'ImageJ-linux64')

            if not os.path.exists(imagej_exe):
                # Try alternative names
                possible_names = ['fiji', 'ImageJ', 'imagej']
                for name in possible_names:
                    test_path = os.path.join(self.ts_fiji_installation, name)
                    if os.path.exists(test_path):
                        imagej_exe = test_path
                        break
                else:
                    raise FileNotFoundError(
                        f"ImageJ executable not found in {self.ts_fiji_installation}\n"
                        f"Looked for: {imagej_exe}"
                    )

            self.ts_status_label.setText("Launching ImageJ...")
            QApplication.processEvents()

            # Run ImageJ with macro
            cmd = [imagej_exe, '-macro', macro_path]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.ts_input_dir
            )

            self.ts_status_label.setText("ImageJ is running... (this may take a while)")
            self.ts_status_label.setStyleSheet("color: blue; padding: 5px;")

            # Show dialog that processing is happening
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "ImageJ Launched",
                f"ImageJ has been launched with the macro.\n\n"
                f"The processing will continue in ImageJ.\n"
                f"Results will be saved to: {self.ts_input_dir}\n\n"
                f"You can continue using FLIKA while ImageJ runs."
            )

            self.ts_status_label.setText("ImageJ is processing in the background...")

        except FileNotFoundError as e:
            raise FileNotFoundError(str(e))
        except Exception as e:
            raise

    def add_thunderstorm_attributes_to_init(self):
        """Initialize thunderSTORM-specific attributes"""
        self.ts_input_dir = None
        self.ts_fiji_installation = None

    def close(self):
        """Enhanced close method that properly closes file logging"""
        try:
            if self.file_logger:
                self.file_logger.close()
        except Exception as e:
            print(f"Error closing file logger: {e}")
        super().close()

# Plugin instance management
spt_batch_analysis_instance = None


def launch_spt_analysis():
    """Launch the enhanced SPT batch analysis plugin with geometric methods and autocorrelation"""
    global spt_batch_analysis_instance

    if spt_batch_analysis_instance is None or not spt_batch_analysis_instance.isVisible():
        spt_batch_analysis_instance = SPTBatchAnalysis()

    spt_batch_analysis_instance.show()
    spt_batch_analysis_instance.raise_()
    spt_batch_analysis_instance.activateWindow()

def launch_docs():
    """Launch documentation"""
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    url = 'https://github.com/flika-org/flika_plugin_template'
    QDesktopServices.openUrl(QUrl(url))

def check_utrack_availability():
    """Check if U-Track linking with mixed motion is available"""
    try:
        plugin_dir = get_plugin_directory()
        utrack_file = os.path.join(plugin_dir, 'utrack_linking.py')

        if not os.path.exists(utrack_file):
            return False, "utrack_linking.py not found in plugin directory"

        # Try importing basic U-Track
        import sys
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        from utrack_linking import UTrackLinker

        # Try importing mixed motion components
        try:
            UTrackLinkerAdapter
            return True, "U-Track with mixed motion models available"
        except NameError:
            return True, "U-Track available (basic functionality only)"

    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"
