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

# ThunderSTORM integration
try:
    from .thunderstorm_integration import (
        ThunderSTORMDetector,
        is_thunderstorm_available,
        get_available_filters,
        get_available_detectors,
        get_available_fitters,
        get_default_threshold_expressions
    )
    THUNDERSTORM_AVAILABLE = is_thunderstorm_available()
except ImportError:
    THUNDERSTORM_AVAILABLE = False
    print("Warning: ThunderSTORM integration module not found")




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

    def __init__(self, file_paths, detector_params, output_dir, pixel_size, show_results=False, detection_method='utrack'):
        super().__init__()
        self.file_paths = file_paths
        self.detector_params = detector_params
        self.output_dir = output_dir
        self.pixel_size = pixel_size
        self.show_results = show_results
        self.detection_method = detection_method  # 'utrack' or 'thunderstorm'

    def run(self):
        """Run detection on all files"""
        try:
            # Route to appropriate detection method
            if self.detection_method == 'thunderstorm':
                self._run_thunderstorm_detection()
            else:  # 'utrack' or default
                self._run_utrack_detection()

        except Exception as e:
            self.detection_error.emit(f"Detection failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _run_utrack_detection(self):
        """Run U-Track based detection on all files"""
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
                    # Swap x and y to undo the rot90+fliplr transpose applied
                    # to images before detection, restoring original image coords.
                    combined_detections['x [nm]'] = combined_detections['y'] * self.pixel_size
                    combined_detections['y [nm]'] = combined_detections['x'] * self.pixel_size
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

            self.detection_complete.emit("U-Track detection completed successfully!")

        except Exception as e:
            self.detection_error.emit(f"U-Track detection failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def _run_thunderstorm_detection(self):
        """Run ThunderSTORM based detection on all files"""
        try:
            # Check if thunderSTORM is available
            if not THUNDERSTORM_AVAILABLE:
                self.detection_error.emit("ThunderSTORM is not available. Please install thunderstorm_python package.")
                return

            total_files = len(self.file_paths)

            for file_idx, file_path in enumerate(self.file_paths):
                self.progress_update.emit(f"ThunderSTORM processing {file_idx + 1}/{total_files}: {os.path.basename(file_path)}")

                try:
                    # Load image sequence
                    images = skio.imread(file_path, plugin='tifffile')
                    if images.ndim == 2:
                        images = images[np.newaxis, ...]  # Add time dimension

                    # Apply same transformations as in main plugin
                    images = np.rot90(images, axes=(1,2))
                    images = np.fliplr(images)

                except Exception as e:
                    self.detection_error.emit(f"Error loading {file_path}: {e}")
                    continue

                # Create ThunderSTORM detector with parameters
                # Pass all ts_* prefixed params from the GUI through to the detector
                gui_params = {'pixel_size': self.pixel_size}
                for key, value in self.detector_params.items():
                    if key.startswith('ts_'):
                        gui_params[key] = value

                detector = ThunderSTORMDetector.create_from_gui_parameters(gui_params)

                # Run detection on all frames
                n_frames = images.shape[0]
                self.progress_update.emit(f"Running ThunderSTORM on {n_frames} frames...")

                localizations = detector.detect_and_fit(images, show_progress=False)

                # The input images have been rot90+fliplr transformed (which
                # transposes rows/cols). Swap x and y to restore coordinates
                # to the original image space.
                if 'x' in localizations and 'y' in localizations:
                    localizations['x'], localizations['y'] = (
                        localizations['y'].copy(), localizations['x'].copy()
                    )
                if 'sigma_x' in localizations and 'sigma_y' in localizations:
                    localizations['sigma_x'], localizations['sigma_y'] = (
                        localizations['sigma_y'].copy(), localizations['sigma_x'].copy()
                    )

                # Update progress after detection
                self.frame_progress.emit(100)

                # Save results in SPT-compatible format with all columns
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(self.output_dir, f"{base_name}_locsID.csv")

                detector.save_localizations(localizations, output_file, image_stack=images)

                n_detections = len(localizations['x'])
                self.progress_update.emit(f"ThunderSTORM: Saved {n_detections} detections to {os.path.basename(output_file)}")

                # Emit signal for visualization if requested
                if self.show_results:
                    self.file_processed.emit(file_path, output_file)

            self.detection_complete.emit("ThunderSTORM detection completed successfully!")

        except Exception as e:
            self.detection_error.emit(f"ThunderSTORM detection failed: {str(e)}")
            import traceback
            traceback.print_exc()



class ValidationWorker(QThread):
    """Worker thread for running validation tests (comparison and ground truth)."""

    progress_update = Signal(str)
    step_progress = Signal(int)  # 0-100 progress
    test_complete = Signal(dict)  # results dict
    test_error = Signal(str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.setStackSize(16 * 1024 * 1024)  # 16 MB — needed for OpenBLAS/numpy linalg ops
        self.task_type = task_type  # 'comparison', 'synthetic_generate', 'synthetic_run', 'ground_truth', 'full_validation'
        self.kwargs = kwargs
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self.task_type == 'comparison':
                self._run_comparison_tests()
            elif self.task_type == 'synthetic_generate':
                self._run_synthetic_generation()
            elif self.task_type == 'synthetic_run':
                self._run_synthetic_flika()
            elif self.task_type == 'ground_truth':
                self._run_ground_truth_comparison()
            elif self.task_type == 'full_validation':
                self._run_full_validation()
            else:
                self.test_error.emit(f"Unknown task type: {self.task_type}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.test_error.emit(f"Validation failed: {str(e)}")

    def _run_comparison_tests(self):
        """Run FLIKA tests on real data and compare against ImageJ results."""
        input_path = self.kwargs.get('input_path')
        output_dir = Path(self.kwargs.get('output_dir'))
        imagej_dir = self.kwargs.get('imagej_dir')
        test_names = self.kwargs.get('test_names')
        match_radius = self.kwargs.get('match_radius', 200.0)
        generate_plots = self.kwargs.get('generate_plots', True)

        # Import comparison modules
        script_dir = Path(__file__).parent / 'tests' / 'comparison'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))

        from generate_comparison_macros import TEST_CONFIGS, CAMERA_DEFAULTS, run_flika_tests, generate_all_macros, save_test_metadata
        from compare_results import run_comparison, load_thunderstorm_csv, match_localizations, compute_comparison_stats, plot_comparison, plot_summary_chart, load_timing_data, plot_speed_comparison

        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate macros and save metadata
        self.progress_update.emit("Saving test metadata...")
        save_test_metadata(str(output_dir), input_path)

        # Step 2: Generate ImageJ macros
        self.progress_update.emit("Generating ImageJ macros...")
        generate_all_macros(input_path, str(output_dir), test_names)
        self.step_progress.emit(10)

        # Step 3: Run FLIKA tests
        self.progress_update.emit("Running FLIKA ThunderSTORM tests...")
        configs = TEST_CONFIGS
        if test_names:
            configs = {k: v for k, v in configs.items() if k in test_names}

        total = len(configs)
        flika_results = run_flika_tests(input_path, str(output_dir), test_names)
        self.step_progress.emit(60)

        if self._cancelled:
            return

        # Step 4: Copy ImageJ results if provided
        if imagej_dir and Path(imagej_dir).exists():
            ij_dest = output_dir / "imagej_results"
            ij_dest.mkdir(exist_ok=True)
            import shutil
            for csv_file in Path(imagej_dir).glob("*.csv"):
                shutil.copy2(csv_file, ij_dest / csv_file.name)
            for txt_file in Path(imagej_dir).glob("*.txt"):
                shutil.copy2(txt_file, ij_dest / txt_file.name)
            self.progress_update.emit(f"Copied ImageJ results from {imagej_dir}")

        # Step 5: Run comparison
        self.progress_update.emit("Comparing FLIKA vs ImageJ results...")
        all_stats = run_comparison(
            str(output_dir),
            test_names=test_names,
            match_radius_nm=match_radius,
            do_plot=generate_plots
        )
        self.step_progress.emit(90)

        # Step 6: Generate HTML report
        self.progress_update.emit("Generating report...")
        report_path = self._generate_html_report(output_dir, all_stats, 'comparison')
        self.step_progress.emit(100)

        results = {
            'type': 'comparison',
            'stats': all_stats,
            'output_dir': str(output_dir),
            'report_path': str(report_path) if report_path else None,
            'n_tests': len(all_stats),
            'flika_results': flika_results,
        }
        self.test_complete.emit(results)

    def _run_synthetic_generation(self):
        """Generate synthetic datasets."""
        output_dir = Path(self.kwargs.get('output_dir'))
        config_names = self.kwargs.get('config_names')
        custom_configs = self.kwargs.get('custom_configs')
        seed = self.kwargs.get('seed', 42)

        script_dir = Path(__file__).parent / 'tests' / 'synthetic'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))

        from generate_synthetic_data import (SYNTHETIC_CONFIGS, MAGNIFICATION_PRESETS,
                                              CAMERA_PRESETS, generate_synthetic_dataset)

        output_dir.mkdir(parents=True, exist_ok=True)
        if config_names is not None:
            dataset_configs = {k: v for k, v in SYNTHETIC_CONFIGS.items() if k in config_names}
        else:
            dataset_configs = dict(SYNTHETIC_CONFIGS)

        # Add custom configs — register any inline optics/camera presets
        if custom_configs:
            for cname, ccfg in custom_configs.items():
                if ccfg.get('optics') == '__custom__' and 'custom_optics' in ccfg:
                    MAGNIFICATION_PRESETS['__custom__'] = ccfg['custom_optics']
                if ccfg.get('camera') == '__custom__' and 'custom_camera' in ccfg:
                    CAMERA_PRESETS['__custom__'] = ccfg['custom_camera']
                dataset_configs[cname] = ccfg

        all_meta = {}
        total = len(dataset_configs)
        for i, (name, cfg) in enumerate(dataset_configs.items()):
            if self._cancelled:
                return
            self.progress_update.emit(f"Generating synthetic dataset {i+1}/{total}: {name}")
            meta = generate_synthetic_dataset(name, cfg, output_dir, seed=seed)
            all_meta[name] = meta
            self.step_progress.emit(int((i + 1) / total * 100))

        # Save combined metadata
        with open(str(output_dir / "all_metadata.json"), 'w') as fp:
            json.dump(all_meta, fp, indent=2)

        self.test_complete.emit({
            'type': 'synthetic_generate',
            'output_dir': str(output_dir),
            'n_datasets': len(all_meta),
            'datasets': list(all_meta.keys()),
        })

    def _run_synthetic_flika(self):
        """Run FLIKA analysis on synthetic datasets."""
        data_dir = Path(self.kwargs.get('data_dir'))
        results_dir = Path(self.kwargs.get('results_dir'))
        config_names = self.kwargs.get('config_names')
        algo_names = self.kwargs.get('algo_names')

        script_dir = Path(__file__).parent / 'tests' / 'synthetic'
        comp_dir = Path(__file__).parent / 'tests' / 'comparison'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        if str(comp_dir) not in sys.path:
            sys.path.insert(0, str(comp_dir))
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))

        from generate_synthetic_data import SYNTHETIC_CONFIGS, MAGNIFICATION_PRESETS, CAMERA_PRESETS, run_flika_on_synthetic
        from generate_comparison_macros import TEST_CONFIGS

        results_dir.mkdir(parents=True, exist_ok=True)
        flika_dir = results_dir / "flika_results"
        flika_dir.mkdir(parents=True, exist_ok=True)

        dataset_configs = SYNTHETIC_CONFIGS
        if config_names:
            dataset_configs = {k: v for k, v in dataset_configs.items() if k in config_names}
        algo_configs = TEST_CONFIGS
        if algo_names:
            algo_configs = {k: v for k, v in algo_configs.items() if k in algo_names}

        import tifffile
        total_tests = len(dataset_configs) * len(algo_configs)
        test_count = 0
        flika_results = {}

        for ds_name, ds_cfg in dataset_configs.items():
            tiff_path = data_dir / f"{ds_name}.tif"
            if not tiff_path.exists():
                self.progress_update.emit(f"Skipping {ds_name}: data not found")
                test_count += len(algo_configs)
                continue

            image_stack = tifffile.imread(str(tiff_path))
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]

            for algo_name, algo_cfg in algo_configs.items():
                if self._cancelled:
                    return
                test_name = f"{ds_name}__{algo_name}"
                test_count += 1
                self.progress_update.emit(f"[{test_count}/{total_tests}] {ds_name} / {algo_name}")
                try:
                    result = run_flika_on_synthetic(
                        test_name, ds_cfg, algo_cfg['flika'],
                        tiff_path, flika_dir, image_stack=image_stack)
                    flika_results[test_name] = result
                except Exception as e:
                    self.progress_update.emit(f"  ERROR: {test_name}: {e}")
                self.step_progress.emit(int(test_count / total_tests * 100))

        # Save summary
        with open(str(results_dir / "flika_synthetic_summary.json"), 'w') as fp:
            json.dump(flika_results, fp, indent=2)

        self.test_complete.emit({
            'type': 'synthetic_run',
            'results_dir': str(results_dir),
            'n_tests': len(flika_results),
            'results': flika_results,
        })

    def _run_ground_truth_comparison(self):
        """Compare FLIKA (and optionally ImageJ) results against ground truth."""
        data_dir = Path(self.kwargs.get('data_dir'))
        results_dir = Path(self.kwargs.get('results_dir'))
        config_names = self.kwargs.get('config_names')
        algo_names = self.kwargs.get('algo_names')
        match_radius = self.kwargs.get('match_radius', 200.0)
        generate_plots = self.kwargs.get('generate_plots', True)

        script_dir = Path(__file__).parent / 'tests' / 'synthetic'
        comp_dir = Path(__file__).parent / 'tests' / 'comparison'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        if str(comp_dir) not in sys.path:
            sys.path.insert(0, str(comp_dir))
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))

        from generate_synthetic_data import SYNTHETIC_CONFIGS, MAGNIFICATION_PRESETS, compare_to_ground_truth
        from generate_comparison_macros import TEST_CONFIGS

        analysis_dir = results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        flika_dir = results_dir / "flika_results"
        imagej_dir = results_dir / "imagej_results"

        dataset_configs = SYNTHETIC_CONFIGS
        if config_names:
            dataset_configs = {k: v for k, v in dataset_configs.items() if k in config_names}
        algo_configs = TEST_CONFIGS
        if algo_names:
            algo_configs = {k: v for k, v in algo_configs.items() if k in algo_names}

        all_results = []
        total = len(dataset_configs) * len(algo_configs)
        count = 0

        for ds_name, ds_cfg in dataset_configs.items():
            optics = MAGNIFICATION_PRESETS[ds_cfg['optics']]
            gt_csv = data_dir / f"{ds_name}_ground_truth.csv"
            if not gt_csv.exists():
                count += len(algo_configs)
                continue

            for algo_name in algo_configs:
                if self._cancelled:
                    return
                test_name = f"{ds_name}__{algo_name}"
                count += 1
                self.progress_update.emit(f"[{count}/{total}] Comparing {test_name}")

                # Compare FLIKA results
                for suffix in ['_flika.csv', '_flika_locsID.csv']:
                    flika_csv = flika_dir / f"{test_name}{suffix}"
                    if flika_csv.exists():
                        break
                if flika_csv.exists():
                    stats = compare_to_ground_truth(
                        test_name, str(flika_csv), str(gt_csv),
                        optics['pixel_size_nm'], match_radius_nm=match_radius, label='FLIKA')
                    if stats:
                        stats['dataset'] = ds_name
                        stats['algorithm'] = algo_name
                        all_results.append(stats)

                # Compare ImageJ results if available
                imagej_csv = imagej_dir / f"{test_name}_imagej.csv"
                if imagej_csv.exists():
                    stats = compare_to_ground_truth(
                        test_name, str(imagej_csv), str(gt_csv),
                        optics['pixel_size_nm'], match_radius_nm=match_radius, label='ImageJ')
                    if stats:
                        stats['dataset'] = ds_name
                        stats['algorithm'] = algo_name
                        all_results.append(stats)

                self.step_progress.emit(int(count / total * 100))

        # Save results
        with open(str(analysis_dir / "ground_truth_comparison.json"), 'w') as fp:
            json.dump(all_results, fp, indent=2)

        # Generate plots
        if generate_plots and all_results:
            self.progress_update.emit("Generating ground truth comparison figures...")
            self._plot_ground_truth_summary(all_results, analysis_dir)

        # Generate HTML report
        report_path = self._generate_html_report(analysis_dir, all_results, 'ground_truth')

        self.test_complete.emit({
            'type': 'ground_truth',
            'results': all_results,
            'analysis_dir': str(analysis_dir),
            'report_path': str(report_path) if report_path else None,
            'n_results': len(all_results),
        })

    def _run_full_validation(self):
        """Run everything: generate synthetic, run FLIKA, compare to ground truth, plus real data comparison."""
        # Step 1: Synthetic generation
        data_dir = Path(self.kwargs.get('synthetic_data_dir'))
        results_dir = Path(self.kwargs.get('synthetic_results_dir'))
        config_names = self.kwargs.get('config_names')
        algo_names = self.kwargs.get('algo_names')
        seed = self.kwargs.get('seed', 42)
        match_radius = self.kwargs.get('match_radius', 200.0)

        script_dir = Path(__file__).parent / 'tests' / 'synthetic'
        comp_dir = Path(__file__).parent / 'tests' / 'comparison'
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        if str(comp_dir) not in sys.path:
            sys.path.insert(0, str(comp_dir))
        if str(Path(__file__).parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent))

        from generate_synthetic_data import SYNTHETIC_CONFIGS, MAGNIFICATION_PRESETS, CAMERA_PRESETS, generate_synthetic_dataset, run_flika_on_synthetic, compare_to_ground_truth
        from generate_comparison_macros import TEST_CONFIGS, CAMERA_DEFAULTS, run_flika_tests, generate_all_macros, save_test_metadata
        from compare_results import run_comparison

        # Phase 1: Generate synthetic data (20%)
        data_dir.mkdir(parents=True, exist_ok=True)
        dataset_configs = SYNTHETIC_CONFIGS
        if config_names:
            dataset_configs = {k: v for k, v in dataset_configs.items() if k in config_names}

        skip_existing = self.kwargs.get('skip_existing_synthetic', False)
        all_exist = False
        if skip_existing:
            all_exist = all(
                (data_dir / f"{name}.tif").exists() and
                (data_dir / f"{name}_ground_truth.csv").exists()
                for name in dataset_configs
            )

        if all_exist:
            self.progress_update.emit("Phase 1/4: Synthetic data already exists — skipping generation")
            # Load existing metadata if available
            all_meta = {}
            meta_path = data_dir / "all_metadata.json"
            if meta_path.exists():
                with open(str(meta_path)) as fp:
                    all_meta = json.load(fp)
        else:
            self.progress_update.emit("Phase 1/4: Generating synthetic data...")
            all_meta = {}
            for i, (name, cfg) in enumerate(dataset_configs.items()):
                if self._cancelled:
                    return
                self.progress_update.emit(f"  Generating {name}...")
                meta = generate_synthetic_dataset(name, cfg, data_dir, seed=seed)
                all_meta[name] = meta
            with open(str(data_dir / "all_metadata.json"), 'w') as fp:
                json.dump(all_meta, fp, indent=2)
        self.step_progress.emit(20)

        # Phase 2: Run FLIKA on synthetic (60%)
        self.progress_update.emit("Phase 2/4: Running FLIKA on synthetic data...")
        results_dir.mkdir(parents=True, exist_ok=True)
        flika_dir = results_dir / "flika_results"
        flika_dir.mkdir(parents=True, exist_ok=True)

        algo_configs = TEST_CONFIGS
        if algo_names:
            algo_configs = {k: v for k, v in algo_configs.items() if k in algo_names}

        import tifffile
        total_tests = len(dataset_configs) * len(algo_configs)
        test_count = 0
        for ds_name, ds_cfg in dataset_configs.items():
            tiff_path = data_dir / f"{ds_name}.tif"
            if not tiff_path.exists():
                test_count += len(algo_configs)
                continue
            image_stack = tifffile.imread(str(tiff_path))
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]
            for algo_name, algo_cfg in algo_configs.items():
                if self._cancelled:
                    return
                test_name = f"{ds_name}__{algo_name}"
                test_count += 1
                self.progress_update.emit(f"  [{test_count}/{total_tests}] {ds_name} / {algo_name}")
                try:
                    run_flika_on_synthetic(test_name, ds_cfg, algo_cfg['flika'],
                                           tiff_path, flika_dir, image_stack=image_stack)
                except Exception as e:
                    self.progress_update.emit(f"    ERROR: {e}")
                self.step_progress.emit(20 + int(test_count / total_tests * 40))
        self.step_progress.emit(60)

        # Phase 3: Ground truth comparison (80%)
        self.progress_update.emit("Phase 3/4: Comparing to ground truth...")
        analysis_dir = results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        all_gt_results = []
        count = 0
        for ds_name, ds_cfg in dataset_configs.items():
            optics = MAGNIFICATION_PRESETS[ds_cfg['optics']]
            gt_csv = data_dir / f"{ds_name}_ground_truth.csv"
            if not gt_csv.exists():
                count += len(algo_configs)
                continue
            for algo_name in algo_configs:
                test_name = f"{ds_name}__{algo_name}"
                count += 1
                for suffix in ['_flika.csv', '_flika_locsID.csv']:
                    flika_csv = flika_dir / f"{test_name}{suffix}"
                    if flika_csv.exists():
                        break
                if flika_csv.exists():
                    stats = compare_to_ground_truth(
                        test_name, str(flika_csv), str(gt_csv),
                        optics['pixel_size_nm'], match_radius_nm=match_radius, label='FLIKA')
                    if stats:
                        stats['dataset'] = ds_name
                        stats['algorithm'] = algo_name
                        all_gt_results.append(stats)

        with open(str(analysis_dir / "ground_truth_comparison.json"), 'w') as fp:
            json.dump(all_gt_results, fp, indent=2)
        self._plot_ground_truth_summary(all_gt_results, analysis_dir)
        self.step_progress.emit(80)

        # Phase 4: Real data comparison (100%)
        real_stats = {}
        real_input = self.kwargs.get('real_input_path')
        real_output = self.kwargs.get('real_output_dir')
        imagej_dir_path = self.kwargs.get('imagej_dir')
        if real_input and Path(real_input).exists():
            self.progress_update.emit("Phase 4/4: Running real data comparison...")
            real_output = Path(real_output)
            real_output.mkdir(parents=True, exist_ok=True)
            save_test_metadata(str(real_output), real_input)
            generate_all_macros(real_input, str(real_output))
            run_flika_tests(real_input, str(real_output))

            if imagej_dir_path and Path(imagej_dir_path).exists():
                import shutil
                ij_dest = real_output / "imagej_results"
                ij_dest.mkdir(exist_ok=True)
                if Path(imagej_dir_path).resolve() != ij_dest.resolve():
                    for f in Path(imagej_dir_path).glob("*"):
                        if f.is_file():
                            shutil.copy2(f, ij_dest / f.name)

            real_stats = run_comparison(str(real_output), do_plot=True)

        self.step_progress.emit(100)

        # Generate combined report
        report_path = self._generate_full_report(
            analysis_dir, all_gt_results, real_stats,
            real_output_dir=real_output if real_input else None
        )

        self.test_complete.emit({
            'type': 'full_validation',
            'gt_results': all_gt_results,
            'real_stats': real_stats,
            'report_path': str(report_path) if report_path else None,
            'analysis_dir': str(analysis_dir),
        })

    def _plot_ground_truth_summary(self, all_results, output_dir):
        """Generate ground truth comparison summary figures."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            flika_results = [r for r in all_results if r.get('label') == 'FLIKA']
            if not flika_results:
                return

            # Group by algorithm
            algo_f1 = {}
            for r in flika_results:
                algo = r.get('algorithm', 'unknown')
                algo_f1.setdefault(algo, []).append(r['f1'])

            # Figure 1: F1 by algorithm (box plot)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Ground Truth Validation Summary', fontsize=14, fontweight='bold')

            ax = axes[0, 0]
            algo_names_sorted = sorted(algo_f1.keys(), key=lambda k: np.mean(algo_f1[k]), reverse=True)
            data = [algo_f1[a] for a in algo_names_sorted]
            bp = ax.boxplot(data, labels=algo_names_sorted, vert=True, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#4CAF50')
                patch.set_alpha(0.7)
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score by Algorithm')
            ax.set_xticklabels(algo_names_sorted, rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='0.95')
            ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='0.99')
            ax.legend(fontsize=8)

            # Figure 2: F1 by dataset
            ax = axes[0, 1]
            ds_f1 = {}
            for r in flika_results:
                ds = r.get('dataset', 'unknown')
                ds_f1.setdefault(ds, []).append(r['f1'])
            ds_sorted = sorted(ds_f1.keys())
            data = [ds_f1[d] for d in ds_sorted]
            bp = ax.boxplot(data, labels=ds_sorted, vert=True, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('#2196F3')
                patch.set_alpha(0.7)
            ax.set_ylabel('F1 Score')
            ax.set_title('F1 Score by Dataset')
            ax.set_xticklabels(ds_sorted, rotation=45, ha='right', fontsize=8)

            # Figure 3: RMSE distribution
            ax = axes[1, 0]
            rmse_vals = [r['rmse_nm'] for r in flika_results if not np.isnan(r.get('rmse_nm', float('nan')))]
            if rmse_vals:
                ax.hist(rmse_vals, bins=30, edgecolor='black', alpha=0.7, color='#FF9800')
                ax.axvline(np.median(rmse_vals), color='red', linestyle='--',
                           label=f'median={np.median(rmse_vals):.1f}nm')
                ax.set_xlabel('RMSE (nm)')
                ax.set_ylabel('Count')
                ax.set_title('Position RMSE Distribution')
                ax.legend()

            # Figure 4: Precision vs Recall scatter
            ax = axes[1, 1]
            prec = [r['precision'] for r in flika_results]
            rec = [r['recall'] for r in flika_results]
            ax.scatter(rec, prec, s=15, alpha=0.5, c='#9C27B0')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision vs Recall')
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)

            plt.tight_layout()
            fig_path = output_dir / "ground_truth_summary.png"
            plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close()

            # Heatmap: F1 by dataset x algorithm
            fig, ax = plt.subplots(figsize=(14, 8))
            datasets = sorted(set(r.get('dataset', '') for r in flika_results))
            algorithms = sorted(set(r.get('algorithm', '') for r in flika_results))
            f1_matrix = np.full((len(datasets), len(algorithms)), np.nan)
            for r in flika_results:
                di = datasets.index(r.get('dataset', ''))
                ai = algorithms.index(r.get('algorithm', ''))
                f1_matrix[di, ai] = r['f1']

            im = ax.imshow(f1_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
            ax.set_xticks(range(len(algorithms)))
            ax.set_xticklabels(algorithms, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(datasets)))
            ax.set_yticklabels(datasets, fontsize=9)
            ax.set_title('F1 Score Heatmap: Dataset x Algorithm')
            plt.colorbar(im, ax=ax, label='F1 Score')
            # Add text annotations
            for i in range(len(datasets)):
                for j in range(len(algorithms)):
                    val = f1_matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if val < 0.7 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=7, color=color)
            plt.tight_layout()
            fig_path = output_dir / "ground_truth_heatmap.png"
            plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            self.progress_update.emit(f"Warning: Could not generate plots: {e}")

    def _generate_html_report(self, output_dir, results_data, report_type):
        """Generate an HTML report for comparison or ground truth results."""
        output_dir = Path(output_dir)
        report_path = output_dir / f"validation_report_{report_type}.html"

        try:
            html = ['<!DOCTYPE html><html><head>',
                    '<meta charset="utf-8">',
                    f'<title>FLIKA ThunderSTORM Validation Report - {report_type.title()}</title>',
                    '<style>',
                    'body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }',
                    '.container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
                    'h1 { color: #2c5aa0; }',
                    'h2 { color: #444; border-bottom: 2px solid #2c5aa0; padding-bottom: 5px; }',
                    'table { border-collapse: collapse; width: 100%; margin: 15px 0; }',
                    'th, td { padding: 8px 12px; text-align: left; border: 1px solid #ddd; }',
                    'th { background: #2c5aa0; color: white; }',
                    'tr:nth-child(even) { background: #f9f9f9; }',
                    '.good { color: #2ecc71; font-weight: bold; }',
                    '.warn { color: #f39c12; font-weight: bold; }',
                    '.bad { color: #e74c3c; font-weight: bold; }',
                    '.summary-box { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0; }',
                    'img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }',
                    '.timestamp { color: #888; font-size: 0.9em; }',
                    '</style></head><body><div class="container">']

            html.append(f'<h1>FLIKA ThunderSTORM Validation Report</h1>')
            html.append(f'<p class="timestamp">Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>')

            if report_type == 'comparison' and isinstance(results_data, dict):
                # Comparison report
                f1_scores = [s['f1'] for s in results_data.values()]
                pos_errors = [s.get('position_error_median_nm', float('nan'))
                              for s in results_data.values() if s.get('n_matched', 0) > 0]

                html.append('<div class="summary-box">')
                html.append(f'<h2>Summary</h2>')
                html.append(f'<p><strong>Tests run:</strong> {len(results_data)}</p>')
                if f1_scores:
                    html.append(f'<p><strong>Mean F1:</strong> {np.mean(f1_scores):.3f} (min: {np.min(f1_scores):.3f}, max: {np.max(f1_scores):.3f})</p>')
                if pos_errors:
                    html.append(f'<p><strong>Mean position error:</strong> {np.nanmean(pos_errors):.1f} nm</p>')
                html.append('</div>')

                html.append('<h2>Per-Configuration Results</h2>')
                html.append('<table><tr><th>Test</th><th>ImageJ</th><th>FLIKA</th><th>Matched</th><th>F1</th><th>Pos Error (nm)</th><th>Intensity Ratio</th></tr>')
                for name, stats in results_data.items():
                    f1 = stats['f1']
                    f1_class = 'good' if f1 >= 0.99 else 'warn' if f1 >= 0.95 else 'bad'
                    pos_err = f"{stats.get('position_error_median_nm', 0):.1f}" if stats.get('n_matched', 0) > 0 else 'N/A'
                    int_rat = f"{stats.get('intensity_ratio_mean', 0):.3f}" if 'intensity_ratio_mean' in stats else 'N/A'
                    html.append(f'<tr><td>{name}</td><td>{stats["n_ref"]}</td><td>{stats["n_test"]}</td>'
                                f'<td>{stats["n_matched"]}</td><td class="{f1_class}">{f1:.3f}</td>'
                                f'<td>{pos_err}</td><td>{int_rat}</td></tr>')
                html.append('</table>')

                # Embed figures
                for fig_name in ['summary_chart.png', 'speed_comparison.png']:
                    fig_path = output_dir / fig_name
                    if not fig_path.exists():
                        # Check in comparison_results subdir
                        fig_path = output_dir / "comparison_results" / fig_name
                    if fig_path.exists():
                        import base64
                        with open(fig_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        html.append(f'<h2>{fig_name.replace("_", " ").replace(".png", "").title()}</h2>')
                        html.append(f'<img src="data:image/png;base64,{img_data}" />')

            elif report_type == 'ground_truth' and isinstance(results_data, list):
                # Ground truth report
                flika_results = [r for r in results_data if r.get('label') == 'FLIKA']
                f1_scores = [r['f1'] for r in flika_results]
                rmse_vals = [r['rmse_nm'] for r in flika_results if not np.isnan(r.get('rmse_nm', float('nan')))]

                html.append('<div class="summary-box">')
                html.append(f'<h2>Summary</h2>')
                html.append(f'<p><strong>Total tests (FLIKA):</strong> {len(flika_results)}</p>')
                if f1_scores:
                    within_001 = sum(1 for f in f1_scores if f >= 0.99)
                    within_005 = sum(1 for f in f1_scores if f >= 0.95)
                    html.append(f'<p><strong>Mean F1:</strong> {np.mean(f1_scores):.3f}</p>')
                    html.append(f'<p><strong>Within 0.01 of perfect (F1 >= 0.99):</strong> {within_001}/{len(f1_scores)} ({100*within_001/len(f1_scores):.1f}%)</p>')
                    html.append(f'<p><strong>Within 0.05 of perfect (F1 >= 0.95):</strong> {within_005}/{len(f1_scores)} ({100*within_005/len(f1_scores):.1f}%)</p>')
                if rmse_vals:
                    html.append(f'<p><strong>Median RMSE:</strong> {np.median(rmse_vals):.1f} nm</p>')
                html.append('</div>')

                html.append('<h2>Per-Test Results</h2>')
                html.append('<table><tr><th>Dataset</th><th>Algorithm</th><th>Method</th><th>F1</th><th>Precision</th><th>Recall</th><th>RMSE (nm)</th><th>Detected</th><th>Ground Truth</th></tr>')
                for r in results_data:
                    f1 = r['f1']
                    f1_class = 'good' if f1 >= 0.95 else 'warn' if f1 >= 0.8 else 'bad'
                    html.append(f'<tr><td>{r.get("dataset", "")}</td><td>{r.get("algorithm", "")}</td>'
                                f'<td>{r.get("label", "")}</td><td class="{f1_class}">{f1:.3f}</td>'
                                f'<td>{r["precision"]:.3f}</td><td>{r["recall"]:.3f}</td>'
                                f'<td>{r["rmse_nm"]:.1f}</td>'
                                f'<td>{r["n_detected"]}</td><td>{r["n_ground_truth"]}</td></tr>')
                html.append('</table>')

                # Embed figures
                for fig_name in ['ground_truth_summary.png', 'ground_truth_heatmap.png']:
                    fig_path = output_dir / fig_name
                    if fig_path.exists():
                        import base64
                        with open(fig_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        html.append(f'<h2>{fig_name.replace("_", " ").replace(".png", "").title()}</h2>')
                        html.append(f'<img src="data:image/png;base64,{img_data}" />')

            html.append('</div></body></html>')

            with open(str(report_path), 'w') as f:
                f.write('\n'.join(html))
            return report_path

        except Exception as e:
            self.progress_update.emit(f"Warning: Could not generate HTML report: {e}")
            return None

    def _generate_full_report(self, analysis_dir, gt_results, real_stats, real_output_dir=None):
        """Generate a combined HTML report covering both synthetic and real data."""
        report_path = analysis_dir / "full_validation_report.html"
        try:
            html = ['<!DOCTYPE html><html><head>',
                    '<meta charset="utf-8">',
                    '<title>FLIKA ThunderSTORM Full Validation Report</title>',
                    '<style>',
                    'body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }',
                    '.container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }',
                    'h1 { color: #2c5aa0; }',
                    'h2 { color: #444; border-bottom: 2px solid #2c5aa0; padding-bottom: 5px; }',
                    'h3 { color: #666; }',
                    'table { border-collapse: collapse; width: 100%; margin: 15px 0; }',
                    'th, td { padding: 8px 12px; text-align: left; border: 1px solid #ddd; }',
                    'th { background: #2c5aa0; color: white; }',
                    'tr:nth-child(even) { background: #f9f9f9; }',
                    '.good { color: #2ecc71; font-weight: bold; }',
                    '.warn { color: #f39c12; font-weight: bold; }',
                    '.bad { color: #e74c3c; font-weight: bold; }',
                    '.summary-box { background: #e8f4fd; padding: 15px; border-radius: 5px; margin: 15px 0; }',
                    'img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }',
                    '.timestamp { color: #888; font-size: 0.9em; }',
                    '</style></head><body><div class="container">']

            html.append('<h1>FLIKA ThunderSTORM Full Validation Report</h1>')
            html.append(f'<p class="timestamp">Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>')

            # Real data section
            if real_stats:
                html.append('<h2>Real Data Comparison (FLIKA vs ImageJ)</h2>')
                f1_scores = [s['f1'] for s in real_stats.values()]
                html.append('<div class="summary-box">')
                html.append(f'<p><strong>Mean F1:</strong> {np.mean(f1_scores):.3f}</p>')
                html.append(f'<p><strong>Tests:</strong> {len(real_stats)}</p>')
                html.append('</div>')

                html.append('<table><tr><th>Config</th><th>F1</th><th>Pos Error (nm)</th><th>ImageJ</th><th>FLIKA</th></tr>')
                for name, stats in real_stats.items():
                    f1 = stats['f1']
                    f1_class = 'good' if f1 >= 0.99 else 'warn' if f1 >= 0.95 else 'bad'
                    pos_err = f"{stats.get('position_error_median_nm', 0):.1f}" if stats.get('n_matched', 0) > 0 else 'N/A'
                    html.append(f'<tr><td>{name}</td><td class="{f1_class}">{f1:.3f}</td><td>{pos_err}</td>'
                                f'<td>{stats["n_ref"]}</td><td>{stats["n_test"]}</td></tr>')
                html.append('</table>')

                # Embed real data figures
                if real_output_dir:
                    for fig_name in ['summary_chart.png', 'speed_comparison.png']:
                        for subdir in ['comparison_results', '']:
                            fig_path = Path(real_output_dir) / subdir / fig_name if subdir else Path(real_output_dir) / fig_name
                            if fig_path.exists():
                                import base64
                                with open(fig_path, 'rb') as f:
                                    img_data = base64.b64encode(f.read()).decode('utf-8')
                                html.append(f'<img src="data:image/png;base64,{img_data}" />')
                                break

            # Ground truth section
            if gt_results:
                flika_gt = [r for r in gt_results if r.get('label') == 'FLIKA']
                html.append('<h2>Synthetic Data - Ground Truth Comparison</h2>')
                f1_scores = [r['f1'] for r in flika_gt]
                if f1_scores:
                    html.append('<div class="summary-box">')
                    html.append(f'<p><strong>Tests:</strong> {len(flika_gt)}</p>')
                    html.append(f'<p><strong>Mean F1:</strong> {np.mean(f1_scores):.3f}</p>')
                    within_001 = sum(1 for f in f1_scores if f >= 0.99)
                    within_005 = sum(1 for f in f1_scores if f >= 0.95)
                    html.append(f'<p><strong>F1 >= 0.99:</strong> {within_001}/{len(f1_scores)}</p>')
                    html.append(f'<p><strong>F1 >= 0.95:</strong> {within_005}/{len(f1_scores)}</p>')
                    html.append('</div>')

                # Embed ground truth figures
                for fig_name in ['ground_truth_summary.png', 'ground_truth_heatmap.png']:
                    fig_path = analysis_dir / fig_name
                    if fig_path.exists():
                        import base64
                        with open(fig_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                        html.append(f'<img src="data:image/png;base64,{img_data}" />')

            html.append('</div></body></html>')
            with open(str(report_path), 'w') as f:
                f.write('\n'.join(html))
            return report_path
        except Exception as e:
            self.progress_update.emit(f"Warning: Could not generate report: {e}")
            return None


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

    # Kalman filter parameters
    measurement_noise_std: float = 1.0
    process_noise_std_brownian: float = 3.0
    process_noise_std_linear: float = 1.0
    process_noise_std_confined: float = 2.0

    # Adaptive gating
    gating_n_sigma: float = 3.0

    # Merge/split intensity validation
    merge_split_intensity_validation: bool = True
    intensity_ratio_penalty_weight: float = 2.0

    # Gap-closing mobility scaling
    gap_closing_use_mobility_scaling: bool = True

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


class KalmanFilter2D:
    """Standard 2-D Kalman filter with state [x, y, vx, vy].

    Supports three motion types via different transition matrices:
    - **brownian**: velocity decays to zero each step (random walk).
    - **linear**: constant-velocity model with persistence.
    - **confined**: Ornstein-Uhlenbeck restoring force towards a centre.
    """

    def __init__(self, motion_type: str = 'brownian',
                 process_noise_std: float = 3.0,
                 measurement_noise_std: float = 1.0,
                 dt: float = 1.0,
                 velocity_persistence: float = 0.8,
                 confinement_strength: float = 0.1):
        self.motion_type = motion_type
        self.dt = dt
        self.velocity_persistence = velocity_persistence
        self.confinement_strength = confinement_strength
        self.confinement_center: Optional[np.ndarray] = None

        # State and covariance
        self.x = np.zeros(4)          # [x, y, vx, vy]
        self.P = np.eye(4) * 100.0    # large initial uncertainty

        # Measurement model: observe [x, y]
        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Measurement noise
        self.R = np.eye(2) * measurement_noise_std ** 2

        # Process noise
        q = process_noise_std ** 2
        self.Q = np.diag([q * dt, q * dt, q, q])

        # Cached prediction state
        self._x_pred: Optional[np.ndarray] = None
        self._P_pred: Optional[np.ndarray] = None

    def _get_transition_matrix(self) -> np.ndarray:
        """Build the state transition matrix F for the current motion type."""
        dt = self.dt
        F = np.eye(4)

        if self.motion_type == 'brownian':
            # Position propagates, velocity → 0
            F[0, 2] = dt
            F[1, 3] = dt
            F[2, 2] = 0.0
            F[3, 3] = 0.0

        elif self.motion_type == 'linear':
            # Constant velocity with persistence
            F[0, 2] = dt
            F[1, 3] = dt
            F[2, 2] = self.velocity_persistence
            F[3, 3] = self.velocity_persistence

        elif self.motion_type == 'confined':
            k = self.confinement_strength
            F[0, 2] = dt
            F[1, 3] = dt
            F[2, 2] = 0.8  # velocity damping
            F[3, 3] = 0.8
            # Restoring force towards confinement centre
            if self.confinement_center is not None:
                # The restoring term is handled as a control input in predict()
                pass
            F[0, 0] = 1.0 - k * dt
            F[1, 1] = 1.0 - k * dt

        return F

    def initialize(self, z: np.ndarray):
        """Initialize state from first measurement [x, y]."""
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.eye(4) * 100.0

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Standard Kalman predict step.  Returns (x_pred, P_pred)."""
        F = self._get_transition_matrix()
        self._x_pred = F @ self.x

        # Confined model: add restoring force control input
        if self.motion_type == 'confined' and self.confinement_center is not None:
            k = self.confinement_strength
            cx, cy = self.confinement_center
            self._x_pred[0] += k * self.dt * cx  # counterpart to -(k*dt)*x in F
            self._x_pred[1] += k * self.dt * cy
            self._x_pred[2] -= k * (self.x[0] - cx)
            self._x_pred[3] -= k * (self.x[1] - cy)

        self._P_pred = F @ self.P @ F.T + self.Q
        return self._x_pred.copy(), self._P_pred.copy()

    def innovation(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute innovation residual y and covariance S."""
        if self._x_pred is None:
            self.predict()
        y = z - self.H @ self._x_pred
        S = self.H @ self._P_pred @ self.H.T + self.R
        return y, S

    def innovation_likelihood(self, z: np.ndarray) -> float:
        """Gaussian likelihood N(y; 0, S) for GPB1 model selection."""
        y, S = self.innovation(z)
        try:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                return 1e-300
            S_inv = np.linalg.inv(S)
            mahal = float(y @ S_inv @ y)
            log_lik = -0.5 * (logdet + mahal + 2 * np.log(2 * np.pi))
            return max(np.exp(log_lik), 1e-300)
        except np.linalg.LinAlgError:
            return 1e-300

    def update(self, z: np.ndarray) -> np.ndarray:
        """Standard Kalman update.  Returns updated state."""
        if self._x_pred is None:
            self.predict()
        y, S = self.innovation(z)
        try:
            K = self._P_pred @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((4, 2))
        self.x = self._x_pred + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self._P_pred @ I_KH.T + K @ self.R @ K.T  # Joseph form
        # Clear prediction cache
        self._x_pred = None
        self._P_pred = None
        return self.x.copy()

    def get_search_radius(self, n_sigma: float = 3.0) -> float:
        """Return adaptive search radius from innovation covariance S."""
        if self._P_pred is None:
            return n_sigma * 5.0  # fallback
        S = self.H @ self._P_pred @ self.H.T + self.R
        try:
            eigvals = np.linalg.eigvalsh(S)
            return n_sigma * np.sqrt(max(eigvals.max(), 0.01))
        except np.linalg.LinAlgError:
            return n_sigma * 5.0

    def copy(self) -> 'KalmanFilter2D':
        """Deep copy for parallel filter banks."""
        import copy as _copy
        return _copy.deepcopy(self)


class MixedMotionPredictor:
    """
    Multi-model motion predictor with GPB1 (Generalised Pseudo-Bayesian
    first-order) model selection and legacy heuristic fallback.

    Maintains per-track filter banks (3 parallel KalmanFilter2D instances)
    that are updated at each frame via GPB1 probability mixing.
    """

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.logger = get_logger('mixed_motion_predictor')

        # Default (global) model weights — used by legacy path
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

        # B1: Per-track GPB1 filter banks
        self._track_filters: Dict[int, Dict[str, 'KalmanFilter2D']] = {}
        self._track_model_probs: Dict[int, Dict[str, float]] = {}
        self._track_prev_model: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # B2: GPB1 filter initialisation
    # ------------------------------------------------------------------

    def init_track_filters(self, track_id: int, z: np.ndarray):
        """Create 3 parallel KalmanFilter2D instances for a new track.

        Parameters
        ----------
        track_id : int
            Unique track identifier.
        z : ndarray shape (2,)
            First measurement [x, y].
        """
        meas_noise = getattr(self.config, 'measurement_noise_std', 1.0)
        q_brownian = getattr(self.config, 'process_noise_std_brownian', 3.0)
        q_linear = getattr(self.config, 'process_noise_std_linear', 1.0)
        q_confined = getattr(self.config, 'process_noise_std_confined', 2.0)
        vel_persist = self.config.linear_velocity_persistence

        filters = {}
        for mtype, q_std in [('brownian', q_brownian),
                              ('linear', q_linear),
                              ('confined', q_confined)]:
            kf = KalmanFilter2D(
                motion_type=mtype,
                process_noise_std=q_std,
                measurement_noise_std=meas_noise,
                dt=1.0,
                velocity_persistence=vel_persist,
                confinement_strength=0.1,
            )
            kf.initialize(z)
            filters[mtype] = kf

        self._track_filters[track_id] = filters
        self._track_model_probs[track_id] = {
            'brownian': 0.4, 'linear': 0.4, 'confined': 0.2
        }
        self._track_prev_model[track_id] = 'brownian'

    # ------------------------------------------------------------------
    # B3: GPB1 predict
    # ------------------------------------------------------------------

    def predict_gpb1(self, track_id: int,
                     confinement_center: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """Run predict() on all 3 filters and return predictions dict.

        Each entry has keys 'state', 'covariance', 'likelihood' (model
        probability) and an extra 'kf' reference to the KalmanFilter2D
        for downstream use.
        """
        filters = self._track_filters[track_id]
        probs = self._track_model_probs[track_id]
        predictions = {}

        for mtype, kf in filters.items():
            if mtype == 'confined' and confinement_center is not None:
                kf.confinement_center = confinement_center
            x_pred, P_pred = kf.predict()
            predictions[mtype] = {
                'state': x_pred.copy(),
                'covariance': P_pred.copy(),
                'likelihood': probs.get(mtype, 0.33),
                'kf': kf,
            }

        return predictions

    # ------------------------------------------------------------------
    # B4: GPB1 update
    # ------------------------------------------------------------------

    def update_gpb1(self, track_id: int,
                    z: np.ndarray) -> Tuple[str, np.ndarray, np.ndarray]:
        """GPB1 measurement update for all 3 filters.

        Returns
        -------
        best_model : str
        best_state : ndarray shape (4,)
        best_P     : ndarray shape (4, 4)
        """
        filters = self._track_filters[track_id]
        probs = self._track_model_probs[track_id]
        models = list(filters.keys())

        # Compute innovation likelihoods
        likelihoods = {}
        for mtype in models:
            likelihoods[mtype] = filters[mtype].innovation_likelihood(z)

        # GPB1 probability update
        new_probs = {}
        for j in models:
            mixed = 0.0
            for i in models:
                tp = self.transition_probs.get((i, j), 0.0)
                mixed += tp * probs.get(i, 0.33)
            new_probs[j] = mixed * likelihoods[j]

        total = sum(new_probs.values())
        if total > 0:
            for j in models:
                new_probs[j] /= total
        else:
            new_probs = {m: 1.0 / len(models) for m in models}

        self._track_model_probs[track_id] = new_probs

        # Update all filters with measurement
        for mtype in models:
            filters[mtype].update(z)

        # Best model
        best_model = max(new_probs, key=new_probs.get)

        # Regime change detection: reinitialize non-winning filters
        prev_model = self._track_prev_model.get(track_id)
        if prev_model is not None and best_model != prev_model:
            winner_kf = filters[best_model]
            for mtype in models:
                if mtype != best_model:
                    filters[mtype].x = winner_kf.x.copy()
                    filters[mtype].P = winner_kf.P.copy()
        self._track_prev_model[track_id] = best_model

        best_kf = filters[best_model]
        return best_model, best_kf.x.copy(), best_kf.P.copy()

    # ------------------------------------------------------------------
    # B5: predict_multiple_models — GPB1 delegation with legacy fallback
    # ------------------------------------------------------------------

    def predict_multiple_models(self, state_history: List[np.ndarray],
                              dt: float = 1.0,
                              track_id: Optional[int] = None) -> Dict[str, Dict]:
        """Generate predictions from multiple motion models.

        When *track_id* is provided and GPB1 filters exist for it, the
        GPB1 path is used.  Otherwise falls through to legacy heuristic.
        """
        # GPB1 path
        if track_id is not None and track_id in self._track_filters:
            if len(state_history) >= 3:
                positions = np.array([[s[0], s[1]] for s in state_history])
                center = positions.mean(axis=0)
            else:
                center = None
            return self.predict_gpb1(track_id, confinement_center=center)

        # Legacy heuristic path
        predictions = {}

        if len(state_history) < 1:
            state = np.array([0, 0, 0, 0])
            predictions['brownian'] = self._predict_brownian(state, dt)
            predictions['linear'] = self._predict_linear(state, dt)
            predictions['confined'] = self._predict_confined(state, dt)
        else:
            last_state = state_history[-1]
            predictions['brownian'] = self._predict_brownian(last_state, dt)
            predictions['linear'] = self._predict_linear(last_state, dt)

            if len(state_history) >= 3:
                positions = np.array([[s[0], s[1]] for s in state_history])
                center = positions.mean(axis=0)
            else:
                center = None
            predictions['confined'] = self._predict_confined(last_state, dt, center=center)

            if len(state_history) >= 3:
                self._update_model_weights(state_history)

        return predictions

    # ------------------------------------------------------------------
    # B6: cleanup
    # ------------------------------------------------------------------

    def cleanup_track(self, track_id: int):
        """Remove GPB1 state for a pruned track."""
        self._track_filters.pop(track_id, None)
        self._track_model_probs.pop(track_id, None)
        self._track_prev_model.pop(track_id, None)

    # ------------------------------------------------------------------
    # B7: Legacy prediction helpers (unchanged)
    # ------------------------------------------------------------------

    def _predict_brownian(self, state: np.ndarray, dt: float) -> Dict:
        """Brownian motion prediction (deterministic — noise is in covariance)."""
        x, y, vx, vy = state
        noise_scale = self.config.brownian_noise_multiplier

        predicted_state = np.array([
            x + vx * dt,
            y + vy * dt,
            0.0,
            0.0
        ])

        covariance = np.diag([noise_scale**2 * dt, noise_scale**2 * dt,
                             noise_scale**2, noise_scale**2])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['brownian']
        }

    def _predict_linear(self, state: np.ndarray, dt: float) -> Dict:
        """Linear/directed motion prediction (deterministic)."""
        x, y, vx, vy = state
        velocity_persistence = self.config.linear_velocity_persistence
        noise_scale = 1.0

        predicted_state = np.array([
            x + vx * dt,
            y + vy * dt,
            vx * velocity_persistence,
            vy * velocity_persistence
        ])

        covariance = np.diag([noise_scale * dt, noise_scale * dt,
                             noise_scale * 0.1, noise_scale * 0.1])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['linear']
        }

    def _predict_confined(self, state: np.ndarray, dt: float,
                          center: Optional[np.ndarray] = None) -> Dict:
        """Confined motion prediction (deterministic)."""
        x, y, vx, vy = state
        confinement_strength = 0.1
        noise_scale = 2.0

        if center is None:
            cx, cy = x, y
        else:
            cx, cy = center

        predicted_state = np.array([
            x + vx * dt - confinement_strength * (x - cx) * dt,
            y + vy * dt - confinement_strength * (y - cy) * dt,
            vx * 0.8 - confinement_strength * (x - cx),
            vy * 0.8 - confinement_strength * (y - cy)
        ])

        covariance = np.diag([noise_scale * dt, noise_scale * dt,
                             noise_scale * 0.5, noise_scale * 0.5])

        return {
            'state': predicted_state,
            'covariance': covariance,
            'likelihood': self.model_weights['confined']
        }

    def _update_model_weights(self, state_history: List[np.ndarray]):
        """Update model weights based on recent motion patterns (legacy)."""
        if len(state_history) < 3:
            return

        recent_states = state_history[-3:]
        velocities = []
        for i in range(len(recent_states)-1):
            dx = recent_states[i+1][0] - recent_states[i][0]
            dy = recent_states[i+1][1] - recent_states[i][1]
            velocities.append([dx, dy])

        velocities = np.array(velocities)

        if len(velocities) >= 2:
            velocity_consistency = np.mean([
                np.dot(velocities[i], velocities[i+1]) /
                (np.linalg.norm(velocities[i]) * np.linalg.norm(velocities[i+1]) + 1e-6)
                for i in range(len(velocities)-1)
            ])

            speeds = np.linalg.norm(velocities, axis=1)
            speed_variability = np.std(speeds) / (np.mean(speeds) + 1e-6)

            if velocity_consistency > 0.5:
                self.model_weights['linear'] *= 1.2
            else:
                self.model_weights['linear'] *= 0.8

            if speed_variability > 1.0:
                self.model_weights['brownian'] *= 1.2
            else:
                self.model_weights['brownian'] *= 0.9

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
        """Main tracking function with mixed motion model support.

        Step 0 (optional): Drift correction on detections.
        Step 1: Frame-to-frame linking via forward tracking pass (with
                optional iterative smoothing for mixed-motion mode).
        Step 2: Gap closing, merge, and split detection via a second LAP,
                matching the U-Track 2.5 algorithm.
        """
        if len(detections) < 1:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        total_particles = sum(len(det) for det in detections)
        self.logger.info(f"Processing {len(detections)} frames with {total_particles} "
                         f"total particles using {self.config.motion_model} motion model")

        # Step 0 — optional drift correction
        if getattr(self.config, 'enable_drift_correction', False):
            detections = self._apply_drift_correction(detections)

        try:
            # Step 1 — frame-to-frame linking
            if self.config.motion_model == 'mixed' and self.config.enable_iterative_smoothing:
                tracks_df = self._track_with_iterative_smoothing(detections)
            else:
                tracks_df = self._track_with_single_model(detections)

            # Step 2 — gap closing / merge / split
            if len(tracks_df) > 0:
                tracks_df = self._close_gaps(tracks_df)

            return tracks_df

        except Exception as e:
            self.logger.error(f"Mixed motion tracking failed: {e}")
            return self._fallback_simple_tracking(detections)

    def _track_with_iterative_smoothing(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """
        PMMS-style tracking with iterative smoothing (Forward-Reverse-Forward)
        and Fraser-Potter fusion between forward and backward passes.
        """
        self.logger.info("Running PMMS-style tracking with iterative smoothing")

        best_tracks = pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        for round_num in range(self.config.num_tracking_rounds):
            self.logger.debug(f"Tracking round {round_num + 1}/{self.config.num_tracking_rounds}")

            if round_num == 0:
                # F1: Forward pass — store filter estimates for fusion
                tracks = self._forward_tracking_pass(detections, store_filter_estimates=True)
            elif round_num == 1:
                # F1: Reverse pass — store filter estimates, then fuse
                tracks = self._reverse_tracking_pass(
                    detections, best_tracks, store_filter_estimates=True)
                tracks = self._fuse_forward_backward(tracks)
            else:
                # Final forward pass with regime information
                tracks = self._final_forward_pass(detections, best_tracks)

            if len(tracks) > len(best_tracks):
                best_tracks = tracks
                self.logger.debug(f"Improved tracks: {len(tracks)} points")

        return best_tracks

    def _fuse_forward_backward(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Fraser-Potter covariance-intersection fusion of forward and backward estimates.

        For each (track_id, frame) where both forward and backward Kalman
        estimates exist, compute:
            P_fused = (P_f^-1 + P_b^-1)^-1
            x_fused = P_fused @ (P_f^-1 @ x_f + P_b^-1 @ x_b)
        and update x, y in the tracks DataFrame.
        """
        fwd = getattr(self, '_forward_estimates', {})
        bwd = getattr(self, '_backward_estimates', {})
        if not fwd or not bwd:
            return tracks_df

        tracks_df = tracks_df.copy()
        fused_count = 0

        for key in fwd:
            if key not in bwd:
                continue
            tid, frame = key
            x_f, P_f = fwd[key]
            x_b, P_b = bwd[key]

            try:
                P_f_inv = np.linalg.inv(P_f)
                P_b_inv = np.linalg.inv(P_b)
                P_fused = np.linalg.inv(P_f_inv + P_b_inv)
                x_fused = P_fused @ (P_f_inv @ x_f + P_b_inv @ x_b)

                # Update x, y in tracks_df for this (track_id, frame)
                mask = (tracks_df['track_id'] == tid) & (tracks_df['frame'] == frame)
                if mask.any():
                    tracks_df.loc[mask, 'x'] = x_fused[0]
                    tracks_df.loc[mask, 'y'] = x_fused[1]
                    fused_count += 1
            except np.linalg.LinAlgError:
                continue

        self.logger.debug(f"Fraser-Potter fusion: {fused_count} points fused")
        return tracks_df

    def _apply_drift_correction(self, detections: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """Apply cross-correlation drift correction to detections before tracking.

        Concatenates all frame detections into one DataFrame, computes drift,
        applies correction, then splits back into per-frame DataFrames.
        """
        try:
            from thunderstorm_python.postprocessing import DriftCorrector

            # Concatenate all detections with frame column
            all_dets = []
            for frame_idx, frame_dets in enumerate(detections):
                if len(frame_dets) == 0:
                    continue
                fd = frame_dets.copy()
                fd['frame'] = frame_idx
                all_dets.append(fd)

            if not all_dets:
                return detections

            combined = pd.concat(all_dets, ignore_index=True)

            corrector = DriftCorrector(method='cross_correlation', smoothing=0.25)
            corrected_df, drift_x, drift_y = corrector.apply_drift_correction_df(combined)

            self._drift_vectors = (drift_x, drift_y)
            self.logger.info(f"Drift correction applied: max drift = "
                             f"({np.max(np.abs(drift_x)):.2f}, {np.max(np.abs(drift_y)):.2f})")

            # Split back into per-frame DataFrames
            corrected_dets = []
            for frame_idx in range(len(detections)):
                frame_mask = corrected_df['frame'] == frame_idx
                frame_data = corrected_df[frame_mask].drop(columns=['frame'], errors='ignore')
                corrected_dets.append(frame_data.reset_index(drop=True))

            return corrected_dets

        except Exception as e:
            self.logger.warning(f"Drift correction failed, using uncorrected: {e}")
            return detections

    def _forward_tracking_pass(self, detections: List[pd.DataFrame],
                              store_filter_estimates: bool = False) -> pd.DataFrame:
        """Forward tracking pass with mixed motion prediction.

        When *store_filter_estimates* is True the per-(track, frame) Kalman
        state and covariance are stored in ``self._forward_estimates`` for
        subsequent Fraser-Potter fusion.
        """
        use_gpb1 = (self.config.motion_model == 'mixed')
        n_sigma = getattr(self.config, 'gating_n_sigma', 3.0)

        tracks = []
        active_tracks: Dict[int, Dict] = {}
        self.active_tracks = active_tracks  # expose for _solve_assignment_problem
        next_track_id = 1

        if store_filter_estimates:
            self._forward_estimates: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        for frame_idx, frame_detections in enumerate(detections):
            if len(frame_detections) == 0:
                continue

            current_detections = frame_detections.copy().reset_index(drop=True)
            frame_assignments = {}

            # Predict existing track positions using mixed motion models
            predicted_positions = {}
            track_search_radii = {}
            for track_id, track_info in active_tracks.items():
                if use_gpb1 and track_id in getattr(self.motion_predictor, '_track_filters', {}):
                    # GPB1 Kalman prediction path
                    positions = np.array([[s[0], s[1]] for s in track_info['state_history']])
                    center = positions.mean(axis=0) if len(positions) >= 3 else None
                    predictions = self.motion_predictor.predict_gpb1(track_id, confinement_center=center)
                    combined_prediction = self._combine_motion_predictions(predictions)
                    predicted_positions[track_id] = combined_prediction
                    # Extract Kalman search radius from best model's filter
                    best_probs = self.motion_predictor._track_model_probs.get(track_id, {})
                    if best_probs:
                        best_m = max(best_probs, key=best_probs.get)
                        kf = self.motion_predictor._track_filters[track_id].get(best_m)
                        if kf is not None:
                            track_search_radii[track_id] = kf.get_search_radius(n_sigma)
                elif self.config.motion_model == 'mixed':
                    predictions = self.motion_predictor.predict_multiple_models(
                        track_info['state_history'])
                    combined_prediction = self._combine_motion_predictions(predictions)
                    predicted_positions[track_id] = combined_prediction
                else:
                    last_state = track_info['state_history'][-1] if track_info['state_history'] else np.array([0, 0, 0, 0])
                    predicted_positions[track_id] = self._predict_single_model(last_state)

            # Assignment using LAP
            if active_tracks and len(current_detections) > 0:
                assignments = self._solve_assignment_problem(
                    predicted_positions, current_detections, frame_idx,
                    track_search_radii=track_search_radii if track_search_radii else None)
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

                    z = np.array([detection['x'], detection['y']])

                    # GPB1 Kalman update or legacy heuristic
                    if use_gpb1 and track_id in getattr(self.motion_predictor, '_track_filters', {}):
                        best_model, best_state, best_P = self.motion_predictor.update_gpb1(track_id, z)
                        new_state = best_state.copy()
                        if store_filter_estimates:
                            self._forward_estimates[(track_id, frame_idx)] = (best_state.copy(), best_P.copy())
                    else:
                        new_state = np.array([detection['x'], detection['y'], 0, 0])
                        if track_id in active_tracks:
                            if len(active_tracks[track_id]['state_history']) > 0:
                                prev_state = active_tracks[track_id]['state_history'][-1]
                                new_state[2] = detection['x'] - prev_state[0]
                                new_state[3] = detection['y'] - prev_state[1]

                    if track_id in active_tracks:
                        active_tracks[track_id]['state_history'].append(new_state)
                        active_tracks[track_id]['last_seen'] = frame_idx
                        active_tracks[track_id]['last_intensity'] = detection.get('intensity', 100.0)

                    if track_id not in self.track_histories:
                        self.track_histories[track_id] = []
                    self.track_histories[track_id].append(new_state)

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

                init_state = np.array([detection['x'], detection['y'], 0, 0])
                active_tracks[next_track_id] = {
                    'state_history': [init_state],
                    'last_seen': frame_idx,
                    'last_intensity': detection.get('intensity', 100.0)
                }
                self.track_histories[next_track_id] = [init_state]

                # Initialize GPB1 filters for new track
                if use_gpb1:
                    self.motion_predictor.init_track_filters(
                        next_track_id, np.array([detection['x'], detection['y']]))

                next_track_id += 1

            # Remove old tracks
            pruned_tids = [tid for tid, info in active_tracks.items()
                           if frame_idx - info['last_seen'] > self.config.max_gap_frames]
            for tid in pruned_tids:
                if use_gpb1:
                    self.motion_predictor.cleanup_track(tid)
                del active_tracks[tid]

        return pd.DataFrame(tracks) if tracks else pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

    def _reverse_tracking_pass(self, detections: List[pd.DataFrame],
                             forward_tracks: pd.DataFrame,
                             store_filter_estimates: bool = False) -> pd.DataFrame:
        """Reverse tracking pass to improve track continuity.

        Processes frames from last to first, seeded by the forward-pass
        tracks.  When *store_filter_estimates* is True, stores per-(track,
        frame) Kalman state and covariance in ``self._backward_estimates``
        for subsequent Fraser-Potter fusion.
        """
        if len(forward_tracks) == 0 or len(detections) < 2:
            return forward_tracks

        use_gpb1 = (self.config.motion_model == 'mixed')
        n_sigma = getattr(self.config, 'gating_n_sigma', 3.0)

        n_frames = len(detections)
        tracks = []
        active_tracks: Dict[int, Dict] = {}
        next_track_id = int(forward_tracks['track_id'].max()) + 1

        if store_filter_estimates:
            self._backward_estimates: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

        # Initialize a fresh backward predictor with its own GPB1 banks
        bwd_predictor = MixedMotionPredictor(self.config)

        # Seed active tracks from forward-pass last observations
        for tid in forward_tracks['track_id'].unique():
            tdata = forward_tracks[forward_tracks['track_id'] == tid].sort_values('frame')
            if len(tdata) == 0:
                continue
            last_pt = tdata.iloc[-1]
            # Build state history (reversed, so most recent = first entry)
            history = []
            for _, row in tdata.iloc[::-1].iterrows():
                vx, vy = 0.0, 0.0
                history.append(np.array([row['x'], row['y'], vx, vy]))
            for k in range(len(history) - 1):
                history[k][2] = history[k][0] - history[k + 1][0]
                history[k][3] = history[k][1] - history[k + 1][1]

            active_tracks[tid] = {
                'state_history': history,
                'last_seen_reverse': int(last_pt['frame']),
                'last_intensity': float(last_pt.get('intensity', 100.0)),
            }

            # Initialize GPB1 filters for backward direction
            if use_gpb1:
                z = np.array([last_pt['x'], last_pt['y']])
                bwd_predictor.init_track_filters(tid, z)

        # Process frames in reverse
        for rev_idx in range(n_frames - 1, -1, -1):
            frame_dets = detections[rev_idx]
            if len(frame_dets) == 0:
                continue

            current_dets = frame_dets.copy().reset_index(drop=True)

            # Predict backward for each active track
            predicted_positions = {}
            track_search_radii = {}
            for tid, tinfo in active_tracks.items():
                if not tinfo['state_history']:
                    continue
                last_state = tinfo['state_history'][-1]
                if use_gpb1 and tid in getattr(bwd_predictor, '_track_filters', {}):
                    positions = np.array([[s[0], s[1]] for s in tinfo['state_history']])
                    center = positions.mean(axis=0) if len(positions) >= 3 else None
                    preds = bwd_predictor.predict_gpb1(tid, confinement_center=center)
                    predicted_positions[tid] = self._combine_motion_predictions(preds)
                    # Extract Kalman search radius
                    best_probs = bwd_predictor._track_model_probs.get(tid, {})
                    if best_probs:
                        best_m = max(best_probs, key=best_probs.get)
                        kf = bwd_predictor._track_filters[tid].get(best_m)
                        if kf is not None:
                            track_search_radii[tid] = kf.get_search_radius(n_sigma)
                elif self.config.motion_model == 'mixed':
                    preds = bwd_predictor.predict_multiple_models(
                        tinfo['state_history'], dt=1.0)
                    predicted_positions[tid] = self._combine_motion_predictions(preds)
                else:
                    predicted_positions[tid] = self._predict_single_model(last_state)

            # Assignment
            if active_tracks and len(current_dets) > 0:
                assignments = self._solve_assignment_problem(
                    predicted_positions, current_dets, rev_idx,
                    track_search_radii=track_search_radii if track_search_radii else None)
            else:
                assignments = {}

            unassigned = set(range(len(current_dets)))

            for tid, det_idx in assignments.items():
                if det_idx in unassigned:
                    det = current_dets.iloc[det_idx]
                    tracks.append({
                        'particle_id': det.get('particle_id', det_idx),
                        'track_id': tid,
                        'frame': rev_idx,
                        'x': det['x'],
                        'y': det['y'],
                        'intensity': det.get('intensity', 100.0),
                    })

                    z = np.array([det['x'], det['y']])

                    if use_gpb1 and tid in getattr(bwd_predictor, '_track_filters', {}):
                        best_model, best_state, best_P = bwd_predictor.update_gpb1(tid, z)
                        new_state = best_state.copy()
                        if store_filter_estimates:
                            self._backward_estimates[(tid, rev_idx)] = (best_state.copy(), best_P.copy())
                    else:
                        new_state = np.array([det['x'], det['y'], 0.0, 0.0])
                        if tid in active_tracks and active_tracks[tid]['state_history']:
                            prev = active_tracks[tid]['state_history'][-1]
                            new_state[2] = det['x'] - prev[0]
                            new_state[3] = det['y'] - prev[1]

                    if tid in active_tracks:
                        active_tracks[tid]['state_history'].append(new_state)
                        active_tracks[tid]['last_seen_reverse'] = rev_idx
                        active_tracks[tid]['last_intensity'] = det.get('intensity', 100.0)
                    unassigned.remove(det_idx)

            # New tracks for unassigned detections
            for det_idx in unassigned:
                det = current_dets.iloc[det_idx]
                tracks.append({
                    'particle_id': det.get('particle_id', det_idx),
                    'track_id': next_track_id,
                    'frame': rev_idx,
                    'x': det['x'],
                    'y': det['y'],
                    'intensity': det.get('intensity', 100.0),
                })
                init_state = np.array([det['x'], det['y'], 0.0, 0.0])
                active_tracks[next_track_id] = {
                    'state_history': [init_state],
                    'last_seen_reverse': rev_idx,
                    'last_intensity': det.get('intensity', 100.0),
                }
                if use_gpb1:
                    bwd_predictor.init_track_filters(next_track_id, np.array([det['x'], det['y']]))
                next_track_id += 1

            # Prune tracks not seen recently (in reverse direction)
            pruned_tids = [tid for tid, info in active_tracks.items()
                           if abs(rev_idx - info['last_seen_reverse']) > self.config.max_gap_frames]
            for tid in pruned_tids:
                if use_gpb1:
                    bwd_predictor.cleanup_track(tid)
                del active_tracks[tid]

        reverse_df = pd.DataFrame(tracks) if tracks else pd.DataFrame(
            columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        # Merge: for each track ID present in both passes, keep the one
        # with more points; add any reverse-only tracks as new.
        if len(reverse_df) == 0:
            return forward_tracks

        fwd_counts = forward_tracks.groupby('track_id').size()
        rev_counts = reverse_df.groupby('track_id').size()

        result_parts = []
        all_tids = set(fwd_counts.index) | set(rev_counts.index)
        for tid in all_tids:
            fwd_n = fwd_counts.get(tid, 0)
            rev_n = rev_counts.get(tid, 0)
            if rev_n > fwd_n:
                result_parts.append(reverse_df[reverse_df['track_id'] == tid])
            elif fwd_n > 0:
                result_parts.append(forward_tracks[forward_tracks['track_id'] == tid])
            else:
                result_parts.append(reverse_df[reverse_df['track_id'] == tid])

        return pd.concat(result_parts, ignore_index=True) if result_parts else forward_tracks

    def _final_forward_pass(self, detections: List[pd.DataFrame],
                           previous_tracks: pd.DataFrame) -> pd.DataFrame:
        """Final forward pass incorporating motion regime information.

        Uses GPB1 model probabilities (when available) to bias per-track
        weights, falling back to sliding-window MotionRegimeDetector for
        tracks without GPB1 state.
        """
        if len(previous_tracks) == 0:
            return self._forward_tracking_pass(detections)

        # Detect motion regimes for each track and set model weights
        for tid in previous_tracks['track_id'].unique():
            # Group I: prefer GPB1 probabilities when available
            gpb1_probs = getattr(self.motion_predictor, '_track_model_probs', {})
            if tid in gpb1_probs:
                probs = gpb1_probs[tid]
                best_model = max(probs, key=probs.get)
                self.motion_predictor.model_weights = {
                    'brownian': 0.6 if best_model == 'brownian' else 0.2,
                    'linear': 0.6 if best_model == 'linear' else 0.2,
                    'confined': 0.6 if best_model == 'confined' else 0.2,
                }
            else:
                # Fallback to sliding-window regime detector
                tdata = previous_tracks[previous_tracks['track_id'] == tid].sort_values('frame')
                if len(tdata) < 4:
                    continue
                trajectory = tdata[['frame', 'x', 'y']].values
                regimes = self.regime_detector.detect_motion_regimes(trajectory)
                if regimes:
                    last_regime = regimes[-1]['type']
                    self.motion_predictor.model_weights = {
                        'brownian': 0.6 if last_regime == 'brownian' else 0.2,
                        'linear': 0.6 if last_regime == 'linear' else 0.2,
                        'confined': 0.6 if last_regime == 'confined' else 0.2,
                    }

        # Re-run forward pass with regime-informed weights
        result = self._forward_tracking_pass(detections)

        # Reset weights to defaults
        self.motion_predictor.model_weights = {
            'brownian': 0.4, 'linear': 0.4, 'confined': 0.2
        }

        return result if len(result) >= len(previous_tracks) else previous_tracks

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
                                detections: pd.DataFrame, frame_idx: int,
                                track_search_radii: Optional[Dict[int, float]] = None) -> Dict[int, int]:
        """Solve augmented LAP between predictions and detections.

        Uses the same U-Track 2.5-style augmented cost matrix as
        UTrackLinker: adaptive birth/death costs based on the 90th
        percentile of valid linking costs, and local-density-scaled
        search radii.  When *track_search_radii* is provided (from
        Kalman innovation covariance), those radii take precedence.
        """
        if not predicted_positions or len(detections) == 0:
            return {}

        track_ids = list(predicted_positions.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        max_dist = self.config.max_linking_distance

        # Build detection position array for vectorised operations
        det_pos = detections[['x', 'y']].values  # (n_dets, 2)

        # G2: Per-track search radius — Kalman-derived when available
        time_reach_conf = getattr(self.config, 'time_reach_conf_b', 4)
        per_track_radius = np.full(n_tracks, max_dist)
        for i, tid in enumerate(track_ids):
            if track_search_radii is not None and tid in track_search_radii:
                # Use Kalman-derived radius, clamped to [2.0, max_dist]
                per_track_radius[i] = np.clip(track_search_radii[tid], 2.0, max_dist)
            elif tid in self.track_histories:
                age = len(self.track_histories[tid])
                if time_reach_conf > 0 and age < time_reach_conf:
                    confidence = age / time_reach_conf
                    kalman_radius = max_dist * 0.5
                    per_track_radius[i] = (1.0 - confidence) * max_dist + confidence * kalman_radius

        # Local density scaling: compute per-detection NN distance
        search_limits = np.full(n_dets, max_dist)
        if n_dets > 1:
            from scipy.spatial.distance import cdist as _cdist
            dd = _cdist(det_pos, det_pos)
            np.fill_diagonal(dd, np.inf)
            nn_dist = dd.min(axis=1)
            # Clamp search radius to half NN distance
            density_limit = np.clip(nn_dist * 0.5, 2.0, max_dist)
            search_limits = np.minimum(search_limits, density_limit)

        # G3: Base cost matrix — Kalman innovation covariance when GPB1 available
        base_cost = np.full((n_tracks, n_dets), np.inf)
        gpb1_filters = getattr(self.motion_predictor, '_track_filters', {})

        for i, tid in enumerate(track_ids):
            pred_xy = predicted_positions[tid][:2]
            dists = np.linalg.norm(det_pos - pred_xy, axis=1)

            # Try to get innovation covariance S from GPB1 Kalman filters
            pred_cov = None
            if tid in gpb1_filters:
                # Use the best model's predicted innovation covariance
                best_model_probs = getattr(self.motion_predictor, '_track_model_probs', {}).get(tid, {})
                if best_model_probs:
                    best_model = max(best_model_probs, key=best_model_probs.get)
                    kf = gpb1_filters[tid].get(best_model)
                    if kf is not None and hasattr(kf, '_P_pred') and kf._P_pred is not None:
                        H = kf.H
                        S = H @ kf._P_pred @ H.T + kf.R
                        eigvals = np.linalg.eigvalsh(S)
                        if np.all(eigvals > 0):
                            pred_cov = S

            # Fallback: empirical covariance from position history
            if pred_cov is None and tid in self.track_histories and len(self.track_histories.get(tid, [])) >= 3:
                hist = self.track_histories[tid]
                recent_pos = np.array([[s[0], s[1]] for s in hist[-min(5, len(hist)):]])
                residuals = np.diff(recent_pos, axis=0)
                if len(residuals) >= 2:
                    cov_est = np.cov(residuals.T)
                    eigvals = np.linalg.eigvalsh(cov_est)
                    if np.all(eigvals > 0):
                        pred_cov = cov_est

            effective_limit = per_track_radius[i]
            for j in range(n_dets):
                if dists[j] <= min(search_limits[j], effective_limit):
                    if pred_cov is not None:
                        dx = det_pos[j] - pred_xy
                        try:
                            S_inv = np.linalg.inv(pred_cov)
                            base_cost[i, j] = dx @ S_inv @ dx
                        except np.linalg.LinAlgError:
                            base_cost[i, j] = dists[j] ** 2
                    else:
                        base_cost[i, j] = dists[j] ** 2

        # Intensity-based cost component
        use_intensity = getattr(self.config, 'use_intensity_costs', False)
        intensity_weight = getattr(self.config, 'intensity_weight', 0.1)
        if use_intensity and intensity_weight > 0 and 'intensity' in detections.columns:
            det_intensity = detections['intensity'].values
            for i, tid in enumerate(track_ids):
                # Get track's last known intensity
                track_intensity = None
                if tid in self.track_histories and self.track_histories[tid]:
                    # State history doesn't store intensity directly, use from active tracks
                    pass  # intensity comparison skipped if not stored
                if tid in getattr(self, 'active_tracks', {}):
                    tinfo = self.active_tracks[tid]
                    if 'last_intensity' in tinfo:
                        track_intensity = tinfo['last_intensity']
                if track_intensity is not None and track_intensity > 0:
                    for j in range(n_dets):
                        if np.isfinite(base_cost[i, j]) and det_intensity[j] > 0:
                            log_ratio = abs(np.log(det_intensity[j] / track_intensity))
                            base_cost[i, j] += intensity_weight * log_ratio

        # Velocity angle constraints for linear motion
        use_velocity = getattr(self.config, 'use_velocity_costs', False)
        velocity_weight = getattr(self.config, 'velocity_weight', 0.1)
        if use_velocity and velocity_weight > 0:
            for i, tid in enumerate(track_ids):
                if tid in self.track_histories and len(self.track_histories[tid]) >= 2:
                    hist = self.track_histories[tid]
                    prev = hist[-2]
                    curr = hist[-1]
                    vel = np.array([curr[0] - prev[0], curr[1] - prev[1]])
                    speed = np.linalg.norm(vel)
                    if speed < 1e-6:
                        continue
                    pred_xy = predicted_positions[tid][:2]
                    for j in range(n_dets):
                        if not np.isfinite(base_cost[i, j]):
                            continue
                        displacement = det_pos[j] - pred_xy
                        disp_norm = np.linalg.norm(displacement)
                        if disp_norm < 1e-6:
                            continue
                        cos_theta = np.dot(vel, displacement) / (speed * disp_norm)
                        cos_theta = np.clip(cos_theta, -1.0, 1.0)
                        angle_cost = (1.0 - cos_theta) * speed
                        base_cost[i, j] += velocity_weight * angle_cost

        # Adaptive birth/death costs (90th percentile of valid costs)
        valid = base_cost[np.isfinite(base_cost)]
        if len(valid) > 0:
            alt_cost = np.percentile(valid, 90) * 1.05
            alt_cost = max(alt_cost, 4.0)  # minimum 2px^2
        else:
            alt_cost = max_dist ** 2

        # Build augmented cost matrix
        matrix_size = n_tracks + n_dets
        full_cost = np.full((matrix_size, matrix_size), np.inf)

        # Upper-left: linking costs
        full_cost[:n_tracks, :n_dets] = base_cost

        # Upper-right: death costs (diagonal)
        for i in range(n_tracks):
            full_cost[i, n_dets + i] = alt_cost

        # Lower-left: birth costs (diagonal)
        for j in range(n_dets):
            full_cost[n_tracks + j, j] = alt_cost

        # Lower-right: dummy
        dummy = np.min(valid) if len(valid) > 0 else alt_cost
        full_cost[n_tracks:, n_dets:] = dummy

        # Solve LAP
        try:
            # Replace inf with large finite for solver
            solve_cost = full_cost.copy()
            inf_mask = np.isinf(solve_cost)
            if np.any(~inf_mask):
                max_finite = np.max(solve_cost[~inf_mask])
                solve_cost[inf_mask] = max_finite * 1000
            else:
                solve_cost[inf_mask] = 1e12

            row_ind, col_ind = linear_sum_assignment(solve_cost)

            assignments = {}
            for row, col in zip(row_ind, col_ind):
                if row < n_tracks and col < n_dets and np.isfinite(base_cost[row, col]):
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

    # ------------------------------------------------------------------
    # Step 2: Gap closing / merge / split  (post-processing)
    # ------------------------------------------------------------------

    def _close_gaps(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Gap closing, merge, and split detection via a second LAP.

        Takes the DataFrame produced by the forward tracking pass,
        converts it into track segments, runs a gap-closing LAP (same
        algorithm as UTrackLinker), and returns a new DataFrame with
        gap-closed tracks.
        """
        time_window = self.config.max_gap_frames
        max_gap_radius = self.config.max_linking_distance
        gap_penalty = 1.5
        min_track_len = getattr(self.config, 'min_track_length', 2)

        # Extract per-track segment info
        track_ids = sorted(tracks_df['track_id'].unique())
        segments = []  # list of dicts with start/end frame+pos, and the df subset

        for tid in track_ids:
            tdata = tracks_df[tracks_df['track_id'] == tid].sort_values('frame')
            if len(tdata) < min_track_len:
                continue
            first = tdata.iloc[0]
            last = tdata.iloc[-1]

            # H1: compute start/end intensity and mean step size
            n_pts = len(tdata)
            intensities = tdata['intensity'].values if 'intensity' in tdata.columns else np.full(n_pts, 100.0)
            start_intensity = float(np.mean(intensities[:min(3, n_pts)]))
            end_intensity = float(np.mean(intensities[max(0, n_pts - 3):]))

            positions = tdata[['x', 'y']].values
            if n_pts >= 2:
                steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                mean_step = float(np.mean(steps)) if len(steps) > 0 else 1.0
            else:
                mean_step = 1.0
            mean_step = max(mean_step, 0.1)  # avoid division by zero

            segments.append({
                'tid': tid,
                'start_frame': int(first['frame']),
                'end_frame': int(last['frame']),
                'start_pos': np.array([first['x'], first['y']]),
                'end_pos': np.array([last['x'], last['y']]),
                'start_intensity': start_intensity,
                'end_intensity': end_intensity,
                'mean_step': mean_step,
                'data': tdata,
            })

        n_seg = len(segments)
        if n_seg < 2:
            return tracks_df

        # H2: Build gap-closing cost matrix with mobility scaling
        use_mobility = getattr(self.config, 'gap_closing_use_mobility_scaling', False)
        gc_cost = np.full((n_seg, n_seg), np.inf)
        for i in range(n_seg):
            for j in range(n_seg):
                if i == j:
                    continue
                dt = segments[j]['start_frame'] - segments[i]['end_frame']
                if dt < 1 or dt > time_window:
                    continue
                d = np.sqrt(np.sum((segments[i]['end_pos'] - segments[j]['start_pos']) ** 2))
                if d > max_gap_radius * np.sqrt(dt):
                    continue
                if use_mobility:
                    avg_step = 0.5 * (segments[i]['mean_step'] + segments[j]['mean_step'])
                    avg_step = max(avg_step, 0.1)
                    gc_cost[i, j] = (d / avg_step) ** 2 * (gap_penalty ** (dt - 1))
                else:
                    gc_cost[i, j] = d ** 2 * (gap_penalty ** (dt - 1))

        # Merge/split candidates
        do_merge = getattr(self.config, 'enable_merging', False)
        do_split = getattr(self.config, 'enable_splitting', False)

        merge_candidates = []
        split_candidates = []

        if do_merge or do_split:
            # Build interior lookup per segment
            seg_interior = []
            for seg in segments:
                interior = {}
                for _, row in seg['data'].iterrows():
                    f = int(row['frame'])
                    interior[f] = np.array([row['x'], row['y']])
                seg_interior.append(interior)

            # H3/H4: intensity validation for merge/split
            do_intensity_val = getattr(self.config, 'merge_split_intensity_validation', False)
            intensity_penalty_w = getattr(self.config, 'intensity_ratio_penalty_weight', 2.0)

            # Build per-segment per-frame intensity lookup
            seg_intensity = []
            for seg in segments:
                frame_int = {}
                for _, row in seg['data'].iterrows():
                    f = int(row['frame'])
                    frame_int[f] = float(row.get('intensity', 100.0))
                seg_intensity.append(frame_int)

            if do_merge:
                for i in range(n_seg):
                    for j in range(n_seg):
                        if i == j:
                            continue
                        end_f = segments[i]['end_frame']
                        for abs_f, pos_j in seg_interior[j].items():
                            dt = abs_f - end_f
                            if dt < 0 or dt > 1:
                                continue
                            d2 = np.sum((segments[i]['end_pos'] - pos_j) ** 2)
                            if np.sqrt(d2) > max_gap_radius:
                                continue
                            cost = d2
                            # H3: merge intensity validation
                            if do_intensity_val:
                                I_seg_i_end = segments[i]['end_intensity']
                                I_seg_j_at_merge = seg_intensity[j].get(abs_f, 100.0)
                                I_after = seg_intensity[j].get(abs_f, 100.0)
                                expected = I_seg_i_end + I_seg_j_at_merge
                                if expected > 0:
                                    rho = I_after / expected
                                    cost *= 1.0 + intensity_penalty_w * abs(rho - 1.0)
                            merge_candidates.append((i, j, abs_f, cost))

            if do_split:
                for j in range(n_seg):
                    for i in range(n_seg):
                        if i == j:
                            continue
                        start_f = segments[j]['start_frame']
                        for abs_f, pos_i in seg_interior[i].items():
                            dt = start_f - abs_f
                            if dt < 0 or dt > 1:
                                continue
                            d2 = np.sum((pos_i - segments[j]['start_pos']) ** 2)
                            if np.sqrt(d2) > max_gap_radius:
                                continue
                            cost = d2
                            # H4: split intensity validation
                            if do_intensity_val:
                                I_before = seg_intensity[i].get(abs_f, 100.0)
                                I_i_after = seg_intensity[i].get(abs_f + 1, segments[i]['end_intensity'])
                                I_j_start = segments[j]['start_intensity']
                                if I_before > 0:
                                    rho = (I_i_after + I_j_start) / I_before
                                    cost *= 1.0 + intensity_penalty_w * abs(rho - 1.0)
                            split_candidates.append((i, j, abs_f, cost))

        n_merge = len(merge_candidates)
        n_split = len(split_candidates)

        # Build augmented cost matrix
        n_rows = n_seg + n_split
        n_cols = n_seg + n_merge
        matrix_size = n_rows + n_cols
        gc_full = np.full((matrix_size, matrix_size), np.inf)

        # Gap closing block
        gc_full[:n_seg, :n_seg] = gc_cost

        # Merge costs
        for m_idx, (end_i, _, _, cost) in enumerate(merge_candidates):
            gc_full[end_i, n_seg + m_idx] = cost

        # Split costs
        for s_idx, (_, start_j, _, cost) in enumerate(split_candidates):
            gc_full[n_seg + s_idx, start_j] = cost

        # Adaptive alternative cost
        all_block = gc_full[:n_rows, :n_cols]
        valid_costs = all_block[np.isfinite(all_block)]

        if len(valid_costs) > 0:
            alt_cost = np.percentile(valid_costs, 90) * 1.05
            alt_cost = max(alt_cost, 4.0)
        else:
            # No valid gap-closing candidates — skip
            return tracks_df

        for r in range(n_rows):
            gc_full[r, n_cols + r] = alt_cost
        for c in range(n_cols):
            gc_full[n_rows + c, c] = alt_cost
        dummy = np.min(valid_costs) if len(valid_costs) > 0 else alt_cost
        gc_full[n_rows:, n_cols:] = dummy

        # Solve LAP
        solve_cost = gc_full.copy()
        inf_mask = np.isinf(solve_cost)
        if np.any(~inf_mask):
            max_finite = np.max(solve_cost[~inf_mask])
            solve_cost[inf_mask] = max_finite * 1000
        else:
            return tracks_df

        try:
            row_ind, col_ind = linear_sum_assignment(solve_cost)
        except Exception as e:
            self.logger.warning(f"Gap closing LAP failed: {e}")
            return tracks_df

        # Parse gap-closing assignments
        merge_map = {}  # seg_i -> seg_j (gap close)
        for row, col in zip(row_ind, col_ind):
            if row < n_seg and col < n_seg and np.isfinite(gc_cost[row, col]):
                if row != col:
                    merge_map[row] = col

        self.logger.debug(f"Gap closing: {len(merge_map)} closures from {n_seg} segments, "
                          f"{n_merge} merge candidates, {n_split} split candidates")

        if not merge_map:
            return tracks_df

        # Chain merged segments and renumber tracks
        merged_into = set(merge_map.values())
        visited = set()
        new_tracks = []
        new_tid = 1

        for seed in range(n_seg):
            if seed in visited or seed in merged_into:
                continue
            chain = [seed]
            visited.add(seed)
            current = seed
            while current in merge_map:
                nxt = merge_map[current]
                if nxt in visited:
                    break
                chain.append(nxt)
                visited.add(nxt)
                current = nxt

            # Combine DataFrames from chain
            combined = pd.concat([segments[idx]['data'] for idx in chain], ignore_index=True)
            combined['track_id'] = new_tid
            new_tracks.append(combined)
            new_tid += 1

        # Add unvisited (isolated) segments
        for idx in range(n_seg):
            if idx not in visited:
                seg_data = segments[idx]['data'].copy()
                seg_data['track_id'] = new_tid
                new_tracks.append(seg_data)
                new_tid += 1
                visited.add(idx)

        result = pd.concat(new_tracks, ignore_index=True)
        self.logger.info(f"Gap closing: {n_seg} segments -> {result['track_id'].nunique()} tracks "
                         f"({len(merge_map)} gap closures)")
        return result

    # ------------------------------------------------------------------
    # Post-tracking analysis methods
    # ------------------------------------------------------------------

    def add_msd_analysis(self, tracks_df, pixel_size=1.0, frame_interval=1.0,
                         classification_config=None):
        """Compute per-track MSD, diffusion coefficient, anomalous exponent.

        Adds columns: D_um2s, alpha, motion_type, msd_r_squared.

        Parameters
        ----------
        tracks_df : DataFrame
            Must contain track_number (or track_id), frame, x, y.
        pixel_size : float
            Pixel size in µm (positions are multiplied by this).
        frame_interval : float
            Time between frames in seconds.
        classification_config : dict or None
            Optional overrides: msd_min_points, msd_fitting_points,
            anomalous_alpha_low, anomalous_alpha_high.

        Returns
        -------
        tracks_df : DataFrame
            Input DataFrame with new columns added.
        """
        cfg = classification_config or {}
        min_pts = cfg.get('msd_min_points', 5)
        n_fit = cfg.get('msd_fitting_points', 10)
        alpha_low = cfg.get('anomalous_alpha_low', 0.7)
        alpha_high = cfg.get('anomalous_alpha_high', 1.3)

        # Support both track_number (main pipeline) and track_id (U-Track internal)
        tid_col = 'track_number' if 'track_number' in tracks_df.columns else 'track_id'

        tracks_df = tracks_df.copy()
        tracks_df['D_um2s'] = np.nan
        tracks_df['alpha'] = np.nan
        tracks_df['motion_type'] = 'unknown'
        tracks_df['msd_r_squared'] = np.nan

        for tid in tracks_df[tid_col].unique():
            mask = tracks_df[tid_col] == tid
            tdata = tracks_df.loc[mask].sort_values('frame')
            if len(tdata) < min_pts:
                continue

            xy = tdata[['x', 'y']].values * pixel_size
            n = len(xy)

            # Compute MSD for each lag
            max_lag = min(n - 1, max(n_fit, 10))
            lags = np.arange(1, max_lag + 1)
            msd = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                displacements = xy[lag:] - xy[:-lag]
                msd[i] = np.mean(np.sum(displacements ** 2, axis=1))

            if len(msd) < 3:
                continue

            # Fit D from linear MSD: MSD = 4*D*t + offset (2D diffusion)
            tau = lags[:n_fit] * frame_interval
            msd_fit = msd[:n_fit]
            if len(tau) >= 2:
                coeffs = np.polyfit(tau, msd_fit, 1)
                slope = coeffs[0]
                D = max(slope / 4.0, 0.0)
                # R² for linear fit
                msd_pred = np.polyval(coeffs, tau)
                ss_res = np.sum((msd_fit - msd_pred) ** 2)
                ss_tot = np.sum((msd_fit - np.mean(msd_fit)) ** 2)
                r2_linear = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            else:
                D = 0.0
                r2_linear = 0.0

            # Fit anomalous exponent: log(MSD) = alpha*log(tau) + log(4*D_alpha)
            valid = msd[:n_fit] > 0
            if np.sum(valid) >= 2:
                log_tau = np.log(tau[valid])
                log_msd = np.log(msd[:n_fit][valid])
                coeffs_log = np.polyfit(log_tau, log_msd, 1)
                alpha_val = coeffs_log[0]
            else:
                alpha_val = 1.0

            # Classify motion type
            if alpha_val < alpha_low:
                motion = 'confined'
            elif alpha_val > alpha_high:
                motion = 'directed'
            else:
                motion = 'brownian'

            tracks_df.loc[mask, 'D_um2s'] = D
            tracks_df.loc[mask, 'alpha'] = alpha_val
            tracks_df.loc[mask, 'motion_type'] = motion
            tracks_df.loc[mask, 'msd_r_squared'] = r2_linear

        return tracks_df

    def classify_track_motion(self, tracks_df, method='msd',
                              pixel_size=1.0, frame_interval=1.0,
                              classification_config=None):
        """Classify motion type per track.

        Parameters
        ----------
        tracks_df : DataFrame
        method : str
            'msd' uses MSDAnalyzer, 'velocity' uses directional persistence.
        pixel_size, frame_interval : float
        classification_config : ClassificationConfig or None

        Returns
        -------
        tracks_df : DataFrame with 'motion_type' column added/updated.
        """
        if method == 'msd':
            return self.add_msd_analysis(tracks_df, pixel_size, frame_interval,
                                         classification_config)
        else:
            # Existing directional persistence method
            tid_col = 'track_number' if 'track_number' in tracks_df.columns else 'track_id'
            tracks_df = tracks_df.copy()
            for tid in tracks_df[tid_col].unique():
                mask = tracks_df[tid_col] == tid
                tdata = tracks_df.loc[mask].sort_values('frame')
                if len(tdata) < 4:
                    tracks_df.loc[mask, 'motion_type'] = 'unknown'
                    continue
                trajectory = tdata[['frame', 'x', 'y']].values
                regimes = self.regime_detector.detect_motion_regimes(trajectory)
                if regimes:
                    tracks_df.loc[mask, 'motion_type'] = regimes[-1]['type']
                else:
                    tracks_df.loc[mask, 'motion_type'] = 'brownian'
            return tracks_df

    def compute_track_quality(self, tracks_df, classification_config=None):
        """Compute per-track quality score (0–1).

        Score is based on:
        - Track length (longer = higher)
        - Gap fraction (fewer gaps = higher)
        - Intensity consistency (lower CoV = higher)
        - MSD fit R² (if available)

        Adds 'quality_score' column.  Optionally filters by min_quality_score.
        """
        cfg = classification_config or {}
        gap_penalty = cfg.get('gap_penalty', 0.1)
        intensity_w = cfg.get('intensity_consistency_weight', 0.2)
        min_quality = cfg.get('min_quality_score', 0.0)

        # Support both track_number (main pipeline) and track_id (U-Track internal)
        tid_col = 'track_number' if 'track_number' in tracks_df.columns else 'track_id'

        tracks_df = tracks_df.copy()
        tracks_df['quality_score'] = 0.0

        for tid in tracks_df[tid_col].unique():
            mask = tracks_df[tid_col] == tid
            tdata = tracks_df.loc[mask].sort_values('frame')
            n_points = len(tdata)

            if n_points == 0:
                continue

            # Length score (sigmoid-like, saturates around 20 frames)
            length_score = min(1.0, n_points / 20.0)

            # Gap fraction score
            if n_points >= 2:
                frame_span = tdata['frame'].max() - tdata['frame'].min() + 1
                gap_frac = 1.0 - n_points / max(frame_span, 1)
                gap_score = max(0.0, 1.0 - gap_penalty * gap_frac * 10)
            else:
                gap_score = 1.0

            # Intensity consistency score
            intensity_score = 1.0
            if 'intensity' in tdata.columns and n_points >= 3:
                intensities = tdata['intensity'].values
                mean_int = np.mean(intensities)
                if mean_int > 0:
                    cov = np.std(intensities) / mean_int
                    intensity_score = max(0.0, 1.0 - cov)

            # MSD R² score (if available)
            msd_score = 0.5  # neutral default
            if 'msd_r_squared' in tdata.columns:
                r2 = tdata['msd_r_squared'].iloc[0]
                if not np.isnan(r2):
                    msd_score = r2

            # Combined weighted score
            quality = (
                0.3 * length_score
                + 0.3 * gap_score
                + intensity_w * intensity_score
                + (0.4 - intensity_w) * msd_score
            )
            quality = np.clip(quality, 0.0, 1.0)
            tracks_df.loc[mask, 'quality_score'] = quality

        # Optional filtering
        if min_quality > 0:
            keep_tids = tracks_df.groupby(tid_col)['quality_score'].first()
            keep_tids = keep_tids[keep_tids >= min_quality].index
            tracks_df = tracks_df[tracks_df[tid_col].isin(keep_tids)].copy()

        return tracks_df

    def add_velocity_persistence(self, tracks_df, pixel_size=1.0):
        """Add velocity autocorrelation and persistence metrics.

        Adds columns: persistence_ratio, velocity_autocorr_tau1, confinement_ratio.
        """
        # Support both track_number (main pipeline) and track_id (U-Track internal)
        tid_col = 'track_number' if 'track_number' in tracks_df.columns else 'track_id'

        tracks_df = tracks_df.copy()
        tracks_df['persistence_ratio'] = np.nan
        tracks_df['velocity_autocorr_tau1'] = np.nan
        tracks_df['confinement_ratio'] = np.nan

        for tid in tracks_df[tid_col].unique():
            mask = tracks_df[tid_col] == tid
            tdata = tracks_df.loc[mask].sort_values('frame')
            if len(tdata) < 4:
                continue

            xy = tdata[['x', 'y']].values * pixel_size

            # Directional persistence: mean cos(angle) between consecutive steps
            steps = np.diff(xy, axis=0)
            step_norms = np.linalg.norm(steps, axis=1)
            valid = step_norms > 1e-12
            if np.sum(valid[:-1] & valid[1:]) > 0:
                cos_angles = []
                for i in range(len(steps) - 1):
                    if step_norms[i] > 1e-12 and step_norms[i + 1] > 1e-12:
                        cos_a = np.dot(steps[i], steps[i + 1]) / (step_norms[i] * step_norms[i + 1])
                        cos_angles.append(np.clip(cos_a, -1.0, 1.0))
                persistence = float(np.mean(cos_angles)) if cos_angles else 0.0
            else:
                persistence = 0.0

            # Velocity autocorrelation at lag 1
            if len(steps) >= 2:
                v_mean = steps - np.mean(steps, axis=0)
                v_var = np.sum(v_mean ** 2) / len(v_mean)
                if v_var > 1e-12:
                    autocorr_1 = np.sum(v_mean[:-1] * v_mean[1:]) / ((len(v_mean) - 1) * v_var)
                    tau1 = float(autocorr_1)
                else:
                    tau1 = 0.0
            else:
                tau1 = 0.0

            # Confinement ratio: max displacement / total path length
            total_path = np.sum(step_norms)
            max_disp = np.max(np.linalg.norm(xy - xy[0], axis=1))
            confinement = float(max_disp / total_path) if total_path > 1e-12 else 0.0

            tracks_df.loc[mask, 'persistence_ratio'] = persistence
            tracks_df.loc[mask, 'velocity_autocorr_tau1'] = tau1
            tracks_df.loc[mask, 'confinement_ratio'] = confinement

        return tracks_df


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

        # U-Track 2.5 cost and drift parameters
        config.use_intensity_costs = getattr(self.parameters, 'utrack_use_intensity_costs', True)
        config.intensity_weight = 0.1
        config.use_velocity_costs = getattr(self.parameters, 'utrack_use_velocity_costs', False)
        config.velocity_weight = 0.1
        config.enable_drift_correction = getattr(self.parameters, 'utrack_enable_drift_correction', False)
        config.time_reach_conf_b = 4
        config.time_reach_conf_l = 4

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

class AutoBackgroundDetector:
    """Automatic background ROI detection for TIFF image stacks.

    Finds the region with lowest intensity and highest temporal stability
    to use as a background ROI. Outputs ROI files in the same format as
    ROIBackgroundSubtractor.parse_roi_file expects.
    """

    @staticmethod
    def load_and_transform_image(file_path):
        """Load TIFF and apply same transforms as analysis pipeline."""
        A = skio.imread(file_path, plugin='tifffile')
        A = np.rot90(A, axes=(1, 2))
        A = np.fliplr(A)
        return A

    @staticmethod
    def compute_max_projection(image_array):
        """Compute maximum intensity projection across all frames."""
        return np.max(image_array, axis=0)

    @staticmethod
    def scan_candidates(max_proj, roi_width, roi_height, stride):
        """Stage 1: Scan all positions on max projection, rank by mean intensity (ascending).

        Returns list of (mean_intensity, y, x) sorted lowest-first.
        """
        h, w = max_proj.shape
        candidates = []
        for y in range(0, h - roi_height + 1, stride):
            for x in range(0, w - roi_width + 1, stride):
                region = max_proj[y:y + roi_height, x:x + roi_width]
                candidates.append((float(np.mean(region)), y, x))
        candidates.sort(key=lambda c: c[0])
        return candidates

    @staticmethod
    def verify_temporal_stability(image_array, candidates, roi_width, roi_height, top_n):
        """Stage 2: From top-N lowest-intensity candidates, pick the one with lowest temporal std.

        Returns (best_candidate, all_evaluated) where best_candidate is
        (temporal_std, mean_intensity, y, x).
        """
        evaluated = []
        for mean_int, y, x in candidates[:top_n]:
            roi_stack = image_array[:, y:y + roi_height, x:x + roi_width]
            frame_means = np.mean(roi_stack, axis=(1, 2))
            temporal_std = float(np.std(frame_means))
            evaluated.append((temporal_std, mean_int, y, x))
        evaluated.sort(key=lambda e: e[0])
        return evaluated[0], evaluated

    @staticmethod
    def write_roi_file(output_path, y, x, width, height):
        """Write ROI file in ROIBackgroundSubtractor.parse_roi_file format.

        Format:
            rectangle
            y x
            width height
        """
        with open(output_path, 'w') as f:
            f.write('rectangle\n')
            f.write(f'{y} {x}\n')
            f.write(f'{width} {height}\n')

    @staticmethod
    def detect_background_roi(file_path, roi_width=5, roi_height=5, stride=5,
                              top_n=10, enable_temporal_check=True):
        """Full detection pipeline. Returns result dict and saves ROI file.

        The ROI file is saved as ROI_{basename}.txt in the same directory
        as the input file, matching the naming convention expected by
        add_background_subtraction().
        """
        image_array = AutoBackgroundDetector.load_and_transform_image(file_path)
        max_proj = AutoBackgroundDetector.compute_max_projection(image_array)

        candidates = AutoBackgroundDetector.scan_candidates(max_proj, roi_width, roi_height, stride)

        if not candidates:
            return None

        if enable_temporal_check and len(candidates) > 1:
            actual_top_n = min(top_n, len(candidates))
            best, evaluated = AutoBackgroundDetector.verify_temporal_stability(
                image_array, candidates, roi_width, roi_height, actual_top_n)
            temporal_std, mean_intensity, best_y, best_x = best
        else:
            mean_intensity, best_y, best_x = candidates[0]
            temporal_std = None

        # Build output path: ROI_{basename}.txt
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(os.path.dirname(file_path), f'ROI_{base_name}.txt')

        AutoBackgroundDetector.write_roi_file(output_path, best_y, best_x, roi_width, roi_height)

        result = {
            'file_path': file_path,
            'roi_file': output_path,
            'y': best_y,
            'x': best_x,
            'width': roi_width,
            'height': roi_height,
            'mean_intensity': mean_intensity,
            'temporal_std': temporal_std,
            'image_shape': image_array.shape,
            'candidates_scanned': len(candidates),
        }
        return result


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

        # Auto background detection parameters
        self.auto_background_detection_mode = 'manual'  # 'manual' or 'auto'
        self.auto_bg_roi_width = 5
        self.auto_bg_roi_height = 5
        self.auto_bg_scan_stride = 5
        self.auto_bg_top_n = 10
        self.auto_bg_enable_temporal_check = True

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
        # ThunderSTORM detection parameters
        self.detection_method = 'utrack'  # 'utrack' or 'thunderstorm'
        # ThunderSTORM filter parameters
        self.ts_filter_type = 'wavelet'
        self.ts_filter_scale = 2          # wavelet scale (1-5)
        self.ts_filter_order = 3          # B-spline order (1-5, 3=cubic)
        self.ts_filter_sigma = 1.6        # Gaussian/Lowered Gaussian sigma
        self.ts_filter_sigma1 = 1.0       # DoG sigma1
        self.ts_filter_sigma2 = 1.6       # DoG sigma2
        self.ts_filter_size1 = 3          # box/median/diff_avg size 1
        self.ts_filter_size2 = 5          # diff_avg size 2
        self.ts_filter_pattern = 'box'    # median pattern: 'box' or 'cross'
        # ThunderSTORM detector parameters
        self.ts_detector_type = 'local_maximum'
        self.ts_detector_threshold = 'std(Wave.F1)'
        self.ts_detector_connectivity = '8-neighbourhood'
        self.ts_detector_radius = 1       # NMS dilation radius
        # ThunderSTORM sub-pixel localization parameters
        self.ts_psf_model = 'integrated_gaussian'  # PSF model / estimator
        self.ts_fitting_method = 'least_squares'    # optimizer (for PSF models)
        self.ts_fitter_type = 'gaussian_lsq'        # internal combined key (derived)
        self.ts_fit_radius = 3
        self.ts_initial_sigma = 1.3
        # Sigma / FWHM range constraint (fit quality gate)
        self.ts_sigma_min = None          # min sigma in pixels (None = no limit)
        self.ts_sigma_max = None          # max sigma in pixels (None = no limit)
        self.ts_fwhm_min_nm = None        # min FWHM in nm (alternative)
        self.ts_fwhm_max_nm = None        # max FWHM in nm (alternative)
        # Advanced ThunderSTORM options
        self.ts_use_watershed = True      # watershed segmentation for centroid detector
        self.ts_multi_emitter_enabled = False  # multi-emitter fitting
        self.ts_multi_emitter_max = 5     # max emitters per region
        self.ts_multi_emitter_pvalue = 1e-6  # model selection p-value
        self.ts_multi_emitter_keep_same_intensity = True  # force similar intensity across emitters
        self.ts_multi_emitter_fixed_intensity = False      # constrain intensity range
        self.ts_multi_emitter_intensity_min = 500          # min photons (if fixed)
        self.ts_multi_emitter_intensity_max = 2500         # max photons (if fixed)
        self.ts_mfa_fitting_method = 'wlsq'               # MFA fitting: 'wlsq' or 'mle'
        self.ts_mfa_model_selection_iterations = 50        # iterations during model comparison
        self.ts_mfa_enable_final_refit = True              # refit winning model with unlimited iterations
        # ThunderSTORM camera parameters
        self.ts_photons_per_adu = 3.6
        self.ts_baseline = 100.0
        self.ts_is_em_gain = True
        self.ts_em_gain = 100.0
        self.ts_quantum_efficiency = 1.0


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

        # U-Track 2.5 post-tracking analysis parameters
        self.utrack_enable_msd_analysis = True
        self.utrack_enable_quality_scoring = True
        self.utrack_enable_velocity_persistence = True
        self.utrack_enable_drift_correction = False
        self.utrack_pixel_size_um = 0.16    # µm per pixel for MSD D calculation
        self.utrack_frame_interval = 0.05   # seconds between frames
        self.utrack_use_intensity_costs = True
        self.utrack_use_velocity_costs = False
        self.utrack_analysis_available = False  # dynamic flag for export control

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
            elif 'intensity [AU]' in missing_locs.columns:
                missing_locs['intensity'] = missing_locs['intensity [AU]']
                print(f"    Using intensity [AU] values from localization file")
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
        self.create_background_tab()
        self.create_geometric_analysis_tab()
        self.create_autocorrelation_tab()
        self.create_progress_tab()
        self.create_export_control_tab()
        self.create_thunderstorm_tab()  # ThunderSTORM macro generation

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
        """Create detection configuration tab with U-Track and ThunderSTORM options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # ========== Detection Method Selection ==========
        method_group = QGroupBox("Detection Method Selection")
        method_layout = QVBoxLayout(method_group)

        # Radio buttons for detection method
        self.detection_method_group = QButtonGroup()

        self.utrack_method_radio = QRadioButton("U-Track Based Detection")
        self.utrack_method_radio.setChecked(self.parameters.detection_method == 'utrack')
        self.detection_method_group.addButton(self.utrack_method_radio, 0)
        method_layout.addWidget(self.utrack_method_radio)

        self.thunderstorm_method_radio = QRadioButton("ThunderSTORM Detection")
        self.thunderstorm_method_radio.setChecked(self.parameters.detection_method == 'thunderstorm')
        self.thunderstorm_method_radio.setEnabled(THUNDERSTORM_AVAILABLE)
        if not THUNDERSTORM_AVAILABLE:
            ts_warning = QLabel("⚠ ThunderSTORM package not available. Install thunderstorm_python to enable.")
            ts_warning.setStyleSheet("color: #ff6b6b; font-style: italic; margin-left: 20px;")
            method_layout.addWidget(ts_warning)
        self.detection_method_group.addButton(self.thunderstorm_method_radio, 1)
        method_layout.addWidget(self.thunderstorm_method_radio)

        # Connect method selection to update panels
        self.utrack_method_radio.toggled.connect(self.on_detection_method_changed)

        scroll_layout.addWidget(method_group)

        # ========== Enable/Disable Detection ==========
        enable_group = QGroupBox("Enable Particle Detection")
        enable_layout = QVBoxLayout(enable_group)

        self.detection_enable_checkbox = QCheckBox("Enable Particle Detection for Analysis")
        self.detection_enable_checkbox.setChecked(self.parameters.enable_detection)
        self.detection_enable_checkbox.toggled.connect(self.on_detection_enable_toggled)
        enable_layout.addWidget(self.detection_enable_checkbox)

        info_label = QLabel("Detects particles in images and saves as localization CSV files")
        info_label.setStyleSheet("color: #666; font-style: italic; margin-left: 20px;")
        enable_layout.addWidget(info_label)

        scroll_layout.addWidget(enable_group)

        # ========== Create Horizontal Splitter for Two Detection Panels ==========
        detection_splitter = QSplitter(Qt.Horizontal)

        # === LEFT PANEL: U-Track Detection ===
        self.utrack_panel = QWidget()
        utrack_layout = QVBoxLayout(self.utrack_panel)

        utrack_title = QLabel("<b>U-Track Detection Parameters</b>")
        utrack_title.setStyleSheet("font-size: 12pt; color: #2E86AB; margin-bottom: 10px;")
        utrack_layout.addWidget(utrack_title)

        utrack_params_group = QGroupBox("U-Track Detection Settings")
        utrack_params_layout = QFormLayout(utrack_params_group)

        # PSF Sigma
        self.detection_psf_sigma_spin = QDoubleSpinBox()
        self.detection_psf_sigma_spin.setRange(0.5, 10.0)
        self.detection_psf_sigma_spin.setDecimals(2)
        self.detection_psf_sigma_spin.setValue(self.parameters.detection_psf_sigma)
        self.detection_psf_sigma_spin.setSuffix(" pixels")
        utrack_params_layout.addRow("PSF Sigma:", self.detection_psf_sigma_spin)

        # Alpha threshold
        self.detection_alpha_spin = QDoubleSpinBox()
        self.detection_alpha_spin.setRange(0.001, 0.5)
        self.detection_alpha_spin.setDecimals(3)
        self.detection_alpha_spin.setValue(self.parameters.detection_alpha_threshold)
        utrack_params_layout.addRow("Significance Threshold (α):", self.detection_alpha_spin)

        # Minimum intensity
        self.detection_min_intensity_spin = QDoubleSpinBox()
        self.detection_min_intensity_spin.setRange(0, 10000)
        self.detection_min_intensity_spin.setValue(self.parameters.detection_min_intensity)
        utrack_params_layout.addRow("Minimum Intensity:", self.detection_min_intensity_spin)

        utrack_layout.addWidget(utrack_params_group)

        # U-Track description
        utrack_desc = QTextEdit()
        utrack_desc.setReadOnly(True)
        utrack_desc.setMaximumHeight(180)
        utrack_desc.setText("""U-Track Based Particle Detection:

• Background Estimation: Robust percentile-based methods
• Pre-filtering: Gaussian filtering matching PSF size
• Local Maxima Detection: Morphological peak detection
• Statistical Testing: Significance against background noise
• Sub-pixel Localization: Intensity-weighted centroid refinement

Parameters:
• PSF Sigma: Expected PSF width (1.0-2.0 pixels typically)
• Significance Threshold: Statistical confidence (0.05 = 95%)
• Minimum Intensity: Absolute intensity cutoff
        """)
        utrack_layout.addWidget(utrack_desc)

        detection_splitter.addWidget(self.utrack_panel)

        # === RIGHT PANEL: ThunderSTORM Detection ===
        self.thunderstorm_panel = QWidget()
        ts_layout = QVBoxLayout(self.thunderstorm_panel)

        ts_title = QLabel("<b>ThunderSTORM Detection Parameters</b>")
        ts_title.setStyleSheet("font-size: 12pt; color: #A23B72; margin-bottom: 10px;")
        ts_layout.addWidget(ts_title)

        # Wrap everything in a scroll area so it all fits
        ts_scroll = QScrollArea()
        ts_scroll.setWidgetResizable(True)
        ts_scroll_widget = QWidget()
        ts_scroll_layout = QVBoxLayout(ts_scroll_widget)

        # ---- Image Filtering group ----
        ts_filter_group = QGroupBox("Image Filtering")
        ts_filter_layout = QFormLayout(ts_filter_group)

        self.ts_filter_combo = QComboBox()
        self.ts_filter_combo.addItems(['wavelet', 'gaussian', 'dog', 'lowered_gaussian',
                                       'diff_avg', 'median', 'box', 'none'])
        self.ts_filter_combo.setCurrentText(self.parameters.ts_filter_type)
        self.ts_filter_combo.currentTextChanged.connect(self._on_ts_filter_type_changed)
        ts_filter_layout.addRow("Filter Type:", self.ts_filter_combo)

        # Wavelet: scale + order
        self.ts_filter_scale_spin = QSpinBox()
        self.ts_filter_scale_spin.setRange(1, 5)
        self.ts_filter_scale_spin.setValue(self.parameters.ts_filter_scale)
        self.ts_filter_scale_spin.setToolTip("Wavelet decomposition level (higher = coarser features)")
        ts_filter_layout.addRow("B-Spline Scale:", self.ts_filter_scale_spin)

        self.ts_filter_order_spin = QSpinBox()
        self.ts_filter_order_spin.setRange(1, 5)
        self.ts_filter_order_spin.setValue(self.parameters.ts_filter_order)
        self.ts_filter_order_spin.setToolTip("B-spline order: 1=linear, 3=cubic (default), 5=quintic")
        ts_filter_layout.addRow("B-Spline Order:", self.ts_filter_order_spin)

        # Gaussian / Lowered Gaussian: sigma
        self.ts_filter_sigma_spin = QDoubleSpinBox()
        self.ts_filter_sigma_spin.setRange(0.1, 20.0)
        self.ts_filter_sigma_spin.setDecimals(2)
        self.ts_filter_sigma_spin.setSingleStep(0.1)
        self.ts_filter_sigma_spin.setValue(self.parameters.ts_filter_sigma)
        self.ts_filter_sigma_spin.setSuffix(" px")
        self.ts_filter_sigma_spin.setToolTip("Gaussian sigma (for Gaussian, Lowered Gaussian filters)")
        ts_filter_layout.addRow("Filter Sigma:", self.ts_filter_sigma_spin)

        # DoG: sigma1, sigma2
        self.ts_filter_sigma1_spin = QDoubleSpinBox()
        self.ts_filter_sigma1_spin.setRange(0.1, 20.0)
        self.ts_filter_sigma1_spin.setDecimals(2)
        self.ts_filter_sigma1_spin.setSingleStep(0.1)
        self.ts_filter_sigma1_spin.setValue(self.parameters.ts_filter_sigma1)
        self.ts_filter_sigma1_spin.setSuffix(" px")
        self.ts_filter_sigma1_spin.setToolTip("Smaller Gaussian sigma (DoG)")
        ts_filter_layout.addRow("DoG Sigma 1:", self.ts_filter_sigma1_spin)

        self.ts_filter_sigma2_spin = QDoubleSpinBox()
        self.ts_filter_sigma2_spin.setRange(0.1, 20.0)
        self.ts_filter_sigma2_spin.setDecimals(2)
        self.ts_filter_sigma2_spin.setSingleStep(0.1)
        self.ts_filter_sigma2_spin.setValue(self.parameters.ts_filter_sigma2)
        self.ts_filter_sigma2_spin.setSuffix(" px")
        self.ts_filter_sigma2_spin.setToolTip("Larger Gaussian sigma (DoG, must be > Sigma 1)")
        ts_filter_layout.addRow("DoG Sigma 2:", self.ts_filter_sigma2_spin)

        # Box/Median/DiffAvg: size1, size2
        self.ts_filter_size1_spin = QSpinBox()
        self.ts_filter_size1_spin.setRange(1, 31)
        self.ts_filter_size1_spin.setSingleStep(2)
        self.ts_filter_size1_spin.setValue(self.parameters.ts_filter_size1)
        self.ts_filter_size1_spin.setSuffix(" px")
        self.ts_filter_size1_spin.setToolTip("Kernel size (Box, Median) or small kernel (Diff Avg)")
        ts_filter_layout.addRow("Kernel Size 1:", self.ts_filter_size1_spin)

        self.ts_filter_size2_spin = QSpinBox()
        self.ts_filter_size2_spin.setRange(1, 31)
        self.ts_filter_size2_spin.setSingleStep(2)
        self.ts_filter_size2_spin.setValue(self.parameters.ts_filter_size2)
        self.ts_filter_size2_spin.setSuffix(" px")
        self.ts_filter_size2_spin.setToolTip("Large kernel size (Diff Avg only, must be > Size 1)")
        ts_filter_layout.addRow("Kernel Size 2:", self.ts_filter_size2_spin)

        # Median: pattern
        self.ts_filter_pattern_combo = QComboBox()
        self.ts_filter_pattern_combo.addItems(['box', 'cross'])
        self.ts_filter_pattern_combo.setCurrentText(self.parameters.ts_filter_pattern)
        self.ts_filter_pattern_combo.setToolTip("Median neighbourhood shape: box (square) or cross (plus)")
        ts_filter_layout.addRow("Median Pattern:", self.ts_filter_pattern_combo)

        ts_scroll_layout.addWidget(ts_filter_group)

        # Show/hide filter-specific rows on init
        self._on_ts_filter_type_changed(self.parameters.ts_filter_type)

        # ---- Approximate Localization (Detection) group ----
        ts_detect_group = QGroupBox("Approximate Localization")
        ts_detect_layout = QFormLayout(ts_detect_group)

        self.ts_detector_combo = QComboBox()
        self.ts_detector_combo.addItems(['local_maximum', 'non_maximum_suppression', 'centroid'])
        self.ts_detector_combo.setCurrentText(self.parameters.ts_detector_type)
        self.ts_detector_combo.currentTextChanged.connect(self._on_ts_detector_type_changed)
        ts_detect_layout.addRow("Detector Method:", self.ts_detector_combo)

        self.ts_threshold_combo = QComboBox()
        self.ts_threshold_combo.setEditable(True)
        self.ts_threshold_combo.addItems(['std(Wave.F1)', '2*std(Wave.F1)', '3*std(Wave.F1)',
                                          'mean(Wave.F1) + 3*std(Wave.F1)', '100'])
        self.ts_threshold_combo.setCurrentText(self.parameters.ts_detector_threshold)
        self.ts_threshold_combo.setToolTip(
            "Threshold expression. Variables: I (raw image), F (filtered),\n"
            "Wave.F1 (wavelet detail), DoG.G1/G2 etc.\n"
            "Functions: std, mean, median, var, sum, abs, min, max.\n"
            "Operators: +, -, *, /, ^ (power)."
        )
        ts_detect_layout.addRow("Threshold:", self.ts_threshold_combo)

        self.ts_detector_connectivity_combo = QComboBox()
        self.ts_detector_connectivity_combo.addItems(['8-neighbourhood', '4-neighbourhood'])
        self.ts_detector_connectivity_combo.setCurrentText(self.parameters.ts_detector_connectivity)
        self.ts_detector_connectivity_combo.setToolTip(
            "Pixel connectivity for peak detection.\n"
            "8-neighbourhood: includes diagonals (default)\n"
            "4-neighbourhood: only horizontal/vertical neighbours"
        )
        ts_detect_layout.addRow("Connectivity:", self.ts_detector_connectivity_combo)

        self.ts_detector_radius_spin = QSpinBox()
        self.ts_detector_radius_spin.setRange(1, 10)
        self.ts_detector_radius_spin.setValue(self.parameters.ts_detector_radius)
        self.ts_detector_radius_spin.setSuffix(" px")
        self.ts_detector_radius_spin.setToolTip(
            "Dilation radius for non-maximum suppression.\n"
            "Larger = peaks must dominate a wider area.\n"
            "Only used with 'non_maximum_suppression' detector."
        )
        ts_detect_layout.addRow("NMS Radius:", self.ts_detector_radius_spin)

        self.ts_use_watershed_checkbox = QCheckBox("Watershed segmentation")
        self.ts_use_watershed_checkbox.setChecked(self.parameters.ts_use_watershed)
        self.ts_use_watershed_checkbox.setToolTip(
            "Split touching molecules using watershed transform.\n"
            "Only applies with 'centroid' detector."
        )
        ts_detect_layout.addRow(self.ts_use_watershed_checkbox)

        ts_scroll_layout.addWidget(ts_detect_group)

        # Show/hide detector-specific rows on init
        self._on_ts_detector_type_changed(self.parameters.ts_detector_type)

        # ---- Sub-pixel Localization (Fitting) group ----
        ts_fit_group = QGroupBox("Sub-pixel Localization")
        ts_fit_layout = QFormLayout(ts_fit_group)

        # PSF Model / Estimator (matches thunderSTORM's top-level choice)
        self.ts_psf_model_combo = QComboBox()
        self.ts_psf_model_combo.addItems([
            'PSF: Integrated Gaussian',
            'PSF: Gaussian',
            'PSF: Elliptical Gaussian (3D astigmatism)',
            'Phasor',
            'Radial symmetry',
            'Centroid of local neighbourhood',
            'No estimator',
        ])
        # Restore from saved parameter
        psf_display_map = {
            'integrated_gaussian': 'PSF: Integrated Gaussian',
            'gaussian': 'PSF: Gaussian',
            'elliptical_gaussian': 'PSF: Elliptical Gaussian (3D astigmatism)',
            'phasor': 'Phasor',
            'radial_symmetry': 'Radial symmetry',
            'centroid': 'Centroid of local neighbourhood',
            'no_estimator': 'No estimator',
        }
        saved_model = self.parameters.ts_psf_model
        self.ts_psf_model_combo.setCurrentText(
            psf_display_map.get(saved_model, 'PSF: Integrated Gaussian'))
        self.ts_psf_model_combo.currentTextChanged.connect(self._on_ts_psf_model_changed)
        ts_fit_layout.addRow("Method:", self.ts_psf_model_combo)

        # Fitting method / optimizer (only for PSF models)
        self.ts_fitting_method_combo = QComboBox()
        self.ts_fitting_method_combo.addItems([
            'Least squares',
            'Weighted least squares',
            'Maximum likelihood estimation',
        ])
        fitting_method_display = {
            'least_squares': 'Least squares',
            'weighted_least_squares': 'Weighted least squares',
            'maximum_likelihood': 'Maximum likelihood estimation',
        }
        saved_method = self.parameters.ts_fitting_method
        self.ts_fitting_method_combo.setCurrentText(
            fitting_method_display.get(saved_method, 'Least squares'))
        self.ts_fitting_method_combo.setToolTip(
            "Least squares: Fast, good default (Levenberg-Marquardt)\n"
            "Weighted least squares: Poisson variance weighting\n"
            "Maximum likelihood: Optimal for low SNR data"
        )
        ts_fit_layout.addRow("Fitting Method:", self.ts_fitting_method_combo)

        # Fitting radius (shown for all methods, label changes)
        self.ts_fit_radius_spin = QSpinBox()
        self.ts_fit_radius_spin.setRange(2, 10)
        self.ts_fit_radius_spin.setValue(self.parameters.ts_fit_radius)
        self.ts_fit_radius_spin.setSuffix(" px")
        self.ts_fit_radius_spin.setToolTip(
            "Size of the ROI around each detection for fitting/estimation.\n"
            "ROI is (2*radius+1) x (2*radius+1) pixels."
        )
        self.ts_fit_radius_label = QLabel("Fitting Radius:")
        ts_fit_layout.addRow(self.ts_fit_radius_label, self.ts_fit_radius_spin)

        # Initial sigma (only for PSF models)
        self.ts_initial_sigma_spin = QDoubleSpinBox()
        self.ts_initial_sigma_spin.setRange(0.5, 5.0)
        self.ts_initial_sigma_spin.setDecimals(2)
        self.ts_initial_sigma_spin.setSingleStep(0.1)
        self.ts_initial_sigma_spin.setValue(self.parameters.ts_initial_sigma)
        self.ts_initial_sigma_spin.setSuffix(" px")
        self.ts_initial_sigma_spin.setToolTip(
            "Initial PSF sigma guess for the optimizer.\n"
            "Should be close to the actual PSF width."
        )
        self.ts_initial_sigma_label = QLabel("Initial Sigma:")
        ts_fit_layout.addRow(self.ts_initial_sigma_label, self.ts_initial_sigma_spin)

        # FWHM / sigma range constraint (only for PSF models)
        self.ts_sigma_range_label = QLabel("Sigma range (fit rejection):")
        self.ts_sigma_range_label.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        ts_fit_layout.addRow(self.ts_sigma_range_label)

        self.ts_sigma_min_spin = QDoubleSpinBox()
        self.ts_sigma_min_spin.setRange(0.0, 20.0)
        self.ts_sigma_min_spin.setDecimals(2)
        self.ts_sigma_min_spin.setSingleStep(0.1)
        self.ts_sigma_min_spin.setSpecialValueText("No limit")
        self.ts_sigma_min_spin.setValue(self.parameters.ts_sigma_min if self.parameters.ts_sigma_min is not None else 0.0)
        self.ts_sigma_min_spin.setSuffix(" px")
        self.ts_sigma_min_spin.setToolTip(
            "Minimum fitted sigma (pixels). Fits below this are rejected.\n"
            "Set to 0 for no lower limit."
        )
        self.ts_sigma_min_label = QLabel("Min Sigma:")
        ts_fit_layout.addRow(self.ts_sigma_min_label, self.ts_sigma_min_spin)

        self.ts_sigma_max_spin = QDoubleSpinBox()
        self.ts_sigma_max_spin.setRange(0.0, 50.0)
        self.ts_sigma_max_spin.setDecimals(2)
        self.ts_sigma_max_spin.setSingleStep(0.1)
        self.ts_sigma_max_spin.setSpecialValueText("No limit")
        self.ts_sigma_max_spin.setValue(self.parameters.ts_sigma_max if self.parameters.ts_sigma_max is not None else 0.0)
        self.ts_sigma_max_spin.setSuffix(" px")
        self.ts_sigma_max_spin.setToolTip(
            "Maximum fitted sigma (pixels). Fits above this are rejected.\n"
            "Set to 0 for no upper limit."
        )
        self.ts_sigma_max_label = QLabel("Max Sigma:")
        ts_fit_layout.addRow(self.ts_sigma_max_label, self.ts_sigma_max_spin)

        ts_scroll_layout.addWidget(ts_fit_group)

        # Set initial visibility based on PSF model
        self._on_ts_psf_model_changed(self.ts_psf_model_combo.currentText())

        # ---- Multi-Emitter Analysis (MFA) group ----
        ts_mfa_group = QGroupBox("Multi-Emitter Analysis (MFA)")
        ts_mfa_layout = QFormLayout(ts_mfa_group)

        self.ts_multi_emitter_checkbox = QCheckBox("Enable multi-emitter fitting")
        self.ts_multi_emitter_checkbox.setChecked(self.parameters.ts_multi_emitter_enabled)
        self.ts_multi_emitter_checkbox.setToolTip(
            "Fit multiple overlapping emitters in each ROI using\n"
            "chi-squared log-likelihood ratio test for model selection.\n"
            "Improves detection in crowded regions but is slower."
        )
        self.ts_multi_emitter_checkbox.toggled.connect(self._on_ts_multi_emitter_toggled)
        ts_mfa_layout.addRow(self.ts_multi_emitter_checkbox)

        self.ts_multi_emitter_max_spin = QSpinBox()
        self.ts_multi_emitter_max_spin.setRange(2, 10)
        self.ts_multi_emitter_max_spin.setValue(self.parameters.ts_multi_emitter_max)
        self.ts_multi_emitter_max_spin.setToolTip("Maximum number of emitters per fitting region")
        self.ts_multi_emitter_max_spin.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow("Max Emitters (Nmax):", self.ts_multi_emitter_max_spin)

        self.ts_multi_emitter_pvalue_combo = QComboBox()
        self.ts_multi_emitter_pvalue_combo.setEditable(True)
        self.ts_multi_emitter_pvalue_combo.addItems(['1e-6', '1e-4', '1e-3', '0.01', '0.05'])
        self.ts_multi_emitter_pvalue_combo.setCurrentText(str(self.parameters.ts_multi_emitter_pvalue))
        self.ts_multi_emitter_pvalue_combo.setToolTip(
            "P-value threshold for chi-squared model selection.\n"
            "Lower = more conservative (fewer multi-emitter fits).\n"
            "Default 1e-6 matches thunderSTORM."
        )
        self.ts_multi_emitter_pvalue_combo.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow("P-value:", self.ts_multi_emitter_pvalue_combo)

        self.ts_mfa_keep_intensity_checkbox = QCheckBox("Keep same intensity")
        self.ts_mfa_keep_intensity_checkbox.setChecked(self.parameters.ts_multi_emitter_keep_same_intensity)
        self.ts_mfa_keep_intensity_checkbox.setToolTip(
            "Force all emitters in a multi-fit region to have\n"
            "similar intensity values."
        )
        self.ts_mfa_keep_intensity_checkbox.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow(self.ts_mfa_keep_intensity_checkbox)

        self.ts_mfa_fixed_intensity_checkbox = QCheckBox("Fixed intensity range")
        self.ts_mfa_fixed_intensity_checkbox.setChecked(self.parameters.ts_multi_emitter_fixed_intensity)
        self.ts_mfa_fixed_intensity_checkbox.setToolTip(
            "Constrain fitted intensity to a user-specified\n"
            "photon count range."
        )
        self.ts_mfa_fixed_intensity_checkbox.setEnabled(self.parameters.ts_multi_emitter_enabled)
        self.ts_mfa_fixed_intensity_checkbox.toggled.connect(self._on_ts_mfa_fixed_intensity_toggled)
        ts_mfa_layout.addRow(self.ts_mfa_fixed_intensity_checkbox)

        self.ts_mfa_intensity_min_spin = QSpinBox()
        self.ts_mfa_intensity_min_spin.setRange(1, 100000)
        self.ts_mfa_intensity_min_spin.setValue(self.parameters.ts_multi_emitter_intensity_min)
        self.ts_mfa_intensity_min_spin.setSuffix(" photons")
        self.ts_mfa_intensity_min_spin.setEnabled(
            self.parameters.ts_multi_emitter_enabled and self.parameters.ts_multi_emitter_fixed_intensity)
        ts_mfa_layout.addRow("Intensity Min:", self.ts_mfa_intensity_min_spin)

        self.ts_mfa_intensity_max_spin = QSpinBox()
        self.ts_mfa_intensity_max_spin.setRange(1, 100000)
        self.ts_mfa_intensity_max_spin.setValue(self.parameters.ts_multi_emitter_intensity_max)
        self.ts_mfa_intensity_max_spin.setSuffix(" photons")
        self.ts_mfa_intensity_max_spin.setEnabled(
            self.parameters.ts_multi_emitter_enabled and self.parameters.ts_multi_emitter_fixed_intensity)
        ts_mfa_layout.addRow("Intensity Max:", self.ts_mfa_intensity_max_spin)

        # MFA fitting method
        self.ts_mfa_fitting_method_combo = QComboBox()
        self.ts_mfa_fitting_method_combo.addItems(['Weighted least squares', 'Maximum likelihood'])
        mfa_method_display = {
            'wlsq': 'Weighted least squares',
            'mle': 'Maximum likelihood',
        }
        self.ts_mfa_fitting_method_combo.setCurrentText(
            mfa_method_display.get(self.parameters.ts_mfa_fitting_method, 'Weighted least squares'))
        self.ts_mfa_fitting_method_combo.setToolTip(
            "Fitting method used during multi-emitter model selection.\n"
            "WLSQ: Matches ImageJ ThunderSTORM (recommended).\n"
            "MLE: Higher statistical power but may over-split."
        )
        self.ts_mfa_fitting_method_combo.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow("MFA Fitting Method:", self.ts_mfa_fitting_method_combo)

        # Model selection iterations
        self.ts_mfa_iterations_spin = QSpinBox()
        self.ts_mfa_iterations_spin.setRange(10, 1000)
        self.ts_mfa_iterations_spin.setValue(self.parameters.ts_mfa_model_selection_iterations)
        self.ts_mfa_iterations_spin.setToolTip(
            "Maximum optimizer iterations during model comparison.\n"
            "ImageJ default: 50. Lower values are faster but may\n"
            "give less accurate model selection."
        )
        self.ts_mfa_iterations_spin.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow("Model Selection Iterations:", self.ts_mfa_iterations_spin)

        # Final refit toggle
        self.ts_mfa_final_refit_checkbox = QCheckBox("Final refit (unlimited iterations)")
        self.ts_mfa_final_refit_checkbox.setChecked(self.parameters.ts_mfa_enable_final_refit)
        self.ts_mfa_final_refit_checkbox.setToolTip(
            "After model selection, refit the winning model with\n"
            "unlimited iterations for maximum accuracy.\n"
            "Matches ImageJ ThunderSTORM behavior."
        )
        self.ts_mfa_final_refit_checkbox.setEnabled(self.parameters.ts_multi_emitter_enabled)
        ts_mfa_layout.addRow(self.ts_mfa_final_refit_checkbox)

        ts_scroll_layout.addWidget(ts_mfa_group)

        # ---- Camera Parameters group ----
        ts_camera_group = QGroupBox("Camera Parameters")
        ts_camera_layout = QFormLayout(ts_camera_group)

        self.ts_photons_per_adu_spin = QDoubleSpinBox()
        self.ts_photons_per_adu_spin.setRange(0.01, 100.0)
        self.ts_photons_per_adu_spin.setDecimals(2)
        self.ts_photons_per_adu_spin.setValue(self.parameters.ts_photons_per_adu)
        self.ts_photons_per_adu_spin.setToolTip("Photoelectrons per A/D count (from camera spec sheet)")
        ts_camera_layout.addRow("Photons/ADU:", self.ts_photons_per_adu_spin)

        self.ts_baseline_spin = QDoubleSpinBox()
        self.ts_baseline_spin.setRange(0, 10000)
        self.ts_baseline_spin.setDecimals(1)
        self.ts_baseline_spin.setValue(self.parameters.ts_baseline)
        self.ts_baseline_spin.setSuffix(" ADU")
        self.ts_baseline_spin.setToolTip("Camera baseline offset in ADU counts (subtracted before fitting)")
        ts_camera_layout.addRow("Baseline Offset:", self.ts_baseline_spin)

        self.ts_is_em_gain_checkbox = QCheckBox("EM Gain enabled (EMCCD)")
        self.ts_is_em_gain_checkbox.setChecked(self.parameters.ts_is_em_gain)
        self.ts_is_em_gain_checkbox.toggled.connect(self.on_ts_em_gain_toggled)
        ts_camera_layout.addRow(self.ts_is_em_gain_checkbox)

        self.ts_em_gain_spin = QDoubleSpinBox()
        self.ts_em_gain_spin.setRange(1, 5000)
        self.ts_em_gain_spin.setDecimals(1)
        self.ts_em_gain_spin.setValue(self.parameters.ts_em_gain)
        self.ts_em_gain_spin.setToolTip("EM multiplication gain (for EMCCD cameras)")
        self.ts_em_gain_spin.setEnabled(self.parameters.ts_is_em_gain)
        ts_camera_layout.addRow("EM Gain:", self.ts_em_gain_spin)

        self.ts_quantum_efficiency_spin = QDoubleSpinBox()
        self.ts_quantum_efficiency_spin.setRange(0.01, 1.0)
        self.ts_quantum_efficiency_spin.setDecimals(2)
        self.ts_quantum_efficiency_spin.setValue(self.parameters.ts_quantum_efficiency)
        self.ts_quantum_efficiency_spin.setToolTip("Quantum efficiency of the sensor (0-1)")
        ts_camera_layout.addRow("Quantum Efficiency:", self.ts_quantum_efficiency_spin)

        camera_info = QLabel(
            "photons = (ADU - offset) x photons/ADU / QE / EM_gain"
        )
        camera_info.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        camera_info.setWordWrap(True)
        ts_camera_layout.addRow(camera_info)

        ts_scroll_layout.addWidget(ts_camera_group)
        ts_scroll_layout.addStretch()

        ts_scroll.setWidget(ts_scroll_widget)
        ts_layout.addWidget(ts_scroll)

        detection_splitter.addWidget(self.thunderstorm_panel)

        # Set initial splitter sizes
        detection_splitter.setSizes([400, 400])

        scroll_layout.addWidget(detection_splitter)

        # ========== Output Options ==========
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

        # ========== Detection Controls ==========
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

        # Enable/disable based on initial state
        self.on_detection_enable_toggled()
        self.on_detection_method_changed()

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Detection")


    def on_detection_enable_toggled(self):
        """Handle detection enable/disable"""
        enabled = self.detection_enable_checkbox.isChecked()

        # Enable/disable both detection panels
        if hasattr(self, 'utrack_panel'):
            self.utrack_panel.setEnabled(enabled)
        if hasattr(self, 'thunderstorm_panel'):
            self.thunderstorm_panel.setEnabled(enabled)

        # Enable/disable output and controls groups
        self.detection_output_group.setEnabled(enabled)
        self.detection_controls_group.setEnabled(enabled)

        # Update start button availability
        if enabled and hasattr(self, 'file_paths') and self.file_paths:
            self.start_detection_btn.setEnabled(True)
        else:
            self.start_detection_btn.setEnabled(False)

    def on_detection_method_changed(self):
        """Handle detection method selection change"""
        is_utrack = self.utrack_method_radio.isChecked()
        detection_enabled = self.detection_enable_checkbox.isChecked()

        # Enable/disable panels based on selection AND overall detection state
        # Only enable the selected panel if detection is enabled
        if detection_enabled:
            self.utrack_panel.setEnabled(is_utrack)
            self.thunderstorm_panel.setEnabled(not is_utrack)
        else:
            # If detection is disabled, keep both panels disabled
            self.utrack_panel.setEnabled(False)
            self.thunderstorm_panel.setEnabled(False)

        # Update visual feedback (highlighting works regardless of enabled state)
        if is_utrack:
            self.utrack_panel.setStyleSheet("QWidget { background-color: #f0f8ff; }")
            self.thunderstorm_panel.setStyleSheet("")
        else:
            self.utrack_panel.setStyleSheet("")
            self.thunderstorm_panel.setStyleSheet("QWidget { background-color: #fff0f5; }")

    def on_ts_em_gain_toggled(self):
        """Enable/disable EM gain spinner based on checkbox"""
        self.ts_em_gain_spin.setEnabled(self.ts_is_em_gain_checkbox.isChecked())

    def _on_ts_multi_emitter_toggled(self, checked):
        """Enable/disable multi-emitter fitting options based on checkbox"""
        self.ts_multi_emitter_max_spin.setEnabled(checked)
        self.ts_multi_emitter_pvalue_combo.setEnabled(checked)
        self.ts_mfa_keep_intensity_checkbox.setEnabled(checked)
        self.ts_mfa_fixed_intensity_checkbox.setEnabled(checked)
        self.ts_mfa_fitting_method_combo.setEnabled(checked)
        self.ts_mfa_iterations_spin.setEnabled(checked)
        self.ts_mfa_final_refit_checkbox.setEnabled(checked)
        fixed = checked and self.ts_mfa_fixed_intensity_checkbox.isChecked()
        self.ts_mfa_intensity_min_spin.setEnabled(fixed)
        self.ts_mfa_intensity_max_spin.setEnabled(fixed)

    def _on_ts_mfa_fixed_intensity_toggled(self, checked):
        """Enable/disable MFA intensity range spinners"""
        enabled = self.ts_multi_emitter_checkbox.isChecked() and checked
        self.ts_mfa_intensity_min_spin.setEnabled(enabled)
        self.ts_mfa_intensity_max_spin.setEnabled(enabled)

    def _on_ts_filter_type_changed(self, filter_type):
        """Show/hide filter-specific parameter rows based on selected filter."""
        is_wavelet = (filter_type == 'wavelet')
        is_gauss_like = (filter_type in ('gaussian', 'lowered_gaussian'))
        is_dog = (filter_type == 'dog')
        is_diff_avg = (filter_type == 'diff_avg')
        is_median = (filter_type == 'median')
        is_box = (filter_type == 'box')

        # Wavelet params
        self.ts_filter_scale_spin.setVisible(is_wavelet)
        self.ts_filter_order_spin.setVisible(is_wavelet)
        # Find labels in the form layout and hide them too
        for spin, visible in [
            (self.ts_filter_scale_spin, is_wavelet),
            (self.ts_filter_order_spin, is_wavelet),
            (self.ts_filter_sigma_spin, is_gauss_like),
            (self.ts_filter_sigma1_spin, is_dog),
            (self.ts_filter_sigma2_spin, is_dog),
            (self.ts_filter_size1_spin, is_diff_avg or is_median or is_box),
            (self.ts_filter_size2_spin, is_diff_avg),
            (self.ts_filter_pattern_combo, is_median),
        ]:
            spin.setVisible(visible)
            # Also hide the label in the form layout
            parent_layout = spin.parentWidget().layout() if spin.parentWidget() else None
            if parent_layout and isinstance(parent_layout, QFormLayout):
                label = parent_layout.labelForField(spin)
                if label:
                    label.setVisible(visible)

    def _on_ts_psf_model_changed(self, model_text):
        """Show/hide sub-pixel localization options based on selected PSF model."""
        is_psf = model_text.startswith('PSF:')
        is_no_estimator = ('No estimator' in model_text)

        # Fitting method, initial sigma, sigma range: only for PSF models
        for widget in [self.ts_fitting_method_combo, self.ts_initial_sigma_spin,
                       self.ts_sigma_range_label, self.ts_sigma_min_spin,
                       self.ts_sigma_max_spin]:
            widget.setVisible(is_psf)
        for label_widget in [self.ts_initial_sigma_label, self.ts_sigma_min_label,
                             self.ts_sigma_max_label]:
            label_widget.setVisible(is_psf)
        # Fitting method label
        parent_layout = self.ts_fitting_method_combo.parentWidget().layout() if self.ts_fitting_method_combo.parentWidget() else None
        if parent_layout and isinstance(parent_layout, QFormLayout):
            label = parent_layout.labelForField(self.ts_fitting_method_combo)
            if label:
                label.setVisible(is_psf)

        # Fit/estimation radius: shown for all except 'No estimator'
        self.ts_fit_radius_spin.setVisible(not is_no_estimator)
        self.ts_fit_radius_label.setVisible(not is_no_estimator)

        # Update radius label text
        if is_psf:
            self.ts_fit_radius_label.setText("Fitting Radius:")
        else:
            self.ts_fit_radius_label.setText("Estimation Radius:")

    def _get_ts_fitter_type(self):
        """Derive internal fitter_type from the PSF model + fitting method combos.

        Maps the user-facing thunderSTORM-style selections to the internal
        fitter_type keys used by the fitting module.

        Returns
        -------
        fitter_type : str
            Internal fitter key (e.g. 'gaussian_lsq', 'centroid', etc.)
        psf_model : str
            Canonical PSF model key for parameter storage.
        fitting_method : str
            Canonical fitting method key for parameter storage.
        """
        model_text = self.ts_psf_model_combo.currentText()
        method_text = self.ts_fitting_method_combo.currentText()

        # Map model display text → internal key
        model_map = {
            'PSF: Integrated Gaussian': 'integrated_gaussian',
            'PSF: Gaussian': 'gaussian',
            'PSF: Elliptical Gaussian (3D astigmatism)': 'elliptical_gaussian',
            'Phasor': 'phasor',
            'Radial symmetry': 'radial_symmetry',
            'Centroid of local neighbourhood': 'centroid',
            'No estimator': 'no_estimator',
        }
        psf_model = model_map.get(model_text, 'integrated_gaussian')

        # Map fitting method display text → internal key
        method_map = {
            'Least squares': 'least_squares',
            'Weighted least squares': 'weighted_least_squares',
            'Maximum likelihood estimation': 'maximum_likelihood',
        }
        fitting_method = method_map.get(method_text, 'least_squares')

        # Derive the combined fitter_type
        if psf_model in ('phasor', 'radial_symmetry', 'centroid'):
            fitter_type = psf_model
        elif psf_model == 'no_estimator':
            fitter_type = 'centroid'  # fallback
        elif psf_model == 'elliptical_gaussian':
            fitter_type = 'elliptical_gaussian_mle'
        elif psf_model in ('integrated_gaussian', 'gaussian'):
            # Map fitting method to suffix
            method_suffix = {
                'least_squares': 'lsq',
                'weighted_least_squares': 'wlsq',
                'maximum_likelihood': 'mle',
            }
            fitter_type = 'gaussian_' + method_suffix.get(fitting_method, 'lsq')
        else:
            fitter_type = 'gaussian_lsq'

        return fitter_type, psf_model, fitting_method

    def _on_ts_detector_type_changed(self, detector_type):
        """Show/hide detector-specific parameter rows."""
        is_nms = (detector_type == 'non_maximum_suppression')
        is_centroid = (detector_type == 'centroid')

        self.ts_detector_radius_spin.setVisible(is_nms)
        self.ts_use_watershed_checkbox.setVisible(is_centroid)

        for widget, visible in [
            (self.ts_detector_radius_spin, is_nms),
            (self.ts_use_watershed_checkbox, is_centroid),
        ]:
            parent_layout = widget.parentWidget().layout() if widget.parentWidget() else None
            if parent_layout and isinstance(parent_layout, QFormLayout):
                label = parent_layout.labelForField(widget)
                if label:
                    label.setVisible(visible)

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
            # U-Track parameters (always included)
            'psf_sigma': self.detection_psf_sigma_spin.value(),
            'alpha_threshold': self.detection_alpha_spin.value(),
            'min_intensity': self.detection_min_intensity_spin.value()
        }

        # Add ThunderSTORM parameters if thunderSTORM is selected
        if hasattr(self, 'ts_filter_combo'):
            # Sigma range: 0 means "no limit" → convert to None
            sigma_min_val = self.ts_sigma_min_spin.value()
            sigma_max_val = self.ts_sigma_max_spin.value()
            # Derive combined fitter_type from PSF model + fitting method
            fitter_type, psf_model, fitting_method = self._get_ts_fitter_type()
            detector_params.update({
                # Filter
                'ts_filter_type': self.ts_filter_combo.currentText(),
                'ts_filter_scale': self.ts_filter_scale_spin.value(),
                'ts_filter_order': self.ts_filter_order_spin.value(),
                'ts_filter_sigma': self.ts_filter_sigma_spin.value(),
                'ts_filter_sigma1': self.ts_filter_sigma1_spin.value(),
                'ts_filter_sigma2': self.ts_filter_sigma2_spin.value(),
                'ts_filter_size1': self.ts_filter_size1_spin.value(),
                'ts_filter_size2': self.ts_filter_size2_spin.value(),
                'ts_filter_pattern': self.ts_filter_pattern_combo.currentText(),
                # Detector
                'ts_detector_type': self.ts_detector_combo.currentText(),
                'ts_detector_threshold': self.ts_threshold_combo.currentText(),
                'ts_detector_connectivity': self.ts_detector_connectivity_combo.currentText(),
                'ts_detector_radius': self.ts_detector_radius_spin.value(),
                # Sub-pixel localization
                'ts_psf_model': psf_model,
                'ts_fitting_method': fitting_method,
                'ts_fitter_type': fitter_type,
                'ts_fit_radius': self.ts_fit_radius_spin.value(),
                'ts_initial_sigma': self.ts_initial_sigma_spin.value(),
                'ts_sigma_min': sigma_min_val if sigma_min_val > 0 else None,
                'ts_sigma_max': sigma_max_val if sigma_max_val > 0 else None,
                # Camera
                'ts_photons_per_adu': self.ts_photons_per_adu_spin.value(),
                'ts_baseline': self.ts_baseline_spin.value(),
                'ts_is_em_gain': self.ts_is_em_gain_checkbox.isChecked(),
                'ts_em_gain': self.ts_em_gain_spin.value(),
                'ts_quantum_efficiency': self.ts_quantum_efficiency_spin.value(),
                # Detection options
                'ts_use_watershed': self.ts_use_watershed_checkbox.isChecked(),
                # Multi-emitter
                'ts_multi_emitter_enabled': self.ts_multi_emitter_checkbox.isChecked(),
                'ts_multi_emitter_max': self.ts_multi_emitter_max_spin.value(),
                'ts_multi_emitter_pvalue': float(self.ts_multi_emitter_pvalue_combo.currentText()),
                'ts_multi_emitter_keep_same_intensity': self.ts_mfa_keep_intensity_checkbox.isChecked(),
                'ts_multi_emitter_fixed_intensity': self.ts_mfa_fixed_intensity_checkbox.isChecked(),
                'ts_multi_emitter_intensity_min': self.ts_mfa_intensity_min_spin.value(),
                'ts_multi_emitter_intensity_max': self.ts_mfa_intensity_max_spin.value(),
                # MFA advanced options
                'ts_mfa_fitting_method': 'mle' if 'likelihood' in self.ts_mfa_fitting_method_combo.currentText().lower() else 'wlsq',
                'ts_mfa_model_selection_iterations': self.ts_mfa_iterations_spin.value(),
                'ts_mfa_enable_final_refit': self.ts_mfa_final_refit_checkbox.isChecked(),
            })

        # Start detection worker with detection method
        self.detection_worker = DetectionWorker(
            files_to_process,
            detector_params,
            output_dir,
            self.parameters.pixel_size,
            show_results=self.parameters.detection_show_results,
            detection_method=self.parameters.detection_method  # Pass the selected method
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
                                # FLIKA addPoint format: [frame, x, y]
                                window.addPoint([int(frame), float(x), float(y)])
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

            # Route to appropriate detection method
            if self.parameters.detection_method == 'thunderstorm' and THUNDERSTORM_AVAILABLE:
                return self._run_thunderstorm_detection_for_file(images, detection_file, file_path)

            # U-Track detection (default)
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
                # Swap x and y to undo the rot90+fliplr transpose applied
                # to images before detection, restoring original image coords.
                combined_detections['x [nm]'] = combined_detections['y'] * self.parameters.pixel_size
                combined_detections['y [nm]'] = combined_detections['x'] * self.parameters.pixel_size
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

    def _run_thunderstorm_detection_for_file(self, images, detection_file, file_path):
        """Run ThunderSTORM detection for a single file during main analysis pipeline"""
        try:
            # Build gui_params from all ts_* attributes on self.parameters
            gui_params = {'pixel_size': self.parameters.pixel_size}
            for attr in dir(self.parameters):
                if attr.startswith('ts_'):
                    gui_params[attr] = getattr(self.parameters, attr)

            detector = ThunderSTORMDetector.create_from_gui_parameters(gui_params)
            localizations = detector.detect_and_fit(images, show_progress=False)

            # The input images have been rot90+fliplr transformed (which
            # transposes rows/cols). Swap x and y to restore coordinates
            # to the original image space so that the CSV matches real
            # thunderSTORM output and is consistent with the tracking pipeline.
            if 'x' in localizations and 'y' in localizations:
                localizations['x'], localizations['y'] = (
                    localizations['y'].copy(), localizations['x'].copy()
                )
            if 'sigma_x' in localizations and 'sigma_y' in localizations:
                localizations['sigma_x'], localizations['sigma_y'] = (
                    localizations['sigma_y'].copy(), localizations['sigma_x'].copy()
                )

            n_detections = len(localizations.get('x', []))
            if n_detections > 0:
                detector.save_localizations(localizations, detection_file, image_stack=images)
                self.log_message(f"    ThunderSTORM: Saved {n_detections} detections to {os.path.basename(detection_file)}")
            else:
                self.log_message(f"    ThunderSTORM: No detections found in {os.path.basename(file_path)}")

            return True

        except Exception as e:
            self.log_message(f"    ThunderSTORM detection failed: {e}")
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

        # U-Track 2.5 Post-Tracking Analysis
        self.utrack_analysis_group = QGroupBox("U-Track 2.5 Post-Tracking Analysis")
        utrack_analysis_layout = QFormLayout(self.utrack_analysis_group)

        self.utrack_msd_checkbox = QCheckBox("MSD Analysis (D, α, motion type)")
        self.utrack_msd_checkbox.setChecked(self.parameters.utrack_enable_msd_analysis)
        utrack_analysis_layout.addRow(self.utrack_msd_checkbox)

        self.utrack_quality_checkbox = QCheckBox("Track Quality Scoring")
        self.utrack_quality_checkbox.setChecked(self.parameters.utrack_enable_quality_scoring)
        utrack_analysis_layout.addRow(self.utrack_quality_checkbox)

        self.utrack_persistence_checkbox = QCheckBox("Velocity Persistence Analysis")
        self.utrack_persistence_checkbox.setChecked(self.parameters.utrack_enable_velocity_persistence)
        utrack_analysis_layout.addRow(self.utrack_persistence_checkbox)

        self.utrack_drift_checkbox = QCheckBox("Drift Correction (before tracking)")
        self.utrack_drift_checkbox.setChecked(self.parameters.utrack_enable_drift_correction)
        utrack_analysis_layout.addRow(self.utrack_drift_checkbox)

        self.utrack_pixel_size_spin = QDoubleSpinBox()
        self.utrack_pixel_size_spin.setRange(0.01, 10.0)
        self.utrack_pixel_size_spin.setDecimals(3)
        self.utrack_pixel_size_spin.setValue(self.parameters.utrack_pixel_size_um)
        self.utrack_pixel_size_spin.setSuffix(" µm")
        utrack_analysis_layout.addRow("Pixel Size:", self.utrack_pixel_size_spin)

        self.utrack_frame_interval_spin = QDoubleSpinBox()
        self.utrack_frame_interval_spin.setRange(0.001, 10.0)
        self.utrack_frame_interval_spin.setDecimals(4)
        self.utrack_frame_interval_spin.setValue(self.parameters.utrack_frame_interval)
        self.utrack_frame_interval_spin.setSuffix(" s")
        utrack_analysis_layout.addRow("Frame Interval:", self.utrack_frame_interval_spin)

        self.utrack_intensity_costs_checkbox = QCheckBox("Use Intensity Costs in Linking")
        self.utrack_intensity_costs_checkbox.setChecked(self.parameters.utrack_use_intensity_costs)
        utrack_analysis_layout.addRow(self.utrack_intensity_costs_checkbox)

        self.utrack_velocity_costs_checkbox = QCheckBox("Use Velocity Angle Costs in Linking")
        self.utrack_velocity_costs_checkbox.setChecked(self.parameters.utrack_use_velocity_costs)
        utrack_analysis_layout.addRow(self.utrack_velocity_costs_checkbox)

        utrack_layout.addWidget(self.utrack_analysis_group)
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

    def create_background_tab(self):
        """Create the Background ROI detection configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # --- Mode Selection Group ---
        mode_group = QGroupBox("Background ROI Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.bg_mode_button_group = QButtonGroup(self)
        self.bg_mode_manual_radio = QRadioButton("Manual — use existing ROI files (default)")
        self.bg_mode_auto_radio = QRadioButton("Auto — detect background ROI automatically")
        self.bg_mode_button_group.addButton(self.bg_mode_manual_radio)
        self.bg_mode_button_group.addButton(self.bg_mode_auto_radio)
        self.bg_mode_manual_radio.setChecked(self.parameters.auto_background_detection_mode == 'manual')
        self.bg_mode_auto_radio.setChecked(self.parameters.auto_background_detection_mode == 'auto')
        self.bg_mode_manual_radio.toggled.connect(self.on_bg_mode_changed)

        mode_layout.addWidget(self.bg_mode_manual_radio)
        mode_layout.addWidget(self.bg_mode_auto_radio)
        scroll_layout.addWidget(mode_group)

        # --- Detection Parameters Group ---
        self.bg_auto_params_group = QGroupBox("Auto Detection Parameters")
        params_layout = QFormLayout(self.bg_auto_params_group)

        self.bg_roi_width_spin = QSpinBox()
        self.bg_roi_width_spin.setRange(1, 100)
        self.bg_roi_width_spin.setValue(self.parameters.auto_bg_roi_width)
        params_layout.addRow("ROI Width (pixels):", self.bg_roi_width_spin)

        self.bg_roi_height_spin = QSpinBox()
        self.bg_roi_height_spin.setRange(1, 100)
        self.bg_roi_height_spin.setValue(self.parameters.auto_bg_roi_height)
        params_layout.addRow("ROI Height (pixels):", self.bg_roi_height_spin)

        self.bg_scan_stride_spin = QSpinBox()
        self.bg_scan_stride_spin.setRange(1, 50)
        self.bg_scan_stride_spin.setValue(self.parameters.auto_bg_scan_stride)
        params_layout.addRow("Scan Stride (pixels):", self.bg_scan_stride_spin)

        self.bg_temporal_check_checkbox = QCheckBox("Enable temporal stability check")
        self.bg_temporal_check_checkbox.setChecked(self.parameters.auto_bg_enable_temporal_check)
        self.bg_temporal_check_checkbox.toggled.connect(self.on_bg_temporal_check_toggled)
        params_layout.addRow(self.bg_temporal_check_checkbox)

        self.bg_top_n_spin = QSpinBox()
        self.bg_top_n_spin.setRange(1, 100)
        self.bg_top_n_spin.setValue(self.parameters.auto_bg_top_n)
        params_layout.addRow("Top-N candidates for temporal check:", self.bg_top_n_spin)

        self.bg_auto_params_group.setEnabled(self.parameters.auto_background_detection_mode == 'auto')
        scroll_layout.addWidget(self.bg_auto_params_group)

        # --- Test Detection Group ---
        test_group = QGroupBox("Test Detection")
        test_layout = QVBoxLayout(test_group)

        file_select_layout = QHBoxLayout()
        self.bg_test_file_edit = QLineEdit()
        self.bg_test_file_edit.setPlaceholderText("Select a TIFF file to test detection...")
        self.bg_test_file_edit.setReadOnly(True)
        file_select_layout.addWidget(self.bg_test_file_edit)

        self.bg_select_file_btn = QPushButton("Browse...")
        self.bg_select_file_btn.clicked.connect(self.select_bg_test_file)
        file_select_layout.addWidget(self.bg_select_file_btn)
        test_layout.addLayout(file_select_layout)

        btn_layout = QHBoxLayout()
        self.bg_detect_single_btn = QPushButton("Detect on Selected File")
        self.bg_detect_single_btn.clicked.connect(self.run_bg_detection_single)
        btn_layout.addWidget(self.bg_detect_single_btn)

        self.bg_detect_all_btn = QPushButton("Detect on All Files")
        self.bg_detect_all_btn.clicked.connect(self.run_bg_detection_all)
        btn_layout.addWidget(self.bg_detect_all_btn)
        btn_layout.addStretch()
        test_layout.addLayout(btn_layout)

        self.bg_results_text = QTextEdit()
        self.bg_results_text.setReadOnly(True)
        self.bg_results_text.setMaximumHeight(200)
        test_layout.addWidget(self.bg_results_text)

        scroll_layout.addWidget(test_group)

        # --- Algorithm Description Group ---
        desc_group = QGroupBox("Algorithm Description")
        desc_layout = QVBoxLayout(desc_group)
        desc_text = QTextEdit()
        desc_text.setReadOnly(True)
        desc_text.setMaximumHeight(120)
        desc_text.setText(
            "The auto-detection algorithm finds the best background region in two stages:\n\n"
            "Stage 1 — Intensity scan: Computes a max-intensity projection of the image stack, "
            "then scans all ROI-sized windows at the given stride. Windows are ranked by mean "
            "intensity (lowest first).\n\n"
            "Stage 2 — Temporal stability (optional): The top-N lowest-intensity candidates are "
            "evaluated across all frames. The candidate with the lowest temporal standard deviation "
            "is selected, ensuring the background region is stable over time.\n\n"
            "The output ROI file uses the same format as manually created ROI files, so the "
            "existing background subtraction pipeline picks it up automatically."
        )
        desc_layout.addWidget(desc_text)
        scroll_layout.addWidget(desc_group)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Background")

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
            },
            'U-Track Analysis': {
                'columns': [
                    'D_um2s', 'alpha', 'motion_type', 'msd_r_squared',
                    'quality_score',
                    'persistence_ratio', 'velocity_autocorr_tau1', 'confinement_ratio',
                    'linking_method', 'motion_model_used',
                ],
                'depends_on': 'utrack_analysis_available',
                'description': 'U-Track 2.5 MSD analysis, quality scoring, and motion metrics (requires U-Track linking)'
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

        # Dynamically set U-Track analysis availability based on linking method
        self.parameters.utrack_analysis_available = (
            getattr(self.parameters, 'linking_method', '') == 'utrack')

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

    def on_bg_mode_changed(self):
        """Enable/disable auto params group based on mode selection"""
        is_auto = self.bg_mode_auto_radio.isChecked()
        self.bg_auto_params_group.setEnabled(is_auto)

    def on_bg_temporal_check_toggled(self):
        """Enable/disable top-N spinner based on temporal check toggle"""
        self.bg_top_n_spin.setEnabled(self.bg_temporal_check_checkbox.isChecked())

    def select_bg_test_file(self):
        """Open file dialog to select a single TIFF for background detection test"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select TIFF File", "", "TIFF Files (*.tif *.tiff)")
        if file_path:
            self.bg_test_file_edit.setText(file_path)

    def run_bg_detection_single(self):
        """Run auto background detection on the selected test file"""
        file_path = self.bg_test_file_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.bg_results_text.setText("Please select a valid TIFF file first.")
            return

        self.bg_results_text.setText("Running detection...")
        QApplication.processEvents()

        try:
            result = AutoBackgroundDetector.detect_background_roi(
                file_path,
                roi_width=self.bg_roi_width_spin.value(),
                roi_height=self.bg_roi_height_spin.value(),
                stride=self.bg_scan_stride_spin.value(),
                top_n=self.bg_top_n_spin.value(),
                enable_temporal_check=self.bg_temporal_check_checkbox.isChecked(),
            )
            if result is None:
                self.bg_results_text.setText("Detection failed — no valid candidates found.")
                return

            lines = [
                f"Detection complete for: {os.path.basename(file_path)}",
                f"Image shape: {result['image_shape']}",
                f"Candidates scanned: {result['candidates_scanned']}",
                f"Best ROI position: y={result['y']}, x={result['x']}",
                f"ROI size: {result['width']} x {result['height']}",
                f"Mean intensity (max proj): {result['mean_intensity']:.2f}",
            ]
            if result['temporal_std'] is not None:
                lines.append(f"Temporal std: {result['temporal_std']:.4f}")
            lines.append(f"ROI file saved: {result['roi_file']}")
            self.bg_results_text.setText('\n'.join(lines))

        except Exception as e:
            self.bg_results_text.setText(f"Error during detection:\n{str(e)}")

    def run_bg_detection_all(self):
        """Run auto background detection on all files from the Files tab"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            self.bg_results_text.setText("No files loaded. Add files in the Files tab first.")
            return

        self.bg_results_text.setText(f"Running detection on {len(self.file_paths)} files...")
        QApplication.processEvents()

        results_lines = []
        for i, file_path in enumerate(self.file_paths):
            try:
                result = AutoBackgroundDetector.detect_background_roi(
                    file_path,
                    roi_width=self.bg_roi_width_spin.value(),
                    roi_height=self.bg_roi_height_spin.value(),
                    stride=self.bg_scan_stride_spin.value(),
                    top_n=self.bg_top_n_spin.value(),
                    enable_temporal_check=self.bg_temporal_check_checkbox.isChecked(),
                )
                if result:
                    results_lines.append(
                        f"[{i+1}/{len(self.file_paths)}] {os.path.basename(file_path)}: "
                        f"y={result['y']}, x={result['x']}, "
                        f"mean={result['mean_intensity']:.2f}"
                    )
                else:
                    results_lines.append(
                        f"[{i+1}/{len(self.file_paths)}] {os.path.basename(file_path)}: FAILED"
                    )
            except Exception as e:
                results_lines.append(
                    f"[{i+1}/{len(self.file_paths)}] {os.path.basename(file_path)}: ERROR - {e}"
                )
            QApplication.processEvents()

        results_lines.insert(0, f"Detection complete for {len(self.file_paths)} files:\n")
        self.bg_results_text.setText('\n'.join(results_lines))

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

        # Auto background detection parameters
        self.parameters.auto_background_detection_mode = 'auto' if self.bg_mode_auto_radio.isChecked() else 'manual'
        self.parameters.auto_bg_roi_width = self.bg_roi_width_spin.value()
        self.parameters.auto_bg_roi_height = self.bg_roi_height_spin.value()
        self.parameters.auto_bg_scan_stride = self.bg_scan_stride_spin.value()
        self.parameters.auto_bg_top_n = self.bg_top_n_spin.value()
        self.parameters.auto_bg_enable_temporal_check = self.bg_temporal_check_checkbox.isChecked()

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

        # Determine which detection method is selected
        if hasattr(self, 'utrack_method_radio'):
            self.parameters.detection_method = 'utrack' if self.utrack_method_radio.isChecked() else 'thunderstorm'

        # ThunderSTORM parameters
        if hasattr(self, 'ts_filter_combo'):
            self.parameters.ts_filter_type = self.ts_filter_combo.currentText()
            self.parameters.ts_filter_scale = self.ts_filter_scale_spin.value()
            self.parameters.ts_filter_order = self.ts_filter_order_spin.value()
            self.parameters.ts_filter_sigma = self.ts_filter_sigma_spin.value()
            self.parameters.ts_filter_sigma1 = self.ts_filter_sigma1_spin.value()
            self.parameters.ts_filter_sigma2 = self.ts_filter_sigma2_spin.value()
            self.parameters.ts_filter_size1 = self.ts_filter_size1_spin.value()
            self.parameters.ts_filter_size2 = self.ts_filter_size2_spin.value()
            self.parameters.ts_filter_pattern = self.ts_filter_pattern_combo.currentText()
            self.parameters.ts_detector_type = self.ts_detector_combo.currentText()
            self.parameters.ts_detector_threshold = self.ts_threshold_combo.currentText()
            self.parameters.ts_detector_connectivity = self.ts_detector_connectivity_combo.currentText()
            self.parameters.ts_detector_radius = self.ts_detector_radius_spin.value()
            fitter_type, psf_model, fitting_method = self._get_ts_fitter_type()
            self.parameters.ts_fitter_type = fitter_type
            self.parameters.ts_psf_model = psf_model
            self.parameters.ts_fitting_method = fitting_method
            self.parameters.ts_fit_radius = self.ts_fit_radius_spin.value()
            self.parameters.ts_initial_sigma = self.ts_initial_sigma_spin.value()
            sigma_min_val = self.ts_sigma_min_spin.value()
            sigma_max_val = self.ts_sigma_max_spin.value()
            self.parameters.ts_sigma_min = sigma_min_val if sigma_min_val > 0 else None
            self.parameters.ts_sigma_max = sigma_max_val if sigma_max_val > 0 else None

        # ThunderSTORM advanced options
        if hasattr(self, 'ts_use_watershed_checkbox'):
            self.parameters.ts_use_watershed = self.ts_use_watershed_checkbox.isChecked()
            self.parameters.ts_multi_emitter_enabled = self.ts_multi_emitter_checkbox.isChecked()
            self.parameters.ts_multi_emitter_max = self.ts_multi_emitter_max_spin.value()
            try:
                self.parameters.ts_multi_emitter_pvalue = float(self.ts_multi_emitter_pvalue_combo.currentText())
            except ValueError:
                self.parameters.ts_multi_emitter_pvalue = 1e-6
            self.parameters.ts_multi_emitter_keep_same_intensity = self.ts_mfa_keep_intensity_checkbox.isChecked()
            self.parameters.ts_multi_emitter_fixed_intensity = self.ts_mfa_fixed_intensity_checkbox.isChecked()
            self.parameters.ts_multi_emitter_intensity_min = self.ts_mfa_intensity_min_spin.value()
            self.parameters.ts_multi_emitter_intensity_max = self.ts_mfa_intensity_max_spin.value()
            # MFA advanced options
            mfa_method_text = self.ts_mfa_fitting_method_combo.currentText()
            self.parameters.ts_mfa_fitting_method = 'mle' if 'likelihood' in mfa_method_text.lower() else 'wlsq'
            self.parameters.ts_mfa_model_selection_iterations = self.ts_mfa_iterations_spin.value()
            self.parameters.ts_mfa_enable_final_refit = self.ts_mfa_final_refit_checkbox.isChecked()

        # ThunderSTORM camera parameters
        if hasattr(self, 'ts_photons_per_adu_spin'):
            self.parameters.ts_photons_per_adu = self.ts_photons_per_adu_spin.value()
            self.parameters.ts_baseline = self.ts_baseline_spin.value()
            self.parameters.ts_is_em_gain = self.ts_is_em_gain_checkbox.isChecked()
            self.parameters.ts_em_gain = self.ts_em_gain_spin.value()
            self.parameters.ts_quantum_efficiency = self.ts_quantum_efficiency_spin.value()

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

        # Switch to progress tab (find by name to avoid hardcoded index issues)
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == "Progress":
                self.tab_widget.setCurrentIndex(i)
                break

        # Disable start button
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        self.log_message("🚀 Starting integrated SPT batch analysis pipeline...")

        # Log what will be processed
        self.log_message(f"📁 Processing {len(self.file_paths)} files")

        if self.parameters.enable_detection:
            detection_method_name = "ThunderSTORM" if self.parameters.detection_method == 'thunderstorm' else "U-Track"
            self.log_message(f"🔍 Detection enabled - will run {detection_method_name} detection first")

        if self.parameters.enable_autocorrelation_analysis:
            self.log_message("📊 Autocorrelation analysis enabled - will analyze tracks after main pipeline")

        if self.parameters.auto_detect_experiment_names:
            self.log_message("🏷️ Using auto-detection: experiment names will be extracted from subfolder names")
        else:
            self.log_message(f"🏷️ Using manual experiment name: '{self.parameters.experiment_name}'")

        try:
            total_files = len(self.file_paths)
            successful_files = []

            # Phase 0: Auto background ROI detection (if enabled)
            if (self.parameters.enable_background_subtraction and
                    self.parameters.auto_background_detection_mode == 'auto'):
                self.log_message("\n" + "="*60)
                self.log_message("PHASE 0: AUTO BACKGROUND ROI DETECTION")
                self.log_message("="*60)
                for i, file_path in enumerate(self.file_paths):
                    file_name = os.path.basename(file_path)
                    self.status_label.setText(f"Phase 0: Detecting background ROI for {file_name}")
                    self.log_message(f"  [{i+1}/{total_files}] {file_name}...")
                    QApplication.processEvents()
                    try:
                        result = AutoBackgroundDetector.detect_background_roi(
                            file_path,
                            roi_width=self.parameters.auto_bg_roi_width,
                            roi_height=self.parameters.auto_bg_roi_height,
                            stride=self.parameters.auto_bg_scan_stride,
                            top_n=self.parameters.auto_bg_top_n,
                            enable_temporal_check=self.parameters.auto_bg_enable_temporal_check,
                        )
                        if result:
                            self.log_message(
                                f"    ROI at y={result['y']}, x={result['x']}, "
                                f"mean={result['mean_intensity']:.2f} -> {os.path.basename(result['roi_file'])}")
                        else:
                            self.log_message(f"    WARNING: No background ROI found")
                    except Exception as e:
                        self.log_message(f"    ERROR: {e}")
                self.log_message("Phase 0 complete.\n")

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
                    detection_method_name = "ThunderSTORM" if self.parameters.detection_method == 'thunderstorm' else "U-Track"
                    self.log_message(f"  🔍 Running {detection_method_name} particle detection...")
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
                # Fix background subtraction for newly interpolated points
                if self.parameters.enable_background_subtraction:
                    tracks_df = self._fix_interpolated_background(tracks_df)

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

            # U-Track 2.5 post-tracking analysis
            if self.parameters.linking_method == 'utrack' and (
               self.parameters.utrack_enable_msd_analysis or
               self.parameters.utrack_enable_quality_scoring or
               self.parameters.utrack_enable_velocity_persistence):
                self.log_message("  🔬 Running U-Track 2.5 post-tracking analysis...")
                utrack_linker = UTrackLinkerWithMixedMotion()

                if self.parameters.utrack_enable_msd_analysis:
                    self.log_message("    📊 Computing MSD analysis...")
                    tracks_df = utrack_linker.add_msd_analysis(
                        tracks_df,
                        pixel_size=self.parameters.utrack_pixel_size_um,
                        frame_interval=self.parameters.utrack_frame_interval)

                if self.parameters.utrack_enable_quality_scoring:
                    self.log_message("    ⭐ Computing track quality scores...")
                    tracks_df = utrack_linker.compute_track_quality(tracks_df)

                if self.parameters.utrack_enable_velocity_persistence:
                    self.log_message("    🔄 Computing velocity persistence metrics...")
                    tracks_df = utrack_linker.add_velocity_persistence(
                        tracks_df,
                        pixel_size=self.parameters.utrack_pixel_size_um)

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
                # Fix background subtraction for newly interpolated points
                if self.parameters.enable_background_subtraction:
                    tracks_df_interpolated = self._fix_interpolated_background(tracks_df_interpolated)

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
                    # Fix background subtraction for newly interpolated points
                    if self.parameters.enable_background_subtraction:
                        tracks_df = self._fix_interpolated_background(tracks_df)
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

                    if self.parameters.linking_method == 'utrack':
                        core_columns.extend(['linking_method', 'motion_model_used'])
                        if getattr(self.parameters, 'utrack_enable_msd_analysis', False):
                            core_columns.extend(['D_um2s', 'alpha', 'motion_type', 'msd_r_squared'])
                        if getattr(self.parameters, 'utrack_enable_quality_scoring', False):
                            core_columns.append('quality_score')
                        if getattr(self.parameters, 'utrack_enable_velocity_persistence', False):
                            core_columns.extend(['persistence_ratio', 'velocity_autocorr_tau1', 'confinement_ratio'])

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
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # Search for localization file in multiple locations:
            # 1. Custom detection output directory (if set)
            # 2. Same directory as the image file
            search_dirs = []
            if (hasattr(self, 'parameters') and
                    self.parameters.detection_output_directory):
                search_dirs.append(self.parameters.detection_output_directory)
            search_dirs.append(os.path.dirname(file_path))

            locs_file = None
            for search_dir in search_dirs:
                candidate = os.path.join(search_dir, f"{base_name}_locsID.csv")
                if os.path.exists(candidate):
                    locs_file = candidate
                    break
                candidate = os.path.join(search_dir, f"{base_name}_locs.csv")
                if os.path.exists(candidate):
                    locs_file = candidate
                    break

            if locs_file is None:
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

        # Auto background detection parameters
        self.bg_mode_manual_radio.setChecked(self.parameters.auto_background_detection_mode == 'manual')
        self.bg_mode_auto_radio.setChecked(self.parameters.auto_background_detection_mode == 'auto')
        self.bg_roi_width_spin.setValue(self.parameters.auto_bg_roi_width)
        self.bg_roi_height_spin.setValue(self.parameters.auto_bg_roi_height)
        self.bg_scan_stride_spin.setValue(self.parameters.auto_bg_scan_stride)
        self.bg_temporal_check_checkbox.setChecked(self.parameters.auto_bg_enable_temporal_check)
        self.bg_top_n_spin.setValue(self.parameters.auto_bg_top_n)
        self.on_bg_mode_changed()
        self.on_bg_temporal_check_toggled()

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

        # ThunderSTORM parameters
        if hasattr(self, 'ts_filter_combo'):
            # Filter
            self.ts_filter_combo.setCurrentText(self.parameters.ts_filter_type)
            self.ts_filter_scale_spin.setValue(self.parameters.ts_filter_scale)
            self.ts_filter_order_spin.setValue(self.parameters.ts_filter_order)
            self.ts_filter_sigma_spin.setValue(self.parameters.ts_filter_sigma)
            self.ts_filter_sigma1_spin.setValue(self.parameters.ts_filter_sigma1)
            self.ts_filter_sigma2_spin.setValue(self.parameters.ts_filter_sigma2)
            self.ts_filter_size1_spin.setValue(self.parameters.ts_filter_size1)
            self.ts_filter_size2_spin.setValue(self.parameters.ts_filter_size2)
            self.ts_filter_pattern_combo.setCurrentText(self.parameters.ts_filter_pattern)
            # Detector
            self.ts_detector_combo.setCurrentText(self.parameters.ts_detector_type)
            self.ts_threshold_combo.setCurrentText(self.parameters.ts_detector_threshold)
            self.ts_detector_connectivity_combo.setCurrentText(self.parameters.ts_detector_connectivity)
            self.ts_detector_radius_spin.setValue(self.parameters.ts_detector_radius)
            self.ts_use_watershed_checkbox.setChecked(self.parameters.ts_use_watershed)
            # Sub-pixel localization
            psf_display_map = {
                'integrated_gaussian': 'PSF: Integrated Gaussian',
                'gaussian': 'PSF: Gaussian',
                'elliptical_gaussian': 'PSF: Elliptical Gaussian (3D astigmatism)',
                'phasor': 'Phasor',
                'radial_symmetry': 'Radial symmetry',
                'centroid': 'Centroid of local neighbourhood',
                'no_estimator': 'No estimator',
            }
            fitting_method_display = {
                'least_squares': 'Least squares',
                'weighted_least_squares': 'Weighted least squares',
                'maximum_likelihood': 'Maximum likelihood estimation',
            }
            self.ts_psf_model_combo.setCurrentText(
                psf_display_map.get(self.parameters.ts_psf_model, 'PSF: Integrated Gaussian'))
            self.ts_fitting_method_combo.setCurrentText(
                fitting_method_display.get(self.parameters.ts_fitting_method, 'Least squares'))
            self.ts_fit_radius_spin.setValue(self.parameters.ts_fit_radius)
            self.ts_initial_sigma_spin.setValue(self.parameters.ts_initial_sigma)
            self.ts_sigma_min_spin.setValue(self.parameters.ts_sigma_min if self.parameters.ts_sigma_min is not None else 0.0)
            self.ts_sigma_max_spin.setValue(self.parameters.ts_sigma_max if self.parameters.ts_sigma_max is not None else 0.0)
            # Multi-emitter
            self.ts_multi_emitter_checkbox.setChecked(self.parameters.ts_multi_emitter_enabled)
            self.ts_multi_emitter_max_spin.setValue(self.parameters.ts_multi_emitter_max)
            self.ts_multi_emitter_pvalue_combo.setCurrentText(str(self.parameters.ts_multi_emitter_pvalue))
            self.ts_mfa_keep_intensity_checkbox.setChecked(self.parameters.ts_multi_emitter_keep_same_intensity)
            self.ts_mfa_fixed_intensity_checkbox.setChecked(self.parameters.ts_multi_emitter_fixed_intensity)
            self.ts_mfa_intensity_min_spin.setValue(self.parameters.ts_multi_emitter_intensity_min)
            self.ts_mfa_intensity_max_spin.setValue(self.parameters.ts_multi_emitter_intensity_max)
            # Camera
            self.ts_photons_per_adu_spin.setValue(self.parameters.ts_photons_per_adu)
            self.ts_baseline_spin.setValue(self.parameters.ts_baseline)
            self.ts_is_em_gain_checkbox.setChecked(self.parameters.ts_is_em_gain)
            self.ts_em_gain_spin.setValue(self.parameters.ts_em_gain)
            self.ts_quantum_efficiency_spin.setValue(self.parameters.ts_quantum_efficiency)
            # Refresh visibility states
            self.on_ts_em_gain_toggled()
            self._on_ts_filter_type_changed(self.parameters.ts_filter_type)
            self._on_ts_detector_type_changed(self.parameters.ts_detector_type)
            self._on_ts_multi_emitter_toggled(self.parameters.ts_multi_emitter_enabled)
            self._on_ts_psf_model_changed(self.ts_psf_model_combo.currentText())

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

                # Store frame-specific ROI intensity array for post-interpolation fix-up
                if frame_specific and isinstance(roi_intensity, np.ndarray):
                    self._roi_intensity_array = roi_intensity
                    self._camera_black = camera_black
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

    def _fix_interpolated_background(self, tracks_df):
        """Apply frame-specific background subtraction to interpolated points.

        After interpolation adds new points, their background columns are either
        NaN or copied from the first point of the track. This method corrects them
        using the stored per-frame ROI intensity array.
        """
        frame_specific = getattr(self.parameters, 'roi_frame_specific_background', False)
        roi_intensity_array = getattr(self, '_roi_intensity_array', None)
        camera_black = getattr(self, '_camera_black', 0.0)

        if not frame_specific or roi_intensity_array is None:
            return tracks_df

        if 'is_interpolated' not in tracks_df.columns:
            return tracks_df

        interpolated_mask = tracks_df['is_interpolated'] == 1
        if not interpolated_mask.any():
            return tracks_df

        n_roi_frames = len(roi_intensity_array)
        mean_roi = roi_intensity_array.mean()

        # Look up the frame-specific ROI intensity for each interpolated point
        frames = tracks_df.loc[interpolated_mask, 'frame'].astype(int)
        bg_values = frames.apply(
            lambda f: roi_intensity_array[f] if f < n_roi_frames else mean_roi)

        tracks_df.loc[interpolated_mask, 'roi_intensity'] = bg_values.values
        tracks_df.loc[interpolated_mask, 'background_signal_used'] = bg_values.values
        tracks_df.loc[interpolated_mask, 'intensity_bg_subtracted'] = (
            tracks_df.loc[interpolated_mask, 'intensity'].values - bg_values.values)
        tracks_df.loc[interpolated_mask, 'background_subtracted'] = True
        tracks_df.loc[interpolated_mask, 'background_method'] = 'frame_specific'
        tracks_df.loc[interpolated_mask, 'camera_black_estimate'] = camera_black

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

            # U-Track 2.5 post-tracking analysis
            self.parameters.utrack_enable_msd_analysis = self.utrack_msd_checkbox.isChecked()
            self.parameters.utrack_enable_quality_scoring = self.utrack_quality_checkbox.isChecked()
            self.parameters.utrack_enable_velocity_persistence = self.utrack_persistence_checkbox.isChecked()
            self.parameters.utrack_enable_drift_correction = self.utrack_drift_checkbox.isChecked()
            self.parameters.utrack_pixel_size_um = self.utrack_pixel_size_spin.value()
            self.parameters.utrack_frame_interval = self.utrack_frame_interval_spin.value()
            self.parameters.utrack_use_intensity_costs = self.utrack_intensity_costs_checkbox.isChecked()
            self.parameters.utrack_use_velocity_costs = self.utrack_velocity_costs_checkbox.isChecked()

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
            'Lowered Gaussian filter',
            'Median filter',
            'Averaging filter (Box)',
            'Difference of averaging filters',
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

        self.ts_export_id = QCheckBox("id")
        self.ts_export_id.setChecked(True)
        export_cols_layout.addWidget(self.ts_export_id, 3, 1)

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
        if self.ts_export_id.isChecked():
            export_cols.append("id=true")

        export_string = " ".join(export_cols)

        # Build Run analysis command
        run_analysis_parts = []

        # Filter
        filter_type = self.ts_filter_type.currentText()
        run_analysis_parts.append(f'filter=[{filter_type}]')

        if 'Wavelet' in filter_type:
            run_analysis_parts.append(f'scale={self.ts_wavelet_scale.value()}')
            run_analysis_parts.append(f'order={self.ts_wavelet_order.value()}')
        elif filter_type in ('Gaussian filter', 'Lowered Gaussian filter'):
            run_analysis_parts.append(f'sigma={self.ts_sigma.value()}')
        elif filter_type == 'Difference of Gaussians':
            run_analysis_parts.append(f'sigma1={self.ts_sigma.value() * 0.625}')
            run_analysis_parts.append(f'sigma2={self.ts_sigma.value()}')
        elif filter_type == 'Median filter':
            run_analysis_parts.append(f'size={int(self.ts_wavelet_scale.value() * 2 + 1)}')
        elif filter_type in ('Averaging filter (Box)', 'Difference of averaging filters'):
            run_analysis_parts.append(f'size={int(self.ts_wavelet_scale.value() * 2 + 1)}')

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

# ==============================================================================
# THUNDERSTORM VALIDATION GUI - Standalone window for validation & comparison
# ==============================================================================

class ThunderSTORMValidation(QWidget):
    """Standalone GUI for ThunderSTORM validation, comparison testing, and ground truth analysis.

    Provides separate tabs for:
    1. Simulation - Generate synthetic SMLM datasets with known ground truth
    2. ImageJ Macros - Generate macros for running ImageJ ThunderSTORM on test data
    3. Real Data Comparison - Compare FLIKA vs ImageJ ThunderSTORM on real data
    4. Ground Truth Testing - Compare FLIKA detections against synthetic ground truth
    5. Full Validation - One-click comprehensive validation suite
    """

    def __init__(self):
        super().__init__()
        self.validation_worker = None
        self._last_output_dir = None
        self._last_report_path = None
        self._last_figures_dir = None
        self._last_results = None
        self._plugin_dir = str(Path(__file__).parent)
        self._setup_paths()
        self._setup_ui()

    def _setup_paths(self):
        """Ensure test module paths are importable."""
        for subdir in ['tests/comparison', 'tests/synthetic']:
            p = os.path.join(self._plugin_dir, subdir)
            if p not in sys.path:
                sys.path.insert(0, p)
        if self._plugin_dir not in sys.path:
            sys.path.insert(0, self._plugin_dir)

    def _setup_ui(self):
        self.setWindowTitle("ThunderSTORM Validation & Comparison Testing")
        self.setMinimumSize(1000, 750)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self._create_simulation_tab()
        self._create_imagej_macros_tab()
        self._create_real_comparison_tab()
        self._create_ground_truth_tab()
        self._create_full_validation_tab()

        # Shared progress and log area at the bottom
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(5, 0, 5, 5)

        # Progress
        prog_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        prog_layout.addWidget(self.progress_bar)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setFixedWidth(80)
        self.cancel_btn.clicked.connect(self._cancel)
        prog_layout.addWidget(self.cancel_btn)
        bottom_layout.addLayout(prog_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; padding: 2px;")
        self.status_label.setWordWrap(True)
        bottom_layout.addWidget(self.status_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        self.log_text.setPlaceholderText("Output log...")
        bottom_layout.addWidget(self.log_text)

        # Result buttons
        result_layout = QHBoxLayout()
        self.view_report_btn = QPushButton("View Report")
        self.view_report_btn.setEnabled(False)
        self.view_report_btn.clicked.connect(self._view_report)
        result_layout.addWidget(self.view_report_btn)

        self.open_output_btn = QPushButton("Open Output Directory")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.clicked.connect(self._open_output_dir)
        result_layout.addWidget(self.open_output_btn)

        self.view_figures_btn = QPushButton("View Figures")
        self.view_figures_btn.setEnabled(False)
        self.view_figures_btn.clicked.connect(self._view_figures)
        result_layout.addWidget(self.view_figures_btn)
        bottom_layout.addLayout(result_layout)

        layout.addWidget(bottom)

    # ========================================================================
    # TAB 1: Simulation - Generate synthetic SMLM data
    # ========================================================================

    def _create_scrollable_tab(self, tab_name):
        """Create a tab with a scrollable content area. Returns (content_layout, tab_widget)."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_widget = QWidget()
        content_layout = QVBoxLayout(scroll_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        scroll.setWidget(scroll_widget)
        tab_layout.addWidget(scroll)

        self.tab_widget.addTab(tab, tab_name)
        return content_layout

    def _create_simulation_tab(self):
        layout = self._create_scrollable_tab("Simulation")

        # Instructions
        info = QLabel(
            "<b>Generate Synthetic SMLM Data</b><br><br>"
            "This tab creates simulated single-molecule localization microscopy (SMLM) image stacks "
            "with known ground truth molecule positions. The synthetic data uses realistic imaging "
            "models including:<br>"
            "<ul>"
            "<li><b>Integrated Gaussian PSF</b> with erf-based pixel integration (matching ThunderSTORM)</li>"
            "<li><b>Realistic camera models</b>: EMCCD (with EM gain) and sCMOS</li>"
            "<li><b>Poisson shot noise</b>, readout noise, and background fluorescence</li>"
            "<li><b>Stochastic blinking</b>: on/off/bleaching dynamics per molecule</li>"
            "</ul>"
            "<b>Available datasets</b> vary in molecule density (sparse/medium/dense), "
            "signal-to-noise ratio (low/high SNR), pixel size (60x/100x/150x objectives), "
            "and camera type (EMCCD/sCMOS).<br><br>"
            "<b>Output:</b> For each dataset, a TIFF image stack, a ground truth CSV "
            "(frame, x_nm, y_nm, intensity, molecule_id), and metadata JSON are saved.<br><br>"
            "<i>After generating data, proceed to the 'ImageJ Macros' tab to create macros for "
            "ImageJ comparison, or to 'Ground Truth' tab to run FLIKA and compare against ground truth.</i>"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        # Settings
        settings = QGroupBox("Settings")
        settings_layout = QGridLayout(settings)

        settings_layout.addWidget(QLabel("Output Directory:"), 0, 0)
        self.sim_output_label = QLabel(str(Path(self._plugin_dir) / "test_data" / "synthetic"))
        self.sim_output_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        settings_layout.addWidget(self.sim_output_label, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.sim_output_label, "Select Synthetic Data Output Directory"))
        settings_layout.addWidget(btn, 0, 2)

        settings_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        self.sim_seed_spin = QSpinBox()
        self.sim_seed_spin.setRange(0, 999999)
        self.sim_seed_spin.setValue(42)
        settings_layout.addWidget(self.sim_seed_spin, 1, 1)

        layout.addWidget(settings)

        # Dataset selection
        ds_group = QGroupBox("Select Datasets to Generate")
        ds_layout = QVBoxLayout(ds_group)
        self.sim_dataset_list = QListWidget()
        self.sim_dataset_list.setSelectionMode(QListWidget.MultiSelection)
        self._populate_dataset_list(self.sim_dataset_list)
        ds_layout.addWidget(self.sim_dataset_list)

        sel_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._select_all(self.sim_dataset_list))
        sel_layout.addWidget(btn_all)
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._select_none(self.sim_dataset_list))
        sel_layout.addWidget(btn_none)
        sel_layout.addStretch()
        ds_layout.addLayout(sel_layout)
        layout.addWidget(ds_group)

        # ---- Custom dataset ----
        custom_group = QGroupBox("Custom Dataset (Optional)")
        custom_group.setCheckable(True)
        custom_group.setChecked(False)
        custom_layout = QGridLayout(custom_group)

        custom_layout.addWidget(QLabel(
            "<i>Define a custom synthetic dataset with your own imaging parameters. "
            "When enabled, this will be generated alongside any selected presets above.</i>"
        ), 0, 0, 1, 6)

        row = 1
        custom_layout.addWidget(QLabel("Dataset Name:"), row, 0)
        self.sim_custom_name = QLineEdit("custom_dataset")
        self.sim_custom_name.setPlaceholderText("e.g. my_custom_test")
        custom_layout.addWidget(self.sim_custom_name, row, 1, 1, 2)

        # -- Image parameters --
        row = 2
        custom_layout.addWidget(QLabel("<b>Image</b>"), row, 0)
        custom_layout.addWidget(QLabel("Width (px):"), row, 1)
        self.sim_custom_width = QSpinBox()
        self.sim_custom_width.setRange(32, 2048)
        self.sim_custom_width.setValue(128)
        custom_layout.addWidget(self.sim_custom_width, row, 2)
        custom_layout.addWidget(QLabel("Height (px):"), row, 3)
        self.sim_custom_height = QSpinBox()
        self.sim_custom_height.setRange(32, 2048)
        self.sim_custom_height.setValue(128)
        custom_layout.addWidget(self.sim_custom_height, row, 4)

        row = 3
        custom_layout.addWidget(QLabel("Frames:"), row, 1)
        self.sim_custom_frames = QSpinBox()
        self.sim_custom_frames.setRange(1, 100000)
        self.sim_custom_frames.setValue(100)
        custom_layout.addWidget(self.sim_custom_frames, row, 2)

        # -- Molecule parameters --
        row = 4
        custom_layout.addWidget(QLabel("<b>Molecules</b>"), row, 0)
        custom_layout.addWidget(QLabel("Total molecules:"), row, 1)
        self.sim_custom_n_molecules = QSpinBox()
        self.sim_custom_n_molecules.setRange(1, 1000000)
        self.sim_custom_n_molecules.setValue(1000)
        custom_layout.addWidget(self.sim_custom_n_molecules, row, 2)
        custom_layout.addWidget(QLabel("Photons/molecule:"), row, 3)
        self.sim_custom_photons = QSpinBox()
        self.sim_custom_photons.setRange(10, 1000000)
        self.sim_custom_photons.setValue(1500)
        custom_layout.addWidget(self.sim_custom_photons, row, 4)

        row = 5
        custom_layout.addWidget(QLabel("Background (photons/px):"), row, 1)
        self.sim_custom_background = QSpinBox()
        self.sim_custom_background.setRange(0, 10000)
        self.sim_custom_background.setValue(20)
        custom_layout.addWidget(self.sim_custom_background, row, 2)

        # -- Optics --
        row = 6
        custom_layout.addWidget(QLabel("<b>Optics</b>"), row, 0)
        custom_layout.addWidget(QLabel("Preset:"), row, 1)
        self.sim_custom_optics = QComboBox()
        self.sim_custom_optics.addItem("Custom...", "custom")
        self.sim_custom_optics.addItem("60x oil (NA 1.4) — 267 nm/px", "60x")
        self.sim_custom_optics.addItem("100x oil (NA 1.49) — 160 nm/px", "100x")
        self.sim_custom_optics.addItem("100x bin2 — 320 nm/px", "100x_bin2")
        self.sim_custom_optics.addItem("150x TIRF (NA 1.49) — 107 nm/px", "150x")
        self.sim_custom_optics.addItem("Match test data — 108 nm/px", "108nm")
        self.sim_custom_optics.setCurrentIndex(5)  # default to 108nm
        self.sim_custom_optics.currentIndexChanged.connect(self._on_optics_preset_changed)
        custom_layout.addWidget(self.sim_custom_optics, row, 2, 1, 3)

        row = 7
        custom_layout.addWidget(QLabel("Pixel size (nm):"), row, 1)
        self.sim_custom_pixel_size = QDoubleSpinBox()
        self.sim_custom_pixel_size.setRange(1.0, 10000.0)
        self.sim_custom_pixel_size.setDecimals(1)
        self.sim_custom_pixel_size.setValue(108.0)
        self.sim_custom_pixel_size.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_pixel_size, row, 2)
        custom_layout.addWidget(QLabel("PSF sigma (nm):"), row, 3)
        self.sim_custom_psf_sigma = QDoubleSpinBox()
        self.sim_custom_psf_sigma.setRange(1.0, 10000.0)
        self.sim_custom_psf_sigma.setDecimals(1)
        self.sim_custom_psf_sigma.setValue(173.0)
        self.sim_custom_psf_sigma.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_psf_sigma, row, 4)

        # -- Camera --
        row = 8
        custom_layout.addWidget(QLabel("<b>Camera</b>"), row, 0)
        custom_layout.addWidget(QLabel("Preset:"), row, 1)
        self.sim_custom_camera = QComboBox()
        self.sim_custom_camera.addItem("Custom...", "custom")
        self.sim_custom_camera.addItem("EMCCD (Andor iXon)", "emccd")
        self.sim_custom_camera.addItem("sCMOS (Hamamatsu Orca)", "scmos")
        self.sim_custom_camera.addItem("Match test data", "match_test")
        self.sim_custom_camera.setCurrentIndex(3)  # default to match_test
        self.sim_custom_camera.currentIndexChanged.connect(self._on_camera_preset_changed)
        custom_layout.addWidget(self.sim_custom_camera, row, 2, 1, 3)

        row = 9
        custom_layout.addWidget(QLabel("Baseline (ADU):"), row, 1)
        self.sim_custom_baseline = QDoubleSpinBox()
        self.sim_custom_baseline.setRange(0, 65535)
        self.sim_custom_baseline.setValue(100)
        self.sim_custom_baseline.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_baseline, row, 2)
        custom_layout.addWidget(QLabel("Photons/ADU:"), row, 3)
        self.sim_custom_photons_per_adu = QDoubleSpinBox()
        self.sim_custom_photons_per_adu.setRange(0.01, 1000.0)
        self.sim_custom_photons_per_adu.setDecimals(2)
        self.sim_custom_photons_per_adu.setValue(3.6)
        self.sim_custom_photons_per_adu.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_photons_per_adu, row, 4)

        row = 10
        custom_layout.addWidget(QLabel("Readout noise (e⁻):"), row, 1)
        self.sim_custom_readout = QDoubleSpinBox()
        self.sim_custom_readout.setRange(0, 1000)
        self.sim_custom_readout.setDecimals(1)
        self.sim_custom_readout.setValue(1.0)
        self.sim_custom_readout.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_readout, row, 2)
        custom_layout.addWidget(QLabel("EM gain:"), row, 3)
        self.sim_custom_em_gain = QDoubleSpinBox()
        self.sim_custom_em_gain.setRange(1, 10000)
        self.sim_custom_em_gain.setValue(100)
        self.sim_custom_em_gain.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_em_gain, row, 4)

        row = 11
        self.sim_custom_is_emccd = QCheckBox("EMCCD (apply EM gain)")
        self.sim_custom_is_emccd.setChecked(True)
        self.sim_custom_is_emccd.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_is_emccd, row, 1, 1, 2)
        custom_layout.addWidget(QLabel("QE:"), row, 3)
        self.sim_custom_qe = QDoubleSpinBox()
        self.sim_custom_qe.setRange(0.01, 1.0)
        self.sim_custom_qe.setDecimals(2)
        self.sim_custom_qe.setValue(1.0)
        self.sim_custom_qe.setSingleStep(0.05)
        self.sim_custom_qe.setEnabled(False)
        custom_layout.addWidget(self.sim_custom_qe, row, 4)

        # -- Blinking dynamics --
        row = 12
        custom_layout.addWidget(QLabel("<b>Blinking</b>"), row, 0)
        custom_layout.addWidget(QLabel("P(on):"), row, 1)
        self.sim_custom_p_on = QDoubleSpinBox()
        self.sim_custom_p_on.setRange(0.001, 1.0)
        self.sim_custom_p_on.setDecimals(3)
        self.sim_custom_p_on.setSingleStep(0.01)
        self.sim_custom_p_on.setValue(0.05)
        self.sim_custom_p_on.setToolTip("Probability of dark→on transition per frame")
        custom_layout.addWidget(self.sim_custom_p_on, row, 2)
        custom_layout.addWidget(QLabel("P(off):"), row, 3)
        self.sim_custom_p_off = QDoubleSpinBox()
        self.sim_custom_p_off.setRange(0.001, 1.0)
        self.sim_custom_p_off.setDecimals(3)
        self.sim_custom_p_off.setSingleStep(0.01)
        self.sim_custom_p_off.setValue(0.4)
        self.sim_custom_p_off.setToolTip("Probability of on→dark transition per frame")
        custom_layout.addWidget(self.sim_custom_p_off, row, 4)

        row = 13
        custom_layout.addWidget(QLabel("P(bleach):"), row, 1)
        self.sim_custom_p_bleach = QDoubleSpinBox()
        self.sim_custom_p_bleach.setRange(0.0, 1.0)
        self.sim_custom_p_bleach.setDecimals(4)
        self.sim_custom_p_bleach.setSingleStep(0.001)
        self.sim_custom_p_bleach.setValue(0.005)
        self.sim_custom_p_bleach.setToolTip("Probability of permanent bleaching per frame (0 = no bleaching)")
        custom_layout.addWidget(self.sim_custom_p_bleach, row, 2)

        # Estimated density label
        row = 14
        self.sim_custom_density_label = QLabel("")
        self.sim_custom_density_label.setStyleSheet("color: #555; font-style: italic;")
        custom_layout.addWidget(self.sim_custom_density_label, row, 0, 1, 6)
        self._update_custom_density_estimate()

        # Open in FLIKA option
        row = 15
        self.sim_custom_open_flika = QCheckBox("Open generated data in a FLIKA window")
        self.sim_custom_open_flika.setChecked(False)
        self.sim_custom_open_flika.setToolTip(
            "After generation, automatically open the synthetic image stack in a FLIKA viewer window"
        )
        custom_layout.addWidget(self.sim_custom_open_flika, row, 0, 1, 6)

        # Connect signals for density estimate update
        for spin in [self.sim_custom_n_molecules, self.sim_custom_frames,
                     self.sim_custom_width, self.sim_custom_height]:
            spin.valueChanged.connect(self._update_custom_density_estimate)
        self.sim_custom_p_on.valueChanged.connect(self._update_custom_density_estimate)
        self.sim_custom_p_off.valueChanged.connect(self._update_custom_density_estimate)

        layout.addWidget(custom_group)
        self.sim_custom_group = custom_group

        # Run button
        self.sim_run_btn = QPushButton("Generate Synthetic Datasets")
        self.sim_run_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px; font-size: 11pt;")
        self.sim_run_btn.setMinimumHeight(40)
        self.sim_run_btn.clicked.connect(self._start_synth_generate)
        layout.addWidget(self.sim_run_btn)

        layout.addStretch()

    # ========================================================================
    # TAB 2: ImageJ Macros
    # ========================================================================

    def _create_imagej_macros_tab(self):
        layout = self._create_scrollable_tab("ImageJ Macros")

        info = QLabel(
            "<b>Generate ImageJ ThunderSTORM Macros</b><br><br>"
            "This tab generates ImageJ macro files (.ijm) that run ThunderSTORM analysis on your "
            "data using the <b>original ImageJ ThunderSTORM plugin</b>. This allows direct "
            "comparison between FLIKA's ThunderSTORM implementation and the reference ImageJ version.<br><br>"
            "<b>How it works:</b><br>"
            "<ol>"
            "<li>Select your input TIFF file (real data or synthetic data from the Simulation tab)</li>"
            "<li>Choose which algorithm configurations to test (13 available, covering different "
            "filters, detectors, fitters, and parameter variations)</li>"
            "<li>Click 'Generate Macros' to create the .ijm files</li>"
            "<li>Open <b>Fiji/ImageJ</b> with the ThunderSTORM plugin installed</li>"
            "<li>Run the generated <code>run_all_tests.ijm</code> macro (Plugins > Macros > Run...)</li>"
            "<li>ImageJ will process each configuration and save CSV results</li>"
            "</ol>"
            "<b>Test configurations</b> cover:<br>"
            "<ul>"
            "<li><b>Filters:</b> Wavelet B-Spline (default & custom), Difference of Gaussians, Lowered Gaussian</li>"
            "<li><b>Detectors:</b> Local maximum, Non-maximum suppression, Centroid of connected components</li>"
            "<li><b>Fitters:</b> Integrated Gaussian (LSQ, WLSQ, MLE), Radial symmetry</li>"
            "<li><b>Options:</b> Multi-emitter fitting, different thresholds, fit radii</li>"
            "</ul>"
            "<i>After running the ImageJ macro, use the 'Real Data Comparison' tab to compare "
            "ImageJ results against FLIKA.</i>"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        settings = QGroupBox("Settings")
        s_layout = QGridLayout(settings)

        s_layout.addWidget(QLabel("Input TIFF:"), 0, 0)
        default_tif = str(Path(self._plugin_dir) / "test_data" / "real" / "Endothelial_NonBapta_bin10_crop.tif")
        self.macro_input_label = QLabel(default_tif if Path(default_tif).exists() else "No file selected")
        self.macro_input_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.macro_input_label, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.macro_input_label, "Select Input TIFF", "TIFF Files (*.tif *.tiff)"))
        s_layout.addWidget(btn, 0, 2)

        s_layout.addWidget(QLabel("Output Directory:"), 1, 0)
        self.macro_output_label = QLabel(str(Path(self._plugin_dir) / "tests" / "comparison" / "results"))
        self.macro_output_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.macro_output_label, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.macro_output_label, "Select Macro Output Directory"))
        s_layout.addWidget(btn, 1, 2)

        layout.addWidget(settings)

        # Algorithm selection
        algo_group = QGroupBox("Select Algorithm Configurations")
        algo_layout = QVBoxLayout(algo_group)
        self.macro_algo_list = QListWidget()
        self.macro_algo_list.setSelectionMode(QListWidget.MultiSelection)
        self._populate_algo_list(self.macro_algo_list)
        algo_layout.addWidget(self.macro_algo_list)

        sel_layout = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._select_all(self.macro_algo_list))
        sel_layout.addWidget(btn_all)
        btn_none = QPushButton("Select None")
        btn_none.clicked.connect(lambda: self._select_none(self.macro_algo_list))
        sel_layout.addWidget(btn_none)
        sel_layout.addStretch()
        algo_layout.addLayout(sel_layout)
        layout.addWidget(algo_group)

        # Synthetic data macros option
        synth_group = QGroupBox("Synthetic Data Macros (Optional)")
        synth_layout = QGridLayout(synth_group)

        self.macro_also_synth = QCheckBox("Also generate macros for synthetic data")
        self.macro_also_synth.setChecked(False)
        self.macro_also_synth.toggled.connect(self._on_macro_synth_toggled)
        synth_layout.addWidget(self.macro_also_synth, 0, 0, 1, 3)

        synth_layout.addWidget(QLabel("Synthetic Data Dir:"), 1, 0)
        self.macro_synth_dir_label = QLabel(str(Path(self._plugin_dir) / "test_data" / "synthetic"))
        self.macro_synth_dir_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.macro_synth_dir_label.setEnabled(False)
        synth_layout.addWidget(self.macro_synth_dir_label, 1, 1)
        self.macro_synth_dir_btn = QPushButton("Browse...")
        self.macro_synth_dir_btn.setEnabled(False)
        self.macro_synth_dir_btn.clicked.connect(lambda: self._browse_dir(self.macro_synth_dir_label, "Select Synthetic Data Directory"))
        synth_layout.addWidget(self.macro_synth_dir_btn, 1, 2)

        synth_note = QLabel(
            "<i>Points to the directory containing .tif and _ground_truth.csv files "
            "generated in the Simulation tab. Macros will be generated for each synthetic "
            "dataset found in this directory.</i>"
        )
        synth_note.setWordWrap(True)
        synth_note.setTextFormat(Qt.RichText)
        synth_note.setStyleSheet("color: #666; font-size: 9pt;")
        synth_layout.addWidget(synth_note, 2, 0, 1, 3)

        layout.addWidget(synth_group)

        # Generate button
        self.macro_generate_btn = QPushButton("Generate ImageJ Macros")
        self.macro_generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 11pt;")
        self.macro_generate_btn.setMinimumHeight(40)
        self.macro_generate_btn.clicked.connect(self._generate_macros)
        layout.addWidget(self.macro_generate_btn)

        # Macro preview
        preview_group = QGroupBox("Generated Macro Preview")
        p_layout = QVBoxLayout(preview_group)
        self.macro_preview = QTextEdit()
        self.macro_preview.setReadOnly(True)
        self.macro_preview.setMaximumHeight(200)
        self.macro_preview.setPlaceholderText("Macro content will appear here after generation...")
        p_layout.addWidget(self.macro_preview)
        layout.addWidget(preview_group)

        layout.addStretch()

    def _on_macro_synth_toggled(self, checked):
        """Enable/disable synthetic data directory controls."""
        self.macro_synth_dir_label.setEnabled(checked)
        self.macro_synth_dir_btn.setEnabled(checked)

    # ========================================================================
    # TAB 3: Real Data Comparison
    # ========================================================================

    def _create_real_comparison_tab(self):
        layout = self._create_scrollable_tab("Real Data Comparison")

        info = QLabel(
            "<b>Compare FLIKA vs ImageJ ThunderSTORM on Real Data</b><br><br>"
            "This tab runs FLIKA's ThunderSTORM implementation on real microscopy data and "
            "compares the results against ImageJ ThunderSTORM output (if available).<br><br>"
            "<b>What it does:</b><br>"
            "<ol>"
            "<li>Runs <b>all 13 algorithm configurations</b> through FLIKA's ThunderSTORM pipeline</li>"
            "<li>If ImageJ results are provided, performs <b>per-localization matching</b> "
            "(nearest-neighbour within the match radius)</li>"
            "<li>Computes <b>F1 score, precision, recall</b> — treating ImageJ as reference</li>"
            "<li>Measures <b>position error</b> (nm), intensity ratio, sigma error, uncertainty ratio</li>"
            "<li>Generates <b>comparison plots</b> and an <b>HTML report</b></li>"
            "</ol>"
            "<b>To get ImageJ results:</b> Use the 'ImageJ Macros' tab first to generate and run "
            "macros in Fiji, then point the 'ImageJ Results Dir' to the <code>imagej_results/</code> "
            "folder.<br><br>"
            "<b>Without ImageJ results:</b> FLIKA tests still run and save their output. "
            "You can add ImageJ results later and re-run the comparison."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        settings = QGroupBox("Settings")
        s_layout = QGridLayout(settings)

        s_layout.addWidget(QLabel("Input TIFF:"), 0, 0)
        default_tif = str(Path(self._plugin_dir) / "test_data" / "real" / "Endothelial_NonBapta_bin10_crop.tif")
        self.real_input_label = QLabel(default_tif if Path(default_tif).exists() else "No file selected")
        self.real_input_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.real_input_label, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_file(self.real_input_label, "Select Input TIFF", "TIFF Files (*.tif *.tiff)"))
        s_layout.addWidget(btn, 0, 2)

        s_layout.addWidget(QLabel("ImageJ Results Dir:"), 1, 0)
        default_ij = str(Path(self._plugin_dir) / "tests" / "comparison" / "results" / "imagej_results")
        self.real_imagej_label = QLabel(default_ij if Path(default_ij).exists() else "Optional - select after running ImageJ macros")
        self.real_imagej_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.real_imagej_label, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.real_imagej_label, "Select ImageJ Results Directory"))
        s_layout.addWidget(btn, 1, 2)

        s_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        self.real_output_label = QLabel(str(Path(self._plugin_dir) / "tests" / "comparison" / "results"))
        self.real_output_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.real_output_label, 2, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.real_output_label, "Select Output Directory"))
        s_layout.addWidget(btn, 2, 2)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Match radius (nm):"))
        self.real_match_spin = QDoubleSpinBox()
        self.real_match_spin.setRange(10.0, 1000.0)
        self.real_match_spin.setValue(200.0)
        opts.addWidget(self.real_match_spin)
        self.real_plots_check = QCheckBox("Generate plots")
        self.real_plots_check.setChecked(True)
        opts.addWidget(self.real_plots_check)
        opts.addStretch()
        s_layout.addLayout(opts, 3, 0, 1, 3)

        layout.addWidget(settings)

        self.real_run_btn = QPushButton("Run FLIKA Tests && Compare Against ImageJ")
        self.real_run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; font-size: 11pt;")
        self.real_run_btn.setMinimumHeight(40)
        self.real_run_btn.clicked.connect(self._start_comparison)
        layout.addWidget(self.real_run_btn)

        layout.addStretch()

    # ========================================================================
    # TAB 4: Ground Truth Testing
    # ========================================================================

    def _create_ground_truth_tab(self):
        layout = self._create_scrollable_tab("Ground Truth")

        info = QLabel(
            "<b>Ground Truth Testing with Synthetic Data</b><br><br>"
            "This tab runs FLIKA's ThunderSTORM on <b>synthetic datasets</b> (generated in the "
            "Simulation tab) and compares detected localizations against the <b>known ground truth</b> "
            "molecule positions.<br><br>"
            "<b>This is the gold standard test</b> because the true positions are known exactly. "
            "It measures how well each algorithm configuration can:<br>"
            "<ul>"
            "<li><b>Detect</b> molecules (recall — fraction of true molecules found)</li>"
            "<li><b>Avoid false positives</b> (precision — fraction of detections that are real)</li>"
            "<li><b>Localize accurately</b> (RMSE — root mean square position error in nm)</li>"
            "</ul>"
            "<b>Workflow:</b><br>"
            "<ol>"
            "<li>Select datasets and algorithm configurations to test</li>"
            "<li>Click 'Run FLIKA Analysis' to process all dataset/algorithm combinations</li>"
            "<li>Click 'Compare to Ground Truth' to compute F1, precision, recall, and RMSE</li>"
            "</ol>"
            "<b>Output:</b> JSON results, summary figures (F1 heatmap, box plots, RMSE distribution), "
            "and an HTML report.<br><br>"
            "<i>Tip: If you also ran ImageJ macros on the synthetic data, those results will be "
            "compared too (if found in the results directory).</i>"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        settings = QGroupBox("Directories")
        s_layout = QGridLayout(settings)

        s_layout.addWidget(QLabel("Synthetic Data Dir:"), 0, 0)
        self.gt_data_label = QLabel(str(Path(self._plugin_dir) / "test_data" / "synthetic"))
        self.gt_data_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.gt_data_label, 0, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.gt_data_label, "Select Synthetic Data Directory"))
        s_layout.addWidget(btn, 0, 2)

        s_layout.addWidget(QLabel("Results Dir:"), 1, 0)
        self.gt_results_label = QLabel(str(Path(self._plugin_dir) / "tests" / "synthetic" / "results"))
        self.gt_results_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        s_layout.addWidget(self.gt_results_label, 1, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse_dir(self.gt_results_label, "Select Results Directory"))
        s_layout.addWidget(btn, 1, 2)

        opts = QHBoxLayout()
        opts.addWidget(QLabel("Match radius (nm):"))
        self.gt_match_spin = QDoubleSpinBox()
        self.gt_match_spin.setRange(10.0, 1000.0)
        self.gt_match_spin.setValue(200.0)
        opts.addWidget(self.gt_match_spin)
        self.gt_plots_check = QCheckBox("Generate plots")
        self.gt_plots_check.setChecked(True)
        opts.addWidget(self.gt_plots_check)
        opts.addStretch()
        s_layout.addLayout(opts, 2, 0, 1, 3)

        layout.addWidget(settings)

        # Dataset selection
        ds_group = QGroupBox("Select Datasets")
        ds_layout = QVBoxLayout(ds_group)
        self.gt_dataset_list = QListWidget()
        self.gt_dataset_list.setSelectionMode(QListWidget.MultiSelection)
        self.gt_dataset_list.setMaximumHeight(130)
        self._populate_dataset_list(self.gt_dataset_list)
        ds_layout.addWidget(self.gt_dataset_list)
        layout.addWidget(ds_group)

        # Algorithm selection
        algo_group = QGroupBox("Select Algorithm Configurations")
        algo_layout = QVBoxLayout(algo_group)
        self.gt_algo_list = QListWidget()
        self.gt_algo_list.setSelectionMode(QListWidget.MultiSelection)
        self.gt_algo_list.setMaximumHeight(130)
        self._populate_algo_list(self.gt_algo_list)
        algo_layout.addWidget(self.gt_algo_list)
        layout.addWidget(algo_group)

        # Buttons
        btn_layout = QHBoxLayout()

        self.gt_run_flika_btn = QPushButton("Step 1: Run FLIKA Analysis")
        self.gt_run_flika_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 8px;")
        self.gt_run_flika_btn.setMinimumHeight(35)
        self.gt_run_flika_btn.clicked.connect(self._start_synth_run)
        btn_layout.addWidget(self.gt_run_flika_btn)

        self.gt_compare_btn = QPushButton("Step 2: Compare to Ground Truth")
        self.gt_compare_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px;")
        self.gt_compare_btn.setMinimumHeight(35)
        self.gt_compare_btn.clicked.connect(self._start_ground_truth)
        btn_layout.addWidget(self.gt_compare_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    # ========================================================================
    # TAB 5: Full Validation
    # ========================================================================

    def _create_full_validation_tab(self):
        layout = self._create_scrollable_tab("Full Validation")

        info = QLabel(
            "<b>Full Validation Suite (One-Click)</b><br><br>"
            "This tab runs the <b>complete validation pipeline</b> in one operation:<br><br>"
            "<b>Phase 1 — Synthetic Data Generation:</b><br>"
            "Generates all selected synthetic datasets with known ground truth positions.<br><br>"
            "<b>Phase 2 — FLIKA Analysis on Synthetic Data:</b><br>"
            "Runs every selected algorithm configuration on each synthetic dataset "
            "(up to 9 datasets x 13 algorithms = 117 tests).<br><br>"
            "<b>Phase 3 — Ground Truth Comparison:</b><br>"
            "Matches FLIKA detections against ground truth, computing F1, precision, recall, "
            "and RMSE for each test. Generates summary figures and heatmaps.<br><br>"
            "<b>Phase 4 — Real Data Comparison (optional):</b><br>"
            "If a real data TIFF and ImageJ results are available, also runs FLIKA on real data "
            "and compares against ImageJ ThunderSTORM.<br><br>"
            "<b>Output:</b> A comprehensive HTML report with all results, comparison tables, "
            "and embedded figures is generated and can be opened immediately.<br><br>"
            "<i>This may take several minutes depending on the number of datasets and algorithms selected.</i>"
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        layout.addWidget(info)

        # Settings (reuse paths from other tabs via getters)
        settings = QGroupBox("Paths (uses settings from other tabs)")
        s_layout = QFormLayout(settings)

        paths_info = QLabel(
            "Synthetic data dir: <i>from Simulation tab</i><br>"
            "Results dir: <i>from Ground Truth tab</i><br>"
            "Real data & ImageJ results: <i>from Real Data Comparison tab</i><br>"
            "Datasets & algorithms: <i>from Ground Truth tab selections</i>"
        )
        paths_info.setTextFormat(Qt.RichText)
        paths_info.setWordWrap(True)
        paths_info.setStyleSheet("color: #555;")
        s_layout.addRow(paths_info)
        layout.addWidget(settings)

        # Options
        opts_group = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_group)
        self.full_skip_existing = QCheckBox("Skip synthetic data generation if all datasets already exist")
        self.full_skip_existing.setChecked(True)
        self.full_skip_existing.setToolTip(
            "If checked, Phase 1 will be skipped when the synthetic data directory already "
            "contains TIFF files and ground truth CSVs for all selected datasets. "
            "Uncheck to force regeneration."
        )
        opts_layout.addWidget(self.full_skip_existing)
        layout.addWidget(opts_group)

        self.full_run_btn = QPushButton("Run Full Validation Suite")
        self.full_run_btn.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold; padding: 14px; font-size: 13pt;"
        )
        self.full_run_btn.setMinimumHeight(55)
        self.full_run_btn.clicked.connect(self._start_full_validation)
        layout.addWidget(self.full_run_btn)

        layout.addStretch()

    # ========================================================================
    # Helpers
    # ========================================================================

    def _populate_dataset_list(self, list_widget):
        try:
            from generate_synthetic_data import SYNTHETIC_CONFIGS
            for name, cfg in SYNTHETIC_CONFIGS.items():
                list_widget.addItem(f"{name} - {cfg['description']}")
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(True)
        except Exception:
            list_widget.addItem("(Could not load dataset configs)")

    def _populate_algo_list(self, list_widget):
        try:
            from generate_comparison_macros import TEST_CONFIGS
            for name, cfg in TEST_CONFIGS.items():
                list_widget.addItem(f"{name} - {cfg['description']}")
            for i in range(list_widget.count()):
                list_widget.item(i).setSelected(True)
        except Exception:
            list_widget.addItem("(Could not load algorithm configs)")

    def _select_all(self, list_widget):
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(True)

    def _select_none(self, list_widget):
        for i in range(list_widget.count()):
            list_widget.item(i).setSelected(False)

    def _get_selected(self, list_widget):
        items = list_widget.selectedItems()
        if not items:
            return None
        return [item.text().split(' - ')[0] for item in items]

    def _browse_dir(self, label, title):
        d = QFileDialog.getExistingDirectory(self, title)
        if d:
            label.setText(d)

    def _browse_file(self, label, title, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        if path:
            label.setText(path)

    def _on_optics_preset_changed(self, index):
        """Update optics fields when preset changes."""
        preset_key = self.sim_custom_optics.currentData()
        is_custom = (preset_key == "custom")
        self.sim_custom_pixel_size.setEnabled(is_custom)
        self.sim_custom_psf_sigma.setEnabled(is_custom)
        if not is_custom:
            try:
                from generate_synthetic_data import MAGNIFICATION_PRESETS
                p = MAGNIFICATION_PRESETS[preset_key]
                self.sim_custom_pixel_size.setValue(p['pixel_size_nm'])
                self.sim_custom_psf_sigma.setValue(p['psf_sigma_nm'])
            except Exception:
                pass

    def _on_camera_preset_changed(self, index):
        """Update camera fields when preset changes."""
        preset_key = self.sim_custom_camera.currentData()
        is_custom = (preset_key == "custom")
        for w in [self.sim_custom_baseline, self.sim_custom_photons_per_adu,
                   self.sim_custom_readout, self.sim_custom_em_gain,
                   self.sim_custom_is_emccd, self.sim_custom_qe]:
            w.setEnabled(is_custom)
        if not is_custom:
            try:
                from generate_synthetic_data import CAMERA_PRESETS
                c = CAMERA_PRESETS[preset_key]
                self.sim_custom_baseline.setValue(c['baseline_adu'])
                self.sim_custom_photons_per_adu.setValue(c['photons_per_adu'])
                self.sim_custom_readout.setValue(c['readout_noise_e'])
                self.sim_custom_em_gain.setValue(c['em_gain'])
                self.sim_custom_is_emccd.setChecked(c['is_emccd'])
                self.sim_custom_qe.setValue(c['quantum_efficiency'])
            except Exception:
                pass

    def _update_custom_density_estimate(self):
        """Update the estimated molecule density label."""
        try:
            n_mol = self.sim_custom_n_molecules.value()
            p_on = self.sim_custom_p_on.value()
            p_off = self.sim_custom_p_off.value()
            # Steady-state fraction of on molecules: p_on / (p_on + p_off)
            frac_on = p_on / (p_on + p_off) if (p_on + p_off) > 0 else 0
            avg_per_frame = n_mol * frac_on
            w = self.sim_custom_width.value()
            h = self.sim_custom_height.value()
            area = w * h
            density_per_px = avg_per_frame / area if area > 0 else 0
            self.sim_custom_density_label.setText(
                f"Estimated ~{avg_per_frame:.1f} active molecules/frame "
                f"(density: {density_per_px:.4f} molecules/pixel)"
            )
        except Exception:
            pass

    def _build_custom_config(self):
        """Build a synthetic dataset config dict from the custom UI fields."""
        optics_key = self.sim_custom_optics.currentData()
        camera_key = self.sim_custom_camera.currentData()

        config = {
            'description': f'Custom dataset ({self.sim_custom_name.text()})',
            'image_size': (self.sim_custom_height.value(), self.sim_custom_width.value()),
            'n_frames': self.sim_custom_frames.value(),
            'n_molecules': self.sim_custom_n_molecules.value(),
            'photons_per_molecule': self.sim_custom_photons.value(),
            'background_per_pixel': self.sim_custom_background.value(),
            'density_description': f'custom ({self.sim_custom_n_molecules.value()} molecules)',
            'blinking': {
                'p_on': self.sim_custom_p_on.value(),
                'p_off': self.sim_custom_p_off.value(),
                'p_bleach': self.sim_custom_p_bleach.value(),
            },
        }

        # Optics: use preset key or inline custom values
        if optics_key != "custom":
            config['optics'] = optics_key
        else:
            config['optics'] = '__custom__'
            config['custom_optics'] = {
                'description': 'Custom optics',
                'magnification': 100,
                'pixel_size_nm': self.sim_custom_pixel_size.value(),
                'na': 1.49,
                'emission_wavelength_nm': 680.0,
                'psf_sigma_nm': self.sim_custom_psf_sigma.value(),
            }

        # Camera: use preset key or inline custom values
        if camera_key != "custom":
            config['camera'] = camera_key
        else:
            config['camera'] = '__custom__'
            config['custom_camera'] = {
                'description': 'Custom camera',
                'baseline_adu': self.sim_custom_baseline.value(),
                'readout_noise_e': self.sim_custom_readout.value(),
                'photons_per_adu': self.sim_custom_photons_per_adu.value(),
                'em_gain': self.sim_custom_em_gain.value(),
                'is_emccd': self.sim_custom_is_emccd.isChecked(),
                'quantum_efficiency': self.sim_custom_qe.value(),
                'bit_depth': 16,
            }

        return self.sim_custom_name.text(), config

    def _open_custom_in_flika(self, results):
        """Open the custom synthetic dataset in a FLIKA window."""
        try:
            output_dir = results.get('output_dir', '')
            name = self.sim_custom_name.text().strip()
            if not name or not output_dir:
                return
            tiff_path = Path(output_dir) / f"{name}.tif"
            if not tiff_path.exists():
                return
            from flika.window import Window
            import tifffile
            stack = tifffile.imread(str(tiff_path))
            Window(stack, name=f"Synthetic: {name}")
            self.log_text.append(f"Opened '{name}' in FLIKA window")
        except Exception as e:
            self.log_text.append(f"Could not open in FLIKA: {e}")

    def _open_path(self, path):
        """Open a file or directory with the system default handler."""
        try:
            if sys.platform == 'darwin':
                subprocess.call(['open', str(path)])
            elif sys.platform.startswith('win'):
                os.startfile(str(path))
            else:
                subprocess.call(['xdg-open', str(path)])
        except Exception as e:
            QMessageBox.warning(self, "Cannot Open", f"Could not open:\n{e}\n\nPath: {path}")

    # ========================================================================
    # Worker management
    # ========================================================================

    def _set_running(self, running):
        """Toggle UI state for running/idle."""
        for btn in [self.sim_run_btn, self.macro_generate_btn, self.real_run_btn,
                     self.gt_run_flika_btn, self.gt_compare_btn, self.full_run_btn]:
            btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.progress_bar.setValue(0)
            self.view_report_btn.setEnabled(False)
            self.open_output_btn.setEnabled(False)
            self.view_figures_btn.setEnabled(False)

    def _start_worker(self, task_type, **kwargs):
        self._set_running(True)
        self.log_text.clear()
        self.validation_worker = ValidationWorker(task_type, **kwargs)
        self.validation_worker.progress_update.connect(self._on_progress)
        self.validation_worker.step_progress.connect(self._on_step_progress)
        self.validation_worker.test_complete.connect(self._on_complete)
        self.validation_worker.test_error.connect(self._on_error)
        self.validation_worker.start()

    def _cancel(self):
        if self.validation_worker and self.validation_worker.isRunning():
            self.validation_worker.cancel()
            self.status_label.setText("Cancelling...")
            self.status_label.setStyleSheet("color: orange; padding: 2px;")

    # ========================================================================
    # Task launchers
    # ========================================================================

    def _start_synth_generate(self):
        self._last_output_dir = self.sim_output_label.text()
        config_names = self._get_selected(self.sim_dataset_list) or []
        custom_configs = None
        if self.sim_custom_group.isChecked():
            name, cfg = self._build_custom_config()
            if not name.strip():
                QMessageBox.warning(self, "Missing Name", "Please enter a name for the custom dataset.")
                return
            custom_configs = {name: cfg}
        if not config_names and not custom_configs:
            QMessageBox.warning(self, "Nothing Selected",
                                "Please select at least one preset dataset or enable the Custom Dataset option.")
            return
        self._start_worker(
            'synthetic_generate',
            output_dir=self.sim_output_label.text(),
            config_names=config_names,
            seed=self.sim_seed_spin.value(),
            custom_configs=custom_configs,
        )

    def _generate_macros(self):
        """Generate ImageJ macros (runs synchronously — fast)."""
        input_path = self.macro_input_label.text()
        output_dir = self.macro_output_label.text()
        algo_names = self._get_selected(self.macro_algo_list)

        if not Path(input_path).exists():
            QMessageBox.warning(self, "Missing Input", "Please select a valid input TIFF file.")
            return

        try:
            from generate_comparison_macros import generate_all_macros, save_test_metadata, TEST_CONFIGS

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_test_metadata(output_dir, input_path)
            macro_path = generate_all_macros(input_path, output_dir, algo_names)

            # Show macro content in preview
            with open(str(macro_path)) as f:
                content = f.read()
            self.macro_preview.setPlainText(content)

            n_configs = len(algo_names) if algo_names else len(TEST_CONFIGS)
            self.status_label.setText(f"Generated {n_configs} macros. Main macro: {macro_path}")
            self.status_label.setStyleSheet("color: green; padding: 2px; font-weight: bold;")
            self.log_text.append(f"Generated macros in: {output_dir}")
            self.log_text.append(f"Run in Fiji: Plugins > Macros > Run... > {macro_path}")

            self._last_output_dir = output_dir
            self.open_output_btn.setEnabled(True)

            # Also generate synthetic macros if requested
            if self.macro_also_synth.isChecked():
                from generate_synthetic_data import SYNTHETIC_CONFIGS, MAGNIFICATION_PRESETS, CAMERA_PRESETS, generate_imagej_macros_for_dataset
                synth_dir = self.macro_synth_dir_label.text()
                imagej_dir = Path(output_dir) / "imagej_results"
                imagej_dir.mkdir(exist_ok=True)

                ds_configs = SYNTHETIC_CONFIGS
                ds_names = self._get_selected(self.sim_dataset_list)
                if ds_names:
                    ds_configs = {k: v for k, v in ds_configs.items() if k in ds_names}

                all_macros = []
                for ds_name, ds_cfg in ds_configs.items():
                    tiff_path = Path(synth_dir) / f"{ds_name}.tif"
                    if tiff_path.exists():
                        macros = generate_imagej_macros_for_dataset(
                            ds_name, ds_cfg, tiff_path, imagej_dir,
                            {k: v for k, v in TEST_CONFIGS.items() if not algo_names or k in algo_names})
                        for _, macro_text in macros:
                            all_macros.append(macro_text)

                if all_macros:
                    synth_macro_path = Path(output_dir) / "macros" / "run_all_synthetic.ijm"
                    synth_macro_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(str(synth_macro_path), 'w') as f:
                        f.write('\n\n'.join(all_macros))
                    self.log_text.append(f"Synthetic macros ({len(all_macros)} tests): {synth_macro_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate macros:\n{e}")
            import traceback
            traceback.print_exc()

    def _start_comparison(self):
        input_path = self.real_input_label.text()
        if not Path(input_path).exists():
            QMessageBox.warning(self, "Missing Input", "Please select a valid input TIFF file.")
            return

        imagej_dir = self.real_imagej_label.text()
        if not Path(imagej_dir).exists():
            imagej_dir = None

        self._last_output_dir = self.real_output_label.text()
        self._start_worker(
            'comparison',
            input_path=input_path,
            output_dir=self.real_output_label.text(),
            imagej_dir=imagej_dir,
            match_radius=self.real_match_spin.value(),
            generate_plots=self.real_plots_check.isChecked(),
        )

    def _start_synth_run(self):
        self._last_output_dir = self.gt_results_label.text()
        self._start_worker(
            'synthetic_run',
            data_dir=self.gt_data_label.text(),
            results_dir=self.gt_results_label.text(),
            config_names=self._get_selected(self.gt_dataset_list),
            algo_names=self._get_selected(self.gt_algo_list),
        )

    def _start_ground_truth(self):
        self._last_output_dir = str(Path(self.gt_results_label.text()) / "analysis")
        self._start_worker(
            'ground_truth',
            data_dir=self.gt_data_label.text(),
            results_dir=self.gt_results_label.text(),
            config_names=self._get_selected(self.gt_dataset_list),
            algo_names=self._get_selected(self.gt_algo_list),
            match_radius=self.gt_match_spin.value(),
            generate_plots=self.gt_plots_check.isChecked(),
        )

    def _start_full_validation(self):
        self._last_output_dir = str(Path(self.gt_results_label.text()) / "analysis")

        real_input = self.real_input_label.text()
        if not Path(real_input).exists():
            real_input = None
        imagej_dir = self.real_imagej_label.text() if hasattr(self, 'real_imagej_label') else None
        if imagej_dir and not Path(imagej_dir).exists():
            imagej_dir = None

        self._start_worker(
            'full_validation',
            synthetic_data_dir=self.sim_output_label.text(),
            synthetic_results_dir=self.gt_results_label.text(),
            config_names=self._get_selected(self.gt_dataset_list),
            algo_names=self._get_selected(self.gt_algo_list),
            seed=self.sim_seed_spin.value(),
            match_radius=self.gt_match_spin.value(),
            real_input_path=real_input,
            real_output_dir=self.real_output_label.text(),
            imagej_dir=imagej_dir,
            skip_existing_synthetic=self.full_skip_existing.isChecked(),
        )

    # ========================================================================
    # Signal handlers
    # ========================================================================

    def _on_progress(self, msg):
        self.status_label.setText(msg)
        self.log_text.append(msg)

    def _on_step_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_complete(self, results):
        self._set_running(False)
        self.progress_bar.setValue(100)
        self._last_results = results

        task_type = results.get('type', '')
        report_path = results.get('report_path')

        if task_type == 'comparison':
            stats = results.get('stats', {})
            if stats:
                f1_scores = [s['f1'] for s in stats.values()]
                msg = f"Comparison complete: {len(stats)} tests, mean F1={np.mean(f1_scores):.3f}"
            else:
                msg = f"FLIKA tests complete: {results.get('n_tests', 0)} configs (no ImageJ results for comparison)"
        elif task_type == 'synthetic_generate':
            msg = f"Generated {results.get('n_datasets', 0)} synthetic datasets"
            # Open custom dataset in FLIKA if requested
            if (self.sim_custom_group.isChecked() and
                    self.sim_custom_open_flika.isChecked()):
                self._open_custom_in_flika(results)
        elif task_type == 'synthetic_run':
            msg = f"FLIKA analysis complete: {results.get('n_tests', 0)} tests"
        elif task_type == 'ground_truth':
            msg = f"Ground truth comparison complete: {results.get('n_results', 0)} results"
        elif task_type == 'full_validation':
            gt = results.get('gt_results', [])
            rs = results.get('real_stats', {})
            msg = f"Full validation complete: {len(gt)} ground truth tests"
            if rs:
                msg += f", {len(rs)} real data comparisons"
            # Log summary of ground truth results
            if gt:
                f1_vals = [r['f1'] for r in gt if 'f1' in r]
                if f1_vals:
                    self.log_text.append(f"\nGround Truth Summary:")
                    self.log_text.append(f"  Mean F1:  {np.mean(f1_vals):.3f}")
                    self.log_text.append(f"  Min F1:   {np.min(f1_vals):.3f}")
                    self.log_text.append(f"  Max F1:   {np.max(f1_vals):.3f}")
                    rmse_vals = [r['rmse_nm'] for r in gt if 'rmse_nm' in r and not np.isnan(r['rmse_nm'])]
                    if rmse_vals:
                        self.log_text.append(f"  Mean RMSE: {np.mean(rmse_vals):.1f} nm")
        else:
            msg = "Complete"

        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: green; padding: 2px; font-weight: bold;")
        self.log_text.append(f"\n{'='*60}\n{msg}")

        self.open_output_btn.setEnabled(True)
        if report_path and Path(report_path).exists():
            self.view_report_btn.setEnabled(True)
            self._last_report_path = report_path

        output_dir = results.get('output_dir') or results.get('analysis_dir') or ''
        if output_dir:
            figs = list(Path(output_dir).glob("*.png"))
            for subdir in ['comparison_results', 'analysis']:
                sd = Path(output_dir) / subdir
                if sd.exists():
                    figs.extend(sd.glob("*.png"))
            if figs:
                self.view_figures_btn.setEnabled(True)
                self._last_figures_dir = str(output_dir)

        # Auto-open HTML report in browser for tasks that generate one
        if report_path and Path(report_path).exists():
            self._open_path(report_path)

    def _on_error(self, error_msg):
        self._set_running(False)
        self.status_label.setText(f"ERROR: {error_msg}")
        self.status_label.setStyleSheet("color: red; padding: 2px; font-weight: bold;")
        self.log_text.append(f"\nERROR: {error_msg}")
        QMessageBox.critical(self, "Validation Error", error_msg)

    # ========================================================================
    # Result viewing
    # ========================================================================

    def _view_report(self):
        if self._last_report_path and Path(self._last_report_path).exists():
            self._open_path(self._last_report_path)

    def _open_output_dir(self):
        if self._last_output_dir and Path(self._last_output_dir).exists():
            self._open_path(self._last_output_dir)

    def _view_figures(self):
        if not self._last_figures_dir:
            return
        png_files = sorted(Path(self._last_figures_dir).glob("*.png"))
        for subdir in ['comparison_results', 'analysis']:
            sd = Path(self._last_figures_dir) / subdir
            if sd.exists():
                png_files.extend(sorted(sd.glob("*.png")))
        for fig_path in png_files[:10]:
            self._open_path(str(fig_path))


# Plugin instance management
spt_batch_analysis_instance = None
validation_instance = None


def launch_spt_analysis():
    """Launch the enhanced SPT batch analysis plugin with geometric methods and autocorrelation"""
    global spt_batch_analysis_instance

    if spt_batch_analysis_instance is None or not spt_batch_analysis_instance.isVisible():
        spt_batch_analysis_instance = SPTBatchAnalysis()

    spt_batch_analysis_instance.show()
    spt_batch_analysis_instance.raise_()
    spt_batch_analysis_instance.activateWindow()

def launch_validation():
    """Launch the ThunderSTORM Validation & Comparison Testing GUI"""
    global validation_instance

    if validation_instance is None or not validation_instance.isVisible():
        validation_instance = ThunderSTORMValidation()

    validation_instance.show()
    validation_instance.raise_()
    validation_instance.activateWindow()

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
