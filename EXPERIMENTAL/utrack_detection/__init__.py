#!/usr/bin/env python3
"""
U-Track Particle Detection Plugin for FLIKA (Modified for Points)

This plugin integrates u-track subresolution feature detection methodology
with FLIKA's visualization and analysis capabilities. Modified to use points
instead of ROIs for marking detected particles.

Based on:
Ellefsen, K., Settle, B., Parker, I. & Smith, I. An algorithm for automated
detection, localization and measurement of local calcium signals from
camera-based imaging. Cell Calcium. 56:147-156, 2014

Author: [Your Name]
Version: 1.0.1 (Modified for Points)
"""

import numpy as np
import os
import sys
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Add current plugin directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import FLIKA modules with proper fallbacks
FLIKA_AVAILABLE = False
BaseProcess = None
SliderLabel = None
CheckBox = None
ComboBox = None
WindowSelector = None
save_file_gui = None
open_file_gui = None
g = None

# Try multiple import paths for FLIKA BaseProcess
try:
    from flika import global_vars as g
    from flika.window import Window
    FLIKA_AVAILABLE = True
    print("✓ FLIKA core modules imported successfully")

    # Try different BaseProcess import paths
    try:
        from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
        print("✓ BaseProcess imported from flika.process.BaseProcess")
    except ImportError:
        try:
            from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
            print("✓ BaseProcess imported from flika.utils.BaseProcess")
        except ImportError:
            try:
                from flika.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
                print("✓ BaseProcess imported from flika.BaseProcess")
            except ImportError:
                print("⚠ Could not import BaseProcess from any known location")
                FLIKA_AVAILABLE = False

    # Try utils imports
    try:
        from flika.utils.misc import save_file_gui, open_file_gui
        print("✓ Utils module imported")
    except ImportError:
        print("⚠ Could not import utils module")
        def save_file_gui(*args, **kwargs):
            return "test_output.csv"
        def open_file_gui(*args, **kwargs):
            return "test_input.csv"

except ImportError as e:
    print(f"⚠ FLIKA core modules not available: {e}")
    FLIKA_AVAILABLE = False

# Create dummy classes if FLIKA not available
if not FLIKA_AVAILABLE:
    print("Creating dummy FLIKA classes...")

    class BaseProcess:
        def __init__(self):
            self.tif = None
            self.oldname = "test"
            self.items = []

        def start(self, keepSourceWindow):
            pass

        def end(self):
            return None

        def gui_reset(self):
            pass

        def gui(self):
            pass

    # Create dummy classes for GUI elements
    class SliderLabel:
        def setRange(self, min_val, max_val): pass
        def setValue(self, val): pass
        def setSingleStep(self, step): pass
        def value(self): return 1.0

    class CheckBox:
        def setChecked(self, checked): pass
        def isChecked(self): return True

    class ComboBox:
        def addItem(self, item): pass
        def setCurrentText(self, text): pass
        def currentText(self): return 'g'

    class WindowSelector:
        pass

    def save_file_gui(*args, **kwargs):
        return "test_output.csv"

    def open_file_gui(*args, **kwargs):
        return "test_input.csv"

    # Create dummy g object
    class GlobalVars:
        currentWindow = None
        windows = []
        def alert(self, msg):
            print(f"ALERT: {msg}")

    g = GlobalVars()

# Import U-track detection modules with better error handling
UTRACK_AVAILABLE = False
utrack_import_error = None

try:
    # Use relative imports since files are in same directory
    from .detect_sub_res_features_2d import detect_sub_res_features_2d_standalone
    from .parameter_classes import DetectionParam, MovieParam
    from .image_processing import filter_gauss_2d, locmax_2d
    from .utils import progress_text
    UTRACK_AVAILABLE = True
    print("✓ U-Track detection modules imported successfully")
except ImportError:
    try:
        # Fallback: try direct imports
        from detect_sub_res_features_2d import detect_sub_res_features_2d_standalone
        from parameter_classes import DetectionParam, MovieParam
        from image_processing import filter_gauss_2d, locmax_2d
        from utils import progress_text
        UTRACK_AVAILABLE = True
        print("✓ U-Track detection modules imported successfully (direct import)")
    except ImportError as e:
        utrack_import_error = str(e)
        print(f"⚠ U-Track detection modules not available: {e}")
        print("Please ensure u-track modules are in the plugin directory")

        # Create dummy classes to prevent further import errors
        class DetectionParam:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class MovieParam:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def detect_sub_res_features_2d_standalone(*args, **kwargs):
            raise ImportError("U-Track modules not available")

        def filter_gauss_2d(*args, **kwargs):
            raise ImportError("U-Track modules not available")

        def locmax_2d(*args, **kwargs):
            raise ImportError("U-Track modules not available")

        def progress_text(*args, **kwargs):
            pass

# Import scikit-image for TIFF handling
try:
    from skimage import io
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠ scikit-image not available - TIFF saving will not work")


class UTrackDetection(BaseProcess):
    """
    U-Track subresolution particle detection for FLIKA.

    This process applies the u-track detection algorithm to detect and localize
    subresolution particles in fluorescence microscopy images. Results are
    displayed as points instead of ROIs for better performance.
    """

    def __init__(self):
        super().__init__()
        self.detection_results = None
        self.temp_dir = None
        self.detection_points = []  # Store detected points

    def get_init_settings_dict(self):
        """Initialize default parameter values"""
        return {
            # Core Detection Parameters (scaled to integers for SliderLabel)
            'psf_sigma': 10,           # Will be divided by 10 (1.0 pixels)
            'calc_method': 'g',        # 'g', 'gv', or 'c'
            'alpha_loc_max': 10,       # Will be divided by 1000 (0.01)

            # Statistical Test Parameters (scaled to integers)
            'alpha_r': 50,             # Will be divided by 1000 (0.05)
            'alpha_a': 50,             # Will be divided by 1000 (0.05)
            'alpha_d': 5,              # Will be divided by 1000 (0.005)
            'alpha_f': 0,              # Will be divided by 1000 (0.0)

            # Processing Options
            'do_mmf': True,            # Use mixture model fitting
            'bit_depth': 16,
            'num_sigma_iter': 0,       # PSF sigma estimation iterations
            'integ_window': 0,         # Temporal integration window

            # Advanced Parameters (scaled to integers)
            'filter_sigma': 3,         # Will be divided by 10 (0.3)
            'locmax_threshold': 30,    # Will be divided by 100 (0.3)
            'locmax_window': 5,        # Local maxima window size

            # Point Output Options
            'create_points': True,      # Create points for detections
            'points_current_frame_only': False,  # Only create points for current frame
            'points_max_per_frame': 50, # Maximum points per frame (0 = unlimited)
            'show_results_window': True, # Show results in new window
            'save_to_file': True,      # Save results to file
            'save_points_file': True,  # Save points to FLIKA-compatible format
            'show_statistics': True,   # Print detection statistics

            # Visualization Options
            'show_amplitudes': True,   # Show amplitude info
            'min_amplitude': 0,        # Minimum amplitude to display
        }

    def get_name(self):
        return 'U-Track Particle Detection'

    def get_menu_path(self):
        return 'Plugins>Particle Detection>U-Track Detection'

    def gui(self):
        """Create the GUI for parameter adjustment"""
        self.gui_reset()

        if not UTRACK_AVAILABLE:
            error_msg = (f"U-Track detection modules not available!\n\n"
                       f"Error: {utrack_import_error}\n\n"
                       f"Please ensure these files are in the plugin directory:\n"
                       f"- detect_sub_res_features_2d.py\n"
                       f"- detection.py\n"
                       f"- image_processing.py\n"
                       f"- fitting.py\n"
                       f"- utils.py\n"
                       f"- parameter_classes.py")
            if FLIKA_AVAILABLE:
                g.alert(error_msg)
            else:
                print(error_msg)
            return

        if FLIKA_AVAILABLE and g.currentWindow is None:
            g.alert("No image window open!\nPlease open an image first.")
            return

        # Get current window info
        self.gui_reset()

        # PSF Parameters (using integer ranges)
        psf_sigma = SliderLabel()
        psf_sigma.setRange(1, 100)  # Will represent 0.1 to 10.0 (divide by 10)
        psf_sigma.setValue(10)
        psf_sigma.setSingleStep(1)

        calc_method = ComboBox()
        calc_method.addItem('Gaussian MMF (g)')
        calc_method.addItem('Variable Sigma MMF (gv)')
        calc_method.addItem('Centroid (c)')
        calc_method.setCurrentText('Gaussian MMF (g)')

        alpha_loc_max = SliderLabel()
        alpha_loc_max.setRange(1, 100)  # Will represent 0.001 to 0.1 (divide by 1000)
        alpha_loc_max.setValue(10)
        alpha_loc_max.setSingleStep(1)

        # Statistical Tests (using integer ranges)
        alpha_r = SliderLabel()
        alpha_r.setRange(1, 200)  # Will represent 0.001 to 0.2 (divide by 1000)
        alpha_r.setValue(50)
        alpha_r.setSingleStep(1)

        alpha_a = SliderLabel()
        alpha_a.setRange(1, 200)  # Will represent 0.001 to 0.2 (divide by 1000)
        alpha_a.setValue(50)
        alpha_a.setSingleStep(1)

        alpha_d = SliderLabel()
        alpha_d.setRange(1, 100)  # Will represent 0.001 to 0.1 (divide by 1000)
        alpha_d.setValue(5)
        alpha_d.setSingleStep(1)

        alpha_f = SliderLabel()
        alpha_f.setRange(0, 100)  # Will represent 0.0 to 0.1 (divide by 1000)
        alpha_f.setValue(0)
        alpha_f.setSingleStep(1)

        # Processing options
        do_mmf = CheckBox()
        do_mmf.setChecked(True)

        bit_depth = SliderLabel()
        bit_depth.setRange(8, 32)
        bit_depth.setValue(16)
        bit_depth.setSingleStep(1)

        num_sigma_iter = SliderLabel()
        num_sigma_iter.setRange(0, 20)
        num_sigma_iter.setValue(0)
        num_sigma_iter.setSingleStep(1)

        integ_window = SliderLabel()
        integ_window.setRange(0, 10)
        integ_window.setValue(0)
        integ_window.setSingleStep(1)

        # Advanced parameters (using integer ranges)
        filter_sigma = SliderLabel()
        filter_sigma.setRange(1, 30)  # Will represent 0.1 to 3.0 (divide by 10)
        filter_sigma.setValue(3)
        filter_sigma.setSingleStep(1)

        locmax_threshold = SliderLabel()
        locmax_threshold.setRange(5, 100)  # Will represent 0.05 to 1.0 (divide by 100)
        locmax_threshold.setValue(30)
        locmax_threshold.setSingleStep(1)

        locmax_window = SliderLabel()
        locmax_window.setRange(3, 15)
        locmax_window.setValue(5)
        locmax_window.setSingleStep(2)

        # Point output options
        create_points = CheckBox()
        create_points.setChecked(True)

        points_current_frame_only = CheckBox()
        points_current_frame_only.setChecked(False)

        points_max_per_frame = SliderLabel()
        points_max_per_frame.setRange(0, 200)  # 0 = unlimited, up to 200 per frame
        points_max_per_frame.setValue(50)
        points_max_per_frame.setSingleStep(10)

        show_results_window = CheckBox()
        show_results_window.setChecked(True)

        save_to_file = CheckBox()
        save_to_file.setChecked(True)

        save_points_file = CheckBox()
        save_points_file.setChecked(True)

        show_statistics = CheckBox()
        show_statistics.setChecked(True)

        # Visualization options
        show_amplitudes = CheckBox()
        show_amplitudes.setChecked(True)

        min_amplitude = SliderLabel()
        min_amplitude.setRange(0, 10000)
        min_amplitude.setValue(0)
        min_amplitude.setSingleStep(100)

        # Add items to GUI with proper descriptions
        self.items.append({'name': 'psf_sigma', 'string': 'PSF Sigma (×0.1 pixels)', 'object': psf_sigma})
        self.items.append({'name': 'calc_method', 'string': 'Calculation Method', 'object': calc_method})
        self.items.append({'name': 'alpha_loc_max', 'string': 'Alpha Local Max (×0.001)', 'object': alpha_loc_max})

        self.items.append({'name': 'alpha_r', 'string': 'Alpha R (×0.001)', 'object': alpha_r})
        self.items.append({'name': 'alpha_a', 'string': 'Alpha A (×0.001)', 'object': alpha_a})
        self.items.append({'name': 'alpha_d', 'string': 'Alpha D (×0.001)', 'object': alpha_d})
        self.items.append({'name': 'alpha_f', 'string': 'Alpha F (×0.001)', 'object': alpha_f})

        self.items.append({'name': 'do_mmf', 'string': 'Use MMF', 'object': do_mmf})
        self.items.append({'name': 'bit_depth', 'string': 'Bit Depth', 'object': bit_depth})
        self.items.append({'name': 'num_sigma_iter', 'string': 'Sigma Estimation Iterations', 'object': num_sigma_iter})
        self.items.append({'name': 'integ_window', 'string': 'Integration Window', 'object': integ_window})

        self.items.append({'name': 'filter_sigma', 'string': 'Pre-filter Sigma (×0.1)', 'object': filter_sigma})
        self.items.append({'name': 'locmax_threshold', 'string': 'Local Max Threshold (×0.01)', 'object': locmax_threshold})
        self.items.append({'name': 'locmax_window', 'string': 'Local Max Window', 'object': locmax_window})

        self.items.append({'name': 'create_points', 'string': 'Create Points', 'object': create_points})
        self.items.append({'name': 'show_results_window', 'string': 'Show Results Window', 'object': show_results_window})
        self.items.append({'name': 'save_to_file', 'string': 'Save Results to File', 'object': save_to_file})
        self.items.append({'name': 'save_points_file', 'string': 'Save Points File', 'object': save_points_file})
        self.items.append({'name': 'show_statistics', 'string': 'Show Statistics', 'object': show_statistics})

        self.items.append({'name': 'show_amplitudes', 'string': 'Show Amplitudes', 'object': show_amplitudes})
        self.items.append({'name': 'min_amplitude', 'string': 'Min Amplitude Threshold', 'object': min_amplitude})

        super().gui()

    def __call__(self, psf_sigma=10, calc_method='g', alpha_loc_max=10,
                 alpha_r=50, alpha_a=50, alpha_d=5, alpha_f=0,
                 do_mmf=True, bit_depth=16, num_sigma_iter=0, integ_window=0,
                 filter_sigma=3, locmax_threshold=30, locmax_window=5,
                 create_points=True, show_results_window=True, save_to_file=True,
                 save_points_file=True, show_statistics=True, show_amplitudes=True,
                 min_amplitude=0, keepSourceWindow=True):
        """
        Execute the u-track particle detection algorithm
        """

        if not UTRACK_AVAILABLE:
            error_msg = "U-Track detection modules not available!"
            if FLIKA_AVAILABLE:
                g.alert(error_msg)
            else:
                print(error_msg)
            return None

        if FLIKA_AVAILABLE and g.currentWindow is None:
            error_msg = "No current window!"
            if FLIKA_AVAILABLE:
                g.alert(error_msg)
            else:
                print(error_msg)
            return None

        # Start processing
        self.start(keepSourceWindow)

        try:
            # Convert scaled integer values back to floats
            psf_sigma_float = psf_sigma / 10.0
            alpha_loc_max_float = alpha_loc_max / 1000.0
            alpha_r_float = alpha_r / 1000.0
            alpha_a_float = alpha_a / 1000.0
            alpha_d_float = alpha_d / 1000.0
            alpha_f_float = alpha_f / 1000.0
            filter_sigma_float = filter_sigma / 10.0
            locmax_threshold_float = locmax_threshold / 100.0

            # Convert calc_method from display string to code
            method_map = {
                'Gaussian MMF (g)': 'g',
                'Variable Sigma MMF (gv)': 'gv',
                'Centroid (c)': 'c'
            }
            calc_code = method_map.get(calc_method, 'g')

            # Get image data
            if FLIKA_AVAILABLE:
                image_data = self.tif.copy()
            else:
                # For testing without FLIKA
                image_data = np.random.randint(0, 1000, (5, 100, 100)).astype(np.uint16)

            if show_statistics:
                print(f"U-Track Detection Parameters:")
                print(f"  PSF Sigma: {psf_sigma_float:.2f}")
                print(f"  Method: {calc_code}")
                print(f"  Alpha Local Max: {alpha_loc_max_float:.4f}")
                print(f"  Alpha R: {alpha_r_float:.4f}")
                print(f"  Alpha A: {alpha_a_float:.4f}")
                print(f"  Alpha D: {alpha_d_float:.4f}")
                print(f"  MMF: {do_mmf}")
                print(f"  Image shape: {image_data.shape}")
                print(f"  Image range: {np.min(image_data):.1f} to {np.max(image_data):.1f}")

            # Create temporary directory for frame files
            self.temp_dir = tempfile.mkdtemp(prefix="flika_utrack_")

            # Save frames as individual TIFF files (required by u-track)
            self._save_frames_to_temp_dir(image_data)

            # Create parameter objects
            movie_param = self._create_movie_param(len(image_data))
            detection_param = self._create_detection_param(
                psf_sigma_float, calc_code, alpha_loc_max_float, alpha_r_float,
                alpha_a_float, alpha_d_float, alpha_f_float, do_mmf, bit_depth,
                num_sigma_iter, integ_window
            )

            # Run u-track detection
            if show_statistics:
                print("Running u-track detection...")

            start_time = time.time()

            movie_info, exceptions, local_maxima, background, final_sigma = \
                detect_sub_res_features_2d_standalone(
                    movie_param,
                    detection_param,
                    save_results=None,
                    verbose=show_statistics
                )

            detection_time = time.time() - start_time

            # Store results
            self.detection_results = {
                'movie_info': movie_info,
                'exceptions': exceptions,
                'local_maxima': local_maxima,
                'background': background,
                'final_sigma': final_sigma,
                'detection_time': detection_time,
                'parameters': {
                    'psf_sigma': psf_sigma_float,
                    'calc_method': calc_code,
                    'alpha_loc_max': alpha_loc_max_float,
                    'alpha_r': alpha_r_float,
                    'alpha_a': alpha_a_float,
                    'alpha_d': alpha_d_float,
                    'alpha_f': alpha_f_float,
                    'do_mmf': do_mmf,
                    'bit_depth': bit_depth
                }
            }

            # Process and display results
            self._process_detection_results(
                movie_info, create_points, show_results_window, save_to_file,
                save_points_file, show_statistics, show_amplitudes, min_amplitude,
                final_sigma, detection_time
            )

            # Create output window with detection overlay if requested
            if show_results_window:
                self.newtif = self._create_detection_overlay(
                    image_data, movie_info, min_amplitude
                )
                self.newname = f"{self.oldname} - U-Track Detection"
            else:
                self.newtif = image_data.copy()
                self.newname = self.oldname

        except Exception as e:
            error_msg = f"Detection failed: {str(e)}"
            if FLIKA_AVAILABLE:
                g.alert(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None

        finally:
            # Clean up temporary files
            self._cleanup_temp_dir()

        if FLIKA_AVAILABLE:
            g.utrack_detection_results = self.detection_results

        return self.end()

    def _save_frames_to_temp_dir(self, image_data):
        """Save image frames as individual TIFF files for u-track"""
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for TIFF saving")

        for i in range(len(image_data)):
            frame = image_data[i]

            # Ensure frame is in appropriate format
            if frame.dtype != np.uint16:
                if frame.max() <= 1.0:
                    # Assume normalized, scale to 16-bit
                    frame_uint16 = (frame * 65535).astype(np.uint16)
                else:
                    # Clip to 16-bit range
                    frame_uint16 = np.clip(frame, 0, 65535).astype(np.uint16)
            else:
                frame_uint16 = frame

            frame_path = os.path.join(self.temp_dir, f"frame_{i+1:05d}.tif")
            io.imsave(frame_path, frame_uint16)

    def _create_movie_param(self, num_frames):
        """Create MovieParam object for u-track detection"""
        movie_param = MovieParam()
        movie_param.image_dir = self.temp_dir
        movie_param.filename_base = "frame_"
        movie_param.digits_4_enum = 5
        movie_param.first_image_num = 1
        movie_param.last_image_num = num_frames
        return movie_param

    def _create_detection_param(self, psf_sigma, calc_method, alpha_loc_max,
                               alpha_r, alpha_a, alpha_d, alpha_f, do_mmf,
                               bit_depth, num_sigma_iter, integ_window):
        """Create DetectionParam object for u-track detection"""
        detection_param = DetectionParam()
        detection_param.psf_sigma = psf_sigma
        detection_param.calc_method = calc_method
        detection_param.alpha_loc_max = alpha_loc_max
        detection_param.test_alpha = {
            'alphaR': alpha_r,
            'alphaA': alpha_a,
            'alphaD': alpha_d,
            'alphaF': alpha_f
        }
        detection_param.do_mmf = do_mmf
        detection_param.bit_depth = bit_depth
        detection_param.num_sigma_iter = num_sigma_iter
        detection_param.integ_window = integ_window
        detection_param.visual = False
        return detection_param

    def _process_detection_results(self, movie_info, create_points, show_results_window,
                                  save_to_file, save_points_file, show_statistics,
                                  show_amplitudes, min_amplitude, final_sigma,
                                  detection_time):
        """Process and display detection results"""

        # Count total detections
        total_detections = 0
        frame_counts = []
        all_amplitudes = []

        for frame_idx, frame_info in enumerate(movie_info):
            count = 0
            if (frame_info and hasattr(frame_info, 'xCoord') and
                frame_info.xCoord is not None and len(frame_info.xCoord) > 0):
                count = len(frame_info.xCoord)
                total_detections += count

                # Collect amplitudes
                if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                    if frame_info.amp.ndim == 2:
                        amplitudes = frame_info.amp[:, 0]
                    else:
                        amplitudes = frame_info.amp
                    all_amplitudes.extend(amplitudes)

            frame_counts.append(count)

        # Show statistics
        if show_statistics:
            print(f"\n=== U-TRACK DETECTION RESULTS ===")
            print(f"Detection time: {detection_time:.2f} seconds")
            print(f"Final PSF sigma: {final_sigma:.3f} pixels")
            print(f"Total detections: {total_detections}")
            print(f"Frames processed: {len(movie_info)}")
            print(f"Average detections/frame: {np.mean(frame_counts):.1f}")
            print(f"Max detections/frame: {np.max(frame_counts)}")

            if all_amplitudes:
                print(f"Amplitude range: {np.min(all_amplitudes):.1f} - {np.max(all_amplitudes):.1f}")
                print(f"Mean amplitude: {np.mean(all_amplitudes):.1f}")

        # Create points if requested
        if create_points and total_detections > 0:
            self._create_detection_points(movie_info, min_amplitude, show_amplitudes)

        # Save points file if requested
        if save_points_file and total_detections > 0:
            self._save_points_file(movie_info, min_amplitude)

        # Save to file if requested
        if save_to_file:
            self._save_detection_results(movie_info, final_sigma)

    def _create_detection_points(self, movie_info, min_amplitude, show_amplitudes):
        """Create and store detection points, and add them to FLIKA window"""

        self.detection_points = []
        scatter_pts = []  # For FLIKA scatter points [frame, x, y] format
        point_count = 0

        for frame_idx, frame_info in enumerate(movie_info):
            if (frame_info and hasattr(frame_info, 'xCoord') and
                frame_info.xCoord is not None and len(frame_info.xCoord) > 0):

                # Extract coordinates and amplitudes
                if frame_info.xCoord.ndim == 2:
                    x_coords = frame_info.xCoord[:, 0]
                    y_coords = frame_info.yCoord[:, 0]
                else:
                    x_coords = frame_info.xCoord
                    y_coords = frame_info.yCoord

                if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                    if frame_info.amp.ndim == 2:
                        amplitudes = frame_info.amp[:, 0]
                    else:
                        amplitudes = frame_info.amp
                else:
                    amplitudes = np.ones(len(x_coords))

                # Create points for each detection
                for i, (x, y, amp) in enumerate(zip(x_coords, y_coords, amplitudes)):
                    if amp >= min_amplitude:
                        # Store detailed point info
                        point = {
                            'frame': frame_idx + 1,  # FLIKA uses 1-based frame indexing
                            'x': float(x),
                            'y': float(y),
                            'amplitude': float(amp),
                            'detection_id': point_count + 1
                        }
                        self.detection_points.append(point)

                        # Add to FLIKA scatter points format [frame, x, y]
                        # Note: FLIKA expects x=column, y=row (swap from detection results)
                        scatter_pts.append([frame_idx, float(y), float(x)])
                        point_count += 1

                        # Show amplitude information if requested
                        if show_amplitudes and point_count <= 10:  # Limit output
                            print(f"Point {point_count}: Frame {frame_idx+1}, "
                                  f"x={x:.2f}, y={y:.2f}, amp={amp:.1f}")

        if point_count > 0:
            print(f"Created {point_count} detection points")

            # Add points to current FLIKA window if available
            if FLIKA_AVAILABLE and g.currentWindow is not None:
                try:
                    # Convert to numpy array format that FLIKA expects
                    scatter_array = np.array(scatter_pts)

                    # Add scatter points to current window
                    if hasattr(g.currentWindow, 'scatterPts'):
                        # Append to existing points
                        if g.currentWindow.scatterPts is not None and len(g.currentWindow.scatterPts) > 0:
                            g.currentWindow.scatterPts = np.vstack([g.currentWindow.scatterPts, scatter_array])
                        else:
                            g.currentWindow.scatterPts = scatter_array
                    else:
                        # Create new scatter points attribute
                        g.currentWindow.scatterPts = scatter_array

                    # Force refresh of the window display
                    try:
                        if hasattr(g.currentWindow, 'updateWindow'):
                            g.currentWindow.updateWindow()
                        elif hasattr(g.currentWindow, 'update'):
                            g.currentWindow.update()
                        elif hasattr(g.currentWindow, 'repaint'):
                            g.currentWindow.repaint()
                    except:
                        pass  # If refresh fails, points should still be there

                    print(f"Added {point_count} points to FLIKA window")
                    g.alert(f"Added {point_count} detection points to window")

                except Exception as e:
                    print(f"Warning: Could not add points to FLIKA window: {e}")
                    g.alert(f"Created {point_count} detection points\n(Could not add to window display)")

            # Store points in global variable for access
            if FLIKA_AVAILABLE:
                g.utrack_detection_points = self.detection_points
        else:
            print("No points created (no detections above threshold)")

    def _save_points_file(self, movie_info, min_amplitude):
        """Save detection points to FLIKA-compatible text file"""
        try:
            # Generate default filename
            if FLIKA_AVAILABLE and hasattr(g.currentWindow, 'name'):
                base_name = g.currentWindow.name
            else:
                base_name = "utrack_detections"

            default_filename = f"{base_name}_points.txt"

            # Get filename from user
            filename = save_file_gui("Save Detection Points",
                                   directory=default_filename,
                                   filetypes="Text files (*.txt);;CSV files (*.csv);;All files (*)")

            if filename:
                self._write_points_file(filename, movie_info, min_amplitude)
                print(f"Detection points saved to: {filename}")

                if FLIKA_AVAILABLE:
                    # Ask user if they want to load the points into current window
                    load_msg = f"Points saved to {filename}\n\nLoad points into current window?"
                    # For now, just inform the user
                    g.alert(f"Points saved to {filename}\n\nUse File > Open Points to load them into FLIKA")

        except Exception as e:
            error_msg = f"Failed to save points file: {str(e)}"
            if FLIKA_AVAILABLE:
                g.alert(error_msg)
            else:
                print(error_msg)

    def _write_points_file(self, filename, movie_info, min_amplitude):
        """Write points to file in FLIKA-compatible format"""

        # Determine file format from extension
        is_csv = filename.lower().endswith('.csv')

        if is_csv:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header for CSV (detailed format with amplitude)
                writer.writerow(['Frame', 'X', 'Y', 'Amplitude'])

                # Write points
                for frame_idx, frame_info in enumerate(movie_info):
                    if (frame_info and hasattr(frame_info, 'xCoord') and
                        frame_info.xCoord is not None and len(frame_info.xCoord) > 0):

                        # Extract coordinates and amplitudes
                        if frame_info.xCoord.ndim == 2:
                            x_coords = frame_info.xCoord[:, 0]
                            y_coords = frame_info.yCoord[:, 0]
                        else:
                            x_coords = frame_info.xCoord
                            y_coords = frame_info.yCoord

                        if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                            if frame_info.amp.ndim == 2:
                                amplitudes = frame_info.amp[:, 0]
                            else:
                                amplitudes = frame_info.amp
                        else:
                            amplitudes = np.ones(len(x_coords))

                        # Write each detection
                        for x, y, amp in zip(x_coords, y_coords, amplitudes):
                            if amp >= min_amplitude:
                                # Note: Keep original x,y order for CSV (detailed format)
                                writer.writerow([frame_idx + 1, f"{x:.6f}", f"{y:.6f}", f"{amp:.6f}"])
        else:
            # Write as simple text file (FLIKA points format - NO COMMENTS, just frame x y)
            with open(filename, 'w') as f:
                # Write points in simple format: frame x y (space-separated, no header)
                for frame_idx, frame_info in enumerate(movie_info):
                    if (frame_info and hasattr(frame_info, 'xCoord') and
                        frame_info.xCoord is not None and len(frame_info.xCoord) > 0):

                        # Extract coordinates and amplitudes
                        if frame_info.xCoord.ndim == 2:
                            x_coords = frame_info.xCoord[:, 0]
                            y_coords = frame_info.yCoord[:, 0]
                        else:
                            x_coords = frame_info.xCoord
                            y_coords = frame_info.yCoord

                        if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                            if frame_info.amp.ndim == 2:
                                amplitudes = frame_info.amp[:, 0]
                            else:
                                amplitudes = frame_info.amp
                        else:
                            amplitudes = np.ones(len(x_coords))

                        # Write each detection in simple format
                        for x, y, amp in zip(x_coords, y_coords, amplitudes):
                            if amp >= min_amplitude:
                                # FLIKA expects: frame x y (space-separated, 0-based frame indexing)
                                # Note: Swap x,y to match FLIKA coordinate system
                                f.write(f"{frame_idx} {y:.6f} {x:.6f}\n")

    def _create_detection_overlay(self, image_data, movie_info, min_amplitude):
        """Create image with detection overlay"""
        overlay_image = image_data.copy().astype(np.float32)

        for frame_idx, frame_info in enumerate(movie_info):
            if (frame_info and hasattr(frame_info, 'xCoord') and
                frame_info.xCoord is not None and len(frame_info.xCoord) > 0):

                # Extract coordinates
                if frame_info.xCoord.ndim == 2:
                    x_coords = frame_info.xCoord[:, 0]
                    y_coords = frame_info.yCoord[:, 0]
                else:
                    x_coords = frame_info.xCoord
                    y_coords = frame_info.yCoord

                # Get amplitudes
                if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                    if frame_info.amp.ndim == 2:
                        amplitudes = frame_info.amp[:, 0]
                    else:
                        amplitudes = frame_info.amp
                else:
                    amplitudes = np.ones(len(x_coords))

                # Mark detections in image
                for x, y, amp in zip(x_coords, y_coords, amplitudes):
                    if amp >= min_amplitude:
                        # Convert to pixel coordinates (round to nearest integer)
                        px, py = int(round(x)), int(round(y))

                        # Ensure coordinates are within image bounds
                        if (0 <= px < overlay_image.shape[2] and
                            0 <= py < overlay_image.shape[1]):

                            # Mark with bright cross pattern
                            frame = overlay_image[frame_idx]
                            max_val = np.max(frame)

                            # Horizontal line
                            for dx in range(-2, 3):
                                if 0 <= px + dx < frame.shape[1]:
                                    frame[py, px + dx] = max_val

                            # Vertical line
                            for dy in range(-2, 3):
                                if 0 <= py + dy < frame.shape[0]:
                                    frame[py + dy, px] = max_val

        return overlay_image

    def _save_detection_results(self, movie_info, final_sigma):
        """Save detection results to file"""
        try:
            filename = save_file_gui("Save Detection Results",
                                   filetypes="CSV files (*.csv);;MAT files (*.mat);;All files (*)")

            if filename:
                if filename.endswith('.csv'):
                    self._save_as_csv(movie_info, filename)
                elif filename.endswith('.mat'):
                    self._save_as_mat(movie_info, final_sigma, filename)
                else:
                    # Default to CSV
                    self._save_as_csv(movie_info, filename + '.csv')

                print(f"Detection results saved to: {filename}")

        except Exception as e:
            if FLIKA_AVAILABLE:
                g.alert(f"Failed to save results: {str(e)}")
            else:
                print(f"Failed to save results: {str(e)}")

    def _save_as_csv(self, movie_info, filename):
        """Save detection results as CSV"""
        import csv

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Detection_ID', 'X', 'Y', 'Amplitude', 'X_Std', 'Y_Std'])

            for frame_idx, frame_info in enumerate(movie_info):
                if (frame_info and hasattr(frame_info, 'xCoord') and
                    frame_info.xCoord is not None and len(frame_info.xCoord) > 0):

                    # Extract data
                    if frame_info.xCoord.ndim == 2:
                        x_coords = frame_info.xCoord[:, 0]
                        x_stds = frame_info.xCoord[:, 1] if frame_info.xCoord.shape[1] > 1 else np.zeros(len(x_coords))
                        y_coords = frame_info.yCoord[:, 0]
                        y_stds = frame_info.yCoord[:, 1] if frame_info.yCoord.shape[1] > 1 else np.zeros(len(y_coords))
                    else:
                        x_coords = frame_info.xCoord
                        y_coords = frame_info.yCoord
                        x_stds = np.zeros(len(x_coords))
                        y_stds = np.zeros(len(y_coords))

                    if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                        if frame_info.amp.ndim == 2:
                            amplitudes = frame_info.amp[:, 0]
                        else:
                            amplitudes = frame_info.amp
                    else:
                        amplitudes = np.ones(len(x_coords))

                    # Write detections
                    for i, (x, y, amp, x_std, y_std) in enumerate(zip(x_coords, y_coords, amplitudes, x_stds, y_stds)):
                        writer.writerow([frame_idx + 1, i + 1, f"{x:.6f}", f"{y:.6f}",
                                       f"{amp:.6f}", f"{x_std:.6f}", f"{y_std:.6f}"])

    def _save_as_mat(self, movie_info, final_sigma, filename):
        """Save detection results as MATLAB file"""
        try:
            import scipy.io

            # Convert to MATLAB-compatible format
            matlab_data = {
                'movieInfo': [],
                'detectionParameters': self.detection_results['parameters'],
                'finalSigma': final_sigma,
                'detectionTime': self.detection_results['detection_time']
            }

            for frame_info in movie_info:
                if (frame_info and hasattr(frame_info, 'xCoord') and
                    frame_info.xCoord is not None):
                    frame_dict = {
                        'xCoord': frame_info.xCoord,
                        'yCoord': frame_info.yCoord,
                        'amp': frame_info.amp if hasattr(frame_info, 'amp') else np.array([])
                    }
                else:
                    frame_dict = {
                        'xCoord': np.array([]),
                        'yCoord': np.array([]),
                        'amp': np.array([])
                    }
                matlab_data['movieInfo'].append(frame_dict)

            scipy.io.savemat(filename, matlab_data)

        except ImportError:
            if FLIKA_AVAILABLE:
                g.alert("scipy not available for MAT file saving")
            else:
                print("scipy not available for MAT file saving")

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")


# Create the main plugin instance (FLIKA expects this)
# Only create if both FLIKA and U-Track are available
if FLIKA_AVAILABLE and UTRACK_AVAILABLE:
    utrack_detection = UTrackDetection()
    print("✓ UTrackDetection instance created successfully")
elif FLIKA_AVAILABLE:
    # Create a stub instance that shows error messages
    class UTrackDetectionStub:
        def gui(self):
            g.alert("U-Track detection modules not available!\n\n"
                   f"Error: {utrack_import_error}\n\n"
                   "Please ensure all u-track Python files are in the plugin directory.")

        def __call__(self, *args, **kwargs):
            g.alert("U-Track detection modules not available!")
            return None

    utrack_detection = UTrackDetectionStub()
    print("✓ UTrackDetection stub created (modules not available)")
else:
    # Create a minimal stub for standalone mode
    class UTrackDetectionMinimal:
        def gui(self):
            print("FLIKA not available - cannot show GUI")

        def __call__(self, *args, **kwargs):
            print("FLIKA not available - cannot run detection")
            return None

    utrack_detection = UTrackDetectionMinimal()
    print("✓ UTrackDetection minimal stub created (FLIKA not available)")


def clear_detection_points():
    """Clear all detection points from current window"""
    if not FLIKA_AVAILABLE:
        print("FLIKA not available")
        return

    if g.currentWindow is None:
        g.alert("No current window open!")
        return

    try:
        # Clear scatter points from current window
        if hasattr(g.currentWindow, 'scatterPts'):
            g.currentWindow.scatterPts = np.array([]).reshape(0, 3)

        # Force refresh of the window display
        try:
            if hasattr(g.currentWindow, 'updateWindow'):
                g.currentWindow.updateWindow()
            elif hasattr(g.currentWindow, 'update'):
                g.currentWindow.update()
            elif hasattr(g.currentWindow, 'repaint'):
                g.currentWindow.repaint()
        except:
            pass  # If refresh fails, points should still be cleared

        print("Cleared detection points from current window")
        g.alert("Detection points cleared")

    except Exception as e:
        g.alert(f"Failed to clear points: {str(e)}")


def load_detection_points():
    """Load detection points from file into current window"""
    if not FLIKA_AVAILABLE:
        print("FLIKA not available")
        return

    if g.currentWindow is None:
        g.alert("No current window open!")
        return

    try:
        filename = open_file_gui("Load Detection Points",
                               filetypes="Text files (*.txt);;CSV files (*.csv);;All files (*)")

        if filename:
            # Load points manually to ensure correct format
            points = []

            if filename.lower().endswith('.csv'):
                import csv
                with open(filename, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header if present
                    for row in reader:
                        if len(row) >= 3:
                            try:
                                frame = float(row[0])
                                x = float(row[1])  # Keep original order for CSV
                                y = float(row[2])
                                points.append([frame, x, y])
                            except ValueError:
                                continue
            else:
                # Text file format (coordinates are swapped in saved text files)
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    frame = float(parts[0])
                                    x = float(parts[1])  # Text files saved as frame y x, so read as x
                                    y = float(parts[2])  # and y
                                    points.append([frame, x, y])
                                except ValueError:
                                    continue

            if points:
                # Convert to numpy array and add to current window
                points_array = np.array(points)

                if hasattr(g.currentWindow, 'scatterPts'):
                    if g.currentWindow.scatterPts is not None and len(g.currentWindow.scatterPts) > 0:
                        g.currentWindow.scatterPts = np.vstack([g.currentWindow.scatterPts, points_array])
                    else:
                        g.currentWindow.scatterPts = points_array
                else:
                    g.currentWindow.scatterPts = points_array

                # Force refresh of the window display
                try:
                    if hasattr(g.currentWindow, 'updateWindow'):
                        g.currentWindow.updateWindow()
                    elif hasattr(g.currentWindow, 'update'):
                        g.currentWindow.update()
                    elif hasattr(g.currentWindow, 'repaint'):
                        g.currentWindow.repaint()
                except:
                    pass  # If refresh fails, points should still be there

                print(f"Loaded {len(points)} detection points from: {filename}")
                g.alert(f"Loaded {len(points)} points from {filename}")
            else:
                g.alert("No valid points found in file")

    except Exception as e:
        g.alert(f"Failed to load points: {str(e)}")


def show_detection_summary():
    """Show summary of last detection results"""
    if not FLIKA_AVAILABLE:
        print("FLIKA not available")
        return

    if not hasattr(g, 'utrack_detection_results') or g.utrack_detection_results is None:
        g.alert("No detection results available!\nRun U-Track detection first.")
        return

    results = g.utrack_detection_results
    movie_info = results.get('movie_info', [])

    # Count detections
    total_detections = 0
    frame_counts = []
    all_amplitudes = []

    for frame_idx, frame_info in enumerate(movie_info):
        count = 0
        if (frame_info and hasattr(frame_info, 'xCoord') and
            frame_info.xCoord is not None and len(frame_info.xCoord) > 0):
            count = len(frame_info.xCoord)
            total_detections += count

            # Collect amplitudes
            if hasattr(frame_info, 'amp') and frame_info.amp is not None:
                if frame_info.amp.ndim == 2:
                    amplitudes = frame_info.amp[:, 0]
                else:
                    amplitudes = frame_info.amp
                all_amplitudes.extend(amplitudes)

        frame_counts.append(count)

    # Create summary message
    summary = f"""U-TRACK DETECTION SUMMARY

Detection Time: {results.get('detection_time', 0):.2f} seconds
Final PSF Sigma: {results.get('final_sigma', 0):.3f} pixels

Total Detections: {total_detections}
Frames Processed: {len(movie_info)}
Average Detections/Frame: {np.mean(frame_counts):.1f}
Max Detections/Frame: {np.max(frame_counts)}

Parameters Used:
  PSF Sigma: {results['parameters']['psf_sigma']:.2f}
  Method: {results['parameters']['calc_method']}
  Alpha R: {results['parameters']['alpha_r']:.4f}
  Alpha A: {results['parameters']['alpha_a']:.4f}
  Alpha D: {results['parameters']['alpha_d']:.4f}
"""

    if all_amplitudes:
        summary += f"""
Amplitude Statistics:
  Range: {np.min(all_amplitudes):.1f} - {np.max(all_amplitudes):.1f}
  Mean: {np.mean(all_amplitudes):.1f}
  Std: {np.std(all_amplitudes):.1f}"""

    print(summary)
    g.alert(summary)


def test_plugin():
    """Test if the plugin is working properly"""
    print("=== U-Track Plugin Test ===")
    print(f"FLIKA available: {FLIKA_AVAILABLE}")
    print(f"U-Track modules available: {UTRACK_AVAILABLE}")
    print(f"scikit-image available: {SKIMAGE_AVAILABLE}")

    if not UTRACK_AVAILABLE:
        print(f"U-Track import error: {utrack_import_error}")
        print("\nRequired files should be in plugin directory:")
        required_files = [
            "detect_sub_res_features_2d.py",
            "detection.py",
            "image_processing.py",
            "fitting.py",
            "utils.py",
            "parameter_classes.py"
        ]
        for f in required_files:
            file_path = os.path.join(current_dir, f)
            exists = "✓" if os.path.exists(file_path) else "✗"
            print(f"  {exists} {f}")

    if FLIKA_AVAILABLE:
        if g.currentWindow is not None:
            print(f"Current window: {g.currentWindow.name}")
            print(f"Image shape: {g.currentWindow.image.shape}")
        else:
            print("No current window - open an image to test detection")
    else:
        print("FLIKA not available - running in standalone mode")

    print("=== Test Complete ===")


def apply_detection_presets(preset_name='balanced'):
    """
    Apply predefined detection parameter presets

    Args:
        preset_name: 'strict', 'balanced', or 'lenient'
    """

    if not UTRACK_AVAILABLE:
        error_msg = "U-Track modules not available!"
        if FLIKA_AVAILABLE:
            g.alert(error_msg)
        else:
            print(error_msg)
        return None

    if not FLIKA_AVAILABLE:
        print("FLIKA not available")
        return None

    # Check if we have a working UTrackDetection instance
    if not hasattr(utrack_detection, '__call__') or isinstance(utrack_detection, (UTrackDetectionStub, UTrackDetectionMinimal)):
        error_msg = "UTrackDetection not properly initialized!"
        g.alert(error_msg)
        return None

    # Convert presets to scaled integer values for SliderLabel compatibility
    presets = {
        'strict': {
            'psf_sigma': 8,         # 0.8 pixels (8/10)
            'alpha_loc_max': 5,     # 0.005 (5/1000)
            'alpha_r': 10,          # 0.01 (10/1000)
            'alpha_a': 10,          # 0.01 (10/1000)
            'alpha_d': 1,           # 0.001 (1/1000)
            'locmax_threshold': 40  # 0.4 (40/100)
        },
        'balanced': {
            'psf_sigma': 10,        # 1.0 pixels (10/10)
            'alpha_loc_max': 10,    # 0.01 (10/1000)
            'alpha_r': 50,          # 0.05 (50/1000)
            'alpha_a': 50,          # 0.05 (50/1000)
            'alpha_d': 5,           # 0.005 (5/1000)
            'locmax_threshold': 30  # 0.3 (30/100)
        },
        'lenient': {
            'psf_sigma': 12,        # 1.2 pixels (12/10)
            'alpha_loc_max': 50,    # 0.05 (50/1000)
            'alpha_r': 100,         # 0.1 (100/1000)
            'alpha_a': 100,         # 0.1 (100/1000)
            'alpha_d': 20,          # 0.02 (20/1000)
            'locmax_threshold': 20  # 0.2 (20/100)
        }
    }

    if preset_name in presets:
        params = presets[preset_name]
        # Apply the preset by calling the detection with these parameters
        print(f"Applying {preset_name} detection preset...")
        return utrack_detection(**params)
    else:
        error_msg = f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
        if FLIKA_AVAILABLE:
            g.alert(error_msg)
        else:
            print(error_msg)


def batch_detection_analysis():
    """
    Run detection analysis on all open windows
    """
    if not FLIKA_AVAILABLE:
        print("FLIKA not available")
        return

    if not UTRACK_AVAILABLE:
        g.alert("U-Track modules not available!")
        return

    # Check if we have a working UTrackDetection instance
    if not hasattr(utrack_detection, '__call__') or isinstance(utrack_detection, (UTrackDetectionStub, UTrackDetectionMinimal)):
        g.alert("UTrackDetection not properly initialized!")
        return

    windows = [w for w in g.windows if w.closed == False]

    if not windows:
        g.alert("No open windows found")
        return

    print(f"Running batch detection on {len(windows)} windows...")

    results = []
    for i, window in enumerate(windows):
        print(f"Processing window {i+1}/{len(windows)}: {window.name}")

        # Set as current window
        window.setAsCurrentWindow()

        # Run detection with balanced preset
        result = apply_detection_presets('balanced')

        if result is not None:
            detection_count = 0
            if hasattr(utrack_detection, 'detection_results') and utrack_detection.detection_results:
                movie_info = utrack_detection.detection_results.get('movie_info', [])
                for frame_info in movie_info:
                    if (frame_info and hasattr(frame_info, 'xCoord') and
                        frame_info.xCoord is not None and len(frame_info.xCoord) > 0):
                        detection_count += len(frame_info.xCoord)

            results.append({
                'window_name': window.name,
                'detection_count': detection_count
            })

    # Summary
    print(f"\nBatch analysis complete!")
    for result in results:
        print(f"  {result['window_name']}: {result['detection_count']} detections")


# Set menu paths for the functions (FLIKA will use these)
test_plugin.menu_path = 'Plugins>Particle Detection>Test Plugin'
apply_detection_presets.menu_path = 'Plugins>Particle Detection>Apply Preset'
batch_detection_analysis.menu_path = 'Plugins>Particle Detection>Batch Analysis'
load_detection_points.menu_path = 'Plugins>Particle Detection>Load Points'
clear_detection_points.menu_path = 'Plugins>Particle Detection>Clear Points'
show_detection_summary.menu_path = 'Plugins>Particle Detection>Show Summary'


# Plugin information for FLIKA
__version__ = '1.0.1'
__author__ = 'Your Name'
__description__ = 'U-Track subresolution particle detection for FLIKA (Points version)'

# Plugin status message
if UTRACK_AVAILABLE and FLIKA_AVAILABLE:
    print("✓ U-Track Particle Detection Plugin loaded successfully! (Points version)")
    print("Available functions:")
    print("  - U-Track Detection (main detection process)")
    print("  - Test Plugin (check plugin status)")
    print("  - Apply Detection Presets (strict/balanced/lenient)")
    print("  - Batch Detection Analysis (analyze all open windows)")
    print("  - Load Detection Points (load points from file)")
    print("  - Clear Detection Points (remove all points from window)")
    print("  - Show Detection Summary (display last results)")
elif FLIKA_AVAILABLE:
    print("⚠ U-Track Particle Detection Plugin loaded with missing dependencies")
    print("  - Plugin structure loaded but detection modules unavailable")
    print("  - Use 'Test Plugin' menu item to check status")
    if utrack_import_error:
        print(f"  - Error: {utrack_import_error}")
else:
    print("⚠ U-Track Plugin running in standalone mode (FLIKA not available)")

print(f"\nPlugin directory: {current_dir}")
print("Restart FLIKA after adding missing files if needed.")
