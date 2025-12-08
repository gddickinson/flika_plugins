"""
ThunderSTORM for FLIKA
======================

A comprehensive FLIKA plugin implementing thunderSTORM functionality for
Single Molecule Localization Microscopy (SMLM) data analysis.

This plugin provides:
- Complete SMLM analysis pipeline (filtering, detection, fitting)
- Multiple PSF fitting methods (LSQ, WLSQ, MLE, Radial Symmetry)
- Post-processing (drift correction, merging, filtering)
- Super-resolution rendering with multiple methods
- Simulation tools for testing and validation
- Performance evaluation against ground truth

Author: George
Reference: Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., & Hagen, G. M. (2014).
          ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis
          and super-resolution imaging. Bioinformatics, 30(16), 2389-2390.
"""

__version__ = '1.0.0'
__author__ = 'George'

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
from flika.process.file_ import save_file_gui, open_file_gui
from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QLineEdit, QGroupBox, QFileDialog,
                            QProgressBar, QTextEdit, QTabWidget, QMessageBox, QInputDialog)
from qtpy.QtCore import Qt, Signal
import numpy as np
import os
from pathlib import Path


# Create a LineEdit wrapper class that works with BaseProcess
class LineEdit(QLineEdit):
    """Wrapper for QLineEdit that adds setValue() and value() methods for BaseProcess compatibility"""

    def __init__(self, parent=None):
        QLineEdit.__init__(self, parent)

    def setValue(self, value):
        """Set the text value"""
        self.setText(str(value))

    def value(self):
        """Get the text value"""
        return self.text()

# Import thunderSTORM modules
# We'll assume the thunderstorm_python package is in the plugin directory
import sys
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

try:
    from thunderstorm_python import ThunderSTORM, create_default_pipeline, quick_analysis
    from thunderstorm_python import filters, detection, fitting, postprocessing, visualization, simulation, utils
    from thunderstorm_python.simulation import SMLMSimulator, PerformanceEvaluator, create_test_pattern
    THUNDERSTORM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import thunderstorm_python: {e}")
    print("Please ensure the thunderstorm_python package is in the plugin directory")
    THUNDERSTORM_AVAILABLE = False


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

class ThunderSTORM_RunAnalysis(BaseProcess):
    """
    Main ThunderSTORM analysis pipeline.

    Provides complete SMLM analysis workflow:
    1. Image filtering (wavelet, gaussian, DoG, etc.)
    2. Molecule detection (local maximum, non-maximum suppression)
    3. PSF fitting (LSQ, WLSQ, MLE, Radial Symmetry)
    4. Output localization table and super-resolution image
    """

    def __init__(self):
        super().__init__()

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available. Please check installation.")
            self.close()
            return

        self.pipeline = None
        self.localizations = None

    def get_init_settings_dict(self):
        return {
            # Filtering
            'filter_type': 'wavelet',
            'wavelet_scale': 2,
            'wavelet_order': 3,
            'gaussian_sigma': 1.6,

            # Detection
            'detector_type': 'local_maximum',
            'connectivity': '8-neighbourhood',
            'threshold_expr': 'std(Wave.F1)',

            # Fitting
            'fitter_type': 'gaussian_lsq',
            'fit_radius': 3,
            'initial_sigma': 1.3,
            'integrated': True,
            'elliptical': False,

            # Camera
            'pixel_size': 100.0,
            'photons_per_adu': 1.0,
            'baseline': 100.0,
            'em_gain': 1.0,

            # Rendering
            'render_pixel_size': 10.0,
            'renderer_type': 'gaussian'
        }

    def gui(self):
        """Create the GUI for the analysis pipeline"""
        self.gui_reset()

        # Create tabs for organized interface
        tabs = QTabWidget()

        # ===== Tab 1: Filtering =====
        filter_tab = QWidget()
        filter_layout = QVBoxLayout()

        filter_group = QGroupBox("Image Filtering")
        filter_group_layout = QVBoxLayout()

        filter_type = ComboBox()
        filter_type.addItem('wavelet')
        filter_type.addItem('gaussian')
        filter_type.addItem('dog')
        filter_type.addItem('lowered_gaussian')
        filter_type.addItem('median')
        filter_type.addItem('none')

        wavelet_scale = SliderLabel(0)
        wavelet_scale.setRange(1, 5)

        wavelet_order = SliderLabel(0)
        wavelet_order.setRange(1, 5)

        gaussian_sigma = SliderLabel(3)
        gaussian_sigma.setRange(0.5, 5.0)

        self.items.append({'name': 'filter_type', 'string': 'Filter Type', 'object': filter_type})
        self.items.append({'name': 'wavelet_scale', 'string': 'Wavelet Scale', 'object': wavelet_scale})
        self.items.append({'name': 'wavelet_order', 'string': 'Wavelet Order', 'object': wavelet_order})
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Sigma', 'object': gaussian_sigma})

        # ===== Tab 2: Detection =====
        detector_type = ComboBox()
        detector_type.addItem('local_maximum')
        detector_type.addItem('non_maximum_suppression')
        detector_type.addItem('centroid')

        connectivity = ComboBox()
        connectivity.addItem('8-neighbourhood')
        connectivity.addItem('4-neighbourhood')

        threshold_expr = LineEdit()
        threshold_expr.setPlaceholderText('std(Wave.F1)')  # Show default as placeholder
        threshold_expr.setText('std(Wave.F1)')  # Set initial value directly

        self.items.append({'name': 'detector_type', 'string': 'Detector Type', 'object': detector_type})
        self.items.append({'name': 'connectivity', 'string': 'Connectivity', 'object': connectivity})
        self.items.append({'name': 'threshold_expr', 'string': 'Threshold Expression', 'object': threshold_expr})

        # ===== Tab 3: Fitting =====
        fitter_type = ComboBox()
        fitter_type.addItem('gaussian_lsq')
        fitter_type.addItem('gaussian_wlsq')
        fitter_type.addItem('gaussian_mle')
        fitter_type.addItem('radial_symmetry')
        fitter_type.addItem('centroid')

        fit_radius = SliderLabel(0)
        fit_radius.setRange(2, 8)

        initial_sigma = SliderLabel(3)
        initial_sigma.setRange(0.5, 3.0)

        integrated = CheckBox()
        elliptical = CheckBox()

        self.items.append({'name': 'fitter_type', 'string': 'Fitter Type', 'object': fitter_type})
        self.items.append({'name': 'fit_radius', 'string': 'Fit Radius (pixels)', 'object': fit_radius})
        self.items.append({'name': 'initial_sigma', 'string': 'Initial Sigma', 'object': initial_sigma})
        self.items.append({'name': 'integrated', 'string': 'Integrated Gaussian', 'object': integrated})
        self.items.append({'name': 'elliptical', 'string': 'Elliptical Gaussian', 'object': elliptical})

        # ===== Tab 4: Camera =====
        pixel_size = SliderLabel(3)
        pixel_size.setRange(10.0, 200.0)

        photons_per_adu = SliderLabel(3)
        photons_per_adu.setRange(0.1, 10.0)

        baseline = SliderLabel(3)
        baseline.setRange(0.0, 500.0)

        em_gain = SliderLabel(3)
        em_gain.setRange(1.0, 300.0)

        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)', 'object': pixel_size})
        self.items.append({'name': 'photons_per_adu', 'string': 'Photons per ADU', 'object': photons_per_adu})
        self.items.append({'name': 'baseline', 'string': 'Baseline (ADU)', 'object': baseline})
        self.items.append({'name': 'em_gain', 'string': 'EM Gain', 'object': em_gain})

        # ===== Tab 5: Rendering =====
        render_pixel_size = SliderLabel(3)
        render_pixel_size.setRange(1.0, 50.0)

        renderer_type = ComboBox()
        renderer_type.addItem('gaussian')
        renderer_type.addItem('histogram')
        renderer_type.addItem('ash')
        renderer_type.addItem('scatter')

        self.items.append({'name': 'render_pixel_size', 'string': 'Render Pixel Size (nm)', 'object': render_pixel_size})
        self.items.append({'name': 'renderer_type', 'string': 'Renderer Type', 'object': renderer_type})

        super().gui()

    def __call__(self, filter_type='wavelet', wavelet_scale=2, wavelet_order=3, gaussian_sigma=1.6,
                 detector_type='local_maximum', connectivity='8-neighbourhood', threshold_expr='std(Wave.F1)',
                 fitter_type='gaussian_lsq', fit_radius=3, initial_sigma=1.3, integrated=True, elliptical=False,
                 pixel_size=100.0, photons_per_adu=1.0, baseline=100.0, em_gain=1.0,
                 render_pixel_size=10.0, renderer_type='gaussian', keepSourceWindow=False):
        """Run the complete analysis pipeline"""

        self.start(keepSourceWindow)

        try:
            g.m.statusBar().showMessage("Running ThunderSTORM analysis...")

            # Get image stack
            image_stack = self.tif
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]

            print(f"Processing {image_stack.shape[0]} frames...")

            # Validate threshold expression - use default if empty
            if not threshold_expr or threshold_expr.strip() == '':
                threshold_expr = 'std(Wave.F1)'
                print(f"Using default threshold expression: {threshold_expr}")

            # Create pipeline using factory pattern
            filter_params = {}
            if filter_type == 'wavelet':
                filter_params = {'scale': int(wavelet_scale), 'order': int(wavelet_order)}
            elif filter_type in ['gaussian', 'lowered_gaussian']:
                filter_params = {'sigma': float(gaussian_sigma)}
            elif filter_type == 'dog':
                filter_params = {'sigma1': float(gaussian_sigma), 'sigma2': float(gaussian_sigma)*1.6}
            elif filter_type == 'median':
                filter_params = {'size': 3}

            # Detector parameters
            conn_value = 8 if connectivity == '8-neighbourhood' else 4
            detector_params = {}
            if detector_type == 'local_maximum':
                detector_params = {'connectivity': connectivity, 'min_distance': 1}
            elif detector_type == 'non_maximum_suppression':
                detector_params = {'connectivity': conn_value}
            elif detector_type == 'centroid':
                detector_params = {'connectivity': conn_value}

            # Fitter parameters
            fitter_params = {}
            if fitter_type in ['gaussian_lsq', 'gaussian_wlsq']:
                fitter_params = {
                    'initial_sigma': float(initial_sigma),
                    'integrated': bool(integrated),
                    'elliptical': bool(elliptical)
                }
            elif fitter_type == 'gaussian_mle':
                fitter_params = {'initial_sigma': float(initial_sigma)}

            # Create pipeline
            self.pipeline = ThunderSTORM(
                filter_type=filter_type,
                filter_params=filter_params,
                detector_type=detector_type,
                detector_params=detector_params,
                fitter_type=fitter_type,
                fitter_params=fitter_params,
                threshold_expression=threshold_expr,
                pixel_size=float(pixel_size),
                photons_per_adu=float(photons_per_adu),
                baseline=float(baseline),
                em_gain=float(em_gain)
            )

            # Process all frames (pass fit_radius to the analysis)
            self.localizations = self.pipeline.analyze_stack(
                image_stack,
                fit_radius=int(fit_radius),
                show_progress=True
            )

            n_locs = len(self.localizations['x'])
            print(f"Analysis complete: {n_locs} molecules detected")

            # Render super-resolution image
            if renderer_type == 'gaussian':
                renderer = visualization.GaussianRenderer(sigma=1.5)
            elif renderer_type == 'histogram':
                renderer = visualization.HistogramRenderer()
            elif renderer_type == 'ash':
                renderer = visualization.AverageShiftedHistogram()
            else:
                renderer = visualization.ScatterRenderer()

            sr_image = renderer.render(self.localizations, pixel_size=float(render_pixel_size))

            g.m.statusBar().showMessage(f"Analysis complete: {n_locs} molecules detected", 3000)

            # Save localizations option
            save_locs = QMessageBox.question(
                None, "Save Localizations",
                "Save localization results?",
                QMessageBox.Yes | QMessageBox.No
            )

            if save_locs == QMessageBox.Yes:
                filename = save_file_gui("Save Localizations", filetypes='*.csv')
                if filename:
                    utils.save_localizations_csv(self.localizations, filename)
                    g.m.statusBar().showMessage(f"Localizations saved to {filename}", 3000)

            self.newtif = sr_image
            self.newname = f"{self.oldname}_ThunderSTORM_SR"
            return self.end()

        except Exception as e:
            g.alert(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Post-Processing
# ============================================================================

class ThunderSTORM_PostProcessing(BaseProcess):
    """
    Post-processing for localization data.

    Operations include:
    - Quality filtering (uncertainty, intensity)
    - Duplicate merging
    - Z-filtering (for 3D data)
    """

    def __init__(self):
        super().__init__()
        self.localizations = None

    def get_init_settings_dict(self):
        return {
            'max_uncertainty': 50.0,
            'min_intensity': 100.0,
            'merge_radius': 50.0,
            'merge_frames': 0
        }

    def gui(self):
        """Create the GUI for post-processing"""
        self.gui_reset()

        max_uncertainty = SliderLabel(3)
        max_uncertainty.setRange(10.0, 200.0)

        min_intensity = SliderLabel(3)
        min_intensity.setRange(0.0, 10000.0)

        merge_radius = SliderLabel(3)
        merge_radius.setRange(0.0, 200.0)

        merge_frames = SliderLabel(0)
        merge_frames.setRange(0, 10)

        self.items.append({'name': 'max_uncertainty', 'string': 'Max Uncertainty (nm)', 'object': max_uncertainty})
        self.items.append({'name': 'min_intensity', 'string': 'Min Intensity (photons)', 'object': min_intensity})
        self.items.append({'name': 'merge_radius', 'string': 'Merge Radius (nm)', 'object': merge_radius})
        self.items.append({'name': 'merge_frames', 'string': 'Merge Frames', 'object': merge_frames})

        super().gui()

    def __call__(self, max_uncertainty=50.0, min_intensity=100.0, merge_radius=50.0,
                 merge_frames=0, keepSourceWindow=False):
        """Run post-processing - this operates on stored localizations, not an image"""

        if self.localizations is None:
            # Try to load from CSV
            filename = open_file_gui("Load Localizations CSV", filetypes='*.csv')
            if filename:
                try:
                    self.localizations = utils.load_localizations_csv(filename)
                    n = len(self.localizations['x'])
                    g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
                except Exception as e:
                    g.alert(f"Error loading file: {str(e)}")
                    return None
            else:
                g.alert("No localizations loaded. Load a CSV file first.")
                return None

        try:
            g.m.statusBar().showMessage("Running post-processing...")

            # Quality filtering using LocalizationFilter class
            loc_filter = postprocessing.LocalizationFilter(
                max_uncertainty=float(max_uncertainty),
                min_intensity=float(min_intensity)
            )
            filtered = loc_filter.filter(self.localizations)

            n_before = len(self.localizations['x'])
            n_after = len(filtered['x'])
            print(f"Filtering: {n_before} -> {n_after} localizations")

            # Merging using MolecularMerger class
            if float(merge_radius) > 0:
                merger = postprocessing.MolecularMerger(
                    max_distance=float(merge_radius),
                    max_frame_gap=int(merge_frames)
                )
                merged = merger.merge(filtered)
                n_merged = len(merged['x'])
                print(f"Merging: {n_after} -> {n_merged} localizations")
                self.localizations = merged
            else:
                self.localizations = filtered

            g.m.statusBar().showMessage(f"Post-processing complete: {len(self.localizations['x'])} localizations", 3000)

            # Offer to save
            save_locs = QMessageBox.question(
                None, "Save Processed Localizations",
                "Save processed localization results?",
                QMessageBox.Yes | QMessageBox.No
            )

            if save_locs == QMessageBox.Yes:
                filename = save_file_gui("Save Processed Localizations", filetypes='*.csv')
                if filename:
                    utils.save_localizations_csv(self.localizations, filename)
                    g.m.statusBar().showMessage(f"Localizations saved to {filename}", 3000)

            return None  # Post-processing doesn't create an image

        except Exception as e:
            g.alert(f"Error in post-processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Drift Correction
# ============================================================================

class ThunderSTORM_DriftCorrection(BaseProcess):
    """
    Drift correction using cross-correlation.

    Corrects sample drift between frames using fiducial markers
    or cross-correlation of localization density.
    """

    def __init__(self):
        super().__init__()
        self.localizations = None

    def get_init_settings_dict(self):
        return {
            'magnification': 5,
            'n_bins': 5,
            'smoothing_bandwidth': 0.25
        }

    def gui(self):
        """Create the GUI for drift correction"""
        self.gui_reset()

        magnification = SliderLabel(0)
        magnification.setRange(1, 20)

        n_bins = SliderLabel(0)
        n_bins.setRange(2, 20)

        smoothing_bandwidth = SliderLabel(3)
        smoothing_bandwidth.setRange(0.1, 1.0)

        self.items.append({'name': 'magnification', 'string': 'Magnification', 'object': magnification})
        self.items.append({'name': 'n_bins', 'string': 'Number of Bins', 'object': n_bins})
        self.items.append({'name': 'smoothing_bandwidth', 'string': 'Smoothing Bandwidth', 'object': smoothing_bandwidth})

        super().gui()

    def __call__(self, magnification=5, n_bins=5, smoothing_bandwidth=0.25, keepSourceWindow=False):
        """Run drift correction - operates on stored localizations"""

        if self.localizations is None:
            # Try to load from CSV
            filename = open_file_gui("Load Localizations CSV", filetypes='*.csv')
            if filename:
                try:
                    self.localizations = utils.load_localizations_csv(filename)
                    n = len(self.localizations['x'])
                    g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
                except Exception as e:
                    g.alert(f"Error loading file: {str(e)}")
                    return None
            else:
                g.alert("No localizations loaded. Load a CSV file first.")
                return None

        try:
            g.m.statusBar().showMessage("Running drift correction...")

            # Create DriftCorrector with smoothing parameter
            drift_corrector = postprocessing.DriftCorrector(
                method='cross_correlation',
                smoothing=float(smoothing_bandwidth)
            )

            # Compute drift
            frames = np.unique(self.localizations['frame'])
            drift_x, drift_y = drift_corrector.compute_drift_xcorr(
                self.localizations,
                frames,
                pixel_size=100.0 / int(magnification),  # Adjust pixel size by magnification
                segment_frames=max(50, len(frames) // int(n_bins))  # Use n_bins to determine segment size
            )

            # Apply drift correction
            self.localizations = drift_corrector.apply_drift_correction(self.localizations)

            max_drift = np.sqrt(drift_x**2 + drift_y**2).max()
            print(f"Drift correction complete. Max drift: {max_drift:.2f} nm")
            g.m.statusBar().showMessage(f"Drift correction complete. Max drift: {max_drift:.2f} nm", 3000)

            # Offer to save
            save_locs = QMessageBox.question(
                None, "Save Corrected Localizations",
                "Save drift-corrected localization results?",
                QMessageBox.Yes | QMessageBox.No
            )

            if save_locs == QMessageBox.Yes:
                filename = save_file_gui("Save Corrected Localizations", filetypes='*.csv')
                if filename:
                    utils.save_localizations_csv(self.localizations, filename)
                    g.m.statusBar().showMessage(f"Localizations saved to {filename}", 3000)

            return None  # Drift correction doesn't create an image

        except Exception as e:
            g.alert(f"Error in drift correction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Rendering
# ============================================================================

class ThunderSTORM_Rendering(BaseProcess):
    """
    Render super-resolution images from localization data.

    Supports multiple rendering methods:
    - Gaussian rendering
    - Histogram
    - Average Shifted Histogram (ASH)
    - Scatter plot
    """

    def __init__(self):
        super().__init__()
        self.localizations = None

    def get_init_settings_dict(self):
        return {
            'pixel_size': 10.0,
            'renderer': 'gaussian',
            'gaussian_sigma': 1.5,
            'jitter': True,
            'n_averages': 10,
            'ash_shifts': 2
        }

    def gui(self):
        """Create the GUI for rendering"""
        self.gui_reset()

        pixel_size = SliderLabel(3)
        pixel_size.setRange(1.0, 50.0)

        renderer = ComboBox()
        renderer.addItem('gaussian')
        renderer.addItem('histogram')
        renderer.addItem('ash')
        renderer.addItem('scatter')

        gaussian_sigma = SliderLabel(3)
        gaussian_sigma.setRange(0.5, 5.0)

        jitter = CheckBox()

        n_averages = SliderLabel(0)
        n_averages.setRange(1, 50)

        ash_shifts = SliderLabel(0)
        ash_shifts.setRange(1, 10)

        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)', 'object': pixel_size})
        self.items.append({'name': 'renderer', 'string': 'Renderer Type', 'object': renderer})
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Sigma', 'object': gaussian_sigma})
        self.items.append({'name': 'jitter', 'string': 'Jittering', 'object': jitter})
        self.items.append({'name': 'n_averages', 'string': 'Number of Averages', 'object': n_averages})
        self.items.append({'name': 'ash_shifts', 'string': 'ASH Shifts', 'object': ash_shifts})

        super().gui()

    def __call__(self, pixel_size=10.0, renderer='gaussian', gaussian_sigma=1.5,
                 jitter=True, n_averages=10, ash_shifts=2, keepSourceWindow=False):
        """Render super-resolution image - operates on stored localizations"""

        if self.localizations is None:
            # Try to load from CSV
            filename = open_file_gui("Load Localizations CSV", filetypes='*.csv')
            if filename:
                try:
                    self.localizations = utils.load_localizations_csv(filename)
                    n = len(self.localizations['x'])
                    g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
                except Exception as e:
                    g.alert(f"Error loading file: {str(e)}")
                    return None
            else:
                g.alert("No localizations loaded. Load a CSV file first.")
                return None

        try:
            g.m.statusBar().showMessage(f"Rendering with {renderer}...")

            # Create renderer
            if renderer == 'gaussian':
                render_obj = visualization.GaussianRenderer(sigma=float(gaussian_sigma))
            elif renderer == 'histogram':
                render_obj = visualization.HistogramRenderer(
                    jittering=bool(jitter),
                    n_averages=int(n_averages)
                )
            elif renderer == 'ash':
                render_obj = visualization.AverageShiftedHistogram(n_shifts=int(ash_shifts))
            else:  # scatter
                render_obj = visualization.ScatterRenderer()

            # Render
            sr_image = render_obj.render(self.localizations, pixel_size=float(pixel_size))

            g.m.statusBar().showMessage("Rendering complete", 3000)

            # Since this is BaseProcess, we need to return through the normal pattern
            # Create a dummy window from the current window, then set newtif
            if g.win is not None:
                self.start(False)
                self.newtif = sr_image
                self.newname = f"{self.oldname}_SR_{renderer}"
                return self.end()
            else:
                # No source window, create a new window directly
                result_window = Window(sr_image, name=f"SR_{renderer}")
                return result_window

        except Exception as e:
            g.alert(f"Error in rendering: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Quick Analysis
# ============================================================================

class ThunderSTORM_QuickAnalysis(BaseProcess):
    """Quick analysis with default parameters"""

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {}

    def gui(self):
        """Create minimal GUI for quick analysis"""
        self.gui_reset()
        # No parameters needed - uses defaults
        super().gui()

    def __call__(self, keepSourceWindow=False):
        """Run quick analysis with default parameters"""

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available.")
            return None

        self.start(keepSourceWindow)

        try:
            g.m.statusBar().showMessage("Running quick analysis...")

            # Get image stack
            image_stack = self.tif
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]

            print(f"Quick analysis starting on {image_stack.shape[0]} frames...")

            # Run quick analysis
            localizations, sr_image, pipeline = quick_analysis(image_stack)

            n_locs = len(localizations['x'])
            print(f"Quick analysis complete: {n_locs} molecules detected")
            g.m.statusBar().showMessage(f"Quick analysis complete: {n_locs} molecules", 3000)

            self.newtif = sr_image
            self.newname = f"{self.oldname}_quick_SR"
            return self.end()

        except Exception as e:
            error_msg = f"Error in quick analysis: {str(e)}"
            g.alert(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            g.m.statusBar().showMessage("Analysis failed", 3000)
            return None


# ============================================================================
# Simulation
# ============================================================================

class ThunderSTORM_SimulateData(BaseProcess):
    """Generate simulated SMLM data for testing"""

    def __init__(self):
        super().__init__()

    def get_init_settings_dict(self):
        return {
            'image_size': 256,
            'pixel_size': 100.0,
            'psf_sigma': 150.0,
            'photons_per_molecule': 1000,
            'background_photons': 20,
            'n_frames': 500,
            'pattern': 'siemens_star'
        }

    def gui(self):
        """Create GUI for simulation parameters"""
        self.gui_reset()

        image_size = SliderLabel(0)
        image_size.setRange(128, 1024)

        pixel_size = SliderLabel(3)
        pixel_size.setRange(50.0, 200.0)

        psf_sigma = SliderLabel(3)
        psf_sigma.setRange(100.0, 300.0)

        photons_per_molecule = SliderLabel(0)
        photons_per_molecule.setRange(100, 5000)

        background_photons = SliderLabel(0)
        background_photons.setRange(0, 100)

        n_frames = SliderLabel(0)
        n_frames.setRange(10, 2000)

        pattern = ComboBox()
        pattern.addItem('siemens_star')
        pattern.addItem('grid')
        pattern.addItem('circle')
        pattern.addItem('random')

        self.items.append({'name': 'image_size', 'string': 'Image Size (pixels)', 'object': image_size})
        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)', 'object': pixel_size})
        self.items.append({'name': 'psf_sigma', 'string': 'PSF Sigma (nm)', 'object': psf_sigma})
        self.items.append({'name': 'photons_per_molecule', 'string': 'Photons per Molecule', 'object': photons_per_molecule})
        self.items.append({'name': 'background_photons', 'string': 'Background Photons', 'object': background_photons})
        self.items.append({'name': 'n_frames', 'string': 'Number of Frames', 'object': n_frames})
        self.items.append({'name': 'pattern', 'string': 'Pattern Type', 'object': pattern})

        super().gui()

    def __call__(self, image_size=256, pixel_size=100.0, psf_sigma=150.0,
                 photons_per_molecule=1000, background_photons=20,
                 n_frames=500, pattern='siemens_star', keepSourceWindow=False):
        """Generate simulated SMLM data"""

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available.")
            return None

        try:
            # Create simulator
            simulator = SMLMSimulator(
                image_size=(int(image_size), int(image_size)),
                pixel_size=float(pixel_size),
                psf_sigma=float(psf_sigma),
                photons_per_molecule=int(photons_per_molecule),
                background_photons=int(background_photons)
            )

            # Create pattern
            g.m.statusBar().showMessage(f"Generating {pattern} pattern...")
            mask = create_test_pattern(pattern, size=int(image_size))

            # Generate movie
            g.m.statusBar().showMessage(f"Simulating {n_frames} frames...")
            movie, ground_truth = simulator.generate_movie(
                n_frames=int(n_frames),
                mask=mask,
                blinking=True
            )

            # Save ground truth
            save_gt = QMessageBox.question(
                None, "Save Ground Truth",
                "Save ground truth localizations?",
                QMessageBox.Yes | QMessageBox.No
            )

            if save_gt == QMessageBox.Yes:
                filename = save_file_gui("Save Ground Truth", filetypes='*.csv')
                if filename:
                    # Combine all frames
                    gt_combined = {
                        'x': np.concatenate([gt['x'] for gt in ground_truth]),
                        'y': np.concatenate([gt['y'] for gt in ground_truth]),
                        'frame': np.concatenate([
                            np.full(len(gt['x']), i) for i, gt in enumerate(ground_truth)
                        ])
                    }
                    utils.save_localizations_csv(gt_combined, filename)
                    g.m.statusBar().showMessage(f"Ground truth saved to {filename}", 3000)

            g.m.statusBar().showMessage(f"Simulation complete: {n_frames} frames", 3000)

            # Return as a window - since we're not processing an existing window, create directly
            result_window = Window(movie, name=f"Simulated_{pattern}")
            return result_window

        except Exception as e:
            g.alert(f"Error in simulation: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Module-level Instances (FLIKA BaseProcess pattern)
# ============================================================================

# Create instances at module level - FLIKA will call these directly
# When menu item is clicked, FLIKA calls instance.gui() which shows the dialog
thunderstorm_run_analysis = ThunderSTORM_RunAnalysis()
thunderstorm_post_processing = ThunderSTORM_PostProcessing()
thunderstorm_drift_correction = ThunderSTORM_DriftCorrection()
thunderstorm_rendering = ThunderSTORM_Rendering()
thunderstorm_quick_analysis = ThunderSTORM_QuickAnalysis()
thunderstorm_simulate_data = ThunderSTORM_SimulateData()


# ============================================================================
# Plugin Initialization
# ============================================================================

def initialize_plugin():
    """Initialize the ThunderSTORM plugin"""
    if THUNDERSTORM_AVAILABLE:
        print("="*60)
        print("ThunderSTORM for FLIKA v" + __version__)
        print("Loaded successfully!")
        print("="*60)
        print("\nAvailable tools:")
        print("  - Run Analysis: Complete SMLM analysis pipeline")
        print("  - Quick Analysis: Fast analysis with default settings")
        print("  - Post-Processing: Quality filtering and merging")
        print("  - Drift Correction: Cross-correlation based drift correction")
        print("  - Rendering: Multiple super-resolution rendering methods")
        print("  - Simulate Data: Generate test datasets")
        print("\nAccess via: Plugins > ThunderSTORM")
        print("="*60)
    else:
        print("="*60)
        print("WARNING: ThunderSTORM modules not available")
        print("Please ensure thunderstorm_python package is installed")
        print("="*60)

# Initialize on import
initialize_plugin()
