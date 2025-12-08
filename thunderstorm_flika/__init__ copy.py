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
                            QProgressBar, QTextEdit, QTabWidget, QMessageBox)
from qtpy.QtCore import Qt, Signal
import numpy as np
import os
from pathlib import Path

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

    def setupGUI(self):
        super().setupGUI()

        # Create tabs for organized interface
        tabs = QTabWidget()

        # ===== Tab 1: Filtering =====
        filter_tab = QWidget()
        filter_layout = QVBoxLayout()

        filter_group = QGroupBox("Image Filtering")
        filter_group_layout = QVBoxLayout()

        self.filter_type = ComboBox()
        self.filter_type.addItem('wavelet')
        self.filter_type.addItem('gaussian')
        self.filter_type.addItem('dog')
        self.filter_type.addItem('lowered_gaussian')
        self.filter_type.addItem('median')
        self.filter_type.addItem('none')
        filter_group_layout.addWidget(QLabel('Filter Type:'))
        filter_group_layout.addWidget(self.filter_type)

        self.wavelet_scale = SliderLabel(0)
        self.wavelet_scale.setRange(1, 5)
        filter_group_layout.addWidget(QLabel('Wavelet Scale:'))
        filter_group_layout.addWidget(self.wavelet_scale)

        self.wavelet_order = SliderLabel(0)
        self.wavelet_order.setRange(1, 5)
        filter_group_layout.addWidget(QLabel('Wavelet Order:'))
        filter_group_layout.addWidget(self.wavelet_order)

        self.gaussian_sigma = SliderLabel(3)
        self.gaussian_sigma.setRange(0.5, 5.0)
        filter_group_layout.addWidget(QLabel('Gaussian Sigma:'))
        filter_group_layout.addWidget(self.gaussian_sigma)

        filter_group.setLayout(filter_group_layout)
        filter_layout.addWidget(filter_group)
        filter_layout.addStretch()
        filter_tab.setLayout(filter_layout)

        # ===== Tab 2: Detection =====
        detect_tab = QWidget()
        detect_layout = QVBoxLayout()

        detect_group = QGroupBox("Molecule Detection")
        detect_group_layout = QVBoxLayout()

        self.detector_type = ComboBox()
        self.detector_type.addItem('local_maximum')
        self.detector_type.addItem('non_maximum_suppression')
        self.detector_type.addItem('centroid')
        detect_group_layout.addWidget(QLabel('Detector Type:'))
        detect_group_layout.addWidget(self.detector_type)

        self.connectivity = ComboBox()
        self.connectivity.addItem('8-neighbourhood')
        self.connectivity.addItem('4-neighbourhood')
        detect_group_layout.addWidget(QLabel('Connectivity:'))
        detect_group_layout.addWidget(self.connectivity)

        self.threshold_expr = QLineEdit()
        detect_group_layout.addWidget(QLabel('Threshold Expression:'))
        detect_group_layout.addWidget(self.threshold_expr)
        detect_group_layout.addWidget(QLabel("Examples: 'std(Wave.F1)', '2*std(Wave.F1)', '100'"))

        detect_group.setLayout(detect_group_layout)
        detect_layout.addWidget(detect_group)
        detect_layout.addStretch()
        detect_tab.setLayout(detect_layout)

        # ===== Tab 3: Fitting =====
        fit_tab = QWidget()
        fit_layout = QVBoxLayout()

        fit_group = QGroupBox("PSF Fitting")
        fit_group_layout = QVBoxLayout()

        self.fitter_type = ComboBox()
        self.fitter_type.addItem('gaussian_lsq')
        self.fitter_type.addItem('gaussian_wlsq')
        self.fitter_type.addItem('gaussian_mle')
        self.fitter_type.addItem('radial_symmetry')
        self.fitter_type.addItem('centroid')
        fit_group_layout.addWidget(QLabel('Fitter Type:'))
        fit_group_layout.addWidget(self.fitter_type)

        self.fit_radius = SliderLabel(0)
        self.fit_radius.setRange(2, 8)
        fit_group_layout.addWidget(QLabel('Fit Radius (pixels):'))
        fit_group_layout.addWidget(self.fit_radius)

        self.initial_sigma = SliderLabel(3)
        self.initial_sigma.setRange(0.5, 3.0)
        fit_group_layout.addWidget(QLabel('Initial Sigma:'))
        fit_group_layout.addWidget(self.initial_sigma)

        self.integrated = CheckBox()
        fit_group_layout.addWidget(QLabel('Integrated Gaussian:'))
        fit_group_layout.addWidget(self.integrated)

        self.elliptical = CheckBox()
        fit_group_layout.addWidget(QLabel('Elliptical Gaussian:'))
        fit_group_layout.addWidget(self.elliptical)

        fit_group.setLayout(fit_group_layout)
        fit_layout.addWidget(fit_group)
        fit_layout.addStretch()
        fit_tab.setLayout(fit_layout)

        # ===== Tab 4: Camera & Rendering =====
        camera_tab = QWidget()
        camera_layout = QVBoxLayout()

        camera_group = QGroupBox("Camera Parameters")
        camera_group_layout = QVBoxLayout()

        self.pixel_size = SliderLabel(3)
        self.pixel_size.setRange(50.0, 200.0)
        camera_group_layout.addWidget(QLabel('Pixel Size (nm):'))
        camera_group_layout.addWidget(self.pixel_size)

        self.photons_per_adu = SliderLabel(3)
        self.photons_per_adu.setRange(0.1, 10.0)
        camera_group_layout.addWidget(QLabel('Photons per ADU:'))
        camera_group_layout.addWidget(self.photons_per_adu)

        self.baseline = SliderLabel(1)
        self.baseline.setRange(0.0, 500.0)
        camera_group_layout.addWidget(QLabel('Baseline:'))
        camera_group_layout.addWidget(self.baseline)

        self.em_gain = SliderLabel(1)
        self.em_gain.setRange(1.0, 1000.0)
        camera_group_layout.addWidget(QLabel('EM Gain:'))
        camera_group_layout.addWidget(self.em_gain)

        camera_group.setLayout(camera_group_layout)
        camera_layout.addWidget(camera_group)

        render_group = QGroupBox("Rendering")
        render_group_layout = QVBoxLayout()

        self.render_pixel_size = SliderLabel(1)
        self.render_pixel_size.setRange(5.0, 50.0)
        render_group_layout.addWidget(QLabel('Render Pixel Size (nm):'))
        render_group_layout.addWidget(self.render_pixel_size)

        self.renderer_type = ComboBox()
        self.renderer_type.addItem('gaussian')
        self.renderer_type.addItem('histogram')
        self.renderer_type.addItem('ash')
        self.renderer_type.addItem('scatter')
        render_group_layout.addWidget(QLabel('Renderer:'))
        render_group_layout.addWidget(self.renderer_type)

        render_group.setLayout(render_group_layout)
        camera_layout.addWidget(render_group)
        camera_layout.addStretch()
        camera_tab.setLayout(camera_layout)

        # Add tabs
        tabs.addTab(filter_tab, "Filtering")
        tabs.addTab(detect_tab, "Detection")
        tabs.addTab(fit_tab, "Fitting")
        tabs.addTab(camera_tab, "Camera/Render")

        self.items.append({'name': 'tabs', 'string': '', 'object': tabs})

        # Save button
        save_btn = QPushButton('Save Localizations')
        save_btn.clicked.connect(self.save_localizations)
        self.items.append({'name': 'save_btn', 'string': '', 'object': save_btn})

    def get_params_dict(self):
        params = super().get_params_dict()

        # Get all parameters
        params['filter_type'] = self.filter_type.currentText()
        params['wavelet_scale'] = int(self.wavelet_scale.value())
        params['wavelet_order'] = int(self.wavelet_order.value())
        params['gaussian_sigma'] = self.gaussian_sigma.value()

        params['detector_type'] = self.detector_type.currentText()
        params['connectivity'] = self.connectivity.currentText()
        params['threshold_expr'] = self.threshold_expr.text()

        params['fitter_type'] = self.fitter_type.currentText()
        params['fit_radius'] = int(self.fit_radius.value())
        params['initial_sigma'] = self.initial_sigma.value()
        params['integrated'] = self.integrated.isChecked()
        params['elliptical'] = self.elliptical.isChecked()

        params['pixel_size'] = self.pixel_size.value()
        params['photons_per_adu'] = self.photons_per_adu.value()
        params['baseline'] = self.baseline.value()
        params['em_gain'] = self.em_gain.value()

        params['render_pixel_size'] = self.render_pixel_size.value()
        params['renderer_type'] = self.renderer_type.currentText()

        return params

    def get_name(self):
        return 'ThunderSTORM Analysis'

    def get_menu_path(self):
        return 'Plugins>ThunderSTORM>Run Analysis'

    def __call__(self, keepSourceWindow=False):
        """Run ThunderSTORM analysis on the current window

        Gets parameters directly from GUI widgets since they're not in self.items
        """
        self.start(keepSourceWindow)

        if self.tif is None:
            g.alert("No image data!")
            return None

        g.m.statusBar().showMessage("Running ThunderSTORM analysis...")

        try:
            # Get parameters directly from widgets
            filter_type = str(self.filter_type.currentText())
            wavelet_scale = int(self.wavelet_scale.value())
            wavelet_order = int(self.wavelet_order.value())
            gaussian_sigma = float(self.gaussian_sigma.value())

            detector_type = str(self.detector_type.currentText())
            connectivity = str(self.connectivity.currentText())
            threshold_expr = str(self.threshold_expr.text())

            fitter_type = str(self.fitter_type.currentText())
            fit_radius = int(self.fit_radius.value())
            initial_sigma = float(self.initial_sigma.value())
            integrated = bool(self.integrated.isChecked())
            elliptical = bool(self.elliptical.isChecked())

            pixel_size = float(self.pixel_size.value())
            photons_per_adu = float(self.photons_per_adu.value())
            baseline = float(self.baseline.value())
            em_gain = float(self.em_gain.value())

            render_pixel_size = float(self.render_pixel_size.value())
            renderer_type = str(self.renderer_type.currentText())

            # Get image stack
            image_stack = self.tif
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]

            # Create filter parameters
            if filter_type == 'wavelet':
                filter_params = {
                    'scale': wavelet_scale,
                    'order': wavelet_order
                }
            elif filter_type == 'gaussian':
                filter_params = {'sigma': gaussian_sigma}
            else:
                filter_params = {}

            # Create detector parameters
            detector_params = {}
            if detector_type == 'local_maximum':
                detector_params['connectivity'] = connectivity

            # Create fitter parameters
            fitter_params = {
                'initial_sigma': initial_sigma,
                'integrated': integrated,
                'elliptical': elliptical
            }

            # Create pipeline
            self.pipeline = ThunderSTORM(
                filter_type=filter_type,
                filter_params=filter_params,
                detector_type=detector_type,
                detector_params=detector_params,
                fitter_type=fitter_type,
                fitter_params=fitter_params,
                threshold_expression=threshold_expr,
                pixel_size=pixel_size,
                photons_per_adu=photons_per_adu,
                baseline=baseline,
                em_gain=em_gain
            )

            # Analyze stack
            g.m.statusBar().showMessage(f"Analyzing {len(image_stack)} frames...")
            self.localizations = self.pipeline.analyze_stack(
                image_stack,
                fit_radius=fit_radius,
                show_progress=True
            )

            n_locs = len(self.localizations['x'])
            g.m.statusBar().showMessage(f"Found {n_locs} molecules. Rendering...")

            # Render super-resolution image
            sr_image = self.pipeline.render(
                renderer_type=renderer_type,
                pixel_size=render_pixel_size
            )

            # Store result for output
            self.newtif = sr_image
            self.newname = f"{self.oldname}_ThunderSTORM"

            # Show statistics
            stats = self.pipeline.get_statistics()
            msg = f"ThunderSTORM Analysis Complete\n"
            msg += f"Molecules detected: {stats['n_localizations']}\n"
            if 'mean_intensity' in stats:
                msg += f"Mean intensity: {stats['mean_intensity']:.1f} photons\n"
            if 'mean_uncertainty' in stats:
                msg += f"Mean uncertainty: {stats['mean_uncertainty']:.2f} nm\n"

            g.m.statusBar().showMessage(msg, 5000)
            print(msg)

            return self.end()

        except Exception as e:
            g.alert(f"Error in ThunderSTORM analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            g.m.statusBar().showMessage("Analysis failed", 3000)
            return None

    def closeEvent(self, event):
        BaseProcess.closeEvent(self, event)

    def save_localizations(self):
        """Save localizations to CSV file"""
        if self.localizations is None:
            g.alert("No localizations to save. Run analysis first.")
            return

        filename = save_file_gui("Save Localizations", filetypes='*.csv')
        if filename:
            try:
                self.pipeline.save(filename)
                g.m.statusBar().showMessage(f"Localizations saved to {filename}", 3000)
            except Exception as e:
                g.alert(f"Error saving localizations: {str(e)}")


# ============================================================================
# Post-Processing Tools
# ============================================================================

class ThunderSTORM_PostProcessing(BaseProcess):
    """
    Post-processing tools for localization data.

    Provides:
    - Quality filtering (intensity, uncertainty, sigma)
    - Density-based filtering
    - Molecule merging (handle blinking)
    - Duplicate removal
    """

    def __init__(self):
        super().__init__()

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available.")
            self.close()
            return

        self.localizations = None

    def get_init_settings_dict(self):
        return {
            # Filtering
            'min_intensity': 0,
            'max_intensity': 100000,
            'max_uncertainty': 100,
            'min_sigma': 0.5,
            'max_sigma': 3.0,

            # Density filter
            'density_radius': 100,
            'min_neighbors': 3,

            # Merging
            'merge_distance': 50,
            'merge_frame_gap': 2,

            'apply_filtering': True,
            'apply_density': False,
            'apply_merging': True
        }

    def setupGUI(self):
        super().setupGUI()

        # Load localizations button
        load_group = QGroupBox("Load Localizations")
        load_layout = QVBoxLayout()
        load_btn = QPushButton('Load CSV File')
        load_btn.clicked.connect(self.load_localizations)
        load_layout.addWidget(load_btn)
        load_group.setLayout(load_layout)
        self.items.append({'name': 'load_group', 'string': '', 'object': load_group})

        # Quality filtering
        filter_group = QGroupBox("Quality Filtering")
        filter_layout = QVBoxLayout()

        self.apply_filtering = CheckBox()
        filter_layout.addWidget(QLabel('Enable Quality Filtering:'))
        filter_layout.addWidget(self.apply_filtering)

        self.min_intensity = SliderLabel(0)
        self.min_intensity.setRange(0, 10000)
        filter_layout.addWidget(QLabel('Min Intensity (photons):'))
        filter_layout.addWidget(self.min_intensity)

        self.max_intensity = SliderLabel(0)
        self.max_intensity.setRange(0, 100000)
        filter_layout.addWidget(QLabel('Max Intensity (photons):'))
        filter_layout.addWidget(self.max_intensity)

        self.max_uncertainty = SliderLabel(1)
        self.max_uncertainty.setRange(10, 200)
        filter_layout.addWidget(QLabel('Max Uncertainty (nm):'))
        filter_layout.addWidget(self.max_uncertainty)

        self.min_sigma = SliderLabel(3)
        self.min_sigma.setRange(0.1, 2.0)
        filter_layout.addWidget(QLabel('Min Sigma (pixels):'))
        filter_layout.addWidget(self.min_sigma)

        self.max_sigma = SliderLabel(3)
        self.max_sigma.setRange(1.0, 5.0)
        filter_layout.addWidget(QLabel('Max Sigma (pixels):'))
        filter_layout.addWidget(self.max_sigma)

        filter_group.setLayout(filter_layout)
        self.items.append({'name': 'filter_group', 'string': '', 'object': filter_group})

        # Density filtering
        density_group = QGroupBox("Density Filtering")
        density_layout = QVBoxLayout()

        self.apply_density = CheckBox()
        density_layout.addWidget(QLabel('Enable Density Filtering:'))
        density_layout.addWidget(self.apply_density)

        self.density_radius = SliderLabel(1)
        self.density_radius.setRange(20, 500)
        density_layout.addWidget(QLabel('Search Radius (nm):'))
        density_layout.addWidget(self.density_radius)

        self.min_neighbors = SliderLabel(0)
        self.min_neighbors.setRange(1, 20)
        density_layout.addWidget(QLabel('Min Neighbors:'))
        density_layout.addWidget(self.min_neighbors)

        density_group.setLayout(density_layout)
        self.items.append({'name': 'density_group', 'string': '', 'object': density_group})

        # Merging
        merge_group = QGroupBox("Molecule Merging")
        merge_layout = QVBoxLayout()

        self.apply_merging = CheckBox()
        merge_layout.addWidget(QLabel('Enable Merging:'))
        merge_layout.addWidget(self.apply_merging)

        self.merge_distance = SliderLabel(1)
        self.merge_distance.setRange(10, 200)
        merge_layout.addWidget(QLabel('Max Distance (nm):'))
        merge_layout.addWidget(self.merge_distance)

        self.merge_frame_gap = SliderLabel(0)
        self.merge_frame_gap.setRange(1, 10)
        merge_layout.addWidget(QLabel('Max Frame Gap:'))
        merge_layout.addWidget(self.merge_frame_gap)

        merge_group.setLayout(merge_layout)
        self.items.append({'name': 'merge_group', 'string': '', 'object': merge_group})

        # Save button
        save_btn = QPushButton('Save Filtered Localizations')
        save_btn.clicked.connect(self.save_localizations)
        self.items.append({'name': 'save_btn', 'string': '', 'object': save_btn})

    def get_name(self):
        return 'Post-Processing'

    def get_menu_path(self):
        return 'Plugins>ThunderSTORM>Post-Processing'

    def __call__(self, min_intensity=0, max_intensity=100000, max_uncertainty=100,
                 min_sigma=0.5, max_sigma=3.0, density_radius=100, min_neighbors=3,
                 merge_distance=50, merge_frame_gap=2,
                 apply_filtering=True, apply_density=False, apply_merging=True,
                 keepSourceWindow=False):
        """Process localizations with filtering, density, and merging

        This is called when user clicks 'Run' - delegates to process()
        """
        return self.process()

    def load_localizations(self):
        """Load localizations from CSV"""
        filename = open_file_gui("Load Localizations", filetypes='*.csv')
        if filename:
            try:
                self.localizations = utils.load_localizations_csv(filename)
                n = len(self.localizations['x'])
                g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
            except Exception as e:
                g.alert(f"Error loading file: {str(e)}")

    def process(self):
        if self.localizations is None:
            g.alert("No localizations loaded. Load a CSV file first or run analysis.")
            return None

        try:
            params = self.get_params_dict()
            locs = self.localizations.copy()
            initial_count = len(locs['x'])

            # Create pipeline for post-processing
            pipeline = ThunderSTORM()
            pipeline.localizations = locs

            # Apply filtering
            if params['apply_filtering']:
                g.m.statusBar().showMessage("Applying quality filters...")
                locs = pipeline.filter_localizations(
                    min_intensity=params['min_intensity'],
                    max_intensity=params['max_intensity'],
                    max_uncertainty=params['max_uncertainty'],
                    min_sigma=params['min_sigma'],
                    max_sigma=params['max_sigma']
                )
                pipeline.localizations = locs

            # Apply density filtering
            if params['apply_density']:
                g.m.statusBar().showMessage("Applying density filter...")
                locs = pipeline.filter_by_density(
                    radius=params['density_radius'],
                    min_neighbors=params['min_neighbors']
                )
                pipeline.localizations = locs

            # Apply merging
            if params['apply_merging']:
                g.m.statusBar().showMessage("Merging molecules...")
                locs = pipeline.merge_molecules(
                    max_distance=params['merge_distance'],
                    max_frame_gap=params['merge_frame_gap']
                )
                pipeline.localizations = locs

            # Store result
            self.localizations = locs
            final_count = len(locs['x'])

            # Render
            sr_image = pipeline.render(renderer_type='gaussian', pixel_size=10)

            msg = f"Post-processing complete\n"
            msg += f"Initial: {initial_count} molecules\n"
            msg += f"Final: {final_count} molecules\n"
            msg += f"Removed: {initial_count - final_count} molecules"

            g.m.statusBar().showMessage(msg, 5000)
            print(msg)

            # Create result window
            if g.win is not None:
                result_window = Window(sr_image, name=f"{g.win.name}_filtered")
            else:
                result_window = Window(sr_image, name="filtered_SR")

            return result_window

        except Exception as e:
            g.alert(f"Error in post-processing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_localizations(self):
        """Save filtered localizations"""
        if self.localizations is None:
            g.alert("No localizations to save.")
            return

        filename = save_file_gui("Save Filtered Localizations", filetypes='*.csv')
        if filename:
            try:
                utils.save_localizations_csv(self.localizations, filename)
                g.m.statusBar().showMessage(f"Saved to {filename}", 3000)
            except Exception as e:
                g.alert(f"Error saving: {str(e)}")


# ============================================================================
# Drift Correction
# ============================================================================

class ThunderSTORM_DriftCorrection(BaseProcess):
    """
    Drift correction for localization data.

    Methods:
    - Cross-correlation based drift estimation
    - Fiducial marker tracking
    """

    def __init__(self):
        super().__init__()

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available.")
            self.close()
            return

        self.localizations = None

    def get_init_settings_dict(self):
        return {
            'method': 'cross_correlation',
            'bin_size': 5,  # frames
            'pixel_size': 10  # nm for reconstruction
        }

    def setupGUI(self):
        super().setupGUI()

        # Load button
        load_btn = QPushButton('Load Localizations CSV')
        load_btn.clicked.connect(self.load_localizations)
        self.items.append({'name': 'load_btn', 'string': '', 'object': load_btn})

        # Method selection
        self.method = ComboBox()
        self.method.addItem('cross_correlation')
        self.method.addItem('fiducial')
        self.items.append({'name': 'method', 'string': 'Drift Correction Method', 'object': self.method})

        self.bin_size = SliderLabel(0)
        self.bin_size.setRange(1, 20)
        self.items.append({'name': 'bin_size', 'string': 'Bin Size (frames)', 'object': self.bin_size})

        self.pixel_size = SliderLabel(1)
        self.pixel_size.setRange(5, 50)
        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)', 'object': self.pixel_size})

        # Save button
        save_btn = QPushButton('Save Corrected Localizations')
        save_btn.clicked.connect(self.save_localizations)
        self.items.append({'name': 'save_btn', 'string': '', 'object': save_btn})

    def get_name(self):
        return 'Drift Correction'

    def get_menu_path(self):
        return 'Plugins>ThunderSTORM>Drift Correction'

    def __call__(self, method='cross_correlation', bin_size=5, pixel_size=10,
                 keepSourceWindow=False):
        """Apply drift correction to localizations

        This is called when user clicks 'Run' - delegates to process()
        """
        return self.process()

    def load_localizations(self):
        filename = open_file_gui("Load Localizations", filetypes='*.csv')
        if filename:
            try:
                self.localizations = utils.load_localizations_csv(filename)
                n = len(self.localizations['x'])
                g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
            except Exception as e:
                g.alert(f"Error loading file: {str(e)}")

    def process(self):
        if self.localizations is None:
            g.alert("No localizations loaded. Load a CSV file first.")
            return None

        try:
            params = self.get_params_dict()

            # Create pipeline
            pipeline = ThunderSTORM()
            pipeline.localizations = self.localizations.copy()

            # Dummy images for drift correction (needed for cross-correlation method)
            if 'frame' in self.localizations:
                n_frames = int(np.max(self.localizations['frame'])) + 1
                pipeline.images = np.zeros((n_frames, 128, 128))  # Dummy

            # Apply drift correction
            g.m.statusBar().showMessage(f"Computing drift ({params['method']})...")
            corrected = pipeline.apply_drift_correction(
                method=params['method'],
                bin_size=params['bin_size']
            )

            self.localizations = corrected

            # Render before/after
            renderer = visualization.GaussianRenderer(sigma=20)
            before_img = renderer.render(self.localizations, pixel_size=params['pixel_size'])

            g.m.statusBar().showMessage("Drift correction complete", 3000)

            result_window = Window(before_img, name="drift_corrected")
            return result_window

        except Exception as e:
            g.alert(f"Error in drift correction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def save_localizations(self):
        if self.localizations is None:
            g.alert("No localizations to save.")
            return

        filename = save_file_gui("Save Drift Corrected", filetypes='*.csv')
        if filename:
            try:
                utils.save_localizations_csv(self.localizations, filename)
                g.m.statusBar().showMessage(f"Saved to {filename}", 3000)
            except Exception as e:
                g.alert(f"Error saving: {str(e)}")


# ============================================================================
# Visualization/Rendering
# ============================================================================

class ThunderSTORM_Rendering(BaseProcess):
    """
    Super-resolution rendering from localization data.

    Methods:
    - Gaussian rendering
    - Histogram (with jittering)
    - Average Shifted Histogram (ASH)
    - Scatter plot
    """

    def __init__(self):
        super().__init__()

        if not THUNDERSTORM_AVAILABLE:
            g.alert("ThunderSTORM modules not available.")
            self.close()
            return

        self.localizations = None

    def get_init_settings_dict(self):
        return {
            'renderer': 'gaussian',
            'pixel_size': 10.0,
            'gaussian_sigma': 20.0,
            'ash_shifts': 4,
            'jitter': False,
            'n_averages': 10
        }

    def setupGUI(self):
        super().setupGUI()

        # Load button
        load_btn = QPushButton('Load Localizations CSV')
        load_btn.clicked.connect(self.load_localizations)
        self.items.append({'name': 'load_btn', 'string': '', 'object': load_btn})

        # Renderer selection
        self.renderer = ComboBox()
        self.renderer.addItem('gaussian')
        self.renderer.addItem('histogram')
        self.renderer.addItem('ash')
        self.renderer.addItem('scatter')
        self.items.append({'name': 'renderer', 'string': 'Renderer Type', 'object': self.renderer})

        self.pixel_size = SliderLabel(1)
        self.pixel_size.setRange(1, 50)
        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)', 'object': self.pixel_size})

        # Gaussian options
        self.gaussian_sigma = SliderLabel(1)
        self.gaussian_sigma.setRange(10, 100)
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Sigma (nm)', 'object': self.gaussian_sigma})

        # ASH options
        self.ash_shifts = SliderLabel(0)
        self.ash_shifts.setRange(2, 8)
        self.items.append({'name': 'ash_shifts', 'string': 'ASH Shifts', 'object': self.ash_shifts})

        # Histogram options
        self.jitter = CheckBox()
        self.items.append({'name': 'jitter', 'string': 'Jittering (Histogram)', 'object': self.jitter})

        self.n_averages = SliderLabel(0)
        self.n_averages.setRange(1, 20)
        self.items.append({'name': 'n_averages', 'string': 'Jitter Averages', 'object': self.n_averages})

    def get_name(self):
        return 'Rendering'

    def get_menu_path(self):
        return 'Plugins>ThunderSTORM>Rendering'

    def __call__(self, renderer='gaussian', pixel_size=10.0,
                 keepSourceWindow=False):
        """Render super-resolution image from localizations

        This is called when user clicks 'Run' - delegates to process()
        """
        return self.process()

    def load_localizations(self):
        filename = open_file_gui("Load Localizations", filetypes='*.csv')
        if filename:
            try:
                self.localizations = utils.load_localizations_csv(filename)
                n = len(self.localizations['x'])
                g.m.statusBar().showMessage(f"Loaded {n} localizations", 3000)
            except Exception as e:
                g.alert(f"Error loading file: {str(e)}")

    def process(self):
        if self.localizations is None:
            g.alert("No localizations loaded. Load a CSV file first.")
            return None

        try:
            params = self.get_params_dict()

            g.m.statusBar().showMessage(f"Rendering with {params['renderer']}...")

            # Create renderer
            if params['renderer'] == 'gaussian':
                renderer = visualization.GaussianRenderer(sigma=params['gaussian_sigma'])
            elif params['renderer'] == 'histogram':
                renderer = visualization.HistogramRenderer(
                    jittering=params['jitter'],
                    n_averages=params['n_averages']
                )
            elif params['renderer'] == 'ash':
                renderer = visualization.AverageShiftedHistogram(
                    n_shifts=params['ash_shifts']
                )
            else:  # scatter
                renderer = visualization.ScatterRenderer()

            # Render
            sr_image = renderer.render(self.localizations, pixel_size=params['pixel_size'])

            g.m.statusBar().showMessage("Rendering complete", 3000)

            result_window = Window(sr_image, name=f"SR_{params['renderer']}")
            return result_window

        except Exception as e:
            g.alert(f"Error in rendering: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


# ============================================================================
# Simulation Tools
# ============================================================================

def simulate_smlm_data():
    """Generate simulated SMLM data for testing"""
    if not THUNDERSTORM_AVAILABLE:
        g.alert("ThunderSTORM modules not available.")
        return

    try:
        # Create simulator
        simulator = SMLMSimulator(
            image_size=(256, 256),
            pixel_size=100.0,
            psf_sigma=150.0,
            photons_per_molecule=1000,
            background_photons=20
        )

        # Ask for pattern type
        from qtpy.QtWidgets import QInputDialog
        pattern_types = ['siemens_star', 'grid', 'circle', 'random']
        pattern, ok = QInputDialog.getItem(
            None, "Select Pattern", "Pattern type:", pattern_types, 0, False
        )

        if not ok:
            return

        # Create pattern
        g.m.statusBar().showMessage(f"Generating {pattern} pattern...")
        mask = create_test_pattern(pattern, size=256)

        # Generate movie
        n_frames = 500
        g.m.statusBar().showMessage(f"Simulating {n_frames} frames...")
        movie, ground_truth = simulator.generate_movie(
            n_frames=n_frames,
            mask=mask,
            blinking=True
        )

        # Create window
        result_window = Window(movie, name=f"Simulated_{pattern}")

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
        return result_window

    except Exception as e:
        g.alert(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def quick_analysis_simple():
    """Quick analysis with default parameters"""
    if not THUNDERSTORM_AVAILABLE:
        g.alert("ThunderSTORM modules not available.")
        return None

    if g.win is None:
        g.alert("No window open! Please open an SMLM movie first.")
        return None

    try:
        g.m.statusBar().showMessage("Running quick analysis...")

        # Get image stack
        image_stack = g.win.image
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

        print(f"Quick analysis starting on {image_stack.shape[0]} frames...")

        # Run quick analysis
        localizations, sr_image, pipeline = quick_analysis(image_stack)

        n_locs = len(localizations['x'])
        print(f"Quick analysis complete: {n_locs} molecules detected")
        g.m.statusBar().showMessage(f"Quick analysis complete: {n_locs} molecules", 3000)

        # Create result window
        result_window = Window(sr_image, name=f"{g.win.name}_quick_SR")
        return result_window

    except Exception as e:
        error_msg = f"Error in quick analysis: {str(e)}"
        g.alert(error_msg)
        print(error_msg)
        import traceback
        traceback.print_exc()
        g.m.statusBar().showMessage("Analysis failed", 3000)
        return None


# ============================================================================
# Module-level Instances (FLIKA BaseProcess pattern)
# ============================================================================

# Create instances at module level - FLIKA will call these directly
# When menu item is clicked, FLIKA calls the instance which invokes __call__()
thunderstorm_run_analysis = ThunderSTORM_RunAnalysis()
thunderstorm_post_processing = ThunderSTORM_PostProcessing()
thunderstorm_drift_correction = ThunderSTORM_DriftCorrection()
thunderstorm_rendering = ThunderSTORM_Rendering()


# ============================================================================
# Menu Integration
# ============================================================================

# Menu items are now defined in info.xml using the <menu_layout> section
# This provides better integration with FLIKA's plugin manager

# The following menu items are available:
# - Run Analysis (launch_run_analysis)
# - Quick Analysis (quick_analysis_simple)
# - Post-Processing (launch_post_processing)
# - Drift Correction (launch_drift_correction)
# - Rendering (launch_rendering)
# - Simulate Data (simulate_smlm_data)


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
