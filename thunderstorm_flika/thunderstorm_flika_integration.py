#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ThunderSTORM FLIKA Integration Module
======================================

This module integrates the ThunderSTORM Python implementation with FLIKA,
providing GUI interfaces for analysis and visualization.

Features:
- Run Analysis GUI with all thunderSTORM options
- Quick Analysis for fast processing
- Automatic localization display on original images
- Results Viewer with spreadsheet, filtering, and plotting
- Export and rendering options
- Integration with FLIKA's window system

Author: George K (with Claude)
Date: 2025-12-13 (Updated with Results Viewer)
"""

import logging
import numpy as np
from typing import Optional, Dict, Tuple
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QMainWindow, QLabel, QPushButton, QSpinBox,
                           QDoubleSpinBox, QCheckBox, QComboBox, QMessageBox,
                           QProgressBar, QFileDialog)

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g
from distutils.version import StrictVersion

# Version-specific imports
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                         ComboBox, BaseProcess_noPriorWindow,
                                         WindowSelector, save_file_gui)
else:
    from flika.utils.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                       ComboBox, BaseProcess_noPriorWindow,
                                       WindowSelector, save_file_gui)

# ThunderSTORM imports
try:
    from .thunderstorm_python import pipeline, filters, detection, fitting
    from .localization_display import LocalizationDisplay
    from .localization_results_viewer import LocalizationResultsViewer, show_results_viewer
except ImportError:
    # Fallback for development
    import sys
    from pathlib import Path
    module_path = Path(__file__).parent
    if module_path not in sys.path:
        sys.path.insert(0, str(module_path))

    import pipeline
    from localization_display import LocalizationDisplay
    from localization_results_viewer import LocalizationResultsViewer, show_results_viewer

# Set up logging
logger = logging.getLogger(__name__)


class ThunderSTORM_RunAnalysis(BaseProcess_noPriorWindow):
    """
    Run full ThunderSTORM analysis with all options.

    This plugin provides a comprehensive GUI for configuring and running
    SMLM analysis on the current FLIKA window.
    """

    def __init__(self):
        super().__init__()
        self.ts_pipeline: Optional[pipeline.ThunderSTORM] = None
        self.localizations: Optional[Dict] = None

    def gui(self):
        """Create comprehensive analysis GUI."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # === Image Filtering ===
        filter_type = ComboBox()
        for ftype in ['wavelet', 'gaussian', 'dog', 'lowered_gaussian', 'none']:
            filter_type.addItem(ftype)
        filter_type.setValue('wavelet')

        # Wavelet scale
        wavelet_scale = QSpinBox()
        wavelet_scale.setRange(1, 5)
        wavelet_scale.setValue(2)

        # Gaussian sigma
        gaussian_sigma = QDoubleSpinBox()
        gaussian_sigma.setRange(0.1, 10.0)
        gaussian_sigma.setValue(1.6)
        gaussian_sigma.setSingleStep(0.1)

        # === Detection ===
        detector_type = ComboBox()
        for dtype in ['local_maximum', 'non_maximum_suppression', 'centroid']:
            detector_type.addItem(dtype)
        detector_type.setValue('local_maximum')

        # Threshold expression
        threshold_expr = QComboBox()
        threshold_expr.addItem('std(Wave.F1)')
        threshold_expr.addItem('1.5*std(Wave.F1)')
        threshold_expr.addItem('2*std(Wave.F1)')
        threshold_expr.addItem('mean(F1) + 3*std(F1)')
        threshold_expr.setEditable(True)
        threshold_expr.setCurrentText('std(Wave.F1)')

        # === PSF Fitting ===
        fitter_type = ComboBox()
        for ftype in ['gaussian_lsq', 'gaussian_mle', 'radial_symmetry',
                     'centroid', 'phasor']:
            fitter_type.addItem(ftype)
        fitter_type.setValue('gaussian_lsq')

        fit_radius = QSpinBox()
        fit_radius.setRange(1, 10)
        fit_radius.setValue(3)
        fit_radius.setToolTip("Fitting radius in pixels")

        # === Camera Parameters ===
        pixel_size = QDoubleSpinBox()
        pixel_size.setRange(1.0, 1000.0)
        pixel_size.setValue(100.0)
        pixel_size.setSingleStep(1.0)
        pixel_size.setToolTip("Camera pixel size in nm")

        photons_per_adu = QDoubleSpinBox()
        photons_per_adu.setRange(0.01, 100.0)
        photons_per_adu.setValue(1.0)
        photons_per_adu.setSingleStep(0.1)

        # === Display Options ===
        show_localizations = CheckBox()
        show_localizations.setChecked(True)
        show_localizations.setToolTip("Display localizations on image after analysis")

        color_mode = ComboBox()
        for mode in ['green', 'red', 'intensity', 'frame']:
            color_mode.addItem(mode)
        color_mode.setValue('green')

        point_size = QSpinBox()
        point_size.setRange(1, 20)
        point_size.setValue(4)

        # === Output Options ===
        save_csv = CheckBox()
        save_csv.setChecked(False)

        render_sr = CheckBox()
        render_sr.setChecked(True)
        render_sr.setToolTip("Create super-resolution rendered image")

        # === NEW: Results Viewer Option ===
        show_results_viewer = CheckBox()
        show_results_viewer.setChecked(False)
        show_results_viewer.setToolTip("Open results spreadsheet viewer after analysis")

        # Add all items
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})

        self.items.append({'name': 'filter_type', 'string': 'Filter Type',
                          'object': filter_type})
        self.items.append({'name': 'wavelet_scale', 'string': 'Wavelet Scale',
                          'object': wavelet_scale})
        self.items.append({'name': 'gaussian_sigma', 'string': 'Gaussian Sigma',
                          'object': gaussian_sigma})

        self.items.append({'name': 'detector_type', 'string': 'Detector',
                          'object': detector_type})
        self.items.append({'name': 'threshold_expr', 'string': 'Threshold',
                          'object': threshold_expr})

        self.items.append({'name': 'fitter_type', 'string': 'PSF Fitter',
                          'object': fitter_type})
        self.items.append({'name': 'fit_radius', 'string': 'Fit Radius',
                          'object': fit_radius})

        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)',
                          'object': pixel_size})
        self.items.append({'name': 'photons_per_adu', 'string': 'Photons/ADU',
                          'object': photons_per_adu})

        self.items.append({'name': 'show_localizations',
                          'string': 'Display Localizations',
                          'object': show_localizations})
        self.items.append({'name': 'color_mode', 'string': 'Point Color',
                          'object': color_mode})
        self.items.append({'name': 'point_size', 'string': 'Point Size',
                          'object': point_size})

        self.items.append({'name': 'save_csv', 'string': 'Save CSV',
                          'object': save_csv})
        self.items.append({'name': 'render_sr', 'string': 'Render SR Image',
                          'object': render_sr})
        self.items.append({'name': 'show_results_viewer', 'string': 'Open Results Viewer',
                          'object': show_results_viewer})

        super().gui()

    def __call__(self, window, filter_type, wavelet_scale, gaussian_sigma,
                detector_type, threshold_expr, fitter_type, fit_radius,
                pixel_size, photons_per_adu, show_localizations, color_mode,
                point_size, save_csv, render_sr, show_results_viewer,
                keepSourceWindow=False):
        """
        Run ThunderSTORM analysis on image window.

        Parameters
        ----------
        window : Window
            FLIKA image window
        filter_type : str
            Image filter type
        wavelet_scale : int
            Wavelet scale (for wavelet filter)
        gaussian_sigma : float
            Gaussian sigma (for Gaussian-based filters)
        detector_type : str
            Detection method
        threshold_expr : str
            Threshold expression
        fitter_type : str
            PSF fitting method
        fit_radius : int
            Fitting radius in pixels
        pixel_size : float
            Camera pixel size in nm
        photons_per_adu : float
            Conversion factor from ADU to photons
        show_localizations : bool
            Display localizations on original image
        color_mode : str
            Color mode for localization display
        point_size : int
            Point size for localization display
        save_csv : bool
            Save results to CSV
        render_sr : bool
            Create super-resolution rendered image
        show_results_viewer : bool
            Open results viewer after analysis
        keepSourceWindow : bool
            Not used (noPriorWindow process)
        """
        self.start()

        try:
            # Get image data
            image = window.image

            if image.ndim == 2:
                image = image[np.newaxis, ...]

            # Setup camera parameters
            camera_params = {
                'pixel_size': pixel_size,
                'photons_per_adu': photons_per_adu
            }

            # Setup filter parameters
            filter_params = {
                'filter_type': filter_type,
                'wavelet_scale': wavelet_scale,
                'gaussian_sigma': gaussian_sigma
            }

            # Setup detection parameters
            detection_params = {
                'detector': detector_type,
                'threshold': threshold_expr
            }

            # Setup fitting parameters
            fitting_params = {
                'fitter': fitter_type,
                'fit_radius': fit_radius
            }

            # Create pipeline
            logger.info("Creating ThunderSTORM pipeline...")
            self.ts_pipeline = pipeline.ThunderSTORM(
                camera_params=camera_params,
                filter_params=filter_params,
                detection_params=detection_params,
                fitting_params=fitting_params
            )

            # Run analysis
            logger.info(f"Processing {len(image)} frames...")
            g.m.statusBar().showMessage(f"Running ThunderSTORM analysis on {len(image)} frames...")

            self.localizations = self.ts_pipeline.run(image)

            # Store localizations in window and global
            window.thunderstorm_localizations = self.localizations
            g.thunderstorm_localizations = self.localizations

            n_locs = len(self.localizations['x'])
            logger.info(f"Analysis complete: {n_locs} localizations found")

            # Display localizations
            if show_localizations:
                display = LocalizationDisplay(window, color_mode=color_mode,
                                             point_size=point_size)
                display.set_localizations(self.localizations)
                display.show_points()
                window.thunderstorm_display = display

            # Save CSV
            if save_csv:
                try:
                    from . import utils
                except ImportError:
                    import utils

                filename = f"{window.name}_localizations.csv"
                utils.save_localizations_csv(self.localizations, filename,
                                           flika_compatible=True)
                logger.info(f"Saved localizations to {filename}")

            # Render super-resolution image
            if render_sr:
                try:
                    from . import rendering
                except ImportError:
                    import rendering

                sr_image = rendering.render_histogram(
                    self.localizations,
                    pixel_size=pixel_size/10.0,  # 10x super-resolution
                    image_size=image.shape[1:]
                )

                # Create new window
                from flika.window import Window
                sr_window = Window(sr_image, name=f"{window.name}_SR")
                logger.info("Created super-resolution image")

            # Open results viewer
            if show_results_viewer:
                try:
                    viewer = LocalizationResultsViewer()
                    viewer.set_data(self.localizations)
                    viewer.show()
                    g.thunderstorm_results_viewer = viewer
                    logger.info("Opened results viewer")
                except Exception as e:
                    logger.error(f"Error opening results viewer: {e}")

            g.m.statusBar().showMessage(f"Analysis complete: {n_locs} localizations found")

            QMessageBox.information(
                None, "Analysis Complete",
                f"ThunderSTORM analysis complete!\n\n"
                f"Found {n_locs} localizations\n"
                f"Filter: {filter_type}\n"
                f"Detector: {detector_type}\n"
                f"Fitter: {fitter_type}"
            )

        except Exception as e:
            logger.error(f"Error in ThunderSTORM analysis: {e}")
            QMessageBox.critical(
                None, "Analysis Error",
                f"Error during analysis:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

        return window


class ThunderSTORM_QuickAnalysis(BaseProcess_noPriorWindow):
    """
    Quick ThunderSTORM analysis with default parameters.

    Fast analysis using recommended default settings:
    - Wavelet filter (scale 2)
    - Local maximum detection
    - Gaussian LSQ fitting
    - Auto display and rendering
    """

    def __init__(self):
        super().__init__()
        self.ts_pipeline: Optional[pipeline.ThunderSTORM] = None
        self.localizations: Optional[Dict] = None

    def gui(self):
        """Create simple quick analysis GUI."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # Camera pixel size (most important parameter)
        pixel_size = QDoubleSpinBox()
        pixel_size.setRange(1.0, 1000.0)
        pixel_size.setValue(100.0)
        pixel_size.setSingleStep(1.0)
        pixel_size.setToolTip("Camera pixel size in nm")

        # Display options
        show_localizations = CheckBox()
        show_localizations.setChecked(True)

        # NEW: Results viewer option
        show_results_viewer = CheckBox()
        show_results_viewer.setChecked(False)

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'pixel_size', 'string': 'Pixel Size (nm)',
                          'object': pixel_size})
        self.items.append({'name': 'show_localizations',
                          'string': 'Display Localizations',
                          'object': show_localizations})
        self.items.append({'name': 'show_results_viewer',
                          'string': 'Open Results Viewer',
                          'object': show_results_viewer})

        super().gui()

    def __call__(self, window, pixel_size, show_localizations,
                show_results_viewer, keepSourceWindow=False):
        """Quick analysis with defaults."""
        self.start()

        try:
            # Get image data
            image = window.image
            if image.ndim == 2:
                image = image[np.newaxis, ...]

            # Default parameters
            camera_params = {'pixel_size': pixel_size, 'photons_per_adu': 1.0}
            filter_params = {'filter_type': 'wavelet', 'wavelet_scale': 2}
            detection_params = {'detector': 'local_maximum',
                              'threshold': 'std(Wave.F1)'}
            fitting_params = {'fitter': 'gaussian_lsq', 'fit_radius': 3}

            # Create pipeline
            logger.info("Running quick ThunderSTORM analysis...")
            g.m.statusBar().showMessage("Running quick ThunderSTORM analysis...")

            self.ts_pipeline = pipeline.ThunderSTORM(
                camera_params=camera_params,
                filter_params=filter_params,
                detection_params=detection_params,
                fitting_params=fitting_params
            )

            # Run analysis
            self.localizations = self.ts_pipeline.run(image)

            # Store results
            window.thunderstorm_localizations = self.localizations
            g.thunderstorm_localizations = self.localizations

            n_locs = len(self.localizations['x'])
            logger.info(f"Quick analysis complete: {n_locs} localizations found")

            # Display localizations
            if show_localizations:
                display = LocalizationDisplay(window, color_mode='green',
                                             point_size=4)
                display.set_localizations(self.localizations)
                display.show_points()
                window.thunderstorm_display = display

            # Auto-render SR image
            try:
                from . import rendering
            except ImportError:
                import rendering

            sr_image = rendering.render_histogram(
                self.localizations,
                pixel_size=pixel_size/10.0,
                image_size=image.shape[1:]
            )

            from flika.window import Window
            sr_window = Window(sr_image, name=f"{window.name}_SR")

            # Open results viewer if requested
            if show_results_viewer:
                try:
                    viewer = LocalizationResultsViewer()
                    viewer.set_data(self.localizations)
                    viewer.show()
                    g.thunderstorm_results_viewer = viewer
                except Exception as e:
                    logger.error(f"Error opening results viewer: {e}")

            g.m.statusBar().showMessage(f"Quick analysis complete: {n_locs} localizations")

            QMessageBox.information(
                None, "Quick Analysis Complete",
                f"Found {n_locs} localizations\n\n"
                f"Used default settings:\n"
                f"• Wavelet filter (scale 2)\n"
                f"• Local maximum detection\n"
                f"• Gaussian LSQ fitting"
            )

        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            QMessageBox.critical(
                None, "Analysis Error",
                f"Error during quick analysis:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

        return window


class ThunderSTORM_ToggleDisplay(BaseProcess_noPriorWindow):
    """
    Toggle localization display visibility.

    Shows/hides localization points on the current window.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        """Create toggle display GUI."""
        self.gui_reset()

        window_selector = WindowSelector()

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})

        super().gui()

    def __call__(self, window, keepSourceWindow=False):
        """Toggle display."""
        try:
            if not hasattr(window, 'thunderstorm_display'):
                QMessageBox.warning(None, "No Display",
                                  "No localization display found for this window.")
                return window

            display = window.thunderstorm_display
            display.toggle_points()

            if display.pointsVisible:
                g.m.statusBar().showMessage("Localizations shown")
            else:
                g.m.statusBar().showMessage("Localizations hidden")

        except Exception as e:
            logger.error(f"Error toggling display: {e}")

        return window


class ThunderSTORM_ViewResults(BaseProcess_noPriorWindow):
    """
    View Localization Results in interactive spreadsheet viewer.

    Opens a comprehensive viewer for analyzing, filtering, and plotting
    localization data. Can load from current analysis or CSV file.
    """

    def __init__(self):
        super().__init__()
        self.viewer = None

    def gui(self):
        """Create results viewer GUI."""
        # Check if we have localizations from a previous analysis
        has_current_data = (
            hasattr(g, 'thunderstorm_localizations') and
            g.thunderstorm_localizations is not None
        )

        self.gui_reset()

        # Option to load from current analysis or file
        data_source = ComboBox()
        if has_current_data:
            data_source.addItem('Current Analysis Results')
        data_source.addItem('Load from CSV File')

        self.items.append({'name': 'data_source', 'string': 'Data Source',
                          'object': data_source})

        super().gui()

    def __call__(self, data_source):
        """Open results viewer with specified data source."""
        # FIXED: Remove keepSourceWindow argument - BaseProcess_noPriorWindow.start() takes no arguments
        self.start()

        try:
            # Create viewer if it doesn't exist or was closed
            if self.viewer is None or not self.viewer.isVisible():
                self.viewer = LocalizationResultsViewer()
                self.viewer.show()
                g.thunderstorm_results_viewer = self.viewer
            else:
                # Bring existing viewer to front
                self.viewer.raise_()
                self.viewer.activateWindow()

            # Load data based on source
            if data_source == 'Current Analysis Results':
                if hasattr(g, 'thunderstorm_localizations'):
                    self.viewer.set_data(g.thunderstorm_localizations)
                    n_locs = len(g.thunderstorm_localizations.get('x', []))
                    self.viewer.info_label.setText(
                        f"Current Analysis Results ({n_locs} localizations)"
                    )
                    g.m.statusBar().showMessage(f"Loaded {n_locs} localizations into viewer")
                else:
                    QMessageBox.warning(
                        None, "No Data",
                        "No analysis results available. Please run an analysis first."
                    )

            elif data_source == 'Load from CSV File':
                self.viewer.load_csv()

        except Exception as e:
            logger.error(f"Error opening results viewer: {e}")
            QMessageBox.critical(
                None, "Viewer Error",
                f"Error opening results viewer:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

        return None


class ThunderSTORM_ExportResults(BaseProcess_noPriorWindow):
    """
    Export localization results to CSV file.

    Quick export function for saving analysis results.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        """Create export GUI."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # FLIKA compatible format option
        flika_compatible = CheckBox()
        flika_compatible.setChecked(True)
        flika_compatible.setToolTip(
            "Swap x/y coordinates to match ImageJ thunderSTORM format for FLIKA display"
        )

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'flika_compatible', 'string': 'FLIKA Compatible Format',
                          'object': flika_compatible})

        super().gui()

    def __call__(self, window, flika_compatible, keepSourceWindow=False):
        """Export localizations to CSV."""
        try:
            if not hasattr(window, 'thunderstorm_localizations'):
                QMessageBox.warning(
                    None, "No Data",
                    "No localization data found for this window."
                )
                return window

            filename = QFileDialog.getSaveFileName(
                None, "Export Localizations",
                f"{window.name}_localizations.csv",
                "CSV Files (*.csv)")[0]

            if not filename:
                return window

            try:
                from . import utils
            except ImportError:
                import utils

            utils.save_localizations_csv(
                window.thunderstorm_localizations,
                filename,
                flika_compatible=flika_compatible
            )

            n_locs = len(window.thunderstorm_localizations['x'])
            g.m.statusBar().showMessage(
                f"Exported {n_locs} localizations to {filename}"
            )

            QMessageBox.information(
                None, "Export Complete",
                f"Exported {n_locs} localizations to:\n{filename}"
            )

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            QMessageBox.critical(
                None, "Export Error",
                f"Error exporting results:\n{str(e)}"
            )

        return window


# ============================================================================
# Create plugin instances
# ============================================================================

thunderstorm_run_analysis = ThunderSTORM_RunAnalysis()
thunderstorm_quick_analysis = ThunderSTORM_QuickAnalysis()
thunderstorm_toggle_display = ThunderSTORM_ToggleDisplay()
thunderstorm_view_results = ThunderSTORM_ViewResults()
thunderstorm_export_results = ThunderSTORM_ExportResults()


# ============================================================================
# Menu Structure (for reference - add to FLIKA menus.py)
# ============================================================================
"""
thunderstorm_menu = [
    {'Analysis': {
        'Run ThunderSTORM Analysis': thunderstorm_run_analysis,
        'Quick Analysis': thunderstorm_quick_analysis,
    }},
    {'Visualization': {
        'Toggle Localization Display': thunderstorm_toggle_display,
    }},
    {'Results': {
        'View Results Spreadsheet': thunderstorm_view_results,
        'Export Results CSV': thunderstorm_export_results,
    }},
]
"""
