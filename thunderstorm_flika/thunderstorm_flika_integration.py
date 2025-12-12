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
- Export and rendering options
- Integration with FLIKA's window system

Author: George K (with Claude)
Date: 2025-12-11
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
except ImportError:
    # Fallback for development
    import sys
    from pathlib import Path
    module_path = Path(__file__).parent
    if module_path not in sys.path:
        sys.path.insert(0, str(module_path))

    import pipeline
    from localization_display import LocalizationDisplay

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

        super().gui()

    def __call__(self, window, filter_type, wavelet_scale, gaussian_sigma,
                detector_type, threshold_expr, fitter_type, fit_radius,
                pixel_size, photons_per_adu, show_localizations, color_mode,
                point_size, save_csv, render_sr, keepSourceWindow=False):
        """
        Run ThunderSTORM analysis on image window.

        Parameters
        ----------
        window : Window
            FLIKA image window
        (other parameters correspond to GUI items)

        Returns
        -------
        window : Window
            Original window with localizations attached
        """
        try:
            # Get image data
            if window.mt == 1:
                # Single image
                images = window.image[np.newaxis, :, :]
            else:
                # Image stack
                images = window.image

            g.m.statusBar().showMessage("Setting up ThunderSTORM pipeline...")

            # Configure filter parameters
            if filter_type == 'wavelet':
                filter_params = {'scale': wavelet_scale, 'order': 3}
            elif filter_type == 'gaussian':
                filter_params = {'sigma': gaussian_sigma}
            else:
                filter_params = {}

            # Create pipeline
            self.ts_pipeline = pipeline.ThunderSTORM(
                filter_type=filter_type,
                filter_params=filter_params,
                detector_type=detector_type,
                detector_params={'connectivity': '8-neighbourhood',
                               'exclude_border': True},
                fitter_type=fitter_type,
                fitter_params={'integrated': True},
                threshold_expression=threshold_expr,
                pixel_size=pixel_size,
                photons_per_adu=photons_per_adu,
                baseline=100.0,
                em_gain=1.0
            )

            g.m.statusBar().showMessage("Running ThunderSTORM analysis...")

            # Run analysis
            self.localizations = self.ts_pipeline.analyze_stack(
                images,
                fit_radius=fit_radius,
                show_progress=True
            )

            n_locs = len(self.localizations['x'])

            if n_locs == 0:
                QMessageBox.warning(None, "No Detections",
                                  "No molecules were detected. Try adjusting threshold.")
                return window

            # Store localizations in window
            window.thunderstorm_localizations = self.localizations
            window.thunderstorm_pipeline = self.ts_pipeline

            g.m.statusBar().showMessage(f"Analysis complete: {n_locs} molecules detected")

            # Display localizations if requested
            if show_localizations:
                display = LocalizationDisplay(window, self.localizations)
                display.show_points(color_mode, point_size, 200)
                display.show_control_window()
                window.thunderstorm_display = display

            # Render super-resolution image if requested
            if render_sr:
                g.m.statusBar().showMessage("Rendering super-resolution image...")
                sr_image = self.ts_pipeline.render(
                    renderer_type='gaussian',
                    pixel_size=10.0  # 10nm pixels for SR image
                )

                # Create new window for SR image
                sr_window = Window(sr_image, name=f"{window.name}_SR")
                sr_window.thunderstorm_localizations = self.localizations

            # Save CSV if requested
            if save_csv:
                filename = QFileDialog.getSaveFileName(
                    None, "Save Localizations", "",
                    "CSV Files (*.csv)")[0]

                if filename:
                    try:
                        from . import utils
                    except ImportError:
                        import utils

                    utils.save_localizations_csv(self.localizations, filename,
                                                flika_compatible=True)
                    g.m.statusBar().showMessage(f"Localizations saved to {filename}")

            # Show summary
            stats = self.ts_pipeline.get_statistics()
            summary = (f"ThunderSTORM Analysis Complete\n\n"
                      f"Localizations: {n_locs}\n"
                      f"Frames: {stats.get('n_frames', 'N/A')}\n"
                      f"Mean Intensity: {stats.get('mean_intensity', 0):.0f} photons\n"
                      f"Mean Uncertainty: {stats.get('mean_uncertainty', 0):.1f} nm")

            QMessageBox.information(None, "Analysis Complete", summary)

        except Exception as e:
            logger.error(f"Error in ThunderSTORM analysis: {e}")
            QMessageBox.critical(None, "Analysis Error",
                               f"Error running analysis: {str(e)}")
            import traceback
            traceback.print_exc()

        return window


class ThunderSTORM_QuickAnalysis(BaseProcess_noPriorWindow):
    """
    Quick ThunderSTORM analysis with default parameters.

    Simplified interface for fast SMLM analysis using optimized defaults.
    """

    def __init__(self):
        super().__init__()

    def gui(self):
        """Create simplified quick analysis GUI."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # Basic options
        show_localizations = CheckBox()
        show_localizations.setChecked(True)

        color_mode = ComboBox()
        for mode in ['green', 'red', 'intensity']:
            color_mode.addItem(mode)

        render_sr = CheckBox()
        render_sr.setChecked(True)

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'show_localizations',
                          'string': 'Display Localizations',
                          'object': show_localizations})
        self.items.append({'name': 'color_mode', 'string': 'Point Color',
                          'object': color_mode})
        self.items.append({'name': 'render_sr', 'string': 'Create SR Image',
                          'object': render_sr})

        super().gui()

    def __call__(self, window, show_localizations, color_mode, render_sr,
                keepSourceWindow=False):
        """Run quick analysis with defaults."""
        try:
            # Get image data
            if window.mt == 1:
                images = window.image[np.newaxis, :, :]
            else:
                images = window.image

            g.m.statusBar().showMessage("Running quick ThunderSTORM analysis...")

            # Create default pipeline
            try:
                from . import pipeline as ts_pipeline
            except ImportError:
                import pipeline as ts_pipeline

            ts = ts_pipeline.create_default_pipeline()

            # Run analysis
            localizations = ts.analyze_stack(images, show_progress=True)

            n_locs = len(localizations['x'])

            if n_locs == 0:
                QMessageBox.warning(None, "No Detections",
                                  "No molecules detected.")
                return window

            # Store in window
            window.thunderstorm_localizations = localizations
            window.thunderstorm_pipeline = ts

            g.m.statusBar().showMessage(f"Quick analysis: {n_locs} molecules")

            # Display localizations
            if show_localizations:
                display = LocalizationDisplay(window, localizations)
                display.show_points(color_mode, 4, 200)
                display.show_control_window()
                window.thunderstorm_display = display

            # Render SR image
            if render_sr:
                sr_image = ts.render(renderer_type='gaussian', pixel_size=10.0)
                Window(sr_image, name=f"{window.name}_SR")

            QMessageBox.information(None, "Quick Analysis Complete",
                                  f"Detected {n_locs} molecules")

        except Exception as e:
            logger.error(f"Error in quick analysis: {e}")
            QMessageBox.critical(None, "Error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

        return window


class ThunderSTORM_ToggleDisplay(BaseProcess_noPriorWindow):
    """
    Toggle localization display on/off for a window.

    Useful for quickly showing/hiding detected molecules on the image.
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


# Create plugin instances
thunderstorm_run_analysis = ThunderSTORM_RunAnalysis()
thunderstorm_quick_analysis = ThunderSTORM_QuickAnalysis()
thunderstorm_toggle_display = ThunderSTORM_ToggleDisplay()
