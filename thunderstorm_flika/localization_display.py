#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Localization Display Module for ThunderSTORM FLIKA Plugin
==========================================================

This module provides functionality to display detected particle localizations
as an overlay on the original image window. It includes customizable appearance
options and interactive features.

Features:
- Display localizations as scatter points on original images
- Color-coding options (by intensity, frame, uncertainty, etc.)
- Adjustable point size and opacity
- Toggle visibility on/off
- Export localization coordinates

Author: George K (with Claude)
Date: 2025-12-11
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QMainWindow, QLabel, QPushButton, QCheckBox,
                           QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox)

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
                                         WindowSelector)
else:
    from flika.utils.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                       ComboBox, BaseProcess_noPriorWindow,
                                       WindowSelector)

# PyQtGraph dock imports
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# Set up logging
logger = logging.getLogger(__name__)


class LocalizationDisplay:
    """
    Display system for showing detected particle localizations on images.

    This class overlays scatter points representing detected molecules on
    the original image stack, with options for customization and interaction.

    Attributes:
        imageWindow: FLIKA image window containing the original data
        localizations: Dictionary of localization data with 'x', 'y', etc.
        scatterPlot: PyQtGraph ScatterPlotItem for displaying points
        pointsVisible: Boolean tracking visibility state
    """

    def __init__(self, imageWindow: Window, localizations: Dict[str, np.ndarray],
                 pixel_size: float = 100.0):
        """
        Initialize the localization display.

        Parameters
        ----------
        imageWindow : Window
            FLIKA image window to display localizations on
        localizations : dict
            Localization data with 'x', 'y' keys (in nanometers)
        pixel_size : float
            Camera pixel size in nanometers (default: 100.0)
        """
        self.imageWindow = imageWindow
        self.localizations = localizations
        self.pixel_size = pixel_size  # nm per pixel
        self.scatterPlot: Optional[pg.ScatterPlotItem] = None
        self.pointsVisible: bool = False

        # Display settings
        self.pointSize: int = 4
        self.pointOpacity: int = 200  # 0-255
        self.pointColor: str = 'green'  # Color mode: 'green', 'intensity', 'frame', 'uncertainty'

        # Frame display mode
        self.displayMode: str = 'all'  # 'all' or 'current_frame'
        self.currentFrame: int = 0
        self.frameChangeConnection = None

        # Control window
        self.controlWindow: Optional[QMainWindow] = None

        logger.debug(f"LocalizationDisplay initialized with pixel_size={pixel_size} nm")

    def show_points(self, color_mode: str = 'green', point_size: int = 4,
                   opacity: int = 200, display_mode: str = 'all') -> None:
        """
        Display localizations as scatter points on the image.

        Parameters
        ----------
        color_mode : str
            How to color points: 'green', 'red', 'white', 'intensity', 'frame', 'uncertainty'
        point_size : int
            Size of scatter points
        opacity : int
            Opacity of points (0-255)
        display_mode : str
            'all' = show all frames, 'current_frame' = show only current frame
        """
        try:
            # Clear existing plot if present
            if self.scatterPlot is not None:
                self.hide_points()

            if len(self.localizations['x']) == 0:
                logger.warning("No localizations to display")
                QMessageBox.warning(None, "No Data",
                                  "No localizations found to display.")
                return

            # Update settings
            self.pointSize = point_size
            self.pointOpacity = opacity
            self.pointColor = color_mode
            self.displayMode = display_mode

            # Get current frame from FLIKA window
            if hasattr(self.imageWindow, 'currentIndex'):
                self.currentFrame = self.imageWindow.currentIndex
            else:
                self.currentFrame = 0

            # Filter localizations based on display mode
            if display_mode == 'current_frame' and 'frame' in self.localizations:
                # Show only points from current frame
                frame_mask = self.localizations['frame'] == self.currentFrame
                if not np.any(frame_mask):
                    logger.warning(f"No localizations in frame {self.currentFrame}")
                    # Still create empty plot and connect to frame changes
                    self._setup_frame_display(color_mode, point_size, opacity)
                    return
            else:
                # Show all frames
                frame_mask = np.ones(len(self.localizations['x']), dtype=bool)

            # Get coordinates in nm and apply frame filter
            x_coords_nm = self.localizations['x'][frame_mask]
            y_coords_nm = self.localizations['y'][frame_mask]

            # Convert from nanometers to pixels
            x_coords = x_coords_nm / self.pixel_size
            y_coords = y_coords_nm / self.pixel_size

            logger.debug(f"Displaying {len(x_coords)} points (mode: {display_mode}, frame: {self.currentFrame})")
            logger.debug(f"Range in pixels: x=[{x_coords.min():.1f}, {x_coords.max():.1f}], "
                        f"y=[{y_coords.min():.1f}, {y_coords.max():.1f}]")

            # Determine colors based on mode
            if color_mode == 'green':
                brush = pg.mkBrush(30, 255, 35, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brush
                )

            elif color_mode == 'red':
                brush = pg.mkBrush(255, 30, 35, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brush
                )

            elif color_mode == 'white':
                brush = pg.mkBrush(255, 255, 255, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brush
                )

            elif color_mode == 'intensity' and 'intensity' in self.localizations:
                # Color by intensity
                intensities = self.localizations['intensity'][frame_mask]
                brushes = self._get_intensity_colors(intensities, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brushes
                )

            elif color_mode == 'frame' and 'frame' in self.localizations:
                # Color by frame number
                frames = self.localizations['frame'][frame_mask]
                brushes = self._get_frame_colors(frames, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brushes
                )

            elif color_mode == 'uncertainty' and 'uncertainty' in self.localizations:
                # Color by localization uncertainty
                uncertainties = self.localizations['uncertainty'][frame_mask]
                brushes = self._get_uncertainty_colors(uncertainties, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brushes
                )

            else:
                # Fallback to green if mode not available
                brush = pg.mkBrush(30, 255, 35, opacity)
                self.scatterPlot = pg.ScatterPlotItem(
                    x=x_coords, y=y_coords,
                    size=point_size,
                    pen=None,
                    brush=brush
                )

            # Add to image view
            self.imageWindow.imageview.view.addItem(self.scatterPlot)
            self.pointsVisible = True

            # Connect to frame changes if in current_frame mode
            if display_mode == 'current_frame':
                self._connect_frame_changes()

            logger.info(f"Displayed {len(x_coords)} localizations with {color_mode} coloring (mode: {display_mode})")

        except Exception as e:
            logger.error(f"Error displaying points: {e}")
            QMessageBox.critical(None, "Display Error",
                               f"Error displaying localizations: {str(e)}")

    def hide_points(self) -> None:
        """Hide localization points from the image."""
        try:
            # Disconnect frame change updates
            self._disconnect_frame_changes()

            if self.scatterPlot is not None:
                self.imageWindow.imageview.view.removeItem(self.scatterPlot)
                self.scatterPlot = None

            self.pointsVisible = False
            logger.debug("Localization points hidden")

        except Exception as e:
            logger.error(f"Error hiding points: {e}")

    def toggle_points(self) -> None:
        """Toggle visibility of localization points."""
        if self.pointsVisible:
            self.hide_points()
        else:
            self.show_points(self.pointColor, self.pointSize, self.pointOpacity, self.displayMode)

    def _connect_frame_changes(self) -> None:
        """Connect to FLIKA window's frame change signal."""
        try:
            # Disconnect any existing connection
            self._disconnect_frame_changes()

            # Connect to the time line position change signal
            if hasattr(self.imageWindow, 'imageview') and hasattr(self.imageWindow.imageview, 'timeLine'):
                self.frameChangeConnection = self.imageWindow.imageview.timeLine.sigPositionChanged.connect(
                    self._on_frame_changed
                )
                logger.debug("Connected to frame change signal")
            else:
                logger.warning("Could not connect to frame change signal")

        except Exception as e:
            logger.error(f"Error connecting to frame changes: {e}")

    def _disconnect_frame_changes(self) -> None:
        """Disconnect from frame change signal."""
        try:
            if self.frameChangeConnection is not None:
                if hasattr(self.imageWindow, 'imageview') and hasattr(self.imageWindow.imageview, 'timeLine'):
                    self.imageWindow.imageview.timeLine.sigPositionChanged.disconnect(
                        self._on_frame_changed
                    )
                self.frameChangeConnection = None
                logger.debug("Disconnected from frame change signal")
        except Exception as e:
            logger.debug(f"Error disconnecting from frame changes: {e}")

    def _on_frame_changed(self) -> None:
        """Handle frame change event from FLIKA."""
        try:
            # Get new frame number
            new_frame = self.imageWindow.currentIndex

            if new_frame != self.currentFrame:
                self.currentFrame = new_frame
                logger.debug(f"Frame changed to {new_frame}, updating display")

                # Update display for new frame
                self._update_frame_display()

        except Exception as e:
            logger.error(f"Error handling frame change: {e}")

    def _update_frame_display(self) -> None:
        """Update the displayed points for the current frame."""
        try:
            if not self.pointsVisible or self.displayMode != 'current_frame':
                return

            if 'frame' not in self.localizations:
                logger.warning("No frame information in localizations")
                return

            # Filter localizations for current frame
            frame_mask = self.localizations['frame'] == self.currentFrame

            if not np.any(frame_mask):
                # No points in this frame - clear display
                if self.scatterPlot is not None:
                    self.scatterPlot.clear()
                logger.debug(f"No localizations in frame {self.currentFrame}")
                return

            # Get coordinates for this frame
            x_coords_nm = self.localizations['x'][frame_mask]
            y_coords_nm = self.localizations['y'][frame_mask]

            # Convert to pixels
            x_coords = x_coords_nm / self.pixel_size
            y_coords = y_coords_nm / self.pixel_size

            # Update scatter plot data
            if self.scatterPlot is not None:
                # For colored modes, we need to regenerate colors
                if self.pointColor == 'intensity' and 'intensity' in self.localizations:
                    intensities = self.localizations['intensity'][frame_mask]
                    brushes = self._get_intensity_colors(intensities, self.pointOpacity)
                    self.scatterPlot.setData(x=x_coords, y=y_coords, brush=brushes)
                elif self.pointColor == 'frame' and 'frame' in self.localizations:
                    frames = self.localizations['frame'][frame_mask]
                    brushes = self._get_frame_colors(frames, self.pointOpacity)
                    self.scatterPlot.setData(x=x_coords, y=y_coords, brush=brushes)
                elif self.pointColor == 'uncertainty' and 'uncertainty' in self.localizations:
                    uncertainties = self.localizations['uncertainty'][frame_mask]
                    brushes = self._get_uncertainty_colors(uncertainties, self.pointOpacity)
                    self.scatterPlot.setData(x=x_coords, y=y_coords, brush=brushes)
                else:
                    # Solid color - just update positions
                    self.scatterPlot.setData(x=x_coords, y=y_coords)

                logger.debug(f"Updated display with {len(x_coords)} points for frame {self.currentFrame}")

        except Exception as e:
            logger.error(f"Error updating frame display: {e}")

    def _setup_frame_display(self, color_mode: str, point_size: int, opacity: int) -> None:
        """Setup empty frame display and connect to frame changes."""
        # Create empty scatter plot
        brush = pg.mkBrush(30, 255, 35, opacity)
        self.scatterPlot = pg.ScatterPlotItem(
            size=point_size,
            pen=None,
            brush=brush
        )
        self.imageWindow.imageview.view.addItem(self.scatterPlot)
        self.pointsVisible = True

        # Connect to frame changes
        self._connect_frame_changes()

    def update_appearance(self, color_mode: Optional[str] = None,
                         point_size: Optional[int] = None,
                         opacity: Optional[int] = None,
                         display_mode: Optional[str] = None) -> None:
        """
        Update the appearance of displayed points.

        Parameters
        ----------
        color_mode : str, optional
            New color mode
        point_size : int, optional
            New point size
        opacity : int, optional
            New opacity
        display_mode : str, optional
            New display mode ('all' or 'current_frame')
        """
        if color_mode is not None:
            self.pointColor = color_mode
        if point_size is not None:
            self.pointSize = point_size
        if opacity is not None:
            self.pointOpacity = opacity
        if display_mode is not None:
            self.displayMode = display_mode

        # Refresh display if points are visible
        if self.pointsVisible:
            self.hide_points()
            self.show_points(self.pointColor, self.pointSize, self.pointOpacity, self.displayMode)

    def show_control_window(self) -> None:
        """Show control window for adjusting display settings."""
        try:
            if self.controlWindow is not None:
                self.controlWindow.show()
                self.controlWindow.raise_()
                self.controlWindow.activateWindow()
                return

            # Create control window
            self.controlWindow = QMainWindow()
            self.controlWindow.setWindowTitle('Localization Display Controls')
            self.controlWindow.resize(350, 300)

            # Create dock area
            area = DockArea()
            self.controlWindow.setCentralWidget(area)

            # Create dock
            dock = Dock("Display Settings", size=(350, 300))
            area.addDock(dock)

            # Create layout widget
            widget = pg.LayoutWidget()

            # Color mode selector
            row = 0
            widget.addWidget(QLabel("Color Mode:"), row=row, col=0)
            self.colorMode_combo = QComboBox()
            self.colorMode_combo.addItems(['green', 'red', 'white', 'intensity',
                                          'frame', 'uncertainty'])
            self.colorMode_combo.setCurrentText(self.pointColor)
            self.colorMode_combo.currentTextChanged.connect(self._on_color_change)
            widget.addWidget(self.colorMode_combo, row=row, col=1)

            # Point size slider
            row += 1
            widget.addWidget(QLabel("Point Size:"), row=row, col=0)
            self.pointSize_spin = QSpinBox()
            self.pointSize_spin.setRange(1, 20)
            self.pointSize_spin.setValue(self.pointSize)
            self.pointSize_spin.valueChanged.connect(self._on_size_change)
            widget.addWidget(self.pointSize_spin, row=row, col=1)

            # Opacity slider
            row += 1
            widget.addWidget(QLabel("Opacity:"), row=row, col=0)
            self.opacity_spin = QSpinBox()
            self.opacity_spin.setRange(0, 255)
            self.opacity_spin.setValue(self.pointOpacity)
            self.opacity_spin.valueChanged.connect(self._on_opacity_change)
            widget.addWidget(self.opacity_spin, row=row, col=1)

            # Pixel size control
            row += 1
            widget.addWidget(QLabel("Pixel Size (nm):"), row=row, col=0)
            self.pixelSize_spin = QDoubleSpinBox()
            self.pixelSize_spin.setRange(1.0, 1000.0)
            self.pixelSize_spin.setValue(self.pixel_size)
            self.pixelSize_spin.setSingleStep(1.0)
            self.pixelSize_spin.setToolTip("Camera pixel size in nanometers")
            self.pixelSize_spin.valueChanged.connect(self._on_pixel_size_change)
            widget.addWidget(self.pixelSize_spin, row=row, col=1)

            # Display mode selector
            row += 1
            widget.addWidget(QLabel("Display Mode:"), row=row, col=0)
            self.displayMode_combo = QComboBox()
            self.displayMode_combo.addItems(['all', 'current_frame'])
            self.displayMode_combo.setCurrentText(self.displayMode)
            self.displayMode_combo.setToolTip("All frames or current frame only")
            self.displayMode_combo.currentTextChanged.connect(self._on_display_mode_change)
            widget.addWidget(self.displayMode_combo, row=row, col=1)

            # Visibility toggle
            row += 1
            self.visibility_button = QPushButton("Hide Points")
            self.visibility_button.clicked.connect(self._on_visibility_toggle)
            widget.addWidget(self.visibility_button, row=row, col=0, colspan=2)

            # Statistics display
            row += 1
            widget.addWidget(QLabel("Statistics:"), row=row, col=0, colspan=2)

            row += 1
            stats_text = self._get_statistics_text()
            self.stats_label = QLabel(stats_text)
            self.stats_label.setWordWrap(True)
            widget.addWidget(self.stats_label, row=row, col=0, colspan=2, rowspan=3)

            # Add widget to dock
            dock.addWidget(widget)

            # Show window
            self.controlWindow.show()

            logger.debug("Control window displayed")

        except Exception as e:
            logger.error(f"Error showing control window: {e}")

    def _on_color_change(self, color_mode: str) -> None:
        """Handle color mode change."""
        self.update_appearance(color_mode=color_mode)

    def _on_size_change(self, size: int) -> None:
        """Handle point size change."""
        self.update_appearance(point_size=size)

    def _on_opacity_change(self, opacity: int) -> None:
        """Handle opacity change."""
        self.update_appearance(opacity=opacity)

    def _on_pixel_size_change(self, pixel_size: float) -> None:
        """Handle pixel size change."""
        self.pixel_size = pixel_size
        # Refresh display with new pixel size
        if self.pointsVisible:
            self.hide_points()
            self.show_points(self.pointColor, self.pointSize, self.pointOpacity, self.displayMode)

    def _on_display_mode_change(self, display_mode: str) -> None:
        """Handle display mode change."""
        self.update_appearance(display_mode=display_mode)
        # Update status message
        if display_mode == 'current_frame':
            logger.info("Switched to current frame display mode")
        else:
            logger.info("Switched to all frames display mode")

    def _on_visibility_toggle(self) -> None:
        """Handle visibility toggle button."""
        self.toggle_points()
        if self.pointsVisible:
            self.visibility_button.setText("Hide Points")
        else:
            self.visibility_button.setText("Show Points")

    def _get_intensity_colors(self, intensities: np.ndarray,
                             opacity: int) -> List[QColor]:
        """
        Generate colors based on intensity values.

        Parameters
        ----------
        intensities : ndarray
            Intensity values
        opacity : int
            Opacity value (0-255)

        Returns
        -------
        brushes : list
            List of QColor brushes
        """
        # Normalize intensities to 0-1
        i_min, i_max = intensities.min(), intensities.max()
        if i_max > i_min:
            norm_intensities = (intensities - i_min) / (i_max - i_min)
        else:
            norm_intensities = np.ones_like(intensities)

        # Create colors (blue to red colormap)
        brushes = []
        for val in norm_intensities:
            r = int(255 * val)
            b = int(255 * (1 - val))
            g = int(128 * (1 - abs(2 * val - 1)))  # Peak at middle
            brushes.append(pg.mkBrush(r, g, b, opacity))

        return brushes

    def _get_frame_colors(self, frames: np.ndarray,
                         opacity: int) -> List[QColor]:
        """
        Generate colors based on frame number.

        Parameters
        ----------
        frames : ndarray
            Frame numbers
        opacity : int
            Opacity value (0-255)

        Returns
        -------
        brushes : list
            List of QColor brushes
        """
        # Normalize frames to 0-1
        f_min, f_max = frames.min(), frames.max()
        if f_max > f_min:
            norm_frames = (frames - f_min) / (f_max - f_min)
        else:
            norm_frames = np.ones_like(frames)

        # Create colors (rainbow colormap)
        brushes = []
        for val in norm_frames:
            hue = int(240 * (1 - val))  # Blue to red
            color = QColor.fromHsv(hue, 255, 255, opacity)
            brushes.append(pg.mkBrush(color))

        return brushes

    def _get_uncertainty_colors(self, uncertainties: np.ndarray,
                               opacity: int) -> List[QColor]:
        """
        Generate colors based on localization uncertainty.

        Lower uncertainty = green, higher uncertainty = red

        Parameters
        ----------
        uncertainties : ndarray
            Uncertainty values (nm)
        opacity : int
            Opacity value (0-255)

        Returns
        -------
        brushes : list
            List of QColor brushes
        """
        # Normalize uncertainties to 0-1
        u_min, u_max = uncertainties.min(), uncertainties.max()
        if u_max > u_min:
            norm_uncertainties = (uncertainties - u_min) / (u_max - u_min)
        else:
            norm_uncertainties = np.ones_like(uncertainties)

        # Create colors (green to red)
        brushes = []
        for val in norm_uncertainties:
            r = int(255 * val)
            g = int(255 * (1 - val))
            brushes.append(pg.mkBrush(r, g, 0, opacity))

        return brushes

    def _get_statistics_text(self) -> str:
        """Generate statistics text for display."""
        n_locs = len(self.localizations['x'])

        stats_lines = [
            f"Total Localizations: {n_locs}",
        ]

        if 'frame' in self.localizations:
            n_frames = len(np.unique(self.localizations['frame']))
            avg_per_frame = n_locs / n_frames if n_frames > 0 else 0
            stats_lines.append(f"Frames: {n_frames}")
            stats_lines.append(f"Avg/Frame: {avg_per_frame:.1f}")

        if 'intensity' in self.localizations:
            mean_int = np.mean(self.localizations['intensity'])
            stats_lines.append(f"Mean Intensity: {mean_int:.0f}")

        if 'uncertainty' in self.localizations:
            mean_unc = np.mean(self.localizations['uncertainty'])
            stats_lines.append(f"Mean Uncertainty: {mean_unc:.1f} nm")

        return '\n'.join(stats_lines)


class ThunderSTORM_DisplayLocalizations(BaseProcess_noPriorWindow):
    """
    FLIKA plugin to display ThunderSTORM localizations on image.

    This plugin allows you to load localization data (from CSV) and display
    it as scatter points on the current FLIKA image window.
    """

    def __init__(self):
        super().__init__()
        self.display: Optional[LocalizationDisplay] = None

    def gui(self):
        """Create GUI for displaying localizations."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # Load from file checkbox
        load_from_file = CheckBox()
        load_from_file.setChecked(False)
        load_from_file.setToolTip("Load localizations from CSV file instead of using window data")

        # Color mode
        color_mode = ComboBox()
        for mode in ['green', 'red', 'white', 'intensity', 'frame', 'uncertainty']:
            color_mode.addItem(mode)

        # Point size
        point_size = QSpinBox()
        point_size.setRange(1, 20)
        point_size.setValue(4)

        # Opacity
        opacity = QSpinBox()
        opacity.setRange(0, 255)
        opacity.setValue(200)

        # Pixel size
        pixel_size_nm = QDoubleSpinBox()
        pixel_size_nm.setRange(1.0, 1000.0)
        pixel_size_nm.setValue(100.0)
        pixel_size_nm.setSingleStep(1.0)
        pixel_size_nm.setDecimals(1)
        pixel_size_nm.setToolTip("Camera pixel size in nanometers")

        # Display mode
        display_mode = ComboBox()
        display_mode.addItem('all')
        display_mode.addItem('current_frame')
        display_mode.setValue('all')
        display_mode.setToolTip("Display all frames or current frame only")

        # Show controls checkbox
        show_controls = CheckBox()
        show_controls.setChecked(True)

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'load_from_file', 'string': 'Load from CSV File',
                          'object': load_from_file})
        self.items.append({'name': 'color_mode', 'string': 'Color Mode',
                          'object': color_mode})
        self.items.append({'name': 'point_size', 'string': 'Point Size',
                          'object': point_size})
        self.items.append({'name': 'opacity', 'string': 'Opacity',
                          'object': opacity})
        self.items.append({'name': 'pixel_size_nm', 'string': 'Pixel Size (nm)',
                          'object': pixel_size_nm})
        self.items.append({'name': 'display_mode', 'string': 'Display Mode',
                          'object': display_mode})
        self.items.append({'name': 'show_controls', 'string': 'Show Controls',
                          'object': show_controls})

        super().gui()

    def __call__(self, window, load_from_file, color_mode, point_size, opacity,
                pixel_size_nm, display_mode, show_controls, keepSourceWindow=False):
        """
        Display localizations on image window.

        Parameters
        ----------
        window : Window
            FLIKA image window
        load_from_file : bool
            Load localizations from CSV file
        color_mode : str
            Color mode for points
        point_size : int
            Size of points
        opacity : int
            Opacity of points (0-255)
        pixel_size_nm : float
            Camera pixel size in nanometers
        display_mode : str
            'all' or 'current_frame'
        show_controls : bool
            Show control window
        """
        try:
            localizations = None

            # Check if we should load from file or use existing data
            if load_from_file:
                # Browse for CSV file
                from qtpy.QtWidgets import QFileDialog
                filename, _ = QFileDialog.getOpenFileName(
                    None,
                    "Select Localizations CSV File",
                    "",
                    "CSV Files (*.csv);;All Files (*.*)"
                )

                if not filename:
                    g.m.statusBar().showMessage("No file selected")
                    return window

                # Load localizations from CSV
                try:
                    from .thunderstorm_python import utils
                except ImportError:
                    from thunderstorm_python import utils

                localizations = utils.load_localizations_csv(filename)

                # Attach to window for future use
                window.thunderstorm_localizations = localizations

                g.m.statusBar().showMessage(f"Loaded {len(localizations['x'])} localizations from {filename}")

            else:
                # Try to use existing localizations from window
                if not hasattr(window, 'thunderstorm_localizations'):
                    # No localizations found - offer to load from file
                    reply = QMessageBox.question(
                        None,
                        "No Localizations Found",
                        "No localizations found for this window.\n\n"
                        "Would you like to load localizations from a CSV file?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )

                    if reply == QMessageBox.Yes:
                        from qtpy.QtWidgets import QFileDialog
                        filename, _ = QFileDialog.getOpenFileName(
                            None,
                            "Select Localizations CSV File",
                            "",
                            "CSV Files (*.csv);;All Files (*.*)"
                        )

                        if not filename:
                            g.m.statusBar().showMessage("No file selected")
                            return window

                        # Load localizations from CSV
                        try:
                            from .thunderstorm_python import utils
                        except ImportError:
                            from thunderstorm_python import utils

                        localizations = utils.load_localizations_csv(filename)

                        # Attach to window
                        window.thunderstorm_localizations = localizations

                        g.m.statusBar().showMessage(f"Loaded {len(localizations['x'])} localizations")
                    else:
                        return window
                else:
                    localizations = window.thunderstorm_localizations

            # Check if we have valid localizations
            if localizations is None or len(localizations.get('x', [])) == 0:
                QMessageBox.warning(None, "No Data",
                                  "No localization data available to display.")
                return window

            # Create display with pixel size
            self.display = LocalizationDisplay(window, localizations, pixel_size=pixel_size_nm)

            # Show points
            self.display.show_points(color_mode, point_size, opacity, display_mode)

            # Show controls if requested
            if show_controls:
                self.display.show_control_window()

            # Store display in window for later access
            window.thunderstorm_display = self.display

            g.m.statusBar().showMessage(
                f"Displayed {len(localizations['x'])} localizations")

        except Exception as e:
            logger.error(f"Error in display localizations: {e}")
            QMessageBox.critical(None, "Error", f"Error displaying localizations: {str(e)}")
            import traceback
            traceback.print_exc()

        return window


class ThunderSTORM_LoadAndDisplay(BaseProcess_noPriorWindow):
    """
    Load localizations from CSV and display on image.

    Simplified plugin specifically for loading saved localization files.
    """

    def __init__(self):
        super().__init__()
        self.display: Optional[LocalizationDisplay] = None

    def gui(self):
        """Create GUI for loading and displaying."""
        self.gui_reset()

        # Window selector
        window_selector = WindowSelector()

        # Color mode
        color_mode = ComboBox()
        for mode in ['green', 'red', 'white', 'intensity', 'frame', 'uncertainty']:
            color_mode.addItem(mode)
        color_mode.setValue('green')

        # Point size
        point_size = QSpinBox()
        point_size.setRange(1, 20)
        point_size.setValue(4)

        # Pixel size
        pixel_size_nm = QDoubleSpinBox()
        pixel_size_nm.setRange(1.0, 1000.0)
        pixel_size_nm.setValue(100.0)
        pixel_size_nm.setSingleStep(1.0)
        pixel_size_nm.setDecimals(1)
        pixel_size_nm.setToolTip("Camera pixel size in nanometers")

        # Display mode
        display_mode = ComboBox()
        display_mode.addItem('all')
        display_mode.addItem('current_frame')
        display_mode.setValue('all')
        display_mode.setToolTip("Display all frames or current frame only")

        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'color_mode', 'string': 'Color Mode',
                          'object': color_mode})
        self.items.append({'name': 'point_size', 'string': 'Point Size',
                          'object': point_size})
        self.items.append({'name': 'pixel_size_nm', 'string': 'Pixel Size (nm)',
                          'object': pixel_size_nm})
        self.items.append({'name': 'display_mode', 'string': 'Display Mode',
                          'object': display_mode})

        super().gui()

    def __call__(self, window, color_mode, point_size, pixel_size_nm, display_mode,
                keepSourceWindow=False):
        """
        Load CSV and display localizations.

        Parameters
        ----------
        window : Window
            FLIKA image window
        color_mode : str
            Color mode for points
        point_size : int
            Size of points
        pixel_size_nm : float
            Camera pixel size in nanometers
        display_mode : str
            'all' or 'current_frame'
        """
        try:
            # Browse for CSV file
            from qtpy.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getOpenFileName(
                None,
                "Select Localizations CSV File",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )

            if not filename:
                g.m.statusBar().showMessage("No file selected")
                return window

            # Load localizations from CSV
            try:
                from .thunderstorm_python import utils
            except ImportError:
                from thunderstorm_python import utils

            g.m.statusBar().showMessage(f"Loading localizations from {filename}...")
            localizations = utils.load_localizations_csv(filename)

            if len(localizations['x']) == 0:
                QMessageBox.warning(None, "No Data",
                                  "No localizations found in the CSV file.")
                return window

            # Attach to window
            window.thunderstorm_localizations = localizations

            # Create display with pixel size
            self.display = LocalizationDisplay(window, localizations, pixel_size=pixel_size_nm)

            # Show points
            self.display.show_points(color_mode, point_size, 200, display_mode)

            # Show control window
            self.display.show_control_window()

            # Store display in window
            window.thunderstorm_display = self.display

            g.m.statusBar().showMessage(
                f"Loaded and displayed {len(localizations['x'])} localizations")

            QMessageBox.information(
                None,
                "Localizations Loaded",
                f"Successfully loaded and displayed {len(localizations['x'])} localizations\n"
                f"from {filename}"
            )

        except Exception as e:
            logger.error(f"Error loading and displaying: {e}")
            QMessageBox.critical(None, "Error",
                               f"Error loading localizations: {str(e)}")
            import traceback
            traceback.print_exc()

        return window


# Create plugin instances
thunderstorm_display_localizations = ThunderSTORM_DisplayLocalizations()
thunderstorm_load_and_display = ThunderSTORM_LoadAndDisplay()
