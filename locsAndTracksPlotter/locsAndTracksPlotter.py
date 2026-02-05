#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIKA Plugin for Localizations and Tracks Plotting

This plugin provides comprehensive tools for analyzing and visualizing single-particle
tracking data from fluorescence microscopy experiments. It supports multiple file formats
and provides various analysis and visualization options.

Features:
- Load and display localization data from various formats
- Track visualization with customizable appearance
- Interactive filtering and analysis tools
- Export capabilities for processed data
- Integration with FLIKA image analysis platform

Created on Sat May 23 10:38:20 2020
@author: george.dickinson@gmail.com
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal, QPointF
from qtpy.QtGui import QColor, QPainter, QPen, QPainterPath
from qtpy.QtWidgets import (QMainWindow, QLabel, QPushButton, QLineEdit,
                           QFileDialog, QMessageBox, QGraphicsPathItem)
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from distutils.version import StrictVersion

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g

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

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=Warning)

# Enable Numba for performance if available
try:
    import numba
    pg.setConfigOption('useNumba', True)
    logging.getLogger(__name__).info("Numba acceleration enabled")
except ImportError:
    logging.getLogger(__name__).warning("Numba not available, performance may be reduced")

# Plugin modules
from .helperFunctions import dictFromList, gammaCorrect
from .io import FileSelector
from .joinTracks import JoinTracks
from .allTracksPlotter import AllTracksPlot
from .chartDock import ChartDock
from .trackPlotter import TrackPlot
from .flowerPlot import FlowerPlotWindow
from .diffusionPlot import DiffusionPlotWindow
from .trackWindow import TrackWindow
from .overlay import Overlay
from .roiZoomPlotter import ROIPLOT

# Set up logging
logger = logging.getLogger(__name__)

# Log PyQtGraph version for debugging
try:
    pg_version = getattr(pg, '__version__', 'unknown')
    logger.info(f"✓ PyQtGraph version: {pg_version}")

    # Test ScatterPlotItem capabilities
    test_scatter = pg.ScatterPlotItem()
    has_setData = hasattr(test_scatter, 'setData') and callable(test_scatter.setData)
    has_setPoints = hasattr(test_scatter, 'setPoints') and callable(test_scatter.setPoints)
    has_addPoints = hasattr(test_scatter, 'addPoints') and callable(test_scatter.addPoints)
    logger.info(f"✓ ScatterPlotItem methods - setData: {has_setData}, setPoints: {has_setPoints}, addPoints: {has_addPoints}")
except Exception as e:
    logger.warning(f"Could not detect PyQtGraph capabilities: {e}")


def set_scatter_data_compat(scatter_item, **kwargs):
    """
    Set scatter data with PyQtGraph version compatibility.

    Handles differences between PyQtGraph versions where some use
    setData() and others use setPoints().
    """
    try:
        if hasattr(scatter_item, 'setData') and callable(scatter_item.setData):
            scatter_item.setData(**kwargs)
        elif hasattr(scatter_item, 'setPoints') and callable(scatter_item.setPoints):
            scatter_item.setPoints(**kwargs)
        else:
            # Fallback - try setData anyway
            logger.warning("ScatterPlotItem has neither setData nor setPoints, attempting setData")
            scatter_item.setData(**kwargs)
    except Exception as e:
        logger.error(f"Error setting scatter data: {e}")
        logger.error(f"PyQtGraph version: {getattr(pg, '__version__', 'unknown')}")
        logger.error(f"kwargs: {kwargs.keys()}")
        raise


class ColorButton(QPushButton):
    """
    Custom Qt Widget to show and select colors.

    Left-clicking shows the color chooser, right-clicking resets to default.

    Signals:
        colorChanged: Emitted when color changes
    """

    colorChanged = Signal(object)

    def __init__(self, *args, color: Optional[str] = None, **kwargs):
        """
        Initialize ColorButton.

        Args:
            color: Default color as hex string (e.g., '#ff0000')
        """
        super().__init__(*args, **kwargs)
        self._color: Optional[str] = None
        self._default: Optional[str] = color
        self.pressed.connect(self._on_color_picker)
        self.setColor(self._default)
        logger.debug(f"ColorButton initialized with default color: {color}")

    def setColor(self, color: Optional[str]) -> None:
        """
        Set the button color.

        Args:
            color: Color as hex string or None for no color
        """
        if color != self._color:
            self._color = color
            self.colorChanged.emit(color)

        if self._color:
            self.setStyleSheet(f"background-color: {self._color};")
        else:
            self.setStyleSheet("")

    def color(self) -> Optional[str]:
        """Get the current color."""
        return self._color

    def _on_color_picker(self) -> None:
        """Show color picker dialog."""
        try:
            from qtpy.QtWidgets import QColorDialog

            dlg = QColorDialog(self)
            if self._color:
                dlg.setCurrentColor(QColor(self._color))

            if dlg.exec_():
                self.setColor(dlg.currentColor().name())
                logger.debug(f"Color selected: {self._color}")
        except Exception as e:
            logger.error(f"Error in color picker: {e}")

    def mousePressEvent(self, event) -> None:
        """Handle mouse press events."""
        if event.button() == Qt.RightButton:
            self.setColor(self._default)
            logger.debug("Color reset to default")
        return super().mousePressEvent(event)


class FilterOptions:
    """
    GUI for setting filter options for points and tracks.

    Provides controls for filtering data based on various criteria
    with support for sequential filtering operations.
    """

    def __init__(self, mainGUI: 'LocsAndTracksPlotter'):
        """
        Initialize filter options GUI.

        Args:
            mainGUI: Reference to main GUI instance
        """
        super().__init__()
        self.mainGUI = mainGUI
        self._setup_ui()
        logger.debug("FilterOptions initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window and layout
            self.win = QMainWindow()
            from pyqtgraph.dockarea import DockArea, Dock

            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.resize(550, 100)
            self.win.setWindowTitle('Filter')

            self.d1 = Dock("Filter Options", size=(550, 100))
            self.area.addDock(self.d1)

            # Create layout widget
            self.w1 = pg.LayoutWidget()

            # Filter controls
            self._create_filter_controls()
            self._create_buttons()
            self._layout_widgets()

            # Add to dock
            self.d1.addWidget(self.w1)

        except Exception as e:
            logger.error(f"Error setting up FilterOptions UI: {e}")

    def _create_filter_controls(self) -> None:
        """Create filter control widgets."""
        # Filter column selector
        self.filterCol_Box = pg.ComboBox()
        self.filtercols = {'None': 'None'}
        self.filterCol_Box.setItems(self.filtercols)

        # Filter operator selector
        self.filterOp_Box = pg.ComboBox()
        self.filterOps = {'=': '==', '<': '<', '>': '>', '!=': '!='}
        self.filterOp_Box.setItems(self.filterOps)

        # Filter value input
        self.filterValue_Box = QLineEdit()

        # Sequential filtering checkbox
        self.sequentialFilter_checkbox = CheckBox()
        self.sequentialFilter_checkbox.setChecked(False)
        self.sequentialFilter_checkbox.stateChanged.connect(self._set_sequential_filter)

        # Labels
        self.filterCol_label = QLabel('Filter column')
        self.filterVal_label = QLabel('Value')
        self.filterOp_label = QLabel('Operator')
        self.sequentialFilter_label = QLabel('Allow sequential filtering')

    def _create_buttons(self) -> None:
        """Create action buttons."""
        self.filterData_button = QPushButton('Filter')
        self.filterData_button.pressed.connect(self.mainGUI.filterData)

        self.clearFilterData_button = QPushButton('Clear Filter')
        self.clearFilterData_button.pressed.connect(self.mainGUI.clearFilterData)

        self.ROIFilterData_button = QPushButton('Filter by ROI(s)')
        self.ROIFilterData_button.pressed.connect(self.mainGUI.ROIFilterData)

        self.clearROIFilterData_button = QPushButton('Clear ROI Filter')
        self.clearROIFilterData_button.pressed.connect(self.mainGUI.clearROIFilterData)

    def _layout_widgets(self) -> None:
        """Arrange widgets in layout."""
        # Row 0
        self.w1.addWidget(self.filterCol_label, row=0, col=0)
        self.w1.addWidget(self.filterCol_Box, row=0, col=1)
        self.w1.addWidget(self.filterOp_label, row=0, col=2)
        self.w1.addWidget(self.filterOp_Box, row=0, col=3)
        self.w1.addWidget(self.filterVal_label, row=0, col=4)
        self.w1.addWidget(self.filterValue_Box, row=0, col=5)

        # Row 1
        self.w1.addWidget(self.filterData_button, row=1, col=0)
        self.w1.addWidget(self.clearFilterData_button, row=1, col=1)
        self.w1.addWidget(self.sequentialFilter_label, row=1, col=2)
        self.w1.addWidget(self.sequentialFilter_checkbox, row=1, col=3)

        # Row 3
        self.w1.addWidget(self.ROIFilterData_button, row=3, col=0)
        self.w1.addWidget(self.clearROIFilterData_button, row=3, col=1)

    def _set_sequential_filter(self) -> None:
        """Handle sequential filter checkbox change."""
        self.mainGUI.sequentialFiltering = self.sequentialFilter_checkbox.isChecked()
        logger.debug(f"Sequential filtering set to: {self.mainGUI.sequentialFiltering}")

    def show(self) -> None:
        """Show the filter options window."""
        self.win.show()

    def close(self) -> None:
        """Close the filter options window."""
        self.win.close()

    def hide(self) -> None:
        """Hide the filter options window."""
        self.win.hide()


class TrackPlotOptions:
    """
    GUI for configuring track and point display options.

    Provides comprehensive controls for customizing the appearance
    of tracks and points in the visualization.
    """

    def __init__(self, mainGUI: 'LocsAndTracksPlotter'):
        """
        Initialize track plot options GUI.

        Args:
            mainGUI: Reference to main GUI instance
        """
        super().__init__()
        self.mainGUI = mainGUI
        self._setup_ui()
        logger.debug("TrackPlotOptions initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window
            self.win = QMainWindow()
            from pyqtgraph.dockarea import DockArea, Dock

            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.setWindowTitle('Display Options')

            # Create docks
            self._create_docks()
            self._create_point_options()
            self._create_track_options()
            self._create_recording_options()
            self._create_background_options()

        except Exception as e:
            logger.error(f"Error setting up TrackPlotOptions UI: {e}")

    def _create_docks(self) -> None:
        """Create dock widgets."""
        from pyqtgraph.dockarea import Dock

        self.d0 = Dock("Point Options")
        self.d1 = Dock("Track Options")
        self.d2 = Dock("Recording Parameters")
        self.d3 = Dock("Background subtraction")

        self.area.addDock(self.d0)
        self.area.addDock(self.d1)
        self.area.addDock(self.d2)
        self.area.addDock(self.d3)

    def _create_point_options(self) -> None:
        """Create point display options."""
        self.w0 = pg.LayoutWidget()

        # Point color selector
        self.pointColour_Box = pg.ComboBox()
        self.pointColours = {
            'green': QColor(Qt.green),
            'red': QColor(Qt.red),
            'blue': QColor(Qt.blue)
        }
        self.pointColour_Box.setItems(self.pointColours)
        self.pointColour_Box_label = QLabel('Point Colour')

        # Point size selector
        self.pointSize_selector = pg.SpinBox(value=5, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(0)
        self.pointSize_selector.setMaximum(100)
        self.pointSize_selector_label = QLabel('Point Size')

        # Unlinked point color
        self.unlinkedpointColour_Box = pg.ComboBox()
        self.unlinkedpointColours = {
            'blue': QColor(Qt.blue),
            'green': QColor(Qt.green),
            'red': QColor(Qt.red)
        }
        self.unlinkedpointColour_Box.setItems(self.unlinkedpointColours)
        self.unlinkedpointColour_Box_label = QLabel('Unlinked Point Colour')

        # Layout point options
        self.w0.addWidget(self.unlinkedpointColour_Box_label, row=0, col=0)
        self.w0.addWidget(self.unlinkedpointColour_Box, row=0, col=1)
        self.w0.addWidget(self.pointColour_Box_label, row=1, col=0)
        self.w0.addWidget(self.pointColour_Box, row=1, col=1)
        self.w0.addWidget(self.pointSize_selector_label, row=2, col=0)
        self.w0.addWidget(self.pointSize_selector, row=2, col=1)

        self.d0.addWidget(self.w0)

    def _create_track_options(self) -> None:
        """Create track display options."""
        self.w1 = pg.LayoutWidget()

        # Track color controls
        self.trackColourCol_Box = pg.ComboBox()
        self.trackcolourcols = {'None': 'None'}
        self.trackColourCol_Box.setItems(self.trackcolourcols)
        self.trackColourCol_Box_label = QLabel('Colour By')
        self.trackColourCol_Box.currentIndexChanged.connect(self.update)

        # Color map controls
        self.colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.colourMap_Box.setItems(self.colourMaps)
        self.colourMap_Box_label = QLabel('Colour Map')

        # Default track color
        self.trackDefaultColour_Box = pg.ComboBox()
        self.trackdefaultcolours = {
            'green': Qt.green,
            'red': Qt.red,
            'blue': Qt.blue
        }
        self.trackDefaultColour_Box.setItems(self.trackdefaultcolours)
        self.trackDefaultColour_Box_label = QLabel('Track Default Colour')

        # Line size
        self.lineSize_selector = pg.SpinBox(value=2, int=True)
        self.lineSize_selector.setSingleStep(1)
        self.lineSize_selector.setMinimum(0)
        self.lineSize_selector.setMaximum(100)
        self.lineSize_selector_label = QLabel('Line Size')

        # Checkboxes
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(False)
        self.trackColour_checkbox_label = QLabel('Set Track Colour')

        self.matplotCM_checkbox = CheckBox()
        self.matplotCM_checkbox.stateChanged.connect(self.mainGUI.setColourMap)
        self.matplotCM_checkbox.setChecked(False)
        self.matplotCM_checkbox_label = QLabel('Use Matplotlib Colour Map')

        # Threshold options
        self.threshold_checkbox = CheckBox()
        self.threshold_checkbox.setChecked(False)
        self.threshold_checkbox_label = QLabel('Colour By Threshold (overrides above)')
        self.threshold_checkbox.stateChanged.connect(self.update)

        self.threshValue_selector = pg.SpinBox(value=20, int=False)
        self.threshValue_selector.setSingleStep(0.1)
        self.threshValue_selector.setMinimum(0)
        self.threshValue_selector.setMaximum(1000)
        self.threshValue_selector_label = QLabel('Threshold Value')
        self.threshValue_selector.valueChanged.connect(self.update)

        # Color buttons for threshold
        self.aboveColour_button = ColorButton(color='#00fdff')
        self.aboveColour_button_label = QLabel('Above Colour (click to set)')
        self.aboveColour_button.pressed.connect(self._set_above_colour)

        self.belowColour_button = ColorButton(color='#ff40ff')
        self.belowColour_button_label = QLabel('Below Colour (click to set)')
        self.belowColour_button.pressed.connect(self._set_below_colour)

        # Layout track options
        self._layout_track_widgets()
        self.d1.addWidget(self.w1)

    def _layout_track_widgets(self) -> None:
        """Layout track option widgets."""
        self.w1.addWidget(self.trackColour_checkbox_label, row=1, col=0)
        self.w1.addWidget(self.trackColour_checkbox, row=1, col=1)
        self.w1.addWidget(self.trackColourCol_Box_label, row=2, col=0)
        self.w1.addWidget(self.trackColourCol_Box, row=2, col=1)
        self.w1.addWidget(self.colourMap_Box_label, row=3, col=0)
        self.w1.addWidget(self.colourMap_Box, row=3, col=1)
        self.w1.addWidget(self.matplotCM_checkbox_label, row=4, col=0)
        self.w1.addWidget(self.matplotCM_checkbox, row=4, col=1)
        self.w1.addWidget(self.trackDefaultColour_Box_label, row=5, col=0)
        self.w1.addWidget(self.trackDefaultColour_Box, row=5, col=1)
        self.w1.addWidget(self.lineSize_selector_label, row=6, col=0)
        self.w1.addWidget(self.lineSize_selector, row=6, col=1)
        self.w1.addWidget(self.threshold_checkbox_label, row=7, col=0)
        self.w1.addWidget(self.threshold_checkbox, row=7, col=1)
        self.w1.addWidget(self.threshValue_selector_label, row=8, col=0)
        self.w1.addWidget(self.threshValue_selector, row=8, col=1)
        self.w1.addWidget(self.aboveColour_button_label, row=9, col=0)
        self.w1.addWidget(self.aboveColour_button, row=9, col=1)
        self.w1.addWidget(self.belowColour_button_label, row=10, col=0)
        self.w1.addWidget(self.belowColour_button, row=10, col=1)

    def _create_recording_options(self) -> None:
        """Create recording parameter options."""
        self.w2 = pg.LayoutWidget()

        # Frame length
        self.frameLength_selector = pg.SpinBox(value=10, int=True)
        self.frameLength_selector.setSingleStep(10)
        self.frameLength_selector.setMinimum(1)
        self.frameLength_selector.setMaximum(100000)
        self.frameLength_selector_label = QLabel('milliseconds per frame')

        # Pixel size
        self.pixelSize_selector = pg.SpinBox(value=108, int=True)
        self.pixelSize_selector.setSingleStep(1)
        self.pixelSize_selector.setMinimum(1)
        self.pixelSize_selector.setMaximum(10000)
        self.pixelSize_selector_label = QLabel('nanometers per pixel')

        # Layout recording options
        self.w2.addWidget(self.frameLength_selector_label, row=1, col=0)
        self.w2.addWidget(self.frameLength_selector, row=1, col=1)
        self.w2.addWidget(self.pixelSize_selector_label, row=2, col=0)
        self.w2.addWidget(self.pixelSize_selector, row=2, col=1)

        self.d2.addWidget(self.w2)

    def _create_background_options(self) -> None:
        """Create background subtraction options."""
        self.w3 = pg.LayoutWidget()

        # Intensity choice
        self.intensityChoice_Box = pg.ComboBox()
        self.intensityChoice = {
            'intensity': 'intensity',
            'intensity - mean roi1': 'intensity - mean roi1',
            'intensity_roiOnMeanXY': 'intensity_roiOnMeanXY',
            'intensity_roiOnMeanXY - mean roi1': 'intensity_roiOnMeanXY - mean roi1',
            'intensity_roiOnMeanXY - mean roi1 and black': 'intensity_roiOnMeanXY - mean roi1 and black',
            'intensity_roiOnMeanXY - smoothed roi_1': 'intensity_roiOnMeanXY - smoothed roi_1',
            'intensity - smoothed roi_1': 'intensity - smoothed roi_1'
        }
        self.intensityChoice_Box.setItems(self.intensityChoice)
        self.intensityChoice_Box_label = QLabel('Intensity plot data')

        # Background subtraction
        self.backgroundSubtract_checkbox = CheckBox()
        self.backgroundSubtract_checkbox.setChecked(False)
        self.backgroundSubtract_label = QLabel('Subtract Background')

        self.background_selector = pg.SpinBox(value=0, int=True)
        self.background_selector.setSingleStep(1)
        self.background_selector.setMinimum(0)
        self.background_selector.setMaximum(10000)
        self.background_selector_label = QLabel('background value')

        self.estimatedCameraBlack = QLabel('')
        self.estimatedCameraBlack_label = QLabel('estimated camera black')

        # Layout background options
        self.w3.addWidget(self.intensityChoice_Box_label, row=0, col=0)
        self.w3.addWidget(self.intensityChoice_Box, row=0, col=1)
        self.w3.addWidget(self.backgroundSubtract_label, row=1, col=0)
        self.w3.addWidget(self.backgroundSubtract_checkbox, row=1, col=1)
        self.w3.addWidget(self.background_selector_label, row=2, col=0)
        self.w3.addWidget(self.background_selector, row=2, col=1)
        self.w3.addWidget(self.estimatedCameraBlack_label, row=3, col=0)
        self.w3.addWidget(self.estimatedCameraBlack, row=3, col=1)

        self.d3.addWidget(self.w3)

    def _set_above_colour(self) -> None:
        """Handle above threshold color setting."""
        self.update()
        threshold = self.threshValue_selector.value()
        color = self.aboveColour_button.color()
        logger.info(f'Above threshold {threshold} colour set to {color}')

    def _set_below_colour(self) -> None:
        """Handle below threshold color setting."""
        self.update()
        threshold = self.threshValue_selector.value()
        color = self.belowColour_button.color()
        logger.info(f'Below threshold {threshold} colour set to {color}')

    def update(self) -> None:
        """Update threshold colors based on current settings."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return

            col_name = self.trackColourCol_Box.value()
            if col_name == 'None' or col_name not in self.mainGUI.data.columns:
                return

            thresh = self.threshValue_selector.value()
            below_colour = QColor(self.belowColour_button.color())
            above_colour = QColor(self.aboveColour_button.color())

            self.mainGUI.data['threshColour'] = np.where(
                self.mainGUI.data[col_name] > thresh,
                above_colour,
                below_colour
            )
            logger.debug(f"Threshold colors updated for column {col_name}")

        except Exception as e:
            logger.error(f"Error updating threshold colors: {e}")

    def show(self) -> None:
        """Show the track plot options window."""
        self.win.show()

    def close(self) -> None:
        """Close the track plot options window."""
        self.win.close()

    def hide(self) -> None:
        """Hide the track plot options window."""
        self.win.hide()


class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    Main plugin class for plotting localization and track data.

    This class provides the primary interface for loading, visualizing,
    and analyzing single-particle tracking data in FLIKA.

    Features:
    - Multiple file format support (CSV, JSON, ThunderSTORM)
    - Interactive track visualization
    - Comprehensive filtering and analysis tools
    - Export capabilities
    - Integration with various analysis modules
    """

    def __init__(self):
        """Initialize the plugin with default settings."""
        logger.info("Initializing LocsAndTracksPlotter")

        # Initialize settings
        self._initialize_settings()

        # Call parent initialization
        BaseProcess_noPriorWindow.__init__(self)

        logger.debug("LocsAndTracksPlotter initialization complete")

    def _initialize_settings(self) -> None:
        """Initialize or verify plugin settings."""
        try:
            if (g.settings.get('locsAndTracksPlotter') is None or
                'set_track_colour' not in g.settings.get('locsAndTracksPlotter', {})):

                default_settings = {
                    'filename': '',
                    'filetype': 'flika',
                    'pixelSize': 108,
                    'set_track_colour': False,
                }
                g.settings['locsAndTracksPlotter'] = default_settings
                logger.debug("Default settings initialized")
            else:
                logger.debug("Using existing settings")

        except Exception as e:
            logger.error(f"Error initializing settings: {e}")

    def __call__(self, filename: str, filetype: str, pixelSize: int,
                 set_track_colour: bool, keepSourceWindow: bool = False) -> None:
        """
        Main entry point for the plugin.

        Args:
            filename: Path to the data file
            filetype: Type of file ('flika', 'thunderstorm', 'xy', 'json')
            pixelSize: Pixel size in nanometers
            set_track_colour: Whether to color tracks by ID
            keepSourceWindow: Whether to keep the source window open
        """
        try:
            logger.info(f"Loading data from {filename} (type: {filetype})")

            # Save parameters to settings
            settings = g.settings['locsAndTracksPlotter']
            settings.update({
                'filename': filename,
                'filetype': filetype,
                'pixelSize': pixelSize,
                'set_track_colour': set_track_colour
            })

            # Show status message
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage("Plotting data...")

            logger.debug("Plugin call completed successfully")

        except Exception as e:
            logger.error(f"Error in plugin call: {e}")
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(f"Error: {str(e)}")

    def closeEvent(self, event) -> None:
        """
        Handle window close event.

        Args:
            event: Close event object
        """
        try:
            logger.info("Closing LocsAndTracksPlotter")
            self.clearPlots()
            BaseProcess_noPriorWindow.closeEvent(self, event)
        except Exception as e:
            logger.error(f"Error during close event: {e}")

    def gui(self) -> None:
        """Set up the graphical user interface."""
        try:
            logger.debug("Setting up GUI")

            # Initialize instance variables
            self._initialize_variables()

            # Initialize sub-windows
            self._initialize_windows()

            # Set up main GUI
            self._setup_main_gui()

            # Call parent GUI setup
            super().gui()

            logger.debug("GUI setup completed")

        except Exception as e:
            logger.error(f"Error setting up GUI: {e}")

    def _initialize_variables(self) -> None:
        """Initialize instance variables."""
        # File and data variables
        self.filename: str = ''
        self.filetype: str = 'flika'
        self.pixelSize: Optional[int] = None
        self.data: Optional[pd.DataFrame] = None
        self.data_unlinked: Optional[pd.DataFrame] = None
        self.filteredData: Optional[pd.DataFrame] = None

        # Plot and display variables
        self.plotWindow: Optional[Window] = None
        self.pathitems: List = []
        self.useFilteredData: bool = False
        self.useFilteredTracks: bool = False
        self.useMatplotCM: bool = False
        self.selectedTrack: Optional[int] = None
        self.displayTrack: Optional[int] = None
        self.unlinkedPoints: Optional[pd.DataFrame] = None
        self.displayUnlinkedPoints: bool = False
        self.estimatedCameraBlackLevel: int = 0

        # Window display flags
        self.displayCharts: bool = False
        self.displayDiffusionPlot: bool = False
        self.displayROIplot: bool = False
        self.displayTrackPlotOptions: bool = False
        self.displayOverlay: bool = False

        # Sub-window references
        self.chartWindow: Optional[ChartDock] = None
        self.diffusionWindow: Optional[DiffusionPlotWindow] = None

        # Filter settings
        self.sequentialFiltering: bool = False

        # Expected columns for data validation
        self.expectedColumns = [
            'frame', 'track_number', 'x', 'y', 'intensity', 'zeroed_X', 'zeroed_Y',
            'lagNumber', 'distanceFromOrigin', 'dy-dt: distance', 'radius_gyration',
            'asymmetry', 'skewness', 'kurtosis', 'fracDimension', 'netDispl',
            'Straight', 'Experiment', 'SVM', 'nnDist_inFrame', 'n_segments', 'lag',
            'meanLag', 'track_length', 'radius_gyration_scaled',
            'radius_gyration_scaled_nSegments', 'radius_gyration_scaled_trackLength',
            'roi_1', 'camera black estimate', 'd_squared', 'lag_squared', 'dt',
            'velocity', 'direction_Relative_To_Origin', 'meanVelocity',
            'intensity - mean roi1', 'intensity - mean roi1 and black',
            'nnCountInFrame_within_3_pixels', 'nnCountInFrame_within_5_pixels',
            'nnCountInFrame_within_10_pixels', 'nnCountInFrame_within_20_pixels',
            'nnCountInFrame_within_30_pixels', 'intensity_roiOnMeanXY',
            'intensity_roiOnMeanXY - mean roi1', 'intensity_roiOnMeanXY - mean roi1 and black',
            'roi_1 smoothed', 'intensity_roiOnMeanXY - smoothed roi_1',
            'intensity - smoothed roi_1'
        ]

    def _initialize_windows(self) -> None:
        """Initialize sub-windows."""
        try:
            # Track plot options window
            self.trackPlotOptions = TrackPlotOptions(self)
            self.trackPlotOptions.hide()

            # Overlay window
            self.overlayWindow = Overlay(self)
            self.overlayWindow.hide()

            # Filter options window
            self.filterOptionsWindow = FilterOptions(self)
            self.filterOptionsWindow.hide()

            # Track window
            self.trackWindow = TrackWindow(self)
            self.trackWindow.hide()

            # Flower plot window
            self.flowerPlotWindow = FlowerPlotWindow(self)
            self.flowerPlotWindow.hide()

            # Single track plot window
            self.singleTrackPlot = TrackPlot(self)
            self.singleTrackPlot.hide()

            # All tracks plot window
            self.allTracksPlot = AllTracksPlot(self)
            self.allTracksPlot.hide()

            # ROI plot window
            self.ROIplot = ROIPLOT(self)
            self.ROIplot.hide()

            logger.debug("Sub-windows initialized")

        except Exception as e:
            logger.error(f"Error initializing windows: {e}")

    def _setup_main_gui(self) -> None:
        """Set up the main GUI elements."""
        try:
            # Reset GUI
            self.gui_reset()

            # Get current settings
            s = g.settings['locsAndTracksPlotter']

            # Create control widgets
            self._create_buttons()
            self._create_checkboxes()
            self._create_comboboxes()
            self._create_file_selector()

            # Define GUI items
            self._define_gui_items()

        except Exception as e:
            logger.error(f"Error setting up main GUI: {e}")

    def _create_buttons(self) -> None:
        """Create button widgets."""
        # Plot control buttons
        self.plotPointData_button = QPushButton('Plot Points')
        self.plotPointData_button.pressed.connect(self.plotPointData)

        self.hidePointData_button = QPushButton('Toggle Points')
        self.hidePointData_button.pressed.connect(self.hidePointData)

        self.toggleUnlinkedPointData_button = QPushButton('Show Unlinked')
        self.toggleUnlinkedPointData_button.pressed.connect(self.toggleUnlinkedPointData)

        self.plotTrackData_button = QPushButton('Plot Tracks')
        self.plotTrackData_button.pressed.connect(self.plotTrackData)

        self.clearTrackData_button = QPushButton('Clear Tracks')
        self.clearTrackData_button.pressed.connect(self.clearTracks)

        # Analysis buttons
        self.saveData_button = QPushButton('Save Tracks')
        self.saveData_button.pressed.connect(self.saveData)

        self.showCharts_button = QPushButton('Show Charts')
        self.showCharts_button.pressed.connect(self.toggleCharts)

        self.showDiffusion_button = QPushButton('Show Diffusion')
        self.showDiffusion_button.pressed.connect(self.toggleDiffusionPlot)

        self.togglePointMap_button = QPushButton('Plot Point Map')
        self.togglePointMap_button.pressed.connect(self.togglePointMap)

        self.overlayOption_button = QPushButton('Overlay')
        self.overlayOption_button.pressed.connect(self.displayOverlayOptions)

    def _create_checkboxes(self) -> None:
        """Create checkbox widgets."""
        # Display option checkboxes
        self.displayFlowPlot_checkbox = CheckBox()
        self.displayFlowPlot_checkbox.stateChanged.connect(self.toggleFlowerPlot)
        self.displayFlowPlot_checkbox.setChecked(False)

        self.displaySingleTrackPlot_checkbox = CheckBox()
        self.displaySingleTrackPlot_checkbox.stateChanged.connect(self.toggleSingleTrackPlot)
        self.displaySingleTrackPlot_checkbox.setChecked(False)

        self.displayAllTracksPlot_checkbox = CheckBox()
        self.displayAllTracksPlot_checkbox.stateChanged.connect(self.toggleAllTracksPlot)
        self.displayAllTracksPlot_checkbox.setChecked(False)

        self.displayFilterOptions_checkbox = CheckBox()
        self.displayFilterOptions_checkbox.stateChanged.connect(self.toggleFilterOptions)
        self.displayFilterOptions_checkbox.setChecked(False)

        self.displayTrackPlotOptions_checkbox = CheckBox()
        self.displayTrackPlotOptions_checkbox.stateChanged.connect(self.toggleTrackPlotOptions)
        self.displayTrackPlotOptions_checkbox.setChecked(False)

        self.displayROIplot_checkbox = CheckBox()
        self.displayROIplot_checkbox.stateChanged.connect(self.toggleROIplot)
        self.displayROIplot_checkbox.setChecked(False)

    def _create_comboboxes(self) -> None:
        """Create combobox widgets."""
        # File type selector
        self.filetype_Box = pg.ComboBox()
        filetypes = {
            'flika': 'flika',
            'thunderstorm': 'thunderstorm',
            'xy': 'xy',
            'json': 'json'
        }
        self.filetype_Box.setItems(filetypes)

        # Column selectors
        self._create_column_selectors()

    def _create_column_selectors(self) -> None:
        """Create column selector comboboxes."""
        default_cols = {'None': 'None'}

        self.xCol_Box = pg.ComboBox()
        self.xCol_Box.setItems(default_cols)

        self.yCol_Box = pg.ComboBox()
        self.yCol_Box.setItems(default_cols)

        self.frameCol_Box = pg.ComboBox()
        self.frameCol_Box.setItems(default_cols)

        self.trackCol_Box = pg.ComboBox()
        self.trackCol_Box.setItems(default_cols)

    def _create_file_selector(self) -> None:
        """Create file selector widget."""
        self.getFile = FileSelector(filetypes='*.csv *.json', mainGUI=self)
        self.getFile.valueChanged.connect(self.loadData)

    def _define_gui_items(self) -> None:
        """Define GUI items for the interface."""
        self.items.extend([
            {'name': 'filename', 'string': '', 'object': self.getFile},
            {'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box},
            {'name': 'hidePoints', 'string': 'PLOT --------------------', 'object': self.hidePointData_button},
            {'name': 'plotPointMap', 'string': '', 'object': self.togglePointMap_button},
            {'name': 'plotUnlinkedPoints', 'string': '', 'object': self.toggleUnlinkedPointData_button},
            {'name': 'trackPlotOptions', 'string': 'Display Options', 'object': self.displayTrackPlotOptions_checkbox},
            {'name': 'displayFlowerPlot', 'string': 'Flower Plot', 'object': self.displayFlowPlot_checkbox},
            {'name': 'displaySingleTrackPlot', 'string': 'Track Plot', 'object': self.displaySingleTrackPlot_checkbox},
            {'name': 'displayAllTracksPlot', 'string': 'All Tracks Plot', 'object': self.displayAllTracksPlot_checkbox},
            {'name': 'displayROIplot', 'string': 'ROI Plot', 'object': self.displayROIplot_checkbox},
            {'name': 'displayFilterOptions', 'string': 'Filter Window', 'object': self.displayFilterOptions_checkbox},
            {'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button},
            {'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button},
            {'name': 'saveTracks', 'string': '', 'object': self.saveData_button},
            {'name': 'showCharts', 'string': '', 'object': self.showCharts_button},
            {'name': 'showDiffusion', 'string': '', 'object': self.showDiffusion_button},
            {'name': 'overlayOptions', 'string': '', 'object': self.overlayOption_button},
        ])

    # Data loading and processing methods

    def loadData(self) -> None:
        """Load data from the selected file."""
        try:
            logger.info("Loading data file")

            # Set the plot window to the global window instance
            if not hasattr(g, 'win') or g.win is None:
                logger.error("No active window found")
                return

            self.plotWindow = g.win
            self.filename = self.getFile.value()

            if not self.filename:
                logger.warning("No filename provided")
                return

            # Load data based on file type
            if self.filetype_Box.value() == 'json':
                self.data = self.loadJSONTracks(self.filename)
            else:
                self.data = pd.read_csv(self.filename)

            if self.data is None or self.data.empty:
                logger.error("Failed to load data or data is empty")
                return

            # Process loaded data
            self._process_loaded_data()

            logger.info(f"Data loaded successfully: {len(self.data)} rows")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(f"Error loading data: {str(e)}")

    def _process_loaded_data(self) -> None:
        """Process the loaded data."""
        try:
            # Ensure frame and track_number are integers
            self.data['frame'] = self.data['frame'].astype(int)

            if 'track_number' in self.data.columns:
                # Separate unlinked points
                self.data_unlinked = self.data[self.data['track_number'].isna()].copy()
                self.data = self.data[~self.data['track_number'].isna()].copy()
                self.data['track_number'] = self.data['track_number'].astype(int)
            else:
                self.data['track_number'] = None
                self.data_unlinked = self.data.copy()

            # Validate frame count
            max_frame = np.max(self.data['frame'])
            if max_frame > g.win.mt:
                error_msg = "Selected window doesn't have enough frames to plot all data points"
                logger.error(error_msg)
                if hasattr(g, 'alert'):
                    g.alert(error_msg)
                self._reset_data()
                return

            # Display data info
            logger.info("Data processing completed")
            logger.debug("Data preview:")
            logger.debug(f"\n{self.data.head()}")

            # Add missing columns
            self._add_missing_columns()

            # Setup GUI elements
            self._setup_data_dependent_gui()

            # Plot initial data
            self.plotPointData()

        except Exception as e:
            logger.error(f"Error processing data: {e}")
            self._reset_data()

    def _reset_data(self) -> None:
        """Reset data variables to None."""
        self.plotWindow = None
        self.filename = None
        self.data = None
        self.data_unlinked = None

    def _add_missing_columns(self) -> None:
        """Add any missing expected columns to the data."""
        try:
            for col in self.expectedColumns:
                if col not in self.data.columns:
                    self.data[col] = np.nan
                if hasattr(self, 'data_unlinked') and col not in self.data_unlinked.columns:
                    self.data_unlinked[col] = np.nan
            logger.debug("Missing columns added")
        except Exception as e:
            logger.error(f"Error adding missing columns: {e}")

    def _setup_data_dependent_gui(self) -> None:
        """Set up GUI elements that depend on loaded data."""
        try:
            # Create column dictionary
            self.columns = self.data.columns
            self.colDict = dictFromList(self.columns)

            # Update combobox items
            self._update_column_selectors()

            # Update sub-windows
            self._update_subwindows()

            # Set estimated camera black level
            self.estimatedCameraBlackLevel = np.min(self.plotWindow.image)
            self.trackPlotOptions.estimatedCameraBlack.setText(str(self.estimatedCameraBlackLevel))

            logger.debug("Data-dependent GUI setup completed")

        except Exception as e:
            logger.error(f"Error setting up data-dependent GUI: {e}")

    def _update_column_selectors(self) -> None:
        """Update column selector comboboxes with data columns."""
        selectors = [
            self.xCol_Box, self.yCol_Box, self.frameCol_Box, self.trackCol_Box,
            self.filterOptionsWindow.filterCol_Box,
            self.trackPlotOptions.trackColourCol_Box
        ]

        for selector in selectors:
            selector.setItems(self.colDict)

    def _update_subwindows(self) -> None:
        """Update sub-windows with loaded data."""
        try:
            # Update track selectors
            self.singleTrackPlot.updateTrackList()
            self.singleTrackPlot.pointCol_Box.setItems(self.colDict)
            self.singleTrackPlot.lineCol_Box.setItems(self.colDict)
            self.singleTrackPlot.setPadArray(self.plotWindow.imageArray())

            self.allTracksPlot.updateTrackList()
            self.ROIplot.updateTrackList()

            # Update track window column selector
            self.trackWindow.update_column_selector()

            # Load data to overlay
            self.overlayWindow.loadData()

        except Exception as e:
            self.logger.error(f"Error updating sub-windows: {e}")

    def loadJSONTracks(self, json_file: str) -> Optional[pd.DataFrame]:
        """
        Import track data from a JSON file format and calculate track metrics.

        Args:
            json_file: Path to JSON file containing track data

        Returns:
            DataFrame containing the track data with calculated metrics
        """
        try:
            logger.info(f"Loading JSON track data from {json_file}")

            with open(json_file, 'r') as f:
                data = json.load(f)

            # Initialize data lists
            frames, track_numbers, x_coords, y_coords = [], [], [], []

            # Convert txy_pts to numpy array
            txy_pts = np.array(data['txy_pts'])

            # Parse tracks from JSON
            for track_idx, track in enumerate(data['tracks']):
                for point_id in track:
                    point_data = txy_pts[point_id]
                    frames.append(point_data[0])
                    x_coords.append(point_data[1])
                    y_coords.append(point_data[2])
                    track_numbers.append(track_idx)

            # Create DataFrame
            df = pd.DataFrame({
                'frame': frames,
                'track_number': track_numbers,
                'x': x_coords,
                'y': y_coords
            })

            # Sort by track number and frame
            df = df.sort_values(['track_number', 'frame'])

            # Calculate track metrics
            df = df.groupby('track_number').apply(self._calculate_track_metrics)

            # Add remaining columns with NaN values
            remaining_cols = ['intensity', 'lagNumber', 'fracDimension', 'Experiment', 'SVM', 'nnDist_inFrame']
            for col in remaining_cols:
                df[col] = np.nan

            logger.info(f"JSON data loaded successfully: {len(df)} points, {df['track_number'].nunique()} tracks")
            return df

        except Exception as e:
            logger.error(f"Error loading JSON tracks: {e}")
            return None

    def _calculate_track_metrics(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate various metrics for each track."""
        try:
            # Zero coordinates (relative to track start)
            start_x, start_y = group['x'].iloc[0], group['y'].iloc[0]
            group['zeroed_X'] = group['x'] - start_x
            group['zeroed_Y'] = group['y'] - start_y

            # Calculate time steps and displacements
            group['dt'] = group['frame'].diff()
            group['dx'] = group['x'].diff()
            group['dy'] = group['y'].diff()
            group['d_squared'] = group['dx']**2 + group['dy']**2

            # Distance from origin
            group['distanceFromOrigin'] = np.sqrt(
                (group['x'] - start_x)**2 + (group['y'] - start_y)**2
            )

            # Velocity and direction
            group['velocity'] = np.sqrt(group['d_squared']) / group['dt']
            group['direction_Relative_To_Origin'] = (
                np.arctan2(group['dy'], group['dx']) * 180 / np.pi
            )

            # Track statistics
            group['meanVelocity'] = group['velocity'].mean()
            group['track_length'] = len(group)
            group['lag'] = group.index - group.index[0]
            group['meanLag'] = group['lag'].mean()

            # Net displacement
            end_x, end_y = group['x'].iloc[-1], group['y'].iloc[-1]
            net_displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            group['netDispl'] = net_displacement

            # Radius of gyration
            mean_x, mean_y = group['x'].mean(), group['y'].mean()
            group['radius_gyration'] = np.sqrt(
                ((group['x'] - mean_x)**2 + (group['y'] - mean_y)**2).mean()
            )

            # Shape metrics
            dx_squared = (group['x'] - mean_x)**2
            dy_squared = (group['y'] - mean_y)**2
            group['asymmetry'] = (
                abs(dx_squared.mean() - dy_squared.mean()) /
                (dx_squared.mean() + dy_squared.mean())
            )

            # Statistical moments
            if len(group['d_squared'].dropna()) > 0:
                group['skewness'] = skew(group['d_squared'].dropna())
                group['kurtosis'] = kurtosis(group['d_squared'].dropna())
            else:
                group['skewness'] = np.nan
                group['kurtosis'] = np.nan

            # Straightness
            total_path_length = np.sqrt(group['d_squared']).sum()
            group['Straight'] = (
                net_displacement / total_path_length if total_path_length > 0 else 0
            )

            # Distance over time metric
            group['dy-dt: distance'] = group['distanceFromOrigin'].diff() / group['dt']

            return group

        except Exception as e:
            logger.error(f"Error calculating track metrics: {e}")
            return group

    # Plotting and visualization methods

    def makePointDataDF(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare point data for plotting based on the file type.

        Args:
            data: Input DataFrame with tracking data

        Returns:
            DataFrame with frame, x, y columns in correct format
        """
        try:
            df = pd.DataFrame()
            filetype = self.filetype_Box.value()

            if filetype == 'thunderstorm':
                # ThunderSTORM format (frames start from 1, coordinates in nm)
                df['frame'] = data['frame'].astype(int) - 1
                df['x'] = data['x [nm]'] / self.trackPlotOptions.pixelSize_selector.value()
                df['y'] = data['y [nm]'] / self.trackPlotOptions.pixelSize_selector.value()

            elif filetype == 'flika':
                # FLIKA format (frames start from 1, coordinates in pixels)
                df['frame'] = data['frame'].astype(int) - 1
                df['x'] = data['x']
                df['y'] = data['y']

            elif filetype == 'json':
                # JSON format (frames as-is, coordinates in pixels)
                df['frame'] = data['frame'].astype(int)
                df['x'] = data['x']
                df['y'] = data['y']

            else:
                logger.warning(f"Unknown filetype: {filetype}, using default format")
                df['frame'] = data['frame'].astype(int)
                df['x'] = data['x']
                df['y'] = data['y']

            logger.debug(f"Point data prepared: {len(df)} points")
            return df

        except Exception as e:
            logger.error(f"Error preparing point data: {e}")
            return pd.DataFrame()

    def plotPointsOnStack(self, points: pd.DataFrame, pointColor: QColor,
                         unlinkedPoints: Optional[pd.DataFrame] = None,
                         unlinkedColour: QColor = QColor(Qt.blue)) -> None:
        """
        Plot points on the image stack.

        Args:
            points: DataFrame with point coordinates
            pointColor: Color for linked points
            unlinkedPoints: Optional DataFrame with unlinked points
            unlinkedColour: Color for unlinked points
        """
        try:
            if self.plotWindow is None:
                logger.error("No plot window available")
                return

            points_byFrame = points[['frame', 'x', 'y']].copy()

            # Debug information
            max_frame = points_byFrame['frame'].max()
            logger.debug(f"Maximum frame in data: {max_frame}")
            logger.debug(f"Number of frames in window: {self.plotWindow.mt}")

            # Align frames with display
            if self.filetype_Box.value() in ['thunderstorm', 'json']:
                # These formats use 0-based indexing
                pass
            else:
                # FLIKA uses 1-based indexing
                points_byFrame['frame'] = points_byFrame['frame'] + 1

            # Convert to numpy array
            pointArray = points_byFrame.to_numpy()

            # Initialize scatter points for each frame
            self.plotWindow.scatterPoints = [[] for _ in range(self.plotWindow.mt)]
            pointSize = self.trackPlotOptions.pointSize_selector.value()

            # Add linked points
            for pt in pointArray:
                frame_idx = int(pt[0])
                if self.plotWindow.mt == 1:
                    frame_idx = 0
                elif frame_idx >= self.plotWindow.mt:
                    continue

                position = [pt[1], pt[2], pointColor, pointSize]
                self.plotWindow.scatterPoints[frame_idx].append(position)

            # Add unlinked points if requested
            if self.displayUnlinkedPoints and unlinkedPoints is not None:
                unlinkedPoints_byFrame = unlinkedPoints[['frame', 'x', 'y']].copy()

                if self.filetype_Box.value() in ['thunderstorm', 'json']:
                    pass
                else:
                    unlinkedPoints_byFrame['frame'] = unlinkedPoints_byFrame['frame'] + 1

                unlinkedPointArray = unlinkedPoints_byFrame.to_numpy()

                for pt in unlinkedPointArray:
                    frame_idx = int(pt[0])
                    if self.plotWindow.mt == 1:
                        frame_idx = 0
                    elif frame_idx >= self.plotWindow.mt:
                        continue

                    position = [pt[1], pt[2], unlinkedColour, pointSize]
                    self.plotWindow.scatterPoints[frame_idx].append(position)

            # Update the display
            self.plotWindow.updateindex()
            logger.debug("Points plotted on stack successfully")

        except Exception as e:
            logger.error(f"Error plotting points on stack: {e}")

    def plotPointData(self) -> None:
        """Plot point data to current window."""
        try:
            if self.data is None:
                logger.warning("No data available to plot")
                return

            # Determine which data to use
            if self.useFilteredData and self.filteredData is not None:
                points_data = self.filteredData
            else:
                points_data = self.data

            # Create point data DataFrame
            self.points = self.makePointDataDF(points_data)

            # Handle unlinked points
            unlinked_points = None
            if self.displayUnlinkedPoints and hasattr(self, 'data_unlinked'):
                unlinked_points = self.makePointDataDF(self.data_unlinked)

            # Plot points
            point_color = self.trackPlotOptions.pointColour_Box.value()
            unlinked_color = self.trackPlotOptions.unlinkedpointColour_Box.value()

            self.plotPointsOnStack(
                self.points,
                point_color,
                unlinkedPoints=unlinked_points,
                unlinkedColour=unlinked_color
            )

            # Update status
            message = 'Point data plotted to current window'
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)
            logger.info(message)

        except Exception as e:
            logger.error(f"Error plotting point data: {e}")

    def hidePointData(self) -> None:
        """Toggle visibility of point data."""
        try:
            if (hasattr(self.plotWindow, 'scatterPlot') and
                hasattr(self.plotWindow, 'imageview') and
                self.plotWindow.scatterPlot in self.plotWindow.imageview.ui.graphicsView.items()):

                self.plotWindow.imageview.ui.graphicsView.removeItem(self.plotWindow.scatterPlot)
                logger.debug("Points hidden")
            else:
                self.plotWindow.imageview.addItem(self.plotWindow.scatterPlot)
                logger.debug("Points shown")

        except Exception as e:
            logger.error(f"Error toggling point visibility: {e}")

    def toggleUnlinkedPointData(self) -> None:
        """Toggle display of unlinked point data."""
        try:
            if not self.displayUnlinkedPoints:
                self.displayUnlinkedPoints = True
                self.plotPointData()
                self.toggleUnlinkedPointData_button.setText('Hide Unlinked')
                message = 'Unlinked point data plotted to current window'
            else:
                self.displayUnlinkedPoints = False
                self.plotPointData()
                self.toggleUnlinkedPointData_button.setText('Show Unlinked')
                message = 'Unlinked point data hidden'

            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)
            logger.info(message)

        except Exception as e:
            logger.error(f"Error toggling unlinked points: {e}")

    def makeTrackDF(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare track data for plotting based on the file type.

        Args:
            data: Input DataFrame with track data

        Returns:
            Grouped DataFrame by track_number
        """
        try:
            df = pd.DataFrame()
            filetype = self.filetype_Box.value()

            if filetype == 'thunderstorm':
                df['frame'] = data['frame'].astype(int) - 1
                df['x'] = data['x [nm]'] / self.trackPlotOptions.pixelSize_selector.value()
                df['y'] = data['y [nm]'] / self.trackPlotOptions.pixelSize_selector.value()
                df['track_number'] = data['track_number']

            elif filetype == 'flika':
                df['frame'] = data['frame'].astype(int) - 1
                df['x'] = data['x']
                df['y'] = data['y']
                df['track_number'] = data['track_number']
                df['zeroed_X'] = data['zeroed_X']
                df['zeroed_Y'] = data['zeroed_Y']

            elif filetype == 'json':
                df['frame'] = data['frame'].astype(int) - 1
                df['x'] = data['x']
                df['y'] = data['y']
                df['track_number'] = data['track_number']
                df['zeroed_X'] = data['zeroed_X']
                df['zeroed_Y'] = data['zeroed_Y']

            # Add colors if track coloring is enabled
            if self.trackPlotOptions.trackColour_checkbox.isChecked():
                self._add_track_colors(df, data)

            # Reset index for JSON format
            if filetype == 'json':
                df = df.reset_index(drop=True)

            # Group by track number
            grouped = df.groupby(['track_number'])
            logger.debug(f"Track data prepared: {len(df)} points in {len(grouped)} tracks")
            return grouped

        except Exception as e:
            logger.error(f"Error preparing track data: {e}")
            return pd.DataFrame().groupby(['track_number'])

    def _add_track_colors(self, df: pd.DataFrame, data: pd.DataFrame) -> None:
        """Add color information to track DataFrame."""
        try:
            color_column = self.trackPlotOptions.trackColourCol_Box.value()

            if color_column == 'None' or color_column not in data.columns:
                return

            # Get colormap
            if self.useMatplotCM:
                cm = pg.colormap.getFromMatplotlib(self.trackPlotOptions.colourMap_Box.value())
            else:
                cm = pg.colormap.get(self.trackPlotOptions.colourMap_Box.value())

            # Map colors
            color_values = data[color_column].to_numpy()
            max_value = np.max(color_values)
            if max_value > 0:
                normalized_values = color_values / max_value
                df['colour'] = cm.mapToQColor(normalized_values)
                df['threshColour'] = data.get('threshColour', df['colour'])

        except Exception as e:
            logger.error(f"Error adding track colors: {e}")

    def clearTracks(self) -> None:
        """Clear track visualization from all windows."""
        try:
            # Clear from main plot window
            if self.plotWindow is not None and not self.plotWindow.closed:
                for pathitem in self.pathitems:
                    self.plotWindow.imageview.view.removeItem(pathitem)

            # Clear from overlay window
            if hasattr(self, 'overlayWindow'):
                for pathitem in self.overlayWindow.pathitems:
                    self.overlayWindow.overlayWindow.view.removeItem(pathitem)
                self.overlayWindow.pathitems = []

            # Reset path items
            self.pathitems = []
            logger.debug("Tracks cleared from all windows")

        except Exception as e:
            logger.error(f"Error clearing tracks: {e}")

    def plotTrackData(self) -> None:
        """Plot track data to current window."""
        try:
            if self.data is None:
                logger.warning("No data available to plot tracks")
                return

            # Determine which data to use
            if self.useFilteredData and self.filteredData is not None:
                track_data = self.filteredData
                self.trackIDs = np.unique(self.filteredData['track_number']).astype(int)
            else:
                track_data = self.data
                self.trackIDs = np.unique(self.data['track_number']).astype(int)

            # Create track DataFrame
            self.tracks = self.makeTrackDF(track_data)

            # Show tracks
            self.showTracks()

            # Connect mouse and keyboard events
            if hasattr(self.plotWindow, 'imageview'):
                self.plotWindow.imageview.scene.sigMouseMoved.connect(self.updateTrackSelector)
            if hasattr(self.plotWindow, 'keyPressSignal'):
                self.plotWindow.keyPressSignal.connect(self.selectTrack)

            # Show track window
            self.trackWindow.show()

            # Show flower plot if enabled
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.show()

            # Show single track plot if enabled
            if self.displaySingleTrackPlot_checkbox.isChecked():
                self.singleTrackPlot.show()

            # Update status
            message = 'Track data plotted to current window'
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)
            logger.info(f"{message}: {len(self.trackIDs)} tracks")

        except Exception as e:
            logger.error(f"Error plotting track data: {e}")

    def showTracks(self) -> None:
        """Update track paths in main view and other windows."""
        try:
            # Clear existing tracks
            self.clearTracks()

            # Clear flower plot tracks if enabled
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.clearTracks()

            # Set up pens for drawing
            pen = QPen(self.trackPlotOptions.trackDefaultColour_Box.value(), 0.4)
            pen.setCosmetic(True)
            pen.setWidth(self.trackPlotOptions.lineSize_selector.value())

            # Determine which tracks to plot
            if self.useFilteredTracks and hasattr(self, 'filteredTrackIds'):
                tracks_to_plot = self.filteredTrackIds
            else:
                tracks_to_plot = self.trackIDs

            logger.debug(f"Plotting {len(tracks_to_plot)} tracks")


            # Plot each track
            for track_idx in tracks_to_plot:
                # Convert numpy types to native Python types
                track_key = int(track_idx)

                try:
                    # Direct index selection from original data (PyInstaller compatible)
                    if 'track_number' in self.data.columns:
                        track_mask = self.data['track_number'] == track_key
                        track = self.data[track_mask]
                    else:
                        logger.warning("No 'track_number' column found in data")
                        continue

                    if len(track) == 0:
                        continue

                    self._plot_single_track(track, pen)

                except Exception as e:
                    logger.warning(f"Could not plot track {track_key}: {e}")
                    continue

            # Plot tracks in flower plot if enabled
            if self.displayFlowPlot_checkbox.isChecked():
                # Determine which data to use
                if self.useFilteredData and self.filteredData is not None:
                    flower_data = self.filteredData
                else:
                    flower_data = self.data
                self.flowerPlotWindow.plotTracks(flower_data)

                print(f"DEBUG: Calling flowerPlotWindow.plotTracks with {len(flower_data)} rows")
                self.flowerPlotWindow.plotTracks(flower_data)
                print("DEBUG: flowerPlotWindow.plotTracks completed")

            logger.debug("Track visualization completed")

        except Exception as e:
            import traceback
            logger.error(f"Error showing tracks: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            print(f"\n{'='*70}")
            print(f"ERROR in showTracks:")
            print(f"{'='*70}")
            print(traceback.format_exc())
            print(f"{'='*70}\n")


    def _plot_single_track(self, track: pd.DataFrame, pen: QPen) -> None:
        """Plot a single track path."""
        try:
            track_id = track['track_number'].iloc[0] if 'track_number' in track.columns else 'unknown'

            # Create path item for main window
            pathitem = QGraphicsPathItem(self.plotWindow.imageview.view)

            # Set color based on track coloring options
            if self.trackPlotOptions.trackColour_checkbox.isChecked():
                color_mode = 'threshColour' if self.trackPlotOptions.threshold_checkbox.isChecked() else 'colour'
                if color_mode in track.columns:
                    pen.setColor(track[color_mode].iloc[0])

            pathitem.setPen(pen)
            self.plotWindow.imageview.view.addItem(pathitem)
            self.pathitems.append(pathitem)

            # Create overlay path if overlay window exists
            if hasattr(self, 'overlayWindow'):
                pathitem_overlay = QGraphicsPathItem(self.overlayWindow.overlayWindow.view)
                pen_overlay = QPen(pen)
                pathitem_overlay.setPen(pen_overlay)
                self.overlayWindow.overlayWindow.view.addItem(pathitem_overlay)
                self.overlayWindow.pathitems.append(pathitem_overlay)

            # Create flower plot path if enabled
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_fp = QGraphicsPathItem(self.flowerPlotWindow.plt)
                pen_fp = QPen(pen)
                pathitem_fp.setPen(pen_fp)
                self.flowerPlotWindow.plt.addItem(pathitem_fp)
                self.flowerPlotWindow.pathitems.append(pathitem_fp)

            # Extract coordinates
            x = track['x'].to_numpy()
            y = track['y'].to_numpy()

            if len(x) < 2:
                return  # Need at least 2 points for a path

            # Create painter path
            path = QPainterPath(QPointF(x[0], y[0]))
            for i in range(1, len(x)):
                path.lineTo(QPointF(x[i], y[i]))

            pathitem.setPath(path)

            # Set overlay path
            if hasattr(self, 'overlayWindow'):
                pathitem_overlay.setPath(path)

            # Set flower plot path with zeroed coordinates
            if self.displayFlowPlot_checkbox.isChecked() and 'zeroed_X' in track.columns:
                zeroed_x = track['zeroed_X'].to_numpy()
                zeroed_y = track['zeroed_Y'].to_numpy()
                if len(zeroed_x) >= 2:
                    path_fp = QPainterPath(QPointF(zeroed_x[0], zeroed_y[0]))
                    for i in range(1, len(zeroed_x)):
                        path_fp.lineTo(QPointF(zeroed_x[i], zeroed_y[i]))
                    pathitem_fp.setPath(path_fp)

        except Exception as e:
            logger.error(f"Error plotting single track: {e}")

    # Filter and analysis methods

    def filterData(self) -> None:
        """Apply data filtering based on user criteria."""
        try:
            if self.data is None:
                logger.warning("No data available to filter")
                return

            # Get filter parameters
            op = self.filterOptionsWindow.filterOp_Box.value()
            filter_col = self.filterOptionsWindow.filterCol_Box.value()

            if filter_col == 'None' or filter_col not in self.data.columns:
                logger.warning(f"Invalid filter column: {filter_col}")
                return

            try:
                value = float(self.filterOptionsWindow.filterValue_Box.text())
            except ValueError:
                logger.error("Invalid filter value - must be numeric")
                return

            # Determine source data
            if self.sequentialFiltering and self.useFilteredData and self.filteredData is not None:
                source_data = self.filteredData
            else:
                source_data = self.data

            # Apply filter operation
            if op == '==':
                self.filteredData = source_data[source_data[filter_col] == value]
            elif op == '<':
                self.filteredData = source_data[source_data[filter_col] < value]
            elif op == '>':
                self.filteredData = source_data[source_data[filter_col] > value]
            elif op == '!=':
                self.filteredData = source_data[source_data[filter_col] != value]
            else:
                logger.error(f"Unknown filter operation: {op}")
                return

            # Update state
            self.useFilteredData = True

            # Update display
            self.plotPointData()
            if hasattr(self, 'allTracksPlot'):
                self.allTracksPlot.updateTrackList()

            # Update status
            message = f'Filter applied: {len(self.filteredData)} points remaining'
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)
            logger.info(message)

        except Exception as e:
            logger.error(f"Error filtering data: {e}")

    def clearFilterData(self) -> None:
        """Clear data filtering."""
        try:
            self.useFilteredData = False
            self.filteredData = None

            # Update display
            self.plotPointData()
            if hasattr(self, 'allTracksPlot'):
                self.allTracksPlot.updateTrackList()

            message = 'Data filter cleared'
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)
            logger.info(message)

        except Exception as e:
            logger.error(f"Error clearing filter: {e}")

    def saveData(self) -> None:
        """Save filtered data to file."""
        try:
            if not self.useFilteredData or self.filteredData is None:
                if hasattr(g, 'alert'):
                    g.alert('Filter data first')
                logger.warning("No filtered data available to save")
                return

            # Get save path
            save_path, _ = QFileDialog.getSaveFileName(
                None, "Save file", "", "CSV Files (*.csv)"
            )

            if not save_path:
                return

            # Save data
            self.filteredData.to_csv(save_path, index=False)

            message = f'Filtered data saved to: {save_path}'
            logger.info(message)

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    # Window management methods

    def toggleFlowerPlot(self) -> None:
        """Toggle flower plot window visibility."""
        try:
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.show()
            else:
                self.flowerPlotWindow.hide()
        except Exception as e:
            logger.error(f"Error toggling flower plot: {e}")

    def toggleSingleTrackPlot(self) -> None:
        """Toggle single track plot window visibility."""
        try:
            if self.displaySingleTrackPlot_checkbox.isChecked():
                self.singleTrackPlot.show()
            else:
                self.singleTrackPlot.hide()
        except Exception as e:
            logger.error(f"Error toggling single track plot: {e}")

    def toggleAllTracksPlot(self) -> None:
        """Toggle all tracks plot window visibility."""
        try:
            if self.displayAllTracksPlot_checkbox.isChecked():
                self.allTracksPlot.show()
            else:
                self.allTracksPlot.hide()
        except Exception as e:
            logger.error(f"Error toggling all tracks plot: {e}")

    def toggleFilterOptions(self) -> None:
        """Toggle filter options window visibility."""
        try:
            if self.displayFilterOptions_checkbox.isChecked():
                self.filterOptionsWindow.show()
            else:
                self.filterOptionsWindow.hide()
        except Exception as e:
            logger.error(f"Error toggling filter options: {e}")

    def toggleTrackPlotOptions(self) -> None:
        """Toggle track plot options window visibility."""
        try:
            if not self.displayTrackPlotOptions:
                self.trackPlotOptions.show()
                self.displayTrackPlotOptions = True
            else:
                self.trackPlotOptions.hide()
                self.displayTrackPlotOptions = False
        except Exception as e:
            logger.error(f"Error toggling track plot options: {e}")

    def toggleROIplot(self) -> None:
        """Toggle ROI plot window visibility."""
        try:
            if not self.displayROIplot:
                self.ROIplot.show()
                self.displayROIplot = True
            else:
                self.ROIplot.hide()
                self.displayROIplot = False
        except Exception as e:
            logger.error(f"Error toggling ROI plot: {e}")

    def toggleCharts(self) -> None:
        """Toggle charts window visibility."""
        try:
            if self.chartWindow is None:
                self.chartWindow = ChartDock(self)
                self.chartWindow.xColSelector.setItems(self.colDict)
                self.chartWindow.yColSelector.setItems(self.colDict)
                self.chartWindow.colSelector.setItems(self.colDict)

            if not self.displayCharts:
                self.chartWindow.show()
                self.displayCharts = True
                self.showCharts_button.setText('Hide Charts')
            else:
                self.chartWindow.hide()
                self.displayCharts = False
                self.showCharts_button.setText('Show Charts')

        except Exception as e:
            logger.error(f"Error toggling charts: {e}")

    def toggleDiffusionPlot(self) -> None:
        """Toggle diffusion plot window visibility."""
        try:
            if self.diffusionWindow is None:
                self.diffusionWindow = DiffusionPlotWindow(self)

            if not self.displayDiffusionPlot:
                self.diffusionWindow.show()
                self.displayDiffusionPlot = True
                self.showDiffusion_button.setText('Hide Diffusion')
            else:
                self.diffusionWindow.hide()
                self.displayDiffusionPlot = False
                self.showDiffusion_button.setText('Show Diffusion')

        except Exception as e:
            logger.error(f"Error toggling diffusion plot: {e}")

    def displayOverlayOptions(self) -> None:
        """Toggle overlay options window visibility."""
        try:
            if not self.displayOverlay:
                self.overlayWindow.show()
                self.displayOverlay = True
                self.overlayOption_button.setText('Hide Overlay')
            else:
                self.overlayWindow.hide()
                self.displayOverlay = False
                self.overlayOption_button.setText('Show Overlay')

        except Exception as e:
            logger.error(f"Error toggling overlay options: {e}")

    def togglePointMap(self) -> None:
        """Toggle point map visualization."""
        try:
            if self.togglePointMap_button.text() == 'Plot Point Map':
                # Determine data source
                if self.useFilteredData and self.filteredData is not None:
                    df = self.filteredData
                else:
                    df = self.data

                # Add unlinked points if displayed
                if self.displayUnlinkedPoints and hasattr(self, 'data_unlinked'):
                    df = pd.concat([df, self.data_unlinked])

                # Create scatter plot with error handling
                try:
                    self.pointMapScatter = pg.ScatterPlotItem(
                        size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255)
                    )
                    self.pointMapScatter.setSize(2, update=False)
                    set_scatter_data_compat(self.pointMapScatter, x=df['x'], y=df['y'])
                    self.plotWindow.imageview.view.addItem(self.pointMapScatter)
                    self.togglePointMap_button.setText('Hide Point Map')
                except Exception as e:
                    logger.error(f"Error creating point map scatter plot: {e}")
                    g.m.statusBar().showMessage(f'Error creating point map: {str(e)}')
                    raise

            else:
                # Remove scatter plot
                if hasattr(self, 'pointMapScatter'):
                    self.plotWindow.imageview.view.removeItem(self.pointMapScatter)
                self.togglePointMap_button.setText('Plot Point Map')

        except Exception as e:
            logger.error(f"Error toggling point map: {e}")

    def setColourMap(self) -> None:
        """Set color map type (matplotlib vs pyqtgraph)."""
        try:
            if self.trackPlotOptions.matplotCM_checkbox.isChecked():
                # Use matplotlib color maps
                self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
                self.trackPlotOptions.colourMap_Box.setItems(self.colourMaps)
                self.useMatplotCM = True
            else:
                # Use pyqtgraph color maps
                self.colourMaps = dictFromList(pg.colormap.listMaps())
                self.trackPlotOptions.colourMap_Box.setItems(self.colourMaps)
                self.useMatplotCM = False

            logger.debug(f"Color map type set to: {'matplotlib' if self.useMatplotCM else 'pyqtgraph'}")

        except Exception as e:
            logger.error(f"Error setting color map: {e}")

    def clearPlots(self) -> None:
        """Clear all matplotlib plots."""
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            logger.debug("All plots cleared")
        except Exception as e:
            logger.debug(f"Could not clear plots: {e}")

    # Additional utility methods for track selection and interaction

    def updateTrackSelector(self, point: QPointF) -> None:
        """Update selected track based on mouse position."""
        try:
            if not hasattr(self.plotWindow, 'imageview'):
                return

            pos = self.plotWindow.imageview.getImageItem().mapFromScene(point)

            # Check which track the mouse is hovering over
            for i, path in enumerate(self.pathitems):
                if path.contains(pos):
                    if i < len(self.trackIDs):
                        self.selectedTrack = self.trackIDs[i]
                        break

        except Exception as e:
            logger.debug(f"Error updating track selector: {e}")

    def selectTrack(self, event) -> None:
        """Handle track selection via keyboard."""
        try:
            if event.key() == Qt.Key_T and self.selectedTrack is not None:
                # Update display track
                if self.selectedTrack != self.displayTrack:
                    self.displayTrack = self.selectedTrack
                    print(f"DEBUG: Selected track {self.displayTrack}")  # Debug line
                    self._update_track_displays()

        except Exception as e:
            logger.error(f"Error in track selection: {e}")

    def _update_track_displays(self) -> None:
        """Update all track-related displays with current track."""
        try:
            if self.displayTrack is None or self.data is None:
                return

            # Get track data
            track_data = self.data[self.data['track_number'] == int(self.displayTrack)]

            if track_data.empty:
                return

            # Extract track information
            frame = track_data['frame'].to_numpy()
            intensity_col = self.trackPlotOptions.intensityChoice_Box.value()
            intensity = track_data[intensity_col].to_numpy()

            # Apply background subtraction if enabled
            if self.trackPlotOptions.backgroundSubtract_checkbox.isChecked():
                intensity = intensity - self.trackPlotOptions.background_selector.value()

            # Get other track metrics
            track_metrics = self._extract_track_metrics(track_data)

            # Update track window
            self.trackWindow.update(
                frame, intensity,
                track_metrics['distance'],
                track_metrics['zeroed_X'],
                track_metrics['zeroed_Y'],
                track_metrics['dydt'],
                track_metrics['direction'],
                track_metrics['velocity'],
                self.displayTrack,
                *track_metrics['nn_counts'],
                track_metrics['svm'],
                track_metrics['length']
            )

            # Explicitly update the column value display
            self.trackWindow._update_column_value(self.displayTrack)

            # Update single track plot
            if hasattr(self, 'singleTrackPlot'):
                self.singleTrackPlot.plotTracks()

            logger.debug(f"Track displays updated for track {self.displayTrack}")

        except Exception as e:
            logger.error(f"Error updating track displays: {e}")

    def _extract_track_metrics(self, track_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract metrics from track data."""
        try:
            metrics = {
                'distance': track_data.get('distanceFromOrigin', np.zeros(len(track_data))).to_numpy(),
                'zeroed_X': track_data.get('zeroed_X', np.zeros(len(track_data))).to_numpy(),
                'zeroed_Y': track_data.get('zeroed_Y', np.zeros(len(track_data))).to_numpy(),
                'dydt': track_data.get('dy-dt: distance', np.zeros(len(track_data))).to_numpy(),
                'direction': track_data.get('direction_Relative_To_Origin', np.zeros(len(track_data))).to_numpy(),
                'velocity': track_data.get('velocity', np.zeros(len(track_data))).to_numpy(),
                'nn_counts': [
                    track_data.get('nnCountInFrame_within_3_pixels', np.zeros(len(track_data))).to_numpy(),
                    track_data.get('nnCountInFrame_within_5_pixels', np.zeros(len(track_data))).to_numpy(),
                    track_data.get('nnCountInFrame_within_10_pixels', np.zeros(len(track_data))).to_numpy(),
                    track_data.get('nnCountInFrame_within_20_pixels', np.zeros(len(track_data))).to_numpy(),
                    track_data.get('nnCountInFrame_within_30_pixels', np.zeros(len(track_data))).to_numpy(),
                ],
                'svm': track_data.get('SVM', [0]).iloc[0] if not track_data.empty else 0,
                'length': track_data.get('n_segments', [0]).iloc[0] if not track_data.empty else 0,
            }
            return metrics

        except Exception as e:
            logger.error(f"Error extracting track metrics: {e}")
            return {}

    def getScatterPointsAsQPoints(self):
        # Get scatter plot data as numpy array
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        # Convert numpy array to list of QPointF objects
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        # Get track IDs for all points in scatter plot
        trackIDs = []

        # Flatten scatter plot data into a single list of points
        flat_ptList = [pt for sublist in self.plotWindow.scatterPoints for pt in sublist]

        # Loop through each point and get track IDs for corresponding data points in DataFrame
        for pt in flat_ptList:
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.data[(self.data['x']==pt[0]) & (self.data['y']==pt[1])]

            trackIDs.extend(ptFilterDF['track_number'])

        # Set filtered track IDs and filtered data
        self.filteredTrackIds = np.unique(trackIDs)
        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]

        # Set flags for using filtered data and filtered tracks
        self.useFilteredData = True
        self.useFilteredTracks = True


    def ROIFilterData(self):
        # Not implemented yet for unlinked points
        if self.displayUnlinkedPoints:
           g.m.statusBar().showMessage('ROI filter not implemented for unliked points - hide them first')
           print('ROI filter not implemented for unliked points - hide them first')
           return
        # initialize variables
        self.roiFilterPoints = []
        self.rois = self.plotWindow.rois
        self.oldScatterPoints = self.plotWindow.scatterPoints

        # loop through all ROIs and all frames to find points inside them
        for roi in self.rois:
            currentFrame = self.plotWindow.currentIndex
            for i in range(0,self.plotWindow.mt):
                # set current frame
                self.plotWindow.setIndex(i)
                # get ROI shape in coordinate system of the scatter plot
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                self.roiFilterPoints.extend((selected))
            # reset current frame
            self.plotWindow.setIndex(currentFrame)

        # clear old scatter points and add new filtered points
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        for pt in self.roiFilterPoints:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(0,255,0)
            position = [pt[1], pt[2], pointColor, pointSize]
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()

        # get filtered data
        self.getDataFromScatterPoints()

        # update status bar and return
        g.m.statusBar().showMessage('ROI filter complete')

        #update allTracks track list
        self.allTracksPlot.updateTrackList()
        self.ROIplot.updateTrackList()
        return

    def clearROIFilterData(self):
        # Reset the scatter plot data to the previous unfiltered scatter plot data
        self.plotWindow.scatterPoints = self.oldScatterPoints
        self.plotWindow.updateindex()

        # Set useFilteredData and useFilteredTracks to False
        self.useFilteredData = False
        self.useFilteredTracks = False

        #update allTracks track list
        self.allTracksPlot.updateTrackList()

        return


# Create plugin instance
locsAndTracksPlotter = LocsAndTracksPlotter()

if __name__ == "__main__":
    logger.info("LocsAndTracksPlotter module loaded")
