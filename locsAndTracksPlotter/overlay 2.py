#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay Module for FLIKA Tracking Plugin

This module provides comprehensive image overlay functionality for the tracking
analysis pipeline. It supports TIFF file overlays, image processing operations,
filament detection, and various visualization options.

Features:
- TIFF file overlay with adjustable opacity and gamma correction
- Binary image processing with thresholding and morphological operations
- Automatic filament detection and analysis
- Track axis detection and visualization
- Interactive ROI-based analysis
- Point filtering based on intensity thresholds

Created on Fri Jun  2 16:32:14 2023
@author: george
"""

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal, QPointF
from qtpy.QtWidgets import (QMainWindow, QLabel, QPushButton, QLineEdit,
                           QCheckBox, QMessageBox, QGraphicsPathItem)
from qtpy.QtGui import QColor, QPainter, QPen, QPainterPath

# Scientific computing imports
import skimage.io as skio
from skimage.filters import (threshold_otsu, threshold_local, threshold_isodata,
                           gaussian)
from skimage import data, color, measure
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
from skimage.color import label2rgb
from skimage.draw import ellipse
from skimage.transform import rotate
from scipy import stats, spatial
from tqdm import tqdm

# Matplotlib imports for analysis
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Arrow

# FLIKA imports
import flika
from flika.window import Window
from flika.process.file_ import open_tiff
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

# PyQtGraph dock imports
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph import HistogramLUTWidget

# Plugin imports
from .io import FileSelector_overlay
from .helperFunctions import gammaCorrect, dictFromList

# Set up logging
logger = logging.getLogger(__name__)


class Overlay:
    """
    Comprehensive overlay system for image analysis and visualization.

    This class provides tools for overlaying single TIFF files on recording
    image stacks, with extensive image processing capabilities including
    binary operations, filament detection, and track analysis.

    Attributes:
        mainGUI: Reference to main GUI instance
        win: Main window containing dock area
        area: Dock area for organizing widgets
        overlayWindow: Main overlay image view
        binaryWindow: Binary image view
        pathitems: List of track path items for overlay
        pathitemsActin: List of actin filament paths
        actinLabels: List of detected actin regions
    """

    def __init__(self, mainGUI):
        """
        Initialize the overlay system.

        Args:
            mainGUI: Reference to main GUI instance
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Initialize data variables
        self._initialize_data_variables()

        # Set up user interface
        self._setup_ui()

        self.logger.debug("Overlay system initialized")

    def _initialize_data_variables(self) -> None:
        """Initialize data-related instance variables."""
        # Image data
        self.dataIMG: Optional[np.ndarray] = None
        self.overlayIMG: Optional[np.ndarray] = None
        self.originalIMG: Optional[np.ndarray] = None
        self.overlayFileName: Optional[str] = None
        self.binary: Optional[np.ndarray] = None

        # Visualization elements
        self.pathitems: List[QGraphicsPathItem] = []
        self.pathitemsActin: List[QGraphicsPathItem] = []
        self.pathitemsActin_binary: List[QGraphicsPathItem] = []
        self.pointMapScatter: Optional[pg.ScatterPlotItem] = None

        # Analysis results
        self.actinLabels: List[np.ndarray] = []
        self.pointsInFilaments: List = []
        self.pointsNotInFilaments: List = []

        # State flags
        self.pointsPlotted: bool = False
        self.showFilaments: bool = False

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window and dock area
            self.win = QMainWindow()
            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.resize(1000, 500)
            self.win.setWindowTitle('Image Overlay Analysis')

            # Create docks
            self._create_docks()

            # Create image views
            self._create_image_views()

            # Create control widgets
            self._create_controls()

            self.logger.debug("Overlay UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up overlay UI: {e}")

    def _create_docks(self) -> None:
        """Create dock widgets for different sections."""
        self.d1 = Dock("Main Overlay", size=(500, 500))
        self.d2 = Dock('Main Controls', size=(250, 500))
        self.d3 = Dock("Binary Image", size=(500, 500))
        self.d4 = Dock('Binary Controls', size=(250, 500))

        # Arrange docks
        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d2, 'right', self.d1)
        self.area.addDock(self.d3, 'below', self.d1)
        self.area.addDock(self.d4, 'below', self.d2)

        # Bring main docks to front
        self.d1.raiseDock()
        self.d2.raiseDock()

    def _create_image_views(self) -> None:
        """Create image view widgets."""
        # Main overlay image view
        self.overlayWindow = pg.ImageView()
        self.overlayWindow.setToolTip("Main overlay view with data points and tracks")
        self.d1.addWidget(self.overlayWindow)

        # Binary image view
        self.binaryWindow = pg.ImageView()
        self.binaryWindow.setToolTip("Binary processed image for filament detection")
        self.d3.addWidget(self.binaryWindow)

    def _create_controls(self) -> None:
        """Create all control widgets."""
        try:
            # Create layout widgets
            self.w2 = pg.LayoutWidget()  # Main controls
            self.w4 = pg.LayoutWidget()  # Binary controls

            # Create control groups
            self._create_file_controls()
            self._create_display_controls()
            self._create_analysis_controls()
            self._create_binary_controls()

            # Layout controls
            self._layout_main_controls()
            self._layout_binary_controls()

            # Add to docks
            self.d2.addWidget(self.w2)
            self.d4.addWidget(self.w4)

        except Exception as e:
            self.logger.error(f"Error creating controls: {e}")

    def _create_file_controls(self) -> None:
        """Create file loading controls."""
        # TIFF file loader
        self.loadTiff_button = FileSelector_overlay(filetypes='*.tif *.tiff')
        self.loadTiff_button.valueChanged.connect(self.loadTiff)

        # Active window loader
        self.loadWindow_button = WindowSelector()
        self.loadWindow_button.valueChanged.connect(self.loadWindow)

        # Data display toggle
        self.showData_button = QPushButton('Show Data Points')
        self.showData_button.pressed.connect(self.toggleData)

        # Filament detection
        self.getFilaments_button = QPushButton('Detect Filaments')
        self.getFilaments_button.pressed.connect(self.showDetectedFilaments)

        # Track axis detection
        self.getTrackAxis_button = QPushButton('Analyze Track Directions')
        self.getTrackAxis_button.pressed.connect(self.detectTrackAxis)

    def _create_display_controls(self) -> None:
        """Create display adjustment controls."""
        # Opacity control
        self.opacity = SliderLabel(1)
        self.opacity.setRange(0, 10)
        self.opacity.setValue(5)
        self.opacity.setSingleStep(1)
        self.opacity.valueChanged.connect(self.updateOpacity)
        self.opacity_label = QLabel('Overlay Opacity')

        # Gamma correction
        self.gammaCorrect = CheckBox()
        self.gamma = SliderLabel(1)
        self.gamma.setRange(1, 200)
        self.gamma.setValue(100)  # 100 = gamma of 1.0
        self.gamma.valueChanged.connect(self.updateGamma)
        self.gamma_label = QLabel('Gamma Correction (x0.01)')
        self.gammaCorrect.stateChanged.connect(self.resetGamma)

    def _create_analysis_controls(self) -> None:
        """Create analysis parameter controls."""
        # Filament size limits
        self.maxSizeLimit = CheckBox()
        self.maxSize_slider = SliderLabel(1)
        self.maxSize_slider.setRange(0, 10000)
        self.maxSize_slider.setValue(1000)
        self.maxSize_slider.valueChanged.connect(self.detectFilaments)
        self.maxSize_label = QLabel('Max Filament Size (pixels)')

        self.minSize_slider = SliderLabel(1)
        self.minSize_slider.setRange(0, 10000)
        self.minSize_slider.setValue(50)
        self.minSize_slider.valueChanged.connect(self.detectFilaments)
        self.minSize_label = QLabel('Min Filament Size (pixels)')

        # Point filtering by intensity
        self.pointThreshold = CheckBox()
        self.pointThreshold_slider = SliderLabel(1)
        self.pointThreshold_slider.setRange(0, 1000)
        self.pointThreshold_slider.setValue(0)
        self.pointThreshold_slider.valueChanged.connect(self.plotPoints)
        self.pointThreshold_label = QLabel('Min Point Intensity')
        self.pointThreshold.stateChanged.connect(self.plotPoints)

    def _create_binary_controls(self) -> None:
        """Create binary image processing controls."""
        # Gaussian blur
        self.gaussianBlur = CheckBox()
        self.gaussian_slider = SliderLabel(1)
        self.gaussian_slider.setRange(0, 20)
        self.gaussian_slider.setValue(1.5)
        self.gaussian_slider.valueChanged.connect(self.updateBinary)
        self.gaussian_label = QLabel('Gaussian Blur Sigma')
        self.gaussianBlur.stateChanged.connect(self.updateBinary)

        # Manual threshold
        self.manualThreshold = CheckBox()
        self.threshold_slider = SliderLabel(1)
        self.threshold_slider.setRange(0, 20)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.updateBinary)
        self.threshold_label = QLabel('Global Threshold')
        self.manualThreshold.stateChanged.connect(self.updateBinary)

        # Local threshold parameters
        self.blocksize_slider = SliderLabel(0)
        self.blocksize_slider.setRange(1, 201)
        self.blocksize_slider.setSingleStep(2)
        self.blocksize_slider.setValue(35)
        self.blocksize_slider.valueChanged.connect(self.updateBinary)
        self.blocksize_label = QLabel('Local Threshold Block Size')

        self.offset_slider = SliderLabel(0)
        self.offset_slider.setRange(-500, 500)
        self.offset_slider.setValue(10)
        self.offset_slider.valueChanged.connect(self.updateBinary)
        self.offset_label = QLabel('Local Threshold Offset')

        # Morphological operations
        self.fillHoles = CheckBox()
        self.hole_slider = SliderLabel(0)
        self.hole_slider.setRange(1, 50)
        self.hole_slider.setValue(3)
        self.hole_slider.valueChanged.connect(self.updateBinary)
        self.fillHoles.stateChanged.connect(self.updateBinary)
        self.hole_label = QLabel('Hole Closing Size')

        self.removeSpeckle = CheckBox()
        self.speckle_slider = SliderLabel(0)
        self.speckle_slider.setRange(1, 50)
        self.speckle_slider.setValue(3)
        self.speckle_slider.valueChanged.connect(self.updateBinary)
        self.removeSpeckle.stateChanged.connect(self.updateBinary)
        self.speckle_label = QLabel('Speckle Removal Size')

    def _layout_main_controls(self) -> None:
        """Layout main control widgets."""
        row = 0

        # File controls
        self.w2.addWidget(QLabel("Load Image:"), row=row, col=0)
        row += 1
        self.w2.addWidget(self.loadTiff_button, row=row, col=0)
        row += 1
        self.w2.addWidget(self.loadWindow_button, row=row, col=0)
        row += 1

        # Display controls
        self.w2.addWidget(self.opacity_label, row=row, col=0)
        row += 1
        self.w2.addWidget(self.opacity, row=row, col=0)
        row += 1

        self.w2.addWidget(self.gamma_label, row=row, col=0)
        self.w2.addWidget(self.gammaCorrect, row=row, col=1)
        row += 1
        self.w2.addWidget(self.gamma, row=row, col=0)
        row += 1

        # Analysis controls
        self.w2.addWidget(self.minSize_label, row=row, col=0)
        row += 1
        self.w2.addWidget(self.minSize_slider, row=row, col=0)
        row += 1

        self.w2.addWidget(self.maxSize_label, row=row, col=0)
        self.w2.addWidget(self.maxSizeLimit, row=row, col=1)
        row += 1
        self.w2.addWidget(self.maxSize_slider, row=row, col=0)
        row += 1

        self.w2.addWidget(self.pointThreshold_label, row=row, col=0)
        self.w2.addWidget(self.pointThreshold, row=row, col=1)
        row += 1
        self.w2.addWidget(self.pointThreshold_slider, row=row, col=0)
        row += 1

        # Action buttons
        self.w2.addWidget(self.showData_button, row=row, col=0)
        row += 1
        self.w2.addWidget(self.getFilaments_button, row=row, col=0)
        row += 1
        self.w2.addWidget(self.getTrackAxis_button, row=row, col=0)

    def _layout_binary_controls(self) -> None:
        """Layout binary processing control widgets."""
        row = 0

        # Gaussian blur
        self.w4.addWidget(self.gaussian_label, row=row, col=0)
        self.w4.addWidget(self.gaussianBlur, row=row, col=1)
        row += 1
        self.w4.addWidget(self.gaussian_slider, row=row, col=0)
        row += 1

        # Local threshold
        self.w4.addWidget(self.blocksize_label, row=row, col=0)
        row += 1
        self.w4.addWidget(self.blocksize_slider, row=row, col=0)
        row += 1

        self.w4.addWidget(self.offset_label, row=row, col=0)
        row += 1
        self.w4.addWidget(self.offset_slider, row=row, col=0)
        row += 1

        # Global threshold
        self.w4.addWidget(self.threshold_label, row=row, col=0)
        self.w4.addWidget(self.manualThreshold, row=row, col=1)
        row += 1
        self.w4.addWidget(self.threshold_slider, row=row, col=0)
        row += 1

        # Morphological operations
        self.w4.addWidget(self.hole_label, row=row, col=0)
        self.w4.addWidget(self.fillHoles, row=row, col=1)
        row += 1
        self.w4.addWidget(self.hole_slider, row=row, col=0)
        row += 1

        self.w4.addWidget(self.speckle_label, row=row, col=0)
        self.w4.addWidget(self.removeSpeckle, row=row, col=1)
        row += 1
        self.w4.addWidget(self.speckle_slider, row=row, col=0)

    # Image loading and processing methods

    def loadTiff(self) -> None:
        """Load TIFF file for overlay."""
        try:
            self.overlayFileName = self.loadTiff_button.value()

            if not self.overlayFileName:
                self.logger.warning("No TIFF file selected")
                return

            if not os.path.exists(self.overlayFileName):
                self.logger.error(f"TIFF file not found: {self.overlayFileName}")
                return

            self.logger.info(f"Loading TIFF file: {self.overlayFileName}")

            # Load overlay file
            self.overlayIMG, metadata = open_tiff(self.overlayFileName, None)

            # Handle stack images (take first frame)
            if len(self.overlayIMG.shape) > 2:
                self.overlayIMG = self.overlayIMG[0]
                self.logger.debug("Extracted first frame from image stack")

            # Store original for processing
            self.originalIMG = self.overlayIMG.copy()

            # Create overlay visualization
            self.overlay()

            # Update binary processing
            self.updateBinary()

            # Set slider ranges based on image intensity
            self._update_slider_ranges()

            # Add intensity information to tracking data
            self.addActinIntensity()

            self.logger.info(f"TIFF overlay loaded successfully: {self.overlayIMG.shape}")

        except Exception as e:
            self.logger.error(f"Error loading TIFF file: {e}")
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(f"Error loading TIFF: {str(e)}")

    def loadWindow(self) -> None:
        """Load image data from active FLIKA window."""
        try:
            self.importWindow = self.loadWindow_button.value()

            if self.importWindow is None:
                self.logger.warning("No window selected for import")
                return

            self.logger.info("Loading image from active window")

            # Load image data
            self.overlayIMG = self.importWindow.image

            # Store original
            self.originalIMG = self.overlayIMG.copy()

            # Create overlay
            self.overlay()

            # Update binary processing
            self.updateBinary()

            # Set slider ranges
            self._update_slider_ranges()

            # Add intensity information
            self.addActinIntensity()

            self.logger.info(f"Window overlay loaded successfully: {self.overlayIMG.shape}")

        except Exception as e:
            self.logger.error(f"Error loading window: {e}")

    def _update_slider_ranges(self) -> None:
        """Update slider ranges based on loaded image."""
        try:
            if self.overlayIMG is not None:
                img_min, img_max = np.min(self.overlayIMG), np.max(self.overlayIMG)

                # Update threshold sliders
                self.threshold_slider.setRange(img_min, img_max)
                self.pointThreshold_slider.setRange(img_min, img_max)

                self.logger.debug(f"Slider ranges updated: {img_min} - {img_max}")

        except Exception as e:
            self.logger.error(f"Error updating slider ranges: {e}")

    def overlay(self) -> None:
        """Create image overlay with histogram controls."""
        try:
            if self.overlayIMG is None:
                self.logger.warning("No overlay image available")
                return

            self.overlayedIMG = self.overlayIMG.copy()

            # Apply gamma correction if enabled
            if self.gammaCorrect.isChecked():
                gamma_value = self.gamma.value() / 100.0  # Convert to actual gamma
                self.overlayedIMG = gammaCorrect(self.overlayedIMG, gamma_value)

            # Create background item
            self.bgItem = pg.ImageItem()
            opacity = self.opacity.value() / 10.0
            self.bgItem.setImage(
                self.overlayedIMG,
                autoRange=False,
                autoLevels=False,
                opacity=opacity
            )

            # Set composition mode for blending
            self.bgItem.setCompositionMode(QPainter.CompositionMode_SourceOver)
            self.overlayWindow.view.addItem(self.bgItem)

            # Add histogram widget
            self.bgItem.hist_lut = HistogramLUTWidget(fillHistogram=False)
            self.bgItem.hist_lut.setMinimumWidth(110)
            self.bgItem.hist_lut.setImageItem(self.bgItem)
            self.overlayWindow.ui.gridLayout.addWidget(
                self.bgItem.hist_lut, 0, 4, 1, 4
            )

            self.logger.debug("Overlay created successfully")

        except Exception as e:
            self.logger.error(f"Error creating overlay: {e}")

    def updateGamma(self) -> None:
        """Update gamma correction."""
        try:
            if not hasattr(self, 'bgItem') or self.bgItem is None:
                return

            # Get current histogram levels
            levels = self.bgItem.hist_lut.getLevels()

            if self.gammaCorrect.isChecked():
                gamma_value = self.gamma.value() / 100.0
                self.overlayIMG = gammaCorrect(self.originalIMG, gamma_value)
            else:
                self.overlayIMG = self.originalIMG.copy()

            # Update image with preserved levels
            opacity = self.opacity.value() / 10.0
            self.bgItem.setImage(
                self.overlayIMG,
                autoLevels=False,
                levels=levels,
                opacity=opacity
            )

            # Update binary processing
            self.updateBinary()

            self.logger.debug(f"Gamma correction updated: {self.gamma.value()/100.0}")

        except Exception as e:
            self.logger.error(f"Error updating gamma: {e}")

    def resetGamma(self) -> None:
        """Reset gamma correction."""
        try:
            if self.gammaCorrect.isChecked():
                self.updateGamma()
            else:
                if hasattr(self, 'bgItem') and self.bgItem is not None:
                    levels = self.bgItem.hist_lut.getLevels()
                    self.overlayIMG = self.originalIMG.copy()
                    opacity = self.opacity.value() / 10.0
                    self.bgItem.setImage(
                        self.overlayIMG,
                        autoLevels=False,
                        levels=levels,
                        opacity=opacity
                    )

            self.logger.debug("Gamma reset")

        except Exception as e:
            self.logger.error(f"Error resetting gamma: {e}")

    def updateOpacity(self) -> None:
        """Update overlay opacity."""
        try:
            if not hasattr(self, 'bgItem') or self.bgItem is None:
                return

            levels = self.bgItem.hist_lut.getLevels()
            opacity = self.opacity.value() / 10.0

            self.bgItem.setImage(
                self.overlayIMG,
                autoLevels=False,
                levels=levels,
                opacity=opacity
            )

            self.logger.debug(f"Opacity updated: {opacity}")

        except Exception as e:
            self.logger.error(f"Error updating opacity: {e}")

    def updateBinary(self) -> None:
        """Update binary image processing."""
        try:
            if self.overlayIMG is None:
                return

            actin_img = self.overlayIMG.copy()

            # Apply Gaussian blur if enabled
            if self.gaussianBlur.isChecked():
                sigma = self.gaussian_slider.value()
                actin_img = gaussian(actin_img, sigma=sigma)

            # Apply thresholding
            thresh_mask = self._calculate_threshold(actin_img)

            # Create binary image
            binary = actin_img > thresh_mask

            # Apply morphological operations
            if self.fillHoles.isChecked():
                kernel_size = int(self.hole_slider.value())
                binary = closing(binary, square(kernel_size))

            if self.removeSpeckle.isChecked():
                min_size = int(self.speckle_slider.value())
                binary = remove_small_objects(binary, min_size=min_size)

            # Convert to int and store
            self.binary = binary.astype(int)

            # Display binary image
            self.binaryWindow.setImage(self.binary)

            # Update filament detection
            self.detectFilaments()

            self.logger.debug("Binary image updated")

        except Exception as e:
            self.logger.error(f"Error updating binary image: {e}")

    def _calculate_threshold(self, image: np.ndarray) -> Union[float, np.ndarray]:
        """Calculate threshold value or array."""
        try:
            if self.manualThreshold.isChecked():
                # Global threshold
                thresh = self.threshold_slider.value()
                if self.gaussianBlur.isChecked():
                    thresh = thresh / 100000  # Adjust for blurred images
                return thresh
            else:
                # Local threshold
                block_size = int(self.blocksize_slider.value())

                # Ensure odd block size
                if block_size % 2 == 0:
                    block_size += 1

                offset = int(self.offset_slider.value())

                # Adjust offset for blurred images
                if self.gaussianBlur.isChecked():
                    offset = offset / 100000

                return threshold_local(
                    image,
                    block_size,
                    method='gaussian',
                    offset=offset
                )

        except Exception as e:
            self.logger.error(f"Error calculating threshold: {e}")
            return 0

    # Data analysis and visualization methods

    def detectFilaments(self) -> None:
        """Detect and analyze filaments in binary image."""
        try:
            if self.binary is None:
                return

            self.logger.debug("Starting filament detection")

            # Prepare images (flip to match PyQtGraph layout)
            actin_img = np.rot90(self.overlayedIMG)
            actin_img = np.flipud(actin_img)
            binary = np.rot90(self.binary)
            binary = np.flipud(binary)

            # Clear existing filament visualizations
            self.clearActinOutlines()
            self.actinLabels = []

            # Label connected components
            labels = measure.label(binary)
            props = measure.regionprops(labels, actin_img)

            orientation_list = []

            # Process each detected region
            for index in range(1, labels.max() + 1):
                if index >= len(props):
                    continue

                prop = props[index - 1]  # props is 0-indexed
                area = prop.area

                # Check size constraints
                size_ok = self._check_filament_size(area)
                if not size_ok:
                    continue

                # Store filament region
                self.actinLabels.append(labels == index)

                # Visualize filament if enabled
                if self.showFilaments:
                    self._visualize_filament(labels == index)

                # Extract orientation
                orientation = prop.orientation
                orientation_list.append(math.degrees(orientation))

            # Process orientations
            corrected_orientations = self._correct_orientations(orientation_list)

            self.logger.info(f"Detected {len(self.actinLabels)} filaments")

        except Exception as e:
            self.logger.error(f"Error in filament detection: {e}")

    def _check_filament_size(self, area: int) -> bool:
        """Check if filament meets size criteria."""
        min_size = self.minSize_slider.value()

        if self.maxSizeLimit.isChecked():
            max_size = self.maxSize_slider.value()
            return min_size <= area <= max_size
        else:
            return area >= min_size

    def _visualize_filament(self, region: np.ndarray) -> None:
        """Visualize a single filament region."""
        try:
            # Find contours
            contours = measure.find_contours(region)
            if not contours:
                return

            contour = contours[0]
            y, x = contour.T

            # Create path items for both windows
            pathitem = QGraphicsPathItem(self.overlayWindow.view)
            pathitem2 = QGraphicsPathItem(self.binaryWindow.view)

            # Set pen properties
            pen = pg.mkPen(color='red', width=1)
            pen2 = pg.mkPen(color='red', width=3)

            pathitem.setPen(pen)
            pathitem2.setPen(pen2)

            # Add to views
            self.overlayWindow.view.addItem(pathitem)
            self.binaryWindow.view.addItem(pathitem2)

            # Store references
            self.pathitemsActin.append(pathitem)
            self.pathitemsActin_binary.append(pathitem2)

            # Create path
            if len(x) > 0:
                path = QPainterPath(QPointF(x[0], y[0]))
                for i in range(1, len(x)):
                    path.lineTo(QPointF(x[i], y[i]))

                pathitem.setPath(path)
                pathitem2.setPath(path)

        except Exception as e:
            self.logger.error(f"Error visualizing filament: {e}")

    def _correct_orientations(self, orientations: List[float]) -> List[float]:
        """Correct orientation values to 0-180 degree range."""
        corrected = []
        for deg in orientations:
            if deg < 0:
                deg = deg + 180
            corrected.append(deg)
        return corrected

    def detectTrackAxis(self) -> None:
        """Analyze track directions using confidence ellipses."""
        try:
            self.logger.info("Starting track axis detection")

            # Get track data
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                track_data = self.mainGUI.filteredData
            else:
                track_data = self.mainGUI.data

            if track_data is None or track_data.empty:
                self.logger.warning("No track data available")
                return

            track_list = track_data['track_number'].unique()
            track_list = track_list[~pd.isna(track_list)]

            if len(track_list) == 0:
                self.logger.warning("No valid tracks found")
                return

            # Create analysis plot
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_aspect('equal', adjustable='box')
            ax.scatter(track_data['zeroed_X'], track_data['zeroed_Y'], alpha=0.6)
            ax.axvline(c='grey', lw=1, alpha=0.5)
            ax.axhline(c='grey', lw=1, alpha=0.5)
            ax.set_xlabel('X Position (relative to origin)')
            ax.set_ylabel('Y Position (relative to origin)')
            ax.set_title('Track Direction Analysis')

            degree_list = []

            # Analyze each track
            for track_id in tqdm(track_list, desc="Analyzing tracks"):
                track = track_data[track_data['track_number'] == track_id]

                if len(track) < 3:  # Need at least 3 points for meaningful analysis
                    continue

                try:
                    _, degree = self._confidence_ellipse(
                        track['zeroed_X'], track['zeroed_Y'], ax,
                        edgecolor='red', alpha=0.3
                    )
                    degree_list.append(degree)
                except Exception as e:
                    self.logger.debug(f"Error analyzing track {track_id}: {e}")
                    continue

            # Process results
            corrected_degrees = []
            for deg in degree_list:
                if deg < 0:
                    deg = deg + 180
                corrected_degrees.append(deg)

            # Show results
            plt.tight_layout()
            plt.show()

            # Create histogram of directions
            if corrected_degrees:
                fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
                ax2.hist(corrected_degrees, bins=20, alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Track Direction (degrees)')
                ax2.set_ylabel('Count')
                ax2.set_title(f'Distribution of Track Directions (n={len(corrected_degrees)})')
                plt.tight_layout()
                plt.show()

            self.logger.info(f"Track axis detection completed: {len(corrected_degrees)} tracks analyzed")

        except Exception as e:
            self.logger.error(f"Error in track axis detection: {e}")

    def _confidence_ellipse(self, x: np.ndarray, y: np.ndarray, ax,
                          nstd: float = 2.0, facecolor: str = 'none', **kwargs):
        """
        Create confidence ellipse for track data.

        Returns:
            Tuple of (axes, major_axis_angle_degrees)
        """
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("Invalid input arrays")

        # Calculate covariance and center
        cov = np.cov(x, y)
        center = (np.mean(x), np.mean(y))

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Calculate angle and dimensions
        vx, vy = eigvecs[:, 0]
        theta = np.arctan2(vy, vx)
        width, height = 2 * nstd * np.sqrt(eigvals)

        # Create ellipse
        ellipse = Ellipse(
            xy=center, width=width, height=height,
            angle=np.degrees(theta), facecolor=facecolor, **kwargs
        )

        # Create direction arrow
        r = max(width, height) / 2
        arrow = Arrow(
            center[0], center[1],
            r * np.cos(theta), r * np.sin(theta),
            width=max(width, height) * 0.1
        )

        ax.add_patch(ellipse)
        ax.add_patch(arrow)

        return ax, np.degrees(theta)

    def addActinIntensity(self) -> None:
        """Add actin intensity values to tracking data."""
        try:
            if self.overlayedIMG is None or not hasattr(self.mainGUI, 'data'):
                return

            self.logger.debug("Adding actin intensity to tracking data")

            # Add intensity to main data
            if self.mainGUI.data is not None:
                self.mainGUI.data['actin_intensity'] = self._get_intensities(
                    self.overlayedIMG,
                    self.mainGUI.data['x'],
                    self.mainGUI.data['y']
                )

            # Add intensity to filtered data
            if (self.mainGUI.useFilteredData and
                self.mainGUI.filteredData is not None):
                self.mainGUI.filteredData['actin_intensity'] = self._get_intensities(
                    self.overlayedIMG,
                    self.mainGUI.filteredData['x'],
                    self.mainGUI.filteredData['y']
                )

            # Add intensity to unlinked data
            if hasattr(self.mainGUI, 'data_unlinked') and self.mainGUI.data_unlinked is not None:
                self.mainGUI.data_unlinked['actin_intensity'] = self._get_intensities(
                    self.overlayedIMG,
                    self.mainGUI.data_unlinked['x'],
                    self.mainGUI.data_unlinked['y']
                )

            self.logger.debug("Actin intensity added successfully")

        except Exception as e:
            self.logger.error(f"Error adding actin intensity: {e}")

    def _get_intensities(self, img: np.ndarray, x_positions: np.ndarray,
                        y_positions: np.ndarray) -> np.ndarray:
        """Extract intensity values at specified positions."""
        try:
            y_max, x_max = img.shape

            # Convert to integers and handle edge cases
            x_pos = np.round(x_positions).astype(int)
            y_pos = np.round(y_positions).astype(int)

            # Clip to image bounds
            x_pos = np.clip(x_pos, 0, x_max - 1)
            y_pos = np.clip(y_pos, 0, y_max - 1)

            # Extract intensities
            intensities = img[y_pos, x_pos]

            return intensities

        except Exception as e:
            self.logger.error(f"Error extracting intensities: {e}")
            return np.zeros(len(x_positions))

    # Point visualization methods

    def toggleData(self) -> None:
        """Toggle data point visibility."""
        if self.pointsPlotted:
            self.hidePoints()
        else:
            self.plotPoints()

    def plotPoints(self) -> None:
        """Plot data points on overlay with optional intensity filtering."""
        try:
            # Clear existing scatter plot
            if self.pointMapScatter is not None:
                self.pointMapScatter.clear()

            # Get data source
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                df = self.mainGUI.filteredData.copy()
            else:
                df = self.mainGUI.data.copy()

            if df is None or df.empty:
                self.logger.warning("No data available for plotting")
                return

            # Add unlinked points if enabled
            if (self.mainGUI.displayUnlinkedPoints and
                hasattr(self.mainGUI, 'data_unlinked') and
                self.mainGUI.data_unlinked is not None):
                df = pd.concat([df, self.mainGUI.data_unlinked])

            # Apply intensity filtering if enabled
            if self.pointThreshold.isChecked() and 'actin_intensity' in df.columns:
                threshold = self.pointThreshold_slider.value()
                df = df[df['actin_intensity'] >= threshold]
                self.logger.debug(f"Applied intensity filter: {len(df)} points remain")

            if df.empty:
                self.logger.warning("No points remain after filtering")
                return

            # Create scatter plot
            self.pointMapScatter = pg.ScatterPlotItem(
                size=4,
                pen=None,
                brush=pg.mkBrush(30, 255, 35, 200)
            )
            self.pointMapScatter.setData(df['x'], df['y'])
            self.overlayWindow.view.addItem(self.pointMapScatter)

            self.pointsPlotted = True
            self.showData_button.setText('Hide Data Points')

            self.logger.info(f"Plotted {len(df)} data points")

        except Exception as e:
            self.logger.error(f"Error plotting points: {e}")

    def hidePoints(self) -> None:
        """Hide data points from overlay."""
        try:
            if self.pointMapScatter is not None:
                self.overlayWindow.view.removeItem(self.pointMapScatter)
                self.pointMapScatter = None

            self.pointsPlotted = False
            self.showData_button.setText('Show Data Points')

            self.logger.debug("Data points hidden")

        except Exception as e:
            self.logger.error(f"Error hiding points: {e}")

    # Cleanup and utility methods

    def clearActinOutlines(self) -> None:
        """Clear actin filament outlines from both windows."""
        try:
            # Clear from overlay window
            for pathitem in self.pathitemsActin:
                try:
                    self.overlayWindow.view.removeItem(pathitem)
                except:
                    pass

            # Clear from binary window
            for pathitem in self.pathitemsActin_binary:
                try:
                    self.binaryWindow.view.removeItem(pathitem)
                except:
                    pass

            # Reset lists
            self.pathitemsActin = []
            self.pathitemsActin_binary = []

            self.logger.debug("Actin outlines cleared")

        except Exception as e:
            self.logger.error(f"Error clearing actin outlines: {e}")

    def clearTracks(self) -> None:
        """Clear track paths from overlay window."""
        try:
            for pathitem in self.pathitems:
                try:
                    self.overlayWindow.view.removeItem(pathitem)
                except:
                    pass

            self.pathitems = []
            self.logger.debug("Track paths cleared")

        except Exception as e:
            self.logger.error(f"Error clearing tracks: {e}")

    def showDetectedFilaments(self) -> None:
        """Toggle filament detection visualization."""
        try:
            self.showFilaments = not self.showFilaments

            if self.showFilaments:
                self.getFilaments_button.setText('Hide Filaments')
            else:
                self.getFilaments_button.setText('Detect Filaments')

            self.detectFilaments()

            self.logger.debug(f"Filament visualization: {'enabled' if self.showFilaments else 'disabled'}")

        except Exception as e:
            self.logger.error(f"Error toggling filament detection: {e}")

    def loadData(self) -> None:
        """Load data image from main GUI."""
        try:
            if hasattr(self.mainGUI, 'plotWindow') and self.mainGUI.plotWindow:
                self.dataIMG = self.mainGUI.plotWindow.image
                self.overlayWindow.setImage(self.dataIMG)
                self.logger.debug("Data image loaded from main GUI")
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")

    def show(self) -> None:
        """Show the overlay window."""
        try:
            self.win.show()
            self.logger.debug("Overlay window shown")
        except Exception as e:
            self.logger.error(f"Error showing overlay window: {e}")

    def close(self) -> None:
        """Close the overlay window."""
        try:
            # Clear visualizations
            self.clearActinOutlines()
            self.clearTracks()
            if self.pointsPlotted:
                self.hidePoints()

            self.win.close()
            self.logger.debug("Overlay window closed")
        except Exception as e:
            self.logger.error(f"Error closing overlay window: {e}")

    def hide(self) -> None:
        """Hide the overlay window."""
        try:
            self.win.hide()
            self.logger.debug("Overlay window hidden")
        except Exception as e:
            self.logger.error(f"Error hiding overlay window: {e}")

    def export_analysis(self, filename: str) -> None:
        """
        Export analysis results to file.

        Args:
            filename: Output filename for analysis results
        """
        try:
            results = {
                'filament_count': len(self.actinLabels),
                'overlay_file': self.overlayFileName,
                'binary_processing': {
                    'gaussian_blur': self.gaussianBlur.isChecked(),
                    'gaussian_sigma': self.gaussian_slider.value() if self.gaussianBlur.isChecked() else None,
                    'manual_threshold': self.manualThreshold.isChecked(),
                    'threshold_value': self.threshold_slider.value() if self.manualThreshold.isChecked() else None,
                    'local_block_size': self.blocksize_slider.value(),
                    'local_offset': self.offset_slider.value(),
                    'hole_filling': self.fillHoles.isChecked(),
                    'speckle_removal': self.removeSpeckle.isChecked()
                },
                'size_filters': {
                    'min_size': self.minSize_slider.value(),
                    'max_size': self.maxSize_slider.value() if self.maxSizeLimit.isChecked() else None
                }
            }

            import json
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)

            self.logger.info(f"Analysis results exported to {filename}")

        except Exception as e:
            self.logger.error(f"Error exporting analysis: {e}")
