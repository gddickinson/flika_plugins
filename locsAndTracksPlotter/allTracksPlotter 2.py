#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All Tracks Plotter Module for FLIKA Tracking Plugin

This module provides visualization and analysis tools for displaying all tracks
simultaneously. It includes intensity analysis, trace extraction, and statistical
visualization for comprehensive track analysis.

Created on Fri Jun  2 15:18:37 2023
@author: george
"""

import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (QMainWindow, QLabel, QPushButton, QMessageBox,
                           QFileDialog, QProgressDialog, QApplication)
from qtpy.QtGui import QFont
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

# PyQtGraph dock imports
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# Plugin imports
from .helperFunctions import dictFromList

# Set up logging
logger = logging.getLogger(__name__)


class AllTracksPlot:
    """
    GUI for visualizing and analyzing all tracked data simultaneously.

    This class provides comprehensive tools for analyzing multiple tracks,
    including intensity traces, statistical analysis, and export capabilities.
    It extracts and displays track data from regions of interest around each
    tracked particle.

    Features:
    - Intensity trace extraction for all tracks
    - Statistical visualization (mean, max projections)
    - Interactive ROI analysis
    - Data export capabilities
    - Progress tracking for long operations

    Attributes:
        mainGUI: Reference to main GUI instance
        win: Main window containing dock area
        area: Dock area for organizing widgets
        trace_data: Dictionary containing extracted trace data
        image_data: Processed image stack data
    """

    def __init__(self, mainGUI):
        """
        Initialize the all tracks plotter.

        Args:
            mainGUI: Reference to main GUI instance containing track data
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Analysis parameters
        self.roi_size: int = 5  # Size of ROI for intensity extraction
        self.image_stack: Optional[np.ndarray] = None
        self.padded_stack: Optional[np.ndarray] = None
        self.cropped_stack: Optional[np.ndarray] = None

        # Data storage
        self.trace_list: List[np.ndarray] = []
        self.time_list: List[np.ndarray] = []
        self.track_list: List[str] = []

        # Analysis results
        self.mean_intensity_img: Optional[np.ndarray] = None
        self.max_intensity_img: Optional[np.ndarray] = None

        self._setup_ui()
        self.logger.debug("AllTracksPlot initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window and dock area
            self.win = QMainWindow()
            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.resize(1400, 550)
            self.win.setWindowTitle('All Tracks Intensity Analysis')

            # Create docks
            self._create_docks()

            # Create control widgets
            self._create_controls()

            # Create image displays
            self._create_image_displays()

            # Create trace plot
            self._create_trace_plot()

            # Create ROI selectors
            self._create_roi_selectors()

            self.logger.debug("AllTracksPlot UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up AllTracksPlot UI: {e}")

    def _create_docks(self) -> None:
        """Create dock widgets for different sections."""
        # Control options dock
        self.d2 = Dock("Analysis Options", size=(500, 100))

        # Mean intensity projection dock
        self.d3 = Dock('Mean Intensity Projection', size=(300, 300))

        # Trace plot dock
        self.d4 = Dock('Intensity Traces', size=(750, 300))

        # Max intensity projection dock
        self.d5 = Dock('Max Intensity Projection', size=(300, 300))

        # Line transect docks
        self.d6 = Dock('Mean Line Profile', size=(300, 250))
        self.d7 = Dock('Max Line Profile', size=(300, 250))

        # Arrange docks
        self.area.addDock(self.d4, 'top')
        self.area.addDock(self.d2, 'bottom')
        self.area.addDock(self.d3, 'right', self.d4)
        self.area.addDock(self.d5, 'bottom', self.d3)
        self.area.addDock(self.d6, 'right', self.d3)
        self.area.addDock(self.d7, 'right', self.d5)

    def _create_controls(self) -> None:
        """Create control widgets."""
        self.w2 = pg.LayoutWidget()

        # Track selector
        self.trackSelector = pg.ComboBox()
        self.tracks = {'None': 'None'}
        self.trackSelector.setItems(self.tracks)
        self.trackSelector_label = QLabel("Select Track ID")

        # Track selection mode
        self.selectTrack_checkbox = CheckBox()
        self.selectTrack_checkbox.setChecked(False)

        # Interpolation option
        self.interpolate_checkbox = CheckBox()
        self.interpolate_checkbox.setChecked(True)
        self.interpolate_label = QLabel("Interpolate missing frames")

        # ROI size selector
        self.roiSize_box = pg.SpinBox(value=5, int=True)
        self.roiSize_box.setSingleStep(1)
        self.roiSize_box.setMinimum(1)
        self.roiSize_box.setMaximum(50)
        self.roiSize_box.valueChanged.connect(self._on_roi_size_changed)
        self.roiSize_label = QLabel("ROI size (pixels)")

        # Action buttons
        self.plot_button = QPushButton('Analyze Tracks')
        self.plot_button.pressed.connect(self.plotTracks)

        self.export_button = QPushButton('Export Traces')
        self.export_button.pressed.connect(self.exportTraces)

        # Layout controls
        self._layout_controls()

        # Add to dock
        self.d2.addWidget(self.w2)

    def _layout_controls(self) -> None:
        """Arrange control widgets in layout."""
        # Row 0: Track selection
        self.w2.addWidget(self.trackSelector_label, row=0, col=0)
        self.w2.addWidget(self.selectTrack_checkbox, row=0, col=1)
        self.w2.addWidget(self.trackSelector, row=0, col=2)

        # Row 1: Options
        self.w2.addWidget(self.interpolate_label, row=1, col=0)
        self.w2.addWidget(self.interpolate_checkbox, row=1, col=1)

        # Row 2: ROI size
        self.w2.addWidget(self.roiSize_label, row=2, col=0)
        self.w2.addWidget(self.roiSize_box, row=2, col=1)

        # Row 3: Buttons
        self.w2.addWidget(self.plot_button, row=3, col=2)
        self.w2.addWidget(self.export_button, row=3, col=3)

    def _create_image_displays(self) -> None:
        """Create image display widgets."""
        # Mean intensity display
        self.meanIntensity = pg.ImageView()
        self.meanIntensity.setWindowTitle("Mean Intensity Projection")
        self.d3.addWidget(self.meanIntensity)

        # Max intensity display
        self.maxIntensity = pg.ImageView()
        self.maxIntensity.setWindowTitle("Max Intensity Projection")
        self.d5.addWidget(self.maxIntensity)

    def _create_trace_plot(self) -> None:
        """Create trace plotting widget."""
        self.tracePlot = pg.PlotWidget(title="Intensity Traces for All Tracks")
        self.tracePlot.setLabel('left', 'Intensity', units='AU')
        self.tracePlot.setLabel('bottom', 'Time', units='Frames')
        self.tracePlot.showGrid(x=True, y=True, alpha=0.3)
        self.tracePlot.setLimits(xMin=0)
        self.d4.addWidget(self.tracePlot)

    def _create_roi_selectors(self) -> None:
        """Create ROI selector widgets for line profiles."""
        # Mean intensity line profile
        self.meanTransect = pg.PlotWidget(title="Mean Intensity Line Profile")
        self.meanTransect.setLabel('left', 'Intensity', units='AU')
        self.meanTransect.setLabel('bottom', 'Position', units='pixels')
        self.meanTransect.showGrid(x=True, y=True, alpha=0.3)
        self.d6.addWidget(self.meanTransect)

        # Max intensity line profile
        self.maxTransect = pg.PlotWidget(title="Max Intensity Line Profile")
        self.maxTransect.setLabel('left', 'Intensity', units='AU')
        self.maxTransect.setLabel('bottom', 'Position', units='pixels')
        self.maxTransect.showGrid(x=True, y=True, alpha=0.3)
        self.d7.addWidget(self.maxTransect)

        # Add ROI controls to image displays
        self.ROI_mean = self._add_roi_to_display(self.meanIntensity)
        self.ROI_mean.sigRegionChanged.connect(self.updateMeanTransect)

        self.ROI_max = self._add_roi_to_display(self.maxIntensity)
        self.ROI_max.sigRegionChanged.connect(self.updateMaxTransect)

    def _add_roi_to_display(self, image_view: pg.ImageView) -> pg.ROI:
        """
        Add ROI selector to image display.

        Args:
            image_view: ImageView widget to add ROI to

        Returns:
            ROI widget
        """
        try:
            roi = pg.ROI([0, 0], [self.roi_size, self.roi_size])
            roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            roi.addScaleHandle([0, 0.5], [0.5, 0.5])
            roi.addRotateFreeHandle([1, 1], [0.5, 0.5])
            image_view.addItem(roi)
            return roi
        except Exception as e:
            self.logger.error(f"Error adding ROI to display: {e}")
            return None

    def updateTrackList(self) -> None:
        """Update the track list based on available data."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return

            # Determine data source
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                source_data = self.mainGUI.filteredData
            else:
                source_data = self.mainGUI.data

            if 'track_number' not in source_data.columns:
                self.logger.warning("No track_number column found in data")
                return

            # Create track dictionary
            track_numbers = source_data['track_number'].dropna().unique()
            self.tracks = dictFromList(track_numbers.astype(str))
            self.trackSelector.setItems(self.tracks)

            self.logger.debug(f"Track list updated with {len(track_numbers)} tracks")

        except Exception as e:
            self.logger.error(f"Error updating track list: {e}")

    def _on_roi_size_changed(self) -> None:
        """Handle ROI size change."""
        self.roi_size = int(self.roiSize_box.value())
        self.logger.debug(f"ROI size changed to: {self.roi_size}")

    def plotTracks(self) -> None:
        """Analyze and plot intensity traces for tracks."""
        try:
            self.logger.info("Starting track analysis")

            # Validate data
            if not self._validate_data_for_analysis():
                return

            # Check track count for performance warning
            if not self._check_track_count():
                return

            # Clear previous results
            self.tracePlot.clear()
            self._reset_analysis_data()

            # Extract intensity traces
            self._extract_intensity_traces()

            # Create projections
            self._create_intensity_projections()

            # Plot traces
            self._plot_intensity_traces()

            # Update transects
            self.updateMeanTransect()
            self.updateMaxTransect()

            self.logger.info("Track analysis completed successfully")

        except Exception as e:
            self.logger.error(f"Error in track analysis: {e}")
            QMessageBox.critical(
                self.win, "Analysis Error",
                f"An error occurred during analysis:\n{str(e)}"
            )

    def _validate_data_for_analysis(self) -> bool:
        """Validate that data is available for analysis."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                QMessageBox.warning(
                    self.win, "No Data",
                    "No track data available for analysis."
                )
                return False

            if not hasattr(self.mainGUI, 'plotWindow') or self.mainGUI.plotWindow is None:
                QMessageBox.warning(
                    self.win, "No Image Window",
                    "No image window available for analysis."
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False

    def _check_track_count(self) -> bool:
        """Check track count and warn if too many tracks."""
        try:
            track_count = self.trackSelector.count()

            if track_count > 2000:
                reply = QMessageBox.question(
                    self.win,
                    "Performance Warning",
                    f"Analyzing {track_count} tracks may take a long time. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                return reply == QMessageBox.Yes

            return True

        except Exception as e:
            self.logger.error(f"Error checking track count: {e}")
            return True

    def _reset_analysis_data(self) -> None:
        """Reset analysis data structures."""
        self.trace_list = []
        self.time_list = []
        self.track_list = []
        self.cropped_stack = None

    def _extract_intensity_traces(self) -> None:
        """Extract intensity traces for selected tracks."""
        try:
            # Determine which tracks to analyze
            if self.selectTrack_checkbox.isChecked():
                self.track_list = [str(self.trackSelector.value())]
            else:
                self.track_list = [self.trackSelector.itemText(i)
                                 for i in range(self.trackSelector.count())]

            # Prepare image stack
            self._prepare_image_stack()

            # Create progress dialog
            progress = QProgressDialog(
                "Extracting intensity traces...", "Cancel",
                0, len(self.track_list), self.win
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # Process each track
            for i, track_id_str in enumerate(self.track_list):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Processing track {track_id_str}...")
                QApplication.processEvents()

                self._extract_single_track_trace(track_id_str, i)

            progress.close()

        except Exception as e:
            self.logger.error(f"Error extracting intensity traces: {e}")
            raise

    def _prepare_image_stack(self) -> None:
        """Prepare padded image stack for ROI extraction."""
        try:
            # Get image array from plot window
            self.image_stack = self.mainGUI.plotWindow.imageArray()

            if self.image_stack is None:
                raise ValueError("No image stack available")

            # Pad the stack to avoid boundary issues
            pad_size = self.roi_size
            self.padded_stack = np.pad(
                self.image_stack,
                ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
                'constant',
                constant_values=0
            )

            # Initialize cropped stack
            n_frames = self.image_stack.shape[0]
            n_tracks = len(self.track_list)
            self.cropped_stack = np.zeros(
                (n_tracks, n_frames, self.roi_size, self.roi_size)
            )

            self.logger.debug(f"Image stack prepared: {self.image_stack.shape}")

        except Exception as e:
            self.logger.error(f"Error preparing image stack: {e}")
            raise

    def _extract_single_track_trace(self, track_id_str: str, track_idx: int) -> None:
        """
        Extract intensity trace for a single track.

        Args:
            track_id_str: Track ID as string
            track_idx: Index in track list
        """
        try:
            track_id = int(track_id_str)

            # Get track data
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                source_data = self.mainGUI.filteredData
            else:
                source_data = self.mainGUI.data

            track_data = source_data[source_data['track_number'] == track_id]

            if track_data.empty:
                self.logger.warning(f"No data found for track {track_id}")
                return

            # Extract coordinates and frames
            frames = track_data['frame'].to_numpy()
            x_coords = track_data['x'].to_numpy()
            y_coords = track_data['y'].to_numpy()

            # Handle interpolation
            if self.interpolate_checkbox.isChecked():
                frames, x_coords, y_coords = self._interpolate_track_data(
                    frames, x_coords, y_coords
                )

            # Extract ROI intensities
            trace, time_points = self._extract_roi_intensities(
                frames, x_coords, y_coords, track_idx
            )

            # Store results
            self.trace_list.append(trace)
            self.time_list.append(time_points)

        except Exception as e:
            self.logger.error(f"Error extracting trace for track {track_id_str}: {e}")

    def _interpolate_track_data(self, frames: np.ndarray, x_coords: np.ndarray,
                              y_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate track data for missing frames.

        Args:
            frames: Frame numbers
            x_coords: X coordinates
            y_coords: Y coordinates

        Returns:
            Tuple of interpolated (frames, x_coords, y_coords)
        """
        try:
            if len(frames) < 2:
                return frames, x_coords, y_coords

            # Create complete frame range
            min_frame, max_frame = int(np.min(frames)), int(np.max(frames))
            complete_frames = np.arange(min_frame, max_frame + 1)

            # Interpolate coordinates
            x_interp = np.interp(complete_frames, frames, x_coords)
            y_interp = np.interp(complete_frames, frames, y_coords)

            return complete_frames, x_interp, y_interp

        except Exception as e:
            self.logger.error(f"Error interpolating track data: {e}")
            return frames, x_coords, y_coords

    def _extract_roi_intensities(self, frames: np.ndarray, x_coords: np.ndarray,
                               y_coords: np.ndarray, track_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract ROI intensities for track positions.

        Args:
            frames: Frame numbers
            x_coords: X coordinates
            y_coords: Y coordinates
            track_idx: Track index for storage

        Returns:
            Tuple of (intensity_trace, time_points)
        """
        try:
            n_frames = self.image_stack.shape[0]
            trace = np.full(n_frames, np.nan)

            roi_half = self.roi_size // 2

            for frame, x, y in zip(frames, x_coords, y_coords):
                frame_idx = int(frame)

                if frame_idx < 0 or frame_idx >= n_frames:
                    continue

                # Calculate ROI bounds (accounting for padding)
                x_center = int(np.round(x)) + self.roi_size
                y_center = int(np.round(y)) + self.roi_size

                x_min = x_center - roi_half
                x_max = x_center + roi_half + (self.roi_size % 2)
                y_min = y_center - roi_half
                y_max = y_center + roi_half + (self.roi_size % 2)

                # Extract ROI
                try:
                    roi = self.padded_stack[frame_idx, y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        # Store cropped ROI
                        if roi.shape == (self.roi_size, self.roi_size):
                            self.cropped_stack[track_idx, frame_idx] = roi
                            # Calculate mean intensity (subtract background)
                            background = np.min(self.image_stack[frame_idx])
                            trace[frame_idx] = np.mean(roi) - background

                except IndexError:
                    continue

            # Create time points
            time_points = np.arange(n_frames)

            return trace, time_points

        except Exception as e:
            self.logger.error(f"Error extracting ROI intensities: {e}")
            return np.array([]), np.array([])

    def _create_intensity_projections(self) -> None:
        """Create mean and max intensity projections."""
        try:
            if self.cropped_stack is None:
                return

            # Replace zeros with NaN for proper averaging
            stack_for_analysis = self.cropped_stack.copy()
            stack_for_analysis[stack_for_analysis == 0] = np.nan

            # Create projections
            with np.errstate(invalid='ignore'):
                self.max_intensity_img = np.nanmax(stack_for_analysis, axis=(0, 1))
                self.mean_intensity_img = np.nanmean(stack_for_analysis, axis=(0, 1))

            # Display projections
            if self.max_intensity_img is not None:
                self.maxIntensity.setImage(self.max_intensity_img)
            if self.mean_intensity_img is not None:
                self.meanIntensity.setImage(self.mean_intensity_img)

        except Exception as e:
            self.logger.error(f"Error creating intensity projections: {e}")

    def _plot_intensity_traces(self) -> None:
        """Plot all intensity traces."""
        try:
            if not self.trace_list:
                return

            # Plot each trace with different colors
            for i, (trace, time_points) in enumerate(zip(self.trace_list, self.time_list)):
                if len(trace) == 0:
                    continue

                # Remove NaN values for plotting
                valid_mask = ~np.isnan(trace)
                if not np.any(valid_mask):
                    continue

                # Create plot curve
                color = pg.intColor(i)
                curve = pg.PlotCurveItem()
                curve.setData(
                    x=time_points[valid_mask],
                    y=trace[valid_mask],
                    pen=pg.mkPen(color, width=1)
                )
                self.tracePlot.addItem(curve)

            self.logger.debug(f"Plotted {len(self.trace_list)} intensity traces")

        except Exception as e:
            self.logger.error(f"Error plotting intensity traces: {e}")

    def updateMeanTransect(self) -> None:
        """Update mean intensity line profile."""
        try:
            if self.mean_intensity_img is None or self.ROI_mean is None:
                return

            selected = self.ROI_mean.getArrayRegion(
                self.mean_intensity_img, self.meanIntensity.imageItem
            )

            if selected.size > 0:
                profile = selected.mean(axis=1)
                self.meanTransect.plot(profile, clear=True,
                                     pen=pg.mkPen('b', width=2))

        except Exception as e:
            self.logger.debug(f"Error updating mean transect: {e}")

    def updateMaxTransect(self) -> None:
        """Update max intensity line profile."""
        try:
            if self.max_intensity_img is None or self.ROI_max is None:
                return

            selected = self.ROI_max.getArrayRegion(
                self.max_intensity_img, self.maxIntensity.imageItem
            )

            if selected.size > 0:
                profile = selected.mean(axis=1)
                self.maxTransect.plot(profile, clear=True,
                                    pen=pg.mkPen('r', width=2))

        except Exception as e:
            self.logger.debug(f"Error updating max transect: {e}")

    def exportTraces(self) -> None:
        """Export intensity traces to CSV file."""
        try:
            if not self.trace_list:
                QMessageBox.warning(
                    self.win, "No Data",
                    "No traces available for export. Run analysis first."
                )
                return

            # Get save path
            if hasattr(self.mainGUI, 'filename') and self.mainGUI.filename:
                default_dir = os.path.dirname(self.mainGUI.filename)
                default_name = os.path.join(default_dir, "intensity_traces.csv")
            else:
                default_name = "intensity_traces.csv"

            file_path, _ = QFileDialog.getSaveFileName(
                self.win, 'Export Intensity Traces',
                default_name, 'CSV Files (*.csv)'
            )

            if not file_path:
                return

            # Prepare export data
            self._export_traces_to_csv(file_path)

            # Update status
            message = f'Traces exported to {os.path.basename(file_path)}'
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage(message)

            self.logger.info(f"Traces exported to: {file_path}")

            QMessageBox.information(
                self.win, "Export Complete",
                f"Intensity traces exported to:\n{file_path}"
            )

        except Exception as e:
            self.logger.error(f"Error exporting traces: {e}")
            QMessageBox.critical(
                self.win, "Export Error",
                f"Failed to export traces:\n{str(e)}"
            )

    def _export_traces_to_csv(self, file_path: str) -> None:
        """
        Export traces to CSV file.

        Args:
            file_path: Path for output file
        """
        try:
            # Determine maximum length
            max_length = max(len(trace) for trace in self.trace_list)

            # Create DataFrame
            export_data = {}

            for i, (trace, track_id) in enumerate(zip(self.trace_list, self.track_list)):
                # Pad shorter traces with NaN
                padded_trace = np.full(max_length, np.nan)
                padded_trace[:len(trace)] = trace
                export_data[f'Track_{track_id}'] = padded_trace

            # Add time column
            export_data['Frame'] = np.arange(max_length)

            # Create DataFrame and save
            df = pd.DataFrame(export_data)

            # Reorder columns to put Frame first
            columns = ['Frame'] + [col for col in df.columns if col != 'Frame']
            df = df[columns]

            df.to_csv(file_path, index=False, na_rep='NaN')

        except Exception as e:
            self.logger.error(f"Error writing CSV file: {e}")
            raise

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from the current analysis.

        Returns:
            Dictionary containing analysis statistics
        """
        try:
            if not self.trace_list:
                return {}

            stats = {
                'n_tracks': len(self.trace_list),
                'roi_size': self.roi_size,
                'interpolation_used': self.interpolate_checkbox.isChecked(),
                'trace_statistics': []
            }

            # Calculate statistics for each trace
            for i, (trace, track_id) in enumerate(zip(self.trace_list, self.track_list)):
                valid_trace = trace[~np.isnan(trace)]

                if len(valid_trace) > 0:
                    trace_stats = {
                        'track_id': track_id,
                        'n_points': len(valid_trace),
                        'mean_intensity': np.mean(valid_trace),
                        'std_intensity': np.std(valid_trace),
                        'min_intensity': np.min(valid_trace),
                        'max_intensity': np.max(valid_trace),
                        'median_intensity': np.median(valid_trace)
                    }
                    stats['trace_statistics'].append(trace_stats)

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating analysis statistics: {e}")
            return {}

    def show(self) -> None:
        """Show the all tracks plot window."""
        try:
            self.win.show()
            self.logger.debug("AllTracksPlot window shown")
        except Exception as e:
            self.logger.error(f"Error showing AllTracksPlot window: {e}")

    def close(self) -> None:
        """Close the all tracks plot window."""
        try:
            self.win.close()
            self.logger.debug("AllTracksPlot window closed")
        except Exception as e:
            self.logger.error(f"Error closing AllTracksPlot window: {e}")

    def hide(self) -> None:
        """Hide the all tracks plot window."""
        try:
            self.win.hide()
            self.logger.debug("AllTracksPlot window hidden")
        except Exception as e:
            self.logger.error(f"Error hiding AllTracksPlot window: {e}")
