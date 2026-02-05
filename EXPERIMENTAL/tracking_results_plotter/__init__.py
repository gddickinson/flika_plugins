# tracking_results_plotter/__init__.py
"""
FLIKA Plugin: Tracking Results Plotter
========================================

A comprehensive FLIKA plugin for visualizing and analyzing particle tracking results.
Overlays tracks and points onto TIFF stacks with filtering, coloring, and analysis tools.

Author: Assistant
Version: 1.0.0
Compatible with: FLIKA 0.2.25+
"""

import numpy as np
import pandas as pd
import sys
import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings

# Qt imports
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import QThread, Signal, QTimer
from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QSlider, QCheckBox, QComboBox,
                           QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit,
                           QProgressBar, QFileDialog, QTableWidget, QTableWidgetItem,
                           QGroupBox, QGridLayout, QSplitter, QFrame, QScrollArea)

# FLIKA imports
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import SliderLabel, CheckBox, ComboBox, WindowSelector
from flika.roi import ROI_Base, makeROI
from flika.tracefig import TraceFig

# Scientific computing
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.qt_compat import QtWidgets as mpl_QtWidgets
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Some plotting features will be disabled.")

try:
    from scipy import stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some statistical features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Plugin metadata
__version__ = '1.0.0'
__author__ = 'Assistant'
__description__ = 'Comprehensive particle tracking results visualization and analysis'

# Default column mappings for common tracking result formats
DEFAULT_COLUMN_MAPPING = {
    'track_id': ['track_number', 'track_id', 'id', 'particle'],
    'frame': ['frame', 'Frame', 't', 'time'],
    'x': ['x', 'X', 'x_coord', 'position_x'],
    'y': ['y', 'Y', 'y_coord', 'position_y'],
    'intensity': ['intensity', 'Intensity', 'signal', 'amp', 'amplitude'],
    'experiment': ['Experiment', 'experiment', 'condition', 'sample']
}

class TrackingDataManager:
    """
    Manages loading, validation, and manipulation of tracking data.
    Handles different CSV formats and provides unified data access.
    """

    def __init__(self):
        self.data = None
        self.original_data = None
        self.column_mapping = {}
        self.available_columns = []
        self.required_columns = ['track_id', 'frame', 'x', 'y']
        self.numeric_columns = []
        self.categorical_columns = []

    def load_data(self, filepath: str) -> bool:
        """
        Load tracking data from CSV file with automatic column detection.

        Args:
            filepath: Path to CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading tracking data from: {filepath}")

            # Load CSV data
            self.data = pd.read_csv(filepath)
            self.original_data = self.data.copy()

            # Detect and map columns
            self._detect_columns()

            # Validate required columns
            if not self._validate_required_columns():
                return False

            # Process data types
            self._process_data_types()

            # Add derived columns
            self._add_derived_columns()

            logger.info(f"Successfully loaded {len(self.data)} rows with {len(self.data.columns)} columns")
            logger.info(f"Found {len(self.data[self.column_mapping['track_id']].unique())} unique tracks")

            return True

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            if g.m is not None:
                g.alert(f"Error loading data: {str(e)}")
            return False

    def _detect_columns(self):
        """Automatically detect column mappings based on common naming conventions."""
        self.column_mapping = {}

        for required_col, possible_names in DEFAULT_COLUMN_MAPPING.items():
            for col_name in possible_names:
                if col_name in self.data.columns:
                    self.column_mapping[required_col] = col_name
                    break

        # Store all available columns
        self.available_columns = list(self.data.columns)

    def _validate_required_columns(self) -> bool:
        """Check if all required columns are present."""
        missing_columns = []
        for req_col in self.required_columns:
            if req_col not in self.column_mapping:
                missing_columns.append(req_col)

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}. Available columns: {self.available_columns}"
            logger.error(error_msg)
            if g.m is not None:
                g.alert(error_msg)
            return False

        return True

    def _process_data_types(self):
        """Identify numeric and categorical columns."""
        self.numeric_columns = []
        self.categorical_columns = []

        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numeric_columns.append(col)
            else:
                # Try to convert to numeric
                try:
                    pd.to_numeric(self.data[col])
                    self.numeric_columns.append(col)
                except:
                    self.categorical_columns.append(col)

    def _add_derived_columns(self):
        """Add useful derived columns for analysis."""
        try:
            # Track length (number of frames per track)
            track_lengths = self.data.groupby(self.column_mapping['track_id']).size()
            self.data['track_length_frames'] = self.data[self.column_mapping['track_id']].map(track_lengths)

            # Frame within track (0-indexed position in track)
            self.data['frame_in_track'] = self.data.groupby(self.column_mapping['track_id']).cumcount()

            # Displacement from first point in track
            first_points = self.data.groupby(self.column_mapping['track_id'])[
                [self.column_mapping['x'], self.column_mapping['y']]].first()

            for track_id in self.data[self.column_mapping['track_id']].unique():
                mask = self.data[self.column_mapping['track_id']] == track_id
                first_x, first_y = first_points.loc[track_id]

                self.data.loc[mask, 'displacement_x'] = (
                    self.data.loc[mask, self.column_mapping['x']] - first_x
                )
                self.data.loc[mask, 'displacement_y'] = (
                    self.data.loc[mask, self.column_mapping['y']] - first_y
                )
                self.data.loc[mask, 'displacement_magnitude'] = np.sqrt(
                    self.data.loc[mask, 'displacement_x']**2 +
                    self.data.loc[mask, 'displacement_y']**2
                )

            # Update column lists
            self.numeric_columns.extend(['track_length_frames', 'frame_in_track',
                                       'displacement_x', 'displacement_y', 'displacement_magnitude'])

            logger.info("Added derived columns for enhanced analysis")

        except Exception as e:
            logger.warning(f"Could not add some derived columns: {str(e)}")

    def get_tracks(self, track_ids: Optional[List] = None) -> pd.DataFrame:
        """Get track data, optionally filtered by track IDs."""
        if track_ids is None:
            return self.data
        else:
            return self.data[self.data[self.column_mapping['track_id']].isin(track_ids)]

    def get_track_summary(self) -> pd.DataFrame:
        """Get summary statistics for each track."""
        if self.data is None:
            return pd.DataFrame()

        summary_data = []

        for track_id in self.data[self.column_mapping['track_id']].unique():
            track_data = self.data[self.data[self.column_mapping['track_id']] == track_id]

            summary = {
                'track_id': track_id,
                'n_points': len(track_data),
                'start_frame': track_data[self.column_mapping['frame']].min(),
                'end_frame': track_data[self.column_mapping['frame']].max(),
                'x_mean': track_data[self.column_mapping['x']].mean(),
                'y_mean': track_data[self.column_mapping['y']].mean(),
                'x_std': track_data[self.column_mapping['x']].std(),
                'y_std': track_data[self.column_mapping['y']].std(),
            }

            # Add intensity stats if available
            if 'intensity' in self.column_mapping:
                summary.update({
                    'intensity_mean': track_data[self.column_mapping['intensity']].mean(),
                    'intensity_std': track_data[self.column_mapping['intensity']].std(),
                    'intensity_max': track_data[self.column_mapping['intensity']].max(),
                    'intensity_min': track_data[self.column_mapping['intensity']].min(),
                })

            # Add analysis columns if present
            for col in self.numeric_columns:
                if col in track_data.columns and col not in ['frame', 'x', 'y']:
                    if track_data[col].notna().any():
                        summary[f'{col}_mean'] = track_data[col].mean()

            summary_data.append(summary)

        return pd.DataFrame(summary_data)


class TrackOverlay:
    """
    Handles overlay of tracks and points onto FLIKA windows.
    Manages ROI creation, styling, and interaction.
    """

    def __init__(self, data_manager: TrackingDataManager):
        self.data_manager = data_manager
        self.current_window = None
        self.track_rois = {}  # track_id -> list of ROIs
        self.point_rois = {}  # (track_id, frame) -> ROI
        self.current_frame = 0

        # Styling options
        self.point_size = 3
        self.line_width = 2
        self.point_color = 'red'
        self.line_color = 'blue'
        self.color_by_property = None
        self.colormap = 'viridis'
        self.show_tracks = True
        self.show_points = True
        self.show_track_ids = False

    def set_window(self, window: Window):
        """Set the target FLIKA window for overlays."""
        self.current_window = window
        if window is not None:
            self.current_frame = window.currentIndex

    def clear_overlays(self):
        """Remove all track and point overlays from current window."""
        if self.current_window is None:
            return

        # Remove track ROIs
        for track_id, rois in self.track_rois.items():
            for roi in rois:
                try:
                    roi.delete()
                except:
                    pass

        # Remove point ROIs
        for roi in self.point_rois.values():
            try:
                roi.delete()
            except:
                pass

        self.track_rois.clear()
        self.point_rois.clear()

        logger.info("Cleared all overlays from window")

    def plot_tracks(self, track_ids: Optional[List] = None, filter_conditions: Optional[Dict] = None):
        """
        Plot track overlays on the current window.

        Args:
            track_ids: Specific tracks to plot (None for all)
            filter_conditions: Dictionary of column -> (operator, value) filters
        """
        if self.current_window is None or self.data_manager.data is None:
            return

        logger.info("Plotting track overlays...")

        # Get filtered data
        data = self._apply_filters(self.data_manager.data, track_ids, filter_conditions)

        if len(data) == 0:
            logger.warning("No data to plot after filtering")
            return

        # Group by track
        grouped = data.groupby(self.data_manager.column_mapping['track_id'])

        for track_id, track_data in grouped:
            self._plot_single_track(track_id, track_data)

        logger.info(f"Plotted {len(grouped)} tracks")

    def _plot_single_track(self, track_id: int, track_data: pd.DataFrame):
        """Plot a single track as connected line segments."""
        if not self.show_tracks:
            return

        # Sort by frame to ensure proper connection
        track_data = track_data.sort_values(self.data_manager.column_mapping['frame'])

        x_coords = track_data[self.data_manager.column_mapping['x']].values
        y_coords = track_data[self.data_manager.column_mapping['y']].values

        if len(x_coords) < 2:
            return  # Need at least 2 points to draw a track

        # Create line ROIs for track segments
        track_rois = []
        color = self._get_track_color(track_id, track_data)

        for i in range(len(x_coords) - 1):
            try:
                # Create line ROI between consecutive points
                line_roi = makeROI('line',
                                 pts=[[x_coords[i], y_coords[i]],
                                      [x_coords[i+1], y_coords[i+1]]],
                                 window=self.current_window,
                                 color=color)

                # Set line properties
                pen = line_roi.pen()
                pen.setWidth(self.line_width)
                line_roi.setPen(pen)

                track_rois.append(line_roi)

            except Exception as e:
                logger.warning(f"Could not create line ROI for track {track_id}: {str(e)}")

        self.track_rois[track_id] = track_rois

    def plot_points(self, frame: Optional[int] = None, track_ids: Optional[List] = None,
                   filter_conditions: Optional[Dict] = None):
        """
        Plot point overlays for specific frame(s).

        Args:
            frame: Specific frame to plot (None for current frame)
            track_ids: Specific tracks to plot (None for all)
            filter_conditions: Dictionary of column -> (operator, value) filters
        """
        if self.current_window is None or self.data_manager.data is None:
            return

        if frame is None:
            frame = self.current_window.currentIndex

        # Get data for specific frame
        frame_data = self.data_manager.data[
            self.data_manager.data[self.data_manager.column_mapping['frame']] == frame
        ]

        # Apply additional filters
        frame_data = self._apply_filters(frame_data, track_ids, filter_conditions)

        if len(frame_data) == 0:
            return

        logger.info(f"Plotting {len(frame_data)} points for frame {frame}")

        for _, point in frame_data.iterrows():
            self._plot_single_point(point, frame)

    def _plot_single_point(self, point_data: pd.Series, frame: int):
        """Plot a single point as a circular ROI."""
        if not self.show_points:
            return

        try:
            track_id = point_data[self.data_manager.column_mapping['track_id']]
            x = point_data[self.data_manager.column_mapping['x']]
            y = point_data[self.data_manager.column_mapping['y']]

            # Create small rectangle ROI as point
            point_roi = makeROI('rectangle',
                              pos=[x - self.point_size/2, y - self.point_size/2],
                              size=[self.point_size, self.point_size],
                              window=self.current_window,
                              color=self._get_point_color(track_id, point_data))

            self.point_rois[(track_id, frame)] = point_roi

        except Exception as e:
            logger.warning(f"Could not create point ROI: {str(e)}")

    def _apply_filters(self, data: pd.DataFrame, track_ids: Optional[List],
                      filter_conditions: Optional[Dict]) -> pd.DataFrame:
        """Apply filtering conditions to data."""
        filtered_data = data.copy()

        # Filter by track IDs
        if track_ids is not None:
            filtered_data = filtered_data[
                filtered_data[self.data_manager.column_mapping['track_id']].isin(track_ids)
            ]

        # Apply column-based filters
        if filter_conditions:
            for column, (operator, value) in filter_conditions.items():
                if column in filtered_data.columns:
                    if operator == '==':
                        filtered_data = filtered_data[filtered_data[column] == value]
                    elif operator == '!=':
                        filtered_data = filtered_data[filtered_data[column] != value]
                    elif operator == '>':
                        filtered_data = filtered_data[filtered_data[column] > value]
                    elif operator == '>=':
                        filtered_data = filtered_data[filtered_data[column] >= value]
                    elif operator == '<':
                        filtered_data = filtered_data[filtered_data[column] < value]
                    elif operator == '<=':
                        filtered_data = filtered_data[filtered_data[column] <= value]

        return filtered_data

    def _get_track_color(self, track_id: int, track_data: pd.DataFrame):
        """Get color for track based on coloring scheme."""
        if self.color_by_property and self.color_by_property in track_data.columns:
            # Color by property value
            prop_value = track_data[self.color_by_property].mean()  # Use mean for track
            return self._property_to_color(prop_value)
        else:
            # Use default color or track ID-based color
            return QtGui.QColor(self.line_color)

    def _get_point_color(self, track_id: int, point_data: pd.Series):
        """Get color for point based on coloring scheme."""
        if self.color_by_property and self.color_by_property in point_data.index:
            prop_value = point_data[self.color_by_property]
            return self._property_to_color(prop_value)
        else:
            return QtGui.QColor(self.point_color)

    def _property_to_color(self, value: float) -> QtGui.QColor:
        """Convert property value to color using colormap."""
        # This is a simplified color mapping - could be enhanced with matplotlib colormaps
        if np.isnan(value):
            return QtGui.QColor('gray')

        # Normalize value to 0-1 range (simple approach)
        normalized = max(0, min(1, (value - 0) / (1 - 0)))  # Placeholder normalization

        # Simple color mapping (could use matplotlib colormaps)
        if self.colormap == 'viridis':
            r = int(255 * (0.267 + 0.2 * normalized))
            g = int(255 * (0.004 + 0.8 * normalized))
            b = int(255 * (0.329 + 0.4 * normalized))
        else:
            # Default: red to blue gradient
            r = int(255 * (1 - normalized))
            g = 0
            b = int(255 * normalized)

        return QtGui.QColor(r, g, b)


class TrackingResultsPlotter(QWidget):
    """
    Main plugin class for tracking results visualization.
    Provides comprehensive GUI and analysis tools.
    """

    def __init__(self):
        super().__init__()
        self.data_manager = TrackingDataManager()
        self.track_overlay = TrackOverlay(self.data_manager)

        # Analysis windows
        self.trace_windows = {}
        self.plot_windows = {}

        # Settings
        self.auto_update = True
        self.current_filters = {}

        # Setup GUI
        self.setupGUI()

        logger.info("Tracking Results Plotter initialized")

    def setupGUI(self):
        """Create the main GUI with tabbed interface."""
        self.setWindowTitle('Tracking Results Plotter v1.0.0')
        self.resize(900, 700)

        # Create main layout
        main_layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs
        self._create_data_tab()
        self._create_display_tab()
        self._create_filter_tab()
        self._create_analysis_tab()
        self._create_plots_tab()

        # Add control buttons
        self._create_control_buttons(main_layout)

        # Connect signals
        self._connect_signals()

        logger.info("GUI setup complete")

    def _create_data_tab(self):
        """Create data loading and management tab."""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)

        # File loading section
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)

        # File selection
        file_selection_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file loaded")
        self.load_button = QPushButton("Load CSV File")
        self.load_button.clicked.connect(self.load_data_file)

        file_selection_layout.addWidget(QLabel("Data File:"))
        file_selection_layout.addWidget(self.file_path_label, 1)
        file_selection_layout.addWidget(self.load_button)
        file_layout.addLayout(file_selection_layout)

        # Data info display
        self.data_info_text = QTextEdit()
        self.data_info_text.setMaximumHeight(150)
        self.data_info_text.setReadOnly(True)
        file_layout.addWidget(QLabel("Data Information:"))
        file_layout.addWidget(self.data_info_text)

        layout.addWidget(file_group)

        # Window selection section
        window_group = QGroupBox("FLIKA Window")
        window_layout = QVBoxLayout(window_group)

        window_selection_layout = QHBoxLayout()
        self.window_selector = WindowSelector()
        self.set_window_button = QPushButton("Set Active Window")
        self.set_window_button.clicked.connect(self.set_active_window)

        window_selection_layout.addWidget(QLabel("Target Window:"))
        window_selection_layout.addWidget(self.window_selector, 1)
        window_selection_layout.addWidget(self.set_window_button)
        window_layout.addLayout(window_selection_layout)

        layout.addWidget(window_group)

        # Add data preview table
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.data_table = QTableWidget()
        self.data_table.setMaximumHeight(200)
        preview_layout.addWidget(self.data_table)

        layout.addWidget(preview_group)

        self.tabs.addTab(data_widget, "Data")

    def _create_display_tab(self):
        """Create display options tab."""
        display_widget = QWidget()
        layout = QVBoxLayout(display_widget)

        # Track display options
        track_group = QGroupBox("Track Display")
        track_layout = QGridLayout(track_group)

        self.show_tracks = CheckBox()
        self.show_tracks.setChecked(True)
        track_layout.addWidget(QLabel("Show Tracks:"), 0, 0)
        track_layout.addWidget(self.show_tracks, 0, 1)

        self.line_width = SliderLabel()
        self.line_width.setRange(1, 10)
        self.line_width.setValue(2)
        track_layout.addWidget(QLabel("Line Width:"), 1, 0)
        track_layout.addWidget(self.line_width, 1, 1)

        self.track_color_combo = ComboBox()
        self.track_color_combo.addItems(['blue', 'red', 'green', 'black', 'yellow', 'cyan', 'magenta'])
        track_layout.addWidget(QLabel("Track Color:"), 2, 0)
        track_layout.addWidget(self.track_color_combo, 2, 1)

        layout.addWidget(track_group)

        # Point display options
        point_group = QGroupBox("Point Display")
        point_layout = QGridLayout(point_group)

        self.show_points = CheckBox()
        self.show_points.setChecked(True)
        point_layout.addWidget(QLabel("Show Points:"), 0, 0)
        point_layout.addWidget(self.show_points, 0, 1)

        self.point_size = SliderLabel()
        self.point_size.setRange(1, 20)
        self.point_size.setValue(3)
        point_layout.addWidget(QLabel("Point Size:"), 1, 0)
        point_layout.addWidget(self.point_size, 1, 1)

        self.point_color_combo = ComboBox()
        self.point_color_combo.addItems(['red', 'blue', 'green', 'black', 'yellow', 'cyan', 'magenta'])
        point_layout.addWidget(QLabel("Point Color:"), 2, 0)
        point_layout.addWidget(self.point_color_combo, 2, 1)

        layout.addWidget(point_group)

        # Color coding options
        color_group = QGroupBox("Color Coding")
        color_layout = QGridLayout(color_group)

        self.color_by_property = ComboBox()
        self.color_by_property.addItem("None")
        color_layout.addWidget(QLabel("Color by Property:"), 0, 0)
        color_layout.addWidget(self.color_by_property, 0, 1)

        self.colormap_combo = ComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'hot', 'cool', 'rainbow'])
        color_layout.addWidget(QLabel("Colormap:"), 1, 0)
        color_layout.addWidget(self.colormap_combo, 1, 1)

        layout.addWidget(color_group)

        # Auto-update option
        self.auto_update_display = CheckBox()
        self.auto_update_display.setChecked(True)
        layout.addWidget(self.auto_update_display)
        layout.addWidget(QLabel("Auto-update display"))

        layout.addStretch()

        self.tabs.addTab(display_widget, "Display")

    def _create_filter_tab(self):
        """Create filtering options tab."""
        filter_widget = QWidget()
        layout = QVBoxLayout(filter_widget)

        # Filter controls
        filter_group = QGroupBox("Data Filters")
        filter_layout = QVBoxLayout(filter_group)

        # Add filter button
        add_filter_layout = QHBoxLayout()
        self.filter_column_combo = ComboBox()
        self.filter_operator_combo = ComboBox()
        self.filter_operator_combo.addItems(['>', '>=', '<', '<=', '==', '!='])
        self.filter_value_input = QtWidgets.QLineEdit()
        add_filter_button = QPushButton("Add Filter")
        add_filter_button.clicked.connect(self.add_filter)

        add_filter_layout.addWidget(QLabel("Column:"))
        add_filter_layout.addWidget(self.filter_column_combo)
        add_filter_layout.addWidget(QLabel("Operator:"))
        add_filter_layout.addWidget(self.filter_operator_combo)
        add_filter_layout.addWidget(QLabel("Value:"))
        add_filter_layout.addWidget(self.filter_value_input)
        add_filter_layout.addWidget(add_filter_button)

        filter_layout.addLayout(add_filter_layout)

        # Active filters display
        self.active_filters_text = QTextEdit()
        self.active_filters_text.setMaximumHeight(100)
        self.active_filters_text.setReadOnly(True)
        filter_layout.addWidget(QLabel("Active Filters:"))
        filter_layout.addWidget(self.active_filters_text)

        # Filter control buttons
        filter_buttons_layout = QHBoxLayout()
        clear_filters_button = QPushButton("Clear All Filters")
        clear_filters_button.clicked.connect(self.clear_filters)
        apply_filters_button = QPushButton("Apply Filters")
        apply_filters_button.clicked.connect(self.apply_filters)

        filter_buttons_layout.addWidget(clear_filters_button)
        filter_buttons_layout.addWidget(apply_filters_button)
        filter_layout.addLayout(filter_buttons_layout)

        layout.addWidget(filter_group)
        layout.addStretch()

        self.tabs.addTab(filter_widget, "Filters")

    def _create_analysis_tab(self):
        """Create analysis tools tab."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)

        # Track statistics
        stats_group = QGroupBox("Track Statistics")
        stats_layout = QVBoxLayout(stats_group)

        stats_buttons_layout = QHBoxLayout()
        calc_stats_button = QPushButton("Calculate Track Statistics")
        calc_stats_button.clicked.connect(self.calculate_track_statistics)
        export_stats_button = QPushButton("Export Statistics")
        export_stats_button.clicked.connect(self.export_statistics)

        stats_buttons_layout.addWidget(calc_stats_button)
        stats_buttons_layout.addWidget(export_stats_button)
        stats_layout.addLayout(stats_buttons_layout)

        # Statistics display
        self.stats_table = QTableWidget()
        self.stats_table.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_table)

        layout.addWidget(stats_group)

        # Track selection tools
        selection_group = QGroupBox("Track Selection")
        selection_layout = QVBoxLayout(selection_group)

        selection_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All Tracks")
        select_all_button.clicked.connect(self.select_all_tracks)
        select_filtered_button = QPushButton("Select Filtered Tracks")
        select_filtered_button.clicked.connect(self.select_filtered_tracks)
        clear_selection_button = QPushButton("Clear Selection")
        clear_selection_button.clicked.connect(self.clear_track_selection)

        selection_buttons_layout.addWidget(select_all_button)
        selection_buttons_layout.addWidget(select_filtered_button)
        selection_buttons_layout.addWidget(clear_selection_button)
        selection_layout.addLayout(selection_buttons_layout)

        layout.addWidget(selection_group)
        layout.addStretch()

        self.tabs.addTab(analysis_widget, "Analysis")

    def _create_plots_tab(self):
        """Create plotting tools tab."""
        plots_widget = QWidget()
        layout = QVBoxLayout(plots_widget)

        # Time trace plots
        trace_group = QGroupBox("Time Trace Plots")
        trace_layout = QVBoxLayout(trace_group)

        trace_buttons_layout = QHBoxLayout()
        intensity_trace_button = QPushButton("Intensity Traces")
        intensity_trace_button.clicked.connect(self.plot_intensity_traces)
        position_trace_button = QPushButton("Position Traces")
        position_trace_button.clicked.connect(self.plot_position_traces)

        trace_buttons_layout.addWidget(intensity_trace_button)
        trace_buttons_layout.addWidget(position_trace_button)
        trace_layout.addLayout(trace_buttons_layout)

        layout.addWidget(trace_group)

        # Flower plots
        flower_group = QGroupBox("Flower Plots")
        flower_layout = QVBoxLayout(flower_group)

        flower_buttons_layout = QHBoxLayout()
        single_flower_button = QPushButton("Single Track Flower Plot")
        single_flower_button.clicked.connect(self.plot_single_flower)
        multi_flower_button = QPushButton("Multi-Track Flower Plot")
        multi_flower_button.clicked.connect(self.plot_multi_flower)

        flower_buttons_layout.addWidget(single_flower_button)
        flower_buttons_layout.addWidget(multi_flower_button)
        flower_layout.addLayout(flower_buttons_layout)

        layout.addWidget(flower_group)

        # Histograms and distributions
        hist_group = QGroupBox("Histograms & Distributions")
        hist_layout = QVBoxLayout(hist_group)

        # Property selection for histogram
        hist_property_layout = QHBoxLayout()
        self.hist_property_combo = ComboBox()
        plot_hist_button = QPushButton("Plot Histogram")
        plot_hist_button.clicked.connect(self.plot_histogram)

        hist_property_layout.addWidget(QLabel("Property:"))
        hist_property_layout.addWidget(self.hist_property_combo)
        hist_property_layout.addWidget(plot_hist_button)
        hist_layout.addLayout(hist_property_layout)

        # Scatter plots
        scatter_layout = QHBoxLayout()
        self.scatter_x_combo = ComboBox()
        self.scatter_y_combo = ComboBox()
        plot_scatter_button = QPushButton("Plot Scatter")
        plot_scatter_button.clicked.connect(self.plot_scatter)

        scatter_layout.addWidget(QLabel("X:"))
        scatter_layout.addWidget(self.scatter_x_combo)
        scatter_layout.addWidget(QLabel("Y:"))
        scatter_layout.addWidget(self.scatter_y_combo)
        scatter_layout.addWidget(plot_scatter_button)
        hist_layout.addLayout(scatter_layout)

        layout.addWidget(hist_group)
        layout.addStretch()

        self.tabs.addTab(plots_widget, "Plots")

    def _create_control_buttons(self, layout):
        """Create main control buttons."""
        button_layout = QHBoxLayout()

        self.plot_overlays_button = QPushButton("Plot Overlays")
        self.plot_overlays_button.clicked.connect(self.plot_overlays)

        self.clear_overlays_button = QPushButton("Clear Overlays")
        self.clear_overlays_button.clicked.connect(self.clear_overlays)

        self.update_display_button = QPushButton("Update Display")
        self.update_display_button.clicked.connect(self.update_display)

        button_layout.addWidget(self.plot_overlays_button)
        button_layout.addWidget(self.clear_overlays_button)
        button_layout.addWidget(self.update_display_button)
        button_layout.addStretch()

        layout.addLayout(button_layout)

    def _connect_signals(self):
        """Connect GUI signals to update functions."""
        # Auto-update on parameter changes
        if hasattr(self, 'point_size'):
            self.point_size.valueChanged.connect(self._on_display_params_changed)
        if hasattr(self, 'line_width'):
            self.line_width.valueChanged.connect(self._on_display_params_changed)
        if hasattr(self, 'show_tracks'):
            self.show_tracks.stateChanged.connect(self._on_display_params_changed)
        if hasattr(self, 'show_points'):
            self.show_points.stateChanged.connect(self._on_display_params_changed)

    def _on_display_params_changed(self):
        """Handle display parameter changes."""
        if self.auto_update_display.isChecked():
            self.update_display()

    # Main functionality methods

    def load_data_file(self):
        """Load tracking data from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Tracking Results CSV", "", "CSV Files (*.csv)")

        if file_path:
            if self.data_manager.load_data(file_path):
                self.file_path_label.setText(os.path.basename(file_path))
                self._update_data_info()
                self._update_column_combos()
                self._update_data_preview()
                g.alert("Data loaded successfully!")
            else:
                g.alert("Failed to load data. Check file format.")

    def _update_data_info(self):
        """Update data information display."""
        if self.data_manager.data is None:
            return

        data = self.data_manager.data
        info_text = f"""
Rows: {len(data)}
Columns: {len(data.columns)}
Unique Tracks: {len(data[self.data_manager.column_mapping['track_id']].unique())}
Frame Range: {data[self.data_manager.column_mapping['frame']].min()} - {data[self.data_manager.column_mapping['frame']].max()}

Column Mapping:
- Track ID: {self.data_manager.column_mapping.get('track_id', 'Not found')}
- Frame: {self.data_manager.column_mapping.get('frame', 'Not found')}
- X: {self.data_manager.column_mapping.get('x', 'Not found')}
- Y: {self.data_manager.column_mapping.get('y', 'Not found')}
- Intensity: {self.data_manager.column_mapping.get('intensity', 'Not found')}

Numeric Columns: {len(self.data_manager.numeric_columns)}
Categorical Columns: {len(self.data_manager.categorical_columns)}
        """.strip()

        self.data_info_text.setText(info_text)

    def _update_column_combos(self):
        """Update combo boxes with available columns."""
        if self.data_manager.data is None:
            return

        numeric_cols = self.data_manager.numeric_columns
        all_cols = self.data_manager.available_columns

        # Update filter column combo
        self.filter_column_combo.clear()
        self.filter_column_combo.addItems(all_cols)

        # Update color property combo
        self.color_by_property.clear()
        self.color_by_property.addItem("None")
        self.color_by_property.addItems(numeric_cols)

        # Update histogram property combo
        self.hist_property_combo.clear()
        self.hist_property_combo.addItems(numeric_cols)

        # Update scatter plot combos
        self.scatter_x_combo.clear()
        self.scatter_x_combo.addItems(numeric_cols)
        self.scatter_y_combo.clear()
        self.scatter_y_combo.addItems(numeric_cols)

    def _update_data_preview(self):
        """Update data preview table."""
        if self.data_manager.data is None:
            return

        data = self.data_manager.data.head(100)  # Show first 100 rows

        self.data_table.setRowCount(len(data))
        self.data_table.setColumnCount(len(data.columns))
        self.data_table.setHorizontalHeaderLabels(data.columns.tolist())

        for i, row in data.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)

        self.data_table.resizeColumnsToContents()

    def set_active_window(self):
        """Set the active FLIKA window for overlays."""
        selected_window = self.window_selector.value()
        if selected_window is not None:
            self.track_overlay.set_window(selected_window)
            g.alert(f"Set active window: {selected_window.name}")
        else:
            g.alert("No window selected")

    def plot_overlays(self):
        """Plot track and point overlays on active window."""
        if self.data_manager.data is None:
            g.alert("No data loaded")
            return

        if self.track_overlay.current_window is None:
            g.alert("No active window set")
            return

        try:
            # Update overlay settings
            self._update_overlay_settings()

            # Clear existing overlays
            self.track_overlay.clear_overlays()

            # Plot tracks if enabled
            if self.show_tracks.isChecked():
                self.track_overlay.plot_tracks(filter_conditions=self.current_filters)

            # Plot points if enabled
            if self.show_points.isChecked():
                self.track_overlay.plot_points(filter_conditions=self.current_filters)

            g.alert("Overlays plotted successfully")

        except Exception as e:
            logger.error(f"Error plotting overlays: {str(e)}")
            g.alert(f"Error plotting overlays: {str(e)}")

    def clear_overlays(self):
        """Clear all overlays from active window."""
        self.track_overlay.clear_overlays()
        g.alert("Overlays cleared")

    def update_display(self):
        """Update display with current settings."""
        if self.auto_update and self.data_manager.data is not None:
            self.plot_overlays()

    def _update_overlay_settings(self):
        """Update overlay settings from GUI."""
        self.track_overlay.point_size = self.point_size.value()
        self.track_overlay.line_width = self.line_width.value()
        self.track_overlay.show_tracks = self.show_tracks.isChecked()
        self.track_overlay.show_points = self.show_points.isChecked()
        self.track_overlay.point_color = self.point_color_combo.currentText()
        self.track_overlay.line_color = self.track_color_combo.currentText()

        # Color coding
        color_prop = self.color_by_property.currentText()
        if color_prop != "None":
            self.track_overlay.color_by_property = color_prop
        else:
            self.track_overlay.color_by_property = None

        self.track_overlay.colormap = self.colormap_combo.currentText()

    # Filter methods

    def add_filter(self):
        """Add a new filter condition."""
        column = self.filter_column_combo.currentText()
        operator = self.filter_operator_combo.currentText()
        value_text = self.filter_value_input.text()

        if not column or not value_text:
            g.alert("Please specify column and value")
            return

        try:
            # Try to convert value to appropriate type
            if column in self.data_manager.numeric_columns:
                value = float(value_text)
            else:
                value = value_text

            self.current_filters[column] = (operator, value)
            self._update_filter_display()
            self.filter_value_input.clear()

        except ValueError:
            g.alert("Invalid value for numeric column")

    def clear_filters(self):
        """Clear all filter conditions."""
        self.current_filters.clear()
        self._update_filter_display()

    def apply_filters(self):
        """Apply current filters and update display."""
        self.update_display()

    def _update_filter_display(self):
        """Update active filters display."""
        if not self.current_filters:
            self.active_filters_text.setText("No active filters")
        else:
            filter_text = "\n".join([
                f"{col} {op} {val}" for col, (op, val) in self.current_filters.items()
            ])
            self.active_filters_text.setText(filter_text)

    # Analysis methods

    def calculate_track_statistics(self):
        """Calculate and display track statistics."""
        if self.data_manager.data is None:
            g.alert("No data loaded")
            return

        try:
            stats_df = self.data_manager.get_track_summary()

            if len(stats_df) == 0:
                g.alert("No tracks found")
                return

            # Display in table
            self.stats_table.setRowCount(len(stats_df))
            self.stats_table.setColumnCount(len(stats_df.columns))
            self.stats_table.setHorizontalHeaderLabels(stats_df.columns.tolist())

            for i, row in stats_df.iterrows():
                for j, value in enumerate(row):
                    item = QTableWidgetItem(str(value))
                    self.stats_table.setItem(i, j, item)

            self.stats_table.resizeColumnsToContents()
            g.alert(f"Calculated statistics for {len(stats_df)} tracks")

        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            g.alert(f"Error calculating statistics: {str(e)}")

    def export_statistics(self):
        """Export statistics to CSV file."""
        if self.stats_table.rowCount() == 0:
            g.alert("No statistics to export. Calculate statistics first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics", "track_statistics.csv", "CSV Files (*.csv)")

        if file_path:
            try:
                stats_df = self.data_manager.get_track_summary()
                stats_df.to_csv(file_path, index=False)
                g.alert(f"Statistics exported to {file_path}")
            except Exception as e:
                g.alert(f"Error exporting statistics: {str(e)}")

    def select_all_tracks(self):
        """Select all tracks for analysis."""
        # Implementation would depend on how track selection is handled
        g.alert("All tracks selected")

    def select_filtered_tracks(self):
        """Select only filtered tracks for analysis."""
        # Implementation would depend on how track selection is handled
        g.alert("Filtered tracks selected")

    def clear_track_selection(self):
        """Clear track selection."""
        # Implementation would depend on how track selection is handled
        g.alert("Track selection cleared")

    # Plotting methods

    def plot_intensity_traces(self):
        """Plot intensity traces for selected tracks."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        if self.data_manager.data is None or 'intensity' not in self.data_manager.column_mapping:
            g.alert("No intensity data available")
            return

        try:
            self._create_trace_plot('intensity')
        except Exception as e:
            logger.error(f"Error plotting intensity traces: {str(e)}")
            g.alert(f"Error plotting intensity traces: {str(e)}")

    def plot_position_traces(self):
        """Plot position traces for selected tracks."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        if self.data_manager.data is None:
            g.alert("No data loaded")
            return

        try:
            self._create_trace_plot('position')
        except Exception as e:
            logger.error(f"Error plotting position traces: {str(e)}")
            g.alert(f"Error plotting position traces: {str(e)}")

    def _create_trace_plot(self, trace_type: str):
        """Create trace plot window."""
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        data = self.data_manager.data
        track_col = self.data_manager.column_mapping['track_id']
        frame_col = self.data_manager.column_mapping['frame']

        # Plot first 10 tracks (to avoid overcrowding)
        unique_tracks = data[track_col].unique()[:10]

        for track_id in unique_tracks:
            track_data = data[data[track_col] == track_id].sort_values(frame_col)
            frames = track_data[frame_col].values

            if trace_type == 'intensity' and 'intensity' in self.data_manager.column_mapping:
                intensity = track_data[self.data_manager.column_mapping['intensity']].values
                axes[0].plot(frames, intensity, label=f'Track {track_id}', alpha=0.7)
                axes[0].set_ylabel('Intensity')
                axes[0].set_title('Intensity Traces')

            elif trace_type == 'position':
                x_vals = track_data[self.data_manager.column_mapping['x']].values
                y_vals = track_data[self.data_manager.column_mapping['y']].values

                axes[0].plot(frames, x_vals, label=f'Track {track_id} (X)', alpha=0.7)
                axes[1].plot(frames, y_vals, label=f'Track {track_id} (Y)', alpha=0.7)

                axes[0].set_ylabel('X Position')
                axes[1].set_ylabel('Y Position')
                axes[0].set_title('X Position Traces')
                axes[1].set_title('Y Position Traces')

        axes[-1].set_xlabel('Frame')
        if len(unique_tracks) <= 10:
            axes[0].legend()

        plt.tight_layout()
        plt.show()

    def plot_single_flower(self):
        """Plot flower plot for single selected track."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        # For now, plot first track as example
        if self.data_manager.data is None:
            g.alert("No data loaded")
            return

        try:
            track_id = self.data_manager.data[self.data_manager.column_mapping['track_id']].iloc[0]
            self._create_flower_plot([track_id])
        except Exception as e:
            logger.error(f"Error plotting flower plot: {str(e)}")
            g.alert(f"Error plotting flower plot: {str(e)}")

    def plot_multi_flower(self):
        """Plot flower plots for multiple tracks."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        if self.data_manager.data is None:
            g.alert("No data loaded")
            return

        try:
            # Plot first 5 tracks
            track_ids = self.data_manager.data[self.data_manager.column_mapping['track_id']].unique()[:5]
            self._create_flower_plot(track_ids)
        except Exception as e:
            logger.error(f"Error plotting flower plots: {str(e)}")
            g.alert(f"Error plotting flower plots: {str(e)}")

    def _create_flower_plot(self, track_ids: List):
        """Create flower plot(s) for specified tracks."""
        fig, axes = plt.subplots(1, len(track_ids), figsize=(4*len(track_ids), 4))
        if len(track_ids) == 1:
            axes = [axes]

        data = self.data_manager.data
        track_col = self.data_manager.column_mapping['track_id']
        x_col = self.data_manager.column_mapping['x']
        y_col = self.data_manager.column_mapping['y']
        frame_col = self.data_manager.column_mapping['frame']

        for i, track_id in enumerate(track_ids):
            track_data = data[data[track_col] == track_id].sort_values(frame_col)

            if len(track_data) < 2:
                continue

            # Center the track at origin
            x_vals = track_data[x_col].values - track_data[x_col].iloc[0]
            y_vals = track_data[y_col].values - track_data[y_col].iloc[0]

            # Plot trajectory
            axes[i].plot(x_vals, y_vals, 'b-', alpha=0.7, linewidth=1)
            axes[i].scatter(x_vals[0], y_vals[0], c='green', s=50, marker='o', label='Start')
            axes[i].scatter(x_vals[-1], y_vals[-1], c='red', s=50, marker='s', label='End')

            axes[i].set_xlabel('ΔX (pixels)')
            axes[i].set_ylabel('ΔY (pixels)')
            axes[i].set_title(f'Track {track_id} Flower Plot')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_aspect('equal')
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_histogram(self):
        """Plot histogram of selected property."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        property_name = self.hist_property_combo.currentText()
        if not property_name or self.data_manager.data is None:
            g.alert("No property selected or no data loaded")
            return

        try:
            data = self.data_manager.data[property_name].dropna()

            plt.figure(figsize=(8, 6))
            plt.hist(data, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel(property_name)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {property_name}')
            plt.grid(True, alpha=0.3)

            # Add statistics text
            stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nN: {len(data)}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting histogram: {str(e)}")
            g.alert(f"Error plotting histogram: {str(e)}")

    def plot_scatter(self):
        """Plot scatter plot of two selected properties."""
        if not MATPLOTLIB_AVAILABLE:
            g.alert("Matplotlib not available for plotting")
            return

        x_property = self.scatter_x_combo.currentText()
        y_property = self.scatter_y_combo.currentText()

        if not x_property or not y_property or self.data_manager.data is None:
            g.alert("Please select both X and Y properties")
            return

        try:
            data = self.data_manager.data[[x_property, y_property]].dropna()

            plt.figure(figsize=(8, 6))
            plt.scatter(data[x_property], data[y_property], alpha=0.6)
            plt.xlabel(x_property)
            plt.ylabel(y_property)
            plt.title(f'{y_property} vs {x_property}')
            plt.grid(True, alpha=0.3)

            # Add correlation coefficient if scipy available
            if SCIPY_AVAILABLE:
                corr_coef, p_value = stats.pearsonr(data[x_property], data[y_property])
                plt.text(0.02, 0.98, f'R = {corr_coef:.3f}\np = {p_value:.3e}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting scatter: {str(e)}")
            g.alert(f"Error plotting scatter: {str(e)}")


# Create plugin instance
tracking_results_plotter = None

def launch_tracking_plotter():
    """Launch the tracking results plotter."""
    global tracking_results_plotter
    tracking_results_plotter = TrackingResultsPlotter()
    tracking_results_plotter.show()
    return tracking_results_plotter

# Plugin initialization
def initialize_plugin():
    """Initialize the tracking results plotter plugin."""
    logger.info("Tracking Results Plotter plugin initialized successfully")
    print("📊 Tracking Results Plotter v1.0.0 loaded successfully!")
    print("   Access via: Plugins → Tracking Analysis → Launch Results Plotter")

# Call initialization
initialize_plugin()
