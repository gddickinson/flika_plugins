#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Window Module for FLIKA Tracking Plugin

This module provides a comprehensive track analysis window that displays
various metrics and visualizations for individual tracks. It includes
intensity traces, position plots, velocity analysis, and nearest neighbor
statistics.

Created on Fri Jun  2 15:40:06 2023
@author: george
"""

import logging
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QGraphicsProxyWidget, QPushButton, QLabel, QListView,
    QWidget, QVBoxLayout, QHBoxLayout,
)
from qtpy.QtGui import QFont
from packaging.version import Version

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g

# Version-specific imports
flika_version = flika.__version__
if Version(flika_version) < Version('0.2.23'):
    from flika.process.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                         ComboBox, BaseProcess_noPriorWindow,
                                         WindowSelector, save_file_gui)
else:
    from flika.utils.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                       ComboBox, BaseProcess_noPriorWindow,
                                       WindowSelector, save_file_gui)

# Plugin imports
from .helperFunctions import rollingFunc

# Set up logging
logger = logging.getLogger(__name__)


class TrackWindow(BaseProcess):
    """
    Track analysis window for displaying individual track metrics.

    This window provides comprehensive visualization and analysis tools
    for individual tracks, including:
    - Intensity traces over time
    - Position tracking (relative to origin)
    - Distance from origin analysis
    - Velocity and movement statistics
    - Nearest neighbor analysis
    - Interactive time indicators

    Attributes:
        mainGUI: Reference to main GUI instance
        win: PyQtGraph graphics layout widget
        plots: Dictionary of plot widgets
        data: Current track data dictionary
        position_indicators: List of position indicator lines
    """

    def __init__(self, mainGUI):
        """
        Initialize the track window.

        Args:
            mainGUI: Reference to main GUI instance
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Track analysis state
        self.showPositionIndicators: bool = False
        self.plotsInitiated: bool = False
        self.data: Dict[int, tuple] = {}  # time -> (x, y) position mapping
        self.r: Optional[pg.RectROI] = None  # Position indicator ROI

        # Plot references
        self.plots: Dict[str, pg.PlotItem] = {}
        self.position_indicators: List[pg.InfiniteLine] = []

        self._setup_ui()
        self.logger.debug("TrackWindow initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Top-level container. We use a normal QWidget so the column
            # selector below can live OUTSIDE the QGraphicsScene — when it
            # was embedded via QGraphicsProxyWidget the dropdown's scrollbar
            # didn't receive mouse / wheel events, so users with many
            # columns couldn't reach the entries beyond the visible window.
            self.win = QWidget()
            self.win.resize(600, 850)
            self.win.setWindowTitle('Track Analysis - Press "T" to select track')

            outer_layout = QVBoxLayout(self.win)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(0)

            # The pyqtgraph layout owns all of the plots and existing
            # proxy-wrapped controls. Code elsewhere in this class still
            # references ``self.plotArea`` (legacy code referenced
            # ``self.win`` for graphics — those usages were redirected to
            # ``self.plotArea``).
            self.plotArea = pg.GraphicsLayoutWidget()
            outer_layout.addWidget(self.plotArea, stretch=1)

            # Native Qt panel for the column selector below the plots.
            self._create_column_selector_panel(outer_layout)

            # Create labels for track info
            self._create_info_labels()

            # Create plots
            self._create_plots()

            # Create control panel
            self._create_controls()

            self.logger.debug("TrackWindow UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up TrackWindow UI: {e}")

    def _create_column_selector_panel(self, parent_layout) -> None:
        """Build the column-selector panel as a native Qt widget.

        The selector lives outside the QGraphicsScene so its dropdown
        receives mouse-wheel and scrollbar-drag events normally, which is
        required to navigate long column lists (40+ entries).
        """
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(8)

        self.columnSelector_Box = pg.ComboBox()
        # Force a QListView popup with a bounded visible-item count and
        # scrollbars when the list overflows. ``combobox-popup: 0`` is
        # required so styles (notably macOS native) honour
        # ``setMaxVisibleItems`` instead of rendering full height.
        list_view = QListView()
        list_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        list_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.columnSelector_Box.setView(list_view)
        self.columnSelector_Box.setMaxVisibleItems(15)
        self.columnSelector_Box.setStyleSheet(
            "QComboBox { combobox-popup: 0; }"
        )
        # Keep the combobox itself a reasonable width even on long labels.
        self.columnSelector_Box.setMinimumWidth(220)
        self.columnSelector_Box.currentIndexChanged.connect(
            self._on_column_selection_changed)

        self.columnValue_label = QLabel('--')
        self.columnValue_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.columnValue_label.setFont(font)
        self.columnValue_label.setMinimumWidth(120)

        layout.addWidget(QLabel('Column:'))
        layout.addWidget(self.columnSelector_Box, stretch=1)
        layout.addSpacing(12)
        layout.addWidget(QLabel('Value:'))
        layout.addWidget(self.columnValue_label, stretch=1)

        parent_layout.addWidget(panel)

    def _create_info_labels(self) -> None:
        """Create information labels."""
        # Track ID label
        self.label = pg.LabelItem(justify='center')
        self.plotArea.addItem(self.label)

        # Track statistics label
        self.label_2 = pg.LabelItem(justify='center')
        self.plotArea.addItem(self.label_2)

        self.plotArea.nextRow()

    def _update_info_labels(self, track_id: int, svm: int, length: int) -> None:
        """Update information labels (track id, SVM, length, sRg)."""
        try:
            self.label.setText(f"<span style='font-size: 16pt'>Track ID = {track_id}</span>")

            # Look up an sRg value if the source data exposes one. Prefer the
            # geometric form ('sRg_geometric') when available; otherwise fall
            # back to the classical scaled radius of gyration.
            srg_text = self._lookup_srg(track_id)
            label_2_html = (
                f"<span style='font-size: 16pt'>SVM = {svm}, "
                f"Length = {length}{srg_text}</span>"
            )
            self.label_2.setText(label_2_html)

            # Update custom column value
            self._update_column_value(track_id)

        except Exception as e:
            self.logger.error(f"Error updating info labels: {e}")

    def _lookup_srg(self, track_id: int) -> str:
        """Return ', sRg = ...' for the given track, or '' if unavailable."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return ''

            if (hasattr(self.mainGUI, 'useFilteredData')
                    and self.mainGUI.useFilteredData
                    and getattr(self.mainGUI, 'filteredData', None) is not None):
                source = self.mainGUI.filteredData
            else:
                source = self.mainGUI.data

            for col in ('sRg_geometric', 'radius_gyration_scaled'):
                if col in source.columns:
                    track_rows = source[source['track_number'] == int(track_id)]
                    if track_rows.empty:
                        continue
                    value = track_rows[col].iloc[0]
                    if pd.isna(value):
                        continue
                    label = 'sRg' if col == 'sRg_geometric' else 'sRg(scaled)'
                    return f", {label} = {float(value):.3f}"
            return ''
        except Exception as e:
            self.logger.error(f"Error looking up sRg: {e}")
            return ''

    def _update_column_value(self, track_id: int) -> None:
        """Update the value display for the selected column."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                self.columnValue_label.setText('--')
                return

            selected_column = self.columnSelector_Box.value()
            if selected_column == 'None':
                self.columnValue_label.setText('--')
                return

            # Determine which data to use (filtered or full dataset)
            if (hasattr(self.mainGUI, 'useFilteredData') and
                self.mainGUI.useFilteredData and
                hasattr(self.mainGUI, 'filteredData') and
                self.mainGUI.filteredData is not None):
                data_source = self.mainGUI.filteredData
            else:
                data_source = self.mainGUI.data

            if selected_column not in data_source.columns:
                self.columnValue_label.setText('--')
                return

            # Get track data
            track_data = data_source[data_source['track_number'] == int(track_id)]

            if track_data.empty:
                self.columnValue_label.setText('--')
                return

            # Get the value (use first occurrence if multiple values)
            value = track_data[selected_column].iloc[0]

            # Format the value appropriately
            if pd.isna(value):
                display_value = 'NaN'
            elif isinstance(value, (int, np.integer)):
                display_value = str(int(value))
            elif isinstance(value, (float, np.floating)):
                display_value = f"{float(value):.4f}"
            else:
                display_value = str(value)

            self.columnValue_label.setText(display_value)

        except Exception as e:
            self.logger.error(f"Error updating column value: {e}")
            self.columnValue_label.setText('--')

    def _on_column_selection_changed(self) -> None:
        """Handle column selection change."""
        try:
            # Update the value display if we have a current track
            if hasattr(self.mainGUI, 'displayTrack') and self.mainGUI.displayTrack is not None:
                self._update_column_value(self.mainGUI.displayTrack)
        except Exception as e:
            self.logger.error(f"Error handling column selection change: {e}")

    def update_column_selector(self) -> None:
        """Update the column selector with available columns from the data."""
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return

            # Get available columns
            columns = list(self.mainGUI.data.columns)

            # Create items dictionary
            column_dict = {col: col for col in columns}
            column_dict['None'] = 'None'

            # Update combo box
            self.columnSelector_Box.setItems(column_dict)

            # Set default selection — prefer scaled-Rg columns so the
            # user sees a meaningful per-track metric immediately.
            preferred = [
                'sRg_geometric',
                'radius_gyration_scaled',
                'radius_gyration',
                'SVM',
            ]
            default_col = next((c for c in preferred if c in columns), None)
            if default_col is not None:
                self.columnSelector_Box.setValue(default_col)
            elif columns:
                self.columnSelector_Box.setValue(columns[0])
            else:
                self.columnSelector_Box.setValue('None')

        except Exception as e:
            self.logger.error(f"Error updating column selector: {e}")

    def _create_plots(self) -> None:
        """Create all analysis plots."""
        try:
            # Intensity plot
            self.plt1 = self.plotArea.addPlot(title='Intensity Trace')
            self.plt1.getAxis('left').enableAutoSIPrefix(False)
            self.plt1.setLabel('left', 'Intensity', units='AU')
            self.plt1.setLabel('bottom', 'Time', units='Frames')
            self.plt1.showGrid(x=True, y=True, alpha=0.3)
            self.plots['intensity'] = self.plt1

            # Track position plot (relative to origin)
            self.plt3 = self.plotArea.addPlot(title='Track Position (Relative to Origin)')
            self.plt3.setAspectLocked()
            self.plt3.showGrid(x=True, y=True, alpha=0.3)
            self.plt3.setXRange(-5, 5)
            self.plt3.setYRange(-5, 5)
            self.plt3.getViewBox().invertY(True)
            self.plt3.setLabel('left', 'Y Position', units='pixels')
            self.plt3.setLabel('bottom', 'X Position', units='pixels')
            self.plots['position'] = self.plt3

            self.plotArea.nextRow()

            # Distance from origin plot
            self.plt2 = self.plotArea.addPlot(title='Distance from Origin')
            self.plt2.getAxis('left').enableAutoSIPrefix(False)
            self.plt2.setLabel('left', 'Distance', units='pixels')
            self.plt2.setLabel('bottom', 'Time', units='Frames')
            self.plt2.showGrid(x=True, y=True, alpha=0.3)
            self.plots['distance'] = self.plt2

            # Nearest neighbor count plot
            self.plt4 = self.plotArea.addPlot(title='Nearest Neighbor Count')
            self.plt4.getAxis('left').enableAutoSIPrefix(False)
            self.plt4.setLabel('left', 'Number of Neighbors', units='count')
            self.plt4.setLabel('bottom', 'Time', units='Frames')
            self.plt4.showGrid(x=True, y=True, alpha=0.3)
            self.plots['neighbors'] = self.plt4

            self.plotArea.nextRow()

            # Instantaneous velocity plot
            self.plt5 = self.plotArea.addPlot(title='Instantaneous Velocity')
            self.plt5.getAxis('left').enableAutoSIPrefix(False)
            self.plt5.setLabel('left', 'Velocity', units='pixels/frame')
            self.plt5.setLabel('bottom', 'Time', units='Frames')
            self.plt5.showGrid(x=True, y=True, alpha=0.3)
            self.plots['velocity'] = self.plt5

            # Intensity variance plot
            self.plt6 = self.plotArea.addPlot(title='Intensity Variance (Rolling)')
            self.plt6.getAxis('left').enableAutoSIPrefix(False)
            self.plt6.setLabel('left', 'Variance', units='intensity')
            self.plt6.setLabel('bottom', 'Time', units='Frames')
            self.plt6.showGrid(x=True, y=True, alpha=0.3)
            self.plots['variance'] = self.plt6

            self.plotArea.nextRow()

        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")

    def _create_controls(self) -> None:
        """Create control panel widgets."""
        try:
            # Position indicator toggle button
            self.optionsPanel = QGraphicsProxyWidget()
            self.positionIndicator_button = QPushButton('Show Position Indicators')
            self.positionIndicator_button.pressed.connect(self.togglePositionIndicator)
            self.optionsPanel.setWidget(self.positionIndicator_button)

            # Nearest neighbor radius selector
            self.optionsPanel2 = QGraphicsProxyWidget()
            self.plotCountSelector = pg.ComboBox()
            self.countTypes = {
                'NN radius: 3': '3',
                'NN radius: 5': '5',
                'NN radius: 10': '10',
                'NN radius: 20': '20',
                'NN radius: 30': '30'
            }
            self.plotCountSelector.setItems(self.countTypes)
            self.plotCountSelector.currentIndexChanged.connect(self._on_nn_radius_changed)
            self.optionsPanel2.setWidget(self.plotCountSelector)

            # Add controls row (position indicator + NN-radius selector).
            # The column selector intentionally lives in a native Qt panel
            # outside the graphics scene — see ``_create_column_selector_panel``.
            self.plotArea.addItem(self.optionsPanel)
            self.plotArea.addItem(self.optionsPanel2)

        except Exception as e:
            self.logger.error(f"Error creating controls: {e}")

    def update(self, time: np.ndarray, intensity: np.ndarray, distance: np.ndarray,
               zeroed_X: np.ndarray, zeroed_Y: np.ndarray, dydt: np.ndarray,
               direction: np.ndarray, velocity: np.ndarray, track_id: int,
               count_3: np.ndarray, count_5: np.ndarray, count_10: np.ndarray,
               count_20: np.ndarray, count_30: np.ndarray, svm: int, length: int) -> None:
        """
        Update all plots with new track data.

        Args:
            time: Time points (frame numbers)
            intensity: Intensity values
            distance: Distance from origin
            zeroed_X: X coordinates relative to origin
            zeroed_Y: Y coordinates relative to origin
            dydt: Change in distance over time
            direction: Direction relative to origin
            velocity: Instantaneous velocity
            track_id: Track identifier
            count_3, count_5, count_10, count_20, count_30: Neighbor counts at different radii
            svm: SVM classification result
            length: Track length
        """
        try:
            self.logger.debug(f"Updating track window for track {track_id}")

            # Validate input data
            if not self._validate_input_data(time, intensity, distance, zeroed_X, zeroed_Y):
                return

            # Update track information labels
            self._update_info_labels(track_id, svm, length)

            # Update plots
            self._update_intensity_plot(time, intensity)
            self._update_distance_plot(time, distance)
            self._update_position_plot(zeroed_X, zeroed_Y)
            self._update_neighbor_plot(time, count_3, count_5, count_10, count_20, count_30)
            self._update_velocity_plot(time, velocity)
            self._update_variance_plot(time, intensity)

            # Update position indicators if enabled
            if self.showPositionIndicators:
                self._setup_position_indicators()

            # Store position data for indicators
            self._store_position_data(time, zeroed_X, zeroed_Y)

            # Update custom column value (add at the very end)
            self._update_column_value(track_id)
            print(f"DEBUG: Updated column value for track {track_id}")  # Debug line

            self.logger.debug("Track window update completed")

        except Exception as e:
            self.logger.error(f"Error updating track window: {e}")

    def _validate_input_data(self, time: np.ndarray, intensity: np.ndarray,
                           distance: np.ndarray, zeroed_X: np.ndarray,
                           zeroed_Y: np.ndarray) -> bool:
        """Validate input data arrays."""
        try:
            arrays = [time, intensity, distance, zeroed_X, zeroed_Y]

            # Check if all arrays exist and have same length
            if not all(arr is not None and len(arr) > 0 for arr in arrays):
                self.logger.warning("Invalid or empty input arrays")
                return False

            lengths = [len(arr) for arr in arrays]
            if not all(length == lengths[0] for length in lengths):
                self.logger.warning("Input arrays have different lengths")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating input data: {e}")
            return False

    def _update_intensity_plot(self, time: np.ndarray, intensity: np.ndarray) -> None:
        """Update intensity trace plot."""
        try:
            # Remove invalid values
            valid_mask = ~(np.isnan(intensity) | np.isinf(intensity))
            if not np.any(valid_mask):
                self.logger.warning("No valid intensity data")
                return

            self.plt1.plot(
                time[valid_mask], intensity[valid_mask],
                pen=pg.mkPen(color=(0, 120, 255), width=2),
                symbol='o',
                symbolSize=3,
                symbolBrush=pg.mkBrush(0, 120, 255, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error updating intensity plot: {e}")

    def _update_distance_plot(self, time: np.ndarray, distance: np.ndarray) -> None:
        """Update distance from origin plot."""
        try:
            # Remove invalid values
            valid_mask = ~(np.isnan(distance) | np.isinf(distance))
            if not np.any(valid_mask):
                self.logger.warning("No valid distance data")
                return

            self.plt2.plot(
                time[valid_mask], distance[valid_mask],
                pen=pg.mkPen(color=(255, 120, 0), width=2),
                symbol='o',
                symbolSize=3,
                symbolBrush=pg.mkBrush(255, 120, 0, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error updating distance plot: {e}")

    def _update_position_plot(self, zeroed_X: np.ndarray, zeroed_Y: np.ndarray) -> None:
        """Update position trace plot."""
        try:
            # Remove invalid values
            valid_mask = ~(np.isnan(zeroed_X) | np.isnan(zeroed_Y) |
                          np.isinf(zeroed_X) | np.isinf(zeroed_Y))
            if not np.any(valid_mask):
                self.logger.warning("No valid position data")
                return

            self.plt3.plot(
                zeroed_X[valid_mask], zeroed_Y[valid_mask],
                pen=pg.mkPen(color=(0, 255, 120), width=2),
                symbol='o',
                symbolSize=4,
                symbolBrush=pg.mkBrush(0, 255, 120, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error updating position plot: {e}")

    def _update_neighbor_plot(self, time: np.ndarray, count_3: np.ndarray,
                            count_5: np.ndarray, count_10: np.ndarray,
                            count_20: np.ndarray, count_30: np.ndarray) -> None:
        """Update nearest neighbor count plot."""
        try:
            # Select data based on current radius setting
            radius = self.plotCountSelector.value()
            count_data = {
                '3': count_3,
                '5': count_5,
                '10': count_10,
                '20': count_20,
                '30': count_30
            }.get(radius, count_5)

            # Remove invalid values
            valid_mask = ~(np.isnan(count_data) | np.isinf(count_data))
            if not np.any(valid_mask):
                self.logger.warning("No valid neighbor count data")
                return

            self.plt4.plot(
                time[valid_mask], count_data[valid_mask],
                pen=pg.mkPen(color=(255, 0, 120), width=2),
                symbol='s',
                symbolSize=4,
                symbolBrush=pg.mkBrush(255, 0, 120, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error updating neighbor plot: {e}")

    def _update_velocity_plot(self, time: np.ndarray, velocity: np.ndarray) -> None:
        """Update instantaneous velocity plot."""
        try:
            # Remove invalid values
            valid_mask = ~(np.isnan(velocity) | np.isinf(velocity))
            if not np.any(valid_mask):
                self.logger.warning("No valid velocity data")
                return

            self.plt5.plot(
                time[valid_mask], velocity[valid_mask],
                pen=pg.mkPen(color=(120, 255, 0), width=2),
                symbol='t',
                symbolSize=4,
                symbolBrush=pg.mkBrush(120, 255, 0, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error updating velocity plot: {e}")

    def _update_variance_plot(self, time: np.ndarray, intensity: np.ndarray) -> None:
        """Update rolling intensity variance plot."""
        try:
            # Calculate rolling variance
            window_size = min(6, len(intensity) // 2) if len(intensity) > 3 else 1

            if window_size > 1:
                rolling_time = rollingFunc(time.tolist(), window_size=window_size, func_type='mean')
                rolling_variance = rollingFunc(intensity.tolist(), window_size=window_size, func_type='variance')

                if rolling_time and rolling_variance:
                    # Remove invalid values
                    rolling_time = np.array(rolling_time)
                    rolling_variance = np.array(rolling_variance)
                    valid_mask = ~(np.isnan(rolling_variance) | np.isinf(rolling_variance))

                    if np.any(valid_mask):
                        self.plt6.plot(
                            rolling_time[valid_mask], rolling_variance[valid_mask],
                            pen=pg.mkPen(color=(255, 120, 255), width=2),
                            symbol='d',
                            symbolSize=3,
                            symbolBrush=pg.mkBrush(255, 120, 255, 180),
                            clear=True
                        )
                    else:
                        self.plt6.clear()
                else:
                    self.plt6.clear()
            else:
                # Not enough data for rolling calculation
                self.plt6.clear()

        except Exception as e:
            self.logger.error(f"Error updating variance plot: {e}")

    def _store_position_data(self, time: np.ndarray, zeroed_X: np.ndarray,
                           zeroed_Y: np.ndarray) -> None:
        """Store position data for indicator display."""
        try:
            # Create time -> position mapping
            self.data = {}
            for t, x, y in zip(time, zeroed_X, zeroed_Y):
                if not (np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y)):
                    self.data[int(t)] = (float(x), float(y))

        except Exception as e:
            self.logger.error(f"Error storing position data: {e}")

    def _on_nn_radius_changed(self) -> None:
        """Handle nearest neighbor radius change."""
        try:
            # This will trigger a replot with the current data
            # The actual replotting will happen when update() is called again
            self.logger.debug(f"NN radius changed to: {self.plotCountSelector.value()}")
        except Exception as e:
            self.logger.error(f"Error handling NN radius change: {e}")

    def togglePositionIndicator(self) -> None:
        """Toggle position indicators on/off."""
        try:
            if not self.showPositionIndicators:
                self._enable_position_indicators()
            else:
                self._disable_position_indicators()

        except Exception as e:
            self.logger.error(f"Error toggling position indicators: {e}")

    def _enable_position_indicators(self) -> None:
        """Enable position indicators."""
        try:
            # Add indicator lines to time-based plots
            self._setup_position_indicators()

            # Connect to time changes
            if hasattr(self.mainGUI, 'plotWindow') and self.mainGUI.plotWindow:
                self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)

            self.showPositionIndicators = True
            self.positionIndicator_button.setText("Hide Position Indicators")

            self.logger.debug("Position indicators enabled")

        except Exception as e:
            self.logger.error(f"Error enabling position indicators: {e}")

    def _disable_position_indicators(self) -> None:
        """Disable position indicators."""
        try:
            # Remove indicator lines
            for line in self.position_indicators:
                for plot in [self.plt1, self.plt2, self.plt4, self.plt5, self.plt6]:
                    try:
                        plot.removeItem(line)
                    except:
                        pass  # Line might not be in this plot

            self.position_indicators.clear()

            # Remove position ROI from position plot
            if self.r is not None:
                try:
                    self.plt3.removeItem(self.r)
                except:
                    pass
                self.r = None

            # Disconnect from time changes
            if hasattr(self.mainGUI, 'plotWindow') and self.mainGUI.plotWindow:
                try:
                    self.mainGUI.plotWindow.sigTimeChanged.disconnect(self.updatePositionIndicators)
                except:
                    pass  # Might not be connected

            self.showPositionIndicators = False
            self.positionIndicator_button.setText("Show Position Indicators")

            self.logger.debug("Position indicators disabled")

        except Exception as e:
            self.logger.error(f"Error disabling position indicators: {e}")

    def _setup_position_indicators(self) -> None:
        """Set up position indicator lines."""
        try:
            # Clear existing indicators
            for line in self.position_indicators:
                for plot in [self.plt1, self.plt2, self.plt4, self.plt5, self.plt6]:
                    try:
                        plot.removeItem(line)
                    except:
                        pass
            self.position_indicators.clear()

            # Create new indicator lines for each time-based plot
            plots_with_indicators = [self.plt1, self.plt2, self.plt4, self.plt5, self.plt6]

            for plot in plots_with_indicators:
                line = pg.InfiniteLine(
                    pos=0,
                    angle=90,
                    pen=pg.mkPen('y', style=Qt.DashLine, width=2),
                    movable=False
                )
                plot.addItem(line)
                self.position_indicators.append(line)

        except Exception as e:
            self.logger.error(f"Error setting up position indicators: {e}")

    def updatePositionIndicators(self, frame_time: int) -> None:
        """
        Update position indicators based on current time.

        Args:
            frame_time: Current frame/time index
        """
        try:
            if not self.showPositionIndicators:
                return

            # Update time indicator lines
            for line in self.position_indicators:
                line.setPos(frame_time)

            # Update position ROI
            if self.r is not None:
                try:
                    self.plt3.removeItem(self.r)
                except:
                    pass
                self.r = None

            # Add new position ROI if data exists for this time
            if frame_time in self.data:
                x, y = self.data[frame_time]
                self.r = pg.RectROI(
                    (x - 0.25, y - 0.25),
                    size=pg.Point(0.5, 0.5),
                    movable=False,
                    rotatable=False,
                    resizable=False,
                    pen=pg.mkPen('r', width=2)
                )
                self.r.handlePen = None
                self.plt3.addItem(self.r)

        except Exception as e:
            self.logger.debug(f"Error updating position indicators: {e}")

    def export_plots(self, base_filename: str) -> None:
        """
        Export all plots to image files.

        Args:
            base_filename: Base filename for exported plots
        """
        try:
            import os

            plot_names = {
                'intensity': self.plt1,
                'distance': self.plt2,
                'position': self.plt3,
                'neighbors': self.plt4,
                'velocity': self.plt5,
                'variance': self.plt6
            }

            for name, plot in plot_names.items():
                try:
                    filename = f"{base_filename}_{name}.png"
                    exporter = pg.exporters.ImageExporter(plot.plotItem)
                    exporter.export(filename)
                    self.logger.info(f"Exported {name} plot to {filename}")
                except Exception as e:
                    self.logger.error(f"Error exporting {name} plot: {e}")

        except Exception as e:
            self.logger.error(f"Error exporting plots: {e}")

    def get_plot_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get current plot data for external analysis.

        Returns:
            Dictionary containing plot data arrays
        """
        try:
            data = {}

            # Extract data from each plot
            for name, plot in self.plots.items():
                try:
                    items = plot.listDataItems()
                    if items:
                        item = items[0]  # Get first data item
                        x_data, y_data = item.getData()
                        data[name] = {
                            'x': np.array(x_data) if x_data is not None else np.array([]),
                            'y': np.array(y_data) if y_data is not None else np.array([])
                        }
                    else:
                        data[name] = {'x': np.array([]), 'y': np.array([])}
                except Exception as e:
                    self.logger.error(f"Error extracting data from {name} plot: {e}")
                    data[name] = {'x': np.array([]), 'y': np.array([])}

            return data

        except Exception as e:
            self.logger.error(f"Error getting plot data: {e}")
            return {}

    def show(self) -> None:
        """Show the track window."""
        try:
            self.win.show()
            self.logger.debug("Track window shown")
        except Exception as e:
            self.logger.error(f"Error showing track window: {e}")

    def close(self) -> None:
        """Close the track window."""
        try:
            # Disable position indicators before closing
            if self.showPositionIndicators:
                self._disable_position_indicators()

            self.win.close()
            self.logger.debug("Track window closed")
        except Exception as e:
            self.logger.error(f"Error closing track window: {e}")

    def hide(self) -> None:
        """Hide the track window."""
        try:
            self.win.hide()
            self.logger.debug("Track window hidden")
        except Exception as e:
            self.logger.error(f"Error hiding track window: {e}")
