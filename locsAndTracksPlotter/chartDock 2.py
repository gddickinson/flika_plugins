#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chart Dock Module for FLIKA Tracking Plugin

This module provides a dockable window for creating various types of plots
and histograms from tracking data. It supports scatter plots, line plots,
and histograms with customizable parameters.

Created on Fri Jun  2 16:28:47 2023
@author: george
"""

import logging
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QLabel, QPushButton
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

# Plugin imports
from .helperFunctions import dictFromList

# PyQtGraph dock imports
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# Set up logging
logger = logging.getLogger(__name__)


class ChartDock:
    """
    A dockable window for displaying analysis charts and histograms.

    This class provides an interface for creating various types of plots
    from tracking data, including scatter plots, line plots, and histograms
    with extensive customization options.

    Attributes:
        mainGUI: Reference to the main GUI instance
        win: Main window containing the dock area
        area: Dock area for organizing plot widgets
        plot_widget: Main plot widget for scatter/line plots
        histogram_widget: Widget for histogram plots
    """

    def __init__(self, mainGUI):
        """
        Initialize the chart dock window.

        Args:
            mainGUI: Reference to main GUI instance containing data
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Initialize data references
        self.xcols: Dict[str, str] = {'None': 'None'}
        self.ycols: Dict[str, str] = {'None': 'None'}
        self.cols: Dict[str, str] = {'None': 'None'}

        self._setup_ui()
        self.logger.debug("ChartDock initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window and dock area
            self.win = QMainWindow()
            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.resize(1000, 500)
            self.win.setWindowTitle('Charts and Analysis')

            # Create docks
            self._create_docks()

            # Create scatter plot controls
            self._create_scatter_plot_controls()

            # Create histogram controls
            self._create_histogram_controls()

            # Create plot widgets
            self._create_plot_widgets()

            self.logger.debug("ChartDock UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up ChartDock UI: {e}")

    def _create_docks(self) -> None:
        """Create dock widgets for different sections."""
        # Plot options dock
        self.d1 = Dock("Plot Options", size=(500, 100))

        # Main plot dock
        self.d2 = Dock("Main Plot", size=(500, 400))

        # Histogram options dock
        self.d3 = Dock("Histogram Options", size=(500, 100))

        # Histogram plot dock
        self.d4 = Dock("Histogram", size=(500, 400))

        # Add docks to area
        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d3, 'right', self.d1)
        self.area.addDock(self.d2, 'bottom', self.d1)
        self.area.addDock(self.d4, 'bottom', self.d3)

    def _create_scatter_plot_controls(self) -> None:
        """Create controls for scatter/line plots."""
        # Create layout widget
        self.w1 = pg.LayoutWidget()

        # Data type selector
        self.pointOrTrackData_selector_plot = pg.ComboBox()
        self.plotDataChoice = {
            'Point Data': 'Point Data',
            'Track Means': 'Track Means'
        }
        self.pointOrTrackData_selector_plot.setItems(self.plotDataChoice)

        # Axis selectors
        self.xlabel = QLabel("X-axis:")
        self.ylabel = QLabel("Y-axis:")

        self.xColSelector = pg.ComboBox()
        self.xColSelector.setItems(self.xcols)

        self.yColSelector = pg.ComboBox()
        self.yColSelector.setItems(self.ycols)

        # Plot type selector
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes = {'scatter': 'scatter', 'line': 'line'}
        self.plotTypeSelector.setItems(self.plotTypes)
        self.selectorLabel = QLabel("Plot type")

        # Point size control
        self.pointSize_selector = pg.SpinBox(value=7, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(20)
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size")

        # Plot button
        self.plot_button = QPushButton('Update Plot')
        self.plot_button.pressed.connect(self.updatePlot)

        # Layout widgets
        self._layout_scatter_widgets()

        # Add to dock
        self.d1.addWidget(self.w1)

    def _layout_scatter_widgets(self) -> None:
        """Arrange scatter plot widgets in layout."""
        # Row 0: Data type
        self.w1.addWidget(QLabel("Data type:"), row=0, col=0)
        self.w1.addWidget(self.pointOrTrackData_selector_plot, row=0, col=1)

        # Row 1: X-axis
        self.w1.addWidget(self.xlabel, row=1, col=0)
        self.w1.addWidget(self.xColSelector, row=1, col=1)

        # Row 2: Y-axis
        self.w1.addWidget(self.ylabel, row=2, col=0)
        self.w1.addWidget(self.yColSelector, row=2, col=1)

        # Row 3: Plot type
        self.w1.addWidget(self.selectorLabel, row=3, col=0)
        self.w1.addWidget(self.plotTypeSelector, row=3, col=1)

        # Row 4: Point size
        self.w1.addWidget(self.pointSizeLabel, row=4, col=0)
        self.w1.addWidget(self.pointSize_selector, row=4, col=1)

        # Row 5: Plot button
        self.w1.addWidget(self.plot_button, row=5, col=1)

    def _create_histogram_controls(self) -> None:
        """Create controls for histogram plots."""
        # Create layout widget
        self.w2 = pg.LayoutWidget()

        # Data type selector for histogram
        self.pointOrTrackData_selector_histo = pg.ComboBox()
        self.histoDataChoice = {
            'Point Data': 'Point Data',
            'Track Means': 'Track Means'
        }
        self.pointOrTrackData_selector_histo.setItems(self.histoDataChoice)

        # Column selector
        self.colSelector = pg.ComboBox()
        self.colSelector.setItems(self.cols)
        self.collabel = QLabel("Column:")

        # Histogram controls
        self.histoBin_selector = pg.SpinBox(value=50, int=True)
        self.histoBin_selector.setSingleStep(1)
        self.histoBin_selector.setMinimum(5)
        self.histoBin_selector.setMaximum(1000)
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        self.histoBin_label = QLabel('Number of bins')

        # Cumulative histogram option
        self.cumulativeTick_label = QLabel('Cumulative')
        self.cumulativeTick = CheckBox()
        self.cumulativeTick.setChecked(False)
        self.cumulativeTick.stateChanged.connect(self.updateHisto)

        # Histogram button
        self.histo_button = QPushButton('Update Histogram')
        self.histo_button.pressed.connect(self.updateHisto)

        # Layout histogram widgets
        self._layout_histogram_widgets()

        # Add to dock
        self.d3.addWidget(self.w2)

    def _layout_histogram_widgets(self) -> None:
        """Arrange histogram widgets in layout."""
        # Row 0: Data type
        self.w2.addWidget(QLabel("Data type:"), row=0, col=0)
        self.w2.addWidget(self.pointOrTrackData_selector_histo, row=0, col=1)

        # Row 1: Column
        self.w2.addWidget(self.collabel, row=1, col=0)
        self.w2.addWidget(self.colSelector, row=1, col=1)

        # Row 2: Bins
        self.w2.addWidget(self.histoBin_label, row=2, col=0)
        self.w2.addWidget(self.histoBin_selector, row=2, col=1)

        # Row 3: Cumulative
        self.w2.addWidget(self.cumulativeTick_label, row=3, col=0)
        self.w2.addWidget(self.cumulativeTick, row=3, col=1)

        # Row 4: Update button
        self.w2.addWidget(self.histo_button, row=4, col=1)

    def _create_plot_widgets(self) -> None:
        """Create plot widgets for main plot and histogram."""
        # Main plot widget
        self.w3 = pg.PlotWidget(title="Data Plot")
        self.w3.setLabel('left', 'Y-axis', units='')
        self.w3.setLabel('bottom', 'X-axis', units='')
        self.w3.showGrid(x=True, y=True, alpha=0.3)
        self.d2.addWidget(self.w3)

        # Histogram plot widget
        self.w4 = pg.PlotWidget(title="Data Histogram")
        self.w4.setLabel('left', 'Count', units='')
        self.w4.setLabel('bottom', 'Value', units='')
        self.w4.showGrid(x=True, y=True, alpha=0.3)
        self.d4.addWidget(self.w4)

    def updatePlot(self) -> None:
        """Update the main scatter/line plot based on current settings."""
        try:
            self.logger.debug("Updating main plot")

            # Clear current plot
            self.w3.clear()

            # Get data
            x_data, y_data = self._get_plot_data()
            if x_data is None or y_data is None:
                self.logger.warning("No valid data for plotting")
                return

            # Validate data
            if len(x_data) == 0 or len(y_data) == 0:
                self.logger.warning("Empty data arrays")
                return

            # Remove invalid values
            valid_mask = ~(np.isnan(x_data) | np.isnan(y_data) |
                          np.isinf(x_data) | np.isinf(y_data))
            x_clean = x_data[valid_mask]
            y_clean = y_data[valid_mask]

            if len(x_clean) == 0:
                self.logger.warning("No valid data points after filtering")
                return

            # Create plot based on type
            plot_type = self.plotTypeSelector.value()

            if plot_type == 'line':
                self._create_line_plot(x_clean, y_clean)
            elif plot_type == 'scatter':
                self._create_scatter_plot(x_clean, y_clean)

            # Update axis labels
            x_col = self.xColSelector.value()
            y_col = self.yColSelector.value()
            self.w3.setLabel('left', y_col, units=None)
            self.w3.setLabel('bottom', x_col, units=None)

            self.logger.debug(f"Plot updated with {len(x_clean)} points")

        except Exception as e:
            self.logger.error(f"Error updating plot: {e}")

    def _get_plot_data(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get data arrays for plotting."""
        try:
            # Check data availability
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return None, None

            x_col = self.xColSelector.value()
            y_col = self.yColSelector.value()

            if x_col == 'None' or y_col == 'None':
                return None, None

            # Determine data source
            data_type = self.pointOrTrackData_selector_plot.value()

            if data_type == 'Point Data':
                # Use raw point data
                if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                    source_data = self.mainGUI.filteredData
                else:
                    source_data = self.mainGUI.data

                if x_col not in source_data.columns or y_col not in source_data.columns:
                    return None, None

                x_data = source_data[x_col].to_numpy()
                y_data = source_data[y_col].to_numpy()

            else:
                # Use track means
                if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                    source_data = self.mainGUI.filteredData
                else:
                    source_data = self.mainGUI.data

                if 'track_number' not in source_data.columns:
                    return None, None

                # Group by track and calculate means
                plot_df = source_data.groupby('track_number', as_index=False).mean()

                if x_col not in plot_df.columns or y_col not in plot_df.columns:
                    return None, None

                x_data = plot_df[x_col].to_numpy()
                y_data = plot_df[y_col].to_numpy()

            return x_data, y_data

        except Exception as e:
            self.logger.error(f"Error getting plot data: {e}")
            return None, None

    def _create_line_plot(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Create a line plot."""
        try:
            # Sort data by x values for proper line plotting
            sort_indices = np.argsort(x_data)
            x_sorted = x_data[sort_indices]
            y_sorted = y_data[sort_indices]

            self.w3.plot(
                x_sorted, y_sorted,
                pen=pg.mkPen(color=(0, 120, 255), width=2),
                symbol='o',
                symbolSize=4,
                symbolBrush=pg.mkBrush(0, 120, 255, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error creating line plot: {e}")

    def _create_scatter_plot(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """Create a scatter plot."""
        try:
            point_size = self.pointSize_selector.value()

            self.w3.plot(
                x_data, y_data,
                pen=None,
                symbol='o',
                symbolPen=pg.mkPen(color=(0, 120, 255), width=1),
                symbolBrush=pg.mkBrush(0, 120, 255, 180),
                symbolSize=point_size,
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")

    def updateHisto(self) -> None:
        """Update the histogram plot based on current settings."""
        try:
            self.logger.debug("Updating histogram")

            # Clear current histogram
            self.w4.clear()

            # Get data
            values = self._get_histogram_data()
            if values is None or len(values) == 0:
                self.logger.warning("No valid data for histogram")
                return

            # Remove invalid values
            valid_values = values[~(np.isnan(values) | np.isinf(values))]

            if len(valid_values) == 0:
                self.logger.warning("No valid values for histogram")
                return

            # Get histogram parameters
            n_bins = self.histoBin_selector.value()
            is_cumulative = self.cumulativeTick.isChecked()

            # Calculate histogram range
            value_min, value_max = np.min(valid_values), np.max(valid_values)

            if value_min == value_max:
                # Handle case where all values are the same
                bins = np.array([value_min - 0.5, value_min + 0.5])
                n_bins = 1
            else:
                bins = np.linspace(value_min, value_max, n_bins + 1)

            if is_cumulative:
                self._create_cumulative_histogram(valid_values, bins)
            else:
                self._create_regular_histogram(valid_values, bins)

            # Update axis label
            col_name = self.colSelector.value()
            self.w4.setLabel('bottom', col_name, units=None)

            y_label = 'Cumulative Probability' if is_cumulative else 'Count'
            self.w4.setLabel('left', y_label, units=None)

            self.logger.debug(f"Histogram updated with {len(valid_values)} values")

        except Exception as e:
            self.logger.error(f"Error updating histogram: {e}")

    def _get_histogram_data(self) -> Optional[np.ndarray]:
        """Get data for histogram plotting."""
        try:
            # Check data availability
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return None

            col_name = self.colSelector.value()
            if col_name == 'None':
                return None

            # Determine data source
            data_type = self.pointOrTrackData_selector_histo.value()

            if data_type == 'Point Data':
                # Use raw point data
                if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                    source_data = self.mainGUI.filteredData
                else:
                    source_data = self.mainGUI.data

                if col_name not in source_data.columns:
                    return None

                values = source_data[col_name].to_numpy()

            else:
                # Use track means
                if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                    source_data = self.mainGUI.filteredData
                else:
                    source_data = self.mainGUI.data

                if 'track_number' not in source_data.columns:
                    return None

                # Group by track and calculate means
                plot_df = source_data.groupby('track_number', as_index=False).mean()

                if col_name not in plot_df.columns:
                    return None

                values = plot_df[col_name].to_numpy()

            return values

        except Exception as e:
            self.logger.error(f"Error getting histogram data: {e}")
            return None

    def _create_regular_histogram(self, values: np.ndarray, bins: np.ndarray) -> None:
        """Create a regular histogram."""
        try:
            y, x = np.histogram(values, bins=bins)

            # Create step plot
            self.w4.plot(
                x, y,
                stepMode=True,
                fillLevel=0,
                brush=pg.mkBrush(0, 120, 255, 120),
                pen=pg.mkPen(0, 120, 255, 200),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error creating regular histogram: {e}")

    def _create_cumulative_histogram(self, values: np.ndarray, bins: np.ndarray) -> None:
        """Create a cumulative histogram."""
        try:
            # Calculate histogram
            counts, bin_edges = np.histogram(values, bins=bins)

            # Calculate cumulative distribution
            pdf = counts / np.sum(counts)
            cdf = np.cumsum(pdf)
            x = bin_edges[1:]  # Use right edges of bins

            # Plot cumulative distribution
            self.w4.plot(
                x, cdf,
                pen=pg.mkPen(color=(0, 120, 255), width=2),
                symbol='o',
                symbolSize=4,
                symbolBrush=pg.mkBrush(0, 120, 255, 180),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error creating cumulative histogram: {e}")

    def update_column_lists(self, columns: Dict[str, str]) -> None:
        """
        Update the available columns for plotting.

        Args:
            columns: Dictionary of column names
        """
        try:
            self.xcols = columns.copy()
            self.ycols = columns.copy()
            self.cols = columns.copy()

            # Update comboboxes
            self.xColSelector.setItems(self.xcols)
            self.yColSelector.setItems(self.ycols)
            self.colSelector.setItems(self.cols)

            self.logger.debug(f"Column lists updated with {len(columns)} columns")

        except Exception as e:
            self.logger.error(f"Error updating column lists: {e}")

    def export_plot(self, plot_type: str = 'main') -> None:
        """
        Export plot to image file.

        Args:
            plot_type: Type of plot to export ('main' or 'histogram')
        """
        try:
            from qtpy.QtWidgets import QFileDialog

            # Select plot widget
            if plot_type == 'histogram':
                plot_widget = self.w4
                default_name = "histogram.png"
            else:
                plot_widget = self.w3
                default_name = "plot.png"

            # Get save path
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                f"Export {plot_type} plot",
                default_name,
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )

            if not file_path:
                return

            # Export plot
            exporter = pg.exporters.ImageExporter(plot_widget.plotItem)
            exporter.export(file_path)

            self.logger.info(f"Plot exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting plot: {e}")

    def reset_zoom(self, plot_type: str = 'main') -> None:
        """
        Reset zoom on specified plot.

        Args:
            plot_type: Type of plot to reset ('main' or 'histogram')
        """
        try:
            if plot_type == 'histogram':
                self.w4.autoRange()
            else:
                self.w3.autoRange()

            self.logger.debug(f"{plot_type} plot zoom reset")

        except Exception as e:
            self.logger.error(f"Error resetting zoom: {e}")

    def show(self) -> None:
        """Show the chart dock window."""
        try:
            self.win.show()
            self.logger.debug("ChartDock window shown")
        except Exception as e:
            self.logger.error(f"Error showing ChartDock window: {e}")

    def close(self) -> None:
        """Close the chart dock window."""
        try:
            self.win.close()
            self.logger.debug("ChartDock window closed")
        except Exception as e:
            self.logger.error(f"Error closing ChartDock window: {e}")

    def hide(self) -> None:
        """Hide the chart dock window."""
        try:
            self.win.hide()
            self.logger.debug("ChartDock window hidden")
        except Exception as e:
            self.logger.error(f"Error hiding ChartDock window: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for currently displayed data.

        Returns:
            Dictionary containing statistics
        """
        try:
            stats = {}

            # Get plot data statistics
            x_data, y_data = self._get_plot_data()
            if x_data is not None and y_data is not None:
                # Remove invalid values
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data) |
                              np.isinf(x_data) | np.isinf(y_data))
                x_clean = x_data[valid_mask]
                y_clean = y_data[valid_mask]

                if len(x_clean) > 0:
                    stats['plot'] = {
                        'n_points': len(x_clean),
                        'x_mean': np.mean(x_clean),
                        'x_std': np.std(x_clean),
                        'x_min': np.min(x_clean),
                        'x_max': np.max(x_clean),
                        'y_mean': np.mean(y_clean),
                        'y_std': np.std(y_clean),
                        'y_min': np.min(y_clean),
                        'y_max': np.max(y_clean),
                        'correlation': np.corrcoef(x_clean, y_clean)[0, 1] if len(x_clean) > 1 else 0
                    }

            # Get histogram data statistics
            hist_data = self._get_histogram_data()
            if hist_data is not None:
                valid_hist = hist_data[~(np.isnan(hist_data) | np.isinf(hist_data))]
                if len(valid_hist) > 0:
                    stats['histogram'] = {
                        'n_values': len(valid_hist),
                        'mean': np.mean(valid_hist),
                        'std': np.std(valid_hist),
                        'min': np.min(valid_hist),
                        'max': np.max(valid_hist),
                        'median': np.median(valid_hist),
                        'q25': np.percentile(valid_hist, 25),
                        'q75': np.percentile(valid_hist, 75)
                    }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
