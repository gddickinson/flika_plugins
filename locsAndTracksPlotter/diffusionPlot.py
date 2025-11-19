#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diffusion Analysis Plot Module for FLIKA Tracking Plugin

This module provides comprehensive tools for analyzing diffusion properties
of tracked particles, including mean squared displacement analysis, histogram
distributions, and cumulative distribution function fitting with exponential
decay models.

Key Features:
- Scatter and line plots of squared displacement vs lag time
- Histogram analysis of step length distributions
- Cumulative distribution function (CDF) analysis
- Single, double, and triple exponential decay fitting
- Interactive parameter adjustment
- Export capabilities for analysis results

Created on Fri Jun  2 15:28:30 2023
@author: george
"""

import logging
from typing import Optional, Dict, Any, Tuple, List, Union, Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow, QLabel, QPushButton
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g

# PyQtGraph dock imports
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# Plugin imports
from .helperFunctions import exp_dec, exp_dec_2, exp_dec_3

# Set up logging
logger = logging.getLogger(__name__)

# Suppress optimization warnings for cleaner output
warnings.filterwarnings("ignore", category=OptimizeWarning)


class DiffusionPlotWindow:
    """
    Comprehensive diffusion analysis window for single-particle tracking data.

    This class provides tools for analyzing diffusion properties including:
    - Mean squared displacement (MSD) analysis
    - Step length distribution (SLD) analysis
    - Cumulative distribution function (CDF) fitting
    - Multi-component exponential decay analysis
    - Interactive parameter selection and fitting

    The analysis helps characterize different diffusion modes:
    - Free diffusion (single exponential)
    - Confined diffusion (multiple exponentials)
    - Directed motion analysis

    Attributes:
        mainGUI: Reference to main GUI instance
        win: Main window containing dock area
        area: Dock area for organizing widgets
        plots: Dictionary of plot widgets
        fit_curves: Dictionary of fitted curve items
        fitting_data: Current data used for fitting
    """

    def __init__(self, mainGUI):
        """
        Initialize the diffusion analysis window.

        Args:
            mainGUI: Reference to main GUI instance containing tracking data
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Analysis state
        self.fitting_data: Dict[str, np.ndarray] = {}
        self.fit_curves: Dict[str, Optional[pg.PlotDataItem]] = {
            'exp1': None,
            'exp2': None,
            'exp3': None
        }
        self.fit_parameters: Dict[str, Dict[str, float]] = {}

        # UI components
        self.plots: Dict[str, pg.PlotWidget] = {}
        self.controls: Dict[str, Any] = {}

        self._setup_ui()
        self.logger.debug("DiffusionPlotWindow initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main window and dock area
            self.win = QMainWindow()
            self.area = DockArea()
            self.win.setCentralWidget(self.area)
            self.win.resize(1400, 600)
            self.win.setWindowTitle('Diffusion Analysis')

            # Create docks
            self._create_docks()

            # Create scatter plot controls and widget
            self._create_scatter_plot_section()

            # Create histogram controls and widget
            self._create_histogram_section()

            # Create CDF controls and widget
            self._create_cdf_section()

            self.logger.debug("DiffusionPlotWindow UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up DiffusionPlotWindow UI: {e}")

    def _create_docks(self) -> None:
        """Create dock widgets for different sections."""
        # Scatter plot section
        self.d1 = Dock("MSD Plot Options", size=(400, 120))
        self.d2 = Dock("Mean Squared Displacement", size=(400, 400))

        # Histogram section
        self.d3 = Dock("Histogram Options", size=(400, 120))
        self.d4 = Dock("Step Length Distribution", size=(400, 400))

        # CDF section
        self.d5 = Dock("CDF Analysis Options", size=(400, 120))
        self.d6 = Dock("Cumulative Distribution Function", size=(400, 400))

        # Arrange docks
        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d3, 'right', self.d1)
        self.area.addDock(self.d5, 'right', self.d3)

        self.area.addDock(self.d2, 'bottom', self.d1)
        self.area.addDock(self.d4, 'bottom', self.d3)
        self.area.addDock(self.d6, 'bottom', self.d5)

    def _create_scatter_plot_section(self) -> None:
        """Create scatter plot controls and widget."""
        # Controls layout
        self.w1 = pg.LayoutWidget()

        # Plot type selector
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes = {
            'scatter': 'scatter',
            'line (slow with many tracks!)': 'line'
        }
        self.plotTypeSelector.setItems(self.plotTypes)
        self.selectorLabel = QLabel("Plot type")

        # Point size control
        self.pointSize_selector = pg.SpinBox(value=3, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10)
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size")

        # Update button
        self.plot_button = QPushButton('Update Plot')
        self.plot_button.pressed.connect(self.updatePlot)

        # Layout controls
        self.w1.addWidget(self.selectorLabel, row=0, col=0)
        self.w1.addWidget(self.plotTypeSelector, row=0, col=1)
        self.w1.addWidget(self.pointSizeLabel, row=1, col=0)
        self.w1.addWidget(self.pointSize_selector, row=1, col=1)
        self.w1.addWidget(self.plot_button, row=2, col=0, colspan=2)

        self.d1.addWidget(self.w1)

        # Plot widget
        self.w3 = pg.PlotWidget(title="Mean Squared Displacement Analysis")
        self.w3.setLabel('left', 'Squared Displacement', units='μm²')
        self.w3.setLabel('bottom', 'Lag Time', units='frames')
        self.w3.showGrid(x=True, y=True, alpha=0.3)
        self.w3.setLogMode(x=False, y=False)
        self.plots['msd'] = self.w3
        self.d2.addWidget(self.w3)

    def _create_histogram_section(self) -> None:
        """Create histogram controls and widget."""
        # Controls layout
        self.w2 = pg.LayoutWidget()

        # Bins control
        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(10)
        self.histoBin_selector.setMinimum(10)
        self.histoBin_selector.setMaximum(1000)
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        self.histoBin_label = QLabel('Number of bins')

        # Update button
        self.histo_button = QPushButton('Update Histogram')
        self.histo_button.pressed.connect(self.updateHisto)

        # Statistics display
        self.stats_label = QLabel('Statistics will appear here')
        self.stats_label.setWordWrap(True)
        self.stats_label.setMaximumHeight(60)

        # Layout controls
        self.w2.addWidget(self.histoBin_label, row=0, col=0)
        self.w2.addWidget(self.histoBin_selector, row=0, col=1)
        self.w2.addWidget(self.histo_button, row=1, col=0, colspan=2)
        self.w2.addWidget(self.stats_label, row=2, col=0, colspan=2)

        self.d3.addWidget(self.w2)

        # Plot widget
        self.w4 = pg.PlotWidget(title="Step Length Distribution")
        self.w4.setLabel('left', 'Count', units='')
        self.w4.setLabel('bottom', 'Step Length', units='μm')
        self.w4.showGrid(x=True, y=True, alpha=0.3)
        self.w4.getAxis('bottom').enableAutoSIPrefix(False)
        self.plots['histogram'] = self.w4
        self.d4.addWidget(self.w4)

    def _create_cdf_section(self) -> None:
        """Create CDF analysis controls and widget."""
        # Controls layout
        self.w5 = pg.LayoutWidget()

        # Bins control for CDF
        self.cdfBin_selector = pg.SpinBox(value=100, int=True)
        self.cdfBin_selector.setSingleStep(10)
        self.cdfBin_selector.setMinimum(10)
        self.cdfBin_selector.setMaximum(1000)
        self.cdfBin_selector.sigValueChanged.connect(self.updateCDF)
        self.cdfBin_label = QLabel('Number of bins')

        # CDF update button
        self.cdf_button = QPushButton('Update CDF')
        self.cdf_button.pressed.connect(self.updateCDF)

        # Fitting buttons
        self.fit_exp_dec_1_button = QPushButton('Fit 1-Component')
        self.fit_exp_dec_1_button.pressed.connect(self.fit_exp_dec_1)

        self.fit_exp_dec_2_button = QPushButton('Fit 2-Component')
        self.fit_exp_dec_2_button.pressed.connect(self.fit_exp_dec_2)

        self.fit_exp_dec_3_button = QPushButton('Fit 3-Component')
        self.fit_exp_dec_3_button.pressed.connect(self.fit_exp_dec_3)

        # Clear fits button
        self.clear_fits_button = QPushButton('Clear Fits')
        self.clear_fits_button.pressed.connect(self.clear_fits)

        # Export button
        self.export_button = QPushButton('Export Data')
        self.export_button.pressed.connect(self.export_analysis_data)

        # Layout controls
        self.w5.addWidget(self.cdfBin_label, row=0, col=0)
        self.w5.addWidget(self.cdfBin_selector, row=0, col=1)
        self.w5.addWidget(self.cdf_button, row=1, col=0, colspan=2)
        self.w5.addWidget(self.fit_exp_dec_1_button, row=2, col=0, colspan=2)
        self.w5.addWidget(self.fit_exp_dec_2_button, row=3, col=0, colspan=2)
        self.w5.addWidget(self.fit_exp_dec_3_button, row=4, col=0, colspan=2)
        self.w5.addWidget(self.clear_fits_button, row=5, col=0)
        self.w5.addWidget(self.export_button, row=5, col=1)

        self.d5.addWidget(self.w5)

        # Plot widget with legend
        self.w6 = pg.PlotWidget(title="Cumulative Distribution Function Analysis")
        self.w6.setLabel('left', 'Cumulative Probability', units='')
        self.w6.setLabel('bottom', 'Squared Step Length', units='μm²')
        self.w6.showGrid(x=True, y=True, alpha=0.3)
        self.w6.getAxis('bottom').enableAutoSIPrefix(False)
        self.plots['cdf'] = self.w6
        self.d6.addWidget(self.w6)

        # Add legend to CDF plot
        self.cdf_legend = self.w6.plotItem.addLegend()

    def updatePlot(self) -> None:
        """Update the mean squared displacement plot."""
        try:
            self.logger.debug("Updating MSD plot")

            # Clear previous plot
            self.w3.clear()

            # Get data
            if not self._validate_data_availability():
                return

            # Determine data source
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                df = self.mainGUI.filteredData
            else:
                df = self.mainGUI.data

            # Check required columns
            required_cols = ['lagNumber', 'd_squared']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Missing required columns: {required_cols}")
                return

            plot_type = self.plotTypeSelector.value()

            if plot_type == 'line':
                self._create_line_plot(df)
            elif plot_type == 'scatter':
                self._create_scatter_plot(df)

            self.logger.debug("MSD plot updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating MSD plot: {e}")

    def _validate_data_availability(self) -> bool:
        """Validate that data is available for analysis."""
        if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
            self.logger.warning("No data available for analysis")
            return False
        return True

    def _create_line_plot(self, df: pd.DataFrame) -> None:
        """Create line plot for MSD analysis."""
        try:
            # Group by track and plot each track separately
            track_groups = df.groupby('track_number')

            for track_id, track_data in track_groups:
                if len(track_data) < 2:
                    continue

                x = track_data['lagNumber'].to_numpy()
                y = track_data['d_squared'].to_numpy()

                # Remove invalid values
                valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
                if not np.any(valid_mask):
                    continue

                x_clean = x[valid_mask]
                y_clean = y[valid_mask]

                if len(x_clean) < 2:
                    continue

                # Create path for this track
                color = pg.intColor(int(track_id))
                self.w3.plot(
                    x_clean, y_clean,
                    pen=pg.mkPen(color, width=1, alpha=150),
                    connect='all'
                )

        except Exception as e:
            self.logger.error(f"Error creating line plot: {e}")

    def _create_scatter_plot(self, df: pd.DataFrame) -> None:
        """Create scatter plot for MSD analysis."""
        try:
            x = df['lagNumber'].to_numpy()
            y = df['d_squared'].to_numpy()

            # Remove invalid values
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if not np.any(valid_mask):
                self.logger.warning("No valid data for scatter plot")
                return

            x_clean = x[valid_mask]
            y_clean = y[valid_mask]

            point_size = self.pointSize_selector.value()

            self.w3.plot(
                x_clean, y_clean,
                pen=None,
                symbol='o',
                symbolPen=pg.mkPen(color=(0, 120, 255), width=0),
                symbolBrush=pg.mkBrush(0, 120, 255, 120),
                symbolSize=point_size
            )

        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {e}")

    def updateHisto(self) -> None:
        """Update the step length distribution histogram."""
        try:
            self.logger.debug("Updating histogram")

            # Clear previous histogram
            self.w4.clear()

            if not self._validate_data_availability():
                return

            # Get data and calculate step lengths
            step_lengths = self._calculate_step_lengths()
            if step_lengths is None or len(step_lengths) == 0:
                return

            # Create histogram
            n_bins = self.histoBin_selector.value()
            self._create_histogram(step_lengths, n_bins)

            # Update statistics
            self._update_histogram_statistics(step_lengths)

            self.logger.debug("Histogram updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating histogram: {e}")

    def _calculate_step_lengths(self) -> Optional[np.ndarray]:
        """Calculate step lengths in micrometers."""
        try:
            # Determine data source
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                df = self.mainGUI.filteredData
            else:
                df = self.mainGUI.data

            # Group by track and calculate mean velocity (equivalent to step length)
            if 'track_number' not in df.columns or 'velocity' not in df.columns:
                self.logger.warning("Missing required columns for step length calculation")
                return None

            track_means = df.groupby('track_number', as_index=False).mean()

            if 'velocity' not in track_means.columns:
                return None

            # Convert to micrometers
            pixel_size_nm = self.mainGUI.trackPlotOptions.pixelSize_selector.value()
            step_lengths_um = track_means['velocity'] * pixel_size_nm / 1000  # nm to μm

            # Remove invalid values
            valid_mask = ~(np.isnan(step_lengths_um) | np.isinf(step_lengths_um))
            return step_lengths_um[valid_mask].to_numpy()

        except Exception as e:
            self.logger.error(f"Error calculating step lengths: {e}")
            return None

    def _create_histogram(self, data: np.ndarray, n_bins: int) -> None:
        """Create histogram plot."""
        try:
            if len(data) == 0:
                return

            # Calculate histogram
            data_min, data_max = np.min(data), np.max(data)

            if data_min == data_max:
                # Handle case where all values are the same
                bins = np.array([data_min - 0.5, data_min + 0.5])
                y = np.array([len(data)])
                x = np.array([data_min])
            else:
                bins = np.linspace(data_min, data_max, n_bins + 1)
                y, x = np.histogram(data, bins=bins)

            # Plot histogram
            self.w4.plot(
                x, y,
                stepMode=True,
                fillLevel=0,
                brush=pg.mkBrush(0, 120, 255, 120),
                pen=pg.mkPen(0, 120, 255, 200),
                clear=True
            )

        except Exception as e:
            self.logger.error(f"Error creating histogram: {e}")

    def _update_histogram_statistics(self, data: np.ndarray) -> None:
        """Update histogram statistics display."""
        try:
            if len(data) == 0:
                self.stats_label.setText("No data available")
                return

            stats_text = (
                f"N = {len(data)}, "
                f"Mean = {np.mean(data):.3f} μm, "
                f"Std = {np.std(data):.3f} μm, "
                f"Median = {np.median(data):.3f} μm"
            )
            self.stats_label.setText(stats_text)

        except Exception as e:
            self.logger.error(f"Error updating histogram statistics: {e}")

    def updateCDF(self) -> None:
        """Update the cumulative distribution function."""
        try:
            self.logger.debug("Updating CDF")

            # Clear previous CDF (but keep fits)
            self._clear_cdf_data()

            if not self._validate_data_availability():
                return

            # Calculate squared step length distributions
            squared_slds = self._calculate_squared_slds()
            if squared_slds is None or len(squared_slds) == 0:
                return

            # Create CDF
            self._create_cdf_plot(squared_slds)

            # Store data for fitting
            self.fitting_data['x'] = self.cdf_x
            self.fitting_data['y'] = self.cdf_y
            self.fitting_data['squared_slds'] = squared_slds

            # Add selection lines for fitting range
            self._add_fitting_range_selectors()

            self.logger.debug("CDF updated successfully")

        except Exception as e:
            self.logger.error(f"Error updating CDF: {e}")

    def _calculate_squared_slds(self) -> Optional[np.ndarray]:
        """Calculate squared step length distributions."""
        try:
            # Determine data source
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                df = self.mainGUI.filteredData
            else:
                df = self.mainGUI.data

            # Group by track and calculate means
            if 'track_number' not in df.columns or 'velocity' not in df.columns:
                return None

            track_means = df.groupby('track_number', as_index=False).mean()

            # Calculate squared step lengths in μm²
            pixel_size_nm = self.mainGUI.trackPlotOptions.pixelSize_selector.value()
            velocity_um = track_means['velocity'] * pixel_size_nm / 1000  # Convert to μm
            squared_slds = np.square(velocity_um)

            # Remove invalid values
            valid_mask = ~(np.isnan(squared_slds) | np.isinf(squared_slds))
            return squared_slds[valid_mask].to_numpy()

        except Exception as e:
            self.logger.error(f"Error calculating squared SLDs: {e}")
            return None

    def _create_cdf_plot(self, data: np.ndarray) -> None:
        """Create cumulative distribution function plot."""
        try:
            n_bins = self.cdfBin_selector.value()

            # Calculate histogram
            data_min, data_max = np.min(data), np.max(data)

            if data_min == data_max:
                self.cdf_x = np.array([data_min])
                self.cdf_y = np.array([1.0])
            else:
                bins = np.linspace(data_min, data_max, n_bins + 1)
                counts, bin_edges = np.histogram(data, bins=bins)

                # Calculate CDF
                pdf = counts / np.sum(counts)
                self.cdf_y = np.cumsum(pdf)
                self.cdf_x = bin_edges[1:]  # Use right edges

            # Get max lags for normalization
            self.nlags = np.max(self.cdf_y) if len(self.cdf_y) > 0 else 1

            # Plot CDF
            cdf_curve = self.w6.plot(
                self.cdf_x, self.cdf_y,
                pen=pg.mkPen(color=(0, 120, 255), width=2),
                symbol='o',
                symbolSize=3,
                symbolBrush=pg.mkBrush(0, 120, 255, 180),
                name='Data CDF'
            )

        except Exception as e:
            self.logger.error(f"Error creating CDF plot: {e}")

    def _add_fitting_range_selectors(self) -> None:
        """Add movable lines for selecting fitting range."""
        try:
            if len(self.fitting_data.get('x', [])) == 0:
                return

            x_min, x_max = np.min(self.fitting_data['x']), np.max(self.fitting_data['x'])

            # Add movable dashed lines for range selection
            self.left_bound_line = pg.InfiniteLine(
                pos=x_min,
                angle=90,
                pen=pg.mkPen('y', style=Qt.DashLine, width=2),
                movable=True,
                bounds=(x_min, x_max)
            )

            self.right_bound_line = pg.InfiniteLine(
                pos=x_max,
                angle=90,
                pen=pg.mkPen('y', style=Qt.DashLine, width=2),
                movable=True,
                bounds=(x_min, x_max)
            )

            self.w6.addItem(self.left_bound_line)
            self.w6.addItem(self.right_bound_line)

        except Exception as e:
            self.logger.error(f"Error adding fitting range selectors: {e}")

    def _clear_cdf_data(self) -> None:
        """Clear CDF data while preserving fits."""
        try:
            # Get all items and remove non-fit items
            items_to_remove = []
            for item in self.w6.listDataItems():
                name = getattr(item, 'name', '')
                if 'Fit' not in name and name != '':
                    items_to_remove.append(item)
                elif name == '' or name == 'Data CDF':
                    items_to_remove.append(item)

            for item in items_to_remove:
                self.w6.removeItem(item)

            # Remove range selectors if they exist
            if hasattr(self, 'left_bound_line'):
                try:
                    self.w6.removeItem(self.left_bound_line)
                    self.w6.removeItem(self.right_bound_line)
                except:
                    pass

        except Exception as e:
            self.logger.error(f"Error clearing CDF data: {e}")

    def fit_exp_dec_1(self) -> None:
        """Fit single exponential decay to CDF data."""
        try:
            self.logger.debug("Fitting single exponential decay")

            if not self._validate_fitting_data():
                return

            # Remove existing single exponential fit
            self._remove_fit_curve('exp1')

            # Get fitting range
            fit_range = self._get_fitting_range()
            if fit_range is None:
                return

            x_fit, y_fit_data = fit_range

            # Perform fit
            try:
                popt, pcov = curve_fit(
                    exp_dec, x_fit, y_fit_data,
                    bounds=([-1.2, 0], [0, 30]),
                    maxfev=5000
                )

                # Calculate diffusion coefficient
                tau_fit = popt[1]
                D_fit = self._tau_to_D(tau_fit)

                # Generate fitted curve
                y_fit = exp_dec(x_fit, *popt)

                # Plot fit
                fit_curve = self.w6.plot(
                    x_fit, y_fit,
                    pen=pg.mkPen('g', width=2),
                    name=f'1-Exp Fit: D = {D_fit:.4g} μm²/s'
                )

                self.fit_curves['exp1'] = fit_curve
                self.fit_parameters['exp1'] = {
                    'D': D_fit,
                    'tau': tau_fit,
                    'A1': popt[0]
                }

                self.logger.info(f'Single exponential fit: D = {D_fit:.4g} μm²/s')

            except Exception as fit_error:
                self.logger.error(f"Fitting failed: {fit_error}")

        except Exception as e:
            self.logger.error(f"Error in single exponential fitting: {e}")

    def fit_exp_dec_2(self) -> None:
        """Fit double exponential decay to CDF data."""
        try:
            self.logger.debug("Fitting double exponential decay")

            if not self._validate_fitting_data():
                return

            # Remove existing double exponential fit
            self._remove_fit_curve('exp2')

            # Get fitting range
            fit_range = self._get_fitting_range()
            if fit_range is None:
                return

            x_fit, y_fit_data = fit_range

            # Perform fit
            try:
                popt, pcov = curve_fit(
                    exp_dec_2, x_fit, y_fit_data,
                    bounds=([-1, 0, 0], [0, 30, 30]),
                    maxfev=5000
                )

                # Extract parameters
                A1, A2 = popt[0], -1 - popt[0]
                tau1_fit, tau2_fit = popt[1], popt[2]
                D1_fit, D2_fit = self._tau_to_D(tau1_fit), self._tau_to_D(tau2_fit)

                # Generate fitted curve
                y_fit = exp_dec_2(x_fit, *popt)

                # Plot fit
                fit_name = f'2-Exp Fit: D1={D1_fit:.3g}, D2={D2_fit:.3g} μm²/s'
                fit_curve = self.w6.plot(
                    x_fit, y_fit,
                    pen=pg.mkPen('r', width=2),
                    name=fit_name
                )

                self.fit_curves['exp2'] = fit_curve
                self.fit_parameters['exp2'] = {
                    'D1': D1_fit, 'D2': D2_fit,
                    'tau1': tau1_fit, 'tau2': tau2_fit,
                    'A1': A1, 'A2': A2
                }

                self.logger.info(f'Double exponential fit: D1={D1_fit:.3g}, D2={D2_fit:.3g} μm²/s')

            except Exception as fit_error:
                self.logger.error(f"Double exponential fitting failed: {fit_error}")

        except Exception as e:
            self.logger.error(f"Error in double exponential fitting: {e}")

    def fit_exp_dec_3(self) -> None:
        """Fit triple exponential decay to CDF data."""
        try:
            self.logger.debug("Fitting triple exponential decay")

            if not self._validate_fitting_data():
                return

            # Remove existing triple exponential fit
            self._remove_fit_curve('exp3')

            # Get fitting range
            fit_range = self._get_fitting_range()
            if fit_range is None:
                return

            x_fit, y_fit_data = fit_range

            # Perform fit
            try:
                popt, pcov = curve_fit(
                    exp_dec_3, x_fit, y_fit_data,
                    bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]),
                    maxfev=5000
                )

                # Extract parameters
                A1, A2, A3 = popt[0], popt[1], -1 - popt[0] - popt[1]
                tau1_fit, tau2_fit, tau3_fit = popt[2], popt[3], popt[4]
                D1_fit = self._tau_to_D(tau1_fit)
                D2_fit = self._tau_to_D(tau2_fit)
                D3_fit = self._tau_to_D(tau3_fit)

                # Generate fitted curve
                y_fit = exp_dec_3(x_fit, *popt)

                # Plot fit
                fit_name = f'3-Exp Fit: D1={D1_fit:.2g}, D2={D2_fit:.2g}, D3={D3_fit:.2g} μm²/s'
                fit_curve = self.w6.plot(
                    x_fit, y_fit,
                    pen=pg.mkPen('y', width=2),
                    name=fit_name
                )

                self.fit_curves['exp3'] = fit_curve
                self.fit_parameters['exp3'] = {
                    'D1': D1_fit, 'D2': D2_fit, 'D3': D3_fit,
                    'tau1': tau1_fit, 'tau2': tau2_fit, 'tau3': tau3_fit,
                    'A1': A1, 'A2': A2, 'A3': A3
                }

                self.logger.info(f'Triple exponential fit: D1={D1_fit:.2g}, D2={D2_fit:.2g}, D3={D3_fit:.2g} μm²/s')

            except Exception as fit_error:
                self.logger.error(f"Triple exponential fitting failed: {fit_error}")

        except Exception as e:
            self.logger.error(f"Error in triple exponential fitting: {e}")

    def _validate_fitting_data(self) -> bool:
        """Validate that fitting data is available."""
        if not self.fitting_data or 'x' not in self.fitting_data or 'y' not in self.fitting_data:
            self.logger.warning("No CDF data available for fitting. Please update CDF first.")
            return False
        return True

    def _get_fitting_range(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get data within the selected fitting range."""
        try:
            if not hasattr(self, 'left_bound_line') or not hasattr(self, 'right_bound_line'):
                self.logger.warning("Fitting range not defined")
                return None

            # Get bounds
            left_bound = min(self.left_bound_line.value(), self.right_bound_line.value())
            right_bound = max(self.left_bound_line.value(), self.right_bound_line.value())

            # Get data
            x_data = self.fitting_data['x']
            y_data = self.fitting_data['y']

            # Select data within bounds
            fit_mask = (left_bound <= x_data) & (x_data <= right_bound)
            x_fit = x_data[fit_mask]
            y_fit = y_data[fit_mask]

            if len(x_fit) < 3:
                self.logger.warning("Insufficient data points in fitting range")
                return None

            return x_fit, y_fit

        except Exception as e:
            self.logger.error(f"Error getting fitting range: {e}")
            return None

    def _remove_fit_curve(self, fit_type: str) -> None:
        """Remove existing fit curve of specified type."""
        try:
            if self.fit_curves.get(fit_type) is not None:
                self.w6.removeItem(self.fit_curves[fit_type])
                self.fit_curves[fit_type] = None

                # Remove from legend
                if hasattr(self, 'cdf_legend'):
                    try:
                        self.cdf_legend.removeItem(fit_type)
                    except:
                        pass

        except Exception as e:
            self.logger.error(f"Error removing fit curve {fit_type}: {e}")

    def clear_fits(self) -> None:
        """Clear all fitted curves."""
        try:
            for fit_type in self.fit_curves.keys():
                self._remove_fit_curve(fit_type)

            self.fit_parameters.clear()
            self.logger.debug("All fits cleared")

        except Exception as e:
            self.logger.error(f"Error clearing fits: {e}")

    def _tau_to_D(self, tau: float) -> float:
        """
        Convert time constant to diffusion coefficient.

        Formula: tau = 4Dt, where t is the lag time duration

        Args:
            tau: Time constant from exponential fit

        Returns:
            Diffusion coefficient in μm²/s
        """
        try:
            # Get frame duration in seconds
            frame_length_ms = self.mainGUI.trackPlotOptions.frameLength_selector.value()
            t = (frame_length_ms / 1000) * self.nlags  # Convert to seconds

            if t <= 0:
                self.logger.warning("Invalid time duration for diffusion calculation")
                return 0

            D = tau / (4 * t)
            return D

        except Exception as e:
            self.logger.error(f"Error converting tau to D: {e}")
            return 0

    def export_analysis_data(self) -> None:
        """Export analysis data and fit results."""
        try:
            from qtpy.QtWidgets import QFileDialog
            import json

            # Get save path
            file_path, _ = QFileDialog.getSaveFileName(
                None,
                "Export Diffusion Analysis",
                "diffusion_analysis.json",
                "JSON Files (*.json);;CSV Files (*.csv)"
            )

            if not file_path:
                return

            # Prepare export data
            export_data = {
                'analysis_type': 'diffusion_analysis',
                'parameters': {
                    'pixel_size_nm': self.mainGUI.trackPlotOptions.pixelSize_selector.value(),
                    'frame_length_ms': self.mainGUI.trackPlotOptions.frameLength_selector.value(),
                },
                'fit_results': self.fit_parameters,
                'raw_data': {
                    'cdf_x': self.fitting_data.get('x', []).tolist() if 'x' in self.fitting_data else [],
                    'cdf_y': self.fitting_data.get('y', []).tolist() if 'y' in self.fitting_data else [],
                    'squared_slds': self.fitting_data.get('squared_slds', []).tolist() if 'squared_slds' in self.fitting_data else []
                }
            }

            # Save data
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                # Save as CSV (simplified)
                if 'x' in self.fitting_data and 'y' in self.fitting_data:
                    df = pd.DataFrame({
                        'cdf_x': self.fitting_data['x'],
                        'cdf_y': self.fitting_data['y']
                    })
                    df.to_csv(file_path, index=False)

            self.logger.info(f"Analysis data exported to: {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting analysis data: {e}")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of current analysis results."""
        try:
            summary = {
                'fit_parameters': self.fit_parameters.copy(),
                'data_points': len(self.fitting_data.get('x', [])),
                'analysis_settings': {
                    'histogram_bins': self.histoBin_selector.value(),
                    'cdf_bins': self.cdfBin_selector.value(),
                    'plot_type': self.plotTypeSelector.value()
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {e}")
            return {}

    def show(self) -> None:
        """Show the diffusion analysis window."""
        try:
            self.win.show()
            self.logger.debug("DiffusionPlotWindow shown")
        except Exception as e:
            self.logger.error(f"Error showing DiffusionPlotWindow: {e}")

    def close(self) -> None:
        """Close the diffusion analysis window."""
        try:
            self.win.close()
            self.logger.debug("DiffusionPlotWindow closed")
        except Exception as e:
            self.logger.error(f"Error closing DiffusionPlotWindow: {e}")

    def hide(self) -> None:
        """Hide the diffusion analysis window."""
        try:
            self.win.hide()
            self.logger.debug("DiffusionPlotWindow hidden")
        except Exception as e:
            self.logger.error(f"Error hiding DiffusionPlotWindow: {e}")
