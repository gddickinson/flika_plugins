#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flower Plot Module for FLIKA Tracking Plugin

This module provides a specialized visualization window that displays all tracks
with their origins centered at (0,0), creating a "flower plot" pattern. This
visualization is useful for analyzing track directionality, displacement patterns,
and overall movement characteristics.

The flower plot normalizes all tracks to start from the origin, making it easy
to compare track shapes, directions, and relative displacements regardless of
their original positions in the field of view.

Created on Fri Jun  2 15:26:49 2023
@author: george
"""

import logging
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QSpinBox
from qtpy.QtGui import QFont, QPen, QColor

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g

# Plugin imports
from .helperFunctions import dictFromList

# Set up logging
logger = logging.getLogger(__name__)


class FlowerPlotWindow:
    """
    Flower plot visualization window for track analysis.

    This window displays all tracks with their origins normalized to (0,0),
    creating a flower-like pattern that reveals directional preferences,
    displacement distributions, and movement patterns across all tracks.

    Features:
    - Origin-centered track display
    - Customizable plot appearance
    - Track filtering and selection
    - Export capabilities
    - Statistical overlays
    - Interactive controls

    Attributes:
        mainGUI: Reference to main GUI instance
        win: PyQtGraph graphics layout widget
        plt: Main plot widget
        pathitems: List of track path items
        controls: Dictionary of control widgets
        settings: Plot display settings
    """

    def __init__(self, mainGUI):
        """
        Initialize the flower plot window.

        Args:
            mainGUI: Reference to main GUI instance containing track data
        """
        super().__init__()
        self.mainGUI = mainGUI
        self.logger = logging.getLogger(__name__)

        # Track visualization components
        self.pathitems: List[pg.GraphicsObject] = []
        self.controls: Dict[str, QWidget] = {}

        # Plot settings
        self.settings = {
            'show_origin_marker': True,
            'show_grid': True,
            'show_axes': True,
            'show_statistics': False,
            'track_width': 2,
            'plot_range': 10,
            'color_mode': 'by_track',
            'background_color': 'black'
        }

        # Statistics overlays
        self.stat_items: List[pg.GraphicsObject] = []

        self._setup_ui()
        self.logger.debug("FlowerPlotWindow initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        try:
            # Create main graphics layout widget
            self.win = pg.GraphicsLayoutWidget()
            self.win.resize(600, 600)
            self.win.setWindowTitle('Flower Plot - Track Origins Centered')

            # Create main plot
            self._create_main_plot()

            # Create control panel
            self._create_control_panel()

            # Set up initial plot appearance
            self._setup_plot_appearance()

            self.logger.debug("FlowerPlotWindow UI setup completed")

        except Exception as e:
            self.logger.error(f"Error setting up FlowerPlotWindow UI: {e}")

    def _create_main_plot(self) -> None:
        """Create the main plot widget."""
        try:
            self.plt = self.win.addPlot(title='Flower Plot - Normalized Track Origins')

            # Configure plot properties
            self.plt.setAspectLocked(True)
            self.plt.showGrid(x=True, y=True, alpha=0.3)
            self.plt.setXRange(-self.settings['plot_range'], self.settings['plot_range'])
            self.plt.setYRange(-self.settings['plot_range'], self.settings['plot_range'])

            # Invert Y-axis to match image coordinates
            self.plt.getViewBox().invertY(True)

            # Set axis labels
            self.plt.setLabel('left', 'Y Displacement', units='pixels')
            self.plt.setLabel('bottom', 'X Displacement', units='pixels')

            # Add origin marker
            self._add_origin_marker()

        except Exception as e:
            self.logger.error(f"Error creating main plot: {e}")

    def _create_control_panel(self) -> None:
        """Create control panel for plot customization."""
        try:
            # Create control dock (simplified for this version)
            # In a full implementation, this could be a separate docked widget

            # Note: For now, controls are minimal since the original was simple
            # Future enhancements could include:
            # - Track filtering controls
            # - Color scheme selection
            # - Statistical overlay options
            # - Export controls
            pass

        except Exception as e:
            self.logger.error(f"Error creating control panel: {e}")

    def _setup_plot_appearance(self) -> None:
        """Set up initial plot appearance."""
        try:
            # Configure background
            if self.settings['background_color'] == 'black':
                self.plt.setBackground('k')
            else:
                self.plt.setBackground('w')

            # Configure grid
            if self.settings['show_grid']:
                self.plt.showGrid(x=True, y=True, alpha=0.3)
            else:
                self.plt.showGrid(x=False, y=False)

            # Configure axes
            if not self.settings['show_axes']:
                self.plt.hideAxis('bottom')
                self.plt.hideAxis('left')

        except Exception as e:
            self.logger.error(f"Error setting up plot appearance: {e}")

    def _add_origin_marker(self) -> None:
        """Add a marker at the origin (0,0)."""
        try:
            if self.settings['show_origin_marker']:
                # Add crosshair at origin
                origin_pen = pg.mkPen(color='r', width=2, style=Qt.DashLine)

                # Vertical line
                v_line = pg.InfiniteLine(pos=0, angle=90, pen=origin_pen)
                self.plt.addItem(v_line)

                # Horizontal line
                h_line = pg.InfiniteLine(pos=0, angle=0, pen=origin_pen)
                self.plt.addItem(h_line)

                # Central point
                origin_scatter = pg.ScatterPlotItem(
                    [0], [0],
                    size=10,
                    pen=pg.mkPen('r', width=2),
                    brush=pg.mkBrush('r'),
                    symbol='+'
                )
                self.plt.addItem(origin_scatter)

        except Exception as e:
            self.logger.error(f"Error adding origin marker: {e}")

    def plotTracks(self, track_data: Optional[pd.DataFrame] = None) -> None:
        """
        Plot tracks in the flower plot format.

        Args:
            track_data: Optional DataFrame with track data. If None, uses mainGUI data.
        """
        try:
            self.logger.debug("Plotting tracks in flower plot")

            # Clear existing tracks
            self.clearTracks()

            # Get track data
            if track_data is None:
                if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                    self.logger.warning("No track data available")
                    return

                # Use filtered data if available
                if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                    data = self.mainGUI.filteredData
                else:
                    data = self.mainGUI.data
            else:
                data = track_data

            # Check for required columns
            required_cols = ['track_number', 'zeroed_X', 'zeroed_Y']
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return

            # Plot each track
            track_ids = data['track_number'].unique()
            valid_track_ids = [tid for tid in track_ids if not pd.isna(tid)]

            self.logger.debug(f"Plotting {len(valid_track_ids)} tracks")

            for track_id in valid_track_ids:
                self._plot_single_track(data, track_id)

            # Add statistics overlay if enabled
            if self.settings['show_statistics']:
                self._add_statistics_overlay(data)

            # Update plot range if needed
            self._update_plot_range(data)

            self.logger.info(f"Flower plot updated with {len(valid_track_ids)} tracks")

        except Exception as e:
            self.logger.error(f"Error plotting tracks: {e}")

    def _plot_single_track(self, data: pd.DataFrame, track_id: Union[int, float]) -> None:
        """
        Plot a single track in the flower plot.

        Args:
            data: DataFrame containing track data
            track_id: ID of the track to plot
        """
        try:
            # Get track data
            track_data = data[data['track_number'] == track_id].copy()

            if len(track_data) < 2:
                # Need at least 2 points for a track
                return

            # Sort by frame to ensure proper track order
            track_data = track_data.sort_values('frame')

            # Get coordinates (already zeroed to origin)
            x_coords = track_data['zeroed_X'].values
            y_coords = track_data['zeroed_Y'].values

            # Remove invalid coordinates
            valid_mask = ~(np.isnan(x_coords) | np.isnan(y_coords) |
                          np.isinf(x_coords) | np.isinf(y_coords))

            if not np.any(valid_mask):
                return

            x_clean = x_coords[valid_mask]
            y_clean = y_coords[valid_mask]

            if len(x_clean) < 2:
                return

            # Determine track color
            color = self._get_track_color(track_id, track_data)

            # Create pen for track
            pen = pg.mkPen(
                color=color,
                width=self.settings['track_width'],
                style=Qt.SolidLine
            )

            # Plot track
            track_item = self.plt.plot(
                x_clean, y_clean,
                pen=pen,
                symbol='o',
                symbolSize=3,
                symbolBrush=pg.mkBrush(color),
                symbolPen=pg.mkPen(color, width=1)
            )

            self.pathitems.append(track_item)

            # Add track start marker
            start_marker = pg.ScatterPlotItem(
                [x_clean[0]], [y_clean[0]],
                size=8,
                pen=pg.mkPen(color, width=2),
                brush=pg.mkBrush(color),
                symbol='s'  # Square for start
            )
            self.plt.addItem(start_marker)
            self.pathitems.append(start_marker)

            # Add track end marker
            end_marker = pg.ScatterPlotItem(
                [x_clean[-1]], [y_clean[-1]],
                size=8,
                pen=pg.mkPen(color, width=2),
                brush=pg.mkBrush(color),
                symbol='t'  # Triangle for end
            )
            self.plt.addItem(end_marker)
            self.pathitems.append(end_marker)

        except Exception as e:
            self.logger.error(f"Error plotting track {track_id}: {e}")

    def _get_track_color(self, track_id: Union[int, float],
                        track_data: pd.DataFrame) -> QColor:
        """
        Get color for a track based on current color mode.

        Args:
            track_id: Track identifier
            track_data: DataFrame with track data

        Returns:
            QColor for the track
        """
        try:
            if self.settings['color_mode'] == 'by_track':
                # Use PyQtGraph's automatic color cycling
                return pg.intColor(int(track_id))

            elif self.settings['color_mode'] == 'by_length':
                # Color by track length
                track_length = len(track_data)
                # Normalize to 0-1 range (assuming max length of 100)
                normalized = min(track_length / 100.0, 1.0)
                return pg.colormap.get('viridis').mapToQColor([normalized])[0]

            elif self.settings['color_mode'] == 'by_displacement':
                # Color by net displacement
                if 'zeroed_X' in track_data.columns and 'zeroed_Y' in track_data.columns:
                    final_x = track_data['zeroed_X'].iloc[-1]
                    final_y = track_data['zeroed_Y'].iloc[-1]
                    displacement = np.sqrt(final_x**2 + final_y**2)
                    # Normalize (assuming max displacement of 50 pixels)
                    normalized = min(displacement / 50.0, 1.0)
                    return pg.colormap.get('plasma').mapToQColor([normalized])[0]

            # Default color
            return QColor(100, 150, 255)

        except Exception as e:
            self.logger.error(f"Error getting track color: {e}")
            return QColor(100, 150, 255)

    def _add_statistics_overlay(self, data: pd.DataFrame) -> None:
        """
        Add statistical overlays to the plot.

        Args:
            data: DataFrame with track data
        """
        try:
            # Clear existing statistics
            self._clear_statistics()

            if 'zeroed_X' not in data.columns or 'zeroed_Y' not in data.columns:
                return

            # Calculate track endpoints
            endpoints = []
            for track_id in data['track_number'].unique():
                if pd.isna(track_id):
                    continue

                track_data = data[data['track_number'] == track_id]
                if len(track_data) > 0:
                    final_x = track_data['zeroed_X'].iloc[-1]
                    final_y = track_data['zeroed_Y'].iloc[-1]
                    if not (np.isnan(final_x) or np.isnan(final_y)):
                        endpoints.append((final_x, final_y))

            if not endpoints:
                return

            endpoints = np.array(endpoints)

            # Add mean displacement vector
            mean_x = np.mean(endpoints[:, 0])
            mean_y = np.mean(endpoints[:, 1])

            mean_arrow = pg.ArrowItem(
                pos=(mean_x, mean_y),
                angle=0,
                headLen=20,
                tipAngle=30,
                pen=pg.mkPen('yellow', width=3),
                brush=pg.mkBrush('yellow')
            )
            self.plt.addItem(mean_arrow)
            self.stat_items.append(mean_arrow)

            # Add displacement distribution ellipse (2 standard deviations)
            std_x = np.std(endpoints[:, 0])
            std_y = np.std(endpoints[:, 1])

            # Simple ellipse approximation
            theta = np.linspace(0, 2*np.pi, 100)
            ellipse_x = mean_x + 2 * std_x * np.cos(theta)
            ellipse_y = mean_y + 2 * std_y * np.sin(theta)

            ellipse_item = self.plt.plot(
                ellipse_x, ellipse_y,
                pen=pg.mkPen('yellow', width=2, style=Qt.DashLine)
            )
            self.stat_items.append(ellipse_item)

        except Exception as e:
            self.logger.error(f"Error adding statistics overlay: {e}")

    def _clear_statistics(self) -> None:
        """Clear statistical overlay items."""
        try:
            for item in self.stat_items:
                self.plt.removeItem(item)
            self.stat_items.clear()
        except Exception as e:
            self.logger.error(f"Error clearing statistics: {e}")

    def _update_plot_range(self, data: pd.DataFrame) -> None:
        """
        Update plot range based on data extent.

        Args:
            data: DataFrame with track data
        """
        try:
            if 'zeroed_X' not in data.columns or 'zeroed_Y' not in data.columns:
                return

            # Calculate data extent
            valid_x = data['zeroed_X'].dropna()
            valid_y = data['zeroed_Y'].dropna()

            if len(valid_x) == 0 or len(valid_y) == 0:
                return

            x_range = max(abs(valid_x.min()), abs(valid_x.max()))
            y_range = max(abs(valid_y.min()), abs(valid_y.max()))
            max_range = max(x_range, y_range)

            # Add 10% padding
            plot_range = max_range * 1.1

            # Only update if significantly different
            current_range = self.settings['plot_range']
            if abs(plot_range - current_range) / current_range > 0.2:
                self.settings['plot_range'] = plot_range
                self.plt.setXRange(-plot_range, plot_range)
                self.plt.setYRange(-plot_range, plot_range)

        except Exception as e:
            self.logger.error(f"Error updating plot range: {e}")

    def clearTracks(self) -> None:
        """Clear all track visualizations from the plot."""
        try:
            # Remove all track path items
            for item in self.pathitems:
                try:
                    self.plt.removeItem(item)
                except:
                    pass  # Item might have been removed already

            self.pathitems.clear()

            # Clear statistics
            self._clear_statistics()

            self.logger.debug("Tracks cleared from flower plot")

        except Exception as e:
            self.logger.error(f"Error clearing tracks: {e}")

    def exportPlot(self, filename: str, width: int = 800, height: int = 800) -> None:
        """
        Export the flower plot to an image file.

        Args:
            filename: Output filename
            width: Image width in pixels
            height: Image height in pixels
        """
        try:
            exporter = pg.exporters.ImageExporter(self.plt.plotItem)
            exporter.parameters()['width'] = width
            exporter.parameters()['height'] = height
            exporter.export(filename)

            self.logger.info(f"Flower plot exported to: {filename}")

        except Exception as e:
            self.logger.error(f"Error exporting flower plot: {e}")

    def getTrackStatistics(self) -> Dict[str, Any]:
        """
        Calculate and return track statistics for the flower plot.

        Returns:
            Dictionary containing various track statistics
        """
        try:
            if not hasattr(self.mainGUI, 'data') or self.mainGUI.data is None:
                return {}

            # Use filtered data if available
            if self.mainGUI.useFilteredData and self.mainGUI.filteredData is not None:
                data = self.mainGUI.filteredData
            else:
                data = self.mainGUI.data

            if 'zeroed_X' not in data.columns or 'zeroed_Y' not in data.columns:
                return {}

            # Calculate track endpoints and statistics
            endpoints = []
            track_lengths = []
            displacements = []

            for track_id in data['track_number'].unique():
                if pd.isna(track_id):
                    continue

                track_data = data[data['track_number'] == track_id]
                if len(track_data) > 1:
                    final_x = track_data['zeroed_X'].iloc[-1]
                    final_y = track_data['zeroed_Y'].iloc[-1]

                    if not (np.isnan(final_x) or np.isnan(final_y)):
                        endpoints.append((final_x, final_y))
                        track_lengths.append(len(track_data))
                        displacement = np.sqrt(final_x**2 + final_y**2)
                        displacements.append(displacement)

            if not endpoints:
                return {}

            endpoints = np.array(endpoints)

            # Calculate statistics
            stats = {
                'n_tracks': len(endpoints),
                'mean_displacement': {
                    'x': float(np.mean(endpoints[:, 0])),
                    'y': float(np.mean(endpoints[:, 1])),
                    'magnitude': float(np.mean(displacements))
                },
                'std_displacement': {
                    'x': float(np.std(endpoints[:, 0])),
                    'y': float(np.std(endpoints[:, 1])),
                    'magnitude': float(np.std(displacements))
                },
                'max_displacement': float(np.max(displacements)),
                'min_displacement': float(np.min(displacements)),
                'mean_track_length': float(np.mean(track_lengths)),
                'displacement_range': {
                    'x': [float(np.min(endpoints[:, 0])), float(np.max(endpoints[:, 0]))],
                    'y': [float(np.min(endpoints[:, 1])), float(np.max(endpoints[:, 1]))]
                }
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating track statistics: {e}")
            return {}

    def updateSettings(self, new_settings: Dict[str, Any]) -> None:
        """
        Update plot settings and refresh display.

        Args:
            new_settings: Dictionary of settings to update
        """
        try:
            self.settings.update(new_settings)

            # Refresh plot appearance
            self._setup_plot_appearance()

            # Replot with new settings
            self.plotTracks()

            self.logger.debug("Flower plot settings updated")

        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")

    def show(self) -> None:
        """Show the flower plot window."""
        try:
            self.win.show()
            self.logger.debug("Flower plot window shown")
        except Exception as e:
            self.logger.error(f"Error showing flower plot window: {e}")

    def close(self) -> None:
        """Close the flower plot window."""
        try:
            self.clearTracks()
            self.win.close()
            self.logger.debug("Flower plot window closed")
        except Exception as e:
            self.logger.error(f"Error closing flower plot window: {e}")

    def hide(self) -> None:
        """Hide the flower plot window."""
        try:
            self.win.hide()
            self.logger.debug("Flower plot window hidden")
        except Exception as e:
            self.logger.error(f"Error hiding flower plot window: {e}")
