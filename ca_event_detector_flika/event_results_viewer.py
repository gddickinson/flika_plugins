#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Results Viewer for Calcium Event Detector
===============================================

Interactive spreadsheet viewer for analyzing detected calcium events.

Author: George Stuyt (with Claude)
Date: 2024-12-25
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

from qtpy.QtCore import Qt, Signal, QAbstractTableModel, QModelIndex
from qtpy.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTableView, QFileDialog,
                            QComboBox, QLineEdit, QMessageBox, QSplitter,
                            QGroupBox, QTabWidget)
from qtpy.QtGui import QColor
import pyqtgraph as pg

import logging
logger = logging.getLogger(__name__)


class EventTableModel(QAbstractTableModel):
    """Table model for displaying event data."""
    
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data
        self._filtered_data = data.copy()
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._filtered_data)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self._filtered_data.columns)
    
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole:
            value = self._filtered_data.iloc[index.row(), index.column()]
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, (float, np.floating)):
                return f"{value:.2f}"
            return str(value)
        
        elif role == Qt.BackgroundRole:
            # Color code by class
            class_col = self._filtered_data.columns.get_loc('class')
            class_val = self._filtered_data.iloc[index.row(), class_col]
            
            if class_val == 'spark':
                return QColor(200, 255, 200)  # Light green
            elif class_val == 'puff':
                return QColor(255, 235, 200)  # Light orange
            elif class_val == 'wave':
                return QColor(255, 200, 200)  # Light red
        
        return None
    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._filtered_data.columns[section])
            else:
                return str(section + 1)
        return None
    
    def set_filter(self, column: str, value: str):
        """Filter data by column value."""
        if not value or value == "All":
            self._filtered_data = self._data.copy()
        else:
            self._filtered_data = self._data[self._data[column] == value].copy()
        
        self.layoutChanged.emit()


class EventResultsViewer(QMainWindow):
    """
    Interactive viewer for calcium event detection results.
    
    Features:
    - Spreadsheet view of all detected events
    - Filter by event type
    - Sort by any column
    - Statistics summary
    - Export to CSV
    - Plot event properties
    """
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.model = None
        
        self.setWindowTitle("Calcium Event Results Viewer")
        self.resize(1200, 800)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # === Top Controls ===
        controls = QHBoxLayout()
        
        self.info_label = QLabel("No data loaded")
        controls.addWidget(self.info_label)
        
        controls.addStretch()
        
        # Filter controls
        controls.addWidget(QLabel("Filter by class:"))
        self.class_filter = QComboBox()
        self.class_filter.addItem("All")
        self.class_filter.currentTextChanged.connect(self._apply_filter)
        controls.addWidget(self.class_filter)
        
        # Buttons
        load_btn = QPushButton("Load CSV")
        load_btn.clicked.connect(self.load_csv)
        controls.addWidget(load_btn)
        
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv)
        controls.addWidget(export_btn)
        
        layout.addLayout(controls)
        
        # === Main Content ===
        splitter = QSplitter(Qt.Vertical)
        
        # Table view
        table_group = QGroupBox("Event Table")
        table_layout = QVBoxLayout(table_group)
        
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        table_layout.addWidget(self.table_view)
        
        splitter.addWidget(table_group)
        
        # Tabs for statistics and plots
        self.tabs = QTabWidget()
        
        # Statistics tab
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.stats_text = QLabel("No statistics available")
        self.stats_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats_text.setWordWrap(True)
        stats_layout.addWidget(self.stats_text)
        self.tabs.addTab(stats_widget, "Statistics")
        
        # Plots tab
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        
        # Plot controls
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Plot:"))
        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "Size Distribution",
            "Duration Distribution",
            "Spatial Distribution (X-Y)",
            "Temporal Distribution"
        ])
        self.plot_selector.currentTextChanged.connect(self._update_plot)
        plot_controls.addWidget(self.plot_selector)
        plot_controls.addStretch()
        plots_layout.addLayout(plot_controls)
        
        # Plot widget
        self.plot_widget = pg.PlotWidget()
        plots_layout.addWidget(self.plot_widget)
        
        self.tabs.addTab(plots_widget, "Plots")
        
        splitter.addWidget(self.tabs)
        splitter.setSizes([400, 400])
        
        layout.addWidget(splitter)
    
    def set_data(self, results: Dict[str, np.ndarray]):
        """
        Set detection results data.
        
        Parameters
        ----------
        results : dict
            Detection results with 'class_mask' and 'instance_mask'
        """
        # Extract event properties
        class_mask = results['class_mask']
        instance_mask = results['instance_mask']
        
        events = []
        class_names = ['background', 'spark', 'puff', 'wave']
        
        for instance_id in range(1, instance_mask.max() + 1):
            mask = instance_mask == instance_id
            if not np.any(mask):
                continue
            
            # Get class
            class_id = int(class_mask[mask][0])
            class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
            
            # Get properties
            size = np.sum(mask)
            coords = np.argwhere(mask)
            
            t_mean = coords[:, 0].mean()
            y_mean = coords[:, 1].mean()
            x_mean = coords[:, 2].mean()
            
            t_min, t_max = coords[:, 0].min(), coords[:, 0].max()
            duration = t_max - t_min + 1
            
            # Calculate extent
            y_extent = coords[:, 1].max() - coords[:, 1].min() + 1
            x_extent = coords[:, 2].max() - coords[:, 2].min() + 1
            
            events.append({
                'ID': instance_id,
                'class': class_name,
                'class_id': class_id,
                'size_pixels': size,
                't_center': t_mean,
                'y_center': y_mean,
                'x_center': x_mean,
                'frame_start': int(t_min),
                'frame_end': int(t_max),
                'duration_frames': int(duration),
                'y_extent': int(y_extent),
                'x_extent': int(x_extent)
            })
        
        self.data = pd.DataFrame(events)
        self._update_view()
    
    def _update_view(self):
        """Update table view and statistics."""
        if self.data is None or len(self.data) == 0:
            self.info_label.setText("No events detected")
            return
        
        # Update table
        self.model = EventTableModel(self.data)
        self.table_view.setModel(self.model)
        
        # Update info
        n_events = len(self.data)
        self.info_label.setText(f"{n_events} events detected")
        
        # Update class filter
        self.class_filter.clear()
        self.class_filter.addItem("All")
        for class_name in self.data['class'].unique():
            self.class_filter.addItem(class_name)
        
        # Update statistics
        self._update_statistics()
        
        # Update plot
        self._update_plot()
    
    def _apply_filter(self, class_name: str):
        """Apply class filter to table."""
        if self.model is not None:
            self.model.set_filter('class', class_name)
            self._update_statistics()
    
    def _update_statistics(self):
        """Update statistics display."""
        if self.data is None or len(self.data) == 0:
            self.stats_text.setText("No statistics available")
            return
        
        # Get filtered data
        if self.model is not None:
            data = self.model._filtered_data
        else:
            data = self.data
        
        # Calculate statistics
        n_total = len(data)
        
        stats = []
        stats.append(f"<h3>Event Summary</h3>")
        stats.append(f"<b>Total events:</b> {n_total}<br>")
        
        # By class
        stats.append(f"<br><b>By class:</b><br>")
        for class_name in ['spark', 'puff', 'wave']:
            n_class = len(data[data['class'] == class_name])
            pct = 100 * n_class / n_total if n_total > 0 else 0
            stats.append(f"  {class_name}: {n_class} ({pct:.1f}%)<br>")
        
        # Size statistics
        stats.append(f"<br><b>Size (pixels):</b><br>")
        stats.append(f"  Mean: {data['size_pixels'].mean():.1f}<br>")
        stats.append(f"  Median: {data['size_pixels'].median():.1f}<br>")
        stats.append(f"  Range: {data['size_pixels'].min():.0f} - {data['size_pixels'].max():.0f}<br>")
        
        # Duration statistics
        if 'duration_frames' in data.columns:
            stats.append(f"<br><b>Duration (frames):</b><br>")
            stats.append(f"  Mean: {data['duration_frames'].mean():.1f}<br>")
            stats.append(f"  Median: {data['duration_frames'].median():.1f}<br>")
            stats.append(f"  Range: {data['duration_frames'].min():.0f} - {data['duration_frames'].max():.0f}<br>")
        
        self.stats_text.setText("".join(stats))
    
    def _update_plot(self):
        """Update plot based on selection."""
        if self.data is None or len(self.data) == 0:
            return
        
        self.plot_widget.clear()
        
        plot_type = self.plot_selector.currentText()
        
        # Get filtered data
        if self.model is not None:
            data = self.model._filtered_data
        else:
            data = self.data
        
        if plot_type == "Size Distribution":
            # Histogram of event sizes by class
            bins = np.logspace(0, np.log10(data['size_pixels'].max()), 30)
            
            for class_name, color in [('spark', 'g'), ('puff', 'orange'), ('wave', 'r')]:
                class_data = data[data['class'] == class_name]['size_pixels']
                if len(class_data) > 0:
                    y, x = np.histogram(class_data, bins=bins)
                    self.plot_widget.plot(x, y, stepMode=True, fillLevel=0,
                                        brush=color, name=class_name)
            
            self.plot_widget.setLabel('bottom', 'Size (pixels)')
            self.plot_widget.setLabel('left', 'Count')
            self.plot_widget.setLogMode(x=True)
            self.plot_widget.addLegend()
        
        elif plot_type == "Duration Distribution":
            # Histogram of durations by class
            bins = np.arange(0, data['duration_frames'].max() + 2)
            
            for class_name, color in [('spark', 'g'), ('puff', 'orange'), ('wave', 'r')]:
                class_data = data[data['class'] == class_name]['duration_frames']
                if len(class_data) > 0:
                    y, x = np.histogram(class_data, bins=bins)
                    self.plot_widget.plot(x, y, stepMode=True, fillLevel=0,
                                        brush=color, name=class_name)
            
            self.plot_widget.setLabel('bottom', 'Duration (frames)')
            self.plot_widget.setLabel('left', 'Count')
            self.plot_widget.addLegend()
        
        elif plot_type == "Spatial Distribution (X-Y)":
            # Scatter plot of event positions
            for class_name, color, symbol in [
                ('spark', 'g', 'o'),
                ('puff', 'orange', 's'),
                ('wave', 'r', 't')
            ]:
                class_data = data[data['class'] == class_name]
                if len(class_data) > 0:
                    self.plot_widget.plot(
                        class_data['x_center'],
                        class_data['y_center'],
                        pen=None, symbol=symbol, symbolPen=None,
                        symbolBrush=color, symbolSize=8,
                        name=class_name
                    )
            
            self.plot_widget.setLabel('bottom', 'X position')
            self.plot_widget.setLabel('left', 'Y position')
            self.plot_widget.addLegend()
        
        elif plot_type == "Temporal Distribution":
            # Histogram of event start times
            bins = np.arange(0, data['frame_start'].max() + 2)
            
            for class_name, color in [('spark', 'g'), ('puff', 'orange'), ('wave', 'r')]:
                class_data = data[data['class'] == class_name]['frame_start']
                if len(class_data) > 0:
                    y, x = np.histogram(class_data, bins=bins)
                    self.plot_widget.plot(x, y, stepMode=True, fillLevel=0,
                                        brush=color, name=class_name)
            
            self.plot_widget.setLabel('bottom', 'Frame')
            self.plot_widget.setLabel('left', 'Event count')
            self.plot_widget.addLegend()
    
    def load_csv(self):
        """Load event data from CSV file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Event Data",
            "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
        
        try:
            self.data = pd.read_csv(filename)
            self._update_view()
            self.info_label.setText(f"Loaded: {Path(filename).name}")
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            QMessageBox.critical(
                self, "Load Error",
                f"Error loading CSV file:\n{str(e)}"
            )
    
    def export_csv(self):
        """Export current data to CSV."""
        if self.data is None or len(self.data) == 0:
            QMessageBox.warning(
                self, "No Data",
                "No data to export"
            )
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Event Data",
            "calcium_events.csv",
            "CSV Files (*.csv)"
        )
        
        if not filename:
            return
        
        try:
            # Export filtered data if filter is active
            if self.model is not None:
                data_to_export = self.model._filtered_data
            else:
                data_to_export = self.data
            
            data_to_export.to_csv(filename, index=False)
            
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {len(data_to_export)} events to:\n{filename}"
            )
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            QMessageBox.critical(
                self, "Export Error",
                f"Error exporting CSV:\n{str(e)}"
            )
