#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Tracking Analysis Application

This script allows running the FLIKA tracking plugin as a standalone application
without requiring the full FLIKA environment. It provides a minimal GUI for
loading and analyzing tracking data.

Usage:
    python standalone_app.py [--debug] [--data-file PATH]

Features:
    - Independent operation without FLIKA
    - Comprehensive logging
    - Command line argument support
    - Error handling and user feedback

Author: george.dickinson@gmail.com
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PyQt imports
try:
    from qtpy.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QHBoxLayout, QWidget, QLabel, QPushButton,
                               QFileDialog, QMessageBox, QTextEdit,
                               QTabWidget, QSplitter, QGroupBox)
    from qtpy.QtCore import Qt, QTimer, QThread, Signal
    from qtpy.QtGui import QFont, QIcon, QPixmap
except ImportError as e:
    print(f"Error importing Qt libraries: {e}")
    print("Please install PyQt5, PySide2, or PySide6")
    sys.exit(1)

# Scientific computing imports
try:
    import numpy as np
    import pandas as pd
    import pyqtgraph as pg
except ImportError as e:
    print(f"Error importing scientific libraries: {e}")
    print("Please install numpy, pandas, and pyqtgraph")
    sys.exit(1)


def setup_logging(debug: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        debug: Enable debug level logging
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "tracking_app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    return logger


class MockGlobalVars:
    """Mock FLIKA global variables for standalone operation."""
    
    def __init__(self):
        self.win = None
        self.m = MockMainWindow()
        self.settings = {
            'locsAndTracksPlotter': {
                'filename': '',
                'filetype': 'flika',
                'pixelSize': 108,
                'set_track_colour': False,
            }
        }
    
    def alert(self, message: str) -> None:
        """Show alert message."""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Alert")
        msg_box.setText(str(message))
        msg_box.exec_()


class MockMainWindow:
    """Mock main window for status bar messages."""
    
    def __init__(self):
        self._status_message = ""
    
    def statusBar(self):
        return MockStatusBar()


class MockStatusBar:
    """Mock status bar for message display."""
    
    def showMessage(self, message: str) -> None:
        logger = logging.getLogger(__name__)
        logger.info(f"Status: {message}")


class MockWindow:
    """Mock FLIKA window for image data."""
    
    def __init__(self):
        self.mt = 1000  # Number of time points
        self.image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
        self.currentIndex = 0
        self.closed = False
        self.rois = []
        self.currentROI = None
        self.scatterPoints = [[] for _ in range(self.mt)]
        self.scatterPlot = None
        
    def imageArray(self) -> np.ndarray:
        """Return mock image array."""
        return np.random.randint(0, 65535, (self.mt, 512, 512), dtype=np.uint16)
    
    def setIndex(self, index: int) -> None:
        """Set current time index."""
        self.currentIndex = max(0, min(index, self.mt - 1))
    
    def updateindex(self) -> None:
        """Update index display."""
        pass


class DataLoader(QThread):
    """Thread for loading data files without blocking UI."""
    
    dataLoaded = Signal(object)  # Emits loaded DataFrame
    errorOccurred = Signal(str)  # Emits error message
    progressUpdate = Signal(str)  # Emits progress messages
    
    def __init__(self, filepath: str, filetype: str):
        super().__init__()
        self.filepath = filepath
        self.filetype = filetype
        self.logger = logging.getLogger(__name__)
    
    def run(self) -> None:
        """Load data in separate thread."""
        try:
            self.progressUpdate.emit("Loading data file...")
            
            if self.filetype == 'json':
                data = self._load_json_data()
            else:
                data = pd.read_csv(self.filepath)
            
            if data is not None and not data.empty:
                self.progressUpdate.emit("Data loaded successfully")
                self.dataLoaded.emit(data)
            else:
                self.errorOccurred.emit("No data found in file")
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            self.errorOccurred.emit(f"Error loading data: {str(e)}")
    
    def _load_json_data(self) -> Optional[pd.DataFrame]:
        """Load JSON format data."""
        import json
        
        with open(self.filepath, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame (simplified version)
        frames, track_numbers, x_coords, y_coords = [], [], [], []
        txy_pts = np.array(data['txy_pts'])
        
        for track_idx, track in enumerate(data['tracks']):
            for point_id in track:
                point_data = txy_pts[point_id]
                frames.append(point_data[0])
                x_coords.append(point_data[1])
                y_coords.append(point_data[2])
                track_numbers.append(track_idx)
        
        return pd.DataFrame({
            'frame': frames,
            'track_number': track_numbers,
            'x': x_coords,
            'y': y_coords
        })


class TrackingMainWindow(QMainWindow):
    """Main application window for standalone tracking analysis."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.data: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None
        
        # Initialize mock FLIKA environment
        self._setup_mock_environment()
        
        # Set up UI
        self._setup_ui()
        
        self.logger.info("Main window initialized")
    
    def _setup_mock_environment(self) -> None:
        """Set up mock FLIKA environment."""
        import sys
        
        # Create mock module
        mock_flika = type(sys)('flika')
        mock_flika.global_vars = MockGlobalVars()
        mock_flika.__version__ = '0.2.24'
        
        # Add to sys.modules
        sys.modules['flika'] = mock_flika
        sys.modules['flika.global_vars'] = mock_flika.global_vars
        
        # Create mock window
        mock_flika.global_vars.win = MockWindow()
        
        self.logger.debug("Mock FLIKA environment created")
    
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Tracking Data Analysis - Standalone")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        self._create_header(main_layout)
        
        # Content area
        self._create_content_area(main_layout)
        
        # Status area
        self._create_status_area(main_layout)
        
        # Set application icon if available
        self._set_app_icon()
    
    def _create_header(self, parent_layout: QVBoxLayout) -> None:
        """Create header section."""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        # Title
        title_label = QLabel("Tracking Data Analysis")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        
        # File controls
        file_group = QGroupBox("Data File")
        file_layout = QHBoxLayout(file_group)
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("QLabel { color: gray; }")
        
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self._load_data_file)
        
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.load_button)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(file_group)
        
        parent_layout.addWidget(header_widget)
    
    def _create_content_area(self, parent_layout: QVBoxLayout) -> None:
        """Create main content area."""
        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Data view tab
        self._create_data_view_tab()
        
        # Visualization tab
        self._create_visualization_tab()
        
        # Analysis tab
        self._create_analysis_tab()
        
        parent_layout.addWidget(self.tab_widget)
    
    def _create_data_view_tab(self) -> None:
        """Create data viewing tab."""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)
        
        # Data info
        info_layout = QHBoxLayout()
        self.data_info_label = QLabel("No data loaded")
        info_layout.addWidget(self.data_info_label)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        # Data table (placeholder)
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setFont(QFont("Courier", 10))
        layout.addWidget(self.data_display)
        
        self.tab_widget.addTab(data_widget, "Data")
    
    def _create_visualization_tab(self) -> None:
        """Create visualization tab."""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # Controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.plot_points_btn = QPushButton("Plot Points")
        self.plot_tracks_btn = QPushButton("Plot Tracks")
        self.clear_plot_btn = QPushButton("Clear")
        
        self.plot_points_btn.clicked.connect(self._plot_points)
        self.plot_tracks_btn.clicked.connect(self._plot_tracks)
        self.clear_plot_btn.clicked.connect(self._clear_plot)
        
        controls_layout.addWidget(self.plot_points_btn)
        controls_layout.addWidget(self.plot_tracks_btn)
        controls_layout.addWidget(self.clear_plot_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Plot area
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Y Position', units='pixels')
        self.plot_widget.setLabel('bottom', 'X Position', units='pixels')
        self.plot_widget.setTitle('Tracking Data Visualization')
        layout.addWidget(self.plot_widget)
        
        self.tab_widget.addTab(viz_widget, "Visualization")
    
    def _create_analysis_tab(self) -> None:
        """Create analysis tab."""
        analysis_widget = QWidget()
        layout = QVBoxLayout(analysis_widget)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis Tools")
        controls_layout = QHBoxLayout(controls_group)
        
        self.stats_btn = QPushButton("Calculate Statistics")
        self.export_btn = QPushButton("Export Results")
        
        self.stats_btn.clicked.connect(self._calculate_statistics)
        self.export_btn.clicked.connect(self._export_results)
        
        controls_layout.addWidget(self.stats_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Results area
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setFont(QFont("Courier", 10))
        layout.addWidget(self.results_display)
        
        self.tab_widget.addTab(analysis_widget, "Analysis")
    
    def _create_status_area(self, parent_layout: QVBoxLayout) -> None:
        """Create status area."""
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        
        self.status_label = QLabel("Ready")
        self.progress_label = QLabel("")
        
        status_layout.addWidget(QLabel("Status:"))
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_label)
        
        parent_layout.addWidget(status_widget)
    
    def _set_app_icon(self) -> None:
        """Set application icon if available."""
        try:
            icon_path = Path(__file__).parent / "icon.png"
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
        except Exception:
            pass  # No icon available
    
    def _load_data_file(self) -> None:
        """Load a data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Tracking Data",
            "",
            "All Supported (*.csv *.json);;CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if file_path:
            self.current_file = file_path
            file_name = Path(file_path).name
            
            # Determine file type
            if file_path.endswith('.json'):
                filetype = 'json'
            else:
                filetype = 'csv'
            
            # Update UI
            self.file_label.setText(file_name)
            self.file_label.setStyleSheet("QLabel { color: black; }")
            self._update_status("Loading data...")
            
            # Start data loading thread
            self.data_loader = DataLoader(file_path, filetype)
            self.data_loader.dataLoaded.connect(self._on_data_loaded)
            self.data_loader.errorOccurred.connect(self._on_load_error)
            self.data_loader.progressUpdate.connect(self._update_status)
            self.data_loader.start()
    
    def _on_data_loaded(self, data: pd.DataFrame) -> None:
        """Handle successful data loading."""
        self.data = data
        self.logger.info(f"Data loaded: {len(data)} rows, {len(data.columns)} columns")
        
        # Update data info
        info_text = f"Loaded: {len(data)} points"
        if 'track_number' in data.columns:
            n_tracks = data['track_number'].nunique()
            info_text += f", {n_tracks} tracks"
        
        self.data_info_label.setText(info_text)
        
        # Display data preview
        preview = data.head(100).to_string()
        self.data_display.setText(preview)
        
        # Enable controls
        self.plot_points_btn.setEnabled(True)
        self.plot_tracks_btn.setEnabled(True)
        self.stats_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        self._update_status("Data loaded successfully")
    
    def _on_load_error(self, error_message: str) -> None:
        """Handle data loading error."""
        self.logger.error(f"Data loading failed: {error_message}")
        QMessageBox.critical(self, "Error", f"Failed to load data:\n{error_message}")
        self._update_status("Error loading data")
    
    def _plot_points(self) -> None:
        """Plot data points."""
        if self.data is None:
            return
        
        try:
            self.plot_widget.clear()
            
            if 'x' in self.data.columns and 'y' in self.data.columns:
                x = self.data['x'].values
                y = self.data['y'].values
                
                # Create scatter plot
                scatter = pg.ScatterPlotItem(
                    x=x, y=y,
                    size=5,
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(255, 0, 0, 120)
                )
                self.plot_widget.addItem(scatter)
                
                self._update_status(f"Plotted {len(x)} points")
            else:
                QMessageBox.warning(self, "Warning", "No x,y coordinates found in data")
                
        except Exception as e:
            self.logger.error(f"Error plotting points: {e}")
            QMessageBox.critical(self, "Error", f"Failed to plot points:\n{str(e)}")
    
    def _plot_tracks(self) -> None:
        """Plot tracks."""
        if self.data is None:
            return
        
        try:
            self.plot_widget.clear()
            
            if 'track_number' not in self.data.columns:
                QMessageBox.warning(self, "Warning", "No track information found in data")
                return
            
            # Plot each track
            for track_id in self.data['track_number'].unique():
                if pd.isna(track_id):
                    continue
                    
                track_data = self.data[self.data['track_number'] == track_id]
                if len(track_data) < 2:
                    continue
                
                x = track_data['x'].values
                y = track_data['y'].values
                
                # Create line plot for track
                color = pg.intColor(int(track_id))
                self.plot_widget.plot(
                    x, y,
                    pen=pg.mkPen(color, width=2),
                    symbol='o',
                    symbolSize=3,
                    symbolBrush=color
                )
            
            n_tracks = self.data['track_number'].nunique()
            self._update_status(f"Plotted {n_tracks} tracks")
            
        except Exception as e:
            self.logger.error(f"Error plotting tracks: {e}")
            QMessageBox.critical(self, "Error", f"Failed to plot tracks:\n{str(e)}")
    
    def _clear_plot(self) -> None:
        """Clear the plot."""
        self.plot_widget.clear()
        self._update_status("Plot cleared")
    
    def _calculate_statistics(self) -> None:
        """Calculate basic statistics."""
        if self.data is None:
            return
        
        try:
            stats_text = "Data Statistics:\n\n"
            
            # Basic info
            stats_text += f"Total points: {len(self.data)}\n"
            
            if 'track_number' in self.data.columns:
                n_tracks = self.data['track_number'].nunique()
                stats_text += f"Number of tracks: {n_tracks}\n"
                
                # Track length statistics
                track_lengths = self.data.groupby('track_number').size()
                stats_text += f"Mean track length: {track_lengths.mean():.1f} points\n"
                stats_text += f"Min track length: {track_lengths.min()} points\n"
                stats_text += f"Max track length: {track_lengths.max()} points\n\n"
            
            # Position statistics
            if 'x' in self.data.columns and 'y' in self.data.columns:
                stats_text += "Position Statistics:\n"
                stats_text += f"X range: {self.data['x'].min():.1f} - {self.data['x'].max():.1f}\n"
                stats_text += f"Y range: {self.data['y'].min():.1f} - {self.data['y'].max():.1f}\n"
                stats_text += f"Mean X: {self.data['x'].mean():.1f}\n"
                stats_text += f"Mean Y: {self.data['y'].mean():.1f}\n\n"
            
            # Column information
            stats_text += f"Available columns:\n"
            for col in self.data.columns:
                stats_text += f"  - {col}\n"
            
            self.results_display.setText(stats_text)
            self._update_status("Statistics calculated")
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            QMessageBox.critical(self, "Error", f"Failed to calculate statistics:\n{str(e)}")
    
    def _export_results(self) -> None:
        """Export analysis results."""
        if self.data is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "tracking_results.csv",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    self.data.to_json(file_path, orient='records', indent=2)
                else:
                    self.data.to_csv(file_path, index=False)
                
                self._update_status(f"Results exported to {Path(file_path).name}")
                QMessageBox.information(self, "Success", f"Results exported to:\n{file_path}")
                
            except Exception as e:
                self.logger.error(f"Error exporting results: {e}")
                QMessageBox.critical(self, "Error", f"Failed to export results:\n{str(e)}")
    
    def _update_status(self, message: str) -> None:
        """Update status message."""
        self.status_label.setText(message)
        self.logger.debug(f"Status: {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone Tracking Data Analysis Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python standalone_app.py
    python standalone_app.py --debug
    python standalone_app.py --data-file tracking_data.csv
        """
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to data file to load on startup'
    )
    
    return parser.parse_args()


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.debug)
    
    try:
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("Tracking Data Analysis")
        app.setApplicationVersion("1.0.0")
        
        # Create main window
        window = TrackingMainWindow()
        window.show()
        
        # Load data file if specified
        if args.data_file:
            if Path(args.data_file).exists():
                window.current_file = args.data_file
                # Trigger file loading
                QTimer.singleShot(100, lambda: window._load_data_file())
            else:
                logger.error(f"Data file not found: {args.data_file}")
        
        logger.info("Application started")
        
        # Run application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
