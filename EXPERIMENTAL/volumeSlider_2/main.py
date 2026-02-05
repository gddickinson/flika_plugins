#!/usr/bin/env python3
"""
FLIKA Volume Slider Plugin - Professional Edition
================================================

A comprehensive 3D/4D lightsheet microscopy analysis plugin for FLIKA with:
- Professional GUI with menu-selectable panels
- Comprehensive error handling and logging
- Built-in test data generation
- Advanced visualization capabilities
- Modular, extensible architecture

Author: Refactored for professionalism
License: GPL v3
"""

import sys
import os
import numpy as np
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime
import json

# Ensure the plugin directory is in the path for imports
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# Qt imports - FIXED: Use Signal instead of pyqtSignal with qtpy
from qtpy.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QSplitter, QTabWidget, QProgressBar, QStatusBar,
                           QMenuBar, QAction, QMessageBox, QFileDialog,
                           QDockWidget, QTextEdit, QLabel, QPushButton,
                           QSpinBox, QDoubleSpinBox, QSlider, QComboBox,
                           QCheckBox, QGroupBox, QFormLayout, QGridLayout,
                           QApplication, QSizePolicy, QDialog)
from qtpy.QtCore import Qt, QTimer, QThread, QObject, Signal, QSettings  # FIXED: Signal instead of pyqtSignal
from qtpy.QtGui import QIcon, QPixmap

# Scientific computing
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# FLIKA imports
try:
    import flika
    from flika import global_vars as g
    from flika.window import Window
    from flika.utils.BaseProcess import BaseProcess_noPriorWindow
    HAS_FLIKA = True
except ImportError:
    HAS_FLIKA = False
    print("FLIKA not found. Some features may be limited.")

# Local imports - with fallbacks for missing modules
try:
    from .core.data_manager import DataManager
except ImportError:
    print("Warning: DataManager not found, using fallback")
    DataManager = None

try:
    from .core.volume_processor import VolumeProcessor
except ImportError:
    print("Warning: VolumeProcessor not found, using fallback")
    VolumeProcessor = None

try:
    from .core.visualization_engine import VisualizationEngine, RenderMode, ColorMap
except ImportError:
    print("Warning: VisualizationEngine not found, using fallback")
    VisualizationEngine = None
    # Create fallback enums
    from enum import Enum
    class RenderMode(Enum):
        POINTS = "points"
        SURFACE = "surface"
        VOLUME = "volume"
    class ColorMap(Enum):
        VIRIDIS = "viridis"
        JET = "jet"

try:
    from .gui.control_panels import ControlPanelManager
except ImportError:
    print("Warning: ControlPanelManager not found, using fallback")
    ControlPanelManager = None

try:
    from .analysis.algorithms import AnalysisEngine
except ImportError:
    print("Warning: AnalysisEngine not found, using fallback")
    AnalysisEngine = None

try:
    from .io.data_loaders import DataLoaderManager
except ImportError:
    print("Warning: DataLoaderManager not found, using fallback")
    DataLoaderManager = None

try:
    from .synthetic.test_data_generator import TestDataGenerator
except ImportError:
    print("Warning: TestDataGenerator not found, using fallback")
    TestDataGenerator = None

try:
    from .utils.error_handler import ErrorHandler
except ImportError:
    print("Warning: ErrorHandler not found, using fallback")
    ErrorHandler = None

try:
    from .utils.logger_config import setup_logging
except ImportError:
    print("Warning: logger_config not found, using basic logging")
    def setup_logging():
        return logging.getLogger(__name__)


class ControlPanelDialog(QDialog):
    """Base dialog class for control panels."""

    def __init__(self, title, panel_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)  # Non-modal so users can interact with main window
        self.resize(350, 500)

        layout = QVBoxLayout(self)
        layout.addWidget(panel_widget)

        # Make it stay on top but not modal
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)


class VolumeSliderPlugin(QMainWindow):
    """
    Main plugin window providing comprehensive 3D/4D volume analysis.

    Features:
    - Unified interface with menu-selectable control panels
    - Real-time 3D visualization
    - Advanced image processing
    - Test data generation
    - Comprehensive error handling
    """

    # Signals for inter-component communication - FIXED: Use Signal
    data_loaded = Signal(np.ndarray)
    volume_changed = Signal(int)
    processing_progress = Signal(int, str)
    error_occurred = Signal(str, str)  # title, message

    def __init__(self, parent=None):
        super().__init__(parent)

        # Initialize logging first
        try:
            self.logger = setup_logging()
        except:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Volume Slider Plugin")

        # Core components - with fallbacks
        self.data_manager = None
        self.volume_processor = None
        self.visualization_engine = None
        self.control_panels = None
        self.analysis_engine = None
        self.data_loader = None
        self.test_data_generator = None
        self.error_handler = None

        # Control panel dialogs - NEW
        self.volume_dialog = None
        self.processing_dialog = None
        self.analysis_dialog = None
        self.visualization_dialog = None
        self.log_dialog = None

        # Settings
        self.settings = QSettings('FLIKA', 'VolumeSlider')

        # UI state
        self.current_volume = 0
        self.is_playing = False
        self.play_timer = QTimer()

        try:
            self._initialize_components()
            self._setup_ui()
            self._connect_signals()
            self._restore_settings()
            self.logger.info("Plugin initialization completed successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize plugin: {str(e)}")
            self._show_critical_error("Initialization Error",
                                    f"Failed to initialize Volume Slider Plugin:\n{str(e)}")

    def _initialize_components(self):
        """Initialize core components with better error handling."""
        try:
            # Initialize error handler first
            if ErrorHandler:
                self.error_handler = ErrorHandler(self.logger)

            # Initialize data manager
            if DataManager:
                self.data_manager = DataManager()
                self.logger.info("DataManager initialized successfully")
            else:
                self.logger.warning("DataManager not available")

            # Initialize volume processor
            if VolumeProcessor:
                self.volume_processor = VolumeProcessor()
                self.logger.info("VolumeProcessor initialized successfully")

            # Initialize visualization engine AFTER creating the GL widget
            # This will be done in _create_main_layout
            self.visualization_engine = None

            # Initialize other components
            if ControlPanelManager:
                self.control_panels = ControlPanelManager(self)

            if AnalysisEngine:
                self.analysis_engine = AnalysisEngine()

            if DataLoaderManager:
                self.data_loader = DataLoaderManager()

            if TestDataGenerator:
                self.test_data_generator = TestDataGenerator()

        except Exception as e:
            error_msg = f"Component initialization failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _setup_ui(self):
        """Setup the main user interface."""
        self.setWindowTitle("FLIKA Volume Slider - Professional Edition")
        self.setMinimumSize(1200, 800)

        # Create menu bar
        self._create_menu_bar()

        # Create status bar with progress
        self._create_status_bar()

        # Create central widget with splitter layout
        self._create_main_layout()

        # Apply stylesheet for professional appearance
        self._apply_styling()

    def _create_menu_bar(self):
        """Create comprehensive menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('&File')

        # Load data actions
        load_group = file_menu.addMenu('&Load Data')
        self._add_action(load_group, '&TIFF Stack...', self.load_tiff_data, 'Ctrl+O')
        self._add_action(load_group, '&NumPy Array...', self.load_numpy_data, 'Ctrl+N')
        if HAS_FLIKA:
            self._add_action(load_group, 'From &FLIKA Window', self.load_from_flika, 'Ctrl+F')

        file_menu.addSeparator()

        # Save data actions
        save_group = file_menu.addMenu('&Save Data')
        self._add_action(save_group, 'Save &Processed Data...', self.save_processed_data, 'Ctrl+S')
        self._add_action(save_group, 'Export &Analysis Results...', self.export_analysis, 'Ctrl+E')

        file_menu.addSeparator()
        self._add_action(file_menu, '&Quit', self.close, 'Ctrl+Q')

        # Data menu
        data_menu = menubar.addMenu('&Data')
        if self.test_data_generator:
            self._add_action(data_menu, 'Generate &Test Data...', self.show_test_data_dialog, 'Ctrl+T')
        self._add_action(data_menu, '&Batch Process...', self.show_batch_dialog, 'Ctrl+B')
        data_menu.addSeparator()
        self._add_action(data_menu, 'Data &Properties...', self.show_data_properties, 'Ctrl+P')

        # Analysis menu
        analysis_menu = menubar.addMenu('&Analysis')
        self._add_action(analysis_menu, '&Motion Correction', self.run_motion_correction)
        self._add_action(analysis_menu, '&Deconvolution', self.run_deconvolution)
        self._add_action(analysis_menu, 'Background &Subtraction', self.run_background_subtraction)
        analysis_menu.addSeparator()
        self._add_action(analysis_menu, 'Calcium &Event Detection', self.run_event_detection)
        self._add_action(analysis_menu, 'ROI &Analysis', self.run_roi_analysis)

        # Visualization menu
        viz_menu = menubar.addMenu('&Visualization')
        self._add_action(viz_menu, 'Reset &Views', self.reset_views, 'Ctrl+R')
        self._add_action(viz_menu, '&Auto Scale', self.auto_scale, 'Ctrl+A')
        self._add_action(viz_menu, 'Toggle &3D View', self.toggle_3d_view, 'F3')

        # NEW: Panels menu for control panels
        panels_menu = menubar.addMenu('&Panels')
        self._add_action(panels_menu, '&Volume Control', self.show_volume_panel, 'Ctrl+1')
        self._add_action(panels_menu, '&Processing', self.show_processing_panel, 'Ctrl+2')
        self._add_action(panels_menu, '&Analysis', self.show_analysis_panel, 'Ctrl+3')
        self._add_action(panels_menu, 'V&isualization', self.show_visualization_panel, 'Ctrl+4')
        panels_menu.addSeparator()
        self._add_action(panels_menu, '&Log Window', self.show_log_panel, 'Ctrl+L')
        panels_menu.addSeparator()
        self._add_action(panels_menu, '&Close All Panels', self.close_all_panels, 'Ctrl+Shift+W')

        # Help menu
        help_menu = menubar.addMenu('&Help')
        self._add_action(help_menu, '&Documentation', self.show_documentation, 'F1')
        self._add_action(help_menu, '&About...', self.show_about, 'Ctrl+H')

    def _add_action(self, menu, text, callback, shortcut=None):
        """Helper to add menu actions."""
        action = menu.addAction(text)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    def _create_status_bar(self):
        """Create status bar with progress indicator."""
        self.status_bar = self.statusBar()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(300)

        # Status labels
        self.status_label = QLabel("Ready")
        self.data_info_label = QLabel("No data loaded")

        # Add to status bar
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addPermanentWidget(self.data_info_label)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def _create_main_layout(self):
        """Create main splitter layout for visualization panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)

        # Left side: 2D views in vertical splitter
        left_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(left_splitter)

        # Create image view widgets
        self.xy_view = self._create_image_view("XY View (Top)")
        self.xz_view = self._create_image_view("XZ View (Side)")
        self.yz_view = self._create_image_view("YZ View (Front)")

        left_splitter.addWidget(self.xy_view)
        left_splitter.addWidget(self.xz_view)
        left_splitter.addWidget(self.yz_view)

        # Right side: 3D view - CREATE AND INITIALIZE PROPERLY
        try:
            self.view_3d = gl.GLViewWidget()
            self.view_3d.setCameraPosition(distance=200, elevation=30, azimuth=45)
            self.view_3d.setBackgroundColor('black')

            # Initialize visualization engine with the GL widget
            if VisualizationEngine:
                self.visualization_engine = VisualizationEngine(self.view_3d)
                self.logger.info("VisualizationEngine initialized successfully")
            else:
                self.logger.warning("VisualizationEngine not available, using fallback")

            main_splitter.addWidget(self.view_3d)

        except Exception as e:
            self.logger.error(f"Failed to create 3D view: {str(e)}")
            # Create a fallback widget
            fallback_widget = QWidget()
            fallback_layout = QVBoxLayout(fallback_widget)
            fallback_layout.addWidget(QLabel("3D View Not Available\nCheck OpenGL support"))
            main_splitter.addWidget(fallback_widget)
            self.view_3d = None
            self.visualization_engine = None

        # Set splitter proportions
        main_splitter.setSizes([400, 600])
        left_splitter.setSizes([150, 150, 150])

    def _create_image_view(self, title):
        """Create a configured ImageView widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Title label
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(title_label)

        # Image view
        image_view = pg.ImageView()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        layout.addWidget(image_view)

        # Store reference to image view
        widget.image_view = image_view

        return widget

    # NEW: Panel management methods
    def show_volume_panel(self):
        """Show/create volume control panel."""
        if self.volume_dialog is None and self.control_panels:
            panel_widget = self.control_panels.create_volume_panel()
            self.volume_dialog = ControlPanelDialog("Volume Control", panel_widget, self)

        if self.volume_dialog:
            self.volume_dialog.show()
            self.volume_dialog.raise_()
            self.volume_dialog.activateWindow()

    def show_processing_panel(self):
        """Show/create processing control panel."""
        if self.processing_dialog is None and self.control_panels:
            panel_widget = self.control_panels.create_processing_panel()
            self.processing_dialog = ControlPanelDialog("Processing Controls", panel_widget, self)

        if self.processing_dialog:
            self.processing_dialog.show()
            self.processing_dialog.raise_()
            self.processing_dialog.activateWindow()

    def show_analysis_panel(self):
        """Show/create analysis control panel."""
        if self.analysis_dialog is None and self.control_panels:
            panel_widget = self.control_panels.create_analysis_panel()
            self.analysis_dialog = ControlPanelDialog("Analysis Controls", panel_widget, self)

        if self.analysis_dialog:
            self.analysis_dialog.show()
            self.analysis_dialog.raise_()
            self.analysis_dialog.activateWindow()

    def show_visualization_panel(self):
        """Show/create visualization control panel."""
        if self.visualization_dialog is None and self.control_panels:
            panel_widget = self.control_panels.create_visualization_panel()
            self.visualization_dialog = ControlPanelDialog("Visualization Controls", panel_widget, self)

        if self.visualization_dialog:
            self.visualization_dialog.show()
            self.visualization_dialog.raise_()
            self.visualization_dialog.activateWindow()

    def show_log_panel(self):
        """Show/create log panel."""
        if self.log_dialog is None:
            log_widget = self._create_log_panel()
            self.log_dialog = ControlPanelDialog("Log Window", log_widget, self)

        if self.log_dialog:
            self.log_dialog.show()
            self.log_dialog.raise_()
            self.log_dialog.activateWindow()

    def close_all_panels(self):
        """Close all open control panels."""
        panels = [self.volume_dialog, self.processing_dialog, self.analysis_dialog,
                 self.visualization_dialog, self.log_dialog]

        for panel in panels:
            if panel and panel.isVisible():
                panel.close()

        self.status_label.setText("All control panels closed")

    def _create_log_panel(self):
        """Create logging panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # Log controls
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        save_log_btn = QPushButton("Save Log...")
        save_log_btn.clicked.connect(self.save_log)

        controls.addWidget(clear_btn)
        controls.addWidget(save_log_btn)
        controls.addStretch()

        layout.addLayout(controls)

        return widget

    def _apply_styling(self):
        """Apply professional styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QDialog {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px 0px;
                padding: 10px 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QPushButton:pressed {
                background-color: #d4d4d4;
            }
        """)

    def _connect_signals(self):
        """Connect inter-component signals."""

        self.data_loaded.connect(self._on_data_loaded)
        self.volume_changed.connect(self._on_volume_changed)
        self.processing_progress.connect(self._on_progress_update)
        self.error_occurred.connect(self._on_error_occurred)

        # Timer for playback
        self.play_timer.timeout.connect(self._advance_volume)

        # Component signals
        if self.data_manager:
            self.data_manager.data_changed.connect(self._update_displays)

        # Control panel signals
        if self.control_panels:
            self.control_panels.visualization_changed.connect(self._on_visualization_changed)

        # Visualization engine signals
        if self.visualization_engine:
            self.visualization_engine.rendering_failed.connect(
                lambda msg: self.logger.error(f"3D rendering failed: {msg}")
            )
            self.visualization_engine.rendering_completed.connect(
                lambda: self.logger.debug("3D rendering completed")
            )

    def _restore_settings(self):
        """Restore application settings."""
        try:
            # Window geometry
            geometry = self.settings.value('geometry')
            if geometry:
                self.restoreGeometry(geometry)

            self.logger.info("Settings restored successfully")

        except Exception as e:
            self.logger.warning(f"Could not restore settings: {str(e)}")

    def _save_settings(self):
        """Save application settings."""
        try:
            self.settings.setValue('geometry', self.saveGeometry())
            self.logger.info("Settings saved successfully")

        except Exception as e:
            self.logger.warning(f"Could not save settings: {str(e)}")

    # Slot implementations
    def _on_data_loaded(self, data):
        """Handle data loading completion."""
        try:
            self.data_info_label.setText(
                f"Data: {data.shape} | {data.dtype} | "
                f"{data.nbytes / 1024**2:.1f} MB"
            )
            self.status_label.setText("Data loaded successfully")
            self._update_displays()

        except Exception as e:
            self.logger.error(f"Error handling loaded data: {str(e)}")

    def _on_volume_changed(self, volume_idx):
        """Handle volume change."""
        self.current_volume = volume_idx
        self._update_displays()

    def _on_visualization_changed(self, setting_name, value):
        """Handle visualization parameter changes from control panel."""
        try:
            if self.visualization_engine and hasattr(self.visualization_engine, 'current_params'):
                params = self.visualization_engine.current_params

                if setting_name == 'threshold':
                    params.threshold = float(value)
                elif setting_name == 'render_mode':
                    if hasattr(RenderMode, value.upper()):
                        params.render_mode = getattr(RenderMode, value.upper())
                elif setting_name == 'point_size':
                    params.point_size = float(value)
                elif setting_name == 'show_axes':
                    params.show_axes = bool(value)
                elif setting_name == 'reset_view':
                    if self.visualization_engine:
                        self.visualization_engine.reset_view()
                    return

                # Update visualization
                self.visualization_engine.set_parameters(params, update_display=True)

            else:
                self.logger.debug(f"Visualization parameter change ignored: {setting_name}={value}")

        except Exception as e:
            self.logger.error(f"Error handling visualization change: {str(e)}")

    def _on_progress_update(self, value, message):
        """Update progress bar."""
        if value < 0:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(value)

        if message:
            self.status_label.setText(message)

    def _on_error_occurred(self, title, message):
        """Handle error display."""
        QMessageBox.critical(self, title, message)
        self.logger.error(f"{title}: {message}")

    # Menu action implementations
    def load_tiff_data(self):
        """Load TIFF data file."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load TIFF Stack", "",
                "TIFF Files (*.tif *.tiff);;All Files (*)"
            )
            if filename and self.data_loader:
                self._load_data_async(filename, 'tiff')
            elif filename:
                QMessageBox.information(self, "Info", "Data loader not available - feature disabled")

        except Exception as e:
            self._handle_error("Load Error", f"Failed to load TIFF file: {str(e)}")

    def load_numpy_data(self):
        """Load NumPy data file."""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load NumPy Array", "",
                "NumPy Files (*.npy);;All Files (*)"
            )
            if filename and self.data_loader:
                self._load_data_async(filename, 'numpy')
            elif filename:
                QMessageBox.information(self, "Info", "Data loader not available - feature disabled")

        except Exception as e:
            self._handle_error("Load Error", f"Failed to load NumPy file: {str(e)}")

    def load_from_flika(self):
        """Load data from current FLIKA window."""
        try:
            if not HAS_FLIKA:
                raise RuntimeError("FLIKA not available")

            if not hasattr(g, 'win') or g.win is None:
                raise RuntimeError("No FLIKA window open")

            data = np.array(g.win.image)
            if self.data_manager:
                self.data_manager.set_data(data)
                self.data_loaded.emit(data)
            else:
                QMessageBox.information(self, "Info", "Data manager not available")

        except Exception as e:
            self._handle_error("FLIKA Load Error", str(e))

    def save_processed_data(self):
        """Save processed data."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                QMessageBox.information(self, "Save Data", "No data available to save")
                return

            filename, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Processed Data", "",
                "NumPy Array (*.npy);;TIFF Stack (*.tif);;All Files (*)"
            )

            if filename:
                try:
                    self.data_manager.save_processed_data(filename)
                    QMessageBox.information(self, "Save Complete",
                                          f"Data saved to:\n{filename}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error",
                                       f"Failed to save data:\n{str(e)}")

        except Exception as e:
            self._handle_error("Save Error", str(e))

    def export_analysis(self):
        """Export analysis results."""
        QMessageBox.information(self, "Info", "Export functionality not yet implemented")

    def show_test_data_dialog(self):
        """Show test data generation dialog."""
        try:
            if not self.test_data_generator:
                QMessageBox.warning(self, "Test Data Generator",
                                  "Test data generator not available")
                return

            # Import and create the dialog
            try:
                from .gui.dialogs.test_data_dialog import TestDataDialog
                dialog = TestDataDialog(self.test_data_generator, self)

                if dialog.exec_() == dialog.Accepted:
                    params = dialog.get_parameters()
                    self._generate_test_data(params)

            except ImportError as e:
                QMessageBox.warning(self, "Dialog Error",
                                  f"Could not load test data dialog: {str(e)}")

        except Exception as e:
            self._handle_error("Test Data Error", str(e))

    def _generate_test_data(self, params):
        """Generate synthetic test data."""
        try:
            if not self.test_data_generator:
                QMessageBox.warning(self, "Error", "Test data generator not available")
                return

            self.processing_progress.emit(0, "Generating test data...")
            self.status_label.setText("Generating test data...")

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Generate data synchronously (for now - could be threaded later)
            try:
                # Connect progress signals if available
                if hasattr(self.test_data_generator, 'progress_updated'):
                    self.test_data_generator.progress_updated.connect(self._on_generation_progress)

                # Generate the data
                data, metadata = self.test_data_generator.generate_data(params)

                # Load into data manager
                if self.data_manager and data is not None:
                    success = self.data_manager.set_data(data, metadata)
                    if success:
                        self.data_loaded.emit(data)
                        self.processing_progress.emit(100, "Test data generated successfully")

                        # Update UI
                        self.status_label.setText("Test data generated successfully")

                        # Show generation summary
                        self.show_generation_summary(data, metadata)
                    else:
                        raise RuntimeError("Failed to load data into data manager")
                else:
                    raise RuntimeError("Data manager not available or data generation failed")

            except Exception as gen_error:
                raise RuntimeError(f"Data generation failed: {str(gen_error)}")

        except Exception as e:
            error_msg = f"Test data generation error: {str(e)}"
            self.logger.error(error_msg)
            self.processing_progress.emit(-1, "Test data generation failed")
            self.status_label.setText("Test data generation failed")
            QMessageBox.critical(self, "Generation Error", error_msg)

        finally:
            # Hide progress bar
            self.progress_bar.setVisible(False)

    def _on_generation_progress(self, percent, message):
        """Handle test data generation progress updates."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

        # Process events to keep UI responsive
        QApplication.processEvents()

    def show_generation_summary(self, data, metadata):
        """Show a summary of the generated test data."""
        try:
            pattern_type = metadata.get('pattern_type', 'unknown')
            params = metadata.get('parameters', {})

            summary_text = f"Generated Test Data Summary\n" + "="*40 + "\n\n"
            summary_text += f"Pattern Type: {pattern_type}\n"
            summary_text += f"Data Shape: {data.shape}\n"
            summary_text += f"Data Type: {data.dtype}\n"
            summary_text += f"Memory Size: {data.nbytes / 1024**2:.1f} MB\n\n"

            summary_text += "Parameters Used:\n" + "-"*20 + "\n"
            for key, value in params.items():
                if isinstance(value, (list, tuple)):
                    summary_text += f"{key}: {value}\n"
                elif isinstance(value, float):
                    summary_text += f"{key}: {value:.3f}\n"
                else:
                    summary_text += f"{key}: {value}\n"

            # Create info dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Test Data Generated")
            msg.setText(f"Successfully generated {pattern_type} test data!")
            msg.setDetailedText(summary_text)
            msg.setIcon(QMessageBox.Information)
            msg.exec_()

        except Exception as e:
            self.logger.error(f"Error showing generation summary: {str(e)}")

    def show_batch_dialog(self):
        """Show batch processing dialog."""
        QMessageBox.information(self, "Info", "Batch processing dialog not yet implemented")

    def show_data_properties(self):
        """Show data properties dialog."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                QMessageBox.information(self, "Data Properties", "No data loaded")
                return

            # Get data statistics
            stats = self.data_manager.data_statistics
            metadata = self.data_manager.metadata
            memory_usage = self.data_manager.get_memory_usage()

            # Create properties text
            properties_text = "Data Properties\n" + "="*40 + "\n\n"

            # Basic info
            shape = self.data_manager.get_data_shape()
            dtype = self.data_manager.get_data_dtype()
            properties_text += f"Shape: {shape}\n"
            properties_text += f"Data Type: {dtype}\n"
            properties_text += f"Total Memory: {memory_usage['total']:.1f} MB\n\n"

            # Statistics
            if stats:
                properties_text += "Statistics:\n" + "-"*20 + "\n"
                properties_text += f"Min: {stats.get('min', 'N/A'):.6f}\n"
                properties_text += f"Max: {stats.get('max', 'N/A'):.6f}\n"
                properties_text += f"Mean: {stats.get('mean', 'N/A'):.6f}\n"
                properties_text += f"Std: {stats.get('std', 'N/A'):.6f}\n"
                properties_text += f"Median: {stats.get('median', 'N/A'):.6f}\n\n"

            # Metadata
            if metadata:
                properties_text += "Metadata:\n" + "-"*20 + "\n"
                for key, value in metadata.items():
                    if key not in ['data_statistics']:  # Skip redundant stats
                        properties_text += f"{key}: {value}\n"

            # Create dialog
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Data Properties")
            dialog.setText("Current Data Properties:")
            dialog.setDetailedText(properties_text)
            dialog.setStandardButtons(QMessageBox.Ok)
            dialog.exec_()

        except Exception as e:
            self._handle_error("Data Properties Error", str(e))

    def show_documentation(self):
        """Show documentation."""
        QMessageBox.information(self, "Documentation", "Documentation not yet implemented")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About Volume Slider",
                         "Volume Slider Plugin v2.0\n"
                         "Professional 3D/4D lightsheet microscopy analysis\n\n"
                         "Built for FLIKA\n\n"
                         "Control panels available via Panels menu")

    def save_log(self):
        """Save log to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Log", "", "Text Files (*.txt);;All Files (*)"
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.status_label.setText("Log saved successfully")
        except Exception as e:
            self._handle_error("Save Log Error", str(e))

    def _load_data_async(self, filename, file_type):
        """Load data asynchronously."""
        try:
            if not self.data_loader:
                QMessageBox.warning(self, "Data Loader", "Data loader not available")
                return

            self.processing_progress.emit(0, f"Loading {file_type} file...")
            self.progress_bar.setVisible(True)

            # For now, load synchronously (could be threaded later)
            try:
                loaded_data = self.data_loader.load_data(filename)

                if loaded_data and loaded_data.data is not None:
                    # Load into data manager
                    success = self.data_manager.set_data(loaded_data.data, loaded_data.metadata)
                    if success:
                        self.data_loaded.emit(loaded_data.data)
                        self.processing_progress.emit(100, f"File loaded successfully")

                        info_msg = (f"Loaded {file_type.upper()} file:\n"
                                   f"Shape: {loaded_data.data.shape}\n"
                                   f"Data type: {loaded_data.data.dtype}\n"
                                   f"Size: {loaded_data.size_mb:.1f} MB")

                        QMessageBox.information(self, "File Loaded", info_msg)
                    else:
                        raise RuntimeError("Failed to load data into data manager")
                else:
                    raise RuntimeError("Failed to load file")

            except Exception as load_error:
                raise RuntimeError(f"File loading failed: {str(load_error)}")

        except Exception as e:
            error_msg = f"Error loading {filename}: {str(e)}"
            self.logger.error(error_msg)
            self.processing_progress.emit(-1, "File loading failed")
            QMessageBox.critical(self, "Loading Error", error_msg)

        finally:
            self.progress_bar.setVisible(False)

    def _update_displays(self):
        """Update all visualization displays with better error handling."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                return

            current_data = self.data_manager.get_current_volume_data(self.current_volume)
            if current_data is None:
                return

            self.logger.debug(f"Updating displays with data shape: {current_data.shape}")

            # Update 2D views with projections
            self._update_2d_views(current_data)

            # Update 3D view
            self._update_3d_view(current_data)

            # Update control panels if they exist
            if self.control_panels:
                try:
                    max_volume = self.data_manager.get_volume_count() - 1
                    self.control_panels.update_volume_range(max_volume)

                    shape = self.data_manager.get_data_shape()
                    dtype = str(self.data_manager.get_data_dtype())
                    memory_mb = self.data_manager.get_memory_usage()['total']
                    self.control_panels.set_data_info(shape, dtype, memory_mb)

                except Exception as e:
                    self.logger.error(f"Error updating control panels: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error updating displays: {str(e)}")

    def _update_2d_views(self, data):
        """Update 2D projection views."""
        try:
            # XY view - max projection along Z
            xy_proj = self.data_manager.get_projection(0, 'max', self.current_volume)
            if xy_proj is not None:
                self.xy_view.image_view.setImage(xy_proj, autoRange=False, autoLevels=False)

            # XZ view - max projection along Y
            xz_proj = self.data_manager.get_projection(1, 'max', self.current_volume)
            if xz_proj is not None:
                self.xz_view.image_view.setImage(xz_proj, autoRange=False, autoLevels=False)

            # YZ view - max projection along X
            yz_proj = self.data_manager.get_projection(2, 'max', self.current_volume)
            if yz_proj is not None:
                self.yz_view.image_view.setImage(yz_proj, autoRange=False, autoLevels=False)

            # Auto-range on first load
            if self.current_volume == 0:
                self.xy_view.image_view.autoRange()
                self.xz_view.image_view.autoRange()
                self.yz_view.image_view.autoRange()

                self.xy_view.image_view.autoLevels()
                self.xz_view.image_view.autoLevels()
                self.yz_view.image_view.autoLevels()

        except Exception as e:
            self.logger.error(f"Error updating 2D views: {str(e)}")

    def _update_3d_view(self, data):
        """Update 3D visualization."""
        try:
            if self.visualization_engine and self.view_3d:
                self.logger.debug("Updating 3D view using VisualizationEngine")
                self.visualization_engine.set_data(data, update_display=True)
            else:
                self.logger.debug("Using fallback 3D visualization")
                self._update_3d_view_simple(data)

        except Exception as e:
            self.logger.error(f"Error updating 3D view: {str(e)}")
            # Try fallback
            try:
                self._update_3d_view_simple(data)
            except Exception as fallback_error:
                self.logger.error(f"Fallback 3D view also failed: {str(fallback_error)}")

    def _update_3d_view_simple(self, data):
        """Simple 3D visualization fallback - IMPROVED VERSION."""
        try:
            if not hasattr(self, 'view_3d') or self.view_3d is None:
                return

            # Clear existing items
            self.view_3d.clear()

            # Check data
            if data is None or data.size == 0:
                self.logger.warning("No data for 3D visualization")
                return

            self.logger.debug(f"Simple 3D view: data shape {data.shape}, dtype {data.dtype}")

            # Use adaptive thresholding
            data_flat = data.flatten()
            data_nonzero = data_flat[data_flat > 0]

            if len(data_nonzero) == 0:
                self.logger.warning("No non-zero data for 3D visualization")
                return

            # Use 90th percentile as threshold for better visualization
            threshold = np.percentile(data_nonzero, 90)
            coords = np.where(data > threshold)

            if len(coords[0]) == 0:
                # Try lower threshold
                threshold = np.percentile(data_nonzero, 70)
                coords = np.where(data > threshold)

            if len(coords[0]) == 0:
                self.logger.warning("No points above threshold for 3D visualization")
                return

            # Limit points for performance
            max_points = 20000
            if len(coords[0]) > max_points:
                indices = np.random.choice(len(coords[0]), max_points, replace=False)
                coords = tuple(coord[indices] for coord in coords)

            # Create position array and center it
            pos = np.column_stack(coords).astype(np.float32)
            center = np.mean(pos, axis=0)
            pos -= center

            # Create colors based on intensity
            intensities = data[coords]
            normalized_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

            # Simple colormap (hot colormap approximation)
            colors = np.zeros((len(pos), 4))
            colors[:, 0] = normalized_intensities  # Red
            colors[:, 1] = np.maximum(0, 2 * normalized_intensities - 1)  # Green
            colors[:, 2] = np.maximum(0, 3 * normalized_intensities - 2)  # Blue
            colors[:, 3] = 0.8  # Alpha

            # Create scatter plot
            scatter = gl.GLScatterPlotItem(
                pos=pos,
                color=colors,
                size=3,
                pxMode=False
            )

            self.view_3d.addItem(scatter)

            self.logger.debug(f"Simple 3D view updated with {len(pos)} points")

        except Exception as e:
            self.logger.error(f"Error in simple 3D view update: {str(e)}")

    def _advance_volume(self):
        """Advance to next volume during playback."""
        if self.data_manager and self.data_manager.has_data:
            max_volume = self.data_manager.get_volume_count() - 1
            if max_volume > 0:
                self.current_volume = (self.current_volume + 1) % (max_volume + 1)
                self.volume_changed.emit(self.current_volume)

                # Update control panels if available
                if self.control_panels and hasattr(self.control_panels, 'volume_panel'):
                    self.control_panels.volume_panel.update_volume_display(self.current_volume)

    def _handle_error(self, title, message):
        """Centralized error handling."""
        if self.error_handler:
            self.error_handler.handle_error(title, message)
        self.error_occurred.emit(title, message)

    def _show_critical_error(self, title, message):
        """Show critical error dialog."""
        QMessageBox.critical(self, title, message)

    # Additional menu implementations
    def run_motion_correction(self):
        """Run motion correction on current data."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                QMessageBox.information(self, "Motion Correction", "No data loaded")
                return

            if not self.volume_processor:
                QMessageBox.warning(self, "Motion Correction", "Volume processor not available")
                return

            # Simple motion correction with default parameters
            reply = QMessageBox.question(self, "Motion Correction",
                                       "Apply motion correction with default settings?",
                                       QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                QMessageBox.information(self, "Motion Correction",
                                      "Motion correction not yet implemented")

        except Exception as e:
            self._handle_error("Motion Correction Error", str(e))

    def run_deconvolution(self):
        """Run deconvolution."""
        QMessageBox.information(self, "Deconvolution", "Deconvolution not yet implemented")

    def run_background_subtraction(self):
        """Run background subtraction."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                QMessageBox.information(self, "Background Subtraction", "No data loaded")
                return

            QMessageBox.information(self, "Background Subtraction",
                                  "Background subtraction not yet implemented")

        except Exception as e:
            self._handle_error("Background Subtraction Error", str(e))

    def run_event_detection(self):
        """Run calcium event detection."""
        try:
            if not self.data_manager or not self.data_manager.has_data:
                QMessageBox.information(self, "Event Detection", "No data loaded")
                return

            if not self.analysis_engine:
                QMessageBox.warning(self, "Event Detection", "Analysis engine not available")
                return

            QMessageBox.information(self, "Event Detection",
                                  "Event detection not yet implemented")

        except Exception as e:
            self._handle_error("Event Detection Error", str(e))

    def run_roi_analysis(self):
        """Run ROI analysis."""
        QMessageBox.information(self, "ROI Analysis", "ROI analysis not yet implemented")

    # View control methods
    def auto_scale(self):
        """Auto-scale all views."""
        try:
            if hasattr(self, 'xy_view'):
                self.xy_view.image_view.autoRange()
                self.xy_view.image_view.autoLevels()

            if hasattr(self, 'xz_view'):
                self.xz_view.image_view.autoRange()
                self.xz_view.image_view.autoLevels()

            if hasattr(self, 'yz_view'):
                self.yz_view.image_view.autoRange()
                self.yz_view.image_view.autoLevels()

            self.status_label.setText("Views auto-scaled")

        except Exception as e:
            self.logger.error(f"Auto-scale error: {str(e)}")

    def reset_views(self):
        """Reset all visualization views."""
        try:
            # Reset 3D view
            if hasattr(self, 'view_3d'):
                self.view_3d.setCameraPosition(distance=200)
                self.view_3d.clear()

            # Reset and update 2D views
            if self.data_manager and self.data_manager.has_data:
                self._update_displays()

            # Auto-scale the views
            self.auto_scale()

            self.status_label.setText("Views reset")

        except Exception as e:
            self.logger.error(f"Reset views error: {str(e)}")

    def toggle_3d_view(self):
        """Toggle 3D view visibility."""
        try:
            if hasattr(self, 'view_3d'):
                visible = self.view_3d.isVisible()
                self.view_3d.setVisible(not visible)

                status = "hidden" if visible else "shown"
                self.status_label.setText(f"3D view {status}")

        except Exception as e:
            self.logger.error(f"Toggle 3D view error: {str(e)}")

    def get_available_test_patterns(self):
        """Get list of available test data patterns."""
        if self.test_data_generator:
            try:
                return self.test_data_generator.get_available_patterns()
            except:
                pass

        # Fallback list
        return ["calcium_puffs", "calcium_waves", "random_blobs", "lightsheet_beads"]

    def closeEvent(self, event):
        """Handle application closure."""
        try:
            self._save_settings()

            # Stop any running timers
            self.play_timer.stop()

            # Close all panel dialogs
            self.close_all_panels()

            self.logger.info("Volume Slider Plugin closed")
            event.accept()

        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")
            event.accept()


# FLIKA Plugin Interface
if HAS_FLIKA:
    class VolumeSliderFLIKAPlugin(BaseProcess_noPriorWindow):
        """
        FLIKA plugin wrapper for the Volume Slider.
        """

        def __init__(self):
            super().__init__()
            self.plugin_window = None

        def __call__(self, **kwargs):
            """Launch the plugin."""
            try:
                if self.plugin_window is None:
                    self.plugin_window = VolumeSliderPlugin()

                self.plugin_window.show()
                self.plugin_window.raise_()

            except Exception as e:
                g.m.statusBar().showMessage(f"Error launching Volume Slider: {str(e)}")

        def gui(self):
            """FLIKA GUI interface."""
            # Create simple GUI for FLIKA's process dialog
            from qtpy.QtWidgets import QLabel
            self.items = []
            self.items.append({
                'name': 'info',
                'string': 'Launch Volume Slider Plugin',
                'object': QLabel("Click OK to launch the Volume Slider interface")
            })
            super().gui()

    # Global instance for FLIKA
    volume_slider_plugin = VolumeSliderFLIKAPlugin()


if __name__ == "__main__":
    # Standalone testing
    app = QApplication(sys.argv)

    # Create test window
    window = VolumeSliderPlugin()
    window.show()

    sys.exit(app.exec_())
