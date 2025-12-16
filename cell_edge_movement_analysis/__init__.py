#!/usr/bin/env python3
"""
Cell Edge Movement Analysis - FLIKA Plugin

Analyzes the relationship between cell edge movement and PIEZO1 protein
intensity using vertical displacement tracking at every x-pixel position.

Author: George Dickinson
Version: 1.1.0
"""

import os
import numpy as np
import pandas as pd
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Signal
import pyqtgraph as pg
from skimage import measure, draw
from scipy import stats, interpolate
import json
import datetime
from pathlib import Path
import tifffile

# FLIKA imports
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.roi import makeROI, ROI_rectangle
from flika import start_flika
import flika
from flika.utils.misc import open_file_gui, save_file_gui

# Import WindowSelector based on FLIKA version
from distutils.version import StrictVersion
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import WindowSelector
else:
    from flika.utils.BaseProcess import WindowSelector

# For plotting
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be limited.")

__version__ = '1.1.0'

# =============================================================================
# ANALYSIS WORKER (for background processing)
# =============================================================================

class AnalysisWorker(QtCore.QObject):
    """Worker thread for running analysis in background."""

    progress = Signal(int, str)  # (percentage, message)
    finished = Signal(dict)  # results
    error = Signal(str)  # error message

    def __init__(self, piezo1_window, mask_window, config, output_config, viz_config):
        super().__init__()
        self.piezo1_window = piezo1_window
        self.mask_window = mask_window
        self.config = config
        self.output_config = output_config
        self.viz_config = viz_config
        self._is_running = True

    def run(self):
        """Run the analysis."""
        try:
            from .analysis_core import run_analysis

            results = run_analysis(
                self.piezo1_window,
                self.mask_window,
                self.config,
                self.output_config,
                self.viz_config,
                progress_callback=self.progress.emit
            )

            if self._is_running:
                self.finished.emit(results)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def stop(self):
        """Stop the analysis."""
        self._is_running = False


# =============================================================================
# MAIN PLUGIN CLASS
# =============================================================================

class CellEdgeMovementAnalysis(QtWidgets.QWidget):
    """Main plugin window for cell edge movement analysis."""

    def __init__(self):
        super().__init__()

        # Initialize state
        self.piezo1_window = None
        self.mask_window = None
        self.analysis_results = None
        self.current_frame_results = None
        self.overlay_rois = []

        # Visualization windows
        self.viz_windows = {}

        # Analysis thread
        self.analysis_thread = None
        self.analysis_worker = None

        # Default configuration
        self.config = {
            'n_points': 12,
            'depth': 200,
            'width': 75,
            'min_cell_coverage': 0.8,
            'try_rotation': True,
            'movement_threshold': 0.1,
            'min_movement_pixels': 5,
            'exclude_endpoints': True,
            'temporal_direction': 'future',
        }

        self.output_config = {
            'save_detailed_csv': True,
            'save_summary_json': True,
            'save_correlation_plots': True,
            'save_frame_plots': True,
        }

        self.viz_config = {
            'plot_every_nth_frame': 1,
            'plot_every_nth_sampling_point': 1,
            'save_sampling_plots': True,
            'save_intensity_plots': True,
            'save_movement_plots': True,
            'save_edge_transition_plots': True,
            'save_movement_overlay_plots': True,
            'save_intensity_overlay_plots': True,
            'save_movement_type_correlation': True,
            'movement_cmap': 'RdBu_r',
            'intensity_cmap': 'viridis',
            'marker_size': 30,
            'figure_dpi': 150,
            'display_stages': {
                'edge_contours': False,
                'movement_map': False,
                'sampling_regions': False,
                'local_intensities': False,
            }
        }

        self.initUI()
        self.setWindowTitle("Cell Edge Movement Analysis")
        self.resize(1000, 800)

    def initUI(self):
        """Initialize the user interface."""
        main_layout = QtWidgets.QVBoxLayout()

        # Create tab widget
        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: Window Selection
        self.setup_tab = self.create_setup_tab()
        self.tabs.addTab(self.setup_tab, "1️⃣ Setup")

        # Tab 2: Analysis Configuration
        self.config_tab = self.create_config_tab()
        self.tabs.addTab(self.config_tab, "2️⃣ Analysis Configuration")

        # Tab 3: Run Analysis
        self.run_tab = self.create_run_tab()
        self.tabs.addTab(self.run_tab, "3️⃣ Run Analysis")

        # Tab 4: Results & Visualization
        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.results_tab, "4️⃣ Results & Visualization")

        main_layout.addWidget(self.tabs)

        # Status bar
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { padding: 5px; background-color: #e8f4f8; }")
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)

    def create_setup_tab(self):
        """Create the setup and mask editor tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Instructions
        instructions = QtWidgets.QLabel(
            "<h3>📋 Analysis Setup</h3>"
            "<p>Select the FLIKA windows containing your PIEZO1 signal and cell masks.</p>"
            "<p><b>Tip:</b> Click on a FLIKA window to make it current, then use the buttons below.</p>"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Window selection group
        window_group = QtWidgets.QGroupBox("Window Selection")
        window_layout = QtWidgets.QGridLayout()

        # PIEZO1 window
        window_layout.addWidget(QtWidgets.QLabel("PIEZO1 Signal Window:"), 0, 0)

        # Only the "Use Current Window" button
        self.piezo1_current_btn = QtWidgets.QPushButton("📌 Use Current FLIKA Window")
        self.piezo1_current_btn.setToolTip("Click a FLIKA window first, then click this button")
        self.piezo1_current_btn.clicked.connect(self.use_current_as_piezo1)
        self.piezo1_current_btn.setStyleSheet("QPushButton { padding: 8px; }")
        window_layout.addWidget(self.piezo1_current_btn, 0, 1)

        # Mask window
        window_layout.addWidget(QtWidgets.QLabel("Cell Mask Window:"), 1, 0)

        # Only the "Use Current Window" button
        self.mask_current_btn = QtWidgets.QPushButton("📌 Use Current FLIKA Window")
        self.mask_current_btn.setToolTip("Click a FLIKA window first, then click this button")
        self.mask_current_btn.clicked.connect(self.use_current_as_mask)
        self.mask_current_btn.setStyleSheet("QPushButton { padding: 8px; }")
        window_layout.addWidget(self.mask_current_btn, 1, 1)

        # Window info
        self.window_info_label = QtWidgets.QLabel("No windows selected")
        self.window_info_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }")
        window_layout.addWidget(self.window_info_label, 2, 0, 1, 2)

        window_group.setLayout(window_layout)
        layout.addWidget(window_group)

        # Visualization stages group
        viz_group = QtWidgets.QGroupBox("🔬 Display Analysis Stages in FLIKA")
        viz_layout = QtWidgets.QGridLayout()

        viz_instructions = QtWidgets.QLabel(
            "Enable stages below to display them as FLIKA windows during analysis:"
        )
        viz_instructions.setWordWrap(True)
        viz_layout.addWidget(viz_instructions, 0, 0, 1, 2)

        self.show_edge_contours_check = QtWidgets.QCheckBox("Show Edge Contours")
        self.show_edge_contours_check.setToolTip("Display detected cell edge contours on each frame")
        viz_layout.addWidget(self.show_edge_contours_check, 1, 0)

        self.show_movement_map_check = QtWidgets.QCheckBox("Show Movement Map")
        self.show_movement_map_check.setToolTip("Display 2D movement map showing displacement at each position")
        viz_layout.addWidget(self.show_movement_map_check, 1, 1)

        self.show_sampling_regions_check = QtWidgets.QCheckBox("Show Sampling Regions")
        self.show_sampling_regions_check.setToolTip("Display rectangular sampling regions along the cell edge")
        viz_layout.addWidget(self.show_sampling_regions_check, 2, 0)

        self.show_local_intensities_check = QtWidgets.QCheckBox("Show Local Intensities")
        self.show_local_intensities_check.setToolTip("Display PIEZO1 intensity at each sampling region")
        viz_layout.addWidget(self.show_local_intensities_check, 2, 1)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_config_tab(self):
        """Create the configuration tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Create scroll area for configuration
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout()

        # Instructions
        instructions = QtWidgets.QLabel(
            "<h3>⚙️ Analysis Configuration</h3>"
            "<p>Configure the parameters for edge detection and movement analysis.</p>"
        )
        instructions.setWordWrap(True)
        scroll_layout.addWidget(instructions)

        # Sampling parameters
        sampling_group = QtWidgets.QGroupBox("Sampling Parameters")
        sampling_layout = QtWidgets.QFormLayout()

        self.n_points_spin = QtWidgets.QSpinBox()
        self.n_points_spin.setRange(3, 100)
        self.n_points_spin.setValue(self.config['n_points'])
        sampling_layout.addRow("Number of sampling points:", self.n_points_spin)

        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(10, 1000)
        self.depth_spin.setValue(self.config['depth'])
        sampling_layout.addRow("Sampling depth (pixels):", self.depth_spin)

        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(10, 500)
        self.width_spin.setValue(self.config['width'])
        sampling_layout.addRow("Sampling width (pixels):", self.width_spin)

        self.min_coverage_spin = QtWidgets.QDoubleSpinBox()
        self.min_coverage_spin.setRange(0.0, 1.0)
        self.min_coverage_spin.setSingleStep(0.05)
        self.min_coverage_spin.setValue(self.config['min_cell_coverage'])
        sampling_layout.addRow("Minimum cell coverage:", self.min_coverage_spin)

        self.try_rotation_check = QtWidgets.QCheckBox()
        self.try_rotation_check.setChecked(self.config['try_rotation'])
        sampling_layout.addRow("Try 180° rotation:", self.try_rotation_check)

        self.exclude_endpoints_check = QtWidgets.QCheckBox()
        self.exclude_endpoints_check.setChecked(self.config['exclude_endpoints'])
        sampling_layout.addRow("Exclude edge endpoints:", self.exclude_endpoints_check)

        sampling_group.setLayout(sampling_layout)
        scroll_layout.addWidget(sampling_group)

        # Movement parameters
        movement_group = QtWidgets.QGroupBox("Movement Detection Parameters")
        movement_layout = QtWidgets.QFormLayout()

        self.movement_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.movement_threshold_spin.setRange(0.01, 10.0)
        self.movement_threshold_spin.setSingleStep(0.05)
        self.movement_threshold_spin.setValue(self.config['movement_threshold'])
        movement_layout.addRow("Movement threshold (pixels):", self.movement_threshold_spin)

        self.min_movement_pixels_spin = QtWidgets.QSpinBox()
        self.min_movement_pixels_spin.setRange(1, 100)
        self.min_movement_pixels_spin.setValue(self.config['min_movement_pixels'])
        movement_layout.addRow("Min valid pixels:", self.min_movement_pixels_spin)

        self.temporal_direction_combo = QtWidgets.QComboBox()
        self.temporal_direction_combo.addItems(['future', 'past'])
        self.temporal_direction_combo.setCurrentText(self.config['temporal_direction'])
        movement_layout.addRow("Temporal direction:", self.temporal_direction_combo)

        # Add explanatory text
        temporal_help = QtWidgets.QLabel(
            "• <b>future</b>: Intensity at frame N predicts movement from N to N+1<br>"
            "• <b>past</b>: Movement from N-1 to N correlates with intensity at N"
        )
        temporal_help.setWordWrap(True)
        movement_layout.addRow("", temporal_help)

        movement_group.setLayout(movement_layout)
        scroll_layout.addWidget(movement_group)

        # Output configuration
        output_group = QtWidgets.QGroupBox("Output Options")
        output_layout = QtWidgets.QFormLayout()

        self.save_csv_check = QtWidgets.QCheckBox()
        self.save_csv_check.setChecked(self.output_config['save_detailed_csv'])
        output_layout.addRow("Save detailed CSV:", self.save_csv_check)

        self.save_json_check = QtWidgets.QCheckBox()
        self.save_json_check.setChecked(self.output_config['save_summary_json'])
        output_layout.addRow("Save summary JSON:", self.save_json_check)

        self.save_correlation_check = QtWidgets.QCheckBox()
        self.save_correlation_check.setChecked(self.output_config['save_correlation_plots'])
        output_layout.addRow("Save correlation plots:", self.save_correlation_check)

        self.save_frame_plots_check = QtWidgets.QCheckBox()
        self.save_frame_plots_check.setChecked(self.output_config['save_frame_plots'])
        output_layout.addRow("Save frame plots:", self.save_frame_plots_check)

        output_group.setLayout(output_layout)
        scroll_layout.addWidget(output_group)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self.save_config_btn = QtWidgets.QPushButton("💾 Save Configuration")
        self.save_config_btn.clicked.connect(self.save_configuration)

        self.load_config_btn = QtWidgets.QPushButton("📂 Load Configuration")
        self.load_config_btn.clicked.connect(self.load_configuration)

        self.reset_config_btn = QtWidgets.QPushButton("🔄 Reset to Defaults")
        self.reset_config_btn.clicked.connect(self.reset_configuration)

        button_layout.addWidget(self.save_config_btn)
        button_layout.addWidget(self.load_config_btn)
        button_layout.addWidget(self.reset_config_btn)

        scroll_layout.addLayout(button_layout)
        scroll_layout.addStretch()

        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)

        tab_layout = QtWidgets.QVBoxLayout()
        tab_layout.addWidget(scroll)
        tab.setLayout(tab_layout)

        return tab

    def create_run_tab(self):
        """Create the run analysis tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Instructions
        instructions = QtWidgets.QLabel(
            "<h3>▶️ Run Analysis</h3>"
            "<p>Start the cell edge movement analysis. Progress will be displayed below.</p>"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Output directory selection
        output_group = QtWidgets.QGroupBox("Output Directory")
        output_layout = QtWidgets.QHBoxLayout()

        self.output_dir_edit = QtWidgets.QLineEdit("xaxis_movement_analysis_results")
        self.output_dir_browse_btn = QtWidgets.QPushButton("📁 Browse")
        self.output_dir_browse_btn.clicked.connect(self.browse_output_dir)

        output_layout.addWidget(QtWidgets.QLabel("Save results to:"))
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.output_dir_browse_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Run button
        self.run_btn = QtWidgets.QPushButton("▶️ Start Analysis")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; background-color: #4CAF50; color: white; }")
        layout.addWidget(self.run_btn)

        self.stop_btn = QtWidgets.QPushButton("⏹️ Stop Analysis")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { font-size: 16px; padding: 10px; background-color: #f44336; color: white; }")
        layout.addWidget(self.stop_btn)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.progress_label = QtWidgets.QLabel("Ready to start")
        self.progress_label.setStyleSheet("QLabel { padding: 5px; }")
        layout.addWidget(self.progress_label)

        # Log output
        log_group = QtWidgets.QGroupBox("Analysis Log")
        log_layout = QtWidgets.QVBoxLayout()

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_results_tab(self):
        """Create the results visualization tab."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Instructions
        instructions = QtWidgets.QLabel(
            "<h3>📊 Results & Visualization</h3>"
            "<p>View and interact with analysis results. Navigate frames to see overlays.</p>"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Frame navigation
        nav_group = QtWidgets.QGroupBox("Frame Navigation")
        nav_layout = QtWidgets.QHBoxLayout()

        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)

        self.frame_label = QtWidgets.QLabel("Frame: 0/0")

        self.prev_frame_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.prev_frame_btn.setEnabled(False)

        self.next_frame_btn = QtWidgets.QPushButton("Next ▶")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_frame_btn)
        nav_layout.addWidget(self.frame_slider)
        nav_layout.addWidget(self.frame_label)
        nav_layout.addWidget(self.next_frame_btn)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)

        # Overlay controls
        overlay_group = QtWidgets.QGroupBox("Overlay Controls")
        overlay_layout = QtWidgets.QGridLayout()

        self.show_edge_check = QtWidgets.QCheckBox("Show cell edge")
        self.show_edge_check.setChecked(True)
        self.show_edge_check.stateChanged.connect(self.update_overlay)

        self.show_sampling_check = QtWidgets.QCheckBox("Show sampling regions")
        self.show_sampling_check.setChecked(True)
        self.show_sampling_check.stateChanged.connect(self.update_overlay)

        self.show_movement_check = QtWidgets.QCheckBox("Show movement vectors")
        self.show_movement_check.setChecked(True)
        self.show_movement_check.stateChanged.connect(self.update_overlay)

        self.clear_overlays_btn = QtWidgets.QPushButton("🗑️ Clear All Overlays")
        self.clear_overlays_btn.clicked.connect(self.clear_all_overlays)

        overlay_layout.addWidget(self.show_edge_check, 0, 0)
        overlay_layout.addWidget(self.show_sampling_check, 0, 1)
        overlay_layout.addWidget(self.show_movement_check, 1, 0)
        overlay_layout.addWidget(self.clear_overlays_btn, 1, 1)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # Current frame info
        info_group = QtWidgets.QGroupBox("Current Frame Information")
        info_layout = QtWidgets.QVBoxLayout()

        self.frame_info_text = QtWidgets.QTextEdit()
        self.frame_info_text.setReadOnly(True)
        self.frame_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.frame_info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Summary statistics
        stats_group = QtWidgets.QGroupBox("Summary Statistics")
        stats_layout = QtWidgets.QVBoxLayout()

        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Export buttons
        export_layout = QtWidgets.QHBoxLayout()

        self.export_results_btn = QtWidgets.QPushButton("📊 Export Results CSV")
        self.export_results_btn.clicked.connect(self.export_results)
        self.export_results_btn.setEnabled(False)

        self.export_plots_btn = QtWidgets.QPushButton("📈 Generate Summary Plots")
        self.export_plots_btn.clicked.connect(self.generate_summary_plots)
        self.export_plots_btn.setEnabled(False)

        export_layout.addWidget(self.export_results_btn)
        export_layout.addWidget(self.export_plots_btn)

        layout.addLayout(export_layout)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    # =========================================================================
    # Setup Tab Methods
    # =========================================================================

    def use_current_as_piezo1(self):
        """Use the current FLIKA window (g.win) as PIEZO1 window."""
        if g.win is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Window",
                "Please click on a FLIKA window first to make it the current window."
            )
            return

        self.piezo1_window = g.win
        self.update_window_info()
        self.log_message(f"Set PIEZO1 window to: {self.piezo1_window.name}")
        QtWidgets.QMessageBox.information(
            self,
            "Window Set",
            f"PIEZO1 window set to: {self.piezo1_window.name}"
        )

    def use_current_as_mask(self):
        """Use the current FLIKA window (g.win) as mask window."""
        if g.win is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No Window",
                "Please click on a FLIKA window first to make it the current window."
            )
            return

        self.mask_window = g.win
        self.update_window_info()
        self.log_message(f"Set mask window to: {self.mask_window.name}")
        QtWidgets.QMessageBox.information(
            self,
            "Window Set",
            f"Mask window set to: {self.mask_window.name}"
        )

    def update_window_info(self):
        """Update the window information display."""
        if self.piezo1_window is None and self.mask_window is None:
            self.window_info_label.setText("⚠️ No windows selected")
            return

        info_parts = []

        if self.piezo1_window is not None:
            info_parts.append(
                f"<b>PIEZO1 Window:</b> {self.piezo1_window.name}<br>"
                f"Shape: {self.piezo1_window.mt} frames × {self.piezo1_window.my} × {self.piezo1_window.mx} pixels"
            )

        if self.mask_window is not None:
            info_parts.append(
                f"<b>Mask Window:</b> {self.mask_window.name}<br>"
                f"Shape: {self.mask_window.mt} frames × {self.mask_window.my} × {self.mask_window.mx} pixels"
            )

        # Check compatibility
        if self.piezo1_window is not None and self.mask_window is not None:
            if (self.piezo1_window.mt != self.mask_window.mt or
                self.piezo1_window.my != self.mask_window.my or
                self.piezo1_window.mx != self.mask_window.mx):
                info_parts.append("<br><span style='color: red;'>⚠️ Warning: Window dimensions don't match!</span>")
            else:
                info_parts.append("<br><span style='color: green;'>✓ Windows are compatible</span>")

        self.window_info_label.setText("<br><br>".join(info_parts))

    # =========================================================================
    # Configuration Tab Methods
    # =========================================================================

    def get_current_config(self):
        """Get current configuration from UI."""
        self.config['n_points'] = self.n_points_spin.value()
        self.config['depth'] = self.depth_spin.value()
        self.config['width'] = self.width_spin.value()
        self.config['min_cell_coverage'] = self.min_coverage_spin.value()
        self.config['try_rotation'] = self.try_rotation_check.isChecked()
        self.config['exclude_endpoints'] = self.exclude_endpoints_check.isChecked()
        self.config['movement_threshold'] = self.movement_threshold_spin.value()
        self.config['min_movement_pixels'] = self.min_movement_pixels_spin.value()
        self.config['temporal_direction'] = self.temporal_direction_combo.currentText()

        self.output_config['save_detailed_csv'] = self.save_csv_check.isChecked()
        self.output_config['save_summary_json'] = self.save_json_check.isChecked()
        self.output_config['save_correlation_plots'] = self.save_correlation_check.isChecked()
        self.output_config['save_frame_plots'] = self.save_frame_plots_check.isChecked()

        # Update visualization config with display stage settings
        self.viz_config['display_stages']['edge_contours'] = self.show_edge_contours_check.isChecked()
        self.viz_config['display_stages']['movement_map'] = self.show_movement_map_check.isChecked()
        self.viz_config['display_stages']['sampling_regions'] = self.show_sampling_regions_check.isChecked()
        self.viz_config['display_stages']['local_intensities'] = self.show_local_intensities_check.isChecked()

        return self.config, self.output_config

    def save_configuration(self):
        """Save configuration to JSON file."""
        filename = save_file_gui("Save Configuration", None, "*.json")
        if filename:
            config_data = {
                'analysis_config': self.config,
                'output_config': self.output_config,
                'viz_config': self.viz_config
            }

            try:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=2)
                QtWidgets.QMessageBox.information(
                    self, 'Success', f'Configuration saved to:\n{filename}')
                self.log_message(f"Configuration saved to: {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', f'Failed to save configuration:\n{str(e)}')

    def load_configuration(self):
        """Load configuration from JSON file."""
        filename = open_file_gui("Load Configuration", None, "*.json")
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_data = json.load(f)

                # Update configs
                if 'analysis_config' in config_data:
                    self.config.update(config_data['analysis_config'])
                if 'output_config' in config_data:
                    self.output_config.update(config_data['output_config'])
                if 'viz_config' in config_data:
                    self.viz_config.update(config_data['viz_config'])

                # Update UI
                self.n_points_spin.setValue(self.config['n_points'])
                self.depth_spin.setValue(self.config['depth'])
                self.width_spin.setValue(self.config['width'])
                self.min_coverage_spin.setValue(self.config['min_cell_coverage'])
                self.try_rotation_check.setChecked(self.config['try_rotation'])
                self.exclude_endpoints_check.setChecked(self.config['exclude_endpoints'])
                self.movement_threshold_spin.setValue(self.config['movement_threshold'])
                self.min_movement_pixels_spin.setValue(self.config['min_movement_pixels'])
                self.temporal_direction_combo.setCurrentText(self.config['temporal_direction'])

                self.save_csv_check.setChecked(self.output_config['save_detailed_csv'])
                self.save_json_check.setChecked(self.output_config['save_summary_json'])
                self.save_correlation_check.setChecked(self.output_config['save_correlation_plots'])
                self.save_frame_plots_check.setChecked(self.output_config['save_frame_plots'])

                QtWidgets.QMessageBox.information(
                    self, 'Success', 'Configuration loaded successfully')
                self.log_message(f"Configuration loaded from: {filename}")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', f'Failed to load configuration:\n{str(e)}')

    def reset_configuration(self):
        """Reset configuration to defaults."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Reset Configuration',
            'Reset all settings to default values?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            # Reset to defaults
            self.config = {
                'n_points': 12,
                'depth': 200,
                'width': 75,
                'min_cell_coverage': 0.8,
                'try_rotation': True,
                'movement_threshold': 0.1,
                'min_movement_pixels': 5,
                'exclude_endpoints': True,
                'temporal_direction': 'future',
            }

            # Update UI
            self.n_points_spin.setValue(12)
            self.depth_spin.setValue(200)
            self.width_spin.setValue(75)
            self.min_coverage_spin.setValue(0.8)
            self.try_rotation_check.setChecked(True)
            self.exclude_endpoints_check.setChecked(True)
            self.movement_threshold_spin.setValue(0.1)
            self.min_movement_pixels_spin.setValue(5)
            self.temporal_direction_combo.setCurrentText('future')

            self.log_message("Configuration reset to defaults")

    # =========================================================================
    # Run Tab Methods
    # =========================================================================

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if directory:
            self.output_dir_edit.setText(directory)

    def run_analysis(self):
        """Start the analysis."""
        # Validate inputs
        if self.piezo1_window is None or self.mask_window is None:
            QtWidgets.QMessageBox.warning(
                self, 'Missing Windows',
                'Please select both PIEZO1 and mask windows in the Setup tab.')
            return

        # Check window compatibility
        if (self.piezo1_window.mt != self.mask_window.mt or
            self.piezo1_window.my != self.mask_window.my or
            self.piezo1_window.mx != self.mask_window.mx):
            reply = QtWidgets.QMessageBox.question(
                self, 'Dimension Mismatch',
                'The PIEZO1 and mask windows have different dimensions. Continue anyway?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        # Get current configuration
        self.get_current_config()

        # Create output directory
        output_dir = self.output_dir_edit.text()
        os.makedirs(output_dir, exist_ok=True)

        # Disable UI during analysis
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)

        # Clear log
        self.log_text.clear()
        self.log_message("=" * 60)
        self.log_message("Starting Cell Edge Movement Analysis")
        self.log_message("=" * 60)
        self.log_message(f"PIEZO1 window: {self.piezo1_window.name}")
        self.log_message(f"Mask window: {self.mask_window.name}")
        self.log_message(f"Output directory: {output_dir}")
        self.log_message(f"Temporal direction: {self.config['temporal_direction']}")

        # Log visualization stages
        display_stages = self.viz_config['display_stages']
        enabled_stages = [k for k, v in display_stages.items() if v]
        if enabled_stages:
            self.log_message(f"Display stages enabled: {', '.join(enabled_stages)}")
        self.log_message("")

        # Start analysis in background thread
        self.analysis_thread = QtCore.QThread()
        self.analysis_worker = AnalysisWorker(
            self.piezo1_window,
            self.mask_window,
            self.config,
            self.output_config,
            self.viz_config
        )

        self.analysis_worker.moveToThread(self.analysis_thread)

        # Connect signals
        self.analysis_thread.started.connect(self.analysis_worker.run)
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.error.connect(self.on_analysis_error)
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        self.analysis_worker.error.connect(self.analysis_thread.quit)

        # Start thread
        self.analysis_thread.start()

    def stop_analysis(self):
        """Stop the running analysis."""
        if self.analysis_worker is not None:
            self.analysis_worker.stop()
            self.log_message("\n⏹️ Analysis stopped by user")

    def on_analysis_progress(self, percentage, message):
        """Handle analysis progress updates."""
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(message)
        self.log_message(message)

    def on_analysis_finished(self, results):
        """Handle analysis completion."""
        self.analysis_results = results

        # Re-enable UI
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)

        # Update progress
        self.progress_bar.setValue(100)
        self.progress_label.setText("Analysis complete!")

        self.log_message("\n" + "=" * 60)
        self.log_message("ANALYSIS COMPLETE")
        self.log_message("=" * 60)

        # Display summary
        if 'summary_stats' in results:
            stats = results['summary_stats']
            self.log_message(f"\nProcessed transitions: {stats.get('processed_transitions', 0)}")
            self.log_message(f"Valid measurements: {stats.get('valid_measurements', 0)} ({stats.get('valid_measurement_percentage', 0):.1f}%)")

            if 'movement_statistics' in stats:
                mvmt = stats['movement_statistics']
                self.log_message(f"\nMovement distribution:")
                self.log_message(f"  Extending: {mvmt.get('extending_transitions', 0)}")
                self.log_message(f"  Retracting: {mvmt.get('retracting_transitions', 0)}")
                self.log_message(f"  Stable: {mvmt.get('stable_transitions', 0)}")

            if 'correlation_analysis' in stats:
                corr = stats['correlation_analysis']
                if corr.get('r_squared') is not None:
                    self.log_message(f"\nCorrelation R²: {corr['r_squared']:.3f}")
                    self.log_message(f"P-value: {corr['p_value']:.3e}")

        # Enable results tab
        self.enable_results_tab()

        # Switch to results tab
        self.tabs.setCurrentIndex(3)

        QtWidgets.QMessageBox.information(
            self, 'Analysis Complete',
            'Cell edge movement analysis completed successfully!')

    def on_analysis_error(self, error_msg):
        """Handle analysis errors."""
        # Re-enable UI
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)

        self.log_message(f"\n❌ ERROR: {error_msg}")

        QtWidgets.QMessageBox.critical(
            self, 'Analysis Error',
            f'An error occurred during analysis:\n\n{error_msg}')

    # =========================================================================
    # Results Tab Methods
    # =========================================================================

    def enable_results_tab(self):
        """Enable results tab controls after analysis."""
        if self.analysis_results is None:
            return

        # Enable frame navigation
        n_frames = len(self.analysis_results.get('all_results', []))
        if n_frames > 0:
            self.frame_slider.setMaximum(n_frames - 1)
            self.frame_slider.setEnabled(True)
            self.prev_frame_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            self.frame_label.setText(f"Frame: 0/{n_frames - 1}")

        # Enable export buttons
        self.export_results_btn.setEnabled(True)
        self.export_plots_btn.setEnabled(True)

        # Display summary statistics
        self.display_summary_stats()

        # Show first frame
        self.on_frame_changed(0)

    def on_frame_changed(self, frame_idx):
        """Handle frame slider change."""
        if self.analysis_results is None:
            return

        all_results = self.analysis_results.get('all_results', [])
        if frame_idx >= len(all_results):
            return

        self.current_frame_results = all_results[frame_idx]

        # Update frame label
        self.frame_label.setText(f"Frame: {frame_idx}/{len(all_results) - 1}")

        # Update frame info display
        self.display_frame_info()

        # Update overlays
        self.update_overlay()

    def prev_frame(self):
        """Go to previous frame."""
        current = self.frame_slider.value()
        if current > 0:
            self.frame_slider.setValue(current - 1)

    def next_frame(self):
        """Go to next frame."""
        current = self.frame_slider.value()
        if current < self.frame_slider.maximum():
            self.frame_slider.setValue(current + 1)

    def display_frame_info(self):
        """Display information about the current frame."""
        if self.current_frame_results is None:
            return

        results = self.current_frame_results

        info_lines = []
        info_lines.append(f"<b>Movement Type:</b> {results.get('movement_type', 'N/A')}")
        info_lines.append(f"<b>Movement Score:</b> {results.get('movement_score', 0.0):.3f} pixels")
        info_lines.append(f"<b>Valid Sampling Points:</b> {np.sum(results.get('valid_points', []))}/{len(results.get('valid_points', []))}")

        # Intensity statistics
        intensities = results.get('intensities', [])
        valid_intensities = [i for i, v in zip(intensities, results.get('valid_points', [])) if v]
        if valid_intensities:
            info_lines.append(f"<b>Mean Intensity:</b> {np.mean(valid_intensities):.1f}")
            info_lines.append(f"<b>Intensity Range:</b> [{np.min(valid_intensities):.1f}, {np.max(valid_intensities):.1f}]")

        self.frame_info_text.setHtml("<br>".join(info_lines))

    def display_summary_stats(self):
        """Display summary statistics."""
        if self.analysis_results is None:
            return

        stats = self.analysis_results.get('summary_stats', {})

        stats_lines = []
        stats_lines.append("<h4>Analysis Summary</h4>")
        stats_lines.append(f"<b>Total frames:</b> {stats.get('total_frames', 0)}")
        stats_lines.append(f"<b>Processed transitions:</b> {stats.get('processed_transitions', 0)}")
        stats_lines.append(f"<b>Valid measurements:</b> {stats.get('valid_measurements', 0)} ({stats.get('valid_measurement_percentage', 0):.1f}%)")

        if 'movement_statistics' in stats:
            mvmt = stats['movement_statistics']
            stats_lines.append("<br><h4>Movement Statistics</h4>")
            stats_lines.append(f"<b>Extending transitions:</b> {mvmt.get('extending_transitions', 0)}")
            stats_lines.append(f"<b>Retracting transitions:</b> {mvmt.get('retracting_transitions', 0)}")
            stats_lines.append(f"<b>Stable transitions:</b> {mvmt.get('stable_transitions', 0)}")
            stats_lines.append(f"<b>Average movement score:</b> {mvmt.get('average_movement_score', 0):.3f} ± {mvmt.get('movement_score_std', 0):.3f}")

        if 'correlation_analysis' in stats:
            corr = stats['correlation_analysis']
            if corr.get('r_squared') is not None:
                stats_lines.append("<br><h4>Correlation Analysis</h4>")
                stats_lines.append(f"<b>R²:</b> {corr['r_squared']:.3f}")
                stats_lines.append(f"<b>P-value:</b> {corr['p_value']:.3e}")
                stats_lines.append(f"<b>Sample size:</b> {corr['sample_size']}")
                stats_lines.append(f"<b>Slope:</b> {corr.get('slope', 0):.6f}")

        self.stats_text.setHtml("<br>".join(stats_lines))

    def update_overlay(self):
        """Update the overlay visualization on the PIEZO1 window."""
        # Clear existing overlays
        self.clear_all_overlays()

        if self.current_frame_results is None or self.piezo1_window is None:
            return

        results = self.current_frame_results

        # Set the PIEZO1 window to the current frame
        frame_idx = self.frame_slider.value()
        if hasattr(self.piezo1_window, 'setIndex'):
            self.piezo1_window.setIndex(frame_idx)

        # Draw cell edge contour
        if self.show_edge_check.isChecked():
            contour = results.get('current_contour', None)
            if contour is not None and len(contour) > 0:
                # Create a freehand ROI from the contour points
                # Note: contour is (y, x), ROI expects [[x, y], ...]
                pts = [[x, y] for y, x in contour]
                try:
                    roi = makeROI('freehand', pts, self.piezo1_window, color=pg.mkColor('#00FF00'))
                    self.overlay_rois.append(roi)
                except Exception as e:
                    self.log_message(f"Could not draw edge contour: {str(e)}")

        # Draw sampling rectangles
        if self.show_sampling_check.isChecked():
            sampling_rects = results.get('sampling_rects', [])
            valid_points = results.get('valid_points', [])

            for i, (rect, valid) in enumerate(zip(sampling_rects, valid_points)):
                if rect is not None and valid:
                    corners = rect['corners']
                    # Draw rectangle using corners
                    # corners is Nx2 array of [x, y]
                    try:
                        roi = makeROI('freehand', corners.tolist(), self.piezo1_window,
                                    color=pg.mkColor('#FFFF00'))
                        self.overlay_rois.append(roi)
                    except Exception as e:
                        pass  # Skip if can't draw

        # Draw movement vectors
        if self.show_movement_check.isChecked():
            sampling_points = results.get('sampling_points', [])
            local_movement_scores = results.get('local_movement_scores', [])
            valid_points = results.get('valid_points', [])

            for point, movement, valid in zip(sampling_points, local_movement_scores, valid_points):
                if valid and not np.isnan(movement):
                    y, x = point
                    # Draw a line showing movement direction
                    # Positive movement = downward, negative = upward
                    length = abs(movement) * 10  # Scale for visibility
                    if movement > 0:
                        # Retracting (downward)
                        color = pg.mkColor('#FF0000')  # Red
                        end_y = y + length
                    else:
                        # Extending (upward)
                        color = pg.mkColor('#0000FF')  # Blue
                        end_y = y - length

                    try:
                        roi = makeROI('line', [[x, y], [x, end_y]], self.piezo1_window, color=color)
                        self.overlay_rois.append(roi)
                    except Exception as e:
                        pass

    def clear_all_overlays(self):
        """Clear all overlay ROIs."""
        if self.piezo1_window is not None:
            for roi in self.overlay_rois:
                try:
                    roi.delete()
                except:
                    pass
        self.overlay_rois = []

    def export_results(self):
        """Export results to CSV."""
        if self.analysis_results is None:
            return

        from .analysis_core import export_results_to_csv

        try:
            output_dir = self.output_dir_edit.text()
            export_results_to_csv(self.analysis_results, output_dir)

            QtWidgets.QMessageBox.information(
                self, 'Export Complete',
                f'Results exported to:\n{output_dir}')
            self.log_message(f"Results exported to: {output_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 'Export Error',
                f'Failed to export results:\n{str(e)}')

    def generate_summary_plots(self):
        """Generate summary plots."""
        if self.analysis_results is None:
            return

        if not MATPLOTLIB_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, 'Matplotlib Not Available',
                'Please install matplotlib to use plotting features:\npip install matplotlib')
            return

        from .analysis_core import generate_summary_plots

        try:
            output_dir = self.output_dir_edit.text()
            generate_summary_plots(self.analysis_results, output_dir)

            QtWidgets.QMessageBox.information(
                self, 'Plots Generated',
                f'Summary plots saved to:\n{output_dir}')
            self.log_message(f"Summary plots generated in: {output_dir}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, 'Plot Error',
                f'Failed to generate plots:\n{str(e)}')

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def closeEvent(self, event):
        """Handle window close event."""
        # Clean up
        self.clear_all_overlays()

        # Close visualization windows
        for win_name, win in list(self.viz_windows.items()):
            try:
                if hasattr(win, 'close'):
                    win.close()
            except:
                pass

        # Stop analysis if running
        if self.analysis_worker is not None:
            self.analysis_worker.stop()

        event.accept()


# =============================================================================
# Plugin Entry Point
# =============================================================================

def launch_plugin():
    """Launch the Cell Edge Movement Analysis plugin."""
    global plugin_window

    # Create and show plugin window
    plugin_window = CellEdgeMovementAnalysis()
    plugin_window.show()

    return plugin_window


# For FLIKA plugin system
if __name__ == '__main__':
    launch_plugin()
