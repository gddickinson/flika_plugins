#!/usr/bin/env python3
"""
GUI Control Panels Module
=========================

Provides organized control panels for the Volume Slider plugin.
Each panel handles specific functionality with proper signal connections.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                           QGroupBox, QLabel, QPushButton, QSpinBox,
                           QDoubleSpinBox, QSlider, QComboBox, QCheckBox,
                           QProgressBar, QTextEdit, QTabWidget, QSplitter)
from qtpy.QtCore import Qt, QTimer, Signal, QObject
from qtpy.QtGui import QFont


class ControlPanelManager(QObject):
    """
    Manager for all control panels in the Volume Slider plugin.

    Organizes panels by functionality and manages inter-panel communication.
    """

    # Signals for control changes
    volume_changed = Signal(int)
    processing_requested = Signal(str, dict)  # process_type, parameters
    visualization_changed = Signal(str, object)  # setting_name, value
    playback_state_changed = Signal(bool)  # is_playing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.logger = logging.getLogger(__name__)

        # Panel references
        self.volume_panel = None
        self.processing_panel = None
        self.analysis_panel = None
        self.visualization_panel = None

        # Current state
        self.current_volume = 0
        self.is_playing = False
        self.play_speed = 1.0

        # Timer for playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._advance_playback)

    def create_volume_panel(self) -> QWidget:
        """Create volume control panel."""
        self.volume_panel = VolumeControlPanel(self)
        return self.volume_panel

    def create_processing_panel(self) -> QWidget:
        """Create processing control panel."""
        self.processing_panel = ProcessingControlPanel(self)
        return self.processing_panel

    def create_analysis_panel(self) -> QWidget:
        """Create analysis control panel."""
        self.analysis_panel = AnalysisControlPanel(self)
        return self.analysis_panel

    def create_visualization_panel(self) -> QWidget:
        """Create visualization control panel."""
        self.visualization_panel = VisualizationControlPanel(self)
        return self.visualization_panel

    def get_visualization_threshold(self) -> float:
        """Get current visualization threshold."""
        if self.visualization_panel:
            return self.visualization_panel.get_threshold()
        return 0.5

    def get_render_mode(self) -> str:
        """Get current render mode."""
        if self.visualization_panel:
            return self.visualization_panel.get_render_mode()
        return "points"

    def update_volume_range(self, max_volume: int):
        """Update the volume range in controls."""
        if self.volume_panel:
            self.volume_panel.update_volume_range(max_volume)

    def set_data_info(self, shape: tuple, dtype: str, memory_mb: float):
        """Update data information display."""
        if self.volume_panel:
            self.volume_panel.set_data_info(shape, dtype, memory_mb)

    def _advance_playback(self):
        """Advance playback to next volume."""
        if hasattr(self.parent, 'data_manager') and self.parent.data_manager.has_data:
            max_volume = self.parent.data_manager.get_volume_count() - 1
            self.current_volume = (self.current_volume + 1) % (max_volume + 1)
            self.volume_changed.emit(self.current_volume)

            if self.volume_panel:
                self.volume_panel.update_volume_display(self.current_volume)


class VolumeControlPanel(QWidget):
    """Control panel for volume navigation and playback."""

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.logger = logging.getLogger(__name__)

        # UI elements
        self.volume_slider = None
        self.volume_spinbox = None
        self.play_button = None
        self.speed_spinbox = None
        self.data_info_label = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Data information group
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout(info_group)

        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setWordWrap(True)
        info_layout.addRow(self.data_info_label)

        layout.addWidget(info_group)

        # Volume navigation group
        nav_group = QGroupBox("Volume Navigation")
        nav_layout = QFormLayout(nav_group)

        # Volume slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(0)
        self.volume_slider.setValue(0)
        self.volume_slider.setEnabled(False)

        # Volume spinbox
        self.volume_spinbox = QSpinBox()
        self.volume_spinbox.setMinimum(0)
        self.volume_spinbox.setMaximum(0)
        self.volume_spinbox.setValue(0)
        self.volume_spinbox.setEnabled(False)

        nav_layout.addRow("Volume:", self.volume_slider)
        nav_layout.addRow("Volume #:", self.volume_spinbox)

        layout.addWidget(nav_group)

        # Playback controls group
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QVBoxLayout(playback_group)

        # Play button row
        play_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setEnabled(False)

        self.speed_spinbox = QDoubleSpinBox()
        self.speed_spinbox.setRange(0.1, 10.0)
        self.speed_spinbox.setValue(1.0)
        self.speed_spinbox.setSuffix(" fps")
        self.speed_spinbox.setDecimals(1)

        play_row.addWidget(QLabel("Speed:"))
        play_row.addWidget(self.speed_spinbox)
        play_row.addStretch()
        play_row.addWidget(self.play_button)

        playback_layout.addLayout(play_row)

        # Loop checkbox
        self.loop_checkbox = QCheckBox("Loop playback")
        self.loop_checkbox.setChecked(True)
        playback_layout.addWidget(self.loop_checkbox)

        layout.addWidget(playback_group)

        # Navigation buttons
        nav_buttons_group = QGroupBox("Quick Navigation")
        nav_buttons_layout = QHBoxLayout(nav_buttons_group)

        self.first_button = QPushButton("First")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.last_button = QPushButton("Last")

        for btn in [self.first_button, self.prev_button, self.next_button, self.last_button]:
            btn.setEnabled(False)
            nav_buttons_layout.addWidget(btn)

        layout.addWidget(nav_buttons_group)

        layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals."""
        self.volume_slider.valueChanged.connect(self._on_volume_slider_changed)
        self.volume_spinbox.valueChanged.connect(self._on_volume_spinbox_changed)
        self.play_button.toggled.connect(self._on_play_toggled)
        self.speed_spinbox.valueChanged.connect(self._on_speed_changed)

        # Navigation buttons
        self.first_button.clicked.connect(lambda: self._set_volume(0))
        self.prev_button.clicked.connect(self._previous_volume)
        self.next_button.clicked.connect(self._next_volume)
        self.last_button.clicked.connect(self._last_volume)

    def _on_volume_slider_changed(self, value):
        """Handle volume slider change."""
        self.volume_spinbox.blockSignals(True)
        self.volume_spinbox.setValue(value)
        self.volume_spinbox.blockSignals(False)

        self.manager.current_volume = value
        self.manager.volume_changed.emit(value)

    def _on_volume_spinbox_changed(self, value):
        """Handle volume spinbox change."""
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(value)
        self.volume_slider.blockSignals(False)

        self.manager.current_volume = value
        self.manager.volume_changed.emit(value)

    def _on_play_toggled(self, checked):
        """Handle play button toggle."""
        if checked:
            self.play_button.setText("Pause")
            interval = int(1000 / self.speed_spinbox.value())
            self.manager.play_timer.start(interval)
            self.manager.is_playing = True
        else:
            self.play_button.setText("Play")
            self.manager.play_timer.stop()
            self.manager.is_playing = False

        self.manager.playback_state_changed.emit(checked)

    def _on_speed_changed(self, value):
        """Handle playback speed change."""
        self.manager.play_speed = value
        if self.manager.is_playing:
            interval = int(1000 / value)
            self.manager.play_timer.start(interval)

    def _set_volume(self, volume):
        """Set specific volume."""
        max_vol = self.volume_slider.maximum()
        volume = max(0, min(volume, max_vol))
        self.volume_slider.setValue(volume)

    def _previous_volume(self):
        """Go to previous volume."""
        self._set_volume(self.volume_slider.value() - 1)

    def _next_volume(self):
        """Go to next volume."""
        self._set_volume(self.volume_slider.value() + 1)

    def _last_volume(self):
        """Go to last volume."""
        self._set_volume(self.volume_slider.maximum())

    def update_volume_range(self, max_volume):
        """Update volume range controls."""
        self.volume_slider.setMaximum(max_volume)
        self.volume_spinbox.setMaximum(max_volume)

        # Enable controls
        enabled = max_volume > 0
        self.volume_slider.setEnabled(enabled)
        self.volume_spinbox.setEnabled(enabled)
        self.play_button.setEnabled(enabled)

        for btn in [self.first_button, self.prev_button, self.next_button, self.last_button]:
            btn.setEnabled(enabled)

    def update_volume_display(self, volume):
        """Update volume display without triggering signals."""
        self.volume_slider.blockSignals(True)
        self.volume_spinbox.blockSignals(True)

        self.volume_slider.setValue(volume)
        self.volume_spinbox.setValue(volume)

        self.volume_slider.blockSignals(False)
        self.volume_spinbox.blockSignals(False)

    def set_data_info(self, shape, dtype, memory_mb):
        """Update data information display."""
        info_text = f"Shape: {shape}\nType: {dtype}\nMemory: {memory_mb:.1f} MB"
        self.data_info_label.setText(info_text)


class ProcessingControlPanel(QWidget):
    """Control panel for image processing operations."""

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Preprocessing group
        preprocess_group = QGroupBox("Preprocessing")
        preprocess_layout = QFormLayout(preprocess_group)

        # Background subtraction
        self.bg_subtract_btn = QPushButton("Subtract Background")
        self.bg_value_spinbox = QDoubleSpinBox()
        self.bg_value_spinbox.setRange(0, 10000)
        self.bg_value_spinbox.setValue(100)
        self.bg_value_spinbox.setDecimals(1)

        preprocess_layout.addRow("Background:", self.bg_value_spinbox)
        preprocess_layout.addRow(self.bg_subtract_btn)

        # Gaussian filtering
        self.gaussian_btn = QPushButton("Apply Gaussian Filter")
        self.gaussian_sigma_spinbox = QDoubleSpinBox()
        self.gaussian_sigma_spinbox.setRange(0.1, 10.0)
        self.gaussian_sigma_spinbox.setValue(1.0)
        self.gaussian_sigma_spinbox.setDecimals(1)

        preprocess_layout.addRow("Gaussian σ:", self.gaussian_sigma_spinbox)
        preprocess_layout.addRow(self.gaussian_btn)

        layout.addWidget(preprocess_group)

        # Volume processing group
        volume_group = QGroupBox("Volume Processing")
        volume_layout = QFormLayout(volume_group)

        # Reshape controls
        self.frames_per_vol_spinbox = QSpinBox()
        self.frames_per_vol_spinbox.setRange(1, 1000)
        self.frames_per_vol_spinbox.setValue(10)

        self.reshape_btn = QPushButton("Reshape to Volumes")

        volume_layout.addRow("Frames per Volume:", self.frames_per_vol_spinbox)
        volume_layout.addRow(self.reshape_btn)

        # Delete frames
        self.delete_frames_spinbox = QSpinBox()
        self.delete_frames_spinbox.setRange(0, 100)
        self.delete_frames_spinbox.setValue(0)

        volume_layout.addRow("Delete Frames:", self.delete_frames_spinbox)

        layout.addWidget(volume_group)

        # Transform group
        transform_group = QGroupBox("Geometric Transforms")
        transform_layout = QFormLayout(transform_group)

        # Shear transform
        self.shear_angle_spinbox = QDoubleSpinBox()
        self.shear_angle_spinbox.setRange(-90, 90)
        self.shear_angle_spinbox.setValue(45)
        self.shear_angle_spinbox.setDecimals(1)
        self.shear_angle_spinbox.setSuffix("°")

        self.shear_factor_spinbox = QDoubleSpinBox()
        self.shear_factor_spinbox.setRange(0.1, 10.0)
        self.shear_factor_spinbox.setValue(1.0)
        self.shear_factor_spinbox.setDecimals(1)

        self.apply_shear_btn = QPushButton("Apply Shear Transform")

        transform_layout.addRow("Shear Angle:", self.shear_angle_spinbox)
        transform_layout.addRow("Shear Factor:", self.shear_factor_spinbox)
        transform_layout.addRow(self.apply_shear_btn)

        layout.addWidget(transform_group)

        # Intensity processing
        intensity_group = QGroupBox("Intensity Processing")
        intensity_layout = QFormLayout(intensity_group)

        # DF/F0 calculation
        self.f0_start_spinbox = QSpinBox()
        self.f0_start_spinbox.setRange(0, 1000)
        self.f0_start_spinbox.setValue(0)

        self.f0_end_spinbox = QSpinBox()
        self.f0_end_spinbox.setRange(1, 1000)
        self.f0_end_spinbox.setValue(10)

        self.calc_dff0_btn = QPushButton("Calculate ΔF/F₀")

        intensity_layout.addRow("F₀ Start:", self.f0_start_spinbox)
        intensity_layout.addRow("F₀ End:", self.f0_end_spinbox)
        intensity_layout.addRow(self.calc_dff0_btn)

        # Multiplication factor
        self.multiply_factor_spinbox = QDoubleSpinBox()
        self.multiply_factor_spinbox.setRange(0.1, 1000.0)
        self.multiply_factor_spinbox.setValue(1.0)
        self.multiply_factor_spinbox.setDecimals(2)

        self.multiply_btn = QPushButton("Multiply Intensities")

        intensity_layout.addRow("Factor:", self.multiply_factor_spinbox)
        intensity_layout.addRow(self.multiply_btn)

        layout.addWidget(intensity_group)

        layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals."""
        self.bg_subtract_btn.clicked.connect(self._subtract_background)
        self.gaussian_btn.clicked.connect(self._apply_gaussian)
        self.reshape_btn.clicked.connect(self._reshape_volumes)
        self.apply_shear_btn.clicked.connect(self._apply_shear)
        self.calc_dff0_btn.clicked.connect(self._calculate_dff0)
        self.multiply_btn.clicked.connect(self._multiply_intensities)

    def _subtract_background(self):
        """Request background subtraction."""
        params = {'background_value': self.bg_value_spinbox.value()}
        self.manager.processing_requested.emit('subtract_background', params)

    def _apply_gaussian(self):
        """Request Gaussian filtering."""
        params = {'sigma': self.gaussian_sigma_spinbox.value()}
        self.manager.processing_requested.emit('gaussian_filter', params)

    def _reshape_volumes(self):
        """Request volume reshaping."""
        params = {
            'frames_per_volume': self.frames_per_vol_spinbox.value(),
            'delete_frames': self.delete_frames_spinbox.value()
        }
        self.manager.processing_requested.emit('reshape_volumes', params)

    def _apply_shear(self):
        """Request shear transform."""
        params = {
            'angle': self.shear_angle_spinbox.value(),
            'factor': self.shear_factor_spinbox.value()
        }
        self.manager.processing_requested.emit('shear_transform', params)

    def _calculate_dff0(self):
        """Request ΔF/F₀ calculation."""
        params = {
            'f0_start': self.f0_start_spinbox.value(),
            'f0_end': self.f0_end_spinbox.value()
        }
        self.manager.processing_requested.emit('calculate_dff0', params)

    def _multiply_intensities(self):
        """Request intensity multiplication."""
        params = {'factor': self.multiply_factor_spinbox.value()}
        self.manager.processing_requested.emit('multiply_intensities', params)


class AnalysisControlPanel(QWidget):
    """Control panel for analysis operations."""

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Event detection group
        event_group = QGroupBox("Event Detection")
        event_layout = QFormLayout(event_group)

        # Threshold for detection
        self.detection_threshold_spinbox = QDoubleSpinBox()
        self.detection_threshold_spinbox.setRange(0.0, 10.0)
        self.detection_threshold_spinbox.setValue(3.0)
        self.detection_threshold_spinbox.setDecimals(1)

        self.detect_events_btn = QPushButton("Detect Ca²⁺ Events")

        event_layout.addRow("Threshold (σ):", self.detection_threshold_spinbox)
        event_layout.addRow(self.detect_events_btn)

        layout.addWidget(event_group)

        # ROI analysis group
        roi_group = QGroupBox("ROI Analysis")
        roi_layout = QFormLayout(roi_group)

        self.create_rois_btn = QPushButton("Create ROIs")
        self.analyze_rois_btn = QPushButton("Analyze ROIs")
        self.export_rois_btn = QPushButton("Export ROI Data")

        roi_layout.addRow(self.create_rois_btn)
        roi_layout.addRow(self.analyze_rois_btn)
        roi_layout.addRow(self.export_rois_btn)

        layout.addWidget(roi_group)

        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.calculate_stats_btn = QPushButton("Calculate Statistics")
        self.stats_display = QTextEdit()
        self.stats_display.setMaximumHeight(150)
        self.stats_display.setReadOnly(True)

        stats_layout.addWidget(self.calculate_stats_btn)
        stats_layout.addWidget(self.stats_display)

        layout.addWidget(stats_group)

        layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals."""
        self.detect_events_btn.clicked.connect(self._detect_events)
        self.create_rois_btn.clicked.connect(self._create_rois)
        self.analyze_rois_btn.clicked.connect(self._analyze_rois)
        self.export_rois_btn.clicked.connect(self._export_rois)
        self.calculate_stats_btn.clicked.connect(self._calculate_statistics)

    def _detect_events(self):
        """Request event detection."""
        params = {'threshold': self.detection_threshold_spinbox.value()}
        self.manager.processing_requested.emit('detect_events', params)

    def _create_rois(self):
        """Request ROI creation."""
        self.manager.processing_requested.emit('create_rois', {})

    def _analyze_rois(self):
        """Request ROI analysis."""
        self.manager.processing_requested.emit('analyze_rois', {})

    def _export_rois(self):
        """Request ROI data export."""
        self.manager.processing_requested.emit('export_rois', {})

    def _calculate_statistics(self):
        """Request statistics calculation."""
        self.manager.processing_requested.emit('calculate_statistics', {})

    def update_statistics(self, stats_text):
        """Update statistics display."""
        self.stats_display.setText(stats_text)


class VisualizationControlPanel(QWidget):
    """Control panel for visualization settings."""

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.logger = logging.getLogger(__name__)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # 3D rendering group
        render_group = QGroupBox("3D Rendering")
        render_layout = QFormLayout(render_group)

        # Render mode
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["points", "surface", "volume", "wireframe"])
        self.render_mode_combo.setCurrentText("points")

        # Threshold
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)

        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setValue(0.5)
        self.threshold_spinbox.setDecimals(2)

        # Point size
        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(1, 20)
        self.point_size_spinbox.setValue(2)

        render_layout.addRow("Mode:", self.render_mode_combo)
        render_layout.addRow("Threshold:", self.threshold_slider)
        render_layout.addRow("Threshold Value:", self.threshold_spinbox)
        render_layout.addRow("Point Size:", self.point_size_spinbox)

        layout.addWidget(render_group)

        # Color mapping group
        color_group = QGroupBox("Color Mapping")
        color_layout = QFormLayout(color_group)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "jet",
            "hot", "cool", "spring", "summer", "autumn", "winter"
        ])
        self.colormap_combo.setCurrentText("viridis")

        self.invert_colormap_checkbox = QCheckBox()

        color_layout.addRow("Colormap:", self.colormap_combo)
        color_layout.addRow("Invert Colors:", self.invert_colormap_checkbox)

        layout.addWidget(color_group)

        # Display options group
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)

        self.show_axes_checkbox = QCheckBox()
        self.show_axes_checkbox.setChecked(True)

        self.show_grid_checkbox = QCheckBox()
        self.show_grid_checkbox.setChecked(False)

        self.show_scalebar_checkbox = QCheckBox()
        self.show_scalebar_checkbox.setChecked(True)

        self.auto_rotate_checkbox = QCheckBox()
        self.auto_rotate_checkbox.setChecked(False)

        display_layout.addRow("Show Axes:", self.show_axes_checkbox)
        display_layout.addRow("Show Grid:", self.show_grid_checkbox)
        display_layout.addRow("Show Scale Bar:", self.show_scalebar_checkbox)
        display_layout.addRow("Auto Rotate:", self.auto_rotate_checkbox)

        layout.addWidget(display_group)

        # View controls group
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout(view_group)

        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")

        # Preset views
        presets_layout = QHBoxLayout()
        self.view_xy_btn = QPushButton("XY")
        self.view_xz_btn = QPushButton("XZ")
        self.view_yz_btn = QPushButton("YZ")
        self.view_iso_btn = QPushButton("ISO")

        presets_layout.addWidget(self.view_xy_btn)
        presets_layout.addWidget(self.view_xz_btn)
        presets_layout.addWidget(self.view_yz_btn)
        presets_layout.addWidget(self.view_iso_btn)

        view_layout.addWidget(self.reset_view_btn)
        view_layout.addLayout(presets_layout)

        layout.addWidget(view_group)

        layout.addStretch()

    def _connect_signals(self):
        """Connect UI signals."""
        self.render_mode_combo.currentTextChanged.connect(
            lambda text: self.manager.visualization_changed.emit('render_mode', text)
        )

        self.threshold_slider.valueChanged.connect(self._update_threshold_from_slider)
        self.threshold_spinbox.valueChanged.connect(self._update_threshold_from_spinbox)

        self.point_size_spinbox.valueChanged.connect(
            lambda value: self.manager.visualization_changed.emit('point_size', value)
        )

        self.colormap_combo.currentTextChanged.connect(
            lambda text: self.manager.visualization_changed.emit('colormap', text)
        )

        self.invert_colormap_checkbox.toggled.connect(
            lambda checked: self.manager.visualization_changed.emit('invert_colormap', checked)
        )

        # Display options
        self.show_axes_checkbox.toggled.connect(
            lambda checked: self.manager.visualization_changed.emit('show_axes', checked)
        )

        self.show_grid_checkbox.toggled.connect(
            lambda checked: self.manager.visualization_changed.emit('show_grid', checked)
        )

        self.show_scalebar_checkbox.toggled.connect(
            lambda checked: self.manager.visualization_changed.emit('show_scalebar', checked)
        )

        # View controls
        self.reset_view_btn.clicked.connect(
            lambda: self.manager.visualization_changed.emit('reset_view', True)
        )

        self.view_xy_btn.clicked.connect(
            lambda: self.manager.visualization_changed.emit('view_preset', 'xy')
        )

        self.view_xz_btn.clicked.connect(
            lambda: self.manager.visualization_changed.emit('view_preset', 'xz')
        )

        self.view_yz_btn.clicked.connect(
            lambda: self.manager.visualization_changed.emit('view_preset', 'yz')
        )

        self.view_iso_btn.clicked.connect(
            lambda: self.manager.visualization_changed.emit('view_preset', 'iso')
        )

    def _update_threshold_from_slider(self, value):
        """Update threshold from slider."""
        threshold_value = value / 100.0
        self.threshold_spinbox.blockSignals(True)
        self.threshold_spinbox.setValue(threshold_value)
        self.threshold_spinbox.blockSignals(False)

        self.manager.visualization_changed.emit('threshold', threshold_value)

    def _update_threshold_from_spinbox(self, value):
        """Update threshold from spinbox."""
        slider_value = int(value * 100)
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(slider_value)
        self.threshold_slider.blockSignals(False)

        self.manager.visualization_changed.emit('threshold', value)

    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.threshold_spinbox.value()

    def get_render_mode(self) -> str:
        """Get current render mode."""
        return self.render_mode_combo.currentText()

    def get_colormap(self) -> str:
        """Get current colormap."""
        return self.colormap_combo.currentText()

    def get_point_size(self) -> int:
        """Get current point size."""
        return self.point_size_spinbox.value()


class ParameterGroup(QGroupBox):
    """
    Reusable parameter group widget.

    Provides consistent styling and layout for parameter controls.
    """

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.layout = QFormLayout(self)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(10, 15, 10, 10)

        # Apply consistent styling
        font = QFont()
        font.setBold(True)
        self.setFont(font)

    def add_parameter(self, label, widget):
        """Add a parameter control to the group."""
        self.layout.addRow(label, widget)

    def add_widget(self, widget):
        """Add a widget without label."""
        self.layout.addRow(widget)

    def add_button_row(self, buttons):
        """Add a row of buttons."""
        button_layout = QHBoxLayout()
        for button in buttons:
            button_layout.addWidget(button)
        button_layout.addStretch()
        self.layout.addRow(button_layout)
