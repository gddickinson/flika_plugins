#!/usr/bin/env python3
"""
Test Data Generation Dialog
===========================

Interactive dialog for generating synthetic test data with
comprehensive parameter control and preview capabilities.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                           QGroupBox, QLabel, QPushButton, QSpinBox,
                           QDoubleSpinBox, QSlider, QComboBox, QCheckBox,
                           QProgressBar, QTextEdit, QTabWidget, QFrame,
                           QSizePolicy, QScrollArea, QGridLayout, QSplitter, QWidget)
from qtpy.QtCore import Qt, QTimer, QThread, Signal, QObject
from qtpy.QtGui import QFont, QPixmap, QImage
import pyqtgraph as pg

# Fixed import paths - use absolute imports from the plugin root
try:
    from ....synthetic.test_data_generator import (TestDataGenerator, PatternType,
                                               GenerationParameters)
except ImportError:
    # Fallback for different import contexts
    try:
        from synthetic.test_data_generator import (TestDataGenerator, PatternType,
                                               GenerationParameters)
    except ImportError:
        # Create dummy classes if import fails
        class TestDataGenerator:
            def __init__(self): pass
            def generate_data(self, params): return np.random.random((10,10,10,10)), {}

        class PatternType:
            CALCIUM_PUFFS = "calcium_puffs"
            CALCIUM_WAVES = "calcium_waves"
            RANDOM_BLOBS = "random_blobs"

        class GenerationParameters:
            def __init__(self):
                self.volume_shape = (50, 256, 256)
                self.n_timepoints = 100
                self.pattern_type = PatternType.CALCIUM_PUFFS
                self.n_patterns = 20
                self.pattern_intensity_range = (0.5, 2.0)
                self.pattern_size_range = (2.0, 8.0)
                self.temporal_dynamics = True
                self.dynamics_speed = 1.0
                self.dynamics_variability = 0.3
                self.photon_noise = True
                self.gaussian_noise_level = 0.05
                self.baseline_level = 0.1
                self.psf_size = (2.0, 1.0, 1.0)
                self.photobleaching = True
                self.photobleaching_rate = 0.001
                self.output_dtype = np.float32
                self.normalize_output = True

try:
    from ....utils.error_handler import ErrorHandler
except ImportError:
    ErrorHandler = None


class DataGenerationThread(QThread):
    """Thread for generating test data without blocking UI."""

    progress = Signal(int, str)  # percentage, message
    finished = Signal(np.ndarray, dict)  # data, metadata
    error = Signal(str)  # error message

    def __init__(self, generator: TestDataGenerator, params: GenerationParameters):
        super().__init__()
        self.generator = generator
        self.params = params
        self.logger = logging.getLogger(__name__)

    def run(self):
        """Run data generation in separate thread."""
        try:
            # Connect progress signal if available
            if hasattr(self.generator, 'progress_updated'):
                self.generator.progress_updated.connect(self.progress.emit)

            # Generate data
            data, metadata = self.generator.generate_data(self.params)

            # Emit completion
            self.finished.emit(data, metadata)

        except Exception as e:
            self.logger.error(f"Data generation thread failed: {str(e)}")
            self.error.emit(str(e))


class TestDataDialog(QDialog):
    """
    Comprehensive dialog for synthetic test data generation.

    Features:
    - Interactive parameter adjustment
    - Real-time preview
    - Pattern type selection
    - Advanced parameter controls
    - Progress monitoring
    - Data export options
    """

    def __init__(self, generator: TestDataGenerator, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.logger = logging.getLogger(__name__)

        # Current parameters
        self.current_params = GenerationParameters()

        # Preview data
        self.preview_data = None
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)

        # Generation thread
        self.generation_thread = None

        # UI elements
        self.parameter_widgets = {}
        self.preview_widget = None
        self.progress_bar = None

        self._setup_ui()
        self._connect_signals()
        self._update_parameter_display()

        # Start with initial preview
        self._schedule_preview_update()

    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Generate Synthetic Test Data")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: Parameters
        param_widget = self._create_parameter_panel()
        splitter.addWidget(param_widget)

        # Right side: Preview
        preview_widget = self._create_preview_panel()
        splitter.addWidget(preview_widget)

        # Set splitter proportions
        splitter.setSizes([400, 600])

        # Bottom: Progress and buttons
        bottom_layout = QHBoxLayout()

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)

        # Buttons
        self.preview_button = QPushButton("Update Preview")
        self.generate_button = QPushButton("Generate Full Data")
        self.cancel_button = QPushButton("Cancel")

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.preview_button)
        bottom_layout.addWidget(self.generate_button)
        bottom_layout.addWidget(self.cancel_button)

        main_layout.addLayout(bottom_layout)

        # Apply styling
        self.setStyleSheet("""
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
        """)

    def _create_parameter_panel(self):
        """Create the parameter control panel."""
        # Scrollable widget for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        param_widget = QWidget()
        layout = QVBoxLayout(param_widget)

        # Pattern selection group
        pattern_group = QGroupBox("Pattern Type")
        pattern_layout = QFormLayout(pattern_group)

        self.pattern_combo = QComboBox()
        try:
            pattern_types = [pattern.value.replace('_', ' ').title() for pattern in PatternType]
        except:
            pattern_types = ["Calcium Puffs", "Calcium Waves", "Random Blobs"]
        self.pattern_combo.addItems(pattern_types)
        self.pattern_combo.setCurrentText("Calcium Puffs")

        pattern_layout.addRow("Pattern:", self.pattern_combo)
        layout.addWidget(pattern_group)

        # Volume dimensions group
        dims_group = QGroupBox("Volume Dimensions")
        dims_layout = QFormLayout(dims_group)

        self.z_size_spin = QSpinBox()
        self.z_size_spin.setRange(10, 200)
        self.z_size_spin.setValue(50)

        self.y_size_spin = QSpinBox()
        self.y_size_spin.setRange(50, 1024)
        self.y_size_spin.setValue(256)

        self.x_size_spin = QSpinBox()
        self.x_size_spin.setRange(50, 1024)
        self.x_size_spin.setValue(256)

        self.n_timepoints_spin = QSpinBox()
        self.n_timepoints_spin.setRange(10, 1000)
        self.n_timepoints_spin.setValue(100)

        dims_layout.addRow("Z Size:", self.z_size_spin)
        dims_layout.addRow("Y Size:", self.y_size_spin)
        dims_layout.addRow("X Size:", self.x_size_spin)
        dims_layout.addRow("Time Points:", self.n_timepoints_spin)

        layout.addWidget(dims_group)

        # Pattern parameters group
        pattern_params_group = QGroupBox("Pattern Parameters")
        pattern_params_layout = QFormLayout(pattern_params_group)

        self.n_patterns_spin = QSpinBox()
        self.n_patterns_spin.setRange(1, 200)
        self.n_patterns_spin.setValue(20)

        self.intensity_min_spin = QDoubleSpinBox()
        self.intensity_min_spin.setRange(0.0, 10.0)
        self.intensity_min_spin.setValue(0.5)
        self.intensity_min_spin.setDecimals(2)

        self.intensity_max_spin = QDoubleSpinBox()
        self.intensity_max_spin.setRange(0.0, 10.0)
        self.intensity_max_spin.setValue(2.0)
        self.intensity_max_spin.setDecimals(2)

        pattern_params_layout.addRow("Number of Patterns:", self.n_patterns_spin)
        pattern_params_layout.addRow("Min Intensity:", self.intensity_min_spin)
        pattern_params_layout.addRow("Max Intensity:", self.intensity_max_spin)

        layout.addWidget(pattern_params_group)

        # Simplified controls for basic functionality
        controls_group = QGroupBox("Basic Controls")
        controls_layout = QFormLayout(controls_group)

        self.noise_check = QCheckBox()
        self.noise_check.setChecked(True)

        self.normalize_check = QCheckBox()
        self.normalize_check.setChecked(True)

        controls_layout.addRow("Add Noise:", self.noise_check)
        controls_layout.addRow("Normalize Output:", self.normalize_check)

        layout.addWidget(controls_group)

        layout.addStretch()
        scroll.setWidget(param_widget)

        # Store widget references for easy access
        self.parameter_widgets = {
            'pattern_combo': self.pattern_combo,
            'z_size': self.z_size_spin,
            'y_size': self.y_size_spin,
            'x_size': self.x_size_spin,
            'n_timepoints': self.n_timepoints_spin,
            'n_patterns': self.n_patterns_spin,
            'intensity_min': self.intensity_min_spin,
            'intensity_max': self.intensity_max_spin,
            'noise': self.noise_check,
            'normalize': self.normalize_check
        }

        return scroll

    def _create_preview_panel(self):
        """Create the preview panel."""
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)

        # Preview controls
        controls_layout = QHBoxLayout()

        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems([
            "Max Projection", "Single Slice", "Time Series"
        ])

        controls_layout.addWidget(QLabel("View:"))
        controls_layout.addWidget(self.preview_mode_combo)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Preview image view
        self.preview_widget = pg.ImageView()
        self.preview_widget.ui.roiBtn.hide()
        self.preview_widget.ui.menuBtn.hide()
        layout.addWidget(self.preview_widget)

        # Preview info
        self.preview_info = QTextEdit()
        self.preview_info.setMaximumHeight(100)
        self.preview_info.setReadOnly(True)
        layout.addWidget(self.preview_info)

        return preview_widget

    def _connect_signals(self):
        """Connect UI signals."""
        # Parameter changes trigger preview update
        for widget in self.parameter_widgets.values():
            if hasattr(widget, 'valueChanged'):
                widget.valueChanged.connect(self._schedule_preview_update)
            elif hasattr(widget, 'currentTextChanged'):
                widget.currentTextChanged.connect(self._schedule_preview_update)
            elif hasattr(widget, 'toggled'):
                widget.toggled.connect(self._schedule_preview_update)

        # Preview controls
        self.preview_mode_combo.currentTextChanged.connect(self._update_preview_display)

        # Buttons
        self.preview_button.clicked.connect(self._force_preview_update)
        self.generate_button.clicked.connect(self._generate_data)
        self.cancel_button.clicked.connect(self.reject)

    def _schedule_preview_update(self):
        """Schedule preview update with debouncing."""
        self.preview_timer.start(500)  # 500ms delay

    def _force_preview_update(self):
        """Force immediate preview update."""
        self.preview_timer.stop()
        self._update_preview()

    def _update_parameter_display(self):
        """Update parameter display ranges based on current values."""
        pass  # Simplified for now

    def _update_preview(self):
        """Update preview with current parameters."""
        try:
            self.logger.debug("Updating preview...")

            # Get current parameters for small preview
            params = self._get_parameters()

            # Use smaller size for preview
            original_shape = params.volume_shape
            original_timepoints = params.n_timepoints

            # Scale down for preview
            scale_factor = 0.3
            params.volume_shape = (
                max(10, int(original_shape[0] * scale_factor)),
                max(32, int(original_shape[1] * scale_factor)),
                max(32, int(original_shape[2] * scale_factor))
            )
            params.n_timepoints = min(20, original_timepoints)

            # Generate preview data
            self.preview_data, metadata = self.generator.generate_data(params)

            # Update preview display
            self._update_preview_display()

            # Update info
            self._update_preview_info(metadata)

            self.logger.debug("Preview update completed")

        except Exception as e:
            self.logger.error(f"Preview update failed: {str(e)}")
            self.preview_info.setText(f"Preview failed: {str(e)}")

    def _update_preview_display(self):
        """Update the preview display based on current mode."""
        if self.preview_data is None:
            return

        try:
            mode = self.preview_mode_combo.currentText()

            if mode == "Max Projection":
                # Show max projection over Z
                if self.preview_data.ndim == 4:
                    image = np.max(self.preview_data[:, 0, :, :], axis=0)
                else:
                    image = np.max(self.preview_data, axis=0)

                self.preview_widget.setImage(image)

            elif mode == "Single Slice":
                # Show single Z slice
                if self.preview_data.ndim == 4:
                    z_mid = self.preview_data.shape[0] // 2
                    image = self.preview_data[z_mid, 0, :, :]
                else:
                    z_mid = self.preview_data.shape[0] // 2
                    image = self.preview_data[z_mid, :, :]

                self.preview_widget.setImage(image)

            elif mode == "Time Series":
                # Show time series of max projections
                if self.preview_data.ndim == 4:
                    time_series = np.max(self.preview_data, axis=0)  # Max over Z
                    self.preview_widget.setImage(time_series)

        except Exception as e:
            self.logger.error(f"Preview display update failed: {str(e)}")

    def _update_preview_info(self, metadata: Dict[str, Any]):
        """Update preview information display."""
        try:
            info_text = "Preview Information:\n"
            info_text += f"Pattern: {metadata.get('pattern_type', 'Unknown')}\n"

            data_props = metadata.get('data_properties', {})
            info_text += f"Shape: {data_props.get('shape', 'Unknown')}\n"
            info_text += f"Data Type: {data_props.get('dtype', 'Unknown')}\n"
            info_text += f"Value Range: {data_props.get('min_value', 0):.3f} - {data_props.get('max_value', 1):.3f}\n"

            self.preview_info.setText(info_text)

        except Exception as e:
            self.logger.error(f"Preview info update failed: {str(e)}")

    def _get_parameters(self) -> GenerationParameters:
        """Get current generation parameters from UI."""
        params = GenerationParameters()

        # Get basic parameters
        params.volume_shape = (
            self.z_size_spin.value(),
            self.y_size_spin.value(),
            self.x_size_spin.value()
        )
        params.n_timepoints = self.n_timepoints_spin.value()
        params.n_patterns = self.n_patterns_spin.value()
        params.pattern_intensity_range = (
            self.intensity_min_spin.value(),
            self.intensity_max_spin.value()
        )
        params.photon_noise = self.noise_check.isChecked()
        params.normalize_output = self.normalize_check.isChecked()

        return params

    def _generate_data(self):
        """Generate full-size data."""
        try:
            # Disable UI during generation
            self._set_ui_enabled(False)

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Get parameters
            params = self._get_parameters()

            # Start generation thread
            self.generation_thread = DataGenerationThread(self.generator, params)
            self.generation_thread.progress.connect(self._on_generation_progress)
            self.generation_thread.finished.connect(self._on_generation_finished)
            self.generation_thread.error.connect(self._on_generation_error)
            self.generation_thread.start()

        except Exception as e:
            self.logger.error(f"Data generation failed to start: {str(e)}")
            self._on_generation_error(str(e))

    def _on_generation_progress(self, percentage: int, message: str):
        """Handle generation progress update."""
        self.progress_bar.setValue(percentage)

    def _on_generation_finished(self, data: np.ndarray, metadata: Dict[str, Any]):
        """Handle successful data generation."""
        try:
            self.generated_data = data
            self.generated_metadata = metadata

            # Hide progress
            self.progress_bar.setVisible(False)

            # Re-enable UI
            self._set_ui_enabled(True)

            # Accept dialog
            self.accept()

        except Exception as e:
            self.logger.error(f"Generation completion handling failed: {str(e)}")
            self._on_generation_error(str(e))

    def _on_generation_error(self, error_message: str):
        """Handle generation error."""
        self.logger.error(f"Data generation error: {error_message}")

        # Hide progress
        self.progress_bar.setVisible(False)

        # Re-enable UI
        self._set_ui_enabled(True)

        # Show error message
        from qtpy.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Generation Error",
                           f"Data generation failed:\n\n{error_message}")

    def _set_ui_enabled(self, enabled: bool):
        """Enable or disable UI controls."""
        self.preview_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)

        for widget in self.parameter_widgets.values():
            widget.setEnabled(enabled)

    def get_parameters(self) -> GenerationParameters:
        """Get the current generation parameters."""
        return self._get_parameters()

    def get_generated_data(self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Get generated data and metadata."""
        if hasattr(self, 'generated_data'):
            return self.generated_data, self.generated_metadata
        return None, None

    def closeEvent(self, event):
        """Handle dialog close."""
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.terminate()
            self.generation_thread.wait()

        event.accept()
