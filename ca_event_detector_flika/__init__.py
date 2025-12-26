"""
Calcium Event Detector for FLIKA
=================================

A comprehensive FLIKA plugin for deep learning-based detection and classification
of local Ca²⁺ release events in confocal imaging.

This plugin provides:
- Automatic detection of Ca²⁺ sparks, puffs, and waves
- 3D U-Net neural network architecture
- Interactive visualization with color-coded overlays
- Results viewer with filtering and statistics
- Export functionality for masks and event properties
- Model training interface with robust error handling
- Comprehensive diagnostics and testing tools

Features integrated from train_patches_resume_mps.py:
- Robust checkpoint finding by iteration number
- Data integrity verification (iCloud stub detection)
- Robust dataset wrapper for corrupted file handling
- Enhanced progress monitoring with ETA
- Emergency checkpoint saving
- MPS optimization for Apple Silicon

Author: George Stuyt
Reference: Dotti, P., et al. (2024). Cell Calcium, 121, 102893.
"""

__version__ = '1.1.0'  # Updated with diagnostics & testing
__author__ = 'George'

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, WindowSelector, CheckBox, ComboBox, BaseProcess_noPriorWindow
from flika.process.file_ import save_file_gui, open_file_gui
from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QMessageBox,
                            QProgressDialog, QSpinBox, QDoubleSpinBox, QTextEdit,
                            QTabWidget, QTableWidget, QTableWidgetItem)
from qtpy.QtCore import Qt
import numpy as np
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import plugin modules
from .ca_event_detector_flika_integration import (
    ca_event_detector_run_detection,
    ca_event_detector_quick_detection,
    ca_event_detector_toggle_display
)

from .event_display import (
    EventDisplay,
    ca_event_detector_display_results
)

from .event_results_viewer import EventResultsViewer

# Import diagnostics and testing modules
from . import diagnostics
from . import testing
from . import training_utils


# ============================================================================
# View Results Plugin
# ============================================================================

class CA_EventDetector_ViewResults(BaseProcess):
    """
    View detection results in interactive spreadsheet viewer.
    """
    
    def __init__(self):
        super().__init__()
        self.viewer = None
    
    def gui(self):
        """Create results viewer GUI."""
        self.gui_reset()
        
        window_selector = WindowSelector()
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        
        super().gui()
    
    def __call__(self, window, keepSourceWindow=False):
        """Open results viewer."""
        try:
            if not hasattr(window, 'ca_event_results'):
                QMessageBox.warning(
                    None, "No Results",
                    "No detection results found for this window.\n"
                    "Please run detection first."
                )
                return window
            
            # Create viewer if it doesn't exist
            if self.viewer is None or not self.viewer.isVisible():
                self.viewer = EventResultsViewer()
                self.viewer.show()
                g.ca_event_results_viewer = self.viewer
            else:
                self.viewer.raise_()
                self.viewer.activateWindow()
            
            # Load results
            self.viewer.set_data(window.ca_event_results)
            
            n_events = len(self.viewer.data) if self.viewer.data is not None else 0
            self.viewer.info_label.setText(
                f"Window: {window.name} ({n_events} events)"
            )
            
            g.m.statusBar().showMessage(f"Opened results viewer with {n_events} events")
            
        except Exception as e:
            import logging
            logging.error(f"Error opening results viewer: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Viewer Error",
                f"Error opening results viewer:\n{str(e)}"
            )
        
        return window


# ============================================================================
# Export Results Plugin
# ============================================================================

class CA_EventDetector_ExportResults(BaseProcess):
    """
    Export detection results to files.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create export GUI."""
        self.gui_reset()
        
        window_selector = WindowSelector()
        
        export_masks = CheckBox()
        export_masks.setChecked(True)
        export_masks.setToolTip("Export class and instance masks as TIFF files")
        
        export_csv = CheckBox()
        export_csv.setChecked(True)
        export_csv.setToolTip("Export event properties to CSV file")
        
        export_probabilities = CheckBox()
        export_probabilities.setChecked(False)
        export_probabilities.setToolTip("Export probability maps (4D array, can be large)")
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'export_masks', 'string': 'Export Masks (TIFF)',
                          'object': export_masks})
        self.items.append({'name': 'export_csv', 'string': 'Export Events (CSV)',
                          'object': export_csv})
        self.items.append({'name': 'export_probabilities', 'string': 'Export Probabilities (TIFF)',
                          'object': export_probabilities})
        
        super().gui()
    
    def __call__(self, window, export_masks, export_csv, export_probabilities,
                 keepSourceWindow=False):
        """Export results to files."""
        try:
            if not hasattr(window, 'ca_event_results'):
                QMessageBox.warning(
                    None, "No Results",
                    "No detection results found for this window."
                )
                return window
            
            # Select output directory
            save_dir = QFileDialog.getExistingDirectory(
                None, "Select Output Directory"
            )
            
            if not save_dir:
                return window
            
            results = window.ca_event_results
            prefix = window.name.replace(' ', '_')
            
            saved_files = []
            
            # Export masks
            if export_masks:
                try:
                    from tifffile import imwrite
                except ImportError:
                    QMessageBox.critical(
                        None, "Import Error",
                        "tifffile not installed. Install with: pip install tifffile"
                    )
                    return window
                
                class_path = Path(save_dir) / f"{prefix}_class_mask.tif"
                instance_path = Path(save_dir) / f"{prefix}_instance_mask.tif"
                
                imwrite(str(class_path), results['class_mask'].astype(np.uint8))
                imwrite(str(instance_path), results['instance_mask'].astype(np.uint16))
                
                saved_files.append(str(class_path))
                saved_files.append(str(instance_path))
            
            # Export probabilities
            if export_probabilities and 'probabilities' in results:
                try:
                    from tifffile import imwrite
                except ImportError:
                    pass
                else:
                    prob_path = Path(save_dir) / f"{prefix}_probabilities.tif"
                    imwrite(str(prob_path), results['probabilities'].astype(np.float32))
                    saved_files.append(str(prob_path))
            
            # Export CSV
            if export_csv:
                try:
                    import pandas as pd
                    from scipy import ndimage
                except ImportError:
                    QMessageBox.critical(
                        None, "Import Error",
                        "pandas or scipy not installed."
                    )
                    return window
                
                csv_path = Path(save_dir) / f"{prefix}_events.csv"
                
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
                    
                    y_extent = coords[:, 1].max() - coords[:, 1].min() + 1
                    x_extent = coords[:, 2].max() - coords[:, 2].min() + 1
                    
                    events.append({
                        'instance_id': instance_id,
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
                
                df = pd.DataFrame(events)
                df.to_csv(csv_path, index=False)
                saved_files.append(str(csv_path))
            
            # Show completion message
            if saved_files:
                files_str = "\n".join([f"  - {Path(f).name}" for f in saved_files])
                QMessageBox.information(
                    None, "Export Complete",
                    f"Exported files to {save_dir}:\n{files_str}"
                )
                g.m.statusBar().showMessage(f"Exported {len(saved_files)} files")
            
        except Exception as e:
            import logging
            logging.error(f"Error exporting results: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Export Error",
                f"Error exporting results:\n{str(e)}"
            )
        
        return window


# ============================================================================
# Run Diagnostics Plugin
# ============================================================================

class CA_EventDetector_RunDiagnostics(BaseProcess_noPriorWindow):
    """
    Run comprehensive diagnostics on data, model, and system.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create diagnostics GUI."""
        self.gui_reset()
        
        # Diagnostic type
        from qtpy.QtWidgets import QComboBox
        
        diag_type = QComboBox()
        diag_type.addItem("Data Integrity Check")
        diag_type.addItem("Data Balance Analysis")
        diag_type.addItem("System Capabilities")
        diag_type.addItem("Model Checkpoint Inspection")
        diag_type.addItem("Training History Analysis")
        
        # Directory selection
        target_dir = QPushButton("Select Directory/File...")
        target_dir.clicked.connect(self._select_target)
        self._target = None
        
        self.items.append({'name': 'diag_type', 'string': 'Diagnostic Type',
                          'object': diag_type})
        self.items.append({'name': 'target_dir', 'string': 'Target',
                          'object': target_dir})
        
        super().gui()
    
    def _select_target(self):
        """Select target directory or file."""
        # First try directory
        directory = QFileDialog.getExistingDirectory(
            None, "Select Directory"
        )
        if directory:
            self._target = directory
            g.m.statusBar().showMessage(f"Target: {directory}")
        else:
            # Try file
            filename, _ = QFileDialog.getOpenFileName(
                None, "Select File",
                "", "Checkpoint Files (*.pth);;All Files (*)"
            )
            if filename:
                self._target = filename
                g.m.statusBar().showMessage(f"Target: {filename}")
    
    def __call__(self, diag_type, target_dir, keepSourceWindow=False):
        """Run diagnostics."""
        if self._target is None and diag_type != "System Capabilities":
            QMessageBox.warning(
                None, "No Target",
                "Please select a target directory or file."
            )
            return None
        
        try:
            # Create results window
            results_window = QWidget()
            results_window.setWindowTitle(f"Diagnostics: {diag_type}")
            results_window.resize(800, 600)
            
            layout = QVBoxLayout(results_window)
            
            results_text = QTextEdit()
            results_text.setReadOnly(True)
            layout.addWidget(results_text)
            
            # Run diagnostic
            results = []
            
            if diag_type == "Data Integrity Check":
                is_valid, corrupted = diagnostics.verify_data_integrity(
                    Path(self._target), num_samples=20
                )
                results.append(f"Data Integrity Check\n{'='*60}\n")
                results.append(f"Directory: {self._target}\n")
                results.append(f"Status: {'✅ PASSED' if is_valid else '❌ FAILED'}\n")
                if corrupted:
                    results.append(f"\nCorrupted files ({len(corrupted)}):\n")
                    for f in corrupted:
                        results.append(f"  - {Path(f).name}\n")
            
            elif diag_type == "Data Balance Analysis":
                stats = diagnostics.check_data_balance(Path(self._target))
                results.append(f"Data Balance Analysis\n{'='*60}\n")
                results.append(f"Directory: {self._target}\n")
                results.append(f"Total pixels: {stats.get('total_pixels', 0):,}\n")
                results.append(f"Number of files: {stats.get('num_files', 0)}\n\n")
                results.append(f"Class Distribution:\n")
                for class_name, pct in stats.get('class_percentages', {}).items():
                    results.append(f"  {class_name}: {pct:.2f}%\n")
                if 'imbalance_ratio' in stats:
                    results.append(f"\nImbalance ratio: {stats['imbalance_ratio']:.1f}:1\n")
            
            elif diag_type == "System Capabilities":
                caps = diagnostics.check_system_capabilities()
                results.append(f"System Capabilities\n{'='*60}\n")
                for key, value in caps.items():
                    results.append(f"{key}: {value}\n")
            
            elif diag_type == "Model Checkpoint Inspection":
                info = diagnostics.inspect_checkpoint(Path(self._target))
                results.append(f"Checkpoint Inspection\n{'='*60}\n")
                results.append(f"File: {Path(self._target).name}\n\n")
                for key, value in info.items():
                    results.append(f"{key}: {value}\n")
            
            elif diag_type == "Training History Analysis":
                analysis = diagnostics.analyze_training_history(Path(self._target))
                results.append(f"Training History Analysis\n{'='*60}\n")
                for key, value in analysis.items():
                    results.append(f"{key}: {value}\n")
            
            results_text.setText(''.join(results))
            results_window.show()
            
            g.m.statusBar().showMessage(f"Diagnostics complete: {diag_type}")
            
        except Exception as e:
            logging.error(f"Error running diagnostics: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Diagnostics Error",
                f"Error running diagnostics:\n{str(e)}"
            )
        
        return None


# ============================================================================
# Run Tests Plugin
# ============================================================================

class CA_EventDetector_RunTests(BaseProcess_noPriorWindow):
    """
    Run comprehensive tests on trained models.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create testing GUI."""
        self.gui_reset()
        
        # Test type
        from qtpy.QtWidgets import QComboBox
        
        test_type = QComboBox()
        test_type.addItem("Ground Truth Comparison")
        test_type.addItem("Threshold Optimization")
        test_type.addItem("Inference Speed Benchmark")
        test_type.addItem("Memory Profiling")
        
        # Model selection
        model_path = QPushButton("Select Model...")
        model_path.clicked.connect(self._select_model)
        self._model_path = None
        
        # Test data selection
        test_data_dir = QPushButton("Select Test Data...")
        test_data_dir.clicked.connect(self._select_test_data)
        self._test_data = None
        
        self.items.append({'name': 'test_type', 'string': 'Test Type',
                          'object': test_type})
        self.items.append({'name': 'model_path', 'string': 'Model File',
                          'object': model_path})
        self.items.append({'name': 'test_data_dir', 'string': 'Test Data',
                          'object': test_data_dir})
        
        super().gui()
    
    def _select_model(self):
        """Select model file."""
        filename, _ = QFileDialog.getOpenFileName(
            None, "Select Model",
            "", "Model Files (*.pth *.pt);;All Files (*)"
        )
        if filename:
            self._model_path = filename
            g.m.statusBar().showMessage(f"Model: {filename}")
    
    def _select_test_data(self):
        """Select test data directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Test Data Directory"
        )
        if directory:
            self._test_data = directory
            g.m.statusBar().showMessage(f"Test data: {directory}")
    
    def __call__(self, test_type, model_path, test_data_dir, keepSourceWindow=False):
        """Run tests."""
        if self._model_path is None:
            QMessageBox.warning(
                None, "No Model",
                "Please select a model file."
            )
            return None
        
        try:
            # Create progress dialog
            progress = QProgressDialog(f"Running {test_type}...", "Cancel", 0, 100)
            progress.setWindowTitle("Testing")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            # Run test
            results = None
            
            if test_type == "Ground Truth Comparison":
                if self._test_data is None:
                    QMessageBox.warning(None, "No Test Data", "Please select test data directory.")
                    return None
                
                progress.setLabelText("Comparing with ground truth...")
                progress.setValue(30)
                QApplication.processEvents()
                
                results = testing.test_with_ground_truth(
                    Path(self._model_path), Path(self._test_data)
                )
            
            elif test_type == "Threshold Optimization":
                if self._test_data is None:
                    QMessageBox.warning(None, "No Test Data", "Please select validation data directory.")
                    return None
                
                progress.setLabelText("Optimizing threshold...")
                progress.setValue(30)
                QApplication.processEvents()
                
                results = testing.optimize_threshold(
                    Path(self._model_path), Path(self._test_data)
                )
            
            elif test_type == "Inference Speed Benchmark":
                progress.setLabelText("Benchmarking inference speed...")
                progress.setValue(30)
                QApplication.processEvents()
                
                results = testing.benchmark_inference_speed(Path(self._model_path))
            
            elif test_type == "Memory Profiling":
                progress.setLabelText("Profiling memory usage...")
                progress.setValue(30)
                QApplication.processEvents()
                
                results = testing.profile_memory_usage(Path(self._model_path))
            
            progress.setValue(90)
            QApplication.processEvents()
            
            # Show results
            if results:
                results_window = QWidget()
                results_window.setWindowTitle(f"Test Results: {test_type}")
                results_window.resize(800, 600)
                
                layout = QVBoxLayout(results_window)
                
                results_text = QTextEdit()
                results_text.setReadOnly(True)
                
                import json
                results_text.setText(json.dumps(results, indent=2))
                
                layout.addWidget(results_text)
                results_window.show()
                
                g.m.statusBar().showMessage(f"Testing complete: {test_type}")
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            logging.error(f"Error running tests: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Testing Error",
                f"Error running tests:\n{str(e)}"
            )
        
        return None


# ============================================================================
# Enhanced Train Model Plugin (with robust features from train_patches_resume_mps.py)
# ============================================================================

class CA_EventDetector_TrainModel_Enhanced(BaseProcess_noPriorWindow):
    """
    Train calcium event detection model with robust error handling.
    
    Integrated features from train_patches_resume_mps.py:
    - Finds latest checkpoint by iteration number (not just interrupted.pth)
    - Verifies data integrity before training
    - Handles corrupted/iCloud stub files gracefully
    - Enhanced progress monitoring with ETA
    - Emergency checkpoint saving on errors
    - MPS optimization for Apple Silicon
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create enhanced training GUI."""
        self.gui_reset()
        
        # Data directory
        data_dir = QPushButton("Select Data Directory...")
        data_dir.clicked.connect(self._select_data_dir)
        self._data_dir = None
        
        # Output directory
        output_dir = QPushButton("Select Output Directory...")
        output_dir.clicked.connect(self._select_output_dir)
        self._output_dir = None
        
        # Resume from checkpoint option
        resume_training = CheckBox()
        resume_training.setChecked(True)
        resume_training.setToolTip("Automatically find and resume from latest checkpoint")
        
        # Training parameters
        num_iterations = QSpinBox()
        num_iterations.setRange(1000, 1000000)
        num_iterations.setValue(100000)
        num_iterations.setToolTip("Total training iterations")
        
        batch_size = QSpinBox()
        batch_size.setRange(1, 64)
        batch_size.setValue(4)
        batch_size.setToolTip("Batch size (4 recommended for MPS)")
        
        learning_rate = QDoubleSpinBox()
        learning_rate.setRange(0.00001, 0.1)
        learning_rate.setValue(0.001)
        learning_rate.setDecimals(5)
        learning_rate.setSingleStep(0.0001)
        
        # Device selection
        use_gpu = CheckBox()
        use_gpu.setChecked(True)
        use_gpu.setToolTip("Use GPU (CUDA/MPS) if available")
        
        # Validation frequency
        val_frequency = QSpinBox()
        val_frequency.setRange(100, 10000)
        val_frequency.setValue(500)
        val_frequency.setToolTip("Validate every N iterations")
        
        # Data verification
        verify_data = CheckBox()
        verify_data.setChecked(True)
        verify_data.setToolTip("Verify data integrity before training (recommended!)")
        
        self.items.append({'name': 'data_dir', 'string': 'Data Directory',
                          'object': data_dir})
        self.items.append({'name': 'output_dir', 'string': 'Output Directory',
                          'object': output_dir})
        self.items.append({'name': 'resume_training', 'string': 'Resume from Checkpoint',
                          'object': resume_training})
        self.items.append({'name': 'num_iterations', 'string': 'Training Iterations',
                          'object': num_iterations})
        self.items.append({'name': 'batch_size', 'string': 'Batch Size',
                          'object': batch_size})
        self.items.append({'name': 'learning_rate', 'string': 'Learning Rate',
                          'object': learning_rate})
        self.items.append({'name': 'use_gpu', 'string': 'Use GPU',
                          'object': use_gpu})
        self.items.append({'name': 'val_frequency', 'string': 'Validation Frequency',
                          'object': val_frequency})
        self.items.append({'name': 'verify_data', 'string': 'Verify Data Integrity',
                          'object': verify_data})
        
        super().gui()
    
    def _select_data_dir(self):
        """Select data directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Training Data Directory"
        )
        if directory:
            self._data_dir = directory
            g.m.statusBar().showMessage(f"Data directory: {directory}")
    
    def _select_output_dir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Output Directory"
        )
        if directory:
            self._output_dir = directory
            g.m.statusBar().showMessage(f"Output directory: {directory}")
    
    def __call__(self, data_dir, output_dir, resume_training, num_iterations,
                 batch_size, learning_rate, use_gpu, val_frequency, verify_data):
        """Train model with robust error handling."""
        if self._data_dir is None or self._output_dir is None:
            QMessageBox.warning(
                None, "Missing Directories",
                "Please select both data and output directories."
            )
            return None
        
        try:
            # Import modules
            try:
                from ca_event_detector.configs.config import Config
                from ca_event_detector.training.train import Trainer
            except ImportError as e:
                QMessageBox.critical(
                    None, "Import Error",
                    f"Could not import ca_event_detector:\n{str(e)}\n\n"
                    "Please ensure the package is installed:\n"
                    "pip install ca_event_detector"
                )
                return None
            
            # Setup MPS environment if needed
            if use_gpu:
                training_utils.setup_mps_environment()
            
            # Setup config
            config = Config()
            config.data.data_dir = self._data_dir
            config.training.num_iterations = num_iterations
            config.training.batch_size = batch_size
            config.training.learning_rate = learning_rate
            config.training.val_frequency = val_frequency
            
            # Determine device
            if use_gpu:
                import torch
                if torch.cuda.is_available():
                    config.training.device = 'cuda'
                elif torch.backends.mps.is_available():
                    config.training.device = 'mps'
                else:
                    config.training.device = 'cpu'
                    logging.warning("GPU requested but not available, using CPU")
            else:
                config.training.device = 'cpu'
            
            # Verify data integrity
            if verify_data:
                logging.info("Verifying data integrity...")
                is_valid, corrupted_files = diagnostics.verify_data_integrity(
                    Path(self._data_dir), num_samples=20
                )
                
                if not is_valid:
                    msg = (
                        f"Data integrity check failed!\n\n"
                        f"Found {len(corrupted_files)} corrupted or undownloaded files.\n\n"
                        f"This is likely an iCloud issue. Please:\n"
                        f"1. Open the data folder in Finder\n"
                        f"2. Right-click → 'Download Now'\n"
                        f"3. Wait for all files to download\n"
                        f"4. Re-run training"
                    )
                    QMessageBox.critical(None, "Data Integrity Error", msg)
                    return None
            
            # Find existing checkpoint if resuming
            checkpoint_path = None
            if resume_training:
                checkpoint_path = diagnostics.find_latest_checkpoint_by_iteration(
                    Path(self._output_dir)
                )
                
                if checkpoint_path:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    current_iteration = checkpoint.get('iteration', 0)
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    
                    logging.info(f"Found checkpoint at iteration {current_iteration:,}")
                    logging.info(f"Best validation loss: {best_val_loss:.4f}")
                    
                    config.training.resume_from = str(checkpoint_path)
                else:
                    logging.info("No checkpoint found, starting from scratch")
            
            # Estimate training time
            if checkpoint_path:
                estimate = training_utils.estimate_training_time(
                    config, checkpoint_path=Path(checkpoint_path)
                )
            else:
                estimate = training_utils.estimate_training_time(config)
            
            # Show confirmation
            msg = (
                f"Training Configuration:\n\n"
                f"Data: {Path(self._data_dir).name}\n"
                f"Output: {Path(self._output_dir).name}\n"
                f"Device: {config.training.device}\n"
                f"Batch size: {batch_size}\n"
                f"Total iterations: {num_iterations:,}\n"
                f"\nProgress:\n"
                f"Current iteration: {estimate['current_iteration']:,}\n"
                f"Remaining: {estimate['remaining_iterations']:,}\n"
                f"Estimated time: {estimate['estimated_time_formatted']}\n\n"
                f"Training will use robust error handling:\n"
                f"✓ Skips corrupted files\n"
                f"✓ Saves emergency checkpoints\n"
                f"✓ Enhanced progress monitoring\n\n"
                f"Start training?"
            )
            
            reply = QMessageBox.question(
                None, "Confirm Training",
                msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return None
            
            # Save config
            config_path = Path(self._output_dir) / "config.json"
            config.save(str(config_path))
            
            # Create robust dataloaders
            logging.info("Creating robust data loaders...")
            train_loader, val_loader = training_utils.create_robust_dataloaders(config)
            
            # Initialize trainer
            logging.info("Initializing trainer...")
            trainer = Trainer(config)
            
            # Run training with robust handling
            logging.info("Starting training...")
            g.m.statusBar().showMessage("Training model... (check console for progress)")
            
            training_utils.train_with_robust_handling(
                trainer, train_loader, val_loader, config, Path(self._output_dir)
            )
            
            QMessageBox.information(
                None, "Training Complete",
                f"Training complete!\n\nModel saved to: {self._output_dir}"
            )
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Training Error",
                f"Error during training:\n{str(e)}"
            )
        
        return None


# ============================================================================
# Old Train Model Plugin (kept for compatibility)
# ============================================================================

class CA_EventDetector_TrainModel(BaseProcess):
    """
    Train a new calcium event detection model.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create training GUI."""
        self.gui_reset()
        
        # Data directory
        data_dir = QPushButton("Select Data Directory...")
        data_dir.clicked.connect(self._select_data_dir)
        self._data_dir = None
        
        # Output directory
        output_dir = QPushButton("Select Output Directory...")
        output_dir.clicked.connect(self._select_output_dir)
        self._output_dir = None
        
        # Training parameters
        num_epochs = QSpinBox()
        num_epochs.setRange(1, 1000)
        num_epochs.setValue(100)
        
        batch_size = QSpinBox()
        batch_size.setRange(1, 64)
        batch_size.setValue(4)
        
        learning_rate = QDoubleSpinBox()
        learning_rate.setRange(0.00001, 0.1)
        learning_rate.setValue(0.001)
        learning_rate.setDecimals(5)
        learning_rate.setSingleStep(0.0001)
        
        use_gpu = CheckBox()
        use_gpu.setChecked(True)
        
        self.items.append({'name': 'data_dir', 'string': 'Data Directory',
                          'object': data_dir})
        self.items.append({'name': 'output_dir', 'string': 'Output Directory',
                          'object': output_dir})
        self.items.append({'name': 'num_epochs', 'string': 'Number of Epochs',
                          'object': num_epochs})
        self.items.append({'name': 'batch_size', 'string': 'Batch Size',
                          'object': batch_size})
        self.items.append({'name': 'learning_rate', 'string': 'Learning Rate',
                          'object': learning_rate})
        self.items.append({'name': 'use_gpu', 'string': 'Use GPU',
                          'object': use_gpu})
        
        super().gui()
    
    def _select_data_dir(self):
        """Select data directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Training Data Directory"
        )
        if directory:
            self._data_dir = directory
            g.m.statusBar().showMessage(f"Data directory: {directory}")
    
    def _select_output_dir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Output Directory"
        )
        if directory:
            self._output_dir = directory
            g.m.statusBar().showMessage(f"Output directory: {directory}")
    
    def __call__(self, data_dir, output_dir, num_epochs, batch_size,
                 learning_rate, use_gpu, keepSourceWindow=False):
        """Train model."""
        if self._data_dir is None or self._output_dir is None:
            QMessageBox.warning(
                None, "Missing Directories",
                "Please select both data and output directories."
            )
            return None
        
        try:
            # Import training module
            try:
                from ca_event_detector.training.train import train_model
                from ca_event_detector.configs.config import Config
            except ImportError as e:
                QMessageBox.critical(
                    None, "Import Error",
                    f"Could not import ca_event_detector:\n{str(e)}"
                )
                return None
            
            # Setup config
            config = Config()
            config.data.data_dir = self._data_dir
            config.training.num_epochs = num_epochs
            config.training.batch_size = batch_size
            config.training.learning_rate = learning_rate
            
            if use_gpu:
                import torch
                if torch.cuda.is_available():
                    config.device = 'cuda'
                elif torch.backends.mps.is_available():
                    config.device = 'mps'
                else:
                    config.device = 'cpu'
            else:
                config.device = 'cpu'
            
            # Show info
            msg = (
                f"Starting model training\n\n"
                f"Data: {self._data_dir}\n"
                f"Output: {self._output_dir}\n"
                f"Epochs: {num_epochs}\n"
                f"Batch size: {batch_size}\n"
                f"Learning rate: {learning_rate}\n"
                f"Device: {config.device}\n\n"
                f"This may take several hours. Continue?"
            )
            
            reply = QMessageBox.question(
                None, "Confirm Training",
                msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return None
            
            # Run training (this will be a long process)
            g.m.statusBar().showMessage("Training model... (check console for progress)")
            
            # Save config
            config_path = Path(self._output_dir) / "config.json"
            config.save(str(config_path))
            
            # Train model
            train_model(config, self._output_dir)
            
            QMessageBox.information(
                None, "Training Complete",
                f"Model training complete!\n\nModel saved to: {self._output_dir}"
            )
            
        except Exception as e:
            import logging
            logging.error(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Training Error",
                f"Error during training:\n{str(e)}"
            )
        
        return None


# ============================================================================
# Create plugin instances
# ============================================================================

ca_event_detector_view_results = CA_EventDetector_ViewResults()
ca_event_detector_export_results = CA_EventDetector_ExportResults()
ca_event_detector_run_diagnostics = CA_EventDetector_RunDiagnostics()
ca_event_detector_run_tests = CA_EventDetector_RunTests()
ca_event_detector_train_model = CA_EventDetector_TrainModel_Enhanced()  # Use enhanced version
ca_event_detector_train_model_simple = CA_EventDetector_TrainModel()  # Keep simple version


# ============================================================================
# Plugin Information
# ============================================================================

__all__ = [
    # Detection
    'ca_event_detector_run_detection',
    'ca_event_detector_quick_detection',
    # Display
    'ca_event_detector_display_results',
    'ca_event_detector_toggle_display',
    # Results
    'ca_event_detector_view_results',
    'ca_event_detector_export_results',
    # Diagnostics & Testing
    'ca_event_detector_run_diagnostics',
    'ca_event_detector_run_tests',
    # Training
    'ca_event_detector_train_model',
    'ca_event_detector_train_model_simple',
    # Classes
    'EventDisplay',
    'EventResultsViewer',
    # Modules
    'diagnostics',
    'testing',
    'training_utils'
]
