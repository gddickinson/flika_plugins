"""
Calcium Event Data Labeler for FLIKA
=====================================

A comprehensive FLIKA plugin for creating, managing, and validating training data
for calcium event detection models.

This plugin provides:
- Interactive manual labeling tools with ROI drawing
- Automated label generation based on signal analysis
- Label quality assessment and validation
- Patch extraction and augmentation
- Dataset creation and management
- Train/val/test splitting with stratification
- Format conversion and export
- Batch processing utilities

Perfect for creating high-quality training datasets for deep learning models!

Author: George (with Claude)
Date: 2024-12-26
"""

__version__ = '1.0.0'
__author__ = 'George'

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, WindowSelector, CheckBox, ComboBox, BaseProcess_noPriorWindow
from flika.process.file_ import save_file_gui, open_file_gui
from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QFileDialog, QMessageBox,
                            QProgressDialog, QSpinBox, QDoubleSpinBox, QTextEdit,
                            QTableWidget, QTableWidgetItem, QSlider, QGroupBox,
                            QRadioButton, QButtonGroup, QListWidget, QSplitter)
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QColor, QPen
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
from .interactive_labeler import InteractiveLabeler
from .automated_labeling import AutomatedLabeler
from .label_quality import LabelQualityChecker
from .dataset_creator import DatasetCreator
from .augmentation import AugmentationEngine
from . import utils


# ============================================================================
# Interactive Manual Labeling Plugin
# ============================================================================

class Ca_Labeler_ManualLabel(BaseProcess):
    """
    Interactive manual labeling tool for calcium events.
    
    Features:
    - Draw ROIs to define event regions
    - Assign classes (spark, puff, wave)
    - Frame-by-frame navigation
    - Undo/redo support
    - Save/load annotations
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create GUI for manual labeling."""
        self.gui_reset()
        
        # Window selector
        window = WindowSelector()
        
        # Output name
        self.output_name = "labeled_mask"
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window})
        
        super().gui()
        
        # Add custom buttons after standard GUI
        self.launch_button = QPushButton("Launch Interactive Labeler")
        self.launch_button.clicked.connect(self._launch_labeler)
        self.layout().addWidget(self.launch_button)
    
    def _launch_labeler(self):
        """Launch the interactive labeling interface."""
        window_name = self.getValue('window')
        
        if not window_name or window_name == 'None':
            QMessageBox.warning(None, "No Window", "Please select an image window.")
            return
        
        window = g.win
        if window is None:
            QMessageBox.warning(None, "No Window", "Could not find selected window.")
            return
        
        # Create interactive labeler
        labeler = InteractiveLabeler(window)
        labeler.show()
        
        # Store reference
        if not hasattr(g, 'ca_labelers'):
            g.ca_labelers = []
        g.ca_labelers.append(labeler)
        
        g.m.statusBar().showMessage("Interactive labeler launched!")
    
    def __call__(self, window, keepSourceWindow=False):
        """Manual labeling (launches interactive tool)."""
        # This is called if user clicks "Run" instead of "Launch Interactive Labeler"
        self._launch_labeler()
        return None


# ============================================================================
# Automated Label Generation Plugin
# ============================================================================

class Ca_Labeler_AutoLabel(BaseProcess):
    """
    Automated label generation using signal analysis.
    
    Creates initial labels based on:
    - Intensity thresholding
    - Temporal filtering
    - Morphological operations
    - Size and duration criteria
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create GUI for automated labeling."""
        self.gui_reset()
        
        # Window selector
        window = WindowSelector()
        
        # Intensity threshold
        threshold = QDoubleSpinBox()
        threshold.setRange(0.0, 1.0)
        threshold.setValue(0.3)
        threshold.setSingleStep(0.05)
        threshold.setDecimals(3)
        threshold.setToolTip("Intensity threshold (0-1, normalized)")
        
        # Temporal filter
        temporal_filter = QSpinBox()
        temporal_filter.setRange(1, 50)
        temporal_filter.setValue(3)
        temporal_filter.setToolTip("Temporal filter size (frames)")
        
        # Min event size
        min_size = QSpinBox()
        min_size.setRange(1, 1000)
        min_size.setValue(10)
        min_size.setToolTip("Minimum event size (pixels)")
        
        # Min duration
        min_duration = QSpinBox()
        min_duration.setRange(1, 100)
        min_duration.setValue(2)
        min_duration.setToolTip("Minimum event duration (frames)")
        
        # Auto-classify
        auto_classify = CheckBox()
        auto_classify.setChecked(True)
        auto_classify.setToolTip("Automatically classify events by size/duration")
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window})
        self.items.append({'name': 'threshold', 'string': 'Intensity Threshold',
                          'object': threshold})
        self.items.append({'name': 'temporal_filter', 'string': 'Temporal Filter',
                          'object': temporal_filter})
        self.items.append({'name': 'min_size', 'string': 'Min Size (pixels)',
                          'object': min_size})
        self.items.append({'name': 'min_duration', 'string': 'Min Duration (frames)',
                          'object': min_duration})
        self.items.append({'name': 'auto_classify', 'string': 'Auto-Classify Events',
                          'object': auto_classify})
        
        super().gui()
    
    def __call__(self, window, threshold, temporal_filter, min_size, 
                 min_duration, auto_classify, keepSourceWindow=False):
        """Generate automated labels."""
        window = g.win
        if window is None:
            g.m.statusBar().showMessage("No window selected")
            return None
        
        try:
            # Get image data
            image = window.image
            
            # Show progress
            progress = QProgressDialog("Generating automated labels...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            # Create automated labeler
            labeler = AutomatedLabeler()
            
            progress.setValue(30)
            QApplication.processEvents()
            
            # Generate labels
            labels = labeler.generate_labels(
                image,
                intensity_threshold=threshold,
                temporal_filter_size=temporal_filter,
                min_event_size=min_size,
                min_event_duration=min_duration,
                auto_classify=auto_classify
            )
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Create new window with labels
            from flika.process.binary import binary_to_labels
            label_window = Window(labels, name=f"{window.name}_auto_labels")
            
            # Store labeling parameters
            label_window.labeling_params = {
                'threshold': threshold,
                'temporal_filter': temporal_filter,
                'min_size': min_size,
                'min_duration': min_duration,
                'auto_classify': auto_classify
            }
            
            progress.setValue(100)
            progress.close()
            
            g.m.statusBar().showMessage(f"Generated automated labels: {np.unique(labels).size - 1} events")
            
            return label_window
            
        except Exception as e:
            logging.error(f"Error in automated labeling: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Automated Labeling Error",
                f"Error generating labels:\n{str(e)}"
            )
            return None


# ============================================================================
# Label Quality Assessment Plugin
# ============================================================================

class Ca_Labeler_CheckQuality(BaseProcess):
    """
    Assess quality of training labels.
    
    Checks:
    - Label completeness
    - Class distribution
    - Spatial coverage
    - Temporal coverage
    - Annotation consistency
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create GUI for quality checking."""
        self.gui_reset()
        
        # Data directory selection
        data_dir = QPushButton("Select Data Directory...")
        data_dir.clicked.connect(self._select_data_dir)
        self._data_dir = None
        
        # Quality metrics to compute
        from qtpy.QtWidgets import QCheckBox
        
        self.check_distribution = QCheckBox("Class Distribution")
        self.check_distribution.setChecked(True)
        
        self.check_coverage = QCheckBox("Spatial/Temporal Coverage")
        self.check_coverage.setChecked(True)
        
        self.check_consistency = QCheckBox("Label Consistency")
        self.check_consistency.setChecked(True)
        
        self.check_completeness = QCheckBox("Annotation Completeness")
        self.check_completeness.setChecked(True)
        
        self.items.append({'name': 'data_dir', 'string': 'Data Directory',
                          'object': data_dir})
        
        super().gui()
        
        # Add checkboxes to layout
        metrics_group = QGroupBox("Quality Metrics")
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.check_distribution)
        metrics_layout.addWidget(self.check_coverage)
        metrics_layout.addWidget(self.check_consistency)
        metrics_layout.addWidget(self.check_completeness)
        metrics_group.setLayout(metrics_layout)
        
        self.layout().addWidget(metrics_group)
    
    def _select_data_dir(self):
        """Select data directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Data Directory with Labels"
        )
        if directory:
            self._data_dir = directory
            g.m.statusBar().showMessage(f"Data directory: {directory}")
    
    def __call__(self, data_dir, keepSourceWindow=False):
        """Check label quality."""
        if self._data_dir is None:
            QMessageBox.warning(
                None, "No Directory",
                "Please select a data directory."
            )
            return None
        
        try:
            # Show progress
            progress = QProgressDialog("Assessing label quality...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            # Create quality checker
            checker = LabelQualityChecker()
            
            progress.setValue(30)
            QApplication.processEvents()
            
            # Run quality checks
            results = checker.assess_quality(
                Path(self._data_dir),
                check_distribution=self.check_distribution.isChecked(),
                check_coverage=self.check_coverage.isChecked(),
                check_consistency=self.check_consistency.isChecked(),
                check_completeness=self.check_completeness.isChecked()
            )
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Display results
            self._show_quality_results(results)
            
            progress.setValue(100)
            progress.close()
            
            g.m.statusBar().showMessage("Quality assessment complete")
            
        except Exception as e:
            logging.error(f"Error in quality check: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Quality Check Error",
                f"Error assessing quality:\n{str(e)}"
            )
        
        return None
    
    def _show_quality_results(self, results):
        """Display quality assessment results."""
        results_window = QWidget()
        results_window.setWindowTitle("Label Quality Assessment")
        results_window.resize(800, 600)
        
        layout = QVBoxLayout(results_window)
        
        results_text = QTextEdit()
        results_text.setReadOnly(True)
        
        # Format results
        text_parts = []
        text_parts.append("="*60 + "\n")
        text_parts.append("LABEL QUALITY ASSESSMENT RESULTS\n")
        text_parts.append("="*60 + "\n\n")
        
        for section, data in results.items():
            text_parts.append(f"{section.upper()}\n")
            text_parts.append("-"*60 + "\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    text_parts.append(f"{key}: {value}\n")
            else:
                text_parts.append(f"{data}\n")
            
            text_parts.append("\n")
        
        results_text.setText(''.join(text_parts))
        layout.addWidget(results_text)
        
        results_window.show()


# ============================================================================
# Dataset Creation Plugin
# ============================================================================

class Ca_Labeler_CreateDataset(BaseProcess_noPriorWindow):
    """
    Create training dataset from labeled data.
    
    Features:
    - Extract patches from full frames
    - Train/val/test splitting
    - Class balancing
    - Data augmentation
    - Format conversion
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create GUI for dataset creation."""
        self.gui_reset()
        
        # Input directory
        input_dir = QPushButton("Select Input Directory...")
        input_dir.clicked.connect(self._select_input_dir)
        self._input_dir = None
        
        # Output directory
        output_dir = QPushButton("Select Output Directory...")
        output_dir.clicked.connect(self._select_output_dir)
        self._output_dir = None
        
        # Patch size
        patch_size = QSpinBox()
        patch_size.setRange(16, 512)
        patch_size.setValue(64)
        patch_size.setToolTip("Size of extracted patches (pixels)")
        
        # Temporal length
        temporal_length = QSpinBox()
        temporal_length.setRange(8, 100)
        temporal_length.setValue(16)
        temporal_length.setToolTip("Temporal length of patches (frames)")
        
        # Train/val/test split
        train_ratio = QDoubleSpinBox()
        train_ratio.setRange(0.0, 1.0)
        train_ratio.setValue(0.7)
        train_ratio.setSingleStep(0.05)
        train_ratio.setDecimals(2)
        
        val_ratio = QDoubleSpinBox()
        val_ratio.setRange(0.0, 1.0)
        val_ratio.setValue(0.15)
        val_ratio.setSingleStep(0.05)
        val_ratio.setDecimals(2)
        
        # Augmentation
        augment = CheckBox()
        augment.setChecked(True)
        augment.setToolTip("Apply data augmentation")
        
        # Balance classes
        balance_classes = CheckBox()
        balance_classes.setChecked(True)
        balance_classes.setToolTip("Balance classes by oversampling")
        
        self.items.append({'name': 'input_dir', 'string': 'Input Directory',
                          'object': input_dir})
        self.items.append({'name': 'output_dir', 'string': 'Output Directory',
                          'object': output_dir})
        self.items.append({'name': 'patch_size', 'string': 'Patch Size',
                          'object': patch_size})
        self.items.append({'name': 'temporal_length', 'string': 'Temporal Length',
                          'object': temporal_length})
        self.items.append({'name': 'train_ratio', 'string': 'Train Ratio',
                          'object': train_ratio})
        self.items.append({'name': 'val_ratio', 'string': 'Validation Ratio',
                          'object': val_ratio})
        self.items.append({'name': 'augment', 'string': 'Apply Augmentation',
                          'object': augment})
        self.items.append({'name': 'balance_classes', 'string': 'Balance Classes',
                          'object': balance_classes})
        
        super().gui()
    
    def _select_input_dir(self):
        """Select input directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Input Directory with Labeled Data"
        )
        if directory:
            self._input_dir = directory
            g.m.statusBar().showMessage(f"Input directory: {directory}")
    
    def _select_output_dir(self):
        """Select output directory."""
        directory = QFileDialog.getExistingDirectory(
            None, "Select Output Directory for Dataset"
        )
        if directory:
            self._output_dir = directory
            g.m.statusBar().showMessage(f"Output directory: {directory}")
    
    def __call__(self, input_dir, output_dir, patch_size, temporal_length,
                 train_ratio, val_ratio, augment, balance_classes):
        """Create dataset."""
        if self._input_dir is None or self._output_dir is None:
            QMessageBox.warning(
                None, "Missing Directories",
                "Please select both input and output directories."
            )
            return None
        
        try:
            # Calculate test ratio
            test_ratio = 1.0 - train_ratio - val_ratio
            
            if test_ratio < 0:
                QMessageBox.warning(
                    None, "Invalid Split",
                    "Train + Val ratios must be <= 1.0"
                )
                return None
            
            # Show confirmation
            msg = (
                f"Create Dataset:\n\n"
                f"Input: {Path(self._input_dir).name}\n"
                f"Output: {Path(self._output_dir).name}\n"
                f"Patch size: {patch_size}x{patch_size}x{temporal_length}\n"
                f"Split: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}\n"
                f"Augmentation: {'Yes' if augment else 'No'}\n"
                f"Class balancing: {'Yes' if balance_classes else 'No'}\n\n"
                f"Proceed?"
            )
            
            reply = QMessageBox.question(
                None, "Confirm Dataset Creation",
                msg,
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return None
            
            # Show progress
            progress = QProgressDialog("Creating dataset...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(10)
            QApplication.processEvents()
            
            # Create dataset creator
            creator = DatasetCreator()
            
            progress.setValue(20)
            QApplication.processEvents()
            
            # Create dataset
            stats = creator.create_dataset(
                input_dir=Path(self._input_dir),
                output_dir=Path(self._output_dir),
                patch_size=(temporal_length, patch_size, patch_size),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                augment=augment,
                balance_classes=balance_classes,
                progress_callback=lambda p: (progress.setValue(20 + int(p * 70)), QApplication.processEvents())
            )
            
            progress.setValue(100)
            progress.close()
            
            # Show results
            result_msg = (
                f"Dataset created successfully!\n\n"
                f"Train samples: {stats.get('train_samples', 0)}\n"
                f"Val samples: {stats.get('val_samples', 0)}\n"
                f"Test samples: {stats.get('test_samples', 0)}\n"
                f"Total patches: {stats.get('total_patches', 0)}\n"
                f"\nOutput: {self._output_dir}"
            )
            
            QMessageBox.information(
                None, "Dataset Created",
                result_msg
            )
            
            g.m.statusBar().showMessage("Dataset creation complete")
            
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Dataset Creation Error",
                f"Error creating dataset:\n{str(e)}"
            )
        
        return None


# ============================================================================
# Data Augmentation Plugin
# ============================================================================

class Ca_Labeler_AugmentData(BaseProcess):
    """
    Apply augmentation to labeled data.
    
    Augmentations:
    - Rotation
    - Flipping
    - Elastic deformation
    - Intensity variations
    - Noise addition
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create GUI for augmentation."""
        self.gui_reset()
        
        # Window selectors
        image_window = WindowSelector()
        label_window = WindowSelector()
        
        # Augmentation options
        rotate = CheckBox()
        rotate.setChecked(True)
        
        flip = CheckBox()
        flip.setChecked(True)
        
        elastic = CheckBox()
        elastic.setChecked(False)
        
        intensity = CheckBox()
        intensity.setChecked(True)
        
        noise = CheckBox()
        noise.setChecked(False)
        
        # Number of augmented copies
        num_copies = QSpinBox()
        num_copies.setRange(1, 20)
        num_copies.setValue(5)
        num_copies.setToolTip("Number of augmented copies per image")
        
        self.items.append({'name': 'image_window', 'string': 'Image Window',
                          'object': image_window})
        self.items.append({'name': 'label_window', 'string': 'Label Window',
                          'object': label_window})
        self.items.append({'name': 'rotate', 'string': 'Rotation',
                          'object': rotate})
        self.items.append({'name': 'flip', 'string': 'Flipping',
                          'object': flip})
        self.items.append({'name': 'elastic', 'string': 'Elastic Deformation',
                          'object': elastic})
        self.items.append({'name': 'intensity', 'string': 'Intensity Variation',
                          'object': intensity})
        self.items.append({'name': 'noise', 'string': 'Add Noise',
                          'object': noise})
        self.items.append({'name': 'num_copies', 'string': 'Augmented Copies',
                          'object': num_copies})
        
        super().gui()
    
    def __call__(self, image_window, label_window, rotate, flip, elastic,
                 intensity, noise, num_copies, keepSourceWindow=False):
        """Apply augmentation."""
        # Implementation in augmentation module
        g.m.statusBar().showMessage("Augmentation feature - see Dataset Creator for full augmentation")
        return None


# ============================================================================
# Create plugin instances
# ============================================================================

ca_labeler_manual_label = Ca_Labeler_ManualLabel()
ca_labeler_auto_label = Ca_Labeler_AutoLabel()
ca_labeler_check_quality = Ca_Labeler_CheckQuality()
ca_labeler_create_dataset = Ca_Labeler_CreateDataset()
ca_labeler_augment_data = Ca_Labeler_AugmentData()


# ============================================================================
# Plugin Information
# ============================================================================

__all__ = [
    # Main tools
    'ca_labeler_manual_label',
    'ca_labeler_auto_label',
    'ca_labeler_check_quality',
    'ca_labeler_create_dataset',
    'ca_labeler_augment_data',
    # Classes
    'InteractiveLabeler',
    'AutomatedLabeler',
    'LabelQualityChecker',
    'DatasetCreator',
    'AugmentationEngine',
    # Modules
    'utils'
]
