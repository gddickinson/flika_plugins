#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcium Event Detector FLIKA Integration Module
================================================

This module integrates the ca_event_detector package with FLIKA,
providing GUI interfaces for calcium event detection and visualization.

Features:
- Run Detection with configurable parameters
- Quick Detection for fast processing
- Display Results with class and instance overlays
- Export detection results to CSV/TIFF
- Model training interface

Author: George Stuyt (with Claude)
Date: 2024-12-25
"""

import logging
import numpy as np
from typing import Optional, Dict
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QMessageBox, QFileDialog, QProgressDialog,
                           QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
                           QPushButton, QApplication)

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g
from distutils.version import StrictVersion

# Version-specific imports
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                         ComboBox, BaseProcess_noPriorWindow,
                                         WindowSelector, save_file_gui)
else:
    from flika.utils.BaseProcess import (BaseProcess, SliderLabel, CheckBox,
                                       ComboBox, BaseProcess_noPriorWindow,
                                       WindowSelector, save_file_gui)

# Set up logging
logger = logging.getLogger(__name__)


class CA_EventDetector_RunDetection(BaseProcess_noPriorWindow):
    """
    Run calcium event detection with full configuration options.
    
    Provides comprehensive GUI for configuring and running deep learning-based
    calcium event detection on FLIKA windows.
    """
    
    def __init__(self):
        super().__init__()
        self.detector = None
        self.results = None
    
    def gui(self):
        """Create comprehensive detection GUI."""
        self.gui_reset()
        
        # Window selector
        window_selector = WindowSelector()
        
        # === Model Configuration ===
        model_path = QPushButton("Select Model...")
        model_path.clicked.connect(self._select_model)
        self._model_path = None
        
        # === Detection Parameters ===
        probability_threshold = QDoubleSpinBox()
        probability_threshold.setRange(0.0, 1.0)
        probability_threshold.setValue(0.5)
        probability_threshold.setSingleStep(0.05)
        probability_threshold.setToolTip("Probability threshold for event detection")
        
        min_event_size = QSpinBox()
        min_event_size.setRange(1, 1000)
        min_event_size.setValue(5)
        min_event_size.setToolTip("Minimum event size in pixels")
        
        # Spark-specific parameters
        spark_min_size = QSpinBox()
        spark_min_size.setRange(1, 100)
        spark_min_size.setValue(5)
        
        spark_max_size = QSpinBox()
        spark_max_size.setRange(10, 500)
        spark_max_size.setValue(100)
        
        # Puff parameters
        puff_min_size = QSpinBox()
        puff_min_size.setRange(10, 500)
        puff_min_size.setValue(50)
        
        puff_max_size = QSpinBox()
        puff_max_size.setRange(50, 5000)
        puff_max_size.setValue(500)
        
        # Wave parameters
        wave_min_size = QSpinBox()
        wave_min_size.setRange(100, 10000)
        wave_min_size.setValue(500)
        
        # === Processing Options ===
        use_gpu = CheckBox()
        use_gpu.setChecked(True)
        use_gpu.setToolTip("Use GPU acceleration if available")
        
        batch_size = QSpinBox()
        batch_size.setRange(1, 64)
        batch_size.setValue(8)
        batch_size.setToolTip("Batch size for processing")
        
        # === Display Options ===
        show_results = CheckBox()
        show_results.setChecked(True)
        show_results.setToolTip("Display results after detection")
        
        display_mode = ComboBox()
        display_mode.addItem('Class')
        display_mode.addItem('Instance')
        display_mode.setValue('Class')
        
        # === Output Options ===
        save_masks = CheckBox()
        save_masks.setChecked(False)
        save_masks.setToolTip("Save class and instance masks to TIFF files")
        
        save_csv = CheckBox()
        save_csv.setChecked(False)
        save_csv.setToolTip("Save event properties to CSV file")
        
        # Add all items
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'model_path', 'string': 'Model File',
                          'object': model_path})
        
        self.items.append({'name': 'probability_threshold', 'string': 'Probability Threshold',
                          'object': probability_threshold})
        self.items.append({'name': 'min_event_size', 'string': 'Min Event Size (pixels)',
                          'object': min_event_size})
        
        self.items.append({'name': 'spark_min_size', 'string': 'Spark Min Size',
                          'object': spark_min_size})
        self.items.append({'name': 'spark_max_size', 'string': 'Spark Max Size',
                          'object': spark_max_size})
        
        self.items.append({'name': 'puff_min_size', 'string': 'Puff Min Size',
                          'object': puff_min_size})
        self.items.append({'name': 'puff_max_size', 'string': 'Puff Max Size',
                          'object': puff_max_size})
        
        self.items.append({'name': 'wave_min_size', 'string': 'Wave Min Size',
                          'object': wave_min_size})
        
        self.items.append({'name': 'use_gpu', 'string': 'Use GPU',
                          'object': use_gpu})
        self.items.append({'name': 'batch_size', 'string': 'Batch Size',
                          'object': batch_size})
        
        self.items.append({'name': 'show_results', 'string': 'Display Results',
                          'object': show_results})
        self.items.append({'name': 'display_mode', 'string': 'Display Mode',
                          'object': display_mode})
        
        self.items.append({'name': 'save_masks', 'string': 'Save Masks (TIFF)',
                          'object': save_masks})
        self.items.append({'name': 'save_csv', 'string': 'Save Results (CSV)',
                          'object': save_csv})
        
        super().gui()
    
    def _select_model(self):
        """Open file dialog to select model file."""
        filename, _ = QFileDialog.getOpenFileName(
            None, "Select Model File",
            "", "PyTorch Models (*.pth *.pt);;All Files (*)"
        )
        if filename:
            self._model_path = filename
            g.m.statusBar().showMessage(f"Model selected: {filename}")
    
    def __call__(self, window, model_path, probability_threshold, min_event_size,
                 spark_min_size, spark_max_size, puff_min_size, puff_max_size,
                 wave_min_size, use_gpu, batch_size, show_results, display_mode,
                 save_masks, save_csv, keepSourceWindow=False):
        """
        Run calcium event detection on image window.
        
        Parameters
        ----------
        window : Window
            FLIKA window containing calcium imaging data
        model_path : str
            Path to trained model file
        probability_threshold : float
            Detection threshold (0-1)
        min_event_size : int
            Minimum event size in pixels
        spark_min_size, spark_max_size : int
            Size range for sparks
        puff_min_size, puff_max_size : int
            Size range for puffs
        wave_min_size : int
            Minimum size for waves
        use_gpu : bool
            Use GPU acceleration
        batch_size : int
            Processing batch size
        show_results : bool
            Display results after detection
        display_mode : str
            'Class' or 'Instance'
        save_masks : bool
            Save masks to TIFF
        save_csv : bool
            Save results to CSV
        keepSourceWindow : bool
            Keep source window (not used)
        """
        try:
            # Check model path
            if self._model_path is None:
                QMessageBox.warning(
                    None, "No Model",
                    "Please select a model file first."
                )
                return window
            
            # Import ca_event_detector
            try:
                from ca_event_detector.inference.detect import CalciumEventDetector
                from ca_event_detector.configs.config import Config
            except ImportError as e:
                QMessageBox.critical(
                    None, "Import Error",
                    f"Could not import ca_event_detector:\n{str(e)}\n\n"
                    "Please ensure the package is installed:\n"
                    "pip install ca_event_detector"
                )
                return window
            
            # Get image data
            image = window.image
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            
            # Create progress dialog
            progress = QProgressDialog("Detecting calcium events...", "Cancel", 0, 100)
            progress.setWindowTitle("Event Detection")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # Update config with parameters
            config = Config()
            config.inference.probability_threshold = probability_threshold
            config.inference.min_event_size = min_event_size
            config.inference.spark_min_size = spark_min_size
            config.inference.spark_max_size = spark_max_size
            config.inference.puff_min_size = puff_min_size
            config.inference.puff_max_size = puff_max_size
            config.inference.wave_min_size = wave_min_size
            config.inference.batch_size = batch_size
            
            # Set device
            if use_gpu:
                import torch
                if torch.cuda.is_available():
                    config.device = 'cuda'
                elif torch.backends.mps.is_available():
                    config.device = 'mps'
                else:
                    config.device = 'cpu'
                    logger.warning("GPU requested but not available, using CPU")
            else:
                config.device = 'cpu'
            
            progress.setValue(10)
            QApplication.processEvents()
            
            # Create detector
            g.m.statusBar().showMessage("Loading model...")
            self.detector = CalciumEventDetector(self._model_path, config)
            
            progress.setValue(30)
            QApplication.processEvents()
            
            # Run detection
            g.m.statusBar().showMessage("Running detection...")
            self.results = self.detector.detect(image)
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Store results in window
            window.ca_event_results = self.results
            window.ca_event_config = config
            
            # Also store in global vars for access by other plugins
            g.ca_event_results = self.results
            
            # Get statistics
            class_mask = self.results['class_mask']
            instance_mask = self.results['instance_mask']
            
            n_sparks = np.sum(class_mask == 1)
            n_puffs = np.sum(class_mask == 2)
            n_waves = np.sum(class_mask == 3)
            n_instances = instance_mask.max()
            
            progress.setValue(90)
            QApplication.processEvents()
            
            # Save outputs if requested
            if save_masks or save_csv:
                save_dir = QFileDialog.getExistingDirectory(
                    None, "Select Output Directory"
                )
                if save_dir:
                    from tifffile import imwrite
                    import pandas as pd
                    
                    prefix = window.name.replace(' ', '_')
                    
                    if save_masks:
                        class_path = Path(save_dir) / f"{prefix}_class_mask.tif"
                        instance_path = Path(save_dir) / f"{prefix}_instance_mask.tif"
                        
                        imwrite(str(class_path), class_mask.astype(np.uint8))
                        imwrite(str(instance_path), instance_mask.astype(np.uint16))
                        
                        logger.info(f"Saved masks to {save_dir}")
                    
                    if save_csv:
                        # Extract event properties
                        csv_path = Path(save_dir) / f"{prefix}_events.csv"
                        event_data = self._extract_event_properties(class_mask, instance_mask)
                        event_data.to_csv(csv_path, index=False)
                        logger.info(f"Saved event data to {csv_path}")
            
            progress.setValue(95)
            QApplication.processEvents()
            
            # Display results if requested
            if show_results:
                try:
                    from .event_display import EventDisplay
                except ImportError:
                    from event_display import EventDisplay
                
                if not hasattr(window, 'ca_event_display'):
                    window.ca_event_display = EventDisplay(window)
                
                window.ca_event_display.set_events(class_mask, instance_mask)
                mode = 'class' if display_mode == 'Class' else 'instance'
                window.ca_event_display.show_overlay(mode)
            
            progress.setValue(100)
            progress.close()
            
            # Show completion message
            message = (
                f"Detection complete!\n\n"
                f"Detected events:\n"
                f"  Ca²⁺ sparks: {n_sparks:,} pixels\n"
                f"  Ca²⁺ puffs: {n_puffs:,} pixels\n"
                f"  Ca²⁺ waves: {n_waves:,} pixels\n"
                f"  Total instances: {n_instances}"
            )
            
            g.m.statusBar().showMessage(
                f"Detected: {n_sparks} sparks, {n_puffs} puffs, {n_waves} waves"
            )
            
            QMessageBox.information(None, "Detection Complete", message)
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Detection Error",
                f"Error during detection:\n{str(e)}"
            )
        
        return window
    
    def _extract_event_properties(self, class_mask: np.ndarray,
                                  instance_mask: np.ndarray) -> 'pd.DataFrame':
        """
        Extract properties of detected events.
        
        Parameters
        ----------
        class_mask : ndarray
            Class labels
        instance_mask : ndarray
            Instance IDs
            
        Returns
        -------
        df : DataFrame
            Event properties
        """
        import pandas as pd
        from scipy import ndimage
        
        events = []
        
        for instance_id in range(1, instance_mask.max() + 1):
            mask = instance_mask == instance_id
            if not np.any(mask):
                continue
            
            # Get class
            class_id = int(class_mask[mask][0])
            class_names = ['background', 'spark', 'puff', 'wave']
            class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
            
            # Get properties
            size = np.sum(mask)
            coords = np.argwhere(mask)
            
            t_mean = coords[:, 0].mean()
            y_mean = coords[:, 1].mean()
            x_mean = coords[:, 2].mean()
            
            t_min, t_max = coords[:, 0].min(), coords[:, 0].max()
            duration = t_max - t_min + 1
            
            events.append({
                'instance_id': instance_id,
                'class': class_name,
                'class_id': class_id,
                'size_pixels': size,
                't_center': t_mean,
                'y_center': y_mean,
                'x_center': x_mean,
                'frame_start': t_min,
                'frame_end': t_max,
                'duration_frames': duration
            })
        
        return pd.DataFrame(events)


class CA_EventDetector_QuickDetection(BaseProcess_noPriorWindow):
    """
    Quick calcium event detection with minimal configuration.
    
    Simplified interface for fast event detection with default parameters.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create quick detection GUI."""
        self.gui_reset()
        
        window_selector = WindowSelector()
        
        model_path = QPushButton("Select Model...")
        model_path.clicked.connect(self._select_model)
        self._model_path = None
        
        show_results = CheckBox()
        show_results.setChecked(True)
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'model_path', 'string': 'Model File',
                          'object': model_path})
        self.items.append({'name': 'show_results', 'string': 'Display Results',
                          'object': show_results})
        
        super().gui()
    
    def _select_model(self):
        """Open file dialog to select model file."""
        filename, _ = QFileDialog.getOpenFileName(
            None, "Select Model File",
            "", "PyTorch Models (*.pth *.pt);;All Files (*)"
        )
        if filename:
            self._model_path = filename
            g.m.statusBar().showMessage(f"Model selected: {filename}")
    
    def __call__(self, window, model_path, show_results, keepSourceWindow=False):
        """Run quick detection with default parameters."""
        try:
            if self._model_path is None:
                QMessageBox.warning(
                    None, "No Model",
                    "Please select a model file first."
                )
                return window
            
            # Import ca_event_detector
            try:
                from ca_event_detector.inference.detect import CalciumEventDetector
            except ImportError as e:
                QMessageBox.critical(
                    None, "Import Error",
                    f"Could not import ca_event_detector:\n{str(e)}"
                )
                return window
            
            # Get image
            image = window.image
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            
            # Create progress
            progress = QProgressDialog("Running quick detection...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress.setValue(20)
            QApplication.processEvents()
            
            # Create detector with defaults
            g.m.statusBar().showMessage("Loading model...")
            detector = CalciumEventDetector(self._model_path)
            
            progress.setValue(40)
            QApplication.processEvents()
            
            # Run detection
            g.m.statusBar().showMessage("Detecting events...")
            results = detector.detect(image)
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Store results
            window.ca_event_results = results
            g.ca_event_results = results
            
            # Display if requested
            if show_results:
                try:
                    from .event_display import EventDisplay
                except ImportError:
                    from event_display import EventDisplay
                
                if not hasattr(window, 'ca_event_display'):
                    window.ca_event_display = EventDisplay(window)
                
                window.ca_event_display.set_events(
                    results['class_mask'],
                    results['instance_mask']
                )
                window.ca_event_display.show_overlay('class')
            
            progress.setValue(100)
            progress.close()
            
            # Show stats
            class_mask = results['class_mask']
            n_sparks = np.sum(class_mask == 1)
            n_puffs = np.sum(class_mask == 2)
            n_waves = np.sum(class_mask == 3)
            
            g.m.statusBar().showMessage(
                f"Detected: {n_sparks} sparks, {n_puffs} puffs, {n_waves} waves"
            )
            
        except Exception as e:
            logger.error(f"Error in quick detection: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                None, "Detection Error",
                f"Error during detection:\n{str(e)}"
            )
        
        return window


class CA_EventDetector_ToggleDisplay(BaseProcess):
    """Toggle event overlay display on/off."""
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create toggle GUI."""
        self.gui_reset()
        
        window_selector = WindowSelector()
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        
        super().gui()
    
    def __call__(self, window, keepSourceWindow=False):
        """Toggle display."""
        try:
            if not hasattr(window, 'ca_event_display'):
                QMessageBox.warning(
                    None, "No Display",
                    "No event display found for this window."
                )
                return window
            
            window.ca_event_display.toggle_overlay()
            
            if window.ca_event_display.overlayVisible:
                g.m.statusBar().showMessage("Event overlay shown")
            else:
                g.m.statusBar().showMessage("Event overlay hidden")
            
        except Exception as e:
            logger.error(f"Error toggling display: {e}")
        
        return window


# ============================================================================
# Create plugin instances
# ============================================================================

ca_event_detector_run_detection = CA_EventDetector_RunDetection()
ca_event_detector_quick_detection = CA_EventDetector_QuickDetection()
ca_event_detector_toggle_display = CA_EventDetector_ToggleDisplay()
