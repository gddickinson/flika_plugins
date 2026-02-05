#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Labeler for Calcium Events
=======================================

Interactive GUI for manual labeling of calcium events in time-series data.

Features:
- Frame-by-frame navigation with slider
- ROI drawing tools (rectangle, ellipse, polygon, freehand)
- Class selection (background, spark, puff, wave)
- Undo/redo functionality
- Save/load annotations in multiple formats
- Overlay visualization
- Keyboard shortcuts

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from qtpy.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QSlider, QComboBox, QSpinBox,
                            QGroupBox, QRadioButton, QButtonGroup, QFileDialog,
                            QMessageBox, QToolBar, QAction, QStatusBar)
from qtpy.QtCore import Qt, Signal, QPoint, QRect
from qtpy.QtGui import QColor, QPen, QPainter, QPixmap, QImage, QKeySequence

from flika import global_vars as g
from flika.window import Window

logger = logging.getLogger(__name__)


class InteractiveLabeler(QMainWindow):
    """
    Interactive labeling interface for calcium events.
    
    Allows frame-by-frame manual annotation of calcium events
    with multiple drawing tools and class selection.
    """
    
    # Class definitions
    CLASS_LABELS = {
        0: 'background',
        1: 'spark',
        2: 'puff',
        3: 'wave'
    }
    
    CLASS_COLORS = {
        0: QColor(0, 0, 0, 0),           # Background - transparent
        1: QColor(0, 255, 0, 128),        # Spark - green
        2: QColor(255, 165, 0, 128),      # Puff - orange
        3: QColor(255, 0, 0, 128)         # Wave - red
    }
    
    def __init__(self, image_window: Window):
        super().__init__()
        
        self.image_window = image_window
        self.image = image_window.image
        self.T, self.H, self.W = self.image.shape
        
        # Initialize label mask
        self.labels = np.zeros((self.T, self.H, self.W), dtype=np.uint8)
        
        # Current state
        self.current_frame = 0
        self.current_class = 1  # Default to spark
        self.draw_tool = 'rectangle'  # rectangle, ellipse, polygon, freehand
        
        # Drawing state
        self.drawing = False
        self.draw_start = None
        self.draw_points = []
        
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        
        self.setup_ui()
        self.update_display()
    
    def setup_ui(self):
        """Setup user interface."""
        self.setWindowTitle(f"Interactive Labeler - {self.image_window.name}")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self.create_toolbar()
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        # Frame navigation
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout()
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, self.T - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.on_frame_changed)
        
        self.frame_label = QLabel(f"Frame: 0 / {self.T - 1}")
        
        nav_layout.addWidget(self.frame_label)
        nav_layout.addWidget(self.frame_slider)
        nav_group.setLayout(nav_layout)
        controls_layout.addWidget(nav_group)
        
        # Class selection
        class_group = QGroupBox("Event Class")
        class_layout = QVBoxLayout()
        
        self.class_buttons = QButtonGroup()
        
        for class_id, class_name in self.CLASS_LABELS.items():
            if class_id == 0:
                continue  # Skip background
            
            radio = QRadioButton(class_name.capitalize())
            radio.setStyleSheet(f"QRadioButton {{ color: {self.CLASS_COLORS[class_id].name()}; }}")
            self.class_buttons.addButton(radio, class_id)
            class_layout.addWidget(radio)
            
            if class_id == 1:
                radio.setChecked(True)
        
        self.class_buttons.buttonClicked.connect(self.on_class_changed)
        class_group.setLayout(class_layout)
        controls_layout.addWidget(class_group)
        
        # Drawing tools
        tool_group = QGroupBox("Drawing Tools")
        tool_layout = QVBoxLayout()
        
        self.tool_buttons = QButtonGroup()
        tools = [
            ('rectangle', 'Rectangle'),
            ('ellipse', 'Ellipse'),
            ('polygon', 'Polygon'),
            ('freehand', 'Freehand'),
            ('erase', 'Erase')
        ]
        
        for tool_id, tool_name in tools:
            radio = QRadioButton(tool_name)
            self.tool_buttons.addButton(radio)
            radio.clicked.connect(lambda checked, t=tool_id: self.set_draw_tool(t))
            tool_layout.addWidget(radio)
            
            if tool_id == 'rectangle':
                radio.setChecked(True)
        
        tool_group.setLayout(tool_layout)
        controls_layout.addWidget(tool_group)
        
        main_layout.addLayout(controls_layout)
        
        # Image display area
        display_layout = QHBoxLayout()
        
        # Original image view
        orig_group = QGroupBox("Original Image")
        orig_layout = QVBoxLayout()
        self.original_label = QLabel()
        self.original_label.setMinimumSize(400, 400)
        orig_layout.addWidget(self.original_label)
        orig_group.setLayout(orig_layout)
        display_layout.addWidget(orig_group)
        
        # Labeled image view
        label_group = QGroupBox("Labels Overlay")
        label_layout = QVBoxLayout()
        self.overlay_label = QLabel()
        self.overlay_label.setMinimumSize(400, 400)
        self.overlay_label.setMouseTracking(True)
        self.overlay_label.mousePressEvent = self.on_mouse_press
        self.overlay_label.mouseMoveEvent = self.on_mouse_move
        self.overlay_label.mouseReleaseEvent = self.on_mouse_release
        label_layout.addWidget(self.overlay_label)
        label_group.setLayout(label_layout)
        display_layout.addWidget(label_group)
        
        main_layout.addLayout(display_layout)
        
        # Bottom controls
        bottom_layout = QHBoxLayout()
        
        # Statistics
        self.stats_label = QLabel("Events: 0")
        bottom_layout.addWidget(self.stats_label)
        
        bottom_layout.addStretch()
        
        # Action buttons
        save_btn = QPushButton("Save Labels")
        save_btn.clicked.connect(self.save_labels)
        bottom_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Labels")
        load_btn.clicked.connect(self.load_labels)
        bottom_layout.addWidget(load_btn)
        
        export_btn = QPushButton("Export to Window")
        export_btn.clicked.connect(self.export_to_window)
        bottom_layout.addWidget(export_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready - Draw ROIs on the overlay image")
    
    def create_toolbar(self):
        """Create toolbar with actions."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Undo
        undo_action = QAction("Undo", self)
        undo_action.setShortcut(QKeySequence.Undo)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)
        
        # Redo
        redo_action = QAction("Redo", self)
        redo_action.setShortcut(QKeySequence.Redo)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)
        
        toolbar.addSeparator()
        
        # Clear frame
        clear_action = QAction("Clear Frame", self)
        clear_action.triggered.connect(self.clear_frame)
        toolbar.addAction(clear_action)
        
        # Clear all
        clear_all_action = QAction("Clear All", self)
        clear_all_action.triggered.connect(self.clear_all)
        toolbar.addAction(clear_all_action)
    
    def on_frame_changed(self, value):
        """Handle frame slider change."""
        self.current_frame = value
        self.frame_label.setText(f"Frame: {value} / {self.T - 1}")
        self.update_display()
    
    def on_class_changed(self):
        """Handle class selection change."""
        self.current_class = self.class_buttons.checkedId()
        logger.info(f"Selected class: {self.CLASS_LABELS[self.current_class]}")
    
    def set_draw_tool(self, tool: str):
        """Set current drawing tool."""
        self.draw_tool = tool
        logger.info(f"Selected tool: {tool}")
        self.statusBar.showMessage(f"Tool: {tool}")
    
    def update_display(self):
        """Update image displays."""
        # Get current frame
        frame = self.image[self.current_frame]
        
        # Normalize to 0-255
        frame_norm = ((frame - frame.min()) / (frame.max() - frame.min() + 1e-8) * 255).astype(np.uint8)
        
        # Convert to QImage
        height, width = frame_norm.shape
        bytes_per_line = width
        
        # Original image
        q_img = QImage(frame_norm.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.original_label.setPixmap(pixmap.scaled(
            self.original_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Overlay image
        overlay = self.create_overlay(frame_norm)
        self.overlay_label.setPixmap(overlay.scaled(
            self.overlay_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        
        # Update statistics
        self.update_statistics()
    
    def create_overlay(self, base_image):
        """Create overlay with labels."""
        # Convert grayscale to RGB
        rgb = np.stack([base_image] * 3, axis=-1)
        
        # Get current labels
        label_mask = self.labels[self.current_frame]
        
        # Apply colored overlays
        for class_id in [1, 2, 3]:  # Skip background
            mask = label_mask == class_id
            if np.any(mask):
                color = self.CLASS_COLORS[class_id]
                rgb[mask] = (
                    rgb[mask] * (1 - color.alphaF()) + 
                    np.array([color.red(), color.green(), color.blue()]) * color.alphaF()
                ).astype(np.uint8)
        
        # Convert to QImage
        height, width = rgb.shape[:2]
        bytes_per_line = width * 3
        q_img = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_img)
    
    def on_mouse_press(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            pos = self.get_image_coordinates(event.pos())
            self.draw_start = pos
            self.draw_points = [pos]
            
            # Save state for undo
            self.save_state()
    
    def on_mouse_move(self, event):
        """Handle mouse move."""
        if self.drawing:
            pos = self.get_image_coordinates(event.pos())
            
            if self.draw_tool == 'freehand':
                self.draw_points.append(pos)
                # Draw incrementally
                self.draw_line(self.draw_points[-2], pos)
                self.update_display()
            
            # Update cursor position
            self.statusBar.showMessage(f"Position: ({pos.x()}, {pos.y()})")
    
    def on_mouse_release(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            pos = self.get_image_coordinates(event.pos())
            
            if self.draw_tool == 'rectangle':
                self.draw_rectangle(self.draw_start, pos)
            elif self.draw_tool == 'ellipse':
                self.draw_ellipse(self.draw_start, pos)
            elif self.draw_tool == 'polygon':
                self.draw_points.append(pos)
                # Polygon completes on double-click (simplified: complete on release)
                self.draw_polygon(self.draw_points)
            
            self.update_display()
    
    def get_image_coordinates(self, widget_pos: QPoint) -> QPoint:
        """Convert widget coordinates to image coordinates."""
        # Get pixmap size and widget size
        pixmap = self.overlay_label.pixmap()
        if pixmap is None:
            return QPoint(0, 0)
        
        widget_size = self.overlay_label.size()
        pixmap_size = pixmap.size()
        
        # Calculate scaling
        scale_x = self.W / pixmap_size.width()
        scale_y = self.H / pixmap_size.height()
        
        # Calculate offset (pixmap is centered)
        offset_x = (widget_size.width() - pixmap_size.width()) / 2
        offset_y = (widget_size.height() - pixmap_size.height()) / 2
        
        # Convert coordinates
        image_x = int((widget_pos.x() - offset_x) * scale_x)
        image_y = int((widget_pos.y() - offset_y) * scale_y)
        
        # Clamp to image bounds
        image_x = max(0, min(image_x, self.W - 1))
        image_y = max(0, min(image_y, self.H - 1))
        
        return QPoint(image_x, image_y)
    
    def draw_rectangle(self, start: QPoint, end: QPoint):
        """Draw rectangle ROI."""
        x1, y1 = min(start.x(), end.x()), min(start.y(), end.y())
        x2, y2 = max(start.x(), end.x()), max(start.y(), end.y())
        
        value = self.current_class if self.draw_tool != 'erase' else 0
        self.labels[self.current_frame, y1:y2+1, x1:x2+1] = value
    
    def draw_ellipse(self, start: QPoint, end: QPoint):
        """Draw ellipse ROI."""
        center_x = (start.x() + end.x()) / 2
        center_y = (start.y() + end.y()) / 2
        radius_x = abs(end.x() - start.x()) / 2
        radius_y = abs(end.y() - start.y()) / 2
        
        # Create coordinate grids
        y, x = np.ogrid[:self.H, :self.W]
        
        # Ellipse equation
        mask = ((x - center_x)**2 / (radius_x**2 + 1e-8) + 
                (y - center_y)**2 / (radius_y**2 + 1e-8)) <= 1
        
        value = self.current_class if self.draw_tool != 'erase' else 0
        self.labels[self.current_frame][mask] = value
    
    def draw_polygon(self, points: List[QPoint]):
        """Draw polygon ROI."""
        if len(points) < 3:
            return
        
        # Create polygon mask using scanline algorithm
        from skimage.draw import polygon
        
        y_coords = [p.y() for p in points]
        x_coords = [p.x() for p in points]
        
        rr, cc = polygon(y_coords, x_coords, shape=(self.H, self.W))
        
        value = self.current_class if self.draw_tool != 'erase' else 0
        self.labels[self.current_frame, rr, cc] = value
    
    def draw_line(self, start: QPoint, end: QPoint):
        """Draw line (for freehand)."""
        from skimage.draw import line
        
        rr, cc = line(start.y(), start.x(), end.y(), end.x())
        
        # Make line thicker
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                rr_thick = np.clip(rr + dy, 0, self.H - 1)
                cc_thick = np.clip(cc + dx, 0, self.W - 1)
                
                value = self.current_class if self.draw_tool != 'erase' else 0
                self.labels[self.current_frame, rr_thick, cc_thick] = value
    
    def save_state(self):
        """Save current state for undo."""
        self.undo_stack.append(self.labels.copy())
        self.redo_stack.clear()
        
        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
    
    def undo(self):
        """Undo last action."""
        if self.undo_stack:
            self.redo_stack.append(self.labels.copy())
            self.labels = self.undo_stack.pop()
            self.update_display()
            self.statusBar.showMessage("Undo")
    
    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            self.undo_stack.append(self.labels.copy())
            self.labels = self.redo_stack.pop()
            self.update_display()
            self.statusBar.showMessage("Redo")
    
    def clear_frame(self):
        """Clear current frame."""
        reply = QMessageBox.question(
            self, "Clear Frame",
            "Clear all labels on current frame?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_state()
            self.labels[self.current_frame] = 0
            self.update_display()
    
    def clear_all(self):
        """Clear all labels."""
        reply = QMessageBox.question(
            self, "Clear All",
            "Clear all labels on ALL frames?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.save_state()
            self.labels[:] = 0
            self.update_display()
    
    def update_statistics(self):
        """Update statistics display."""
        # Count events per class
        unique, counts = np.unique(self.labels, return_counts=True)
        stats_dict = dict(zip(unique, counts))
        
        stats_text = "Events: "
        for class_id in [1, 2, 3]:
            count = stats_dict.get(class_id, 0)
            name = self.CLASS_LABELS[class_id]
            stats_text += f"{name}={count} "
        
        self.stats_label.setText(stats_text)
    
    def save_labels(self):
        """Save labels to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Labels",
            "",
            "TIFF Files (*.tif *.tiff);;NumPy Files (*.npy);;JSON Files (*.json)"
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    np.save(filename, self.labels)
                elif filename.endswith('.json'):
                    # Save as JSON with metadata
                    data = {
                        'labels': self.labels.tolist(),
                        'shape': self.labels.shape,
                        'class_names': self.CLASS_LABELS
                    }
                    with open(filename, 'w') as f:
                        json.dump(data, f)
                else:
                    # Save as TIFF
                    from tifffile import imwrite
                    imwrite(filename, self.labels)
                
                self.statusBar.showMessage(f"Labels saved to {filename}")
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Save Error",
                    f"Error saving labels:\n{str(e)}"
                )
    
    def load_labels(self):
        """Load labels from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Labels",
            "",
            "TIFF Files (*.tif *.tiff);;NumPy Files (*.npy);;JSON Files (*.json)"
        )
        
        if filename:
            try:
                if filename.endswith('.npy'):
                    labels = np.load(filename)
                elif filename.endswith('.json'):
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    labels = np.array(data['labels'], dtype=np.uint8)
                else:
                    from tifffile import imread
                    labels = imread(filename)
                
                # Verify shape
                if labels.shape == self.labels.shape:
                    self.save_state()
                    self.labels = labels
                    self.update_display()
                    self.statusBar.showMessage(f"Labels loaded from {filename}")
                else:
                    QMessageBox.warning(
                        self, "Shape Mismatch",
                        f"Loaded labels shape {labels.shape} doesn't match image shape {self.labels.shape}"
                    )
                
            except Exception as e:
                QMessageBox.critical(
                    self, "Load Error",
                    f"Error loading labels:\n{str(e)}"
                )
    
    def export_to_window(self):
        """Export labels to FLIKA window."""
        try:
            label_window = Window(self.labels, name=f"{self.image_window.name}_labels")
            self.statusBar.showMessage("Labels exported to new window")
            
            QMessageBox.information(
                self, "Export Complete",
                f"Labels exported to window: {label_window.name}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Error exporting labels:\n{str(e)}"
            )
