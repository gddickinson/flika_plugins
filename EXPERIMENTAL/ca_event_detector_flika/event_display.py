#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Display Module for Calcium Event Detector
================================================

Provides visualization of detected calcium events overlaid on FLIKA windows.

Author: George Stuyt (with Claude)
Date: 2024-12-25
"""

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from typing import Dict, Optional, Tuple
import logging

# FLIKA imports
import flika
from flika.window import Window
import flika.global_vars as g
from distutils.version import StrictVersion

# Version-specific imports
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector

logger = logging.getLogger(__name__)


class EventDisplay:
    """
    Display detected calcium events on FLIKA windows.
    
    Visualizes class masks (sparks, puffs, waves) and instance masks
    with color-coded overlays.
    """
    
    # Default colors for each event type
    COLORS = {
        'background': (0, 0, 0, 0),          # Transparent
        'spark': (0, 255, 0, 180),           # Green
        'puff': (255, 165, 0, 180),          # Orange  
        'wave': (255, 0, 0, 180),            # Red
        'instance': None                      # Will use colormap
    }
    
    def __init__(self, window: Window):
        """
        Initialize event display for a FLIKA window.
        
        Parameters
        ----------
        window : Window
            FLIKA window to display events on
        """
        self.window = window
        self.image_item = None
        self.class_overlay = None
        self.instance_overlay = None
        
        self.class_mask = None
        self.instance_mask = None
        
        self.display_mode = 'class'  # 'class' or 'instance'
        self.overlayVisible = False
        
        # Get the image item from the window
        if hasattr(window, 'imageview'):
            self.image_item = window.imageview.imageItem
        else:
            logger.warning("Window does not have imageview attribute")
    
    def set_events(self, class_mask: np.ndarray, instance_mask: np.ndarray):
        """
        Set the event masks to display.
        
        Parameters
        ----------
        class_mask : ndarray
            (T, H, W) array with class labels (0=bg, 1=spark, 2=puff, 3=wave)
        instance_mask : ndarray
            (T, H, W) array with instance IDs
        """
        self.class_mask = class_mask
        self.instance_mask = instance_mask
        
        # Create overlay images
        self._create_overlays()
    
    def _create_overlays(self):
        """Create RGBA overlay images from masks."""
        if self.class_mask is None:
            return
        
        T, H, W = self.class_mask.shape
        
        # Create class overlay
        self.class_overlay = np.zeros((T, H, W, 4), dtype=np.uint8)
        
        # Apply colors for each class
        for class_id, color in enumerate(self.COLORS.values()):
            if color is None or class_id == 0:  # Skip background
                continue
            mask = self.class_mask == class_id
            self.class_overlay[mask] = color
        
        # Create instance overlay with colormap
        if self.instance_mask is not None and self.instance_mask.max() > 0:
            self.instance_overlay = self._create_instance_overlay()
    
    def _create_instance_overlay(self) -> np.ndarray:
        """
        Create colored overlay for instance mask using distinct colors.
        
        Returns
        -------
        overlay : ndarray
            (T, H, W, 4) RGBA overlay
        """
        T, H, W = self.instance_mask.shape
        overlay = np.zeros((T, H, W, 4), dtype=np.uint8)
        
        # Get unique instance IDs (excluding background)
        instance_ids = np.unique(self.instance_mask)
        instance_ids = instance_ids[instance_ids > 0]
        
        # Generate distinct colors for each instance
        n_instances = len(instance_ids)
        if n_instances == 0:
            return overlay
        
        # Use HSV to generate distinct hues
        hues = np.linspace(0, 360, n_instances, endpoint=False)
        
        for i, instance_id in enumerate(instance_ids):
            mask = self.instance_mask == instance_id
            color = self._hsv_to_rgb(hues[i], 0.8, 0.9, alpha=180)
            overlay[mask] = color
        
        return overlay
    
    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float, alpha: int = 255) -> Tuple[int, int, int, int]:
        """
        Convert HSV to RGB color.
        
        Parameters
        ----------
        h : float
            Hue (0-360)
        s : float
            Saturation (0-1)
        v : float
            Value (0-1)
        alpha : int
            Alpha channel (0-255)
            
        Returns
        -------
        rgba : tuple
            (R, G, B, A) values (0-255)
        """
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
        return (r, g, b, alpha)
    
    def show_overlay(self, mode: str = 'class'):
        """
        Display overlay on the window.
        
        Parameters
        ----------
        mode : str
            'class' or 'instance'
        """
        if mode not in ['class', 'instance']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'class' or 'instance'")
        
        self.display_mode = mode
        
        if mode == 'class' and self.class_overlay is not None:
            overlay = self.class_overlay
        elif mode == 'instance' and self.instance_overlay is not None:
            overlay = self.instance_overlay
        else:
            logger.warning(f"No {mode} overlay available")
            return
        
        # Add overlay to window
        if hasattr(self.window, 'imageview'):
            # Create overlay item if it doesn't exist
            if not hasattr(self.window, 'event_overlay_item'):
                self.window.event_overlay_item = pg.ImageItem()
                self.window.imageview.addItem(self.window.event_overlay_item)
            
            # Update overlay
            current_frame = self.window.currentIndex
            if current_frame < overlay.shape[0]:
                self.window.event_overlay_item.setImage(overlay[current_frame])
                self.window.event_overlay_item.setZValue(10)  # Display on top
            
            self.overlayVisible = True
            
            # Connect to frame change event
            if not hasattr(self.window, '_event_overlay_connected'):
                self.window.sigTimeChanged.connect(self._update_overlay_frame)
                self.window._event_overlay_connected = True
    
    def _update_overlay_frame(self):
        """Update overlay when frame changes."""
        if not self.overlayVisible or not hasattr(self.window, 'event_overlay_item'):
            return
        
        overlay = self.class_overlay if self.display_mode == 'class' else self.instance_overlay
        if overlay is None:
            return
        
        current_frame = self.window.currentIndex
        if current_frame < overlay.shape[0]:
            self.window.event_overlay_item.setImage(overlay[current_frame])
    
    def hide_overlay(self):
        """Hide the overlay."""
        if hasattr(self.window, 'event_overlay_item'):
            self.window.event_overlay_item.setVisible(False)
            self.overlayVisible = False
    
    def toggle_overlay(self):
        """Toggle overlay visibility."""
        if self.overlayVisible:
            self.hide_overlay()
        else:
            self.show_overlay(self.display_mode)
    
    def switch_mode(self, mode: str):
        """
        Switch between class and instance display modes.
        
        Parameters
        ----------
        mode : str
            'class' or 'instance'
        """
        if self.overlayVisible:
            self.hide_overlay()
        self.show_overlay(mode)
    
    def get_event_stats(self) -> Dict:
        """
        Get statistics about detected events.
        
        Returns
        -------
        stats : dict
            Dictionary with event counts and properties
        """
        stats = {
            'n_sparks': 0,
            'n_puffs': 0,
            'n_waves': 0,
            'n_total_instances': 0
        }
        
        if self.class_mask is not None:
            stats['n_sparks'] = int(np.sum(self.class_mask == 1))
            stats['n_puffs'] = int(np.sum(self.class_mask == 2))
            stats['n_waves'] = int(np.sum(self.class_mask == 3))
        
        if self.instance_mask is not None:
            stats['n_total_instances'] = int(self.instance_mask.max())
        
        return stats


class CA_EventDetector_DisplayResults(BaseProcess):
    """
    Display detection results on current window.
    
    Shows color-coded overlays of detected calcium events.
    """
    
    def __init__(self):
        super().__init__()
    
    def gui(self):
        """Create display GUI."""
        self.gui_reset()
        
        window_selector = WindowSelector()
        
        from qtpy.QtWidgets import QComboBox, QCheckBox
        
        display_mode = QComboBox()
        display_mode.addItem('Class (Sparks/Puffs/Waves)')
        display_mode.addItem('Instance (Individual Events)')
        
        self.items.append({'name': 'window', 'string': 'Image Window',
                          'object': window_selector})
        self.items.append({'name': 'display_mode', 'string': 'Display Mode',
                          'object': display_mode})
        
        super().gui()
    
    def __call__(self, window, display_mode, keepSourceWindow=False):
        """Display results on window."""
        try:
            if not hasattr(window, 'ca_event_results'):
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.warning(None, "No Results",
                                  "No detection results found for this window.\n"
                                  "Please run detection first.")
                return window
            
            # Get results
            results = window.ca_event_results
            class_mask = results['class_mask']
            instance_mask = results['instance_mask']
            
            # Create or get display
            if not hasattr(window, 'ca_event_display'):
                window.ca_event_display = EventDisplay(window)
            
            # Set events and show
            window.ca_event_display.set_events(class_mask, instance_mask)
            
            mode = 'class' if 'Class' in display_mode else 'instance'
            window.ca_event_display.show_overlay(mode)
            
            # Show stats
            stats = window.ca_event_display.get_event_stats()
            g.m.statusBar().showMessage(
                f"Detected: {stats['n_sparks']} sparks, {stats['n_puffs']} puffs, "
                f"{stats['n_waves']} waves ({stats['n_total_instances']} total events)"
            )
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
            import traceback
            traceback.print_exc()
        
        return window


# Create plugin instance
ca_event_detector_display_results = CA_EventDetector_DisplayResults()
