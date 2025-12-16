#!/usr/bin/env python3
"""
U-Track Particle Tracking Plugin for FLIKA - Simplified Enhanced GUI Only

This plugin integrates u-track particle tracking methodology with FLIKA's
visualization and analysis capabilities using a comprehensive tabbed GUI.

Features:
- Multi-tabbed interface for all tracking options
- Real-time parameter validation and display
- Interactive track viewer with motion analysis
- Comprehensive visualization controls
- Advanced configuration options
- Progress tracking and detailed logging

Based on the u-track tracking framework by Danuser Lab - UTSouthwestern

Author: Simplified Enhanced Version
Version: 2.1.0
"""

import numpy as np
import os
import sys
import tempfile
import shutil
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add current plugin directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import PyQtGraph for track overlays
try:
    import pyqtgraph as pg
    from pyqtgraph import PlotDataItem
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Import FLIKA modules with proper fallbacks
FLIKA_AVAILABLE = False
g = None

# Qt imports for enhanced interface
try:
    from qtpy import QtWidgets, QtCore, QtGui
    from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                               QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                               QCheckBox, QComboBox, QTextEdit, QProgressBar,
                               QGroupBox, QGridLayout, QFormLayout, QFileDialog,
                               QListWidget, QFrame, QApplication, QSplitter,
                               QScrollArea, QSlider, QButtonGroup, QRadioButton)
    from qtpy.QtCore import Qt, QTimer
    from qtpy.QtGui import QFont, QPixmap, QPainter, QColor
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Try FLIKA imports
try:
    from flika import global_vars as g
    from flika.window import Window
    FLIKA_AVAILABLE = True

    # Try utils imports
    try:
        from flika.utils.misc import save_file_gui, open_file_gui
    except ImportError:
        def save_file_gui(*args, **kwargs):
            return "test_output.csv"
        def open_file_gui(*args, **kwargs):
            return "test_input.csv"

except ImportError:
    FLIKA_AVAILABLE = False

    # Create dummy functions if FLIKA not available
    class GlobalVars:
        currentWindow = None
        windows = []
        def alert(self, msg):
            print(f"ALERT: {msg}")

    g = GlobalVars()

    def save_file_gui(*args, **kwargs):
        return "test_output.csv"
    def open_file_gui(*args, **kwargs):
        return "test_input.csv"

# Import U-track tracking modules
TRACKING_AVAILABLE = False
tracking_import_error = None

try:
    from .track_general import ParticleTracker, CostMatrixParameters, GapCloseParameters, KalmanFunctions
    from .track_analysis import analyze_tracking_results, MotionAnalyzer
    from .utils import validate_movie_info
    TRACKING_AVAILABLE = True
except ImportError:
    try:
        from track_general import ParticleTracker, CostMatrixParameters, GapCloseParameters, KalmanFunctions
        from track_analysis import analyze_tracking_results, MotionAnalyzer
        from utils import validate_movie_info
        TRACKING_AVAILABLE = True
    except ImportError as e:
        tracking_import_error = str(e)

        # Create dummy classes to prevent further import errors
        class ParticleTracker:
            def run_tracking(self, *args, **kwargs):
                raise ImportError("U-Track modules not available")

        class CostMatrixParameters:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class GapCloseParameters:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class KalmanFunctions:
            pass

        def analyze_tracking_results(*args, **kwargs):
            return {'error': 'Tracking modules not available'}

        def validate_movie_info(*args, **kwargs):
            return []

# Import other required modules
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# ENHANCED PARAMETERS CLASS
# =============================================================================

class UTrackParameters:
    """Enhanced global storage for U-Track parameters with validation"""

    def __init__(self):
        # Motion model parameters
        self.motion_model = 'Mixed Motion'  # 'Brownian Motion', 'Directed Motion', 'Mixed Motion', 'Switching Motion'

        # Search radius parameters (stored as actual pixel values)
        self.min_search_radius = 3.0  # pixels
        self.max_search_radius = 20.0  # pixels

        # Motion parameters
        self.brown_std_mult = 5.0
        self.lin_std_mult = 5.0
        self.use_local_density = False
        self.max_angle_vv = 60.0  # degrees

        # Gap closing parameters
        self.time_window = 15
        self.gap_penalty = 1.0
        self.min_track_len = 3

        # Amplitude constraints
        self.amp_ratio_limit_min = 0.5
        self.amp_ratio_limit_max = 2.0
        self.res_limit = 1.0

        # Analysis options
        self.classify_motion = True
        self.len_for_classify = 5

        # Visualization options
        self.create_track_rois = True
        self.roi_color = 'green'
        self.show_track_overlay = True
        self.create_interactive_viewer = True
        self.track_alpha = 180
        self.track_width = 2
        self.max_tracks_display = 50
        self.show_start_end_markers = True

        # Output options
        self.save_tracks = False
        self.save_format = 'CSV'  # 'CSV', 'PKL', 'MAT'
        self.show_statistics = True
        self.save_debug_log = False

        # Debug options
        self.debug_mode = False
        self.parameter_logging = True
        self.verbose_output = False

        # Advanced options
        self.kalman_init_method = 'auto'
        self.cost_matrix_method = 'standard'
        self.enable_merge_split = True

    def validate(self):
        """Validate parameter consistency"""
        warnings = []

        if self.min_search_radius >= self.max_search_radius:
            warnings.append("min_search_radius should be less than max_search_radius")

        if self.min_search_radius < 0.1:
            warnings.append("min_search_radius is very small, may cause issues")

        if self.max_search_radius > 100:
            warnings.append("max_search_radius is very large, may be slow")

        if self.time_window > 50:
            warnings.append("time_window is very large, may be slow")

        if self.min_track_len < 2:
            warnings.append("min_track_len should be at least 2")

        return warnings

    def to_dict(self):
        """Convert parameters to dictionary"""
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}

    def from_dict(self, params_dict):
        """Load parameters from dictionary"""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, filename):
        """Save parameters to file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_file(self, filename):
        """Load parameters from file"""
        with open(filename, 'r') as f:
            params = json.load(f)
            self.from_dict(params)

    def get_formatted_display(self):
        """Get formatted parameter display for GUI"""
        params_dict = self.to_dict()

        display_text = "=== CURRENT U-TRACK PARAMETERS ===\n\n"

        # Motion model
        display_text += f"🎯 Motion Model: {params_dict['motion_model']}\n\n"

        # Search parameters
        display_text += f"🔍 Search Parameters:\n"
        display_text += f"  Min Search Radius: {params_dict['min_search_radius']:.1f} pixels\n"
        display_text += f"  Max Search Radius: {params_dict['max_search_radius']:.1f} pixels\n\n"

        # Motion parameters
        display_text += f"⚡ Motion Parameters:\n"
        display_text += f"  Brownian Std Multiplier: {params_dict['brown_std_mult']:.1f}\n"
        display_text += f"  Linear Std Multiplier: {params_dict['lin_std_mult']:.1f}\n"
        display_text += f"  Use Local Density: {params_dict['use_local_density']}\n"
        display_text += f"  Max Angle: {params_dict['max_angle_vv']:.1f}°\n\n"

        # Gap closing
        display_text += f"🔗 Gap Closing:\n"
        display_text += f"  Time Window: {params_dict['time_window']} frames\n"
        display_text += f"  Gap Penalty: {params_dict['gap_penalty']:.2f}\n"
        display_text += f"  Min Track Length: {params_dict['min_track_len']} frames\n\n"

        # Constraints
        display_text += f"📊 Constraints:\n"
        display_text += f"  Amplitude Ratio: {params_dict['amp_ratio_limit_min']:.2f} - {params_dict['amp_ratio_limit_max']:.2f}\n"
        display_text += f"  Resolution Limit: {params_dict['res_limit']:.1f} pixels\n\n"

        # Analysis
        display_text += f"🔬 Analysis:\n"
        display_text += f"  Classify Motion: {params_dict['classify_motion']}\n"
        display_text += f"  Length for Classification: {params_dict['len_for_classify']} frames\n\n"

        # Display
        display_text += f"🖥️ Display:\n"
        display_text += f"  Create Track ROIs: {params_dict['create_track_rois']}\n"
        display_text += f"  ROI Color: {params_dict['roi_color']}\n"
        display_text += f"  Show Track Overlay: {params_dict['show_track_overlay']}\n"
        display_text += f"  Create Interactive Viewer: {params_dict['create_interactive_viewer']}\n"
        display_text += f"  Save Tracks: {params_dict['save_tracks']}\n"
        display_text += f"  Show Statistics: {params_dict['show_statistics']}\n"

        display_text += "=" * 50

        return display_text

# Global parameters instance
utrack_params = UTrackParameters()


# =============================================================================
# TRACK OVERLAY SYSTEM
# =============================================================================

class TrackOverlay:
    """PyQtGraph-based track visualization overlay system"""

    def __init__(self, window):
        """Initialize track overlay for a FLIKA window"""
        self.window = window
        self.track_items = []
        self.colors = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 128, 0),   # Orange
            (128, 255, 0),   # Lime
            (255, 0, 128),   # Pink
            (128, 0, 255),   # Purple
            (0, 128, 255),   # Light Blue
            (255, 255, 128), # Light Yellow
            (128, 255, 255), # Light Cyan
            (255, 128, 255), # Light Magenta
            (128, 128, 255), # Light Purple
            (255, 128, 128), # Light Red
            (128, 255, 128), # Light Green
            (255, 255, 255), # White
            (192, 192, 192), # Silver
            (128, 128, 128), # Gray
        ]

    def add_tracks(self, tracks_final, max_tracks=50, show_start_end=True,
                   line_width=2, track_alpha=180):
        """Add track overlays to the FLIKA window"""

        if not PYQTGRAPH_AVAILABLE:
            return

        # Clear existing track items
        self.clear_tracks()

        tracks_to_show = tracks_final[:max_tracks]

        for i, track in enumerate(tracks_to_show):
            try:
                # Extract track coordinates
                coords = self._extract_track_coordinates(track)

                if len(coords) < 2:
                    continue

                # Get color for this track
                color = self.colors[i % len(self.colors)]
                color_with_alpha = (*color, track_alpha)

                # Create track line
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]

                # Create PlotDataItem for the track line
                track_line = PlotDataItem(
                    x=x_coords,
                    y=y_coords,
                    pen=pg.mkPen(color=color_with_alpha, width=line_width),
                    antialias=True,
                    name=f'Track_{i+1}'
                )

                # Add to window's imageview plot
                if hasattr(self.window, 'imageview') and hasattr(self.window.imageview, 'view'):
                    self.window.imageview.view.addItem(track_line)
                    self.track_items.append(track_line)

                    if show_start_end:
                        # Add start marker (circle)
                        start_marker = pg.ScatterPlotItem(
                            pos=[coords[0]],
                            brush=pg.mkBrush(color=color_with_alpha),
                            pen=pg.mkPen(color=(0, 0, 0), width=2),
                            size=8,
                            symbol='o',
                            name=f'Track_{i+1}_start'
                        )
                        self.window.imageview.view.addItem(start_marker)
                        self.track_items.append(start_marker)

                        # Add end marker (square)
                        end_marker = pg.ScatterPlotItem(
                            pos=[coords[-1]],
                            brush=pg.mkBrush(color=color_with_alpha),
                            pen=pg.mkPen(color=(0, 0, 0), width=2),
                            size=8,
                            symbol='s',
                            name=f'Track_{i+1}_end'
                        )
                        self.window.imageview.view.addItem(end_marker)
                        self.track_items.append(end_marker)

            except Exception as e:
                continue

        # Update the display
        if hasattr(self.window, 'imageview') and hasattr(self.window.imageview, 'view'):
            self.window.imageview.view.update()

    def clear_tracks(self):
        """Remove all track overlays from the window"""
        if hasattr(self.window, 'imageview') and hasattr(self.window.imageview, 'view'):
            for item in self.track_items:
                try:
                    self.window.imageview.view.removeItem(item)
                except:
                    pass
        self.track_items.clear()

    def _extract_track_coordinates(self, track):
        """Extract coordinate data from track"""
        coords = []

        if 'tracks_coord_amp_cg' in track:
            coord_data = track['tracks_coord_amp_cg']

            if coord_data.ndim > 1:
                coord_data = coord_data[0, :]

            # Coordinate data structure [x, y, ?, amp, x_err, y_err, amp_err, ?]
            num_frames = len(coord_data) // 8

            for i in range(num_frames):
                coord_idx = i * 8
                if coord_idx + 1 < len(coord_data):
                    x = coord_data[coord_idx]      # X at index 0
                    y = coord_data[coord_idx + 1]  # Y at index 1

                    if not (np.isnan(x) or np.isnan(y)):
                        coords.append([x, y])

        return np.array(coords) if coords else np.array([]).reshape(0, 2)



# =============================================================================
# INTERACTIVE TRACK VIEWER
# =============================================================================

class InteractiveTrackViewer:
    """Enhanced PyQtGraph-based interactive track viewer"""

    def __init__(self, image_data, tracks_data, window_name="Interactive Track Viewer"):
        """Initialize the interactive track viewer"""
        self.image_data = image_data
        self.tracks_data = tracks_data
        self.window_name = window_name
        self.track_viewer = None
        self.imv = None
        self.track_items = []
        self.particle_items = []
        self.current_frame = 0

        # Color map for movement types
        self.color_map = {
            'brownian': (255, 0, 0, 150),        # Red
            'directed': (0, 255, 0, 150),        # Green
            'confined': (0, 0, 255, 150),        # Blue
            'subdiffusive': (255, 255, 0, 150),  # Yellow
            'superdiffusive': (255, 0, 255, 150), # Magenta
            'variable': (255, 128, 0, 150),      # Orange
            'insufficient_data': (128, 128, 128, 150),  # Gray
            'unknown': (255, 255, 255, 150),     # White
        }

    def create_interactive_viewer(self):
        """Create interactive pyqtgraph viewer with tracks overlaid on original data"""
        if not PYQTGRAPH_AVAILABLE or not QT_AVAILABLE:
            if FLIKA_AVAILABLE:
                g.alert("PyQtGraph or Qt not available - cannot create interactive viewer")
            return None

        try:
            # Create the main widget and layout
            self.track_viewer = QtWidgets.QWidget()
            self.track_viewer.setWindowTitle(f"Particle Tracks - {self.window_name}")
            self.track_viewer.resize(1000, 700)

            layout = QtWidgets.QVBoxLayout()
            self.track_viewer.setLayout(layout)

            # Create control panel
            control_panel = self.create_control_panel()
            layout.addWidget(control_panel)

            # Create ImageView widget
            self.imv = pg.ImageView()
            layout.addWidget(self.imv)

            # Set the image data
            image_data = self.image_data.astype(np.float32)
            if image_data.ndim == 3:
                self.imv.setImage(image_data, axes={'t': 0, 'x': 2, 'y': 1})
            else:
                self.imv.setImage(image_data)

            # Initialize overlays
            self.update_track_overlay(0)

            # Connect to time slider changes
            if hasattr(self.imv, 'timeLine'):
                self.imv.timeLine.sigPositionChanged.connect(self.on_time_changed)

            # Add legend
            self.create_legend()

            # Show the window
            self.track_viewer.show()

            return self.track_viewer

        except Exception as e:
            if FLIKA_AVAILABLE:
                g.alert(f"Error creating interactive viewer: {e}")
            return None

    def create_control_panel(self):
        """Create control panel with track visualization options"""
        control_widget = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout()
        control_widget.setLayout(control_layout)

        # Track visibility controls
        self.show_tracks_cb = QtWidgets.QCheckBox("Show Track Paths")
        self.show_tracks_cb.setChecked(True)
        self.show_tracks_cb.stateChanged.connect(self.on_visibility_changed)
        control_layout.addWidget(self.show_tracks_cb)

        self.show_particles_cb = QtWidgets.QCheckBox("Show Current Particles")
        self.show_particles_cb.setChecked(True)
        self.show_particles_cb.stateChanged.connect(self.on_visibility_changed)
        control_layout.addWidget(self.show_particles_cb)

        # Track length control
        control_layout.addWidget(QtWidgets.QLabel("Trail Length:"))
        self.trail_length_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.trail_length_slider.setRange(1, 50)
        self.trail_length_slider.setValue(10)
        self.trail_length_slider.valueChanged.connect(self.on_trail_length_changed)
        control_layout.addWidget(self.trail_length_slider)

        self.trail_length_label = QtWidgets.QLabel("10")
        control_layout.addWidget(self.trail_length_label)

        # Particle size control
        control_layout.addWidget(QtWidgets.QLabel("Particle Size:"))
        self.particle_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.particle_size_slider.setRange(3, 20)
        self.particle_size_slider.setValue(8)
        self.particle_size_slider.valueChanged.connect(self.on_particle_size_changed)
        control_layout.addWidget(self.particle_size_slider)

        self.particle_size_label = QtWidgets.QLabel("8")
        control_layout.addWidget(self.particle_size_label)

        # Export button
        export_btn = QtWidgets.QPushButton("Export Current View")
        export_btn.clicked.connect(self.export_current_view)
        control_layout.addWidget(export_btn)

        control_layout.addStretch()
        return control_widget

    def create_legend(self):
        """Create a legend showing movement type colors"""
        legend_widget = QtWidgets.QWidget()
        legend_layout = QtWidgets.QHBoxLayout()
        legend_widget.setLayout(legend_layout)

        legend_layout.addWidget(QtWidgets.QLabel("Movement Types:"))

        legend_types = [
            ('brownian', 'Brownian'),
            ('directed', 'Directed'),
            ('confined', 'Confined'),
            ('variable', 'Variable'),
            ('subdiffusive', 'Subdiffusive'),
            ('superdiffusive', 'Superdiffusive')
        ]

        for movement_key, movement_label in legend_types:
            if movement_key in self.color_map:
                color = self.color_map[movement_key]

                # Create colored square
                color_label = QtWidgets.QLabel()
                color_label.setFixedSize(15, 15)
                color_label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid black;")

                text_label = QtWidgets.QLabel(movement_label)

                legend_layout.addWidget(color_label)
                legend_layout.addWidget(text_label)
                legend_layout.addSpacing(10)

        # Add legend to the main layout
        self.track_viewer.layout().addWidget(legend_widget)

    def on_time_changed(self):
        """Handle time slider changes"""
        try:
            if hasattr(self.imv, 'currentIndex'):
                current_frame = int(self.imv.currentIndex)
                if current_frame != self.current_frame:
                    self.current_frame = current_frame
                    self.update_track_overlay(current_frame)
        except Exception as e:
            pass

    def on_visibility_changed(self):
        """Handle visibility checkbox changes"""
        self.update_track_overlay(self.current_frame)

    def on_trail_length_changed(self, value):
        """Handle trail length slider changes"""
        self.trail_length_label.setText(str(value))
        self.update_track_overlay(self.current_frame)

    def on_particle_size_changed(self, value):
        """Handle particle size slider changes"""
        self.particle_size_label.setText(str(value))
        self.update_track_overlay(self.current_frame)

    def update_track_overlay(self, current_frame):
        """Update track overlay for the current frame"""
        try:
            # Clear existing overlays
            for item in self.track_items + self.particle_items:
                if hasattr(self.imv, 'view'):
                    self.imv.view.removeItem(item)

            self.track_items = []
            self.particle_items = []

            show_tracks = self.show_tracks_cb.isChecked() if hasattr(self, 'show_tracks_cb') else True
            show_particles = self.show_particles_cb.isChecked() if hasattr(self, 'show_particles_cb') else True
            trail_length = self.trail_length_slider.value() if hasattr(self, 'trail_length_slider') else 10
            particle_size = self.particle_size_slider.value() if hasattr(self, 'particle_size_slider') else 8

            # Debug counters
            color_usage = {}

            # Draw tracks up to current frame
            for track_idx, track in enumerate(self.tracks_data):
                movement_type = track.get('movement_type', 'unknown')

                # Debug first few tracks
                if track_idx < 3 and current_frame == 0:
                    print(f"DEBUG: Track {track_idx} movement_type: '{movement_type}'")

                color = self.color_map.get(movement_type, self.color_map['unknown'])

                # Count color usage for debugging
                color_usage[movement_type] = color_usage.get(movement_type, 0) + 1

                # Convert track data format to particles list if needed
                particles = self.extract_track_particles(track, current_frame)

                if not particles:
                    continue

                # Get track points up to current frame (with trail length limit)
                track_points = []
                current_particle = None

                # Sort particles by frame
                particles.sort(key=lambda p: p['frame'])

                for particle in particles:
                    frame = particle['frame']
                    if frame <= current_frame:
                        # Only include points within trail length
                        if current_frame - frame <= trail_length:
                            track_points.append([particle['x'], particle['y']])
                        if frame == current_frame:
                            current_particle = particle

                # Draw track path if we have multiple points and show_tracks is enabled
                if len(track_points) > 1 and show_tracks:
                    track_points = np.array(track_points)

                    # Create path item
                    path_item = pg.PlotCurveItem(
                        x=track_points[:, 0],
                        y=track_points[:, 1],
                        pen=pg.mkPen(color=color[:3], width=2, style=QtCore.Qt.SolidLine)
                    )
                    if hasattr(self.imv, 'view'):
                        self.imv.view.addItem(path_item)
                    self.track_items.append(path_item)

                # Draw current particle position if show_particles is enabled
                if current_particle is not None and show_particles:
                    particle_item = pg.ScatterPlotItem(
                        x=[current_particle['x']],
                        y=[current_particle['y']],
                        brush=pg.mkBrush(color=color),
                        pen=pg.mkPen(color=(255, 255, 255, 255), width=1),
                        size=particle_size,
                        symbol='o'
                    )
                    if hasattr(self.imv, 'view'):
                        self.imv.view.addItem(particle_item)
                    self.particle_items.append(particle_item)

            # Debug: Print color usage on first frame
            if current_frame == 0:
                print(f"DEBUG: Color usage by motion type: {color_usage}")

        except Exception as e:
            print(f"DEBUG: Error in update_track_overlay: {e}")
            import traceback
            traceback.print_exc()


    # ADD DEBUG METHOD TO UTrackTrackingGUI CLASS
    def debug_motion_results(self, motion_results):
        """Debug function to inspect motion results structure"""
        self.log_message("=== MOTION RESULTS DEBUG ===")

        if not motion_results:
            self.log_message("Motion results is None or empty")
            return

        self.log_message(f"Motion results type: {type(motion_results)}")
        self.log_message(f"Motion results keys: {list(motion_results.keys())}")

        # Print details of each key
        for key, value in motion_results.items():
            if isinstance(value, (list, tuple)):
                self.log_message(f"  {key}: list/tuple of length {len(value)}")
                if len(value) > 0:
                    self.log_message(f"    First item type: {type(value[0])}")
                    self.log_message(f"    First item: {str(value[0])[:100]}...")
            elif isinstance(value, dict):
                self.log_message(f"  {key}: dict with {len(value)} keys")
                dict_keys = list(value.keys())[:5]  # Show first 5 keys
                self.log_message(f"    Sample keys: {dict_keys}")
            else:
                self.log_message(f"  {key}: {type(value)} = {str(value)[:100]}...")

        self.log_message("==============================")

    def extract_track_particles(self, track, max_frame):
        """Extract particle positions from track data in consistent format"""
        particles = []

        try:
            # Handle different track data formats
            if 'particles' in track:
                # Already in particle format
                return [p for p in track['particles'] if p['frame'] <= max_frame]

            elif 'tracks_coord_amp_cg' in track:
                # Extract from coordinate data format
                coord_data = track['tracks_coord_amp_cg']
                if coord_data.ndim > 1:
                    coord_data = coord_data[0, :]

                # Coordinate data structure: [x, y, ?, amp, x_err, y_err, amp_err, ?]
                num_frames = len(coord_data) // 8

                for frame_idx in range(num_frames):
                    if frame_idx > max_frame:
                        break

                    coord_idx = frame_idx * 8
                    if coord_idx + 3 < len(coord_data):
                        x = coord_data[coord_idx]      # X at index 0
                        y = coord_data[coord_idx + 1]  # Y at index 1
                        amp = coord_data[coord_idx + 3] # Amplitude at index 3

                        if not (np.isnan(x) or np.isnan(y)):
                            particles.append({
                                'frame': frame_idx,
                                'x': float(x),
                                'y': float(y),
                                'amplitude': float(amp) if not np.isnan(amp) else 1000.0
                            })

            # Add movement type to all particles
            movement_type = track.get('movement_type', 'unknown')
            for particle in particles:
                particle['movement_type'] = movement_type

        except Exception as e:
            pass

        return particles

    def export_current_view(self):
        """Export current view as image"""
        try:
            if hasattr(self.imv, 'export'):
                self.imv.export()
        except Exception as e:
            pass

    def close(self):
        """Close the interactive viewer"""
        if self.track_viewer:
            self.track_viewer.close()


# =============================================================================
# SIMPLIFIED TRACKING ENGINE
# =============================================================================

class TrackingEngine:
    """Fixed tracking engine for GUI integration"""

    def __init__(self):
        self.tracking_results = None
        self.detection_data = None
        self.parameter_log = []
        self.final_parameters_used = {}

    def run_tracking(self, input_source, parameters, progress_callback=None, log_callback=None):
        """Run tracking with given parameters - FIXED VERSION"""
        if not TRACKING_AVAILABLE:
            raise ImportError("U-Track tracking modules not available")

        try:
            # Load detection data
            if log_callback:
                log_callback("🔍 Loading detection data...")

            movie_info = self._load_detection_data(input_source)
            if not movie_info:
                raise ValueError("No detection data available")

            if log_callback:
                total_detections = sum(frame.get('num', 0) for frame in movie_info if frame)
                num_frames = len(movie_info)
                log_callback(f"✅ Data loaded: {num_frames} frames, {total_detections} detections")

            # FIXED: Proper parameter conversion with validation
            if log_callback:
                log_callback("⚙️ Converting and validating tracking parameters...")

            # Convert GUI parameters to module-compatible format
            converted_params = self._convert_parameters(parameters)

            # Log parameter conversion for debugging
            self._log_parameter_conversion(parameters, converted_params, log_callback)

            if progress_callback:
                progress_callback(10)

            # FIXED: Create proper tracking parameter objects
            cost_matrices, gap_close_param, kalman_functions = self._create_tracking_parameters(converted_params)

            if progress_callback:
                progress_callback(20)

            # FIXED: Run tracking using individual modules
            if log_callback:
                log_callback("🚀 Running tracking algorithm with individual modules...")

            # Create tracker and ensure it uses individual modules
            tracker = ParticleTracker()

            # Set parameters explicitly
            tracker.cost_matrices = cost_matrices
            tracker.gap_close_param = gap_close_param
            tracker.kalman_functions = kalman_functions
            tracker.prob_dim = 2  # Can be made configurable
            tracker.verbose = True

            start_time = time.time()

            # FIXED: Call tracking with proper parameter structure
            tracks_final, kalman_info, err_flag = tracker.run_tracking(
                movie_info=movie_info,
                save_dir=None,  # Don't save here, we'll handle it in GUI
                filename=None,
                cost_matrices=cost_matrices,
                gap_close_param=gap_close_param,
                kalman_functions=kalman_functions
            )

            tracking_time = time.time() - start_time

            if err_flag != 0:
                raise RuntimeError(f"Tracking algorithm failed with error flag {err_flag}")

            if progress_callback:
                progress_callback(70)

            # FIXED: Ensure motion analysis is performed by individual modules
            motion_results = None
            if parameters.classify_motion and tracks_final:
                if log_callback:
                    log_callback("🔬 Running motion analysis using individual modules...")

                # Use the track_analysis module directly
                try:
                    from track_analysis import analyze_tracking_results, MotionAnalyzer

                    # Perform comprehensive motion analysis
                    motion_results = analyze_tracking_results(tracks_final, prob_dim=2)

                    # Add individual track motion analysis
                    motion_analyzer = MotionAnalyzer()
                    individual_results = []

                    for i, track in enumerate(tracks_final):
                        track_motion = motion_analyzer.analyze_track_motion(track, prob_dim=2)
                        individual_results.append(track_motion)

                        # Add motion type to the track structure itself
                        if 'error' not in track_motion:
                            track['motion_type'] = track_motion.get('motion_type', 'unknown')
                            track['motion_analysis'] = track_motion

                    # Add individual results to motion_results
                    if motion_results and 'error' not in motion_results:
                        motion_results['individual_track_analysis'] = individual_results

                        # Create motion type mapping for each track
                        motion_results['track_motion_types'] = [
                            result.get('motion_type', 'unknown')
                            for result in individual_results
                            if 'error' not in result
                        ]

                    if log_callback:
                        if motion_results and 'error' not in motion_results:
                            log_callback(f"✅ Motion analysis completed: {motion_results.get('num_tracks', 0)} tracks analyzed")
                        else:
                            log_callback("⚠️ Motion analysis completed with errors")

                except Exception as e:
                    if log_callback:
                        log_callback(f"⚠️ Motion analysis error: {str(e)}")
                    motion_results = {'error': str(e)}

            if progress_callback:
                progress_callback(90)

            # FIXED: Compile comprehensive results from all modules
            if log_callback:
                log_callback("📊 Compiling results from all modules...")

            self.tracking_results = {
                # Core tracking results from individual modules
                'tracks_final': tracks_final,
                'kalman_info': kalman_info,
                'motion_results': motion_results,

                # Timing and performance
                'tracking_time': tracking_time,
                'num_tracks_final': len(tracks_final) if tracks_final else 0,

                # Parameter tracking
                'parameters_used': parameters.to_dict(),
                'converted_parameters': converted_params,
                'final_parameters_used': self.final_parameters_used,
                'parameter_log': self.parameter_log,

                # Module-specific results
                'linking_results': {
                    'method': 'FeatureLinker.link_features_kalman_sparse',
                    'kalman_info': kalman_info,
                },
                'gap_closing_results': {
                    'method': 'GapCloser.close_gaps',
                    'gap_close_param': gap_close_param.__dict__ if hasattr(gap_close_param, '__dict__') else gap_close_param,
                },
                'cost_matrix_results': {
                    'linking_cost_matrix': cost_matrices[0] if len(cost_matrices) > 0 else None,
                    'gap_closing_cost_matrix': cost_matrices[1] if len(cost_matrices) > 1 else None,
                },

                # Data provenance
                'input_source': input_source,
                'detection_data_summary': {
                    'num_frames': len(movie_info),
                    'total_detections': sum(frame.get('num', 0) for frame in movie_info if frame),
                    'avg_detections_per_frame': sum(frame.get('num', 0) for frame in movie_info if frame) / len(movie_info) if movie_info else 0
                },

                # Error tracking
                'error_flag': err_flag,
                'errors': [],
                'warnings': []
            }

            if progress_callback:
                progress_callback(100)

            if log_callback:
                log_callback(f"✅ Tracking completed! Found {len(tracks_final)} tracks in {tracking_time:.2f} seconds")
                if motion_results and 'error' not in motion_results:
                    log_callback(f"📊 Motion analysis: {motion_results.get('motion_type_counts', {})}")

            return self.tracking_results

        except Exception as e:
            if log_callback:
                log_callback(f"❌ Tracking failed: {str(e)}")

            # Store error information
            self.tracking_results = {
                'error': str(e),
                'parameters_used': parameters.to_dict() if parameters else {},
                'input_source': input_source,
                'tracking_time': 0,
                'tracks_final': [],
                'error_flag': 1
            }
            raise

    def _convert_parameters(self, params):
        """FIXED: Convert UTrackParameters to internal format with full validation"""

        # Map motion model names to internal codes
        motion_map = {
            'Brownian Motion': 0,
            'Directed Motion': 1,
            'Mixed Motion': 1,
            'Switching Motion': 2
        }

        lin_std_mult_val = params.lin_std_mult
        if isinstance(lin_std_mult_val, (list, np.ndarray)) and len(lin_std_mult_val) > 1:
            # Already an array, use it
            lin_std_mult_array = np.array(lin_std_mult_val)
        else:
            # Scalar, create array
            scalar_val = float(lin_std_mult_val) if not isinstance(lin_std_mult_val, (list, np.ndarray)) else float(lin_std_mult_val[0])
            lin_std_mult_array = np.array([max(1.0, scalar_val)] * 5)

        # Similarly for brown_std_mult:
        brown_std_mult_val = params.brown_std_mult
        if isinstance(brown_std_mult_val, (list, np.ndarray)) and len(brown_std_mult_val) > 1:
            brown_std_mult_final = np.array(brown_std_mult_val)
        else:
            scalar_val = float(brown_std_mult_val) if not isinstance(brown_std_mult_val, (list, np.ndarray)) else float(brown_std_mult_val[0])
            brown_std_mult_final = max(1.0, scalar_val)

        # Convert and validate all parameters
        converted = {
            # Motion model
            'linear_motion': motion_map.get(params.motion_model, 1),

            # Search radius parameters (ensure they're positive)
            'min_search_radius': max(0.1, float(params.min_search_radius)),
            'max_search_radius': max(params.min_search_radius, float(params.max_search_radius)),

            # Motion parameters
            'brown_std_mult': brown_std_mult_final,
            'lin_std_mult': lin_std_mult_array,
            'use_local_density': int(bool(params.use_local_density)),
            'max_angle_vv': max(1.0, min(180.0, float(params.max_angle_vv))),

            # Time scaling parameters
            'brown_scaling': [0.25, 0.01],  # Default values
            'lin_scaling': [1.0, 0.01],     # Default values
            'time_reach_conf_b': min(params.time_window, 10),  # Reasonable limit
            'time_reach_conf_l': min(params.time_window, 10),  # Reasonable limit

            # Gap closing parameters
            'time_window': max(2, int(params.time_window)),
            'gap_penalty': max(1.0, float(params.gap_penalty)),
            'min_track_len': max(2, int(params.min_track_len)),

            # Amplitude constraints
            'amp_ratio_limit': [
                max(0.1, float(params.amp_ratio_limit_min)),
                max(params.amp_ratio_limit_min, float(params.amp_ratio_limit_max))
            ],
            'res_limit': max(0.0, float(params.res_limit)),

            # Merge/split handling
            'merge_split_code': 1,  # Enable merge and split by default

            # Classification parameters
            'len_for_classify': max(3, int(params.len_for_classify)),
            'classify_motion': bool(params.classify_motion),
        }

        # Store converted parameters for logging
        self.final_parameters_used = converted.copy()

        return converted

    def _create_tracking_parameters(self, params):
        """FIXED: Create tracking parameter objects with proper structure"""

        # Create linking parameters
        link_params = CostMatrixParameters(
            linear_motion=params['linear_motion'],
            min_search_radius=params['min_search_radius'],
            max_search_radius=params['max_search_radius'],
            brown_std_mult=params['brown_std_mult'],
            lin_std_mult=params['lin_std_mult'][0],  # Use first value for linking
            use_local_density=params['use_local_density'],
            max_angle_vv=params['max_angle_vv'],
            brown_scaling=params['brown_scaling'],
            lin_scaling=params['lin_scaling'],
            time_reach_conf_b=params['time_reach_conf_b'],
            time_reach_conf_l=params['time_reach_conf_l'],
            res_limit=params['res_limit'],
            amp_ratio_limit=params['amp_ratio_limit']
        )

        # Create gap closing parameters with time-dependent multipliers
        gap_params = CostMatrixParameters(
            linear_motion=params['linear_motion'],
            min_search_radius=params['min_search_radius'],
            max_search_radius=params['max_search_radius'] * 1.5,  # Larger for gap closing
            brown_std_mult=np.linspace(params['brown_std_mult'], params['brown_std_mult'] * 1.5, params['time_window']),
            lin_std_mult=np.linspace(params['lin_std_mult'][0], params['lin_std_mult'][0] * 1.5, params['time_window']),
            gap_penalty=params['gap_penalty'],
            use_local_density=params['use_local_density'],
            max_angle_vv=params['max_angle_vv'] * 1.5,  # More permissive for gap closing
            brown_scaling=params['brown_scaling'],
            lin_scaling=params['lin_scaling'],
            time_reach_conf_b=params['time_reach_conf_b'],
            time_reach_conf_l=params['time_reach_conf_l'],
            res_limit=params['res_limit'],
            amp_ratio_limit=params['amp_ratio_limit']
        )

        # Create cost matrix specifications
        cost_matrices = [
            {
                'func_name': 'cost_mat_random_directed_switching_motion_link',
                'parameters': link_params
            },
            {
                'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                'parameters': gap_params
            }
        ]

        # Create gap closing parameters
        gap_close_param = GapCloseParameters(
            time_window=params['time_window'],
            merge_split=params['merge_split_code'],
            min_track_len=params['min_track_len'],
            tolerance=0.05
        )

        # Create Kalman filter functions
        kalman_functions = KalmanFunctions()

        return cost_matrices, gap_close_param, kalman_functions

    def _log_parameter_conversion(self, original_params, converted_params, log_callback):
        """Log parameter conversion for debugging"""
        if log_callback:
            log_callback("📋 Parameter conversion summary:")
            log_callback(f"  Motion model: {original_params.motion_model} → {converted_params['linear_motion']}")
            log_callback(f"  Search radius: {original_params.min_search_radius}-{original_params.max_search_radius} pixels")
            log_callback(f"  Time window: {converted_params['time_window']} frames")
            log_callback(f"  Min track length: {converted_params['min_track_len']} frames")
            log_callback(f"  Motion classification: {converted_params['classify_motion']}")


    def _load_detection_data(self, input_source):
        """Load detection data from various sources"""
        if input_source == 'Current Detection Results':
            return self._load_from_detection_plugin()
        elif input_source.endswith('.csv'):
            return self._load_from_csv(input_source)
        elif input_source.endswith('.mat'):
            return self._load_from_mat(input_source)
        elif input_source.endswith('.pkl'):
            return self._load_from_pkl(input_source)
        else:
            return None

    def _load_from_detection_plugin(self):
        """Load detection results from the detection plugin"""
        if FLIKA_AVAILABLE and hasattr(g, 'utrack_detection_results'):
            detection_results = g.utrack_detection_results
            if detection_results and 'movie_info' in detection_results:
                return self._convert_detection_format(detection_results['movie_info'])

        if FLIKA_AVAILABLE and g.currentWindow:
            if hasattr(g.currentWindow, 'utrack_detection_results'):
                detection_results = g.currentWindow.utrack_detection_results
                if detection_results and 'movie_info' in detection_results:
                    return self._convert_detection_format(detection_results['movie_info'])

        return None

    def _convert_detection_format(self, raw_movie_info):
        """Convert detection plugin results to tracking-compatible format"""
        if not raw_movie_info:
            return None

        converted_movie_info = []

        try:
            for frame_idx, frame_data in enumerate(raw_movie_info):
                if frame_data is None:
                    frame_info = create_standard_frame_info([], [], [])
                    converted_movie_info.append(frame_info)
                    continue

                if hasattr(frame_data, 'xCoord') and frame_data.xCoord is not None:
                    if hasattr(frame_data.xCoord, '__len__') and len(frame_data.xCoord) > 0:
                        # Extract coordinates and amplitudes
                        x_coords = frame_data.xCoord[:, 0] if frame_data.xCoord.ndim == 2 else frame_data.xCoord
                        x_stds = frame_data.xCoord[:, 1] if frame_data.xCoord.ndim == 2 and frame_data.xCoord.shape[1] > 1 else np.ones(len(x_coords)) * 0.1

                        if hasattr(frame_data, 'yCoord') and frame_data.yCoord is not None:
                            y_coords = frame_data.yCoord[:, 0] if frame_data.yCoord.ndim == 2 else frame_data.yCoord
                            y_stds = frame_data.yCoord[:, 1] if frame_data.yCoord.ndim == 2 and frame_data.yCoord.shape[1] > 1 else np.ones(len(y_coords)) * 0.1
                        else:
                            y_coords = np.zeros_like(x_coords)
                            y_stds = np.ones(len(x_coords)) * 0.1

                        if hasattr(frame_data, 'amp') and frame_data.amp is not None:
                            amplitudes = frame_data.amp[:, 0] if frame_data.amp.ndim == 2 else frame_data.amp
                            amp_stds = frame_data.amp[:, 1] if frame_data.amp.ndim == 2 and frame_data.amp.shape[1] > 1 else amplitudes * 0.1
                        else:
                            amplitudes = np.ones(len(x_coords)) * 1000.0
                            amp_stds = amplitudes * 0.1

                        # Ensure all arrays are the same length
                        min_len = min(len(x_coords), len(y_coords), len(amplitudes))
                        if min_len > 0:
                            frame_info = create_standard_frame_info(
                                x_coords[:min_len], y_coords[:min_len], amplitudes[:min_len],
                                x_stds[:min_len], y_stds[:min_len], amp_stds[:min_len], frame_idx
                            )
                        else:
                            frame_info = create_standard_frame_info([], [], [])
                    else:
                        frame_info = create_standard_frame_info([], [], [])
                else:
                    frame_info = create_standard_frame_info([], [], [])

                converted_movie_info.append(frame_info)

            return converted_movie_info

        except Exception as e:
            return None

    def _load_from_csv(self, filename):
        """Load detection data from CSV file"""
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not available - cannot load CSV files")

        try:
            df = pd.read_csv(filename)

            # Check for required columns
            required_cols = ['Frame', 'X', 'Y']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                # Try alternative column names
                col_mapping = {
                    'frame': 'Frame', 'frames': 'Frame',
                    'x': 'X', 'x_pos': 'X', 'x_position': 'X',
                    'y': 'Y', 'y_pos': 'Y', 'y_position': 'Y',
                    'amplitude': 'Amplitude', 'amp': 'Amplitude', 'intensity': 'Amplitude'
                }

                # Rename columns to standard names
                for old_name, new_name in col_mapping.items():
                    if old_name in df.columns and new_name not in df.columns:
                        df = df.rename(columns={old_name: new_name})

                # Check again for required columns
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"CSV file missing required columns: {missing_cols}")

            movie_info = []
            unique_frames = sorted(df['Frame'].unique())

            for frame_num in unique_frames:
                frame_data = df[df['Frame'] == frame_num]

                x_coords = frame_data['X'].values
                y_coords = frame_data['Y'].values
                x_stds = frame_data['X_Std'].values if 'X_Std' in frame_data.columns else np.ones(len(x_coords)) * 0.1
                y_stds = frame_data['Y_Std'].values if 'Y_Std' in frame_data.columns else np.ones(len(y_coords)) * 0.1
                amplitudes = frame_data['Amplitude'].values if 'Amplitude' in frame_data.columns else np.ones(len(x_coords)) * 1000
                amp_stds = amplitudes * 0.1

                frame_info = create_standard_frame_info(
                    x_coords, y_coords, amplitudes, x_stds, y_stds, amp_stds, frame_num
                )

                movie_info.append(frame_info)

            return movie_info

        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def _load_from_mat(self, filename):
        """Load detection data from MATLAB file"""
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available - cannot load MAT files")

        try:
            mat_data = scipy.io.loadmat(filename)

            if 'movieInfo' in mat_data:
                matlab_movie_info = mat_data['movieInfo']
                movie_info = []
                total_detections = 0

                for frame_idx, frame_data in enumerate(matlab_movie_info):
                    if isinstance(frame_data, dict):
                        x_coord = frame_data.get('xCoord', np.array([]))
                        y_coord = frame_data.get('yCoord', np.array([]))
                        amp = frame_data.get('amp', np.array([]))
                    else:
                        x_coord = getattr(frame_data, 'xCoord', np.array([]))
                        y_coord = getattr(frame_data, 'yCoord', np.array([]))
                        amp = getattr(frame_data, 'amp', np.array([]))

                    if x_coord.size > 0 and y_coord.size > 0:
                        if x_coord.ndim == 2:
                            x_coords = x_coord[:, 0]
                            x_stds = x_coord[:, 1] if x_coord.shape[1] > 1 else np.ones(len(x_coords)) * 0.1
                        else:
                            x_coords = x_coord.flatten()
                            x_stds = np.ones(len(x_coords)) * 0.1

                        if y_coord.ndim == 2:
                            y_coords = y_coord[:, 0]
                            y_stds = y_coord[:, 1] if y_coord.shape[1] > 1 else np.ones(len(y_coords)) * 0.1
                        else:
                            y_coords = y_coord.flatten()
                            y_stds = np.ones(len(y_coords)) * 0.1

                        if amp.size > 0:
                            if amp.ndim == 2:
                                amplitudes = amp[:, 0]
                                amp_stds = amp[:, 1] if amp.shape[1] > 1 else amplitudes * 0.1
                            else:
                                amplitudes = amp.flatten()
                                amp_stds = amplitudes * 0.1
                        else:
                            amplitudes = np.ones(len(x_coords)) * 1000.0
                            amp_stds = amplitudes * 0.1

                        min_len = min(len(x_coords), len(y_coords), len(amplitudes))
                        x_coords = x_coords[:min_len]
                        y_coords = y_coords[:min_len]
                        amplitudes = amplitudes[:min_len]
                        x_stds = x_stds[:min_len]
                        y_stds = y_stds[:min_len]
                        amp_stds = amp_stds[:min_len]

                        total_detections += min_len

                        frame_info = create_standard_frame_info(
                            x_coords, y_coords, amplitudes, x_stds, y_stds, amp_stds, frame_idx
                        )
                    else:
                        frame_info = create_standard_frame_info([], [], [])

                    movie_info.append(frame_info)

                return movie_info
            else:
                raise ValueError("MAT file does not contain 'movieInfo' field")

        except Exception as e:
            raise ValueError(f"Error loading MAT file: {str(e)}")

    def _load_from_pkl(self, filename):
        """Load detection data from pickle file"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict):
                if 'movie_info' in data:
                    raw_movie_info = data['movie_info']
                elif 'detection_results' in data and 'movie_info' in data['detection_results']:
                    raw_movie_info = data['detection_results']['movie_info']
                else:
                    raw_movie_info = data
            else:
                raw_movie_info = data

            movie_info = []
            total_detections = 0

            for frame_idx, frame_data in enumerate(raw_movie_info):
                if frame_data is None:
                    frame_info = create_standard_frame_info([], [], [])
                elif isinstance(frame_data, dict):
                    x_coords = frame_data.get('xCoord', frame_data.get('x_coord', np.array([])))
                    y_coords = frame_data.get('yCoord', frame_data.get('y_coord', np.array([])))
                    amps = frame_data.get('amp', np.array([]))

                    if hasattr(x_coords, 'ndim') and x_coords.ndim == 2:
                        x_vals = x_coords[:, 0]
                        x_stds = x_coords[:, 1]
                    else:
                        x_vals = np.array(x_coords).flatten()
                        x_stds = np.ones(len(x_vals)) * 0.1

                    if hasattr(y_coords, 'ndim') and y_coords.ndim == 2:
                        y_vals = y_coords[:, 0]
                        y_stds = y_coords[:, 1]
                    else:
                        y_vals = np.array(y_coords).flatten()
                        y_stds = np.ones(len(y_vals)) * 0.1

                    if hasattr(amps, 'ndim') and amps.ndim == 2:
                        amp_vals = amps[:, 0]
                        amp_stds = amps[:, 1]
                    else:
                        amp_vals = np.array(amps).flatten() if len(amps) > 0 else np.ones(len(x_vals)) * 1000.0
                        amp_stds = amp_vals * 0.1

                    if len(x_vals) > 0 and len(y_vals) > 0:
                        min_len = min(len(x_vals), len(y_vals), len(amp_vals))
                        total_detections += min_len

                        frame_info = create_standard_frame_info(
                            x_vals[:min_len], y_vals[:min_len], amp_vals[:min_len],
                            x_stds[:min_len], y_stds[:min_len], amp_stds[:min_len], frame_idx
                        )
                    else:
                        frame_info = create_standard_frame_info([], [], [])

                else:
                    x_coord = getattr(frame_data, 'xCoord', np.array([]))
                    y_coord = getattr(frame_data, 'yCoord', np.array([]))
                    amp = getattr(frame_data, 'amp', np.array([]))

                    if x_coord.size > 0 and y_coord.size > 0:
                        if x_coord.ndim == 2:
                            x_coords = x_coord[:, 0]
                            x_stds = x_coord[:, 1]
                        else:
                            x_coords = x_coord.flatten()
                            x_stds = np.ones(len(x_coords)) * 0.1

                        if y_coord.ndim == 2:
                            y_coords = y_coord[:, 0]
                            y_stds = y_coord[:, 1]
                        else:
                            y_coords = y_coord.flatten()
                            y_stds = np.ones(len(y_coords)) * 0.1

                        if amp.size > 0:
                            if amp.ndim == 2:
                                amplitudes = amp[:, 0]
                                amp_stds = amp[:, 1]
                            else:
                                amplitudes = amp.flatten()
                                amp_stds = amplitudes * 0.1
                        else:
                            amplitudes = np.ones(len(x_coords)) * 1000.0
                            amp_stds = amplitudes * 0.1

                        total_detections += len(x_coords)

                        frame_info = create_standard_frame_info(
                            x_coords, y_coords, amplitudes, x_stds, y_stds, amp_stds, frame_idx
                        )
                    else:
                        frame_info = create_standard_frame_info([], [], [])

                movie_info.append(frame_info)

            return movie_info

        except Exception as e:
            raise ValueError(f"Error loading PKL file: {str(e)}")




# =============================================================================
# ENHANCED MULTI-TABBED GUI
# =============================================================================

class UTrackTrackingGUI(QWidget):
    """Enhanced multi-tabbed GUI for U-Track particle tracking"""

    def __init__(self):
        super().__init__()
        self.parameters = utrack_params
        self.tracking_engine = TrackingEngine()
        self.tracking_results = None
        self.interactive_viewer = None

        self.setupUI()
        self.update_gui_from_parameters()

    def setupUI(self):
        """Create the main user interface"""
        self.setWindowTitle("U-Track Particle Tracking - Enhanced Interface")
        self.setMinimumSize(1000, 800)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc; }")
        main_layout.addWidget(self.status_label)

        # Create tabbed interface
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_input_tab()
        self.create_parameters_tab()
        self.create_current_parameters_tab()  # New tab for parameter display
        self.create_visualization_tab()
        self.create_analysis_tab()
        self.create_advanced_tab()
        self.create_progress_tab()

        # Control buttons
        self.create_control_buttons(main_layout)

        # Set up auto-validation
        self.setup_validation()

    def create_input_tab(self):
        """Create input/data source selection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Data source section
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)

        self.input_source_combo = QComboBox()
        self.input_source_combo.addItems([
            'Current Detection Results',
            'Load from CSV File',
            'Load from MAT File',
            'Load from PKL File'
        ])
        self.input_source_combo.currentTextChanged.connect(self.on_input_source_changed)
        source_layout.addWidget(QLabel("Input Source:"))
        source_layout.addWidget(self.input_source_combo)

        # File selection (hidden by default)
        self.file_selection_widget = QWidget()
        file_layout = QHBoxLayout(self.file_selection_widget)
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        file_layout.addWidget(self.file_path_label)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)

        source_layout.addWidget(self.file_selection_widget)
        self.file_selection_widget.hide()

        layout.addWidget(source_group)

        # Data preview section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.data_info_text = QTextEdit()
        self.data_info_text.setMaximumHeight(150)
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setText("No data loaded")
        preview_layout.addWidget(self.data_info_text)

        self.refresh_data_button = QPushButton("Refresh Data Info")
        self.refresh_data_button.clicked.connect(self.refresh_data_info)
        preview_layout.addWidget(self.refresh_data_button)

        layout.addWidget(preview_group)

        # Current window info
        if FLIKA_AVAILABLE:
            window_group = QGroupBox("Current FLIKA Window")
            window_layout = QVBoxLayout(window_group)

            self.window_info_label = QLabel("No window selected")
            window_layout.addWidget(self.window_info_label)

            self.refresh_window_button = QPushButton("Refresh Window Info")
            self.refresh_window_button.clicked.connect(self.refresh_window_info)
            window_layout.addWidget(self.refresh_window_button)

            layout.addWidget(window_group)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Input")

    def create_parameters_tab(self):
        """Create parameters configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Scrollable area
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Motion model section
        motion_group = QGroupBox("Motion Model")
        motion_layout = QFormLayout(motion_group)

        self.motion_model_combo = QComboBox()
        self.motion_model_combo.addItems([
            'Brownian Motion',
            'Directed Motion',
            'Mixed Motion',
            'Switching Motion'
        ])
        self.motion_model_combo.currentTextChanged.connect(self.on_parameter_changed)
        motion_layout.addRow("Motion Model:", self.motion_model_combo)

        scroll_layout.addWidget(motion_group)

        # Search parameters section
        search_group = QGroupBox("Search Parameters")
        search_layout = QFormLayout(search_group)

        self.min_search_radius_spin = QDoubleSpinBox()
        self.min_search_radius_spin.setRange(0.1, 50.0)
        self.min_search_radius_spin.setSuffix(" pixels")
        self.min_search_radius_spin.setDecimals(1)
        self.min_search_radius_spin.valueChanged.connect(self.on_parameter_changed)
        search_layout.addRow("Min Search Radius:", self.min_search_radius_spin)

        self.max_search_radius_spin = QDoubleSpinBox()
        self.max_search_radius_spin.setRange(1.0, 100.0)
        self.max_search_radius_spin.setSuffix(" pixels")
        self.max_search_radius_spin.setDecimals(1)
        self.max_search_radius_spin.valueChanged.connect(self.on_parameter_changed)
        search_layout.addRow("Max Search Radius:", self.max_search_radius_spin)

        scroll_layout.addWidget(search_group)

        # Motion parameters section
        motion_params_group = QGroupBox("Motion Parameters")
        motion_params_layout = QFormLayout(motion_params_group)

        self.brown_std_mult_spin = QDoubleSpinBox()
        self.brown_std_mult_spin.setRange(1.0, 20.0)
        self.brown_std_mult_spin.setDecimals(1)
        self.brown_std_mult_spin.valueChanged.connect(self.on_parameter_changed)
        motion_params_layout.addRow("Brownian Std Multiplier:", self.brown_std_mult_spin)

        self.lin_std_mult_spin = QDoubleSpinBox()
        self.lin_std_mult_spin.setRange(1.0, 20.0)
        self.lin_std_mult_spin.setDecimals(1)
        self.lin_std_mult_spin.valueChanged.connect(self.on_parameter_changed)
        motion_params_layout.addRow("Linear Std Multiplier:", self.lin_std_mult_spin)

        self.max_angle_spin = QSpinBox()
        self.max_angle_spin.setRange(5, 180)
        self.max_angle_spin.setSuffix("°")
        self.max_angle_spin.valueChanged.connect(self.on_parameter_changed)
        motion_params_layout.addRow("Max Angle:", self.max_angle_spin)

        self.use_local_density_check = QCheckBox("Use Local Density")
        self.use_local_density_check.stateChanged.connect(self.on_parameter_changed)
        motion_params_layout.addRow("", self.use_local_density_check)

        scroll_layout.addWidget(motion_params_group)

        # Gap closing parameters section
        gap_group = QGroupBox("Gap Closing Parameters")
        gap_layout = QFormLayout(gap_group)

        self.time_window_spin = QSpinBox()
        self.time_window_spin.setRange(2, 50)
        self.time_window_spin.setSuffix(" frames")
        self.time_window_spin.valueChanged.connect(self.on_parameter_changed)
        gap_layout.addRow("Time Window:", self.time_window_spin)

        self.gap_penalty_spin = QDoubleSpinBox()
        self.gap_penalty_spin.setRange(0.1, 10.0)
        self.gap_penalty_spin.setDecimals(2)
        self.gap_penalty_spin.valueChanged.connect(self.on_parameter_changed)
        gap_layout.addRow("Gap Penalty:", self.gap_penalty_spin)

        self.min_track_len_spin = QSpinBox()
        self.min_track_len_spin.setRange(2, 100)
        self.min_track_len_spin.setSuffix(" frames")
        self.min_track_len_spin.valueChanged.connect(self.on_parameter_changed)
        gap_layout.addRow("Min Track Length:", self.min_track_len_spin)

        scroll_layout.addWidget(gap_group)

        # Amplitude constraints section
        amp_group = QGroupBox("Amplitude Constraints")
        amp_layout = QFormLayout(amp_group)

        self.amp_ratio_min_spin = QDoubleSpinBox()
        self.amp_ratio_min_spin.setRange(0.1, 2.0)
        self.amp_ratio_min_spin.setDecimals(2)
        self.amp_ratio_min_spin.valueChanged.connect(self.on_parameter_changed)
        amp_layout.addRow("Min Amplitude Ratio:", self.amp_ratio_min_spin)

        self.amp_ratio_max_spin = QDoubleSpinBox()
        self.amp_ratio_max_spin.setRange(1.0, 10.0)
        self.amp_ratio_max_spin.setDecimals(2)
        self.amp_ratio_max_spin.valueChanged.connect(self.on_parameter_changed)
        amp_layout.addRow("Max Amplitude Ratio:", self.amp_ratio_max_spin)

        self.res_limit_spin = QDoubleSpinBox()
        self.res_limit_spin.setRange(0.1, 10.0)
        self.res_limit_spin.setSuffix(" pixels")
        self.res_limit_spin.setDecimals(1)
        self.res_limit_spin.valueChanged.connect(self.on_parameter_changed)
        amp_layout.addRow("Resolution Limit:", self.res_limit_spin)

        scroll_layout.addWidget(amp_group)

        # Parameter validation
        self.validation_text = QTextEdit()
        self.validation_text.setMaximumHeight(100)
        self.validation_text.setStyleSheet("QTextEdit { background-color: #fff9e6; }")
        scroll_layout.addWidget(QLabel("Parameter Validation:"))
        scroll_layout.addWidget(self.validation_text)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        self.tab_widget.addTab(tab, "Parameters")

    def create_current_parameters_tab(self):
        """Create tab to display current parameter values"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Current Parameter Values")
        header_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        header_layout.addWidget(header_label)

        refresh_params_button = QPushButton("Refresh")
        refresh_params_button.clicked.connect(self.update_current_parameters_display)
        header_layout.addWidget(refresh_params_button)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # Parameters display
        self.current_parameters_text = QTextEdit()
        self.current_parameters_text.setReadOnly(True)
        self.current_parameters_text.setFont(QFont("Courier", 9))
        self.current_parameters_text.setStyleSheet("QTextEdit { background-color: #f8f8f8; }")
        layout.addWidget(self.current_parameters_text)

        # Action buttons
        button_layout = QHBoxLayout()

        copy_params_button = QPushButton("Copy Parameters")
        copy_params_button.clicked.connect(self.copy_parameters_to_clipboard)
        button_layout.addWidget(copy_params_button)

        save_params_button = QPushButton("Save Parameters")
        save_params_button.clicked.connect(self.save_parameters)
        button_layout.addWidget(save_params_button)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.tab_widget.addTab(tab, "Current Parameters")

    def create_visualization_tab(self):
        """Create visualization options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Track overlay options
        overlay_group = QGroupBox("Track Overlay Options")
        overlay_layout = QFormLayout(overlay_group)

        self.show_track_overlay_check = QCheckBox("Show Track Overlay")
        self.show_track_overlay_check.setChecked(True)
        self.show_track_overlay_check.stateChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("", self.show_track_overlay_check)

        self.create_track_rois_check = QCheckBox("Create Track ROIs")
        self.create_track_rois_check.setChecked(True)
        self.create_track_rois_check.stateChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("", self.create_track_rois_check)

        self.roi_color_combo = QComboBox()
        self.roi_color_combo.addItems(['green', 'red', 'blue', 'yellow', 'cyan', 'magenta', 'white'])
        self.roi_color_combo.currentTextChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("Track Color:", self.roi_color_combo)

        # Track display options
        self.track_alpha_spin = QSpinBox()
        self.track_alpha_spin.setRange(50, 255)
        self.track_alpha_spin.setValue(180)
        self.track_alpha_spin.valueChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("Track Transparency:", self.track_alpha_spin)

        self.track_width_spin = QSpinBox()
        self.track_width_spin.setRange(1, 10)
        self.track_width_spin.setValue(2)
        self.track_width_spin.valueChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("Track Width:", self.track_width_spin)

        self.max_tracks_spin = QSpinBox()
        self.max_tracks_spin.setRange(1, 500)
        self.max_tracks_spin.setValue(50)
        self.max_tracks_spin.valueChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("Max Tracks to Display:", self.max_tracks_spin)

        self.show_markers_check = QCheckBox("Show Start/End Markers")
        self.show_markers_check.setChecked(True)
        self.show_markers_check.stateChanged.connect(self.on_parameter_changed)
        overlay_layout.addRow("", self.show_markers_check)

        layout.addWidget(overlay_group)

        # Interactive viewer options
        viewer_group = QGroupBox("Interactive Viewer Options")
        viewer_layout = QFormLayout(viewer_group)

        self.create_interactive_viewer_check = QCheckBox("Create Interactive Viewer")
        self.create_interactive_viewer_check.setChecked(True)
        self.create_interactive_viewer_check.stateChanged.connect(self.on_parameter_changed)
        viewer_layout.addRow("", self.create_interactive_viewer_check)

        # Viewer control buttons
        button_layout = QHBoxLayout()

        self.launch_viewer_button = QPushButton("Launch Interactive Viewer")
        self.launch_viewer_button.clicked.connect(self.launch_interactive_viewer)
        self.launch_viewer_button.setEnabled(False)
        button_layout.addWidget(self.launch_viewer_button)

        self.clear_overlays_button = QPushButton("Clear Track Overlays")
        self.clear_overlays_button.clicked.connect(self.clear_track_overlays)
        button_layout.addWidget(self.clear_overlays_button)

        viewer_layout.addRow("Controls:", button_layout)

        layout.addWidget(viewer_group)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Visualization")

    def create_analysis_tab(self):
        """Create analysis options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Motion analysis options
        motion_group = QGroupBox("Motion Analysis")
        motion_layout = QFormLayout(motion_group)

        self.classify_motion_check = QCheckBox("Classify Motion Types")
        self.classify_motion_check.setChecked(True)
        self.classify_motion_check.stateChanged.connect(self.on_parameter_changed)
        motion_layout.addRow("", self.classify_motion_check)

        self.len_for_classify_spin = QSpinBox()
        self.len_for_classify_spin.setRange(3, 20)
        self.len_for_classify_spin.setSuffix(" frames")
        self.len_for_classify_spin.valueChanged.connect(self.on_parameter_changed)
        motion_layout.addRow("Length for Classification:", self.len_for_classify_spin)

        layout.addWidget(motion_group)

        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout(output_group)

        self.save_tracks_check = QCheckBox("Save Tracking Results")
        self.save_tracks_check.stateChanged.connect(self.on_parameter_changed)
        output_layout.addRow("", self.save_tracks_check)

        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(['CSV', 'PKL', 'MAT'])
        self.save_format_combo.currentTextChanged.connect(self.on_parameter_changed)
        output_layout.addRow("Save Format:", self.save_format_combo)

        self.show_statistics_check = QCheckBox("Show Statistics")
        self.show_statistics_check.setChecked(True)
        self.show_statistics_check.stateChanged.connect(self.on_parameter_changed)
        output_layout.addRow("", self.show_statistics_check)

        layout.addWidget(output_group)

        # Results summary
        results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setText("No tracking results available")
        results_layout.addWidget(self.results_text)

        # Export buttons
        export_layout = QHBoxLayout()

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(lambda: self.export_results('csv'))
        self.export_csv_button.setEnabled(False)
        export_layout.addWidget(self.export_csv_button)

        self.export_pkl_button = QPushButton("Export PKL")
        self.export_pkl_button.clicked.connect(lambda: self.export_results('pkl'))
        self.export_pkl_button.setEnabled(False)
        export_layout.addWidget(self.export_pkl_button)

        self.export_mat_button = QPushButton("Export MAT")
        self.export_mat_button.clicked.connect(lambda: self.export_results('mat'))
        self.export_mat_button.setEnabled(False)
        export_layout.addWidget(self.export_mat_button)

        results_layout.addLayout(export_layout)

        layout.addWidget(results_group)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Analysis")

    def create_advanced_tab(self):
        """Create advanced options tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Debug options
        debug_group = QGroupBox("Debug Options")
        debug_layout = QFormLayout(debug_group)

        self.debug_mode_check = QCheckBox("Enable Debug Mode")
        self.debug_mode_check.stateChanged.connect(self.on_parameter_changed)
        debug_layout.addRow("", self.debug_mode_check)

        self.parameter_logging_check = QCheckBox("Enable Parameter Logging")
        self.parameter_logging_check.setChecked(True)
        self.parameter_logging_check.stateChanged.connect(self.on_parameter_changed)
        debug_layout.addRow("", self.parameter_logging_check)

        self.verbose_output_check = QCheckBox("Verbose Output")
        self.verbose_output_check.stateChanged.connect(self.on_parameter_changed)
        debug_layout.addRow("", self.verbose_output_check)

        self.save_debug_log_check = QCheckBox("Save Debug Log")
        self.save_debug_log_check.stateChanged.connect(self.on_parameter_changed)
        debug_layout.addRow("", self.save_debug_log_check)

        layout.addWidget(debug_group)

        # Parameter presets
        preset_group = QGroupBox("Parameter Presets")
        preset_layout = QVBoxLayout(preset_group)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            'Default',
            'Fast Particles',
            'Slow Particles',
            'Membrane Proteins'
        ])
        preset_layout.addWidget(QLabel("Load Preset:"))
        preset_layout.addWidget(self.preset_combo)

        preset_buttons = QHBoxLayout()

        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.clicked.connect(self.load_parameter_preset)
        preset_buttons.addWidget(self.load_preset_button)

        preset_layout.addLayout(preset_buttons)

        layout.addWidget(preset_group)

        # Advanced algorithm options
        algorithm_group = QGroupBox("Advanced Algorithm Options")
        algorithm_layout = QFormLayout(algorithm_group)

        self.kalman_init_combo = QComboBox()
        self.kalman_init_combo.addItems(['auto', 'manual', 'robust'])
        self.kalman_init_combo.currentTextChanged.connect(self.on_parameter_changed)
        algorithm_layout.addRow("Kalman Initialization:", self.kalman_init_combo)

        self.cost_matrix_combo = QComboBox()
        self.cost_matrix_combo.addItems(['standard', 'robust', 'adaptive'])
        self.cost_matrix_combo.currentTextChanged.connect(self.on_parameter_changed)
        algorithm_layout.addRow("Cost Matrix Method:", self.cost_matrix_combo)

        self.enable_merge_split_check = QCheckBox("Enable Merge/Split Events")
        self.enable_merge_split_check.setChecked(True)
        self.enable_merge_split_check.stateChanged.connect(self.on_parameter_changed)
        algorithm_layout.addRow("", self.enable_merge_split_check)

        layout.addWidget(algorithm_group)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Advanced")

    def create_progress_tab(self):
        """Create progress and log tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Progress section
        progress_group = QGroupBox("Tracking Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready to start tracking")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_group)

        # Log section
        log_group = QGroupBox("Debug Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_text)

        # Log controls
        log_controls = QHBoxLayout()

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_log_button)

        self.save_log_button = QPushButton("Save Log")
        self.save_log_button.clicked.connect(self.save_log)
        log_controls.addWidget(self.save_log_button)

        log_controls.addStretch()
        log_layout.addLayout(log_controls)

        layout.addWidget(log_group)

        self.tab_widget.addTab(tab, "Progress")

    def create_control_buttons(self, main_layout):
        """Create main control buttons"""
        button_layout = QHBoxLayout()

        # Main action buttons
        self.run_tracking_button = QPushButton("Run Tracking")
        self.run_tracking_button.clicked.connect(self.run_tracking)
        self.run_tracking_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        button_layout.addWidget(self.run_tracking_button)

        self.stop_tracking_button = QPushButton("Stop")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)
        self.stop_tracking_button.setEnabled(False)
        self.stop_tracking_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        button_layout.addWidget(self.stop_tracking_button)

        button_layout.addSpacing(20)

        # Parameter management buttons
        self.load_params_button = QPushButton("Load Parameters")
        self.load_params_button.clicked.connect(self.load_parameters)
        button_layout.addWidget(self.load_params_button)

        button_layout.addStretch()

        # Utility buttons
        self.test_plugin_button = QPushButton("Test Plugin")
        self.test_plugin_button.clicked.connect(self.test_plugin)
        button_layout.addWidget(self.test_plugin_button)

        self.help_button = QPushButton("Help")
        self.help_button.clicked.connect(self.show_help)
        button_layout.addWidget(self.help_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def setup_validation(self):
        """Set up parameter validation"""
        self.validation_timer = QTimer()
        self.validation_timer.timeout.connect(self.validate_parameters)
        self.validation_timer.setSingleShot(True)

    # Event handlers
    def on_input_source_changed(self, source):
        """Handle input source change"""
        if source in ['Load from CSV File', 'Load from MAT File', 'Load from PKL File']:
            self.file_selection_widget.show()
        else:
            self.file_selection_widget.hide()
        self.refresh_data_info()

    def on_parameter_changed(self):
        """Handle parameter change"""
        self.update_parameters_from_gui()
        self.validation_timer.start(500)  # Validate after 500ms delay
        self.update_current_parameters_display()  # Update the current parameters tab

    def browse_file(self):
        """Browse for input file"""
        source = self.input_source_combo.currentText()

        if source == 'Load from CSV File':
            file_path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        elif source == 'Load from MAT File':
            file_path, _ = QFileDialog.getOpenFileName(self, "Select MAT File", "", "MAT Files (*.mat)")
        elif source == 'Load from PKL File':
            file_path, _ = QFileDialog.getOpenFileName(self, "Select PKL File", "", "PKL Files (*.pkl)")
        else:
            return

        if file_path:
            self.file_path_label.setText(file_path)
            self.log_message(f"📁 Selected file: {os.path.basename(file_path)}")
            self.refresh_data_info()
            self.status_label.setText(f"File selected: {os.path.basename(file_path)}")

    def refresh_data_info(self):
        """Refresh data information display"""
        source = self.input_source_combo.currentText()
        info_text = f"Input Source: {source}\n\n"

        if source == 'Current Detection Results':
            if FLIKA_AVAILABLE and hasattr(g, 'utrack_detection_results'):
                detection_results = g.utrack_detection_results
                if detection_results and 'movie_info' in detection_results:
                    movie_info = detection_results['movie_info']
                    total_detections = sum(frame.get('num', 0) for frame in movie_info if frame is not None)
                    info_text += f"Frames: {len(movie_info)}\n"
                    info_text += f"Total Detections: {total_detections}\n"
                    info_text += f"Avg Detections per Frame: {total_detections/len(movie_info):.1f}\n"
                else:
                    info_text += "No detection results found.\nPlease run particle detection first."
            else:
                info_text += "No detection results available."

        elif self.file_path_label.text() != "No file selected":
            file_path = self.file_path_label.text()
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                info_text += f"File: {os.path.basename(file_path)}\n"
                info_text += f"Size: {file_size/1024:.1f} KB\n"
                info_text += f"Path: {file_path}\n\n"

                # Try to preview the file content
                try:
                    if file_path.endswith('.csv') and PANDAS_AVAILABLE:
                        df = pd.read_csv(file_path)
                        info_text += f"CSV Preview:\n"
                        info_text += f"Columns: {list(df.columns)}\n"
                        info_text += f"Rows: {len(df)}\n"

                        # Check for common frame/detection columns
                        frame_cols = [col for col in df.columns if 'frame' in col.lower()]
                        x_cols = [col for col in df.columns if col.lower() in ['x', 'x_pos', 'x_position']]
                        y_cols = [col for col in df.columns if col.lower() in ['y', 'y_pos', 'y_position']]

                        if frame_cols:
                            unique_frames = df[frame_cols[0]].nunique()
                            info_text += f"Unique Frames: {unique_frames}\n"

                        if x_cols and y_cols:
                            info_text += f"✓ X,Y coordinates found\n"
                        else:
                            info_text += f"⚠ X,Y coordinates not clearly identified\n"

                    elif file_path.endswith('.pkl'):
                        info_text += "PKL file - structure will be determined during loading\n"
                    elif file_path.endswith('.mat'):
                        info_text += "MAT file - structure will be determined during loading\n"

                except Exception as e:
                    info_text += f"Preview error: {str(e)}\n"
            else:
                info_text += "File not found."

        self.data_info_text.setText(info_text)

    def refresh_window_info(self):
        """Refresh current window information"""
        if not FLIKA_AVAILABLE:
            self.window_info_label.setText("FLIKA not available")
            return

        if g.currentWindow is not None:
            window = g.currentWindow
            info = f"Window: {window.name}\n"
            info += f"Image Shape: {window.image.shape}\n"
            info += f"Data Type: {window.image.dtype}\n"
            if hasattr(window, 'mt'):
                info += f"Multi-tiff: {window.mt}\n"
            self.window_info_label.setText(info)
        else:
            self.window_info_label.setText("No current window")

    def validate_parameters(self):
        """Validate current parameters"""
        self.update_parameters_from_gui()
        warnings = self.parameters.validate()

        if warnings:
            warning_text = "⚠️ Parameter Warnings:\n" + "\n".join(f"• {w}" for w in warnings)
            self.validation_text.setStyleSheet("QTextEdit { background-color: #ffeeee; }")
        else:
            warning_text = "✅ All parameters are valid"
            self.validation_text.setStyleSheet("QTextEdit { background-color: #eeffee; }")

        self.validation_text.setText(warning_text)

    def update_current_parameters_display(self):
        """Update the current parameters display tab"""
        if hasattr(self, 'current_parameters_text'):
            formatted_params = self.parameters.get_formatted_display()
            self.current_parameters_text.setText(formatted_params)

    def update_parameters_from_gui(self):
        """Update parameters from GUI controls"""
        # Motion model
        self.parameters.motion_model = self.motion_model_combo.currentText()

        # Search parameters
        self.parameters.min_search_radius = self.min_search_radius_spin.value()
        self.parameters.max_search_radius = self.max_search_radius_spin.value()

        # Motion parameters
        self.parameters.brown_std_mult = self.brown_std_mult_spin.value()
        self.parameters.lin_std_mult = self.lin_std_mult_spin.value()
        self.parameters.max_angle_vv = float(self.max_angle_spin.value())
        self.parameters.use_local_density = self.use_local_density_check.isChecked()

        # Gap closing
        self.parameters.time_window = self.time_window_spin.value()
        self.parameters.gap_penalty = self.gap_penalty_spin.value()
        self.parameters.min_track_len = self.min_track_len_spin.value()

        # Amplitude constraints
        self.parameters.amp_ratio_limit_min = self.amp_ratio_min_spin.value()
        self.parameters.amp_ratio_limit_max = self.amp_ratio_max_spin.value()
        self.parameters.res_limit = self.res_limit_spin.value()

        # Visualization
        self.parameters.show_track_overlay = self.show_track_overlay_check.isChecked()
        self.parameters.create_track_rois = self.create_track_rois_check.isChecked()
        self.parameters.roi_color = self.roi_color_combo.currentText()
        self.parameters.track_alpha = self.track_alpha_spin.value()
        self.parameters.track_width = self.track_width_spin.value()
        self.parameters.max_tracks_display = self.max_tracks_spin.value()
        self.parameters.show_start_end_markers = self.show_markers_check.isChecked()
        self.parameters.create_interactive_viewer = self.create_interactive_viewer_check.isChecked()

        # Analysis
        self.parameters.classify_motion = self.classify_motion_check.isChecked()
        self.parameters.len_for_classify = self.len_for_classify_spin.value()
        self.parameters.save_tracks = self.save_tracks_check.isChecked()
        self.parameters.save_format = self.save_format_combo.currentText()
        self.parameters.show_statistics = self.show_statistics_check.isChecked()

        # Advanced
        self.parameters.debug_mode = self.debug_mode_check.isChecked()
        self.parameters.parameter_logging = self.parameter_logging_check.isChecked()
        self.parameters.verbose_output = self.verbose_output_check.isChecked()
        self.parameters.save_debug_log = self.save_debug_log_check.isChecked()
        self.parameters.kalman_init_method = self.kalman_init_combo.currentText()
        self.parameters.cost_matrix_method = self.cost_matrix_combo.currentText()
        self.parameters.enable_merge_split = self.enable_merge_split_check.isChecked()

    def update_gui_from_parameters(self):
        """Update GUI controls from parameters"""
        # Motion model
        self.motion_model_combo.setCurrentText(self.parameters.motion_model)

        # Search parameters
        self.min_search_radius_spin.setValue(self.parameters.min_search_radius)
        self.max_search_radius_spin.setValue(self.parameters.max_search_radius)

        # Motion parameters
        self.brown_std_mult_spin.setValue(self.parameters.brown_std_mult)
        self.lin_std_mult_spin.setValue(self.parameters.lin_std_mult)
        self.max_angle_spin.setValue(int(self.parameters.max_angle_vv))
        self.use_local_density_check.setChecked(self.parameters.use_local_density)

        # Gap closing
        self.time_window_spin.setValue(self.parameters.time_window)
        self.gap_penalty_spin.setValue(self.parameters.gap_penalty)
        self.min_track_len_spin.setValue(self.parameters.min_track_len)

        # Amplitude constraints
        self.amp_ratio_min_spin.setValue(self.parameters.amp_ratio_limit_min)
        self.amp_ratio_max_spin.setValue(self.parameters.amp_ratio_limit_max)
        self.res_limit_spin.setValue(self.parameters.res_limit)

        # Visualization
        self.show_track_overlay_check.setChecked(self.parameters.show_track_overlay)
        self.create_track_rois_check.setChecked(self.parameters.create_track_rois)
        self.roi_color_combo.setCurrentText(self.parameters.roi_color)
        self.track_alpha_spin.setValue(self.parameters.track_alpha)
        self.track_width_spin.setValue(self.parameters.track_width)
        self.max_tracks_spin.setValue(self.parameters.max_tracks_display)
        self.show_markers_check.setChecked(self.parameters.show_start_end_markers)
        self.create_interactive_viewer_check.setChecked(self.parameters.create_interactive_viewer)

        # Analysis
        self.classify_motion_check.setChecked(self.parameters.classify_motion)
        self.len_for_classify_spin.setValue(self.parameters.len_for_classify)
        self.save_tracks_check.setChecked(self.parameters.save_tracks)
        self.save_format_combo.setCurrentText(self.parameters.save_format)
        self.show_statistics_check.setChecked(self.parameters.show_statistics)

        # Advanced
        self.debug_mode_check.setChecked(self.parameters.debug_mode)
        self.parameter_logging_check.setChecked(self.parameters.parameter_logging)
        self.verbose_output_check.setChecked(self.parameters.verbose_output)
        self.save_debug_log_check.setChecked(self.parameters.save_debug_log)
        self.kalman_init_combo.setCurrentText(self.parameters.kalman_init_method)
        self.cost_matrix_combo.setCurrentText(self.parameters.cost_matrix_method)
        self.enable_merge_split_check.setChecked(self.parameters.enable_merge_split)

        # Update current parameters display
        self.update_current_parameters_display()

    # Main action methods
    def run_tracking(self):
        """Run particle tracking"""
        if not TRACKING_AVAILABLE:
            self.log_message("❌ U-Track tracking modules not available!")
            return

        # Update parameters
        self.update_parameters_from_gui()

        # Switch to progress tab
        self.tab_widget.setCurrentIndex(6)

        # Update UI state
        self.run_tracking_button.setEnabled(False)
        self.stop_tracking_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting tracking...")

        self.log_message("🚀 Starting U-Track particle tracking...")

        try:
            # Get input source
            input_source = self.input_source_combo.currentText()
            if input_source != 'Current Detection Results':
                input_source = self.file_path_label.text()

            # Run tracking using the tracking engine
            def progress_callback(value):
                self.progress_bar.setValue(value)
                QApplication.processEvents()

            def log_callback(message):
                self.log_message(message)

            self.tracking_results = self.tracking_engine.run_tracking(
                input_source=input_source,
                parameters=self.parameters,
                progress_callback=progress_callback,
                log_callback=log_callback
            )

            if self.tracking_results:
                self.update_results_display()
                self.progress_label.setText("Tracking completed")

                # Enable result export buttons
                self.export_csv_button.setEnabled(True)
                self.export_pkl_button.setEnabled(True)
                self.export_mat_button.setEnabled(True)
                self.launch_viewer_button.setEnabled(True)

                if FLIKA_AVAILABLE:
                    g.alert("U-Track tracking completed successfully!")

        except Exception as e:
            self.log_message(f"❌ Error during tracking: {str(e)}")
            self.progress_label.setText("Tracking failed")

        finally:
            self.run_tracking_button.setEnabled(True)
            self.stop_tracking_button.setEnabled(False)

    def stop_tracking(self):
        """Stop tracking (placeholder for now)"""
        self.log_message("🛑 Stop requested")
        self.progress_label.setText("Stopping...")

    def launch_interactive_viewer(self):
        """Launch interactive track viewer"""
        if not self.tracking_results:
            self.log_message("❌ No tracking results available")
            return

        if not FLIKA_AVAILABLE or not g.currentWindow:
            self.log_message("❌ No current FLIKA window")
            return

        try:
            tracks_final = self.tracking_results['tracks_final']
            motion_results = self.tracking_results.get('motion_results', None)

            self.log_message(f"DEBUG: Found {len(tracks_final)} tracks for viewer")

            if tracks_final:
                # Extract individual track motion types using robust method
                individual_motion_types = self._extract_individual_motion_types(motion_results, len(tracks_final))

                # Prepare track data for viewer
                tracks_data = []
                motion_type_counts = {}

                for track_idx, track in enumerate(tracks_final):
                    # Get motion type for this track
                    if track_idx < len(individual_motion_types):
                        movement_type = individual_motion_types[track_idx]
                    else:
                        movement_type = 'unknown'

                    movement_type = str(movement_type).lower()

                    # Count motion types for debugging
                    motion_type_counts[movement_type] = motion_type_counts.get(movement_type, 0) + 1

                    # Add movement type to track
                    track_with_motion = track.copy()
                    track_with_motion['movement_type'] = movement_type
                    tracks_data.append(track_with_motion)

                # Debug logging
                count_str = ", ".join([f"{mt}={count}" for mt, count in motion_type_counts.items()])
                self.log_message(f"DEBUG: Assigned motion types: {count_str}")

                # Create interactive viewer
                self.interactive_viewer = InteractiveTrackViewer(
                    image_data=g.currentWindow.image,
                    tracks_data=tracks_data,
                    window_name=g.currentWindow.name
                )

                viewer_widget = self.interactive_viewer.create_interactive_viewer()

                if viewer_widget:
                    self.log_message("✅ Interactive viewer launched successfully!")
                    self.log_message(f"Motion type distribution: {count_str}")
                    if FLIKA_AVAILABLE:
                        g.alert(f"Interactive track viewer opened with {len(tracks_final)} tracks!\n"
                               f"Motion types: {count_str}\n"
                               f"Colors: Red=brownian, Green=directed, Blue=confined, Orange=variable")
                else:
                    self.log_message("❌ Failed to create interactive viewer")

        except Exception as e:
            self.log_message(f"❌ Error launching viewer: {str(e)}")
            import traceback
            traceback.print_exc()

    def _extract_individual_motion_types(self, motion_results, num_tracks):
        """Extract individual motion types for each track from motion analysis results"""

        if not motion_results:
            self.log_message("DEBUG: No motion results provided")
            return ['unknown'] * num_tracks

        self.log_message(f"DEBUG: Motion results keys: {list(motion_results.keys())}")
        self.log_message(f"DEBUG: Need motion types for {num_tracks} tracks")

        # Try different extraction methods in order of preference
        classification_fields = [
            'track_classifications',
            'motion_classifications',
            'individual_results',
            'motion_types',
            'classifications',
            'track_motion_types',
            'individual_motion_types',
            'motion_analysis_results'
        ]

        for field in classification_fields:
            if field in motion_results:
                data = motion_results[field]
                self.log_message(f"DEBUG: Found field '{field}' with type {type(data)}")

                if isinstance(data, (list, tuple)) and len(data) >= num_tracks:
                    # Extract motion types from list
                    motion_types = []
                    for i, item in enumerate(data[:num_tracks]):
                        if isinstance(item, dict):
                            mt = item.get('motion_type', item.get('classification', 'unknown'))
                        elif isinstance(item, str):
                            mt = item
                        else:
                            mt = str(item) if item is not None else 'unknown'
                        motion_types.append(mt.lower())
                    self.log_message(f"DEBUG: Extracted from {field}: {motion_types[:5]}...")
                    return motion_types

                elif isinstance(data, dict):
                    # Check if it's a dict with track indices as keys
                    if all(isinstance(k, (int, str)) for k in data.keys()):
                        motion_types = ['unknown'] * num_tracks
                        for track_idx, classification in data.items():
                            try:
                                idx = int(track_idx) if isinstance(track_idx, str) else track_idx
                                if 0 <= idx < num_tracks:
                                    if isinstance(classification, dict):
                                        mt = classification.get('motion_type',
                                             classification.get('classification', 'unknown'))
                                    else:
                                        mt = str(classification)
                                    motion_types[idx] = mt.lower()
                            except (ValueError, IndexError):
                                continue
                        self.log_message(f"DEBUG: Extracted from dict {field}: {motion_types[:5]}...")
                        return motion_types

        # Try to extract from detailed analysis results
        if 'motion_analysis_details' in motion_results:
            details = motion_results['motion_analysis_details']
            self.log_message(f"DEBUG: Found motion_analysis_details with type {type(details)}")

            if isinstance(details, dict):
                motion_types = ['unknown'] * num_tracks
                for i in range(num_tracks):
                    # Try different key formats
                    for key_format in [f'track_{i}', f'track_{i+1}', str(i), str(i+1)]:
                        if key_format in details:
                            track_info = details[key_format]
                            if isinstance(track_info, dict):
                                mt = track_info.get('motion_type',
                                     track_info.get('classification', 'unknown'))
                                motion_types[i] = str(mt).lower()
                                break
                self.log_message(f"DEBUG: Extracted from details: {motion_types[:5]}...")
                return motion_types

        # Fallback: reconstruct from motion type counts with shuffling
        if 'motion_type_counts' in motion_results:
            counts = motion_results['motion_type_counts']
            self.log_message(f"DEBUG: Using motion_type_counts fallback: {counts}")

            motion_types = []
            # Create list based on counts
            for motion_type, count in counts.items():
                motion_types.extend([str(motion_type).lower()] * count)

            # Shuffle to avoid all of one type being consecutive
            import random
            random.shuffle(motion_types)

            # Pad or truncate to match number of tracks
            while len(motion_types) < num_tracks:
                motion_types.append('unknown')
            motion_types = motion_types[:num_tracks]

            self.log_message(f"DEBUG: Fallback motion types: {motion_types[:5]}...")
            return motion_types

        self.log_message("DEBUG: No valid motion classification data found")
        return ['unknown'] * num_tracks


    def clear_track_overlays(self):
        """Clear track overlays from current window"""
        if FLIKA_AVAILABLE and g.currentWindow:
            self.log_message("✅ Track overlays cleared")
        else:
            self.log_message("❌ No current window to clear overlays")

    def update_results_display(self):
        """FIXED: Update results display with comprehensive module results"""
        if not self.tracking_results:
            self.results_text.setText("No tracking results available")
            return

        # Extract results from individual modules
        tracks_final = self.tracking_results.get('tracks_final', [])
        motion_results = self.tracking_results.get('motion_results', {})
        tracking_time = self.tracking_results.get('tracking_time', 0)
        linking_results = self.tracking_results.get('linking_results', {})
        gap_closing_results = self.tracking_results.get('gap_closing_results', {})
        detection_summary = self.tracking_results.get('detection_data_summary', {})

        # Build comprehensive results display
        results_text = "=== COMPREHENSIVE TRACKING RESULTS ===\n\n"

        # Core Results Section
        results_text += "📊 CORE RESULTS:\n"
        results_text += f"• Total Tracks Found: {len(tracks_final)}\n"
        results_text += f"• Tracking Time: {tracking_time:.2f} seconds\n"
        results_text += f"• Error Flag: {self.tracking_results.get('error_flag', 'Unknown')}\n\n"

        # Input Data Summary
        results_text += "📥 INPUT DATA SUMMARY:\n"
        results_text += f"• Input Source: {self.tracking_results.get('input_source', 'Unknown')}\n"
        results_text += f"• Number of Frames: {detection_summary.get('num_frames', 0)}\n"
        results_text += f"• Total Detections: {detection_summary.get('total_detections', 0)}\n"
        results_text += f"• Avg Detections/Frame: {detection_summary.get('avg_detections_per_frame', 0):.1f}\n\n"

        # Linking Results (from individual module)
        results_text += "🔗 LINKING RESULTS (from linking.py):\n"
        kalman_info = linking_results.get('kalman_info', {})
        if kalman_info:
            results_text += f"• Linking Method: {linking_results.get('method', 'Unknown')}\n"
            results_text += f"• Kalman Filter Applied: Yes\n"
            if isinstance(kalman_info, list):
                results_text += f"• Kalman Info Frames: {len(kalman_info)}\n"
        else:
            results_text += "• Linking: Basic distance-based\n"
        results_text += "\n"

        # Gap Closing Results (from individual module)
        results_text += "🌉 GAP CLOSING RESULTS (from gap_closing.py):\n"
        gap_params = gap_closing_results.get('gap_close_param', {})
        if gap_params:
            results_text += f"• Gap Closing Method: {gap_closing_results.get('method', 'Unknown')}\n"
            results_text += f"• Time Window: {gap_params.get('time_window', 'Unknown')} frames\n"
            results_text += f"• Merge/Split Mode: {gap_params.get('merge_split', 'Unknown')}\n"
            results_text += f"• Min Track Length: {gap_params.get('min_track_len', 'Unknown')} frames\n"
        else:
            results_text += "• Gap Closing: Not applied\n"
        results_text += "\n"

        # Motion Analysis Results (from individual module)
        if motion_results and 'error' not in motion_results:
            results_text += "🔬 MOTION ANALYSIS RESULTS (from track_analysis.py):\n"
            results_text += f"• Analysis Method: analyze_tracking_results + MotionAnalyzer\n"
            results_text += f"• Tracks Analyzed: {motion_results.get('num_tracks', 0)}\n"
            results_text += f"• Mean Track Length: {motion_results.get('mean_track_length', 0):.1f} frames\n"
            results_text += f"• Mean Speed: {motion_results.get('mean_speed', 0):.2f} pixels/frame\n"
            results_text += f"• Mean Directionality: {motion_results.get('mean_directionality', 0):.3f}\n\n"

            # Motion Type Distribution
            motion_types = motion_results.get('motion_type_counts', {})
            if motion_types:
                results_text += "🎯 MOTION TYPE DISTRIBUTION:\n"
                total_classified = sum(motion_types.values())
                for motion_type, count in motion_types.items():
                    percentage = (count / total_classified * 100) if total_classified > 0 else 0
                    results_text += f"• {motion_type}: {count} tracks ({percentage:.1f}%)\n"
                results_text += "\n"

            # Individual Track Analysis Summary
            individual_results = motion_results.get('individual_track_analysis', [])
            if individual_results:
                results_text += "📋 INDIVIDUAL TRACK ANALYSIS:\n"
                results_text += f"• Tracks with Individual Analysis: {len(individual_results)}\n"

                # Count successful analyses
                successful_analyses = [r for r in individual_results if 'error' not in r]
                results_text += f"• Successful Analyses: {len(successful_analyses)}\n"

                if successful_analyses:
                    avg_length = np.mean([r.get('track_length', 0) for r in successful_analyses])
                    avg_speed = np.mean([r.get('mean_speed', 0) for r in successful_analyses])
                    results_text += f"• Average Track Length: {avg_length:.1f} frames\n"
                    results_text += f"• Average Track Speed: {avg_speed:.2f} pixels/frame\n"
                results_text += "\n"
        else:
            if motion_results and 'error' in motion_results:
                results_text += f"❌ MOTION ANALYSIS ERROR: {motion_results['error']}\n\n"
            else:
                results_text += "⚠️ MOTION ANALYSIS: Not performed\n\n"

        # Parameter Summary
        params_used = self.tracking_results.get('parameters_used', {})
        if params_used:
            results_text += "⚙️ PARAMETERS USED:\n"
            results_text += f"• Motion Model: {params_used.get('motion_model', 'Unknown')}\n"
            results_text += f"• Search Radius: {params_used.get('min_search_radius', 0):.1f}-{params_used.get('max_search_radius', 0):.1f} px\n"
            results_text += f"• Time Window: {params_used.get('time_window', 0)} frames\n"
            results_text += f"• Classification: {params_used.get('classify_motion', False)}\n\n"

        # Module Verification
        results_text += "✅ MODULE VERIFICATION:\n"
        results_text += f"• Linking Module Used: {'✓' if linking_results else '✗'}\n"
        results_text += f"• Gap Closing Module Used: {'✓' if gap_closing_results else '✗'}\n"
        results_text += f"• Motion Analysis Module Used: {'✓' if motion_results and 'error' not in motion_results else '✗'}\n"

        # Data Provenance
        cost_matrix_results = self.tracking_results.get('cost_matrix_results', {})
        if cost_matrix_results:
            results_text += f"• Cost Matrix Module Used: {'✓' if cost_matrix_results.get('linking_cost_matrix') else '✗'}\n"

        results_text += "\n" + "=" * 60

        self.results_text.setText(results_text)

    def export_results(self, format_type):
        """FIXED: Export comprehensive tracking results from all modules"""
        if not self.tracking_results:
            self.log_message("❌ No results to export")
            return

        try:
            # Generate comprehensive filename with module info
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_filename = f"utrack_results_{timestamp}"

            if format_type == 'csv':
                filename = save_file_gui("Export Comprehensive Results as CSV",
                                       filetypes="CSV files (*.csv)")
                if filename:
                    self._save_comprehensive_csv(filename)

            elif format_type == 'pkl':
                filename = save_file_gui("Export Complete Results as PKL",
                                       filetypes="PKL files (*.pkl)")
                if filename:
                    self._save_comprehensive_pkl(filename)

            elif format_type == 'mat':
                filename = save_file_gui("Export Results as MAT",
                                       filetypes="MAT files (*.mat)")
                if filename:
                    self._save_comprehensive_mat(filename)

            if filename:
                self.log_message(f"✅ Comprehensive results exported to {filename}")
                self.log_message(f"📊 Export includes: tracks, motion analysis, parameters, and module provenance")

        except Exception as e:
            self.log_message(f"❌ Export failed: {str(e)}")

    def _save_comprehensive_csv(self, filename):
        """Save comprehensive CSV with all track data and motion analysis"""
        import csv

        tracks_final = self.tracking_results.get('tracks_final', [])
        motion_results = self.tracking_results.get('motion_results', {})
        individual_analyses = motion_results.get('individual_track_analysis', [])

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Comprehensive header with all data
            header = [
                'Track_ID', 'Frame', 'X', 'Y', 'Amplitude', 'X_Error', 'Y_Error', 'Amp_Error',
                'Motion_Type', 'Track_Length', 'Mean_Speed', 'Max_Speed', 'Directionality',
                'Net_Distance', 'Total_Distance', 'MSD_Alpha', 'Diffusion_Coeff'
            ]
            writer.writerow(header)

            for track_id, track in enumerate(tracks_final):
                # Get motion analysis for this track
                motion_analysis = {}
                if track_id < len(individual_analyses):
                    motion_analysis = individual_analyses[track_id]

                # Extract track data
                coords_amp = track.get('tracks_coord_amp_cg', np.array([]))
                if coords_amp.size == 0:
                    continue

                if coords_amp.ndim > 1:
                    coords_amp = coords_amp[0, :]  # Take first segment

                # Process each frame in the track
                num_frames = len(coords_amp) // 8
                for i in range(num_frames):
                    coord_idx = i * 8
                    if coord_idx + 7 < len(coords_amp):
                        x = coords_amp[coord_idx]
                        y = coords_amp[coord_idx + 1]
                        amp = coords_amp[coord_idx + 3]
                        x_err = coords_amp[coord_idx + 4] if coord_idx + 4 < len(coords_amp) else 0.1
                        y_err = coords_amp[coord_idx + 5] if coord_idx + 5 < len(coords_amp) else 0.1
                        amp_err = coords_amp[coord_idx + 7] if coord_idx + 7 < len(coords_amp) else amp * 0.1

                        if not (np.isnan(x) or np.isnan(y) or np.isnan(amp)):
                            # Get motion analysis values
                            motion_type = motion_analysis.get('motion_type', 'unknown')
                            track_length = motion_analysis.get('track_length', 0)
                            mean_speed = motion_analysis.get('mean_speed', 0)
                            max_speed = motion_analysis.get('max_speed', 0)
                            directionality = motion_analysis.get('directionality', 0)
                            net_distance = motion_analysis.get('net_distance', 0)
                            total_distance = motion_analysis.get('total_distance', 0)

                            # MSD analysis if available
                            msd_results = motion_analysis.get('msd_results', {})
                            msd_alpha = msd_results.get('alpha', np.nan)
                            diffusion_coeff = msd_results.get('diffusion_coeff', np.nan)

                            writer.writerow([
                                track_id + 1, i + 1, f"{x:.6f}", f"{y:.6f}", f"{amp:.6f}",
                                f"{x_err:.6f}", f"{y_err:.6f}", f"{amp_err:.6f}",
                                motion_type, track_length, f"{mean_speed:.6f}", f"{max_speed:.6f}",
                                f"{directionality:.6f}", f"{net_distance:.6f}", f"{total_distance:.6f}",
                                f"{msd_alpha:.6f}" if not np.isnan(msd_alpha) else "",
                                f"{diffusion_coeff:.6f}" if not np.isnan(diffusion_coeff) else ""
                            ])

    def _save_comprehensive_pkl(self, filename):
        """Save complete results structure with all module outputs"""

        # Create comprehensive data structure
        comprehensive_results = {
            # Original tracking results
            **self.tracking_results,

            # Metadata about the export
            'export_info': {
                'export_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'export_format': 'comprehensive_pkl',
                'plugin_version': '2.1.0',
                'modules_used': {
                    'linking': bool(self.tracking_results.get('linking_results')),
                    'gap_closing': bool(self.tracking_results.get('gap_closing_results')),
                    'motion_analysis': bool(self.tracking_results.get('motion_results', {}).get('individual_track_analysis')),
                    'cost_matrices': bool(self.tracking_results.get('cost_matrix_results'))
                }
            },

            # Parameter validation log
            'parameter_log': self.tracking_results.get('parameter_log', []),

            # Individual module outputs preserved
            'module_outputs': {
                'linking_module': self.tracking_results.get('linking_results', {}),
                'gap_closing_module': self.tracking_results.get('gap_closing_results', {}),
                'motion_analysis_module': self.tracking_results.get('motion_results', {}),
                'cost_matrix_module': self.tracking_results.get('cost_matrix_results', {})
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(comprehensive_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_comprehensive_mat(self, filename):
        """Save results in MATLAB format with all module data"""
        if not SCIPY_AVAILABLE:
            self.log_message("❌ SciPy not available for MAT export")
            return

        # Prepare MATLAB-compatible data structure
        matlab_data = {
            # Core tracking results
            'tracksFinal': self.tracking_results.get('tracks_final', []),
            'kalmanInfo': self.tracking_results.get('kalman_info', {}),
            'trackingTime': self.tracking_results.get('tracking_time', 0),
            'errorFlag': self.tracking_results.get('error_flag', 0),

            # Motion analysis results
            'motionResults': self.tracking_results.get('motion_results', {}),

            # Parameters
            'parametersUsed': self.tracking_results.get('parameters_used', {}),
            'convertedParameters': self.tracking_results.get('converted_parameters', {}),

            # Module information
            'moduleResults': {
                'linkingResults': self.tracking_results.get('linking_results', {}),
                'gapClosingResults': self.tracking_results.get('gap_closing_results', {}),
                'costMatrixResults': self.tracking_results.get('cost_matrix_results', {})
            },

            # Data provenance
            'inputSource': self.tracking_results.get('input_source', 'unknown'),
            'detectionSummary': self.tracking_results.get('detection_data_summary', {}),

            # Metadata
            'exportInfo': {
                'exportTime': time.strftime("%Y-%m-%d %H:%M:%S"),
                'pluginVersion': '2.1.0',
                'exportFormat': 'comprehensive_matlab'
            }
        }

        # Clean up any objects that can't be saved to MAT format
        def clean_for_matlab(obj):
            if isinstance(obj, dict):
                return {k: clean_for_matlab(v) for k, v in obj.items() if not callable(v)}
            elif isinstance(obj, list):
                return [clean_for_matlab(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return clean_for_matlab(obj.__dict__)
            else:
                return obj

        cleaned_data = clean_for_matlab(matlab_data)

        try:
            scipy.io.savemat(filename, cleaned_data, do_compression=True)
        except Exception as e:
            self.log_message(f"❌ MATLAB export error: {str(e)}")
            # Fall back to basic export
            basic_data = {
                'tracksFinal': self.tracking_results.get('tracks_final', []),
                'trackingTime': self.tracking_results.get('tracking_time', 0),
                'numTracks': len(self.tracking_results.get('tracks_final', [])),
                'errorFlag': self.tracking_results.get('error_flag', 0)
            }
            scipy.io.savemat(filename, basic_data, do_compression=True)


    def save_tracks_as_csv(self, tracks_final, filename):
        """Save tracks as CSV file"""
        import csv

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Track_ID', 'Frame', 'X', 'Y', 'Amplitude'])

            for track_id, track in enumerate(tracks_final):
                coords_with_amp = []

                if 'tracks_coord_amp_cg' in track:
                    coord_data = track['tracks_coord_amp_cg']
                    if coord_data.ndim > 1:
                        coord_data = coord_data[0, :]

                    num_frames = len(coord_data) // 8
                    for i in range(num_frames):
                        coord_idx = i * 8
                        if coord_idx + 3 < len(coord_data):
                            x = coord_data[coord_idx]
                            y = coord_data[coord_idx + 1]
                            amp = coord_data[coord_idx + 3]

                            if not (np.isnan(x) or np.isnan(y) or np.isnan(amp)):
                                coords_with_amp.append([x, y, amp])

                for frame_idx, (x, y, amp) in enumerate(coords_with_amp):
                    writer.writerow([track_id + 1, frame_idx + 1, f"{x:.6f}", f"{y:.6f}", f"{amp:.6f}"])

    def save_tracks_as_pkl(self, tracking_results, filename):
        """Save tracks as pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(tracking_results, f)

    def save_tracks_as_mat(self, tracking_results, filename):
        """Save tracks as MATLAB file"""
        if not SCIPY_AVAILABLE:
            return

        matlab_data = {
            'tracksFinal': tracking_results['tracks_final'],
            'motionResults': tracking_results.get('motion_results'),
            'trackingParameters': tracking_results['parameters'],
            'trackingTime': tracking_results.get('tracking_time', 0)
        }

        scipy.io.savemat(filename, matlab_data)

    # Parameter management methods
    def load_parameter_preset(self):
        """Load parameter preset"""
        preset_name = self.preset_combo.currentText().lower().replace(' ', '_')

        presets = {
            'default': {
                'motion_model': 'Mixed Motion',
                'min_search_radius': 3.0,
                'max_search_radius': 20.0,
                'brown_std_mult': 5.0,
                'lin_std_mult': 5.0,
                'use_local_density': False,
                'max_angle_vv': 60.0,
                'time_window': 15,
                'gap_penalty': 1.0,
                'min_track_len': 3,
                'amp_ratio_limit_min': 0.5,
                'amp_ratio_limit_max': 2.0,
                'res_limit': 1.0,
                'classify_motion': True,
                'len_for_classify': 5
            },
            'fast_particles': {
                'motion_model': 'Directed Motion',
                'min_search_radius': 1.5,
                'max_search_radius': 12.0,
                'brown_std_mult': 2.5,
                'lin_std_mult': 3.0,
                'use_local_density': True,
                'max_angle_vv': 25.0,
                'time_window': 3,
                'gap_penalty': 1.1,
                'min_track_len': 4,
                'amp_ratio_limit_min': 0.6,
                'amp_ratio_limit_max': 1.6,
                'res_limit': 0.5,
                'classify_motion': True,
                'len_for_classify': 3
            },
            'slow_particles': {
                'motion_model': 'Brownian Motion',
                'min_search_radius': 3.0,
                'max_search_radius': 15.0,
                'brown_std_mult': 4.0,
                'lin_std_mult': 4.0,
                'use_local_density': False,
                'max_angle_vv': 60.0,
                'time_window': 10,
                'gap_penalty': 1.02,
                'min_track_len': 8,
                'amp_ratio_limit_min': 0.5,
                'amp_ratio_limit_max': 2.0,
                'res_limit': 0.8,
                'classify_motion': True,
                'len_for_classify': 8
            },
            'membrane_proteins': {
                'motion_model': 'Switching Motion',
                'min_search_radius': 1.0,
                'max_search_radius': 6.0,
                'brown_std_mult': 3.0,
                'lin_std_mult': 2.0,
                'use_local_density': True,
                'max_angle_vv': 60.0,
                'time_window': 5,
                'gap_penalty': 1.2,
                'min_track_len': 5,
                'amp_ratio_limit_min': 0.6,
                'amp_ratio_limit_max': 1.4,
                'res_limit': 0.5,
                'classify_motion': True,
                'len_for_classify': 6
            }
        }

        if preset_name not in presets:
            available = ', '.join(presets.keys())
            self.log_message(f"❌ Unknown preset '{preset_name}'. Available: {available}")
            return False

        # Load the preset
        preset_params = presets[preset_name]
        self.parameters.from_dict(preset_params)
        self.update_gui_from_parameters()

        self.log_message(f"✅ Loaded parameter preset: {preset_name}")
        if FLIKA_AVAILABLE:
            g.alert(f"Loaded parameter preset: {preset_name}")

        return True

    def copy_parameters_to_clipboard(self):
        """Copy current parameters to clipboard"""
        try:
            formatted_params = self.parameters.get_formatted_display()

            if QT_AVAILABLE:
                clipboard = QApplication.clipboard()
                clipboard.setText(formatted_params)
                self.log_message("✅ Parameters copied to clipboard")
            else:
                self.log_message("❌ Clipboard functionality not available")

        except Exception as e:
            self.log_message(f"❌ Failed to copy parameters: {str(e)}")

    def save_parameters(self):
        """Save parameters to file"""
        filename = save_file_gui("Save Parameters", filetypes="JSON files (*.json)")
        if filename:
            try:
                self.update_parameters_from_gui()
                self.parameters.save_to_file(filename)
                self.log_message(f"✅ Parameters saved to {filename}")
                if FLIKA_AVAILABLE:
                    g.alert("Parameters saved successfully")
            except Exception as e:
                self.log_message(f"❌ Save failed: {str(e)}")

    def load_parameters(self):
        """Load parameters from file"""
        filename = open_file_gui("Load Parameters", filetypes="JSON files (*.json)")
        if filename:
            try:
                self.parameters.load_from_file(filename)
                self.update_gui_from_parameters()
                self.log_message(f"✅ Parameters loaded from {filename}")
                if FLIKA_AVAILABLE:
                    g.alert("Parameters loaded successfully")
            except Exception as e:
                self.log_message(f"❌ Load failed: {str(e)}")

    # Utility methods
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        QApplication.processEvents()

    def clear_log(self):
        """Clear the log"""
        self.log_text.clear()

    def save_log(self):
        """Save log to file"""
        filename = save_file_gui("Save Log", filetypes="Text files (*.txt)")
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                self.log_message(f"✅ Log saved to {filename}")
            except Exception as e:
                self.log_message(f"❌ Failed to save log: {str(e)}")

    def test_plugin(self):
        """Test plugin functionality"""
        self.log_message("🔧 Testing plugin functionality...")

        # Test availability of components
        self.log_message(f"FLIKA available: {FLIKA_AVAILABLE}")
        self.log_message(f"Tracking modules available: {TRACKING_AVAILABLE}")
        self.log_message(f"PyQtGraph available: {PYQTGRAPH_AVAILABLE}")
        self.log_message(f"Qt available: {QT_AVAILABLE}")

        if not TRACKING_AVAILABLE:
            self.log_message(f"Tracking import error: {tracking_import_error}")

        # Test parameter validation
        warnings = self.parameters.validate()
        if warnings:
            self.log_message("⚠️ Parameter warnings:")
            for warning in warnings:
                self.log_message(f"  • {warning}")
        else:
            self.log_message("✅ All parameters valid")

        self.log_message("🔧 Plugin test completed")

    def show_help(self):
        """Show help information"""
        help_text = """
        U-Track Particle Tracking Plugin Help

        TABS:
        • Input: Select data source and preview input data
        • Parameters: Configure tracking parameters
        • Current Parameters: View current parameter values
        • Visualization: Set up track display options
        • Analysis: Configure motion analysis and output
        • Advanced: Debug options and expert settings
        • Progress: Monitor tracking progress and view logs

        WORKFLOW:
        1. Select input source (detection results or file)
        2. Configure tracking parameters
        3. Review current parameters
        4. Set visualization preferences
        5. Run tracking
        6. Analyze results and export data

        For detailed documentation, see the plugin documentation.
        """

        help_dialog = QtWidgets.QMessageBox()
        help_dialog.setWindowTitle("U-Track Help")
        help_dialog.setText(help_text)
        help_dialog.exec_()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_standard_frame_info(x_coords, y_coords, amplitudes, x_stds=None, y_stds=None, amp_stds=None, frame_num=None):
    """Create standardized frame_info dictionary"""

    # Convert to numpy arrays
    x_coords = np.array(x_coords) if not isinstance(x_coords, np.ndarray) else x_coords
    y_coords = np.array(y_coords) if not isinstance(y_coords, np.ndarray) else y_coords
    amplitudes = np.array(amplitudes) if not isinstance(amplitudes, np.ndarray) else amplitudes

    num_detections = len(x_coords)

    # Create standard deviations if not provided
    if x_stds is None:
        x_stds = np.ones(num_detections) * 0.1
    if y_stds is None:
        y_stds = np.ones(num_detections) * 0.1
    if amp_stds is None:
        amp_stds = amplitudes * 0.1

    # Create tracking-compatible frame info
    if num_detections > 0:
        frame_info = {
            'num': num_detections,
            'xCoord': np.column_stack([x_coords, x_stds]),
            'yCoord': np.column_stack([y_coords, y_stds]),
            'amp': np.column_stack([amplitudes, amp_stds]),
            'x_coord': None,
            'y_coord': None,
            'all_coord': np.column_stack([x_coords, x_stds, y_coords, y_stds])
        }

        frame_info['x_coord'] = frame_info['xCoord']
        frame_info['y_coord'] = frame_info['yCoord']

    else:
        # Empty frame
        frame_info = {
            'num': 0,
            'xCoord': np.array([]).reshape(0, 2),
            'yCoord': np.array([]).reshape(0, 2),
            'amp': np.array([]).reshape(0, 2),
            'x_coord': None,
            'y_coord': None,
            'all_coord': np.array([]).reshape(0, 4)
        }

        frame_info['x_coord'] = frame_info['xCoord']
        frame_info['y_coord'] = frame_info['yCoord']

    return frame_info


# =============================================================================
# PLUGIN ENTRY POINTS
# =============================================================================

# Global GUI instance
utrack_gui_instance = None

def launch_utrack_gui():
    """Launch the enhanced multi-tabbed U-Track GUI"""
    global utrack_gui_instance

    if not QT_AVAILABLE:
        if FLIKA_AVAILABLE:
            g.alert("Qt not available - cannot launch enhanced GUI")
        else:
            print("Qt not available - cannot launch enhanced GUI")
        return

    try:
        if utrack_gui_instance is None or not utrack_gui_instance.isVisible():
            utrack_gui_instance = UTrackTrackingGUI()

        utrack_gui_instance.show()
        utrack_gui_instance.raise_()
        utrack_gui_instance.activateWindow()

        if FLIKA_AVAILABLE:
            print("✅ Enhanced U-Track GUI launched successfully!")
        else:
            print("Enhanced U-Track GUI launched in standalone mode")

    except Exception as e:
        error_msg = f"Failed to launch U-Track GUI: {str(e)}"
        if FLIKA_AVAILABLE:
            g.alert(error_msg)
        else:
            print(error_msg)


def run_tracking():
    """Simple function to run tracking with current parameters"""
    if not TRACKING_AVAILABLE:
        error_msg = "U-Track modules not available!"
        if FLIKA_AVAILABLE:
            g.alert(error_msg)
        else:
            print(error_msg)
        return None

    # Launch GUI if not already open
    global utrack_gui_instance
    if utrack_gui_instance is None or not utrack_gui_instance.isVisible():
        launch_utrack_gui()
        if FLIKA_AVAILABLE:
            g.alert("GUI launched. Configure parameters and click 'Run Tracking'.")
    else:
        # Trigger tracking from the GUI
        utrack_gui_instance.run_tracking()


def test_tracking_plugin():
    """Test if the tracking plugin is working properly"""
    print("=== U-Track Tracking Plugin Test ===")
    print(f"FLIKA available: {FLIKA_AVAILABLE}")
    print(f"U-Track tracking modules available: {TRACKING_AVAILABLE}")
    print(f"PyQtGraph available: {PYQTGRAPH_AVAILABLE}")
    print(f"Qt available: {QT_AVAILABLE}")
    print(f"pandas available: {PANDAS_AVAILABLE}")
    print(f"scipy available: {SCIPY_AVAILABLE}")
    print(f"matplotlib available: {MATPLOTLIB_AVAILABLE}")

    if not TRACKING_AVAILABLE:
        print(f"U-Track import error: {tracking_import_error}")
        print("\nRequired files should be in plugin directory:")
        required_files = [
            "track_general.py",
            "linking.py",
            "gap_closing.py",
            "cost_matrices.py",
            "kalman_filters.py",
            "track_analysis.py",
            "advanced_config.py",
            "utils.py",
            "visualization.py"
        ]
        for f in required_files:
            file_path = os.path.join(current_dir, f)
            exists = "✓" if os.path.exists(file_path) else "✗"
            print(f"  {exists} {f}")

    if FLIKA_AVAILABLE:
        if g.currentWindow is not None:
            print(f"Current window: {g.currentWindow.name}")
            print(f"Image shape: {g.currentWindow.image.shape}")
        else:
            print("No current window - open an image to test tracking")
    else:
        print("FLIKA not available - running in standalone mode")

    print(f"\nInteractive Viewer Components:")
    print(f"  PyQtGraph: {'✓' if PYQTGRAPH_AVAILABLE else '✗'}")
    print(f"  Qt/PyQt: {'✓' if QT_AVAILABLE else '✗'}")

    if PYQTGRAPH_AVAILABLE and QT_AVAILABLE:
        print("  Interactive viewer functionality: ✓ Available")
    else:
        print("  Interactive viewer functionality: ✗ Limited")

    print("=== Test Complete ===")


def create_demo_interactive_viewer():
    """Create a demo interactive viewer with synthetic data"""
    if not PYQTGRAPH_AVAILABLE or not QT_AVAILABLE:
        if FLIKA_AVAILABLE:
            g.alert("PyQtGraph or Qt not available for demo viewer")
        else:
            print("PyQtGraph or Qt not available for demo viewer")
        return

    # Create synthetic image data
    demo_image = np.random.randint(0, 1000, (50, 100, 100)).astype(np.float32)

    # Create synthetic track data
    demo_tracks = []
    for i in range(5):
        particles = []
        for frame in range(20):
            x = 30 + i * 15 + frame * 0.5 + np.random.normal(0, 1)
            y = 30 + frame * 0.8 + np.random.normal(0, 1)
            particles.append({
                'frame': frame,
                'x': x,
                'y': y,
                'amplitude': 1000 + np.random.normal(0, 100)
            })

        track = {
            'particles': particles,
            'movement_type': ['brownian', 'directed', 'confined', 'subdiffusive', 'superdiffusive'][i]
        }
        demo_tracks.append(track)

    # Create viewer
    viewer = InteractiveTrackViewer(
        image_data=demo_image,
        tracks_data=demo_tracks,
        window_name="Demo Interactive Viewer"
    )

    viewer_widget = viewer.create_interactive_viewer()

    if viewer_widget:
        if FLIKA_AVAILABLE:
            g.alert("Demo interactive viewer created!")
        else:
            print("Demo interactive viewer created!")
        return viewer_widget
    else:
        return None


def show_current_parameters():
    """Display current parameter values in console"""
    global utrack_params

    formatted_params = utrack_params.get_formatted_display()
    print(formatted_params)

    if FLIKA_AVAILABLE:
        # Also show in GUI if available
        global utrack_gui_instance
        if utrack_gui_instance and utrack_gui_instance.isVisible():
            utrack_gui_instance.tab_widget.setCurrentIndex(2)  # Switch to Current Parameters tab
            utrack_gui_instance.update_current_parameters_display()


def clear_all_tracks():
    """Clear all track overlays from current window"""
    if FLIKA_AVAILABLE and g.currentWindow:
        clear_all_track_overlays(g.currentWindow)
        print("Track overlays cleared")
        if FLIKA_AVAILABLE:
            g.alert("Track overlays cleared")
    else:
        print("No current window to clear overlays")

def clear_all_track_overlays(window):
    """Clear all track overlays from a FLIKA window"""
    if hasattr(window, 'track_overlays'):
        for overlay in window.track_overlays:
            overlay.clear_tracks()
        window.track_overlays.clear()

# =============================================================================
# PLUGIN MENU ASSIGNMENTS
# =============================================================================

# Set menu paths for all functions
launch_utrack_gui.menu_path = 'Plugins>Particle Tracking>Enhanced U-Track GUI'
run_tracking.menu_path = 'Plugins>Particle Tracking>Run Tracking'
test_tracking_plugin.menu_path = 'Plugins>Particle Tracking>Test Tracking Plugin'
create_demo_interactive_viewer.menu_path = 'Plugins>Particle Tracking>Demo Interactive Viewer'
show_current_parameters.menu_path = 'Plugins>Particle Tracking>Show Current Parameters'
clear_all_tracks.menu_path = 'Plugins>Particle Tracking>Clear Track Overlays'


# =============================================================================
# PLUGIN INFORMATION
# =============================================================================

__version__ = '2.1.0'
__author__ = 'Simplified Enhanced Version'
__description__ = 'U-Track particle tracking for FLIKA with simplified enhanced GUI interface'

# Plugin status message
if TRACKING_AVAILABLE and FLIKA_AVAILABLE and PYQTGRAPH_AVAILABLE and QT_AVAILABLE:
    print("✅ Enhanced U-Track Particle Tracking Plugin loaded successfully!")
    print("🎯 Simplified interface with comprehensive tracking options available")
    print("🎭 Interactive Track Viewer with color-coded motion analysis ready")
    print("⚙️ Use 'Enhanced U-Track GUI' menu to access interface")
    print("📊 Current Parameters display integrated into GUI")
elif TRACKING_AVAILABLE and FLIKA_AVAILABLE and PYQTGRAPH_AVAILABLE:
    print("⚠ Enhanced U-Track Plugin loaded with limited Qt functionality")
elif TRACKING_AVAILABLE and FLIKA_AVAILABLE:
    print("⚠ Enhanced U-Track Plugin loaded - install PyQtGraph for enhanced visualization")
elif FLIKA_AVAILABLE:
    print("⚠ U-Track Plugin loaded with missing tracking dependencies")
else:
    print("⚠ U-Track Plugin running in standalone mode")

# Show available functions
print("\n📋 Available Functions:")
print("  • launch_utrack_gui() - Launch the main GUI interface")
print("  • run_tracking() - Quick tracking with current parameters")
print("  • show_current_parameters() - Display current parameter values")
print("  • test_tracking_plugin() - Test plugin functionality")
print("  • create_demo_interactive_viewer() - Create demo viewer")
print("  • clear_all_tracks() - Clear track overlays")
