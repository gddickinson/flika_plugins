#!/usr/bin/env python3
"""
Fixed 3D Visualization Engine
=============================

Simplified and more robust version of the 3D rendering engine.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from enum import Enum
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from qtpy.QtCore import QObject, Signal, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QWidget
import warnings
import traceback

# Optional imports for advanced features
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    warnings.warn("scikit-image not available, some visualization features disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available, colormap features limited")


class RenderMode(Enum):
    """Available rendering modes."""
    POINTS = "points"
    SURFACE = "surface"
    VOLUME = "volume"
    WIREFRAME = "wireframe"


class ColorMap(Enum):
    """Available colormaps."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    JET = "jet"
    HOT = "hot"
    GRAYSCALE = "gray"


class VisualizationParameters:
    """Simple and robust visualization parameters."""

    def __init__(self):
        # Basic rendering parameters
        self.render_mode = RenderMode.POINTS
        self.threshold = 0.5
        self.point_size = 2.0
        self.opacity = 1.0

        # Color mapping
        self.colormap = ColorMap.VIRIDIS
        self.color_range = (0.0, 1.0)
        self.invert_colormap = False

        # Display options
        self.show_axes = True
        self.show_grid = False
        self.show_scalebar = True
        self.auto_rotate = False

        # Camera settings
        self.camera_distance = 200.0
        self.camera_elevation = 30.0
        self.camera_azimuth = 45.0

        # Performance settings
        self.max_points = 50000  # Reduced for better performance


class VisualizationEngine(QObject):
    """
    Simplified and robust 3D visualization engine.
    """

    # Signals
    rendering_started = Signal(str)  # render_mode
    rendering_completed = Signal()
    rendering_failed = Signal(str)  # error_message
    parameters_changed = Signal(dict)  # parameters

    def __init__(self, gl_widget: Optional[gl.GLViewWidget] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize parameters first (simplified approach)
        self.current_params = VisualizationParameters()

        # 3D view widget
        self.gl_widget = gl_widget
        self.current_data: Optional[np.ndarray] = None
        self.current_items: List[gl.GLGraphicsItem] = []

        # Animation support
        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self._auto_rotate)
        self.rotation_speed = 1.0

        # Try to setup GL widget
        if self.gl_widget is not None:
            try:
                self._setup_gl_widget()
                self.logger.info("GL widget setup successful")
            except Exception as e:
                self.logger.error(f"GL widget setup failed: {e}")
                self.gl_widget = None

        self.logger.info("VisualizationEngine initialized")

    def _setup_gl_widget(self):
        """Setup the OpenGL widget with default settings."""
        if self.gl_widget is None:
            return

        try:
            # Basic setup
            self.gl_widget.setBackgroundColor('black')

            # Set camera position
            self.gl_widget.setCameraPosition(
                distance=self.current_params.camera_distance,
                elevation=self.current_params.camera_elevation,
                azimuth=self.current_params.camera_azimuth
            )

            # Enable depth testing
            self.gl_widget.opts['depth_test'] = True

        except Exception as e:
            self.logger.error(f"Error setting up GL widget: {e}")
            raise

    def set_data(self, data: np.ndarray, update_display: bool = True):
        """Set volume data for visualization."""
        try:
            if data is None or data.size == 0:
                raise ValueError("Data is empty")

            if data.ndim not in [3, 4]:
                raise ValueError(f"Data must be 3D or 4D, got {data.ndim}D")

            # Store data
            self.current_data = data.astype(np.float32)

            self.logger.info(f"Data set for visualization: shape={data.shape}, dtype={data.dtype}")

            if update_display:
                self.render()

        except Exception as e:
            error_msg = f"Failed to set visualization data: {str(e)}"
            self.logger.error(error_msg)
            self.rendering_failed.emit(error_msg)

    def set_parameters(self, params, update_display: bool = True):
        """Update visualization parameters."""
        try:
            if params is not None:
                self.current_params = params

            # Update auto-rotation
            if self.current_params.auto_rotate:
                self.rotation_timer.start(50)  # 20 FPS
            else:
                self.rotation_timer.stop()

            if update_display and self.current_data is not None:
                self.render()

            # Convert params to dict for signal
            params_dict = self._params_to_dict()
            self.parameters_changed.emit(params_dict)

        except Exception as e:
            error_msg = f"Failed to set visualization parameters: {str(e)}"
            self.logger.error(error_msg)

    def render(self, volume_index: int = 0):
        """Render the current data with current parameters."""
        try:
            if self.current_data is None:
                self.logger.warning("No data to render")
                return

            if self.gl_widget is None:
                self.logger.warning("No GL widget available for rendering")
                return

            # Get render mode
            render_mode = self.current_params.render_mode.value if hasattr(self.current_params.render_mode, 'value') else str(self.current_params.render_mode)

            self.logger.debug(f"Starting render with mode: {render_mode}")
            self.rendering_started.emit(render_mode)

            # Clear existing items
            self.clear()

            # Get data for visualization (3D volume)
            if self.current_data.ndim == 4:
                if volume_index >= self.current_data.shape[1]:
                    volume_index = 0
                data_3d = self.current_data[:, volume_index, :, :]
            else:
                data_3d = self.current_data

            # Apply threshold
            thresholded_data = self._apply_threshold(data_3d)

            self.logger.debug(f"Data shape after threshold: {thresholded_data.shape}, non-zero elements: {np.count_nonzero(thresholded_data)}")

            # Route to appropriate rendering method
            if render_mode.lower() == "points":
                self._render_points(thresholded_data)
            elif render_mode.lower() == "surface" and HAS_SKIMAGE:
                self._render_surface(thresholded_data)
            else:
                # Default to points
                self._render_points(thresholded_data)

            # Add coordinate axes if requested
            if self.current_params.show_axes:
                self._add_coordinate_axes(data_3d.shape)

            self.logger.debug("Rendering completed successfully")
            self.rendering_completed.emit()

        except Exception as e:
            error_msg = f"Rendering failed: {str(e)}"
            self.logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
            self.rendering_failed.emit(error_msg)

    def _apply_threshold(self, data: np.ndarray) -> np.ndarray:
        """Apply threshold to data."""
        if self.current_params.threshold <= 0:
            return data

        # Use percentile-based thresholding for better results
        if self.current_params.threshold < 1.0:
            threshold_value = np.percentile(data, self.current_params.threshold * 100)
        else:
            # Absolute threshold
            threshold_value = self.current_params.threshold

        return np.where(data >= threshold_value, data, 0)

    def _render_points(self, data: np.ndarray):
        """Render data as point cloud."""
        try:
            if self.gl_widget is None:
                return

            # Find points above zero
            coords = np.where(data > 0)
            if len(coords[0]) == 0:
                self.logger.warning("No points to render after thresholding")
                return

            # Limit number of points for performance
            n_points = len(coords[0])
            if n_points > self.current_params.max_points:
                # Randomly sample points
                indices = np.random.choice(n_points, self.current_params.max_points, replace=False)
                coords = tuple(coord[indices] for coord in coords)
                n_points = self.current_params.max_points

            # Create position array (swap Z and Y for proper orientation)
            pos = np.column_stack(coords).astype(np.float32)

            # Center the data
            center = np.mean(pos, axis=0)
            pos -= center

            # Get intensities for coloring
            intensities = data[coords]

            # Create colors
            colors = self._get_colors(intensities)

            self.logger.debug(f"Rendering {n_points} points")

            # Create scatter plot item
            scatter = gl.GLScatterPlotItem(
                pos=pos,
                color=colors,
                size=self.current_params.point_size,
                pxMode=False
            )

            self.gl_widget.addItem(scatter)
            self.current_items.append(scatter)

        except Exception as e:
            self.logger.error(f"Points rendering failed: {str(e)}")
            raise

    def _render_surface(self, data: np.ndarray):
        """Render data as isosurface using marching cubes."""
        if not HAS_SKIMAGE:
            self.logger.warning("scikit-image not available for surface rendering, using points")
            self._render_points(data)
            return

        try:
            # Use a reasonable threshold for surface generation
            threshold_value = np.percentile(data[data > 0], 70) if np.any(data > 0) else 0.5

            if threshold_value <= 0:
                self.logger.warning("No valid threshold for surface generation")
                self._render_points(data)
                return

            # Generate isosurface
            vertices, faces, normals, values = measure.marching_cubes(
                data,
                level=threshold_value,
                step_size=2,  # Reduce resolution for performance
                allow_degenerate=False
            )

            if len(vertices) == 0:
                self.logger.warning("No surface generated, falling back to points")
                self._render_points(data)
                return

            # Center vertices
            center = np.mean(vertices, axis=0)
            vertices -= center

            # Create colors for vertices
            vertex_colors = self._get_vertex_colors(vertices + center, data)

            # Create mesh
            mesh_data = gl.MeshData(vertexes=vertices.astype(np.float32),
                                  faces=faces.astype(np.uint32))
            mesh_data.setVertexColors(vertex_colors)

            mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                drawEdges=False,
                shader='shaded'
            )

            self.gl_widget.addItem(mesh_item)
            self.current_items.append(mesh_item)

            self.logger.debug(f"Surface rendered with {len(vertices)} vertices")

        except Exception as e:
            self.logger.error(f"Surface rendering failed: {str(e)}, falling back to points")
            self._render_points(data)

    def _get_colors(self, values: np.ndarray) -> np.ndarray:
        """Generate colors for values using current colormap."""
        if len(values) == 0:
            return np.array([]).reshape(0, 4)

        # Normalize values to [0, 1]
        v_min, v_max = self.current_params.color_range
        if v_max > v_min:
            normalized = (values - v_min) / (v_max - v_min)
        else:
            # Auto-range
            if np.max(values) > np.min(values):
                normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
            else:
                normalized = np.ones_like(values) * 0.5

        normalized = np.clip(normalized, 0, 1)

        if self.current_params.invert_colormap:
            normalized = 1 - normalized

        # Apply colormap
        return self._apply_colormap(normalized)

    def _apply_colormap(self, normalized_values: np.ndarray) -> np.ndarray:
        """Apply colormap to normalized values."""
        colormap_name = self.current_params.colormap.value if hasattr(self.current_params.colormap, 'value') else str(self.current_params.colormap)

        # Use matplotlib colormaps if available
        if HAS_MATPLOTLIB:
            try:
                cmap = cm.get_cmap(colormap_name)
                colors = cmap(normalized_values)
                colors[:, 3] = self.current_params.opacity  # Set alpha
                return colors.astype(np.float32)
            except Exception as e:
                self.logger.debug(f"Matplotlib colormap failed: {e}")

        # Fallback to built-in colormaps
        return self._builtin_colormap(normalized_values)

    def _builtin_colormap(self, values: np.ndarray) -> np.ndarray:
        """Apply built-in colormap."""
        colors = np.zeros((len(values), 4), dtype=np.float32)
        colors[:, 3] = self.current_params.opacity  # Alpha channel

        # Simple viridis approximation
        colors[:, 0] = 0.267 * values**2 + 0.105 * values + 0.012  # Red
        colors[:, 1] = 0.971 * values - 0.334 * values**2 + 0.022  # Green
        colors[:, 2] = 0.334 + 0.645 * values - 0.203 * values**2  # Blue

        return np.clip(colors, 0, 1)

    def _get_vertex_colors(self, vertices: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Get colors for surface vertices based on data values."""
        vertex_values = np.zeros(len(vertices))

        for i, vertex in enumerate(vertices):
            z, y, x = np.clip(vertex.astype(int), 0, np.array(data.shape) - 1)
            vertex_values[i] = data[z, y, x]

        return self._get_colors(vertex_values)

    def _add_coordinate_axes(self, shape: Tuple[int, int, int]):
        """Add coordinate axes to the visualization."""
        if self.gl_widget is None:
            return

        try:
            axes = gl.GLAxisItem()
            # Scale axes to data size
            max_dim = max(shape)
            axes.setSize(max_dim/4, max_dim/4, max_dim/4)
            self.gl_widget.addItem(axes)
            self.current_items.append(axes)
        except Exception as e:
            self.logger.debug(f"Failed to add axes: {e}")

    def _auto_rotate(self):
        """Auto-rotate the view."""
        if self.gl_widget and hasattr(self.gl_widget, 'orbit'):
            try:
                self.gl_widget.orbit(self.rotation_speed, 0)
            except Exception as e:
                self.logger.debug(f"Auto-rotate failed: {e}")

    def clear(self):
        """Clear all visualization items."""
        if self.gl_widget is None:
            return

        for item in self.current_items:
            try:
                self.gl_widget.removeItem(item)
            except Exception as e:
                self.logger.debug(f"Failed to remove item: {e}")

        self.current_items.clear()

    def reset_view(self):
        """Reset camera to default position."""
        if self.gl_widget is None:
            return

        try:
            self.gl_widget.setCameraPosition(
                distance=self.current_params.camera_distance,
                elevation=self.current_params.camera_elevation,
                azimuth=self.current_params.camera_azimuth
            )
        except Exception as e:
            self.logger.debug(f"Reset view failed: {e}")

    def _params_to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        try:
            return {
                'render_mode': self.current_params.render_mode.value if hasattr(self.current_params.render_mode, 'value') else str(self.current_params.render_mode),
                'threshold': self.current_params.threshold,
                'point_size': self.current_params.point_size,
                'opacity': self.current_params.opacity,
                'colormap': self.current_params.colormap.value if hasattr(self.current_params.colormap, 'value') else str(self.current_params.colormap),
                'show_axes': self.current_params.show_axes,
                'auto_rotate': self.current_params.auto_rotate
            }
        except Exception as e:
            self.logger.error(f"Error converting params to dict: {e}")
            return {}
