#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:23:24 2024

@author: george
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QDockWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
from PyQt5.QtCore import Qt
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import blob_log
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
import tifffile
from scipy.ndimage import rotate
from typing import Optional, List, Tuple, Dict, Any

import logging

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_sphere(center, radius, size):
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    return ((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2) <= radius**2

def create_cube(center, side_length, size):
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    return ((np.abs(x - center[0]) <= side_length/2) &
            (np.abs(y - center[1]) <= side_length/2) &
            (np.abs(z - center[2]) <= side_length/2))

def create_pyramid(center, base_length, height, size):
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    base = ((np.abs(x - center[0]) <= base_length/2) &
            (np.abs(y - center[1]) <= base_length/2) &
            (z <= center[2]))
    slope = (base_length/2) / height
    pyramid = base & (np.abs(x - center[0]) <= (center[2] + height - z) * slope) & \
                     (np.abs(y - center[1]) <= (center[2] + height - z) * slope) & \
                     (z >= center[2]) & (z <= center[2] + height)
    return pyramid

def create_persistent_shape(shape_type: str, size: Tuple[int, int, int], shape_size: float) -> np.ndarray:
    """
    Create a single volumetric shape centered at (0, 0, 0) that persists throughout the time series.

    :param shape_type: Type of shape ('sphere', 'cube', or 'pyramid')
    :param size: Size of the volume (depth, height, width)
    :param shape_size: Size of the shape
    :return: 3D numpy array containing the shape
    """
    depth, height, width = size
    center = (depth // 2, height // 2, width // 2)

    if shape_type == 'sphere':
        return create_sphere(center, shape_size / 2, size)
    elif shape_type == 'cube':
        return create_cube(center, shape_size, size)
    elif shape_type == 'pyramid':
        return create_pyramid(center, shape_size, shape_size, size)
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def generate_lightsheet_test_data(size=(64, 64, 64), num_timepoints=10, num_blobs=5, num_channels=1,
                                  size_range=(3, 8), intensity_range=(0.5, 1.0), angle=45, num_slices=30,
                                  use_persistent_shape=False, persistent_shape_type='cube', persistent_shape_size=10):
    data = np.zeros((num_channels, num_timepoints * num_slices, size[1], size[2]))
    volume = np.zeros(size)

    # Create persistent shape if enabled
    if use_persistent_shape:
        persistent_shape = create_persistent_shape(persistent_shape_type, size, persistent_shape_size)
    else:
        persistent_shape = np.zeros(size)

    for c in range(num_channels):
        for t in range(num_timepoints):
            # Start with the persistent shape if enabled, otherwise start with zeros
            if use_persistent_shape:
                volume = persistent_shape.copy() * np.random.uniform(*intensity_range)
            else:
                volume = np.zeros(size)

            # Add random blobs
            for _ in range(num_blobs):
                center = np.random.randint(0, min(size), 3)
                intensity = np.random.uniform(*intensity_range)
                blob_size = np.random.uniform(*size_range)

                # Randomly choose a shape
                shape_type = np.random.choice(['sphere', 'cube', 'pyramid'])

                if shape_type == 'sphere':
                    blob = create_sphere(center, blob_size/2, size)
                elif shape_type == 'cube':
                    blob = create_cube(center, blob_size, size)
                else:  # pyramid
                    blob = create_pyramid(center, blob_size, blob_size, size)

                volume += intensity * blob.astype(float)

            # # Add some Gaussian noise
            # volume += np.random.normal(0, 0.05, size)
            # volume = np.clip(volume, 0, 1)  # Ensure values are between 0 and 1

            # Rotate the volume
            rotated_volume = rotate(volume, angle, axes=(0, 2), reshape=False, mode='constant', cval=0)

            # Extract slices
            for i in range(num_slices):
                slice_pos = int(i * size[0] / num_slices)
                data_index = t * num_slices + i
                data[c, data_index] = rotated_volume[slice_pos]

    return data


def detect_blobs(image, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.1):
    # Check if the image is 2D or 3D
    if image.ndim == 2:
        # If 2D, use it directly
        image_max = image
    elif image.ndim == 3:
        # If 3D, use maximum intensity projection across all slices
        image_max = np.max(image, axis=0)
    else:
        raise ValueError(f"Unexpected image dimensions: {image.ndim}")

    # Detect blobs
    blobs = blob_log(image_max, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold)

    print(f"Detected {len(blobs)} blobs")

    if len(blobs) == 0:
        return np.empty((0, 3)), np.array([]), np.array([])

    # Extract coordinates and sizes
    coordinates = blobs[:, :2][:, ::-1]  # Reverse to get (x, y)
    sizes = blobs[:, 2] if blobs.shape[1] > 2 else np.array([])

    # Calculate intensities
    intensities = image_max[tuple(coordinates.T.astype(int))]

    # Add z-coordinate (set to middle slice if 3D, or 0 if 2D)
    z_coords = np.full(len(coordinates), image.shape[0] // 2 if image.ndim == 3 else 0)
    coordinates = np.column_stack((coordinates, z_coords))

    return coordinates, sizes, intensities

def calculate_blob_distances(blobs_ch1, blobs_ch2):
    coords_ch1, _, _ = blobs_ch1
    coords_ch2, _, _ = blobs_ch2

    # Calculate all pairwise distances
    all_distances = cdist(coords_ch1, coords_ch2).flatten()

    # Calculate nearest neighbor distances
    nn_distances_ch1 = np.min(cdist(coords_ch1, coords_ch2), axis=1)
    nn_distances_ch2 = np.min(cdist(coords_ch2, coords_ch1), axis=1)
    nn_distances = np.concatenate([nn_distances_ch1, nn_distances_ch2])

    return all_distances, nn_distances

def plot_3d_frame(data, blobs, time_point, x_slice, y_slice, z_slice, angle, num_slices=30):
    print(f"Data shape: {data.shape}")
    print(f"Number of channels: {data.shape[0]}")
    print(f"Number of slices: {data.shape[1]}")
    fig = plt.figure(figsize=(15, 7))

    # Plot the volume
    ax1 = fig.add_subplot(121, projection='3d')

    # Check if data is 3D or 4D and adjust accordingly
    if data.ndim == 3:
        num_channels = 1
        data_to_plot = data[np.newaxis, :]  # Add channel dimension
    elif data.ndim == 4:
        num_channels, _, _, _ = data.shape
        data_to_plot = data
    else:
        raise ValueError(f"Data should be 3D or 4D, but has {data.ndim} dimensions")

    _, total_slices, height, width = data_to_plot.shape
    start_slice = time_point * num_slices
    end_slice = min(start_slice + num_slices, total_slices)

    colormaps = [cm.Reds, cm.Greens]  # Use 'Reds' for channel 1, 'Greens' for channel 2
    for c in range(num_channels):
        try:
            # Only plot voxels above a certain threshold to reduce clutter
            data_slice = data_to_plot[c, start_slice:end_slice]
            print(f"Channel {c} data slice shape: {data_slice.shape}")
            print(f"Channel {c} data slice min: {data_slice.min()}, max: {data_slice.max()}")

            if data_slice.size > 0:
                threshold = np.max(data_slice) * 0.1
                mask = data_slice > threshold

                z, y, x = np.where(mask)
                print(f"Channel {c}: {len(x)} voxels above threshold")

                if len(x) > 0:
                    # Rotate the coordinates
                    angle_rad = np.radians(angle)
                    x_rot = x
                    z_rot = z * np.cos(angle_rad) - y * np.sin(angle_rad)
                    y_rot = z * np.sin(angle_rad) + y * np.cos(angle_rad)

                    c_values = data_slice[mask]

                    scatter = ax1.scatter(x_rot, y_rot, z_rot,
                                          c=c_values, cmap=colormaps[c],
                                          alpha=0.1, s=1)
                    fig.colorbar(scatter, ax=ax1, label=f'Channel {c+1} Intensity')
                else:
                    print(f"No voxels above threshold for channel {c}")
            else:
                print(f"Empty data slice for channel {c}")
        except Exception as e:
            print(f"Error plotting channel {c}: {str(e)}")

    # Plot slice planes
    ax1.plot([x_slice, x_slice], [0, height], [0, 0], 'b-', alpha=0.5)
    ax1.plot([x_slice, x_slice], [0, 0], [0, num_slices], 'b-', alpha=0.5)
    ax1.plot([x_slice, x_slice], [height, height], [0, num_slices], 'b-', alpha=0.5)
    ax1.plot([0, width], [y_slice, y_slice], [0, 0], 'g-', alpha=0.5)
    ax1.plot([0, 0], [y_slice, y_slice], [0, num_slices], 'g-', alpha=0.5)
    ax1.plot([width, width], [y_slice, y_slice], [0, num_slices], 'g-', alpha=0.5)
    ax1.plot([0, width], [0, height], [z_slice, z_slice], 'r-', alpha=0.5)

    ax1.set_title(f'Volume at time point {time_point}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot the detected blobs
    ax2 = fig.add_subplot(122, projection='3d')
    blob_colormaps = [plt.cm.viridis, plt.cm.plasma]  # Use more contrasting colormaps
    angle_rad = np.radians(angle)  # Define angle_rad here
    plotted_artists = []
    for c in range(len(blobs)):
        try:
            coords, sizes, intensities = blobs[c]
            print(f"Channel {c}: Plotting {len(coords)} blobs")
            if len(coords) > 0:
                # Rotate blob coordinates
                x_blob = coords[:, 0]
                y_blob = coords[:, 1] * np.cos(angle_rad) - coords[:, 2] * np.sin(angle_rad)
                z_blob = coords[:, 1] * np.sin(angle_rad) + coords[:, 2] * np.cos(angle_rad)

                scatter = ax2.scatter(x_blob, y_blob, z_blob,
                                      c=intensities, s=sizes*100, cmap=blob_colormaps[c],
                                      alpha=0.7, label=f'Channel {c+1}')
                fig.colorbar(scatter, ax=ax2, label=f'Channel {c+1} Intensity')
                plotted_artists.append(scatter)
            else:
                print(f"No blobs to plot for channel {c}")
        except Exception as e:
            print(f"Error plotting blobs for channel {c}: {str(e)}")

    # Set axis limits explicitly
    ax2.set_xlim(0, width)
    ax2.set_ylim(0, height)
    ax2.set_zlim(0, num_slices)
    ax2.set_title(f'Detected blobs at time point {time_point}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Only create legend if there are plotted artists
    if plotted_artists:
        ax2.legend()

    # Adjust the view angle for better visibility
    ax2.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

def plot_distance_histograms(all_distances, nn_distances):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot histogram of all distances
    ax1.hist(all_distances, bins=50, edgecolor='black')
    ax1.set_title('Histogram of All Distances between Red and Green Blobs')
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Frequency')

    # Plot histogram of nearest neighbor distances
    ax2.hist(nn_distances, bins=50, edgecolor='black')
    ax2.set_title('Histogram of Nearest Neighbor Distances between Red and Green Blobs')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

class Viewer(QMainWindow):
    def __init__(self, data, angle):
        super().__init__()
        self.data = data
        self.angle = angle
        self.num_channels, total_slices, self.height, self.width = data.shape
        self.num_timepoints = total_slices // 30  # Assuming 30 slices per timepoint
        self.num_slices = 30
        self.current_time = 0
        self.current_slice = 0
        self.current_x = self.width // 2
        self.current_y = self.height // 2
        self.blobs = [detect_blobs(self.data[c, :self.num_slices]) for c in range(self.num_channels)]
        self.initUI()

    def initUI(self):
        self.setWindowTitle('4D Lightsheet Microscopy Viewer')
        self.setGeometry(100, 100, 1200, 800)

        # Create docks
        self.create_3d_dock()
        self.create_xy_dock()
        self.create_xz_dock()
        self.create_yz_dock()
        self.create_controls_dock()

        # Initialize views
        self.update_view()

    def create_3d_dock(self):
        dock = QDockWidget("3D View", self)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.gl_widget = gl.GLViewWidget()
        layout.addWidget(self.gl_widget)
        dock.setWidget(widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def create_xy_dock(self):
        dock = QDockWidget("XY Slice", self)
        self.imv_xy = pg.ImageView()
        dock.setWidget(self.imv_xy)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def create_xz_dock(self):
        dock = QDockWidget("XZ Slice", self)
        self.imv_xz = pg.ImageView()
        dock.setWidget(self.imv_xz)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def create_yz_dock(self):
        dock = QDockWidget("YZ Slice", self)
        self.imv_yz = pg.ImageView()
        dock.setWidget(self.imv_yz)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def create_controls_dock(self):
        dock = QDockWidget("Controls", self)
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Time slider
        layout.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.num_timepoints - 1)
        self.time_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.time_slider)

        # Slice slider
        layout.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(self.num_slices - 1)
        self.slice_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.slice_slider)

        # X and Y position sliders
        layout.addWidget(QLabel("X Position:"))
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(self.width - 1)
        self.x_slider.setValue(self.width // 2)
        self.x_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.x_slider)

        layout.addWidget(QLabel("Y Position:"))
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.setMaximum(self.height - 1)
        self.y_slider.setValue(self.height // 2)
        self.y_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.y_slider)

        # Buttons
        self.plot_button = QPushButton('Show Matplotlib 3D Plot')
        self.plot_button.clicked.connect(self.show_3d_plot)
        layout.addWidget(self.plot_button)

        self.histogram_button = QPushButton('Show Distance Histograms')
        self.histogram_button.clicked.connect(self.show_distance_histograms)
        layout.addWidget(self.histogram_button)

        self.reset_view_button = QPushButton('Reset 3D View')
        self.reset_view_button.clicked.connect(self.reset_3d_view)
        layout.addWidget(self.reset_view_button)

        dock.setWidget(widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def update_view(self):
        self.current_time = self.time_slider.value()
        self.current_slice = self.slice_slider.value()
        self.current_x = self.x_slider.value()
        self.current_y = self.y_slider.value()

        start_slice = self.current_time * self.num_slices
        end_slice = start_slice + self.num_slices
        current_data = self.data[:, start_slice:end_slice]

        self.blobs = [detect_blobs(current_data[c]) for c in range(current_data.shape[0])]

        self.update_3d_scatter()
        self.update_orthogonal_views()

    def update_3d_scatter(self):
        self.gl_widget.clear()
        colors = [(1, 0, 0, 1), (0, 1, 0, 1)]  # Red and Green
        all_coords = []

        print(f"Number of channels: {len(self.blobs)}")
        for c, blob_data in enumerate(self.blobs):
            coords, sizes, intensities = blob_data
            print(f"Channel {c}: {len(coords)} blobs detected")
            print(f"Coordinates: {coords}")
            print(f"Sizes: {sizes}")
            print(f"Intensities: {intensities}")
            if len(coords) > 0:
                all_coords.append(coords)
                size = sizes * 10  # Increased size multiplier
                color = np.tile(colors[c], (len(coords), 1))
                color[:, 3] = intensities / intensities.max()  # Alpha channel based on intensity
                scatter = gl.GLScatterPlotItem(pos=coords, size=size, color=color, pxMode=True)
                self.gl_widget.addItem(scatter)

        if all_coords:
            all_coords = np.vstack(all_coords)
            center = all_coords.mean(axis=0)
            max_range = (all_coords.max(axis=0) - all_coords.min(axis=0)).max()

            print(f"All coordinates: {all_coords}")
            print(f"Center: {center}")
            print(f"Max range: {max_range}")

            # Add the check for max_range here
            if max_range > 0:
                scale = 2.0 / max_range
            else:
                scale = 1.0

            # Set the view
            self.gl_widget.opts['center'] = pg.Vector(center)
            self.gl_widget.opts['distance'] = max_range * 2
            self.gl_widget.opts['fov'] = 60
            self.gl_widget.setCameraPosition(distance=max_range * 2, elevation=30, azimuth=45)

        else:
            print("No blobs detected in any channel")

        # Rest of the method remains the same...

        # Add coordinate axes
        axis_length = max(self.data.shape[2:])
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_length, 0, 0]]), color=(1, 0, 0, 1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_length, 0]]), color=(0, 1, 0, 1), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_length]]), color=(0, 0, 1, 1), width=2)
        self.gl_widget.addItem(x_axis)
        self.gl_widget.addItem(y_axis)
        self.gl_widget.addItem(z_axis)

        self.gl_widget.update()

    def reset_3d_view(self):
        self.update_3d_scatter()

    def update_orthogonal_views(self):
        current_index = self.current_time * self.num_slices + self.current_slice

        if self.num_channels == 1:
            self.imv_xy.setImage(self.data[0, current_index])
            self.imv_xz.setImage(self.data[0, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, :, self.current_y])
            self.imv_yz.setImage(self.data[0, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, self.current_x, :])
        else:
            combined_xy = np.zeros((self.height, self.width, 3))
            combined_xy[:, :, 0] = self.data[0, current_index]
            combined_xy[:, :, 1] = self.data[1, current_index]
            self.imv_xy.setImage(combined_xy)

            combined_xz = np.zeros((self.num_slices, self.width, 3))
            combined_xz[:, :, 0] = self.data[0, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, :, self.current_y]
            combined_xz[:, :, 1] = self.data[1, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, :, self.current_y]
            self.imv_xz.setImage(combined_xz)

            combined_yz = np.zeros((self.num_slices, self.height, 3))
            combined_yz[:, :, 0] = self.data[0, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, self.current_x, :]
            combined_yz[:, :, 1] = self.data[1, self.current_time*self.num_slices:(self.current_time+1)*self.num_slices, self.current_x, :]
            self.imv_yz.setImage(combined_yz)

    def show_3d_plot(self):
        start_slice = self.current_time * self.num_slices
        end_slice = start_slice + self.num_slices
        current_data = self.data[:, start_slice:end_slice]

        # Detect blobs for the current time point
        current_blobs = [detect_blobs(current_data[c]) for c in range(current_data.shape[0])]

        plot_3d_frame(current_data, current_blobs, self.current_time, self.width // 2, self.height // 2, self.current_slice, self.angle)

    def show_distance_histograms(self):
        all_distances, nn_distances = calculate_blob_distances(self.blobs[0], self.blobs[1])
        plot_distance_histograms(all_distances, nn_distances)

if __name__ == '__main__':
    # Generate test data
    size = (64, 64, 64)
    num_timepoints = 10
    num_channels = 2
    angle = 45  # angle in degrees
    num_slices = 30

    data = generate_lightsheet_test_data(size=size,
                                     num_timepoints = num_timepoints,
                                     num_channels = num_channels,
                                     angle = angle,
                                     num_slices= num_slices,
                                     persistent_shape_type='sphere',
                                     persistent_shape_size=30,
                                     num_blobs=2, intensity_range=(0.3, 1.0))

    print(f"Generated data shape: {data.shape}")

    # Save the array as a TIFF file
    data_out = data.reshape(-1, size[1], size[2])
    tifffile.imwrite('/Users/george/Desktop/lightsheet_test_shapes.tif', data_out)

    print(f"Saved tiff to: {'/Users/george/Desktop/lightsheet_test_shapes.tif'}")


    # Create and show the viewer
    app = QApplication([])
    viewer = Viewer(data,angle)
    viewer.show()

    # Plot the first frame to verify data
    blobs = []
    for c in range(data.shape[0]):
        channel_data = data[c]
        if channel_data.ndim == 3:
            # If 3D, pass the entire channel data
            blob_data = detect_blobs(channel_data, threshold=0.05)
        elif channel_data.ndim == 2:
            # If 2D, pass it directly
            blob_data = detect_blobs(channel_data, threshold=0.05)
        else:
            raise ValueError(f"Unexpected channel data dimensions: {channel_data.ndim}")
        blobs.append(blob_data)
        try:
            print("Blobs to be plotted:")
            for c, blob_data in enumerate(blobs):
                coords, sizes, intensities = blob_data
                print(f"Channel {c}: {len(coords)} blobs")
                print(f"Coordinates: {coords}")
                print(f"Sizes: {sizes}")
                print(f"Intensities: {intensities}")

            plot_3d_frame(data, blobs, 0, size[1]//2, size[2]//2, num_slices//2, angle)
        except Exception as e:
            logger.error(f"Error calling plot_3d_frame: {str(e)}")

    # Calculate distances between blobs
    all_distances, nn_distances = calculate_blob_distances(blobs[0], blobs[1])

    # Plot distance histograms
    plot_distance_histograms(all_distances, nn_distances)

    app.exec_()
