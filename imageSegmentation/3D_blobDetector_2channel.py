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

#With Numba
#import numba

# @numba.jit(nopython=True)
# def generate_test_data(size=(64, 64, 64), num_timepoints=10, num_blobs=5, num_channels=1,
#                        size_range=(1, 5), intensity_range=(0.5, 1.0)):
#     data = np.zeros((num_channels, num_timepoints, *size))
#     for c in range(num_channels):
#         for t in range(num_timepoints):
#             for _ in range(num_blobs):
#                 center = np.random.randint(0, size[0], 3)
#                 intensity = np.random.uniform(intensity_range[0], intensity_range[1])
#                 blob_size = np.random.uniform(size_range[0], size_range[1])

#                 for x in range(size[0]):
#                     for y in range(size[1]):
#                         for z in range(size[2]):
#                             distance = ((x - center[0])**2 +
#                                         (y - center[1])**2 +
#                                         (z - center[2])**2)
#                             data[c, t, x, y, z] += intensity * np.exp(-distance / (2 * blob_size**2))

#     return data

#Without Numba
def generate_test_data(size=(64, 64, 64), num_timepoints=10, num_blobs=5, num_channels=1,
                       size_range=(1, 5), intensity_range=(0.5, 1.0)):
    data = np.zeros((num_channels, num_timepoints, *size))
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]

    for c in range(num_channels):
        for t in range(num_timepoints):
            for _ in range(num_blobs):
                center = np.random.randint(0, size[0], 3)
                intensity = np.random.uniform(*intensity_range)
                blob_size = np.random.uniform(*size_range)

                distance = ((x - center[0])**2 +
                            (y - center[1])**2 +
                            (z - center[2])**2)
                data[c, t] += intensity * np.exp(-distance / (2 * blob_size**2))

    return data

def detect_blobs(image, min_sigma=1, max_sigma=10, num_sigma=10, threshold=0.1):
    # Detect blobs
    blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma,
                     num_sigma=num_sigma, threshold=threshold)

    print(f"Detected {len(blobs)} blobs")

    # Extract coordinates, sizes, and intensities
    coordinates = blobs[:, :3]
    sizes = blobs[:, 3]
    intensities = image[coordinates[:, 0].astype(int),
                        coordinates[:, 1].astype(int),
                        coordinates[:, 2].astype(int)]

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


def plot_3d_frame(data, blobs, time_point, x_slice, y_slice, z_slice):
    fig = plt.figure(figsize=(15, 7))

    # Plot the volume
    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = np.meshgrid(np.arange(data.shape[1]),
                          np.arange(data.shape[2]),
                          np.arange(data.shape[3]),
                          indexing='ij')

    colormaps = [cm.Reds, cm.Greens]  # Use 'Reds' for channel 1, 'Greens' for channel 2
    for c in range(data.shape[0]):
        # Only plot voxels above a certain threshold to reduce clutter
        threshold = np.max(data[c]) * 0.1
        mask = data[c] > threshold
        scatter = ax1.scatter(x[mask], y[mask], z[mask],
                              c=data[c][mask], cmap=colormaps[c],
                              alpha=0.1, s=1)
        fig.colorbar(scatter, ax=ax1, label=f'Channel {c+1} Intensity')

    # Plot slice planes
    ax1.plot([x_slice, x_slice], [0, data.shape[2]], [0, 0], 'b-', alpha=0.5)
    ax1.plot([x_slice, x_slice], [0, 0], [0, data.shape[3]], 'b-', alpha=0.5)
    ax1.plot([x_slice, x_slice], [data.shape[2], data.shape[2]], [0, data.shape[3]], 'b-', alpha=0.5)
    ax1.plot([0, data.shape[1]], [y_slice, y_slice], [0, 0], 'g-', alpha=0.5)
    ax1.plot([0, 0], [y_slice, y_slice], [0, data.shape[3]], 'g-', alpha=0.5)
    ax1.plot([data.shape[1], data.shape[1]], [y_slice, y_slice], [0, data.shape[3]], 'g-', alpha=0.5)
    ax1.plot([0, data.shape[1]], [0, data.shape[2]], [z_slice, z_slice], 'r-', alpha=0.5)

    ax1.set_title(f'Volume at time point {time_point}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot the detected blobs
    ax2 = fig.add_subplot(122, projection='3d')
    blob_colormaps = [cm.Reds, cm.Greens]  # Use 'Reds' for channel 1, 'Greens' for channel 2
    for c in range(len(blobs)):
        coords, sizes, intensities = blobs[c]
        if len(coords) > 0:
            scatter = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                                  c=intensities, s=sizes*10, cmap=blob_colormaps[c],
                                  label=f'Channel {c+1}')
            fig.colorbar(scatter, ax=ax2, label=f'Channel {c+1} Intensity')

    ax2.set_xlim(0, data.shape[1])
    ax2.set_ylim(0, data.shape[2])
    ax2.set_zlim(0, data.shape[3])
    ax2.set_title(f'Detected blobs at time point {time_point}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

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
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.num_channels = data.shape[0]
        self.current_z = data.shape[-1] // 2
        self.current_y = data.shape[-2] // 2
        self.current_x = data.shape[-3] // 2
        self.blobs = [detect_blobs(self.data[c, 0]) for c in range(self.num_channels)]
        self.initUI()

    def initUI(self):
        self.setWindowTitle('4D Fluorescence Microscopy Viewer')
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
        self.time_slider.setMaximum(self.data.shape[1] - 1)
        self.time_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.time_slider)

        # X-slice slider
        layout.addWidget(QLabel("X-slice:"))
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(self.data.shape[-3] - 1)
        self.x_slider.setValue(self.current_x)
        self.x_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.x_slider)

        # Y-slice slider
        layout.addWidget(QLabel("Y-slice:"))
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setMinimum(0)
        self.y_slider.setMaximum(self.data.shape[-2] - 1)
        self.y_slider.setValue(self.current_y)
        self.y_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.y_slider)

        # Z-slice slider
        layout.addWidget(QLabel("Z-slice:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(self.data.shape[-1] - 1)
        self.z_slider.setValue(self.current_z)
        self.z_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.z_slider)

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
        t = self.time_slider.value()
        self.current_x = self.x_slider.value()
        self.current_y = self.y_slider.value()
        self.current_z = self.z_slider.value()

        self.blobs = [detect_blobs(self.data[c, t]) for c in range(self.num_channels)]

        # Update 3D scatter plot
        self.update_3d_scatter()

        # Update orthogonal views
        self.update_orthogonal_views(self.data[:, t])


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
            scale = 2.0 / max_range

            print(f"All coordinates: {all_coords}")
            print(f"Center: {center}")
            print(f"Max range: {max_range}")

            # Set the view
            self.gl_widget.opts['center'] = pg.Vector(center)
            self.gl_widget.opts['distance'] = max_range * 2
            self.gl_widget.opts['fov'] = 60
            self.gl_widget.setCameraPosition(distance=max_range * 2, elevation=30, azimuth=45)

        else:
            print("No blobs detected in any channel")

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

    def update_orthogonal_views(self, data):
        if self.num_channels == 1:
            self.imv_xy.setImage(data[0, :, :, self.current_z].T)
            self.imv_xz.setImage(data[0, :, self.current_y, :].T)
            self.imv_yz.setImage(data[0, self.current_x, :, :].T)
        else:
            combined_xy = np.zeros((*data.shape[1:3], 3))
            combined_xy[:, :, 0] = data[0, :, :, self.current_z].T
            combined_xy[:, :, 1] = data[1, :, :, self.current_z].T
            self.imv_xy.setImage(combined_xy)

            combined_xz = np.zeros((*data.shape[1::2], 3))
            combined_xz[:, :, 0] = data[0, :, self.current_y, :].T
            combined_xz[:, :, 1] = data[1, :, self.current_y, :].T
            self.imv_xz.setImage(combined_xz)

            combined_yz = np.zeros((*data.shape[2:], 3))
            combined_yz[:, :, 0] = data[0, self.current_x, :, :].T
            combined_yz[:, :, 1] = data[1, self.current_x, :, :].T
            self.imv_yz.setImage(combined_yz)

    def show_3d_plot(self):
        t = self.time_slider.value()
        plot_3d_frame(self.data[:, t], self.blobs, t, self.current_x, self.current_y, self.current_z)

    def show_distance_histograms(self):
        all_distances, nn_distances = calculate_blob_distances(self.blobs[0], self.blobs[1])
        plot_distance_histograms(all_distances, nn_distances)

if __name__ == '__main__':
    # Generate test data (uncomment the desired option)
    # data = generate_test_data(num_channels=1)  # 1-channel data
    data = generate_test_data(num_channels=2)  # 2-channel data

    #data = np.memmap('large_dataset.npy', dtype='float32', mode='r', shape=(num_channels, num_timepoints, *size))

    # Save the array as a TIFF file
    data_out = data[0]
    data_out = np.reshape(data_out, (640,64,64))
    tifffile.imwrite('/Users/george/Desktop/test.tif', data_out)

    # Create and show the viewer
    app = QtWidgets.QApplication([])
    viewer = Viewer(data)
    viewer.show()

    # Plot the first frame to verify data
    blobs = [detect_blobs(data[c, 0]) for c in range(data.shape[0])]
    plot_3d_frame(data[:, 0], blobs, 0, data.shape[2]//2, data.shape[3]//2, data.shape[4]//2)

    # Calculate distances between blobs
    all_distances, nn_distances = calculate_blob_distances(blobs[0], blobs[1])

    # Plot distance histograms
    plot_distance_histograms(all_distances, nn_distances)

    app.exec_()
