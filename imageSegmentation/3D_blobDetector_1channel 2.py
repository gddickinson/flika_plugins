#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:41:47 2024

@author: george
"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate test data
def generate_test_data(size=(64, 64, 64), num_timepoints=10, num_blobs=5):
    data = np.zeros((num_timepoints, *size))
    for t in range(num_timepoints):
        for _ in range(num_blobs):
            x, y, z = np.random.randint(0, size[0], 3)
            intensity = np.random.uniform(0.5, 1.0)
            data[t] += intensity * np.exp(-((np.indices(size) - np.array([x, y, z])[:, None, None, None])**2).sum(0) / 20)
    return data

# 3D blob detector
def detect_blobs(image, threshold=0.3, min_distance=5):
    filtered_image = ndimage.gaussian_filter(image, sigma=1)
    local_max = ndimage.maximum_filter(filtered_image, size=min_distance) == filtered_image
    thresholded = filtered_image > threshold
    blobs = local_max & thresholded
    coordinates = np.column_stack(np.nonzero(blobs))
    return coordinates

# New function to plot a single time frame using matplotlib
def plot_3d_frame(data, blobs, time_point):
    fig = plt.figure(figsize=(12, 5))

    # Plot the volume
    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = np.meshgrid(np.arange(data.shape[0]),
                          np.arange(data.shape[1]),
                          np.arange(data.shape[2]))
    ax1.scatter(x, y, z, c=data, alpha=0.1, s=1)
    ax1.set_title(f'Volume at time point {time_point}')

    # Plot the detected blobs
    ax2 = fig.add_subplot(122, projection='3d')
    if len(blobs) > 0:
        ax2.scatter(blobs[:, 0], blobs[:, 1], blobs[:, 2], c='r', s=50)
    ax2.set_xlim(0, data.shape[0])
    ax2.set_ylim(0, data.shape[1])
    ax2.set_zlim(0, data.shape[2])
    ax2.set_title(f'Detected blobs at time point {time_point}')

    plt.tight_layout()
    plt.show()

# Main application
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.initUI()

    def initUI(self):
        self.setWindowTitle('4D Fluorescence Microscopy Viewer')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)

        # Create ImageView for 2D slice viewing
        self.imv = pg.ImageView()
        layout.addWidget(self.imv)

        # Time slider
        self.time_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.data.shape[0] - 1)
        self.time_slider.valueChanged.connect(self.update_view)
        layout.addWidget(self.time_slider)

        # Button to show 3D plot
        self.plot_button = QtWidgets.QPushButton('Show 3D Plot')
        self.plot_button.clicked.connect(self.show_3d_plot)
        layout.addWidget(self.plot_button)

        # Initialize view
        self.update_view()

    def update_view(self):
        t = self.time_slider.value()
        self.imv.setImage(self.data[t])

    def show_3d_plot(self):
        t = self.time_slider.value()
        blobs = detect_blobs(self.data[t])
        plot_3d_frame(self.data[t], blobs, t)

if __name__ == '__main__':
    # Generate test data
    data = generate_test_data()

    # Create and show the viewer
    app = QtWidgets.QApplication([])
    viewer = Viewer(data)
    viewer.show()

    # Plot the first frame to verify data
    blobs = detect_blobs(data[0])
    plot_3d_frame(data[0], blobs, 0)

    app.exec_()
