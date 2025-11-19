#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:37:46 2023

@author: george
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import cv2
from matplotlib import pyplot as plt

def update_plot():
    # Generate random scatter plot data
    num_points = 100
    x = np.random.normal(size=num_points)
    y = np.random.normal(size=num_points)
    size = np.random.randint(10, 100, size=num_points)
    color = np.random.randint(0, 255, size=(num_points, 3))

    # Clear previous plot data
    scatter_plot.setData(x=[], y=[])

    # Update scatter plot
    scatter_plot.setData(x=x, y=y, size=size, brush=color)


# Create application and window
app = QtGui.QApplication([])
win = QtWidgets.QMainWindow()

# Create ImageView widget
image_view = pg.ImageView(view=pg.PlotItem())
win.setCentralWidget(image_view)
win.show()

# Create ScatterPlotItem
scatter_plot = pg.ScatterPlotItem()
image_view.getView().addItem(scatter_plot)

# Set up the image view parameters
image_view.view.invertY(False)
image_view.ui.histogram.hide()
image_view.ui.roiBtn.hide()
image_view.ui.menuBtn.hide()

# Set up the plot parameters
scatter_plot.setSymbol("o")
scatter_plot.setSize(10)

# Create a movie writer
frame_rate = 1  # Number of frames per second
duration = 50  # Duration of the movie in seconds
total_frames = frame_rate * duration
frame_counter = 0

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
output_file = r"/Users/george/Desktop/videoTest/scatter_movie.mp4"
frame_size = (233, 222)  # Size of the frames
#video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

# Start the update timer
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(1000 / frame_rate)  # Interval in milliseconds


def QImageToCvMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr

image_list = []

while frame_counter < total_frames:
    QtGui.QApplication.processEvents()

    # Render the plot to an image
    win.repaint()
    screenshot = win.grab()
    image = screenshot.toImage()
    frame = QImageToCvMat(image)
    image_list.append(frame)
    # Write the frame to the video file
    #video_writer.write(frame)

    frame_counter += 1

# Cleanup
timer.stop()
#video_writer.release()
app.quit()
