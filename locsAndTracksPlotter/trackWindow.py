#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:40:06 2023

@author: george
"""

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os

# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

from .helperFunctions import *

class TrackWindow(BaseProcess):
    def __init__(self, mainGUI):
        super().__init__()

        self.mainGUI = mainGUI

        # Setup window
        self.win = pg.GraphicsLayoutWidget()  # Create a PyqtGraph GraphicsWindow object
        self.win.resize(600, 800)  # Set the size of the window
        self.win.setWindowTitle('Track Display - press "t" to add track')  # Set the title of the window

        # Add widgets to the window
        self.label = pg.LabelItem(justify='center')  # Create a PyqtGraph LabelItem object for displaying text
        self.label_2 = pg.LabelItem(justify='center')

        self.win.addItem(self.label)  # Add the label to the window
        self.win.addItem(self.label_2)

        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying intensity data
        self.plt1 = self.win.addPlot(title='intensity')  # Create a PyqtGraph PlotItem object
        self.plt1.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        # Create a plot for displaying the track
        self.plt3 = self.win.addPlot(title='track')  # Create a PyqtGraph PlotItem object
        self.plt3.setAspectLocked()  # Keep the aspect ratio of the plot fixed
        self.plt3.showGrid(x=True, y=True)  # Show a grid on the plot
        self.plt3.setXRange(-5,5)  # Set the x-axis limits of the plot
        self.plt3.setYRange(-5,5)  # Set the y-axis limits of the plot
        self.plt3.getViewBox().invertY(True)  # Invert the y-axis of the plot

        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying the distance from the origin
        self.plt2 = self.win.addPlot(title='distance from origin')  # Create a PyqtGraph PlotItem object
        self.plt2.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

# =============================================================================
#         # Create a plot for displaying the polar coordinates of the track (direction and velocity)
#         self.plt4 = self.win.addPlot(title='polar (direction and velocity)')  # Create a PyqtGraph PlotItem object
#         self.plt4.getViewBox().invertY(True)  # Invert the y-axis of the plot
#         self.plt4.setAspectLocked()  # Keep the aspect ratio of the plot fixed
#         self.plt4.setXRange(-4,4)  # Set the x-axis limits of the plot
#         self.plt4.setYRange(-4,4)  # Set the y-axis limits of the plot
#         self.plt4.hideAxis('bottom')  # Hide the x-axis of the plot
#         self.plt4.hideAxis('left')  # Hide the y-axis of the plot
# =============================================================================

        # Create a plot for displaying the nearest neighbour counts
        self.plt4 = self.win.addPlot(title='nearest neighbobur count')  # Create a PyqtGraph PlotItem object
        self.plt4.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying the direction relative to the origin of the
        self.plt6 = self.win.addPlot(title='intensity varience')
        self.plt6.getAxis('left').enableAutoSIPrefix(False)

        # Create a plot for displaying the instantaneous velocity of the track
        self.plt5 = self.win.addPlot(title='instantaneous velocity')  # Create a PyqtGraph PlotItem object
        self.plt5.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        self.win.nextRow()

        # Set plot labels for each of the six plots
        self.plt1.setLabel('left', 'Intensity', units ='Arbitary')
        self.plt1.setLabel('bottom', 'Time', units ='Frames')

        self.plt2.setLabel('left', 'Distance', units ='pixels')
        self.plt2.setLabel('bottom', 'Time', units ='Frames')

        self.plt3.setLabel('left', 'y', units ='pixels')
        self.plt3.setLabel('bottom', 'x', units ='pixels')

        self.plt4.setLabel('left', '# of neighbours', units ='count')
        self.plt4.setLabel('bottom', 'Time', units ='Frames')

        self.plt5.setLabel('left', 'velocity', units ='pixels/frame')
        self.plt5.setLabel('bottom', 'Time', units ='Frames')

        self.plt6.setLabel('left', 'rolling average varience', units ='intensity')
        self.plt6.setLabel('bottom', 'Time', units ='Frames')

        self.win.nextRow()

        # Add a button to toggle the position indicator display
        self.optionsPanel = QGraphicsProxyWidget()
        self.positionIndicator_button = QPushButton('Show position info')
        self.positionIndicator_button.pressed.connect(self.togglePoistionIndicator)
        self.optionsPanel.setWidget(self.positionIndicator_button)

        # Create a ComboBox for selecting the nearest neighbour count.
        self.optionsPanel2 = QGraphicsProxyWidget()
        self.plotCountSelector = pg.ComboBox()
        self.countTypes= {'NN radius: 3':'3','NN radius: 5':'5','NN radius: 10':'10','NN radius: 20':'20','NN radius: 30':'30'}
        self.plotCountSelector.setItems(self.countTypes)
        self.countLabel = QLabel("NN count radius")
        self.optionsPanel2.setWidget(self.plotCountSelector)

        #add options panel to win
        self.win.addItem(self.optionsPanel)
        self.win.addItem(self.optionsPanel2)


        #status flags
        self.showPositionIndicators = False
        self.plotsInitiated = False


        self.r = None

    def update(self, time, intensity, distance, zeroed_X, zeroed_Y, dydt, direction, velocity, ID, count_3, count_5, count_10, count_20, count_30, svm, length):

        ##Update track ID
        self.label.setText("<span style='font-size: 16pt'>track ID={}".format(ID))

        ##Update track svm
        self.label_2.setText("<span style='font-size: 16pt'>SVM={} Length={}".format(int(svm), int(length)))

        #update intensity plot
        self.plt1.plot(time, intensity, stepMode=False, brush=(0,0,255,150), clear=True)

        #update distance plot
        self.plt2.plot(time, distance, stepMode=False, brush=(0,0,255,150), clear=True)

        #update position relative to 0 plot
        self.plt3.plot(zeroed_X, zeroed_Y, stepMode=False, brush=(0,0,255,150), clear=True)

# =============================================================================
#         #update polar
#         self.updatePolarPlot(direction,velocity)
# =============================================================================
        #update nearest neighbour count

        if self.plotCountSelector.value() == '3':
            countRadius = count_3
        elif self.plotCountSelector.value() == '5':
            countRadius = count_5
        elif self.plotCountSelector.value() == '10':
            countRadius = count_10
        elif self.plotCountSelector.value() == '20':
            countRadius = count_20
        elif self.plotCountSelector.value() == '30':
            countRadius = count_30


        self.plt4.plot(time, countRadius, stepMode=False, brush=(0,0,255,150), clear=True)

        #update dydt
        self.plt5.plot(time, velocity, stepMode=False, brush=(0,0,255,150), clear=True)

        #update direction
        #self.plt6.plot(time, direction, stepMode=False, brush=(0,0,255,150), clear=True)

        #update roling intensity varience
        rollingTime = rollingFunc(time, func_type='mean')
        rollingVariance = rollingFunc(intensity, func_type='variance')
        self.plt6.plot(rollingTime, rollingVariance, stepMode=False, brush=(0,0,255,150), clear=True)

        # if self.autoscaleX:
        #     self.plt1.setXRange(np.min(x),np.max(x),padding=0)
        # if self.autoscaleY:
        #     self.plt1.setYRange(np.min(y),np.max(y),padding=0)

        # If enabled, show the position indicators
        if self.showPositionIndicators:
            # Add vertical lines to each plot that indicate the current time
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt4_line = self.plt4.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)

            # Connect the signal that is emitted when the current time changes to the function
            # that updates the position indicators
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)

        # Store the data as a dictionary with time as keys and position as values
        keys = time
        values = zip(zeroed_X, zeroed_Y)
        self.data = dict(zip(keys,values))

        # Reset the zoom of the polar plot
        self.r = None

# =============================================================================
#     def updatePolarPlot(self, direction,velocity):
#         # Clear the polar plot
#         self.plt4.clear()
#
#         # Add polar grid lines
#         self.plt4.addLine(x=0, pen=1)
#         self.plt4.addLine(y=0, pen=1)
#         for r in range(10, 50, 10):
#             r = r/10
#             circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
#             circle.setPen(pg.mkPen('w', width=0.5))
#             self.plt4.addItem(circle)
#
#         # Convert direction and velocity to cartesian coordinates
#         theta = np.radians(direction)
#         radius = velocity
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#
#         # Plot lines in the polar plot for each direction and velocity point
#         for i in range(len(x)):
#             path = QPainterPath(QPointF(0,0))
#             path.lineTo(QPointF(x[i],y[i]))
#             item = pg.QtGui.QGraphicsPathItem(path)
#             item.setPen(pg.mkPen('r', width=5))
#             self.plt4.addItem(item)
#
#         # Add position labels to the polar plot
#         labels = [0,90,180,270]
#         d = 6
#         pos = [ (d,0),(0,d),(-d,0),(0,-d) ]
#         for i,label in enumerate(labels):
#             text = pg.TextItem(str(label), color=(200,200,0))
#             self.plt4.addItem(text)
#             text.setPos(pos[i][0],pos[i][1])
#
#         # Add scale to the polar plot
#         for r in range(10, 50, 10):
#             r = r/10
#             text = pg.TextItem(str(r))
#             self.plt4.addItem(text)
#             text.setPos(0,r)
#
#         return
# =============================================================================

    def togglePoistionIndicator(self):
        # If position indicators are not shown, add them to the plots
        if self.showPositionIndicators == False:
            # Add dashed lines to the plots
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt4_line = self.plt4.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)

            # Connect the updatePositionIndicators method to the signal for time changes
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)

            # Update the flag and button text
            self.showPositionIndicators = True
            self.positionIndicator_button.setText("Hide position info")

        # If position indicators are shown, remove them from the plots
        else:
            # Remove the dashed lines from the plots
            self.plt1.removeItem(self.plt1_line)
            self.plt2.removeItem(self.plt2_line)
            self.plt4.removeItem(self.plt4_line)
            self.plt5.removeItem(self.plt5_line)
            self.plt6.removeItem(self.plt6_line)

            # Disconnect the updatePositionIndicators method from the signal for time changes
            self.mainGUI.plotWindow.sigTimeChanged.disconnect(self.updatePositionIndicators)

            # Update the flag and button text
            self.showPositionIndicators = False
            self.positionIndicator_button.setText("Show position info")



    def updatePositionIndicators(self, t):
        #match frames to flika window numbering
        #t = t+1
        # Set the position of the position indicators in all four plots to the current time t
        self.plt1_line.setPos(t)
        self.plt2_line.setPos(t)
        self.plt4_line.setPos(t)
        self.plt5_line.setPos(t)
        self.plt6_line.setPos(t)

        # If a rectangular ROI exists in plot 3, remove it
        if self.r != None:
            self.plt3.removeItem(self.r)

        # If the current time t is in the data dictionary, create a new rectangular ROI and add it to plot 3
        if t in self.data:
            self.r = pg.RectROI((self.data[t][0]-0.25,self.data[t][1]-0.25), size = pg.Point(0.5,0.5), movable=False,rotatable=False,resizable=False, pen=pg.mkPen('r',width=1))
            self.r.handlePen = None
            self.plt3.addItem(self.r)

    def show(self):
        # Show the plot window
        self.win.show()

    def close(self):
        # Close the plot window
        self.win.close()

    def hide(self):
        # Hide the plot window
        self.win.hide()
