#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:28:47 2023

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

class ChartDock():
    """
    A class for creating a dockable window for displaying analysis charts.

    Attributes:
    -----------
    mainGUI : QtGui.QMainWindow
        The main window of the GUI containing this chart window.

    win : QtGui.QMainWindow
        The QMainWindow instance for the chart window.

    area : DockArea
        The DockArea instance for the chart window, which contains the individual docks.

    d1 : Dock
        The dock for plot options.

    d2 : Dock
        The dock for the main plot.

    d3 : Dock
        The dock for histogram options.

    d4 : Dock
        The dock for the histogram plot.
    """
    def __init__(self, mainGUI):
        super().__init__()

        self.mainGUI = mainGUI

        # Create a new QMainWindow instance
        self.win = QMainWindow()

        # Create a new DockArea instance and set it as the central widget of the main window
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        # Set the size and title of the main window
        self.win.resize(1000,500)
        self.win.setWindowTitle('Plots')

        ## Create docks, place them into the window one at a time.

        # Create a dock for plot options and set its size
        self.d1 = Dock("plot options", size=(500, 100))

        # Create a dock for the main plot and set its size
        self.d2 = Dock("plot", size=(500,400))

        # Create a dock for histogram options and set its size
        self.d3 = Dock("histogram options", size=(500,100))

        # Create a dock for the histogram plot and set its size
        self.d4 = Dock("histogram", size=(500,400))

        # Add the plot options dock to the left side of the window
        self.area.addDock(self.d1, 'left')

        # Add the histogram options dock to the right side of the window, next to the plot options dock
        self.area.addDock(self.d3, 'right', self.d1)

        # Add the main plot dock to the bottom of the window, next to the plot options dock
        self.area.addDock(self.d2, 'bottom', self.d1)

        # Add the histogram plot dock to the bottom of the window, next to the histogram options dock
        self.area.addDock(self.d4, 'bottom', self.d3)

        #### SCATTER PLOT
        # Create a new layout widget for the scatter plot controls
        self.w1 = pg.LayoutWidget()

        # Create a combo box to choose between point and track data
        self.pointOrTrackData_selector_plot = pg.ComboBox()

        # Define the options for the plot data
        self.plotDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}

        # Set the options for the combo box
        self.pointOrTrackData_selector_plot.setItems(self.plotDataChoice)

        # Create labels for the x and y axes
        self.xlabel = QLabel("x:")
        self.ylabel = QLabel("y:")

        # Create a combo box for the x-axis data
        self.xColSelector = pg.ComboBox()

        # Define the default option for x-axis data
        self.xcols = {'None':'None'}

        # Set the options for the x-axis combo box
        self.xColSelector.setItems(self.xcols)

        # Create a combo box for the y-axis data
        self.yColSelector = pg.ComboBox()

        # Define the default option for y-axis data
        self.ycols = {'None':'None'}

        # Set the options for the y-axis combo box
        self.yColSelector.setItems(self.ycols)

        # Create a combo box for selecting the type of plot
        self.plotTypeSelector = pg.ComboBox()

        # Define the options for the plot type combo box
        self.plotTypes= {'scatter':'scatter', 'line':'line'}

        # Set the options for the plot type combo box
        self.plotTypeSelector.setItems(self.plotTypes)

        # Create a label for the plot type selector
        self.selectorLabel = QLabel("Plot type")

        # Create a spin box for selecting the point size
        self.pointSize_selector = pg.SpinBox(value=7, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10)
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)

        # Create a label for the point size selector
        self.pointSizeLabel = QLabel("Point size")

        # Create a button for plotting the data
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)

        # Add all the widgets to the layout widget
        self.w1.addWidget(self.pointOrTrackData_selector_plot , row=0, col=1)
        self.w1.addWidget(self.xColSelector, row=1, col=1)
        self.w1.addWidget(self.yColSelector, row=2, col=1)
        self.w1.addWidget(self.xlabel, row=1, col=0)
        self.w1.addWidget(self.ylabel, row=2, col=0)
        self.w1.addWidget(self.plotTypeSelector, row=3,col=1)
        self.w1.addWidget(self.selectorLabel, row=3,col=0)
        self.w1.addWidget(self.pointSizeLabel, row=4, col=0)
        self.w1.addWidget(self.pointSize_selector, row=4, col=1)
        self.w1.addWidget(self.plot_button, row=5, col=1)

        # Add the layout widget to the dock
        self.d1.addWidget(self.w1)

        #### HISTOGRAM
        # create a layout widget to hold the histogram controls
        self.w2 = pg.LayoutWidget()

        # create a ComboBox to select whether to plot point data or track means
        self.pointOrTrackData_selector_histo = pg.ComboBox()
        self.histoDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}
        self.pointOrTrackData_selector_histo.setItems(self.histoDataChoice)

        # create a ComboBox to select which column to plot
        self.colSelector = pg.ComboBox()
        self.cols = {'None':'None'}
        self.colSelector.setItems(self.cols)

        # create a label for the column selector ComboBox
        self.collabel = QLabel("col:")

        # create a button to plot the histogram
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)

        # create a SpinBox to select the number of bins in the histogram
        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(1)
        self.histoBin_selector.setMinimum(1)
        self.histoBin_selector.setMaximum(100000)
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)

        # create a label for the SpinBox to select the number of bins in the histogram
        self.histoBin_label = QLabel('# of bins')

        # create a label for the checkbox to plot the cumulative histogram
        self.cumulativeTick_label = QLabel('cumulative')

        # create a checkbox to plot the cumulative histogram
        self.cumulativeTick = CheckBox()
        self.cumulativeTick.setChecked(False)

        # add the widgets to the layout widget
        self.w2.addWidget(self.pointOrTrackData_selector_histo , row=0, col=1)
        self.w2.addWidget(self.colSelector, row=1, col=1)
        self.w2.addWidget(self.collabel, row=1, col=0)
        self.w2.addWidget(self.histoBin_selector, row=2, col=1)
        self.w2.addWidget(self.histoBin_label, row=2, col=0)
        self.w2.addWidget(self.cumulativeTick_label, row=3, col=0)
        self.w2.addWidget(self.cumulativeTick, row=3, col=1)
        self.w2.addWidget(self.histo_button, row=4, col=1)

        # add the layout widget to the histogram dock
        self.d3.addWidget(self.w2)

        self.w3 = pg.PlotWidget(title="plot")
        self.w3.plot()
        self.w3.setLabel('left', 'y-axis', units ='')
        self.w3.setLabel('bottom', 'x-axis', units ='')
        self.d2.addWidget(self.w3)

        # create a plot widget for the histogram
        self.w4 = pg.PlotWidget(title="histogram")
        self.w4.plot()
        self.w4.setLabel('left', '# of observations', units ='')
        self.w4.setLabel('bottom', 'value', units ='')

        # add the plot widget to the histogram dock
        self.d4.addWidget(self.w4)


    def updatePlot(self):
        # Clear the current plot
        self.w3.clear()

        # Check if user selected point data or track means
        if self.pointOrTrackData_selector_plot.value() == 'Point Data':

            # If not using filtered data, extract x and y data from the main data table
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.data[self.yColSelector.value()].to_numpy()
            # If using filtered data, extract x and y data from the filtered data table
            else:
                x = self.mainGUI.filteredData[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.filteredData[self.yColSelector.value()].to_numpy()

        else:
            # If not using filtered data, group the data by track number and take the mean for each track
            if self.mainGUI.useFilteredData == False:
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()
            # If using filtered data, group the filtered data by track number and take the mean for each track
            else:
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean()

            # Extract x and y data from the track means data table
            x = plotDF[self.xColSelector.value()].to_numpy()
            y = plotDF[self.yColSelector.value()].to_numpy()

        # Check if user selected line or scatter plot
        if self.plotTypeSelector.value() == 'line':
            # Plot line using the selected x and y data, and set stepMode=False to draw a continuous line
            self.w3.plot(x, y, stepMode=False, brush=(0,0,255,150), clear=True)
        elif self.plotTypeSelector.value() == 'scatter':
            # Plot scatter plot using the selected x and y data
            self.w3.plot(x, y,
                         pen=None,                                              # Set pen=None to remove the line border around each point, ,
                         symbol='o',                                            # set symbol='o' to use circles as symbols
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),        # set symbolPen=pg.mkPen(color=(0, 0, 255), width=0) to remove the border around each circle,
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),                # set symbolBrush=pg.mkBrush(0, 0, 255, 255) to fill the circles with blue color, and
                         symbolSize=self.pointSize_selector.value())            # set symbolSize=self.pointSize_selector.value() to set the size of the circles

        # Set the labels for the x and y axes of the plot
        self.w3.setLabel('left', self.yColSelector.value(), units=None)
        self.w3.setLabel('bottom', self.xColSelector.value(), units=None)
        return

    def updateHisto(self):
        # clear the plot
        self.w4.clear()

        # check if point or track data is selected
        if self.pointOrTrackData_selector_histo.value() == 'Point Data':
            # use either data or filteredData based on the useFilteredData flag
            if self.mainGUI.useFilteredData == False:
                vals = self.mainGUI.data[self.colSelector.value()]
            else:
                vals = self.mainGUI.filteredData[self.colSelector.value()]

        else:
            # group the data by track number and calculate the mean of each group
            # use either data or filteredData based on the useFilteredData flag
            if self.mainGUI.useFilteredData == False:
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()
            else:
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean()

            # extract the values of the selected column from the plot dataframe
            vals = plotDF[self.colSelector.value()]

        # define the range and number of bins for the histogram
        start=0
        end=np.max(vals)
        n=self.histoBin_selector.value()

        # check if the cumulative tick is selected
        if self.cumulativeTick.isChecked():
            # calculate the cumulative distribution function (CDF)
            count,bins_count = np.histogram(vals, bins=np.linspace(start, end, n))
            pdf = count / sum(count)
            y = np.cumsum(pdf)
            x = bins_count[1:]
            # plot the CDF
            self.w4.plot(x, y, brush=(0,0,255,150), clear=True)

        else:
            # calculate the histogram
            y,x = np.histogram(vals, bins=np.linspace(start, end, n))
            # plot the histogram
            self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)

        # set the label of the x-axis to the selected column name
        self.w4.setLabel('bottom', self.colSelector.value(), units = None)
        return

    def show(self):
        # show the window
        self.win.show()

    def close(self):
        # close the window
        self.win.close()

    def hide(self):
        # hide the window
        self.win.hide()
