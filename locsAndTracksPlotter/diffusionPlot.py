#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:28:30 2023

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

import flika
from flika.window import Window
import flika.global_vars as g

from scipy.optimize import curve_fit

from .helperFunctions import *

class DiffusionPlotWindow():
    """
    A class for creating the diffusion analysis window .

    Attributes:
    -----------
    mainGUI : MainGUI object
        The parent object that instantiated this class.
    win : QMainWindow
        The main window for the diffusion analysis.
    area : DockArea
        The area of the window where dock widgets can be placed.
    d1 : Dock
        The dock widget for the plot options.
    d2 : Dock
        The dock widget for the distance plot.
    d3 : Dock
        The dock widget for the histogram options.
    d4 : Dock
        The dock widget for the lag histogram.
    d5 : Dock
        The dock widget for the CDF options.
    d6 : Dock
        The dock widget for the CDF plot.
    """
    def __init__(self, mainGUI):
        super().__init__()

        self.mainGUI = mainGUI

        self.win =QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1200,500)
        self.win.setWindowTitle('Diffusion Analysis')

        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("plot options", size=(400, 100))
        self.d2 = Dock("distance plot", size=(400,400))
        self.d3 = Dock("histo options", size=(400,100))
        self.d4 = Dock("lag histogram", size=(400,400))
        self.d5 = Dock("CDF options", size=(400,100))
        self.d6 = Dock("CDF", size=(400,400))

        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d3, 'right', self.d1)
        self.area.addDock(self.d2, 'bottom', self.d1)
        self.area.addDock(self.d4, 'bottom', self.d3)

        self.area.addDock(self.d5, 'right', self.d3)
        self.area.addDock(self.d6, 'right', self.d4)

        #### DISTANCE SCATTER PLOT
        # Create a LayoutWidget for the options for the distance scatter plot.
        self.w1 = pg.LayoutWidget()

        # Create a ComboBox for selecting the type of plot.
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes= {'scatter':'scatter','line (slow with many tracks!)':'line'}
        self.plotTypeSelector.setItems(self.plotTypes)
        self.selectorLabel = QLabel("Plot type")

        # Create a SpinBox for selecting the size of the points.
        self.pointSize_selector = pg.SpinBox(value=3, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10)
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size")

        # Create a button for updating the plot.
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)

        # Add the widgets to the LayoutWidget.
        self.w1.addWidget(self.plotTypeSelector, row=0,col=1)
        self.w1.addWidget(self.selectorLabel, row=0,col=0)
        self.w1.addWidget(self.pointSizeLabel, row=1, col=0)
        self.w1.addWidget(self.pointSize_selector, row=1, col=1)
        self.w1.addWidget(self.plot_button, row=2, col=1)

        # Add the LayoutWidget to the dock.
        self.d1.addWidget(self.w1)

        # Create a PlotWidget for the distance plot.
        self.w3 = pg.PlotWidget(title="square of distance from origin")
        self.w3.plot()
        self.w3.setLabel('left', 'd squared', units ='')
        self.w3.setLabel('bottom', 'lags', units ='')
        self.d2.addWidget(self.w3)


        #### LAG HISTOGRAM
        # Create a LayoutWidget for the options for the lag histogram.
        self.w2 = pg.LayoutWidget()

        # Create a button for updating the histogram.
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)

        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(1)
        self.histoBin_selector.setMinimum(1)
        self.histoBin_selector.setMaximum(100000)
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)

        self.histoBin_label = QLabel('# of bins')

        self.w2.addWidget(self.histoBin_selector, row=0, col=1)
        self.w2.addWidget(self.histoBin_label, row=0, col=0)
        self.w2.addWidget(self.histo_button, row=1, col=1)

        self.d3.addWidget(self.w2)

        self.w4 = pg.PlotWidget(title="Distribution of mean SLDs")
        self.w4.plot()
        self.w4.setLabel('left', 'Count', units ='')
        self.w4.setLabel('bottom', 'mean sld per track', units ='nm')
        self.w4.getAxis('bottom').enableAutoSIPrefix(False)
        self.d4.addWidget(self.w4)

        ### CDF
        # Create a new layout widget for the CDF plot and its controls
        self.w5 = pg.LayoutWidget()

        # Add a button for updating the CDF plot
        self.cdf_button = QPushButton('Plot CDF')
        self.cdf_button.pressed.connect(self.updateCDF)

        # Add a spinbox for selecting the number of bins in the CDF plot
        self.cdfBin_selector = pg.SpinBox(value=100, int=True)
        self.cdfBin_selector.setSingleStep(1)
        self.cdfBin_selector.setMinimum(1)
        self.cdfBin_selector.setMaximum(100000)
        self.cdfBin_selector.sigValueChanged.connect(self.updateCDF)

        # Add a label for the bin selector
        self.cdfBin_label = QLabel('# of bins')

        # Add buttons for fitting one, two, or three-component exponential curves to the CDF
        self.fit_exp_dec_1_button = QPushButton('Fit 1 component exponential')
        self.fit_exp_dec_1_button.pressed.connect(self.fit_exp_dec_1)
        self.fit_exp_dec_2_button = QPushButton('Fit 2 component exponential')
        self.fit_exp_dec_2_button.pressed.connect(self.fit_exp_dec_2)
        self.fit_exp_dec_3_button = QPushButton('Fit 3 component exponential')
        self.fit_exp_dec_3_button.pressed.connect(self.fit_exp_dec_3)

        # Add the controls to the layout widget
        self.w5.addWidget(self.cdfBin_selector, row=0, col=1)
        self.w5.addWidget(self.cdfBin_label, row=0, col=0)
        self.w5.addWidget(self.cdf_button, row=1, col=1)
        self.w5.addWidget(self.fit_exp_dec_1_button , row=2, col=1)
        self.w5.addWidget(self.fit_exp_dec_2_button , row=3, col=1)
        self.w5.addWidget(self.fit_exp_dec_3_button , row=4, col=1)

        # Add the layout widget to the CDF plot dock
        self.d5.addWidget(self.w5)

        # Create a new plot widget for the CDF
        self.w6 = pg.PlotWidget(title="CDF")
        self.w6.plot()

        # Set the axis labels and disable auto SI prefix for the x-axis
        self.w6.setLabel('left', 'CDF', units ='')
        self.w6.setLabel('bottom', 'mean sld^2', units ='micron^2')
        self.w6.getAxis('bottom').enableAutoSIPrefix(False)

        # Add the CDF plot widget to its dock
        self.d6.addWidget(self.w6)

        # Add a legend to the CDF plot widget
        self.cdf_legend = self.w6.plotItem.addLegend()

        # Initialize the curve objects for the exponential fits
        self.exp_dec_1_curve = None
        self.exp_dec_2_curve = None
        self.exp_dec_3_curve = None

    def updatePlot(self):
        # Clear plot before updating
        self.w3.clear()

        # Check plot type and get data accordingly
        if self.plotTypeSelector.value() == 'line':
            # Group data by track number to plot lines
            if self.mainGUI.useFilteredData == False:
                df = self.mainGUI.data
            else:
                df = self.mainGUI.filteredData

            # Get lag number and d squared for each track
            x = df.groupby('track_number')['lagNumber'].apply(list)
            y = df.groupby('track_number')['d_squared'].apply(list)

            # Get unique track IDs
            trackID_list = np.unique(df['track_number']).astype(np.int)

            # Plot each track as a line
            for txid in trackID_list:
                path = pg.arrayToQPath(np.array(x[txid]),np.array(y[txid]))
                item = pg.QtGui.QGraphicsPathItem(path)
                item.setPen(pg.mkPen('w'))
                self.w3.addItem(item)

        elif self.plotTypeSelector.value() == 'scatter':
            # Get x and y data for scatter plot
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data['lagNumber'].to_numpy()
                y = self.mainGUI.data['d_squared'].to_numpy()
            else:
                x = self.mainGUI.filteredData['lagNumber'].to_numpy()
                y = self.mainGUI.filteredData['d_squared'].to_numpy()

            # Plot scatter plot with points
            self.w3.plot(x, y,
                         pen=None,
                         symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),
                         symbolSize=self.pointSize_selector.value())


        return

    def updateHisto(self):
        # Clear the histogram plot
        self.w4.clear()

        # Check if filtered data is being used or not
        if self.mainGUI.useFilteredData == False:
            plotDF = self.mainGUI.data.groupby('track_number').mean()
        else:
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean()

        # Calculate the mean lag in microns
        #meanLag = plotDF['lag'] * self.mainGUI.trackPlotOptions.pixelSize_selector.value()
        meanLag = plotDF['velocity'] * self.mainGUI.trackPlotOptions.pixelSize_selector.value()      #velocity is equivalent to lag distance for 1 frame

        # Set the start and end of the histogram bins and the number of bins to use
        start = 0
        end = np.max(meanLag)
        n = self.histoBin_selector.value()

        # Create the histogram data
        y, x = np.histogram(meanLag, bins=np.linspace(start, end, n))

        # Plot the histogram data
        self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), clear=True)

        # Return
        return

    def updateCDF(self):
        # Clear the widget to start with a blank slate
        self.w6.clear()

        # Check whether to use the filtered data or not
        if self.mainGUI.useFilteredData == False:
            # Use the original data
            plotDF = self.mainGUI.data.groupby('track_number').mean()
        else:
            # Use the filtered data
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean()

        # Calculate the squared lag distances in microns
        #self.squared_SLDs = plotDF['lag_squared'] * np.square(self.mainGUI.trackPlotOptions.pixelSize_selector.value()/1000)
        self.squared_SLDs = np.square(plotDF['velocity'] * (self.mainGUI.trackPlotOptions.pixelSize_selector.value()/1000))

        # Set the start and end points of the histogram, and the number of bins
        start=0
        end=np.max(self.squared_SLDs)
        n=self.cdfBin_selector.value()

        # Calculate the histogram using numpy
        count, bins_count = np.histogram(self.squared_SLDs, bins=np.linspace(start, end, n))

        # Calculate the probability density function and the cumulative distribution function
        pdf = count / sum(count)
        self.cdf_y = np.cumsum(pdf)
        self.cdf_x = bins_count[1:]

        # Get the maximum number of lags for normalization
        self.nlags = np.max(self.cdf_y)

        # Plot the CDF
        self.w6.plot(self.cdf_x, self.cdf_y, brush=(0,0,255,150), clear=True)

        # Add movable dashed lines to select a range on the CDF
        self.left_bound_line = self.w6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))
        self.right_bound_line = self.w6.addLine(x=np.max(self.squared_SLDs), pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))

        return

    def fit_exp_dec_1(self):
        # Remove any existing fitted curve and its legend item
        if self.exp_dec_1_curve is not None:
            self.w6.removeItem(self.exp_dec_1_curve)
            self.cdf_legend.removeItem(self.exp_dec_1_curve.name())

        # Get the left and right bounds of the fitting range
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])

        # Get the x and y data for the CDF plot
        xdata = self.cdf_x
        ydata = self.cdf_y

        # Select the data points within the fitting range
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]

        # Fit an exponential decay function to the selected data
        popt, pcov = curve_fit(exp_dec, xfit, ydata[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
        tau_fit = popt[1]
        D_fit = self.tau_to_D(tau_fit)

        # Print the fitted diffusion coefficient
        print('D = {0:.4g} um^2 s^-1'.format(D_fit))

        # Generate the fitted curve and add it to the plot
        yfit = exp_dec(xfit, *popt)
        self.exp_dec_1_curve = self.w6.plot(xfit, yfit, pen='g', name=' Fit. D = {0:.4g} um^2 s^-1'.format(D_fit))

        # TODO: Residual plot implementation commented out.
        # Generate a residual plot (optional)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_2(self):
        if self.exp_dec_2_curve is not None:
            self.w6.removeItem(self.exp_dec_2_curve)
            self.cdf_legend.removeItem(self.exp_dec_2_curve.name())

        # Determine the bounds for the fitting based on the position of the two vertical lines
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])

        # Get the data to fit from the CDF plot
        xdata = self.cdf_x
        ydata = self.cdf_y

        # Mask the data to fit within the bounds
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]

        # Perform the curve fitting using the double-exponential decay function (exp_dec_2)
        # and the masked data
        popt, pcov = curve_fit(exp_dec_2, xfit, ydata[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))

        # Extract the fitted parameters
        A1 = popt[0]
        A2 = -1 - A1
        tau1_fit = popt[1]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[2]
        D2_fit = self.tau_to_D(tau2_fit)

        # Print the fitted diffusion coefficients and amplitudes
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2. A1={2:.2g} A2={3:.2g}'.format(D1_fit, D2_fit, A1, A2)
        print(msg)

        # Calculate the fit line and plot it on the CDF plot
        yfit = exp_dec_2(xfit, *popt)
        self.exp_dec_2_curve = self.w6.plot(xfit, yfit, pen='r', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_3(self):
        # Check if an existing plot for the fit exists and remove it
        if self.exp_dec_3_curve is not None:
            self.w6.removeItem(self.exp_dec_3_curve)
            self.cdf_legend.removeItem(self.exp_dec_3_curve.name())

        # Get the left and right bounds for the fit from the GUI sliders
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])

        # Get the x and y data for the CDF plot from the GUI
        xdata = self.cdf_x
        ydata = self.cdf_y

        # Create a mask to only fit data within the selected bounds
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]

        # Fit the data using the three-exponential decay function and bounds on the parameters
        popt, pcov = curve_fit(exp_dec_3, xfit, ydata[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))

        # Extract the fitted parameters and compute diffusion coefficients
        A1 = popt[0]
        A2 = popt[1]
        A3 = -1 - A1 - A2
        tau1_fit = popt[2]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[3]
        D2_fit = self.tau_to_D(tau2_fit)
        tau3_fit = popt[4]
        D3_fit = self.tau_to_D(tau3_fit)

        # Create a string summarizing the fit parameters
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2, D3 = {2:.4g} um2/2. A1={3:.2g} A2={4:.2g}, A3={5:.2g}'.format(D1_fit, D2_fit, D3_fit, A1, A2, A3)

        # Generate the fitted curve and add it to the plot with a label containing the fit parameters
        yfit = exp_dec_3(xfit, *popt)
        self.exp_dec_3_curve = self.w6.plot(xfit, yfit, pen='y', name=' Fit. '+msg)

        # Uncomment these lines to generate a plot of the residuals
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))


    def tau_to_D(self, tau):
        """
        tau = 4Dt
        tau is decay constant of exponential fit
        D is diffusion coefficient
        t is duration of one lag (exposure time) in seconds
        """
        t = (self.mainGUI.trackPlotOptions.frameLength_selector.value()/1000) * self.nlags
        D = tau / (4 * t)
        return D


    def show(self):
        self.win.show()

    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()
