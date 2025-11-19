#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:28:30 2023

@author: george
"""

import logging
import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.dockarea import DockArea, Dock
from scipy.optimize import curve_fit
import flika.global_vars as g
from .helperFunctions import exp_dec, exp_dec_2, exp_dec_3

logger = logging.getLogger(__name__)

class DiffusionPlotWindow(QtWidgets.QMainWindow):
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Diffusion Analysis')
        self.resize(1200, 500)

        self.area = DockArea()
        self.setCentralWidget(self.area)

        # Create docks
        self.d1 = Dock("Plot Options", size=(400, 100))
        self.d2 = Dock("Distance Plot", size=(400, 400))
        self.d3 = Dock("Histogram Options", size=(400, 100))
        self.d4 = Dock("Lag Histogram", size=(400, 400))
        self.d5 = Dock("CDF Options", size=(400, 100))
        self.d6 = Dock("CDF", size=(400, 400))

        # Add docks to the layout
        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d3, 'right', self.d1)
        self.area.addDock(self.d2, 'bottom', self.d1)
        self.area.addDock(self.d4, 'bottom', self.d3)
        self.area.addDock(self.d5, 'right', self.d3)
        self.area.addDock(self.d6, 'right', self.d4)

        self.setupDistancePlot()
        self.setupHistogram()
        self.setupCDF()

    def setupDistancePlot(self):
        self.w1 = pg.LayoutWidget()
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes = {'scatter': 'scatter', 'line (slow with many tracks!)': 'line'}
        self.plotTypeSelector.setItems(self.plotTypes)
        self.selectorLabel = QtWidgets.QLabel("Plot type")

        self.pointSize_selector = pg.SpinBox(value=3, int=True, step=1, bounds=(1, 10))
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QtWidgets.QLabel("Point size")

        self.plot_button = QtWidgets.QPushButton('Plot')
        self.plot_button.clicked.connect(self.updatePlot)

        self.w1.addWidget(self.selectorLabel, row=0, col=0)
        self.w1.addWidget(self.plotTypeSelector, row=0, col=1)
        self.w1.addWidget(self.pointSizeLabel, row=1, col=0)
        self.w1.addWidget(self.pointSize_selector, row=1, col=1)
        self.w1.addWidget(self.plot_button, row=2, col=1)

        self.d1.addWidget(self.w1)

        self.w3 = pg.PlotWidget(title="Square of Distance from Origin")
        self.w3.setLabel('left', 'd squared', units='')
        self.w3.setLabel('bottom', 'lags', units='')
        self.d2.addWidget(self.w3)

    def setupHistogram(self):
        self.w2 = pg.LayoutWidget()
        self.histo_button = QtWidgets.QPushButton('Plot Histogram')
        self.histo_button.clicked.connect(self.updateHisto)

        self.histoBin_selector = pg.SpinBox(value=100, int=True, step=1, bounds=(1, 100000))
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        self.histoBin_label = QtWidgets.QLabel('# of bins')

        self.w2.addWidget(self.histoBin_label, row=0, col=0)
        self.w2.addWidget(self.histoBin_selector, row=0, col=1)
        self.w2.addWidget(self.histo_button, row=1, col=1)

        self.d3.addWidget(self.w2)

        self.w4 = pg.PlotWidget(title="Distribution of Mean SLDs")
        self.w4.setLabel('left', 'Count', units='')
        self.w4.setLabel('bottom', 'mean sld per track', units='nm')
        self.w4.getAxis('bottom').enableAutoSIPrefix(False)
        self.d4.addWidget(self.w4)

    def setupCDF(self):
        self.w5 = pg.LayoutWidget()
        self.cdf_button = QtWidgets.QPushButton('Plot CDF')
        self.cdf_button.clicked.connect(self.updateCDF)

        self.cdfBin_selector = pg.SpinBox(value=100, int=True, step=1, bounds=(1, 100000))
        self.cdfBin_selector.sigValueChanged.connect(self.updateCDF)
        self.cdfBin_label = QtWidgets.QLabel('# of bins')

        self.fit_exp_dec_1_button = QtWidgets.QPushButton('Fit 1 component exponential')
        self.fit_exp_dec_1_button.clicked.connect(self.fit_exp_dec_1)
        self.fit_exp_dec_2_button = QtWidgets.QPushButton('Fit 2 component exponential')
        self.fit_exp_dec_2_button.clicked.connect(self.fit_exp_dec_2)
        self.fit_exp_dec_3_button = QtWidgets.QPushButton('Fit 3 component exponential')
        self.fit_exp_dec_3_button.clicked.connect(self.fit_exp_dec_3)

        self.w5.addWidget(self.cdfBin_label, row=0, col=0)
        self.w5.addWidget(self.cdfBin_selector, row=0, col=1)
        self.w5.addWidget(self.cdf_button, row=1, col=1)
        self.w5.addWidget(self.fit_exp_dec_1_button, row=2, col=1)
        self.w5.addWidget(self.fit_exp_dec_2_button, row=3, col=1)
        self.w5.addWidget(self.fit_exp_dec_3_button, row=4, col=1)

        self.d5.addWidget(self.w5)

        self.w6 = pg.PlotWidget(title="CDF")
        self.w6.setLabel('left', 'CDF', units='')
        self.w6.setLabel('bottom', 'mean sld^2', units='micron^2')
        self.w6.getAxis('bottom').enableAutoSIPrefix(False)
        self.d6.addWidget(self.w6)

        self.cdf_legend = self.w6.addLegend()

        self.exp_dec_1_curve = None
        self.exp_dec_2_curve = None
        self.exp_dec_3_curve = None

    def updatePlot(self):
        self.w3.clear()
        data = self.mainGUI.filteredData if self.mainGUI.useFilteredData else self.mainGUI.data

        try:
            if self.plotTypeSelector.value() == 'line':
                x = data.groupby('track_number')['lagNumber'].apply(list)
                y = data.groupby('track_number')['d_squared'].apply(list)
                trackID_list = np.unique(data['track_number']).astype(np.int)

                for txid in trackID_list:
                    path = pg.arrayToQPath(np.array(x[txid]), np.array(y[txid]))
                    item = QtWidgets.QGraphicsPathItem(path)
                    item.setPen(pg.mkPen('w'))
                    self.w3.addItem(item)
            else:
                x = data['lagNumber'].to_numpy()
                y = data['d_squared'].to_numpy()
                self.w3.plot(x, y, pen=None, symbol='o', symbolPen=None,
                             symbolBrush=(0, 0, 255, 255),
                             symbolSize=self.pointSize_selector.value())

            logger.info("Distance plot updated successfully")
        except Exception as e:
            logger.error(f"Error in updatePlot: {str(e)}")
            g.m.statusBar().showMessage(f"Error updating plot: {str(e)}")

    def updateHisto(self):
        self.w4.clear()

        try:
            data = self.mainGUI.filteredData if self.mainGUI.useFilteredData else self.mainGUI.data

            data['track_number'] = pd.to_numeric(data['track_number'], errors='coerce')
            data['velocity'] = pd.to_numeric(data['velocity'], errors='coerce')
            data = data.dropna(subset=['track_number', 'velocity'])

            plotDF = data.groupby('track_number').mean()
            meanLag = plotDF['velocity'] * self.mainGUI.trackPlotOptions.pixelSize_selector.value()

            start, end = 0, np.max(meanLag)
            n = self.histoBin_selector.value()

            y, x = np.histogram(meanLag, bins=np.linspace(start, end, n))
            self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

            logger.info("Histogram updated successfully")
        except Exception as e:
            logger.error(f"Error in updateHisto: {str(e)}")
            g.m.statusBar().showMessage(f"Error updating histogram: {str(e)}")

    def updateCDF(self):
        self.w6.clear()

        try:
            data = self.mainGUI.filteredData if self.mainGUI.useFilteredData else self.mainGUI.data
            plotDF = data.groupby('track_number').mean()

            self.squared_SLDs = np.square(plotDF['velocity'] * (self.mainGUI.trackPlotOptions.pixelSize_selector.value() / 1000))

            start, end = 0, np.max(self.squared_SLDs)
            n = self.cdfBin_selector.value()

            count, bins_count = np.histogram(self.squared_SLDs, bins=np.linspace(start, end, n))
            pdf = count / sum(count)
            self.cdf_y = np.cumsum(pdf)
            self.cdf_x = bins_count[1:]

            self.nlags = np.max(self.cdf_y)

            self.w6.plot(self.cdf_x, self.cdf_y, brush=(0, 0, 255, 150))

            self.left_bound_line = pg.InfiniteLine(pos=0, angle=90, movable=True, bounds=(start, end))
            self.right_bound_line = pg.InfiniteLine(pos=np.max(self.squared_SLDs), angle=90, movable=True, bounds=(start, end))
            self.w6.addItem(self.left_bound_line)
            self.w6.addItem(self.right_bound_line)

            logger.info("CDF updated successfully")
        except Exception as e:
            logger.error(f"Error in updateCDF: {str(e)}")
            g.m.statusBar().showMessage(f"Error updating CDF: {str(e)}")

    def fit_exp_dec_1(self):
        self._fit_exp_dec(exp_dec, 1)

    def fit_exp_dec_2(self):
        self._fit_exp_dec(exp_dec_2, 2)

    def fit_exp_dec_3(self):
        self._fit_exp_dec(exp_dec_3, 3)

    def _fit_exp_dec(self, func, n_components):
        if hasattr(self, f'exp_dec_{n_components}_curve'):
            curve = getattr(self, f'exp_dec_{n_components}_curve')
            if curve is not None:
                self.w6.removeItem(curve)
                self.cdf_legend.removeItem(curve.name())

        try:
            left_bound = min(self.left_bound_line.value(), self.right_bound_line.value())
            right_bound = max(self.left_bound_line.value(), self.right_bound_line.value())

            x_fit_mask = (left_bound <= self.cdf_x) & (self.cdf_x <= right_bound)
            xfit = self.cdf_x[x_fit_mask]

            if n_components == 1:
                popt, _ = curve_fit(func, xfit, self.cdf_y[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
                tau_fit = popt[1]
                D_fit = self.tau_to_D(tau_fit)
                msg = f'D = {D_fit:.4g} um^2 s^-1'
            elif n_components == 2:
                popt, _ = curve_fit(func, xfit, self.cdf_y[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))
                A1, tau1_fit, tau2_fit = popt
                A2 = -1 - A1
                D1_fit, D2_fit = self.tau_to_D(tau1_fit), self.tau_to_D(tau2_fit)
                msg = f'D1 = {D1_fit:.4g} um2/2, D2 = {D2_fit:.4g} um2/2. A1={A1:.2g} A2={A2:.2g}'
            else:
                popt, _ = curve_fit(func, xfit, self.cdf_y[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))
                A1, A2, tau1_fit, tau2_fit, tau3_fit = popt
                A3 = -1 - A1 - A2
                D1_fit, D2_fit, D3_fit = self.tau_to_D(tau1_fit), self.tau_to_D(tau2_fit), self.tau_to_D(tau3_fit)
                msg = f'D1 = {D1_fit:.4g} um2/2, D2 = {D2_fit:.4g} um2/2, D3 = {D3_fit:.4g} um2/2. A1={A1:.2g} A2={A2:.2g}, A3={A3:.2g}'

            yfit = func(xfit, *popt)
            curve = self.w6.plot(xfit, yfit, pen=pg.mkPen(color=(n_components * 85, 255, 0), width=2),
                                 name=f' Fit. {msg}')
            setattr(self, f'exp_dec_{n_components}_curve', curve)

            logger.info(f"Exponential decay fit ({n_components} component(s)) completed successfully")
            print(msg)
        except Exception as e:
            logger.error(f"Error in exponential decay fit ({n_components} component(s)): {str(e)}")
            g.m.statusBar().showMessage(f"Error in fit: {str(e)}")

    def tau_to_D(self, tau):
        """
        Convert tau to diffusion coefficient.

        tau = 4Dt
        tau is decay constant of exponential fit
        D is diffusion coefficient
        t is duration of one lag (exposure time) in seconds
        """
        t = (self.mainGUI.trackPlotOptions.frameLength_selector.value() / 1000) * self.nlags
        D = tau / (4 * t)
        return D

    def show(self):
        super().show()
        logger.info("DiffusionPlotWindow displayed")

    def close(self):
        super().close()
        logger.info("DiffusionPlotWindow closed")

    def hide(self):
        super().hide()
        logger.info("DiffusionPlotWindow hidden")

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    mainGUI = type('obj', (object,), {'trackPlotOptions': type('obj', (object,), {'pixelSize_selector': type('obj', (object,), {'value': lambda: 108})(),
                                                                                  'frameLength_selector': type('obj', (object,), {'value': lambda: 10})()})(),
                                      'filteredData': pd.DataFrame({'track_number': range(100), 'velocity': np.random.rand(100)}),
                                      'useFilteredData': True})

    diffusion_plot = DiffusionPlotWindow(mainGUI)
    diffusion_plot.show()
    sys.exit(app.exec_())
