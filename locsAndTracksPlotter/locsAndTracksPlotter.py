# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

@author: george.dickinson@gmail.com
"""
import warnings
warnings.simplefilter(action='ignore', category=Warning)

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from os.path import expanduser
import os
import math
import sys
from scipy.optimize import curve_fit
import time

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

import numba
pg.setConfigOption('useNumba', True)

import pandas as pd
from matplotlib import pyplot as plt

from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea


def dictFromList(l):
    # Create a zip object from two lists
    zipbObj = zip(l, l)
    return dict(zipbObj)


def exp_dec(x, A1, tau):
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x, A1, tau1, tau2):
    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x, A1, A2, tau1, tau2, tau3):
    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)


def open_file_gui(prompt="Open File", directory=None, filetypes=''):
    """ File dialog for opening an existing file, isolated to handle tuple/string return value
    
    Args:
        prompt (str): string to display at the top of the window
        directory (str): initial directory to open
        filetypes (str): argument for filtering file types separated by ;; (*.png) or (Images *.png);;(Other *.*)
    
    Returns:
        str: the file (path+file+extension) selected, or None
    """
    filename = None
    if directory is None:
        filename = g.settings['filename']
        try:
            directory = os.path.dirname(filename)
        except:
            directory = None
    if directory is None or filename is None:
        filename = QFileDialog.getOpenFileName(g.m, prompt, '', filetypes)
    else:
        filename = QFileDialog.getOpenFileName(g.m, prompt, filename, filetypes)
    if isinstance(filename, tuple):
        filename, ext = filename
        if ext and '.' not in filename:
            filename += '.' + ext.rsplit('.')[-1]
    if filename is None or str(filename) == '':
        g.m.statusBar().showMessage('No File Selected')
        return None
    else:
        return str(filename)

    
class FileSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a file to save.  Once you do, it sets self.filename and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Load Data')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.filename = ''
        self.columns = []
        self.pixelSize = 108    #nanometers

        
    def buttonclicked(self):
        if g.win == None:
            g.alert('Load tiff stack and set as current window first')
            return
        prompt = 'testing fileSelector'
        self.filename = open_file_gui(prompt, filetypes=self.filetypes)
        self.label.setText('...'+os.path.split(self.filename)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filename

    def setValue(self, filename):
        self.filename = str(filename)
        self.label.setText('...' + os.path.split(self.filename)[-1][-20:])    



class FlowerPlotWindow():
    def __init__(self, mainGUI):
        super().__init__()  

        self.mainGUI = mainGUI

        #setup window
        self.win = pg.GraphicsWindow()
        self.win.resize(500, 500)
        self.win.setWindowTitle('Flower Plot')

        self.plt = self.win.addPlot(title='plot')  
        self.plt.setAspectLocked()
        self.plt.showGrid(x=True, y=True)
        self.plt.setXRange(-10,10)
        self.plt.setYRange(-10,10)
        self.plt.getViewBox().invertY(True)        
        
        self.plt.setLabel('left', 'y', units ='pixels')
        self.plt.setLabel('bottom', 'x', units ='pixels') 
        
        self.pathitems = []

       
    def clearTracks(self):
        if self.win is not None and not self.win.closed:
            for pathitem in self.pathitems:
                self.plt.removeItem(pathitem)
        self.pathitems = [] 

    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()


class DiffusionPlotWindow():
    def __init__(self, mainGUI):
        super().__init__()  

        self.mainGUI = mainGUI
        
        self.win =QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1200,500)
        self.win.setWindowTitle('Diffusion Analysis')

        self.pixelSize = self.mainGUI.pixelSize
        
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
        self.w1 = pg.LayoutWidget()
    
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes= {'scatter':'scatter','line (slow with many tracks!)':'line'}
        self.plotTypeSelector.setItems(self.plotTypes)  
        self.selectorLabel = QLabel("Plot type")  

        self.pointSize_selector = pg.SpinBox(value=3, int=True)
        self.pointSize_selector.setSingleStep(1)       
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10) 
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size") 
        
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)
        
        self.w1.addWidget(self.plotTypeSelector, row=0,col=1)
        self.w1.addWidget(self.selectorLabel, row=0,col=0) 
        self.w1.addWidget(self.pointSizeLabel, row=1, col=0)         
        self.w1.addWidget(self.pointSize_selector, row=1, col=1)        
        self.w1.addWidget(self.plot_button, row=2, col=1)         
        
        self.d1.addWidget(self.w1)    

        self.w3 = pg.PlotWidget(title="square of distance from origin")
        self.w3.plot()
        self.w3.setLabel('left', 'd squared', units ='')
        self.w3.setLabel('bottom', 'lags', units ='')  
        self.d2.addWidget(self.w3)  

        
        #### LAG HISTOGRAM
        self.w2 = pg.LayoutWidget()             
        
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
        self.w5 = pg.LayoutWidget()             
        
        self.cdf_button = QPushButton('Plot CDF')
        self.cdf_button.pressed.connect(self.updateCDF)

        self.cdfBin_selector = pg.SpinBox(value=100, int=True)
        self.cdfBin_selector.setSingleStep(1)       
        self.cdfBin_selector.setMinimum(1)
        self.cdfBin_selector.setMaximum(100000) 
        self.cdfBin_selector.sigValueChanged.connect(self.updateCDF)
        
        self.cdfBin_label = QLabel('# of bins')
        
        self.fit_exp_dec_1_button = QPushButton('Fit 1 component exponential')
        self.fit_exp_dec_1_button.pressed.connect(self.fit_exp_dec_1)
        self.fit_exp_dec_2_button = QPushButton('Fit 2 component exponential')
        self.fit_exp_dec_2_button.pressed.connect(self.fit_exp_dec_2)
        self.fit_exp_dec_3_button = QPushButton('Fit 3component exponential')
        self.fit_exp_dec_3_button.pressed.connect(self.fit_exp_dec_3)
        
        self.w5.addWidget(self.cdfBin_selector, row=0, col=1)
        self.w5.addWidget(self.cdfBin_label, row=0, col=0)        
        self.w5.addWidget(self.cdf_button, row=1, col=1) 

        self.w5.addWidget(self.fit_exp_dec_1_button , row=2, col=1) 
        self.w5.addWidget(self.fit_exp_dec_2_button , row=3, col=1) 
        self.w5.addWidget(self.fit_exp_dec_3_button , row=4, col=1) 
        
        self.d5.addWidget(self.w5)       
    
        self.w6 = pg.PlotWidget(title="CDF")
        self.w6.plot()
        self.w6.setLabel('left', 'CDF', units ='')
        self.w6.setLabel('bottom', 'mean sld^2', units ='micron^2')          
        self.w6.getAxis('bottom').enableAutoSIPrefix(False) 
        self.d6.addWidget(self.w6)   
                
        self.cdf_legend = self.w6.plotItem.addLegend()
        
        self.exp_dec_1_curve = None
        self.exp_dec_2_curve = None
        self.exp_dec_3_curve = None 
        

        

    def updatePlot(self):
        self.w3.clear()
                     
        if self.plotTypeSelector.value() == 'line':
            
            if self.mainGUI.useFilteredData == False:            
                df = self.mainGUI.data
            else:           
                df = self.mainGUI.filteredData
            
            x = df.groupby('track_number')['lagNumber'].apply(list)
            y = df.groupby('track_number')['d_squared'].apply(list)        

            trackID_list = np.unique(df['track_number']).astype(np.int)

            
            for txid in trackID_list:
                
                path = pg.arrayToQPath(np.array(x[txid]),np.array(y[txid]))
                item = pg.QtGui.QGraphicsPathItem(path)
                item.setPen(pg.mkPen('w'))                
                self.w3.addItem(item)
            
            
        elif self.plotTypeSelector.value() == 'scatter':
            
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data['lagNumber'].to_numpy()
                y = self.mainGUI.data['d_squared'].to_numpy() 
            else:
                x = self.mainGUI.filteredData['lagNumber'].to_numpy()
                y = self.mainGUI.filteredData['d_squared'].to_numpy() 
            
            self.w3.plot(x, y,
                         pen = None,
                         symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),                                      
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),
                         symbolSize=self.pointSize_selector.value())    
            
            
        return
    
    def updateHisto(self):
        self.w4.clear()
             
        if self.mainGUI.useFilteredData == False:                
            plotDF = self.mainGUI.data.groupby('track_number').mean()                
        else:                
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean() 
        
        # in microns
        meanLag = plotDF['lag'] * self.pixelSize

        start=0
        end=np.max(meanLag)
        n=self.histoBin_selector.value()

        y,x = np.histogram(meanLag, bins=np.linspace(start, end, n))     
        self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)     
        return


    def updateCDF(self):
        self.w6.clear()
             
        if self.mainGUI.useFilteredData == False:                
            plotDF = self.mainGUI.data.groupby('track_number').mean()                
        else:                
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean() 
        
        # in microns squared   
        self.squared_SLDs = plotDF['lag_squared'] * np.square(self.pixelSize/1000)

        start=0
        end=np.max(self.squared_SLDs)
        n=self.cdfBin_selector.value()

        count,bins_count = np.histogram(self.squared_SLDs, bins=np.linspace(start, end, n)) 
        
        pdf = count / sum(count)
        self.cdf_y = np.cumsum(pdf)        
        self.cdf_x = bins_count[1:]
        
        self.nlags = np.max(self.cdf_y)
        
        self.w6.plot(self.cdf_x, self.cdf_y, brush=(0,0,255,150), clear=True) 
        
        self.left_bound_line = self.w6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))
        self.right_bound_line = self.w6.addLine(x=np.max(self.squared_SLDs), pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))
        
        return


    def fit_exp_dec_1(self):
        if self.exp_dec_1_curve is not None:
            self.w6.removeItem(self.exp_dec_1_curve)
            self.cdf_legend.removeItem(self.exp_dec_1_curve.name())

        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
               
        xdata = self.cdf_x
        ydata = self.cdf_y 

        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]

        popt, pcov = curve_fit(exp_dec, xfit, ydata[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
        tau_fit = popt[1]
        D_fit = self.tau_to_D(tau_fit)
        print('D = {0:.4g} um^2 s^-1'.format(D_fit))
        yfit = exp_dec(xfit, *popt)
        self.exp_dec_1_curve = self.w6.plot(xfit, yfit, pen='g', name=' Fit. D = {0:.4g} um^2 s^-1'.format(D_fit))
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))


    def fit_exp_dec_2(self):
        if self.exp_dec_2_curve is not None:
            self.w6.removeItem(self.exp_dec_2_curve)
            self.cdf_legend.removeItem(self.exp_dec_2_curve.name())

        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])

        xdata = self.cdf_x
        ydata = self.cdf_y 

        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]
        
        popt, pcov = curve_fit(exp_dec_2, xfit, ydata[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))
        A1 = popt[0]
        A2 = -1 - A1
        tau1_fit = popt[1]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[2]
        D2_fit = self.tau_to_D(tau2_fit)
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2. A1={2:.2g} A2={3:.2g}'.format(D1_fit, D2_fit, A1, A2)
        print(msg)
        yfit = exp_dec_2(xfit, *popt)
        self.exp_dec_2_curve = self.w6.plot(xfit, yfit, pen='r', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_3(self):
        if self.exp_dec_3_curve is not None:
            self.w6.removeItem(self.exp_dec_3_curve)
            self.cdf_legend.removeItem(self.exp_dec_3_curve.name())

        left_bound =  np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
            
        xdata = self.cdf_x
        ydata = self.cdf_y 
        
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]        
        
        popt, pcov = curve_fit(exp_dec_3, xfit, ydata[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))
        A1 = popt[0]
        A2 = popt[1]
        A3 = -1 - A1 - A2
        tau1_fit = popt[2]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[3]
        D2_fit = self.tau_to_D(tau2_fit)
        tau3_fit = popt[4]
        D3_fit = self.tau_to_D(tau3_fit)
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2, D3 = {2:.4g} um2/2. A1={3:.2g} A2={4:.2g}, A3={5:.2g}'.format(D1_fit, D2_fit, D3_fit, A1, A2, A3)
        print(msg)
        yfit = exp_dec_3(xfit, *popt)
        self.exp_dec_3_curve = self.w6.plot(xfit, yfit, pen='y', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def tau_to_D(self, tau):
        """ 
        tau = 4Dt
        tau is decay constant of exponential fit
        D is diffusion coefficient
        t is duration of one lag (exposure time) in seconds
        """
        t = (self.mainGUI.frameLength_selector.value()/1000) * self.nlags
        D = tau / (4 * t)
        return D
    
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()


class TrackWindow(BaseProcess):
    def __init__(self, mainGUI):
        super().__init__()
        
        self.mainGUI = mainGUI
        
        #setup window
        self.win = pg.GraphicsWindow()
        self.win.resize(600, 800)
        self.win.setWindowTitle('Track Display - press "t" to add track')
        
        #add widgets
        self.label = pg.LabelItem(justify='center')
        self.win.addItem(self.label)
        self.win.nextRow()
        
        self.plt1 = self.win.addPlot(title='intensity')
        self.plt1.getAxis('left').enableAutoSIPrefix(False)         
        

        self.plt3 = self.win.addPlot(title='track')  
        self.plt3.setAspectLocked()
        self.plt3.showGrid(x=True, y=True)
        self.plt3.setXRange(-5,5)
        self.plt3.setYRange(-5,5)
        self.plt3.getViewBox().invertY(True)
        
        self.win.nextRow()
        
        self.plt2 = self.win.addPlot(title='distance from origin')
        self.plt2.getAxis('left').enableAutoSIPrefix(False)        

        self.plt4 = self.win.addPlot(title='polar (direction and velocity)')  
        self.plt4.getViewBox().invertY(True)
        self.plt4.setAspectLocked()
        self.plt4.setXRange(-4,4)
        self.plt4.setYRange(-4,4)
        self.plt4.hideAxis('bottom')
        self.plt4.hideAxis('left')

        self.win.nextRow()
 
        self.plt5 = self.win.addPlot(title='instantaneous velocity')
        self.plt5.getAxis('left').enableAutoSIPrefix(False)
        
        self.plt6 = self.win.addPlot(title='direction relative to origin')
        self.plt6.getAxis('left').enableAutoSIPrefix(False)       

        self.win.nextRow()       
 
       
        #add plot labels
        self.plt1.setLabel('left', 'Intensity', units ='Arbitary')
        self.plt1.setLabel('bottom', 'Time', units ='Frames')        

        self.plt2.setLabel('left', 'Distance', units ='pixels')
        self.plt2.setLabel('bottom', 'Time', units ='Frames') 
        
        self.plt3.setLabel('left', 'y', units ='pixels')
        self.plt3.setLabel('bottom', 'x', units ='pixels') 
        
        self.plt5.setLabel('left', 'velocity', units ='pixels/frame')
        self.plt5.setLabel('bottom', 'Time', units ='Frames')      
        
        self.plt6.setLabel('left', 'direction moved', units ='degrees')
        self.plt6.setLabel('bottom', 'Time', units ='Frames')       
        
        
        self.win.nextRow()
        
        self.optionsPanel = QGraphicsProxyWidget()                  
        self.positionIndicator_button = QPushButton('Show position info')
        self.positionIndicator_button.pressed.connect(self.togglePoistionIndicator)        
        self.optionsPanel.setWidget(self.positionIndicator_button)
        
        self.win.addItem(self.optionsPanel)
             
        self.showPositionIndicators = False
        self.plotsInitiated = False  
        
        self.r = None

    def update(self, time, intensity, distance, zeroed_X, zeroed_Y, dydt, direction, velocity, ID):  
        ##Update track ID
        self.label.setText("<span style='font-size: 16pt'>track ID={}".format(ID))        
        #update intensity plot
        self.plt1.plot(time, intensity, stepMode=False, brush=(0,0,255,150), clear=True) 
        #update distance plot        
        self.plt2.plot(time, distance, stepMode=False, brush=(0,0,255,150), clear=True)
        #update position relative to 0 plot          
        self.plt3.plot(zeroed_X, zeroed_Y, stepMode=False, brush=(0,0,255,150), clear=True) 
        
        #update polar
        self.updatePolarPlot(direction,velocity)
        
        #update dydt
        self.plt5.plot(time, velocity, stepMode=False, brush=(0,0,255,150), clear=True)
        #update drection
        self.plt6.plot(time, direction, stepMode=False, brush=(0,0,255,150), clear=True)       

        
                
        # if self.autoscaleX:
        #     self.plt1.setXRange(np.min(x),np.max(x),padding=0)
            
        # if self.autoscaleY:
        #     self.plt1.setYRange(np.min(y),np.max(y),padding=0)

        if self.showPositionIndicators:
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)           
                        
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)   
        
        keys = time
        values = zip(zeroed_X, zeroed_Y)
        self.data = dict(zip(keys,values))
        self.r = None


    def updatePolarPlot(self, direction,velocity):
        self.plt4.clear()
        
        # Add polar grid lines
        self.plt4.addLine(x=0, pen=1)
        self.plt4.addLine(y=0, pen=1)
        for r in range(10, 50, 10):
            r = r/10
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen('w', width=0.5))
            self.plt4.addItem(circle)
        
        theta = np.radians(direction)
        radius = velocity
                
        # Transform to cartesian and self.plt4
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
                
        for i in range(len(x)):        
            path = QPainterPath(QPointF(0,0))
            path.lineTo(QPointF(x[i],y[i]))            
            item = pg.QtGui.QGraphicsPathItem(path)
            item.setPen(pg.mkPen('r', width=5))                
            self.plt4.addItem(item) 

        #position label
        labels = [0,90,180,270]
        d = 6
        pos = [ (d,0),(0,d),(-d,0),(0,-d) ]
        for i,label in enumerate(labels):
            text = pg.TextItem(str(label), color=(200,200,0))
            self.plt4.addItem(text)
            text.setPos(pos[i][0],pos[i][1])
        
        # scale
        for r in range(10, 50, 10):
            r = r/10
            text = pg.TextItem(str(r))
            self.plt4.addItem(text)
            text.setPos(0,r)

        return


    def togglePoistionIndicator(self):
        if self.showPositionIndicators == False:
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)  
 
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False) 
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False) 
            
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)  
            self.showPositionIndicators = True
            self.positionIndicator_button.setText("Hide position info")
            
        else:
            self.plt1.removeItem(self.plt1_line)
            self.plt2.removeItem(self.plt2_line) 
            self.plt5.removeItem(self.plt5_line)
            self.plt6.removeItem(self.plt6_line)             
            
            self.mainGUI.plotWindow.sigTimeChanged.disconnect(self.updatePositionIndicators)  
            self.showPositionIndicators = False
            self.positionIndicator_button.setText("Show position info")            
        

    def updatePositionIndicators(self, t):
        #print('x axis range: {}'.format(self.plt1_axis.range))
        self.plt1_line.setPos(t)
        self.plt2_line.setPos(t)
        self.plt5_line.setPos(t)
        self.plt6_line.setPos(t)        
        
        
        #update scatter plot position indicator
        if self.r != None:
            self.plt3.removeItem(self.r)
            
        if t in self.data:    
            self.r = pg.RectROI((self.data[t][0]-0.25,self.data[t][1]-0.25), size = pg.Point(0.5,0.5), movable=False,rotatable=False,resizable=False, pen=pg.mkPen('r',width=1))
            self.r.handlePen = None
            self.plt3.addItem(self.r)
            
                            
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()

class ChartDock():
    def __init__(self, mainGUI):
        super().__init__()    
        
        self.mainGUI = mainGUI
        
        self.win =QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1000,500)
        self.win.setWindowTitle('Plots')
        
        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("plot options", size=(500, 100))
        self.d2 = Dock("plot", size=(500,400))
        self.d3 = Dock("histogram options", size=(500,100))
        self.d4 = Dock("histogram", size=(500,400))
        
        self.area.addDock(self.d1, 'left') 
        self.area.addDock(self.d3, 'right', self.d1)       
        self.area.addDock(self.d2, 'bottom', self.d1)     

        self.area.addDock(self.d4, 'bottom', self.d3)     
    
        #### SCATTER PLOT
        self.w1 = pg.LayoutWidget()
        
        self.pointOrTrackData_selector_plot = pg.ComboBox()
        self.plotDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}
        self.pointOrTrackData_selector_plot.setItems(self.plotDataChoice)  

        self.xlabel = QLabel("x:")  
        self.ylabel = QLabel("y:")  
    
        self.xColSelector = pg.ComboBox()
        self.xcols = {'None':'None'}
        self.xColSelector.setItems(self.xcols)
    
        self.yColSelector = pg.ComboBox()
        self.ycols = {'None':'None'}
        self.yColSelector.setItems(self.ycols)    

        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes= {'scatter':'scatter', 'line':'line'}
        self.plotTypeSelector.setItems(self.plotTypes)  
        self.selectorLabel = QLabel("Plot type")  


        self.pointSize_selector = pg.SpinBox(value=7, int=True)
        self.pointSize_selector.setSingleStep(1)       
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10) 
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size") 
        
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)
        
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
        
        self.d1.addWidget(self.w1)    
        
        #### HISTOGRAM
        self.w2 = pg.LayoutWidget()
        
        self.pointOrTrackData_selector_histo = pg.ComboBox()
        self.histoDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}
        self.pointOrTrackData_selector_histo.setItems(self.histoDataChoice)  
    
        self.colSelector = pg.ComboBox()
        self.cols = {'None':'None'}
        self.colSelector.setItems(self.cols)
        
        self.collabel = QLabel("col:") 
        
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)

        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(1)       
        self.histoBin_selector.setMinimum(1)
        self.histoBin_selector.setMaximum(100000) 
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        
        self.histoBin_label = QLabel('# of bins')
        
        self.cumulativeTick_label = QLabel('cumulative')  
        self.cumulativeTick = CheckBox() 
        self.cumulativeTick.setChecked(False) 
        
        self.w2.addWidget(self.pointOrTrackData_selector_histo , row=0, col=1)
        self.w2.addWidget(self.colSelector, row=1, col=1)
        self.w2.addWidget(self.collabel, row=1, col=0)  
        self.w2.addWidget(self.histoBin_selector, row=2, col=1)
        self.w2.addWidget(self.histoBin_label, row=2, col=0)    
        self.w2.addWidget(self.cumulativeTick_label, row=3, col=0)
        self.w2.addWidget(self.cumulativeTick, row=3, col=1)         
        self.w2.addWidget(self.histo_button, row=4, col=1)         
        
        self.d3.addWidget(self.w2)      
    
        self.w3 = pg.PlotWidget(title="plot")
        self.w3.plot()
        self.w3.setLabel('left', 'y-axis', units ='')
        self.w3.setLabel('bottom', 'x-axis', units ='')  
        self.d2.addWidget(self.w3)    
    
        self.w4 = pg.PlotWidget(title="histogram")
        self.w4.plot()
        self.w4.setLabel('left', '# of observations', units ='')
        self.w4.setLabel('bottom', 'value', units ='')          
        
        self.d4.addWidget(self.w4)      

    def updatePlot(self):
        self.w3.clear()

        if self.pointOrTrackData_selector_plot.value() == 'Point Data':
        
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.data[self.yColSelector.value()].to_numpy() 
            else:
                x = self.mainGUI.filteredData[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.filteredData[self.yColSelector.value()].to_numpy()             

        else:
            
            if self.mainGUI.useFilteredData == False:                
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()                
            else:                
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean()
                
            x = plotDF[self.xColSelector.value()].to_numpy()
            y = plotDF[self.yColSelector.value()].to_numpy() 


        if self.plotTypeSelector.value() == 'line':
            self.w3.plot(x, y, stepMode=False, brush=(0,0,255,150), clear=True) 
        elif self.plotTypeSelector.value() == 'scatter':
            self.w3.plot(x, y,
                         pen = None,
                         symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),                                      
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),
                         symbolSize=self.pointSize_selector.value())    
            
        
        self.w3.setLabel('left', self.yColSelector.value(), units = None)
        self.w3.setLabel('bottom', self.xColSelector.value(), units = None)             
            
        return
    
    def updateHisto(self):
        self.w4.clear()

        if self.pointOrTrackData_selector_histo.value() == 'Point Data':
             
            if self.mainGUI.useFilteredData == False:
                vals = self.mainGUI.data[self.colSelector.value()]
            else:
                vals = self.mainGUI.filteredData[self.colSelector.value()]         

        else:
            if self.mainGUI.useFilteredData == False:                
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()                
            else:                
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean() 
                
            vals = plotDF[self.colSelector.value()] 


        start=0
        end=np.max(vals)
        n=self.histoBin_selector.value()
        
        if self.cumulativeTick.isChecked():
            count,bins_count = np.histogram(vals, bins=np.linspace(start, end, n)) 
            pdf = count / sum(count)
            y = np.cumsum(pdf)        
            x = bins_count[1:]  
            self.w4.plot(x, y, brush=(0,0,255,150), clear=True) 
        
        else:
            y,x = np.histogram(vals, bins=np.linspace(start, end, n))            
            self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True) 
        
        self.w4.setLabel('bottom', self.colSelector.value(), units = None)     
        return

    
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()
    
    

class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    plots loc and track data onto current window
    """
    def __init__(self):
        if g.settings['locsAndTracksPlotter'] is None or 'set_track_colour' not in g.settings['locsAndTracksPlotter']:
            s = dict()            
            s['filename'] = '' 
            s['filetype'] = 'flika'   
            s['pixelSize'] = 108                         
            s['set_track_colour'] = False
            g.settings['locsAndTracksPlotter'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
        

    def __call__(self, filename, filetype, pixelSize, set_track_colour,  keepSourceWindow=False):
        '''
        '''

        #currently not saving parameter changes on call
        g.settings['locsAndTracksPlotter']['filename'] = filename 
        g.settings['locsAndTracksPlotter']['filetype'] = filetype       
        g.settings['locsAndTracksPlotter']['pixelSize'] = pixelSize 
        g.settings['locsAndTracksPlotter']['set_track_colour'] = set_track_colour      
        
        g.m.statusBar().showMessage("plotting data...")
        return


    def closeEvent(self, event):
        self.clearPlots()
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):      
        self.filename = '' 
        self.filetype = 'flika'   
        self.pixelSize= 108  
        self.plotWindow = None
        self.pathitems = []
        self.useFilteredData = False
        self.useFilteredTracks = False
        
        #self.filteredTrackIds = None
        
        self.useMatplotCM = False
        
        self.selectedTrack = None
        self.displayTrack = None
        
        self.chartWindow = None
        self.displayCharts = False
        
        self.diffusionWindow = None
        self.displayDiffusionPlot = False        
        
        #initiate track plot
        self.trackWindow = TrackWindow(self)
        self.trackWindow.hide()
        
        #initiate flower plot
        self.flowerPlotWindow = FlowerPlotWindow(self)
        self.flowerPlotWindow.hide()  
        
        
        self.gui_reset()        
        s=g.settings['locsAndTracksPlotter']  
        
        #buttons      
        self.plotPointData_button = QPushButton('Plot Points')
        self.plotPointData_button.pressed.connect(self.plotPointData)  
        
        self.hidePointData_button = QPushButton('Toggle Points')
        self.hidePointData_button.pressed.connect(self.hidePointData)         
            
        self.plotTrackData_button = QPushButton('Plot Tracks')
        self.plotTrackData_button.pressed.connect(self.plotTrackData)  
        
        self.clearTrackData_button = QPushButton('Clear Tracks')
        self.clearTrackData_button.pressed.connect(self.clearTracks)  
        
        self.filterData_button = QPushButton('Filter')
        self.filterData_button.pressed.connect(self.filterData)          
        
        self.clearFilterData_button = QPushButton('Clear Filter')
        self.clearFilterData_button.pressed.connect(self.clearFilterData)  

        self.ROIFilterData_button = QPushButton(' Filter by ROI(s)')
        self.ROIFilterData_button.pressed.connect(self.ROIFilterData)  

        self.clearROIFilterData_button = QPushButton('Clear ROI Filter')
        self.clearROIFilterData_button.pressed.connect(self.clearROIFilterData)  
        
        self.saveData_button = QPushButton('Save Tracks')
        self.saveData_button.pressed.connect(self.saveData)    
        
        self.showCharts_button = QPushButton('Show Charts')
        self.showCharts_button.pressed.connect(self.toggleCharts)    
        
        self.showDiffusion_button = QPushButton('Show Diffusion')
        self.showDiffusion_button.pressed.connect(self.toggleDiffusionPlot) 

                         
        #checkbox
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(s['set_track_colour'])
        
        self.matplotCM_checkbox = CheckBox() 
        self.matplotCM_checkbox.stateChanged.connect(self.setColourMap)
        self.matplotCM_checkbox.setChecked(False)  

        self.displayFlowPlot_checkbox = CheckBox() 
        self.displayFlowPlot_checkbox.stateChanged.connect(self.toggleFlowerPlot)
        self.displayFlowPlot_checkbox.setChecked(True)  

        #comboboxes
        self.filetype_Box = pg.ComboBox()
        filetypes = {'flika' : 'flika', 'thunderstorm':'thunderstorm', 'xy':'xy'}
        self.filetype_Box.setItems(filetypes)

        self.xCol_Box = pg.ComboBox()
        self.xcols = {'None':'None'}
        self.xCol_Box.setItems(self.xcols)

        self.yCol_Box = pg.ComboBox()
        self.ycols = {'None':'None'}
        self.yCol_Box.setItems(self.ycols)
        
        self.frameCol_Box = pg.ComboBox()
        self.framecols = {'None':'None'}
        self.frameCol_Box.setItems(self.framecols)        
 
        self.trackCol_Box = pg.ComboBox()
        self.trackcols = {'None':'None'}
        self.trackCol_Box.setItems(self.trackcols)   
        
        self.filterCol_Box = pg.ComboBox()
        self.filtercols = {'None':'None'}
        self.filterCol_Box.setItems(self.filtercols)  

        self.trackColourCol_Box = pg.ComboBox()
        self.trackcolourcols = {'None':'None'}
        self.trackColourCol_Box.setItems(self.trackcolourcols)  
        
        self.colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.colourMap_Box.setItems(self.colourMaps)         

        self.filterOp_Box = pg.ComboBox()
        self.filterOps = {'=':'==', '<':'<', '>':'>'}
        self.filterOp_Box.setItems(self.filterOps)  
        
        self.filterValue_Box = QLineEdit()     
        
        self.trackDefaultColour_Box = pg.ComboBox()
        self.trackdefaultcolours = {'green': Qt.green, 'red': Qt.red, 'blue': Qt.blue}
        self.trackDefaultColour_Box.setItems(self.trackdefaultcolours)

        #spinbox
        self.frameLength_selector = pg.SpinBox(value=10, int=True)
        self.frameLength_selector.setSingleStep(10)       #########TODDO! FIX THIS - AND INDEXING PROBLEM IN CALL  BY ADDING OBJECT NAMES ABOVE
        self.frameLength_selector.setMinimum(1)
        self.frameLength_selector.setMaximum(100000) 
        
        
        #data file selector
        self.getFile = FileSelector(filetypes='*.csv')
        
        #connections
        self.getFile.valueChanged.connect(self.loadData)
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        #self.items.append({'name': 'blank1 ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'filename ', 'string': 'LOAD  -----------------------', 'object': self.getFile})    
        self.items.append({'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box})  
        self.items.append({'name': 'frameLength', 'string': 'milliseconds per frame', 'object': self.frameLength_selector})          
       
        #self.items.append({'name': 'blank ', 'string': '-------------------------------------------', 'object': None})   
        #self.items.append({'name': 'frameCol', 'string': 'Frame col', 'object': self.frameCol_Box})          
        #self.items.append({'name': 'xCol', 'string': 'X col', 'object': self.xCol_Box})          
        #self.items.append({'name': 'yCol', 'string': 'Y col', 'object': self.yCol_Box})   
        #self.items.append({'name': 'trackCol', 'string': 'Track col', 'object': self.trackCol_Box})   
        #self.items.append({'name': 'blank ', 'string': '---- FILTER -----', 'object': None})                  
        self.items.append({'name': 'filterCol', 'string': 'FILTER    -------- Filter col', 'object': self.filterCol_Box})
        self.items.append({'name': 'filterOp', 'string': 'Operator', 'object': self.filterOp_Box})  
        self.items.append({'name': 'filterValue', 'string': 'Value', 'object': self.filterValue_Box})         
        self.items.append({'name': 'filterData', 'string': '', 'object': self.filterData_button })         
        self.items.append({'name': 'clearFilterData', 'string': '', 'object': self.clearFilterData_button })  

        #self.items.append({'name': 'blank ', 'string': '--- ROI FILTER ----', 'object': None})                  
        self.items.append({'name': 'filterROI', 'string': '', 'object': self.ROIFilterData_button})
        self.items.append({'name': 'clearFilterROI', 'string': '', 'object': self.clearROIFilterData_button})  
        
        
        #self.items.append({'name': 'blank ', 'string': '----  PLOT  -----', 'object': None})           
        #self.items.append({'name': 'plotPoints', 'string': '', 'object': self.plotPointData_button }) 
        self.items.append({'name': 'hidePoints', 'string': 'PLOT    --------------------', 'object': self.hidePointData_button })
        self.items.append({'name': 'trackDefaultColour', 'string': 'Track Default Colour', 'object': self.trackDefaultColour_Box })        
        self.items.append({'name': 'trackColour', 'string': 'Set Track Colour', 'object': self.trackColour_checkbox})           
        self.items.append({'name': 'trackColourCol', 'string': 'Colour by', 'object': self.trackColourCol_Box})
        self.items.append({'name': 'trackColourMap', 'string': 'Colour Map', 'object': self.colourMap_Box})   
        self.items.append({'name': 'matplotClourMap', 'string': 'Use matplot map', 'object': self.matplotCM_checkbox}) 
        self.items.append({'name': 'displayFlowerPlot', 'string': 'Flower Plot', 'object': self.displayFlowPlot_checkbox})         
        self.items.append({'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button })         
        self.items.append({'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button })     
        self.items.append({'name': 'saveTracks', 'string': '', 'object': self.saveData_button })  
        self.items.append({'name': 'showCharts', 'string': '', 'object': self.showCharts_button })
        self.items.append({'name': 'showDiffusion', 'string': '', 'object': self.showDiffusion_button })          
        
        super().gui()
        ######################################################################
        return


    def loadData(self):
        self.filename = self.getFile.value()
        self.data = pd.read_csv(self.filename)
        
        print('-------------------------------------')
        print('Data loaded (first 5 rows displayed):')
        print(self.data.head())
        print('-------------------------------------')
        
        self.columns = self.data.columns
        self.colDict= dictFromList(self.columns)

        
        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)
        self.frameCol_Box.setItems(self.colDict)        
        self.trackCol_Box.setItems(self.colDict)   
        self.filterCol_Box.setItems(self.colDict)  
        self.trackColourCol_Box.setItems(self.colDict)  
        
        #format points add to image window
        self.plotPointData()
        

    def makePointDataDF(self, data):   
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.pixelSize
            df['y'] = data['y [nm]']/self.pixelSize   

        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']

        return df


    def plotPointsOnStack(self):
            
        points_byFrame = self.points[['frame','x','y']]
    
        #points_byFrame['point_color'] = QColor(g.m.settings['point_color'])
        #points_byFrame['point_size'] = g.m.settings['point_size']
        pointArray = points_byFrame.to_numpy()
        
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        
        
        for pt in pointArray:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(g.m.settings['point_color'])
            #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
            position = [pt[1], pt[2], pointColor, pointSize]    
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()
        

    def hidePointData(self):
        if self.plotWindow.scatterPlot in self.plotWindow.imageview.ui.graphicsView.items():
            self.plotWindow.imageview.ui.graphicsView.removeItem(self.plotWindow.scatterPlot)
        else:
            self.plotWindow.imageview.addItem(self.plotWindow.scatterPlot)

    def plotPointData(self):
        ### plot point data to current window
        if self.useFilteredData == False:
            self.points = self.makePointDataDF(self.data)
        else:
            self.points = self.makePointDataDF(self.filteredData)
        self.plotWindow = g.win
        self.plotPointsOnStack()
        

        g.m.statusBar().showMessage('point data plotted to current window') 
        print('point data plotted to current window')    
        return



    def makeTrackDF(self, data):
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.pixelSize
            df['y'] = data['y [nm]']/self.pixelSize  
            df['track_number'] = data['track_number']

        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']
            df['track_number'] = data['track_number']
            
            df['zeroed_X'] = data['zeroed_X']
            df['zeroed_Y'] = data['zeroed_Y']            
            
            
            if self.trackColour_checkbox.isChecked():
                if self.useMatplotCM:
                    cm = pg.colormap.getFromMatplotlib(self.colourMap_Box.value()) #cm goes from 0-1, need to scale input values   
                else:    
                    cm = pg.colormap.get(self.colourMap_Box.value()) #cm goes from 0-1, need to scale input values
                
                df['colour'] = cm.mapToQColor(data[self.trackColourCol_Box.value()].to_numpy()/max(data[self.trackColourCol_Box.value()]))
        
                     
        return df.groupby(['track_number'])


    def clearTracks(self):
        if self.plotWindow is not None and not self.plotWindow.closed:
            for pathitem in self.pathitems:
                self.plotWindow.imageview.view.removeItem(pathitem)
        self.pathitems = []        

    def showTracks(self):
        '''Updates track paths in main view and Flower Plot'''
        # clear self.pathitems
        self.clearTracks()
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.clearTracks()
        
        #setup pens
        pen = QPen(self.trackDefaultColour_Box.value(), .4)
        pen.setCosmetic(True)
        pen.setWidth(2)
        
        pen_FP = QPen(self.trackDefaultColour_Box.value(), .4)
        pen_FP.setCosmetic(True)
        pen_FP.setWidth(1)       
        
        if self.useFilteredTracks:
            trackIDs = self.filteredTrackIds
            
        else:
            trackIDs = self.trackIDs

        print('tracks to plot {}'.format(trackIDs))
        
        for track_idx in trackIDs:
            tracks = self.tracks.get_group(track_idx)
            pathitem = QGraphicsPathItem(self.plotWindow.imageview.view)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP = QGraphicsPathItem(self.flowerPlotWindow.plt)            
            
            if self.trackColour_checkbox.isChecked():
                #print(tracks['colour'].to_list()[0].rgb())
                pen.setColor(tracks['colour'].to_list()[0])
                pen_FP.setColor(tracks['colour'].to_list()[0])                
            
            pathitem.setPen(pen)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPen(pen_FP)
            
            self.plotWindow.imageview.view.addItem(pathitem)
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.plt.addItem(pathitem_FP)
            
            self.pathitems.append(pathitem)
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.pathitems.append(pathitem_FP)
            
            x = tracks['x'].to_numpy()
            y = tracks['y'].to_numpy() 

            if self.displayFlowPlot_checkbox.isChecked():
                zeroed_X = tracks['zeroed_X'].to_numpy()
                zeroed_Y = tracks['zeroed_Y'].to_numpy()            
            
            #x = pts[:, 1]+.5; y = pts[:,2]+.5
            path = QPainterPath(QPointF(x[0],y[0]))
            if self.displayFlowPlot_checkbox.isChecked():
                path_FP = QPainterPath(QPointF(zeroed_X[0],zeroed_Y[0]))
            for i in np.arange(1, len(x)):
                path.lineTo(QPointF(x[i],y[i]))
                if self.displayFlowPlot_checkbox.isChecked():
                    path_FP.lineTo(QPointF(zeroed_X[i],zeroed_Y[i]))                
            
            pathitem.setPath(path)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPath(path_FP)            


    def plotTrackData(self):
        ### plot track data to current window
        self.plotWindow = g.win
        
        if self.useFilteredData == False:            
            self.trackIDs = np.unique(self.data['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.data)
        else:
            self.trackIDs = np.unique(self.filteredData['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.filteredData)           
        
        self.showTracks()
        
        #get mouse events from plot window
        self.plotWindow.imageview.scene.sigMouseMoved.connect(self.updateTrackSelector)
        
        #use key press to select tracks to display
        self.plotWindow.keyPressSignal.connect(self.selectTrack)
        
        
        #display track window with plots for individual tracks
        self.trackWindow.show()
        
        #display flower plot with all tracks origins set to 0,0
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()
        
        g.m.statusBar().showMessage('track data plotted to current window') 
        print('track data plotted to current window')    
        return


    def updateTrackSelector(self, point):
        pos =  self.plotWindow.imageview.getImageItem().mapFromScene(point)

        #print('x: {}, y: {}'.format(pos.x(),pos.y()))
                
        for i, path in enumerate(self.pathitems):
            if path.contains(pos):
                #print('mouse at {}'.format(pos))
                #print('track ID:  {}'.format(self.trackIDs[i]))
                #print('track pos {}{}'.format(path.pos().x(),path.pos().y()))
                self.selectedTrack = self.trackIDs[i]                 
                

    def selectTrack(self,ev):
        if ev.key() == Qt.Key_T:
            if self.selectedTrack != self.displayTrack:
                self.displayTrack = self.selectedTrack    
                #print(self.selectedTrack)
                
                #get track data for plots
                trackData = self.data[self.data['track_number'] == int(self.displayTrack)]
                frame = trackData['frame'].to_numpy()
                intensity = trackData['intensity'].to_numpy() 
                distance = trackData['distanceFromOrigin'].to_numpy() 
                zeroed_X = trackData['zeroed_X'].to_numpy()
                zeroed_Y = trackData['zeroed_Y'].to_numpy()  

                dydt =  trackData['dy-dt: distance'].to_numpy() 
                direction = trackData['direction_Relative_To_Origin'].to_numpy()    
                velocity =  trackData['velocity'].to_numpy()                    
                
                #update plots in track display               
                self.trackWindow.update(frame, intensity, distance, zeroed_X, zeroed_Y, dydt, direction, velocity, self.displayTrack)
                

        if ev.key() == Qt.Key_R:
                       
            roiFilterPoints = []
            roi = self.plotWindow.currentROI
            
            currentFrame = self.plotWindow.currentIndex
            
            for i in range(0,self.plotWindow.mt):
                # get ROI shape in coordinate system of the scatter plot
                self.plotWindow.setIndex(i)
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                roiFilterPoints.extend((selected))
            
            self.plotWindow.setIndex(currentFrame)
             
            trackIDs = []
            
            #flat_ptList = [pt for sublist in roiFilterPoints for pt in sublist]
            
            for pt in roiFilterPoints:            
                #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))    
                ptFilterDF = self.data[(self.data['x']==pt[1]) & (self.data['y']==pt[2])]                
                trackIDs.extend(ptFilterDF['track_number'])
               
            selectedTracks = np.unique(trackIDs)    
            self.joinedData = self.data[self.data['track_number'].isin(selectedTracks)]
            print(selectedTracks)
            g.m.statusBar().showMessage('Track join complete') 
            


    def filterData(self):
        
        op = self.filterOp_Box.value()
        filterCol = self.filterCol_Box.value()
        dtype = self.data[filterCol].dtype 
        value = float(self.filterValue_Box.text())
        
        
        if op == '==':
            self.filteredData = self.data[self.data[filterCol] == value]
 
        elif op == '<':
            self.filteredData = self.data[self.data[filterCol] < value]
        
        elif op == '>':
             self.filteredData = self.data[self.data[filterCol] > value]           
            
        
        print(self.filteredData.head())
        g.m.statusBar().showMessage('filter complete') 
        self.useFilteredData = True
        
        #update point data plot
        self.plotPointData()
        
        return


    def clearFilterData(self):
        self.useFilteredData = False
        
        self.plotPointData()
        return

    def getScatterPointsAsQPoints(self):
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        trackIDs = []
        
        flat_ptList = [pt for sublist in self.plotWindow.scatterPoints for pt in sublist]
        
        for pt in flat_ptList:            
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.data[(self.data['x']==pt[0]) & (self.data['y']==pt[1])]
            
            trackIDs.extend(ptFilterDF['track_number'])

        
        self.filteredTrackIds = np.unique(trackIDs)

        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        
        #self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        
        #self.filteredTrackIds = np.unique(self.filteredData['track_number'])

        self.useFilteredData = True
        self.useFilteredTracks = True
        

    def ROIFilterData(self):
        self.roiFilterPoints = []
        self.rois = self.plotWindow.rois
        
        self.oldScatterPoints = self.plotWindow.scatterPoints
        
        for roi in self.rois:
            currentFrame = self.plotWindow.currentIndex
            for i in range(0,self.plotWindow.mt):
                # get ROI shape in coordinate system of the scatter plot
                self.plotWindow.setIndex(i)
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                self.roiFilterPoints.extend((selected))
            self.plotWindow.setIndex(currentFrame)
        
        
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        
        
        for pt in self.roiFilterPoints:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(0,255,0)
            #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
            position = [pt[1], pt[2], pointColor, pointSize]  
            #print(position)
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()

        self.getDataFromScatterPoints()
        
        g.m.statusBar().showMessage('ROI filter complete') 
        
        return

    def clearROIFilterData(self):
        self.plotWindow.scatterPoints = self.oldScatterPoints 
        self.plotWindow.updateindex()
        self.useFilteredData = False
        self.useFilteredTracks = False
        return
    

    def setColourMap(self):
        if self.matplotCM_checkbox.isChecked():
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
            self.colourMap_Box.setItems(self.colourMaps)  
            self.useMatplotCM = True
        else:
            self.colourMaps = dictFromList(pg.colormap.listMaps())
            self.colourMap_Box.setItems(self.colourMaps) 
            self.useMatplotCM = False
            

    def toggleFlowerPlot(self):
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()
        else:
            self.flowerPlotWindow.hide()            

    def toggleCharts(self):
        if self.chartWindow == None:
            #create chart plot window
            self.chartWindow = ChartDock(self)
            self.chartWindow.xColSelector.setItems(self.colDict)
            self.chartWindow.yColSelector.setItems(self.colDict)
            self.chartWindow.colSelector.setItems(self.colDict)    
            
            self.chartWindow.xcols = self.colDict
            self.chartWindow.ycols = self.colDict            
            self.chartWindow.cols = self.colDict
            
            
        if self.displayCharts == False:
            self.chartWindow.show()
            self.displayCharts = True
            self.showCharts_button.setText('Hide Charts')
        else:
            self.chartWindow.hide()
            self.displayCharts = False   
            self.showCharts_button.setText('Show Charts')

    def toggleDiffusionPlot(self):
        if self.diffusionWindow == None:
            #create diffusion plot window
            self.diffusionWindow = DiffusionPlotWindow(self)   
                        
        if self.displayDiffusionPlot == False:
            self.diffusionWindow.show()
            self.displayDiffusionPlot = True
            self.showDiffusion_button.setText('Hide Diffusion')
        else:
            self.chartWindow.hide()
            self.displayDiffusionPlot = False   
            self.showDiffusion_button.setText('Show Diffusion')


    def createStatsDFs(self):
            self.meanDF = self.data.groupby('track_number', as_index=False).mean()
            self.stdDF = self.data.groupby('track_number', as_index=False).std()        

    def createStatsDFs_filtered(self):
            self.meanDF_filtered = self.filteredData.groupby('track_number', as_index=False).mean()
            self.stdDF_filtered = self.filteredData.groupby('track_number', as_index=False).std() 


    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return

    def saveData(self):      
        if self.useFilteredData == False:
            print('filter data first')
            g.alert('Filter data first')
            return
        
        #set export path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")        

        #write file
        try:
            # writing the data into the file 
            self.filteredData.to_csv(savePath)
            
            print('Filtered data saved to: {}'.format(savePath))
        except BaseException as e:
            print(e)
            print('Export of filtered data failed')



locsAndTracksPlotter = LocsAndTracksPlotter()
	

if __name__ == "__main__":
    pass


