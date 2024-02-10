# -*- coding: utf-8 -*-
"""
Created on Jan 29 2024

@author: george.dickinson@gmail.com

This program is a Python script developed to analyze the motion of intracellular Piezo1 proteins labeled with a fluorescent tag.
It allows the user to load raw data from a series of image files and track the movement of individual particles over time.
The script includes several data analysis and visualization tools, including the ability to filter data by various parameters, plot tracks, generate scatter and line plots, and create statistics for track speed and displacement.
Additional features include the ability to toggle between different color maps, plot diffusion maps, and save filtered data to a CSV file.

"""

# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# import necessary modules
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
import os, shutil, subprocess
import math
import sys
import glob

from scipy.ndimage import center_of_mass, gaussian_filter, binary_fill_holes, binary_closing, label, find_objects, zoom
from skimage.filters import threshold_otsu
from scipy.ndimage import rotate as nd_rotate
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale

import pandas as pd

#from pyqtgraph.canvas import Canvas, CanvasItem

from pyqtgraph import Point

from tqdm import tqdm

import numba
pg.setConfigOption('useNumba', True)

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from .io import *


def rotate_around_point(x,y, angle, origin=(0,0)):
    radians = angle * math.pi / 180
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    dx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    dy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return dx, dy


def dictFromList(lst):
    return {str(x): str(x) for x in lst}

class FolderSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Select Folder')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''

    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])

class ControlPanel():
    '''
    Controls for manipulating data
    '''
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        #self.win.resize(250, 200)
        self.win.setWindowTitle('Control Panel')

        ## Create docks
        self.d0 = Dock("Controls")
        self.d1 = Dock("Parameters")
        self.d2 = Dock("File Selector")

        self.area.addDock(self.d0)
        self.area.addDock(self.d1)
        self.area.addDock(self.d2)

        #control widget
        self.w0 = pg.LayoutWidget()

        self.leftButton = QPushButton('Left')
        self.leftButton.pressed.connect(self.moveLeft)

        self.rightButton = QPushButton('Right')
        self.rightButton.pressed.connect(self.moveRight)

        self.upButton = QPushButton('Up')
        self.upButton.pressed.connect(self.moveUp)

        self.downButton = QPushButton('Down')
        self.downButton.pressed.connect(self.moveDown)

        self.counterClockButton = QPushButton('Counter')
        self.counterClockButton.pressed.connect(self.rotateCounter)

        self.clockButton = QPushButton('Clock')
        self.clockButton.pressed.connect(self.rotateClock)

        self.saveButton = QPushButton('Save New Positions')
        self.saveButton.pressed.connect(self.save)

        #layout
        self.w0.addWidget(self.upButton, row=0,col=1)
        self.w0.addWidget(self.leftButton, row=1,col=0)
        self.w0.addWidget(self.rightButton, row=1,col=2)
        self.w0.addWidget(self.downButton, row=2,col=1)
        self.w0.addWidget(self.counterClockButton, row=3,col=0)
        self.w0.addWidget(self.clockButton, row=3,col=2)
        self.w0.addWidget(self.saveButton, row=4,col=0, colspan=3)


        #add layout widget to dock
        self.d0.addWidget(self.w0)

        #param widget
        self.w1 = pg.LayoutWidget()

        self.moveSize_label = QLabel('Step size:')
        self.moveSize = pg.SpinBox(int=True, step=1)
        self.moveSize.setValue(1)
        self.moveSize.setMinimum(1)

        self.multiply_label = QLabel('Multiply by:')
        self.multiply = pg.SpinBox(int=True, step=100)
        self.multiply.setValue(1)
        self.multiply.setMinimum(1)

        self.degrees_label = QLabel('Rotation degrees')
        self.degrees = pg.SpinBox(int=True, step=1)
        self.degrees.setValue(1)
        self.degrees.setMinimum(1)
        self.degrees.setMaximum(360)

        #layout
        self.w1.addWidget(self.moveSize_label, row=0,col=0)
        self.w1.addWidget(self.moveSize, row=0,col=1)
        self.w1.addWidget(self.multiply_label, row=1,col=0)
        self.w1.addWidget(self.multiply, row=1,col=1)
        self.w1.addWidget(self.degrees_label, row=2,col=0)
        self.w1.addWidget(self.degrees, row=2,col=1)


        #add layout widget to dock
        self.d1.addWidget(self.w1)

        #file selector widget
        self.w2 = pg.LayoutWidget()

        self.fileSelector_label = QLabel('File:')
        self.fileSelector_box = pg.ComboBox()
        self.fileList = {'None':'None'}
        self.fileSelector_box.setItems(self.fileList)
        self.fileSelector_box.activated.connect(self.update)

        #layout
        self.w2.addWidget(self.fileSelector_label, row=0,col=0)
        self.w2.addWidget(self.fileSelector_box, row=0,col=1)

        #add layout widget to dock
        self.d2.addWidget(self.w2)

    def moveLeft(self):
        shiftValue = self.moveSize.value() * self.multiply.value()
        self.mainGUI.move(shiftValue, 'l')
        return

    def moveRight(self):
        shiftValue = self.moveSize.value() * self.multiply.value()
        self.mainGUI.move(shiftValue, 'r')
        return

    def moveUp(self):
        shiftValue = self.moveSize.value() * self.multiply.value()
        self.mainGUI.move(shiftValue, 'u')
        return

    def moveDown(self):
        shiftValue = self.moveSize.value() * self.multiply.value()
        self.mainGUI.move(shiftValue, 'd')
        return

    def rotateClock(self):
        self.mainGUI.rotatePoints(-self.degrees.value())
        return

    def rotateCounter(self):
        self.mainGUI.rotatePoints(self.degrees.value())
        return

    def update(self):
        self.mainGUI.update()
        return

    def updateFileList(self, files):
        self.fileList = dictFromList(files)
        self.fileSelector_box.setItems(self.fileList)

    def save(self):
        fileName = self.mainGUI.selectedFile + '_newPos_transform.csv'
        saveName = os.path.join(self.mainGUI.foldername, fileName)
        self.mainGUI.selectedData.to_csv(saveName, index=None)
        print('new positions saved to: {}'.format(saveName))


    def show(self):
        """
        Shows the main window.
        """
        self.win.show()

    def close(self):
        """
        Closes the main window.
        """
        self.win.close()

    def hide(self):
        """
        Hides the main window.
        """
        self.win.hide()




'''
#####################################################################################################################################
######################################   Main LOCSANDTRACKSPLOTTER CLASS   ##########################################################
#####################################################################################################################################
'''

class OverlayMultipleRecordings(BaseProcess_noPriorWindow):
    """
    Generate alignment for rotation, translation and scaling of point data
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['overlayMultipleRecordings'] is None or 'heatmapBins' not in g.settings['overlayMultipleRecordings']:
            s = dict()
            #s['pixelSize'] = 108
            s['heatmapBins'] = 100
            g.settings['overlayMultipleRecordings'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, heatmapBins,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        pixelSize: int - pixel size of image data

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        #g.settings['overlayMultipleRecordings']['pixelSize'] = pixelSize
        g.settings['overlayMultipleRecordings']['heatmapBins'] = heatmapBins


        return


    def closeEvent(self, event):
        '''
        This function is called when the user closes the locs and tracks plotter window. It clears any plots that have been
        generated and calls the closeEvent function for the BaseProcess_noPriorWindow class.

        Parameters:
        event: object - object representing the close event

        Returns: None
        '''

        # Call the closeEvent function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return


    def gui(self):
        self.gui_reset()
        s=g.settings['overlayMultipleRecordings']
        self.dataWindow = WindowSelector()


        self.loadButton = QPushButton('Load Data')
        self.loadButton.pressed.connect(self.loadData)

        #data file selector
        self.getFolder = FolderSelector()


        self.heatmapButton = QPushButton('Make heatmap')
        self.heatmapButton.pressed.connect(self.plotHeatMap)

        self.heatmapBinSelector = pg.SpinBox(int=True, step=1)
        self.heatmapBinSelector.setValue(s['heatmapBins'])

        self.items.append({'name': 'dataWindow', 'string': 'Image Window', 'object': self.dataWindow})
        self.items.append({'name': 'foldername ', 'string': 'Data Folder', 'object': self.getFolder})

        self.items.append({'name': 'loadButton', 'string': '', 'object': self.loadButton})
        self.items.append({'name': 'heatmapBins', 'string': 'Number of bins in heatmap', 'object': self.heatmapBinSelector})
        self.items.append({'name': 'heatmapButton', 'string': '', 'object': self.heatmapButton})


        super().gui()

        #initialize display window and hide it
        self.controlPanel = ControlPanel(self)
        self.controlPanel.show()

        self.data = None
        self.delectedData = None

        self.foldername = None
        self.filenames = []
        self.selectedFile = None

        self.plotWindow = None

        self.pointMapScatter = None
        self.selectedPointMapScatter = None

        self.heatMap = None

        return


    def update(self):
        self.selectedFile = self.controlPanel.fileSelector_box.value()
        self.selectedData = self.data [self.data ['file'] == self.selectedFile]
        self.plotSelectedDataPoints()


    def loadData(self):
        #Check folder selected
        if self.getValue('dataWindow') == None:
            print('Select Plot Window')
            return

        #set plot window as selected window
        self.plotWindow = self.getValue('dataWindow')

        #get folders '*_transform.csv' file path using glob
        self.foldername = self.getFolder.value()
        print('folder: {}'.format(self.foldername))
        self.filenames = glob.glob(self.foldername + '/*_transform.csv', recursive = False)
        print('files loaded: {}'.format(self.filenames))
        # Load xy data from the selected files using Pandas
        for file in tqdm(self.filenames):
            tempDF = pd.read_csv(file, usecols=['x_transformed', 'y_transformed'])
            truncFileName = os.path.basename(file).split('_locsID')[0]
            tempDF['file'] = truncFileName
            self.data = pd.concat([self.data, tempDF])

        print('Data loaded')

        #plot data on image
        self.plotDataPoints()

        #update control panel file list
        self.controlPanel.updateFileList(self.data['file'].tolist())

        self.update()



    def plotDataPoints(self):
        #plot all points
        if self.plotWindow == None:
            print('First Load Data')
            return

        if self.pointMapScatter != None:
            self.plotWindow.imageview.view.removeItem(self.pointMapScatter)

        # Create a ScatterPlotItem and add it to the ImageView
        self.pointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
        self.pointMapScatter.setSize(2, update=False)
        self.pointMapScatter.setData(self.data['x_transformed'], self.data['y_transformed'])
        self.plotWindow.imageview.view.addItem(self.pointMapScatter)


    def plotSelectedDataPoints(self):
        #plot filterd points
        if self.selectedPointMapScatter != None:
            self.plotWindow.imageview.view.removeItem(self.selectedPointMapScatter)

        # Create a ScatterPlotItem and add it to the ImageView
        self.selectedPointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(255, 0, 0, 255))
        self.selectedPointMapScatter.setSize(2, update=False)
        self.selectedPointMapScatter.setData(self.selectedData['x_transformed'], self.selectedData['y_transformed'])
        self.plotWindow.imageview.view.addItem(self.selectedPointMapScatter)


    def plotHeatMap(self):
        #mkae 2D histogram
        H, yedges, xedges = np.histogram2d(self.data['x_transformed'], self.data['y_transformed'], bins=self.getValue('heatmapBins') )
        self.heatMap = H
        #plot in new window
        self.heatmapWindow = Window(self.heatMap)


    def move(self, shiftValue, direction):
        #translate data points
        if direction == 'l':
            self.selectedData['x_transformed'] = self.selectedData['x_transformed'] - shiftValue
        if direction == 'r':
            self.selectedData['x_transformed'] = self.selectedData['x_transformed'] + shiftValue
        if direction == 'u':
            self.selectedData['y_transformed'] = self.selectedData['y_transformed'] - shiftValue
        if direction == 'd':
            self.selectedData['y_transformed'] = self.selectedData['y_transformed'] + shiftValue
        #plot new positions
        self.plotSelectedDataPoints()

    def rotatePoints(self, angle):
        #get origin for filtered points
        origin = (np.mean(self.selectedData['x_transformed']),np.mean(self.selectedData['y_transformed']))
        #rotate points
        newX,newY = rotate_around_point(self.selectedData['x_transformed'], self.selectedData['y_transformed'], angle, origin=origin)
        self.selectedData['x_transformed'] = newX
        self.selectedData['y_transformed'] = newY
        #plot new positions
        self.plotSelectedDataPoints()

# Instantiate the LocsAndTracksPlotter class
overlayMultipleRecordings = OverlayMultipleRecordings()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











