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

class DisplayParams():
    '''
    Display translation parameters
    '''
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        #self.win.resize(250, 200)
        self.win.setWindowTitle('Parameters')

        ## Create docks
        self.d0 = Dock("ROI parameters")

        self.area.addDock(self.d0)


        #point options widget
        self.w0 = pg.LayoutWidget()

        self.centerPos_text = QLabel('--,--')
        self.centerPos_label = QLabel('Center:')

        self.angle_text = QLabel('--')
        self.angle_label = QLabel('Angle:')

        self.size_text = QLabel('--')
        self.size_label = QLabel('Size:')

        #layout
        self.w0.addWidget(self.centerPos_label , row=0,col=0)
        self.w0.addWidget(self.centerPos_text, row=0,col=1)

        self.w0.addWidget(self.angle_label , row=1,col=0)
        self.w0.addWidget(self.angle_text, row=1,col=1)

        self.w0.addWidget(self.size_label , row=2,col=0)
        self.w0.addWidget(self.size_text, row=2,col=1)

        #add layout widget to dock
        self.d0.addWidget(self.w0)


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
        if g.settings['overlayMultipleRecordings'] is None or 'pixelSize' not in g.settings['overlayMultipleRecordings']:
            s = dict()
            s['pixelSize'] = 108
            g.settings['overlayMultipleRecordings'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, pixelSize,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        pixelSize: int - pixel size of image data

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['overlayMultipleRecordings']['pixelSize'] = pixelSize


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
        self.dataWindow = WindowSelector()


        self.loadButton = QPushButton('Load Data')
        self.loadButton.pressed.connect(self.loadData)

        #self.endButton = QPushButton('Save Alignment')
        #self.endButton.pressed.connect(self.endAlign)

        #self.clearButton = QPushButton('Clear Alignment')
        #self.clearButton.pressed.connect(self.clearTemplate)

        # self.templateBox = pg.ComboBox()
        # self.templates = {'Disk': 'disk',
        #                   'Square': 'square',
        #                   'Crossbow': 'crossbow',
        #                   'Y-shape': 'y-shape',
        #                   'H-shape': 'h-shape'
        #                   }
        # self.templateBox.setItems(self.templates)
        # self.templateBox.activated.connect(self.update)

        #data file selector
        self.getFolder = FolderSelector()

        #self.transformDataButton = QPushButton('Transform data')
        #self.transformDataButton.pressed.connect(self.transformData)


        self.items.append({'name': 'dataWindow', 'string': 'Image Window', 'object': self.dataWindow})
        self.items.append({'name': 'foldername ', 'string': 'Data Folder', 'object': self.getFolder})

        self.items.append({'name': 'loadButton', 'string': '', 'object': self.loadButton})

        #self.items.append({'name': 'endButton', 'string': '', 'object': self.endButton})
        #self.items.append({'name': 'clearButton', 'string': '', 'object': self.clearButton})
        #self.items.append({'name': 'clearButton', 'string': '', 'object': self.clearButton})
        #self.items.append({'name': 'transformButton', 'string': 'Transform data file', 'object': self.transformDataButton})

        super().gui()

        #initialize display window and hide it
        self.displayParams = DisplayParams(self)
        self.displayParams.show()

        self.data = None

        self.foldername = None
        self.filenames = []

        self.plotWindow = None

        self.pointMapScatter = None
        self.heatMap = None

        return


    def update(self):
        ...

    def loadData(self):
        #Check folder selected
        if self.getValue('dataWindow') == None:
            print('Select Plot Window')
            return

        #set plot window as selected window
        self.plotWindow = self.getValue('dataWindow')

        #get folders '*_transform.csv' file path using glob
        self.foldername = self.getFolder.value()
        print(self.foldername)
        self.filenames = glob.glob(self.foldername + '/*_transform.csv', recursive = False)
        print(self.filenames)
        # Load xy data from the selected files using Pandas
        for file in tqdm(self.filenames):
            tempDF = pd.read_csv(file, usecols=['x_transformed', 'y_transformed'])
            truncFileName = os.path.basename(file).split('_locsID')[0]
            tempDF['file'] = truncFileName
            self.data = pd.concat([self.data, tempDF])


        print('Data loaded')

        #plot data on image
        self.plotDataPoints()


    def plotDataPoints(self):
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




# Instantiate the LocsAndTracksPlotter class
overlayMultipleRecordings = OverlayMultipleRecordings()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











