# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

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

from tqdm import tqdm


# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# import custom module to join tracks

from .helperFunctions import *
from .roiZoomPlotter import *


'''
#####################################################################################################################################
######################################   Main LOCSANDTRACKSPLOTTER CLASS   ##########################################################
#####################################################################################################################################
'''

class VideoExporter(BaseProcess_noPriorWindow):
    """
    export stack as video
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['videoExporter'] is None or 'framelength' not in g.settings['videoExporter']:
            s = dict()
            s['pixelSize'] = 108
            s['framelength'] = 100
            g.settings['videoExporter'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, pixelSize, framelength,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        framelength
        pixelSize: int - pixel size of image data

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['videoExporter']['pixelSize'] = pixelSize
        g.settings['videoExporter']['framelength'] = framelength

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
        # Initialize class variables
        self.pixelSize= None
        self.framelength= None
        self.displayROIplot = False
        self.plotWindow = g.win

        self.framelength=pg.SpinBox(int=True)
        self.framelength.setRange(0,1000000)
        #self.framelength.setDecimals(10)

        self.pixelSize=pg.SpinBox(int=True)
        self.pixelSize.setRange(0,1000000)
        #self.pixelSize.setDecimals(10)



        # Call gui_reset function
        self.gui_reset()
        # Get settings for locsAndTracksPlotter
        s=g.settings['videoExporter']

        #buttons

        #checkbox
        self.displayROIplot_checkbox = CheckBox()
        self.displayROIplot_checkbox.stateChanged.connect(self.toggleROIplot)
        self.displayROIplot_checkbox.setChecked(False)


        #comboboxes


        #################################################################
        # Define the items that will appear in the GUI, and associate them with the appropriate functions.
        self.items.append({'name':'pixelSize','string':'default pixel size (microns)','object':self.pixelSize})
        self.items.append({'name':'framelength','string':'default Frame length (ms)','object':self.framelength})
        self.items.append({'name': 'displayROIplot', 'string': 'ROI Video Exporter', 'object': self.displayROIplot_checkbox})

        super().gui()
        ######################################################################

        # Initialize all-tracks plot window and hide it
        self.ROIplot = ROIPLOT(self)
        self.ROIplot.hide()

        #link spinbox change
        self.pixelSize.valueChanged.connect(self.update)
        self.framelength.valueChanged.connect(self.update)
        return


    def toggleROIplot(self):
        if self.ROIplot == None:
            # Create a new instance of the DiffusionPlotWindow class
            self.ROIplot = ROIPLOT(self)

        if self.displayROIplot == False:
            # Show the window if it is not already displayed
            self.ROIplot.show()
            self.displayROIplot= True

        else:
            # Hide the window if it is already displayed
            self.ROIplot.hide()
            self.displayROIplot= False

    def update(self):
        self.ROIplot.pixelSize_box.setValue(self.pixelSize.value())
        self.ROIplot.framelength_box.setValue(self.framelength.value())

# Instantiate the LocsAndTracksPlotter class
videoExporter = VideoExporter()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











