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
from scipy.optimize import curve_fit
import time
import skimage.io as skio
from tqdm import tqdm

from skimage.filters import threshold_otsu
from skimage import data, color, measure
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from math import cos, sin, degrees


# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

from pyqtgraph import HistogramLUTWidget

# enable Numba for performance
import numba
pg.setConfigOption('useNumba', True)

# import pandas and matplotlib for generating graphs
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Arrow

# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# import custom module to join tracks
from .joinTracks import JoinTracks
from .helperFunctions import *
from .io import *
from .roiZoomPlotter import *
from .allTracksPlotter import AllTracksPlot
from .trackPlotter import *
from .flowerPlot import *
from .diffusionPlot import *
from .trackWindow import TrackWindow
from .chartDock import ChartDock
from .overlay import Overlay


class FilterOptions():
    """
    A class for a GUI setting filter options for points and tracks.
    """
    def __init__(self, mainGUI):
        super().__init__()

        # Initialize the main GUI instance
        self.mainGUI = mainGUI

        # Create a dock window and add a dock
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(550,100)
        self.win.setWindowTitle('Filter')
        self.d1 = Dock("Filter Options", size=(550, 100))
        self.area.addDock(self.d1)

        # Create a layout widget
        self.w1 = pg.LayoutWidget()

        # Create widgets for filter column, operator, and value selection
        self.filterCol_Box = pg.ComboBox()
        self.filtercols = {'None':'None'}
        self.filterCol_Box.setItems(self.filtercols)
        self.filterOp_Box = pg.ComboBox()
        self.filterOps = {'=':'==', '<':'<', '>':'>', '!=':'!='}
        self.filterOp_Box.setItems(self.filterOps)
        self.filterValue_Box = QLineEdit()

        # Create a checkbox for enabling/disabling sequential filtering
        self.sequentialFlter_checkbox = CheckBox()
        self.sequentialFlter_checkbox.setChecked(False)
        self.sequentialFlter_checkbox.stateChanged.connect(self.setSequentialFilter)

        # Create labels for filter column, operator, and value selection, as well as for the sequential filtering checkbox
        self.filterCol_label = QLabel('Filter column')
        self.filterVal_label = QLabel('Value')
        self.filterOp_label = QLabel('Operator')
        self.sequentialFilter_label = QLabel('Allow sequential filtering')

        #buttons
        self.filterData_button = QPushButton('Filter')
        self.filterData_button.pressed.connect(self.mainGUI.filterData)
        self.clearFilterData_button = QPushButton('Clear Filter')
        self.clearFilterData_button.pressed.connect(self.mainGUI.clearFilterData)
        self.ROIFilterData_button = QPushButton(' Filter by ROI(s)')
        self.ROIFilterData_button.pressed.connect(self.mainGUI.ROIFilterData)
        self.clearROIFilterData_button = QPushButton('Clear ROI Filter')
        self.clearROIFilterData_button.pressed.connect(self.mainGUI.clearROIFilterData)

        ## Add widgets to layout
        #row0
        self.w1.addWidget(self.filterCol_label, row=0,col=0)
        self.w1.addWidget(self.filterCol_Box, row=0,col=1)
        self.w1.addWidget(self.filterOp_label, row=0,col=2)
        self.w1.addWidget(self.filterOp_Box, row=0,col=3)
        self.w1.addWidget(self.filterVal_label, row=0,col=4)
        self.w1.addWidget(self.filterValue_Box, row=0,col=5)
        #row1
        self.w1.addWidget(self.filterData_button, row=1,col=0)
        self.w1.addWidget(self.clearFilterData_button, row=1,col=1)
        self.w1.addWidget(self.sequentialFilter_label, row=1,col=2)
        self.w1.addWidget(self.sequentialFlter_checkbox, row=1,col=3)
        #row3
        self.w1.addWidget(self.ROIFilterData_button, row=3,col=0)
        self.w1.addWidget(self.clearROIFilterData_button, row=3,col=1)

        #add layout to dock
        self.d1.addWidget(self.w1)


    def setSequentialFilter(self):
        if self.sequentialFlter_checkbox.isChecked():
            self.mainGUI.sequentialFiltering = True
        else:
            self.mainGUI.sequentialFiltering = False

    def show(self):
        self.win.show()

    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()


class TrackPlotOptions():
    '''
    Choose colours etc for track plots
    '''
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        #self.win.resize(250, 200)
        self.win.setWindowTitle('Display Options')

        ## Create docks
        self.d0 = Dock("Point Options")
        self.d1 = Dock("Track Options")
        self.d2 = Dock("Recording Parameters")
        self.d3 = Dock("Background subtraction")
        self.area.addDock(self.d0)
        self.area.addDock(self.d1)
        self.area.addDock(self.d2)
        self.area.addDock(self.d3)

        #point options widget
        self.w0 = pg.LayoutWidget()

        self.pointColour_Box = pg.ComboBox()
        self.pointColours = {'green': QColor(Qt.green), 'red': QColor(Qt.red), 'blue': QColor(Qt.blue)}
        self.pointColour_Box.setItems(self.pointColours)
        self.pointColour_Box_label = QLabel('Point Colour')

        self.pointSize_selector = pg.SpinBox(value=5, int=True)
        self.pointSize_selector.setSingleStep(1)
        self.pointSize_selector.setMinimum(0)
        self.pointSize_selector.setMaximum(100)
        self.pointSize_selector_label = QLabel('Point Size')

        self.unlinkedpointColour_Box = pg.ComboBox()
        self.unlinkedpointColours = {'blue': QColor(Qt.blue), 'green': QColor(Qt.green), 'red': QColor(Qt.red)}
        self.unlinkedpointColour_Box.setItems(self.unlinkedpointColours)
        self.unlinkedpointColour_Box_label = QLabel('Unlinked Point Colour')


        #layout
        self.w0.addWidget(self.unlinkedpointColour_Box_label , row=0,col=0)
        self.w0.addWidget(self.unlinkedpointColour_Box, row=0,col=1)

        self.w0.addWidget(self.pointColour_Box_label , row=1,col=0)
        self.w0.addWidget(self.pointColour_Box, row=1,col=1)

        self.w0.addWidget(self.pointSize_selector_label , row=2,col=0)
        self.w0.addWidget(self.pointSize_selector , row=2,col=1)

        #add layout widget to dock
        self.d0.addWidget(self.w0)

        #track options widget
        self.w1 = pg.LayoutWidget()

        #combo boxes
        self.trackColourCol_Box = pg.ComboBox()
        self.trackcolourcols = {'None':'None'}
        self.trackColourCol_Box.setItems(self.trackcolourcols)
        self.trackColourCol_Box_label = QLabel('Colour By')

        self.colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.colourMap_Box.setItems(self.colourMaps)
        self.colourMap_Box_label = QLabel('Colour Map')

        self.trackDefaultColour_Box = pg.ComboBox()
        self.trackdefaultcolours = {'green': Qt.green, 'red': Qt.red, 'blue': Qt.blue}
        self.trackDefaultColour_Box.setItems(self.trackdefaultcolours)
        self.trackDefaultColour_Box_label = QLabel('Track Default Colour')

        self.lineSize_selector = pg.SpinBox(value=2, int=True)
        self.lineSize_selector.setSingleStep(1)
        self.lineSize_selector.setMinimum(0)
        self.lineSize_selector.setMaximum(100)
        self.lineSize_selector_label = QLabel('Line Size')

        #check boxes
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(False)
        self.trackColour_checkbox_label = QLabel('Set Track Colour')

        self.matplotCM_checkbox = CheckBox()
        self.matplotCM_checkbox.stateChanged.connect(self.mainGUI.setColourMap)
        self.matplotCM_checkbox.setChecked(False)
        self.matplotCM_checkbox_label = QLabel('Use Matplot Colour Map')

        #layout
        self.w1.addWidget(self.trackColour_checkbox_label , row=1,col=0)
        self.w1.addWidget(self.trackColour_checkbox, row=1,col=1)

        self.w1.addWidget(self.trackColourCol_Box_label , row=2,col=0)
        self.w1.addWidget(self.trackColourCol_Box , row=2,col=1)

        self.w1.addWidget(self.colourMap_Box_label , row=3,col=0)
        self.w1.addWidget(self.colourMap_Box , row=3,col=1)

        self.w1.addWidget(self.matplotCM_checkbox_label , row=4,col=0)
        self.w1.addWidget(self.matplotCM_checkbox , row=4,col=1)

        self.w1.addWidget(self.trackDefaultColour_Box_label , row=5,col=0)
        self.w1.addWidget(self.trackDefaultColour_Box , row=5,col=1)

        self.w1.addWidget(self.lineSize_selector_label, row=6,col=0)
        self.w1.addWidget(self.lineSize_selector, row=6,col=1)

        #add layout widget to dock
        self.d1.addWidget(self.w1)

        #recording options widget
        self.w2 = pg.LayoutWidget()

        #spinbox
        self.frameLength_selector = pg.SpinBox(value=10, int=True)
        self.frameLength_selector.setSingleStep(10)
        self.frameLength_selector.setMinimum(1)
        self.frameLength_selector.setMaximum(100000)
        self.frameLength_selector_label = QLabel('milliseconds per frame')

        self.pixelSize_selector = pg.SpinBox(value=108, int=True)
        self.pixelSize_selector.setSingleStep(1)
        self.pixelSize_selector.setMinimum(1)
        self.pixelSize_selector.setMaximum(10000)
        self.pixelSize_selector_label = QLabel('nanometers per pixel')

        #layout
        self.w2.addWidget(self.frameLength_selector_label , row=1,col=0)
        self.w2.addWidget(self.frameLength_selector, row=1,col=1)

        self.w2.addWidget(self.pixelSize_selector_label, row=2,col=0)
        self.w2.addWidget(self.pixelSize_selector, row=2,col=1)

        #add layout widget to dock
        self.d2.addWidget(self.w2)

        #background options widget
        self.w3 = pg.LayoutWidget()

        self.intensityChoice_Box = pg.ComboBox()
        self.intensityChoice = {'intensity':'intensity',
                                'intensity - mean roi1':'intensity - mean roi1',
                                'intensity_roiOnMeanXY': 'intensity_roiOnMeanXY',
                                'intensity_roiOnMeanXY - mean roi1': 'intensity_roiOnMeanXY - mean roi1',
                                'intensity_roiOnMeanXY - mean roi1 and black': 'intensity_roiOnMeanXY - mean roi1 and black',
                                'intensity_roiOnMeanXY - smoothed roi_1': 'intensity_roiOnMeanXY - smoothed roi_1',
                                'intensity - smoothed roi_1': 'intensity - smoothed roi_1'}
        self.intensityChoice_Box.setItems(self.intensityChoice)
        self.intensityChoice_Box_label = QLabel('Intensity plot data')

        self.backgroundSubtract_checkbox = CheckBox()
        self.backgroundSubtract_checkbox.setChecked(False)
        self.backgroundSubtract_label = QLabel('Subtract Background')

        self.background_selector = pg.SpinBox(value=0, int=True)
        self.background_selector.setSingleStep(1)
        self.background_selector.setMinimum(0)
        self.background_selector.setMaximum(10000)
        self.background_selector_label = QLabel('background value')

        self.estimatedCameraBlack = QLabel('')
        self.estimatedCameraBlack_label = QLabel('estimated camera black')

        self.w3.addWidget(self.intensityChoice_Box, row=0,col=1)
        self.w3.addWidget(self.intensityChoice_Box_label, row=0,col=0)

        self.w3.addWidget(self.backgroundSubtract_checkbox, row=1,col=1)
        self.w3.addWidget(self.backgroundSubtract_label, row=1,col=0)

        self.w3.addWidget(self.background_selector, row=2,col=1)
        self.w3.addWidget(self.background_selector_label, row=2,col=0)

        self.w3.addWidget(self.estimatedCameraBlack, row=3,col=1)
        self.w3.addWidget(self.estimatedCameraBlack_label, row=3,col=0)

        #add layout widget to dock
        self.d3.addWidget(self.w3)


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

class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    plots loc and track data onto current window
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['locsAndTracksPlotter'] is None or 'set_track_colour' not in g.settings['locsAndTracksPlotter']:
            s = dict()
            s['filename'] = ''
            s['filetype'] = 'flika'
            s['pixelSize'] = 108
            s['set_track_colour'] = False
            g.settings['locsAndTracksPlotter'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, filename, filetype, pixelSize, set_track_colour,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        filename: str - path to file
        filetype: str - file type (flika or thunderstorm)
        pixelSize: int - pixel size of image data
        set_track_colour: bool - whether to set track colour based on track ID

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['locsAndTracksPlotter']['filename'] = filename
        g.settings['locsAndTracksPlotter']['filetype'] = filetype
        g.settings['locsAndTracksPlotter']['pixelSize'] = pixelSize
        g.settings['locsAndTracksPlotter']['set_track_colour'] = set_track_colour

        # Show a message in the status bar that plotting data is in progress
        g.m.statusBar().showMessage("plotting data...")
        return


    def closeEvent(self, event):
        '''
        This function is called when the user closes the locs and tracks plotter window. It clears any plots that have been
        generated and calls the closeEvent function for the BaseProcess_noPriorWindow class.

        Parameters:
        event: object - object representing the close event

        Returns: None
        '''
        # Clear any plots that have been generated
        self.clearPlots()

        # Call the closeEvent function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return


    def gui(self):
        # Initialize class variables
        self.filename = ''
        self.filetype = 'flika'
        self.pixelSize= None
        self.plotWindow = None
        self.pathitems = []
        self.useFilteredData = False
        self.useFilteredTracks = False
        self.filteredData = None
        self.useMatplotCM = False
        self.selectedTrack = None
        self.displayTrack = None
        self.chartWindow = None
        self.displayCharts = False
        self.diffusionWindow = None
        self.displayDiffusionPlot = False
        self.unlinkedPoints = None
        self.displayUnlinkedPoints = False
        self.estimatedCameraBlackLevel = 0
        self.displayROIplot = False

        #initialize track plot options window and hide it
        self.trackPlotOptions = TrackPlotOptions(self)
        self.trackPlotOptions.hide()
        self.displayTrackPlotOptions = False

        # Initialize overlay window and hide it
        self.overlayWindow = Overlay(self)
        self.overlayWindow.hide()
        self.displayOverlay = False

        # Initialize filter options window and hide it
        self.filterOptionsWindow = FilterOptions(self)
        self.filterOptionsWindow.hide()
        self.sequentialFiltering = False

        # Initialize track plot window and hide it
        self.trackWindow = TrackWindow(self)
        self.trackWindow.hide()

        # Initialize flower plot window and hide it
        self.flowerPlotWindow = FlowerPlotWindow(self)
        self.flowerPlotWindow.hide()

        # Initialize single track plot window and hide it
        self.singleTrackPlot= TrackPlot(self)
        self.singleTrackPlot.hide()

        # Initialize all-tracks plot window and hide it
        self.allTracksPlot = AllTracksPlot(self)
        self.allTracksPlot.hide()

        # Initialize all-tracks plot window and hide it
        self.ROIplot = ROIPLOT(self)
        self.ROIplot.hide()

        # Call gui_reset function
        self.gui_reset()
        # Get settings for locsAndTracksPlotter
        s=g.settings['locsAndTracksPlotter']

        #buttons
        self.plotPointData_button = QPushButton('Plot Points')
        self.plotPointData_button.pressed.connect(self.plotPointData)

        self.hidePointData_button = QPushButton('Toggle Points')
        self.hidePointData_button.pressed.connect(self.hidePointData)

        self.toggleUnlinkedPointData_button = QPushButton('Show Unlinked')
        self.toggleUnlinkedPointData_button.pressed.connect(self.toggleUnlinkedPointData)


        self.plotTrackData_button = QPushButton('Plot Tracks')
        self.plotTrackData_button.pressed.connect(self.plotTrackData)

        self.clearTrackData_button = QPushButton('Clear Tracks')
        self.clearTrackData_button.pressed.connect(self.clearTracks)


        self.saveData_button = QPushButton('Save Tracks')
        self.saveData_button.pressed.connect(self.saveData)

        self.showCharts_button = QPushButton('Show Charts')
        self.showCharts_button.pressed.connect(self.toggleCharts)

        self.showDiffusion_button = QPushButton('Show Diffusion')
        self.showDiffusion_button.pressed.connect(self.toggleDiffusionPlot)

        self.togglePointMap_button = QPushButton('Plot Point Map')
        self.togglePointMap_button.pressed.connect(self.togglePointMap)

        self.overlayOption_button = QPushButton('Overlay')
        self.overlayOption_button.pressed.connect(self.displayOverlayOptions)

        #checkbox
        self.displayFlowPlot_checkbox = CheckBox()
        self.displayFlowPlot_checkbox.stateChanged.connect(self.toggleFlowerPlot)
        self.displayFlowPlot_checkbox.setChecked(False)

        self.displaySingleTrackPlot_checkbox = CheckBox()
        self.displaySingleTrackPlot_checkbox.stateChanged.connect(self.toggleSingleTrackPlot)
        self.displaySingleTrackPlot_checkbox.setChecked(False)

        self.displayAllTracksPlot_checkbox = CheckBox()
        self.displayAllTracksPlot_checkbox.stateChanged.connect(self.toggleAllTracksPlot)
        self.displayAllTracksPlot_checkbox.setChecked(False)

        self.displayFilterOptions_checkbox = CheckBox()
        self.displayFilterOptions_checkbox.stateChanged.connect(self.toggleFilterOptions)
        self.displayFilterOptions_checkbox.setChecked(False)

        self.displayTrackPlotOptions_checkbox = CheckBox()
        self.displayTrackPlotOptions_checkbox.stateChanged.connect(self.toggleTrackPlotOptions)
        self.displayTrackPlotOptions_checkbox.setChecked(False)

        self.displayROIplot_checkbox = CheckBox()
        self.displayROIplot_checkbox.stateChanged.connect(self.toggleROIplot)
        self.displayROIplot_checkbox.setChecked(False)


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

        #data file selector
        self.getFile = FileSelector(filetypes='*.csv', mainGUI=self)

        #connections
        self.getFile.valueChanged.connect(self.loadData)

        #add blank columns for missing analysis steps to prevent crashes
        self.expectedColumns = ['frame', 'track_number', 'x', 'y', 'intensity', 'zeroed_X', 'zeroed_Y',
       'lagNumber', 'distanceFromOrigin', 'dy-dt: distance', 'radius_gyration',
       'asymmetry', 'skewness', 'kurtosis', 'fracDimension', 'netDispl',
       'Straight', 'Experiment', 'SVM', 'nnDist_inFrame', 'n_segments', 'lag',
       'meanLag', 'track_length', 'radius_gyration_scaled',
       'radius_gyration_scaled_nSegments',
       'radius_gyration_scaled_trackLength', 'roi_1', 'camera black estimate',
       'd_squared', 'lag_squared', 'dt', 'velocity',
       'direction_Relative_To_Origin', 'meanVelocity', 'intensity - mean roi1',
       'intensity - mean roi1 and black', 'nnCountInFrame_within_3_pixels',
       'nnCountInFrame_within_5_pixels', 'nnCountInFrame_within_10_pixels',
       'nnCountInFrame_within_20_pixels', 'nnCountInFrame_within_30_pixels',
        'intensity_roiOnMeanXY','intensity_roiOnMeanXY - mean roi1',
        'intensity_roiOnMeanXY - mean roi1 and black','roi_1 smoothed',
        'intensity_roiOnMeanXY - smoothed roi_1',
        'intensity - smoothed roi_1']


        #################################################################
        # Define the items that will appear in the GUI, and associate them with the appropriate functions.
        #self.exportFolder = FolderSelector('*.txt')

        self.items.append({'name': 'filename ', 'string': '', 'object': self.getFile})
        self.items.append({'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box})
        self.items.append({'name': 'hidePoints', 'string': 'PLOT    --------------------', 'object': self.hidePointData_button })
        self.items.append({'name': 'plotPointMap', 'string': '', 'object': self.togglePointMap_button })
        self.items.append({'name': 'plotUnlinkedPoints', 'string': '', 'object': self.toggleUnlinkedPointData_button })
        self.items.append({'name': 'trackPlotOptions', 'string': 'Display Options', 'object': self.displayTrackPlotOptions_checkbox })
        self.items.append({'name': 'displayFlowerPlot', 'string': 'Flower Plot', 'object': self.displayFlowPlot_checkbox})
        self.items.append({'name': 'displaySingleTrackPlot', 'string': 'Track Plot', 'object': self.displaySingleTrackPlot_checkbox})
        self.items.append({'name': 'displayAllTracksPlot', 'string': 'All Tracks Plot', 'object': self.displayAllTracksPlot_checkbox})
        self.items.append({'name': 'displayROIplot', 'string': 'ROI Plot', 'object': self.displayROIplot_checkbox})
        self.items.append({'name': 'displayFilterOptions', 'string': 'Filter Window', 'object': self.displayFilterOptions_checkbox})
        self.items.append({'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button })
        self.items.append({'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button })
        self.items.append({'name': 'saveTracks', 'string': '', 'object': self.saveData_button })
        self.items.append({'name': 'showCharts', 'string': '', 'object': self.showCharts_button })
        self.items.append({'name': 'showDiffusion', 'string': '', 'object': self.showDiffusion_button })
        self.items.append({'name': 'overlayOptions', 'string': '', 'object': self.overlayOption_button })


        super().gui()
        ######################################################################
        return

    def loadData(self):
        # Set the plot window to the global window instance
        self.plotWindow = g.win

        # Get the filename from the GUI
        self.filename = self.getFile.value()

        # Load the data from the selected file using Pandas
        self.data = pd.read_csv(self.filename)

        ### TODO! #Check if analysed columns are in df - if missing add and display error message
        #g.m.statusBar().showMessage('{} columns missing - blanks aded'.format())
        #print('{} columns missing - blanks aded'.format())

        #make sure track number and frame are int
        self.data['frame'] = self.data['frame'].astype(int)

        if 'track_number' in self.data.columns:
            # filter any points that dont have track_numbers to seperate df
            self.data_unlinked = self.data[self.data['track_number'].isna()]
            self.data  = self.data[~self.data['track_number'].isna()]

            self.data['track_number'] = self.data['track_number'].astype(int)

        else:
            self.data['track_number'] = None
            self.data_unlinked = self.data

        # Check that there are enough frames in the stack to plot all data points
        if np.max(self.data['frame']) > g.win.mt:
            g.alert("Selected window doesn't have enough frames to plot all data points")
            self.plotWindow = None
            self.filename = None
            self.data = None
            return

        # Print a message to the console indicating that the data has been loaded
        # and display the first 5 rows of the data
        print('-------------------------------------')
        print('Data loaded (first 5 rows displayed):')
        print(self.data.head())
        print('-------------------------------------')

        #add background subtracted intensity column
        if 'intensity - background' in self.data.columns:
            self.data['intensity - background'] = self.data['intensity']

        # Create a dictionary from the column names of the data
        self.columns = self.data.columns
        self.colDict= dictFromList(self.columns)

        # Set the options for the various dropdown menus in the GUI based on the column names
        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)
        self.frameCol_Box.setItems(self.colDict)
        self.trackCol_Box.setItems(self.colDict)
        self.filterOptionsWindow.filterCol_Box.setItems(self.colDict)
        self.trackPlotOptions.trackColourCol_Box.setItems(self.colDict)

        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)

        # Format the points and add them to the image window
        self.plotPointData()

        # Update the track plot track selector and set the options for the color dropdowns
        self.singleTrackPlot.updateTrackList()
        self.singleTrackPlot.pointCol_Box.setItems(self.colDict)
        self.singleTrackPlot.lineCol_Box.setItems(self.colDict)

        # Set the padding array for the single track plot based on the image array
        self.singleTrackPlot.setPadArray(self.plotWindow.imageArray())

        # Update the all-tracks plot track selector
        self.allTracksPlot.updateTrackList()
        self.ROIplot.updateTrackList()


        # add image data to overlay
        self.overlayWindow.loadData()

        # set estimated camera black level
        self.estimatedCameraBlackLevel = np.min(self.plotWindow.image)
        self.trackPlotOptions.estimatedCameraBlack.setText(str(self.estimatedCameraBlackLevel))

        for col in self.expectedColumns:
            if col not in self.columns:
                self.data[col] = np.nan
                self.data_unlinked[col] = np.nan


    def makePointDataDF(self, data):
        # Check the filetype selected in the GUI
        if self.filetype_Box.value() == 'thunderstorm':
            # Load thunderstorm data into a pandas dataframe
            df = pd.DataFrame()
            # Convert frame data from float to int and adjust by -1 (since ThunderSTORM starts counting from 1)
            df['frame'] = data['frame'].astype(int)-1
            # Convert x data from nanometers to pixels
            df['x'] = data['x [nm]'] / self.trackPlotOptions.pixelSize_selector.value()
            # Convert y data from nanometers to pixels
            df['y'] = data['y [nm]'] / self.trackPlotOptions.pixelSize_selector.value()

        elif self.filetype_Box.value() == 'flika':
            # Load FLIKA pyinsight data into a pandas dataframe
            df = pd.DataFrame()
            # Convert frame data from float to int and adjust by -1 (since FLIKA starts counting from 1)
            df['frame'] = data['frame'].astype(int)-1
            # Use x data as-is (FLIKA data is already in pixel units)
            df['x'] = data['x']
            # Use y data as-is (FLIKA data is already in pixel units)
            df['y'] = data['y']


        # Return the completed pandas dataframe
        return df

    def plotPointsOnStack(self, points, pointColor, unlinkedPoints=None, unlinkedColour=QColor(Qt.blue)):
        points_byFrame = points[['frame','x','y']]
        #align frames with display
        if self.filetype_Box.value() == 'thunderstorm':
            points_byFrame['frame'] =  points_byFrame['frame']
        else:
            points_byFrame['frame'] =  points_byFrame['frame']+1
        # Convert the points DataFrame into a numpy array
        pointArray = points_byFrame.to_numpy()
        # Create an empty list for each frame in the stack
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        #set pointsize
        pointSize = self.trackPlotOptions.pointSize_selector.value()


        # Iterate through each point in the point array and add it to the appropriate frame's list
        for pt in pointArray:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            #pointSize = g.m.settings['point_size']
            #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
            position = [pt[1], pt[2], pointColor, pointSize]

            self.plotWindow.scatterPoints[t].append(position)


        if self.displayUnlinkedPoints:
            unlinkedPoints_byFrame = unlinkedPoints[['frame','x','y']]
            if self.filetype_Box.value() == 'thunderstorm':
                unlinkedPoints_byFrame['frame'] =  unlinkedPoints_byFrame['frame']
            else:
                points_byFrame['frame'] =  points_byFrame['frame']+1
            # Convert the points DataFrame into a numpy array
            unlinkedPointArray = unlinkedPoints_byFrame.to_numpy()


            # Iterate through each point in the point array and add it to the appropriate frame's list
            for pt in unlinkedPointArray:
                t = int(pt[0])
                if self.plotWindow.mt == 1:
                    t = 0
                #pointSize = g.m.settings['point_size']
                #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
                position = [pt[1], pt[2], unlinkedColour, pointSize]

                self.plotWindow.scatterPoints[t].append(position)



        # Update the index of the image stack to include the new points
        self.plotWindow.updateindex()


    def hidePointData(self):
        if self.plotWindow.scatterPlot in self.plotWindow.imageview.ui.graphicsView.items():
            # If the scatter plot is currently in the graphics view, remove it
            self.plotWindow.imageview.ui.graphicsView.removeItem(self.plotWindow.scatterPlot)
        else:
            # Otherwise, add it back to the graphics view
            self.plotWindow.imageview.addItem(self.plotWindow.scatterPlot)


    def plotPointData(self):
        ### plot point data to current window
        # Create a pandas DataFrame containing the point data
        if self.useFilteredData == False:
            self.points = self.makePointDataDF(self.data)
        else:
            self.points = self.makePointDataDF(self.filteredData)

        if self.displayUnlinkedPoints:
            # Create a pandas DataFrame containing the point data
            self.unlinkedPoints = self.makePointDataDF(self.data_unlinked)
        else:
            self.unlinkedPoints = None

        # Plot the points on the image stack using the plotPointsOnStack() method
        self.plotPointsOnStack(self.points, self.trackPlotOptions.pointColour_Box.value(), unlinkedPoints=self.unlinkedPoints, unlinkedColour=self.trackPlotOptions.unlinkedpointColour_Box.value())

        # Display a message in the status bar indicating that the point data has been plotted
        g.m.statusBar().showMessage('point data plotted to current window')
        print('point data plotted to current window')

        return


    def toggleUnlinkedPointData(self):
        if self.displayUnlinkedPoints == False:
            ### plot unlinked point data to current window
            self.displayUnlinkedPoints = True
            self.plotPointData()
            self.toggleUnlinkedPointData_button.setText('Hide Unlinked')
            # Display a message in the status bar indicating that the point data has been plotted
            g.m.statusBar().showMessage('unlinked point data plotted to current window')
            print('unlinked point data plotted to current window')

        else:
            self.displayUnlinkedPoints = False
            self.plotPointData()
            self.toggleUnlinkedPointData_button.setText('Show Unlinked')
        return


    def makeTrackDF(self, data):
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.trackPlotOptions.pixelSize_selector.value()
            df['y'] = data['y [nm]']/self.trackPlotOptions.pixelSize_selector.value()
            df['track_number'] = data['track_number']

            colsPresent = df.columns

            for col in self.expectedColumns:
                if col not in colsPresent:
                    df[col] = np.nan


        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']
            df['track_number'] = data['track_number']

            df['zeroed_X'] = data['zeroed_X']
            df['zeroed_Y'] = data['zeroed_Y']

            # Add a color column to the DataFrame based on the selected color map and column
            if self.trackPlotOptions.trackColour_checkbox.isChecked():
                if self.useMatplotCM:
                    cm = pg.colormap.getFromMatplotlib(self.trackPlotOptions.colourMap_Box.value()) # Get the colormap from Matplotlib and convert it to a PyqtGraph colormap
                else:
                    cm = pg.colormap.get(self.trackPlotOptions.colourMap_Box.value()) # Get the PyqtGraph colormap

                # Map the values from the selected color column to a QColor using the selected colormap
                df['colour'] = cm.mapToQColor(data[self.trackPlotOptions.trackColourCol_Box.value()].to_numpy()/max(data[self.trackPlotOptions.trackColourCol_Box.value()]))

        # Group the data by track number
        return df.groupby(['track_number'])


    def clearTracks(self):
        # Check that there is an open plot window
        if self.plotWindow is not None and not self.plotWindow.closed:
            # Remove each path item from the plot window
            for pathitem in self.pathitems:
                self.plotWindow.imageview.view.removeItem(pathitem)

            for pathitem in self.overlayWindow.pathitems:
                self.overlayWindow.overlayWindow.view.removeItem(pathitem)
        # Reset the path items list to an empty list
        self.pathitems = []
        self.overlayWindow.pathitems=[]


    def showTracks(self):
        '''Updates track paths in main view and Flower Plot'''

        # clear self.pathitems
        self.clearTracks()

        self.overlayWindow.clearTracks()

        # clear track paths in Flower Plot window if displayFlowPlot_checkbox is checked
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.clearTracks()

        # setup pens
        pen = QPen(self.trackPlotOptions.trackDefaultColour_Box.value(), .4)
        pen.setCosmetic(True)
        pen.setWidth(self.trackPlotOptions.lineSize_selector.value())

        pen_FP = QPen(self.trackPlotOptions.trackDefaultColour_Box.value(), .4)
        pen_FP.setCosmetic(True)
        pen_FP.setWidth(1)

        pen_overlay = QPen(self.trackPlotOptions.trackDefaultColour_Box.value(), .4)
        pen_overlay.setCosmetic(True)
        pen_overlay.setWidth(1)

        # determine which track IDs to plot based on whether filtered tracks are being used
        if self.useFilteredTracks:
            trackIDs = self.filteredTrackIds
        else:
            trackIDs = self.trackIDs

        print('tracks to plot {}'.format(trackIDs))

        for track_idx in trackIDs:
            tracks = self.tracks.get_group(track_idx)
            pathitem = QGraphicsPathItem(self.plotWindow.imageview.view)

            pathitem_overlay = QGraphicsPathItem(self.overlayWindow.overlayWindow.view)

            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP = QGraphicsPathItem(self.flowerPlotWindow.plt)

            # set the color of the pen based on the track color
            if self.trackPlotOptions.trackColour_checkbox.isChecked():
                pen.setColor(tracks['colour'].to_list()[0])
                pen_overlay.setColor(tracks['colour'].to_list()[0])
                pen_FP.setColor(tracks['colour'].to_list()[0])

            # set the pen for the path items
            pathitem.setPen(pen)

            pathitem_overlay.setPen(pen_overlay)

            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPen(pen_FP)

            # add the path items to the view(s)
            self.plotWindow.imageview.view.addItem(pathitem)

            self.overlayWindow.overlayWindow.view.addItem(pathitem_overlay)

            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.plt.addItem(pathitem_FP)

            # keep track of the path items
            self.pathitems.append(pathitem)

            self.overlayWindow.pathitems.append(pathitem_overlay)

            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.pathitems.append(pathitem_FP)

            # extract the x and y coordinates for the track
            x = tracks['x'].to_numpy()
            y = tracks['y'].to_numpy()

            # extract the zeroed x and y coordinates for the track, if displayed
            if self.displayFlowPlot_checkbox.isChecked():
                zeroed_X = tracks['zeroed_X'].to_numpy()
                zeroed_Y = tracks['zeroed_Y'].to_numpy()

            # create a QPainterPath for the track and set the path for the path item
            path = QPainterPath(QPointF(x[0],y[0]))

            path_overlay = QPainterPath(QPointF(x[0],y[0]))

            if self.displayFlowPlot_checkbox.isChecked():
                path_FP = QPainterPath(QPointF(zeroed_X[0],zeroed_Y[0]))
            for i in np.arange(1, len(x)):
                path.lineTo(QPointF(x[i],y[i]))

                path_overlay.lineTo(QPointF(x[i],y[i]))

                if self.displayFlowPlot_checkbox.isChecked():
                    path_FP.lineTo(QPointF(zeroed_X[i],zeroed_Y[i]))

            pathitem.setPath(path)

            pathitem_overlay.setPath(path_overlay)

            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPath(path_FP)


    def plotTrackData(self):
        ### plot track data to current window

        # check whether to use filtered data or not, get unique track IDs and create DataFrame of tracks
        if self.useFilteredData == False:
            self.trackIDs = np.unique(self.data['track_number']).astype(int)
            self.tracks = self.makeTrackDF(self.data)
        else:
            self.trackIDs = np.unique(self.filteredData['track_number']).astype(int)
            self.tracks = self.makeTrackDF(self.filteredData)

        # show tracks in main view and flower plot
        self.showTracks()

        # connect to mouse and key press events in the main view
        self.plotWindow.imageview.scene.sigMouseMoved.connect(self.updateTrackSelector)
        self.plotWindow.keyPressSignal.connect(self.selectTrack)

        # display track window with plots for individual tracks
        self.trackWindow.show()

        # display flower plot with all tracks origins set to 0,0
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()

        # display plot for a single selected track
        if self.displaySingleTrackPlot_checkbox.isChecked():
            self.singleTrackPlot.show()

        # update status bar message and print confirmation
        g.m.statusBar().showMessage('track data plotted to current window')
        print('track data plotted to current window')
        return

    def updateTrackSelector(self, point):
        pos =  self.plotWindow.imageview.getImageItem().mapFromScene(point)

        # Map mouse position to image coordinates and check which track the mouse is hovering over
        for i, path in enumerate(self.pathitems):
            if path.contains(pos):
                self.selectedTrack = self.trackIDs[i]


    def selectTrack(self,ev):
        # Listen for key press events and select track when the "T" key is pressed
        if ev.key() == Qt.Key_T:
            # Check if the selected track is different from the current display track
            if self.selectedTrack != self.displayTrack:
                self.displayTrack = self.selectedTrack

                # Extract track data for the selected track
                trackData = self.data[self.data['track_number'] == int(self.displayTrack)]
                frame = trackData['frame'].to_numpy()

                #intensity trace choice from trackPlot options
                intensity = trackData[self.trackPlotOptions.intensityChoice_Box.value()].to_numpy()
                #use background subtracted intensity if option selected
                if self.trackPlotOptions.backgroundSubtract_checkbox.isChecked():
                    intensity = intensity - self.trackPlotOptions.background_selector.value()

                distance = trackData['distanceFromOrigin'].to_numpy()
                zeroed_X = trackData['zeroed_X'].to_numpy()
                zeroed_Y = trackData['zeroed_Y'].to_numpy()
                dydt =  trackData['dy-dt: distance'].to_numpy()
                direction = trackData['direction_Relative_To_Origin'].to_numpy()
                velocity =  trackData['velocity'].to_numpy()
                svm = trackData['SVM'].iloc[0]
                length = trackData['n_segments'].iloc[0]


                count_3 = trackData['nnCountInFrame_within_3_pixels'].to_numpy()
                count_5 = trackData['nnCountInFrame_within_5_pixels'].to_numpy()
                count_10 = trackData['nnCountInFrame_within_10_pixels'].to_numpy()
                count_20 = trackData['nnCountInFrame_within_20_pixels'].to_numpy()
                count_30 = trackData['nnCountInFrame_within_30_pixels'].to_numpy()


                # Update plots in the track display window
                self.trackWindow.update(frame, intensity, distance,
                                        zeroed_X, zeroed_Y, dydt,
                                        direction, velocity, self.displayTrack,
                                        count_3, count_5, count_10, count_20, count_30,
                                        svm, length)

                # Update the individual track display
                self.singleTrackPlot.plotTracks()


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

            for pt in roiFilterPoints:
                # filter data by point coordinates
                ptFilterDF = self.data[(self.data['x']==pt[1]) & (self.data['y']==pt[2])]
                trackIDs.extend(ptFilterDF['track_number'])

            # get unique track IDs for tracks that pass through the ROI
            selectedTracks = np.unique(trackIDs)

            # join data for selected tracks
            self.joinROItracks(selectedTracks)

            # display message in status bar to indicate completion
            g.m.statusBar().showMessage('Track join complete')


    def joinROItracks(self, selectedTracks):
        # create an instance of the JoinTracks class
        joinTracks = JoinTracks()
        # create a list of selected tracks
        IDlist = [selectedTracks]
        # display the number of tracks to be joined in the status bar
        g.m.statusBar().showMessage('Tracks to join: {}'.format(IDlist))
        # use the joinTracks object to join the selected tracks
        newDF = joinTracks.join(self.data, IDlist)
        # replace the track data with the updated DataFrame that includes the joined track
        self.data = newDF
        # print the new DataFrame to the console (for debugging purposes)
        print(newDF)
        # display a message in the status bar to indicate that the track join is complete
        g.m.statusBar().showMessage('track join complete')

    def filterData(self):
        # get the filter options from the filterOptionsWindow
        op = self.filterOptionsWindow.filterOp_Box.value()
        filterCol = self.filterOptionsWindow.filterCol_Box.value()
        dtype = self.data[filterCol].dtype
        value = float(self.filterOptionsWindow.filterValue_Box.text())

        # if sequential filtering is enabled and filtered data is being used
        if self.sequentialFiltering and self.useFilteredData:
            data = self.filteredData
        else:
            data = self.data

        # apply the selected filter operation to the selected column
        if op == '==':
            self.filteredData = data[data[filterCol] == value]
        elif op == '<':
            self.filteredData = data[data[filterCol] < value]
        elif op == '>':
            self.filteredData = data[data[filterCol] > value]
        elif op == '!=':
            self.filteredData = data[data[filterCol] != value]

        # display a message in the status bar to indicate that the filter is complete
        g.m.statusBar().showMessage('filter complete')
        # set useFilteredData to True
        self.useFilteredData = True

        # update the point data plot with the filtered data
        self.plotPointData()

        #update allTracks track list
        self.allTracksPlot.updateTrackList()

        return

    def clearFilterData(self):
        # Set variables to default values to clear filtered data
        self.useFilteredData = False
        self.filteredData = None

        # Update point data plot
        self.plotPointData()

        #update allTracks track list
        self.allTracksPlot.updateTrackList()
        return

    def getScatterPointsAsQPoints(self):
        # Get scatter plot data as numpy array
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        # Convert numpy array to list of QPointF objects
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        # Get track IDs for all points in scatter plot
        trackIDs = []

        # Flatten scatter plot data into a single list of points
        flat_ptList = [pt for sublist in self.plotWindow.scatterPoints for pt in sublist]

        # Loop through each point and get track IDs for corresponding data points in DataFrame
        for pt in flat_ptList:
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.data[(self.data['x']==pt[0]) & (self.data['y']==pt[1])]

            trackIDs.extend(ptFilterDF['track_number'])

        # Set filtered track IDs and filtered data
        self.filteredTrackIds = np.unique(trackIDs)
        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]

        # Set flags for using filtered data and filtered tracks
        self.useFilteredData = True
        self.useFilteredTracks = True


    def ROIFilterData(self):
        # Not implemented yet for unlinked points
        if self.displayUnlinkedPoints:
           g.m.statusBar().showMessage('ROI filter not implemented for unliked points - hide them first')
           print('ROI filter not implemented for unliked points - hide them first')
           return
        # initialize variables
        self.roiFilterPoints = []
        self.rois = self.plotWindow.rois
        self.oldScatterPoints = self.plotWindow.scatterPoints

        # loop through all ROIs and all frames to find points inside them
        for roi in self.rois:
            currentFrame = self.plotWindow.currentIndex
            for i in range(0,self.plotWindow.mt):
                # set current frame
                self.plotWindow.setIndex(i)
                # get ROI shape in coordinate system of the scatter plot
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                self.roiFilterPoints.extend((selected))
            # reset current frame
            self.plotWindow.setIndex(currentFrame)

        # clear old scatter points and add new filtered points
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        for pt in self.roiFilterPoints:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(0,255,0)
            position = [pt[1], pt[2], pointColor, pointSize]
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()

        # get filtered data
        self.getDataFromScatterPoints()

        # update status bar and return
        g.m.statusBar().showMessage('ROI filter complete')

        #update allTracks track list
        self.allTracksPlot.updateTrackList()
        self.ROIplot.updateTrackList()
        return

    def clearROIFilterData(self):
        # Reset the scatter plot data to the previous unfiltered scatter plot data
        self.plotWindow.scatterPoints = self.oldScatterPoints
        self.plotWindow.updateindex()

        # Set useFilteredData and useFilteredTracks to False
        self.useFilteredData = False
        self.useFilteredTracks = False

        #update allTracks track list
        self.allTracksPlot.updateTrackList()

        return

    def setColourMap(self):
        # If the matplotCM_checkbox is checked, use matplotlib color maps
        if self.trackPlotOptions.matplotCM_checkbox.isChecked():
            # Create a dictionary of matplotlib color maps
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
            # Set the color map options in the dropdown box to the matplotlib color maps
            self.trackPlotOptions.colourMap_Box.setItems(self.colourMaps)
            self.useMatplotCM = True
        else:
            # If the matplotCM_checkbox is unchecked, use pyqtgraph color maps
            # Create a dictionary of pyqtgraph color maps
            self.colourMaps = dictFromList(pg.colormap.listMaps())
            # Set the color map options in the dropdown box to the pyqtgraph color maps
            self.trackPlotOptions.colourMap_Box.setItems(self.colourMaps)
            self.useMatplotCM = False

    def toggleFlowerPlot(self):
        # If the displayFlowPlot_checkbox is checked, show the flower plot window
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()
        else:
            # If the displayFlowPlot_checkbox is unchecked, hide the flower plot window
            self.flowerPlotWindow.hide()


    def toggleSingleTrackPlot(self):
        if self.displaySingleTrackPlot_checkbox.isChecked():
            # show the single track plot if checkbox is checked
            self.singleTrackPlot.show()
        else:
            # hide the single track plot if checkbox is unchecked
            self.singleTrackPlot.hide()

    def toggleAllTracksPlot(self):
        if self.displayAllTracksPlot_checkbox.isChecked():
            # show the single track plot if checkbox is checked
            self.allTracksPlot.show()
        else:
            # hide the single track plot if checkbox is unchecked
            self.allTracksPlot.hide()

    def toggleFilterOptions(self):
        if self.displayFilterOptions_checkbox.isChecked():
            # show the filter options window if checkbox is checked
            self.filterOptionsWindow.show()
        else:
            # hide the filter options window if checkbox is unchecked
            self.filterOptionsWindow.hide()

    def toggleCharts(self):
        if self.chartWindow == None:
            # create chart plot window and set items for column selectors
            self.chartWindow = ChartDock(self)
            self.chartWindow.xColSelector.setItems(self.colDict)
            self.chartWindow.yColSelector.setItems(self.colDict)
            self.chartWindow.colSelector.setItems(self.colDict)

            self.chartWindow.xcols = self.colDict
            self.chartWindow.ycols = self.colDict
            self.chartWindow.cols = self.colDict

        if self.displayCharts == False:
            # show the chart plot window if not currently displayed
            self.chartWindow.show()
            self.displayCharts = True
            self.showCharts_button.setText('Hide Charts')
        else:
            # hide the chart plot window if currently displayed
            self.chartWindow.hide()
            self.displayCharts = False
            self.showCharts_button.setText('Show Charts')

    def toggleDiffusionPlot(self):
        if self.diffusionWindow == None:
            # Create a new instance of the DiffusionPlotWindow class
            self.diffusionWindow = DiffusionPlotWindow(self)

        if self.displayDiffusionPlot == False:
            # Show the window if it is not already displayed
            self.diffusionWindow.show()
            self.displayDiffusionPlot = True
            self.showDiffusion_button.setText('Hide Diffusion')
        else:
            # Hide the window if it is already displayed
            self.diffusionWindow.hide()
            self.displayDiffusionPlot = False
            self.showDiffusion_button.setText('Show Diffusion')


    def toggleTrackPlotOptions(self):
        if self.trackPlotOptions == None:
            # Create a new instance of the DiffusionPlotWindow class
            self.trackPlotOptions = TrackPlotOptions(self)

        if self.displayTrackPlotOptions == False:
            # Show the window if it is not already displayed
            self.trackPlotOptions.show()
            self.displayTrackPlotOptions = True

        else:
            # Hide the window if it is already displayed
            self.trackPlotOptions.hide()
            self.displayTrackPlotOptions = False

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


    def togglePointMap(self):
        if self.togglePointMap_button.text() == 'Plot Point Map':
            # Check if filtered data is being used, if not use the original data
            if self.useFilteredData == False:
                df = self.data
            else:
                df = self.filteredData

            #add in unlinked points if displayed
            if self.displayUnlinkedPoints:
                df = df.append(self.data_unlinked)

            # Create a ScatterPlotItem and add it to the ImageView
            self.pointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
            self.pointMapScatter.setSize(2, update=False)
            self.pointMapScatter.setData(df['x'], df['y'])
            self.plotWindow.imageview.view.addItem(self.pointMapScatter)
            self.togglePointMap_button.setText('Hide Point Map')
        else:
            # Remove the ScatterPlotItem from the ImageView
            self.plotWindow.imageview.view.removeItem(self.pointMapScatter)
            self.togglePointMap_button.setText('Plot Point Map')


    def createStatsDFs(self):
            # Calculate mean and standard deviation for each track in the original data
            self.meanDF = self.data.groupby('track_number', as_index=False).mean()
            self.stdDF = self.data.groupby('track_number', as_index=False).std()

    def createStatsDFs_filtered(self):
            # Calculate mean and standard deviation for each track in the filtered data
            self.meanDF_filtered = self.filteredData.groupby('track_number', as_index=False).mean()
            self.stdDF_filtered = self.filteredData.groupby('track_number', as_index=False).std()

    def clearPlots(self):
        try:
            plt.close('all')
        except:
            pass
        return


    def displayOverlayOptions(self):
        if self.overlayWindow  == None:
            # Create a new instance of the DiffusionPlotWindow class
            self.overlayWindow = Overlay(self)

        if self.displayOverlay == False:
            # Show the window if it is not already displayed
            self.overlayWindow.show()
            self.displayOverlay = True
            self.overlayOption_button.setText('Hide Overlay')
        else:
            # Hide the window if it is already displayed
            self.overlayWindow.hide()
            self.displayOverlay = False
            self.overlayOption_button.setText('Show Overlay')

    def saveData(self):
        if self.useFilteredData == False:
            print('filter data first')
            g.alert('Filter data first')
            return

        # Prompt user to select a save path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")

        # Write the filtered data to a CSV file
        try:
            self.filteredData.to_csv(savePath)
            print('Filtered data saved to: {}'.format(savePath))
        except BaseException as e:
            print(e)
            print('Export of filtered data failed')

# Instantiate the LocsAndTracksPlotter class
locsAndTracksPlotter = LocsAndTracksPlotter()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











