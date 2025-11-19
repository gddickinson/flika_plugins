#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:26:49 2023

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

from .helperFunctions import *


class FlowerPlotWindow():
    """
    This class creates a new window for the flower plot visualization.
    It initializes a new window using the pyqtgraph library, sets the size
    and title of the window, and assigns a reference to the main GUI window
    as an attribute.

    Args:
        mainGUI: A reference to the main GUI window.
    """
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        # Setup window
        self.win = pg.GraphicsLayoutWidget()
        self.win.resize(500, 500)
        self.win.setWindowTitle('Flower Plot')

        # Add plot to window and set attributes
        self.plt = self.win.addPlot(title='plot')
        self.plt.setAspectLocked()
        self.plt.showGrid(x=True, y=True)
        self.plt.setXRange(-10,10)
        self.plt.setYRange(-10,10)
        self.plt.getViewBox().invertY(True)

        # Set labels for axes
        self.plt.setLabel('left', 'y', units ='pixels')
        self.plt.setLabel('bottom', 'x', units ='pixels')

        # List to store plot items representing tracks
        self.pathitems = []

    def clearTracks(self):
        # Remove all plot items representing tracks
        self.plt.clear()
        self.pathitems = []

    def show(self):
        # Show the window
        self.win.show()

    def close(self):
        # Close the window
        self.win.close()

    def hide(self):
        # Hide the window
        self.win.hide()
