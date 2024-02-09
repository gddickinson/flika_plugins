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

from scipy.ndimage import center_of_mass, gaussian_filter, binary_fill_holes, binary_closing, label, find_objects, zoom
from skimage.filters import threshold_otsu
from scipy.ndimage import rotate as nd_rotate
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
import skimage.io as skio

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

crossbowShape = np.array([
 [0,0,0,1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,1,1,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,1,1,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0],
 [0,0,0,1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0,0,0,0]])

hShape = np.array([
 [0,0,0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,0,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,1,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,0,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [0,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,0,0],
 [0,0,1,1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,1,0,0],
 [0,0,0,1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,0,0,0]])

yShape = np.array([
[1,1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,1],
[1,1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,1],
[1,1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,1],
[0,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0],
[0,0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,0],
[0,0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0],
[0,0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,0],
[0,0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,0],
[0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,0],
[0,0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0]
])

squareShape = np.array([
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

discShape = np.array([
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
 [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])




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



class FileSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a file to save.  Once you do, it sets self.filename and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*', mainGUI=None):
        QWidget.__init__(self)

        self.mainGUI = mainGUI

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



class TemplateROI(pg.RectROI):
    def __init__(self, template = 'disc', display= None, *args, **kwds):
        pg.RectROI.__init__(self, aspectLocked=True, *args, **kwds)
        self.addRotateHandle([1,0], [0.5, 0.5])
        self.template = template

        self.outlinePen = pg.mkPen('y', width=1, style=Qt.DashLine)
        self.centerPointPen = pg.mkPen('g', width=20)


    def paint(self, p, opt, widget):
        radius = self.getState()['size'][1]
        if self.template == 'disc':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            #get center of ROI
            br = self.boundingRect()
            center = br.center()
            #draw shape
            p.drawEllipse(center,int(radius/2),int(radius/2))


        elif self.template == 'square':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawRect(int(0), int(0), int(radius), int(radius))

        elif self.template == 'crossbow':
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            #p.drawLine(Point(radius/2, -radius/2), Point(radius/2, radius/2))
            p.drawLine(Point(0, radius/2), Point(radius, radius/2))

            pen2 = QPen()
            pen2.setWidth(20)
            pen2.setColor(Qt.green)
            p.setPen(pen2)
            p.drawPoint(int(radius-10),int(radius/2))


        elif self.template == 'y-shape':
            #get equilatoral triangle vertices
            angles = np.linspace(0, np.pi * 4 / 3, 3)
            verticies = (np.array((np.sin(angles), np.cos(angles))).T * radius) / 2.0
            #scale by roi size
            verticies = np.array([[i[0] + radius/2, i[1] + radius/2] for i in verticies])
            #create poly
            poly = QPolygonF()
            for pt in verticies:
                poly.append(QPointF(*pt))

            #draw shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawPolygon(poly)


        elif self.template == 'h-shape':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawLine(Point(0, radius/5), Point(radius, radius/5))
            p.drawLine(Point(0, radius-(radius/5)), Point(radius, radius-(radius/5)))


        #draw center point
        p.setPen(self.centerPointPen)
        p.drawPoint(int(radius/2),int(radius/2))


        #outline
        # Note: don't use self.boundingRect here, because subclasses may need to redefine it.
        r = QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(self.outlinePen)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

        self.updateDisplay()

    def getPattern(self):
        return self.template

    def getCenter(self):
        posX,posY = self.pos()
        sizeX,sizeY = self.size()
        return (int(posX+sizeX/2), int(posY+sizeY/2))


    def _rotate(self, angle):
        newAngle = self.angle()+angle
        newState = self.stateCopy()
        #origin = self.transformOriginPoint()

        br = self.boundingRect()
        center = br.center()

        self.setTransformOriginPoint(center)
        self.setRotation(newAngle)
        self.setPos(newState['pos'], update=True)
        self.setAngle(newAngle, update=True)


        self.stateChanged(finish=True)

    def stateRect(self, state):
        r = QtCore.QRectF(0, 0, state['size'][0], state['size'][1])
        tr = QtGui.QTransform()
        tr.rotate(state['angle'])
        r = tr.mapRect(r)
        return r.adjusted(state['pos'][0], state['pos'][1], state['pos'][0], state['pos'][1])

    def getAngle(self):
        return self.angle() % 360

    def getSize(self):
        return self.size()[0]

    def addDisplay(self, display):
        self.display = display

    def updateDisplay(self):
        if self.display != None:
            x,y = self.getCenter()
            angle = self.getAngle()
            size = self.getSize()
            self.display.centerPos_text.setText('{},{}'.format(x,y))
            self.display.angle_text.setText('{}'.format(round(angle, 2)))
            self.display.size_text.setText('{}'.format(round(size, 2)))





'''
#####################################################################################################################################
######################################   Main LOCSANDTRACKSPLOTTER CLASS   ##########################################################
#####################################################################################################################################
'''

class TranslateAndScale(BaseProcess_noPriorWindow):
    """
    Generate alignment for rotation, translation and scaling of point data
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['translateAndScale'] is None or 'template' not in g.settings['translateAndScale']:
            s = dict()
            s['template'] = 'square'
            g.settings['translateAndScale'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, template,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        pixelSize: int - pixel size of image data

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['translateAndScale']['template'] = template


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
        # Get settings for locsAndTracksPlotter
        s=g.settings['translateAndScale']
        self.dataWindow = WindowSelector()
        self.pixelSize = None

        self.currentTemplate = None

        self.startButton = QPushButton('Restart Alignment')
        self.startButton.pressed.connect(self.startAlign)

        self.endButton = QPushButton('Save Alignment')
        self.endButton.pressed.connect(self.endAlign)

        self.clearButton = QPushButton('Clear Alignment')
        self.clearButton.pressed.connect(self.clearTemplate)

        self.templateBox = pg.ComboBox()
        self.templates = {'Disc': 'disc',
                          'Square': 'square',
                          'Crossbow': 'crossbow',
                          'Y-shape': 'y-shape',
                          'H-shape': 'h-shape'
                          }
        self.templateBox.setItems(self.templates)

        index = self.templateBox.findText(s['template'], Qt.MatchFixedString)
        if index >= 0:
            self.templateBox.setCurrentIndex(index)

        self.templateBox.activated.connect(self.update)

        #data file selector
        self.getFile = FileSelector(filetypes='*.csv', mainGUI=self)

        #connections
        self.getFile.valueChanged.connect(self.loadData)

        self.transformDataButton = QPushButton('Transform data')
        self.transformDataButton.pressed.connect(self.transformData)

        self.saveTransformDataButton = QPushButton('Save Transform')
        self.saveTransformDataButton.pressed.connect(self.saveTransformedData)


        self.items.append({'name': 'dataWindow', 'string': 'Image Window', 'object': self.dataWindow})
        self.items.append({'name': 'filename', 'string': 'Data File', 'object': self.getFile})
        self.items.append({'name': 'template', 'string': 'Choose template', 'object': self.templateBox})
        self.items.append({'name': 'startButton', 'string': '', 'object': self.startButton})
        self.items.append({'name': 'endButton', 'string': '', 'object': self.endButton})
        self.items.append({'name': 'clearButton', 'string': '', 'object': self.clearButton})
        self.items.append({'name': 'clearButton', 'string': '', 'object': self.clearButton})
        self.items.append({'name': 'transformButton', 'string': 'Transform data', 'object': self.transformDataButton})
        self.items.append({'name': 'saveTransformButton', 'string': 'Save Transformed data', 'object': self.saveTransformDataButton})

        super().gui()

        #initialize display window and hide it
        self.displayParams = DisplayParams(self)
        self.displayParams.show()

        self.df = None

        self.filename = ''
        self.dataDF = None
        self.plotWindow = None

        self.pointMapScatter = None

        return

    def startAlign(self):
        roiSize = [400,400]
        startPosition = [80, 50]
        #initiate template
        if self.currentTemplate == None:
            template = TemplateROI(pos=startPosition, size=roiSize, template = self.getValue('template'), pen=(0,9))
            self.currentTemplate = template

        self.currentTemplate.addDisplay(self.displayParams)
        self.getValue('dataWindow').imageview.addItem(self.currentTemplate)


        #autodetect micropattern
        img = self.getValue('dataWindow').image

        #print(img.shape)

        #get max projection if stack
        if len(img.shape) > 2:
            img = np.max(img,0)

        #print(img.shape)

        #blur
        img = gaussian_filter(img, 3)

        #set threshold
        threshold = threshold_otsu(img)

        # Create a mask
        mask = np.zeros_like(img)
        mask[img > threshold] = 1

        # Apply the mask to the image stack
        img = img * mask

        #img = binary_fill_holes(img).astype(int)
        img = binary_closing(img, iterations=50).astype(int)

        #blur again
        img = gaussian_filter(img, 3)

        # Find the location of micropattern (assuming 1 per image)
        objs = find_objects(img)

        # Get the height and width
        height = int(objs[0][0].stop - objs[0][0].start)
        #width = int(objs[0][1].stop - objs[0][1].start)

        center = center_of_mass(img)
        if len(center) == 2:
            x,y = center[0], center[1]
        else:
            x,y = center[1], center[2]
        #print(x,y)

        #display autodetected micropattern binary
        #Window((img))

        #attempt to scale
        self.currentTemplate.scale(height/self.currentTemplate.size()[0])

        #attempt to center on micropattern
        self.currentTemplate.setPos((x-(self.currentTemplate.size()[0]/2),y-(self.currentTemplate.size()[0]/2)))

        #use phasediff to guess rotation
        #generate image for comparison
        rotated = np.zeros_like(img)

        if self.currentTemplate.template == 'square':
            print(self.currentTemplate.template)
            shapeSize = 300
            rotShape = zoom(squareShape, shapeSize/squareShape.shape[0])
            #rotShape = np.ones((shapeSize ,shapeSize )),

        elif self.currentTemplate.template == 'h-shape':
            print(self.currentTemplate.template)
            shapeSize = 350
            rotShape = zoom(hShape, shapeSize/hShape.shape[0])

        elif self.currentTemplate.template == 'y-shape':
            print(self.currentTemplate.template)
            shapeSize = 450
            rotShape = zoom(yShape, shapeSize/yShape.shape[0])

        elif self.currentTemplate.template == 'crossbow':
            print(self.currentTemplate.template)
            shapeSize = 350
            rotShape = zoom(crossbowShape, shapeSize/yShape.shape[0])

        elif self.currentTemplate.template == 'disc':
            print(self.currentTemplate.template)
            shapeSize = 350
            rotShape = zoom(discShape, shapeSize/discShape.shape[0])
            return

        self.currentTemplate.setSize([shapeSize,shapeSize])

        startx = int(img.shape[0]/2-(shapeSize/2) )
        starty = int(img.shape[1]/2-(shapeSize/2) )
        endx = startx+rotShape.shape[0]
        endy = startx+rotShape.shape[1]

        # Add the value 10 to the index (1, 1)
        rotated[startx:endx,starty:endy] = rotShape

        shifts, error, phasediff = phase_cross_correlation(img,
                                                           rotated,
                                                           normalization=None)
        angleToRotate = shifts[0]

        print('rotation: {}'.format(shifts[0]))

        #attempt to rotate to match micropattern template

        #move rotation handle to rotate roi - using this approach as pyqt rotate function appears to have a bug in which it only rotates around origin
        handles = self.currentTemplate.getHandles()
        print(handles)
        handlesPos = self.currentTemplate.getLocalHandlePositions()
        print(handlesPos)
        rotX,rotY = (handlesPos[1][1][0],handlesPos[1][1][1])
        print(rotX,rotY)
        newX, newY = rotate_around_point(rotX,rotY, angleToRotate, origin=(0,0))
        print(newX,newY)

        self.currentTemplate.movePoint(handles[1],QPointF(newX,newY))

        #Window(rotated)

        return

    def endAlign(self):
        #export rotation file
        fileName = self.getValue('dataWindow').filename

        d = {'center_x' : self.currentTemplate.getCenter()[0],
        'center_y': self.currentTemplate.getCenter()[1],
        'angle' : self.currentTemplate.getAngle(),
        'size' : self.currentTemplate.getSize(),
        'pattern' : self.currentTemplate.getPattern(),
        'file' : fileName}

        self.df = pd.DataFrame(data=d, index=[0])


        saveName = os.path.splitext(fileName)[0] + '_align.txt'
        self.df.to_csv(saveName)
        print('\n alignment file exported to {}'.format(saveName))


        return

    def clearTemplate(self):
        #clear roi
        if self.currentTemplate != None:
            self.getValue('dataWindow').imageview.removeItem(self.currentTemplate)
        self.currentTemplate = None

    def update(self):
        self.clearTemplate()
        self.startAlign()

    def loadData(self):
        # Set the plot window to the global window instance
        img = self.getValue('dataWindow').image
        if len(img.shape) > 2:
            img = np.max(img,0)

        self.plotWindow = Window(img)

        # Get the filename from the GUI
        self.filename = self.getFile.value()

        # Load the data from the selected file using Pandas
        self.data = pd.read_csv(self.filename)
        # create new df to store transform points
        self.transformDF = self.data[['x','y']]

        #plot data on image
        self.plotDataPoints()

        #initiate alignment
        self.update()


    def plotDataPoints(self):
        if self.pointMapScatter != None:
            self.plotWindow.imageview.view.removeItem(self.pointMapScatter)


        # Create a ScatterPlotItem and add it to the ImageView
        self.pointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
        self.pointMapScatter.setSize(2, update=False)
        self.pointMapScatter.setData(self.transformDF['x'], self.transformDF['y'])
        self.plotWindow.imageview.view.addItem(self.pointMapScatter)



    def transformData(self):
        target_X = 0
        target_Y = 0
        target_angle = 0
        target_size = 400

        #get angle to rotate
        angle = target_angle + self.currentTemplate.getAngle()
        # Get the center of the dataframe
        center = self.data[['x', 'y']].mean()
        #rotate data points around center
        self.transformDF['x'],self.transformDF['y'] = rotate_around_point(self.data['x'],self.data['y'], angle, origin=center)

        #center and rotate img
        img = self.getValue('dataWindow').image
        if len(img.shape) > 2:
            img = np.max(img,0)

        #pad image with 2000x2000 pixels to provide room for cropping to center of micropattern
        padSize = 2000
        img = np.pad(img, padSize, mode='constant')

        #crop micropattern based on point data center
        crop_width = 400
        img = img[int(center[0]-crop_width)+padSize:int(center[0]+crop_width+padSize), int(center[1]-crop_width)+padSize: int(center[1]+crop_width)+padSize]

        #pad edges
        pad_width = 400
        img = np.pad(img, pad_width, mode='constant')

        #rotate image
        img= nd_rotate(img, angle= -angle, reshape=False)

        #crop img
        #img = img[padX:padX+h,padY:padY+w]

        #center points on image
        imgCenter = (img.shape[0]/2, img.shape[1]/2)
        self.transformDF['x'] = self.transformDF['x'] + imgCenter[0] - center[0]
        self.transformDF['y'] = self.transformDF['y'] + imgCenter[1] - center[1]

        self.plotWindow.imageview.setImage(img)

        #replot
        self.plotDataPoints()

        return

    def saveTransformedData(self):
        #export transformed img and data points
        baseName = os.path.splitext(self.filename)[0]

        #export tif
        exportIMG = self.plotWindow.imageview.image
        saveName_img = baseName + '_transform.tif'
        skio.imsave(saveName_img, exportIMG)
        print('transformed image file saved as: {}'.format(saveName_img))

        #export transformDF
        exportDF = self.data
        exportDF['x_transformed'] = self.transformDF['x']
        exportDF['y_transformed'] = self.transformDF['y']
        saveName_DF = baseName + '_transform.csv'

        exportDF.to_csv(saveName_DF, index=None)

        print('transformed point file saved as: {}'.format(saveName_DF))

        return


# Instantiate the LocsAndTracksPlotter class
translateAndScale = TranslateAndScale()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











