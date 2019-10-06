import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
import os
from os import listdir
from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
from copy import deepcopy
from numpy import moveaxis
from skimage.transform import rescale
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from qtpy.QtCore import Signal

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .helperFunctions import *
from .pyqtGraph_classOverwrites import *
from .scalebar_classOverwrite import Scale_Bar_volumeView
from .histogramExtension import HistogramLUTWidget_Overlay
from .texturePlot import *

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm


class OverlayOptions(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(OverlayOptions, self).__init__(parent)
        
        self.viewer = viewerInstance

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        self.opacity = 50
        self.lut = 'grey'
        self.overlay1 = None
        self.overlay2 = None
        self.overlay3 = None
        self.overlay4 = None        

        #buttons
        self.linkOverlayButton = QtWidgets.QPushButton("Set All Overlays") 
        self.linkOverlayButton.clicked.connect(self.setOverlays)

        self.transferButton1 = QtWidgets.QPushButton("Transfer Top Main") 
        self.transferButton1.clicked.connect(self.transfer1)        
        
        self.transferButton2 = QtWidgets.QPushButton("Transfer Top Overlay") 
        self.transferButton2.clicked.connect(self.transfer2)   


        #combo boxes
        self.cmSelectorBox = QtWidgets.QComboBox()
        #self.cmSelectorBox.addItems(["inferno", "magma", "plasma","viridis","Reds","Greens","Blues", "binary","bone","Greys",
        #                             "hot","Set1","RdBu","Accent","autumn","jet","hsv","Spectral"])
        self.cmSelectorBox.addItems(['grey','thermal','flame','yellowy','bipolar','spectrum','cyclic','greyclip'])
        self.cmSelectorBox.setCurrentIndex(0)
        self.cmSelectorBox.currentIndexChanged.connect(self.setColorMap)

        self.modeSelectorBox = QtWidgets.QComboBox()
        self.modeSelectorBox.addItems(["Source Overlay", "Overlay", "Plus", "Mutiply"])
        self.modeSelectorBox.setCurrentIndex(0)
        self.modeSelectorBox.currentIndexChanged.connect(self.setMode)

        #sliders
        self.sliderOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderOpacity.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sliderOpacity.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sliderOpacity.setMinimum(0)
        self.sliderOpacity.setMaximum(100)
        self.sliderOpacity.setTickInterval(10)
        self.sliderOpacity.setSingleStep(1)
        self.sliderOpacity.setValue(self.opacity)
        self.sliderOpacity.valueChanged.connect(self.setOpacity)

        #checkboxes
        self.linkCheck = QtWidgets.QCheckBox()
        self.linkCheck.setChecked(True)
        self.linkCheck.stateChanged.connect(self.linkCheckValueChange)
        
        #labels
        self.labelCM = QtWidgets.QLabel("Color Map Presets")
        self.labelOverlayMode = QtWidgets.QLabel("Overlay Mode")
        self.labelOpacity = QtWidgets.QLabel("Opacity (%)")
        self.labelLinkCheck = QtWidgets.QLabel("Link Histogram Sliders")         
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.labelCM, 1, 0)
        layout.addWidget(self.cmSelectorBox, 1, 1)
        layout.addWidget(self.linkOverlayButton, 1, 2)  
        
        layout.addWidget(self.transferButton1, 3, 0)  
        layout.addWidget(self.transferButton2, 3, 1)          
        
        layout.addWidget(self.labelOverlayMode, 5, 0)
        layout.addWidget(self.modeSelectorBox, 5, 1)
        layout.addWidget(self.labelOpacity, 6, 0)
        layout.addWidget(self.sliderOpacity, 6, 1,1,4)  
        layout.addWidget(self.labelLinkCheck, 7, 0)    
        layout.addWidget(self.linkCheck, 7, 1)          

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Overlay Options")


        return

    def setColorMap(self):
        self.lut = self.cmSelectorBox.currentText()
        return

    def setOverlays(self):     
        self.viewer.viewer.gradientPreset = self.lut
        self.viewer.viewer.usePreset = True
        self.viewer.viewer.updateAllOverlayWins()
        self.viewer.viewer.usePreset = False
        return

    def transfer1(self):
        lut = self.viewer.viewer.imv1.getHistogramWidget().item.gradient.saveState()
        levels = self.viewer.viewer.imv1.getHistogramWidget().item.getLevels()
        #set lut
        self.viewer.viewer.imv2.getHistogramWidget().item.gradient.restoreState(lut)
        self.viewer.viewer.imv3.getHistogramWidget().item.gradient.restoreState(lut)
        self.viewer.viewer.imv4.getHistogramWidget().item.gradient.restoreState(lut)
        self.viewer.viewer.imv6.getHistogramWidget().item.gradient.restoreState(lut)    
        #set levels
        self.viewer.viewer.imv2.getHistogramWidget().item.setLevels(levels[0],levels[1])
        self.viewer.viewer.imv3.getHistogramWidget().item.setLevels(levels[0],levels[1])
        self.viewer.viewer.imv4.getHistogramWidget().item.setLevels(levels[0],levels[1])
        self.viewer.viewer.imv6.getHistogramWidget().item.setLevels(levels[0],levels[1])           
        return

    def transfer2(self):
        self.viewer.viewer.sharedState = self.viewer.viewer.bgItem_imv1.hist_luttt.item.gradient.saveState()
        self.viewer.viewer.sharedLevels = self.viewer.viewer.bgItem_imv1.hist_luttt.item.getLevels()
        self.viewer.viewer.useSharedState = True
        self.viewer.viewer.updateAllOverlayWins()
        self.viewer.viewer.useSharedState = False
        return

    
    def setMode(self):
        self.viewer.viewer.setOverlayMode(self.modeSelectorBox.currentText())
        self.viewer.viewer.updateAllOverlayWins()
        return
    
    def setOpacity(self,value):
        self.opacity = value
        self.viewer.viewer.OverlayOPACITY = (value/100)
        self.viewer.viewer.updateAllOverlayWins()
        return

    def linkCheckValueChange(self, value):
        self.viewer.viewer.histogramsLinked = value
        return
    
