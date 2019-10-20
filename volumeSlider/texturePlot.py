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

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm

#########################################################################################
#############            3D Texture plot            #####################################
#########################################################################################
class plotTexture(QtWidgets.QDialog):
    def __init__(self, data, parent = None):
        super(plotTexture, self).__init__()

        self.data = data

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.setWindowTitle('3D slice - texture plot')

        self.shape = self.data.shape

        #set intensity
        self.levels = (0, 1000)

        ## slice out three center planes, convert to RGBA for OpenGL texture
        self.slice1 = int(self.shape[0]/2)
        self.slice2 = int(self.shape[1]/2)
        self.slice3 = int(self.shape[2]/2)

        ## Create three image items from textures, add to view
        self.updateYZ()
        self.updateXZ()
        self.updateXY()
        self.addBorder(self.shape[0],self.shape[1],self.shape[2])


    def updateYZ(self):
        try:
            self.w.removeItem(self.v1)
        except:
            pass
        tex1 = pg.makeRGBA(self.data[self.slice1], levels=self.levels)[0]       # yz plane
        self.v1 = gl.GLImageItem(tex1)
        #self.v1.translate(-self.slice2, -self.slice3, int(self.shape[0]/2)-self.slice1)
        self.v1.translate(-self.slice2, -self.slice3, int(self.shape[0]/2)-self.slice1)        
        self.v1.rotate(90, 0,0,1)
        self.v1.rotate(-90, 0,1,0)
        self.w.addItem(self.v1)
        return

    def updateXZ(self):
        try:
            self.w.removeItem(self.v2)
        except:
            pass
        tex2 = pg.makeRGBA(self.data[:,self.slice2], levels=self.levels)[0]     # xz plane
        self.v2 = gl.GLImageItem(tex2)
        #self.v2.translate(-self.slice1, -self.slice3, int(self.shape[1]/2)-self.slice2)
        self.v2.translate(-self.slice1, -self.slice3, -int(self.shape[1]/2)+self.slice2)        
        self.v2.rotate(-90, 1,0,0)
        self.w.addItem(self.v2)
        return

    def updateXY(self):
        try:
            self.w.removeItem(self.v3)
        except:
            pass
        tex3 = pg.makeRGBA(self.data[:,:,self.slice3], levels=self.levels)[0]   # xy plane
        self.v3 = gl.GLImageItem(tex3)
        #self.v3.translate(-self.slice1, -self.slice2, int(self.shape[2]/2)-self.slice3)
        self.v3.translate(-self.slice1, -self.slice2, 0)        
        self.w.addItem(self.v3)
        return

    def addBorder(self,x,y,z):
        self.ax = GLBorderItem()
        self.ax.setSize(x=x,y=y,z=z)
        self.w.addItem(self.ax)

    def getSliceValues(self):
        return [self.slice1,self.slice2,self.slice3]

    def setSliceValues(self,slice1,slice2,slice3):
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        return

    def getShape(self):
        return self.shape

class textureDialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, shape, X,Y,Z, parent = None):
        super(textureDialog_win, self).__init__(parent)

        self.viewer = viewerInstance

        #window geometry
        windowGeometry(self, left=300, top=300, height=200, width=500)

        #sliders
        self.sliderX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.sliderX, minimum=0, maximum=shape[0], tickInterval=1, singleStep=1, value=X)

        self.sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.sliderY, minimum=0, maximum=shape[1], tickInterval=1, singleStep=1, value=Y)        

        self.sliderZ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.sliderZ, minimum=0, maximum=shape[2], tickInterval=1, singleStep=1, value=Z)

        #labels
        self.label_X = QtWidgets.QLabel("X:")
        self.label_Y = QtWidgets.QLabel("Y:")
        self.label_Z = QtWidgets.QLabel("Z:")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.label_X, 0, 0)
        layout.addWidget(self.label_Y, 1, 0)
        layout.addWidget(self.label_Z, 2, 0)   
        layout.addWidget(self.sliderX, 0, 1)
        layout.addWidget(self.sliderY, 1, 1)
        layout.addWidget(self.sliderZ, 2, 1)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Texture Control")

        #connect sliders
        self.sliderX.valueChanged.connect(self.sliderXValueChange)
        self.sliderY.valueChanged.connect(self.sliderYValueChange)
        self.sliderZ.valueChanged.connect(self.sliderZValueChange)
        return

    def sliderXValueChange(self, value):
        self.viewer.viewer.texturePlot.slice1 = value
        self.viewer.viewer.texturePlot.updateYZ()
        return

    def sliderYValueChange(self, value):
        self.viewer.viewer.texturePlot.slice2 = value
        self.viewer.viewer.texturePlot.updateXZ()
        return

    def sliderZValueChange(self, value):
        self.viewer.viewer.texturePlot.slice3 = value
        self.viewer.viewer.texturePlot.updateXY()
        return
