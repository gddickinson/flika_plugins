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
from .volumeSlider_3DViewer import *
from .volumeSlider_Main_GUI import *

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm

### disable messages from PyQt ################
def handler(msg_type, msg_log_context, msg_string):
    pass

QtCore.qInstallMessageHandler(handler)
#####################################################


#########################################################################################
#############                  volumeViewer class                ########################
#########################################################################################
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.nVols = 1

        #create dtype dict
        self.dTypeDict = {
                'float16': np.float16,
                'float32': np.float32,
                'float64': np.float64,
                'int8': np.uint8,
                'int16': np.uint16,
                'int32': np.uint32,
                'int64': np.uint64                                  }

        self.dataType = dataType

        #create array order dict
        self.arrayDict = {
                '[0, 1, 2, 3]': [0, 1, 2, 3],
                '[0, 1, 3, 2]': [0, 1, 3, 2],
                '[0, 2, 1, 3]': [0, 2, 1, 3],
                '[0, 2, 3, 1]': [0, 2, 3, 1],
                '[0, 3, 1, 2]': [0, 3, 1, 2],
                '[0, 3, 2, 1]': [0, 3, 2, 1],
                '[1, 0, 2, 3]': [1, 0, 2, 3],
                '[1, 0, 3, 2]': [1, 0, 3, 2],
                '[1, 2, 0, 3]': [1, 2, 0, 3],
                '[1, 2, 3, 0]': [1, 2, 3, 0],
                '[1, 3, 0, 2]': [1, 3, 0, 2],
                '[1, 3, 2, 0]': [1, 3, 2, 0],
                '[2, 0, 1, 3]': [2, 0, 1, 3],
                '[2, 0, 3, 1]': [2, 0, 3, 1],
                '[2, 1, 0, 3]': [2, 1, 0, 3],
                '[2, 1, 3, 0]': [2, 1, 3, 0],
                '[2, 3, 0, 1]': [2, 3, 0, 1],
                '[2, 3, 1, 0]': [2, 3, 1, 0],
                '[3, 0, 1, 2]': [3, 0, 1, 2],
                '[3, 0, 2, 1]': [3, 0, 2, 1],
                '[3, 1, 0, 2]': [3, 1, 0, 2],
                '[3, 1, 2, 0]': [3, 1, 2, 0],
                '[3, 2, 0, 1]': [3, 2, 0, 1],
                '[3, 2, 1, 0]': [3, 2, 1, 0]
                }

        self.inputArrayOrder = [0, 3, 1, 2]
        self.displayArrayOrder = [3, 0, 1, 2]

        #array savepath
        self.savePath = ''
        return

    def startVolumeSlider(self, A=[], keepWindow=False):
        if A == []:
            #copy selected window
            self.A =  np.array(deepcopy(g.win.image),dtype=self.dataType)
            if keepWindow == False:
                g.win.close()
            
        else:
            #load numpy array
            self.B = A
            self.nFrames, self.nVols, self.x, self.y = self.B.shape
            self.dialogbox = Form2(camVolumeSlider)
            self.viewer = SliceViewer(camVolumeSlider, self.B)            
            return
            
        self.B = []
        #get shape
        self.nFrames, self.x, self.y = self.A.shape
        self.framesPerVol = int(self.nFrames/self.nVols)
        #setup display window
        self.displayWindow = Window(self.A,'Volume Slider Window')
        #open gui
        self.dialogbox = Form2(camVolumeSlider)
        self.dialogbox.show()        
        return

    def updateDisplay_volumeSizeChange(self):
        #remove final volume if it dosen't contain the full number of frames
        numberFramesToRemove = self.nFrames%self.getFramesPerVol()
        if numberFramesToRemove != 0:
            self.B = self.A[:-numberFramesToRemove,:,:]
        else:
            self.B = self.A

        self.nFrames, self.x, self.y = self.B.shape

        #reshape to 4D
        self.B = np.reshape(self.B, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
        self.displayWindow.imageview.setImage(self.B[0],autoLevels=False)
        return

    def updateDisplay_sliceNumberChange(self, index):
        displayIndex = self.displayWindow.imageview.currentIndex
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        self.displayWindow.imageview.setCurrentIndex(displayIndex)
        return

    def getNFrames(self):
        return self.nFrames

    def getNVols(self):
        return self.nVols

    def getFramesPerVol(self):
        return self.framesPerVol

    def updateVolsandFramesPerVol(self, nVols, framesPerVol):
        self.nVols = nVols
        self.framesPerVol = framesPerVol

    def closeEvent(self, event):
        event.accept()

    def getArrayShape(self):
        if self.B == []:
            return self.A.shape
        return self.B.shape

    def subtractBaseline(self):
        index = self.displayWindow.imageview.currentIndex
        baseline = self.dialogbox.getBaseline()
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.B = self.B - baseline
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)

    def averageByVol(self):
        index = self.displayWindow.imageview.currentIndex
        #TODO
        return

    def ratioDFF0(self):
        index = self.displayWindow.imageview.currentIndex
        ratioStart, ratioEnd , volStart, volEnd = self.dialogbox.getF0()
        
        if ratioStart >= ratioEnd:
            print('invalid F0 selection')
            return
        
        if volStart >= volEnd:
            print('invalid F0 Volume selection')
            return

        #get mean of vols used to make ratio
        ratioVol = self.B[:,ratioStart:ratioEnd,:,]
        ratioVolmean = np.mean(ratioVol, axis=1,keepdims=True)
        #get vols to be ratio-ed
        volsToRatio = self.B[:,volStart:volEnd,:,]
        #make ratio
        ratio = np.divide(volsToRatio, ratioVolmean, out=np.zeros_like(volsToRatio), where=ratioVolmean!=0, dtype=self.dataType)
        #replace original array data with raio values
        self.B[:,volStart:volEnd,:,] = ratio

        #self.B = np.divide(self.B, ratioVolmean, dtype=self.dataType)
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def exportToWindow(self):
        Window(np.reshape(self.B, (self.nFrames, self.x, self.y), order='F'))
        return

    def exportArray(self, vol='None'):
        if vol == 'None':
            np.save(self.savePath, self.B)
        else:
            f,v,x,y = self.B.shape
            np.save(self.savePath, self.B[:,vol,:,:].reshape((f,1,x,y)))
        return

    def getVolumeArray(self, vol):
        f,v,x,y = self.B.shape
        return self.B[:,vol,:,:].reshape((f,1,x,y))

    def getMaxPixel(self):
        if self.B == []:
            return np.max(self.A)
        else:
            return np.max(self.B)

    def setDType(self, newDataType):
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.dataType = self.dTypeDict[newDataType]
            self.B = self.B.astype(self.dataType)
            self.dialogbox.dataTypeText.setText(self.getDataType())
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def getDataType(self):
        return str(self.dataType).split(".")[-1].split("'")[0]

    def getArrayKeys(self):
        return list(self.arrayDict.keys())

    def getInputArrayOrder(self):
        return self.inputArrayOrder

    def getDisplayArrayOrder(self):
        return self.displayArrayOrder

    def setInputArrayOrder(self, value):
        self.inputArrayOrder = self.arrayDict[value]
        return

    def setDisplayArrayOrder(self, value):
        self.displayArrayOrder = self.arrayDict[value]
        return

    def multiplyByFactor(self, factor):
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.B = self.B * float(factor)
            print(self.B.shape)
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def startViewer(self):
        self.viewer = SliceViewer(camVolumeSlider, self.B)
        return

    def closeViewer(self):
        self.viewer.close()
        return

camVolumeSlider = CamVolumeSlider()



