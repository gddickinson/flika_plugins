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
from .tiffLoader import openTiff

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm
import gc

#########################################################################################
#############                  volumeViewer GUI setup            ########################
#########################################################################################
class Form2(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(Form2, self).__init__(parent)
        self.batch = False
        self.viewer = viewerInstance

        self.s = g.settings['volumeSlider']

        self.arraySavePath = self.viewer.savePath
        self.arrayImportPath = "None"

        #window geometry
        windowGeometry(self, left=300, top=300, height=600, width=400)

        self.slicesPerVolume = self.s['slicesPerVolume']
        
        self.slicesDeletedPerVolume = self.s['slicesDeletedPerVolume']       
        
        self.baselineValue = self.s['baselineValue']
        self.f0Start = self.s['f0Start']
        self.f0End = self.s['f0End']
        self.f0VolStart = self.s['f0VolStart']
        self.f0VolEnd = self.s['f0VolEnd']                
        self.multiplicationFactor = self.s['multiplicationFactor']
        self.currentDataType = self.s['currentDataType']
        self.newDataType = self.s['newDataType']
        self.inputArrayOrder = self.s['inputArrayOrder']
        self.displayArrayOrder = self.s['displayArrayOrder'] = 16
        self.theta = self.s['theta']
        self.shiftFactor = self.s['shiftFactor']
        self.trim_last_frame = self.s['trimLastFrame']

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #")
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,self.viewer.getNFrames())
        self.SpinBox1.setValue(0)

        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,self.viewer.getNFrames())
        if self.slicesPerVolume < self.viewer.getNFrames():
            self.SpinBox2.setValue(self.slicesPerVolume)
        else:
            self.SpinBox2.setValue(1)

        self.spinLabel13 = QtWidgets.QLabel("# of slices removed per volume: ")
        self.SpinBox13 = QtWidgets.QSpinBox()
        self.SpinBox13.setRange(0,self.viewer.getNFrames())
        if self.slicesDeletedPerVolume < self.viewer.getNFrames():
            self.SpinBox13.setValue(self.slicesDeletedPerVolume)
        else:
            self.SpinBox13.setValue(0)


        self.spinLabel4 = QtWidgets.QLabel("baseline value: ")
        self.SpinBox4 = QtWidgets.QSpinBox()
        self.SpinBox4.setRange(0,self.viewer.getMaxPixel())
        if self.baselineValue < self.viewer.getMaxPixel():
            self.SpinBox4.setValue(self.baselineValue)
        else:
            self.SpinBox4.setValue(0)           

        self.spinLabel6 = QtWidgets.QLabel("F0 start volume: ")
        self.SpinBox6 = QtWidgets.QSpinBox()
        self.SpinBox6.setRange(0,self.viewer.getNVols())
        if self.f0Start < self.viewer.getNVols():
            self.SpinBox6.setValue(self.f0Start)
        else:
            self.SpinBox6.setValue(0)            

        self.spinLabel7 = QtWidgets.QLabel("F0 end volume: ")
        self.SpinBox7 = QtWidgets.QSpinBox()
        self.SpinBox7.setRange(0,self.viewer.getNVols())
        if self.f0End < self.viewer.getNVols():
            self.SpinBox7.setValue(self.f0End)
        else:
            self.SpinBox7.setValue(0)

        self.spinLabel8 = QtWidgets.QLabel("factor to multiply by: ")
        self.SpinBox8 = QtWidgets.QSpinBox()
        self.SpinBox8.setRange(0,10000)
        self.SpinBox8.setValue(self.multiplicationFactor)

        self.spinLabel9 = QtWidgets.QLabel("theta: ")
        self.SpinBox9 = QtWidgets.QSpinBox()
        self.SpinBox9.setRange(0,360)
        self.SpinBox9.setValue(self.theta)

        self.spinLabel10 = QtWidgets.QLabel("shift factor: ")
        self.SpinBox10 = QtWidgets.QSpinBox()
        self.SpinBox10.setRange(0,100)
        self.SpinBox10.setValue(self.shiftFactor)


        self.SpinBox11 = QtWidgets.QSpinBox()
        self.SpinBox11.setRange(0,self.viewer.getNVols())
        if self.f0Start < self.viewer.getNVols():
            self.SpinBox11.setValue(self.f0VolStart)
        else:
            self.SpinBox6.setValue(0)            


        self.SpinBox12 = QtWidgets.QSpinBox()
        self.SpinBox12.setRange(0,self.viewer.getNVols())
        if self.f0End <= self.viewer.getNVols():
            self.SpinBox12.setValue(self.f0VolEnd)
        else:
            self.SpinBox12.setValue(self.viewer.getNVols())


        #sliders
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.slider1, minimum=0, maximum=self.viewer.getNFrames(), tickInterval=1, singleStep=1, value=0)


        #ComboBox
        self.dTypeSelectorBox = QtWidgets.QComboBox()
        self.dTypeSelectorBox.addItems(["float16", "float32", "float64","int8","int16","int32","int64"])
        self.inputArraySelectorBox = QtWidgets.QComboBox()
        self.inputArraySelectorBox.addItems(self.viewer.getArrayKeys())
        self.inputArraySelectorBox.setCurrentIndex(4)
        self.inputArraySelectorBox.currentIndexChanged.connect(self.inputArraySelectionChange)

        self.displayArraySelectorBox = QtWidgets.QComboBox()
        self.displayArraySelectorBox.addItems(self.viewer.getArrayKeys())
        self.displayArraySelectorBox.setCurrentIndex(18)
        self.displayArraySelectorBox.currentIndexChanged.connect(self.displayArraySelectionChange)

        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")
        self.button2 = QtWidgets.QPushButton("Set Slices")
        #self.button3 = QtWidgets.QPushButton("Average Volumes")
        self.button4 = QtWidgets.QPushButton("subtract baseline")
        self.button5 = QtWidgets.QPushButton("run DF/F0")
        self.button6 = QtWidgets.QPushButton("export to Window")
        self.button7 = QtWidgets.QPushButton("set data Type")
        self.button8 = QtWidgets.QPushButton("multiply")
        self.button9 = QtWidgets.QPushButton("export to array")        

        self.button12 = QtWidgets.QPushButton("open 3D viewer")
        self.button13 = QtWidgets.QPushButton("close 3D viewer")
        
        self.button14 = QtWidgets.QPushButton("load new file")
        
        self.button15 = QtWidgets.QPushButton("Set overlay to current volume")       
        

        #labels
        self.ratioVolStartLabel = QtWidgets.QLabel("ratio start volume: ")
        self.ratioVolEndLabel = QtWidgets.QLabel("ratio End volume: ")        
        self.volumeLabel = QtWidgets.QLabel("# of volumes: ")
        self.volumeText = QtWidgets.QLabel("  ")
        
        self.currentVolumeLabel = QtWidgets.QLabel("current volume: ")
        self.currentVolumeText = QtWidgets.QLabel("0")        
        

        self.shapeLabel = QtWidgets.QLabel("array shape: ")
        self.shapeText = QtWidgets.QLabel(str(self.viewer.getArrayShape()))

        self.dataTypeLabel = QtWidgets.QLabel("current data type: ")
        self.dataTypeText = QtWidgets.QLabel(str(self.viewer.getDataType()))
        self.dataTypeChangeLabel = QtWidgets.QLabel("new data type: ")

        self.inputArrayLabel = QtWidgets.QLabel("input array order: ")
        self.displayArrayLabel = QtWidgets.QLabel("display array order: ")

        self.arraySavePathLabel = QtWidgets.QLabel(str(self.arraySavePath))

        self.trim_last_frameLabel = QtWidgets.QLabel("Trim Last Frame: ")
        self.trim_last_frame_checkbox = CheckBox()
        self.trim_last_frame_checkbox.setChecked(self.trim_last_frame)
        self.trim_last_frame_checkbox.stateChanged.connect(self.trim_last_frameClicked)


        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)
        layout.addWidget(self.SpinBox1, 1, 1)

        layout.addWidget(self.slider1, 2, 0, 2, 5)

        layout.addWidget(self.spinLabel2, 4, 0)
        layout.addWidget(self.SpinBox2, 4, 1)
        layout.addWidget(self.button2, 4, 4)

        layout.addWidget(self.spinLabel13, 4, 2)
        layout.addWidget(self.SpinBox13, 4, 3)


        layout.addWidget(self.spinLabel4, 6, 0)
        layout.addWidget(self.SpinBox4, 6, 1)
        layout.addWidget(self.button4, 6, 2)

        layout.addWidget(self.spinLabel6, 7, 0)
        layout.addWidget(self.SpinBox6, 7, 1)
        layout.addWidget(self.spinLabel7, 7, 2)
        layout.addWidget(self.SpinBox7, 7, 3)
        layout.addWidget(self.button5, 7, 4)

        layout.addWidget(self.ratioVolStartLabel, 8, 0)
        layout.addWidget(self.SpinBox11, 8, 1)
        layout.addWidget(self.ratioVolEndLabel, 8, 2)
        layout.addWidget(self.SpinBox12, 8, 3)

        layout.addWidget(self.volumeLabel, 9, 0)
        layout.addWidget(self.volumeText, 9, 1)
        
        layout.addWidget(self.currentVolumeLabel, 9, 2)
        layout.addWidget(self.currentVolumeText, 9, 3)        
        
        layout.addWidget(self.shapeLabel, 10, 0)
        layout.addWidget(self.shapeText, 10, 1)

        layout.addWidget(self.spinLabel8, 11, 0)
        layout.addWidget(self.SpinBox8, 11, 1)
        layout.addWidget(self.button8, 11, 2)
        
        layout.addWidget(self.button15, 12, 0)       
        

        layout.addWidget(self.dataTypeLabel, 13, 0)
        layout.addWidget(self.dataTypeText, 13, 1)
        layout.addWidget(self.dataTypeChangeLabel, 13, 2)
        layout.addWidget(self.dTypeSelectorBox, 13,3)
        layout.addWidget(self.button7, 13, 4)

        layout.addWidget(self.button6, 15, 0)
        layout.addWidget(self.button1, 15, 4)
        layout.addWidget(self.button6, 15, 0)
        layout.addWidget(self.button1, 15, 4)
        layout.addWidget(self.button9, 16, 0)
      
        layout.addWidget(self.arraySavePathLabel, 16, 1, 1, 4)      
        
        
        layout.addWidget(self.spinLabel9, 18, 0)
        layout.addWidget(self.SpinBox9, 18, 1)

        layout.addWidget(self.spinLabel10, 19, 0)
        layout.addWidget(self.SpinBox10, 19, 1)
        layout.addWidget(self.trim_last_frameLabel, 20, 0)
        layout.addWidget(self.trim_last_frame_checkbox, 20, 1)

        layout.addWidget(self.inputArrayLabel, 21, 0)
        layout.addWidget(self.inputArraySelectorBox, 21, 1)

        layout.addWidget(self.displayArrayLabel, 21, 2)
        layout.addWidget(self.displayArraySelectorBox, 21, 3)

        layout.addWidget(self.button12, 22, 0)
        layout.addWidget(self.button13, 22, 1)
        layout.addWidget(self.button14, 22, 2)  

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Volume Slider GUI")

        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox9.valueChanged.connect(self.setTheta)
        self.SpinBox10.valueChanged.connect(self.setShiftFactor)

        #connect buttons
        self.button1.clicked.connect(self.autoLevel)
        self.button2.clicked.connect(self.updateVolumeValue)
        #self.button3.clicked.connect(self.averageByVol)
        self.button4.clicked.connect(self.subtractBaseline)
        self.button5.clicked.connect(self.ratioDFF0)
        self.button6.clicked.connect(self.exportToWindow)
        self.button7.clicked.connect(self.dTypeSelectionChange)
        self.button8.clicked.connect(self.multiplyByFactor)
        self.button9.clicked.connect(self.exportArray)            
        self.button12.clicked.connect(self.startViewer)
        self.button13.clicked.connect(self.closeViewer)
        self.button14.clicked.connect(lambda: self.loadNewFile(''))   
        self.button15.clicked.connect(self.setOverlay)

        return

     #volume changes with slider & spinbox
    def loadNewFile(self, fileName):
        if self.batch == False:
            fileName = QtWidgets.QFileDialog.getOpenFileName(self,'Open File', os.path.expanduser("~/Desktop"), 'tiff files (*.tif *.tiff)')
            fileName = str(fileName[0])
            
        A, _, _ = openTiff(fileName)
        self.viewer.updateVolumeSlider(A)
        self.viewer.displayWindow.imageview.setImage(A)        
        return


    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return

    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        self.viewer.updateDisplay_sliceNumberChange(value)
        return

    def autoLevel(self):
        self.viewer.displayWindow.imageview.autoLevels()
        return

    def updateVolumeValue(self):
        self.slicesPerVolume = self.SpinBox2.value()
        noVols = int(self.viewer.getNFrames()/self.slicesPerVolume)
        self.framesToDelete = self.SpinBox13.value()
        self.viewer.updateVolsandFramesPerVol(noVols, self.slicesPerVolume, framesToDelete = self.framesToDelete)
        self.volumeText.setText(str(noVols))

        self.viewer.updateDisplay_volumeSizeChange()
        self.shapeText.setText(str(self.viewer.getArrayShape()))

        if (self.slicesPerVolume)%2 == 0:
            self.SpinBox1.setRange(0,self.slicesPerVolume - 1 - self.framesToDelete) #if even, display the last volume
            self.slider1.setMaximum(self.slicesPerVolume - 1 - self.framesToDelete)
        else:
            self.SpinBox1.setRange(0,self.slicesPerVolume - 2 - self.framesToDelete) #else, don't display the last volume
            self.slider1.setMaximum(self.slicesPerVolume - 2 - self.framesToDelete)

        self.updateVolSpinBoxes()
        return

    def updateVolSpinBoxes(self):
        rangeValue = self.viewer.getNVols()-1
        #self.SpinBox3.setRange(0,self.viewer.getNVols())        
        self.SpinBox6.setRange(0,rangeValue)
        self.SpinBox7.setRange(0,rangeValue)
        self.SpinBox11.setRange(0,rangeValue)
        self.SpinBox12.setRange(0,rangeValue) 
        self.SpinBox12.setValue(rangeValue)             
        return

    def getBaseline(self):
        return self.SpinBox4.value()

    def getF0(self):
        return self.SpinBox6.value(), self.SpinBox7.value(), self.SpinBox11.value(), self.SpinBox12.value()

    def subtractBaseline(self):
        self.viewer.subtractBaseline()
        return

    def ratioDFF0(self):
        self.viewer.ratioDFF0()
        return

    def exportToWindow(self):
        self.viewer.savePath = self.arraySavePath
        self.viewer.exportToWindow()
        return

    def dTypeSelectionChange(self):
        self.viewer.setDType(self.dTypeSelectorBox.currentText())
        self.dataTypeText = QtWidgets.QLabel(str(self.viewer.getDataType()))
        return

    def multiplyByFactor(self):
        self.viewer.multiplyByFactor(self.SpinBox8.value())
        return

    def exportArray(self):
        if self.viewer.B == []:
            print('first set number of frames per volume')
            g.m.statusBar().showMessage("first set number of frames per volume")
            return
        self.arraySavePath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', self.arraySavePath, 'Numpy array (*.npy)')
        self.arraySavePath = str(self.arraySavePath[0])
        self.viewer.savePath = self.arraySavePath
        self.arraySavePathLabel.setText(self.arraySavePath)
        self.viewer.exportArray()
        return

    def startViewer(self):
        self.saveSettings()
        self.viewer.startViewer()
        return

    def setOverlay(self):
        self.viewer.setOverlay()

    def closeViewer(self):
        self.viewer.closeViewer()
        return

    def setTheta(self):
        self.theta = self.SpinBox9.value()

    def setShiftFactor(self):
        self.shiftFactor = self.SpinBox10.value()

    def trim_last_frameClicked(self):
        self.trim_last_frame = self.trim_last_frame_checkbox.isChecked()

    def inputArraySelectionChange(self, value):
        self.viewer.setInputArrayOrder(self.inputArraySelectorBox.currentText())
        return

    def displayArraySelectionChange(self, value):
        self.viewer.setDisplayArrayOrder(self.displayArraySelectorBox.currentText())
        return

    def saveSettings(self):
        self.s['theta'] = self.theta
        self.s['slicesPerVolume'] = self.slicesPerVolume
        self.s['slicesDeletedPerVolume'] = self.slicesDeletedPerVolume      
        self.s['baselineValue'] = self.baselineValue
        self.s['f0Start'] = self.f0Start
        self.s['f0End'] = self.f0End
        self.s['multiplicationFactor'] = self.multiplicationFactor
        self.s['currentDataType'] = self.currentDataType
        self.s['newDataType'] = self.newDataType
        self.s['shiftFactor'] = self.shiftFactor
        self.s['trimLastFrame'] = self.trim_last_frame      
        self.s['f0VolStart'] = self.f0VolStart
        self.s['f0VolEnd'] = self.f0VolEnd 
        
        g.settings['volumeSlider'] = self.s
        
        return


    def close(self):
        self.saveSettings()
        self.viewer.closeViewer()
        self.viewer.displayWindow.close()
        self.viewer.dialogbox.destroy()
        self.viewer.end()
        self.closeAllWindows()
        gc.collect()
        return

    def clearData(self):
        self.viewer.A = []
        self.viewer.B = []

        

