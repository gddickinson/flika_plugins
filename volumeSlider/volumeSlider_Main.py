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
import glob
import gc

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
from .tiffLoader import openTiff

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
        
        self.batch = False

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

    def startVolumeSlider(self, A=[], keepWindow=False, batch=False):
        if batch:
            self.batch = True
            self.batchOptions = BatchOptions()
            self.batchOptions.start()
            return
        
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

    def updateVolumeSlider(self, A):
        self.A = A
        self.B = []
        self.nFrames, self.x, self.y = self.A.shape
        self.framesPerVol = int(self.nFrames/self.nVols)
        

    def batchProcess(self, paramDict):
        print(paramDict)
        #get filenames from import folder
        tiffFiles = []
        for file in glob.glob(os.path.join(paramDict['inputDirectory'],"*.tiff")):
            tiffFiles.append(file)
        for file in glob.glob(os.path.join(paramDict['inputDirectory'],"*.tif")):
            tiffFiles.append(file)
        print(tiffFiles)
        #loop through files
        i = 0
        for tiff_file in tiffFiles:
            self.imsPath = tiff_file
            #load tiff
            if i == 0:
                self.A, _, _ = openTiff(tiff_file)
                self.B = []
            else:
                self.dialogbox.loadNewFile(tiff_file)
            #get shape
            self.nFrames, self.x, self.y = self.A.shape
            self.framesPerVol = paramDict['slicesPerVolume']
            self.displayWindow = Window(self.A,'Volume Slider Window')
            self.displayWindow.hide()
            if i == 0:
                self.dialogbox = Form2(camVolumeSlider)
            self.dialogbox.batch = True
            #self.dialogbox.show() 
            #shape tiff to 4D
            self.dialogbox.slicesPerVolume = self.framesPerVol
            self.dialogbox.SpinBox2.setValue(self.framesPerVol)
            self.dialogbox.button2.click()
            #subtract baseline
            self.dialogbox.baselineValue = paramDict['baselineValue']
            self.dialogbox.SpinBox4.setValue(self.dialogbox.baselineValue)            
            if paramDict['subtractBaseline']:
                self.dialogbox.button4.click()
            #DF/F0    
            self.dialogbox.f0Start = paramDict['f0Start']
            self.dialogbox.SpinBox6.setValue(self.dialogbox.f0Start)            
            self.dialogbox.f0End = paramDict['f0End']
            self.dialogbox.SpinBox7.setValue(self.dialogbox.f0End)            
            self.dialogbox.f0VolStart = paramDict['f0VolStart']
            self.dialogbox.SpinBox11.setValue(self.dialogbox.f0VolStart)            
            self.dialogbox.f0VolEnd = paramDict['f0VolEnd']
            self.dialogbox.SpinBox12.setValue(self.dialogbox.f0VolEnd)                 
            if paramDict['runDFF0']:
                self.dialogbox.button5.click() 
            #scale by multiplication factor
            self.dialogbox.multiplicationFactor = paramDict['multiplicationFactor']
            self.dialogbox.SpinBox8.setValue(self.dialogbox.multiplicationFactor)  
            if paramDict['runMultiplication']:
                self.dialogbox.button8.click() 
            #theta
            self.dialogbox.theta = paramDict['theta']
            self.dialogbox.SpinBox9.setValue(self.dialogbox.theta)
            #shift factor
            self.dialogbox.shiftFactor = paramDict['shiftFactor']
            self.dialogbox.SpinBox10.setValue(self.dialogbox.shiftFactor)
            #trim last frame
            self.dialogbox.trim_last_frame = paramDict['trim_last_frame']
            self.dialogbox.trim_last_frame_checkbox.setChecked(self.dialogbox.trim_last_frame)
            #start volumeSlider3D
            if i == 0:
                self.dialogbox.button12.click() 
            else:
                self.viewer.changeMainImage(self.B)
                self.viewer.runBatchStep(self.imsPath)   
            #update counter
            i = i+1
            

        g.m.statusBar().showMessage('finished batch processing')   
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
        print(self.B.shape)
        
        #delete frames from start of each volume
        if self.framesToDelete != 0:
            self.B = self.B[self.framesToDelete:-1,:,:,:]
                
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

    def updateVolsandFramesPerVol(self, nVols, framesPerVol, framesToDelete = 0):
        self.nVols = nVols
        self.framesPerVol = framesPerVol
        self.framesToDelete = framesToDelete

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
            g.m.statusBar().showMessage('first set number of frames per volume')
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
            g.m.statusBar().showMessage('invalid F0 selection')
            return
        
        if volStart >= volEnd:
            print('invalid F0 Volume selection')
            g.m.statusBar().showMessage('invalid F0 Volume selection')
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
        if self.B == []:
            print('first set number of frames per volume')
            g.m.statusBar().showMessage("first set number of frames per volume")
        else:
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
            g.m.statusBar().showMessage("first set number of frames per volume")
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
            g.m.statusBar().showMessage("first set number of frames per volume")
            return
        else:
            self.B = self.B * float(factor)
            print(self.B.shape)
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def startViewer(self):
        if self.B == []:
            print('first set number of frames per volume')
            g.m.statusBar().showMessage("first set number of frames per volume")
        else:
            if self.batch:
                self.viewer = SliceViewer(camVolumeSlider, self.B, batch=True, imsExportPath=self.imsPath)
                return
            self.viewer = SliceViewer(camVolumeSlider, self.B)
        return

    def closeViewer(self):
        self.viewer.close()
        return

camVolumeSlider = CamVolumeSlider()

class BatchOptions(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(BatchOptions, self).__init__(parent)

        self.s = g.settings['volumeSlider']
        
        self.slicesPerVolume = self.s['slicesPerVolume']
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

        self.subtractBaseline = False 
        self.runDFF0 = False
        self.runMultiplication = False

        self.inputDirectory = ''
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #labels
        self.label_slicesPerVolume = QtWidgets.QLabel("slices per volume:") 
        self.label_theta = QtWidgets.QLabel("theta:") 
        self.label_baselineValue = QtWidgets.QLabel('baseline Value:')
        self.label_f0Start = QtWidgets.QLabel('f0 Start:')
        self.label_f0End = QtWidgets.QLabel('f0 End:')
        self.label_f0VolStart = QtWidgets.QLabel('f0Vol Start:')
        self.label_f0VolEnd = QtWidgets.QLabel('f0Vol End:')            
        self.label_multiplicationFactor = QtWidgets.QLabel('multiplication Factor:')
        self.label_shiftFactor = QtWidgets.QLabel('shift Factor:')
        self.label_trim_last_frame = QtWidgets.QLabel('trim Last Frame:') 
        self.label_inputDirectory = QtWidgets.QLabel('input directory:') 
      
        self.label_subtractBaseline = QtWidgets.QLabel('subtract baseline:') 
        self.label_runDFF0 = QtWidgets.QLabel('run DF/F0:') 
        self.label_runMultiplication = QtWidgets.QLabel('scale by multiplication factor:')         
        
        #spinboxes/comboboxes
        self.volBox = QtWidgets.QSpinBox()
        self.volBox.setRange(0,10000)
        self.volBox.setValue(self.slicesPerVolume)

        self.thetaBox = QtWidgets.QSpinBox()
        self.thetaBox.setRange(0,360)
        self.thetaBox.setValue(self.theta)

        self.baselineBox = QtWidgets.QSpinBox()
        self.baselineBox.setRange(0,100000)
        self.baselineBox.setValue(self.baselineValue)

        self.f0StartBox = QtWidgets.QSpinBox()
        self.f0StartBox.setRange(0,100000)
        self.f0StartBox.setValue(self.f0Start)

        self.f0EndBox = QtWidgets.QSpinBox()
        self.f0EndBox.setRange(0,100000)
        self.f0EndBox.setValue(self.f0End)

        self.f0VolStartBox = QtWidgets.QSpinBox()
        self.f0VolStartBox.setRange(0,100000)
        self.f0VolStartBox.setValue(self.f0VolStart)

        self.f0VolEndBox = QtWidgets.QSpinBox()
        self.f0VolEndBox.setRange(0,100000)
        self.f0VolEndBox.setValue(self.f0VolEnd) 
        
        self.multiplicationFactorBox = QtWidgets.QSpinBox()
        self.multiplicationFactorBox.setRange(0,100000)
        self.multiplicationFactorBox.setValue(self.multiplicationFactor)         
        
        self.shiftFactorBox = QtWidgets.QSpinBox()
        self.shiftFactorBox.setRange(0,100000)
        self.shiftFactorBox.setValue(self.shiftFactor)         

        self.trim_last_frame_checkbox = CheckBox()
        self.trim_last_frame_checkbox.setChecked(self.trim_last_frame)

        self.subtractBaseline_checkbox = CheckBox()
        self.subtractBaseline_checkbox.setChecked(self.subtractBaseline)

        self.runDFF0_checkbox = CheckBox()
        self.runDFF0_checkbox.setChecked(self.runDFF0)

        self.runMultiplication_checkbox = CheckBox()
        self.runMultiplication_checkbox.setChecked(self.runMultiplication)

        
        self.inputDirectory_display = QtWidgets.QLabel(self.inputDirectory)

        #buttons
        self.button_setInputDirectory = QtWidgets.QPushButton("Set Folder")
        self.button_startBatch = QtWidgets.QPushButton("Go")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.label_slicesPerVolume, 0, 0)        
        layout.addWidget(self.volBox, 0, 1)
        layout.addWidget(self.label_theta, 1, 0)        
        layout.addWidget(self.thetaBox, 1, 1)
        layout.addWidget(self.label_baselineValue, 2, 0)        
        layout.addWidget(self.baselineBox, 2, 1)        
        layout.addWidget(self.label_f0Start, 3, 0)        
        layout.addWidget(self.f0StartBox, 3, 1)         
        layout.addWidget(self.label_f0End, 4, 0)        
        layout.addWidget(self.f0EndBox, 4, 1)         
        layout.addWidget(self.label_f0VolStart, 5, 0)        
        layout.addWidget(self.f0VolStartBox, 5, 1)         
        layout.addWidget(self.label_f0VolEnd, 6, 0)        
        layout.addWidget(self.f0VolEndBox, 6, 1)  
        layout.addWidget(self.label_multiplicationFactor, 7, 0)        
        layout.addWidget(self.multiplicationFactorBox, 7, 1)  
        layout.addWidget(self.label_shiftFactor, 8, 0)        
        layout.addWidget(self.shiftFactorBox, 8, 1) 
        layout.addWidget(self.label_trim_last_frame, 9, 0)        
        layout.addWidget(self.trim_last_frame_checkbox, 9, 1) 
        
        layout.addWidget(self.label_subtractBaseline, 10, 0)        
        layout.addWidget(self.subtractBaseline_checkbox, 10, 1) 

        layout.addWidget(self.label_runDFF0, 11, 0)        
        layout.addWidget(self.runDFF0_checkbox, 11, 1)        
        layout.addWidget(self.label_runMultiplication, 12, 0)        
        layout.addWidget(self.runMultiplication_checkbox, 12, 1)        
        
        layout.addWidget(self.label_inputDirectory, 13, 0)        
        layout.addWidget(self.inputDirectory_display, 13, 1) 
        layout.addWidget(self.button_setInputDirectory, 13, 2) 
        layout.addWidget(self.button_startBatch, 14, 2) 
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Batch Options")

        #connect spinboxes/comboboxes
        self.volBox.valueChanged.connect(self.set_slicesPerVolume) 
        self.thetaBox.valueChanged.connect(self.set_theta) 
        self.baselineBox.valueChanged.connect(self.set_baselineValue) 
        self.f0StartBox.valueChanged.connect(self.set_f0Start) 
        self.f0EndBox.valueChanged.connect(self.set_f0End) 
        self.f0VolStartBox.valueChanged.connect(self.set_f0VolStart) 
        self.f0VolEndBox.valueChanged.connect(self.set_f0VolEnd)  
        self.multiplicationFactorBox.valueChanged.connect(self.set_multiplicationFactor) 
        self.shiftFactorBox.valueChanged.connect(self.set_shiftFactor) 
        self.trim_last_frame_checkbox.stateChanged.connect(self.set_trim_last_frame)
        self.subtractBaseline_checkbox.stateChanged.connect(self.set_subtractBaseline)          
        self.runDFF0_checkbox.stateChanged.connect(self.set_runDFF0)        
        self.runMultiplication_checkbox.stateChanged.connect(self.set_runMultiplication)         
        self.button_setInputDirectory.pressed.connect(lambda: self.setInput_button())
        self.button_startBatch.pressed.connect(lambda: self.start_button())        
        return


    def set_slicesPerVolume(self,value):
        self.slicesPerVolume = value
        
    def set_baselineValue(self,value):        
        self.baselineValue = value

    def set_f0Start (self,value):                  
        self.f0Start = value
        
    def set_f0End (self,value):        
        self.f0End = value
        
    def set_f0VolStart (self,value):        
        self.f0VolStart = value
                
    def set_f0VolEnd(self,value):            
        self.f0VolEnd = value

    def set_multiplicationFactor(self,value):               
        self.multiplicationFactor = value

    def set_theta(self,value):  
        self.theta = value

    def set_shiftFactor(self,value):         
        self.shiftFactor = value
        
    def set_trim_last_frame(self):           
        self.trim_last_frame = self.trim_last_frame_checkbox.isChecked()     
 
    def set_subtractBaseline(self):           
        self.subtractBaseline = self.subtractBaseline_checkbox.isChecked() 

    def set_runDFF0(self):           
        self.runDFF0 = self.runDFF0_checkbox.isChecked()

    def set_runMultiplication(self):           
        self.runMultiplication = self.runMultiplication_checkbox.isChecked()
       
    def setInput_button(self):
        self.inputDirectory = QtWidgets.QFileDialog.getExistingDirectory()
        self.inputDirectory_display.setText('...\\' + os.path.basename(self.inputDirectory))
        return

    def start_button(self):
        paramDict = {
                     'slicesPerVolume': self.slicesPerVolume,
                     'theta': self.theta,
                     'baselineValue': self.baselineValue,
                     'f0Start': self.f0Start,
                     'f0End': self.f0End,
                     'f0VolStart': self.f0VolStart,
                     'f0VolEnd': self.f0VolEnd   ,      
                     'multiplicationFactor': self.multiplicationFactor,
                     'shiftFactor': self.shiftFactor ,
                     'trim_last_frame': self.trim_last_frame,
                     'inputDirectory': self.inputDirectory,                    
                     'subtractBaseline': self.subtractBaseline,                     
                     'runDFF0': self.runDFF0,                     
                     'runMultiplication': self.runMultiplication                     
                     }
        
        self.hide()
        camVolumeSlider.batchProcess(paramDict)
        return
    
    def start(self):
        self.show()

