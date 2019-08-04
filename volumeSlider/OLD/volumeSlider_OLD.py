#from __future__ import (absolute_import, division,print_function, unicode_literals)
#from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
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

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox

class VolumeSlider():

    def __init__(self):
        #initialize list to store volumes
        self.A_list = []
        self.displayedVol = 0
        self.maxFrames = 0
        self.width = 0
        self.height = 0
        self.displayedFrame = 0
        return        

    def startVolumeSlider(self):
        #get volume arrays
        self.getVolumes()
        #display 1st volume
        self.displayWindow = Window(self.A_list[0],'Volume Slider Window')
        self.displayedVol = 0
        self.maxFrames, self.width, self.height = self.A_list[0].shape
        #open gui
        self.dialogbox = Form()
        self.dialogbox.show() 
        #connect window time slider to volumeSlider GUI
        self.displayWindow.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider)
        return        
    
    def getVolumes(self):
        #clear volume list
        self.A_list = []
        #get path of volume folder
        volume_path = QtWidgets.QFileDialog.getExistingDirectory(g.m, "Select a parent folder to save into.", expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        #get volume files in folder
        vols = [f for f in listdir(volume_path) if isfile(join(volume_path, f))]
        #add volumes to volume list
        for i in range(len(vols)):
            file = join(volume_path, vols[i])
            self.A_list.append(self.openTiff(file))
        print(len(self.A_list))
        return
    
    def openTiff(self, filename):
        Tiff = tifffile.TiffFile(str(filename))
        A = Tiff.asarray()
        Tiff.close()
        axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]
        if set(axes) == set(['series', 'height', 'width']):  # single channel, multi-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
        return A
    
    def getNumberVols(self):
        return len(self.A_list)
    
    def getVolumeLength(self):
        return self.displayedVol
    
    def getNumberFrames(self):
        return self.maxFrames

    def setDisplayFrame(self, value):
        self.displayedFrame = value
        return
        
    def getDisplayFrame(self):
        return self.displayWindow.imageview.currentIndex 
        
    def updateVol(self, newVol):
        #store current frame index
        frame = self.getDisplayFrame()
        # update volume index
        self.displayedVol = newVol
        #update volume
        self.displayWindow.imageview.setImage(self.A_list[newVol],autoLevels=False)
        #update frame
        self.displayedFrame = frame
        self.displayWindow.imageview.setCurrentIndex(frame)
        return 

    def updateFrame(self, newFrame):
        #update frame
        self.displayWindow.imageview.setCurrentIndex(newFrame)
        return
        
    def updateFrameSlider(self):
        self.displayedFrame = self.getDisplayFrame()
        self.dialogbox.slider2.setValue(self.displayedFrame)
        self.dialogbox.SpinBox2.setValue(self.displayedFrame)
        return
        

volumeSlider = VolumeSlider()

class Form(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 250

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Volume #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,volumeSlider.getNumberVols())
        self.SpinBox1.setValue(0)
        
        self.spinLabel2 = QtWidgets.QLabel("Frame #")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,volumeSlider.getNumberFrames())
        self.SpinBox2.setValue(0) 
        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Volume #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(volumeSlider.getNumberVols())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        self.sliderLabel2 = QtWidgets.QLabel("Frame #")
        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider2.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(volumeSlider.getNumberFrames())
        self.slider2.setTickInterval(10)
        self.slider2.setSingleStep(1)     

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0,4,5)
        layout.addWidget(self.spinLabel2, 7, 0) 
        layout.addWidget(self.SpinBox2, 7, 1)
        layout.addWidget(self.slider2, 8, 0,4,5)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.slider2.valueChanged.connect(self.slider2ValueChange)
        self.SpinBox2.valueChanged.connect(self.spinBox2ValueChange)

        return
 
    #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        volumeSlider.updateVol(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        volumeSlider.updateVol(value)
        return       
    
    #frame changes with slider & spinbox
    def slider2ValueChange(self, value):
        self.SpinBox2.setValue(value)
        volumeSlider.updateFrame(value)
        return
            
    def spinBox2ValueChange(self, value):
        self.slider2.setValue(value)
        volumeSlider.updateFrame(value)
        return  
     
  
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        #initialize list to store volumes
        self.A_list = []
        self.displayedVol = 0
        self.maxFrames = 0
        self.width = 0
        self.height = 0
        self.displayedFrame = 0
        self.slicesPerVolume = 0
        self.numberOfVolumes = 0
        return        

    def startVolumeSlider(self):
        #copy selected window
        self.A = g.win.image
        #get dimesnsions
        self.mt, self.mx, self.my = self.A.shape
        #add array to A_list
        self.A_list.append(self.A)
        #set slices per volume
        self.slicesPerVolume = self.mt
        #set number of volumes
        self.numberOfVolumes = 1
        #display 1st volume
        self.displayWindow = Window(self.A_list[0],'Volume Slider Window')
        self.displayedVol = 0
        self.maxFrames, self.width, self.height = self.A_list[0].shape
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show() 
        #connect window time slider to volumeSlider GUI
        self.displayWindow.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider)
        return        
    
    
    def getNumberVols(self):
        return len(self.A_list)
    
    def getVolumeLength(self):
        return self.displayedVol
    
    def getNumberFrames(self):
        mt, mx, my = self.A_list[0].shape
        return mt

    def setDisplayFrame(self, value):
        self.displayedFrame = value
        return
        
    def getDisplayFrame(self):
        return self.displayWindow.imageview.currentIndex 

    def getSlicesPerVolume(self):
        return self.slicesPerVolume
        
    def updateVol(self, newVol):
        #store current frame index
        frame = self.getDisplayFrame()
        # update volume index
        self.displayedVol = newVol
        #update volume
        self.displayWindow.imageview.setImage(self.A_list[newVol],autoLevels=False)
        #update frame
        self.displayedFrame = frame
        self.displayWindow.imageview.setCurrentIndex(frame)
        return 

    def updateFrame(self, newFrame):
        #update frame
        self.displayWindow.imageview.setCurrentIndex(newFrame)
        return
        
    def updateFrameSlider(self):
        self.displayedFrame = self.getDisplayFrame()
        self.dialogbox.slider2.setValue(self.displayedFrame)
        self.dialogbox.SpinBox2.setValue(self.displayedFrame)
        return
 
 
    # def updateByVolume(self, volumeNumber):
        # self.numberOfVolumes = volumeNumber
        # return

    def updateBySlice(self, sliceNumber):
        self.slicesPerVolume = sliceNumber
        arrayList = []
        mv = self.mt // self.slicesPerVolume  # number of volumes
        A = self.A[:mv * self.slicesPerVolume]
        B = np.reshape(A, (mv, self.slicesPerVolume, self.mx, self.my))
        
        for i in range(1, mv):
            arrayList.append(B[i])
        
        self.A_list = arrayList
        self.numberOfVolumes = mv 
        return        

camVolumeSlider = CamVolumeSlider()  

class Form2(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form2, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 250

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Volume #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider.getNumberVols())
        self.SpinBox1.setValue(0)
        
        self.spinLabel2 = QtWidgets.QLabel("Frame #")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider.getNumberFrames())
        self.SpinBox2.setValue(0) 
        
        self.spinLabel3 = QtWidgets.QLabel("Number of slices per volume")
        self.SpinBox3 = QtWidgets.QSpinBox()
        self.SpinBox3.setRange(1,camVolumeSlider.getSlicesPerVolume())
        self.SpinBox3.setValue(camVolumeSlider.getSlicesPerVolume())        

        self.spinLabel4 = QtWidgets.QLabel("Total number of volumes: 1")  

        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Volume #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider.getNumberVols())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        self.sliderLabel2 = QtWidgets.QLabel("Frame #")
        self.slider2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider2.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider2.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider2.setMinimum(0)
        self.slider2.setMaximum(camVolumeSlider.getNumberFrames())
        self.slider2.setTickInterval(10)
        self.slider2.setSingleStep(1)     

        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")
        
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0,4,5)
        layout.addWidget(self.spinLabel2, 7, 0) 
        layout.addWidget(self.SpinBox2, 7, 1)
        layout.addWidget(self.slider2, 8, 0,4,5)
        layout.addWidget(self.spinLabel3, 13, 0)        
        layout.addWidget(self.SpinBox3, 13, 1,1,2)        
        layout.addWidget(self.spinLabel4, 14, 0)        
        layout.addWidget(self.button1, 14, 4,1,1) 
        
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.slider2.valueChanged.connect(self.slider2ValueChange)
        self.SpinBox2.valueChanged.connect(self.spinBox2ValueChange)
        self.SpinBox3.valueChanged.connect(self.spinBox3ValueChange)
        
        #connect buttons
        self.button1.clicked.connect(self.autoLevel)

        return
 
    #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        camVolumeSlider.updateVol(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider.updateVol(value)
        return       
    
    #frame changes with slider & spinbox
    def slider2ValueChange(self, value):
        self.SpinBox2.setValue(value)
        camVolumeSlider.updateFrame(value)
        return
            
    def spinBox2ValueChange(self, value):
        self.slider2.setValue(value)
        camVolumeSlider.updateFrame(value)
        return  
 
    def spinBox3ValueChange(self, value):
        camVolumeSlider.updateBySlice(value)
        self.updateUI()
        return 
    
    def updateUI(self):
        #update spinbox ranges
        self.SpinBox1.setRange(0,camVolumeSlider.getNumberVols())
        self.SpinBox2.setRange(0,camVolumeSlider.getNumberFrames())
        #update splider ranges
        self.slider1.setMaximum(camVolumeSlider.getNumberVols())
        self.slider2.setMaximum(camVolumeSlider.getNumberFrames())
        #update number of volumes displayed
        newText = "Total number of volumes: " + str(camVolumeSlider.getNumberVols())
        self.spinLabel4.setText(newText)
        return
    
    def autoLevel(self):
        camVolumeSlider.displayWindow.imageview.autoLevels()
        return
 