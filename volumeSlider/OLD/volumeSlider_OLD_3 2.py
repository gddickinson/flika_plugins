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

class VolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.numberOfTimeSlices = 0
        self.displayedTimeSlice = 0
        
        return        

    def startVolumeSlider(self):
        #get volume arrays
        self.getVolumes()
        #update image
        self.initiateImage()
        #display image
        self.displayWindow = Window(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],'Volume Slider Window')
        #open gui
        self.dialogbox = Form()
        self.dialogbox.show() 
        return        
  
    def initiateImage(self):   
        self.displayImage = self.interleave(np.array(self.A_list))
        #print(self.displayImage.shape)
        return

    def interleave(self, A):
        self.nVols, self.nFrames, self.x, self.y = A.shape
        #print(self.nVols, self.nFrames, self.x, self.y )
        interleaved = np.zeros((self.nVols*self.nFrames,self.x,self.y))
        #print(interleaved.shape)
        
        z = 0
        for i in np.arange(self.nFrames):
           for j in np.arange(self.nVols):
               #print(z, i, j)
               interleaved[z] = A[j%self.nVols][i] 
               z = z +1
    
        return interleaved

    def updateImage(self):
        self.displayWindow.imageview.setImage(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False)
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
        self.numberOfTimeSlices = (len(self.A_list))
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

    def getNFrames(self):
        return self.nFrames
 
    def getFramesPerSlice(self):
        return self.numberOfTimeSlices
 
    def getDisplayFrame(self):
        return self.displayWindow.imageview.currentIndex         

    def setDisplayFrame(self, value):
        self.displayedFrame = value
        self.displayWindow.imageview.setCurrentIndex(self.displayedFrame)
        return
        
    def updateSlice(self, value):
        index = self.getDisplayFrame()
        self.displayedTimeSlice = value * self.getFramesPerSlice()
        #print(self.displayedTimeSlice)
        self.updateImage()
        self.setDisplayFrame(index)
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
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,volumeSlider.getNFrames())
        self.SpinBox1.setValue(0)
        
        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(volumeSlider.getNFrames())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")        

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0,2,5)
        layout.addWidget(self.button1, 4, 4,1,1) 
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        
        #connect buttons
        self.button1.clicked.connect(self.autoLevel)

        return
 
    #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        #volumeSlider.updateSlice(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        volumeSlider.updateSlice(value)
        return       
    
    def autoLevel(self):
        volumeSlider.displayWindow.imageview.autoLevels()
        return 

