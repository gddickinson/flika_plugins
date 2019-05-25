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

        
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.nVols = 1
        return        

    def startVolumeSlider(self):
        #copy selected window
        self.A = g.win.image
        #get shape
        self.nFrames, self.x, self.y = self.A.shape
        #setup display window
        self.displayWindow = Window(self.A,'Volume Slider Window')
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show() 
        return        
    
    def updateDisplay_volumeSizeChange(self):
        self.B = np.reshape(self.A, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
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
        return int(self.nFrames/self.nVols)
        
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
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox1.setValue(0)
 
        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox2.setValue(camVolumeSlider.getNFrames())

        self.spinLabel3 = QtWidgets.QLabel("# of volumes to average by: ") 
        self.SpinBox3 = QtWidgets.QSpinBox()
        self.SpinBox3.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox3.setValue(0)

        self.spinLabel4 = QtWidgets.QLabel("baseline start volume: ") 
        self.SpinBox4 = QtWidgets.QSpinBox()
        self.SpinBox4.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox4.setValue(0)
        
        self.spinLabel5 = QtWidgets.QLabel("baseline end volume: ") 
        self.SpinBox5 = QtWidgets.QSpinBox()
        self.SpinBox5.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox5.setValue(0)
         
        self.spinLabel6 = QtWidgets.QLabel("F0 start volume: ") 
        self.SpinBox6 = QtWidgets.QSpinBox()
        self.SpinBox6.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox6.setValue(0)
        
        self.spinLabel7 = QtWidgets.QLabel("F0 start volume: ") 
        self.SpinBox7 = QtWidgets.QSpinBox()
        self.SpinBox7.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox7.setValue(0)
        

        
        #sliders
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider.getNFrames())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
             
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")       
        self.button2 = QtWidgets.QPushButton("Set Slices")      
        self.button3 = QtWidgets.QPushButton("Average Volumes")  
        self.button4 = QtWidgets.QPushButton("subtract baseline")          
        self.button5 = QtWidgets.QPushButton("run DF/F0")  

        #labels
        self.volumeLabel = QtWidgets.QLabel("# of volumes")
        self.volumeText = QtWidgets.QLabel("  ")
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        layout.addWidget(self.spinLabel2, 3, 0)
        layout.addWidget(self.SpinBox2, 3, 1)
        layout.addWidget(self.button2, 3, 2) 
        
        layout.addWidget(self.spinLabel3, 4, 0)
        layout.addWidget(self.SpinBox3, 4, 1)
        layout.addWidget(self.button3, 4, 2) 

        layout.addWidget(self.spinLabel4, 5, 0)
        layout.addWidget(self.SpinBox4, 5, 1)
        layout.addWidget(self.spinLabel5, 5, 2)
        layout.addWidget(self.SpinBox5, 5, 3)        
        layout.addWidget(self.button4, 5, 4) 

        layout.addWidget(self.spinLabel6, 6, 0)
        layout.addWidget(self.SpinBox6, 6, 1)
        layout.addWidget(self.spinLabel7, 6, 2)
        layout.addWidget(self.SpinBox7, 6, 3)        
        layout.addWidget(self.button5, 6, 4)         

        layout.addWidget(self.volumeLabel, 7, 0)         
        layout.addWidget(self.volumeText, 7, 1) 
        
        layout.addWidget(self.button1, 8, 4, 1, 1)  
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        
        #connect buttons
        self.button1.clicked.connect(self.autoLevel)
        self.button2.clicked.connect(self.updateVolumeValue)

        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider.updateDisplay_sliceNumberChange(value)
        return       
    
    def autoLevel(self):
        camVolumeSlider.displayWindow.imageview.autoLevels()
        return 
    
    def updateVolumeValue(self):
        value = self.SpinBox2.value()
        noVols = int(camVolumeSlider.getNFrames()/value)
        camVolumeSlider.nVols = noVols
        self.volumeText.setText(str(noVols))
        
        camVolumeSlider.updateDisplay_volumeSizeChange()
        
        if (value)%2 == 0:
            self.SpinBox1.setRange(0,value-1) #if even, display the last volume 
            self.slider1.setMaximum(value-1)
        else:
            self.SpinBox1.setRange(0,value-2) #else, don't display the last volume 
            self.slider1.setMaximum(value-2)
            
        self.SpinBox3.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox4.setRange(0,camVolumeSlider.getNVols())
        return

    def averageByVol(self):
        return
    
    def ratioDFF0(self):
        return