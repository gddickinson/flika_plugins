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


dataType = np.float16

class Load_tiff (BaseProcess):
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        self.nChannels = 0
        self.nVolumes = 0
        self.nFrames = 0
        self.errorFlag = False
        self.channel2_array = None

    def gui(self):
        filetypes = 'Image Files (*.tif *.tiff);;All Files (*.*)'
        prompt = 'Open File'
        filename = open_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
        
        self.openTiff(filename) 
        
        if self.errorFlag:
            return None
            
        
        return (self.nChannels, self.nVolumes, self.sclicesPerVolume, self.channel1_array, self.channel2_array)
            
    def openTiff(self, filename):
        Tiff = tifffile.TiffFile(str(filename))

        A = Tiff.asarray()
        Tiff.close()
        axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]
        print(axes)

        if set(axes) == set(['time', 'depth', 'height', 'width']):  # single channel, multi-volume
            target_axes = ['time', 'depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nScans, nFrames, x, y = A.shape

            self.channel1_array = A.reshape(nScans*nFrames,x,y)
            #newWindow = Window(self.channel1_array,'Loaded Tiff')
            
            nFrames, x, y = self.channel1_array.shape
            
            self.nChannels = 1
            self.nVolumes = nScans
            self.nFrames = nFrames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return 
            
        elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            
            self.channel1_array = A.reshape(nFrames,x,y)
            
            #uncomment to display original
            #newWindow = Window(self.channel1_array,'Loaded Tiff')
            
            self.nChannels = 1
            self.nVolumes = 1
            self.nFrames = nFrames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return 
            
        elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
            target_axes = ['time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            
            self.channel1_array = A.reshape(nFrames,x,y)
            
            #uncomment to display original
            #newWindow = Window(self.channel1_array,'Loaded Tiff')  
                        
            self.nChannels = 1
            self.nVolumes = 1
            self.nFrames = nFrames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))   
            return
            
        elif set(axes) == set(['time', 'depth', 'channel', 'height', 'width']):  # multi-channel, multi-volume
            target_axes = ['channel','time','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            self.channel1_array = A[0]
            self.channel2_array = A[1]

            nChannels, nScans, nFrames, x, y = A.shape
            
            n1Scans, n1Frames, x1, y1 = self.channel1_array.shape
            n2Scans, n2Frames, x2, y2 = self.channel2_array.shape

            self.channel1_array = self.channel1_array.reshape(n1Scans*n1Frames,x1,y1)
            self.channel2_array = self.channel2_array.reshape(n2Scans*n2Frames,x2,y2)
            
            n1Frames, x1, y1 = self.channel1_array.shape
            n2Frames, x2, y2 = self.channel2_array.shape

            #uncomment to display original
            #self.channel_1 = Window(self.channel1_array,'Channel 1')
            #self.channel_2 = Window(self.channel2_array,'Channel 2')
            
            #original shape before splitting channel
            self.nChannels = nChannels
            self.nVolumes = nScans
            self.nFrames = n1Frames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return 

        elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            self.channel1_array = A[0]
            self.channel2_array = A[1]
            
            nChannels, nFrames, x, y = A.shape
            
            n1Frames, x1, y1 = self.channel1_array.shape
            n2Frames, x2, y2 = self.channel2_array.shape

            #uncomment to display original
            #self.channel_1 = Window(self.channel1_array,'Channel 1')
            #self.channel_2 = Window(self.channel2_array,'Channel 2')
            
            #original shape before splitting channel            
            self.nChannels = nChannels
            self.nVolumes = 1
            self.nFrames = n1Frames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return

        elif set(axes) == set(['time', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            self.channel1_array = A[0]
            self.channel2_array = A[1]
            
            nChannels, nFrames, x, y = A.shape
            
            n1Frames, x1, y1 = self.channel1_array.shape
            n2Frames, x2, y2 = self.channel2_array.shape

            #uncomment to display original
            #self.channel_1 = Window(self.channel1_array,'Channel 1')
            #self.channel_2 = Window(self.channel2_array,'Channel 2')
            
            #original shape before splitting channel            
            self.nChannels = nChannels
            self.nVolumes = 1
            self.nFrames = n1Frames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return    
            

        elif set(axes) == set(['depth', 'height', 'width']):  # single channel, single-volume
            target_axes = ['depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            
            self.channel1_array = A.reshape(nFrames,x,y)
            
            #uncomment to display original
            #newWindow = Window(self.channel1_array,'Loaded Tiff')
            
            self.nChannels = 1
            self.nVolumes = 1
            self.nFrames = nFrames
            self.sclicesPerVolume = int(self.nFrames / self.nVolumes)
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return 
            
        else:
            print('tif axes header not recognized')
            self.errorFlag = True
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return

            
load_tiff = Load_tiff()
 
class CamVolumeSlider2(BaseProcess):

    def __init__(self):
        super().__init__()
        self.nVols = 1
        self.nChannels = 0
        self.overlayFlag = False
        return        

    def startVolumeSlider(self):
        #open file
        self.nChannels, self.nVols, self.slicesPerVolume, self.A, self.B = load_tiff.gui()
        print(self.nChannels, self.nVols, self.slicesPerVolume)

        self.C = []
        self.D = []        
        #copy selected window
        #self.A = g.win.image
        #self.A = self.channel1_array
        
        #get shape
        self.nFrames, self.x, self.y = self.A.shape
        self.framesPerVol = int(self.nFrames/self.nVols)
        text_1 = 'Volume Slider Window'
        if self.nChannels == 2:
            #self.B = self.channel2_array
            text_1 = 'Volume Slider Channel 2'
            text_2 = 'Volume Slider Channel 1'

        #Reshape original array(s) for display
        self.C = np.reshape(self.A, (self.slicesPerVolume,self.nVols,self.x,self.y), order='F') 
        if self.nChannels == 2:
            self.D = np.reshape(self.B, (self.slicesPerVolume,self.nVols,self.x,self.y), order='F') 
            
        #display image
        self.displayWindow = Window(self.C[0],text_1)
        if self.nChannels == 2:
            self.displayWindow_2 = Window(self.D[0],text_2)
        
        #open gui
        self.dialogbox = Form3()
        self.dialogbox.show() 
        
        
        #link 2 channel display window time sliders
        if self.nChannels == 2:
            self.displayWindow.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider)
            self.displayWindow_2.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider_2)
        return        
 

    def updateDisplay_volumeSizeChange(self):
        #remove final volume if it dosen't contain the full number of frames
        numberFramesToRemove = self.nFrames%self.getFramesPerVol()
        if numberFramesToRemove != 0:
            self.C = self.A[:-numberFramesToRemove,:,:]   
        else:
            self.C = self.A
        
        self.nFrames, self.x, self.y = self.C.shape
        
        #reshape to 4D
        self.C = np.reshape(self.C, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
        self.displayWindow.imageview.setImage(self.C[0],autoLevels=False)
        
        if self.nChannels == 2:
            if numberFramesToRemove != 0:
                self.D = self.B[:-numberFramesToRemove,:,:]   
            else:
                self.D = self.B
           
            #reshape to 4D            
            self.D = np.reshape(self.D, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
            self.displayWindow_2.imageview.setImage(self.D[0],autoLevels=False)         
        return 
 
    def updateDisplay_sliceNumberChange(self, index):
        displayIndex = self.displayWindow.imageview.currentIndex  
        self.displayWindow.imageview.setImage(self.C[index],autoLevels=False)
        self.displayWindow.imageview.setCurrentIndex(displayIndex)
        
        if self.nChannels == 2:
            self.displayWindow_2.imageview.setImage(self.D[index],autoLevels=False)
            self.displayWindow_2.imageview.setCurrentIndex(displayIndex)       
        
        if self.overlayFlag == True:
            self.overlay()         
        return
         
    def updateFrameSlider(self):
        self.displayWindow_2.imageview.setCurrentIndex(self.displayWindow.imageview.currentIndex)
        return
        
    def updateFrameSlider_2(self):
        self.displayWindow.imageview.setCurrentIndex(self.displayWindow_2.imageview.currentIndex)
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

    def setChannelNumber(self, value):
        self.nChannels = value
        return
                
    def overlay(self):
        red = self.displayWindow.imageview.getProcessedImage()
        green = self.displayWindow_2.imageview.getProcessedImage()
        
        self.overlayed = np.zeros((red.shape[0], red.shape[1], red.shape[2], 3))
        
        self.overlayed[:,:,:,0] = red
        self.overlayed[:,:,:,1] = green
        
        #print(self.overlayed.shape)
        if self.overlayFlag == False:
            self.displayWindow_Overlay = Window(self.overlayed[0],'Overlay')
            self.overlayFlag = True
        else:
            self.displayWindow_Overlay.imageview.setImage(self.overlayed[0],autoLevels=False)
        return

    def closeEvent(self, event):
        BaseProcess.closeEvent(self, event)
        self.dialogbox.close()

    def getArrayShape(self):
        if self.nChannels == 2:
            if self.C == []:
                return self.A.shape, self.B.shape
            else:
                return self.C.shape, self.D.shape
        else:                
            if self.C == []:
                return self.A.shape
            else:
                return self.C.shape

    def subtractBaseline(self):
        index = self.displayWindow.imageview.currentIndex
        baseline = self.dialogbox.getBaseline()
        #print(baseToSubtract.shape)
        #print(baseToSubtract)

        self.C = self.C - baseline
        self.displayWindow.imageview.setImage(self.C[index],autoLevels=False)
        
        if self.nChannels == 2:    
            self.D = self.D - baseline
            self.displayWindow_2.imageview.setImage(self.D[index],autoLevels=False)
        return

    def averageByVol(self):
        index = self.displayWindow.imageview.currentIndex
        #TODO
        return

    def ratioDFF0(self):
        index = self.displayWindow.imageview.currentIndex
        ratioStart, ratioEnd = self.dialogbox.getF0()
        if ratioStart >= ratioEnd:
            print('invalid F0 selection')
            return
        
        ratioVol = self.C[:,ratioStart:ratioEnd,:,]
        ratioVol = np.mean(ratioVol, axis=1,keepdims=True)        

        self.C = np.divide(self.C,ratioVol, dtype=dataType)
        self.displayWindow.imageview.setImage(self.C[index],autoLevels=False) 

        if self.nChannels == 2:
            ratioVol2 = self.D[:,ratioStart:ratioEnd,:,]
            ratioVol2 = np.mean(ratioVol2, axis=1,keepdims=True)        
    
            self.D = np.divide(self.D,ratioVol2,dtype=dataType)
            self.displayWindow_2.imageview.setImage(self.D[index],autoLevels=False) 
               
        return

    def exportToWindow(self):
        Window(np.reshape(self.C, (self.nFrames, self.x, self.y), order='F'))
        if self.nChannels == 2:
            Window(np.reshape(self.D, (self.nFrames, self.x, self.y), order='F'))
        return

    def getMaxPixel(self):
        if self.C == []:
            return np.max(self.A)
        else:
            return np.max(self.C)
 
camVolumeSlider2 = CamVolumeSlider2()  

class Form3(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form3, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 400

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider2.getFramesPerVol()-1)
        self.SpinBox1.setValue(0)
 
        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider2.getNFrames())
        self.SpinBox2.setValue(camVolumeSlider2.getFramesPerVol())
        
        self.spinLabel3 = QtWidgets.QLabel("# of volumes to average by: ") 
        self.SpinBox3 = QtWidgets.QSpinBox()
        self.SpinBox3.setRange(0,camVolumeSlider2.getNVols())
        self.SpinBox3.setValue(0)

        self.spinLabel4 = QtWidgets.QLabel("baseline value: ") 
        self.SpinBox4 = QtWidgets.QSpinBox()
        self.SpinBox4.setRange(0,camVolumeSlider2.getMaxPixel())
        self.SpinBox4.setValue(0)
         
        self.spinLabel6 = QtWidgets.QLabel("F0 start volume: ") 
        self.SpinBox6 = QtWidgets.QSpinBox()
        self.SpinBox6.setRange(0,camVolumeSlider2.getNVols())
        self.SpinBox6.setValue(0)
        
        self.spinLabel7 = QtWidgets.QLabel("F0 start volume: ") 
        self.SpinBox7 = QtWidgets.QSpinBox()
        self.SpinBox7.setRange(0,camVolumeSlider2.getNVols())
        self.SpinBox7.setValue(0)
 
        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider2.getFramesPerVol()-1)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")       
        self.button2 = QtWidgets.QPushButton("Set Slices")   
        self.button3 = QtWidgets.QPushButton("Overlay")  
        self.button4 = QtWidgets.QPushButton("Average Volumes")  
        self.button5 = QtWidgets.QPushButton("subtract baseline")          
        self.button6 = QtWidgets.QPushButton("run DF/F0")  
        self.button7 = QtWidgets.QPushButton("export to Window")  

        #labels
        self.volumeLabel = QtWidgets.QLabel("# of volumes")
        self.volumeText = QtWidgets.QLabel(str(int(camVolumeSlider2.getNFrames()/self.SpinBox2.value())))
        
        self.shapeLabel = QtWidgets.QLabel("array shape: ")
        self.shapeText = QtWidgets.QLabel(str(camVolumeSlider2.getArrayShape())) 

        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        
        layout.addWidget(self.spinLabel2, 4, 0)
        layout.addWidget(self.SpinBox2, 4, 1)
        layout.addWidget(self.button2, 4, 2) 
        
        layout.addWidget(self.spinLabel3, 5, 0)
        layout.addWidget(self.SpinBox3, 5, 1)
        layout.addWidget(self.button4, 5, 2) 

        layout.addWidget(self.spinLabel4, 6, 0)
        layout.addWidget(self.SpinBox4, 6, 1)
     
        layout.addWidget(self.button5, 6, 2) 

        layout.addWidget(self.spinLabel6, 7, 0)
        layout.addWidget(self.SpinBox6, 7, 1)
        layout.addWidget(self.spinLabel7, 7, 2)
        layout.addWidget(self.SpinBox7, 7, 3)        
        layout.addWidget(self.button6, 7, 4)         

        layout.addWidget(self.volumeLabel, 8, 0)         
        layout.addWidget(self.volumeText, 8, 1) 
        layout.addWidget(self.shapeLabel, 9, 0)         
        layout.addWidget(self.shapeText, 9, 1) 
        
        layout.addWidget(self.button7, 10, 0)
        layout.addWidget(self.button3, 10, 3)          
        layout.addWidget(self.button1, 10, 4)  

 
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
        self.button3.clicked.connect(self.overlay)
        self.button4.clicked.connect(self.averageByVol)
        self.button5.clicked.connect(self.subtractBaseline)
        self.button6.clicked.connect(self.ratioDFF0)
        self.button7.clicked.connect(self.exportToWindow)
        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider2.updateDisplay_sliceNumberChange(value)
        return       
    
    def autoLevel(self):
        camVolumeSlider2.displayWindow.imageview.autoLevels()
        if camVolumeSlider2.nChannels == 2:
            camVolumeSlider2.displayWindow_2.imageview.autoLevels()
        return 
    
    def updateVolumeValue(self):
        value = self.SpinBox2.value()
        noVols = int(camVolumeSlider2.getNFrames()/value)
        camVolumeSlider2.updateVolsandFramesPerVol(noVols, value)
        self.volumeText.setText(str(noVols))
       
        camVolumeSlider2.updateDisplay_volumeSizeChange()
        
        if (value)%2 == 0:
            self.SpinBox1.setRange(0,value-1) #if even, display the last volume 
            self.slider1.setMaximum(value-1)
        else:
            self.SpinBox1.setRange(0,value-2) #else, don't display the last volume 
            self.slider1.setMaximum(value-2)
            
        self.updateVolSpinBoxes()     
            
        return
        
    def overlay(self):
        camVolumeSlider2.overlay()
        return

    def updateVolSpinBoxes(self):     
        self.SpinBox3.setRange(0,camVolumeSlider2.getNVols())

        self.SpinBox6.setRange(0,camVolumeSlider2.getNVols())
        self.SpinBox7.setRange(0,camVolumeSlider2.getNVols())
        return
   
    def getBaseline(self):
        return self.SpinBox4.value()

    def getF0(self):
        return self.SpinBox6.value(), self.SpinBox7.value()

    def getVolsToAverage(self):
        return self.SpinBox3.value()

    def averageByVol(self):
        camVolumeSlider2.averageByVol()
        return

    def subtractBaseline(self):
        camVolumeSlider2.subtractBaseline()
        return
    
    def ratioDFF0(self):
        camVolumeSlider2.ratioDFF0()
        return
    
    def exportToWindow(self):
        camVolumeSlider2.exportToWindow()
        return    