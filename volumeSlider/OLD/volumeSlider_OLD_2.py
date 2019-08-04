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


#########################################################################################################
        
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.numberOfTimeSlices = 0
        self.displayedTimeSlice = 0
        self.nVols = 1
        self.nChannels = 0
        return        

    def startVolumeSlider(self):
        #copy selected window
        self.A = g.win.image
        #update image
        self.initiateImage()
        #display image
        self.displayWindow = Window(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],'Volume Slider Window')
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show() 
        return        
    
    def initiateImage(self):   
        self.displayImage = self.interleave(self.A)
        #print(self.displayImage.shape)
        return

    def interleave(self, A):
        self.nFrames, self.x, self.y = A.shape
        self.numberOfTimeSlices = self.nFrames
        print(self.nVols, self.nFrames, self.x, self.y )
        
        if self.nVols == 1:
            return A
                
        else:
            self.numberOfTimeSlices = self.getslicesPerVolume()
            interleaved = np.zeros((self.nFrames,self.x,self.y))
            print(interleaved.shape)
            print(A.shape)
            z = 0
            for i in np.arange(self.getNumberVols()):
               for j in np.arange(self.getslicesPerVolume()):
                   #print(z, i, j)
                   interleaved[z] = A[(i*self.getslicesPerVolume()) + j%self.getslicesPerVolume()]
                   z = z +1
        return interleaved    
        

        
    def updateImage(self):
        self.displayWindow.imageview.setImage(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False)
        return    
        
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
    
    def setNumberVols(self,value):
        self.nVols = value
        return

    def getslicesPerVolume(self):
        return self.slicesPerVolume
        
    def getNumberVols(self):
        return self.nVols  
        
    def updateVolumeSize(self, value):
        #set slices per volume
        self.slicesPerVolume = value 
        #set number of volumes
        self.setNumberVols(int(self.getNFrames()/value))
        #update image
        self.initiateImage()
        self.updateImage()
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
        self.SpinBox1.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox1.setValue(0)
 
        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox2.setValue(camVolumeSlider.getNFrames())
 
        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
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

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        layout.addWidget(self.spinLabel2, 3, 0)
        layout.addWidget(self.SpinBox2, 3, 1)
        layout.addWidget(self.button2, 3, 2)         
        layout.addWidget(self.button1, 4, 4, 1, 1) 
        
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
        #camVolumeSlider.updateSlice(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider.updateSlice(value)
        return       
    
    def autoLevel(self):
        camVolumeSlider.displayWindow.imageview.autoLevels()
        return 
    
    def updateVolumeValue(self):
        value = self.SpinBox2.value()
        camVolumeSlider.updateVolumeSize(value)
        if (camVolumeSlider.getNFrames()/camVolumeSlider.getNumberVols())%2 == 0:
            self.SpinBox1.setRange(0,camVolumeSlider.getNumberVols()-1) #if even, display the last volume 
            self.slider1.setMaximum(camVolumeSlider.getNumberVols()-1)
        else:
            self.SpinBox1.setRange(0,camVolumeSlider.getNumberVols()-2) #else, don't display the last volume 
            self.slider1.setMaximum(camVolumeSlider.getNumberVols()-2)
        return

        
#########################################################################################################
        
        
class Load_tiff (BaseProcess):
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        self.nChannels = 0
        self.nVolumes = 0
        self.nFrames = 0


    def gui(self):
        filetypes = 'Image Files (*.tif *.tiff);;All Files (*.*)'
        prompt = 'Open File'
        filename = open_file_gui(prompt, filetypes=filetypes)
        if filename is None:
            return None
        
        self.openTiff(filename) 
        
        self.sclicesPerVolume = int(self.nFrames*self.nChannels / self.nVolumes)
        return (self.nChannels, self.nVolumes, self.sclicesPerVolume)
            
    def openTiff(self, filename):
        Tiff = tifffile.TiffFile(str(filename))

        A = Tiff.asarray()
        Tiff.close()
        axes = [tifffile.AXES_LABELS[ax] for ax in Tiff.series[0].axes]

        if set(axes) == set(['time', 'depth', 'height', 'width']):  # single channel, multi-volume
            target_axes = ['time', 'depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nScans, nFrames, x, y = A.shape

            A = A.reshape(nScans*nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')
            
            self.nChannels = 1
            self.nVolumes = nScans
            self.nFrames = nFrames
            return 
            
        elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')
            
            self.nChannels = 1
            self.nVolumes = 1
            self.nFrames = nFrames
            
            return 
            
        elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
            target_axes = ['time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            newWindow = Window(A,'Loaded Tiff')  
                        
            self.nChannels = 1
            self.nVolumes = 1
            self.nFrames = nFrames
            
            return
            
        elif set(axes) == set(['time', 'depth', 'channel', 'height', 'width']):  # multi-channel, multi-volume
            target_axes = ['channel','time','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            nChannels, nScans, nFrames, x, y = A.shape
            
            n1Scans, n1Frames, x1, y1 = B.shape
            n2Scans, n2Frames, x2, y2 = C.shape

            B = B.reshape(n1Scans*n1Frames,x1,y1)
            C = C.reshape(n2Scans*n2Frames,x2,y2)

            self.channel_1 = Window(B,'Channel 1')
            self.channel_2 = Window(C,'Channel 2')
            
            #original shape before splitting channel
            self.nChannels = nChannels
            self.nVolumes = nScans
            self.nFrames = nFrames

            return 

        elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]
            
            nChannels, nFrames, x, y = A.shape
            
            n1Frames, x1, y1 = B.shape
            n2Frames, x2, y2 = C.shape

            self.channel_1 = Window(B,'Channel 1')
            self.channel_2 = Window(C,'Channel 2')
            
            #original shape before splitting channel            
            self.nChannels = nChannels
            self.nVolumes = 1
            self.nFrames = nFrames
            
            return
 
load_tiff = Load_tiff()
 
class CamVolumeSlider2(BaseProcess):

    def __init__(self):
        super().__init__()
        self.numberOfTimeSlices = 0
        self.displayedTimeSlice = 0
        self.nVols = 1
        self.nChannels = 0
        self.overlayFlag = False
        self.stackedFlag = False
        return        

    def startVolumeSlider(self):
        #open file
        self.nChannels, self.nVols, self.slicesPerVolume = load_tiff.gui()
        #print(self.nChannels, self.nVols, self.slicesPerVolume)
        #copy selected window
        self.A = g.win.image
        text_1 = 'Volume Slider Window'
        if self.nChannels == 2:
            self.B = load_tiff.channel_1.imageview.getProcessedImage()
            text_1 = 'Volume Slider Channel 2'
            text_2 = 'Volume Slider Channel 1'
        #initiate image
        self.initiateImage()
        #display image
        self.displayWindow = Window(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],text_1)
        if self.nChannels == 2:
            self.displayWindow_2 = Window(self.displayImage_2[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],text_2)
        #open gui
        self.dialogbox = Form3()
        self.dialogbox.show() 
        #link 2 channel display window time sliders
        if self.nChannels == 2:
            self.displayWindow.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider)
            self.displayWindow_2.imageview.timeLine.sigPositionChanged.connect(self.updateFrameSlider_2)
        return        
    
    def initiateImage(self):   
        self.displayImage = self.interleave(self.A)
        #print(self.displayImage.shape)
        if self.nChannels == 2:
            self.displayImage_2 = self.interleave(self.B)
        return

    def interleave(self, A):
        self.nFrames, self.x, self.y = A.shape

        #print(self.nVols, self.nFrames, self.x, self.y )
        
        if self.nVols == 1:
            self.numberOfTimeSlices = self.nFrames
            return A
                
        else:
            self.numberOfTimeSlices = self.getslicesPerVolume()
            interleaved = np.zeros((self.nFrames,self.x,self.y))
            #print(interleaved.shape)
            #print(A.shape)
            z = 0
            for i in np.arange(self.getNumberVols()):
               for j in np.arange(self.getslicesPerVolume()):
                   #print(z, i, j)
                   interleaved[z] = A[(i*self.getslicesPerVolume()) + j%self.getslicesPerVolume()]
                   z = z +1
        return interleaved    
        
       
    def updateImage(self):
        self.displayWindow.imageview.setImage(self.displayImage[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False)
        if self.nChannels == 2:
            self.displayWindow_2.imageview.setImage(self.displayImage_2[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False) 

        if self.overlayFlag == True:
            self.displayWindow_Overlay.imageview.setImage(self.overlayed[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False)
        
        if self.stackedFlag == True:
            self.displayWindow_Overlay.imageview.setImage(self.overlayed[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],autoLevels=False)        
        self.stackedFlag
        return    
        
    def getNFrames(self):
        return self.nFrames
 
    def getFramesPerSlice(self):
        return self.numberOfTimeSlices
 
    def getDisplayFrame(self):
        return self.displayWindow.imageview.currentIndex      

    def getDisplayFrame_2(self):
        return self.displayWindow_2.imageview.currentIndex          

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
    
    def setNumberVols(self,value):
        self.nVols = value
        return

    def getslicesPerVolume(self):
        return self.slicesPerVolume
        
    def getNumberVols(self):
        return self.nVols  
        
    def updateVolumeSize(self, value):
        #set slices per volume
        self.slicesPerVolume = value 
        #set number of volumes
        self.setNumberVols(int(self.getNFrames()/value))
        #update image
        self.initiateImage()
        self.updateImage()
        return

    def setChannelNumber(self, value):
        self.nChannels = value
        return
        
    def updateFrameSlider(self):
        self.displayWindow_2.imageview.setCurrentIndex(self.getDisplayFrame())
        return
        
    def updateFrameSlider_2(self):
        self.displayWindow.imageview.setCurrentIndex(self.getDisplayFrame_2())
        return
        
    def overlay(self):
        red = self.displayImage
        green = self.displayImage_2
        
        self.overlayed = np.zeros((red.shape[0], red.shape[1], red.shape[2], 3))
        
        self.overlayed[:,:,:,0] = red
        self.overlayed[:,:,:,1] = green
        
        #print(self.overlayed.shape)
        self.displayWindow_Overlay = Window(self.overlayed[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],'Overlay')
        self.overlayFlag = True
        return

    def stacked(self):
        self.stacked = np.dstack((self.displayImage, self.displayImage_2))
        #print(self.stacked.shape)
        self.displayWindow_Stacked = Window(self.stacked[self.displayedTimeSlice:(self.displayedTimeSlice+self.numberOfTimeSlices)],'Stacked')
        self.stackedFlag = True
        return
 
camVolumeSlider2 = CamVolumeSlider2()  

class Form3(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form3, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 250

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Volume #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider2.getNumberVols()-1)
        self.SpinBox1.setValue(0)
 
        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider2.getNFrames())
        self.SpinBox2.setValue(camVolumeSlider2.getslicesPerVolume())
 
        
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider2.getNumberVols()-1)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")       
        self.button2 = QtWidgets.QPushButton("Set Slices")   
        self.button3 = QtWidgets.QPushButton("Overlay")   
        self.button4 = QtWidgets.QPushButton("Stacked")  

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        layout.addWidget(self.spinLabel2, 3, 0)
        layout.addWidget(self.SpinBox2, 3, 1)
        layout.addWidget(self.button2, 3, 2)   
        layout.addWidget(self.button3, 4, 0)
        layout.addWidget(self.button4, 4, 1)        
        layout.addWidget(self.button1, 4, 4) 
        
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
        self.button4.clicked.connect(self.stacked)

        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        #camVolumeSlider2.updateSlice(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider2.updateSlice(value)
        return       
    
    def autoLevel(self):
        camVolumeSlider2.displayWindow.imageview.autoLevels()
        if camVolumeSlider2.nChannels == 2:
            camVolumeSlider2.displayWindow_2.imageview.autoLevels()
        return 
    
    def updateVolumeValue(self):
        value = self.SpinBox2.value()
        camVolumeSlider2.updateVolumeSize(value)
        if (camVolumeSlider2.getNFrames()/camVolumeSlider2.getNumberVols())%2 == 0:
            self.SpinBox1.setRange(0,camVolumeSlider2.getNumberVols()-1) #if even, display the last volume 
            self.slider1.setMaximum(camVolumeSlider2.getNumberVols()-1)
        else:
            self.SpinBox1.setRange(0,camVolumeSlider2.getNumberVols()-2) #else, don't display the last volume 
            self.slider1.setMaximum(camVolumeSlider2.getNumberVols()-2)
        return
        
    def overlay(self):
        camVolumeSlider2.overlay()
        return
        
    def stacked(self):
        camVolumeSlider2.stacked()
        return