from __future__ import (absolute_import, division,print_function, unicode_literals)
from future.builtins import (bytes, dict, int, list, object, range, str, ascii, chr, hex, input, next, oct, open, pow, round, super, filter, map, zip)
import numpy as np
import cv2
from qtpy import QtWidgets, QtCore, QtGui
import flika
flika_version = flika.__version__
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
from distutils.version import StrictVersion


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox

#Get OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print("OpenCV version %s.%s.%s" % (major_ver, minor_ver, subminor_ver))

#set global variable
global globimg
globimg = 'OFF'



class ZeroSpinBox(QtWidgets.QSpinBox):
    
    zeros = 0
    atzero = QtCore.Signal(int)
    
    
    def __init__(self, parent = None):
        super(ZeroSpinBox, self).__init__(parent)
        self.valueChanged[int].connect(self.checkzero)
        
    def checkzero(self):
        if self.value() == 0:
            self.zeros +=1
            self.atzero.emit(self.zeros)


class Form(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form, self).__init__(parent)

        self.filterBox = QtWidgets.QComboBox()
        self.filterBox.addItem("No Filter")
        self.filterBox.addItem("Canny Filter")
        self.filterBox.addItem("2D Convolution - Average")
        self.filterBox.addItem("2D Convolution - Smooth")
        self.filterBox.addItem("2D Convolution - Gaussian")
        self.filterBox.addItem("2D Convolution - Median")
        self.filterBox.addItem("2D Convolution - Bilateral")
        self.filterBox.addItem("Invert")
        self.filterBox.addItem("Adaptive Threshold")
        self.filterBox.addItem("Laplacian Edge")
        self.filterBox.addItem("Background Subtract")
        
        self.SpinBox1 = QtWidgets.QDoubleSpinBox()
        self.SpinBox1.setRange(0,1000)
        self.SpinBox1.setValue(1.00)

        self.SpinBox2 = QtWidgets.QDoubleSpinBox()
        self.SpinBox2.setRange(0,1000)
        self.SpinBox2.setValue(1.00) 


        self.filterLabel = QtWidgets.QLabel("No Filter")       
        self.filterFlag = 'No Filter'
        
        self.dial = QtWidgets.QDial()
        self.dial.setNotchesVisible(True)
        self.zerospinbox = ZeroSpinBox()
        
        self.button1 = QtWidgets.QPushButton("Click to Take Picture")
        self.getImageFlag = False
        self.button2 = QtWidgets.QPushButton("Quit Live Camera")
        self.liveCameraFlag = False        
        self.button3 = QtWidgets.QPushButton("Black & White")
        self.blackandwhiteFlag = False
        self.button4 = QtWidgets.QPushButton("Start Recording")
        self.recordingFlag = False

		

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.dial)
        layout.addWidget(self.zerospinbox)
        layout.addWidget(self.button4)
        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)

        layout.addWidget(self.filterBox)
        layout.addWidget(self.filterLabel)        
        layout.addWidget(self.SpinBox1)
        layout.addWidget(self.SpinBox2)


        self.setLayout(layout)
 
        self.dial.valueChanged[int].connect(self.zerospinbox.setValue)
        self.zerospinbox.valueChanged[int].connect(self.dial.setValue)      
        self.zerospinbox.atzero.connect(self.announce)
        self.button1.clicked.connect(self.one)
        self.button2.clicked.connect(self.two)   
        self.button3.clicked.connect(self.three)  
        self.button4.clicked.connect(self.four) 
        
        self.filterBox.currentIndexChanged[int].connect(self.updateUi)
        

        self.setWindowTitle("Camera Record Options")
  
    def one(self):
        if self.getImageFlag == False:
            self.getImageFlag = True
            self.button1.setText("Click to Take Picture")

        else:
            self.getImageFlag = False
            openCVcam.getImage()
            self.button1.setText("Click to Take Picture")

    def two(self):
        if self.liveCameraFlag == False:
            self.liveCameraFlag = True            
        else:
            self.liveCameraFlag = False
               
    def three(self):
        if self.blackandwhiteFlag == False:
            self.blackandwhiteFlag = True
            self.button3.setText("Colour")
        else:
            self.blackandwhiteFlag = False
            self.button3.setText("Black & White")

    def four(self):
        if self.recordingFlag == False:
            self.recordingFlag = True
			# Start time
            openCVcam.startRecordingTime = time.time()
            self.button4.setText("Stop Recording")
        else:
            self.recordingFlag = False
			# End time
            openCVcam.endRecordingTime = time.time()
            openCVcam.getRecording()
            self.button4.setText("Start Recording")


			
            
    #def anyButton(self, who):
    #    self.label.setText("You clicked button '%s" % who)

    def announce(self,zeros):
        print ("ZeroSpinBox has been at zero %d times" %zeros)

    def updateUi(self):
        self.filterType = str(self.filterBox.currentText())
        self.filterLabel.setText(self.filterType)
        self.filterFlag = str(self.filterType)

class openCVcam():

    def __init__(self):
        self.recordingList = []
        self.capturedImage = []
        self.startRecordingTime = 0
        self.endRecordingTime = 0

    def startCamera(self):
        '''start Camera
        '''
        cap = cv2.VideoCapture(0)
        
        dialogbox = Form()
        dialogbox.show()
        dialogbox.liveCameraFlag = True 

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          
            if dialogbox.filterFlag == "2D Convolution - Average":
                kernel = np.ones((5,5),np.float32)/25
                gray = cv2.filter2D(gray,-1,kernel)            
            
            if dialogbox.filterFlag == "2D Convolution - Smooth":
                gray = cv2.blur(gray,(5,5))             
            
            if dialogbox.filterFlag == "2D Convolution - Gaussian":
                gray = cv2.GaussianBlur(gray,(5,5),0)
 
            if dialogbox.filterFlag == "2D Convolution - Median": 
                gray = cv2.medianBlur(gray,5)

            if dialogbox.filterFlag == "2D Convolution - Bilateral":                
                gray = cv2.bilateralFilter(gray,9,75,75)
           
            if dialogbox.filterFlag == "Canny Filter":
                gray = cv2.Canny(gray,100,20)

            if dialogbox.filterFlag == "Invert":
                gray = (255-gray)

            if dialogbox.filterFlag =="Adaptive Threshold":
                gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
                gray = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 1)

            if dialogbox.filterFlag=="Laplacian Edge":
                # remove noise
                img = cv2.GaussianBlur(gray,(3,3),0)
                # convolute with proper kernels
                gray = cv2.Laplacian(img,cv2.CV_64F)

            if dialogbox.filterFlag=="Background Subtract":
                fgbg = cv2.BackgroundSubtractorMOG()
                history = 10
                while dialogbox.filterFlag=="Background Subtract":
                    retVal, frame = cap.read()
                    fgmask = fgbg.apply(frame, learningRate=1.0/history)
                    cv2.imshow('Live Camera', fgmask)
                    globimg = gray

                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        globimg = 'OFF'

                        dialogbox.close()
                        break

            if dialogbox.getImageFlag == True:
                if dialogbox.blackandwhiteFlag == True:
                    self.capturedImage = (self.convertImage(gray))
                else:
                    self.capturedImage = (self.convertImage(frame))  

				
            if dialogbox.recordingFlag == True:	
                if dialogbox.blackandwhiteFlag == True:
                    self.recordingList.append(self.convertImage(gray))
                else:
                    self.recordingList.append(self.convertImage(frame))
				

            if dialogbox.blackandwhiteFlag == True:        
            # Display the resulting frame in grey
                cv2.imshow('Live Camera',gray)
                globimg = gray
               
            else: 
            # Display the resulting frame in colour
                cv2.imshow('Live Camera',frame)
                globimg = frame

                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
            if dialogbox.liveCameraFlag == False:
                break
            

        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        globimg = 'OFF'

    def getImage(self):
        img = np.array(self.capturedImage)
        Window(img,'Capture')
        self.capturedImage = []
        return
		
		
    def getRecording(self):

        recordingArray = np.array(self.recordingList)

        # Time elapsed
        seconds = self.endRecordingTime - self.startRecordingTime
        # Estimate frames per second
        fps  = len(self.recordingList) / seconds		
        # Add recording to new window
        message = 'Recording (duration: ' + str(round(seconds,2)) + ' s, fps: ' + str(round(fps,2)) + ')'
        Window(recordingArray, message)
        self.recordingList = []
        return
	
    def convertImage(self, img):
        img = np.rot90(img,k=1)
        img = np.flipud(img)
        return img

openCVcam = openCVcam()


class OpenFile(BaseProcess):

    def __init__(self):
        self.currentFile = None
        self.fileType = None
        self.recordingList = []
        return        

    def startFileOpener(self):
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show() 
        return        
       
    def convertFileToTiff(self):
        if self.fileType == 'avi':
            cap = cv2.VideoCapture(self.currentFile)
            while(cap.isOpened()):
                try:
                    ret, frame = cap.read()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('frame',gray)
                    self.recordingList.append(self.convertImage(gray))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    break
                        
            cap.release()
            cv2.destroyAllWindows()
        
            recordingArray = np.array(self.recordingList)
            self.displayWindow = Window(recordingArray, 'Image Window')
            self.recordingList = []
        
        
        else:
            img = cv2.imread(self.currentFile)
            self.displayWindow = Window(img, 'Image Window')
            self.flipImageLR()
            self.rotateImageClock()
        return

    def convertImage(self, img):
        img = np.rot90(img,k=1)
        img = np.flipud(img)
        return img
        
    def rotateImageCounter(self):
        img = self.displayWindow.imageview.getProcessedImage()
        img = np.rot90(img,k=3)
        self.displayWindow.imageview.setImage(img)       
        return

    def rotateImageClock(self):
        img = self.displayWindow.imageview.getProcessedImage()
        img = np.rot90(img,k=1)
        self.displayWindow.imageview.setImage(img)
        return

    def flipImageLR(self):
        img = self.displayWindow.imageview.getProcessedImage()
        img = np.fliplr(img)
        self.displayWindow.imageview.setImage(img)
        return

    def flipImageUD(self):
        img = self.displayWindow.imageview.getProcessedImage()
        img = np.flipud(img)
        self.displayWindow.imageview.setImage(img)
        return

    def zoomImage(self):
        img = self.displayWindow.imageview.getProcessedImage()
        r = 1.5
        dim = (int(img.shape[1] * r), int(img.shape[0] * r)) 
        # perform the actual resizing of the image and show it
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        self.displayWindow.imageview.setImage(img)        
        return

    def shrinkImage(self):
        img = self.displayWindow.imageview.getProcessedImage()
        r = 1.5
        dim = (int(img.shape[1] / r), int(img.shape[0] / r)) 
        # perform the actual resizing of the image and show it
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        self.displayWindow.imageview.setImage(img)        
        return
        
        

openFile = OpenFile() 

class Form2(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form2, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 500
        self.height = 150

        #buttons
        self.button1 = QtWidgets.QPushButton("select file")
        self.button2 = QtWidgets.QPushButton("convert file")
        
        self.button3 = QtWidgets.QPushButton("rotate 90")
        self.button4 = QtWidgets.QPushButton("rotate -90")
        self.button5 = QtWidgets.QPushButton("flip LR")
        self.button6 = QtWidgets.QPushButton("flip UD")
        
        
        #labels
        self.label1 = QtWidgets.QLabel("Selected file: ")
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)
      
        layout.addWidget(self.button1, 0,0,2,2)
        layout.addWidget(self.button2, 0,3,2,2)
        layout.addWidget(self.label1, 2,0,2,4) 
        layout.addWidget(self.button3, 4,0,2,2)
        layout.addWidget(self.button4, 4,3,2,2)        
        layout.addWidget(self.button5, 6,0,2,2)
        layout.addWidget(self.button6, 6,3,2,2) 

        
          
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("openCV file converter")
        
        #connect buttons
        self.button1.clicked.connect(self.getFileName)
        self.button2.clicked.connect(openFile.convertFileToTiff)
        self.button3.clicked.connect(openFile.rotateImageClock)
        self.button4.clicked.connect(openFile.rotateImageCounter)       
        self.button5.clicked.connect(openFile.flipImageLR)
        self.button6.clicked.connect(openFile.flipImageUD)        
        
        

        return
        
    def getFileName(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Open File", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            openFile.currentFile = fileName
            newText = "Selected file: " + openFile.currentFile
            self.label1.setText(newText)
            openFile.fileType = fileName.split('.')[-1]
            print(openFile.fileType)
        return
 

    
   
            


