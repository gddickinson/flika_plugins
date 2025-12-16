# -*- coding: utf-8 -*-
import os, time
import datetime
from os.path import expanduser
import numpy as np
from numpy import moveaxis
from scipy.ndimage.interpolation import zoom
from qtpy import QtGui, QtWidgets, QtCore
from time import time
from distutils.version import StrictVersion
import pyqtgraph as pg
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.images import image_path
from skimage.transform import rescale
from flika.process.file_ import get_permutation_tuple

from PyQt5.QtCore import QTime, QTimer
from PyQt5.QtWidgets import QApplication, QLCDNumber
from time import sleep

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow

    
def get_transformation_matrix(theta=45):
    """
    theta is the angle of the light sheet
    Look at the pdf in this folder.
    """

    theta = theta/360 * 2 * np.pi # in radians
    hx = np.cos(theta)
    sy = np.sin(theta)
 
    S = np.array([[1, hx, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    return S


def get_transformation_coordinates(I, theta):
    negative_new_max = False
    S = get_transformation_matrix(theta)
    S_inv = np.linalg.inv(S)
    mx, my = I.shape

    four_corners = np.matmul(S, np.array([[0, 0, mx, mx],
                                          [0, my, 0, my],
                                          [1, 1, 1, 1]]))[:-1,:]
    range_x = np.round(np.array([np.min(four_corners[0]), np.max(four_corners[0])])).astype(np.int)
    range_y = np.round(np.array([np.min(four_corners[1]), np.max(four_corners[1])])).astype(np.int)
    all_new_coords = np.meshgrid(np.arange(range_x[0], range_x[1]), np.arange(range_y[0], range_y[1]))
    new_coords = [all_new_coords[0].flatten(), all_new_coords[1].flatten()]
    new_homog_coords = np.stack([new_coords[0], new_coords[1], np.ones(len(new_coords[0]))])
    old_coords = np.matmul(S_inv, new_homog_coords)
    old_coords = old_coords[:-1, :]
    old_coords = old_coords
    old_coords[0, old_coords[0, :] >= mx-1] = -1
    old_coords[1, old_coords[1, :] >= my-1] = -1
    old_coords[0, old_coords[0, :] < 1] = -1
    old_coords[1, old_coords[1, :] < 1] = -1
    new_coords[0] -= np.min(new_coords[0])
    keep_coords = np.logical_not(np.logical_or(old_coords[0] == -1, old_coords[1] == -1))
    new_coords = [new_coords[0][keep_coords], new_coords[1][keep_coords]]
    old_coords = [old_coords[0][keep_coords], old_coords[1][keep_coords]]
    return old_coords, new_coords


def setup_test():
    A = g.win.image
    mt, mx, my = A.shape
    nSteps = 128
    shift_factor = 2
    mv = mt // nSteps  # number of volumes
    A = A[:mv * nSteps]
    B = np.reshape(A, (mv, nSteps, mx, my))

def perform_shear_transform(A, shift_factor, interpolate, datatype, theta):
    A = moveaxis(A, [1, 3, 2, 0], [0, 1, 2, 3])
    m1, m2, m3, m4 = A.shape
    if interpolate:
        A_rescaled = np.zeros((m1*int(shift_factor), m2, m3, m4))
        for v in np.arange(m4):
            print('Upsampling Volume #{}/{}'.format(v+1, m4))
            A_rescaled[:, :, :, v] = rescale(A[:, :, :, v], (shift_factor, 1.), mode='constant', preserve_range=True)
    else:
        A_rescaled = np.repeat(A, shift_factor, axis=0)
    mx, my, mz, mt = A_rescaled.shape
    I = A_rescaled[:, :, 0, 0]
    old_coords, new_coords = get_transformation_coordinates(I, theta)
    old_coords = np.round(old_coords).astype(np.int)
    new_mx, new_my = np.max(new_coords[0]) + 1, np.max(new_coords[1]) + 1
    # I_transformed = np.zeros((new_mx, new_my))
    # I_transformed[new_coords[0], new_coords[1]] = I[old_coords[0], old_coords[1]]
    # Window(I_transformed)
    D = np.zeros((new_mx, new_my, mz, mt))
    D[new_coords[0], new_coords[1], :, :] = A_rescaled[old_coords[0], old_coords[1], :, :]
    E = moveaxis(D, [0, 1, 2, 3], [3, 1, 2, 0])
    E = np.flip(E, 1)
    #Window(E[0, :, :, :])
    E = E.astype(datatype)
    return E
       

        
class FolderSelector(QtWidgets.QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=QtCore.Signal()
    def __init__(self,filetypes='*.*'):
        QtWidgets.QWidget.__init__(self)
        self.button=QtWidgets.QPushButton('Select Folder')
        self.label=QtWidgets.QLabel('None')
        self.window=None
        self.layout=QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''
        
    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QtWidgets.QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    
    
    
    
class OnTheFly(BaseProcess_noPriorWindow):
    """
    Continuous light sheet analysis
    """
    def __init__(self):
        if g.settings['on_the_fly'] is None or 'updateRateVolViewer' not in g.settings['on_the_fly']:
            s = dict()
            s['sliceNumber'] = 0
            s['batchSize'] = 1
            s['updateRate'] = 2 #seconds
            s['updateRateVolViewer'] = 120 #seconds            
            s['nSteps'] = 1
            s['displaySlice'] = 0
            s['shift_factor'] = 1
            s['theta'] = 45
            s['triangle_scan'] = False
            s['interpolate'] = False
            s['trim_last_frame'] = False
            s['nChannels'] = 1
            g.settings['on_the_fly'] = s
        
        BaseProcess_noPriorWindow.__init__(self)
        

    def __call__(self, recordingFolder, batchSize, updateRate, updateRateVolViewer, nSteps, displaySlice, shift_factor, theta, triangle_scan, interpolate, trim_last_frame, zscan, nChannels, keepSourceWindow=False):
        g.settings['on_the_fly']['recordingFolder'] = recordingFolder
        g.settings['on_the_fly']['batchSize'] = batchSize
        g.settings['on_the_fly']['updateRate'] = updateRate
        g.settings['on_the_fly']['updateRateVolViewer'] = updateRateVolViewer     
        g.settings['on_the_fly']['nSteps']=nSteps
        g.settings['on_the_fly']['displaySlice']=displaySlice      
        g.settings['on_the_fly']['shift_factor']=shift_factor
        g.settings['on_the_fly']['theta']=theta
        g.settings['on_the_fly']['triangle_scan'] = triangle_scan
        g.settings['on_the_fly']['interpolate'] = interpolate
        g.settings['on_the_fly']['trim_last_frame'] = trim_last_frame 
        g.settings['on_the_fly']['zscan'] = zscan
        g.settings['on_the_fly']['nChannels'] = nChannels
        g.m.statusBar().showMessage("Starting ...")
        t = time()
        
        self.mainGUI = MainGUI()
        self.mainGUI.show()


    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)

 
    def gui(self):
        s=g.settings['on_the_fly']
        self.gui_reset()
        
        self.batchSize = pg.SpinBox(int=True, step=1)
        self.batchSize.setMinimum(1)
        self.batchSize.setValue(s['batchSize'])
        
        self.nSteps = pg.SpinBox(int=True, step=1)
        self.nSteps.setMinimum(1)
        self.nSteps.setValue(s['nSteps'])
              
        self.shift_factor = pg.SpinBox(int=False, step=.1)
        self.shift_factor.setValue(s['shift_factor'])

        self.theta = pg.SpinBox(int=True, step=1)
        self.theta.setValue(s['theta'])

        self.triangle_scan = CheckBox()
        self.triangle_scan.setValue(s['triangle_scan'])

        self.interpolate = CheckBox()
        self.interpolate.setValue(s['interpolate'])

        self.trim_last_frame = CheckBox()
        self.trim_last_frame.setValue(s['trim_last_frame'])

        self.zscan = CheckBox()
        self.zscan.setValue(s['trim_last_frame'])

        self.recordingFolder = FolderSelector('*.txt')
        
        self.displaySlice = pg.SpinBox(int=True, step=1)
        self.displaySlice.setMinimum(0)
        self.displaySlice.setValue(s['displaySlice'])
        
        self.updateRate = pg.SpinBox(int=True, step=1)
        self.updateRate.setMinimum(1)
        self.updateRate.setValue(s['updateRate'])

        self.updateRateVolViewer = pg.SpinBox(int=True, step=1)
        self.updateRateVolViewer.setMinimum(1)
        self.updateRateVolViewer.setValue(s['updateRateVolViewer'])
        
        self.nChannels = pg.SpinBox(int=True, step=1)
        self.nChannels.setMinimum(1)
        self.nChannels.setMaximum(2)       
        self.nChannels.setValue(s['nChannels'])        

        self.items.append({'name':'recordingFolder','string':'Results Folder Location','object': self.recordingFolder})       
        self.items.append({'name': 'batchSize', 'string': 'Batch Size', 'object': self.batchSize})
        self.items.append({'name': 'updateRate', 'string': 'Update Rate (seconds)', 'object': self.updateRate}) 
        self.items.append({'name': 'updateRateVolViewer', 'string': 'Volume Viewer Update Rate (seconds)', 'object': self.updateRateVolViewer})         
        self.items.append({'name': 'nSteps', 'string': 'Number of Steps Per Volume', 'object': self.nSteps})
        self.items.append({'name': 'displaySlice', 'string': 'Display Slice#', 'object': self.displaySlice})        
        self.items.append({'name': 'shift_factor', 'string': 'Shift Factor', 'object': self.shift_factor})
        self.items.append({'name': 'theta', 'string': 'Theta', 'object': self.theta})
        self.items.append({'name': 'triangle_scan', 'string': 'Triangle Scan', 'object': self.triangle_scan})
        self.items.append({'name': 'interpolate', 'string': 'Interpolate', 'object': self.interpolate})
        self.items.append({'name': 'trim_last_frame', 'string': 'Trim Last Frame', 'object': self.trim_last_frame})
        self.items.append({'name': 'zscan', 'string': 'Z Scan', 'object': self.zscan})       
        self.items.append({'name': 'nChannels', 'string': 'Number of Channels', 'object': self.nChannels}) 
        
        super().gui()
        
onTheFly = OnTheFly()

class MainGUI(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(MainGUI, self).__init__(parent)
        s=g.settings['on_the_fly']
        
        self.nSteps = s['nSteps']
        sliderMax = self.nSteps
        self.folderPath = s['recordingFolder']
        self.sliceNumber = None
        self.recordingArray_1 = None
        self.recordingArray_FLAG = False
        self.initializeArray = np.zeros([1,100,100])
        self.updateRate = s['updateRate'] #seconds
        self.updateRateVolViewer= s['updateRateVolViewer'] #seconds       
        self.folderContents = {}
        self.loading = False
        self.displayWindow_FLAG = False
        
        self.displaySlice = s['displaySlice']
        
        self.nChannels = s['nChannels']

        self.shift_factor = s['shift_factor']
        self.theta = s['theta']
        self.triangle_scan = s['triangle_scan']
        self.interpolate = s['interpolate']
        self.trim_last_frame = s['trim_last_frame']        
        self.zscan = s['zscan']       

        self.volumeViewer_FLAG = False

        self.numberVolsAdded_channel1 = 0
        self.numberVolsAdded_channel2 = 0        

        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 700
        self.height = 150

        #labels
        self.folderLabel = QtWidgets.QLabel(s['recordingFolder'])
        
        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,sliderMax)
        self.SpinBox1.setValue(0)
 
        #buttons
        self.startButton = QtWidgets.QPushButton('Start')
        self.startButton.pressed.connect(self.start)
        
        self.stopButton = QtWidgets.QPushButton('Stop')
        self.stopButton.pressed.connect(self.stop)
        
        self.showElapsedTimeButton = QtWidgets.QPushButton('Show Elapsed Time')
        self.showElapsedTimeButton.pressed.connect(self.showElapsedTime)

        self.startVolumeViewer_button = QtWidgets.QPushButton('Run Volume Viewer Now')
        self.startVolumeViewer_button.pressed.connect(self.startVolumeViewer)

        self.stopVolumeViewer_button = QtWidgets.QPushButton('Close Volume Viewer')
        self.stopVolumeViewer_button.pressed.connect(self.stopVolumeViewer) 

        self.startAutoVolumeViewer_button = QtWidgets.QPushButton('Start Auto Volume Viewer')
        self.startAutoVolumeViewer_button.pressed.connect(self.startAutoVolumeViewer)
        
        self.pauseVolumeViewer_button = QtWidgets.QPushButton('Pause Auto Volume Viewer')
        self.pauseVolumeViewer_button.pressed.connect(self.pauseVolumeViewer) 
 
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(sliderMax)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(self.folderLabel, 1, 0,1,5) 
        layout.addWidget(self.startButton, 2, 0) 
        layout.addWidget(self.stopButton, 2, 1) 
        layout.addWidget(self.showElapsedTimeButton, 2, 2) 
        layout.addWidget(self.spinLabel1, 4, 0)        
        layout.addWidget(self.SpinBox1, 4, 1)
        layout.addWidget(self.slider1, 5, 0, 2, 5)
        layout.addWidget(self.startVolumeViewer_button, 7, 0)
        layout.addWidget(self.stopVolumeViewer_button, 7, 1)
        layout.addWidget(self.startAutoVolumeViewer_button, 7, 2)       
        layout.addWidget(self.pauseVolumeViewer_button, 7, 3)
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("On-the-fly Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)

        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        self.displaySlice = value
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        return       
 
    def start(self):
        self.clock = Clock()
        self.initializeFolderCheck()
        
        #start display window(s)
        #single-channel
        self.displayWindow_1 = Window(self.initializeArray ,'Channel 1')
        #self.volumeArrayWindow_1 = Window(self.initializeArray ,'Channel 1 for v-Viewer')
        #2-channel
        if self.nChannels == 2:
            self.displayWindow_2 = Window(self.initializeArray ,'Channel 2')    
        
        #initialize FLAG settings
        self.displayWindow_FLAG = True
        self.autolevelsFlag_1 = True
        self.autolevelsFlag_2 = True
        return

        
    def getTimeSortedDirList(self):
        name_list = os.listdir(self.folderPath)
        full_list = [self.folderPath + '/' + i for i in name_list]
        time_sorted_list = sorted(full_list, key=os.path.getmtime)
        sorted_filename_list = [ os.path.basename(i) for i in time_sorted_list]
        #print(sorted_filename_list)
        return sorted_filename_list
        
    def initializeFolderCheck(self):
        self.folderContents = dict([f,None] for f in self.getTimeSortedDirList())
        return
       
    def checkFolder(self):
        content = self.getTimeSortedDirList()
        if content != '':
            updatedContents = dict([f,None] for f in content)
            added = [f for f in updatedContents if not f in self.folderContents]
            for folder in added:
                #pathName = os.path.join(self.folderPath , folder)
                pathName = self.folderPath + '/' + folder
                file = os.listdir(pathName)[0]
                #filename = os.path.join(pathName, file)
                filename = pathName + '/' + file
                self.addVolumeToArray(filename)
            self.folderContents = {**self.folderContents,**updatedContents}   
            return
        else:
            return
        
    def addVolumeToArray(self, filename):
        if self.loading == True:
            return
    
        if self.recordingArray_FLAG != False: #continue adding frames to recording array
            self.loading = True
            #wrap loading in a try/except in case loading from disk fails - this might be done better with client/server
            try:
                newVol_1, newVol_2 = load_tiff.openTiff(filename)
            except:
                sleep(3) #sleep and try again to give file time to be unlocked
                newVol_1, newVol_2 = load_tiff.openTiff(filename)
            
            #single-channel recording
            self.recordingArray_1 = np.vstack((self.recordingArray_1, newVol_1)) #to go to batch for light-sheet analysis
            self.numberVolsAdded_channel1 += 1 
            #print('Channel 1: ' + str(self.recordingArray_1.shape))
            
            #2-channel recording
            if self.nChannels == 2:
                self.recordingArray_2 = np.vstack((self.recordingArray_2, newVol_2))
                self.numberVolssAdded_channel2 += 1 
                #print('Channel 2: ' + str(self.recordingArray_2.shape))
            
            self.loading = False
            
        else: #initiate recording array
            self.loading = True
            try:
                self.recordingArray_1, self.recordingArray_2 = load_tiff.openTiff(filename)
            except:
                sleep(3)   #sleep and try again to give file time to be unlocked
                self.recordingArray_1, self.recordingArray_2 = load_tiff.openTiff(filename)
            
            self.loading = False
            self.recordingArray_FLAG = True
        return

        
    def setFrameIndex(self):
        #create list of frames to be displayed
        nFrames, x, y = self.recordingArray_1.shape
        self.frameIndex = np.arange(self.displaySlice,nFrames,self.nSteps)
        #print(self.frameIndex)
        return

    def stop(self):
        self.clock.stopClock()
        return        

    def showElapsedTime(self):
        self.clock.show()
        return
 
    def updateCall(self):
        self.checkFolder()
        #self.test()
        
        #update list of frames to be displayed
        if self.numberVolsAdded_channel1 > 0:
            
            self.setFrameIndex() 
            
            if self.displayWindow_FLAG == True:
                index = self.displayWindow_1.imageview.currentIndex 
                levels_1 = self.displayWindow_1.imageview.getHistogramWidget().getLevels()
                
                #single-channel
                self.displayWindow_1.imageview.setImage(self.recordingArray_1[self.frameIndex], levels=levels_1)
                
                #window for volume Array (full recording)
                #index_vArray = self.volumeArrayWindow_1.imageview.currentIndex 
                #levels_vArray = self.volumeArrayWindow_1.imageview.getHistogramWidget().getLevels()
                #self.volumeArrayWindow_1.imageview.setImage(self.recordingArray_1, levels=levels_vArray)
                #self.volumeArrayWindow_1.imageview.setCurrentIndex(index_vArray)
                print(self.recordingArray_1.shape)
                
                #on first display run autolevel
                if self.autolevelsFlag_1:
                    self.displayWindow_1.imageview.autoLevels()
                    self.autolevelsFlag_1 = False
                
                self.displayWindow_1.imageview.setCurrentIndex(index)
                
                #2-channel
                if self.nChannels == 2:
                    #index = self.displayWindow_1.imageview.currentIndex 
                    levels_2 = self.displayWindow_2.imageview.getHistogramWidget().getLevels()
                
                
                    self.displayWindow_2.imageview.setImage(self.recordingArray_2[self.frameIndex], levels=levels_2)
                    #on first display run autolevel
                    if self.autolevelsFlag_2:
                        self.displayWindow_2.imageview.autoLevels()
                        self.autolevelsFlag_2 = False
                
                    self.displayWindow_2.imageview.setCurrentIndex(index)
                
                
        return
        
        
    def test(self):
        for key, value in self.folderContents.items() :
            print (key)
        if self.recordingArray_FLAG != False:
            print(self.recordingArray_1.shape)
        else:
            print('no array')
        return
 
    def startVolumeViewer(self):

        A = np.copy(self.recordingArray_1)
        
        if self.zscan:
            A = A.swapaxes(1,2)
        
        mt, mx, my = A.shape
        
        if self.triangle_scan:
            for i in np.arange(mt // (self.nSteps * 2)):
                t0 = i * self.nSteps * 2 + self.nSteps
                tf = (i + 1) * self.nSteps * 2
                A[t0:tf] = A[tf:t0:-1]
        
        mv = mt // self.nSteps  # number of volumes
        
        A = A[:mv * self.nSteps]
        B = np.reshape(A, (mv, self.nSteps, mx, my))
        A_dataType = A.dtype
        
        A = np.zeros((2,2)) #clear array to save memory
        
        if self.trim_last_frame:
            B = B[:, :-1, :, :]

        D = perform_shear_transform(B, self.shift_factor, self.interpolate, A_dataType, self.theta)

        B = np.zeros((2,2)) #clear array to save memory
        
        w = Window(np.squeeze(D[:, 0, :, :]), name='Volume View')
        w.volume = D

        D = np.zeros((2,2)) #clear array to save memory
        
        self.volumeViewer = Volume_Viewer(w)
        return 
    
    def stopVolumeViewer(self):
        self.volumeViewer.closeAll()
        self.volumeViewer = None
        return
        
    def pauseVolumeViewer(self):
        self.volumeViewer_FLAG = False
        return
    
    def startAutoVolumeViewer(self):
        self.volumeViewer_FLAG = True
        return
        
class Clock(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        layout = QtWidgets.QStackedLayout()
        self.iterations = 0
        #Background area test
        self.background = QtWidgets.QLabel(self)
        self.background.show()

        #Setup text area for clock
        newfont = QtGui.QFont("Consolas",120, QtGui.QFont.Bold)
        self.lbl1 = QtWidgets.QLabel()
        self.lbl1.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl1.setFont(newfont)
        self.lbl1.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        #add window title
        self.setWindowTitle("onTheFly Time Elapsed")
        
        #Timer to refresh connection
        self.timer = QtCore.QTimer(self)
        #timer.timeout.connect(self.showTime)
        self.timer.timeout.connect(self.updates)
        self.timer.start(1000) #update every 1s
        
        #Start QTime
        self.t = QtCore.QTime()
        self.t.start()
        
        #run first update
        self.updates()
        
        #layout area for widgets
        layout.addWidget(self.background)
        layout.addWidget(self.lbl1)
        layout.setCurrentIndex(1)
        self.setLayout(layout)
        self.setGeometry(300,300,250,150)
        self.show()

        
    def updates(self):
        #update timer display
        self.showElapsedTime()
        
        #use updateRate to limit number of updateCalls made
        if self.iterations % onTheFly.mainGUI.updateRate == 0:
            onTheFly.mainGUI.updateCall()
            print('update')
            
        #activate volumeVIewer according to it's update rate
        if onTheFly.mainGUI.volumeViewer_FLAG:
            if self.iterations % onTheFly.mainGUI.updateRateVolViewer == 0:
                try:
                    onTheFly.mainGUI.stopVolumeViewer()
                except:
                    pass
                onTheFly.mainGUI.startVolumeViewer()
                print('Volume Viewer update')       
            
        self.iterations += 1 #approx 1 iteration/sec
        return
        
    def showTime(self):
        time = QtCore.QTime.currentTime()
        text = time.toString('hh:mm:ss')
        if (time.second() % 2) == 0:
            text = text[:2] + ' ' + text[3:]
        self.lbl1.setText(text)
 
    def showElapsedTime(self):
        self.secondsElapsed = self.t.elapsed()/1000
        m, s = divmod(self.secondsElapsed , 60)
        h, m = divmod(m, 60)
        
        if h < 10:
            hours = '0'+ str(int(h))    
        else: hours = str(int(h))
            
        if m < 10:
            minutes = '0'+ str(int(m))    
        else: minutes = str(int(m))
        
        if s < 10:
            seconds = '0'+ str(int(s))
        else: seconds = str(int(s))
        
        text = hours + ':' + minutes + ':' + seconds
        #print(text)
        self.lbl1.setText(text)
    
    def stopClock(self):
        #print('stop')
        #self.t.stop()
        self.timer.timeout.disconnect(self.updates)
        self.timer.stop() 
        self.close()
        return

 
class Volume_Viewer(QtWidgets.QWidget):
    closeSignal=QtCore.Signal()

    def show_wo_focus(self):
        self.show()
        self.window.activateWindow()  # for Windows
        self.window.raise_()  # for MacOS

    def __init__(self,window=None,parent=None):
        super(Volume_Viewer,self).__init__(parent)  # Create window with ImageView widget
        g.m.volume_viewer = self
        window.lostFocusSignal.connect(self.hide)
        window.gainedFocusSignal.connect(self.show_wo_focus)
        self.window=window
        self.setWindowTitle('Light Sheet Volume View Controller')
        self.setWindowIcon(QtGui.QIcon(image_path('favicon.png')))
        self.setGeometry(QtCore.QRect(422, 35, 222, 86))
        self.layout = QtWidgets.QVBoxLayout()
        self.vol_shape=window.volume.shape
        mv,mz,mx,my=window.volume.shape
        self.currentAxisOrder=[0,1,2,3]
        self.current_v_Index=0
        self.current_z_Index=0
        self.current_x_Index=0
        self.current_y_Index=0
        self.formlayout=QtWidgets.QFormLayout()
        self.formlayout.setLabelAlignment(QtCore.Qt.AlignRight)
        self.xzy_position_label = QtWidgets.QLabel('Z position')
        self.zSlider=SliderLabel(0)
        self.zSlider.setRange(0,mz-1)
        self.zSlider.label.valueChanged.connect(self.zSlider_updated)
        self.zSlider.slider.mouseReleaseEvent=self.zSlider_release_event
        
        self.sideViewOn=CheckBox()
        self.sideViewOn.setChecked(False)
        self.sideViewOn.stateChanged.connect(self.sideViewOnClicked)
        
        self.sideViewSide = QtWidgets.QComboBox(self)
        self.sideViewSide.addItem("X")
        self.sideViewSide.addItem("Y")
        
        self.MaxProjButton = QtWidgets.QPushButton('Max Intenstiy Projection')
        self.MaxProjButton.pressed.connect(self.make_maxintensity)
        
        self.exportVolButton = QtWidgets.QPushButton('Export Volume')
        self.exportVolButton.pressed.connect(self.export_volume)
        
        self.formlayout.addRow(self.xzy_position_label,self.zSlider)
        self.formlayout.addRow('Side View On',self.sideViewOn)
        self.formlayout.addRow('Side View Side',self.sideViewSide)
        self.formlayout.addRow('', self.MaxProjButton)
        self.formlayout.addRow('', self.exportVolButton)
        
        self.layout.addWidget(self.zSlider)
        self.layout.addLayout(self.formlayout)
        self.setLayout(self.layout)
        self.setGeometry(QtCore.QRect(381, 43, 416, 110))
        self.show()

    def closeEvent(self, event):
        event.accept() # let the window close
        
    def zSlider_updated(self,z_val):
        self.current_v_Index=self.window.currentIndex
        vol=self.window.volume
        testimage=np.squeeze(vol[self.current_v_Index,z_val,:,:])
        viewRect = self.window.imageview.view.targetRect()
        self.window.imageview.setImage(testimage,autoLevels=False)
        self.window.imageview.view.setRange(viewRect, padding = 0)
        self.window.image = testimage
        
    def zSlider_release_event(self,ev):
        vol=self.window.volume
        if self.currentAxisOrder[1]==1: # 'z'
            self.current_z_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_z_Index,:,:])
        elif self.currentAxisOrder[1]==2: # 'x'
            self.current_x_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_x_Index,:,:])
        elif self.currentAxisOrder[1]==3: # 'y'
            self.current_y_Index=self.zSlider.value()
            image=np.squeeze(vol[:,self.current_y_Index,:,:])

        viewRect = self.window.imageview.view.viewRect()
        self.window.imageview.setImage(image,autoLevels=False)
        self.window.imageview.view.setRange(viewRect, padding=0)
        self.window.image = image
        if self.window.imageview.axes['t'] is not None:
            self.window.imageview.setCurrentIndex(self.current_v_Index)
        self.window.activateWindow()  # for Windows
        self.window.raise_()  # for MacOS
        QtWidgets.QSlider.mouseReleaseEvent(self.zSlider.slider, ev)
    
    def sideViewOnClicked(self, checked):
        self.current_v_Index=self.window.currentIndex
        vol=self.window.volume
        if checked==2: #checked=True
            assert self.currentAxisOrder==[0,1,2,3]
            side = self.sideViewSide.currentText()
            if side=='X':
                vol=vol.swapaxes(1,2)
                self.currentAxisOrder=[0,2,1,3]
                vol=vol.swapaxes(2,3)
                self.currentAxisOrder=[0,2,3,1]
            elif side=='Y':
                vol=vol.swapaxes(1,3)
                self.currentAxisOrder=[0,3,2,1]
        else: #checked=False
            if self.currentAxisOrder == [0,3,2,1]:
                vol=vol.swapaxes(1,3)
                self.currentAxisOrder=[0,1,2,3]
            elif self.currentAxisOrder == [0,2,3,1]:
                vol=vol.swapaxes(2,3)
                vol=vol.swapaxes(1,2)
                self.currentAxisOrder=[0,1,2,3]
        if self.currentAxisOrder[1]==1: # 'z'
            idx=self.current_z_Index
            self.xzy_position_label.setText('Z position')
            self.zSlider.setRange(0,self.vol_shape[1]-1)
        elif self.currentAxisOrder[1]==2: # 'x'
            idx=self.current_x_Index
            self.xzy_position_label.setText('X position')
            self.zSlider.setRange(0,self.vol_shape[2]-1)
        elif self.currentAxisOrder[1]==3: # 'y'
            idx=self.current_y_Index
            self.xzy_position_label.setText('Y position')
            self.zSlider.setRange(0,self.vol_shape[3]-1)
        image=np.squeeze(vol[:,idx,:,:])
        self.window.imageview.setImage(image,autoLevels=False)
        self.window.volume=vol
        self.window.imageview.setCurrentIndex(self.current_v_Index)
        self.zSlider.setValue(idx)

    def make_maxintensity(self):
        vol=self.window.volume
        new_vol=np.max(vol,1)
        if self.currentAxisOrder[1]==1: # 'z'
            name='Max Z projection'
        elif self.currentAxisOrder[1]==2: # 'x'
            name = 'Max X projection'
        elif self.currentAxisOrder[1]==3: # 'y'
            name = 'Max Y projection'
        Window(new_vol, name=name)
        
    def export_volume(self):
        vol=self.window.volume
        export_path = QtWidgets.QFileDialog.getExistingDirectory(g.m, "Select a parent folder to save into.", expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        export_path = os.path.join(export_path, 'light_sheet_vols')
        i=0
        while os.path.isdir(export_path+str(i)):
            i+=1
        export_path=export_path+str(i)
        os.mkdir(export_path) 
        for v in np.arange(len(vol)):
            A=vol[v]
            filename=os.path.join(export_path,str(v)+'.tiff')
            if len(A.shape)==3:
                A=np.transpose(A,(0,2,1)) # This keeps the x and the y the same as in FIJI
            elif len(A.shape)==2:
                A=np.transpose(A,(1,0))
            tifffile.imsave(filename, A)

    def closeAll(self):
        self.window.close()
        self.close()
        return

class Load_tiff ():
    """ load_tiff()
    This function loads tiff files from lightsheet experiments with multiple channels and volumes.

    """

    def __init__(self):
        pass
  
            
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
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
        elif set(axes) == set(['series', 'height', 'width']):  # single channel, single-volume
            target_axes = ['series', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
        elif set(axes) == set(['time', 'height', 'width']):  # single channel, single-volume
            target_axes = ['time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            nFrames, x, y = A.shape
            A = A.reshape(nFrames,x,y)
            #newWindow = Window(A,'Loaded Tiff')
            return A, None
            
        elif set(axes) == set(['time', 'depth', 'channel', 'height', 'width']):  # multi-channel, multi-volume
            target_axes = ['channel','time','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            n1Scans, n1Frames, x1, y1 = B.shape
            n2Scans, n2Frames, x2, y2 = C.shape

            B = B.reshape(n1Scans*n1Frames,x1,y1)
            C = C.reshape(n2Scans*n2Frames,x2,y2)

            #channel_1 = Window(B,'Channel 1')
            #channel_2 = Window(C,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
            
        elif set(axes) == set(['depth', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','depth', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]

            #channel_1 = Window(B,'Channel 1')
            #channel_2 = Window(C,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
        
        elif set(axes) == set(['time', 'channel', 'height', 'width']):  # multi-channel, single volume
            target_axes = ['channel','time', 'width', 'height']
            perm = get_permutation_tuple(axes, target_axes)
            A = np.transpose(A, perm)
            B = A[0]
            C = A[1]
            
            nChannels, nFrames, x, y = A.shape
            
            n1Frames, x1, y1 = B.shape
            n2Frames, x2, y2 = C.shape

            #uncomment to display original
            #self.channel_1 = Window(B,'Channel 1')
            #self.channel_2 = Window(B,'Channel 2')
            
            #clear A array to reduce memory use
            A = np.zeros((2,2))
            return B, C
        
load_tiff = Load_tiff()   
