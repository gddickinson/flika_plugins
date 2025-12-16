from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
#from pyqtgraph.Point import Point
from os.path import expanduser
import os
import math

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

#from flika.process.file_ import open_file
from flika.roi import makeROI



class ScaledAverageSubtract(BaseProcess_noPriorWindow):
    """
    Subtract scaled average image of peak response from stack
    
    Automatically detects peak of response using rolling average
    
    Scales averaged image from 0-1 and subtracts scaled image from every frame
    
    ------------------------------------------------------------------------
    
    Parameters:
    
    number of frames for rolling average window
    number of frames for average peak
    
    User selects window to analyse 
    Current ROI used to make rolling average 
    (if no current ROI drawn by user a central rectangular ROI is generated)
    
    Returns:
    
    New stack (with scaled averaged peak image subtracted)   
    Plot of ROI trace, overlayed with rolling average and peak frame identified
    
    """
    def __init__(self):
        if g.settings['scaledAverageSubtract'] is None or 'averageSize' not in g.settings['scaledAverageSubtract']:
            s = dict()            
            s['windowSize'] = 50
            s['averageSize'] = 100
                 
            g.settings['scaledAverageSubtract'] = s
            
        self.current_ROI = None
        
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, windowSize, averageSize, keepSourceWindow = True):
        '''
        save parameters
        '''
        self.start()
        g.settings['scaledAverageSubtract']['windowSize'] = windowSize
        g.settings['scaledAverageSubtract']['averageSize'] = averageSize
        
        self.runAnalysis()
        
        return 

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        event.accept()
        return

    def gui(self):
        self.gui_reset()
        s=g.settings['scaledAverageSubtract']
        
        #buttons
        self.analysisWin = WindowSelector()
        
        #spinboxes
        self.windowSize_Box = pg.SpinBox(int=True, step=1)
        self.windowSize_Box.setMinimum(1)
        self.windowSize_Box.setMaximum(100000)        
        self.windowSize_Box.setValue(s['windowSize'])  
        
        self.averageSize_Box = pg.SpinBox(int=True, step=1)
        self.averageSize_Box.setMinimum(1)
        self.averageSize_Box.setMaximum(100000)        
        self.averageSize_Box.setValue(s['averageSize']) 

        self.items.append({'name': 'windowSize', 'string': '# of frames for rolling average:', 'object': self.windowSize_Box})           
        self.items.append({'name': 'averageSize', 'string': '# of frames for peak average:', 'object': self.averageSize_Box})           
        self.items.append({'name': 'analysisWindow', 'string': 'Choose window (with ROI) to analyse', 'object': self.analysisWin})         
     
        super().gui()

    def runAnalysis(self):
        #get params
        windowSize = self.getValue('windowSize')
        averageSize = self.getValue('averageSize')
        
        #get data window
        dataWindow = self.getValue('analysisWindow')
        
        #get image array
        A = dataWindow.image
        
        #get array shape
        frames, height ,width = A.shape 
        #print(A.shape)
        
        #plot roi in center
        if dataWindow.currentROI == None:
            centerROI = makeROI('rectangle',[[10, 10], [height-20, width-20]], window=dataWindow)
            
        else:
            centerROI = dataWindow.currentROI
        

        #plot roi average trace
        plot = centerROI.plot()
        
        #get trace data
        trace = centerROI.getTrace()
        
        #define function to get moving average from trace
        def moving_average(x, n=windowSize) :
            return np.convolve(x, np.ones((n,))/n, mode='valid')
            
        #get trace moving average
        movingAverage = moving_average(trace, n=windowSize)   
        
        #set moving average values <= 0 to 0.0000001
        movingAverage[movingAverage <= 0] = 0.0000001
        
        #add moving average trace to plot
        plot.p1.plot(movingAverage , pen=(1,3), symbol=None)
        
        #identify peak of trace
        peakFrame = np.argmax(movingAverage)
        
        #plot peak
        plot.p1.plot(np.array([peakFrame]),np.array([movingAverage[peakFrame]]) , pen=None, symbol='o', symbolSize =20)
        
        #average frames around peak
        start = int(peakFrame - averageSize/2)
        end = int(peakFrame + averageSize/2)
        
        #prevent indexing from reaching past ends of trace
        if start < 0:
            start = 0
            
        if end > frames:
            end = frames
        
        #get peak images
        averageImages = A[start:end]
        
        #display average stack
        #averageImageStack_Win = Window(averageImages, 'Peak Images')
        
        #average peak images
        averageImage = np.mean(averageImages, axis=0)
        
        #display average
        #averageImage_Win = Window(averageImage,'Averaged Peak Images')
        
        #make stack of averaged images
        scaleImage_stack = np.ones_like(A)
        scaleImage_stack = scaleImage_stack * averageImage
        scaleImage_stack = scaleImage_stack[int(windowSize/2):-(int(windowSize/2)-1)]
        
        #scale moving average 0 -> 1
        scale = movingAverage/max(movingAverage)
        
        #scale averaged image stack
        scaleImage_stack = np.multiply(scaleImage_stack,scale[:,None,None])
        
        #show scale images
        #scaleWindow = Window(scaleImage_stack,'Scaled Average Images')
        
        #subtract scaled average from original stack
        A_subtract = np.zeros_like(A)
        A_subtract[int(windowSize/2):-(int(windowSize/2)-1)]  = (A[int(windowSize/2):-(int(windowSize/2)-1)] - scaleImage_stack)
        
        #show subtracted image stack
        subtractWindow = Window(A_subtract,'Subtracted Images')
       
 
scaledAverageSubtract = ScaledAverageSubtract()
