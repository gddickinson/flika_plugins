from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy import interpolate
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
from copy import deepcopy

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

from flika.roi import makeROI
from tqdm import tqdm

class Flash_remover(BaseProcess_noPriorWindow):
    """
    Remove flash artifact from movies. 
    
    """
    def __init__(self):

        if g.settings['flash_remover'] is None or 'windowSize' not in g.settings['flash_remover']:
            s = dict()
            s['flashRangeStart'] = 0
            s['flashRangeEnd'] = 1000
            s['windowSize'] = 100            
 
            g.settings['flash_remover'] = s
        super().__init__()


    def __call__(self, keepSourceWindow=False):
        g.settings['flash_remover']['flashRangeStart'] = self.getValue('flashRangeStart')
        g.settings['flash_remover']['flashRangeEnd'] = self.getValue('flashRangeEnd')
        g.settings['flash_remover']['windowSize'] = self.getValue('windowSize')        
        return

    def removeFlash(self):
        method = self.getValue('method')
        if method == 1:
            #interpolation method
            print('linear interpolation')
            self.removeFlash_interpolate()
        elif method == 2:
            #subtraction method
            print('subtraction')
            self.removeFlash_subtraction()
        else:
            print('no method')
        return

    def autodetectFlash(self):
        rangeStart =  self.getValue('flashRangeStart')
        rangeEnd = self.getValue('flashRangeEnd') 
        windowSize =  self.getValue('windowSize')
        showAverage = self.getValue('showAverage')
        useROI = self.getValue('useROI')
        
        #get data window
        dataWindow = self.getValue('window')
        
        #get image array
        A = dataWindow.image
        
        #get array shape
        frames, height ,width = A.shape 
                
        #plot roi in center
        if useROI:
            if dataWindow.currentROI == None:
                print('No ROI detected: generating center ROI')
                centerROI = makeROI('rectangle',[[10, 10], [height-20, width-20]], window=dataWindow)
                
            else:
                centerROI = dataWindow.currentROI
        
        else:   
            centerROI = makeROI('rectangle',[[10, 10], [height-20, width-20]], window=dataWindow)
        
        #plot roi average trace
        if showAverage:
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
        if showAverage: 
            plot.p1.plot(movingAverage , pen=(1,3), symbol=None)
        
        #identify peak of trace - first frame of flash
        flashStart = rangeStart + np.argmax(movingAverage[rangeStart:rangeEnd])
        
        #identify 1st drop after peak - end of flash
        flashEnd = flashStart + np.argmin(movingAverage[flashStart:rangeEnd])
        
        #plot flash start and end
        if showAverage:
            plot.p1.plot(np.array([flashStart]),np.array([movingAverage[flashStart]]) , pen=None, symbol='o', symbolSize =20)
            plot.p1.plot(np.array([flashEnd]),np.array([movingAverage[flashEnd]]) , pen=None, symbol='o', symbolSize =20)
        
        #remove center ROI
        if showAverage == False:
            centerROI.delete()
        
        return flashStart, flashEnd
        

    def removeFlash_subtraction(self): 
        #print('Not implemented')
        #return
        img = deepcopy(self.getValue('window').image)        

        if self.getValue('manualFlash'):
            #manual
            flashStart = self.getValue('flashStart')
            flashEnd = self.getValue('flashEnd')  
            
        else:
            #autodetect
            flashStart, flashEnd = self.autodetectFlash()        

        #buffer flash time ends
        flashStart = flashStart -1 
        flashEnd = flashEnd +2       
                    
        flash = img[flashStart:flashEnd]

        #TODO
        #get baseline from 100 frames before flash
        baseline = np.mean(img[flashStart-102:flashStart-2], axis=0)
        #self.baseline_win = Window(baseline,'baseline')   
        
        #determine noise of baseline
        baseNoise = np.std(img[flashStart-102:flashStart-2])
        print('baseNoise: ',baseNoise)
        
        #get mean inital flash increase
        flashIncrease = np.mean(flash[2:12], axis=0) - baseline
        #self.flashIncrease_win = Window(flashIncrease,'flashIncrease') 
        
        #get noise of flash
        flashNoise = np.std(flash[2:12])
        print('flashNoise: ',flashNoise)

        #flashNoise:baseNoise ratio
        noiseRatio = flashNoise/baseNoise
        
        #subtract flashIncrease from img
        #flashReplace = flash - flashIncrease
        #self.flashReplace_win = Window(flashReplace,'flashReplace')         
        
        #scaled reduction of flash
        flashReplace = np.divide(img[flashStart-1:flashEnd+1], noiseRatio)
        #self.flashReplace_win = Window(flashReplace,'flashReplace')          
        img[flashStart-1:flashEnd+1] = flashReplace
        
        #display stack in new window
        self.flashRemoved_win = Window(img,'Flash Removed (subtraction)')
        return

                    
    def removeFlash_interpolate(self):
        img = deepcopy(self.getValue('window').image)
        
        if self.getValue('manualFlash'):
            #manual
            flashStart = self.getValue('flashStart')
            flashEnd = self.getValue('flashEnd')  
            
        else:
            #autodetect
            flashStart, flashEnd = self.autodetectFlash()
            
            
        #buffer flash time ends
        flashStart = flashStart -1 
        flashEnd = flashEnd +2       
                    
        flash = img[flashStart:flashEnd]
        n, r, c = flash.shape
        
        flashReplace = np.zeros_like(flash)
    
        # update each pixel in image with interpolated values  
        for row in range(r):
            for col in range(c):
                points = range(0,n)
                xp = [0,n]
                fp = [flash[0,row,col],flash[-1,row,col]]
                
                interp_data = np.interp(points,xp,fp)

                flashReplace[0:n,row,col] = interp_data

        
        #add noise
        if self.getValue('addNoise'):
            flashLength = flashEnd - flashStart
            if flashStart-flashLength <0:
                print('Not enough trace to sample before flash to get noise')
                noise = np.zeros_like(flash)
            else:
                noise = self.generateNoise(flashStart-flashLength,flashStart)
        else:
            noise = np.zeros_like(flash)
            
                        
        img[flashStart:flashEnd] = (flashReplace + (noise-np.mean(noise,axis=0)))
            
        #display stack in new window
        self.flashRemoved_win = Window(img,'Flash Removed (linear interpolation)')
        return
                    
    def generateNoise(self, start, end):
        img = deepcopy(self.getValue('window').image)
        beforeFlashImg = img[start:end]

        #TODO - change TOO SLOW
        # #simulating noise
        # n, r, c = beforeFlashImg.shape      
        # randomImg = np.zeros_like(beforeFlashImg)
        
        # print('Generating Noise')
        # for row in tqdm(range(r)):
        #     for col in range(c):
        #         for pixel in range(n):
        #             minVal = min(beforeFlashImg[0:n,row,col])
        #             maxVal = max(beforeFlashImg[0:n,row,col])
        #             randomNoise = np.random.randint(minVal,maxVal)
        #             randomImg[pixel,row,col] = randomNoise 
        
        #return randomImg
        
        #for now returning stack before flash as noise substitue
        return beforeFlashImg


    def gui(self):
        s=g.settings['flash_remover']
        self.gui_reset()
        self.window = WindowSelector()
        self.removeFlash_button = QPushButton('Remove Flash')
        self.removeFlash_button.pressed.connect(self.removeFlash)
        
        self.manuallySetFlash_check = CheckBox()
        self.manuallySetFlash_check.setValue(False)
        
        self.addNoise_check = CheckBox()
        self.addNoise_check.setValue(True)  
        
        self.flashStart_slider = pg.SpinBox(int=True, step=1)
        self.flashStart_slider.setValue(0)

        self.flashEnd_slider = pg.SpinBox(int=True, step=1)
        self.flashEnd_slider.setValue(1)  
        
        self.flashRangeStart_slider = pg.SpinBox(int=True, step=1)
        self.flashRangeStart_slider.setValue(s['flashRangeStart'])

        self.flashRangeEnd_slider = pg.SpinBox(int=True, step=1)
        self.flashRangeEnd_slider.setValue(s['flashRangeEnd'])   
        
        self.removeMethod = pg.ComboBox()
        self.methods = {'linear interpolation': 1, 'subtraction': 2}
        self.removeMethod.setItems(self.methods)
        
        self.movingAverageWindow_slider = pg.SpinBox(int=True, step=1)
        self.movingAverageWindow_slider.setValue(s['windowSize'])     
        
        self.plotAverage_check = CheckBox()
        self.plotAverage_check.setValue(False)
        
        self.useROI_check = CheckBox()
        self.useROI_check.setValue(False)   
               
        
        self.items.append({'name': 'window', 'string': 'Select Window', 'object': self.window})
        self.items.append({'name': 'method', 'string': 'Select Method', 'object': self.removeMethod})  
        self.items.append({'name': 'addNoise', 'string': 'Add noise (linear interpolation only)', 'object': self.addNoise_check})          
        self.items.append({'name': 'blank', 'string': '----- Manual Flash -----', 'object': None})        
        self.items.append({'name': 'manualFlash', 'string': 'Manualy Set Flash', 'object': self.manuallySetFlash_check})          
        self.items.append({'name': 'flashStart', 'string': 'Select Flash Start', 'object': self.flashStart_slider})        
        self.items.append({'name': 'flashEnd', 'string': 'Select Flash End', 'object': self.flashEnd_slider})        
        self.items.append({'name': 'blank', 'string': '----- Automatic Flash Detection -----', 'object': None})  
        self.items.append({'name': 'flashRangeStart', 'string': 'Select Flash Range Start', 'object': self.flashRangeStart_slider})        
        self.items.append({'name': 'flashRangeEnd', 'string': 'Select Flash Range End', 'object': self.flashRangeEnd_slider})
        self.items.append({'name': 'windowSize', 'string': 'Select Moving Average Window Size', 'object': self.movingAverageWindow_slider})
        self.items.append({'name': 'useROI', 'string': 'User defined ROI for average', 'object': self.useROI_check})        
        self.items.append({'name': 'showAverage', 'string': 'Plot flash detection result', 'object': self.plotAverage_check})

        
        self.items.append({'name': 'removeFlash_button', 'string': '', 'object': self.removeFlash_button})

        super().gui()
flash_remover = Flash_remover()