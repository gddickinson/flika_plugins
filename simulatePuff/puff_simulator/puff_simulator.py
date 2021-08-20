# -*- coding: utf-8 -*-
"""
Created on Fri Oct 02 12:23:06 2015

@author: George Dickinson adpated from Kyle Ellefsen
"""
import numpy as np
import os
from qtpy.QtCore import Signal
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
import inspect
import numpy.random as random
import shutil
from distutils.version import StrictVersion
from copy import deepcopy
import pyqtgraph as pg
from matplotlib import pyplot as plt 
import csv 
import pandas as pd

import flika
try:
    flika_version = flika.__version__
except AttributeError:
    flika_version = '0.0.0'
if StrictVersion(flika_version) < StrictVersion('0.1.0'):
    import global_vars as g
    from window import Window
    from process.file_ import close
    import tifffile
    from process.filters import gaussian_blur
    from process.binary import threshold
    from process.roi import set_value
    from roi import makeROI
    from process.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector
    #from plugins.detect_puffs.threshold_cluster import threshold_cluster

else:
    from flika import global_vars as g
    from flika.window import Window
    from flika.process.file_ import close
    from flika.process.filters import gaussian_blur
    from flika.process.binary import threshold
    from flika.process.roi import set_value
    from flika.roi import makeROI
    from flika.utils.io import tifffile
    #from ..threshold_cluster import threshold_cluster
    if StrictVersion(flika_version) < StrictVersion('0.2.23'):
        from flika.process.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector, CheckBox
    else:
        from flika.utils.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector, CheckBox


cwd = os.path.dirname(os.path.abspath(__file__)) # cwd=r'C:\Users\Kyle Ellefsen\Documents\GitHub\Flika\plugins\puff_simulator'


class WindowSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to click a Window object.  Once you do, it sets self.window to be the window, and it sets the label to be the widget name.
    """
    valueChanged=Signal()
    def __init__(self):
        QWidget.__init__(self)
        self.button = QPushButton('Select Window')
        self.button.setCheckable(True)
        self.label = QLabel('None')
        self.window = None
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)

    def buttonclicked(self):
        if self.button.isChecked() is False:
            g.m.setCurrentWindowSignal.sig.disconnect(self.setWindow)
        else:
            g.m.setCurrentWindowSignal.sig.connect(self.setWindow)

    def setWindow(self, window=None):
        if window is None:
            try:
                g.m.setCurrentWindowSignal.sig.disconnect(self.setWindow)
            except TypeError:
                pass
            self.window = g.win
        else:
            self.window = window
        self.button.setChecked(False)
        self.label.setText('...'+os.path.split(self.window.name)[-1][-20:])
        self.valueChanged.emit()
        self.parent().raise_()
        
        #update sliders
        simulate_puff.update()

    
    def value(self):
        return self.window
    
    def setValue(self, window):
        ''' This function is written to satify the requirement that all items have a setValue function to recall from settings the last set value. '''
        self.setWindow(window)

       
class Simulate_Puff(BaseProcess_noPriorWindow):
    """
    Add a simulated puff to an image stack
    
    """
    def __init__(self):
        super().__init__()
        self.currentWin = None
        self.currentROI = None
        self.data = None

    def get_init_settings_dict(self):
        s = dict()
        s['nFrames'] = 100
        s['puffAmplitude'] = 5
        s['x'] = 128
        s['y'] = 128
        s['startFrame'] = 100
        s['sigma'] = 20
        s['useROI'] = False
        s['meanExp'] = 5.0
        s['nPuffs'] = 0    
        s['nSites'] = 10
        

        return s

    def gui(self):
        self.gui_reset()
        self.nFrames = SliderLabel(0)
        self.nFrames.setRange(0,10000)
        self.startFrame = SliderLabel(0)
        self.startFrame.setRange(0,10000)        
        
        self.puffAmplitude = pg.SpinBox(int=False, step=.01)
        self.puffAmplitude.setValue(1.0) 
        self.sigma = SliderLabel(0)
        self.sigma.setRange(1,1000)
        self.x = SliderLabel(0)
        self.x.setRange(1, 10000)
        self.y = SliderLabel(0)
        self.y.setRange(1, 10000)
        self.useROI = CheckBox()
        self.useFrame = CheckBox()
        self.active_window = WindowSelector()
        self.previewButton = QPushButton('Preview Puff')
        self.previewButton.pressed.connect(self.previewPuff) 
        self.puffButton = QPushButton('Add Puff')
        self.puffButton.pressed.connect(self.addPuff)  
        
        self.randomPuffsAdded = False
        
        self.nSites = 10
        columnNames = ['time of puff', 'site']
        self.timesAdded = pd.DataFrame(columns=columnNames)
        self.siteNumber = int(0)

        self.nPuffs_slider = SliderLabel(0)
        self.nPuffs_slider.setRange(1, 1000)      
        
        self.meanExp_slider = pg.SpinBox(int=False, step=.01)
        self.meanExp_slider.setValue(5.0) 
        
        self.plotHistoTimes = CheckBox()
        self.plotHistoTimes.setValue(False)
        
        self.addPuffsSequentially = CheckBox()
        self.addPuffsSequentially.setValue(False)
        
        self.exportTimes_button = QPushButton('Export times')
        self.exportTimes_button.pressed.connect(self.exportTimes)  
        
        self.randomPuffButton = QPushButton('Add Puffs to site')
        self.randomPuffButton.pressed.connect(self.addRandomPuffs)     
        
        self.multipleRandomPuffButton = QPushButton('Add Puffs inside ROI')
        self.multipleRandomPuffButton.pressed.connect(self.addMultipleRandomPuffs)          
        
        self.nSites_box = QSpinBox()
        self.nSites_box.setRange(0,10000)
        self.nSites_box.setValue(self.nSites)

        
        self.items.append({'name': 'active_window', 'string': 'Select Window', 'object': self.active_window})             
        self.items.append({'name':'nFrames','string':'Duration (frames)','object':self.nFrames})
        self.items.append({'name':'startFrame','string':'Start Frame','object':self.startFrame}) 
        #self.items.append({'name': 'useCurrentFrame', 'string': 'Use Current Frame For Start', 'object': self.useFrame})          
        self.items.append({'name':'puffAmplitude','string':'Amplitude','object':self.puffAmplitude})
        self.items.append({'name': 'x', 'string': 'x', 'object': self.x})
        self.items.append({'name': 'y', 'string': 'y', 'object': self.y})
        #self.items.append({'name': 'useROI', 'string': 'Use ROI for position', 'object': self.useROI})                
        self.items.append({'name': 'sigma', 'string': 'sigma', 'object': self.sigma})  
        self.items.append({'name': 'preview_Button', 'string': 'Click to preview Puff', 'object': self.previewButton})         
        self.items.append({'name': 'puff_Button', 'string': 'Click to add Puff', 'object': self.puffButton}) 
        self.items.append({'name': 'blank', 'string': '---------- RANDOM PUFFS ---------------------------', 'object': None}) 
        #self.items.append({'name': 'nPuffs', 'string': 'Number of puffs to add', 'object': self.nPuffs_slider})  
        self.items.append({'name': 'meanExp', 'string': 'Mean of exponential distibution', 'object': self.meanExp_slider}) 
        self.items.append({'name': 'puffsSequential', 'string': 'Wait until puff ends before adding next puff:', 'object': self.addPuffsSequentially})        
        self.items.append({'name': 'histoTimes', 'string': 'Plot histogram of puff start times:', 'object': self.plotHistoTimes})
        self.items.append({'name': 'random_puff_Button', 'string': 'Click to add randomly distibuted puffs at one site', 'object': self.randomPuffButton}) 
        self.items.append({'name': 'multipleRandom_puff_Button', 'string': 'Click to add randomly distibuted puffs at multiple sites', 'object': self.multipleRandomPuffButton})
        self.items.append({'name': 'nSites', 'string': 'Number of Sites to add', 'object': self.nSites_box})
        self.items.append({'name': 'blank', 'string': '---------- Export Puffs ---------------------------', 'object': None})                 
        self.items.append({'name': 'listTimes', 'string': 'Export list of puff start times', 'object': self.exportTimes_button})          

        super().gui()

    def __call__(self):
        pass
        return 

    def update(self):
        self.currentWin = self.getValue('active_window')
        #get image data
        self.data =  np.array(deepcopy(self.currentWin.image))                
        self.dt,self.dy,self.dx = self.data.shape        
        self.x.setRange(1,self.dx-1)
        self.y.setRange(1,self.dy-1) 
        
        self.x.setValue(int((self.dx-1)/2))
        self.y.setValue(int((self.dy-1)/2))        
        
        self.nFrames.setRange(0,self.dt)
        self.startFrame.setRange(0,self.dt-1)        
        
        
        self.sigma.setRange(1,int(self.dx/7))

    def addPuff(self, time=False, singleSite = True, x=None, y=None):
        '''add synthetic blip to image stack'''
        #select window
        self.currentWin = self.getValue('active_window')
        if self.currentWin == None:
            g.alert('First select window')
            return
     
        #generate blip
        sigma = self.getValue('sigma')
        amp = self.getValue('puffAmplitude')
        duration = self.getValue('nFrames')
        
        blip = generateBlip(sigma=sigma,amplitude=amp,duration=duration)
        
        blip_time, blip_x_size, blip_y_size = blip.shape
        
        x_size = int(blip_x_size/2)
        y_size = int(blip_y_size/2)        
        
        
        #add blip to stack
        if time == False:
            t = self.getValue('startFrame')
        else:
            t = time
            
        if singleSite:
            x = self.getValue('x')
            y = self.getValue('y')
            
        
        tt = np.arange(t,t+duration,dtype=np.int)
        xx = np.arange(x-x_size-1, x+x_size, dtype=np.int)
        yy = np.arange(y-y_size-1, y+y_size, dtype=np.int)
        
        
        if time==False:
            try:
                self.data[np.ix_(tt,xx,yy)] = self.data[np.ix_(tt,xx,yy)] + blip
            except:
                g.alert('Puff too large, too long or too close to edge')
        else:
            self.data[np.ix_(tt,xx,yy)] = self.data[np.ix_(tt,xx,yy)] + blip
        
        
        frame = self.currentWin.currentIndex  
        self.currentWin.imageview.setImage(self.data)
        self.currentWin.image = self.data
        self.currentWin.setIndex(frame)
        
        return


    def addRandomPuffs(self, singleSite=True, x=None, y=None):
        if self.getValue('active_window') == None:
            g.alert('First select window')
            return                
        
        mean = self.getValue('meanExp')
        #nPuffs = self.getValue('nPuffs')
        
        puffsAdded = 0
        puffsOutsideOfRange = 0
        
        # add first puff
        try:
            time = int(np.random.exponential(scale=mean, size=1) + self.getValue('startFrame'))
            self.addPuff(time = time, singleSite=singleSite, x=x, y=y)
        except:
            puffsOutsideOfRange += 1 
            print('{} puffs added, {} puffs out of range'.format(puffsAdded,puffsOutsideOfRange))
            print('1st puff outside of range, aborting')
            return
       
        # add puff after each time selection untill end of stack
        # casting the exponential continous value as an int to select frame        
        while time < self.dt-self.getValue('nFrames'):
            try:
                time = int(time + np.random.exponential(scale=mean, size=1))
                
                #if sequental, add puff duration to time
                if self.getValue('puffsSequential'):
                    time = time + self.getValue('nFrames')                    
                    
                #add puff at time    
                self.addPuff(time = time, singleSite=singleSite, x=x, y=y)                    

                #update puff time log                
                self.timesAdded = self.timesAdded.append({'time of puff': time, 'site': int(self.siteNumber)},ignore_index=True)                     
                    
                puffsAdded +=1 
            except:
                puffsOutsideOfRange += 1    
        
        print('{} puffs added, {} puffs out of range'.format(puffsAdded,puffsOutsideOfRange))
        
        if self.plotHistoTimes.isChecked():
            plt.hist(self.timesAdded)
            plt.xlabel('Time puff added (frames)')
            plt.ylabel('Number of puffs added')
            plt.show()

        self.randomPuffsAdded = True
        if singleSite:
            self.siteNumber += 1

        return

    def addMultipleRandomPuffs(self):
        nSites = self.getValue('nSites')
        
        if self.currentWin == None:
            g.alert('First select window')
            return

        #select current ROI
        self.currentROI = self.currentWin.currentROI   

        if self.currentWin.currentROI == None:
            g.alert('First draw an ROI')      
            return
        bounds = self.currentROI.parentBounds()
        
        topLeft_x = bounds.topLeft().x()
        topLeft_y = bounds.topLeft().y()        
        
        bottomRight_x = bounds.bottomRight().x()
        bottomRight_y = bounds.bottomRight().y()        
        
        print('adding puffs to {} sites from {},{} to {},{}'.format(nSites, topLeft_x, topLeft_y, bottomRight_x, bottomRight_y))
        
        sites = self.getRandomSites(topLeft_x, bottomRight_x, topLeft_y, bottomRight_y, nSites)
        
        
        for site in sites:
            print("Puff site: ", self.siteNumber)
            self.addRandomPuffs(singleSite=False, x=site[0], y=site[1])
            self.siteNumber += 1
            
        print('Finished adding sites')
        
        return


    def getRandomSites(self, xStart, xEnd, yStart, yEnd, qty, radius=0):
        import random

        rangeX = (xStart, xEnd)
        rangeY = (yStart, yEnd)
        
        deltas = set()
        for x in range(-radius, radius+1):
            for y in range(-radius, radius+1):
                if x*x + y*y <= radius*radius:
                    deltas.add((x,y))
        
        randPoints = []
        excluded = set()
        i = 0

        while i < qty:
            x = random.randrange(*rangeX)
            y = random.randrange(*rangeY)
            if (x,y) in excluded: continue
            randPoints.append((x,y))
            i += 1
            excluded.update((x+dx, y+dy) for (dx,dy) in deltas)
        
        return randPoints

    def exportTimes(self):
        if self.getValue('active_window') == None:
            g.alert('First select window')
            return  
        
        if self.randomPuffsAdded == False:
            g.alert('Add puffs first')
            return
        
        #set export path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")        

        #write file
        self.timesAdded.to_csv(savePath)
        print('List of times saved to: {}'.format(savePath))
        
        # #write file
        # try:
        #     # opening the csv file in 'w+' mode 
        #     file = open(savePath, 'w+', newline ='') 
              
        #     # writing the data into the file 
        #     with file:     
        #         write = csv.writer(file) 
        #         write.writerows(map(lambda x: [x], self.timesAdded)) 
            
        #     print('List of times saved to: {}'.format(savePath))
        # except BaseException as e:
        #     print(e)
        #     print('Export of times failed, printing times to console')
        #     print(self.timesAdded)

    def previewPuff(self):
        '''preview blip to be added'''
        sigma = self.getValue('sigma')
        amp = self.getValue('puffAmplitude')
        duration = self.getValue('nFrames')
        
        blip = generateBlip(sigma=sigma,amplitude=amp,duration=duration)
        Window(blip)
        
        return

    def generateBlip(sigma=1, amplitude=1, duration=1):
        sigma = int(sigma)
        width = sigma*8+1
        xorigin = sigma*4
        yorigin = sigma*4
        x = np.arange(width)
        y = np.arange(width)
        x = x[:, None]
        y = y[None, :]
        gaussian = amplitude*(np.exp(-(x-xorigin)**2/(2.*sigma**2))*np.exp(-(y-yorigin)**2/(2.*sigma**2)))
        blip = np.repeat(gaussian[None,:,:],repeats=duration, axis=0)
        return blip        



simulate_puff = Simulate_Puff()


    

class Simulate_Blips(BaseProcess_noPriorWindow):
    def __init__(self):
        super().__init__()
    def gui(self):
        self.gui_reset()
        nFrames = SliderLabel(0)
        nFrames.setRange(0,10000)
        self.items.append({'name':'nFrames','string':'Movie Duration (frames)','object':nFrames})
        super().gui()
    def __call__(self, nFrames=10000):
        print('called')
        self.start()
        self.newtif = generateBlipImage()
        self.newname = ' Simulated Blips '
        return self.end()

simulate_blips = Simulate_Blips()

def generateBlip(sigma=1, amplitude=1, duration=1):
    sigma = int(sigma)
    width = sigma*8+1
    xorigin = sigma*4
    yorigin = sigma*4
    x = np.arange(width)
    y = np.arange(width)
    x = x[:, None]
    y = y[None, :]
    gaussian = amplitude*(np.exp(-(x-xorigin)**2/(2.*sigma**2))*np.exp(-(y-yorigin)**2/(2.*sigma**2)))
    blip = np.repeat(gaussian[None,:,:],repeats=duration, axis=0)
    return blip
    

def generateBlipImage(amplitude=1):
    A = random.randn(1100,128,128) # tif.asarray() # A[t,y,x]
    
    #puffArray=[ x,    y,   ti, sigma, amplitude, duration] to create puffs
    puffArray = [[ 10,    115,  50, 2, .2, 20],
               [ 40,   22,   100, 2, .2, 20],  
               [ 110,  10,   150, 2, .2, 20], 
               [ 118,  76,   200, 2, .2, 20], 
               [ 50,   50,   250, 2, .2, 20], 
               [ 113,  10,   300, 2, .2, 20],
               [ 11,   117,  350, 2, .2, 20],
               [ 15,   22,   400, 2, .2, 20],
               [ 65,   64,   450, 2, .2, 20],
               [ 114,  110,  500, 2, .2, 20],
               [ 10,   115,  550, 2, .2, 20], 
               [ 40,   22,   600, 2, .2, 20],  
               [ 110,  10,   650, 2, .2, 20], 
               [ 118,  76,   700, 2, .2, 20], 
               [ 50,   50,   750, 2, .2, 20], 
               [ 113,  10,   800, 2, .2, 20],
               [ 11,   117,  850, 2, .2, 20],
               [ 15,   22,   900, 2, .2, 20],
               [ 65,   64,   950, 2, .2, 20],
               [ 114,  110, 1000, 2, .2, 20]]
               
    for p in puffArray:
        x, y, ti, sigma, amp, duration=p
        amp *= 2
        blip = 3*generateBlip(sigma,amp,duration)
        dt, dx, dy = blip.shape
        dx = (dx-1)/2
        dy = (dy-1)/2
        t = np.arange(ti,ti+duration,dtype=np.int)
        y = np.arange(y-dy,y+dy+1,dtype=np.int)
        x = np.arange(x-dx,x+dx+1,dtype=np.int)
        A[np.ix_(t,y,x)] = A[np.ix_(t,y,x)]+amplitude*blip
    
    return Window(A)

