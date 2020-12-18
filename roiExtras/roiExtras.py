from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage.interpolation import shift
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

import pyqtgraph as pg
from copy import deepcopy

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

from flika.process.file_ import open_file


class HistoWindow(BaseProcess):
    def __init__(self):
        super().__init__()
               
        self.win = pg.GraphicsWindow()
        self.win.resize(300, 300)
        self.win.setWindowTitle('ROI Extras Display')
        self.plt1 = self.win.addPlot()  
        
        self.label = pg.LabelItem(justify='right')
        self.win.addItem(self.label)
        
        self.autoscaleX = roiExtras.autoscaleX.isChecked()
        self.autoscaleY = roiExtras.autoscaleY.isChecked()
        

    def update(self, vals, start=-3, end=8, n=50, n_pixels=0):  
        ## compute standard histogram
        y,x = np.histogram(vals, bins=np.linspace(start, end, n))     
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True) 
        self.label.setText("<span style='font-size: 12pt'>pixels={}".format(n_pixels))
        
        if self.autoscaleX:
            self.plt1.setXRange(np.min(x),np.max(x),padding=0)
        if self.autoscaleY:
            self.plt1.setYRange(np.min(y),np.max(y),padding=0)
            
            
    
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()



class RoiExtras(BaseProcess_noPriorWindow):
    """
    Extends ROI menu features
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        
        self.currentWin = None
        self.currentROI = None
        self.displayStarted = False
        self.data = None
        self.img = None
        
        self.startScale = -3
        self.endScale = 50
        self.n = 50
        
        self.frame = 0
        self.ROIwindowExists = False
        

    def __call__(self):
        '''
        
        '''
        self.closeAction()     
        return

    def closeAction(self):
        self.currentWin = None
        self.currentROI = None
        self.displayStarted = False
        self.data = None
        self.img = None
        self.ROIwindowExists = False
        try:
            self.currentROI.sigRegionChanged.disconnect()
        except:
            pass
        try:
            self.currentWin.sigTimeChanged.disconnect()
        except:
            pass
        try:
            self.ROIwindow.close()
        except:
            pass
        
        try:
            self.histoWindow.close() 
        except:
            pass


    def closeEvent(self, event):
        self.closeAction()
        #BaseProcess_noPriorWindow.closeEvent(self, event)
        event.accept()
                                   

    def gui(self):
        self.gui_reset()
        self.active_window = WindowSelector()    
        
        self.startButton = QPushButton('Start')
        self.startButton.pressed.connect(self.start)  
        
        self.autoscaleX = CheckBox()
        self.autoscaleY = CheckBox()    
        
        self.autoscaleX.setChecked(True)
        self.autoscaleY.setChecked(True)
        
        self.autoscaleX.stateChanged.connect(self.updateX)
        self.autoscaleY.stateChanged.connect(self.updateY)     
        
        
        self.items.append({'name': 'active_window', 'string': 'Select Window', 'object': self.active_window})
        self.items.append({'name': 'autoScaleX', 'string': 'Autoscale Histogram X-axis', 'object': self.autoscaleX})    
        self.items.append({'name': 'autoScaleY', 'string': 'Autoscale Histogram Y-axis', 'object': self.autoscaleY})            
        self.items.append({'name': 'start_button', 'string': 'Click to select current ROI', 'object': self.startButton})  
        
        super().gui()

    def start(self):
        #select window
        self.currentWin = self.getValue('active_window')
        if self.currentWin == None:
            g.alert('First select window')
            return
        
        #disconnect previous ROI (if exists)
        try:
            self.currentROI.sigRegionChanged.disconnect()
        except:
            pass

        #disconnect previous time update (if exists)
        try:
            self.currentWin.sigTimeChanged.disconnect(self.update)
        except:
            pass        
        
        #select current ROI
        self.currentROI = self.currentWin.currentROI                
        if self.currentWin.currentROI == None:
            g.alert('First draw an ROI')      
            return
        
        #set updates
        self.currentROI.sigRegionChanged.connect(self.update)
        self.currentWin.sigTimeChanged.connect(self.update)
        
        #start plotting
        if self.displayStarted == False:
            self.startPlot()
            self.displayStarted = True
        
        #get stack data
        self.data =  np.array(deepcopy(self.currentWin.image))
        
        #get histo range from whole stack
        self.startScale = np.min(self.data)
        self.endScale = np.max(self.data)
                   
        self.update()
        
        #create window to plot ROI
        self.ROIwindow = Window(self.selected, name='ROI')
        self.ROIwindowExists = True
        
    def update(self): 
        #get frame index
        self.frame = self.currentWin.currentIndex  
        #get roi region
        self.selected = self.currentROI.getArrayRegion(self.data[self.frame], self.currentWin.imageview.getImageItem())
        #print(self.selected)
        
        #get roi mask (needed for freehand roi which returns square array)     
        mask = self.currentROI.getMask()
        #reset indices to cropped roi
        mask_norm = (mask[0]-min(mask[0]),mask[1]-min(mask[1]) )      
        #invert mask
        invertMask = np.ones_like(self.selected, dtype=bool)
        invertMask[mask_norm] = False
        #select region within roi boundary
        self.selected[invertMask] = 0 #using 0 for now, np.nan is slow
        
        #count number of pixels
        n_pixels = (self.selected>0).sum()
        
        #update plot
        self.histoWindow.update(self.selected, start=self.startScale, end=self.endScale, n=self.n, n_pixels=n_pixels)
        #update roi window
        if self.ROIwindowExists:
            self.ROIwindow.imageview.setImage(self.selected)

    def updateX(self):
        try:
            self.histoWindow.autoscaleX = self.autoscaleX.isChecked()
        except:
            pass

    def updateY(self):
        try:
            self.histoWindow.autoscaleY = self.autoscaleY.isChecked()
        except:
            pass

 
    def startPlot(self):
        self.histoWindow = HistoWindow()
        self.histoWindow.show()

roiExtras = RoiExtras()