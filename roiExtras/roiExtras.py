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

    def update(self, vals, start=-3, end=8, n=50):  
        ## compute standard histogram
        y,x = np.histogram(vals, bins=np.linspace(start, end, n))     
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True) 
    
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
        
        

    def __call__(self):
        '''
        
        '''
        self.currentWin = None
        self.currentROI = None
        self.displayStarted = False
        self.data = None
        self.img = None
        try:
            self.currentROI.sigRegionChanged.disconnect()
        except:
            pass
        try:
            self.currentWin.sigTimeChanged.disconnect()
        except:
            pass        
        self.histoWindow.close()       
        return

    def closeEvent(self, event):
        self.currentWin = None
        self.currentROI = None
        self.displayStarted = False
        self.data = None
        self.img = None
        try:
            self.currentROI.sigRegionChanged.disconnect()
        except:
            pass
        try:
            self.currentWin.sigTimeChanged.disconnect()
        except:
            pass        
        self.histoWindow.close()        
        BaseProcess_noPriorWindow.closeEvent(self, event)
                                   

    def gui(self):
        self.gui_reset()
        self.active_window = WindowSelector()    
        
        self.startButton = QPushButton('Start')
        self.startButton.pressed.connect(self.start)    
        
        self.items.append({'name': 'active_window', 'string': 'Select Window', 'object': self.active_window})
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
        
    def update(self): 
        #get frame index
        self.frame = self.currentWin.currentIndex  
        #get roi region
        self.selected = self.currentROI.getArrayRegion(self.data[self.frame], self.currentWin.imageview.getImageItem())
        #print(self.selected)
        #update plot
        self.histoWindow.update(self.selected, self.startScale, self.endScale, self.n)

 
    def startPlot(self):
        self.histoWindow = HistoWindow()
        self.histoWindow.show()

roiExtras = RoiExtras()