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

        ## make interesting distribution of values
        vals = np.array([0,0,0,0])
        
        ## compute standard histogram
        y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))
        
        ## Using stepMode=True causes the plot to draw two lines for each sample.
        ## notice that len(x) == len(y)+1
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)        

    def update(self, vals):        
        ## compute standard histogram
        y,x = np.histogram(vals, bins=np.linspace(-3, 8, 40))        
        self.plt1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True) 
    
    def show(self):
        self.win.show()

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
        


    def __call__(self):
        '''
        
        '''
        pass
        return

    def closeEvent(self, event):
        try:
            self.currentROI.sigRegionChanged.disconnect()
        except:
            pass
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
        
        #select current ROI
        self.currentROI = self.currentWin.currentROI                
        if self.currentWin.currentROI == None:
            g.alert('First draw an ROI')      
            return
        
        #set updates
        self.currentROI.sigRegionChanged.connect(self.update)
        
        #start plotting
        if self.displayStarted == False:
            self.startPlot()
            self.displayStarted = True
            
        self.update()
        
    def update(self):
        self.data =  np.array(deepcopy(self.currentWin.image))
        self.selected = self.currentROI.getArrayRegion(self.data, self.currentWin.imageview.getImageItem())
        #print(self.selected)
        self.histoWindow.update(self.selected)
        
    def startPlot(self):
        self.histoWindow = HistoWindow()
        self.histoWindow.show()

roiExtras = RoiExtras()