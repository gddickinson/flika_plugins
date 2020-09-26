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


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

#from flika.process.file_ import open_file
from .roi_GD import makeROI

class RoiScaler(BaseProcess_noPriorWindow):
    """
    Creates 2 linked ROI
        Center ROI creates trace
        Outer ROI used for scaling or background subtraction
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.center_ROI = None
        self.surround_ROI = None

    def __call__(self):
        '''
        
        '''
        pass
        return

    def closeEvent(self, event):
        self.unlink_frames(self.current_red, self.current_green)
        BaseProcess_noPriorWindow.closeEvent(self, event)


    def unlink_frames(self, *windows):
        for window in windows:
            if window != None:
                try:
                    window.sigTimeChanged.disconnect(self.indexChanged)
                except:
                    pass

    def displaySurround(self):
        print('clicked')                    

    def startROItrace(self):
        self.win = self.getValue('active_window')
        #get window shape
        height = self.win.my
        width = self.win.mx
        
        #create rois
        self.center_ROI = makeROI('rectangle',[[height/2, width/2], [10, 10]], color=QColor(255, 0, 0, 127), window=self.win)
        self.surround_ROI = makeROI('surround',[[height/2, width/2], [20, 20]], color=QColor(0, 0, 255, 127), window=self.win)
        
        #link rois
        self.center_ROI.addSurround(self.surround_ROI)
        
        #plot roi average trace
        self.plotROI = self.center_ROI.plot()        
        
        #get trace data
        self.traceCenter = self.center_ROI.getTrace()                    

    def gui(self):
        self.gui_reset()
        self.active_window = WindowSelector()

        self.displaySurroundButton = QPushButton('Display Surround ROI')
        self.displaySurroundButton.pressed.connect(self.displaySurround)
        
        self.startButton = QPushButton('Start')
        self.startButton.pressed.connect(self.startROItrace)        
        
        
        self.scaleImages = CheckBox()

        self.items.append({'name': 'active_window', 'string': 'Select Window', 'object': self.active_window})
        self.items.append({'name': 'scaleImages', 'string': 'Scale trace', 'object': self.scaleImages})
        self.items.append({'name': 'displaySurround_button', 'string': '          ', 'object': self.displaySurroundButton})
        self.items.append({'name': 'start_button', 'string': '          ', 'object': self.startButton})        
        

        super().gui()

roiScaler = RoiScaler()