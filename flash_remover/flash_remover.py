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

class Flash_remover(BaseProcess_noPriorWindow):
    """
    Remove flash artifact from movies. 
    
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self):
        '''
        
        '''
        pass
        return

                    
    def removeFlash(self):
        img = deepcopy(self.getValue('window').image)
        flashStart = self.getValue('flashStart')-1
        flashEnd = self.getValue('flashEnd')+2      
        
        flash = img[flashStart:flashEnd]
        n, r, c = flash.shape
        
        flashReplace = np.zeros_like(flash)
    
        for row in range(r):
            for col in range(c):
                points = range(0,n)
                xp = [0,n]
                fp = [flash[0,row,col],flash[-1,row,col]]
                
                interp_data = np.interp(points,xp,fp)

                flashReplace[0:n,row,col] = interp_data

        img[flashStart:flashEnd] = flashReplace
        
        self.flashRemoved_win = Window(img,'Flash Removed')
        return
                    

    def gui(self):
        self.gui_reset()
        self.window = WindowSelector()
        self.removeFlash_button = QPushButton('Remove Flash')
        self.removeFlash_button.pressed.connect(self.removeFlash)
        
        self.flashStart_slider = pg.SpinBox(int=True, step=1)
        self.flashStart_slider.setValue(313)

        self.flashEnd_slider = pg.SpinBox(int=True, step=1)
        self.flashEnd_slider.setValue(344)        
        
        self.items.append({'name': 'window', 'string': 'Select Window', 'object': self.window})
        self.items.append({'name': 'flashStart', 'string': 'Select Flash Start', 'object': self.flashStart_slider})        
        self.items.append({'name': 'flashEnd', 'string': 'Select Flash End', 'object': self.flashEnd_slider})          
        self.items.append({'name': 'removeFlash_button', 'string': '', 'object': self.removeFlash_button})

        super().gui()
flash_remover = Flash_remover()