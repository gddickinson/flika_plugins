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

class Frame_remover(BaseProcess_noPriorWindow):
    """
    Removes frames from movies. 
    
    """
    def __init__(self):

        if g.settings['frame_remover'] is None or 'length' not in g.settings['frame_remover']:
            s = dict()
            s['start'] = 0
            s['end'] = 1000
            s['length'] = 100
            s['interval'] = 100            
 
            g.settings['frame_remover'] = s
        super().__init__()


    def __call__(self, keepSourceWindow=False):
        g.settings['frame_remover']['start'] = self.getValue('start')
        g.settings['frame_remover']['end'] = self.getValue('end')
        g.settings['frame_remover']['length'] = self.getValue('length') 
        g.settings['frame_remover']['interval'] = self.getValue('interval')         
        return
        

                    
    def removeFrames(self):
        img = deepcopy(self.getValue('window').image)  

        start = np.arange(self.getValue('start'),self.getValue('end'),self.getValue('interval') )
        toDelete = []
        for i in start:
            toDelete.extend(np.arange(i,i+self.getValue('length')))

        img = np.delete(img,toDelete,0)                            
            
        #display stack in new window
        self.framesRemoved_win = Window(img,'Frames Removed')
        return
                    

    def gui(self):
        s=g.settings['frame_remover']
        self.gui_reset()
        self.window = WindowSelector()
        self.removeFrames_button = QPushButton('Remove Frames')
        self.removeFrames_button.pressed.connect(self.removeFrames)
        
        self.start_slider = pg.SpinBox(int=True, step=1)
        self.start_slider.setValue(s['start'])

        self.end_slider = pg.SpinBox(int=True, step=1)
        self.end_slider.setValue(s['end'])  
        
        self.frames_slider = pg.SpinBox(int=True, step=1)
        self.frames_slider.setValue(s['length'])

        self.interval_slider = pg.SpinBox(int=True, step=1)
        self.interval_slider.setValue(s['interval'])   
        
               
        
        self.items.append({'name': 'window', 'string': 'Select Window', 'object': self.window})

        self.items.append({'name': 'start', 'string': 'Start frame', 'object': self.start_slider})        
        self.items.append({'name': 'end', 'string': 'End frame', 'object': self.end_slider})        
        self.items.append({'name': 'length', 'string': 'Number of frames to remove', 'object': self.frames_slider})        
        self.items.append({'name': 'interval', 'string': 'Interval', 'object': self.interval_slider})
        
        self.items.append({'name': 'removeFrames_button', 'string': '', 'object': self.removeFrames_button})

        super().gui()

frame_remover = Frame_remover()