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

class Overlay(BaseProcess_noPriorWindow):
    """
    Overlay two image stacks. 
    
    If Scale Images is ticked the dimmer image is rescaled to
    match the range of the brighter image.
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.current_red = None
        self.current_green = None

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

                    
    def overlay(self):
        red = self.getValue('red_window').image
        green = self.getValue('green_window').image
        checked = self.getValue('scaleImages')
        
        self.overlayed = np.zeros((red.shape[0], red.shape[1], red.shape[2], 3))
 
        if checked == True:
            if np.max(red) > np.max(green):
                green = green * int((np.max(red)/np.max(green)))
            else:
                red = red * int((np.max(green)/np.max(red)))
            
 
        self.overlayed[:,:,:,0] = red
        self.overlayed[:,:,:,1] = green
       
        
        #print(np.max(red),np.max(green))
        self.displayWindow_Overlay = Window(self.overlayed,'Overlay')
        return
                    

    def gui(self):
        self.gui_reset()
        self.red_window = WindowSelector()
        self.green_window = WindowSelector()
        self.overlayButton = QPushButton('Overlay')
        self.overlayButton.pressed.connect(self.overlay)
        self.scaleImages = CheckBox()

        self.items.append({'name': 'red_window', 'string': 'Select Red Window', 'object': self.red_window})
        self.items.append({'name': 'green_window', 'string': 'Select Green Window', 'object': self.green_window})
        self.items.append({'name': 'scaleImages', 'string': 'Scale Images', 'object': self.scaleImages})
        self.items.append({'name': 'start_button', 'string': '', 'object': self.overlayButton})

        super().gui()
overlay = Overlay()