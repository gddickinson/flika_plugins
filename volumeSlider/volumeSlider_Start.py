import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
import os
from os import listdir
from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
from copy import deepcopy
from numpy import moveaxis
from skimage.transform import rescale
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from qtpy.QtCore import Signal

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .volumeSlider_Main import *
from .helperFunctions import *
from .tiffLoader import openTiff
from .lightSheet_tiffLoader import Load_tiff
load_tiff = Load_tiff()
#########################################################################################
#############          FLIKA Base Menu             #####################################
#########################################################################################
class VolumeSliderBase(BaseProcess_noPriorWindow):
    """
    Start Volume Slider from differnt sources

        |Select source (current window or saved numpy array)
        
    Returns volumeSlider GUI

    """
    
    def __init__(self):
        if g.settings['volumeSlider'] is None or 'preProcess' not in g.settings['volumeSlider']:
            s = dict() 
            s['inputChoice'] = 'Current Window'              
            s['keepOriginalWindow'] = False   
            s['slicesPerVolume'] =    1
            s['slicesDeletedPerVolume'] =    0            
            s['baselineValue'] = 0
            s['f0Start'] = 0
            s['f0End'] = 0
            s['multiplicationFactor'] = 100
            s['currentDataType'] = 0
            s['newDataType'] = 0
            s['theta'] = 45
            s['shiftFactor'] = 1
            s['trimLastFrame'] = False
            s['inputArrayOrder'] = 4
            s['displayArrayOrder'] = 16
            s['f0VolStart'] = 0
            s['f0VolEnd'] = 0  
            s['IMS_fname'] ='IMS_export.ims'
            s['IMS_subsamp'] = '((1, 1, 1), (1, 2, 2))'
            s['IMS_chunks'] = '((16, 128, 128), (64, 64, 64))'
            s['IMS_compression'] = 'gzip'
            s['IMS_thumbsize'] = '256'
            s['IMS_dx'] = 0.1
            s['IMS_dz'] = 0.25
            s['IMS_Unit'] = 'um'
            s['IMS_GammaCorrection'] = 1
            s['IMS_ColorRange'] =  '0 255'
            s['IMS_LSMEmissionWavelength'] = 0
            s['IMS_LSMExcitationWavelength'] = 0
            s['preProcess'] = False
                            
            g.settings['volumeSlider'] = s
                
        BaseProcess_noPriorWindow.__init__(self)
        
    def __call__(self, inputChoice,keepOriginalWindow,preProcess, slicesPerVolume, slicesDeletedPerVolume, keepSourceWindow=False):
        g.settings['volumeSlider']['inputChoice'] = inputChoice
        g.settings['volumeSlider']['keepOriginalWindow'] = keepOriginalWindow
        g.settings['volumeSlider']['preProcess'] = preProcess        
        g.settings['volumeSlider']['slicesPerVolume'] = slicesPerVolume   
        g.settings['volumeSlider']['slicesDeletedPerVolume'] = slicesDeletedPerVolume        

        g.m.statusBar().showMessage("Starting Volume Slider...")
        
        if inputChoice == 'Current Window':
            camVolumeSlider.startVolumeSlider(keepWindow=keepOriginalWindow, preProcess=preProcess, framesPerVol = slicesPerVolume, framesToDelete = slicesDeletedPerVolume)
            
        elif inputChoice == 'Numpy Array':
            A_path = open_file_gui(directory=os.path.expanduser("~/Desktop"),filetypes='*.npy')
            g.m.statusBar().showMessage("Importing Array: " + A_path)
            A = np.load(str(A_path))
            camVolumeSlider.startVolumeSlider(A=A,keepWindow=keepOriginalWindow)

        elif inputChoice == 'Batch Process':
            g.m.statusBar().showMessage("Starting Batch Processing...")            
            camVolumeSlider.startVolumeSlider(batch=True)
            
        elif inputChoice == 'Load file':
            g.m.statusBar().showMessage("Loading file...")   
            #Open file using tiff loader
            load_tiff.gui()
            #start volumeSlider
            camVolumeSlider.startVolumeSlider(keepWindow=keepOriginalWindow, preProcess=preProcess, framesPerVol = slicesPerVolume, framesToDelete = slicesDeletedPerVolume) 
                        
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)

    def gui(self):
        self.gui_reset()
                       
        #combobox
        inputChoice = ComboBox()
        inputChoice.addItem('Current Window')
        inputChoice.addItem('Load file')        
        inputChoice.addItem('Numpy Array')
        inputChoice.addItem('Batch Process')        
        
        #checkbox
        self.keepOriginalWindow = CheckBox()
        self.keepOriginalWindow.setValue(False)   

        self.preProcess = CheckBox()
        self.preProcess.setValue(g.settings['volumeSlider']['preProcess'])        


        self.framesPerVolume = pg.SpinBox(int=True, step=1)
        self.framesPerVolume.setValue(g.settings['volumeSlider']['slicesPerVolume'])        
        
        self.framesRemoved = pg.SpinBox(int=True, step=1)
        self.framesRemoved.setValue(g.settings['volumeSlider']['slicesDeletedPerVolume'])        
        
        
        #populate GUI
        self.items.append({'name': 'inputChoice', 'string': 'Choose Input Data:', 'object': inputChoice}) 
        self.items.append({'name': 'keepOriginalWindow','string':'Keep Original Window','object': self.keepOriginalWindow}) 
        self.items.append({'name': 'preProcess','string':'Preprocess Image stack','object': self.preProcess})  

        self.items.append({'name': 'slicesPerVolume','string':'Slices per Volume','object': self.framesPerVolume})  
        self.items.append({'name': 'slicesDeletedPerVolume','string':'Frames to remove per volume','object': self.framesRemoved})                                     
        
        super().gui()
        
        
volumeSliderBase = VolumeSliderBase()




  
