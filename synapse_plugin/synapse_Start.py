import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
#import time
#import os
#from os import listdir
#from os.path import expanduser, isfile, join
from distutils.version import StrictVersion

#import pyqtgraph.opengl as gl
#from OpenGL.GL import *
from qtpy.QtCore import Signal

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

#from .BioDocks import *
#import BioDocks
#from Channels import *
#from ClusterMath import *
try:
    from .Synapse3D import *
except:
    from Synapse3D import *
try:
    from .Synapse import *
except:
    from Synapse import *

#########################################################################################
#############          FLIKA Base Menu             #####################################
#########################################################################################
class SynapseStart(BaseProcess_noPriorWindow):
    """
    Start Synapse Plugin

    """
    
    def __init__(self):
        if g.settings['synapse'] is None:
            s = dict() 
            s['version'] = 'synapse3D'                        
                            
            g.settings['synapse'] = s
                
        BaseProcess_noPriorWindow.__init__(self)
        
    def __call__(self, version,keepSourceWindow=False):
        g.settings['synapse']['version'] = version

        g.m.statusBar().showMessage("Starting Synapse...")
        
        if version == 'synapse3D':
            #start synapse3D GUI
            self.synapse3D_app = Synapse3D()
            self.synapse3D_app.start()
        elif version == 'synapse':
            #start synapse GUI
            self.synapse_app = Synapse()
            self.synapse_app.start()            
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)

    def gui(self):
        self.gui_reset()
                       
        #combobox
        versionChoice = ComboBox()
        versionChoice.addItem('synapse3D')
        versionChoice.addItem('synapse')
                         
        #populate GUI
        self.items.append({'name': 'version', 'string': 'Choose Version:', 'object': versionChoice})                                    
        super().gui()
        
        
synapseStart = SynapseStart()