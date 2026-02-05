# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

@author: george.dickinson@gmail.com

This program is a Python script developed to faciliate segmentation of tiff recordings

"""

# ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=Warning)

# import necessary modules
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from os.path import expanduser
import os, shutil, subprocess
import math
import sys



# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

try:
    from .imageSegmenter import *
except:
    print('imageSegmenter path not valid')

'''
#####################################################################################################################################
######################################          Main PLUG-IN CLASS         ##########################################################
#####################################################################################################################################
'''

class ImageSegmentation(BaseProcess_noPriorWindow):
    """
    Segment Tiff Stacks
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['imageSegmentation'] is None or 'framelength' not in g.settings['imageSegmentation']:
            s = dict()
            s['classifier'] = 'Classical ML'
            g.settings['imageSegmentation'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, classifier,  keepSourceWindow=False):
        '''
        save params
        start the RF GUI
        '''

        g.settings['imageSegmentation']['classifier'] = classifier

        return


    def closeEvent(self, event):
        '''
        close plugin GUI
        '''

        # Call the closeEvent function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return


    def startGUI(self):
        #app = QApplication.instance()
        self.plugin = ImageSegmenter()
        plugin_gui = self.plugin.gui()
        plugin_gui.show()
        print("Plugin GUI displayed")
        #app.exec_()


    def gui(self):
        # Initialize class variables
        #self.classifier_choice = ComboBox()
        #self.classifier_choice.addItems(['Classical ML', 'Neural Nets'])

        self.start_button = QPushButton('Start')
        self.start_button.pressed.connect(self.startGUI)

        # Call gui_reset function
        self.gui_reset()

        # Get settings for locsAndTracksPlotter
        s=g.settings['imageSegmentation']


        #################################################################
        # Define the items that will appear in the GUI, and associate them with the appropriate functions.
        #self.items.append({'name':'classifier','string':'classifier','object':self.classifier_choice})
        self.items.append({'name':'startButton','string':'','object':self.start_button})
        super().gui()
        ######################################################################

        return


# Instantiate the plugin class
imageSegmentation = ImageSegmentation()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











