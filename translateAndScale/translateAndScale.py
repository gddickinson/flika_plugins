# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

@author: george.dickinson@gmail.com

This program is a Python script developed to analyze the motion of intracellular Piezo1 proteins labeled with a fluorescent tag.
It allows the user to load raw data from a series of image files and track the movement of individual particles over time.
The script includes several data analysis and visualization tools, including the ability to filter data by various parameters, plot tracks, generate scatter and line plots, and create statistics for track speed and displacement.
Additional features include the ability to toggle between different color maps, plot diffusion maps, and save filtered data to a CSV file.

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

#from pyqtgraph.canvas import Canvas, CanvasItem

from pyqtgraph import Point

from tqdm import tqdm

import numba
pg.setConfigOption('useNumba', True)

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea


class TemplateROI(pg.RectROI):
    def __init__(self,template = 'disk', *args, **kwds):
        pg.RectROI.__init__(self, aspectLocked=True, *args, **kwds)
        self.addRotateHandle([1,0], [0.5, 0.5])
        self.template = template

    def paint(self, p, opt, widget):
        if self.template == 'disk':
            r = self.boundingRect()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.scale(r.width(), r.height())## workaround for GL bug
            r = QRectF(r.x()/r.width(), r.y()/r.height(), 1,1)
            p.drawEllipse(r)

        elif self.template == 'square':
            # Note: don't use self.boundingRect here, because subclasses may need to redefine it.
            r = QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.translate(r.left(), r.top())
            p.scale(r.width(), r.height())
            p.drawRect(0, 0, 1, 1)


        elif self.template == 'crossbow':
            radius = self.getState()['size'][1]
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            #p.drawLine(Point(radius/2, -radius/2), Point(radius/2, radius/2))
            p.drawLine(Point(0, radius/2), Point(radius, radius/2))

            pen2 = QPen()
            pen2.setWidth(20)
            pen2.setColor(Qt.green)
            p.setPen(pen2)
            p.drawPoint(int(radius-10),int(radius/2))

            r = QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.translate(r.left(), r.top())
            p.scale(r.width(), r.height())
            p.drawRect(0, 0, 1, 1)


disk_template = TemplateROI(pos=[80, 50], size=[20, 20], template = 'disk', pen=(0,9))
square_template = TemplateROI(pos=[80, 50], size=[20, 20], template = 'square', pen=(0,9))
crossbow_template = TemplateROI(pos=[80, 50], size=[20, 20], template = 'crossbow', pen=(0,9))

'''
#####################################################################################################################################
######################################   Main LOCSANDTRACKSPLOTTER CLASS   ##########################################################
#####################################################################################################################################
'''

class TranslateAndScale(BaseProcess_noPriorWindow):
    """
    Generate alignment for rotation, translation and scaling of point data
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['translateAndScale'] is None or 'pixelSize' not in g.settings['translateAndScale']:
            s = dict()
            s['pixelSize'] = 108
            g.settings['translateAndScale'] = s

        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)


    def __call__(self, pixelSize,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.

        Parameters:
        pixelSize: int - pixel size of image data

        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['videoExporter']['pixelSize'] = pixelSize


        return


    def closeEvent(self, event):
        '''
        This function is called when the user closes the locs and tracks plotter window. It clears any plots that have been
        generated and calls the closeEvent function for the BaseProcess_noPriorWindow class.

        Parameters:
        event: object - object representing the close event

        Returns: None
        '''

        # Call the closeEvent function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return


    def gui(self):
        self.gui_reset()
        self.dataWindow = WindowSelector()

        self.currentTemplate = None

        self.startButton = QPushButton('Start Alignment')
        self.startButton.pressed.connect(self.startAlign)

        self.endButton = QPushButton('Finished Alignment')
        self.endButton.pressed.connect(self.endAlign)


        self.templateBox = pg.ComboBox()
        self.templates = {'Disk': disk_template,
                          'Square': square_template,
                          'Crossbow': crossbow_template}
        self.templateBox.setItems(self.templates)

        self.items.append({'name': 'dataWindow', 'string': 'Data Window', 'object': self.dataWindow})
        self.items.append({'name': 'template', 'string': 'Choose template', 'object': self.templateBox})
        self.items.append({'name': 'startButton', 'string': '', 'object': self.startButton})
        self.items.append({'name': 'endButton', 'string': '', 'object': self.endButton})

        super().gui()

        return

    def startAlign(self):
        template = self.getValue('template')
        self.getValue('dataWindow').imageview.addItem(template)
        self.currentTemplate = template
        return

    def endAlign(self):
        self.getValue('dataWindow').imageview.removeItem(self.currentTemplate)
        self.currentTemplate = None
        return


# Instantiate the LocsAndTracksPlotter class
translateAndScale = TranslateAndScale()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











