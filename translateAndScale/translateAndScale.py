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

from scipy.ndimage import center_of_mass, gaussian_filter, binary_fill_holes, binary_closing, label, find_objects
from skimage.filters import threshold_otsu

from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

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



class DisplayParams():
    '''
    Display translation parameters
    '''
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        #self.win.resize(250, 200)
        self.win.setWindowTitle('Parameters')

        ## Create docks
        self.d0 = Dock("ROI parameters")

        self.area.addDock(self.d0)


        #point options widget
        self.w0 = pg.LayoutWidget()

        self.centerPos_text = QLabel('--,--')
        self.centerPos_label = QLabel('Center:')

        self.angle_text = QLabel('--')
        self.angle_label = QLabel('Angle:')

        self.size_text = QLabel('--')
        self.size_label = QLabel('Size:')

        #layout
        self.w0.addWidget(self.centerPos_label , row=0,col=0)
        self.w0.addWidget(self.centerPos_text, row=0,col=1)

        self.w0.addWidget(self.angle_label , row=1,col=0)
        self.w0.addWidget(self.angle_text, row=1,col=1)

        self.w0.addWidget(self.size_label , row=2,col=0)
        self.w0.addWidget(self.size_text, row=2,col=1)

        #add layout widget to dock
        self.d0.addWidget(self.w0)


    def show(self):
        """
        Shows the main window.
        """
        self.win.show()

    def close(self):
        """
        Closes the main window.
        """
        self.win.close()

    def hide(self):
        """
        Hides the main window.
        """
        self.win.hide()



class TemplateROI(pg.RectROI):
    def __init__(self, template = 'disk', display= None, *args, **kwds):
        pg.RectROI.__init__(self, aspectLocked=True, *args, **kwds)
        self.addRotateHandle([1,0], [0.5, 0.5])
        self.template = template

        self.outlinePen = pg.mkPen('y', width=1, style=Qt.DashLine)
        self.centerPointPen = pg.mkPen('g', width=20)


    def paint(self, p, opt, widget):
        radius = self.getState()['size'][1]
        if self.template == 'disk':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            #get center of ROI
            br = self.boundingRect()
            center = br.center()
            #draw shape
            p.drawEllipse(center,int(radius/2),int(radius/2))


        elif self.template == 'square':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawRect(int(0), int(0), int(radius), int(radius))

        elif self.template == 'crossbow':
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            #p.drawLine(Point(radius/2, -radius/2), Point(radius/2, radius/2))
            p.drawLine(Point(0, radius/2), Point(radius, radius/2))

            pen2 = QPen()
            pen2.setWidth(20)
            pen2.setColor(Qt.green)
            p.setPen(pen2)
            p.drawPoint(int(radius-10),int(radius/2))


        elif self.template == 'y-shape':
            #get equilatoral triangle vertices
            angles = np.linspace(0, np.pi * 4 / 3, 3)
            verticies = (np.array((np.sin(angles), np.cos(angles))).T * radius) / 2.0
            #scale by roi size
            verticies = np.array([[i[0] + radius/2, i[1] + radius/2] for i in verticies])
            #create poly
            poly = QPolygonF()
            for pt in verticies:
                poly.append(QPointF(*pt))

            #draw shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawPolygon(poly)


        elif self.template == 'h-shape':
            #shape
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(self.currentPen)
            p.drawLine(Point(0, radius/5), Point(radius, radius/5))
            p.drawLine(Point(0, radius-(radius/5)), Point(radius, radius-(radius/5)))


        #draw center point
        p.setPen(self.centerPointPen)
        p.drawPoint(int(radius/2),int(radius/2))


        #outline
        # Note: don't use self.boundingRect here, because subclasses may need to redefine it.
        r = QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(self.outlinePen)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

        self.updateDisplay()

    def getCenter(self):
        posX,posY = self.pos()
        sizeX,sizeY = self.size()
        return (int(posX+sizeX/2), int(posY+sizeY/2))


    def getAngle(self):
        return self.angle() % 360

    def getSize(self):
        return self.size()[0]

    def addDisplay(self, display):
        self.display = display

    def updateDisplay(self):
        if self.display != None:
            x,y = self.getCenter()
            angle = self.getAngle()
            size = self.getSize()
            self.display.centerPos_text.setText('{},{}'.format(x,y))
            self.display.angle_text.setText('{}'.format(round(angle, 2)))
            self.display.size_text.setText('{}'.format(round(size, 2)))

shapeSize = [400,400]
startPosition = [80, 50]

disk_template = TemplateROI(pos=startPosition, size=shapeSize, template = 'disk', pen=(0,9))
square_template = TemplateROI(pos=startPosition, size=shapeSize, template = 'square', pen=(0,9))
crossbow_template = TemplateROI(pos=startPosition, size=shapeSize, template = 'crossbow', pen=(0,9))
yShape_template = TemplateROI(pos=startPosition, size=shapeSize, template = 'y-shape', pen=(0,9))
hShape_template = TemplateROI(pos=startPosition, size=shapeSize, template = 'h-shape', pen=(0,9))

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
                          'Crossbow': crossbow_template,
                          'Y-shape': yShape_template,
                          'H-shape': hShape_template
                          }
        self.templateBox.setItems(self.templates)

        self.items.append({'name': 'dataWindow', 'string': 'Data Window', 'object': self.dataWindow})
        self.items.append({'name': 'template', 'string': 'Choose template', 'object': self.templateBox})
        self.items.append({'name': 'startButton', 'string': '', 'object': self.startButton})
        self.items.append({'name': 'endButton', 'string': '', 'object': self.endButton})

        super().gui()

        #initialize display window and hide it
        self.displayParams = DisplayParams(self)
        self.displayParams.show()

        return

    def startAlign(self):
        #initiate template
        template = self.getValue('template')
        template.addDisplay(self.displayParams)
        self.getValue('dataWindow').imageview.addItem(template)
        self.currentTemplate = template

        #autodetect micropattern
        img = self.getValue('dataWindow').image

        print(img.shape)

        #get max projection if stack
        if len(img.shape) > 2:
            img = np.max(img,0)

        print(img.shape)

        #blur
        img = gaussian_filter(img, 3)

        #set threshold
        threshold = threshold_otsu(img)

        # Create a mask
        mask = np.zeros_like(img)
        mask[img > threshold] = 1

        # Apply the mask to the image stack
        img = img * mask

        #img = binary_fill_holes(img).astype(int)
        img = binary_closing(img, iterations=50).astype(int)

        #blur again
        img = gaussian_filter(img, 3)

        # Find the location of micropattern (assuming 1 per image)
        objs = find_objects(img)

        # Get the height and width
        height = int(objs[0][0].stop - objs[0][0].start)
        width = int(objs[0][1].stop - objs[0][1].start)

        center = center_of_mass(img)
        if len(center) == 2:
            x,y = center[0], center[1]
        else:
            x,y = center[1], center[2]
        print(x,y)

        #display autodetected micropattern binary
        Window((img))

        #attempt to scale
        self.currentTemplate.scale(height/self.currentTemplate.size()[0])

        #attempt to center on micropattern
        self.currentTemplate.setPos((x-(self.currentTemplate.size()[0]/2),y-(self.currentTemplate.size()[0]/2)))

        #use phasediff to guess rotation
        #generate image for comparison
        rotated = np.zeros_like(img)

        shapeSize = 300

        if self.currentTemplate.template == 'square':
            rotShape = np.ones((shapeSize ,shapeSize ))
        else:
            return

        startx = int(img.shape[0]/2-(shapeSize/2) )
        starty = int(img.shape[1]/2-(shapeSize/2) )
        endx = startx+rotShape.shape[0]
        endy = startx+rotShape.shape[1]

        # Add the value 10 to the index (1, 1)
        rotated[startx:endx,starty:endy] = rotShape

        shifts, error, phasediff = phase_cross_correlation(img,
                                                           rotated,
                                                           normalization=None)
        angleToRotate = shifts[0]

        print('rotation: {}'.format(shifts[0]))

        #attempt to rotate to match micropattern template
        self.currentTemplate.rotate(angleToRotate)

        Window(rotated)

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











