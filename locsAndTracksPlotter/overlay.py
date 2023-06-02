#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:32:14 2023

@author: george
"""

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage import data, color, measure
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import math
from math import cos, sin, degrees
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Arrow
from tqdm import tqdm


# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

from pyqtgraph import HistogramLUTWidget

from .helperFunctions import *
from .io import FileSelector_overlay

class Overlay():
    """
    Overlay single tiff and recording image stack.

    """
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI

        self.dataIMG = None

        self.overlayIMG = None
        self.overlayFileName = None

        self.pathitems = []
        self.pathitemsActin = []

        self.actinLabels = []
        self.pointsInFilaments= []
        self.pointsNotInFilaments = []

        self.pointsPlotted = False

        self.pointMapScatter = None

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1000, 500)
        self.win.setWindowTitle('Overlay')

        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("Overlay", size=(500,500))
        self.d2 = Dock('Options', size=(250,500))

        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d2, 'right', self.d1)

        #overlay image view
        self.overlayWindow = pg.ImageView()
        self.d1.addWidget(self.overlayWindow)

        #options widget
        self.w2 = pg.LayoutWidget()

        #load tiff button
        self.loadTiff_button = FileSelector_overlay(filetypes='*.tif')
        self.loadTiff_button.valueChanged.connect(self.loadTiff)

        #overlay button
        self.showData_button = QPushButton('Show Data')
        self.showData_button.pressed.connect(self.toggleData)

        #detect filaments button
        self.getFilaments_button = QPushButton('Detect Filaments')
        self.getFilaments_button.pressed.connect(self.detectFilaments)

        #detect track axis
        self.getTrackAxis_button = QPushButton('Detect Track axis')
        self.getTrackAxis_button.pressed.connect(self.detectTrackAxis)

        #Opacity slider
        self.opacity = SliderLabel(1)
        self.opacity.setRange(0.0,1.0)
        self.opacity.setValue(0.5)
        self.opacity.setSingleStep(0.1)
        self.opacity.valueChanged.connect(self.updateOpacity)
        self.opacity_label = QLabel('Opacity')

        #Gamma correct slider
        self.gammaCorrect = CheckBox()
        self.gamma = SliderLabel(1)
        self.gamma.setRange(0.0,20.0)
        self.gamma.setValue(0.0)
        self.gamma.valueChanged.connect(self.updateGamma)
        self.gamma_label = QLabel('Gamma Correct')
        self.gammaCorrect.stateChanged.connect(self.resetGamma)

        #Threshold slider
        self.manualThreshold = CheckBox()
        self.threshold_slider = SliderLabel(1)
        self.threshold_slider.setRange(0.0,20.0)
        self.threshold_slider.setValue(0.0)
        self.threshold_slider.valueChanged.connect(self.detectFilaments)
        self.threshold_label = QLabel('Manual Threshold For Filament Detection')
        self.manualThreshold.stateChanged.connect(self.detectFilaments)

        #Size limit slider
        self.maxSize_slider = SliderLabel(1)
        self.maxSize_slider.setRange(0,10000)
        self.maxSize_slider.setValue(1000)
        self.maxSize_slider.valueChanged.connect(self.detectFilaments)
        self.maxSize_label = QLabel('Max size For Filament Detection')

        self.minSize_slider = SliderLabel(1)
        self.minSize_slider.setRange(0,10000)
        self.minSize_slider.setValue(50)
        self.minSize_slider.valueChanged.connect(self.detectFilaments)
        self.minSize_label = QLabel('Min size For Filament Detection')


        #Point Plot Threshold slider
        self.pointThreshold = CheckBox()
        self.pointThreshold_slider = SliderLabel(1)
        self.pointThreshold_slider.setRange(0.0,1000.0)
        self.pointThreshold_slider.setValue(0.0)
        self.pointThreshold_slider.valueChanged.connect(self.plotPoints)
        self.pointThreshold_label = QLabel('Filter Points By Actin Intensity')
        self.pointThreshold.stateChanged.connect(self.plotPoints)

        #add buttons etc to layout widget
        self.w2.addWidget(self.loadTiff_button, row=1,col=0)

        self.w2.addWidget(self.opacity_label, row=2,col=0)
        self.w2.addWidget(self.opacity, row=3,col=0)

        self.w2.addWidget(self.gamma_label, row=4,col=0)
        self.w2.addWidget(self.gammaCorrect, row=4,col=1)
        self.w2.addWidget(self.gamma, row=5,col=0)

        self.w2.addWidget(self.threshold_label, row=6,col=0)
        self.w2.addWidget(self.manualThreshold, row=6,col=1)
        self.w2.addWidget(self.threshold_slider, row=7,col=0)

        self.w2.addWidget(self.minSize_label, row=8,col=0)
        self.w2.addWidget(self.minSize_slider, row=9,col=0)

        self.w2.addWidget(self.maxSize_label, row=10,col=0)
        self.w2.addWidget(self.maxSize_slider, row=11,col=0)

        self.w2.addWidget(self.pointThreshold_label, row=12,col=0)
        self.w2.addWidget(self.pointThreshold, row=12,col=1)
        self.w2.addWidget(self.pointThreshold_slider, row=13,col=0)

        self.w2.addWidget(self.showData_button, row=14,col=0)
        self.w2.addWidget(self.getFilaments_button, row=15,col=0)
        self.w2.addWidget(self.getTrackAxis_button, row=16,col=0)

        #add layout widget to dock
        self.d2.addWidget(self.w2)

    def overlay(self):
        '''overlay single tiff file and recording'''
        self.overlayedIMG = self.overlayIMG

        #self.OverlayLUT = 'inferno'
        self.OverlayMODE = QPainter.CompositionMode_SourceOver

        self.bgItem = pg.ImageItem()
        if self.gammaCorrect.isChecked():
            self.overlayedIMG = gammaCorrect(self.overlayedIMG, self.gamma.value())
        self.bgItem.setImage(self.overlayedIMG, autoRange=False, autoLevels=False, opacity=self.opacity.value())
        self.bgItem.setCompositionMode(self.OverlayMODE)
        self.overlayWindow.view.addItem(self.bgItem)

        self.bgItem.hist_luttt = HistogramLUTWidget(fillHistogram = False)
        self.bgItem.hist_luttt.setMinimumWidth(110)
        self.bgItem.hist_luttt.setImageItem(self.bgItem)

        self.overlayWindow.ui.gridLayout.addWidget(self.bgItem.hist_luttt, 0, 4, 1, 4)


    def updateGamma(self):
        '''aply gamma correction using value from slider'''
        if self.gammaCorrect.isChecked():
            levels = self.bgItem.hist_luttt.getLevels()
            gammaCorrrectedImg = gammaCorrect(self.overlayIMG, self.gamma.value())
            self.bgItem.setImage(gammaCorrrectedImg, autoLevels=False, levels=levels, opacity=self.opacity.value())

    def resetGamma(self):
        '''reset the gamma value used to overlay images'''
        if self.gammaCorrect.isChecked():
            self.updateGamma()
        else:
            levels = self.bgItem.hist_luttt.getLevels()
            self.bgItem.setImage(self.overlayIMG, autoLevels=False, levels=levels, opacity=self.opacity.value())

    def updateOpacity(self):
        '''set opacity of overlaid images'''
        green = self.overlayIMG
        levels = self.bgItem.hist_luttt.getLevels()
        if self.gammaCorrect.isChecked():
            green = gammaCorrect(green, self.gamma.value())

        self.bgItem.setImage(green, autoLevels=False, levels=levels, opacity=self.opacity.value())


    def loadTiff(self):
        """ imports the tiff file to overlay """
        #get filename
        self.overlayFileName = self.loadTiff_button.value()

        #load overlay file
        self.overlayIMG = skio.imread(self.overlayFileName)

        #if stack convert to single image
        if len(self.overlayIMG) > 1:
            self.overlayIMG = self.overlayIMG[0] #just 1st image of stack

        #orient image
        self.overlayIMG = np.rot90(self.overlayIMG)
        self.overlayIMG = np.flipud(self.overlayIMG)

        #overlay images
        self.overlay()

        #set manual threshold slider range
        self.threshold_slider.setRange(np.min(self.overlayIMG),np.max(self.overlayIMG))
        self.pointThreshold_slider.setRange(np.min(self.overlayIMG),np.max(self.overlayIMG))

        #add actin image pixel intensity to data df
        self.addActinIntensity()


    def toggleData(self):
        if self.pointsPlotted:
            self.hidePoints()
        else:
            self.plotPoints()

    def loadData(self):
        self.dataIMG = self.mainGUI.plotWindow.image
        self.overlayWindow.setImage(self.dataIMG)


    def confidence_ellipse(self, x, y, ax, nstd=2.0, facecolor='none', **kwargs):
        """
        Return a matplotlib Ellipse patch representing the covariance matrix
        cov centred at centre and scaled by the factor nstd.

        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        cov = np.cov(x, y)
        centre = (np.mean(x), np.mean(y))

        # Find and sort eigenvalues and eigenvectors into descending order
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The anti-clockwise angle to rotate our ellipse by
        vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(eigvals)

        ellipse = Ellipse(xy=centre, width=width, height=height,
                       angle=np.degrees(theta), facecolor=facecolor, **kwargs)

        if width > height:
            r = width
        else:
            r = height

        arrow = Arrow(np.mean(x), np.mean(y), r*cos(theta), r*sin(theta), width=1)

        ax.add_patch(ellipse)
        ax.add_patch(arrow)
        majorAxis_deg  = degrees(theta)
        #majorAxis_deg = (majorAxis_deg + 360) % 360 # +360 for implementations where mod returns negative numbers

        #print(majorAxis_deg)
        #print(ellipse.properties()['angle'])

        return ax, majorAxis_deg

    def detectTrackAxis(self):
        '''determine track direction'''

        if self.mainGUI.useFilteredData:
            track_data = self.mainGUI.filteredData
        else:
            track_data = self.mainGUI.data

        trackList= track_data['track_number'].unique().tolist()

        fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
        axs3.set_aspect('equal', adjustable='box')
        axs3.scatter(track_data['zeroed_X'],track_data['zeroed_Y'])
        axs3.axvline(c='grey', lw=1)
        axs3.axhline(c='grey', lw=1)

        degreeList = []

        for n in tqdm(trackList):
            track = track_data[track_data['track_number'] == n]
            _, degree = self.confidence_ellipse(track['zeroed_X'], track['zeroed_Y'], axs3, edgecolor='red')
            degreeList.append(degree)


        correctedDegList = []
        for deg in degreeList:
            if deg < 0:
                deg = deg + 180
            correctedDegList.append(deg)

        fig3.show()

        #fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
        #axs5.hist(correctedDegList,10)

    def detectFilaments(self):
        '''determine direction of thresholded filaments'''
        actin_img = self.overlayedIMG

        #flip image to match pyqtgraph layout
        actin_img = np.rot90(actin_img)
        actin_img = np.flipud(actin_img)


        #clear plotted outlines and actin label list
        self.clearActinOutlines()
        self.actinLabels = []

        if self.manualThreshold.isChecked():
            thresh = self.threshold_slider.value()

        else:
            thresh = threshold_otsu(actin_img)
        #binary = actin_img > thresh
        #edges = canny(binary, sigma=3)

        bw = closing(actin_img > thresh, square(5))
        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=actin_img, bg_label=0)

        fig6, [axs6,axs7,axs8] = plt.subplots(1, 3, figsize=(15, 5))
        axs6.imshow(actin_img, origin='lower')
        #axs6.scatter(longTracks['x'],longTracks['y'])

        axs7.imshow(cleared, origin='lower')
        axs8.imshow(image_label_overlay, origin='lower')


        orientationList = []

        # for props in regionprops(label_image):
        #     # take regions with large enough areas
        #     if props.area >= self.minSize_slider.value() and props.area < self.maxSize_slider.value():
        #         y0, x0 = props.centroid
        #         orientation = props.orientation
        #         orientationList.append(degrees(orientation))
        #         x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        #         y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        #         x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        #         y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        #         axs8.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        #         axs8.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        #         axs8.plot(x0, y0, '.g', markersize=15)

        # correctedDegList_actin = []
        # for deg in orientationList:
        #     if deg < 0:
        #         deg = deg + 180
        #     correctedDegList_actin.append(deg)

        # axs6.invert_yaxis()
        # axs7.invert_yaxis()
        # axs8.invert_yaxis()
        # #fig6.show()

        # # fig7, axs9 = plt.subplots(1, 1, figsize=(5, 5))
        # # axs9.hist(correctedDegList_actin,10)

        labels = measure.label(cleared)
        props = measure.regionprops(labels, actin_img)

        #table = measure.regionprops_table(actin_img,properties=['label','area'])
        #print(table)

        #add label objects to overlay window
        for index in range(1, labels.max()):

            label_i = props[index].label
            area = props[index].area

            if area >= self.minSize_slider.value() and area < self.maxSize_slider.value():
                self.actinLabels.append(labels == label_i)
                contour = measure.find_contours(labels == label_i, 0.5)[0]
                y, x = contour.T

                pathitem = QGraphicsPathItem(self.overlayWindow.view)

                pen = pg.functions.mkPen(width=1)

                # set the color of the pen based on the track color
                pen.setColor(QColor(Qt.red))

                # set the pen for the path items
                pathitem.setPen(pen)

                # add the path items to the view(s)
                self.overlayWindow.view.addItem(pathitem)

                # keep track of the path items
                self.pathitemsActin.append(pathitem)

                # create a QPainterPath for the track and set the path for the path item
                path = QPainterPath(QPointF(x[0],y[0]))

                path_overlay = QPainterPath(QPointF(x[0],y[0]))

                for i in np.arange(1, len(x)):
                    path.lineTo(QPointF(x[i],y[i]))

                pathitem.setPath(path)

                #get centroids and orientation
                y0, x0 = props[index].centroid
                orientation = props[index].orientation
                orientationList.append(degrees(orientation))
                x1 = x0 + math.cos(orientation) * 0.5 * props[index].minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props[index].minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props[index].major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props[index].major_axis_length


        correctedDegList_actin = []
        for deg in orientationList:
            if deg < 0:
                deg = deg + 180
            correctedDegList_actin.append(deg)


    def getIntensities(self, img, x_positions, y_positions):
        y_max, x_max = img.shape
        #intensities retrieved from image stack using point data (converted from floats to ints)
        y_positions = y_positions.astype(int)
        x_positions = x_positions.astype(int)
        #edge cases
        y_positions[y_positions == y_max] = y_max
        x_positions[x_positions == x_max] = x_max
        intensities = img[x_positions, y_positions]
        return intensities

    def addActinIntensity(self):
        #add column with actin intensitys to data df
        self.mainGUI.data['actin_intensity'] = self.getIntensities(self.overlayedIMG, self.mainGUI.data['x'], self.mainGUI.data['y'])
        if self.mainGUI.useFilteredData:
            self.mainGUI.filteredData['actin_intensity'] = self.getIntensities(self.overlayedIMG, self.mainGUI.filteredData['x'], self.mainGUI.filteredData['y'])
        self.mainGUI.data_unlinked['actin_intensity'] = self.getIntensities(self.overlayedIMG, self.mainGUI.data_unlinked['x'], self.mainGUI.data_unlinked['y'])

    def plotPoints(self):
        #clear scatterplot
        if self.pointMapScatter is not None:
            self.pointMapScatter.clear()

        # Check if filtered data is being used, if not use the original data
        if self.mainGUI.useFilteredData == False:
            df = self.mainGUI.data
        else:
            df = self.mainGUI.filteredData

        #add unlinked points if displayed
        if self.mainGUI.displayUnlinkedPoints:
            df = df.append(self.mainGUI.data_unlinked)

        #filter df based on actin_intensity column
        if self.pointThreshold.isChecked():
            df = df[df['actin_intensity'] > self.pointThreshold_slider.value()]

        # Create a ScatterPlotItem and add it to the ImageView
        self.pointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
        self.pointMapScatter.setSize(2, update=False)
        self.pointMapScatter.setData(df['x'], df['y'])
        self.overlayWindow.view.addItem(self.pointMapScatter)
        #update flag
        self.pointsPlotted = True

    def hidePoints(self):
        # Remove the ScatterPlotItem from the ImageView
        self.overlayWindow.view.removeItem(self.pointMapScatter)
        #update flag
        self.pointsPlotted = False


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


    def clearActinOutlines(self):
        # Remove all plot items representing tracks
        if self.overlayWindow is not None:
            for pathitem in self.pathitemsActin:
                self.overlayWindow.view.removeItem(pathitem)
        self.pathitemsActin = []

    def clearTracks(self):
        # Remove all plot items representing tracks
        if self.overlayWindow is not None:
            for pathitem in self.pathitems:
                self.overlayWindow.view.removeItem(pathitem)
        self.pathitems = []
