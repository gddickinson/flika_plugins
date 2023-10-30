#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:32:14 2023

@author: george
"""

from .io import FileSelector_overlay
from .helperFunctions import *
from pyqtgraph import HistogramLUTWidget
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import skimage.io as skio
from skimage.filters import threshold_otsu, threshold_local, threshold_isodata, gaussian
from skimage import data, color, measure
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
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
from flika.process.file_ import open_tiff
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


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
        self.pathitemsActin_binary = []

        self.actinLabels = []
        self.pointsInFilaments = []
        self.pointsNotInFilaments = []

        self.pointsPlotted = False

        self.pointMapScatter = None

        self.showFilaments = False

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1000, 500)
        self.win.setWindowTitle('Overlay')

        # Create docks, place them into the window one at a time.
        self.d1 = Dock("Overlay", size=(500, 500))
        self.d2 = Dock('Options - main', size=(250, 500))
        self.d3 = Dock("Binary", size=(500, 500))
        self.d4 = Dock('Options - binary', size=(250, 500))

        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d2, 'right', self.d1)
        self.area.addDock(self.d3, 'below', self.d1)
        self.area.addDock(self.d4, 'below', self.d2)

        self.d1.raiseDock()
        self.d2.raiseDock()

        # overlay image view
        self.overlayWindow = pg.ImageView()
        self.d1.addWidget(self.overlayWindow)

        # binary image view
        self.binaryWindow = pg.ImageView()
        self.d3.addWidget(self.binaryWindow)

        # options widget
        self.w2 = pg.LayoutWidget()
        self.w4 = pg.LayoutWidget()

        # load tiff button
        self.loadTiff_button = FileSelector_overlay(filetypes='*.tif')
        self.loadTiff_button.valueChanged.connect(self.loadTiff)

        # load active window button
        self.loadWindow_button = WindowSelector()
        self.loadWindow_button.valueChanged.connect(self.loadWindow)

        # overlay button
        self.showData_button = QPushButton('Show Data')
        self.showData_button.pressed.connect(self.toggleData)

        # detect filaments button
        self.getFilaments_button = QPushButton('Show Detected Filaments')
        self.getFilaments_button.pressed.connect(self.showDetectedFilaments)

        # detect track axis
        self.getTrackAxis_button = QPushButton('Detect Track axis')
        self.getTrackAxis_button.pressed.connect(self.detectTrackAxis)

        # Opacity slider
        self.opacity = SliderLabel(1)
        self.opacity.setRange(0, 10)
        self.opacity.setValue(5)
        self.opacity.setSingleStep(1)
        self.opacity.valueChanged.connect(self.updateOpacity)
        self.opacity_label = QLabel('Opacity')

        # Gamma correct slider
        self.gammaCorrect = CheckBox()
        self.gamma = SliderLabel(1)
        self.gamma.setRange(1, 200)
        self.gamma.setValue(1)
        self.gamma.valueChanged.connect(self.updateGamma)
        self.gamma_label = QLabel('Gamma Correct')
        self.gammaCorrect.stateChanged.connect(self.resetGamma)

        # Gaussian blur slider
        self.gaussianBlur = CheckBox()
        self.gaussian_slider = SliderLabel(1)
        self.gaussian_slider.setRange(0, 20)
        self.gaussian_slider.setValue(1.5)
        self.gaussian_slider.valueChanged.connect(self.updateBinary)
        self.gaussian_label = QLabel('Gaussian Blur sigma')
        self.gaussianBlur.stateChanged.connect(self.updateBinary)

        # Threshold slider
        self.manualThreshold = CheckBox()
        self.threshold_slider = SliderLabel(1)
        self.threshold_slider.setRange(0, 20)
        self.threshold_slider.setValue(0)
        self.threshold_slider.valueChanged.connect(self.updateBinary)
        self.threshold_label = QLabel('Global Threshold cuttoff')
        self.manualThreshold.stateChanged.connect(self.updateBinary)

        # local threshold sliders
        self.blocksize_slider = SliderLabel(0)
        self.blocksize_slider.setRange(1, 201)
        self.blocksize_slider.setSingleStep(2)
        self.blocksize_slider.setValue(35)
        self.blocksize_slider.valueChanged.connect(self.updateBinary)
        self.blocksize_label = QLabel('Local Threshold block size')

        self.offset_slider = SliderLabel(0)
        self.offset_slider.setRange(-500, 500)
        self.offset_slider.setValue(10)
        self.offset_slider.valueChanged.connect(self.updateBinary)
        self.offset_label = QLabel('Local Threshold offset')

        # close hole size selector
        self.fillHoles = CheckBox()
        self.hole_slider = SliderLabel(0)
        self.hole_slider.setRange(1, 50)
        self.hole_slider.setValue(3)
        self.hole_slider.valueChanged.connect(self.updateBinary)
        self.fillHoles.stateChanged.connect(self.updateBinary)
        self.hole_label = QLabel('Hole size to close')

        # remove speckle size selector
        self.removeSpeckle = CheckBox()
        self.speckle_slider = SliderLabel(0)
        self.speckle_slider.setRange(1, 50)
        self.speckle_slider.setValue(3)
        self.speckle_slider.valueChanged.connect(self.updateBinary)
        self.removeSpeckle.stateChanged.connect(self.updateBinary)
        self.speckle_label = QLabel('Speckle size to remove')


        # Size limit slider
        self.maxSizeLimit = CheckBox()
        self.maxSize_slider = SliderLabel(1)
        self.maxSize_slider.setRange(0, 10000)
        self.maxSize_slider.setValue(1000)
        self.maxSize_slider.valueChanged.connect(self.detectFilaments)
        self.maxSize_label = QLabel('Max size For Filament Detection')

        self.minSize_slider = SliderLabel(1)
        self.minSize_slider.setRange(0, 10000)
        self.minSize_slider.setValue(50)
        self.minSize_slider.valueChanged.connect(self.detectFilaments)
        self.minSize_label = QLabel('Min size For Filament Detection')

        # Point Plot Threshold slider
        self.pointThreshold = CheckBox()
        self.pointThreshold_slider = SliderLabel(1)
        self.pointThreshold_slider.setRange(0, 1000)
        self.pointThreshold_slider.setValue(0)
        self.pointThreshold_slider.valueChanged.connect(self.plotPoints)
        self.pointThreshold_label = QLabel('Filter Points By Actin Intensity')
        self.pointThreshold.stateChanged.connect(self.plotPoints)

        # add buttons etc to layout widget
        #main image options
        self.w2.addWidget(self.loadTiff_button, row=1, col=0)
        self.w2.addWidget(self.loadWindow_button, row=2, col=0)

        self.w2.addWidget(self.opacity_label, row=3, col=0)
        self.w2.addWidget(self.opacity, row=4, col=0)

        self.w2.addWidget(self.gamma_label, row=5, col=0)
        self.w2.addWidget(self.gammaCorrect, row=5, col=2)
        self.w2.addWidget(self.gamma, row=6, col=0)

        self.w2.addWidget(self.minSize_label, row=7, col=0)
        self.w2.addWidget(self.minSize_slider, row=8, col=0)

        self.w2.addWidget(self.maxSize_label, row=9, col=0)
        self.w2.addWidget(self.maxSizeLimit, row=9, col=2)
        self.w2.addWidget(self.maxSize_slider, row=10, col=0)

        self.w2.addWidget(self.pointThreshold_label, row=11, col=0)
        self.w2.addWidget(self.pointThreshold, row=11, col=2)
        self.w2.addWidget(self.pointThreshold_slider, row=12, col=0)

        self.w2.addWidget(self.showData_button, row=13, col=0)
        self.w2.addWidget(self.getFilaments_button, row=14, col=0)
        self.w2.addWidget(self.getTrackAxis_button, row=15, col=0)

        #binary options
        self.w4.addWidget(self.gaussian_label, row=1, col=0)
        self.w4.addWidget(self.gaussianBlur, row=1, col=2)
        self.w4.addWidget(self.gaussian_slider, row=2, col=0)

        self.w4.addWidget(self.blocksize_label, row=3, col=0)
        self.w4.addWidget(self.blocksize_slider, row=4, col=0)

        self.w4.addWidget(self.offset_label, row=5, col=0)
        self.w4.addWidget(self.offset_slider, row=6, col=0)

        self.w4.addWidget(self.threshold_label, row=7, col=0)
        self.w4.addWidget(self.manualThreshold, row=7, col=2)
        self.w4.addWidget(self.threshold_slider, row=8, col=0)

        self.w4.addWidget(self.hole_label, row=9, col=0)
        self.w4.addWidget(self.fillHoles, row=9, col=2)
        self.w4.addWidget(self.hole_slider, row=10, col=0)

        self.w4.addWidget(self.speckle_label, row=11, col=0)
        self.w4.addWidget(self.removeSpeckle, row=11, col=2)
        self.w4.addWidget(self.speckle_slider, row=12, col=0)


        # add layout widget to dock
        self.d2.addWidget(self.w2)
        self.d4.addWidget(self.w4)

    def overlay(self):
        '''overlay single tiff file and recording'''
        self.overlayedIMG = self.overlayIMG

        # self.OverlayLUT = 'inferno'
        self.OverlayMODE = QPainter.CompositionMode_SourceOver

        self.bgItem = pg.ImageItem()
        if self.gammaCorrect.isChecked():
            self.overlayedIMG = gammaCorrect(
                self.overlayedIMG, self.gamma.value()/10)
        self.bgItem.setImage(self.overlayedIMG, autoRange=False,
                             autoLevels=False, opacity=self.opacity.value()/10)
        self.bgItem.setCompositionMode(self.OverlayMODE)
        self.overlayWindow.view.addItem(self.bgItem)

        self.bgItem.hist_luttt = HistogramLUTWidget(fillHistogram=False)
        self.bgItem.hist_luttt.setMinimumWidth(110)
        self.bgItem.hist_luttt.setImageItem(self.bgItem)

        self.overlayWindow.ui.gridLayout.addWidget(
            self.bgItem.hist_luttt, 0, 4, 1, 4)

    def updateGamma(self):
        '''apply gamma correction using value from slider'''
        if self.gammaCorrect.isChecked():
            levels = self.bgItem.hist_luttt.getLevels()
            self.overlayIMG = gammaCorrect(
                self.originalIMG, self.gamma.value()/10)
            self.bgItem.setImage(self.overlayIMG, autoLevels=False,
                                 levels=levels, opacity=self.opacity.value()/10)

        else:
            levels = self.bgItem.hist_luttt.getLevels()
            self.overlayIMG = self.originalIMG
            self.bgItem.setImage(self.overlayIMG, autoLevels=False,
                                 levels=levels, opacity=self.opacity.value()/10)

        self.updateBinary()

    def resetGamma(self):
        '''reset the gamma value used to overlay images'''
        if self.gammaCorrect.isChecked():
            self.updateGamma()
        else:
            levels = self.bgItem.hist_luttt.getLevels()
            self.overlayIMG = self.originalIMG
            self.bgItem.setImage(self.overlayIMG, autoLevels=False,
                                 levels=levels, opacity=self.opacity.value()/10)

    def updateOpacity(self):
        '''set opacity of overlaid images'''
        # if self.gammaCorrect.isChecked():
        #     green = gammaCorrect(green, self.gamma.value()/10)

        green = self.overlayIMG
        levels = self.bgItem.hist_luttt.getLevels()

        self.bgItem.setImage(green, autoLevels=False,
                             levels=levels, opacity=self.opacity.value()/10)

    def updateBinary(self):
        # apply gaussian blur
        if self.gaussianBlur.isChecked():
            actin_img = gaussian(self.overlayIMG, sigma=self.gaussian_slider.value())
        else:
            actin_img = self.overlayIMG

        # global thresholding
        if self.manualThreshold.isChecked():
            thresh_mask = self.threshold_slider.value()
            if self.gaussianBlur.isChecked():
                thresh_mask = thresh_mask/100000


        # else:
        #     thresh_mask = threshold_otsu(actin_img)
        # binary = actin_img > thresh_mask
        # edges = canny(binary, sigma=3)

       # local thresholding
        else:
            block_size = int(self.blocksize_slider.value())

            # force block_size to be odd
            block_size -= (1 - block_size % 2)

            offset = int(self.offset_slider.value())

            #offset needs to be rescaled if image is blurred
            #TODO! find a better way to do this - intensity histogram?
            if self.gaussianBlur.isChecked():
                offset = offset/100000


            method = 'gaussian'
            thresh_mask = threshold_local(
                actin_img, block_size, method=method, offset=offset)

        # get binary & close holes
        if self.fillHoles.isChecked():
            self.binary = closing(actin_img > thresh_mask, square(self.hole_slider.value()))
        else:
            self.binary = actin_img > thresh_mask

        if self.removeSpeckle.isChecked():
            self.binary = remove_small_objects(self.binary, min_size=self.speckle_slider.value())

        # convert boolean to int
        self.binary = self.binary.astype(int)

        self.binaryWindow.setImage(self.binary)

        # update filament detection
        self.detectFilaments()

    def loadTiff(self):
        """ imports the tiff file to overlay """
        # get filename
        self.overlayFileName = self.loadTiff_button.value()

        # load overlay file
        #self.overlayIMG = skio.imread(self.overlayFileName)
        self.overlayIMG, metadata = open_tiff(self.overlayFileName, None)

        # if stack convert to single image
        if len(self.overlayIMG.shape) > 2:
            self.overlayIMG = self.overlayIMG[0]  # just 1st image of stack


        # orient image
        #self.overlayIMG = np.rot90(self.overlayIMG)
        #self.overlayIMG = np.flipud(self.overlayIMG)

        # make copy of original
        self.originalIMG = self.overlayIMG

        # overlay images
        self.overlay()

        self.updateBinary()

        # set manual threshold slider range
        self.threshold_slider.setRange(
            np.min(self.overlayIMG), np.max(self.overlayIMG))
        self.pointThreshold_slider.setRange(
            np.min(self.overlayIMG), np.max(self.overlayIMG))

        # add actin image pixel intensity to data df
        self.addActinIntensity()

    def loadWindow(self):
        """ imports window data to overlay """
        # get filename
        self.importWindow = self.loadWindow_button.value()

        # load overlay file
        self.overlayIMG = self.importWindow.image

        # if stack convert to single image
        # if len(self.overlayIMG) > 1:
        #     self.overlayIMG = self.overlayIMG[0]  # just 1st image of stack

        # orient image
        #self.overlayIMG = np.rot90(self.overlayIMG)
        #self.overlayIMG = np.flipud(self.overlayIMG)

        # make copy of original
        self.originalIMG = self.overlayIMG

        # overlay images
        self.overlay()

        self.updateBinary()

        # set manual threshold slider range
        self.threshold_slider.setRange(
            np.min(self.overlayIMG), np.max(self.overlayIMG))
        self.pointThreshold_slider.setRange(
            np.min(self.overlayIMG), np.max(self.overlayIMG))

        # add actin image pixel intensity to data df
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
        vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
        theta = np.arctan2(vy, vx)

        # Width and height of ellipse to draw
        width, height = 2 * nstd * np.sqrt(eigvals)

        ellipse = Ellipse(xy=centre, width=width, height=height,
                          angle=np.degrees(theta), facecolor=facecolor, **kwargs)

        if width > height:
            r = width
        else:
            r = height

        arrow = Arrow(np.mean(x), np.mean(y), r *
                      cos(theta), r*sin(theta), width=1)

        ax.add_patch(ellipse)
        ax.add_patch(arrow)
        majorAxis_deg = degrees(theta)
        # majorAxis_deg = (majorAxis_deg + 360) % 360 # +360 for implementations where mod returns negative numbers

        # print(majorAxis_deg)
        # print(ellipse.properties()['angle'])

        return ax, majorAxis_deg

    def detectTrackAxis(self):
        '''determine track direction'''

        if self.mainGUI.useFilteredData:
            track_data = self.mainGUI.filteredData
        else:
            track_data = self.mainGUI.data

        trackList = track_data['track_number'].unique().tolist()

        fig3, axs3 = plt.subplots(1, 1, figsize=(5, 5))
        axs3.set_aspect('equal', adjustable='box')
        axs3.scatter(track_data['zeroed_X'], track_data['zeroed_Y'])
        axs3.axvline(c='grey', lw=1)
        axs3.axhline(c='grey', lw=1)

        degreeList = []

        for n in tqdm(trackList):
            track = track_data[track_data['track_number'] == n]
            _, degree = self.confidence_ellipse(
                track['zeroed_X'], track['zeroed_Y'], axs3, edgecolor='red')
            degreeList.append(degree)

        correctedDegList = []
        for deg in degreeList:
            if deg < 0:
                deg = deg + 180
            correctedDegList.append(deg)

        fig3.show()

        # fig5, axs5 = plt.subplots(1, 1, figsize=(5, 5))
        # axs5.hist(correctedDegList,10)

    def detectFilaments(self):
        '''determine direction of thresholded filaments'''
        actin_img = self.overlayedIMG
        binary = self.binary

        # flip image to match pyqtgraph layout
        actin_img = np.rot90(actin_img)
        actin_img = np.flipud(actin_img)

        binary = np.rot90(binary)
        binary = np.flipud(binary)

        # clear plotted outlines and actin label list
        self.clearActinOutlines()
        self.actinLabels = []

        # label image regions
        label_image = label(binary)
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(
            label_image, image=actin_img, bg_label=0)

        fig6, [axs6, axs7, axs8] = plt.subplots(1, 3, figsize=(15, 5))
        axs6.imshow(actin_img, origin='lower')
        # axs6.scatter(longTracks['x'],longTracks['y'])

        axs7.imshow(binary, origin='lower')
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

        labels = measure.label(binary)
        props = measure.regionprops(labels, actin_img)

        # table = measure.regionprops_table(actin_img,properties=['label','area'])
        # print(table)

        # add label objects to overlay window
        for index in range(1, labels.max()):

            label_i = props[index].label
            area = props[index].area

            if self.maxSizeLimit.isChecked():
                maxTest = area <= self.maxSize_slider.value()
            else:
                maxTest = True

            if area >= self.minSize_slider.value() and maxTest:
                self.actinLabels.append(labels == label_i)
                contour = measure.find_contours(labels == label_i)[0]
                y, x = contour.T

                if self.showFilaments:

                    pathitem = QGraphicsPathItem(self.overlayWindow.view)
                    pathitem2 = QGraphicsPathItem(self.binaryWindow.view)

                    pen = pg.functions.mkPen(width=1)
                    pen2 = pg.functions.mkPen(width=3)

                    # set the color of the pen based on the track color
                    pen.setColor(QColor(Qt.red))
                    pen2.setColor(QColor(Qt.red))

                    # set the pen for the path items
                    pathitem.setPen(pen)
                    pathitem2.setPen(pen2)

                    # add the path items to the view(s)
                    self.overlayWindow.view.addItem(pathitem)
                    self.binaryWindow.view.addItem(pathitem2)

                    # keep track of the path items
                    self.pathitemsActin.append(pathitem)
                    self.pathitemsActin_binary.append(pathitem2)

                    # create a QPainterPath for the track and set the path for the path item
                    path = QPainterPath(QPointF(x[0], y[0]))

                    path_overlay = QPainterPath(QPointF(x[0], y[0]))

                    for i in np.arange(1, len(x)):
                        path.lineTo(QPointF(x[i], y[i]))

                    pathitem.setPath(path)
                    pathitem2.setPath(path)

                # get centroids and orientation
                y0, x0 = props[index].centroid
                orientation = props[index].orientation
                orientationList.append(degrees(orientation))
                x1 = x0 + math.cos(orientation) * 0.5 * \
                    props[index].minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * \
                    props[index].minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * \
                    props[index].major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * \
                    props[index].major_axis_length

        correctedDegList_actin = []
        for deg in orientationList:
            if deg < 0:
                deg = deg + 180
            correctedDegList_actin.append(deg)

    def getIntensities(self, img, x_positions, y_positions):
        y_max, x_max = img.shape
        # intensities retrieved from image stack using point data (converted from floats to ints)
        y_positions = y_positions.astype(int)
        x_positions = x_positions.astype(int)
        # edge cases
        y_positions[y_positions == y_max] = y_max
        x_positions[x_positions == x_max] = x_max
        intensities = img[x_positions, y_positions]
        return intensities

    def addActinIntensity(self):
        # add column with actin intensitys to data df
        self.mainGUI.data['actin_intensity'] = self.getIntensities(
            self.overlayedIMG, self.mainGUI.data['x'], self.mainGUI.data['y'])
        if self.mainGUI.useFilteredData:
            self.mainGUI.filteredData['actin_intensity'] = self.getIntensities(
                self.overlayedIMG, self.mainGUI.filteredData['x'], self.mainGUI.filteredData['y'])
        self.mainGUI.data_unlinked['actin_intensity'] = self.getIntensities(
            self.overlayedIMG, self.mainGUI.data_unlinked['x'], self.mainGUI.data_unlinked['y'])

    def plotPoints(self):
        # clear scatterplot
        if self.pointMapScatter is not None:
            self.pointMapScatter.clear()

        # Check if filtered data is being used, if not use the original data
        if self.mainGUI.useFilteredData == False:
            df = self.mainGUI.data
        else:
            df = self.mainGUI.filteredData

        # add unlinked points if displayed
        if self.mainGUI.displayUnlinkedPoints:
            df = df.append(self.mainGUI.data_unlinked)

        # filter df based on actin_intensity column
        if self.pointThreshold.isChecked():
            df = df[df['actin_intensity'] >= self.pointThreshold_slider.value()]

        # Create a ScatterPlotItem and add it to the ImageView
        self.pointMapScatter = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
        self.pointMapScatter.setSize(2, update=False)
        self.pointMapScatter.setData(df['x'], df['y'])
        self.overlayWindow.view.addItem(self.pointMapScatter)
        # update flag
        self.pointsPlotted = True

    def hidePoints(self):
        # Remove the ScatterPlotItem from the ImageView
        self.overlayWindow.view.removeItem(self.pointMapScatter)
        # update flag
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

    def showDetectedFilaments(self):
        if self.showFilaments == False:
            self.showFilaments = True
            self.detectFilaments()
        else:
            self.showFilaments = False
            self.detectFilaments()

    def clearActinOutlines(self):
        # Remove all plot items representing tracks
        if self.overlayWindow is not None:
            for pathitem in self.pathitemsActin:
                self.overlayWindow.view.removeItem(pathitem)

        if self.binaryWindow is not None:
            for pathitem in self.pathitemsActin_binary:
                self.binaryWindow.view.removeItem(pathitem)

        self.pathitemsActin = []
        self.pathitemsActin_binary = []

    def clearTracks(self):
        # Remove all plot items representing tracks
        if self.overlayWindow is not None:
            for pathitem in self.pathitems:
                self.overlayWindow.view.removeItem(pathitem)
        self.pathitems = []
