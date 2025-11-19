#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:23:54 2023

@author: george
"""

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
import pandas as pd
import pyqtgraph as pg
import os

# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

from distutils.version import StrictVersion
import flika
from flika.window import Window
from flika.roi import open_rois
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui

from .helperFunctions import *


class ColouredLines(pg.GraphicsObject):
    """
    A subclass of pg.GraphicsObject that displays a series of colored lines and points on a plot.
    """
    def __init__(self, points, colours_line, colours_point, width_line=2, width_point=1, size_symbol=0.05):
        """
        Constructor for ColouredLines.

        Args:
            points (list of tuples): A list of tuples specifying the points to connect with lines.
            colours_line (list of QColor): A list of colors to use for the lines connecting the points.
            colours_point (list of QColor): A list of colors to use for the points.
            width_line (int, optional): The width of the lines connecting the points. Defaults to 2.
            width_point (int, optional): The width of the points. Defaults to 1.
            size_symbol (float, optional): The size of the points as a fraction of the plot. Defaults to 0.05.
        """
        super().__init__()
        self.points = points
        self.colours_line = colours_line
        self.width_line = width_line
        self.colours_point = colours_point
        self.width_point = width_point
        self.size_symbol = size_symbol
        self.generatePicture()

    def generatePicture(self):
        """
        Generates a QPicture of the lines and points.
        """
        self.picture = QPicture()
        painter = QPainter(self.picture)
        pen = pg.functions.mkPen(width=self.width_line)

        # Draw lines connecting the points
        for idx in range(len(self.points) - 1):
            pen.setColor(self.colours_line[idx])
            painter.setPen(pen)
            painter.drawLine(self.points[idx], self.points[idx+1])

        # Draw points at the specified locations
        pen_points = pg.functions.mkPen(width=self.width_point)
        brush_points = pg.mkBrush(0, 0, 255, 255)

        for idx in range(len(self.points) - 1):
            pen_points.setColor(self.colours_point[idx])
            brush_points.setColor(self.colours_point[idx])
            painter.setPen(pen_points)
            painter.setBrush(brush_points)
            painter.drawEllipse(self.points[idx], self.size_symbol,self.size_symbol)

        painter.end()

    def paint(self, p, *args):
        """
        Paints the picture onto the plot.
        """
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """
        Returns the bounding rectangle of the picture.
        """
        return QRectF(self.picture.boundingRect())



class TrackPlot():
    """
    A class representing a GUI for visualizing tracked data for intracellular Piezo1 protein using a fluorescent tag.

    Attributes:
    - mainGUI (object): the main GUI object that contains the tracked data

    """
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI
        self.d = int(16)
        self.A_pad = None
        self.A_crop = None
        self.trackPoints = None
        self.bg_flag = False

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1400, 550)
        self.win.setWindowTitle('Track Plot')

        # Set default colour map types
        self.pointCMtype = 'pg'
        self.lineCMtype = 'pg'

        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("plot", size=(500, 500))
        self.d2 = Dock("options", size=(500,50))
        self.d3 = Dock('signal', size=(500,250))
        self.d4 = Dock('trace', size =(500, 250))
        self.d5 = Dock('max intensity', size=(500,250))
        self.d6 = Dock('mean intensity', size=(500,250))

        self.area.addDock(self.d1, 'left')
        self.area.addDock(self.d3, 'right')
        self.area.addDock(self.d2, 'bottom', self.d1)
        self.area.addDock(self.d4, 'bottom', self.d3)
        self.area.addDock(self.d5, 'below', self.d3)
        self.area.addDock(self.d6, 'below', self.d5)

        self.area.moveDock(self.d3, 'above', self.d5)

        # Set up plot widget
        self.w1 = pg.PlotWidget(title="Track plot")
        self.w1.plot()
        self.w1.setAspectLocked()
        self.w1.showGrid(x=True, y=True)
        self.w1.setXRange(-10,10)
        self.w1.setYRange(-10,10)
        self.w1.getViewBox().invertY(True)
        self.w1.setLabel('left', 'y', units ='pixels')
        self.w1.setLabel('bottom', 'x', units ='pixels')
        self.d1.addWidget(self.w1)

        # Set up options widget
        self.w2 = pg.LayoutWidget()
        self.trackSelector = pg.ComboBox()
        self.tracks= {'None':'None'}
        self.trackSelector.setItems(self.tracks)
        self.trackSelector_label = QLabel("Select Track ID")
        self.selectTrack_checkbox = CheckBox()
        self.selectTrack_checkbox.setChecked(False)
        self.line_colourMap_Box = pg.ComboBox()
        self.point_colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.line_colourMap_Box.setItems(self.colourMaps)
        self.point_colourMap_Box.setItems(self.colourMaps)

        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.plotTracks)


        self.lineCol_Box = pg.ComboBox()
        self.lineCols = {'None':'None'}
        self.lineCol_Box.setItems(self.lineCols)

        self.pointCol_Box = pg.ComboBox()
        self.pointCols = {'None':'None'}
        self.pointCol_Box.setItems(self.pointCols)
        self.lineCol_label = QLabel("Line color by")
        self.pointCol_label = QLabel("Point color by")

        self.plotSegment_checkbox = CheckBox()
        self.plotSegment_checkbox.setChecked(False)
        self.plotSegment_label = QLabel("Plot segment by frame")

        self.pointCM_button = QPushButton('Point cmap PG')
        self.pointCM_button.pressed.connect(self.setPointColourMap)

        self.lineCM_button = QPushButton('Line cmap PG')
        self.lineCM_button.pressed.connect(self.setLineColourMap)

        self.pointSize_box = pg.SpinBox(value=0.2, int=False)
        self.pointSize_box.setSingleStep(0.01)
        self.pointSize_box.setMinimum(0.00)
        self.pointSize_box.setMaximum(5)

        self.lineWidth_box = pg.SpinBox(value=2, int=True)
        self.lineWidth_box.setSingleStep(1)
        self.lineWidth_box.setMinimum(0)
        self.lineWidth_box.setMaximum(100)

        self.pointSize_label = QLabel("Point Size")
        self.lineWidth_label = QLabel("Line Width")

        self.pointSize_box.valueChanged.connect(self.plotTracks)
        self.lineWidth_box.valueChanged.connect(self.plotTracks)

        self.interpolate_checkbox = CheckBox()
        self.interpolate_checkbox.setChecked(True)
        self.interpolate_label = QLabel("Interpolate 'between' frames")

        self.interpolate_checkbox.stateChanged.connect(self.plotTracks)

        self.allFrames_checkbox = CheckBox()
        self.allFrames_checkbox.setChecked(False)
        self.allFrames_label = QLabel("Extend frames")

        self.allFrames_checkbox.stateChanged.connect(self.plotTracks)

        self.subtractBackground_checkbox = CheckBox()
        self.subtractBackground_checkbox.setChecked(False)
        self.subtractBackground_label = QLabel("Subtract background (mean ROI)")

        self.subtractBackground_checkbox.stateChanged.connect(self.plotTracks)

        self.export_button = QPushButton('Export')
        self.export_button.pressed.connect(self.exportTrack)


        #row0
        self.w2.addWidget(self.lineCol_label, row=0,col=0)
        self.w2.addWidget(self.lineCol_Box, row=0,col=1)
        self.w2.addWidget(self.pointCol_label, row=0,col=2)
        self.w2.addWidget(self.pointCol_Box, row=0,col=3)

        #row0
        self.w2.addWidget(self.plotSegment_label, row=1,col=0)
        self.w2.addWidget(self.plotSegment_checkbox, row=1,col=1)

        #row1
        self.w2.addWidget(self.lineCM_button, row=2,col=0)
        self.w2.addWidget(self.line_colourMap_Box, row=2,col=1)
        self.w2.addWidget(self.pointCM_button, row=2,col=2)
        self.w2.addWidget(self.point_colourMap_Box, row=2,col=3)
        #row2
        self.w2.addWidget(self.lineWidth_label, row=3,col=0)
        self.w2.addWidget(self.lineWidth_box, row=3,col=1)
        self.w2.addWidget(self.pointSize_label, row=3,col=2)
        self.w2.addWidget(self.pointSize_box, row=3,col=3)

        #row3
        self.w2.addWidget(self.interpolate_label, row=4,col=0)
        self.w2.addWidget(self.interpolate_checkbox, row=4,col=1)
        self.w2.addWidget(self.allFrames_label, row=4,col=2)
        self.w2.addWidget(self.allFrames_checkbox, row=4,col=3)

        #row4
        self.w2.addWidget(self.subtractBackground_label, row=5,col=0)
        self.w2.addWidget(self.subtractBackground_checkbox, row=5,col=1)
        self.w2.addWidget(self.export_button, row=5,col=3)

        #row5
        self.w2.addWidget(self.trackSelector_label, row=6,col=0)
        self.w2.addWidget(self.selectTrack_checkbox, row=6,col=1)
        self.w2.addWidget(self.trackSelector, row=6,col=2)
        self.w2.addWidget(self.plot_button, row=6,col=3)

        self.d2.addWidget(self.w2)

        #signal image view
        self.signalIMG = pg.ImageView()
        self.d3.addWidget(self.signalIMG)

        #max intensity image view
        self.maxIntensity = pg.ImageView()
        self.d5.addWidget(self.maxIntensity)

        #mean intensity image view
        self.meanIntensity = pg.ImageView()
        self.d6.addWidget(self.meanIntensity)

        # Add ROI for selecting an image region
        roiSize = 3
        self.roi = pg.ROI([(self.d/2)-(1), (self.d/2)-(1)], [roiSize, roiSize])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.signalIMG.addItem(self.roi)
        #self.roi.setZValue(10)  # make sure ROI is drawn above image

        #Trace plot
        self.tracePlot = pg.PlotWidget(title="Signal plot")
        self.tracePlot.plot()
        self.tracePlot.setLimits(xMin=0)
        self.d4.addWidget(self.tracePlot)

        self.roi.sigRegionChanged.connect(self.updateROI)
        self.pathitem = None
        self.trackDF = pd.DataFrame()

        #trace time line
        self.line = self.tracePlot.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=[0,None])
        self.signalIMG.sigTimeChanged.connect(self.updatePositionIndicator)

        self.line.sigPositionChanged.connect(self.updateTimeSlider)

    def updatePositionIndicator(self, t):
        """
        Update the position indicator line on the signal plot.

        Args:
            t (float): The new x-coordinate for the line.
        """
        self.line.setPos(t)

    def updateTimeSlider(self):
        """
        Update the current image displayed in the signal plot based on the position of the time slider.
        """
        t = int(self.line.getXPos())  # Get the current position of the time slider
        self.signalIMG.setCurrentIndex(t)  # Set the current image in the signal plot based on the slider position

    def updateTrackList(self):
        """
        Update the track list displayed in the GUI based on the data loaded into the application.
        """
        self.tracks = dictFromList(self.mainGUI.data['track_number'].to_list())  # Convert a column of track numbers into a dictionary
        self.trackSelector.setItems(self.tracks)  # Set the track list in the GUI

    def setPointColourMap(self):
        """
        Toggle between using the default point colour maps in Pyqtgraph or using the ones from Matplotlib.
        """
        if self.pointCMtype == 'pg':  # If the current point colour map is the default one from Pyqtgraph
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))  # Get a dictionary of Matplotlib colour maps available in Pyqtgraph
            self.point_colourMap_Box.setItems(self.colourMaps)  # Set the available colour maps in the dropdown menu
            self.pointCM_button.setText('Point cmap ML')  # Change the text on the toggle button to indicate that Matplotlib colour maps are currently selected
            self.pointCMtype = 'matplotlib'  # Change the point colour map type to Matplotlib
        else:  # If the current point colour map is from Matplotlib
            self.colourMaps = dictFromList(pg.colormap.listMaps())  # Get a dictionary of default Pyqtgraph colour maps
            self.point_colourMap_Box.setItems(self.colourMaps)  # Set the available colour maps in the dropdown menu
            self.pointCM_button.setText('Point cmap PG')  # Change the text on the toggle button to indicate that default Pyqtgraph colour maps are currently selected
            self.pointCMtype = 'pg'  # Change the point colour map type to the default Pyqtgraph one


    def setLineColourMap(self):
        # Check the current type of line colour map
        if self.lineCMtype == 'pg':
            # Get matplotlib colormaps and set as options in line colour map combo box
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
            self.line_colourMap_Box.setItems(self.colourMaps)
            # Change button text to show user what type of colour map they will switch to
            self.lineCM_button.setText('Line cmap ML')
            # Set lineCMtype to 'matplotlib'
            self.lineCMtype = 'matplotlib'
        else:
            # Get pyqtgraph colormaps and set as options in line colour map combo box
            self.colourMaps = dictFromList(pg.colormap.listMaps())
            self.line_colourMap_Box.setItems(self.colourMaps)
            # Change button text to show user what type of colour map they will switch to
            self.lineCM_button.setText('Line cmap PG')
            # Set lineCMtype to 'pg'
            self.lineCMtype = 'pg'


    def plotTracks(self):
        # Clear the plot widget
        self.w1.clear()

        # Check if user wants to plot a specific track or use the display track
        if self.selectTrack_checkbox.isChecked():
            trackToPlot = int(self.trackSelector.value())
        else:
            trackToPlot = int(self.mainGUI.displayTrack)

        # Get data for the selected track
        self.trackDF = self.mainGUI.data[self.mainGUI.data['track_number'] == trackToPlot]
        #print(self.trackDF)

        # save track ID
        self.trackPlotted = trackToPlot

        # Set the point and line colours to use for the track
        self.setColour()

        # Create a list of points to plot using the track data
        points = [QPointF(*xy.tolist()) for xy in np.column_stack((self.trackDF['zeroed_X'].to_list(), self.trackDF['zeroed_Y'].to_list()))]

        # Create a coloured line item to add to the plot widget
        item = ColouredLines(points, self.colours_line, self.colours_point, width_line=self.lineWidth_box.value(), size_symbol=self.pointSize_box.value())
        self.w1.addItem(item)
        self.pathitem = item

        # Crop the image stack to the points in the track
        self.cropImageStackToPoints()

        #update signal roi
        self.updateROI()


    def setColour(self):
        # Get the name of the column to use for point colour and line colour
        pointCol = self.pointCol_Box.value()
        lineCol = self.lineCol_Box.value()

        # Check the current type of point colour map
        if self.pointCMtype == 'matplotlib':
            # Get matplotlib colormap and map the colour values for the column to use for point colour
            point_cmap = pg.colormap.getFromMatplotlib(self.point_colourMap_Box.value())
        else:
            # Get pyqtgraph colormap and map the colour values for the column to use for point colour
            point_cmap = pg.colormap.get(self.point_colourMap_Box.value())

        # Scale the values in the column to use for point colour to be between 0 and 1
        point_coloursScaled= (self.trackDF[pointCol].to_numpy()) / np.max(self.trackDF[pointCol])
        # Map the scaled values to QColor objects using the chosen colormap
        self.colours_point = point_cmap.mapToQColor(point_coloursScaled)

        # Check the current type of line colour map
        if self.lineCMtype == 'matplotlib':
            # Get matplotlib colormap and map the colour values for the column to use for line colour
            line_cmap = pg.colormap.getFromMatplotlib(self.line_colourMap_Box.value())
        else:
            line_cmap = pg.colormap.get(self.line_colourMap_Box.value())

        # Scale the values in the column to use for line colour to be between 0 and 1
        line_coloursScaled= (self.trackDF[lineCol].to_numpy()) / np.max(self.trackDF[lineCol])
        # Map the scaled values to QColor objects using the chosen colormap
        self.colours_line = line_cmap.mapToQColor(line_coloursScaled)


    def cropImageStackToPoints(self):
        # Initialize an empty array to store the cropped images
        d = self.d # Desired size of cropped image
        self.frames = int(self.A_pad.shape[0])
        self.A_crop = np.zeros((self.frames,d,d))
        x_limit = int(d/2)
        y_limit = int(d/2)

        # Extract x,y,frame data for each point
        points = np.column_stack((self.trackDF['frame'].to_list(), self.trackDF['x'].to_list(), self.trackDF['y'].to_list()))

        if self.interpolate_checkbox.isChecked():
            #interpolate points for missing frames
            allFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
            xinterp = np.interp(allFrames, points[:,0], points[:,1])
            yinterp = np.interp(allFrames, points[:,0], points[:,2])

            points = np.column_stack((allFrames, xinterp, yinterp))

        if self.allFrames_checkbox.isChecked():
            #pad edges with last known position
            xinterp = np.pad(xinterp, (int(min(points[:,0])), self.frames-1 - int(max(points[:,0]))), mode='edge')
            yinterp = np.pad(yinterp, (int(min(points[:,0])), self.frames-1 - int(max(points[:,0]))), mode='edge')

            allFrames = range(0, self.frames)


            points = np.column_stack((allFrames, xinterp, yinterp))

        #store track points for exporting
        self.trackPoints = points

        # Loop through each point and extract a cropped image
        for point in points:
            minX = int(point[1]) - x_limit + d # Determine the limits of the crop
            maxX = int(point[1]) + x_limit + d
            minY = int(point[2]) - y_limit + d
            maxY = int(point[2]) + y_limit + d
            crop = self.A_pad[int(point[0]),minX:maxX,minY:maxY] # Extract the crop
            self.A_crop[int(point[0])] = crop # Store the crop in the array of cropped images

        # subtract background values (if selected)
        if self.subtractBackground_checkbox.isChecked():
            self.A_crop = self.backgroundSubtractStack(self.A_crop)

        # Display the array of cropped images in the signal image widget
        self.signalIMG.setImage(self.A_crop)
        # Display max and mean intensity projections
        self.maxIntensity_IMG = np.max(self.A_crop,axis=0)
        self.maxIntensity.setImage(self.maxIntensity_IMG)

        self.meanIntensity_IMG = np.mean(self.A_crop,axis=0)
        self.meanIntensity.setImage(self.meanIntensity_IMG)


    def exportTrack(self):
        if isinstance(self.trackPoints,(list,pd.core.series.Series,np.ndarray)):
            print('Exporting track ID: {}'.format(self.trackPlotted))
        else:
            print('First load track')
            return

        df = pd.DataFrame(self.trackPoints, columns=['frame', 'x', 'y'])
        df['interpolated'] = 1



        # columns to recaculate - #TODO!
        # 'zeroed_X', 'zeroed_Y',  'camera black estimate'
        # 'lagNumber', 'distanceFromOrigin', 'dy-dt: distance', 'nnDist_inFrame', 'n_segments', 'lag', 'meanLag', 'track_length', 'radius_gyration_scaled',
        # 'radius_gyration_scaled_nSegments','radius_gyration_scaled_trackLength', 'roi_1', 'd_squared', 'lag_squared', 'dt', 'velocity',
        # 'direction_Relative_To_Origin', 'meanVelocity', 'intensity - mean roi1',
        # 'intensity - mean roi1 and black', 'nnCountInFrame_within_3_pixels',
        # 'nnCountInFrame_within_5_pixels', 'nnCountInFrame_within_10_pixels',
        # 'nnCountInFrame_within_20_pixels', 'nnCountInFrame_within_30_pixels',
        # 'intensity_roiOnMeanXY','intensity_roiOnMeanXY - mean roi1',
        # 'intensity_roiOnMeanXY - mean roi1 and black','roi_1 smoothed',
        # 'intensity_roiOnMeanXY - smoothed roi_1',
        # 'intensity - smoothed roi_1'

        #initiate exportTrack_DF from copy of trackDF
        exportTrack_DF = self.trackDF
        #reset index
        exportTrack_DF = exportTrack_DF.reset_index(drop=True)

        #add column for interpolation tag
        exportTrack_DF['interpolated'] = 0

        # print('----   df   ----')
        # print(df.head())
        # print('---- export ----')
        # print(exportTrack_DF.head())

        # add shared track properties to df
        colsToCopy = ['track_number', 'netDispl', 'radius_gyration','asymmetry', 'skewness', 'kurtosis', 'fracDimension', 'Straight', 'Experiment', 'SVM']
        for col in colsToCopy:
            df[col] = exportTrack_DF[col][0]


        #join DFs
        exportTrack_DF = pd.concat([exportTrack_DF, df])

        #drop duplicate rows from df
        exportTrack_DF = exportTrack_DF.drop_duplicates(subset=['frame'], keep='first')

        # sort export_track_DF by frame
        exportTrack_DF = exportTrack_DF.sort_values('frame', ascending=True)

        startFrame = int(min(exportTrack_DF['frame']))
        endFrame = int(max(exportTrack_DF['frame']))+1

        # set roi_1 column
        if self.bg_flag:
            exportTrack_DF['roi_1'] = self.roi_1[startFrame:endFrame]
        else:
            exportTrack_DF['roi_1'] = 0

        # update intensity values
        if self.subtractBackground_checkbox.isChecked():
            exportTrack_DF['intensity - mean roi1'] = self.trace[startFrame:endFrame]
            exportTrack_DF['intensity'] = exportTrack_DF['intensity - mean roi1'] + exportTrack_DF['roi_1']
        else:
            exportTrack_DF['intensity'] = self.trace[startFrame:endFrame]
            exportTrack_DF['intensity - mean roi1'] = exportTrack_DF['intensity'] - exportTrack_DF['roi_1']

        # update lag number
        exportTrack_DF['lagNumber'] = np.arange(len(exportTrack_DF))

        # update zeroed X & Y
        exportTrack_DF['zeroed_X'] = exportTrack_DF['x'] - exportTrack_DF['x'][0]
        exportTrack_DF['zeroed_Y'] = exportTrack_DF['y'] - exportTrack_DF['y'][0]

        # update n_segments
        exportTrack_DF['n_segments'] = len(exportTrack_DF)

        #save df
        saveName = os.path.splitext(self.mainGUI.filename)[0]+'_trackID_{}.csv'.format(self.trackPlotted)
        exportTrack_DF.to_csv(saveName, index=None)
        print('Track file exported as: {}'.format(saveName))



    def backgroundSubtractStack(self, A):
        #get background ROI filename using tiff name
        directory, fileName = os.path.split(self.mainGUI.filename)
        roiFile = 'ROI_' + os.path.basename(fileName).split('_locs')[0] + '.txt'
        path = os.path.join(directory,roiFile)

        #load rois
        try:
            rois = open_rois(path)
        except:
            print('No background file called: {} detected'.format(path))
            return A

        #get trace for first roi in file
        self.roi_1 = rois[0].getTrace()
        # subtract mean roi from A
        A_bgSubtract = np.array([A[i] - self.roi_1[i] for i in range(len(A))])

        #remove roi
        rois[0].delete()

        # set bg flag
        self.bg_flag = True

        return A_bgSubtract

    def updateROI(self):
        # Get the ROI selection as an array of pixel values
        img = self.roi.getArrayRegion(self.A_crop, self.signalIMG.getImageItem(), axes=(1,2))
        # Calculate the average intensity of the selected ROI over time
        self.trace = np.mean(img, axis=(1,2))
        # Clear the current plot and plot the new trace
        self.tracePlot.plot(self.trace, clear=True)
        # Add a vertical line to indicate the current time point
        self.line = self.tracePlot.addLine(x=self.signalIMG.currentIndex, pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=[0,None])
        # When the line is moved, update the time slider
        self.line.sigPositionChanged.connect(self.updateTimeSlider)

    def setPadArray(self, A):
        """
        Pads the array A with zeros to avoid cropping during image registration and ROI selection.

        Args:
        - A (numpy array): the original image stack, with dimensions (frames, height, width).
        """
        self.A_pad = np.pad(A,((0,0),(self.d,self.d),(self.d,self.d)),'constant', constant_values=0)
        #self.updateROI()

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
