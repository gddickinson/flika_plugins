#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:18:37 2023

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
import flika.global_vars as g

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, save_file_gui


from .helperFunctions import *

class AllTracksPlot():
    """
    A class representing a GUI for visualizing all tracked data for intracellular Piezo1 protein using a fluorescent tag.

    Attributes:
    - mainGUI (object): the main GUI object that contains the tracked data

    """
    def __init__(self, mainGUI):
        super().__init__()
        self.mainGUI = mainGUI
        self.d = int(5) #size of cropped image
        self.A_pad = None
        self.A_crop_stack = None
        self.traceList = None

        # Set up main window
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1400, 550)
        self.win.setWindowTitle('All Tracks intensity (background subtracted)')


        ## Create docks, place them into the window one at a time.
        self.d2 = Dock("options", size=(500,50))
        self.d3 = Dock('mean intensity -bg', size=(250,250))
        self.d4 = Dock('trace', size =(750, 250))
        self.d5 = Dock('max intensity -bg', size=(250,250))

        self.d6 = Dock('mean line transect', size = (250,250))
        self.d7 = Dock('max line transect', size = (250,250))

        self.area.addDock(self.d4, 'top')
        self.area.addDock(self.d2, 'bottom')
        self.area.addDock(self.d3, 'right', self.d4)

        self.area.addDock(self.d5, 'bottom', self.d3)

        self.area.addDock(self.d6, 'right', self.d3)
        self.area.addDock(self.d7, 'right', self.d5)


        #self.area.moveDock(self.d3, 'above', self.d5)


        # Set up options widget
        self.w2 = pg.LayoutWidget()
        self.trackSelector = pg.ComboBox()
        self.tracks= {'None':'None'}
        self.trackSelector.setItems(self.tracks)
        self.trackSelector_label = QLabel("Select Track ID")
        self.selectTrack_checkbox = CheckBox()
        self.selectTrack_checkbox.setChecked(False)

        self.interpolate_checkbox = CheckBox()
        self.interpolate_checkbox.setChecked(True)
        self.interpolate_label = QLabel("Interpolate 'between' frames")

        # self.allFrames_checkbox = CheckBox()
        # self.allFrames_checkbox.setChecked(False)
        # self.allFrames_label = QLabel("Extend frames")


        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.plotTracks)

        self.export_button = QPushButton('Export traces')
        self.export_button.pressed.connect(self.exportTraces)

        self.dSize_box = pg.SpinBox(value=5, int=True)
        self.dSize_box.setSingleStep(1)
        self.dSize_box.setMinimum(1)
        self.dSize_box.setMaximum(30)
        self.dSize_label = QLabel("roi width (px)")


        #row0
        self.w2.addWidget(self.trackSelector_label, row=0,col=0)
        self.w2.addWidget(self.selectTrack_checkbox, row=0,col=1)
        self.w2.addWidget(self.trackSelector, row=0,col=2)

        #row1
        self.w2.addWidget(self.interpolate_label, row=1,col=0)
        self.w2.addWidget(self.interpolate_checkbox, row=1,col=1)
        #self.w2.addWidget(self.allFrames_label, row=1, col=2)
        #self.w2.addWidget(self.allFrames_checkbox, row=1,col=3)

        #row2
        self.w2.addWidget(self.dSize_label, row=2,col=0)
        self.w2.addWidget(self.dSize_box, row=2,col=1)
        #row3
        self.w2.addWidget(self.plot_button, row=3,col=2)
        self.w2.addWidget(self.export_button, row=3,col=3)

        self.d2.addWidget(self.w2)

        #signal image view
        self.meanIntensity = pg.ImageView()
        self.d3.addWidget(self.meanIntensity)

        #max intensity image view
        self.maxIntensity = pg.ImageView()
        self.d5.addWidget(self.maxIntensity)


        #Trace plot
        self.tracePlot = pg.PlotWidget(title="Signal plot")
        self.tracePlot.plot()
        self.tracePlot.setLimits(xMin=0)
        self.d4.addWidget(self.tracePlot)

        self.trackDF = pd.DataFrame()

        #line transect plot mean
        self.meanTransect = pg.PlotWidget(title="Mean - Line transect")
        self.d6.addWidget(self.meanTransect)

        #line transect plot max
        self.maxTransect = pg.PlotWidget(title="Max - Line transect")
        self.d7.addWidget(self.maxTransect)

        #transects
        self.ROI_mean = self.addROI(self.meanIntensity)
        self.ROI_mean.sigRegionChanged.connect(self.updateMeanTransect)

        #transects
        self.ROI_max = self.addROI(self.maxIntensity)
        self.ROI_max.sigRegionChanged.connect(self.updateMaxTransect)

    def updateTrackList(self):
        """
        Update the track list displayed in the GUI based on the data loaded into the application.
        """
        if self.mainGUI.useFilteredData == False:
            self.tracks = dictFromList(self.mainGUI.data['track_number'].to_list())  # Convert a column of track numbers into a dictionary
        else:
            self.tracks = dictFromList(self.mainGUI.filteredData['track_number'].to_list())  # Convert a column of track numbers into a dictionary


        self.trackSelector.setItems(self.tracks)  # Set the track list in the GUI

    def cropImageStackToPoints(self):
        # Check if user wants to plot a specific track or use the display track
        if self.selectTrack_checkbox.isChecked():
            self.trackList  = [int(self.trackSelector.value())]
        else:

            self.trackList = [self.trackSelector.itemText(i) for i in range(self.trackSelector.count())]


        self.traceList = []
        self.timeList = []

        # Initialize an empty array to store the cropped images
        self.d = int(self.dSize_box.value()) # Desired size of cropped image

        #pad array image
        self.setPadArray()

        self.frames = int(self.A_pad.shape[0])

        self.A_crop_stack = np.zeros((len(self.trackList),self.frames,self.d,self.d))
        x_limit = int(self.d/2)
        y_limit = int(self.d/2)




        for i, track_number in enumerate(self.trackList):
            #temp array for crop
            A_crop = np.zeros((self.frames,self.d,self.d))
            #get track data
            trackDF = self.mainGUI.data[self.mainGUI.data['track_number'] == int(track_number)]

            # Extract x,y,frame data for each point
            points = np.column_stack((trackDF['frame'].to_list(), trackDF['x'].to_list(), trackDF['y'].to_list()))

            if self.interpolate_checkbox.isChecked():
                #interpolate points for missing frames
                allFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
                xinterp = np.interp(allFrames, points[:,0], points[:,1])
                yinterp = np.interp(allFrames, points[:,0], points[:,2])

                points = np.column_stack((allFrames, xinterp, yinterp))


            # Loop through each point and extract a cropped image
            for point in points:
                minX = round(point[1]) - x_limit + self.d # Determine the limits of the crop including padding
                maxX = round(point[1]) + x_limit + self.d
                minY = round(point[2]) - y_limit + self.d
                maxY = round(point[2]) + y_limit + self.d

                if (self.d % 2) == 0:
                    crop = self.A_pad[int(point[0]),minX:maxX,minY:maxY] - np.min(self.A[int(point[0])])# Extract the crop
                else:
                    crop = self.A_pad[int(point[0]),minX-1:maxX,minY-1:maxY] - np.min(self.A[int(point[0])])# Extract the crop

                A_crop[int(point[0])] = crop

            self.A_crop_stack[i] = A_crop # Store the crop in the array of cropped images

            A_crop[A_crop==0] = np.nan
            trace = np.mean(A_crop, axis=(1,2))

            #extend time series to cover entire recording
            timeSeries = range(0,self.frames)
            times = trackDF['frame'].to_list()

            #if not interpolating points add zeros to to missing time points in traces
            if self.interpolate_checkbox.isChecked() == False:
                trace = trace.tolist()
                missingTrace = []
                for i in timeSeries:
                    if i not in times:
                        missingTrace.append(0)
                    else:
                        missingTrace.append(trace[0])
                        trace.pop(0)
                trace = np.array(missingTrace)

            #add trace to tracelist for plotting and export
            self.traceList.append(trace)
            #ensure all time points accounted for
            missingTimes = [x if x in times else np.nan for x in timeSeries]
            #add time to time list for plotting and export
            self.timeList.append(missingTimes)

        # Display max and mean intensity projections - ignoring zero values
        #convert zero to nan
        self.A_crop_stack[self.A_crop_stack== 0] = np.nan
        #max - using nanmax to ignore nan
        self.maxIntensity_IMG = np.nanmax(self.A_crop_stack,axis=(0,1))
        self.maxIntensity.setImage(self.maxIntensity_IMG)
        #mean - using nanmean to ignore nan
        self.meanIntensity_IMG = np.nanmean(self.A_crop_stack,axis=(0,1))
        self.meanIntensity.setImage(self.meanIntensity_IMG)

        #update transects
        self.updateMeanTransect
        self.updateMaxTransect

    def setPadArray(self):
        """
        Pads the array A with zeros to avoid cropping during image registration and ROI selection.

        Args:
        - A (numpy array): the original image stack, with dimensions (frames, height, width).
        """
        self.A = self.mainGUI.plotWindow.imageArray()
        self.A_pad = np.pad(self.A,((0,0),(self.d,self.d),(self.d,self.d)),'constant', constant_values=0)



    def plotTracks(self):
        #check number of tracks to plot
        if self.trackSelector.count() > 2000:
            warningBox = g.messageBox('Warning!','More than 2000 tracks to plot. Continue?',buttons=QMessageBox.Yes|QMessageBox.No)
            if warningBox == 65536:
                return

        # Clear the plot widget
        self.tracePlot.clear()

        # Crop the image stack to the points in the track
        self.cropImageStackToPoints()

        #add signal traces for all tracks to plot
        for i, trace in enumerate(self.traceList):
            curve = pg.PlotCurveItem()
            curve.setData(x=self.timeList[i],y=trace)
            self.tracePlot.addItem(curve)

        return


    def exportTraces(self):
        #get save path
        fileName = QFileDialog.getSaveFileName(None, 'Save File', os.path.dirname(self.mainGUI.filename),'*.csv')[0]
        print(fileName)

        exportTraceList = []

        #add nan to traces for missing frames
        for i,trace in enumerate(self.traceList):
            startPad =  min(self.timeList[i])
            endPad = self.frames - max(self.timeList[i])

            #paddedTrace = np.pad(trace, (startPad,endPad),mode='constant',constant_values=(np.nan))
            paddedTrace = trace
            exportTraceList.append(paddedTrace)

        exportDF = pd.DataFrame(exportTraceList).T
        exportDF.columns = self.trackList


        # #test
        # #make df to save
        # d = {'track_number': self.trackList, 'frame':self.timeList,'intensity':self.traceList}
        # exportDF = pd.DataFrame(data=d)

        #export traces to file
        exportDF.to_csv(fileName)

        # display save message
        g.m.statusBar().showMessage('trace exported to {}'.format(fileName))
        print('trace exported to {}'.format(fileName))
        return

    def addROI(self, win):
        # Custom ROI for selecting an image region
        roi = pg.ROI([0, 0], [self.d, self.d])
        roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        roi.addRotateFreeHandle([1, 1], [0.5, 0.5])
        win.addItem(roi)
        #roi.setZValue(10)  # make sure ROI is drawn above image
        return roi

    def updateMeanTransect(self):
        selected = self.ROI_mean.getArrayRegion(self.meanIntensity_IMG, self.meanIntensity.imageItem)
        self.meanTransect.plot(selected.mean(axis=1), clear=True)

    def updateMaxTransect(self):
        selected = self.ROI_max.getArrayRegion(self.maxIntensity_IMG, self.maxIntensity.imageItem)
        self.maxTransect.plot(selected.mean(axis=1), clear=True)

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
