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
import os
import math
import sys
from scipy.optimize import curve_fit
import time

# determine which version of flika to use
flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

# enable Numba for performance
import numba
pg.setConfigOption('useNumba', True)

# import pandas and matplotlib for generating graphs
import pandas as pd
from matplotlib import pyplot as plt

# import pyqtgraph modules for dockable windows
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# import custom module to join tracks
from .joinTracks import JoinTracks

# function to create a dictionary from a list
def dictFromList(l):
    # ensure strings
    l = [str(x) for x in l]
    # Create a zip object from two lists
    zipbObj = zip(l, l)
    return dict(zipbObj)

# exponential decay functions for curve fitting
def exp_dec(x, A1, tau):
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x, A1, tau1, tau2):
    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x, A1, A2, tau1, tau2, tau3):
    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)

# function to prompt user to select a file for opening
def open_file_gui(prompt="Open File", directory=None, filetypes=''):
    """ File dialog for opening an existing file, isolated to handle tuple/string return value
    
    Args:
        prompt (str): string to display at the top of the window
        directory (str): initial directory to open
        filetypes (str): argument for filtering file types separated by ;; (*.png) or (Images *.png);;(Other *.*)
    
    Returns:
        str: the file (path+file+extension) selected, or None
    """
    filename = None
    if directory is None:
        filename = g.settings['filename']
        try:
            directory = os.path.dirname(filename)
        except:
            directory = None
    if directory is None or filename is None:
        filename = QFileDialog.getOpenFileName(g.m, prompt, '', filetypes)
    else:
        filename = QFileDialog.getOpenFileName(g.m, prompt, filename, filetypes)
    if isinstance(filename, tuple):
        filename, ext = filename
        if ext and '.' not in filename:
            filename += '.' + ext.rsplit('.')[-1]
    if filename is None or str(filename) == '':
        g.m.statusBar().showMessage('No File Selected')
        return None
    else:
        return str(filename)

class FileSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a file to save.  Once you do, it sets self.filename and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*', mainGUI=None):
        QWidget.__init__(self)
        
        self.mainGUI = mainGUI
        
        self.button=QPushButton('Load Data')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.filename = ''
        self.columns = []
        self.pixelSize = self.mainGUI.pixelSize_selector.value()    #nanometers

        
    def buttonclicked(self):
        if g.win == None:
            g.alert('Load tiff stack and set as current window first')
            return
        prompt = 'testing fileSelector'
        self.filename = open_file_gui(prompt, filetypes=self.filetypes)
        self.label.setText('...'+os.path.split(self.filename)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filename

    def setValue(self, filename):
        self.filename = str(filename)
        self.label.setText('...' + os.path.split(self.filename)[-1][-20:])    

    
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

        #transexts
        self.ROI_mean = self.addROI(self.meanIntensity)
        self.ROI_mean.sigRegionChanged.connect(self.updateMeanTransect)
        
        #transexts
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
        
        self.allFrames_checkbox = CheckBox()
        self.allFrames_checkbox.setChecked(False)
        self.allFrames_label = QLabel("Extend frames")        
        
        

        #row0
        self.w2.addWidget(self.lineCol_label, row=0,col=0)   
        self.w2.addWidget(self.lineCol_Box, row=0,col=1)   
        self.w2.addWidget(self.pointCol_label, row=0,col=2)         
        self.w2.addWidget(self.pointCol_Box, row=0,col=3)        
        #row1
        self.w2.addWidget(self.lineCM_button, row=1,col=0)   
        self.w2.addWidget(self.line_colourMap_Box, row=1,col=1)   
        self.w2.addWidget(self.pointCM_button, row=1,col=2)         
        self.w2.addWidget(self.point_colourMap_Box, row=1,col=3)                  
        #row2
        self.w2.addWidget(self.lineWidth_label, row=2,col=0)   
        self.w2.addWidget(self.lineWidth_box, row=2,col=1)   
        self.w2.addWidget(self.pointSize_label, row=2,col=2)         
        self.w2.addWidget(self.pointSize_box, row=2,col=3) 

        #row3
        self.w2.addWidget(self.interpolate_label, row=3,col=0)
        self.w2.addWidget(self.interpolate_checkbox, row=3,col=1) 
        self.w2.addWidget(self.allFrames_label, row=3,col=2)
        self.w2.addWidget(self.allFrames_checkbox, row=3,col=3)         
                        
        #row4
        self.w2.addWidget(self.trackSelector_label, row=4,col=0)
        self.w2.addWidget(self.selectTrack_checkbox, row=4,col=1)        
        self.w2.addWidget(self.trackSelector, row=4,col=2)
        self.w2.addWidget(self.plot_button, row=4,col=3)  
        
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
        self.roi = pg.ROI([0, 0], [5, 5])
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

        
        # Loop through each point and extract a cropped image
        for point in points:
            minX = int(point[1]) - x_limit + d # Determine the limits of the crop
            maxX = int(point[1]) + x_limit + d
            minY = int(point[2]) - y_limit + d
            maxY = int(point[2]) + y_limit + d
            crop = self.A_pad[int(point[0]),minX:maxX,minY:maxY] # Extract the crop
            self.A_crop[int(point[0])] = crop # Store the crop in the array of cropped images
            
        # Display the array of cropped images in the signal image widget
        self.signalIMG.setImage(self.A_crop)         
        # Display max and mean intensity projections
        self.maxIntensity_IMG = np.max(self.A_crop,axis=0)
        self.maxIntensity.setImage(self.maxIntensity_IMG) 
        
        self.meanIntensity_IMG = np.mean(self.A_crop,axis=0)
        self.meanIntensity.setImage(self.meanIntensity_IMG)       
        
           
    def updateROI(self):
        # Get the ROI selection as an array of pixel values
        img = self.roi.getArrayRegion(self.A_crop, self.signalIMG.getImageItem(), axes=(1,2))
        # Calculate the average intensity of the selected ROI over time
        trace = np.mean(img, axis=(1,2))
        # Clear the current plot and plot the new trace
        self.tracePlot.plot(trace, clear=True) 
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

    
class FlowerPlotWindow():
    """
    This class creates a new window for the flower plot visualization.
    It initializes a new window using the pyqtgraph library, sets the size
    and title of the window, and assigns a reference to the main GUI window
    as an attribute. 
    
    Args:
        mainGUI: A reference to the main GUI window.
    """
    def __init__(self, mainGUI):
        super().__init__()  

        self.mainGUI = mainGUI

        # Setup window
        self.win = pg.GraphicsWindow()
        self.win.resize(500, 500)
        self.win.setWindowTitle('Flower Plot')

        # Add plot to window and set attributes
        self.plt = self.win.addPlot(title='plot')  
        self.plt.setAspectLocked()
        self.plt.showGrid(x=True, y=True)
        self.plt.setXRange(-10,10)
        self.plt.setYRange(-10,10)
        self.plt.getViewBox().invertY(True)        
        
        # Set labels for axes
        self.plt.setLabel('left', 'y', units ='pixels')
        self.plt.setLabel('bottom', 'x', units ='pixels') 
        
        # List to store plot items representing tracks
        self.pathitems = []

       
    def clearTracks(self):
        # Remove all plot items representing tracks
        if self.win is not None and not self.win.closed:
            for pathitem in self.pathitems:
                self.plt.removeItem(pathitem)
        self.pathitems = [] 

    def show(self):
        # Show the window
        self.win.show()
    
    def close(self):
        # Close the window
        self.win.close()

    def hide(self):
        # Hide the window
        self.win.hide()
    
class DiffusionPlotWindow():
    """
    A class for creating the diffusion analysis window .
    
    Attributes:
    -----------
    mainGUI : MainGUI object
        The parent object that instantiated this class.
    win : QMainWindow
        The main window for the diffusion analysis.
    area : DockArea
        The area of the window where dock widgets can be placed.
    d1 : Dock
        The dock widget for the plot options.
    d2 : Dock
        The dock widget for the distance plot.
    d3 : Dock
        The dock widget for the histogram options.
    d4 : Dock
        The dock widget for the lag histogram.
    d5 : Dock
        The dock widget for the CDF options.
    d6 : Dock
        The dock widget for the CDF plot.
    """
    def __init__(self, mainGUI):
        super().__init__()  

        self.mainGUI = mainGUI
        
        self.win =QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1200,500)
        self.win.setWindowTitle('Diffusion Analysis')
        
        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("plot options", size=(400, 100))
        self.d2 = Dock("distance plot", size=(400,400))
        self.d3 = Dock("histo options", size=(400,100))
        self.d4 = Dock("lag histogram", size=(400,400))
        self.d5 = Dock("CDF options", size=(400,100))        
        self.d6 = Dock("CDF", size=(400,400))       
        
        self.area.addDock(self.d1, 'left') 
        self.area.addDock(self.d3, 'right', self.d1)       
        self.area.addDock(self.d2, 'bottom', self.d1)     
        self.area.addDock(self.d4, 'bottom', self.d3) 
        
        self.area.addDock(self.d5, 'right', self.d3)     
        self.area.addDock(self.d6, 'right', self.d4)   
    
        #### DISTANCE SCATTER PLOT
        # Create a LayoutWidget for the options for the distance scatter plot.
        self.w1 = pg.LayoutWidget()
    
        # Create a ComboBox for selecting the type of plot.
        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes= {'scatter':'scatter','line (slow with many tracks!)':'line'}
        self.plotTypeSelector.setItems(self.plotTypes)  
        self.selectorLabel = QLabel("Plot type")  

        # Create a SpinBox for selecting the size of the points.
        self.pointSize_selector = pg.SpinBox(value=3, int=True)
        self.pointSize_selector.setSingleStep(1)       
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10) 
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        self.pointSizeLabel = QLabel("Point size") 
        
        # Create a button for updating the plot.
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)
        
        # Add the widgets to the LayoutWidget.
        self.w1.addWidget(self.plotTypeSelector, row=0,col=1)
        self.w1.addWidget(self.selectorLabel, row=0,col=0) 
        self.w1.addWidget(self.pointSizeLabel, row=1, col=0)         
        self.w1.addWidget(self.pointSize_selector, row=1, col=1)        
        self.w1.addWidget(self.plot_button, row=2, col=1)         
        
        # Add the LayoutWidget to the dock.
        self.d1.addWidget(self.w1)    

        # Create a PlotWidget for the distance plot.
        self.w3 = pg.PlotWidget(title="square of distance from origin")
        self.w3.plot()
        self.w3.setLabel('left', 'd squared', units ='')
        self.w3.setLabel('bottom', 'lags', units ='')  
        self.d2.addWidget(self.w3)  

        
        #### LAG HISTOGRAM
        # Create a LayoutWidget for the options for the lag histogram.
        self.w2 = pg.LayoutWidget()             
        
        # Create a button for updating the histogram.                  
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)

        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(1)       
        self.histoBin_selector.setMinimum(1)
        self.histoBin_selector.setMaximum(100000) 
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        
        self.histoBin_label = QLabel('# of bins')
        
        self.w2.addWidget(self.histoBin_selector, row=0, col=1)
        self.w2.addWidget(self.histoBin_label, row=0, col=0)        
        self.w2.addWidget(self.histo_button, row=1, col=1)         
        
        self.d3.addWidget(self.w2)       
    
        self.w4 = pg.PlotWidget(title="Distribution of mean SLDs")
        self.w4.plot()
        self.w4.setLabel('left', 'Count', units ='')
        self.w4.setLabel('bottom', 'mean sld per track', units ='nm')          
        self.w4.getAxis('bottom').enableAutoSIPrefix(False)         
        self.d4.addWidget(self.w4)  
   
        ### CDF
        # Create a new layout widget for the CDF plot and its controls
        self.w5 = pg.LayoutWidget() 
        
        # Add a button for updating the CDF plot
        self.cdf_button = QPushButton('Plot CDF')
        self.cdf_button.pressed.connect(self.updateCDF)
        
        # Add a spinbox for selecting the number of bins in the CDF plot
        self.cdfBin_selector = pg.SpinBox(value=100, int=True)
        self.cdfBin_selector.setSingleStep(1)       
        self.cdfBin_selector.setMinimum(1)
        self.cdfBin_selector.setMaximum(100000) 
        self.cdfBin_selector.sigValueChanged.connect(self.updateCDF)
                
        # Add a label for the bin selector
        self.cdfBin_label = QLabel('# of bins')
        
        # Add buttons for fitting one, two, or three-component exponential curves to the CDF
        self.fit_exp_dec_1_button = QPushButton('Fit 1 component exponential')
        self.fit_exp_dec_1_button.pressed.connect(self.fit_exp_dec_1)
        self.fit_exp_dec_2_button = QPushButton('Fit 2 component exponential')
        self.fit_exp_dec_2_button.pressed.connect(self.fit_exp_dec_2)
        self.fit_exp_dec_3_button = QPushButton('Fit 3 component exponential')
        self.fit_exp_dec_3_button.pressed.connect(self.fit_exp_dec_3)
        
        # Add the controls to the layout widget
        self.w5.addWidget(self.cdfBin_selector, row=0, col=1)
        self.w5.addWidget(self.cdfBin_label, row=0, col=0)        
        self.w5.addWidget(self.cdf_button, row=1, col=1) 
        self.w5.addWidget(self.fit_exp_dec_1_button , row=2, col=1) 
        self.w5.addWidget(self.fit_exp_dec_2_button , row=3, col=1) 
        self.w5.addWidget(self.fit_exp_dec_3_button , row=4, col=1) 
        
        # Add the layout widget to the CDF plot dock
        self.d5.addWidget(self.w5) 
        
        # Create a new plot widget for the CDF
        self.w6 = pg.PlotWidget(title="CDF")
        self.w6.plot()
        
        # Set the axis labels and disable auto SI prefix for the x-axis
        self.w6.setLabel('left', 'CDF', units ='')
        self.w6.setLabel('bottom', 'mean sld^2', units ='micron^2')          
        self.w6.getAxis('bottom').enableAutoSIPrefix(False) 
        
        # Add the CDF plot widget to its dock
        self.d6.addWidget(self.w6)   
        
        # Add a legend to the CDF plot widget
        self.cdf_legend = self.w6.plotItem.addLegend()
        
        # Initialize the curve objects for the exponential fits
        self.exp_dec_1_curve = None
        self.exp_dec_2_curve = None
        self.exp_dec_3_curve = None 
    
    def updatePlot(self):
        # Clear plot before updating
        self.w3.clear()
        
        # Check plot type and get data accordingly
        if self.plotTypeSelector.value() == 'line':
            # Group data by track number to plot lines
            if self.mainGUI.useFilteredData == False:            
                df = self.mainGUI.data
            else:           
                df = self.mainGUI.filteredData
            
            # Get lag number and d squared for each track
            x = df.groupby('track_number')['lagNumber'].apply(list)
            y = df.groupby('track_number')['d_squared'].apply(list)        
    
            # Get unique track IDs
            trackID_list = np.unique(df['track_number']).astype(np.int)
    
            # Plot each track as a line
            for txid in trackID_list:
                path = pg.arrayToQPath(np.array(x[txid]),np.array(y[txid]))
                item = pg.QtGui.QGraphicsPathItem(path)
                item.setPen(pg.mkPen('w'))                
                self.w3.addItem(item)
                
        elif self.plotTypeSelector.value() == 'scatter':
            # Get x and y data for scatter plot
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data['lagNumber'].to_numpy()
                y = self.mainGUI.data['d_squared'].to_numpy() 
            else:
                x = self.mainGUI.filteredData['lagNumber'].to_numpy()
                y = self.mainGUI.filteredData['d_squared'].to_numpy() 
            
            # Plot scatter plot with points
            self.w3.plot(x, y,
                         pen=None,
                         symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),                                      
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),
                         symbolSize=self.pointSize_selector.value())    
            
    
        return
 
    def updateHisto(self):
        # Clear the histogram plot
        self.w4.clear()
        
        # Check if filtered data is being used or not
        if self.mainGUI.useFilteredData == False:                
            plotDF = self.mainGUI.data.groupby('track_number').mean()                
        else:                
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean() 
            
        # Calculate the mean lag in microns
        meanLag = plotDF['lag'] * self.mainGUI.pixelSize_selector.value()
    
        # Set the start and end of the histogram bins and the number of bins to use
        start = 0
        end = np.max(meanLag)
        n = self.histoBin_selector.value()
    
        # Create the histogram data
        y, x = np.histogram(meanLag, bins=np.linspace(start, end, n))
        
        # Plot the histogram data
        self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150), clear=True)
        
        # Return
        return

    def updateCDF(self):
        # Clear the widget to start with a blank slate
        self.w6.clear()
                 
        # Check whether to use the filtered data or not
        if self.mainGUI.useFilteredData == False:
            # Use the original data
            plotDF = self.mainGUI.data.groupby('track_number').mean()
        else:
            # Use the filtered data
            plotDF = self.mainGUI.filteredData.groupby('track_number').mean() 
            
        # Calculate the squared lag distances in microns
        self.squared_SLDs = plotDF['lag_squared'] * np.square(self.mainGUI.pixelSize_selector.value()/1000)
    
        # Set the start and end points of the histogram, and the number of bins
        start=0
        end=np.max(self.squared_SLDs)
        n=self.cdfBin_selector.value()
    
        # Calculate the histogram using numpy
        count, bins_count = np.histogram(self.squared_SLDs, bins=np.linspace(start, end, n)) 
    
        # Calculate the probability density function and the cumulative distribution function
        pdf = count / sum(count)
        self.cdf_y = np.cumsum(pdf)        
        self.cdf_x = bins_count[1:]
            
        # Get the maximum number of lags for normalization
        self.nlags = np.max(self.cdf_y)
            
        # Plot the CDF
        self.w6.plot(self.cdf_x, self.cdf_y, brush=(0,0,255,150), clear=True) 
            
        # Add movable dashed lines to select a range on the CDF
        self.left_bound_line = self.w6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))
        self.right_bound_line = self.w6.addLine(x=np.max(self.squared_SLDs), pen=pg.mkPen('y', style=Qt.DashLine), movable=True, bounds=(start, end))
            
        return

    def fit_exp_dec_1(self):
        # Remove any existing fitted curve and its legend item
        if self.exp_dec_1_curve is not None:
            self.w6.removeItem(self.exp_dec_1_curve)
            self.cdf_legend.removeItem(self.exp_dec_1_curve.name())
    
        # Get the left and right bounds of the fitting range
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
    
        # Get the x and y data for the CDF plot
        xdata = self.cdf_x
        ydata = self.cdf_y 
    
        # Select the data points within the fitting range
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]
    
        # Fit an exponential decay function to the selected data
        popt, pcov = curve_fit(exp_dec, xfit, ydata[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
        tau_fit = popt[1]
        D_fit = self.tau_to_D(tau_fit)
    
        # Print the fitted diffusion coefficient
        print('D = {0:.4g} um^2 s^-1'.format(D_fit))
    
        # Generate the fitted curve and add it to the plot
        yfit = exp_dec(xfit, *popt)
        self.exp_dec_1_curve = self.w6.plot(xfit, yfit, pen='g', name=' Fit. D = {0:.4g} um^2 s^-1'.format(D_fit))
        
        # TODO: Residual plot implementation commented out.
        # Generate a residual plot (optional)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

    def fit_exp_dec_2(self):
        if self.exp_dec_2_curve is not None:
            self.w6.removeItem(self.exp_dec_2_curve)
            self.cdf_legend.removeItem(self.exp_dec_2_curve.name())
    
        # Determine the bounds for the fitting based on the position of the two vertical lines
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
    
        # Get the data to fit from the CDF plot
        xdata = self.cdf_x
        ydata = self.cdf_y 
    
        # Mask the data to fit within the bounds
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]
    
        # Perform the curve fitting using the double-exponential decay function (exp_dec_2)
        # and the masked data
        popt, pcov = curve_fit(exp_dec_2, xfit, ydata[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))
    
        # Extract the fitted parameters
        A1 = popt[0]
        A2 = -1 - A1
        tau1_fit = popt[1]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[2]
        D2_fit = self.tau_to_D(tau2_fit)
    
        # Print the fitted diffusion coefficients and amplitudes
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2. A1={2:.2g} A2={3:.2g}'.format(D1_fit, D2_fit, A1, A2)
        print(msg)
    
        # Calculate the fit line and plot it on the CDF plot
        yfit = exp_dec_2(xfit, *popt)
        self.exp_dec_2_curve = self.w6.plot(xfit, yfit, pen='r', name=' Fit. '+msg)
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))
        
    def fit_exp_dec_3(self):
        # Check if an existing plot for the fit exists and remove it
        if self.exp_dec_3_curve is not None:
            self.w6.removeItem(self.exp_dec_3_curve)
            self.cdf_legend.removeItem(self.exp_dec_3_curve.name())
    
        # Get the left and right bounds for the fit from the GUI sliders
        left_bound = np.min([self.left_bound_line.value(), self.right_bound_line.value()])
        right_bound = np.max([self.left_bound_line.value(), self.right_bound_line.value()])
    
        # Get the x and y data for the CDF plot from the GUI
        xdata = self.cdf_x
        ydata = self.cdf_y 
    
        # Create a mask to only fit data within the selected bounds
        x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
        xfit = xdata[x_fit_mask]        
    
        # Fit the data using the three-exponential decay function and bounds on the parameters
        popt, pcov = curve_fit(exp_dec_3, xfit, ydata[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))
    
        # Extract the fitted parameters and compute diffusion coefficients
        A1 = popt[0]
        A2 = popt[1]
        A3 = -1 - A1 - A2
        tau1_fit = popt[2]
        D1_fit = self.tau_to_D(tau1_fit)
        tau2_fit = popt[3]
        D2_fit = self.tau_to_D(tau2_fit)
        tau3_fit = popt[4]
        D3_fit = self.tau_to_D(tau3_fit)
    
        # Create a string summarizing the fit parameters
        msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2, D3 = {2:.4g} um2/2. A1={3:.2g} A2={4:.2g}, A3={5:.2g}'.format(D1_fit, D2_fit, D3_fit, A1, A2, A3)
    
        # Generate the fitted curve and add it to the plot with a label containing the fit parameters
        yfit = exp_dec_3(xfit, *popt)
        self.exp_dec_3_curve = self.w6.plot(xfit, yfit, pen='y', name=' Fit. '+msg)
    
        # Uncomment these lines to generate a plot of the residuals
        # residual_plot = pg.plot(title='Single exponential residual')
        # residual_plot.plot(xfit, np.abs(ydata[x_fit_mask] - yfit))

 
    def tau_to_D(self, tau):
        """ 
        tau = 4Dt
        tau is decay constant of exponential fit
        D is diffusion coefficient
        t is duration of one lag (exposure time) in seconds
        """
        t = (self.mainGUI.frameLength_selector.value()/1000) * self.nlags
        D = tau / (4 * t)
        return D


    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()

class TrackWindow(BaseProcess):
    def __init__(self, mainGUI):
        super().__init__()

        self.mainGUI = mainGUI

        # Setup window
        self.win = pg.GraphicsWindow()  # Create a PyqtGraph GraphicsWindow object
        self.win.resize(600, 800)  # Set the size of the window
        self.win.setWindowTitle('Track Display - press "t" to add track')  # Set the title of the window

        # Add widgets to the window
        self.label = pg.LabelItem(justify='center')  # Create a PyqtGraph LabelItem object for displaying text
        self.win.addItem(self.label)  # Add the label to the window
        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying intensity data
        self.plt1 = self.win.addPlot(title='intensity')  # Create a PyqtGraph PlotItem object
        self.plt1.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        # Create a plot for displaying the track
        self.plt3 = self.win.addPlot(title='track')  # Create a PyqtGraph PlotItem object
        self.plt3.setAspectLocked()  # Keep the aspect ratio of the plot fixed
        self.plt3.showGrid(x=True, y=True)  # Show a grid on the plot
        self.plt3.setXRange(-5,5)  # Set the x-axis limits of the plot
        self.plt3.setYRange(-5,5)  # Set the y-axis limits of the plot
        self.plt3.getViewBox().invertY(True)  # Invert the y-axis of the plot

        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying the distance from the origin
        self.plt2 = self.win.addPlot(title='distance from origin')  # Create a PyqtGraph PlotItem object
        self.plt2.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

# =============================================================================
#         # Create a plot for displaying the polar coordinates of the track (direction and velocity)
#         self.plt4 = self.win.addPlot(title='polar (direction and velocity)')  # Create a PyqtGraph PlotItem object
#         self.plt4.getViewBox().invertY(True)  # Invert the y-axis of the plot
#         self.plt4.setAspectLocked()  # Keep the aspect ratio of the plot fixed
#         self.plt4.setXRange(-4,4)  # Set the x-axis limits of the plot
#         self.plt4.setYRange(-4,4)  # Set the y-axis limits of the plot
#         self.plt4.hideAxis('bottom')  # Hide the x-axis of the plot
#         self.plt4.hideAxis('left')  # Hide the y-axis of the plot
# =============================================================================

        # Create a plot for displaying the nearest neighbour counts
        self.plt4 = self.win.addPlot(title='nearest neighbobur count')  # Create a PyqtGraph PlotItem object
        self.plt4.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        self.win.nextRow()  # Move to the next row of the window for adding more widgets

        # Create a plot for displaying the instantaneous velocity of the track
        self.plt5 = self.win.addPlot(title='instantaneous velocity')  # Create a PyqtGraph PlotItem object
        self.plt5.getAxis('left').enableAutoSIPrefix(False)  # Disable auto scientific notation for the y-axis

        # Create a plot for displaying the direction relative to the origin of the
        self.plt6 = self.win.addPlot(title='direction relative to origin')
        self.plt6.getAxis('left').enableAutoSIPrefix(False)       

        self.win.nextRow()  
        
        # Set plot labels for each of the six plots
        self.plt1.setLabel('left', 'Intensity', units ='Arbitary')
        self.plt1.setLabel('bottom', 'Time', units ='Frames')        
        
        self.plt2.setLabel('left', 'Distance', units ='pixels')
        self.plt2.setLabel('bottom', 'Time', units ='Frames') 
        
        self.plt3.setLabel('left', 'y', units ='pixels')
        self.plt3.setLabel('bottom', 'x', units ='pixels') 
        
        self.plt4.setLabel('left', '# of neighbours', units ='count')
        self.plt4.setLabel('bottom', 'Time', units ='Frames')         
        
        self.plt5.setLabel('left', 'velocity', units ='pixels/frame')
        self.plt5.setLabel('bottom', 'Time', units ='Frames')      
        
        self.plt6.setLabel('left', 'direction moved', units ='degrees')
        self.plt6.setLabel('bottom', 'Time', units ='Frames') 

        self.win.nextRow()
        
        # Add a button to toggle the position indicator display
        self.optionsPanel = QGraphicsProxyWidget()                  
        self.positionIndicator_button = QPushButton('Show position info')
        self.positionIndicator_button.pressed.connect(self.togglePoistionIndicator)        
        self.optionsPanel.setWidget(self.positionIndicator_button)     
        
        # Create a ComboBox for selecting the nearest neighbour count.
        self.optionsPanel2 = QGraphicsProxyWidget()      
        self.plotCountSelector = pg.ComboBox()
        self.countTypes= {'NN radius: 3':'3','NN radius: 5':'5','NN radius: 10':'10','NN radius: 20':'20','NN radius: 30':'30'}
        self.plotCountSelector.setItems(self.countTypes)  
        self.countLabel = QLabel("NN count radius") 
        self.optionsPanel2.setWidget(self.plotCountSelector)       
        
        #add options panel to win
        self.win.addItem(self.optionsPanel)
        self.win.addItem(self.optionsPanel2)        
       
       
        #status flags     
        self.showPositionIndicators = False
        self.plotsInitiated = False   
        
        
        self.r = None

    def update(self, time, intensity, distance, zeroed_X, zeroed_Y, dydt, direction, velocity, ID, count_3, count_5, count_10, count_20, count_30):  
        
        ##Update track ID
        self.label.setText("<span style='font-size: 16pt'>track ID={}".format(ID))        
        
        #update intensity plot
        self.plt1.plot(time, intensity, stepMode=False, brush=(0,0,255,150), clear=True) 
        
        #update distance plot        
        self.plt2.plot(time, distance, stepMode=False, brush=(0,0,255,150), clear=True)
        
        #update position relative to 0 plot          
        self.plt3.plot(zeroed_X, zeroed_Y, stepMode=False, brush=(0,0,255,150), clear=True) 
        
# =============================================================================
#         #update polar
#         self.updatePolarPlot(direction,velocity)
# =============================================================================
        #update nearest neighbour count
        
        if self.plotCountSelector.value() == '3':
            countRadius = count_3
        elif self.plotCountSelector.value() == '5':
            countRadius = count_5
        elif self.plotCountSelector.value() == '10':
            countRadius = count_10            
        elif self.plotCountSelector.value() == '20':
            countRadius = count_20            
        elif self.plotCountSelector.value() == '30':
            countRadius = count_30            
            
                   
        self.plt4.plot(time, countRadius, stepMode=False, brush=(0,0,255,150), clear=True)         
        
        #update dydt
        self.plt5.plot(time, velocity, stepMode=False, brush=(0,0,255,150), clear=True)
        
        #update direction
        self.plt6.plot(time, direction, stepMode=False, brush=(0,0,255,150), clear=True)       
            
        # if self.autoscaleX:
        #     self.plt1.setXRange(np.min(x),np.max(x),padding=0)
        # if self.autoscaleY:
        #     self.plt1.setYRange(np.min(y),np.max(y),padding=0)
        
        # If enabled, show the position indicators
        if self.showPositionIndicators:
            # Add vertical lines to each plot that indicate the current time
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt4_line = self.plt4.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)           
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)           
                            
            # Connect the signal that is emitted when the current time changes to the function
            # that updates the position indicators
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)   
        
        # Store the data as a dictionary with time as keys and position as values
        keys = time
        values = zip(zeroed_X, zeroed_Y)
        self.data = dict(zip(keys,values))
        
        # Reset the zoom of the polar plot
        self.r = None

# =============================================================================
#     def updatePolarPlot(self, direction,velocity):
#         # Clear the polar plot
#         self.plt4.clear()
#     
#         # Add polar grid lines
#         self.plt4.addLine(x=0, pen=1)
#         self.plt4.addLine(y=0, pen=1)
#         for r in range(10, 50, 10):
#             r = r/10
#             circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
#             circle.setPen(pg.mkPen('w', width=0.5))
#             self.plt4.addItem(circle)
#     
#         # Convert direction and velocity to cartesian coordinates
#         theta = np.radians(direction)
#         radius = velocity
#         x = radius * np.cos(theta)
#         y = radius * np.sin(theta)
#     
#         # Plot lines in the polar plot for each direction and velocity point
#         for i in range(len(x)):        
#             path = QPainterPath(QPointF(0,0))
#             path.lineTo(QPointF(x[i],y[i]))
#             item = pg.QtGui.QGraphicsPathItem(path)
#             item.setPen(pg.mkPen('r', width=5))
#             self.plt4.addItem(item)
#     
#         # Add position labels to the polar plot
#         labels = [0,90,180,270]
#         d = 6
#         pos = [ (d,0),(0,d),(-d,0),(0,-d) ]
#         for i,label in enumerate(labels):
#             text = pg.TextItem(str(label), color=(200,200,0))
#             self.plt4.addItem(text)
#             text.setPos(pos[i][0],pos[i][1])
#     
#         # Add scale to the polar plot
#         for r in range(10, 50, 10):
#             r = r/10
#             text = pg.TextItem(str(r))
#             self.plt4.addItem(text)
#             text.setPos(0,r)
#     
#         return
# =============================================================================

    def togglePoistionIndicator(self):
        # If position indicators are not shown, add them to the plots
        if self.showPositionIndicators == False:
            # Add dashed lines to the plots
            self.plt1_line = self.plt1.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)
            self.plt2_line = self.plt2.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False) 
            self.plt4_line = self.plt4.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False)             
            self.plt5_line = self.plt5.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False) 
            self.plt6_line = self.plt6.addLine(x=0, pen=pg.mkPen('y', style=Qt.DashLine), movable=False) 
                
            # Connect the updatePositionIndicators method to the signal for time changes
            self.mainGUI.plotWindow.sigTimeChanged.connect(self.updatePositionIndicators)  
            
            # Update the flag and button text
            self.showPositionIndicators = True
            self.positionIndicator_button.setText("Hide position info")
                
        # If position indicators are shown, remove them from the plots
        else:
            # Remove the dashed lines from the plots
            self.plt1.removeItem(self.plt1_line)
            self.plt2.removeItem(self.plt2_line)
            self.plt4.removeItem(self.plt4_line)            
            self.plt5.removeItem(self.plt5_line)
            self.plt6.removeItem(self.plt6_line)             
                
            # Disconnect the updatePositionIndicators method from the signal for time changes
            self.mainGUI.plotWindow.sigTimeChanged.disconnect(self.updatePositionIndicators)  
            
            # Update the flag and button text
            self.showPositionIndicators = False
            self.positionIndicator_button.setText("Show position info")



    def updatePositionIndicators(self, t):
        #match frames to flika window numbering
        #t = t+1
        # Set the position of the position indicators in all four plots to the current time t
        self.plt1_line.setPos(t)
        self.plt2_line.setPos(t)
        self.plt4_line.setPos(t)        
        self.plt5_line.setPos(t)
        self.plt6_line.setPos(t)        
    
        # If a rectangular ROI exists in plot 3, remove it
        if self.r != None:
            self.plt3.removeItem(self.r)
            
        # If the current time t is in the data dictionary, create a new rectangular ROI and add it to plot 3
        if t in self.data:    
            self.r = pg.RectROI((self.data[t][0]-0.25,self.data[t][1]-0.25), size = pg.Point(0.5,0.5), movable=False,rotatable=False,resizable=False, pen=pg.mkPen('r',width=1))
            self.r.handlePen = None
            self.plt3.addItem(self.r)
    
    def show(self):
        # Show the plot window
        self.win.show()
    
    def close(self):
        # Close the plot window
        self.win.close()
    
    def hide(self):
        # Hide the plot window
        self.win.hide()
     
        
        
class ChartDock():
    """
    A class for creating a dockable window for displaying analysis charts.

    Attributes:
    -----------
    mainGUI : QtGui.QMainWindow
        The main window of the GUI containing this chart window.

    win : QtGui.QMainWindow
        The QMainWindow instance for the chart window.

    area : DockArea
        The DockArea instance for the chart window, which contains the individual docks.

    d1 : Dock
        The dock for plot options.

    d2 : Dock
        The dock for the main plot.

    d3 : Dock
        The dock for histogram options.

    d4 : Dock
        The dock for the histogram plot.
    """
    def __init__(self, mainGUI):
        super().__init__()    
        
        self.mainGUI = mainGUI
        
        # Create a new QMainWindow instance
        self.win = QMainWindow()
        
        # Create a new DockArea instance and set it as the central widget of the main window
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        
        # Set the size and title of the main window
        self.win.resize(1000,500)
        self.win.setWindowTitle('Plots')
        
        ## Create docks, place them into the window one at a time.
        
        # Create a dock for plot options and set its size
        self.d1 = Dock("plot options", size=(500, 100))
        
        # Create a dock for the main plot and set its size
        self.d2 = Dock("plot", size=(500,400))
        
        # Create a dock for histogram options and set its size
        self.d3 = Dock("histogram options", size=(500,100))
        
        # Create a dock for the histogram plot and set its size
        self.d4 = Dock("histogram", size=(500,400))
        
        # Add the plot options dock to the left side of the window
        self.area.addDock(self.d1, 'left') 
        
        # Add the histogram options dock to the right side of the window, next to the plot options dock
        self.area.addDock(self.d3, 'right', self.d1)       
                
        # Add the main plot dock to the bottom of the window, next to the plot options dock
        self.area.addDock(self.d2, 'bottom', self.d1)     
 
        # Add the histogram plot dock to the bottom of the window, next to the histogram options dock
        self.area.addDock(self.d4, 'bottom', self.d3)   

        #### SCATTER PLOT
        # Create a new layout widget for the scatter plot controls
        self.w1 = pg.LayoutWidget()
        
        # Create a combo box to choose between point and track data
        self.pointOrTrackData_selector_plot = pg.ComboBox()
        
        # Define the options for the plot data
        self.plotDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}
        
        # Set the options for the combo box
        self.pointOrTrackData_selector_plot.setItems(self.plotDataChoice)
        
        # Create labels for the x and y axes
        self.xlabel = QLabel("x:")  
        self.ylabel = QLabel("y:")  
        
        # Create a combo box for the x-axis data
        self.xColSelector = pg.ComboBox()
        
        # Define the default option for x-axis data
        self.xcols = {'None':'None'}
        
        # Set the options for the x-axis combo box
        self.xColSelector.setItems(self.xcols)
        
        # Create a combo box for the y-axis data
        self.yColSelector = pg.ComboBox()
        
        # Define the default option for y-axis data
        self.ycols = {'None':'None'}
        
        # Set the options for the y-axis combo box
        self.yColSelector.setItems(self.ycols)
        
        # Create a combo box for selecting the type of plot
        self.plotTypeSelector = pg.ComboBox()
        
        # Define the options for the plot type combo box
        self.plotTypes= {'scatter':'scatter', 'line':'line'}
        
        # Set the options for the plot type combo box
        self.plotTypeSelector.setItems(self.plotTypes)
        
        # Create a label for the plot type selector
        self.selectorLabel = QLabel("Plot type")  
        
        # Create a spin box for selecting the point size
        self.pointSize_selector = pg.SpinBox(value=7, int=True)
        self.pointSize_selector.setSingleStep(1)       
        self.pointSize_selector.setMinimum(1)
        self.pointSize_selector.setMaximum(10) 
        self.pointSize_selector.sigValueChanged.connect(self.updatePlot)
        
        # Create a label for the point size selector
        self.pointSizeLabel = QLabel("Point size") 
        
        # Create a button for plotting the data
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)
        
        # Add all the widgets to the layout widget
        self.w1.addWidget(self.pointOrTrackData_selector_plot , row=0, col=1)
        self.w1.addWidget(self.xColSelector, row=1, col=1)
        self.w1.addWidget(self.yColSelector, row=2, col=1)
        self.w1.addWidget(self.xlabel, row=1, col=0)
        self.w1.addWidget(self.ylabel, row=2, col=0) 
        self.w1.addWidget(self.plotTypeSelector, row=3,col=1)
        self.w1.addWidget(self.selectorLabel, row=3,col=0) 
        self.w1.addWidget(self.pointSizeLabel, row=4, col=0)         
        self.w1.addWidget(self.pointSize_selector, row=4, col=1)        
        self.w1.addWidget(self.plot_button, row=5, col=1)     
        
        # Add the layout widget to the dock
        self.d1.addWidget(self.w1) 
        
        #### HISTOGRAM
        # create a layout widget to hold the histogram controls
        self.w2 = pg.LayoutWidget()
        
        # create a ComboBox to select whether to plot point data or track means
        self.pointOrTrackData_selector_histo = pg.ComboBox()
        self.histoDataChoice = {'Point Data':'Point Data', 'Track Means': 'Track Means'}
        self.pointOrTrackData_selector_histo.setItems(self.histoDataChoice)
        
        # create a ComboBox to select which column to plot
        self.colSelector = pg.ComboBox()
        self.cols = {'None':'None'}
        self.colSelector.setItems(self.cols)
        
        # create a label for the column selector ComboBox
        self.collabel = QLabel("col:")
        
        # create a button to plot the histogram
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)
        
        # create a SpinBox to select the number of bins in the histogram
        self.histoBin_selector = pg.SpinBox(value=100, int=True)
        self.histoBin_selector.setSingleStep(1)
        self.histoBin_selector.setMinimum(1)
        self.histoBin_selector.setMaximum(100000)
        self.histoBin_selector.sigValueChanged.connect(self.updateHisto)
        
        # create a label for the SpinBox to select the number of bins in the histogram
        self.histoBin_label = QLabel('# of bins')
        
        # create a label for the checkbox to plot the cumulative histogram
        self.cumulativeTick_label = QLabel('cumulative')
        
        # create a checkbox to plot the cumulative histogram
        self.cumulativeTick = CheckBox()
        self.cumulativeTick.setChecked(False)
        
        # add the widgets to the layout widget
        self.w2.addWidget(self.pointOrTrackData_selector_histo , row=0, col=1)
        self.w2.addWidget(self.colSelector, row=1, col=1)
        self.w2.addWidget(self.collabel, row=1, col=0)
        self.w2.addWidget(self.histoBin_selector, row=2, col=1)
        self.w2.addWidget(self.histoBin_label, row=2, col=0)
        self.w2.addWidget(self.cumulativeTick_label, row=3, col=0)
        self.w2.addWidget(self.cumulativeTick, row=3, col=1)
        self.w2.addWidget(self.histo_button, row=4, col=1)
        
        # add the layout widget to the histogram dock
        self.d3.addWidget(self.w2)
        
        self.w3 = pg.PlotWidget(title="plot")
        self.w3.plot()
        self.w3.setLabel('left', 'y-axis', units ='')
        self.w3.setLabel('bottom', 'x-axis', units ='')  
        self.d2.addWidget(self.w3)  
        
        # create a plot widget for the histogram
        self.w4 = pg.PlotWidget(title="histogram")
        self.w4.plot()
        self.w4.setLabel('left', '# of observations', units ='')
        self.w4.setLabel('bottom', 'value', units ='')
        
        # add the plot widget to the histogram dock
        self.d4.addWidget(self.w4)

        
    def updatePlot(self):
        # Clear the current plot
        self.w3.clear()
    
        # Check if user selected point data or track means
        if self.pointOrTrackData_selector_plot.value() == 'Point Data':
    
            # If not using filtered data, extract x and y data from the main data table
            if self.mainGUI.useFilteredData == False:
                x = self.mainGUI.data[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.data[self.yColSelector.value()].to_numpy() 
            # If using filtered data, extract x and y data from the filtered data table
            else:
                x = self.mainGUI.filteredData[self.xColSelector.value()].to_numpy()
                y = self.mainGUI.filteredData[self.yColSelector.value()].to_numpy()             
    
        else:
            # If not using filtered data, group the data by track number and take the mean for each track
            if self.mainGUI.useFilteredData == False:                
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()                
            # If using filtered data, group the filtered data by track number and take the mean for each track
            else:                
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean()
    
            # Extract x and y data from the track means data table
            x = plotDF[self.xColSelector.value()].to_numpy()
            y = plotDF[self.yColSelector.value()].to_numpy() 
    
        # Check if user selected line or scatter plot
        if self.plotTypeSelector.value() == 'line':
            # Plot line using the selected x and y data, and set stepMode=False to draw a continuous line
            self.w3.plot(x, y, stepMode=False, brush=(0,0,255,150), clear=True) 
        elif self.plotTypeSelector.value() == 'scatter':
            # Plot scatter plot using the selected x and y data
            self.w3.plot(x, y,
                         pen=None,                                              # Set pen=None to remove the line border around each point, ,
                         symbol='o',                                            # set symbol='o' to use circles as symbols
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),        # set symbolPen=pg.mkPen(color=(0, 0, 255), width=0) to remove the border around each circle,
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),                # set symbolBrush=pg.mkBrush(0, 0, 255, 255) to fill the circles with blue color, and                                 
                         symbolSize=self.pointSize_selector.value())            # set symbolSize=self.pointSize_selector.value() to set the size of the circles
    
        # Set the labels for the x and y axes of the plot
        self.w3.setLabel('left', self.yColSelector.value(), units=None)
        self.w3.setLabel('bottom', self.xColSelector.value(), units=None)                 
        return
            
    def updateHisto(self):
        # clear the plot
        self.w4.clear()
    
        # check if point or track data is selected
        if self.pointOrTrackData_selector_histo.value() == 'Point Data':
            # use either data or filteredData based on the useFilteredData flag
            if self.mainGUI.useFilteredData == False:
                vals = self.mainGUI.data[self.colSelector.value()]
            else:
                vals = self.mainGUI.filteredData[self.colSelector.value()]
    
        else:
            # group the data by track number and calculate the mean of each group
            # use either data or filteredData based on the useFilteredData flag
            if self.mainGUI.useFilteredData == False:
                plotDF = self.mainGUI.data.groupby('track_number', as_index=False).mean()
            else:
                plotDF = self.mainGUI.filteredData.groupby('track_number', as_index=False).mean()
    
            # extract the values of the selected column from the plot dataframe
            vals = plotDF[self.colSelector.value()]
    
        # define the range and number of bins for the histogram
        start=0
        end=np.max(vals)
        n=self.histoBin_selector.value()
    
        # check if the cumulative tick is selected
        if self.cumulativeTick.isChecked():
            # calculate the cumulative distribution function (CDF)
            count,bins_count = np.histogram(vals, bins=np.linspace(start, end, n)) 
            pdf = count / sum(count)
            y = np.cumsum(pdf)        
            x = bins_count[1:]  
            # plot the CDF
            self.w4.plot(x, y, brush=(0,0,255,150), clear=True) 
    
        else:
            # calculate the histogram
            y,x = np.histogram(vals, bins=np.linspace(start, end, n))
            # plot the histogram
            self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True)
    
        # set the label of the x-axis to the selected column name
        self.w4.setLabel('bottom', self.colSelector.value(), units = None)     
        return
    
    def show(self):
        # show the window
        self.win.show()
    
    def close(self):
        # close the window
        self.win.close()
    
    def hide(self):
        # hide the window
        self.win.hide()
       
        
class FilterOptions():
    """
    A class for a GUI setting filter options for points and tracks.
    """
    def __init__(self, mainGUI):
        super().__init__()          
        
        # Initialize the main GUI instance
        self.mainGUI = mainGUI
        
        # Create a dock window and add a dock
        self.win = QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(550,100)
        self.win.setWindowTitle('Filter')
        self.d1 = Dock("Filter Options", size=(550, 100))
        self.area.addDock(self.d1)
        
        # Create a layout widget
        self.w1 = pg.LayoutWidget()
        
        # Create widgets for filter column, operator, and value selection
        self.filterCol_Box = pg.ComboBox()
        self.filtercols = {'None':'None'}
        self.filterCol_Box.setItems(self.filtercols)          
        self.filterOp_Box = pg.ComboBox()
        self.filterOps = {'=':'==', '<':'<', '>':'>', '!=':'!='}
        self.filterOp_Box.setItems(self.filterOps)         
        self.filterValue_Box = QLineEdit()
        
        # Create a checkbox for enabling/disabling sequential filtering
        self.sequentialFlter_checkbox = CheckBox() 
        self.sequentialFlter_checkbox.setChecked(False)  
        self.sequentialFlter_checkbox.stateChanged.connect(self.setSequentialFilter)
    
        # Create labels for filter column, operator, and value selection, as well as for the sequential filtering checkbox
        self.filterCol_label = QLabel('Filter column')
        self.filterVal_label = QLabel('Value') 
        self.filterOp_label = QLabel('Operator')
        self.sequentialFilter_label = QLabel('Allow sequential filtering')
                
        #buttons
        self.filterData_button = QPushButton('Filter')
        self.filterData_button.pressed.connect(self.mainGUI.filterData)                 
        self.clearFilterData_button = QPushButton('Clear Filter')
        self.clearFilterData_button.pressed.connect(self.mainGUI.clearFilterData)  
        self.ROIFilterData_button = QPushButton(' Filter by ROI(s)')
        self.ROIFilterData_button.pressed.connect(self.mainGUI.ROIFilterData)  
        self.clearROIFilterData_button = QPushButton('Clear ROI Filter')
        self.clearROIFilterData_button.pressed.connect(self.mainGUI.clearROIFilterData)  
        
        ## Add widgets to layout
        #row0
        self.w1.addWidget(self.filterCol_label, row=0,col=0)   
        self.w1.addWidget(self.filterCol_Box, row=0,col=1)   
        self.w1.addWidget(self.filterOp_label, row=0,col=2)         
        self.w1.addWidget(self.filterOp_Box, row=0,col=3) 
        self.w1.addWidget(self.filterVal_label, row=0,col=4)         
        self.w1.addWidget(self.filterValue_Box, row=0,col=5)  
        #row1      
        self.w1.addWidget(self.filterData_button, row=1,col=0)   
        self.w1.addWidget(self.clearFilterData_button, row=1,col=1)
        self.w1.addWidget(self.sequentialFilter_label, row=1,col=2)   
        self.w1.addWidget(self.sequentialFlter_checkbox, row=1,col=3)                  
        #row3
        self.w1.addWidget(self.ROIFilterData_button, row=3,col=0)   
        self.w1.addWidget(self.clearROIFilterData_button, row=3,col=1)          

        #add layout to dock
        self.d1.addWidget(self.w1)        
        
        
    def setSequentialFilter(self):
        if self.sequentialFlter_checkbox.isChecked():
            self.mainGUI.sequentialFiltering = True
        else:
            self.mainGUI.sequentialFiltering = False            

    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()        
'''        
#####################################################################################################################################
######################################   Main LOCSANDTRACKSPLOTTER CLASS   ##########################################################
#####################################################################################################################################
'''

class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    plots loc and track data onto current window
    """
    def __init__(self):
        # Initialize settings for locs and tracks plotter
        if g.settings['locsAndTracksPlotter'] is None or 'set_track_colour' not in g.settings['locsAndTracksPlotter']:
            s = dict()            
            s['filename'] = '' 
            s['filetype'] = 'flika'   
            s['pixelSize'] = 108                         
            s['set_track_colour'] = False
            g.settings['locsAndTracksPlotter'] = s
                   
        # Call the initialization function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.__init__(self)
        

    def __call__(self, filename, filetype, pixelSize, set_track_colour,  keepSourceWindow=False):
        '''
        Plots loc and track data onto the current window.
        
        Parameters:
        filename: str - path to file
        filetype: str - file type (flika or thunderstorm)
        pixelSize: int - pixel size of image data
        set_track_colour: bool - whether to set track colour based on track ID
        
        Returns: None
        '''

        # Save the input parameters to the locs and tracks plotter settings
        g.settings['locsAndTracksPlotter']['filename'] = filename 
        g.settings['locsAndTracksPlotter']['filetype'] = filetype       
        g.settings['locsAndTracksPlotter']['pixelSize'] = pixelSize 
        g.settings['locsAndTracksPlotter']['set_track_colour'] = set_track_colour      
        
        # Show a message in the status bar that plotting data is in progress
        g.m.statusBar().showMessage("plotting data...")
        return


    def closeEvent(self, event):
        '''
        This function is called when the user closes the locs and tracks plotter window. It clears any plots that have been
        generated and calls the closeEvent function for the BaseProcess_noPriorWindow class.
        
        Parameters:
        event: object - object representing the close event
        
        Returns: None
        '''
        # Clear any plots that have been generated
        self.clearPlots()
        
        # Call the closeEvent function for the BaseProcess_noPriorWindow class
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return
        
        
    def gui(self):      
        # Initialize class variables
        self.filename = '' 
        self.filetype = 'flika'   
        self.pixelSize= None  
        self.plotWindow = None
        self.pathitems = []
        self.useFilteredData = False
        self.useFilteredTracks = False
        self.filteredData = None
        self.useMatplotCM = False
        self.selectedTrack = None
        self.displayTrack = None
        self.chartWindow = None
        self.displayCharts = False
        self.diffusionWindow = None
        self.displayDiffusionPlot = False      
        self.unlinkedPoints = None
        self.displayUnlinkedPoints = False  
            
        # Initialize filter options window and hide it
        self.filterOptionsWindow = FilterOptions(self)
        self.filterOptionsWindow.hide()     
            
        self.sequentialFiltering = False
            
        # Initialize track plot window and hide it
        self.trackWindow = TrackWindow(self)
        self.trackWindow.hide()
            
        # Initialize flower plot window and hide it
        self.flowerPlotWindow = FlowerPlotWindow(self)
        self.flowerPlotWindow.hide()  
            
        # Initialize single track plot window and hide it
        self.singleTrackPlot= TrackPlot(self)
        self.singleTrackPlot.hide() 
        
        # Initialize all-tracks plot window and hide it
        self.allTracksPlot = AllTracksPlot(self)
        self.allTracksPlot.hide()         
            
        # Call gui_reset function
        self.gui_reset()        
        # Get settings for locsAndTracksPlotter
        s=g.settings['locsAndTracksPlotter']
        
        #buttons      
        self.plotPointData_button = QPushButton('Plot Points')
        self.plotPointData_button.pressed.connect(self.plotPointData)  
        
        self.hidePointData_button = QPushButton('Toggle Points')
        self.hidePointData_button.pressed.connect(self.hidePointData)  
        
        self.toggleUnlinkedPointData_button = QPushButton('Show Unlinked')
        self.toggleUnlinkedPointData_button.pressed.connect(self.toggleUnlinkedPointData)  
      
            
        self.plotTrackData_button = QPushButton('Plot Tracks')
        self.plotTrackData_button.pressed.connect(self.plotTrackData)  
        
        self.clearTrackData_button = QPushButton('Clear Tracks')
        self.clearTrackData_button.pressed.connect(self.clearTracks)  
        
        
        self.saveData_button = QPushButton('Save Tracks')
        self.saveData_button.pressed.connect(self.saveData)    
        
        self.showCharts_button = QPushButton('Show Charts')
        self.showCharts_button.pressed.connect(self.toggleCharts)    
        
        self.showDiffusion_button = QPushButton('Show Diffusion')
        self.showDiffusion_button.pressed.connect(self.toggleDiffusionPlot) 

        self.togglePointMap_button = QPushButton('Plot Point Map')
        self.togglePointMap_button.pressed.connect(self.togglePointMap) 
                         
        #checkbox
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(s['set_track_colour'])
        
        self.matplotCM_checkbox = CheckBox() 
        self.matplotCM_checkbox.stateChanged.connect(self.setColourMap)
        self.matplotCM_checkbox.setChecked(False)  

        self.displayFlowPlot_checkbox = CheckBox() 
        self.displayFlowPlot_checkbox.stateChanged.connect(self.toggleFlowerPlot)
        self.displayFlowPlot_checkbox.setChecked(False)  
        
        self.displaySingleTrackPlot_checkbox = CheckBox() 
        self.displaySingleTrackPlot_checkbox.stateChanged.connect(self.toggleSingleTrackPlot)
        self.displaySingleTrackPlot_checkbox.setChecked(False) 
        
        self.displayAllTracksPlot_checkbox = CheckBox() 
        self.displayAllTracksPlot_checkbox.stateChanged.connect(self.toggleAllTracksPlot)
        self.displayAllTracksPlot_checkbox.setChecked(False)         

        self.displayFilterOptions_checkbox = CheckBox() 
        self.displayFilterOptions_checkbox.stateChanged.connect(self.toggleFilterOptions)
        self.displayFilterOptions_checkbox.setChecked(False)          

        #comboboxes
        self.filetype_Box = pg.ComboBox()
        filetypes = {'flika' : 'flika', 'thunderstorm':'thunderstorm', 'xy':'xy'}
        self.filetype_Box.setItems(filetypes)

        self.xCol_Box = pg.ComboBox()
        self.xcols = {'None':'None'}
        self.xCol_Box.setItems(self.xcols)

        self.yCol_Box = pg.ComboBox()
        self.ycols = {'None':'None'}
        self.yCol_Box.setItems(self.ycols)
        
        self.frameCol_Box = pg.ComboBox()
        self.framecols = {'None':'None'}
        self.frameCol_Box.setItems(self.framecols)        
 
        self.trackCol_Box = pg.ComboBox()
        self.trackcols = {'None':'None'}
        self.trackCol_Box.setItems(self.trackcols)   
        

        self.trackColourCol_Box = pg.ComboBox()
        self.trackcolourcols = {'None':'None'}
        self.trackColourCol_Box.setItems(self.trackcolourcols)  
        
        self.colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.colourMap_Box.setItems(self.colourMaps)         
    
        
        self.trackDefaultColour_Box = pg.ComboBox()
        self.trackdefaultcolours = {'green': Qt.green, 'red': Qt.red, 'blue': Qt.blue}
        self.trackDefaultColour_Box.setItems(self.trackdefaultcolours)

        #spinbox
        self.frameLength_selector = pg.SpinBox(value=10, int=True)
        self.frameLength_selector.setSingleStep(10)       #########TODDO! FIX THIS - AND INDEXING PROBLEM IN CALL  BY ADDING OBJECT NAMES ABOVE
        self.frameLength_selector.setMinimum(1)
        self.frameLength_selector.setMaximum(100000) 
        
        self.pixelSize_selector = pg.SpinBox(value=108, int=True)
        self.pixelSize_selector.setSingleStep(1)       
        self.pixelSize_selector.setMinimum(1)
        self.pixelSize_selector.setMaximum(10000) 

        
        #data file selector
        self.getFile = FileSelector(filetypes='*.csv', mainGUI=self)
        
        #connections
        self.getFile.valueChanged.connect(self.loadData)

        
        #################################################################
        # Define the items that will appear in the GUI, and associate them with the appropriate functions.        
        #self.exportFolder = FolderSelector('*.txt')    

        self.items.append({'name': 'filename ', 'string': 'before load, select data window', 'object': self.getFile})    
        self.items.append({'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box})  
        self.items.append({'name': 'pixelSize', 'string': 'nanometers per pixel', 'object': self.pixelSize_selector})                  
        self.items.append({'name': 'frameLength', 'string': 'milliseconds per frame', 'object': self.frameLength_selector})                 
        self.items.append({'name': 'hidePoints', 'string': 'PLOT    --------------------', 'object': self.hidePointData_button })        
        self.items.append({'name': 'plotPointMap', 'string': '', 'object': self.togglePointMap_button })
        self.items.append({'name': 'plotUnlinkedPoints', 'string': '', 'object': self.toggleUnlinkedPointData_button })               
        self.items.append({'name': 'trackDefaultColour', 'string': 'Track Default Colour', 'object': self.trackDefaultColour_Box })        
        self.items.append({'name': 'trackColour', 'string': 'Set Track Colour', 'object': self.trackColour_checkbox})           
        self.items.append({'name': 'trackColourCol', 'string': 'Colour by', 'object': self.trackColourCol_Box})
        self.items.append({'name': 'trackColourMap', 'string': 'Colour Map', 'object': self.colourMap_Box})   
        self.items.append({'name': 'matplotClourMap', 'string': 'Use matplot map', 'object': self.matplotCM_checkbox}) 
        self.items.append({'name': 'displayFlowerPlot', 'string': 'Flower Plot', 'object': self.displayFlowPlot_checkbox})  
        self.items.append({'name': 'displaySingleTrackPlot', 'string': 'Track Plot', 'object': self.displaySingleTrackPlot_checkbox}) 
        self.items.append({'name': 'displayAllTracksPlot', 'string': 'All Tracks Plot', 'object': self.displayAllTracksPlot_checkbox})         
        self.items.append({'name': 'displayFilterOptions', 'string': 'Filter Window', 'object': self.displayFilterOptions_checkbox})  
        self.items.append({'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button })         
        self.items.append({'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button })     
        self.items.append({'name': 'saveTracks', 'string': '', 'object': self.saveData_button })  
        self.items.append({'name': 'showCharts', 'string': '', 'object': self.showCharts_button })
        self.items.append({'name': 'showDiffusion', 'string': '', 'object': self.showDiffusion_button })          
            
        
        super().gui()
        ######################################################################
        return

    def loadData(self):
        # Set the plot window to the global window instance
        self.plotWindow = g.win
        
        # Get the filename from the GUI
        self.filename = self.getFile.value()
        
        # Load the data from the selected file using Pandas
        self.data = pd.read_csv(self.filename)

        ### TODO! #Check if analysed columns are in df - if missing add and display error message
        #g.m.statusBar().showMessage('{} columns missing - blanks aded'.format()) 
        #print('{} columns missing - blanks aded'.format()) 
        
        #make sure track number and frame are int
        self.data['frame'] = self.data['frame'].astype(int)        
        
        if 'track_number' in self.data.columns:            
            # filter any points that dont have track_numbers to seperate df
            self.data_unlinked = self.data[self.data['track_number'].isna()]
            self.data  = self.data[~self.data['track_number'].isna()]
            
            self.data['track_number'] = self.data['track_number'].astype(int)
        
        # Check that there are enough frames in the stack to plot all data points
        if np.max(self.data['frame']) > g.win.mt:
            g.alert("Selected window doesn't have enough frames to plot all data points")
            self.plotWindow = None
            self.filename = None
            self.data = None
            return
        
        # Print a message to the console indicating that the data has been loaded
        # and display the first 5 rows of the data
        print('-------------------------------------')
        print('Data loaded (first 5 rows displayed):')
        print(self.data.head())
        print('-------------------------------------')
        
        # Create a dictionary from the column names of the data
        self.columns = self.data.columns
        self.colDict= dictFromList(self.columns)

        # Set the options for the various dropdown menus in the GUI based on the column names
        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)
        self.frameCol_Box.setItems(self.colDict)        
        self.trackCol_Box.setItems(self.colDict)   
        self.filterOptionsWindow.filterCol_Box.setItems(self.colDict)  
        self.trackColourCol_Box.setItems(self.colDict)  

        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)
        
        # Format the points and add them to the image window
        self.plotPointData()
        
        # Update the track plot track selector and set the options for the color dropdowns
        self.singleTrackPlot.updateTrackList()        
        self.singleTrackPlot.pointCol_Box.setItems(self.colDict)
        self.singleTrackPlot.lineCol_Box.setItems(self.colDict)  

        # Set the padding array for the single track plot based on the image array
        self.singleTrackPlot.setPadArray(self.plotWindow.imageArray())     
        
        # Update the all-tracks plot track selector 
        self.allTracksPlot.updateTrackList()         
        

    def makePointDataDF(self, data):
        # Check the filetype selected in the GUI
        if self.filetype_Box.value() == 'thunderstorm':
            # Load thunderstorm data into a pandas dataframe
            df = pd.DataFrame()
            # Convert frame data from float to int and adjust by -1 (since ThunderSTORM starts counting from 1)
            df['frame'] = data['frame'].astype(int)-1
            # Convert x data from nanometers to pixels
            df['x'] = data['x [nm]'] / self.pixelSize_selector.value()
            # Convert y data from nanometers to pixels
            df['y'] = data['y [nm]'] / self.pixelSize_selector.value()
    
        elif self.filetype_Box.value() == 'flika':
            # Load FLIKA pyinsight data into a pandas dataframe
            df = pd.DataFrame()
            # Convert frame data from float to int and adjust by -1 (since FLIKA starts counting from 1)
            df['frame'] = data['frame'].astype(int)-1
            # Use x data as-is (FLIKA data is already in pixel units)
            df['x'] = data['x']
            # Use y data as-is (FLIKA data is already in pixel units)
            df['y'] = data['y']

    
        # Return the completed pandas dataframe
        return df
            
    def plotPointsOnStack(self, points, pointColor, unlinkedPoints=None, unlinkedColour=QColor(Qt.blue)):                
            points_byFrame = points[['frame','x','y']]   
            #align frames with display
            points_byFrame['frame'] =  points_byFrame['frame']+1
            # Convert the points DataFrame into a numpy array
            pointArray = points_byFrame.to_numpy()
            # Create an empty list for each frame in the stack

            self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)] 
           
                
            # Iterate through each point in the point array and add it to the appropriate frame's list
            for pt in pointArray:
                t = int(pt[0])
                if self.plotWindow.mt == 1:
                    t = 0
                pointSize = g.m.settings['point_size']
                #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
                position = [pt[1], pt[2], pointColor, pointSize] 
             
                self.plotWindow.scatterPoints[t].append(position)
 
                
            if self.displayUnlinkedPoints:
                unlinkedPoints_byFrame = unlinkedPoints[['frame','x','y']] 
                unlinkedPoints_byFrame['frame'] =  unlinkedPoints_byFrame['frame']+1
                # Convert the points DataFrame into a numpy array
                unlinkedPointArray = unlinkedPoints_byFrame.to_numpy()
               
                    
                # Iterate through each point in the point array and add it to the appropriate frame's list
                for pt in unlinkedPointArray:
                    t = int(pt[0])
                    if self.plotWindow.mt == 1:
                        t = 0
                    pointSize = g.m.settings['point_size']
                    #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
                    position = [pt[1], pt[2], unlinkedColour, pointSize] 
                 
                    self.plotWindow.scatterPoints[t].append(position)                
    
 
            
            # Update the index of the image stack to include the new points
            self.plotWindow.updateindex()
        
        
    def hidePointData(self):
        if self.plotWindow.scatterPlot in self.plotWindow.imageview.ui.graphicsView.items():
            # If the scatter plot is currently in the graphics view, remove it
            self.plotWindow.imageview.ui.graphicsView.removeItem(self.plotWindow.scatterPlot)
        else:
            # Otherwise, add it back to the graphics view
            self.plotWindow.imageview.addItem(self.plotWindow.scatterPlot)

    
    def plotPointData(self):
        ### plot point data to current window
        # Create a pandas DataFrame containing the point data
        if self.useFilteredData == False:
            self.points = self.makePointDataDF(self.data)
        else:
            self.points = self.makePointDataDF(self.filteredData)
            
        if self.displayUnlinkedPoints:
            # Create a pandas DataFrame containing the point data
            self.unlinkedPoints = self.makePointDataDF(self.data_unlinked)
        else:
            self.unlinkedPoints = None
                
        # Plot the points on the image stack using the plotPointsOnStack() method
        self.plotPointsOnStack(self.points, QColor(g.m.settings['point_color']), unlinkedPoints=self.unlinkedPoints)
    
        # Display a message in the status bar indicating that the point data has been plotted
        g.m.statusBar().showMessage('point data plotted to current window') 
        print('point data plotted to current window')    
    
        return


    def toggleUnlinkedPointData(self):
        if self.displayUnlinkedPoints == False:    
            ### plot unlinked point data to current window 
            self.displayUnlinkedPoints = True  
            self.plotPointData() 
            self.toggleUnlinkedPointData_button.setText('Hide Unlinked')
            # Display a message in the status bar indicating that the point data has been plotted
            g.m.statusBar().showMessage('unlinked point data plotted to current window') 
            print('unlinked point data plotted to current window') 
             
        else:
            self.displayUnlinkedPoints = False    
            self.plotPointData()             
            self.toggleUnlinkedPointData_button.setText('Show Unlinked')
        return

        
    def makeTrackDF(self, data):
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.pixelSize_selector.value()
            df['y'] = data['y [nm]']/self.pixelSize_selector.value()  
            df['track_number'] = data['track_number']
    
        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']
            df['track_number'] = data['track_number']
            
            df['zeroed_X'] = data['zeroed_X']
            df['zeroed_Y'] = data['zeroed_Y']   
            
            # Add a color column to the DataFrame based on the selected color map and column
            if self.trackColour_checkbox.isChecked():
                if self.useMatplotCM:
                    cm = pg.colormap.getFromMatplotlib(self.colourMap_Box.value()) # Get the colormap from Matplotlib and convert it to a PyqtGraph colormap
                else:    
                    cm = pg.colormap.get(self.colourMap_Box.value()) # Get the PyqtGraph colormap
                
                # Map the values from the selected color column to a QColor using the selected colormap
                df['colour'] = cm.mapToQColor(data[self.trackColourCol_Box.value()].to_numpy()/max(data[self.trackColourCol_Box.value()]))
        
        # Group the data by track number
        return df.groupby(['track_number'])
        
        
    def clearTracks(self):
        # Check that there is an open plot window
        if self.plotWindow is not None and not self.plotWindow.closed:
            # Remove each path item from the plot window
            for pathitem in self.pathitems:
                self.plotWindow.imageview.view.removeItem(pathitem)
        # Reset the path items list to an empty list
        self.pathitems = []
        
        
        
    def showTracks(self):
        '''Updates track paths in main view and Flower Plot'''
        
        # clear self.pathitems
        self.clearTracks()
        
        # clear track paths in Flower Plot window if displayFlowPlot_checkbox is checked
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.clearTracks()
        
        # setup pens
        pen = QPen(self.trackDefaultColour_Box.value(), .4)
        pen.setCosmetic(True)
        pen.setWidth(2)
        
        pen_FP = QPen(self.trackDefaultColour_Box.value(), .4)
        pen_FP.setCosmetic(True)
        pen_FP.setWidth(1)       
        
        # determine which track IDs to plot based on whether filtered tracks are being used
        if self.useFilteredTracks:
            trackIDs = self.filteredTrackIds
        else:
            trackIDs = self.trackIDs

        print('tracks to plot {}'.format(trackIDs))

        for track_idx in trackIDs:
            tracks = self.tracks.get_group(track_idx)
            pathitem = QGraphicsPathItem(self.plotWindow.imageview.view)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP = QGraphicsPathItem(self.flowerPlotWindow.plt) 
        
            # set the color of the pen based on the track color
            if self.trackColour_checkbox.isChecked():
                pen.setColor(tracks['colour'].to_list()[0])
                pen_FP.setColor(tracks['colour'].to_list()[0])                
        
            # set the pen for the path items
            pathitem.setPen(pen)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPen(pen_FP)
        
            # add the path items to the view(s)
            self.plotWindow.imageview.view.addItem(pathitem)
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.plt.addItem(pathitem_FP)
        
            # keep track of the path items
            self.pathitems.append(pathitem)
            if self.displayFlowPlot_checkbox.isChecked():
                self.flowerPlotWindow.pathitems.append(pathitem_FP)
        
            # extract the x and y coordinates for the track
            x = tracks['x'].to_numpy()
            y = tracks['y'].to_numpy() 
        
            # extract the zeroed x and y coordinates for the track, if displayed
            if self.displayFlowPlot_checkbox.isChecked():
                zeroed_X = tracks['zeroed_X'].to_numpy()
                zeroed_Y = tracks['zeroed_Y'].to_numpy() 
        
            # create a QPainterPath for the track and set the path for the path item
            path = QPainterPath(QPointF(x[0],y[0]))
            if self.displayFlowPlot_checkbox.isChecked():
                path_FP = QPainterPath(QPointF(zeroed_X[0],zeroed_Y[0]))
            for i in np.arange(1, len(x)):
                path.lineTo(QPointF(x[i],y[i]))
                if self.displayFlowPlot_checkbox.isChecked():
                    path_FP.lineTo(QPointF(zeroed_X[i],zeroed_Y[i]))                
        
            pathitem.setPath(path)
            if self.displayFlowPlot_checkbox.isChecked():
                pathitem_FP.setPath(path_FP)
        
    def plotTrackData(self):
        ### plot track data to current window
        
        # check whether to use filtered data or not, get unique track IDs and create DataFrame of tracks
        if self.useFilteredData == False:            
            self.trackIDs = np.unique(self.data['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.data)
        else:
            self.trackIDs = np.unique(self.filteredData['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.filteredData)           
        
        # show tracks in main view and flower plot
        self.showTracks()
        
        # connect to mouse and key press events in the main view
        self.plotWindow.imageview.scene.sigMouseMoved.connect(self.updateTrackSelector)
        self.plotWindow.keyPressSignal.connect(self.selectTrack)
        
        # display track window with plots for individual tracks
        self.trackWindow.show()
        
        # display flower plot with all tracks origins set to 0,0
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()
            
        # display plot for a single selected track
        if self.displaySingleTrackPlot_checkbox.isChecked():
            self.singleTrackPlot.show()            
            
        # update status bar message and print confirmation
        g.m.statusBar().showMessage('track data plotted to current window') 
        print('track data plotted to current window')    
        return
        
    def updateTrackSelector(self, point):
        pos =  self.plotWindow.imageview.getImageItem().mapFromScene(point)
    
        # Map mouse position to image coordinates and check which track the mouse is hovering over
        for i, path in enumerate(self.pathitems):
            if path.contains(pos):
                self.selectedTrack = self.trackIDs[i]                 
                    
    
    def selectTrack(self,ev):
        # Listen for key press events and select track when the "T" key is pressed
        if ev.key() == Qt.Key_T:
            # Check if the selected track is different from the current display track
            if self.selectedTrack != self.displayTrack:
                self.displayTrack = self.selectedTrack    
                
                # Extract track data for the selected track
                trackData = self.data[self.data['track_number'] == int(self.displayTrack)]
                frame = trackData['frame'].to_numpy()
                intensity = trackData['intensity'].to_numpy() 
                distance = trackData['distanceFromOrigin'].to_numpy() 
                zeroed_X = trackData['zeroed_X'].to_numpy()
                zeroed_Y = trackData['zeroed_Y'].to_numpy()  
                dydt =  trackData['dy-dt: distance'].to_numpy() 
                direction = trackData['direction_Relative_To_Origin'].to_numpy()    
                velocity =  trackData['velocity'].to_numpy()  
                
                count_3 = trackData['nnCountInFrame_within_3_pixels'].to_numpy() 
                count_5 = trackData['nnCountInFrame_within_5_pixels'].to_numpy()                
                count_10 = trackData['nnCountInFrame_within_10_pixels'].to_numpy()                 
                count_20 = trackData['nnCountInFrame_within_20_pixels'].to_numpy() 
                count_30 = trackData['nnCountInFrame_within_30_pixels'].to_numpy()                
                
                
                # Update plots in the track display window
                self.trackWindow.update(frame, intensity, distance,
                                        zeroed_X, zeroed_Y, dydt,
                                        direction, velocity, self.displayTrack,
                                        count_3, count_5, count_10, count_20, count_30 )
                
                # Update the individual track display
                self.singleTrackPlot.plotTracks()


        if ev.key() == Qt.Key_R:
                               
            roiFilterPoints = []
            roi = self.plotWindow.currentROI
            
            currentFrame = self.plotWindow.currentIndex
            
            for i in range(0,self.plotWindow.mt):
                # get ROI shape in coordinate system of the scatter plot
                self.plotWindow.setIndex(i)
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                roiFilterPoints.extend((selected))
            
            self.plotWindow.setIndex(currentFrame)
             
            trackIDs = []
            
            for pt in roiFilterPoints:            
                # filter data by point coordinates
                ptFilterDF = self.data[(self.data['x']==pt[1]) & (self.data['y']==pt[2])]                
                trackIDs.extend(ptFilterDF['track_number'])
               
            # get unique track IDs for tracks that pass through the ROI
            selectedTracks = np.unique(trackIDs)    
            
            # join data for selected tracks
            self.joinROItracks(selectedTracks)
            
            # display message in status bar to indicate completion
            g.m.statusBar().showMessage('Track join complete')  


    def joinROItracks(self, selectedTracks):
        # create an instance of the JoinTracks class
        joinTracks = JoinTracks()        
        # create a list of selected tracks
        IDlist = [selectedTracks]
        # display the number of tracks to be joined in the status bar
        g.m.statusBar().showMessage('Tracks to join: {}'.format(IDlist)) 
        # use the joinTracks object to join the selected tracks
        newDF = joinTracks.join(self.data, IDlist)
        # replace the track data with the updated DataFrame that includes the joined track
        self.data = newDF
        # print the new DataFrame to the console (for debugging purposes)
        print(newDF)
        # display a message in the status bar to indicate that the track join is complete
        g.m.statusBar().showMessage('track join complete') 
    
    def filterData(self):
        # get the filter options from the filterOptionsWindow
        op = self.filterOptionsWindow.filterOp_Box.value()
        filterCol = self.filterOptionsWindow.filterCol_Box.value()
        dtype = self.data[filterCol].dtype 
        value = float(self.filterOptionsWindow.filterValue_Box.text())
            
        # if sequential filtering is enabled and filtered data is being used
        if self.sequentialFiltering and self.useFilteredData:            
            data = self.filteredData
        else:
            data = self.data
            
        # apply the selected filter operation to the selected column
        if op == '==':
            self.filteredData = data[data[filterCol] == value]
        elif op == '<':
            self.filteredData = data[data[filterCol] < value]
        elif op == '>':
            self.filteredData = data[data[filterCol] > value]           
        elif op == '!=':
            self.filteredData = data[data[filterCol] != value]    
            
        # display a message in the status bar to indicate that the filter is complete
        g.m.statusBar().showMessage('filter complete') 
        # set useFilteredData to True
        self.useFilteredData = True
            
        # update the point data plot with the filtered data
        self.plotPointData()
        
        #update allTracks track list
        self.allTracksPlot.updateTrackList()  
            
        return

    def clearFilterData(self):
        # Set variables to default values to clear filtered data
        self.useFilteredData = False
        self.filteredData = None
        
        # Update point data plot
        self.plotPointData()
        
        #update allTracks track list
        self.allTracksPlot.updateTrackList()  
        return

    def getScatterPointsAsQPoints(self):
        # Get scatter plot data as numpy array
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        # Convert numpy array to list of QPointF objects
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        # Get track IDs for all points in scatter plot
        trackIDs = []
        
        # Flatten scatter plot data into a single list of points
        flat_ptList = [pt for sublist in self.plotWindow.scatterPoints for pt in sublist]
        
        # Loop through each point and get track IDs for corresponding data points in DataFrame
        for pt in flat_ptList:            
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.data[(self.data['x']==pt[0]) & (self.data['y']==pt[1])]
            
            trackIDs.extend(ptFilterDF['track_number'])

        # Set filtered track IDs and filtered data
        self.filteredTrackIds = np.unique(trackIDs)
        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        
        # Set flags for using filtered data and filtered tracks
        self.useFilteredData = True
        self.useFilteredTracks = True


    def ROIFilterData(self):
        # Not implemented yet for unlinked points
        if self.displayUnlinkedPoints:
           g.m.statusBar().showMessage('ROI filter not implemented for unliked points - hide them first') 
           print('ROI filter not implemented for unliked points - hide them first')    
           return
        # initialize variables
        self.roiFilterPoints = []
        self.rois = self.plotWindow.rois
        self.oldScatterPoints = self.plotWindow.scatterPoints
        
        # loop through all ROIs and all frames to find points inside them
        for roi in self.rois:
            currentFrame = self.plotWindow.currentIndex
            for i in range(0,self.plotWindow.mt):
                # set current frame
                self.plotWindow.setIndex(i)
                # get ROI shape in coordinate system of the scatter plot
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                self.roiFilterPoints.extend((selected))
            # reset current frame
            self.plotWindow.setIndex(currentFrame)
        
        # clear old scatter points and add new filtered points
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        for pt in self.roiFilterPoints:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(0,255,0)
            position = [pt[1], pt[2], pointColor, pointSize]  
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()
    
        # get filtered data
        self.getDataFromScatterPoints()
        
        # update status bar and return
        g.m.statusBar().showMessage('ROI filter complete') 
        
        #update allTracks track list
        self.allTracksPlot.updateTrackList()  
        return

    def clearROIFilterData(self):
        # Reset the scatter plot data to the previous unfiltered scatter plot data
        self.plotWindow.scatterPoints = self.oldScatterPoints 
        self.plotWindow.updateindex()
        
        # Set useFilteredData and useFilteredTracks to False
        self.useFilteredData = False
        self.useFilteredTracks = False
        
        #update allTracks track list
        self.allTracksPlot.updateTrackList()  
        
        return
    
    def setColourMap(self):
        # If the matplotCM_checkbox is checked, use matplotlib color maps
        if self.matplotCM_checkbox.isChecked():
            # Create a dictionary of matplotlib color maps
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
            # Set the color map options in the dropdown box to the matplotlib color maps
            self.colourMap_Box.setItems(self.colourMaps)  
            self.useMatplotCM = True
        else:
            # If the matplotCM_checkbox is unchecked, use pyqtgraph color maps
            # Create a dictionary of pyqtgraph color maps
            self.colourMaps = dictFromList(pg.colormap.listMaps())
            # Set the color map options in the dropdown box to the pyqtgraph color maps
            self.colourMap_Box.setItems(self.colourMaps) 
            self.useMatplotCM = False
    
    def toggleFlowerPlot(self):
        # If the displayFlowPlot_checkbox is checked, show the flower plot window
        if self.displayFlowPlot_checkbox.isChecked():
            self.flowerPlotWindow.show()
        else:
            # If the displayFlowPlot_checkbox is unchecked, hide the flower plot window
            self.flowerPlotWindow.hide()


    def toggleSingleTrackPlot(self):
        if self.displaySingleTrackPlot_checkbox.isChecked():
            # show the single track plot if checkbox is checked
            self.singleTrackPlot.show()
        else:
            # hide the single track plot if checkbox is unchecked
            self.singleTrackPlot.hide()   
            
    def toggleAllTracksPlot(self):
        if self.displayAllTracksPlot_checkbox.isChecked():
            # show the single track plot if checkbox is checked
            self.allTracksPlot.show()
        else:
            # hide the single track plot if checkbox is unchecked
            self.allTracksPlot.hide()              

    def toggleFilterOptions(self):
        if self.displayFilterOptions_checkbox.isChecked():
            # show the filter options window if checkbox is checked
            self.filterOptionsWindow.show()
        else:
            # hide the filter options window if checkbox is unchecked
            self.filterOptionsWindow.hide()   

    def toggleCharts(self):
        if self.chartWindow == None:
            # create chart plot window and set items for column selectors
            self.chartWindow = ChartDock(self)
            self.chartWindow.xColSelector.setItems(self.colDict)
            self.chartWindow.yColSelector.setItems(self.colDict)
            self.chartWindow.colSelector.setItems(self.colDict)    
            
            self.chartWindow.xcols = self.colDict
            self.chartWindow.ycols = self.colDict            
            self.chartWindow.cols = self.colDict
            
        if self.displayCharts == False:
            # show the chart plot window if not currently displayed
            self.chartWindow.show()
            self.displayCharts = True
            self.showCharts_button.setText('Hide Charts')
        else:
            # hide the chart plot window if currently displayed
            self.chartWindow.hide()
            self.displayCharts = False   
            self.showCharts_button.setText('Show Charts')

    def toggleDiffusionPlot(self):
        if self.diffusionWindow == None:
            # Create a new instance of the DiffusionPlotWindow class
            self.diffusionWindow = DiffusionPlotWindow(self)   
                            
        if self.displayDiffusionPlot == False:
            # Show the window if it is not already displayed
            self.diffusionWindow.show()
            self.displayDiffusionPlot = True
            self.showDiffusion_button.setText('Hide Diffusion')
        else:
            # Hide the window if it is already displayed
            self.chartWindow.hide()
            self.displayDiffusionPlot = False   
            self.showDiffusion_button.setText('Show Diffusion')
    
    
    def togglePointMap(self):
        if self.togglePointMap_button.text() == 'Plot Point Map':
            # Check if filtered data is being used, if not use the original data
            if self.useFilteredData == False:            
                df = self.data
            else:           
                df = self.filteredData
            
            #add in unlinked points if displayed
            if self.displayUnlinkedPoints:
                df = df.append(self.data_unlinked)
                    
            # Create a ScatterPlotItem and add it to the ImageView
            self.pointMapScatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(30, 255, 35, 255))
            self.pointMapScatter.setSize(2, update=False)
            self.pointMapScatter.setData(df['x'], df['y'])
            self.plotWindow.imageview.view.addItem(self.pointMapScatter)
            self.togglePointMap_button.setText('Hide Point Map')
        else:
            # Remove the ScatterPlotItem from the ImageView
            self.plotWindow.imageview.view.removeItem(self.pointMapScatter)
            self.togglePointMap_button.setText('Plot Point Map')
    

    def createStatsDFs(self):
            # Calculate mean and standard deviation for each track in the original data
            self.meanDF = self.data.groupby('track_number', as_index=False).mean()
            self.stdDF = self.data.groupby('track_number', as_index=False).std()        
    
    def createStatsDFs_filtered(self):
            # Calculate mean and standard deviation for each track in the filtered data
            self.meanDF_filtered = self.filteredData.groupby('track_number', as_index=False).mean()
            self.stdDF_filtered = self.filteredData.groupby('track_number', as_index=False).std() 
    
    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return
    
    def saveData(self):      
        if self.useFilteredData == False:
            print('filter data first')
            g.alert('Filter data first')
            return
        
        # Prompt user to select a save path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")        
    
        # Write the filtered data to a CSV file
        try:
            self.filteredData.to_csv(savePath)
            print('Filtered data saved to: {}'.format(savePath))
        except BaseException as e:
            print(e)
            print('Export of filtered data failed')

# Instantiate the LocsAndTracksPlotter class
locsAndTracksPlotter = LocsAndTracksPlotter()

# Check if this script is being run as the main program
if __name__ == "__main__":
    pass











    