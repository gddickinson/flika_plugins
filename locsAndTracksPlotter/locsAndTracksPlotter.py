# -*- coding: utf-8 -*-
"""
Created on Sat May 23 10:38:20 2020

@author: george.dickinson@gmail.com
"""
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

import time

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

import numba
pg.setConfigOption('useNumba', True)

import pandas as pd
from matplotlib import pyplot as plt

def dictFromList(l):
    # Create a zip object from two lists
    zipbObj = zip(l, l)
    return dict(zipbObj)

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
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
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
        self.pixelSize = 108
        
    def buttonclicked(self):
        prompt = 'testing fileSelector'
        self.filename = open_file_gui(prompt, filetypes=self.filetypes)
        self.label.setText('...'+os.path.split(self.filename)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filename

    def setValue(self, filename):
        self.filename = str(filename)
        self.label.setText('...' + os.path.split(self.filename)[-1][-20:])    

class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    plots loc and track data onto current window
    ------------------
    
    input:      csv file with x, y positions and track info
    
    variables:  
    
    analysis:   
    
    output:     
    """
    def __init__(self):
        if g.settings['locsAndTracksPlotter'] is None or 'set_track_colour' not in g.settings['locsAndTracksPlotter']:
            s = dict()            
            s['filename'] = '' 
            s['filetype'] = 'flika'   
            s['pixelSize'] = 108                         
            s['set_track_colour'] = False
            g.settings['locsAndTracksPlotter'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
            
        

    def __call__(self, filename, filetype, pixelSize, set_track_colour,  keepSourceWindow=False):
        '''
        '''

        #currently not saving parameter changes on call
        g.settings['locsAndTracksPlotter']['filename'] = filename 
        g.settings['locsAndTracksPlotter']['filetype'] = filetype       
        g.settings['locsAndTracksPlotter']['pixelSize'] = pixelSize 
        g.settings['locsAndTracksPlotter']['set_track_colour'] = set_track_colour      
        
        g.m.statusBar().showMessage("plotting data...")
        return


    def closeEvent(self, event):
        self.clearPlots()
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):      
        self.filename = '' 
        self.filetype = 'flika'   
        self.pixelSize= 108      
        self.plotWindow = None
        self.pathitems = []
        self.useFilteredData = False
        self.useFilteredTracks = False
        self.useMatplotCM = False
        self.gui_reset()        
        s=g.settings['locsAndTracksPlotter']  
        
        #buttons      
        self.plotPointData_button = QPushButton('Plot Points')
        self.plotPointData_button.pressed.connect(self.plotPointData)  
        
        self.hidePointData_button = QPushButton('Toggle Points')
        self.hidePointData_button.pressed.connect(self.hidePointData)         
            
        self.plotTrackData_button = QPushButton('Plot Tracks')
        self.plotTrackData_button.pressed.connect(self.plotTrackData)  
        
        self.clearTrackData_button = QPushButton('Clear Tracks')
        self.clearTrackData_button.pressed.connect(self.clearTracks)  
        
        self.filterData_button = QPushButton('Filter')
        self.filterData_button.pressed.connect(self.filterData)          
        
        self.clearFilterData_button = QPushButton('Clear Filter')
        self.clearFilterData_button.pressed.connect(self.clearFilterData)  

        self.ROIFilterData_button = QPushButton(' Filter by ROI(s)')
        self.ROIFilterData_button.pressed.connect(self.ROIFilterData)  

        self.clearROIFilterData_button = QPushButton('Clear ROI Filter')
        self.clearROIFilterData_button.pressed.connect(self.clearROIFilterData)  
        
        self.saveData_button = QPushButton('Save Tracks')
        self.saveData_button.pressed.connect(self.saveData)         

                         
        #checkbox
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(s['set_track_colour'])
        
        self.matplotCM_checkbox = CheckBox()
        self.matplotCM_checkbox.setChecked(False)   
        self.matplotCM_checkbox.stateChanged.connect(self.setColourMap)

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
        
        self.filterCol_Box = pg.ComboBox()
        self.filtercols = {'None':'None'}
        self.filterCol_Box.setItems(self.filtercols)  

        self.trackColourCol_Box = pg.ComboBox()
        self.trackcolourcols = {'None':'None'}
        self.trackColourCol_Box.setItems(self.trackcolourcols)  
        
        self.colourMap_Box = pg.ComboBox()
        self.colourMaps = dictFromList(pg.colormap.listMaps())
        self.colourMap_Box.setItems(self.colourMaps)         

        self.filterOp_Box = pg.ComboBox()
        self.filterOps = {'=':'==', '<':'<', '>':'>'}
        self.filterOp_Box.setItems(self.filterOps)  
        
        self.filterValue_Box = QLineEdit()     
        
        self.trackDefaultColour_Box = pg.ComboBox()
        self.trackdefaultcolours = {'green': Qt.green, 'red': Qt.red, 'blue': Qt.blue}
        self.trackDefaultColour_Box.setItems(self.trackdefaultcolours)
        
        
        #data file selector
        self.getFile = FileSelector()
        
        #connections
        self.getFile.valueChanged.connect(self.loadData)
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        #self.items.append({'name': 'blank1 ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'filepath ', 'string': '', 'object': self.getFile})    
        self.items.append({'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box})  
        
       
        #self.items.append({'name': 'blank ', 'string': '-------------------------------------------', 'object': None})   
        #self.items.append({'name': 'frameCol', 'string': 'Frame col', 'object': self.frameCol_Box})          
        #self.items.append({'name': 'xCol', 'string': 'X col', 'object': self.xCol_Box})          
        #self.items.append({'name': 'yCol', 'string': 'Y col', 'object': self.yCol_Box})   
        #self.items.append({'name': 'trackCol', 'string': 'Track col', 'object': self.trackCol_Box})   
        self.items.append({'name': 'blank ', 'string': '---- FILTER -----', 'object': None})                  
        self.items.append({'name': 'filterCol', 'string': 'Filter col', 'object': self.filterCol_Box})
        self.items.append({'name': 'filterOp', 'string': 'Operator', 'object': self.filterOp_Box})  
        self.items.append({'name': 'filterValue', 'string': 'Value', 'object': self.filterValue_Box})         
        self.items.append({'name': 'filterData', 'string': '', 'object': self.filterData_button })         
        self.items.append({'name': 'clearFilterData', 'string': '', 'object': self.clearFilterData_button })  

        #self.items.append({'name': 'blank ', 'string': '--- ROI FILTER ----', 'object': None})                  
        self.items.append({'name': 'filterROI', 'string': '', 'object': self.ROIFilterData_button})
        self.items.append({'name': 'clearFilterROI', 'string': '', 'object': self.clearROIFilterData_button})  
        
        
        self.items.append({'name': 'blank ', 'string': '----  PLOT  -----', 'object': None})           
        self.items.append({'name': 'plotPoints', 'string': '', 'object': self.plotPointData_button }) 
        self.items.append({'name': 'hidePoints', 'string': '', 'object': self.hidePointData_button })
        self.items.append({'name': 'trackDefaultColour', 'string': 'Track Default Colour', 'object': self.trackDefaultColour_Box })        
        self.items.append({'name': 'trackColour', 'string': 'Set Track Colour', 'object': self.trackColour_checkbox})           
        self.items.append({'name': 'trackColourCol', 'string': 'Colour by', 'object': self.trackColourCol_Box})
        self.items.append({'name': 'trackColourMap', 'string': 'Colour Map', 'object': self.colourMap_Box})   
        self.items.append({'name': 'matplotClourMap', 'string': 'Use matplot map', 'object': self.matplotCM_checkbox})          
        self.items.append({'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button })         
        self.items.append({'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button })     
        self.items.append({'name': 'saveTracks', 'string': '', 'object': self.saveData_button })  
        
        super().gui()
        ######################################################################
        return



    def loadData(self):
        self.filename = self.getFile.value()
        self.data = pd.read_csv(self.filename)
        

        print('-------------------------------------')
        print('Data loaded (first 5 rows displayed):')
        print(self.data.head())
        print('-------------------------------------')
        
        self.columns = self.data.columns
        self.colDict= dictFromList(self.columns)

        
        self.xCol_Box.setItems(self.colDict)
        self.yCol_Box.setItems(self.colDict)
        self.frameCol_Box.setItems(self.colDict)        
        self.trackCol_Box.setItems(self.colDict)   
        self.filterCol_Box.setItems(self.colDict)  
        self.trackColourCol_Box.setItems(self.colDict)  
        

    def makePointDataDF(self, data):   
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.pixelSize
            df['y'] = data['y [nm]']/self.pixelSize   

        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']

        self.indexDF = self.data.set_index(['x', 'y'])
                     
        return df

    def plotPointsOnStack(self):
            
        points_byFrame = self.points[['frame','x','y']]
    
        #points_byFrame['point_color'] = QColor(g.m.settings['point_color'])
        #points_byFrame['point_size'] = g.m.settings['point_size']
        pointArray = points_byFrame.to_numpy()
        
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        
        
        for pt in pointArray:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(g.m.settings['point_color'])
            #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
            position = [pt[1], pt[2], pointColor, pointSize]    
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()


    def hidePointData(self):
        if self.plotWindow.scatterPlot in self.plotWindow.imageview.ui.graphicsView.items():
            self.plotWindow.imageview.ui.graphicsView.removeItem(self.plotWindow.scatterPlot)
        else:
            self.plotWindow.imageview.addItem(self.plotWindow.scatterPlot)

    def plotPointData(self):
        ### plot point data to current window
        if self.useFilteredData == False:
            self.points = self.makePointDataDF(self.data)
        else:
            self.points = self.makePointDataDF(self.filteredData)
        self.plotWindow = g.win
        self.plotPointsOnStack()

        g.m.statusBar().showMessage('point data plotted to current window') 
        print('point data plotted to current window')    
        return



    def makeTrackDF(self, data):
        if self.filetype_Box.value() == 'thunderstorm':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x [nm]']/self.pixelSize
            df['y'] = data['y [nm]']/self.pixelSize  
            df['track_number'] = data['track_number']

        elif self.filetype_Box.value() == 'flika':
            ######### load FLIKA pyinsight data into DF ############
            df = pd.DataFrame()
            df['frame'] = data['frame'].astype(int)-1
            df['x'] = data['x']
            df['y'] = data['y']
            df['track_number'] = data['track_number']
            
            
            if self.trackColour_checkbox.isChecked():
                if self.useMatplotCM:
                    cm = pg.colormap.getFromMatplotlib(self.colourMap_Box.value()) #cm goes from 0-1, need to scale input values   
                else:    
                    cm = pg.colormap.get(self.colourMap_Box.value()) #cm goes from 0-1, need to scale input values
                
                df['colour'] = cm.mapToQColor(self.data[self.trackColourCol_Box.value()].to_numpy()/max(self.data[self.trackColourCol_Box.value()]))
        
                     
        return df.groupby(['track_number'])


    def clearTracks(self):
        if self.plotWindow is not None and not self.plotWindow.closed:
            for pathitem in self.pathitems:
                self.plotWindow.imageview.view.removeItem(pathitem)
        self.pathitems = []        

    def showTracks(self):
        # clear self.pathitems
        self.clearTracks()
        
        #setup pen
        pen = QPen(self.trackDefaultColour_Box.value(), .4)
        pen.setCosmetic(True)
        pen.setWidth(2)
        
        if self.useFilteredTracks:
            trackIDs = self.filteredTrackIds
            
        else:
           trackIDs = self.trackIDs
        
        for track_idx in trackIDs:
            tracks = self.tracks.get_group(track_idx)
            pathitem = QGraphicsPathItem(self.plotWindow.imageview.view)
            
            if self.trackColour_checkbox.isChecked():
                #print(tracks['colour'].to_list()[0].rgb())
                pen.setColor(tracks['colour'].to_list()[0])
                
            
            pathitem.setPen(pen)
            self.plotWindow.imageview.view.addItem(pathitem)
            self.pathitems.append(pathitem)
            x = tracks['x'].to_numpy()
            y = tracks['y'].to_numpy()  
            #x = pts[:, 1]+.5; y = pts[:,2]+.5
            path = QPainterPath(QPointF(x[0],y[0]))
            for i in np.arange(1, len(x)):
                path.lineTo(QPointF(x[i],y[i]))
            pathitem.setPath(path)


    def plotTrackData(self):
        self.plotWindow = g.win
        ### plot track data to current window
        self.trackIDs = np.unique(self.data['track_number']).astype(np.int)
        
        if self.useFilteredData == False:
            self.tracks = self.makeTrackDF(self.data)
        else:
            self.tracks = self.makeTrackDF(self.filteredData)           
        
        self.showTracks()

        g.m.statusBar().showMessage('track data plotted to current window') 
        print('track data plotted to current window')    
        return



    def filterData(self):
        
        op = self.filterOp_Box.value()
        filterCol = self.filterCol_Box.value()
        dtype = self.data[filterCol].dtype 
        value = float(self.filterValue_Box.text())
        
        
        if op == '==':
            self.filteredData = self.data[self.data[filterCol] == value]
 
        elif op == '<':
            self.filteredData = self.data[self.data[filterCol] < value]
        
        elif op == '>':
             self.filteredData = self.data[self.data[filterCol] > value]           
            
        
        print(self.filteredData.head())
        self.useFilteredData = True
        return


    def clearFilterData(self):
        self.useFilteredData = False
        return

    def getScatterPointsAsQPoints(self):
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        trackIDs = []
        for pt in self.plotWindow.scatterPoints:
            try:
                #print((pt[0][0],pt[0][1]))
                trackIDs.append(self.indexDF.at[(pt[0][0],pt[0][1]), 'track_number'])
            except:
                pass

        
        self.filteredTrackIds = np.unique(trackIDs)
        
        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        


        self.useFilteredData = True
        self.useFilteredTracks = True
        

    def ROIFilterData(self):
        self.roiFilterPoints = []
        self.rois = self.plotWindow.rois
        
        self.oldScatterPoints = self.plotWindow.scatterPoints
        
        for roi in self.rois:
            currentFrame = self.plotWindow.currentIndex
            for i in range(0,self.plotWindow.mt):
                # get ROI shape in coordinate system of the scatter plot
                self.plotWindow.setIndex(i)
                roiShape = roi.mapToItem(self.plotWindow.scatterPlot, roi.shape())
                # Get list of all points inside shape
                selected = [[i, pt.x(), pt.y()] for pt in self.getScatterPointsAsQPoints() if roiShape.contains(pt)]
                self.roiFilterPoints.extend((selected))
            self.plotWindow.setIndex(currentFrame)
        
        
        self.plotWindow.scatterPoints = [[] for _ in np.arange(self.plotWindow.mt)]
        
        
        for pt in self.roiFilterPoints:
            t = int(pt[0])
            if self.plotWindow.mt == 1:
                t = 0
            pointSize = g.m.settings['point_size']
            pointColor = QColor(0,255,0)
            #position = [pt[1]+(.5* (1/pixelSize)), pt[2]+(.5* (1/pixelSize)), pointColor, pointSize]
            position = [pt[1], pt[2], pointColor, pointSize]  
            #print(position)
            self.plotWindow.scatterPoints[t].append(position)
        self.plotWindow.updateindex()

        self.getDataFromScatterPoints()
        
        return

    def clearROIFilterData(self):
        self.plotWindow.scatterPoints = self.oldScatterPoints 
        self.plotWindow.updateindex()
        self.useFilteredData = False
        self.useFilteredTracks = False
        return
    
    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return

    def setColourMap(self):
        if self.matplotCM_checkbox.isChecked():
            self.colourMaps = dictFromList(pg.colormap.listMaps('matplotlib'))
            self.colourMap_Box.setItems(self.colourMaps)  
            self.useMatplotCM = True
        else:
            self.colourMaps = dictFromList(pg.colormap.listMaps())
            self.colourMap_Box.setItems(self.colourMaps) 
            self.useMatplotCM = False
            

    def saveData(self):
        pass
        #TODO!
        return


locsAndTracksPlotter = LocsAndTracksPlotter()
	

if __name__ == "__main__":
    pass


