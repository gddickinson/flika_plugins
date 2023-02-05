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

from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea


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



class TrackWindow(BaseProcess):
    def __init__(self):
        super().__init__()
               
        self.win = pg.GraphicsWindow()
        self.win.resize(600, 300)
        self.win.setWindowTitle('Track Display')
        self.plt1 = self.win.addPlot()  
        
        self.label = pg.LabelItem(justify='right')
        self.plt1.setLabel('left', 'Intensity', units ='Arbitary')
        self.plt1.setLabel('bottom', 'Time', units ='Frames')        
        self.win.addItem(self.label)
        
        self.autoscaleX = True
        self.autoscaleY = True
        

    def update(self, x,y,ID):  
        ## compute standard histogram
        self.plt1.plot(x, y, stepMode=False, brush=(0,0,255,150), clear=True) 
        self.label.setText("<span style='font-size: 12pt'>track ID={}".format(ID))
        
        if self.autoscaleX:
            self.plt1.setXRange(np.min(x),np.max(x),padding=0)
        if self.autoscaleY:
            self.plt1.setYRange(np.min(y),np.max(y),padding=0)
                            
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()

class ChartDock():
    def __init__(self, mainGUI):
        super().__init__()    
        
        self.mainGUI = mainGUI
        
        self.win =QMainWindow()
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        self.win.resize(1000,500)
        self.win.setWindowTitle('Plots')
        
        ## Create docks, place them into the window one at a time.
        self.d1 = Dock("plot options", size=(500, 100))
        self.d2 = Dock("plot", size=(500,400))
        self.d3 = Dock("histogram options", size=(500,100))
        self.d4 = Dock("histogram", size=(500,400))
        
        self.area.addDock(self.d1, 'left') 
        self.area.addDock(self.d3, 'right', self.d1)       
        self.area.addDock(self.d2, 'bottom', self.d1)     

        self.area.addDock(self.d4, 'bottom', self.d3)     
    
        self.w1 = pg.LayoutWidget()
        self.plotOptionlabel = QLabel("Select plot options")     

        self.xlabel = QLabel("x:")  
        self.ylabel = QLabel("y:")  
    
        self.xColSelector = pg.ComboBox()
        self.xcols = {'None':'None'}
        self.xColSelector.setItems(self.xcols)
    
        self.yColSelector = pg.ComboBox()
        self.ycols = {'None':'None'}
        self.yColSelector.setItems(self.ycols)    

        self.plotTypeSelector = pg.ComboBox()
        self.plotTypes= {'scatter':'scatter', 'line':'line'}
        self.plotTypeSelector.setItems(self.plotTypes)  
        self.selectorLabel = QLabel("Plot type")  
        
        self.plot_button = QPushButton('Plot')
        self.plot_button.pressed.connect(self.updatePlot)
        
        self.w1.addWidget(self.plotOptionlabel , row=0, col=0)
        self.w1.addWidget(self.xColSelector, row=1, col=1)
        self.w1.addWidget(self.yColSelector, row=2, col=1)
        self.w1.addWidget(self.xlabel, row=1, col=0)
        self.w1.addWidget(self.ylabel, row=2, col=0) 
        self.w1.addWidget(self.plotTypeSelector, row=3,col=1)
        self.w1.addWidget(self.selectorLabel, row=3,col=0)       
        self.w1.addWidget(self.plot_button, row=4, col=1)         
        
        self.d1.addWidget(self.w1)    
    
        self.w2 = pg.LayoutWidget()
        self.histoOptionlabel = QLabel("Select histogram options") 
        self.collabel = QLabel("col:")  
    
        self.colSelector = pg.ComboBox()
        self.cols = {'None':'None'}
        self.colSelector.setItems(self.cols)
        
        self.histo_button = QPushButton('Plot Histo')
        self.histo_button.pressed.connect(self.updateHisto)
        
        self.w2.addWidget(self.histoOptionlabel , row=0, col=0)
        self.w2.addWidget(self.colSelector, row=1, col=1)
        self.w2.addWidget(self.collabel, row=1, col=0)  
        self.w2.addWidget(self.histo_button, row=2, col=1)         
        
        self.d3.addWidget(self.w2)      
    
        self.w3 = pg.PlotWidget(title="plot")
        self.w3.plot()
        self.w3.setLabel('left', 'y-axis', units ='')
        self.w3.setLabel('bottom', 'x-axis', units ='')  
        self.d2.addWidget(self.w3)    
    
        self.w4 = pg.PlotWidget(title="histogram")
        self.w4.plot()
        self.w4.setLabel('left', '# of observations', units ='')
        self.w4.setLabel('bottom', 'value', units ='')          
        
        self.d4.addWidget(self.w4)      

    def updatePlot(self):
        self.w3.clear()
        
        if self.mainGUI.useFilteredData == False:
            x = self.mainGUI.data[self.xColSelector.value()]
            y = self.mainGUI.data[self.yColSelector.value()] 
        else:
            x = self.mainGUI.filteredData[self.xColSelector.value()]
            y = self.mainGUI.filteredData[self.yColSelector.value()]             

        if self.plotTypeSelector.value() == 'line':
            self.w3.plot(x, y, stepMode=False, brush=(0,0,255,150), clear=True) 
        elif self.plotTypeSelector.value() == 'scatter':
            self.w3.plot(x, y,
                         pen = None,
                         symbol='o',
                         symbolPen=pg.mkPen(color=(0, 0, 255), width=0),                                      
                         symbolBrush=pg.mkBrush(0, 0, 255, 255),
                         symbolSize=7)    
            
        
        self.w3.setLabel('left', self.yColSelector.value(), units = None)
        self.w3.setLabel('bottom', self.xColSelector.value(), units = None)             
            
        return
    
    def updateHisto(self):
        self.w4.clear()        
        if self.mainGUI.useFilteredData == False:
            vals = self.mainGUI.data[self.colSelector.value()]
        else:
            vals = self.mainGUI.filteredData[self.colSelector.value()]            

        start=0
        end=np.max(vals)
        n=100

        y,x = np.histogram(vals, bins=np.linspace(start, end, n))     
        self.w4.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150), clear=True) 
        self.w4.setLabel('bottom', self.colSelector.value(), units = None)     
        return

    
    def show(self):
        self.win.show()
    
    def close(self):
        self.win.close()

    def hide(self):
        self.win.hide()
    
    

class LocsAndTracksPlotter(BaseProcess_noPriorWindow):
    """
    plots loc and track data onto current window
    ------------------
    
    input: csv file with x, y positions and track info
    
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
        
        self.selectedTrack = None
        self.displayTrack = None
        
        self.chartWindow = None
        self.displayCharts = False
        
        self.trackWindow = TrackWindow()
        self.trackWindow.hide()
        
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
        
        self.showCharts_button = QPushButton('Show Charts')
        self.showCharts_button.pressed.connect(self.toggleCharts)          

                         
        #checkbox
        self.trackColour_checkbox = CheckBox()
        self.trackColour_checkbox.setChecked(s['set_track_colour'])
        
        self.matplotCM_checkbox = CheckBox() 
        self.matplotCM_checkbox.stateChanged.connect(self.setColourMap)
        self.matplotCM_checkbox.setChecked(False)  

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
        self.getFile = FileSelector(filetypes='*.csv')
        
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
        #self.items.append({'name': 'plotPoints', 'string': '', 'object': self.plotPointData_button }) 
        self.items.append({'name': 'hidePoints', 'string': '', 'object': self.hidePointData_button })
        self.items.append({'name': 'trackDefaultColour', 'string': 'Track Default Colour', 'object': self.trackDefaultColour_Box })        
        self.items.append({'name': 'trackColour', 'string': 'Set Track Colour', 'object': self.trackColour_checkbox})           
        self.items.append({'name': 'trackColourCol', 'string': 'Colour by', 'object': self.trackColourCol_Box})
        self.items.append({'name': 'trackColourMap', 'string': 'Colour Map', 'object': self.colourMap_Box})   
        self.items.append({'name': 'matplotClourMap', 'string': 'Use matplot map', 'object': self.matplotCM_checkbox})          
        self.items.append({'name': 'plotTracks', 'string': '', 'object': self.plotTrackData_button })         
        self.items.append({'name': 'clearTracks', 'string': '', 'object': self.clearTrackData_button })     
        self.items.append({'name': 'saveTracks', 'string': '', 'object': self.saveData_button })  
        self.items.append({'name': 'showCharts', 'string': '', 'object': self.showCharts_button })          
        
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
        
        self.plotPointData()
        
        
        

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


        print('tracks to plot {}'.format(trackIDs))
        
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
        ### plot track data to current window
        self.plotWindow = g.win
        
        if self.useFilteredData == False:            
            self.trackIDs = np.unique(self.data['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.data)
        else:
            self.trackIDs = np.unique(self.filteredData['track_number']).astype(np.int)
            self.tracks = self.makeTrackDF(self.filteredData)           
        
        self.showTracks()
        
        #get mouse events from plot window
        self.plotWindow.imageview.scene.sigMouseMoved.connect(self.updateTrackSelector)
        
        #use key press to select tracks to display
        self.plotWindow.keyPressSignal.connect(self.selectTrack)
        
        self.trackWindow.show()
        
        g.m.statusBar().showMessage('track data plotted to current window') 
        print('track data plotted to current window')    
        return


    def updateTrackSelector(self, point):
        pos =  self.plotWindow.imageview.getImageItem().mapFromScene(point)

        #print('x: {}, y: {}'.format(pos.x(),pos.y()))
                
        for i, path in enumerate(self.pathitems):
            if path.contains(pos):
                #print('mouse at {}'.format(pos))
                #print('track ID:  {}'.format(self.trackIDs[i]))
                #print('track pos {}{}'.format(path.pos().x(),path.pos().y()))
                self.selectedTrack = self.trackIDs[i]                 
                

    def selectTrack(self,ev):
        if ev.key() == Qt.Key_T:
            if self.selectedTrack != self.displayTrack:
                self.displayTrack = self.selectedTrack    
                #print(self.selectedTrack)
                trackData = self.data[self.data['track_number'] == int(self.displayTrack)]
                x = trackData['frame'].to_numpy()
                y = trackData['intensity'].to_numpy()                
                self.trackWindow.update(x,y,self.displayTrack)
        


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
        
        
        self.plotPointData()
        
        return


    def clearFilterData(self):
        self.useFilteredData = False
        
        self.plotPointData()
        return

    def getScatterPointsAsQPoints(self):
        qpoints = np.array(self.plotWindow.scatterPlot.getData()).T
        qpoints = [QPointF(pt[0],pt[1]) for pt in qpoints]
        return qpoints


    def getDataFromScatterPoints(self):
        trackIDs = []
        
        flat_ptList = [pt for sublist in self.plotWindow.scatterPoints for pt in sublist]
        
        for pt in flat_ptList:            
            #print('point x: {} y: {}'.format(pt[0][0],pt[0][1]))

            ptFilterDF = self.data[(self.data['x']==pt[0]) & (self.data['y']==pt[1])]
            
            trackIDs.extend(ptFilterDF['track_number'])

        
        self.filteredTrackIds = np.unique(trackIDs)

        self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        
        #self.filteredData = self.data[self.data['track_number'].isin(self.filteredTrackIds)]
        
        #self.filteredTrackIds = np.unique(self.filteredData['track_number'])

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
        if self.useFilteredData == False:
            print('filter data first')
            g.alert('Filter data first')
            return
        
        #set export path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")        

        #write file
        try:
            # writing the data into the file 
            self.filteredData.to_csv(savePath)
            
            print('Filtered data saved to: {}'.format(savePath))
        except BaseException as e:
            print(e)
            print('Export of filtered data failed')


    def toggleCharts(self):
        if self.chartWindow == None:
            #create plot window
            self.chartWindow = ChartDock(self)
            self.chartWindow.xColSelector.setItems(self.colDict)
            self.chartWindow.yColSelector.setItems(self.colDict)
            self.chartWindow.colSelector.setItems(self.colDict)    
            
            self.chartWindow.xcols = self.colDict
            self.chartWindow.ycols = self.colDict            
            self.chartWindow.cols = self.colDict
            
            
        if self.displayCharts == False:
            self.chartWindow.show()
            self.displayCharts = True
            self.showCharts_button.setText('Hide Charts')
        else:
            self.chartWindow.hide()
            self.displayCharts = False   
            self.showCharts_button.setText('Show Charts')

locsAndTracksPlotter = LocsAndTracksPlotter()
	

if __name__ == "__main__":
    pass


