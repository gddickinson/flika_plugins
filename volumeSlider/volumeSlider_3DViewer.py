import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
#from flika.utils.io import tifffile
#from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
#import time
import os
#from os import listdir
#from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
#from copy import deepcopy
#from numpy import moveaxis
#from skimage.transform import rescale
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from qtpy.QtCore import Signal
from scipy.ndimage import gaussian_filter

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .helperFunctions import *
from .pyqtGraph_classOverwrites import *
from .scalebar_classOverwrite import Scale_Bar_volumeView
from .histogramExtension import HistogramLUTWidget_Overlay
from .texturePlot import *
from .scatterPlot import *
from .overlayOptions import *
from .exportIMS import *

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm
from pathlib import Path

#########################################################################################
#############                  slice viewer (3D Display)                #################
#########################################################################################
class SliceViewer(BaseProcess):

    def __init__(self, viewerInstance, A):
        super().__init__()
        self.app = QtWidgets.QApplication([])
        
        self.viewer = viewerInstance
        self.overlayFlag = False

        self.shift_factor = self.viewer.dialogbox.shiftFactor
        self.interpolate = False
        self.theta = self.viewer.dialogbox.theta
        self.inputArrayOrder = self.viewer.getInputArrayOrder()
        self.displayArrayOrder = self.viewer.getDisplayArrayOrder()

        self.trim_last_frame = self.viewer.dialogbox.trim_last_frame

        print('shift factor: '+ str(self.shift_factor))
        print('theta: ' + str(self.theta))
        print('input Array Order: '+ str(self.inputArrayOrder))
        print('display Array Order: ' + str(self.displayArrayOrder))
        g.m.statusBar().showMessage("3D viewer starting...")

        #3D plot params
        self.prob = 0.001
        self.threshold_3D = 300
        self.plotCube = False
        self.hideAxis = True

        if self.trim_last_frame:
            self.originalData = A[:, :-1, :, :]
        else:
            self.originalData = A

        self.originalDataShape = self.originalData.shape
        self.nVols = self.originalDataShape[1]
        self.nSteps = self.originalDataShape[0]

        self.originalData= perform_shear_transform(self.originalData, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
        self.dataShape = self.originalData.shape

        self.height = self.dataShape[3]
        self.width = self.dataShape[2]

        self.currentVolume = 0

        self.data = self.originalData[:,0,:,:]

        ## Create window with 4 docks
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1000,800)
        self.win.setWindowTitle('Lightsheet 3D display')

        #create Dock area
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        #define docks
        self.dock1 = Dock("Top View - Max Projection", size=(500,400))
        self.dock2 = Dock("X Slice", size=(500,400), closable=True)
        self.dock3 = Dock("Y Slice", size=(500,400), closable=True)
        self.dock4 = Dock("Free ROI", size=(500,400), closable=True)
        self.dock5 = Dock("Time Slider", size=(950,50))
        self.dock6 = Dock("Z Slice", size=(500,400), closable=True)
        self.dock7 = Dock("Quick Buttons",size=(50,50))

        #add docks to area
        self.area.addDock(self.dock1, 'left')                   ## place d1 at left edge of dock area
        self.area.addDock(self.dock2, 'right')                  ## place d2 at right edge of dock area
        self.area.addDock(self.dock3, 'bottom', self.dock1)     ## place d3 at bottom edge of d1
        self.area.addDock(self.dock4, 'bottom', self.dock2)     ## place d4 at bottom edge of d2
        self.area.addDock(self.dock5, 'bottom')                 ## place d4 at bottom edge of d2
        self.area.addDock(self.dock6, 'below', self.dock4)      ## tab below d4
        self.area.addDock(self.dock7, 'right', self.dock5)      ## d7 right of d5

        #initialise image widgets
        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.imv3 = pg.ImageView()
        self.imv4 = pg.ImageView()
        self.imv6 = pg.ImageView()

        self.imageWidgits = [self.imv1,self.imv2,self.imv3,self.imv4,self.imv6]

        #add image widgets to docks
        self.dock1.addWidget(self.imv1)
        self.dock2.addWidget(self.imv3)
        self.dock3.addWidget(self.imv2)
        self.dock4.addWidget(self.imv4)
        self.dock6.addWidget(self.imv6)

        self.imv1.scene.sigMouseMoved.connect(self.mouseMoved_1)
        self.imv2.scene.sigMouseMoved.connect(self.mouseMoved_2)
        self.imv3.scene.sigMouseMoved.connect(self.mouseMoved_3)
        self.imv4.scene.sigMouseMoved.connect(self.mouseMoved_4)
        self.imv6.scene.sigMouseMoved.connect(self.mouseMoved_6)

        #hide dock title-bars at start
        self.dock5.hideTitleBar()
        self.dock7.hideTitleBar()

        #add menu functions
        self.state = self.area.saveState()
        self.menubar = self.win.menuBar()
        
        #=================================================================================================================
        self.fileMenu1 = self.menubar.addMenu('&Options')
        
        self.resetLayout = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Reset Layout')
        setMenuUp(self.resetLayout,self.fileMenu1,shortcut='Ctrl+R',statusTip='Reset Layout',connection=self.reset_layout)

        self.showTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Titles')
        setMenuUp(self.showTitles,self.fileMenu1,shortcut='Ctrl+G',statusTip='Show Titles',connection=self.show_titles)

        self.hideTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Titles')
        setMenuUp(self.hideTitles,self.fileMenu1,shortcut='Ctrl+H',statusTip='Hide Titles',connection=self.hide_titles)        

        self.hideCursors = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Cursors')
        setMenuUp(self.hideCursors,self.fileMenu1,shortcut=None,statusTip='Hide Cursors',connection=self.hide_cursors)         

        self.showCursors = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Cursors')
        setMenuUp(self.showCursors,self.fileMenu1,shortcut=None,statusTip='Show Cursors',connection=self.show_cursors)   

        self.winOptions = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Window Options')
        setMenuUp(self.winOptions,self.fileMenu1,shortcut=None,statusTip='Window Options',connection=self.win_options)                 
        #================================================================================================================
        self.fileMenu2 = self.menubar.addMenu('&3D Plot')

        self.plot = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot')
        setMenuUp(self.plot,self.fileMenu2,shortcut='Ctrl+P',statusTip='3D Plot',connection=self.plot3D)

        self.plotOptions = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot Options')
        setMenuUp(self.plotOptions,self.fileMenu2,shortcut=None,statusTip='3D Plot Options',connection=self.plot3D_options)        

        #================================================================================================================
        self.fileMenu3 = self.menubar.addMenu('&Texture Plot')
        
        self.texturePlot = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Texture Plot')
        setMenuUp(self.texturePlot,self.fileMenu3,shortcut=None,statusTip='Texture Plot',connection=(lambda: self.plotTexture())) 

        self.texturePlotControl = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Texture Plot Control')
        setMenuUp(self.texturePlotControl,self.fileMenu3,shortcut=None,statusTip='Texture Plot Control',connection=self.plotTexture_control)
        #================================================================================================================
        self.fileMenu4 = self.menubar.addMenu('&Export')
        
        self.exportFlika = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export to flika')
        setMenuUp(self.exportFlika,self.fileMenu4,shortcut=None,statusTip='Export to Flikal',connection=self.exportDialog)
        
        self.export = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export Volume to Array')
        setMenuUp(self.export,self.fileMenu4,shortcut='Ctrl+E',statusTip='Export to Array',connection=self.exportCurrentVolToArray) 

        self.exportIMS = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export to Imaris')
        setMenuUp(self.exportIMS,self.fileMenu4,shortcut=None,statusTip='Export to Imaris',connection=self.exportIMSDialog)
                   
        #================================================================================================================                
        self.fileMenu5 = self.menubar.addMenu('&Overlay')
        
        self.overlayArray = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Overlay (from Array)')
        setMenuUp(self.overlayArray,self.fileMenu5,shortcut='Ctrl+O',statusTip='OverlayArray',connection=self.overlayArray_start)          
        
        self.overlayToggle = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Toggle Overlay')
        setMenuUp(self.overlayToggle,self.fileMenu5,shortcut='Ctrl+T',statusTip='OverlayOff',connection=self.toggleOverlay)         
        
        self.overlayArrayWin = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Overlay Options')
        setMenuUp(self.overlayArrayWin,self.fileMenu5,shortcut=None,statusTip='Overlay Options',connection=self.overlayOptions)         
        
        self.overlayScale_win1 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Top)')
        setMenuUp(self.overlayScale_win1,self.fileMenu5,shortcut=None,statusTip='Overlay Scale Bar',connection=(lambda: self.overlayScaleOptions(1)))  
        
        self.overlayScale_win2 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Y)')
        setMenuUp(self.overlayScale_win2,self.fileMenu5,shortcut=None,statusTip='Overlay Scale Bar',connection=(lambda: self.overlayScaleOptions(2)))               
        
        self.overlayScale_win3 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (X)')
        setMenuUp(self.overlayScale_win3,self.fileMenu5,shortcut=None,statusTip='Overlay Scale Bar',connection=(lambda: self.overlayScaleOptions(3)))  

        self.overlayScale_win6 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Z)')
        setMenuUp(self.overlayScale_win6,self.fileMenu5,shortcut=None,statusTip='Overlay Scale Bar',connection=(lambda: self.overlayScaleOptions(6)))     
        #================================================================================================================
        self.fileMenu6 = self.menubar.addMenu('&Filters')
        
        self.gaussian = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Gaussian')
        setMenuUp(self.gaussian,self.fileMenu6,shortcut='Ctrl+F',statusTip='Gaussian',connection=self.gaussianOptions)                
        #================================================================================================================        
        self.fileMenu7 = self.menubar.addMenu('&Quit')
        
        self.quit = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Quit')
        setMenuUp(self.quit,self.fileMenu7,shortcut='Ctrl+Q',statusTip='Quit',connection=self.close)                
        #================================================================================================================  

        #add time slider
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.slider1, minimum=0, maximum=self.nVols-1, tickInterval=1, singleStep=1, value=0)
        self.dock5.addWidget(self.slider1)

        #add buttons to 'quick button' dock
        self.quickOverlayButton = QtWidgets.QPushButton("Overlay") 
        self.dock7.addWidget(self.quickOverlayButton)
        self.quickOverlayButton.clicked.connect(self.quickOverlay)

        #display window
        self.win.show()

        #define single line roi
        self.roi1 = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
        self.imv1.addItem(self.roi1)

        #define crosshair rois
        self.dash = mkPen('w', width=1,style=QtCore.Qt.DashLine)
        self.roi2 = pg.LineSegmentROI([[0, int(self.height/2)], [self.width, int(self.height/2)]], pen='y', maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height))
        self.roi2b = pg.LineSegmentROI([[0, int(self.height/2)], [self.width, int(self.height/2)]], pen=self.dash, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)
        self.roi2c = pg.LineSegmentROI([[0, int(self.height/2)], [int(self.width/6), int(self.height/2)]], pen=self.dash, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)       
        self.imv1.addItem(self.roi2)
        self.imv6.addItem(self.roi2b)
        self.imv3.addItem(self.roi2c)        

        self.roi3 = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen='y', maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0))
        self.roi3b = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen=self.dash, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.roi3c = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen=self.dash, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.imv1.addItem(self.roi3)
        self.imv6.addItem(self.roi3b)
        self.imv2.addItem(self.roi3c)

        #define crosshair center roi
        self.roiCenter = pg.CircleROI([int(self.width/2)-10,int(self.height/2)-10], [20, 20], pen=(4,9))
        self.roiCenter.setPen(None)
        self.imv1.addItem(self.roiCenter)

        #define height indicator rois
        self.dash2 = mkPen('r', width=1,style=QtCore.Qt.DashLine)
        self.roi4 = pg.LineSegmentROI([[0, 0], [self.width, 0]], pen=self.dash2, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)
        self.imv2.addItem(self.roi4)
        self.roi5 = pg.LineSegmentROI([[0, 0], [0, self.height]], pen=self.dash2, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.imv3.addItem(self.roi5)

        #self.win.closeEvent = lambda f: self.app.closeAllWindows()
        #self.win.closeEvent = lambda f: self.app.quitOnLastWindowClosed()
        self.win.closeEvent = lambda f: self.closeEvent(f) 

        #hide default imageview buttons
        def hideButtons(imv):
            imv.ui.roiBtn.hide()
            imv.ui.menuBtn.hide()

        for imv in self.imageWidgits:
            hideButtons(imv)

        #disconnect roi handles
        def disconnectHandles(roi):
            handles = roi.getHandles()
            if len(handles) == 2:
                handles[0].disconnectROI(roi)
                handles[1].disconnectROI(roi)
                handles[0].currentPen = mkPen(None)
                handles[1].currentPen = mkPen(None)
                handles[0].pen = mkPen(None)
                handles[1].pen = mkPen(None)
            elif len(handles) == 1:
                #handles[0].disconnectROI(roi)
                #handles[0].currentPen = mkPen(None)
                #handles[0].pen = mkPen(None)
                roi.removeHandle(-1)

        self.MainROIList = [self.roi2,self.roi3,self.roi2b,self.roi2c,self.roi3b,self.roi3c,self.roi4,self.roi5,self.roiCenter]

        for roi in self.MainROIList:
            disconnectHandles(roi)
       
        #set filter flags
        self.gaussianFlag = False
        self.sigma = 10 #is divided by 10 during filter (this helps with the slider bar)
        self.gaussData = None

        #add max projection data to main window
        self.imv1.setImage(self.maxProjection(self.data)) #display topview (max of slices)

        #add sliceable data to side window
        self.imv6.setImage(self.data)

        #connect roi updates
        self.roi1.sigRegionChanged.connect(lambda: self.update(1))
        self.roi2.sigRegionChanged.connect(lambda: self.update(2))
        self.roi3.sigRegionChanged.connect(lambda: self.update(3))

        self.imv6.sigTimeChanged.connect(lambda: self.update(6))

        self.roi2.sigRegionChangeFinished.connect(self.update_center_fromLines)
        self.roi3.sigRegionChangeFinished.connect(self.update_center_fromLines)        
        self.roiCenter.sigRegionChanged.connect(self.update_center)

        #initial update to populate roi windows
        self.update(1)
        self.update(2)
        self.update(3)

        #autolevel roi windows (not main) at start
        for imv in self.imageWidgits[1:]:
            imv.autoLevels()

        self.slider1.valueChanged.connect(self.timeUpdate)

        #initialize time index for z-slice
        self.index6 = 0        
        #correct roi4 position
        self.update(6)
        
        self.OverlayLUT = 'inferno'
        self.OverlayMODE = QtGui.QPainter.CompositionMode_SourceOver
        self.OverlayOPACITY = 0.5
        self.useOverlayLUT = False
        
        self.gradientPreset = 'grey' #  'thermal','flame','yellowy','bipolar','spectrum','cyclic','greyclip','grey'
        self.usePreset = False
        self.gradientState_1 = None
        self.gradientState_2 = None
        self.gradientState_3 = None
        self.gradientState_4 = None
        self.gradientState_6 = None   
        self.useSharedState = False
        self.sharedState = None
        self.sharedLevels = None
               
        #init overlay levels 
        self.levels_1 = self.imv1.getHistogramWidget().getLevels()
        self.levels_2 = self.imv2.getHistogramWidget().getLevels()
        self.levels_3 = self.imv3.getHistogramWidget().getLevels()
        self.levels_4 = self.imv4.getHistogramWidget().getLevels()     
        self.levels_6 = self.imv6.getHistogramWidget().getLevels()          

        self.overlayFlag = False
        self.overlayArrayLoaded = False
        
        #create lists of objects
        self.dockList = [self.dock1, self.dock2, self.dock3, self.dock4, self.dock6]
        self.roiList = [self.roi2b, self.roi2c, self.roi3b, self.roi3c, self.roi4, self.roi5]

        #link TOP overlay histgramLUT to other windows
        self.imv1.getHistogramWidget().item.sigLevelsChanged.connect(self.setMainLevels)
        self.imv1.getHistogramWidget().item.sigLookupTableChanged.connect(self.setMainLUTs)        
        self.histogramsLinked = True


    #define update calls for each roi
    def update(self, win):
        if win ==1:
            levels = self.imv4.getHistogramWidget().getLevels()
            self.d1 = self.roi1.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
            self.imv4.setImage(self.d1, autoRange=False, autoLevels=False, levels=levels)
            
            if self.overlayFlag:
                self.runOverlayUpdate(1)
                self.runOverlayUpdate(4)            

        elif win ==2:
            levels = self.imv2.getHistogramWidget().getLevels()
            self.d2 = np.rot90(self.roi2.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2)), axes=(1,0))
            self.imv2.setImage(self.d2, autoRange=False, autoLevels=False, levels=levels)
            self.roi2b.setPos(self.roi2.pos(), finish=False)
            self.roi2c.setPos(self.roi2.pos(), finish=False)    
            
            if self.overlayFlag:
                self.runOverlayUpdate(2)

        elif win ==3:
            levels = self.imv3.getHistogramWidget().getLevels()
            self.d3 = self.roi3.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
            self.imv3.setImage(self.d3, autoRange=False, autoLevels=False, levels=levels)
            self.roi3b.setPos(self.roi3.pos(),finish=False)
            self.roi3c.setPos(self.roi3.pos(),finish=False) 
            
            if self.overlayFlag:
                self.runOverlayUpdate(3)

        elif win ==6:
            #self.index6 = self.imv6.currentIndex
            roi4_x, roi4_y = self.roi4.pos()
            roi5_x, roi5_y = self.roi5.pos()
            self.roi4.setPos((roi4_x, self.imv2.imageItem.height()-self.imv6.currentIndex)) #check this is starting at right end
            self.roi5.setPos((self.imv6.currentIndex, roi5_y))
    
            if self.overlayFlag:
                self.runOverlayUpdate(6)

    def update_center_fromLines(self):
        #move center roi after cursor lines moved
        self.roiCenter.setPos((self.roi3.pos()[0]+int(self.width/2)-10,self.roi2.pos()[1]+int(self.height/2)-10), finish=False)  
        self.roiCenter.setPos((self.roi3.pos()[0]+int(self.width/2)-10,self.roi2.pos()[1]+int(self.height/2)-10), finish=False)  
        
    def update_center(self):
        #move cursor lines as center roi is moved
        self.roi2.setPos((self.roi2.pos()[0],self.roiCenter.pos()[1]-int(self.height/2)+10), finish=False)
        self.roi3.setPos((self.roiCenter.pos()[0]-int(self.width/2)+10,self.roi3.pos()[1]), finish=False)

    def runOverlayUpdate(self, win):
            self.updateOverlayLevels()
            self.overlayOff_temp(win)
            self.overlayFlag = True
            self.overlayUpdate(win) 
            return

    def updateAllMainWins(self):
        for win in [1,2,3,6]:
            self.update(win)
        self.update_center()

    #connect time slider
    def timeUpdate(self,value):
        self.currentVolume = value
        self.index6 = self.imv6.currentIndex
        levels1 = self.imv1.getHistogramWidget().getLevels()
        levels6 = self.imv6.getHistogramWidget().getLevels()

        if self.gaussianFlag:
            self.data = self.gaussData[:,value,:,:]
        else:
            self.data = self.originalData[:,value,:,:]
            
        self.imv1.setImage(self.maxProjection(self.data),autoRange=False, levels=levels1)
        self.imv6.setImage(self.data,autoRange=False, levels=levels6)
        self.imv6.setCurrentIndex(self.index6)
        for win in [1,2,3,6]:
            self.update(win)
        return

    def reset_layout(self):
        self.area.restoreState(self.state)

    def hide_titles(self,_):
        for dock in self.dockList:
            dock.hideTitleBar()

    def show_titles(self):
        for dock in self.dockList:
            dock.showTitleBar()
        return

    def hide_cursors(self):
        for roi in self.roiList:
            roi.setPen(None)
        return

    def show_cursors(self):
        self.roi2b.setPen(self.dash)
        self.roi2c.setPen(self.dash)        
        self.roi3b.setPen(self.dash)
        self.roi3c.setPen(self.dash)        
        self.roi4.setPen(self.dash2)
        self.roi5.setPen(self.dash2)
        return

    def maxProjection(self,data):
        return np.max(data,axis=0)

    def plot3D(self):
        vol_downSample = copy.deepcopy(self.data)

        #downsample data
        mask = np.random.choice([False, True], vol_downSample.shape, p=[self.prob, 1-self.prob])
        vol_downSample[mask] = 0

        ##z,x,y = vol.nonzero()
        z,x,y =(vol_downSample > self.threshold_3D).nonzero()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c= 'red', s=1)

        vol_corners = getCorners(self.data)
        z_c,x_c,y_c = vol_corners.nonzero()

        ax.scatter(x_c, y_c, -z_c, zdir='z', c= 'green', s=5)

        x_min,x_max,y_min,y_max,z_min,z_max = getDimensions(self.data)
        maxDim = getMaxDimension(self.data)

        ax.set_xlim(0, maxDim)
        ax.set_ylim(0, maxDim)
        ax.set_zlim(0, -maxDim)

        ax.view_init(0,180)

        #hide axis
        if self.hideAxis:
            ax.axis('off')

        plt.draw()

        #plot array outline
        outline = [(x_min,y_min,z_min),(x_min,y_max,z_min),(x_max,y_min,z_min),(x_min,y_min,-z_max)]

        # plot array shape
        if self.plotCube:
            plot_cube(ax, outline)

        fig.show()

    def plot3D_options(self):
        self.plot3D_optionWin = plot3D_options(self.viewer, self.prob,self.threshold_3D)
        self.plot3D_optionWin.show()
        return

    def plotTexture(self):
        self.texturePlot = plotTexture(self.data)
        self.texturePlot.w.show()

    def plotTexture_control(self):
        X,Y,Z = self.texturePlot.getSliceValues()
        self.textureControl = textureDialog_win(self.viewer, self.texturePlot.getShape(), X, Y, Z)
        self.textureControl.show()
        return

    def close(self):
        try:
            self.roi1.sigRegionChanged.disconnect(self.update)
        except:
            pass
        try:        
            self.roi2.sigRegionChanged.disconnect(self.update)
        except:
            pass
        try:        
            self.roi3.sigRegionChanged.disconnect(self.update)
        except:
            pass            
        
        for imv in self.imageWidgits:
            imv.close()

        try:        
            self.exportDialogWin.close()
        except:
            pass
                
        return

    def closeEvent(self, event):
        self.close()
        event.accept()
        self.app.closeAllWindows()
        self.app.quitOnLastWindowClosed()

    def setProb(self, prob):
        self.prob = prob
        return

    def setThreshold(self, thresh):
        self.threshold_3D = thresh
        return

    def setPlotCube(self, flag):
        self.plotCube = flag
        return

    def setPlotAxis(self, flag):
        self.hideAxis = flag
        return

    def exportDialog(self):
        self.exportDialogWin = exportDialog_win(self.viewer)
        self.exportDialogWin.show()
        return

    def exportIMSDialog(self):
        self.exportIMSdialogWin = exportIMSdialog_win(self.viewer, self.originalData)
        self.exportIMSdialogWin.show()
        return


    def exportCurrentVolToArray(self):
        arraySavePath = QtWidgets.QFileDialog.getSaveFileName(self.win,'Save File', os.path.expanduser("~/Desktop"), 'Numpy array (*.npy)')
        arraySavePath = str(arraySavePath[0])
        self.viewer.savePath = arraySavePath
        self.viewer.exportArray(vol=self.currentVolume)
        return

    def getWin(self,win):
        '''Sends window data to Flika (for selected windows time volume)'''
        if win == 'X':
            return self.data.swapaxes(0,1)
        elif win =='Y':
            return self.data.swapaxes(2,0)
        elif win =='Z':
            return self.data.swapaxes(1,2)
        elif win =='Top':
            return self.data
       # elif win =='Free':
       #     return self.roi3.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))


    def getTimeSeries(self,win):
        '''Sends time series data to Flika (for selected view)''' 
        timeSeries = []
        nVols = self.originalDataShape[1]
        
        if win == 'X':
            for i in range(nVols):
                data = self.originalData[:,i,:,:]
                timeSeries.append(self.maxProjection(data.swapaxes(0,1)))            
        
        elif win =='Y':
            for i in range(nVols):
                data = self.originalData[:,i,:,:]
                timeSeries.append(self.maxProjection(data.swapaxes(2,0)))            
        
        elif win =='Z':
            for i in range(nVols):
                data = self.originalData[:,i,:,:]
                timeSeries.append(self.maxProjection(data.swapaxes(1,2)))            
        
        elif win =='Top':
            for i in range(nVols):
                data = self.originalData[:,i,:,:]
                timeSeries.append(self.maxProjection(data))
        
        return np.array(timeSeries)


    def mouseMoved(self, point, imv):
        '''mouseMoved(self,point)
        Event handler function for mouse movement.
        '''
        point=imv.getImageItem().mapFromScene(point)
        self.point = point
        self.x = point.x()
        self.y = point.y()
        image=imv.getImageItem().image
        if self.x < 0 or self.y < 0 or self.x >= image.shape[0] or self.y >= image.shape[1]:
            pass# if we are outside the image
        else:
            z=imv.currentIndex
            value=image[int(self.x),int(self.y)]
            g.m.statusBar().showMessage('x={}, y={}, z={}, value={}'.format(int(self.x),int(self.y),z,value))

    def mouseMoved_1(self,point):
        self.mouseMoved(point,self.imv1)
        return

    def mouseMoved_2(self,point):
        self.mouseMoved(point,self.imv2)
        return

    def mouseMoved_3(self,point):
        self.mouseMoved(point,self.imv3)
        return

    def mouseMoved_4(self,point):
        self.mouseMoved(point,self.imv4)
        return

    def mouseMoved_6(self,point):
        self.mouseMoved(point,self.imv6)
        return

    def overlayArray_start(self):
        #get array data
        A_path = open_file_gui(directory=os.path.expanduser("~/Desktop"),filetypes='*.npy')
        g.m.statusBar().showMessage("Importing Array: " + A_path)
        self.A_overlay = np.load(str(A_path))
        #perform transform
        self.A_overlay= perform_shear_transform(self.A_overlay, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
        #set flags
        self.overlayFlag = True      
        self.overlayArrayLoaded = True
        #update overlay
        self.overlayUpdate(0)
        #link TOP overlay histgramLUT to other windows
        self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)
        self.bgItem_imv1.hist_luttt.item.sigLookupTableChanged.connect(self.setOverlayLUTs)        
        return


    def getLUT(self, lutNAME = "nipy_spectral"):
        colormap = cm.get_cmap(lutNAME)  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        return lut

    def overlay(self, overlayImage, imv, levels, gradientState):
        bgItem = pg.ImageItem()        
        bgItem.setImage(overlayImage, autoRange=False, autoLevels=False, levels=levels,opacity=self.OverlayOPACITY)
        bgItem.setCompositionMode(self.OverlayMODE)
        imv.view.addItem(bgItem)

        bgItem.hist_luttt = HistogramLUTWidget(fillHistogram = False)
        
        if gradientState == None or self.usePreset == True:
            bgItem.hist_luttt.item.gradient.loadPreset(self.gradientPreset)
        else:
            
            if self.useSharedState:
                bgItem.hist_luttt.item.gradient.restoreState(self.sharedState)
                levels = self.sharedLevels

            else:    
                bgItem.hist_luttt.item.gradient.restoreState(gradientState)
        
        bgItem.hist_luttt.setMinimumWidth(110)
        bgItem.hist_luttt.setImageItem(bgItem)

        #bgItem.hist_luttt.item.levelMode = 'rgba'
        
        #unlinked LUT as placeholder to stop image resizing on update
        state = bgItem.hist_luttt.item.gradient.saveState()
        bgItem.blank_luttt = HistogramLUTWidget_Overlay()
        bgItem.blank_luttt.item.gradient.restoreState(state)        
        
        imv.ui.gridLayout.addWidget(bgItem.blank_luttt, 0, 4, 1, 4)       
        imv.ui.gridLayout.addWidget(bgItem.hist_luttt, 0, 4, 1, 4)        
        bgItem.setLevels(levels)
        bgItem.hist_luttt.item.setLevels(levels[0],levels[1])
        
        if self.useOverlayLUT:
            bgItem.hist_luttt.setLUT(self.getLUT(lutNAME = self.OverlayLUT))
        
        return bgItem



    def overlayOff_temp(self, win):
        def overlay_hide_temp(bgItem,imv):
            bgItem.hist_luttt.hide()
            imv.ui.gridLayout.removeWidget(bgItem.hist_luttt)
            imv.view.removeItem(bgItem)
            return        
        
        if self.overlayFlag:
            if win == 1:
                overlay_hide_temp(self.bgItem_imv1,self.imv1)
            elif win == 2:
                overlay_hide_temp(self.bgItem_imv2,self.imv2) 
            elif win == 3:
                overlay_hide_temp(self.bgItem_imv3,self.imv3)
            elif win == 4:    
                overlay_hide_temp(self.bgItem_imv4,self.imv4) 
            elif win == 6:
                overlay_hide_temp(self.bgItem_imv6,self.imv6)
            self.overlayFlag = False            
        return



    def overlayOff(self):        
        def overlay_hide(bgItem,imv):
            bgItem.hist_luttt.hide()
            bgItem.blank_luttt.hide()
            imv.ui.gridLayout.removeWidget(bgItem.hist_luttt)
            imv.ui.gridLayout.removeWidget(bgItem.blank_luttt)
            imv.view.removeItem(bgItem)  
            #bgItem.hist_luttt = None 
            #bgItem.blank_luttt = None 
            #bgItem = None             
            return
        
        if self.overlayFlag:
            overlay_hide(self.bgItem_imv1,self.imv1)
            overlay_hide(self.bgItem_imv2,self.imv2) 
            overlay_hide(self.bgItem_imv3,self.imv3)
            overlay_hide(self.bgItem_imv4,self.imv4) 
            overlay_hide(self.bgItem_imv6,self.imv6)
            self.overlayFlag = False   
            
        self.resetImages()            
        return


    def toggleOverlay(self):
        if self.overlayArrayLoaded == False:
            print('load array first!')
            return
        
        if self.overlayFlag:
            self.overlayOff()        
        else:
            self.overlayFlag = True
            self.updateAllOverlayWins()
        return


    def setOverlayLevels(self):  
        if self.histogramsLinked == False:
            return
        levels = self.bgItem_imv1.getLevels()
        bgItemList = [self.bgItem_imv2,self.bgItem_imv3,self.bgItem_imv4,self.bgItem_imv6]
        for bgItem in bgItemList:
            bgItem.hist_luttt.item.setLevels(levels[0],levels[1])
        return

    def setOverlayLUTs(self):
        if self.histogramsLinked == False:
            return
        lut = self.bgItem_imv1.hist_luttt.item.gradient.saveState()
        bgItemList = [self.bgItem_imv2,self.bgItem_imv3,self.bgItem_imv4,self.bgItem_imv6]
        for bgItem in bgItemList:
            bgItem.hist_luttt.item.gradient.restoreState(lut)
        return


    def setMainLevels(self):
        if self.histogramsLinked == False:
            return
        levels = self.imv1.getLevels()
        imvList = [self.imv2,self.imv3,self.imv4,self.imv6]
        for imv in imvList:
            imv.getHistogramWidget().item.setLevels(levels[0],levels[1])
        return

    def setMainLUTs(self):
        if self.histogramsLinked == False:
            return
        lut = self.imv1.getHistogramWidget().item.gradient.saveState()
        imvList = [self.imv2,self.imv3,self.imv4,self.imv6]
        for imv in imvList:
            imv.getHistogramWidget().item.gradient.restoreState(lut)
        return


    def updateOverlayLevels(self):
        self.levels_1 = self.bgItem_imv1.getLevels()
        self.levels_2 = self.bgItem_imv2.getLevels()   
        self.levels_3 = self.bgItem_imv3.getLevels() 
        self.levels_4 = self.bgItem_imv4.getLevels()    
        self.levels_6 = self.bgItem_imv6.getLevels()          
        self.gradientState_1 = self.bgItem_imv1.hist_luttt.item.gradient.saveState()
        self.gradientState_2 = self.bgItem_imv2.hist_luttt.item.gradient.saveState()
        self.gradientState_3 = self.bgItem_imv3.hist_luttt.item.gradient.saveState()
        self.gradientState_4 = self.bgItem_imv4.hist_luttt.item.gradient.saveState()
        self.gradientState_6 = self.bgItem_imv6.hist_luttt.item.gradient.saveState()
        #reconnect link TOP overlay histgramLUT to other windows
        self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)

    def overlayUpdate(self,win):
        self.A_overlay_currentVol =self.A_overlay[:,0,:,:] #first volume

        #overlay images
        if win == 0:
            self.bgItem_imv1 = self.overlay(self.maxProjection(self.A_overlay_currentVol), self.imv1, self.levels_1, self.gradientState_1)
            self.bgItem_imv3 = self.overlay(self.roi3.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv3, self.levels_3,self.gradientState_3)
            self.bgItem_imv2 = self.overlay(np.rot90(self.roi2.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), axes=(1,0)), self.imv2,self.levels_2,self.gradientState_2)
            self.bgItem_imv4 = self.overlay(self.roi1.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv4, self.levels_4,self.gradientState_4)
            self.bgItem_imv6 = self.overlay(self.A_overlay_currentVol[self.index6], self.imv6, self.levels_6,self.gradientState_6)        
                
        elif win == 1:
            self.bgItem_imv1 = self.overlay(self.maxProjection(self.A_overlay_currentVol), self.imv1,self.levels_1,self.gradientState_1)
        elif win== 3:
            self.bgItem_imv3 = self.overlay(self.roi3.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv3,self.levels_3,self.gradientState_3)
        elif win == 2:  
            self.bgItem_imv2 = self.overlay(np.rot90(self.roi2.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), axes=(1,0)), self.imv2,self.levels_2,self.gradientState_2)
        elif win == 4:   
            self.bgItem_imv4 = self.overlay(self.roi1.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv4, self.levels_4,self.gradientState_4)
        elif win == 6:    
            self.bgItem_imv6 = self.overlay(self.A_overlay_currentVol[self.imv6.currentIndex], self.imv6, self.levels_6,self.gradientState_6)
        return

    def updateAllOverlayWins(self):
        for winNumber in [1,2,3,4,6]:
            self.runOverlayUpdate(winNumber)
        
    def setOverlayLUT(self, lut):
        self.OverlayLUT = lut

    def overlayOptions(self):
        if self.overlayArrayLoaded == False:
            print('load array first!')
            g.m.statusBar().showMessage("load array first!")
            return
        self.overlayOptionsWin = OverlayOptions(self.viewer)   
        self.overlayOptionsWin.show()
        return

    def setOverlayMode(self,keyText):
        modeDict = {"Source Overlay" : QtGui.QPainter.CompositionMode_SourceOver,
                    "Overlay" : QtGui.QPainter.CompositionMode_Overlay,
                    "Plus" : QtGui.QPainter.CompositionMode_Plus,
                    "Mutiply" : QtGui.QPainter.CompositionMode_Multiply}
        
        self.OverlayMODE = modeDict[keyText]

    def resetImages(self):
        #add max projection data to main window
        self.imv1.setImage(self.maxProjection(self.data))
        #add sliceable data to side window
        self.imv6.setImage(self.data)
        #update to populate roi windows
        self.update(1)
        self.update(2)
        self.update(3) 
        #self.update_4() 
        #correct roi4 position
        self.update(6)


    def overlayScaleOptions(self, win):
        if win == 1:
            scale_bar_1=Scale_Bar_volumeView(self.imv1, self.imv1.image.shape[1], self.imv1.image.shape[0])
            scale_bar_1.gui()
        elif win == 2:
            scale_bar_2=Scale_Bar_volumeView(self.imv2, self.imv2.image.shape[1], self.imv2.image.shape[0])
            scale_bar_2.gui()    
        elif win == 3:
            scale_bar_3=Scale_Bar_volumeView(self.imv3, self.imv3.image.shape[1], self.imv3.image.shape[0])
            scale_bar_3.gui()
        elif win == 4:
            scale_bar_4=Scale_Bar_volumeView(self.imv4, self.imv4.image.shape[1], self.imv4.image.shape[0])
            scale_bar_4.gui()    
        elif win == 6:
            scale_bar_6=Scale_Bar_volumeView(self.imv6, self.imv6.image.shape[1], self.imv6.image.shape[0])
            scale_bar_6.gui()
        return

    def quickOverlay(self):
        if self.overlayArrayLoaded:
            print('Array already loaded')
            g.m.statusBar().showMessage("Array already loaded")
            return
        else:                        
            self.A_overlay = self.viewer.getVolumeArray(self.currentVolume)
            #perform transform
            self.A_overlay= perform_shear_transform(self.A_overlay, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
            #set flags
            self.overlayFlag = True      
            self.overlayArrayLoaded = True
            #update overlay
            self.overlayUpdate(0)
            #link TOP overlay histgramLUT to other windows
            self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)
        return

    def gaussianOptions(self):
        self.gaussianDialogWin = gaussianDialog_win(self.viewer)
        self.gaussianDialogWin.show()
        return


    def win_options(self):
        self.optionsDialogWin = optionsDialog_win(self.viewer)
        self.optionsDialogWin.show()
        return        
    
class exportDialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(exportDialog_win, self).__init__(parent)

        self.viewer = viewerInstance

        #window geometry
        windowGeometry(self, left=300, top=300, height=200, width=200)

        #labels
        self.spaceLabel = QtWidgets.QLabel("|-- Stack in space (displayed vol) --|")
        self.timeLabel = QtWidgets.QLabel("|-- Stack in time (max projections) --|")

        #buttons
        self.buttonTop_space = QtWidgets.QPushButton("Top view")        
        self.buttonZ_space = QtWidgets.QPushButton("Z view")
        self.buttonX_space = QtWidgets.QPushButton("X view")
        self.buttonY_space = QtWidgets.QPushButton("Y view")
        self.buttonTop_time = QtWidgets.QPushButton("Top view")        
        self.buttonZ_time = QtWidgets.QPushButton("Z view")
        self.buttonX_time = QtWidgets.QPushButton("X view")
        self.buttonY_time = QtWidgets.QPushButton("Y view")        
        
        
        #self.buttonFree = QtWidgets.QPushButton("Export Free view")        

        #grid layout
        layout = QtWidgets.QGridLayout()
        #layout.setSpacing(5)
        layout.addWidget(self.spaceLabel, 0, 0)         
        layout.addWidget(self.buttonTop_space, 1, 0)        
        layout.addWidget(self.buttonX_space, 2, 0)
        layout.addWidget(self.buttonY_space, 3, 0)
        layout.addWidget(self.buttonZ_space, 4, 0)
        layout.addWidget(self.timeLabel, 0, 1)         
        layout.addWidget(self.buttonTop_time, 1, 1)        
        layout.addWidget(self.buttonX_time, 2, 1)
        layout.addWidget(self.buttonY_time, 3, 1)
        layout.addWidget(self.buttonZ_time, 4, 1)        
        #layout.addWidget(self.buttonFree, 4, 0)        

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Export Options")

        #connect buttons
        self.buttonTop_space.clicked.connect(lambda: self.exportSpace('Top'))        
        self.buttonZ_space.clicked.connect(lambda: self.exportSpace('Z'))
        self.buttonX_space.clicked.connect(lambda: self.exportSpace('X'))
        self.buttonY_space.clicked.connect(lambda: self.exportSpace('Y'))
        
        self.buttonTop_time.clicked.connect(lambda: self.exportTime('Top'))        
        self.buttonZ_time.clicked.connect(lambda: self.exportTime('Z'))
        self.buttonX_time.clicked.connect(lambda: self.exportTime('X'))
        self.buttonY_time.clicked.connect(lambda: self.exportTime('Y'))        
        
       
        #self.buttonFree.clicked.connect(lambda: self.export('Free'))        
        return

    def exportSpace(self, axis):
        self.displayWindow = Window(self.viewer.viewer.getWin(axis),'{} view'.format(axis))             
        return
    
    def exportTime(self, axis):
        self.displayWindow = Window(self.viewer.viewer.getTimeSeries(axis),'{} view'.format(axis))             
        return    

class optionsDialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(optionsDialog_win, self).__init__(parent)

        self.viewer = viewerInstance
        
        #window geometry
        windowGeometry(self, left=300, top=300, height=300, width=200)
        
        #checkboxes
        self.linkCheck = QtWidgets.QCheckBox()
        self.linkCheck.setChecked(self.viewer.viewer.histogramsLinked)
        self.linkCheck.stateChanged.connect(self.linkCheckValueChange)   
        
        #labels
        self.labelLinkCheck = QtWidgets.QLabel("Link Histogram Sliders") 
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.labelLinkCheck, 1, 0)    
        layout.addWidget(self.linkCheck, 1, 1) 
        self.setLayout(layout)

        #add window title
        self.setWindowTitle("Window Options")        
        
    def linkCheckValueChange(self, value):
        self.viewer.viewer.histogramsLinked = value
        return

class gaussianDialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(gaussianDialog_win, self).__init__(parent)

        self.viewer = viewerInstance
        self.sigma = float(self.viewer.viewer.sigma)
        
        #window geometry
        windowGeometry(self, left=300, top=300, height=300, width=600)

        #sliders
        self.sigmaSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        setSliderUp(self.sigmaSlider, minimum=0, maximum=500, tickInterval=100, singleStep=1, value=self.sigma)
        self.sigmaSlider.valueChanged.connect(self.sigmaValueChange)
        self.sigmaSliderLabel = QtWidgets.QLabel("Sigma: ")
        self.sigmaValueLabel = QtWidgets.QLabel("{:.1f}".format(self.sigma/10))

        #buttons
        self.applyGauss = QtWidgets.QPushButton("Apply Gaussian")
        self.undoGauss = QtWidgets.QPushButton("Undo Gaussian")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.sigmaSliderLabel, 0, 0)
        layout.addWidget(self.sigmaValueLabel, 0, 1)       
        layout.addWidget(self.sigmaSlider, 0, 2)        
        layout.addWidget(self.applyGauss, 1, 2)        
        layout.addWidget(self.undoGauss, 2, 2)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Gaussian Filter Options")

        #connect buttons
        self.applyGauss.clicked.connect(self.applyGaussian)
        self.undoGauss.clicked.connect(self.undoGaussian)
        return

    def gaussByVol(self,A):
        zeroArray = np.zeros_like(A)
        nVols = A.shape[1]
        for i in range(0,nVols):
            g.m.statusBar().showMessage("3D Gaussian running...Vol:{}".format(i))
            zeroArray[:,i,:,:] = gaussian_filter(self.viewer.viewer.originalData[:,i,:,:], self.sigma/10)
        return zeroArray


    def applyGaussian(self):
        g.m.statusBar().showMessage("3D Gaussian running...")
        self.viewer.viewer.gaussData = self.gaussByVol(self.viewer.viewer.originalData)
        self.viewer.viewer.gaussianFlag = True
        self.viewer.viewer.sigma = self.sigma
        self.viewer.viewer.timeUpdate(self.viewer.viewer.currentVolume)
        g.m.statusBar().showMessage("3D Gaussian finished")
        return

    def undoGaussian(self):
        self.viewer.viewer.gaussianFlag = False
        self.viewer.viewer.sigma = self.sigma
        self.viewer.viewer.timeUpdate(self.viewer.viewer.currentVolume)        
        return

    def sigmaValueChange(self):
        self.sigma = self.sigmaSlider.value()
        self.sigmaValueLabel.setText("{:.1f}".format(self.sigma/10))
        return

class exportIMSdialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, A, parent = None):
        super(exportIMSdialog_win, self).__init__(parent)

        self.viewer = viewerInstance
        self.A = A

        subsamp_options = ['((1, 1, 1), (1, 2, 2))']
        chunks_options = ['((16, 128, 128), (64, 64, 64))']
        compression_options = ['gzip', 'lzf', 'szip']
        thumbsize_options = ['256','128']
        dx = g.settings['volumeSlider']['IMS_dx']
        dz = g.settings['volumeSlider']['IMS_dz']
        Unit_options = ['m','mm','um','nm']
        GammaCorrection = g.settings['volumeSlider']['IMS_GammaCorrection']
        ColorRange_options = ['0 255']
        LSMEmissionWavelength = g.settings['volumeSlider']['IMS_LSMEmissionWavelength']
        LSMExcitationWavelength = g.settings['volumeSlider']['IMS_LSMExcitationWavelength']

        #window geometry
        windowGeometry(self, left=300, top=300, height=200, width=500)

        #labels
        self.labelPath = QtWidgets.QLabel('Save Path: ') 
        self.label_IMS_SavePath = QtWidgets.QLabel(str(shorten_path(g.settings['volumeSlider']['IMS_fname'],4))) 
        self.label_IMS_subsamp = QtWidgets.QLabel('subsamp: ') 
        self.label_IMS_chunks = QtWidgets.QLabel('chunks: ') 
        self.label_IMS_compression = QtWidgets.QLabel('compression: ') 
        self.label_IMS_thumbsize = QtWidgets.QLabel('thumbsize: ') 
        self.label_IMS_dx = QtWidgets.QLabel('dx & dy: ') 
        self.label_IMS_dz = QtWidgets.QLabel('dz: ') 
        self.label_IMS_Unit = QtWidgets.QLabel('Unit: ') 
        self.label_IMS_GammaCorrection = QtWidgets.QLabel('Gamma Correction: ') 
        self.label_IMS_ColorRange = QtWidgets.QLabel('Color Range: ') 
        self.label_IMS_LSMEmissionWavelength = QtWidgets.QLabel('LSM Emission Wavelength: ') 
        self.label_IMS_LSMExcitationWavelength = QtWidgets.QLabel('LSM Excitation Wavelength: ') 
        
        #buttons
        self.buttonPath = QtWidgets.QPushButton("Set Path") 
        self.buttonExport = QtWidgets.QPushButton("Export")               

        #ComboBox
        self.subsampSelectorBox = QtWidgets.QComboBox()
        self.subsampSelectorBox.addItems(subsamp_options)
        self.subsampSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_subsamp'])
        self.subsampSelectorBox.currentIndexChanged.connect(self.subsampSelectionChange)        

        self.chunksSelectorBox = QtWidgets.QComboBox()
        self.chunksSelectorBox.addItems(chunks_options)
        self.chunksSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_chunks'])
        self.chunksSelectorBox.currentIndexChanged.connect(self.chunksSelectionChange) 

        self.compressionSelectorBox = QtWidgets.QComboBox()
        self.compressionSelectorBox.addItems(compression_options)
        self.compressionSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_compression'])
        self.compressionSelectorBox.currentIndexChanged.connect(self.compressionSelectionChange) 

        self.thumbsizeSelectorBox = QtWidgets.QComboBox()
        self.thumbsizeSelectorBox.addItems(thumbsize_options)
        self.thumbsizeSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_thumbsize'])
        self.thumbsizeSelectorBox.currentIndexChanged.connect(self.thumbsizeSelectionChange) 

        self.unitSelectorBox = QtWidgets.QComboBox()
        self.unitSelectorBox.addItems(Unit_options)
        self.unitSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_Unit'])
        self.unitSelectorBox.currentIndexChanged.connect(self.unitSelectionChange)

        self.colorRangeSelectorBox = QtWidgets.QComboBox()
        self.colorRangeSelectorBox.addItems(ColorRange_options)
        self.colorRangeSelectorBox.setCurrentText(g.settings['volumeSlider']['IMS_ColorRange'])
        self.colorRangeSelectorBox.currentIndexChanged.connect(self.colorRangeSelectionChange)


        #spinboxes
        self.dxSpinBox = QtWidgets.QDoubleSpinBox()
        self.dxSpinBox.setRange(0.00,100.00)
        self.dxSpinBox.setValue(g.settings['volumeSlider']['IMS_dx'])
        self.dxSpinBox.setSingleStep(0.05)
        self.dxSpinBox.valueChanged.connect(self.dxSpinBoxValueChange)

        self.dzSpinBox = QtWidgets.QDoubleSpinBox()
        self.dzSpinBox.setRange(0.00,100.00)
        self.dzSpinBox.setValue(g.settings['volumeSlider']['IMS_dz'])
        self.dzSpinBox.setSingleStep(0.05)
        self.dzSpinBox.valueChanged.connect(self.dzSpinBoxValueChange)

        self.gammaCorrectionSelectorBox = QtWidgets.QDoubleSpinBox()
        self.gammaCorrectionSelectorBox.setRange(0.0,5.0)
        self.gammaCorrectionSelectorBox.setValue(g.settings['volumeSlider']['IMS_GammaCorrection'])
        self.gammaCorrectionSelectorBox.setSingleStep(0.01)
        self.gammaCorrectionSelectorBox.valueChanged.connect(self.gammaCorrectionBoxValueChange)


        #grid layout
        layout = QtWidgets.QGridLayout()
        #layout.setSpacing(5)
        layout.addWidget(self.labelPath, 0, 0) 
        layout.addWidget(self.label_IMS_SavePath, 0, 1) 
        layout.addWidget(self.buttonPath, 0, 2) 
        layout.addWidget(self.label_IMS_subsamp, 1,0)
        layout.addWidget(self.subsampSelectorBox, 1,1)        
        layout.addWidget(self.label_IMS_chunks, 2,0)
        layout.addWidget(self.chunksSelectorBox, 2,1)          
        layout.addWidget(self.label_IMS_compression, 3,0) 
        layout.addWidget(self.compressionSelectorBox, 3,1)          
        layout.addWidget(self.label_IMS_thumbsize, 4,0) 
        layout.addWidget(self.thumbsizeSelectorBox, 4,1)          
        layout.addWidget(self.label_IMS_dx, 5,0) 
        layout.addWidget(self.dxSpinBox, 5,1)            
        layout.addWidget(self.label_IMS_dz, 6,0) 
        layout.addWidget(self.dzSpinBox, 6,1)         
        layout.addWidget(self.label_IMS_Unit, 7,0) 
        layout.addWidget(self.unitSelectorBox, 7,1)         
        layout.addWidget(self.label_IMS_GammaCorrection, 8,0) 
        layout.addWidget(self.gammaCorrectionSelectorBox, 8,1)         
        layout.addWidget(self.label_IMS_ColorRange, 9,0) 
        layout.addWidget(self.colorRangeSelectorBox, 9,1)           
        layout.addWidget(self.label_IMS_LSMEmissionWavelength, 10,0) 
        
        layout.addWidget(self.label_IMS_LSMExcitationWavelength, 11,0) 
        
        
        layout.addWidget(self.buttonExport, 14, 0)         
       
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Imaris Export Options")

        #connect buttons
        self.buttonExport.clicked.connect(lambda: self.export())  
        self.buttonPath.clicked.connect(lambda: self.getSavePath())               
             
        return

    def export(self):
        makeIMS_flika(self.A) 
        return

    def getSavePath(self): 
        imsSavePath = QtWidgets.QFileDialog.getSaveFileName(None,'Save File', os.path.expanduser("~/Desktop"), 'IMS file (*.ims)')
        g.settings['volumeSlider']['IMS_fname'] = str(imsSavePath[0])
        self.label_IMS_SavePath.setText(str(shorten_path(g.settings['volumeSlider']['IMS_fname'],4))) 
        return
    
    def subsampSelectionChange(self):
        g.settings['volumeSlider']['IMS_subsamp'] = self.subsampSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_subsamp'])
        return
    
    def chunksSelectionChange(self):
        g.settings['volumeSlider']['IMS_chunks'] = self.chunksSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_chunks'])        
        return
    
    def compressionSelectionChange(self):
        g.settings['volumeSlider']['IMS_compression'] = self.compressionSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_compression'])           
        return
    
    def thumbsizeSelectionChange(self,value):
        g.settings['volumeSlider']['IMS_thumbsize'] = self.thumbsizeSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_thumbsize'])         
        return
    
    def dxSpinBoxValueChange(self,value):
        print(value)
        g.settings['volumeSlider']['IMS_dx'] = value        
        return
    
    def dzSpinBoxValueChange(self,value):
        print(value)
        g.settings['volumeSlider']['IMS_dz'] = value        
        return    
    
    def unitSelectionChange(self):
        g.settings['volumeSlider']['IMS_Unit'] = self.unitSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_Unit'])          
        return    

    def gammaCorrectionBoxValueChange(self,value):
        print(value)
        g.settings['volumeSlider']['IMS_GammaCorrection'] = value
        return   
    
    def colorRangeSelectionChange(self):
        g.settings['volumeSlider']['IMS_ColorRange'] = self.colorRangeSelectorBox.currentText()
        print(g.settings['volumeSlider']['IMS_ColorRange'])         
        return    
    
    
    