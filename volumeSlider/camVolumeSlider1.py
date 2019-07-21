import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
import os
from os import listdir
from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
from copy import deepcopy
from numpy import moveaxis
from skimage.transform import rescale
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy
import pyqtgraph.opengl as gl
from OpenGL.GL import *

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox


dataType = np.float16


##############       Helper functions        ###########################################

def get_transformation_matrix(theta=45):
    """
    theta is the angle of the light sheet
    Look at the pdf in this folder.
    """

    theta = theta/360 * 2 * np.pi # in radians
    hx = np.cos(theta)
    sy = np.sin(theta)
 
    S = np.array([[1, hx, 0],
                  [0, sy, 0],
                  [0, 0, 1]])

    return S


def get_transformation_coordinates(I, theta):
    negative_new_max = False
    S = get_transformation_matrix(theta)
    S_inv = np.linalg.inv(S)
    mx, my = I.shape

    four_corners = np.matmul(S, np.array([[0, 0, mx, mx],
                                          [0, my, 0, my],
                                          [1, 1, 1, 1]]))[:-1,:]
    range_x = np.round(np.array([np.min(four_corners[0]), np.max(four_corners[0])])).astype(np.int)
    range_y = np.round(np.array([np.min(four_corners[1]), np.max(four_corners[1])])).astype(np.int)
    all_new_coords = np.meshgrid(np.arange(range_x[0], range_x[1]), np.arange(range_y[0], range_y[1]))
    new_coords = [all_new_coords[0].flatten(), all_new_coords[1].flatten()]
    new_homog_coords = np.stack([new_coords[0], new_coords[1], np.ones(len(new_coords[0]))])
    old_coords = np.matmul(S_inv, new_homog_coords)
    old_coords = old_coords[:-1, :]
    old_coords = old_coords
    old_coords[0, old_coords[0, :] >= mx-1] = -1
    old_coords[1, old_coords[1, :] >= my-1] = -1
    old_coords[0, old_coords[0, :] < 1] = -1
    old_coords[1, old_coords[1, :] < 1] = -1
    new_coords[0] -= np.min(new_coords[0])
    keep_coords = np.logical_not(np.logical_or(old_coords[0] == -1, old_coords[1] == -1))
    new_coords = [new_coords[0][keep_coords], new_coords[1][keep_coords]]
    old_coords = [old_coords[0][keep_coords], old_coords[1][keep_coords]]
    return old_coords, new_coords


def perform_shear_transform(A, shift_factor, interpolate, datatype, theta, inputArrayOrder = [0, 3, 1, 2], displayArrayOrder = [3, 0, 1, 2]):
    A = moveaxis(A, inputArrayOrder, [0, 1, 2, 3]) # INPUT
    m1, m2, m3, m4 = A.shape
    if interpolate:
        A_rescaled = np.zeros((m1*int(shift_factor), m2, m3, m4))
        for v in np.arange(m4):
            print('Upsampling Volume #{}/{}'.format(v+1, m4))
            A_rescaled[:, :, :, v] = rescale(A[:, :, :, v], (shift_factor, 1.), mode='constant', preserve_range=True)
    else:
        A_rescaled = np.repeat(A, shift_factor, axis=1)
    mx, my, mz, mt = A_rescaled.shape
    I = A_rescaled[:, :, 0, 0]
    old_coords, new_coords = get_transformation_coordinates(I, theta)
    old_coords = np.round(old_coords).astype(np.int)
    new_mx, new_my = np.max(new_coords[0]) + 1, np.max(new_coords[1]) + 1

    D = np.zeros((new_mx, new_my, mz, mt))
    D[new_coords[0], new_coords[1], :, :] = A_rescaled[old_coords[0], old_coords[1], :, :]
    E = moveaxis(D, [0, 1, 2, 3], displayArrayOrder) # AXIS INDEX CHANGED FROM INPUT TO MATCH KYLE'S CODE
    #E = np.flip(E, 1)

    return E

def getCorners(vol):
    z,x,y = vol.nonzero()
    z_min = np.min(z)
    z_max = np.max(z)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    newArray = np.zeros(vol.shape)

    newArray[z_min,x_min,y_min] = 1    
    newArray[z_min,x_max,y_min] = 1    
    newArray[z_min,x_min,y_max] = 1    
    newArray[z_min,x_max,y_max] = 1    
    newArray[z_max,x_min,y_min] = 1    
    newArray[z_max,x_max,y_min] = 1    
    newArray[z_max,x_min,y_max] = 1    
    newArray[z_max,x_max,y_max] = 1     
    return newArray

def getDimensions(vol):
    z,x,y = vol.nonzero()
    z_min = np.min(z)
    z_max = np.max(z)
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return x_min,x_max,y_min,y_max,z_min,z_max

def getMaxDimension(vol):
    z,x,y = vol.nonzero()
    z_max = np.max(z)
    x_max = np.max(x)
    y_max = np.max(y)
    return np.max([x_max,y_max,z_max])

def plot_cube(ax, cube_definition):
    cube_definition_array = [
        np.array(list(item))
        for item in cube_definition
    ]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    # Plot faces
    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0,0,0.1,0.1))

    ax.add_collection3d(faces)

    # Plot the points
    ax.scatter(points[:,0], points[:,1], points[:,2], s=0)

    ax.set_aspect('equal')    

###############  New GLGraphicsItem class definition ################################
class GLBorderItem(gl.GLAxisItem):
    """
    **Bases:** :class:`GLGraphicsItem <pyqtgraph.opengl.GLGraphicsItem>`
    Overwrite of GLAxisItem 
    Displays borders of plot data 
    
    """
    
    def setSize(self, x=None, y=None, z=None, size=None):
        """
        Set the size of the axes (in its local coordinate system; this does not affect the transform)
        Arguments can be x,y,z or size=QVector3D().
        """
        if size is not None:
            x = size.x()
            y = size.y()
            z = size.z()
        self.__size = [x,y,z]
        self.update()

        
    def size(self):
        return self.__size[:]
    
        
    def paint(self):

        #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        #glEnable( GL_BLEND )
        #glEnable( GL_ALPHA_TEST )
        self.setupGLState()
        
        if self.antialias:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            
        glBegin( GL_LINES )
        
        x,y,z = self.size()       

        def zFrame(x,y,z,r=1,g=1,b=1,thickness=0.6):
            glColor4f(r, g, b, thickness)  # z 
            glVertex3f(-(int(x)), -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), -(int(y/2)), z-(int(z/2)))
            
            glColor4f(r, g, b, thickness)  # z 
            glVertex3f(-(int(x)), y -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), z-(int(z/2)))        
    
            glColor4f(r, g, b, thickness)  # y 
            glVertex3f(-(int(x)), -(int(y/2)), -(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), -(int(z/2)))
            
            glColor4f(r, g, b, thickness)  # y 
            glVertex3f(-(int(x)), -(int(y/2)), z-(int(z/2)))
            glVertex3f(-(int(x)), y -(int(y/2)), z-(int(z/2))) 

        
        def xFrame(x,y,z,r=1,g=1,b=1,thickness=0.6):
            glColor4f(r, g, b, thickness)  # x is blue
            glVertex3f(x-(int(x/2)), -(int(y)), -(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), -(int(z/2)))
    
            glColor4f(r, g, b, thickness)  # x is blue
            glVertex3f(x-(int(x/2)), -(int(y)), z-(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), z-(int(z/2)))        
            
            glColor4f(r, g, b, thickness)  # z 
            glVertex3f(x-(int(x/2)), -(int(y)), -(int(z/2)))
            glVertex3f(x-(int(x/2)), -(int(y)), z-(int(z/2)))
            
            glColor4f(r, g, b, thickness)  # z 
            glVertex3f((int(x/2))-x, -(int(y)), -(int(z/2)))
            glVertex3f((int(x/2))-x, -(int(y)), z-(int(z/2))) 
            

        def box(x,y,z,r=1,g=1,b=1,thickness=0.6):        
            zFrame(x/2,y,z,r=r,g=g,b=b,thickness=thickness)
            zFrame(x/2-x,y,z,r=r,g=g,b=b,thickness=thickness)        
            xFrame(x,y/2,z,r=r,g=g,b=b,thickness=thickness)
            xFrame(x,-y/2,z,r=r,g=g,b=b,thickness=thickness)        
       

        box(x,y,z)
        
        glEnd()
 
    
    
    
    
    
################################################################################

class SliceViewer(BaseProcess):

    def __init__(self, A):
        super().__init__()

        
        self.shift_factor = camVolumeSlider.dialogbox.shiftFactor
        self.interpolate = False
        self.theta = camVolumeSlider.dialogbox.theta
        self.inputArrayOrder = camVolumeSlider.getInputArrayOrder()
        self.displayArrayOrder = camVolumeSlider.getDisplayArrayOrder()

        self.trim_last_frame = camVolumeSlider.dialogbox.trim_last_frame

        
        print('shift factor: '+ str(self.shift_factor))
        print('theta: ' + str(self.theta))
        print('input Array Order: '+ str(self.inputArrayOrder))
        print('display Array Order: ' + str(self.displayArrayOrder))        
        
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
               
        self.data = self.originalData[:,0,:,:]
        
        #define app
        #self.app = QtWidgets.QApplication([])
        
        ## Create window with 4 docks
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1000,800)
        self.win.setWindowTitle('Lightsheet 3D display')
        
        #create Dock area
        self.area = DockArea()
        self.win.setCentralWidget(self.area)
        
        #define docks
        self.d1 = Dock("Top View - Max Projection", size=(500,400))
        self.d2 = Dock("X Slice", size=(500,400), closable=True)
        self.d3 = Dock("Y Slice", size=(500,400), closable=True)
        self.d4 = Dock("Free ROI", size=(500,400), closable=True)
        self.d5 = Dock("Time Slider", size=(1000,50))
        self.d6 = Dock("Z Slice", size=(500,400), closable=True)
        
        #add docks to area
        self.area.addDock(self.d1, 'left')              ## place d1 at left edge of dock area 
        self.area.addDock(self.d2, 'right')             ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d1)   ## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'bottom', self.d2)   ## place d4 at bottom edge of d2
        self.area.addDock(self.d5, 'bottom')            ## place d4 at bottom edge of d2
        self.area.addDock(self.d6, 'below', self.d4)    ## tab below d4
        
        #initialise image widgets
        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.imv3 = pg.ImageView()
        self.imv4 = pg.ImageView()
        self.imv6 = pg.ImageView()
        
        self.imageWidgits = [self.imv1,self.imv2,self.imv3,self.imv4,self.imv6]
        
        #add image widgets to docks
        self.d1.addWidget(self.imv1)
        self.d2.addWidget(self.imv3)
        self.d3.addWidget(self.imv2)
        self.d4.addWidget(self.imv4)
        self.d6.addWidget(self.imv6)
        
        #hide dock title-bars at start
        self.d5.hideTitleBar()        
        #self.hide_titles(self)
        
        #add menu functions
        self.state = self.area.saveState()
                
        self.menubar = self.win.menuBar()
        
        self.fileMenu1 = self.menubar.addMenu('&Options')
        self.resetLayout = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Reset Layout')
        self.resetLayout.setShortcut('Ctrl+R')
        self.resetLayout.setStatusTip('Reset Layout')
        self.resetLayout.triggered.connect(self.reset_layout)
        self.fileMenu1.addAction(self.resetLayout)
        
        self.showTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Titles')
        self.showTitles.setShortcut('Ctrl+G')
        self.showTitles.setStatusTip('Show Titles')
        self.showTitles.triggered.connect(self.show_titles)
        self.fileMenu1.addAction(self.showTitles)
        
        self.hideTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Titles')
        self.hideTitles.setShortcut('Ctrl+H')
        self.hideTitles.setStatusTip('Hide Titles')
        self.hideTitles.triggered.connect(self.hide_titles)
        self.fileMenu1.addAction(self.hideTitles)
        
        self.fileMenu2 = self.menubar.addMenu('&3D Plot')
        self.plot = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot')
        self.plot.setShortcut('Ctrl+P')
        self.plot.setStatusTip('3D Plot')
        self.plot.triggered.connect(self.plot3D)
        self.fileMenu2.addAction(self.plot)
        
        self.plotOptions = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot Options')
        self.plotOptions.setStatusTip('3D Plot Options')
        self.plotOptions.triggered.connect(self.plot3D_options)
        self.fileMenu2.addAction(self.plotOptions)        
                
        self.fileMenu3 = self.menubar.addMenu('&Texture Plot')
        self.texturePlot = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Texture Plot')
        self.texturePlot.setShortcut('Ctrl+T')
        self.texturePlot.setStatusTip('Texture Plot')
        self.texturePlot.triggered.connect(self.plotTexture)
        self.fileMenu3.addAction(self.texturePlot)        

        self.fileMenu4 = self.menubar.addMenu('&Export')
        self.export = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export')
        self.export.setShortcut('Ctrl+E')
        self.export.setStatusTip('Export')
        self.export.triggered.connect(self.exportDialog)
        self.fileMenu4.addAction(self.export) 
        
        self.fileMenu5 = self.menubar.addMenu('&Quit')
        self.quit = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Quit')
        self.quit.setShortcut('Ctrl+Q')
        self.quit.setStatusTip('Quit')
        self.quit.triggered.connect(self.close)
        self.fileMenu5.addAction(self.quit)
        
        #add time slider
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(self.nVols)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        self.d5.addWidget(self.slider1)
                
        #display window
        self.win.show()
        
        #define single line roi
        self.roi1 = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
        self.imv1.addItem(self.roi1)
        
        #define crosshair rois
        self.roi2 = pg.LineSegmentROI([[0, 0], [self.width, 0]], pen='y', maxBounds=QtCore.QRect(0,0,0,self.height))
        self.imv1.addItem(self.roi2)
        
        self.roi3 = pg.LineSegmentROI([[0, 0], [0, self.height]], pen='y', maxBounds=QtCore.QRect(0,0,self.width,0))
        self.imv1.addItem(self.roi3)

        #hide default imageview buttons
        def hideButtons(imv):    
            imv.ui.roiBtn.hide()
            imv.ui.menuBtn.hide()
        
        for imv in self.imageWidgits:
            hideButtons(imv)

        #disconnect roi handles
        def disconnectHandles(roi):
            handles = roi.getHandles()
            handles[0].disconnectROI(roi)
            handles[1].disconnectROI(roi)
            handles[0].currentPen = mkPen(None)
            handles[1].currentPen = mkPen(None) 
            handles[0].pen = mkPen(None)
            handles[1].pen = mkPen(None)     
            
        disconnectHandles(self.roi2)
        disconnectHandles(self.roi3)

        #add max projection data to main window
        self.imv1.setImage(self.maxProjection(self.data)) #display topview (max of slices)
        
        #add sliceable data to side window
        self.imv6.setImage(self.data)
        
        #connect roi updates
        self.roi1.sigRegionChanged.connect(self.update)
        self.roi2.sigRegionChanged.connect(self.update_2)
        self.roi3.sigRegionChanged.connect(self.update_3)
        
        #initial update to populate roi windows
        self.update()
        self.update_2()
        self.update_3()
        
        #autolevel roi windows at start
        self.imv2.autoLevels()
        self.imv3.autoLevels()
        self.imv4.autoLevels()
        self.imv6.autoLevels()
        
         
        self.slider1.valueChanged.connect(self.timeUpdate)

    #define update calls for each roi
    def update(self):
        self.d1 = self.roi1.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
        self.imv4.setImage(self.d1, autoRange=False, autoLevels=False)
    
    def update_2(self):
        self.d2 = np.rot90(self.roi2.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2)), axes=(1,0))
        self.imv2.setImage(self.d2, autoRange=False, autoLevels=False)
        
    def update_3(self):
        self.d3 = self.roi3.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
        self.imv3.setImage(self.d3, autoRange=False, autoLevels=False)
              

    #connect time slider
    def timeUpdate(self,value):
        #global data
        self.data = self.originalData[:,value,:,:]
        self.imv1.setImage(self.maxProjection(self.data))
        self.imv6.setImage(self.data)
        self.update()
        self.update_2()
        self.update_3()
        return

    def reset_layout(self):
        #global state
        self.area.restoreState(self.state)

    def hide_titles(self,_):
        self.d1.hideTitleBar()
        self.d2.hideTitleBar()
        self.d3.hideTitleBar()
        self.d4.hideTitleBar()
        self.d6.hideTitleBar()
        
    def show_titles(self):
        self.d1.showTitleBar()
        self.d2.showTitleBar()
        self.d3.showTitleBar()
        self.d4.showTitleBar()
        self.d6.showTitleBar()

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
        self.plot3D_optionWin = plot3D_options(self.prob,self.threshold_3D)
        self.plot3D_optionWin.show() 
        return

    def plotTexture(self):
        #app = QtGui.QApplication([])
        w = gl.GLViewWidget()
        w.opts['distance'] = 200
        w.show()
        w.setWindowTitle('3D slice - texture plot')
        
        shape = self.data.shape
                
        ## slice out three planes, convert to RGBA for OpenGL texture
        levels = (0, 1000)
        
        slice1 = int(shape[0]/2)
        slice2 = int(shape[1]/2)
        slice3 = int(shape[2]/2)
        
        tex1 = pg.makeRGBA(self.data[slice1], levels=levels)[0]       # yz plane
        tex2 = pg.makeRGBA(self.data[:,slice2], levels=levels)[0]     # xz plane
        tex3 = pg.makeRGBA(self.data[:,:,slice3], levels=levels)[0]   # xy plane
        
        
        ## Create three image items from textures, add to view
        v1 = gl.GLImageItem(tex1)
        v1.translate(-slice2, -slice3, 0)
        v1.rotate(90, 0,0,1)
        v1.rotate(-90, 0,1,0)
        w.addItem(v1)
        
        v2 = gl.GLImageItem(tex2)
        v2.translate(-slice1, -slice3, 0)
        v2.rotate(-90, 1,0,0)
        w.addItem(v2)
        
        v3 = gl.GLImageItem(tex3)
        v3.translate(-slice1, -slice2, 0)
        w.addItem(v3)
        
        #ax = gl.GLAxisItem()
        ax = GLBorderItem()
        ax.setSize(x=shape[0],y=shape[1],z=shape[2])
        w.addItem(ax)
        
    def close(self):
        self.roi1.sigRegionChanged.disconnect(self.update)
        self.roi2.sigRegionChanged.disconnect(self.update_2)
        self.roi3.sigRegionChanged.disconnect(self.update_3)
        self.imv1.close()
        self.imv2.close()
        self.imv3.close()
        self.imv4.close()
        self.imv6.close()   
        self.exportDialogWin.close() 
        self.win.close()
        self.win.destroy()
        return

    def closeEvent(self, event):
        event.accept()

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
        self.exportDialogWin = exportDialog_win()
        self.exportDialogWin.show() 
        return

    def getXWin(self):
        return self.data.swapaxes(0,1)

    def getYWin(self):
        return self.data.swapaxes(2,0)

    def getZWin(self):
        return self.data.swapaxes(1,2)
       
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.nVols = 1
        
        #create dtype dict
        self.dTypeDict = {
                'float16': np.float16,
                'float32': np.float32,
                'float64': np.float64,
                'int8': np.uint8,
                'int16': np.uint16,                                 
                'int32': np.uint32,
                'int64': np.uint64
                                  }
        
        self.dataType = dataType
        
        #create array order dict
        self.arrayDict = {
                '[0, 1, 2, 3]': [0, 1, 2, 3],
                '[0, 1, 3, 2]': [0, 1, 3, 2],
                '[0, 2, 1, 3]': [0, 2, 1, 3],
                '[0, 2, 3, 1]': [0, 2, 3, 1],
                '[0, 3, 1, 2]': [0, 3, 1, 2],
                '[0, 3, 2, 1]': [0, 3, 2, 1],
                '[1, 0, 2, 3]': [1, 0, 2, 3],
                '[1, 0, 3, 2]': [1, 0, 3, 2],
                '[1, 2, 0, 3]': [1, 2, 0, 3],
                '[1, 2, 3, 0]': [1, 2, 3, 0],
                '[1, 3, 0, 2]': [1, 3, 0, 2],
                '[1, 3, 2, 0]': [1, 3, 2, 0],
                '[2, 0, 1, 3]': [2, 0, 1, 3],
                '[2, 0, 3, 1]': [2, 0, 3, 1],
                '[2, 1, 0, 3]': [2, 1, 0, 3],
                '[2, 1, 3, 0]': [2, 1, 3, 0],
                '[2, 3, 0, 1]': [2, 3, 0, 1],
                '[2, 3, 1, 0]': [2, 3, 1, 0],
                '[3, 0, 1, 2]': [3, 0, 1, 2],
                '[3, 0, 2, 1]': [3, 0, 2, 1],
                '[3, 1, 0, 2]': [3, 1, 0, 2],
                '[3, 1, 2, 0]': [3, 1, 2, 0],
                '[3, 2, 0, 1]': [3, 2, 0, 1],
                '[3, 2, 1, 0]': [3, 2, 1, 0]
                }
        
        self.inputArrayOrder = [0, 3, 1, 2]
        self.displayArrayOrder = [3, 0, 1, 2]
        
        #array savepath
        self.savePath = os.path.join(os.path.expanduser("~/Desktop"),'array_4D_data.npy')
        return        

    def startVolumeSlider(self):
        #copy selected window
        self.A =  np.array(deepcopy(g.win.image),dtype=self.dataType)
        self.B = []
        #get shape
        self.nFrames, self.x, self.y = self.A.shape
        self.framesPerVol = int(self.nFrames/self.nVols)
        #setup display window
        self.displayWindow = Window(self.A,'Volume Slider Window')
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show() 
        return        
    
    def updateDisplay_volumeSizeChange(self):
        #remove final volume if it dosen't contain the full number of frames
        numberFramesToRemove = self.nFrames%self.getFramesPerVol()
        if numberFramesToRemove != 0:
            self.B = self.A[:-numberFramesToRemove,:,:]   
        else:
            self.B = self.A
        
        self.nFrames, self.x, self.y = self.B.shape
        
        #reshape to 4D
        self.B = np.reshape(self.B, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
        self.displayWindow.imageview.setImage(self.B[0],autoLevels=False) 
        return   

    def updateDisplay_sliceNumberChange(self, index):
        displayIndex = self.displayWindow.imageview.currentIndex  
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        self.displayWindow.imageview.setCurrentIndex(displayIndex)
        return
        
    def getNFrames(self):
        return self.nFrames
        
    def getNVols(self):
        return self.nVols

    def getFramesPerVol(self):
        return self.framesPerVol

    def updateVolsandFramesPerVol(self, nVols, framesPerVol):
        self.nVols = nVols
        self.framesPerVol = framesPerVol

    def closeEvent(self, event):
        event.accept()

    def getArrayShape(self):
        if self.B == []:
            return self.A.shape
        return self.B.shape

    def subtractBaseline(self):
        index = self.displayWindow.imageview.currentIndex
        baseline = self.dialogbox.getBaseline()
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.B = self.B - baseline
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)

    def averageByVol(self):
        index = self.displayWindow.imageview.currentIndex
        #TODO
        return

    def ratioDFF0(self):
        index = self.displayWindow.imageview.currentIndex
        ratioStart, ratioEnd = self.dialogbox.getF0()
        if ratioStart >= ratioEnd:
            print('invalid F0 selection')
            return
        
        ratioVol = self.B[:,ratioStart:ratioEnd,:,]
        ratioVol = np.mean(ratioVol, axis=1,keepdims=True)        

        self.B = np.divide(self.B, ratioVol, dtype=self.dataType)
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)                
        return

    def exportToWindow(self):
        Window(np.reshape(self.B, (self.nFrames, self.x, self.y), order='F'))
        return

    def exportArray(self):
        np.save(self.savePath, self.B)
        return

    def getMaxPixel(self):
        if self.B == []:
            return np.max(self.A)
        else:
            return np.max(self.B)

    def setDType(self, newDataType):
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.dataType = self.dTypeDict[newDataType]
            self.B = self.B.astype(self.dataType)   
            self.dialogbox.dataTypeText.setText(self.getDataType())        
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)   
        return
    
    def getDataType(self):
        return str(self.dataType).split(".")[-1].split("'")[0]

    def getArrayKeys(self):
        return list(self.arrayDict.keys())

    def getInputArrayOrder(self):
        return self.inputArrayOrder
    
    def getDisplayArrayOrder(self):
        return self.displayArrayOrder
    
    def setInputArrayOrder(self, value):
        self.inputArrayOrder = self.arrayDict[value]
        return 
    
    def setDisplayArrayOrder(self, value):
        self.displayArrayOrder = self.arrayDict[value]
        return 

    def multiplyByFactor(self, factor):
        #factor = np.array(factor).astype(self.dataType)
        #print(type(factor))
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            #self.B = np.multiply(self.B, factor, dtype=self.dataType)
            self.B = self.B * float(factor)
            print(self.B.shape)
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)             
        return        

    def startViewer(self):
        self.viewer = SliceViewer(self.B)
        return

    def closeViewer(self):
        self.viewer.close()
        return
        
camVolumeSlider = CamVolumeSlider()  



class Form2(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form2, self).__init__(parent)
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 400

        self.theta = 45
        self.shiftFactor = 1
        self.trim_last_frame = False

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox1.setValue(0)
 
        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox2.setValue(camVolumeSlider.getNFrames())

        #self.spinLabel3 = QtWidgets.QLabel("# of volumes to average by: ") 
        #self.SpinBox3 = QtWidgets.QSpinBox()
        #self.SpinBox3.setRange(0,camVolumeSlider.getNVols())
        #self.SpinBox3.setValue(0)

        self.spinLabel4 = QtWidgets.QLabel("baseline value: ") 
        self.SpinBox4 = QtWidgets.QSpinBox()
        self.SpinBox4.setRange(0,camVolumeSlider.getMaxPixel())
        self.SpinBox4.setValue(0)
                 
        self.spinLabel6 = QtWidgets.QLabel("F0 start volume: ") 
        self.SpinBox6 = QtWidgets.QSpinBox()
        self.SpinBox6.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox6.setValue(0)
        
        self.spinLabel7 = QtWidgets.QLabel("F0 end volume: ") 
        self.SpinBox7 = QtWidgets.QSpinBox()
        self.SpinBox7.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox7.setValue(0)
        
        self.spinLabel8 = QtWidgets.QLabel("factor to multiply by: ") 
        self.SpinBox8 = QtWidgets.QSpinBox()
        self.SpinBox8.setRange(0,10000)        
        self.SpinBox8.setValue(100)

        self.spinLabel9 = QtWidgets.QLabel("theta: ") 
        self.SpinBox9 = QtWidgets.QSpinBox()
        self.SpinBox9.setRange(0,360)        
        self.SpinBox9.setValue(self.theta)
        
        self.spinLabel10 = QtWidgets.QLabel("shift factor: ") 
        self.SpinBox10 = QtWidgets.QSpinBox()
        self.SpinBox10.setRange(0,100)  
        self.SpinBox10.setValue(self.shiftFactor)        

        
        #sliders
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider.getNFrames())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)

        #ComboBox
        self.dTypeSelectorBox = QtWidgets.QComboBox()
        self.dTypeSelectorBox.addItems(["float16", "float32", "float64","int8","int16","int32","int64"])
        #self.dTypeSelectorBox.currentIndexChanged.connect(self.dTypeSelectionChange)
        #self.dTypeSelection = self.dTypeSelectorBox.currentText()  
        self.inputArraySelectorBox = QtWidgets.QComboBox()
        self.inputArraySelectorBox.addItems(camVolumeSlider.getArrayKeys()) 
        self.inputArraySelectorBox.setCurrentIndex(4) 
        self.inputArraySelectorBox.currentIndexChanged.connect(self.inputArraySelectionChange)
        
        self.displayArraySelectorBox = QtWidgets.QComboBox()
        self.displayArraySelectorBox.addItems(camVolumeSlider.getArrayKeys())  
        self.displayArraySelectorBox.setCurrentIndex(18)          
        self.displayArraySelectorBox.currentIndexChanged.connect(self.displayArraySelectionChange)
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")       
        self.button2 = QtWidgets.QPushButton("Set Slices")      
        #self.button3 = QtWidgets.QPushButton("Average Volumes")  
        self.button4 = QtWidgets.QPushButton("subtract baseline")          
        self.button5 = QtWidgets.QPushButton("run DF/F0")  
        self.button6 = QtWidgets.QPushButton("export to Window")  
        self.button7 = QtWidgets.QPushButton("set data Type") 
        self.button8 = QtWidgets.QPushButton("multiply") 
        self.button9 = QtWidgets.QPushButton("export array") 
              
        self.button12 = QtWidgets.QPushButton("open 3D viewer") 
        self.button13 = QtWidgets.QPushButton("close 3D viewer")         

        #labels
        self.volumeLabel = QtWidgets.QLabel("# of volumes: ")
        self.volumeText = QtWidgets.QLabel("  ")
        
        self.shapeLabel = QtWidgets.QLabel("array shape: ")
        self.shapeText = QtWidgets.QLabel(str(camVolumeSlider.getArrayShape()))   
        
        self.dataTypeLabel = QtWidgets.QLabel("current data type: ")          
        self.dataTypeText = QtWidgets.QLabel(str(camVolumeSlider.getDataType()))   
        self.dataTypeChangeLabel = QtWidgets.QLabel("new data type: ")          

        self.inputArrayLabel = QtWidgets.QLabel("input array order: ") 
        self.displayArrayLabel = QtWidgets.QLabel("display array order: ") 

        #checkbox
        #self.XYviewerLabel = QtWidgets.QLabel("XY Viewer: ")
        #self.XYviewer=CheckBox()
        #self.XYviewer.setChecked(False)
        #self.XYviewer.stateChanged.connect(self.XYviewerClicked)    
        
        self.trim_last_frameLabel = QtWidgets.QLabel("Trim Last Frame: ")
        self.trim_last_frame_checkbox = CheckBox()
        self.trim_last_frame_checkbox.setChecked(self.trim_last_frame)
        self.trim_last_frame_checkbox.stateChanged.connect(self.trim_last_frameClicked)          
        
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        
        layout.addWidget(self.spinLabel2, 4, 0)
        layout.addWidget(self.SpinBox2, 4, 1)
        layout.addWidget(self.button2, 4, 2) 
        
        #layout.addWidget(self.spinLabel3, 5, 0)
        #layout.addWidget(self.SpinBox3, 5, 1)
        #layout.addWidget(self.button3, 5, 2) 

        layout.addWidget(self.spinLabel4, 6, 0)
        layout.addWidget(self.SpinBox4, 6, 1)      
        layout.addWidget(self.button4, 6, 2) 

        layout.addWidget(self.spinLabel6, 7, 0)
        layout.addWidget(self.SpinBox6, 7, 1)
        layout.addWidget(self.spinLabel7, 7, 2)
        layout.addWidget(self.SpinBox7, 7, 3)        
        layout.addWidget(self.button5, 7, 4)         

        layout.addWidget(self.volumeLabel, 8, 0)         
        layout.addWidget(self.volumeText, 8, 1) 
        layout.addWidget(self.shapeLabel, 9, 0)         
        layout.addWidget(self.shapeText, 9, 1) 
        
        layout.addWidget(self.spinLabel8, 10, 0)
        layout.addWidget(self.SpinBox8, 10, 1)
        layout.addWidget(self.button8, 10, 2)          
        
        layout.addWidget(self.dataTypeLabel, 11, 0)         
        layout.addWidget(self.dataTypeText, 11, 1)   
        layout.addWidget(self.dataTypeChangeLabel, 11, 2)
        layout.addWidget(self.dTypeSelectorBox, 11,3) 
        layout.addWidget(self.button7, 11, 4)   
        
        #layout.addWidget(self.XYviewerLabel, 12, 0) 
        #layout.addWidget(self.XYviewer, 12, 1) 
        
        layout.addWidget(self.button6, 13, 0)          
        layout.addWidget(self.button1, 13, 4)          
        layout.addWidget(self.button6, 13, 0)          
        layout.addWidget(self.button1, 13, 4) 
        layout.addWidget(self.button9, 14, 0)   
        
        
        layout.addWidget(self.spinLabel9, 15, 0)
        layout.addWidget(self.SpinBox9, 15, 1)

        layout.addWidget(self.spinLabel10, 16, 0)
        layout.addWidget(self.SpinBox10, 16, 1)
        layout.addWidget(self.trim_last_frameLabel, 17, 0)         
        layout.addWidget(self.trim_last_frame_checkbox, 17, 1)        
        
        layout.addWidget(self.inputArrayLabel, 18, 0)
        layout.addWidget(self.inputArraySelectorBox, 18, 1)
        
        layout.addWidget(self.displayArrayLabel, 18, 2)
        layout.addWidget(self.displayArraySelectorBox, 18, 3)
        
        layout.addWidget(self.button12, 19, 0)  
        layout.addWidget(self.button13, 19, 1)  
        
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox9.valueChanged.connect(self.setTheta)        
        self.SpinBox10.valueChanged.connect(self.setShiftFactor)       
        
        #connect buttons
        self.button1.clicked.connect(self.autoLevel)
        self.button2.clicked.connect(self.updateVolumeValue)
        #self.button3.clicked.connect(self.averageByVol)
        self.button4.clicked.connect(self.subtractBaseline)
        self.button5.clicked.connect(self.ratioDFF0)
        self.button6.clicked.connect(self.exportToWindow)
        self.button7.clicked.connect(self.dTypeSelectionChange)    
        self.button8.clicked.connect(self.multiplyByFactor)   
        self.button9.clicked.connect(self.exportArray)        
        self.button12.clicked.connect(self.startViewer)           
        self.button13.clicked.connect(self.closeViewer)  
        
        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider.updateDisplay_sliceNumberChange(value)
        return       
    
    def autoLevel(self):
        camVolumeSlider.displayWindow.imageview.autoLevels()
        return 
    
    def updateVolumeValue(self):
        value = self.SpinBox2.value()
        noVols = int(camVolumeSlider.getNFrames()/value)
        camVolumeSlider.updateVolsandFramesPerVol(noVols, value)
        self.volumeText.setText(str(noVols))
        
        camVolumeSlider.updateDisplay_volumeSizeChange()
        self.shapeText.setText(str(camVolumeSlider.getArrayShape())) 
        
        if (value)%2 == 0:
            self.SpinBox1.setRange(0,value-1) #if even, display the last volume 
            self.slider1.setMaximum(value-1)
        else:
            self.SpinBox1.setRange(0,value-2) #else, don't display the last volume 
            self.slider1.setMaximum(value-2)
        
        self.updateVolSpinBoxes()         
        return

    def updateVolSpinBoxes(self):     
        #self.SpinBox3.setRange(0,camVolumeSlider.getNVols())

        self.SpinBox6.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox7.setRange(0,camVolumeSlider.getNVols())
        return

    def getBaseline(self):
        return self.SpinBox4.value()

    def getF0(self):
        return self.SpinBox6.value(), self.SpinBox7.value()

#    def getVolsToAverage(self):
#        return self.SpinBox3.value()

#    def averageByVol(self):
#        camVolumeSlider.averageByVol()
#        return

    def subtractBaseline(self):
        camVolumeSlider.subtractBaseline()
        return
    
    def ratioDFF0(self):
        camVolumeSlider.ratioDFF0()
        return
    
    def exportToWindow(self):
        camVolumeSlider.exportToWindow()
        return
    
    def dTypeSelectionChange(self):
        camVolumeSlider.setDType(self.dTypeSelectorBox.currentText())   
        self.dataTypeText = QtWidgets.QLabel(str(camVolumeSlider.getDataType()))  
        return
    
    def multiplyByFactor(self):
        camVolumeSlider.multiplyByFactor(self.SpinBox8.value())
        return
    
    def exportArray(self):
        camVolumeSlider.exportArray()
        return

    def startViewer(self):
        camVolumeSlider.startViewer()
        return   

    def closeViewer(self):
        camVolumeSlider.closeViewer()
        return   
    
    def setTheta(self):
        self.theta = self.SpinBox9.value()

    def setShiftFactor(self):
        self.shiftFactor = self.SpinBox10.value()    

    def trim_last_frameClicked(self):
        self.trim_last_frame = self.trim_last_frame_checkbox.isChecked()

    def inputArraySelectionChange(self, value):
        camVolumeSlider.setInputArrayOrder(self.inputArraySelectorBox.currentText()) 
        return
    
    def displayArraySelectionChange(self, value):
        camVolumeSlider.setDisplayArrayOrder(self.displayArraySelectorBox.currentText()) 
        return

    def close(self):
        camVolumeSlider.closeViewer()
        camVolumeSlider.displayWindow.close()
        camVolumeSlider.dialogbox.destroy()
        camVolumeSlider.end()
        self.closeAllWindows()
        return        


class plot3D_options(QtWidgets.QDialog):
    def __init__(self, prob, threshold, parent = None):
        super(plot3D_options, self).__init__(parent)
        
        self.prob = prob
        self.threshold = threshold
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Amount of downsampling (0-1)") 
        self.SpinBox1 = QtWidgets.QDoubleSpinBox()
        self.SpinBox1.setDecimals(4)
        self.SpinBox1.setRange(0,1.0000)
        self.SpinBox1.setValue(self.prob)
 
        self.spinLabel2 = QtWidgets.QLabel("Threshold level") 
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,10000)
        self.SpinBox2.setValue(self.threshold)

        #checkboxes
        self.checkBox1_label = QtWidgets.QLabel("Display array outline")  
        self.checkBox1 = QtWidgets.QCheckBox()
        self.checkBox2_label = QtWidgets.QLabel("Display plot axis")  
        self.checkBox2 = QtWidgets.QCheckBox()
        #buttons
        #self.button1 = QtWidgets.QPushButton("Close")       
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.spinLabel2, 2, 0)        
        layout.addWidget(self.SpinBox2, 2, 1)
        layout.addWidget(self.checkBox1_label, 3, 0)        
        layout.addWidget(self.checkBox1, 3, 1) 
        layout.addWidget(self.checkBox2_label, 4, 0)        
        layout.addWidget(self.checkBox2, 4, 1) 
        #layout.addWidget(self.button1, 5, 2) 
               
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("3D Plot Options")
        
        #connect sliders & spinboxes
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox2.valueChanged.connect(self.spinBox2ValueChange)    
        
        #connect checkboxes
        self.checkBox1.stateChanged.connect(self.checkBox1ValueChange)
        self.checkBox2.stateChanged.connect(self.checkBox2ValueChange)
        
        #connect buttons
        #self.button1.clicked.connect(self.close)
         
        return
          
    def spinBox1ValueChange(self, value):
        camVolumeSlider.viewer.setProb(value)
        return       
    
    def spinBox2ValueChange(self, value):
        camVolumeSlider.viewer.setThreshold(value)
        return  

    def checkBox1ValueChange(self, value):
        camVolumeSlider.viewer.setPlotCube(value)
        return

    def checkBox2ValueChange(self, value):
        camVolumeSlider.viewer.setPlotAxis((not value))
        return
        
        return        


class exportDialog_win(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(exportDialog_win, self).__init__(parent)
                
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #buttons
        self.button1 = QtWidgets.QPushButton("Export Z view")     
        self.button2 = QtWidgets.QPushButton("Export X view")  
        self.button3 = QtWidgets.QPushButton("Export Y view")  
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.button1, 0, 0) 
        layout.addWidget(self.button2, 1, 0) 
        layout.addWidget(self.button3, 2, 0) 
               
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Export Options")
        
        #connect buttons
        self.button1.clicked.connect(self.exportZ)
        self.button2.clicked.connect(self.exportX)
        self.button3.clicked.connect(self.exportY)         
        return
          
    def exportZ(self):
        self.z_displayWindow = Window(camVolumeSlider.viewer.getZWin(),'Z view')
        return       
    
    def exportX(self):
        self.x_displayWindow = Window(camVolumeSlider.viewer.getXWin(),'X view')
        return 

    def exportY(self):
        self.y_displayWindow = Window(camVolumeSlider.viewer.getYWin(),'Y view')
        return 
         