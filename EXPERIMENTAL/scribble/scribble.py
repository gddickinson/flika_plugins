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
from pyqtgraph.Point import Point
from os.path import expanduser
import os
import math

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


class FolderSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Select Folder')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''
        
    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    


class canvasGUI(QDialog):
    def __init__(self, parent = None):
        super(canvasGUI, self).__init__(parent)
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 150
                
        #ComboBox
        self.integrationSelectorBoxLabel = QLabel("drawing type") 
        self.integrationSelectorBox = QComboBox()
        self.integrationSelectorBox.addItems(["pen", "???", "???"])
        self.integrationSelectorBox.currentIndexChanged.connect(self.integrationSelectionChange)
        self.integrationSelection = self.integrationSelectorBox.currentText()        
        
        self.channelSelectorBoxLabel = QLabel("channel") 
        self.channelSelectorBox = QComboBox()
        self.channelSelectorBox.addItems(["R", "G", "B"])
        self.channelSelectorBox.currentIndexChanged.connect(self.channelSelectionChange)
        self.channelSelection = self.channelSelectorBox.currentText()
 
        #buttons
        self.drawButton = QPushButton('Draw')
        self.drawButton.pressed.connect(self.draw)
        
        self.stopDrawButton = QPushButton('Stop Drawing')
        self.stopDrawButton.pressed.connect(self.stopDraw)
        
        #self.exportButton = QPushButton('Export')
        #self.exportButton.pressed.connect(self.export)        
        
        #grid layout
        layout = QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(self.integrationSelectorBoxLabel, 4, 0)        
        layout.addWidget(self.integrationSelectorBox, 4, 1)
        layout.addWidget(self.channelSelectorBoxLabel, 5, 0)  
        layout.addWidget(self.channelSelectorBox, 5, 1)
        layout.addWidget(self.drawButton, 7, 0)           
        layout.addWidget(self.stopDrawButton, 8, 0)
                     
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("options GUI")

        self.show()

    def channelSelectionChange(self):
        return

    def integrationSelectionChange(self):
        return

    def stopDraw(self):
        #see pyqtgraph.graphicsItems.ImageItem
        self.img.drawKernel = None
        self.img.drawKernelCenter = None
        self.img.drawMode = None
        self.img.drawMask = None
        return

    def draw(self):
        #see pyqtgraph.graphicsItems.ImageItem
        self.canvasWindow  = scribble.canvasWindow
        self.img = pg.ImageItem(np.zeros((100,100)))
        self.canvasWindow.imageview.view.addItem(self.img)
        #view.setAspectLocked(True)
        kern = np.array([
            [0.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 0.0]
        ])
        self.img.setDrawKernel(kern, mask=kern, center=(1,1), mode='add')
        self.img.setLevels([0, 10])
        
        return

#    def export(self):
#        data = linescan.getData()       
#        return

class Scribble(BaseProcess_noPriorWindow):
    """
    Scribble
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self):
        '''
        
        '''
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)


    def gui(self):
        self.gui_reset()
        #windows
        self.canvasWindow = WindowSelector()

        #buttons
        self.canvasWindowCreateButton = QPushButton('Create Canvas Window')
        self.canvasWindowCreateButton.pressed.connect(self.createCanvasWindow) 
        
        self.startGUIButton = QPushButton('Start GUI')
        self.startGUIButton.pressed.connect(self.startGUI)  
        
        #self.exportFolder = FolderSelector('*.txt')
        self.items.append({'name': 'canvasWindow', 'string': 'Select Canvas Window', 'object': self.canvasWindow})
        self.items.append({'name': 'createCanvasWindow', 'string': 'Create Canvas Window', 'object': self.canvasWindowCreateButton})        
        self.items.append({'name': 'options', 'string': 'Start GUI: ', 'object': self.startGUIButton})       
        super().gui()


    def startGUI(self):
        #create startGUI instance
        self.canvasGUI = canvasGUI()      
        return        

    def createCanvasWindow(self):
        self.canvasArray = np.zeros((100,100,1))
        self.canvasWindow = Window(self.canvasArray)
        return
 
scribble = Scribble()
