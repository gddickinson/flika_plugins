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
from PIL import Image, ImageFont, ImageDraw, ImageOps
from copy import deepcopy

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


class Annotator(BaseProcess_noPriorWindow):
    """
    Annotator
    
    Burns number index into image stack
    
    Requires PIL to be installed ('pip install pillow')
    
    """
    def __init__(self):
        if g.settings['annotator'] is None or 'anchor' not in g.settings['annotator']:
            s = dict()
            s['fontSize'] = 24
            s['fontColour'] = 0
            s['anchor'] = 0          
 
            g.settings['annotator'] = s
        super().__init__()


    def __call__(self, keepSourceWindow=False):
        g.settings['annotator']['fontSize'] = self.getValue('fontSize')
        g.settings['annotator']['fontColour'] = self.getValue('fontColour')
        g.settings['annotator']['anchor'] = self.getValue('anchor')       
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)


    def gui(self):
        s=g.settings['annotator']
        self.gui_reset()
        #windows
        self.window_button = WindowSelector()

        #buttons         
        self.addIndex_button = QPushButton('Add Index')
        self.addIndex_button.pressed.connect(self.addIndex)  
        
        #boxes
        self.fontSize_slider = pg.SpinBox(int=True, step=1)
        self.fontSize_slider.setValue(s['fontSize'])
        
        self.fontColour_combobox = pg.ComboBox()
        self.fontColours = {'white': 1, 'black': 2}
        self.fontColour_combobox.setItems(self.fontColours)  
        
        self.textPosition_combobox = pg.ComboBox()
        self.textPosition = {'ms': 1, 'ma': 2, 'ls': 3, 'mb': 4, 'mt': 5, 'mm': 6, 'md': 7, 'rs': 8}
        self.textPosition_combobox.setItems(self.textPosition)         

        
        #self.exportFolder = FolderSelector('*.txt')
        self.items.append({'name': 'window', 'string': 'Select Window', 'object': self.window_button})
        self.items.append({'name': 'fontSize', 'string': 'Font Size', 'object': self.fontSize_slider})  
        self.items.append({'name': 'fontColour', 'string': 'Font Colour', 'object': self.fontColour_combobox}) 
        self.items.append({'name': 'textPosition', 'string': 'Anchor Position', 'object': self.textPosition_combobox})         
        self.items.append({'name': 'addIndex', 'string': 'Add Numbers to Image Stack ', 'object': self.addIndex_button})       
        super().gui()


    def addIndex(self):
        #get data
        A  = deepcopy(self.getValue('window').image) 
        
        A_labelled = np.zeros_like(A)
        
        font = ImageFont.truetype(r'C:/Windows/Fonts/Arial/arial.ttf', self.getValue('fontSize'))
        colour = self.fontColour_combobox.currentText()
        position = self.textPosition_combobox.currentText()
            
        #print(colour,position)
        
        #add index
        
        for i in range(len(A)):

            img = Image.fromarray(A[i])
            
            img = img.rotate(90, Image.NEAREST, expand = 1)
            img = ImageOps.flip(img)
            
            img_editable = ImageDraw.Draw(img)
            img_editable.text((2, 2), str(i), fill=colour, anchor=position, font=font)
            
            img = ImageOps.flip(img)
            img = img.rotate(-90, Image.NEAREST, expand = 1)
            
            A_labelled[i] = np.array(img)
                
        
        #display stack in new window
        self.indexed_win = Window(A_labelled,'Indexed')
        return
               
 
annotator = Annotator()
