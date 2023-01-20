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


import pandas as pd
from matplotlib import pyplot as plt



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
        if g.settings['locsAndTracksPlotter'] is None or 'filename' not in g.settings['locsAndTracksPlotter']:
            s = dict()            
            s['filename'] = '' 
            s['filetype'] = 'flika'                           
            
            g.settings['locsAndTracksPlotter'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
            
        

    def __call__(self, filename, filetype, keepSourceWindow=False):
        '''
        '''

        #currently not saving parameter changes on call
        g.settings['locsAndTracksPlotter']['filename'] = filename 
        g.settings['locsAndTracksPlotter']['filetype'] = filetype       

        
        g.m.statusBar().showMessage("plotting data...")
        return


    def closeEvent(self, event):
        self.clearPlots()
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):      
        self.filename = '' 
        self.filetype = 'flika'          
        self.gui_reset()        
        s=g.settings['locsAndTracksPlotter']  
        
        #buttons      
        self.plotData_button = QPushButton('Plot')
        self.plotData_button.pressed.connect(self.plotData)  
                         
        #checkbox
        #self.sorted_checkbox = CheckBox()
        #self.sorted_checkbox.setChecked(s['sortedByMax'])

        #comboboxes
        self.filetype_Box = pg.ComboBox()
        filetypes = {'flika' : 'flika', 'thunderstorm':'thunderstorm', 'xy':'xy'}
        self.filetype_Box.setItems(filetypes)

         
        #data file selector
        self.getFile = FileSelector()
        
        #connections
        self.getFile.valueChanged.connect(self.loadData)
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        self.items.append({'name': 'blank1 ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'filepath ', 'string': '', 'object': self.getFile})    
        self.items.append({'name': 'filetype', 'string': 'filetype', 'object': self.filetype_Box})  
        
       
        self.items.append({'name': 'blank ', 'string': '-------------------------------------------', 'object': None})           
        
          
        self.items.append({'name': 'lineplot', 'string': '', 'object': self.plotData_button }) 

        
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



    def plotData(self):
        ### plot point data to current window
        
        
        ### plot track data to current window




        g.m.statusBar().showMessage('point data plotted to current window') 
        print('point data plotted to current window')    
        return



    
    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return




locsAndTracksPlotter = LocsAndTracksPlotter()
	

if __name__ == "__main__":
    pass


