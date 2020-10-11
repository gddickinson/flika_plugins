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
from skimage.measure import block_reduce

# def groupedAvg(myArray, N=2):
#     #no extra libraries needed  
#     cum = np.cumsum(myArray,0)
#     result = cum[N-1::N]/float(N)
#     result[1:] = result[1:] - result[:-1]

#     remainder = myArray.shape[0] % N
#     if remainder != 0:
#         if remainder < myArray.shape[0]:
#             lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
#         else:
#             lastAvg = cum[-1]/float(remainder)
#         result = np.vstack([result, lastAvg])

#     return result

def groupedAvg(A, block_size=(2,1), func=np.mean):
    #uses skimage
    cval=np.mean(A)
    print('x binning every {} values using {}'.format(block_size[0],func.__name__))
    return block_reduce(A, block_size=block_size, func=func, cval=cval)
    


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

class PuffMapper(BaseProcess_noPriorWindow):
    """
    puff heat map generator
    ------------------
    
    input:      csv file with one or more time traces (columns) 
    
    variables:  
    
    analysis:   
    
    output:     
    """
    def __init__(self):
        if g.settings['puffMapper'] is None or 'yExpand' not in g.settings['puffMapper']:
            s = dict()            
            s['nFrames'] = 1000           
            s['startFrame'] = 0 
            s['sortedByMax'] = False             
            s['xBin'] = 0 
            s['yExpand'] = 0                  
            
            g.settings['puffMapper'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
            
        

    def __call__(self, nFrames, startFrame, sortedByMax, xBin, yExpand, keepSourceWindow=False):
        '''
        '''

        #currently not saving parameter changes on call
        g.settings['puffMapper']['nFrames'] = nFrames         
        g.settings['puffMapper']['startFrame'] = startFrame   
        g.settings['puffMapper']['sortedByMax'] = sortedByMax  
        g.settings['puffMapper']['xBin'] = xBin  
        g.settings['puffMapper']['yExpand'] = yExpand         
        
        g.m.statusBar().showMessage("Starting puff mapper...")
        return



    def closeEvent(self, event):
        self.clearPlots()
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):      
        self.filename = ''          
        self.gui_reset()        
        s=g.settings['puffMapper']  
        
        #buttons      
        self.plotChart_button = QPushButton('Create Line plot')
        self.plotChart_button.pressed.connect(self.plotData)  
                
        self.generateHeatmap_button = QPushButton('Create Heatmap')        
        self.generateHeatmap_button.pressed.connect(self.createHeatmap)  

                         
        #checkbox
        self.sorted_checkbox = CheckBox()
        self.sorted_checkbox.setChecked(s['sortedByMax'])

        #comboboxes
        self.xFunc_Box = pg.ComboBox()
        xFuncs = {'mean':np.mean, 'max':np.max, 'median':np.median}
        self.xFunc_Box.setItems(xFuncs)

        
        #spinboxes
        #Frames
        self.startFrame_Box = pg.SpinBox(int=True, step=1)
        self.startFrame_Box.setMinimum(0)
        self.startFrame_Box.setMaximum(1000000)  
        self.startFrame_Box.setValue(s['startFrame'])            
                
        self.nFrames_Box = pg.SpinBox(int=True, step=1)
        self.nFrames_Box.setMinimum(0)
        self.nFrames_Box.setMaximum(1000000)  
        self.nFrames_Box.setValue(s['nFrames'])          

        self.xBin_Box = pg.SpinBox(int=True, step=2)
        self.xBin_Box.setMinimum(1)
        self.xBin_Box.setMaximum(10000)  
        self.xBin_Box.setValue(s['xBin'])   

        self.yExpand_Box = pg.SpinBox(int=True, step=1)
        self.yExpand_Box.setMinimum(1)
        self.yExpand_Box.setMaximum(1000)  
        self.yExpand_Box.setValue(s['yExpand'])   


        #export file selector
        self.getFile = FileSelector()
        
        #connections
        self.getFile.valueChanged.connect(self.loadData)
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        self.items.append({'name': 'blank1 ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'startFrame', 'string': 'Set start frame ', 'object': self.startFrame_Box})          
        self.items.append({'name': 'nFrames', 'string': 'Set number of frames ', 'object': self.nFrames_Box}) 
        self.items.append({'name': 'xBin', 'string': 'Set x-axis binning number ', 'object': self.xBin_Box})  
        self.items.append({'name': 'xBinType', 'string': 'Set x-axis bin function ', 'object': self.xFunc_Box})  
        self.items.append({'name': 'yExpand', 'string': 'Set y-axis expansion value ', 'object': self.yExpand_Box})          
        self.items.append({'name': 'sortedByMax', 'string': 'Sorted by maximum', 'object': self.sorted_checkbox})         
        self.items.append({'name': 'blank ', 'string': '-------------------------------------------', 'object': None})           
        
        self.items.append({'name': 'filepath ', 'string': '', 'object': self.getFile})              

        self.items.append({'name': 'heatmap', 'string': '', 'object': self.generateHeatmap_button })            
        self.items.append({'name': 'lineplot', 'string': '', 'object': self.plotChart_button }) 

        
        super().gui()
        ######################################################################
        return



    def loadData(self):
        self.filename = self.getFile.value()
        self.data = pd.read_csv(self.filename, header=None, skiprows=1, index_col = 0)
        
        #drop extra x columns
        self.data = self.data[self.data.columns[::2]] 
        #drop columns with na values
        self.data = self.data.dropna(axis=1, how='all')        
        #set column names by number
        nCols = len(self.data.columns)
        colNames = list(range(0,nCols))
        self.data.columns = colNames
        #copy df and make one sorted by intensity
        self.dataSorted = self.data.copy()
        self.dataSorted = self.dataSorted.iloc[:, self.dataSorted.max().sort_values(ascending=False).index]

        #set nFrames max
        nRows = len(self.data)
        self.nFrames_Box.setMaximum(nRows)  
        self.nFrames_Box.setValue(nRows)  

        print('-------------------------------------')
        print('Data loaded (first 5 rows displayed):')
        print(self.data.head())
        print('-------------------------------------')



    def plotData(self):
        ### plot test result
        self.data.plot()
        plt.show()
        g.m.statusBar().showMessage('scatter plot created') 
        print('scatter plot created')    
        return

    def createHeatmap(self):
        ### create heatmap image
        if self.sorted_checkbox.isChecked():
            try:
                mapData = self.dataSorted.to_numpy()
            except:
                mapData = self.dataSorted.values
        else:
            try:
                mapData = self.data.to_numpy()
            except:
                mapData = self.data.values              
            

        nRows,nCols = mapData.shape
        print('original data has: {} columns, {} rows'.format(nRows, nCols))
        start = self.startFrame_Box.value()
        end = self.startFrame_Box.value() + self.nFrames_Box.value()
        
        if end > nRows:
            end = nRows

        #crop data
        img = mapData[start:end]
        
        #x binning
        img = groupedAvg(img,block_size=(self.xBin_Box.value(),1),func=self.xFunc_Box.value())

        #y expansion
        img = np.repeat(img,self.yExpand_Box.value(),axis=1)
        print('y-axis expanded by {}'.format(self.yExpand_Box.value()))
        
        #get new img shape
        nRows_img,nCols_img = img.shape
        print('displayed image has: {} columns, {} rows'.format(nRows_img, nCols_img))
      
        self.heatmapImg = Window(img,name='puff map')
        print('puff map created')
        return

    
    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return




puffMapper = PuffMapper()
	

if __name__ == "__main__":
    pass


