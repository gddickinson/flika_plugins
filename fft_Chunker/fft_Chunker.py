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

try:
    from scipy.fft import fft, fftfreq 
except:
    from scipy.fftpack import fft, fftfreq

import time

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


import pandas as pd
from matplotlib import pyplot as plt

#chunking fft function
def fft_chunks(l, n, time_step):
    '''

    Parameters
    ----------
    l : TYPE
        list of numbers.
    n : TYPE
        chunk size.

    Returns
    ------
    TYPE
        dict of lists with fft, power and frequency (Hz) for each quck

    '''
      
    # looping till length l 
    df = pd.DataFrame()
    chunk_num = 1
    
    for i in range(0, len(l), n):
        chunk = l[i:i + n]
        fft_chunk = fft(chunk)
        power_chunk = np.abs(fft_chunk) 
        freq_chunk = fftfreq(chunk.size, d=time_step)
        name1='power_{}'.format(chunk_num)
        name2='frequency_{}'.format(chunk_num)
        name3='chunk_{}'.format(chunk_num)
        d= {name1: power_chunk, name2: freq_chunk, name3: chunk}
        newDF = pd.DataFrame(data=d)
        df = pd.concat([df,newDF], axis=1)
        chunk_num += 1
          
    return df


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
        self.button=QPushButton('Select Filename')
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

class FFT_Chunker(BaseProcess_noPriorWindow):
    """
    fft chunk analysis
    ------------------
    
    input:      csv file with one or more time traces (columns) 
    
    variables:  chunk_size and time_step (s)
    
    analysis:   trace broken into chunks using chunk_size
                FFT analysis on each chunk using scipy.fft -> FFT_chunk
                power reported as abs(FFT_chunk)
                frequency calculated using scipy.fftfreq(chunk.size, time_step)
    
    output:     saves seperate results csv file for each trace in folder of input file 
                each results file has power and frequency results for every chunk (columns)
    """
    def __init__(self):
        if g.settings['fftChunker'] is None or 'timestep' not in g.settings['fftChunker']:
            s = dict()            
            s['timestep'] = 1
            s['chunkSize'] = 128
                  
            g.settings['fftChunker'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
        
        self.timestep_Max = 1000000
        self.chunkSize_Max = 1000000      
        
        self.filename = ''        
        self.exportPath = ''
        self.plotGraph = False
        

    def __call__(self):
        '''
        reset saved parameters
        '''
        g.settings['fftChunker']['timestep'] = 1
        g.settings['fftChunker']['chunkSize'] = 128
             
        #currently not saving parameter changes
        
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):
        self.gui_reset()
        s=g.settings['fftChunker']

        #buttons      
        self.runAnalysis_button = QPushButton('Run Analysis')
        self.runAnalysis_button.pressed.connect(self.runAnalysis)              
              
        #checkbox
        self.plot_checkbox = CheckBox()
        self.plot_checkbox.setChecked(self.plotGraph)
        
        
        #self.setExportFolder_button = FolderSelector('*.csv')
        
        #spinboxes
        #MEPPS
        self.chunkSize_Box = pg.SpinBox(int=True, step=1)
        self.chunkSize_Box.setMinimum(0)
        self.chunkSize_Box.setMaximum(1000000)        
        self.chunkSize_Box.setValue(s['chunkSize'])
        
        self.timestep_Box = pg.SpinBox(int=False, step=0.01)
        self.timestep_Box.setMinimum(0)
        self.timestep_Box.setMaximum(1000000)        
        self.timestep_Box.setValue(s['timestep'])        
        
        self.getFile = FileSelector()
 
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        self.items.append({'name': 'blank ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'chunksize ', 'string': 'Set chunk size ', 'object': self.chunkSize_Box})  
        self.items.append({'name': 'timestep ', 'string': 'Set timestep: ', 'object': self.timestep_Box}) 
        self.items.append({'name': 'filepath ', 'string': '', 'object': self.getFile})              
        self.items.append({'name': 'run', 'string': '', 'object': self.runAnalysis_button }) 
        self.items.append({'name': 'plot', 'string': 'Plot results for 1st Trace', 'object': self.plot_checkbox })         
             
        #self.items.append({'name': 'blank ', 'string': '-----------------   Export Path    -----------------', 'object': None})    
        #self.items.append({'name': 'setPath ', 'string': '', 'object': self.setExportFolder_button })            
        
        super().gui()
        ######################################################################
        return

# #TODO replace with function from flika
#     def getFileName(self):
#         options = QFileDialog.Options()
#         options |= QFileDialog.DontUseNativeDialog
#         fileName, _ = QFileDialog.getOpenFileName(self,"Open File", "","All Files (*);;Python Files (*.py)", options=options)
#         if fileName:
#             self.filename= fileName
#             #newText = "Selected file: " + openFile.currentFile
#             #self.label1.setText(newText)
#             openFile.fileType = fileName.split('.')[-1]
#             print(openFile.fileType)
#         else:
#             self.filename = ''
#         return
        

    def runAnalysis(self):
        self.filename = self.getFile.value()
        chunk_size = self.chunkSize_Box.value()
        timestep = self.timestep_Box.value()
        timeStr = time.strftime("%Y%m%d-%H%M%S")
        
        if self.filename == '':
            print('File not loaded!')
            g.m.statusBar().showMessage('File not loaded!')
            return
        else:
            #import trace/traces
            data = pd.read_csv(self.filename, header = None)
            columns = list(data) 
                        
            #initiate result dict (for multiple traces)
            result_dict = {}
            
            #fft analysis
            for i in columns:   
                # analyse each column
                result = (fft_chunks(data[i].to_numpy(),chunk_size, timestep)) 
                #result = (fft_chunks(sig,chunk_size)) # FOR TESTING     
                # add to result_dict
                result_dict.update( {i : result} )
                print('Trace {} analysed, {} chunks processed'.format(str(i), str((len(data[i].to_numpy() )/chunk_size))))

            #export results
            savePath = os.path.dirname(self.filename)
            
            for key in result_dict:  
                saveName = os.path.join(savePath,'FFT_Chunker_Batch_{}_Trace_{}.csv'.format(timeStr,str(key)))
                result_dict[key].to_csv(saveName)
                print('File {} saved'.format(saveName))
                g.m.statusBar().showMessage('File {} saved'.format(saveName))
                
            if self.plot_checkbox.isChecked():
                ### plot test result
                result_data = result_dict[0] # just results for 1st trace
                numChunks = int(len(list(result_data.keys()))/3)
                if numChunks > 20:
                    g.alert('More than 20 plots would be generated - aborting plotting')
                    return


                for i in range(1,numChunks+1):
                    print('Plotting chunk {}'.format(str(i)))  
                    plt.figure(i)
                    plt.subplot(211)
                    plt.plot(result_data['chunk_{}'.format(str(i))])
                    plt.subplot(212)                    
                    plt.scatter(result_data['frequency_{}'.format(str(i))],result_data['power_{}'.format(str(i))])
                    #plt.plot(result_data['frequency_{}'.format(str(i))],result_data['power_{}'.format(str(i))])                    
                    plt.title("FFT analysis - chunk {}".format(str(i)))
                    plt.xlabel("frequency")
                    plt.ylabel("power")
                    plt.show()
    
 
fft_Chunker = FFT_Chunker()


if __name__ == "__main__":
    pass


