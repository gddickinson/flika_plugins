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
    freq_min = 100000000
    freq_max = 0
    pow_min = 100000000
    pow_max = 0
    
    for i in range(0, len(l), n):
        chunk = l[i:i + n]
        fft_chunk = fft(chunk)
        power_chunk = np.abs(fft_chunk) 
        freq_chunk = fftfreq(chunk.size, d=time_step)


        #replace neqative freq values with NaN - and corresponding power values
        mask = freq_chunk < 0         
        freq_chunk = np.ma.masked_array(data=freq_chunk,mask=mask,fill_value=np.nan).filled()
        power_chunk = np.ma.masked_array(data=power_chunk,mask=mask,fill_value=np.nan).filled()        

        #convert power and freq to 1og10        
        power_chunk = np.log10(power_chunk)
        freq_chunk =  np.log10(freq_chunk)
             
        name1='power_{}'.format(chunk_num)
        name2='frequency_{}'.format(chunk_num)
        name3='chunk_{}'.format(chunk_num)
        d= {name1: power_chunk, name2: freq_chunk, name3: chunk}
        newDF = pd.DataFrame(data=d)
        df = pd.concat([df,newDF], axis=1)
        
        #get min and max values to set plot windows
        if min(freq_chunk) < freq_min:
            freq_min = min(freq_chunk)
            
        if max(freq_chunk) > freq_max:
            freq_max = max(freq_chunk)
            
        if min(power_chunk) < pow_min:
            pow_min = min(power_chunk)
            
        if max(power_chunk) > pow_max:
            pow_max = max(power_chunk)
        
        chunk_num += 1
          
    return df, freq_min, freq_max, pow_min, pow_max, min(l), max(l)


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
        if g.settings['fft_Chunker'] is None or 'closeplots' not in g.settings['fft_Chunker']:
            s = dict()            
            # s['timestep'] = 1
            # s['chunkSize'] = 128
            # s['plot'] = True            

            s['chunkSize'] = 128 
            s['timestep'] = 1 
            s['numFrames'] = None
            s['numChunks'] = None 
            s['baseline_start'] = 0 
            s['baseline_stop'] = 2 
            s['puff1_start'] = 0
            s['puff1_stop'] = 0
            s['puff2_start'] = 0
            s['puff2_stop'] = 0 
            s['puff3_start'] = 0 
            s['puff3_stop'] = 0
            s['puff4_start'] = 0 
            s['puff4_stop'] = 0 
            s['filepath'] = None
            s['run'] = None 
            s['plot'] = True  
            s['closeplots'] = None

                  
            g.settings['fft_Chunker'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
            
        


#chunkSize , timestep , numFrames, numChunks, blank2, blank3, baseline_start , baseline_stop, puff1_start ,puff1_stop, puff2_start, puff2_stop, puff3_start , puff3_stop, puff4_start , puff4_stop, filepath , run, plot, closeplots
    def __call__(self):
        '''
        reset saved parameters
        '''
        
        #currently not saving parameter changes on call - using updateParms()
        self.clearPlots()
        self.updateParams()
        return


    def updateParams(self):
        g.settings['fft_Chunker']['chunkSize'] = self.chunkSize_Box.value()
        g.settings['fft_Chunker']['timestep'] = self.timestep_Box.value()
        # g.settings['fft_Chunker']['baseline_start'] = baseline_start
        # g.settings['fft_Chunker']['baseline_stop'] = baseline_stop
        # g.settings['fft_Chunker']['puff1_start'] = puff1_start
        # g.settings['fft_Chunker']['puff1_stop'] = puff1_stop
        # g.settings['fft_Chunker']['puff2_start'] = puff2_start 
        # g.settings['fft_Chunker']['puff2_stop'] = puff2_stop
        # g.settings['fft_Chunker']['puff3_start'] = puff3_start 
        # g.settings['fft_Chunker']['puff3_stop'] = puff3_stop
        # g.settings['fft_Chunker']['puff4_start'] = puff4_start
        # g.settings['fft_Chunker']['puff4_stop'] = puff4_stop
        # g.settings['fft_Chunker']['filepath'] = filepath
        g.settings['fft_Chunker']['plot'] = self.plot_checkbox.isChecked()     
        
        return


    def closeEvent(self, event):
        self.clearPlots()
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):
        
        self.filename = ''  
        
        self.gui_reset()        
        s=g.settings['fft_Chunker']


        #buttons      
        self.runAnalysis_button = QPushButton('Run Analysis')
        self.runAnalysis_button.pressed.connect(self.runAnalysis)  
        self.setSavename_button = QPushButton('Set SaveName')        
        self.setSavename_button.pressed.connect(self.setSavename)  

        self.closePlots_button = QPushButton('Close plots')
        self.closePlots_button.pressed.connect(self.clearPlots)              
              
        #checkbox
        self.plot_checkbox = CheckBox()
        self.plot_checkbox.setChecked(s['plot'])

        #label
        self.numFrames_label = QLabel()
        self.numFrames_label.setText('No file selected') 
        self.numChunks_label = QLabel()
        self.numChunks_label.setText('No file selected')        
        self.savename_label = QLabel()
        self.savename_label.setText('')                 
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

        self.baseline_start = pg.SpinBox(int=True, step=1)
        self.baseline_start.setMinimum(1)
        self.baseline_start.setMaximum(1000000)        
        self.baseline_start.setValue(1)        
               
        self.baseline_stop = pg.SpinBox(int=True, step=1)  
        self.baseline_stop.setMinimum(0)
        self.baseline_stop.setMaximum(1000000)        
        self.baseline_stop.setValue(0)          
        
        self.puff1_start = pg.SpinBox(int=True, step=1)
        self.puff1_start.setMinimum(1)
        self.puff1_start.setMaximum(1000000)        
        self.puff1_start.setValue(1)        
               
        self.puff1_stop = pg.SpinBox(int=True, step=1)  
        self.puff1_stop.setMinimum(0)
        self.puff1_stop.setMaximum(1000000)        
        self.puff1_stop.setValue(0)         
        
        self.puff2_start = pg.SpinBox(int=True, step=1)
        self.puff2_start.setMinimum(1)
        self.puff2_start.setMaximum(1000000)        
        self.puff2_start.setValue(1)        
               
        self.puff2_stop = pg.SpinBox(int=True, step=1)  
        self.puff2_stop.setMinimum(0)
        self.puff2_stop.setMaximum(1000000)        
        self.puff2_stop.setValue(0)    
        
        self.puff3_start = pg.SpinBox(int=True, step=1)
        self.puff3_start.setMinimum(1)
        self.puff3_start.setMaximum(1000000)        
        self.puff3_start.setValue(1)        
               
        self.puff3_stop = pg.SpinBox(int=True, step=1)  
        self.puff3_stop.setMinimum(0)
        self.puff3_stop.setMaximum(1000000)        
        self.puff3_stop.setValue(0)          
        
        self.puff4_start = pg.SpinBox(int=True, step=1)
        self.puff4_start.setMinimum(1)
        self.puff4_start.setMaximum(1000000)        
        self.puff4_start.setValue(1)        
               
        self.puff4_stop = pg.SpinBox(int=True, step=1)  
        self.puff4_stop.setMinimum(0)
        self.puff4_stop.setMaximum(1000000)        
        self.puff4_stop.setValue(0)          
        
        
        #export file selector
        self.getFile = FileSelector()
        
        
        #connections
        self.chunkSize_Box.valueChanged.connect(self.chunkSizeUpdate)        
        self.getFile.valueChanged.connect(self.loadData) 
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        self.items.append({'name': 'blank1 ', 'string': '-------------   Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'chunksize ', 'string': 'Set chunk size ', 'object': self.chunkSize_Box})  
        self.items.append({'name': 'timestep ', 'string': 'Set timestep: ', 'object': self.timestep_Box}) 
        self.items.append({'name': 'numFrames', 'string': 'Number of frames in trace: ', 'object': self.numFrames_label})        
        self.items.append({'name': 'numChunks', 'string': 'Number of chunks: ', 'object': self.numChunks_label})         
        self.items.append({'name': 'blank2 ', 'string': '-------------    Averaging     ---------------', 'object': None}) 
        self.items.append({'name': 'blank3 ', 'string': '--  If  stop = 0 the group will be ignored  --', 'object': None})         
        self.items.append({'name': 'baseline_start ', 'string': 'Baseline start: ', 'object': self.baseline_start})  
        self.items.append({'name': 'baseline_stop', 'string': 'Baseline stop: ', 'object': self.baseline_stop})          
        self.items.append({'name': 'puff1_start ', 'string': 'Puff Range 1 start: ', 'object': self.puff1_start})  
        self.items.append({'name': 'puff1_stop', 'string': 'Puff Range 1 stop: ', 'object': self.puff1_stop})           
        self.items.append({'name': 'puff2_start ', 'string': 'Puff Range 2 start: ', 'object': self.puff2_start})  
        self.items.append({'name': 'puff2_stop', 'string': 'Puff Range 2 stop: ', 'object': self.puff2_stop})           
        self.items.append({'name': 'puff3_start ', 'string': 'Puff Range 3 start: ', 'object': self.puff3_start})  
        self.items.append({'name': 'puff3_stop', 'string': 'Puff Range 3 stop: ', 'object': self.puff3_stop}) 
        self.items.append({'name': 'puff4_start ', 'string': 'Puff Range 4 start: ', 'object': self.puff4_start})  
        self.items.append({'name': 'puff4_stop', 'string': 'Puff Range 4 stop: ', 'object': self.puff4_stop}) 
        self.items.append({'name': 'blank ', 'string': '-------------------------------------------', 'object': None})           
        self.items.append({'name': 'filepath ', 'string': '', 'object': self.getFile})              
        self.items.append({'name': 'plot', 'string': 'Plot results', 'object': self.plot_checkbox })    
        self.items.append({'name': 'saveName_label', 'string': 'Save Name (if blank, input filename used): ', 'object': self.savename_label})  
        self.items.append({'name': 'saveName', 'string': '', 'object': self.setSavename_button })         
        #self.items.append({'name': 'setPath ', 'string': '', 'object': self.setExportFolder_button })     
        self.items.append({'name': 'run', 'string': '', 'object': self.runAnalysis_button }) 
        self.items.append({'name': 'closeplots', 'string': '', 'object': self.closePlots_button })  
        
        super().gui()
        ######################################################################
        return

    
    def chunkSizeUpdate(self):
        if self.filename == '' :
            return
        
        else:
            self.numChunks = float(self.nFrames/self.chunkSize_Box.value())
            self.numChunks_label.setText(str(self.numChunks)) 
            maxChunks = int(self.numChunks)+1
            self.baseline_start.setMaximum(maxChunks)  
            self.baseline_stop.setMaximum(maxChunks)  
            self.puff1_start.setMaximum(maxChunks)    
            self.puff1_stop.setMaximum(maxChunks)             
            self.puff2_start.setMaximum(maxChunks)    
            self.puff2_stop.setMaximum(maxChunks)              
            self.puff3_start.setMaximum(maxChunks)    
            self.puff3_stop.setMaximum(maxChunks)              
            self.puff4_start.setMaximum(maxChunks)    
            self.puff4_stop.setMaximum(maxChunks)              
                         
        return

    def frameLengthUpdate(self):
        if self.filename == '' :
            return        
        else:
            self.nFrames = len(self.data[0].values)
            self.numFrames_label.setText(str(self.nFrames)) 
            self.chunkSizeUpdate()           
        return


    def loadData(self):
        self.filename = self.getFile.value()
        self.data = pd.read_csv(self.filename, header=None, skiprows=1, index_col = 0)
        self.data = self.data.dropna(axis=1, how='all')
        nCols = len(self.data.columns)
        colNames = list(range(0,nCols))
        self.data.columns = colNames
        self.frameLengthUpdate()


    def setSavename(self):

        text, ok = QInputDialog.getText(None, 'SaveName Dialog', 'Enter save name:')
		
        if ok:

            fft_Chunker.savename_label.setText(str(text))  
        return

    def plotData(self):
        ### plot test result
        result_data = self.result_dict[0] # just results for 1st trace
        numChunks = int(len(list(result_data.keys()))/3)
        if numChunks > 40:
            g.alert('More than 40 plots would be generated - aborting plotting')
            return


        for i in range(1,numChunks+1):
            print('Plotting chunk {}'.format(str(i)))  
            plt.figure(i)
            plt.subplot(211)
            plt.plot(result_data['chunk_{}'.format(str(i))])
            plt.xlabel("frame") 
            plt.ylabel("DF/F0")                    
            plt.ylim(ymin=self.minTime, ymax=self.maxTime)                    
            
            plt.subplot(212)
            x = result_data['frequency_{}'.format(str(i))]
            y = result_data['power_{}'.format(str(i))]
            
            plt.scatter(x,y, s=8, c='blue')
            #plt.plot(result_data['frequency_{}'.format(str(i))],result_data['power_{}'.format(str(i))])                    
            plt.title("FFT analysis - chunk {}".format(str(i)))
            plt.xlabel("frequency")
            plt.ylabel("power")

            
            #add average line
            #y_mean = [np.mean(y)]*len(x)
            #plt.plot(x,y_mean, label='Mean', color='red')
            
            #plt.xscale('log')
            #plt.yscale('log')  
            #plt.xlim(xmin= 0.0001, xmax=self.X_max)
            plt.ylim(ymin=self.Y_min, ymax=self.Y_max)                    
            
            
            plt.show()
         

                
        return


    def plotAverages(self):
        #plot averaged chunks    
        self.fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharex = True, sharey = True)
        self.fig.suptitle('Averaged chunks')
        ax1.plot(self.baseline_FreqAverage,self.baseline_PowerAverage )
        ax2.plot(self.puff1_FreqAverage, self.puff1_PowerAverage)
        ax3.plot(self.puff2_FreqAverage, self.puff2_PowerAverage)    
        ax4.plot(self.puff3_FreqAverage, self.puff3_PowerAverage)
        ax5.plot(self.puff4_FreqAverage, self.puff4_PowerAverage)
        
        
        ax1.set_title('Baseline')
        ax2.set_title('Puff group 1')        
        ax3.set_title('Puff group 2') 
        ax4.set_title('Puff group 3')
        ax5.set_title('Puff group 4')        
        
        plt.show()               
        return


    def clearPlots(self):
        try:
            plt.close('all')  
        except:
            pass
        return

    def makeAverage(self):
        self.averageResults_dict = {}
        
        for key in self.result_dict:
            result_data = self.result_dict[key] 
            
            self.baseline_PowerAverage = []
            self.puff1_PowerAverage = []
            self.puff2_PowerAverage = []
            self.puff3_PowerAverage = []
            self.puff4_PowerAverage = []  
            self.baseline_FreqAverage = []
            self.puff1_FreqAverage = []
            self.puff2_FreqAverage = []
            self.puff3_FreqAverage = []
            self.puff4_FreqAverage = []        
    
            self.average_dict = {}
    
    
            def getAverage(start,end):
                powerList = []
                frequencyList = []
                
                print('Averaging Group {} - {}'.format(start,end))
                
                for i in range(start,end):
                    print('power_{}'.format(str(i)),)
                    #print(result_data['power_{}'.format(str(i))])
                    print('frequency_{}'.format(str(i)),)
                    #print(result_data['frequency_{}'.format(str(i))])
                    powerList.append(result_data['power_{}'.format(str(i))])
                    frequencyList.append(result_data['frequency_{}'.format(str(i))])                
                power = np.mean(powerList, axis=0)
                frequency = np.mean(frequencyList, axis=0)
                print('power mean'.format(str(i)),)
                print(power)
                print('frequency mean'.format(str(i)),)
                print(frequency)                
                print('------------')
                return power, frequency
                
            base_start = self.baseline_start.value()  
            base_stop = self.baseline_stop.value() 
            puff1_start = self.puff1_start.value()  
            puff1_stop = self.puff1_stop.value()         
            puff2_start = self.puff2_start.value()  
            puff2_stop = self.puff2_stop.value()         
            puff3_start = self.puff3_start.value()  
            puff3_stop = self.puff3_stop.value()         
            puff4_start = self.puff4_start.value()  
            puff4_stop = self.puff4_stop.value()         
            
    
            
            #get average group settings
            if base_stop > 0 and base_stop >= base_start:
                self.baseline_PowerAverage, self.baseline_FreqAverage = getAverage(base_start, base_stop) 
                self.average_dict.update( {'baseline_PowerAverage' : self.baseline_PowerAverage, 'baseline_FreqAverage':self.baseline_FreqAverage} )
    
            if puff1_stop> 0 and puff1_stop >= puff1_start:
                self.puff1_PowerAverage, self.puff1_FreqAverage = getAverage(puff1_start, puff1_stop) 
                self.average_dict.update( {'puff1_PowerAverage' : self.puff1_PowerAverage, 'puff1_FreqAverage' :self.puff1_FreqAverage} )
    
            if puff2_stop > 0 and puff2_stop >= puff2_start:
                self.puff2_PowerAverage, self.puff2_FreqAverage = getAverage(puff2_start, puff2_stop) 
                self.average_dict.update( {'puff2_PowerAverage' : self.puff2_PowerAverage, 'puff2_FreqAverage' :self.puff2_FreqAverage} )
                
    
            if puff3_stop > 0 and puff3_stop >= puff3_start:
                self.puff3_PowerAverage, self.puff3_FreqAverage = getAverage(puff3_start, puff3_stop) 
                self.average_dict.update( {'puff3_PowerAverage' : self.puff3_PowerAverage, 'puff3_FreqAverage' :self.puff3_FreqAverage} )
    
    
            if puff4_stop > 0 and puff4_stop >= puff4_start:
                self.puff4_PowerAverage, self.puff4_FreqAverage = getAverage(puff4_start, puff4_stop)
                self.average_dict.update( {'puff4_PowerAverage' : self.puff4_PowerAverage, 'puff4_FreqAverage' :self.puff4_FreqAverage} )
    
        self.averageResults_dict.update( {'AverageResult_{}'.format(str(key)) : self.average_dict} )  
        return

    def exportAverage(self):
        for key in self.averageResults_dict:
            self.average_DF = pd.DataFrame.from_dict(data=self.averageResults_dict[key])
            savePath = os.path.dirname(self.filename)
            if self.savename_label.text() == '':
                saveName = os.path.join(savePath,'FFT_Chunker_Batch_{}_{}.csv'.format(self.timeStr,str(key)))
            else:
                saveName = os.path.join(savePath,'{}_{}.csv'.format(self.savename_label.text(),str(key)))                
            self.average_DF.to_csv(saveName)
            print('average file saved')        
        return


    def exportResult(self):
        savePath = os.path.dirname(self.filename)
        toAverage = []
        
        for key in self.result_dict:
            if self.savename_label.text() == '':
                saveName = os.path.join(savePath,'FFT_Chunker_Batch_{}_Trace_{}.csv'.format(self.timeStr,str(key)))
            else:
                saveName = os.path.join(savePath,'{}_Trace_{}.csv'.format(self.savename_label.text(),str(key)))
            self.result_dict[key].to_csv(saveName)
            print('File {} saved'.format(saveName))
            g.m.statusBar().showMessage('File {} saved'.format(saveName)) 
            toAverage.append(self.result_dict[key])
            
        #if multiple ROI traces - take average of results and export as seperate file
        if len(self.result_dict) > 1:
            if self.savename_label.text() == '':
                saveName = os.path.join(savePath,'FFT_Chunker_Batch_{}_AveragedTraces.csv'.format(self.timeStr))
            else:
                saveName = os.path.join(savePath,'{}_AveragedTraces.csv'.format(self.savename_label.text()))
                                    
            averadedDF = pd.concat(toAverage).groupby(level=0).mean()
            averadedDF.to_csv(saveName)
            print('Averaged ROI File {} saved'.format(saveName))
            g.m.statusBar().showMessage('Averaged ROI File {} saved'.format(saveName))             
        return

        


    def runAnalysis(self):        
        chunk_size = self.chunkSize_Box.value()
        timestep = self.timestep_Box.value()
        self.timeStr = time.strftime("%Y%m%d-%H%M%S")
        
        if self.filename == '':
            print('File not loaded!')
            g.m.statusBar().showMessage('File not loaded!')
            return
        else:
            self.clearPlots()

            #import trace/traces                        
            columns = list(self.data) 
                        
            #initiate result dict (for multiple traces)
            self.result_dict = {}
            
            #fft analysis
            for i in columns:   
                # analyse each column
                result, self.X_min, self.X_max, self.Y_min, self.Y_max, self.minTime, self.maxTime = (fft_chunks(self.data[i].values,chunk_size, timestep)) 
 
                # add to self.result_dict
                self.result_dict.update( {i : result} )
                print('Trace {} analysed, {} chunks processed'.format(str(i), str((len(self.data[i].values )/chunk_size))))

            #export results
            self.exportResult()
            
            #generate averages for groups
            self.makeAverage()
            self.exportAverage()
            
            #plots
            if self.plot_checkbox.isChecked():
                self.plotData()
                self.plotAverages()
                
    
 
fft_Chunker = FFT_Chunker()
	

if __name__ == "__main__":
    pass


