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

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


import pandas as pd
from matplotlib import pyplot as plt

#MEPPS
def mepp_Amplitudes(mu, sigma, n=500):
    '''draw n mepp amplitude values from a guassian ditribution
    mean = mu and std = sigma'''    
    return np.random.normal(mu, sigma, n)

def mepp_nPerInterval(mu,n=500):
    '''draw n mepps/interval values from a poisson ditribution
    mean = mu '''  
    return np.random.poisson(mu, n)

def mepp_Intervals(t, start=0, end=5, n=500):
    '''draw n intervals between mepps values from a exponentially decaying function
    time constant = t '''
    return np.random.exponential(t,size=n)

#EPPS
def epp_Quanta(mu,n=500):
    '''draw n # of quanta values from a poisson ditribution
    mean = mu'''     
    return np.random.poisson(mu, n)

def epp_Amplitudes(mu, sigma, n=500):
    '''draw n epp amplitude values from a guassian ditribution
    mean = mu and std = sigma'''    
    return np.random.normal(mu, sigma, n)    

def epp_Amplitudes_by_quanta(mu, quanta, sigma, n=500):
    '''draw n epp amplitude values from a guassian ditribution
    mean = mu, quanta = quanta, std = sigma'''

    quantaDist = epp_Quanta(quanta,n=n)
    
    unique_quanta, counts_quanta = np.unique(quantaDist, return_counts=True)

    dist = []

    for i in range(len(unique_quanta)):
        dist.append(np.random.normal(unique_quanta[i]*mu,
                                     sigma*np.sqrt(unique_quanta[i]),
                                     counts_quanta[i]))           
 
    

    #flatten list
    flat_dist = [item for sublist in dist for item in sublist]
    #remove zeros
    dist_noNeg = [0 if i <0 else i for i in flat_dist]
    
    return dist_noNeg 



    
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
        neuroLab.exportPath = self.folder
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    
    
        

class NeuroLab(BaseProcess_noPriorWindow):
    """
    neuroLab
    
    *** Generation of MEPP data ***
    MEPP Amplitudes : draw n mepp amplitude values from a gaussian distribution
    Function: np.random.normal(mu, sigma, n)
    Parameters: mean (mu), StD (sigma) and sample size (n)
    
    Number of MEPPs Per Interval: draw n mepps/interval values from a poisson distribution
    Function: np.random.poisson(mu, n)
    Parameters: mean (mu) and sample size (n) 
    
    Number of Intervals between MEPPS: draw n intervals between mepps values from an exponentially decaying function
    Function: np.random.exponential(t, n)
    Parameters: time constant (t) and sample size (n)
    
    *** Generation of EPP data ***
    Number of EPP Quanta values: draw n # of quanta values from a poisson distribution 
    Function: np.random.poisson(mu, n)
    Parameters: mean(mu) and sample size (n)
    
    EPP Amplitudes: draw n epp amplitude values from a gaussian distribution
    Function: np.random.normal(mu, sigma, n)
    Parameters: mean (mu), StD (sigma) and sample size (n) 
    
    EPP Amplitudes by Quanta: draw n epp amplitude values from a gaussian distribution based on poisson distribution of quanta
    for each quantum combine: np.random.normal(quanta*mu, sigma*np.sqrt(quanta), n=number of quanta) 
    Parameters: mean (mu), number of quanta (quanta), StD (sigma) and sample size (n)     
      
    ---------------------------------------------------------------------------------------------------------
    Click 'Select Folder' button to set path for exporting results
    Click 'Generate MEPP Data' or 'Generate EPP Data' to draw random values. Summary histograms should appear
    Click 'Save MEPP Data' or 'Save EPP Data' buttons to export data as csv files to export path
    
    *** If exporting the results doesn't work try clicking the 'Save' again (occasionally the button is unresponsive) ***
 
    """
    def __init__(self):
        if g.settings['neuroLab'] is None or 'eppAmpByQuanta_N' not in g.settings['neuroLab']:
            s = dict()            
            s['meppAmp_mean'] = 10
            s['meppAmp_sigma'] = 0.1
            s['meppAmp_N'] = 500
            s['meppsPerInterval'] = 10
            s['meppsPerInterval_N'] = 500
            s['meppIntervals_time'] = 1
            s['meppIntervals_N'] = 500        
            s['eppQuanta_mean'] = 10
            s['eppQuanta_N'] = 500  
            s['eppAmp_mean'] = 10
            s['eppAmp_sigma'] = 0.1
            s['eppAmp_N'] = 500            
            s['eppAmpByQuanta_mean'] = 10
            s['eppAmpByQuanta_quanta'] = 1           
            s['eppAmpByQuanta_sigma'] = 0.1
            s['eppAmpByQuanta_N'] = 500            
           
            g.settings['neuroLab'] = s
                   
        BaseProcess_noPriorWindow.__init__(self)
        
        self.sampleMax = 1000000
        
        self.exportPath = ''

    def __call__(self):
        '''
        reset saved parameters
        '''
        g.settings['neuroLab']['meppAmp_mean'] = 10
        g.settings['neuroLab']['meppAmp_sigma'] = 0.1
        g.settings['neuroLab']['meppAmp_N'] = 500
        g.settings['neuroLab']['meppsPerInterval'] = 10
        g.settings['neuroLab']['meppsPerInterval_N'] = 500
        g.settings['neuroLab']['meppIntervals_time'] = 1
        g.settings['neuroLab']['meppIntervals_N'] = 500        
        g.settings['neuroLab']['eppQuanta_mean'] = 10
        g.settings['neuroLab']['eppQuanta_N'] = 500  
        g.settings['neuroLab']['eppAmp_mean'] = 10
        g.settings['neuroLab']['eppAmp_sigma'] = 0.1
        g.settings['neuroLab']['eppAmp_N'] = 500 
        g.settings['eppAmpByQuanta_mean'] = 10
        g.settings['eppAmpByQuanta_quanta'] = 1           
        g.settings['eppAmpByQuanta_sigma'] = 0.1
        g.settings['eppAmpByQuanta_N'] = 500         
        
        #currently not saving parameter changes
        
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        return

    def gui(self):
        self.gui_reset()
        s=g.settings['neuroLab']

        #buttons      
        self.generateMeppData_button = QPushButton('Generate MEPP Data')
        self.generateMeppData_button.pressed.connect(self.makeMeppData)  
        self.generateEppData_button = QPushButton('Generate EPP Data')
        self.generateEppData_button.pressed.connect(self.makeEppData)   
        
        self.exportMeppData_button = QPushButton('Save MEPP Data')
        self.exportMeppData_button.pressed.connect(self.exportMeppData)  
        self.exportEppData_button = QPushButton('Save EPP Data')
        self.exportEppData_button.pressed.connect(self.exportEppData)    
               
        
        self.setExportFolder_button = FolderSelector('*.csv')
        
        #spinboxes
        #MEPPS
        self.meppAmp_mean_Box = pg.SpinBox(int=True, step=1)
        self.meppAmp_mean_Box.setMinimum(1)
        self.meppAmp_mean_Box.setMaximum(1000)        
        self.meppAmp_mean_Box.setValue(s['meppAmp_mean'])
        
        self.meppAmp_sigma_Box = pg.SpinBox(int=False, step=0.1)
        self.meppAmp_sigma_Box.setMinimum(0)
        self.meppAmp_sigma_Box.setMaximum(100)        
        self.meppAmp_sigma_Box.setValue(s['meppAmp_sigma'])        
        
        self.meppAmp_N_Box = pg.SpinBox(int=True, step=1)
        self.meppAmp_N_Box.setMinimum(1)
        self.meppAmp_N_Box.setMaximum(self.sampleMax)        
        self.meppAmp_N_Box.setValue(s['meppAmp_N'])         
       
        self.meppsPerInterval_Box = pg.SpinBox(int=False, step=0.1)
        self.meppsPerInterval_Box.setMinimum(1)
        self.meppsPerInterval_Box.setMaximum(1000)        
        self.meppsPerInterval_Box.setValue(s['meppsPerInterval'])          

        self.meppsPerInterval_N_Box = pg.SpinBox(int=True, step=1)
        self.meppsPerInterval_N_Box.setMinimum(1)
        self.meppsPerInterval_N_Box.setMaximum(self.sampleMax)        
        self.meppsPerInterval_N_Box.setValue(s['meppsPerInterval_N'])  

        self.meppIntervals_time_Box = pg.SpinBox(int=True, step=1)
        self.meppIntervals_time_Box.setMinimum(1)
        self.meppIntervals_time_Box.setMaximum(1000)        
        self.meppIntervals_time_Box.setValue(s['meppIntervals_time'])         
        
        self.meppIntervals_N_Box = pg.SpinBox(int=True, step=1)
        self.meppIntervals_N_Box.setMinimum(1)
        self.meppIntervals_N_Box.setMaximum(self.sampleMax)        
        self.meppIntervals_N_Box.setValue(s['meppIntervals_N'])   
        
        #EPPS
        self.eppQuanta_mean_Box = pg.SpinBox(int=False, step=0.1)
        self.eppQuanta_mean_Box.setMinimum(1)
        self.eppQuanta_mean_Box.setMaximum(1000)        
        self.eppQuanta_mean_Box.setValue(s['eppQuanta_mean']) 
        
        self.eppQuanta_N_Box = pg.SpinBox(int=True, step=1)
        self.eppQuanta_N_Box.setMinimum(1)
        self.eppQuanta_N_Box.setMaximum(self.sampleMax)        
        self.eppQuanta_N_Box.setValue(s['eppQuanta_N'])         
        
        self.eppAmp_mean_Box = pg.SpinBox(int=True, step=1)
        self.eppAmp_mean_Box.setMinimum(1)
        self.eppAmp_mean_Box.setMaximum(1000)        
        self.eppAmp_mean_Box.setValue(s['eppAmp_mean'])          
        
        self.eppAmp_sigma_Box = pg.SpinBox(int=False, step=0.1)
        self.eppAmp_sigma_Box.setMinimum(0)
        self.eppAmp_sigma_Box.setMaximum(100)        
        self.eppAmp_sigma_Box.setValue(s['eppAmp_sigma'])          

        self.eppAmp_N_Box = pg.SpinBox(int=True, step=1)
        self.eppAmp_N_Box.setMinimum(1)
        self.eppAmp_N_Box.setMaximum(self.sampleMax)        
        self.eppAmp_N_Box.setValue(s['eppAmp_N'])  

        self.eppAmpByQuanta_mean_Box = pg.SpinBox(int=False, step=0.1)
        self.eppAmpByQuanta_mean_Box.setMinimum(1)
        self.eppAmpByQuanta_mean_Box.setMaximum(1000)        
        self.eppAmpByQuanta_mean_Box.setValue(s['eppAmpByQuanta_mean'])          
 
        self.eppAmpByQuanta_quanta_Box = pg.SpinBox(int=True, step=1)
        self.eppAmpByQuanta_quanta_Box.setMinimum(1)
        self.eppAmpByQuanta_quanta_Box.setMaximum(1000)        
        self.eppAmpByQuanta_quanta_Box.setValue(s['eppAmpByQuanta_quanta'])         
     
        self.eppAmpByQuanta_sigma_Box = pg.SpinBox(int=False, step=0.01)
        self.eppAmpByQuanta_sigma_Box.setMinimum(0.0)
        self.eppAmpByQuanta_sigma_Box.setMaximum(100)        
        self.eppAmpByQuanta_sigma_Box.setValue(s['eppAmpByQuanta_sigma']) 
       
        self.eppAmpByQuanta_N_Box = pg.SpinBox(int=True, step=1)
        self.eppAmpByQuanta_N_Box.setMinimum(1)
        self.eppAmpByQuanta_N_Box.setMaximum(self.sampleMax)        
        self.eppAmpByQuanta_N_Box.setValue(s['eppAmpByQuanta_N'])        
   
           
        #################################################################
        #self.exportFolder = FolderSelector('*.txt')
        #MEPPS
        self.items.append({'name': 'blank ', 'string': '-------------    MEPP Parameters    ---------------', 'object': None}) 
        self.items.append({'name': 'meppAmp_mean ', 'string': 'MEPP Amplitude (mean): ', 'object': self.meppAmp_mean_Box})  
        self.items.append({'name': 'meppAmp_sigma ', 'string': 'MEPP Amplitude (standard deviation): ', 'object': self.meppAmp_sigma_Box}) 
        self.items.append({'name': 'meppAmp_N ', 'string': 'MEPP Amplitude (# of samples): ', 'object': self.meppAmp_N_Box})       
        self.items.append({'name': 'meppsPerInterval ', 'string': '# MEPPs per interval: ', 'object': self.meppsPerInterval_Box})        
        self.items.append({'name': 'meppsPerInterval_N ', 'string': '# MEPPs per interval (# of samples): ', 'object': self.meppsPerInterval_N_Box})         
        self.items.append({'name': 'meppIntervals_time ', 'string': 'MEPP intervals (time constant): ', 'object': self.meppIntervals_time_Box })         
        self.items.append({'name': 'meppIntervals_N ', 'string': 'MEPP intervals (# of samples): ', 'object': self.meppIntervals_N_Box})                 
        self.items.append({'name': 'generateMeppData ', 'string': '', 'object': self.generateMeppData_button})  
        self.items.append({'name': 'exportMeppData ', 'string': '', 'object': self.exportMeppData_button})          
        
        #EPPS        
        self.items.append({'name': 'blank ', 'string': '---------------   EPP Parameters    ---------------', 'object': None})                  
        self.items.append({'name': 'eppQuanta_mean ', 'string': 'EPP Quanta (mean): ', 'object': self.eppQuanta_mean_Box})  
        self.items.append({'name': 'eppQuanta_N ', 'string': 'EPP Quanta (# of samples): ', 'object': self.eppQuanta_N_Box})  
        self.items.append({'name': 'eppAmp_mean ', 'string': 'EPP Amplitude (mean): ', 'object': self.eppAmp_mean_Box})  
        self.items.append({'name': 'eppAmp_sigma ', 'string': 'EPP Amplitude (standard deviation): ', 'object': self.eppAmp_sigma_Box}) 
        self.items.append({'name': 'eppAmp_N ', 'string': 'EPP Amplitude (# of samples): ', 'object': self.eppAmp_N_Box})  
        
        self.items.append({'name': 'eppAmpByQuanta_mean ', 'string': 'EPP Amplitude By Quanta (mean): ', 'object': self.eppAmpByQuanta_mean_Box})  
        self.items.append({'name': 'eppAmpByQuanta_quanta ', 'string': 'EPP Amplitude By Quanta (# of quanta): ', 'object': self.eppAmpByQuanta_quanta_Box})          
        self.items.append({'name': 'eppAmpByQuanta_sigma ', 'string': 'EPP Amplitude By Quanta  (standard deviation): ', 'object': self.eppAmpByQuanta_sigma_Box}) 
        self.items.append({'name': 'eppAmpByQuanta_N ', 'string': 'EPP Amplitude By Quanta  (# of samples): ', 'object': self.eppAmpByQuanta_N_Box})          
        
               
        self.items.append({'name': 'generateEppData ', 'string': '', 'object': self.generateEppData_button}) 
        self.items.append({'name': 'exportEppData ', 'string': '', 'object': self.exportEppData_button}) 
              

        self.items.append({'name': 'blank ', 'string': '-----------------   Export Path    -----------------', 'object': None})    
        self.items.append({'name': 'setPath ', 'string': '', 'object': self.setExportFolder_button })            
        
        super().gui()
        ######################################################################
        return

    def plotMeppData(self):
        try:
            plt.close(self.mepp_fig)
        except:
            pass
        
        self.mepp_fig, (self.mepp_ax1, self.mepp_ax2, self.mepp_ax3) = plt.subplots(1, 3)
        self.mepp_fig.suptitle('Randomly Generated MEPP Data')
        self.mepp_ax1.hist(self.mepp_Amplitudes_dist)
        self.mepp_ax2.hist(self.mepp_nPerInterval_dist)
        self.mepp_ax3.hist(self.mepp_Intervals_dist)
        
        self.mepp_ax1.set_title('MEPP Amplitudes')
        self.mepp_ax2.set_title('# of MEPPs / Interval')
        self.mepp_ax3.set_title('MEPP Intervals')
        
        self.mepp_fig.show()
        return

    def plotEppData(self):
        try:
            plt.close(self.epp_fig)
        except:
            pass
            
        self.epp_fig, (self.epp_ax1, self.epp_ax2, self.epp_ax3) = plt.subplots(1, 3)
        self.epp_fig.suptitle('Randomly Generated EPP Data')
        self.epp_ax1.hist(self.epp_Quanta_dist)
        self.epp_ax2.hist(self.epp_Amplitudes_dist)
        self.epp_ax3.hist(self.epp_Amplitudes_by_quanta_dist,100)
        #noZeroDist = [i for i in self.epp_Amplitudes_by_quanta_dist if i != 0]
        #self.epp_ax4.hist(noZeroDist,100)        
        
        self.epp_ax1.set_title('EPP Quanta')
        self.epp_ax2.set_title('EPP Amplitudes')
        self.epp_ax3.set_title('EPP Amplitudes by Quanta')  
        #self.epp_ax4.set_title('EPP Amplitudes by Quanta (No zeros)')          
     
        self.epp_fig.show()
        return


    def makeMeppData(self):
        self.mepp_Amplitudes_dist = mepp_Amplitudes(self.meppAmp_mean_Box.value(), self.meppAmp_sigma_Box.value(), n=self.meppAmp_N_Box.value())    
        self.mepp_nPerInterval_dist = mepp_nPerInterval(self.meppsPerInterval_Box.value(),n=self.meppsPerInterval_N_Box.value())
        self.mepp_Intervals_dist = mepp_Intervals(self.meppIntervals_time_Box.value(), start=0, end=5, n=self.meppIntervals_N_Box.value())
        self.plotMeppData()
        return

    def makeEppData(self):
        self.epp_Quanta_dist = epp_Quanta(self.eppQuanta_mean_Box.value(),n=self.eppQuanta_N_Box.value()) 
        self.epp_Amplitudes_dist = epp_Amplitudes(self.eppAmp_mean_Box.value(), self.eppAmp_sigma_Box.value(), n=self.eppAmp_N_Box.value())
        self.epp_Amplitudes_by_quanta_dist = epp_Amplitudes_by_quanta(self.eppAmpByQuanta_mean_Box.value(), self.eppAmpByQuanta_quanta_Box.value(), self.eppAmpByQuanta_sigma_Box.value(), self.eppAmpByQuanta_N_Box.value())
        self.plotEppData()        
        return


    def exportMeppData(self):
        if self.exportPath == '':
            print('Set Export Path')
            return
        else:
            np.savetxt(os.path.join(self.exportPath,'mepp_Amplitudes.csv'), self.mepp_Amplitudes_dist, delimiter=',',fmt='%1.3f')
            np.savetxt(os.path.join(self.exportPath,'mepp_nPerInterval.csv'), self.mepp_nPerInterval_dist, delimiter=',',fmt='%1.3f')    
            np.savetxt(os.path.join(self.exportPath,'mepp_Interval.csv'), self.mepp_Intervals_dist, delimiter=',',fmt='%1.3f')
            print('MEPP Data saved')
        return

    def exportEppData(self):
        if self.exportPath == '':
            print('Set Export Path')
            return
        else:
            np.savetxt(os.path.join(self.exportPath,'epp_Quanta.csv'), self.epp_Quanta_dist, delimiter=',',fmt='%1.3f')
            np.savetxt(os.path.join(self.exportPath,'epp_Amplitudes.csv'), self.epp_Amplitudes_dist, delimiter=',',fmt='%1.3f')
            np.savetxt(os.path.join(self.exportPath,'epp_Amplitudes_by_Quanta.csv'), self.epp_Amplitudes_by_quanta_dist, delimiter=',',fmt='%1.3f')            
            print('EPP Data saved')
        return


 
neuroLab = NeuroLab()


if __name__ == "__main__":
    mu = 1.0
    sigma = 1
    quanta = 1.0
    n = 100000
    test = epp_Amplitudes_by_quanta(mu, sigma, quanta, n=n)
    test = [i for i in test  if i != 0]

    plt.hist(test, 100)


    saveTest = np.loadtxt(r"C:\Users\g_dic\OneDrive\Desktop\testing\epp_Amplitudes_by_Quanta.csv", delimiter=',')
    saveTest= [i for i in saveTest if i != 0]

    plt.hist(saveTest, 100)



