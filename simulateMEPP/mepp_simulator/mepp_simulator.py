# -*- coding: utf-8 -*-
"""
Created on Fri Oct 02 12:23:06 2015

@author: George Dickinson
"""
import numpy as np
import os
from qtpy.QtCore import Signal
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
import inspect
import numpy.random as random
import shutil
from distutils.version import StrictVersion
from copy import deepcopy
import pyqtgraph as pg
from matplotlib import pyplot as plt 
import csv 

import flika
try:
    flika_version = flika.__version__
except AttributeError:
    flika_version = '0.0.0'
if StrictVersion(flika_version) < StrictVersion('0.1.0'):
    import global_vars as g
    from window import Window
    from process.file_ import close
    import tifffile
    from process.filters import gaussian_blur
    from process.binary import threshold
    from process.roi import set_value
    from roi import makeROI
    from process.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector


else:
    from flika import global_vars as g
    from flika.window import Window
    from flika.process.file_ import close
    from flika.process.filters import gaussian_blur
    from flika.process.binary import threshold
    from flika.process.roi import set_value
    from flika.roi import makeROI
    from flika.utils.io import tifffile

    if StrictVersion(flika_version) < StrictVersion('0.2.23'):
        from flika.process.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector, CheckBox
    else:
        from flika.utils.BaseProcess import SliderLabel, BaseProcess_noPriorWindow, FileSelector, CheckBox


cwd = os.path.dirname(os.path.abspath(__file__)) # cwd=r'C:\Users\Kyle Ellefsen\Documents\GitHub\Flika\plugins\mepp_simulator'

       
class Simulate_mepp(BaseProcess_noPriorWindow):
    """
    Simulate MEPP events in a noisy time trace
    
    """
    def __init__(self):
        super().__init__()
        self.data = np.array([])

    def get_init_settings_dict(self):
        s = dict()
        s['traceLength'] = 1000
        s['meppAmplitude'] = 5
        s['meppDuration'] = 5        
        s['startTime'] = 0
        s['meanExp'] = 5.0
        s['nmepps'] = 0 
        s['baseline'] = 0.0
        s['noiseSigma'] = 0.1
        
        return s

    def gui(self):
        self.gui_reset()
        self.traceLength = SliderLabel(0)
        self.traceLength.setRange(0,10000)
        self.traceLength.setValue(1000)        
        
        self.startTime = SliderLabel(0)
        self.startTime.setRange(0,10000)        

        self.meppDuration_slider = pg.SpinBox(int=True, step=1)
        self.meppDuration_slider.setValue(10)

        self.baseline_slider = pg.SpinBox(int=False, step=.01)
        self.baseline_slider.setValue(0.0)
        
        self.noiseSigma_slider = pg.SpinBox(int=False, step=.01)
        self.noiseSigma_slider.setValue(0.1)                
        
        self.meppAmplitude_slider = pg.SpinBox(int=False, step=.01)
        self.meppAmplitude_slider.setValue(1.0)         
        
        self.randommeppsAdded = False     
        
        self.meanExp_slider = pg.SpinBox(int=False, step=.01)
        self.meanExp_slider.setValue(100) 
        
        self.plotHistoTimes = CheckBox()
        self.plotHistoTimes.setValue(False)
        
        self.exportTimes_button = QPushButton('Export times')
        self.exportTimes_button.pressed.connect(self.exportTimes)  
        
        self.randommeppButton = QPushButton('Add MEPPs')
        self.randommeppButton.pressed.connect(self.addRandommepps)          

                    
        self.items.append({'name': 'traceLength','string':'Recording length','object':self.traceLength})
        self.items.append({'name': 'startTime','string':'Start time','object':self.startTime})  
        self.items.append({'name': 'meppDuration','string':'MEPP duration','object':self.meppDuration_slider})          
        self.items.append({'name': 'meppAmplitude','string':'Amplitude','object':self.meppAmplitude_slider})                     
        self.items.append({'name': 'meanExp', 'string': 'Mean of exponential distibution', 'object': self.meanExp_slider})  

        self.items.append({'name': 'baseline','string':'Baseline','object':self.baseline_slider})                     
        self.items.append({'name': 'noiseSigma', 'string': 'Noise Sigma', 'object': self.noiseSigma_slider})         
        
        self.items.append({'name': 'histoTimes', 'string': 'Plot histogram of mepp start times:', 'object': self.plotHistoTimes})
        self.items.append({'name': 'random_mepp_Button', 'string': 'Click to add randomly distibuted mepps', 'object': self.randommeppButton}) 
        self.items.append({'name': 'listTimes', 'string': 'Export list of mepp start times', 'object': self.exportTimes_button})          

        super().gui()

    def __call__(self):
        pass
        return 


    def addMEPP(self, time, amplitude, duration):
        self.data[time:time+duration] = self.data[time:time+duration] + amplitude

    def addRandommepps(self):
        self.data = np.array([])
        
        #generate noisy time trace      
        n = int(self.getValue('traceLength')) 

        self.data = np.random.normal(self.getValue('baseline'), self.getValue('noiseSigma'), n)     
        
        #add MEPPS to trace
        mean = self.getValue('meanExp')       
        meppsAdded = 0
        meppsOutsideOfRange = 0
        self.timesAdded = []
        
        amp = self.getValue('meppAmplitude')
        duration = self.getValue('meppDuration')
        
        # add first mepp
        try:
            time = int((np.random.exponential(scale=mean, size=1) + self.getValue('startTime')) )            
            self.addMEPP(time, amp, duration)
        except:
            meppsOutsideOfRange += 1 
            print('{} MEPPs added, {} MEPPs out of time range'.format(meppsAdded,meppsOutsideOfRange))
            print('1st MEPP outside of time range, aborting')
            return
       
        # add mepp after each time selection untill end of stack
        # casting the exponential continous value as an int to select frame        
        while time < self.getValue('traceLength')-self.getValue('meppDuration'):
            try:
                time = int((time + np.random.exponential(scale=mean, size=1)))
                self.addMEPP(time, amp, duration)
                self.timesAdded.append(time)            
                meppsAdded +=1 
            except:
                meppsOutsideOfRange += 1    
        
        print('{} MEPPs added, {} MEPPs out of time range'.format(meppsAdded,meppsOutsideOfRange))
        
        if self.plotHistoTimes.isChecked():
            plt.hist(self.timesAdded)
            plt.xlabel('Time MEPP added')
            plt.ylabel('Number of MEPPs added')
            plt.show()

        self.randommeppsAdded = True
        
        print(self.data)
        plt.plot(self.data)
        plt.show()

        return


    def exportTimes(self):        
        if self.randommeppsAdded == False:
            g.alert('Add MEPPs first')
            return
        
        #set export path
        savePath, _ = QFileDialog.getSaveFileName(None, "Save file","","Text Files (*.csv)")        

        #write file
        try:
            # opening the csv file in 'w+' mode 
            file = open(savePath, 'w+', newline ='') 
              
            # writing the data into the file 
            with file:     
                write = csv.writer(file) 
                write.writerows(map(lambda x: [x], self.timesAdded)) 
            
            print('List of times saved to: {}'.format(savePath))
        except BaseException as e:
            print(e)
            print('Export of times failed, printing times to console')
            print(self.timesAdded)

simulate_mepp = Simulate_mepp()
