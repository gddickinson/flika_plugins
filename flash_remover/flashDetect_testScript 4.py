# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:22:02 2020

@author: george.dickinson@gmail.com
"""

import os, sys, glob

import flika
from flika import *
from flika import global_vars as g
from flika.window import Window
from distutils.version import StrictVersion
import numpy as np

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from flika.process.file_ import open_file
from flika.roi import makeROI


#start flika
# try:
#     %gui qt
# except:
#     pass

start_flika()

#set filename
filename = r"C:\Users\g_dic\OneDrive\Desktop\trial_2_Cal520_cip3.tif"
#filename = r"C:\Users\g_dic\OneDrive\Desktop\2021_02_05_KRAP_800ms001.tif"
windowSize = 100
#averageSize = 100


#open file in window
dataWindow = open_file(filename) 

#get image array
A = dataWindow.imageArray()

#get array shape
frames, height ,width = A.shape 

#plot roi in center
centerROI = makeROI('rectangle',[[10, 10], [height-20, width-20]], window=dataWindow)

#plot roi average trace
plot = centerROI.plot()

#get trace data
trace = centerROI.getTrace()

#define function to get moving average from trace
def moving_average(x, n=windowSize) :
    return np.convolve(x, np.ones((n,))/n, mode='valid')
    
#get trace moving average
movingAverage = moving_average(trace, n=windowSize)   

#set moving average values <= 0 to 0.0000001
movingAverage[movingAverage <= 0] = 0.0000001

#add moving average trace to plot
plot.p1.plot(movingAverage , pen=(1,3), symbol=None)

#identify peak of trace - first frame of flash
flashStart = np.argmax(movingAverage)

#identify 1st drop after peak - end of flash
flashEnd = flashStart + np.argmin(movingAverage[flashStart:-1])

#plot flash start and end
plot.p1.plot(np.array([flashStart]),np.array([movingAverage[flashStart]]) , pen=None, symbol='o', symbolSize =20)
plot.p1.plot(np.array([flashEnd]),np.array([movingAverage[flashEnd]]) , pen=None, symbol='o', symbolSize =20)





