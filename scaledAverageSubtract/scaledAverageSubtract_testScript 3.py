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
filename = r"C:\Users\g_dic\OneDrive\Desktop\testing\stackforGeorge_July3.tif"
windowSize = 50
averageSize = 100


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

#identify peak of trace
peakFrame = np.argmax(movingAverage)

#plot peak
plot.p1.plot(np.array([peakFrame]),np.array([movingAverage[peakFrame]]) , pen=None, symbol='o', symbolSize =20)

#average frames around peak
start = int(peakFrame - averageSize/2)
end = int(peakFrame + averageSize/2)

#prevent indexing from reaching past ends of trace
if start < 0:
    start = 0
    
if end > frames:
    end = frames

#get peak images
averageImages = A[start:end]

#display average stack
#averageImageStack_Win = Window(averageImages, 'Peak Images')

#average peak images
averageImage = np.mean(averageImages, axis=0)

#display average
#averageImage_Win = Window(averageImage,'Averaged Peak Images')

#make stack of averaged images
scaleImage_stack = np.ones_like(A)
scaleImage_stack = scaleImage_stack * averageImage
scaleImage_stack = scaleImage_stack[int(windowSize/2):-(int(windowSize/2)-1)]

#scale moving average 0 -> 1
scale = movingAverage/max(movingAverage)

#scale averaged image stack
scaleImage_stack = np.multiply(scaleImage_stack,scale[:,None,None])

#show scale images
#scaleWindow = Window(scaleImage_stack,'Scaled Average Images')

#subtract scaled average from original stack
A_subtract = np.zeros_like(A)
A_subtract[int(windowSize/2):-(int(windowSize/2)-1)]  = (A[int(windowSize/2):-(int(windowSize/2)-1)] - scaleImage_stack)

#show subtracted image stack
subtractWindow = Window(A_subtract,'Subtracted Images')



