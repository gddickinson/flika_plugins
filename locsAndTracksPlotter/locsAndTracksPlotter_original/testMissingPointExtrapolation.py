#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 12:18:27 2023

@author: george
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
import os, glob

from matplotlib import pyplot as plt

file = '/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount.csv'

#load point data
df = pd.read_csv(file)

frame = max(df['frame'])

#filter for 1 track
df = df[df['track_number']==1500]

#get points and frames as list
points = np.column_stack((df['frame'].to_list(), df['x'].to_list(), df['y'].to_list()))    

#interpolate points in missing frames
interpFrames = range(int(min(points[:,0])), int(max(points[:,0]))+1)
xinterp = np.interp(interpFrames, points[:,0], points[:,1])
yinterp = np.interp(interpFrames, points[:,0], points[:,2])

newPoints = np.column_stack((interpFrames, xinterp, yinterp))  

plt.scatter(newPoints[:,0], newPoints[:,2])

#pad ends
xinterp = np.pad(xinterp, (int(min(points[:,0])), frame - int(max(points[:,0]))), mode='edge')
yinterp = np.pad(yinterp, (int(min(points[:,0])), frame - int(max(points[:,0]))), mode='edge')

allFrames = range(0, frame+1)

newPoints = np.column_stack((allFrames, xinterp, yinterp))   

plt.scatter(newPoints[:,0], newPoints[:,2])


