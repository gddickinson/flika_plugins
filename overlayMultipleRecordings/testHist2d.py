#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:20:20 2024

@author: george
"""

from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

file = r'/Users/george/Data/2024-01-16_micropatterns_for_George/AL_67_2020-07-31-TIRFM_Diff_HT-NSC-CYTOO_H9_1_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity_AllLocs_NNcount_transform.csv'
data= pd.read_csv(file)

#make sure track number and frame are int
data['frame'] = data['frame'].astype(int)

if 'track_number' in data.columns:
    # filter any points that dont have track_numbers to seperate df
    data_unlinked = data[data['track_number'].isna()]
    data  = data[~data['track_number'].isna()]

    data['track_number'] = data['track_number'].astype(int)

else:
    data['track_number'] = None
    data_unlinked = data


#H, xedges, yedges = np.histogram2d(data['x_transformed'], data['y_transformed'], bins=10, density = True, weights=data['radius_gyration'])
H, xedges, yedges = np.histogram2d(data['x_transformed'], data['y_transformed'], bins=99)

H = H.T

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='counts')
plt.imshow(H, interpolation='nearest', origin='lower')



data['bin_x'] = np.digitize(data['x_transformed'],xedges, right=False)
data['bin_y'] = np.digitize(data['y_transformed'],yedges, right=False)

data['bin_x'] -= 1
data['bin_y'] -= 1

groupedDF = data.groupby(['bin_x','bin_y'])

mean_Rg = groupedDF['radius_gyration'].mean().to_frame(name = 'radius_gyration').reset_index()
count = groupedDF['bin_x'].count().to_frame(name = 'count').reset_index()

#convert df columns to image
X = mean_Rg['bin_x'].to_numpy()
Y = mean_Rg['bin_y'].to_numpy()
Z = mean_Rg['radius_gyration'].to_numpy()
Z2 = count['count'].to_numpy()

Xu = np.unique(X)
Yu = np.unique(Y)


img = np.zeros((Xu.size, Yu.size))
img2 = np.zeros((Xu.size, Yu.size))

for i in range(X.size):
    img[np.where(Xu==X[i]), np.where(Yu==Y[i])] = Z[i]
    img2[np.where(Xu==X[i]), np.where(Yu==Y[i])] = Z2[i]

img = img.T
img2 = img2.T

ax = fig.add_subplot(132, title='Rg', aspect='equal')
plt.imshow(img, interpolation='nearest', origin='lower')

ax = fig.add_subplot(133, title='Count', aspect='equal')
plt.imshow(img2, interpolation='nearest', origin='lower')

