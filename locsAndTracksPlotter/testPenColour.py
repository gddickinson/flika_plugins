#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 16:49:26 2023

@author: george
"""

import pyqtgraph as pg
import pandas as pd


# pg.colormap.listMaps()

# cm = pg.colormap.get('CET-L9')

# pen = cm.getPen(span=(0.01,1.0), width=5)

# lut = cm.getLookupTable(start=0.0, stop=1.0, nPts=512, mode='QCOLOR')


# test = [0.1, 20.4, 100]

# mapping = cm.mapToQColor(test)

path = '/Users/george/Data/htag_cutout/GB_199_2022_09_01_tracks/gof/GB_199_2022_09_01_HTEndothelial_locsID_tracksRG_SVMPredicted_NN.csv'
data = pd.read_csv(path)

df = pd.DataFrame()
df['frame'] = data['frame'].astype(int)-1
df['x'] = data['x']
df['y'] = data['y']
df['track_number'] = data['track_number']


cm = pg.colormap.get('plasma')
df['colour'] = cm.mapToQColor(data['track_number'].to_numpy())

# grouped = df.groupby(['track_number'])

# tracks = grouped.get_group(1000)
# colour = tracks['colour'].to_list()[0]
# print(tracks['track_number'])
# print(colour.rgb())

