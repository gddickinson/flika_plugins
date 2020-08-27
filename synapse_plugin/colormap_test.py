# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 15:30:02 2020

@author: g_dic
"""


import pyqtgraph as pg
import numpy as np

pos = np.array([0.0, 0.5, 1.0])
color = np.array([[0,0,0,255], [255,128,0,255], [255,255,0,255]], dtype=np.ubyte)
map = pg.ColorMap(pos, color)
lut = map.getLookupTable(0.0, 1, 1000)