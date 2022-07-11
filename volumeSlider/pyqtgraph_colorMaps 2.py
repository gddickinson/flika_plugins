# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:51:00 2019

@author: George
"""

import pyqtgraph as pg
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

#list the available colormaps
print (Gradients.keys())

#pick one to turn into an actual colormap
spectrumColormap =  pg.ColorMap(*zip(*Gradients["spectrum"]["ticks"]))

#create colormaps from the builtins
for k,v in Gradients.items():
    pos, color = zip(*v["ticks"])
    cmap =  pg.ColorMap(pos, color)