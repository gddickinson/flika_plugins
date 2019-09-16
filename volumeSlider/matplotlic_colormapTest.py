# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 16:04:53 2019

@author: George
"""

from matplotlib import cm
import numpy as np


def getLUT(lutNAME = "nipy_spectral"):
    colormap = cm.get_cmap(lutNAME)  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
    return lut

test = getLUT()