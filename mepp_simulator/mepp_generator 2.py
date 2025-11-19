# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:37:12 2021

@author: g_dic
"""


import numpy as np
from matplotlib import pyplot as plt
import math


#using Segal et al. Biophys J, 1985 MINIATURE ENDPLATE POTENTIAL FREQUENCY AND AMPLITUDE DETERMINED BY AN EXTENSION OF CAMPBELL'S THEOREM
duration = range(0,100)

#decayTimeConstant
dT = 10

#riseTimeConstant
rT = 1

amp = []

for t in duration:
    amp.append( math.exp(-t/dT) - math.exp(-t/rT) )


plt.plot(duration,amp)
    