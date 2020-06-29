# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:28:17 2020

@author: g_dic
"""


#import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft

####fake signal to test
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

np.savetxt(r'C:\Users\g_dic\OneDrive\Desktop\testing\fakeData.csv', y, delimiter=',')


plt.scatter(x,y)
