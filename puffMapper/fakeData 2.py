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

plt.scatter(x,y)

#y = range(0,N)

fakeData = np.column_stack([x,y])
fakeData = fakeData.reshape((N,2))
np.savetxt(r'C:\Users\g_dic\OneDrive\Desktop\testing\fakeData.csv', fakeData, delimiter=',', header='Index,Data')

#test
import pandas as pd

filename = r"C:\Users\g_dic\OneDrive\Desktop\testing\fakeData.csv"

data = pd.read_csv(filename, header=None, skiprows=1, index_col = 0)
data = data.dropna(axis=1, how='all')
nCols = len(data.columns)
colNames = list(range(0,nCols))
data.columns = colNames

columns = list(data) 

print(data.head())


plt.scatter(list(data.index.values),data[0].values)

