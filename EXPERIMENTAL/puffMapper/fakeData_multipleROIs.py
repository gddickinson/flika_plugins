# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:13:03 2020

@author: g_dic
"""


import numpy as np
import pandas as pd
import random

filename = r"C:\Users\g_dic\OneDrive\Desktop\testing\D1T14_fft_test.csv"

data = pd.read_csv(filename, header=None, skiprows=1, index_col = 0)
data = data.dropna(axis=1, how='all')
nCols = len(data.columns)
colNames = list(range(0,nCols))
data.columns = colNames

data.insert(1,1,0.4)
data.insert(2,2,0)
data[2] = np.random.randint(1, 6, data.shape[0])/10

savename = r"C:\Users\g_dic\OneDrive\Desktop\testing\D1T14_fft_test_multipleROI.csv"

data.to_csv(savename)