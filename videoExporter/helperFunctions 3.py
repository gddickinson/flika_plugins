#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:48:13 2023

@author: george
"""

import numpy as np
import pandas as pd

# function to create a dictionary from a list
# def dictFromList(l):
#     # ensure strings
#     l = [str(x) for x in l]
#     # Create a zip object from two lists
#     zipbObj = zip(l, l)
#     return dict(zipbObj)

def dictFromList(lst):
    return {str(x): str(x) for x in lst}

# exponential decay functions for curve fitting
def exp_dec(x, A1, tau):
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x, A1, tau1, tau2):
    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x, A1, A2, tau1, tau2, tau3):
    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)

#rolling window function
# def rollingFunc(arr, window_size=6, func_type='mean'):
#     # Convert array of integers to pandas series
#     numbers_series = pd.Series(arr)

#     # Get the window of series
#     # of observations of specified window size
#     windows = numbers_series.rolling(window_size)

#     # Create a series of moving
#     # averages of each window
#     if func_type == 'mean':
#         moving_averages = windows.mean()
#     if func_type == 'std':
#         moving_averages = windows.std()
#     if func_type == 'variance':
#         moving_averages = np.square(windows.std())

#     # Convert pandas series back to list
#     moving_averages_list = moving_averages.tolist()

#     # Remove null entries from the list
#     final_list = moving_averages_list[window_size - 1:]

#     return final_list

def rollingFunc(arr, window_size=6, func_type='mean'):
    series = pd.Series(arr)
    windows = series.rolling(window_size)
    if func_type == 'mean':
        moving_averages = windows.mean()
    elif func_type == 'std':
        moving_averages = windows.std()
    elif func_type == 'variance':
        moving_averages = windows.var()
    else:
        raise ValueError("Invalid func_type. Must be 'mean', 'std', or 'variance'.")
    final_list = moving_averages[window_size - 1:].tolist()
    return final_list

def gammaCorrect(img, gamma):
    gammaCorrection = 1/gamma
    maxIntensity = np.max(img)
    return np.array(maxIntensity*(img / maxIntensity) ** gammaCorrection)

