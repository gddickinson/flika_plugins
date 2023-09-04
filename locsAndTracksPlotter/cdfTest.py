#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:47:07 2023

@author: george
"""
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

# exponential decay functions for curve fitting
def exp_dec(x, A1, tau):
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x, A1, tau1, tau2):
    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x, A1, A2, tau1, tau2, tau3):
    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)

def tau_to_D(tau):
    """
    tau = 4Dt
    tau is decay constant of exponential fit
    D is diffusion coefficient
    t is duration of one lag (exposure time) in seconds
    """
    t = 100/1000
    D = tau / (4 * t)
    return D


def updateCDF(sld, n=500):
    # Calculate the squared lag distances in microns
    #self.squared_SLDs = plotDF['lag_squared'] * np.square(self.mainGUI.trackPlotOptions.pixelSize_selector.value()/1000)
    squared_SLDs = np.square(sld)

    # Set the start and end points of the histogram, and the number of bins
    start=0
    end=np.max(squared_SLDs)

    # Calculate the histogram using numpy
    count, bins_count = np.histogram(squared_SLDs, bins=np.linspace(start, end, n))

    # Calculate the probability density function and the cumulative distribution function
    pdf = count / sum(count)
    cdf_y = np.cumsum(pdf)
    cdf_x = bins_count[1:]

    # Get the maximum number of lags for normalization
    nlags = np.max(cdf_y)

    return cdf_x, cdf_y

def fit_exp_dec_1(cdf_x, cdf_y, left_bound, right_bound):
    # Get the x and y data for the CDF plot from the GUI
    xdata = cdf_x
    ydata = cdf_y
    # Select the data points within the fitting range
    x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
    xfit = xdata[x_fit_mask]

    # Fit an exponential decay function to the selected data
    popt, pcov = curve_fit(exp_dec, xfit, ydata[x_fit_mask], bounds=([-1.2, 0], [0, 30]))
    tau_fit = popt[1]
    D_fit = tau_to_D(tau_fit)

    # Print the fitted diffusion coefficient
    print('tau = {}'.format(tau_fit))
    print('D = {0:.4g} um^2 s^-1'.format(D_fit))

    # Generate the fitted curve and add it to the plot
    yfit = exp_dec(xfit, *popt)

    return xfit, yfit

def fit_exp_dec_2(cdf_x, cdf_y, left_bound, right_bound):
    # Get the x and y data for the CDF plot from the GUI
    xdata = cdf_x
    ydata = cdf_y
    # Select the data points within the fitting range
    x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
    xfit = xdata[x_fit_mask]

    # Perform the curve fitting using the double-exponential decay function (exp_dec_2)
    # and the masked data
    popt, pcov = curve_fit(exp_dec_2, xfit, ydata[x_fit_mask], bounds=([-1, 0, 0], [0, 30, 30]))

    # Extract the fitted parameters
    A1 = popt[0]
    A2 = -1 - A1
    tau1_fit = popt[1]
    D1_fit = tau_to_D(tau1_fit)
    tau2_fit = popt[2]
    D2_fit = tau_to_D(tau2_fit)

    # Print the fitted diffusion coefficients and amplitudes
    msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2. A1={2:.2g} A2={3:.2g}'.format(D1_fit, D2_fit, A1, A2)
    print('tau1 = {}, tau2 = {}'.format(tau1_fit, tau2_fit))
    print(msg)

    # Calculate the fit line and plot it on the CDF plot
    yfit = exp_dec_2(xfit, *popt)

    return xfit, yfit

def fit_exp_dec_3(cdf_x, cdf_y, left_bound, right_bound):
    # Get the x and y data for the CDF plot from the GUI
    xdata = cdf_x
    ydata = cdf_y
    # Select the data points within the fitting range
    x_fit_mask = (left_bound <= xdata) * (xdata <= right_bound)
    xfit = xdata[x_fit_mask]


    # Fit the data using the three-exponential decay function and bounds on the parameters
    popt, pcov = curve_fit(exp_dec_3, xfit, ydata[x_fit_mask], bounds=([-1, -1, 0, 0, 0], [0, 0, 30, 30, 30]))

    # Extract the fitted parameters and compute diffusion coefficients
    A1 = popt[0]
    A2 = popt[1]
    A3 = -1 - A1 - A2
    tau1_fit = popt[2]
    D1_fit = tau_to_D(tau1_fit)
    tau2_fit = popt[3]
    D2_fit = tau_to_D(tau2_fit)
    tau3_fit = popt[4]
    D3_fit = tau_to_D(tau3_fit)

    # Create a string summarizing the fit parameters
    msg = 'D1 = {0:.4g} um2/2, D2 = {1:.4g} um2/2, D3 = {2:.4g} um2/2. A1={3:.2g} A2={4:.2g}, A3={5:.2g}'.format(D1_fit, D2_fit, D3_fit, A1, A2, A3)
    print('tau1 = {}, tau2 = {}, tau3 = {}'.format(tau1_fit, tau2_fit,  tau3_fit))
    print(msg)

    # Generate the fitted curve and add it to the plot with a label containing the fit parameters
    yfit = exp_dec_3(xfit, *popt)

    return xfit, yfit



file = '/Users/george/Desktop/testing/GB_131_2021_08_10_HTEndothelial_BAPTA_plate1_9_cropped_trackid20_1327_locsID_tracksRG_SVMPredicted_NN_diffusion_velocity.csv'
df = pd.read_csv(file)
sld = df['velocity'] * 0.108 #convert to microns

x,y = updateCDF(sld)

fit1_X, fit1_Y = fit_exp_dec_1(x, y, min(x), max(x))
fit2_X, fit2_Y = fit_exp_dec_2(x, y, min(x), max(x))
#fit3_X, fit3_Y = fit_exp_dec_3(x, y, min(x), max(x))

fig = plt.scatter(x, y)
plt.plot(fit1_X,fit1_Y,'-',c='r')
plt.plot(fit2_X,fit2_Y,'-',c='g')
#plt.plot(fit3_X,fit3_Y,'-',c='y')
