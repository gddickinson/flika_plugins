#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:22:58 2023

@author: george
"""

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

#import numba

import pims
import trackpy as tp

# @pims.pipeline
# def gray(image):
#     return image[:, :, 1]  # Take just the green channel

frames = pims.open('/Users/george/Data/testing/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate1_2_MMStack_Default_bin10.tif')

#plt.imshow(frames[0]);

# =============================================================================
# #get single frame localizations
# f = tp.locate(frames[0], 11, invert=False)
# print(f.head())  # shows the first few rows of data
# tp.annotate(f, frames[0]);
# 
# fig, ax = plt.subplots()
# ax.hist(f['mass'], bins=20)
# 
# # Optionally, label the axes.
# ax.set(xlabel='mass', ylabel='count');
# f = tp.locate(frames[0], 11, invert=True, minmass=5)
# tp.annotate(f, frames[0]);
# tp.subpx_bias(f)
# =============================================================================

# tp.quiet()  # Turn off progress reports for best performance

#get localizations for all frames
f = tp.batch(frames[:], 19, minmass=5, invert=False);
#link
t = tp.link(f, 9, memory=3)
print(t.head())

#export


# =============================================================================
# t1 = tp.filter_stubs(t, 5)
# # Compare the number of particles in the unfiltered and filtered data.
# print('Before:', t['particle'].nunique())
# print('After:', t1['particle'].nunique())
# 
# plt.figure()
# tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass
# 
# t2 = t1[((t1['mass'] > 50) & (t1['size'] < 4.0) &
#          (t1['ecc'] < 0.3))]
# 
# plt.figure()
# tp.annotate(t2[t2['frame'] == 0], frames[0]);
# 
# 
# plt.figure()
# tp.plot_traj(t2);
# 
# 
# d = tp.compute_drift(t2)
# 
# d.plot()
# plt.show()
# 
# tm = tp.subtract_drift(t2.copy(), d)
# 
# 
# ax = tp.plot_traj(tm)
# plt.show()
# 
# im = tp.imsd(tm, 1000/108., 10)  # microns per pixel = 1000/108., frames per second = 10
# 
# fig, ax = plt.subplots()
# ax.plot(im.index, im, 'k-', alpha=0.1)  # black lines, semitransparent
# ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
#        xlabel='lag time $t$')
# ax.set_xscale('log')
# ax.set_yscale('log')
# 
# em = tp.emsd(tm, 1000/108., 10)  # microns per pixel = 1000/108., frames per second = 10
# 
# fig, ax = plt.subplots()
# ax.plot(em.index, em, 'o')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]',
#        xlabel='lag time $t$')
# ax.set(ylim=(10, 100000));
# 
# plt.figure()
# plt.ylabel(r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]')
# plt.xlabel('lag time $t$');
# tp.utils.fit_powerlaw(em)  # performs linear best fit in log space, plots]
# =============================================================================

