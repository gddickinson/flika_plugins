#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:41:37 2023

@author: george
"""

from skimage import data
from skimage.filters import threshold_otsu, threshold_local
from matplotlib import pyplot as plt


img = data.page()

global_thresh = threshold_otsu(img)
binary_global = img > global_thresh

block_size = 35
binary_adaptive_mask = threshold_local(img, block_size, offset=10)

binary_adaptive = img > binary_adaptive_mask

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(img)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()
