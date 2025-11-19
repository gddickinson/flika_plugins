#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:55:49 2023

@author: george
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
import os
import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage import data, color, measure
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.feature import canny
from skimage.draw import circle_perimeter, ellipse_perimeter
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import math
from math import cos, sin, degrees
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Arrow
from tqdm import tqdm


#fileName = r'/Users/george/Data/actin_test/actin/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate2_actin.tif'

fileName = r'/Users/george/actin_test/GB_199_2022_09_01_HTEndothelial_NonBAPTA_plate2_actin_Probabilities_.tiff'

img = skio.imread( fileName )

#if stack convert to single image
if len(img.shape) > 2:
    img= img[0][0] #just 1st image of stack

#orient image
img= np.rot90(img)
img= np.flipud(img)


# #Sci-kit image thresholding
# import matplotlib.pyplot as plt

# from skimage import data
# from skimage.filters import try_all_threshold


# fig2, ax2 = try_all_threshold(img, figsize=(10, 8), verbose=False)
# plt.show()


from skimage import data
from skimage.filters import threshold_otsu, threshold_local, threshold_isodata, gaussian
from matplotlib import pyplot as plt


#img = data.page()

gaussianBlur = False

if gaussianBlur:
    img = gaussian(img,sigma=1.5)

#global
global_thresh = threshold_otsu(img)
binary_global = img> global_thresh

#local
block_size = 35
offset = 0.00001
if gaussianBlur:
    offset = offset/100000

method = 'gaussian'
binary_adaptive_mask = threshold_local(img, block_size, method=method, offset=offset)
binary_adaptive = img > binary_adaptive_mask


fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2  = axes
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

