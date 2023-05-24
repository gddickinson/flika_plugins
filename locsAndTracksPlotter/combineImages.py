#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:57:01 2023

@author: george
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage

images = [r"/Users/george/Desktop/videoTest/main/000.jpg",r"/Users/george/Desktop/videoTest/zoom/000.jpg", r"/Users/george/Desktop/videoTest/trace/000.jpg"]


collection = skio.imread_collection(images)



#m = skimage.util.montage([collection[0],collection[1],collection[2]], multichannel=True)

skio.imshow_collection(collection, )
