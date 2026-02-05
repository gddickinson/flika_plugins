#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:04:39 2024

@author: george
"""

import os

def rename_tif_files(text='',replace='',fileType='',folder_path=''):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .tif file
        if filename.endswith(fileType):
            # Replace text with replace in the filename
            new_filename = filename.replace(text, replace)

            # Construct full file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

# Specify the folder path here
folder_path = '/Users/george/Data/unet_flika/training_images'

# Call the function
rename_tif_files(text='image', replace='image_2',fileType='.tif',folder_path=folder_path)
