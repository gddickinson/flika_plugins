#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:56:38 2024

@author: george
"""

import os

def rename_tif_files(directory):
    # Walk through all subdirectories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file has a .TIF extension
            if filename.endswith('.TIF'):
                # Construct the full file path
                old_file = os.path.join(root, filename)
                # Create the new filename by replacing .TIF with .tif
                new_filename = filename.replace('.TIF', '.tif')
                # Construct the new full file path
                new_file = os.path.join(root, new_filename)
                # Rename the file
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} -> {new_file}')

# Specify the directory containing the image files
directory = r'/Users/george/.FLIKA/plugins/imageSegmentation/data'

# Call the function to rename .TIF files
rename_tif_files(directory)
