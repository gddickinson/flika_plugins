"""
Example scripts for using the Advanced Beam Splitter Plugin
"""

# ============================================================================
# EXAMPLE 1: Basic Alignment
# ============================================================================

from plugins.advanced_beam_splitter import advanced_beam_splitter
import flika.global_vars as g

# Get the current open windows
red_win = g.m.currentWindow  # Or specify by name
green_win = g.m.windows[1]   # Or get by index

# Simple alignment with just translation
result = advanced_beam_splitter(
    red_window=red_win,
    green_window=green_win,
    x_shift=5,              # pixels to shift right
    y_shift=-3,             # pixels to shift up (negative = up)
    rotation=0,             # no rotation
    scale_factor=1.0,       # no scaling
    background_method='none',
    background_radius=50,
    photobleach_correction='none',
    normalize_intensity=False
)


# ============================================================================
# EXAMPLE 2: Full Processing Pipeline for TIRF
# ============================================================================

# Complete workflow with background correction, bleach correction, and normalization
result = advanced_beam_splitter(
    red_window=red_win,
    green_window=green_win,
    x_shift=5,
    y_shift=-3,
    rotation=1.2,           # slight rotation correction
    scale_factor=0.998,     # minor magnification difference
    background_method='rolling_ball',
    background_radius=50,
    photobleach_correction='exponential',
    normalize_intensity=True
)

# Access the result
aligned_image = result.image
result.setName("Aligned_and_Processed")


# ============================================================================
# EXAMPLE 3: Batch Processing Multiple Image Pairs
# ============================================================================

# First, determine optimal parameters from a calibration image
# (Do this manually with the GUI using bead images)

calibration_params = {
    'x_shift': 5,
    'y_shift': -3,
    'rotation': 1.2,
    'scale_factor': 0.998,
    'background_method': 'rolling_ball',
    'background_radius': 50,
    'photobleach_correction': 'exponential',
    'normalize_intensity': True
}

# Load multiple image pairs
import os
from flika import start_flika
from flika.window import Window

# Assume you have pairs of files in a directory
data_dir = "/path/to/your/data"
red_files = sorted([f for f in os.listdir(data_dir) if 'red' in f.lower()])
green_files = sorted([f for f in os.listdir(data_dir) if 'green' in f.lower()])

# Process each pair
for red_file, green_file in zip(red_files, green_files):
    # Load images
    red_path = os.path.join(data_dir, red_file)
    green_path = os.path.join(data_dir, green_file)
    
    red_win = Window(red_path)
    green_win = Window(green_path)
    
    # Apply processing
    result = advanced_beam_splitter(
        red_win, green_win, **calibration_params
    )
    
    # Save result
    output_name = red_file.replace('red', 'aligned')
    result.image  # Access the numpy array if needed
    # Save using FLIKA's save functions or numpy
    
    # Clean up
    red_win.close()
    green_win.close()


# ============================================================================
# EXAMPLE 4: ROI Analysis After Alignment
# ============================================================================

# After alignment, perform ROI-based analysis
result = advanced_beam_splitter(
    red_win, green_win,
    x_shift=5, y_shift=-3,
    rotation=0, scale_factor=1.0,
    background_method='rolling_ball',
    background_radius=50,
    photobleach_correction='none',
    normalize_intensity=True
)

# Now you can create ROIs and measure intensities
# The aligned image ensures both channels are properly registered
from flika.roi import makeROI

# Create a rectangular ROI
roi = makeROI('rectangle', result)
roi.setPos([50, 50])
roi.setSize([100, 100])

# Get intensity trace from ROI
trace = roi.getTrace()

# For dual-channel analysis, you can also analyze the original red channel
# with the same ROI coordinates since alignment is now correct


# ============================================================================
# EXAMPLE 5: Using Auto-Alignment Programmatically
# ============================================================================

# You can also use the auto-alignment feature programmatically
plugin = advanced_beam_splitter

# This will calculate optimal x, y shifts (rotation and scale would need manual tuning)
x_shift, y_shift, rotation, scale = plugin.auto_align_by_correlation(
    red_win.image, 
    green_win.image
)

print(f"Auto-detected alignment: x={x_shift}, y={y_shift}")

# Then apply with detected parameters
result = plugin(
    red_win, green_win,
    x_shift=x_shift,
    y_shift=y_shift,
    rotation=rotation,
    scale_factor=scale,
    background_method='gaussian',
    background_radius=30,
    photobleach_correction='histogram',
    normalize_intensity=False
)


# ============================================================================
# EXAMPLE 6: Processing Only Specific Frames from a Time Series
# ============================================================================

import numpy as np

# If you want to process only certain frames
full_red = red_win.image    # Shape: (t, x, y)
full_green = green_win.image

# Extract specific time range
start_frame = 10
end_frame = 50
subset_red = full_red[start_frame:end_frame]
subset_green = full_green[start_frame:end_frame]

# Create temporary windows
from flika.window import Window
temp_red = Window(subset_red, name="Red_Subset")
temp_green = Window(subset_green, name="Green_Subset")

# Process subset
result = advanced_beam_splitter(
    temp_red, temp_green,
    x_shift=5, y_shift=-3,
    rotation=0, scale_factor=1.0,
    background_method='gaussian',
    background_radius=20,
    photobleach_correction='exponential',
    normalize_intensity=True
)


# ============================================================================
# EXAMPLE 7: Different Background Methods Comparison
# ============================================================================

# Test different background subtraction methods to see which works best

methods = ['none', 'rolling_ball', 'gaussian', 'manual']
results = {}

for method in methods:
    result = advanced_beam_splitter(
        red_win, green_win,
        x_shift=5, y_shift=-3,
        rotation=0, scale_factor=1.0,
        background_method=method,
        background_radius=50,  # or sigma for gaussian, percentile for manual
        photobleach_correction='none',
        normalize_intensity=False
    )
    results[method] = result
    result.setName(f"Method_{method}")

# Now you can visually compare the results


# ============================================================================
# EXAMPLE 8: Custom Processing Pipeline
# ============================================================================

# For advanced users: access individual methods

plugin = advanced_beam_splitter

# Load images
red_img = red_win.image
green_img = green_win.image

# Step 1: Background subtraction
green_bg_corrected = plugin.subtract_background(
    green_img, 
    method='rolling_ball', 
    radius=50
)

# Step 2: Photobleaching correction (for time series only)
if green_img.ndim == 3:
    green_bleach_corrected = plugin.correct_photobleaching(
        green_bg_corrected,
        method='exponential'
    )
else:
    green_bleach_corrected = green_bg_corrected

# Step 3: Apply geometric transformations
green_aligned = plugin.apply_transformations(
    green_bleach_corrected,
    red_img,
    x_shift=5,
    y_shift=-3,
    rotation_angle=1.2,
    scale_factor=0.998
)

# Step 4: Normalize intensity
green_final = plugin.normalize_intensity_channels(
    green_aligned,
    red_img
)

# Create result window
result = Window(green_final, name="Custom_Processed")


# ============================================================================
# NOTES
# ============================================================================

"""
Tips for best results:

1. Always calibrate with fluorescent beads first using the GUI
2. Record your calibration parameters for batch processing
3. Test different background methods on your specific data
4. For time-lapse, only apply photobleaching correction if it's significant
5. Normalize intensity only when doing quantitative comparisons
6. Keep original data - never overwrite raw files
7. Document all processing steps in your methods section

For GUI usage:
- Launch from Plugins menu
- Use keyboard shortcuts for fine control
- Press 'A' for auto-alignment
- Press 'U' to revert if you make a mistake
- Preview updates in real-time

Common parameter ranges:
- x_shift, y_shift: typically -20 to +20 pixels
- rotation: typically -5 to +5 degrees  
- scale_factor: typically 0.95 to 1.05
- background_radius: 20-100 pixels depending on feature size
- Background percentile: 1-10
"""
