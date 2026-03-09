"""
FFT Chunker Plugin - Example Usage
Performs FFT-based spectral analysis on time-series data in chunks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test time series with known frequency components
n_frames, h, w = 1000, 32, 32
t = np.arange(n_frames)
# Signal with 1 Hz and 5 Hz components (assuming 30 fps)
signal = 100 + 20 * np.sin(2 * np.pi * 1.0 * t / 30) + 10 * np.sin(2 * np.pi * 5.0 * t / 30)
test_data = np.random.poisson(50, (n_frames, h, w)).astype(np.float32)
test_data += signal[:, np.newaxis, np.newaxis]

win = Window(test_data, name='Oscillating Signal')

# Launch FFT Chunker GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('fft_Chunker')

from plugins.fft_Chunker import fft_Chunker
fft_Chunker.gui()

# The GUI allows you to:
# - Set chunk size (number of frames per FFT window)
# - Set timestep (frame interval in seconds)
# - Define baseline and puff time ranges
# - Select output file for results
# - Plot frequency spectra
