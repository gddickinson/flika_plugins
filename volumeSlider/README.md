# Volume Slider

A comprehensive 4D volume viewer for light-sheet and volumetric microscopy data. Reshapes 3D stacks into 4D volumes and provides interactive slice-by-slice navigation with analysis tools.

## Features

- **4D volume reshaping** -- converts 3D stacks (frames x H x W) to 4D (volumes x slices x H x W)
- **Slice slider** -- navigate through Z slices within each volume
- **dF/F0 ratio** -- compute fluorescence change relative to baseline
- **Baseline subtraction** -- configurable baseline window
- **Data type conversion** -- float16, float32, float64
- **3D OpenGL visualization** -- interactive 3D volume rendering
- **Max/mean projection** -- project volumes along the Z axis
- **Batch processing** -- process multiple files with the same settings
- **Overlay support** -- overlay multiple volume channels
- **Export** -- save to numpy arrays or new flika windows

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- **PyOpenGL** (optional, for 3D volume viewer)

## Usage

1. Open a 3D image stack in flika (e.g., a light-sheet recording)
2. Go to **Plugins > Volume Slider**
3. Set the number of Z slices per volume (e.g., 10 slices for a 1000-frame stack gives 100 volumes)
4. Use the slice slider to navigate through the volume
5. Apply dF/F0, projections, or other processing as needed

## Parameters

| Parameter | Description |
|-----------|-------------|
| Slices per volume | Number of Z planes in each volume |
| Baseline start/end | Frame range for F0 baseline calculation |
| Data type | Output data type (float16/32/64) |
| Projection | Max or mean intensity projection |

## Programmatic Usage

```python
from flika import *
start_flika()

from flika.process.file_ import open_file
w = open_file('lightsheet_data.tif')

from volumeSlider import volumeSlider_Start
volumeSlider_Start.volumeSlider.gui()
```
