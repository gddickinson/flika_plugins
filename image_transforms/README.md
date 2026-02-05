# Image Transforms Plugin for FLIKA

A plugin providing geometric and intensity transformation operations for FLIKA image stacks, optimized for TIRF microscopy and experimental imaging applications.

## Features

### Rotation Operations
- **Rotate 90°**: Quick clockwise/counterclockwise rotation
- **Rotate Custom Angle**: Rotate by any angle with interpolation control

### Flip & Transpose
- **Flip Image**: Horizontal or vertical mirroring
- **Transpose**: Swap X and Y dimensions

### Intensity Operations
- **Invert Intensity**: Create negative images
- **Normalize Intensity**: Min-max, z-score, and percentile-based normalization

### Spatial Operations
- **Crop to Square**: Extract largest centered square
- **Bin Pixels**: Downsample with averaging or summing
- **Pad Image**: Add borders with various padding modes

## Installation

### Method 1: Direct Installation

1. Download or clone this repository
2. Copy the `image_transforms` folder to your FLIKA plugins directory:
   - **Windows**: `C:\Users\[YourUsername]\.FLIKA\plugins\`
   - **macOS/Linux**: `~/.FLIKA/plugins/`
3. Restart FLIKA

### Method 2: Using the Plugin Manager

1. Open FLIKA
2. Go to **Plugins → Plugin Manager**
3. Click **Install from GitHub**
4. Enter the repository URL
5. Click **Install**

## Usage

### Basic Usage

1. Open an image in FLIKA
2. Navigate to **Plugins → Image Transforms**
3. Select the desired operation from the submenu
4. Adjust parameters in the dialog
5. Click **Apply** to create a new window with the transformed image

### Example Workflows

#### TIRF Microscopy Workflow

```python
# Correct camera orientation and improve SNR
1. Plugins → Image Transforms → Rotation → Rotate 90° (if needed)
2. Plugins → Image Transforms → Spatial → Bin Pixels (2x2, mean)
3. Plugins → Image Transforms → Intensity → Normalize Intensity (percentile)
```

#### Prepare Data for Analysis

```python
1. Plugins → Image Transforms → Spatial → Crop to Square
2. Plugins → Image Transforms → Intensity → Normalize Intensity (minmax)
3. Plugins → Image Transforms → Spatial → Pad Image (to desired size)
```

### Programmatic Usage

You can also use the plugin functions programmatically:

```python
from flika import start_flika
from flika.process.file_ import open_file
import flika.plugins.image_transforms as transforms

# Start FLIKA
start_flika()

# Open an image
window = open_file('path/to/your/image.tif')

# Apply transformations
transforms.rotate_90(direction='clockwise', keepSourceWindow=True)
transforms.bin_pixels(bin_factor=2, method='mean')
transforms.normalize_intensity(method='minmax')
```

## Operations Reference

### Rotate 90°
- **Parameters**:
  - `direction`: 'clockwise' or 'counterclockwise'
- **Output**: New window with rotated image

### Rotate Custom Angle
- **Parameters**:
  - `angle`: Rotation angle in degrees (-180 to 180)
  - `reshape`: Expand canvas to fit rotated image (True/False)
  - `order`: Interpolation order (0=nearest, 1=linear, 3=cubic)
- **Output**: New window with rotated image

### Flip Image
- **Parameters**:
  - `direction`: 'horizontal' or 'vertical'
- **Output**: New window with flipped image

### Transpose
- **Parameters**: None
- **Output**: New window with transposed image

### Invert Intensity
- **Parameters**: None
- **Output**: New window with inverted intensities

### Normalize Intensity
- **Parameters**:
  - `method`: 'minmax', 'zscore', or 'percentile'
  - `percentile_low`: Lower percentile for clipping (0-100)
  - `percentile_high`: Upper percentile for clipping (0-100)
- **Output**: New window with normalized intensities

### Crop to Square
- **Parameters**: None
- **Output**: New window with largest centered square

### Bin Pixels
- **Parameters**:
  - `bin_factor`: Number of pixels to bin (2-8)
  - `method`: 'mean' or 'sum'
- **Output**: New window with binned image

### Pad Image
- **Parameters**:
  - `pad_width`: Number of pixels to add on each side (1-100)
  - `mode`: 'constant', 'edge', 'reflect', or 'symmetric'
  - `constant_value`: Value for constant padding (0-65535)
- **Output**: New window with padded image

## Tips for TIRF Microscopy

1. **Camera Orientation**: Use Rotate 90° to correct camera mounting
2. **Noise Reduction**: Apply 2x2 binning with mean for better SNR
3. **Standardization**: Use percentile normalization to handle outliers
4. **ROI Extraction**: Crop to square for uniform analysis regions
5. **Edge Artifacts**: Use reflect padding when preparing for convolution

## Performance Notes

- All operations work on both single images and multi-frame stacks
- Processing is done frame-by-frame to minimize memory usage
- For very large stacks (>1000 frames), consider processing in batches
- Custom angle rotation is slower than 90° rotation due to interpolation

## Requirements

- FLIKA >= 0.2.25
- NumPy
- SciPy
- Python 3.6+

## Troubleshooting

### Plugin doesn't appear in menu
- Check that all three files (`__init__.py`, `info.xml`, `about.html`) are present
- Verify the plugin is in `~/.FLIKA/plugins/image_transforms/`
- Restart FLIKA

### Rotation creates black borders
- Enable "reshape to fit" option in custom rotation
- Or manually pad the image before rotating

### Binning trims my image
- This is normal - dimensions must be divisible by bin factor
- The plugin automatically trims to the nearest multiple

### Out of memory errors
- Process smaller subsets of frames
- Reduce bin factor for large images
- Close other applications to free memory

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This plugin is distributed under the MIT license.

## Citation

If you use FLIKA in your research, please cite:

Ellefsen, K., Settle, B., Parker, I. & Smith, I. An algorithm for automated detection, localization and measurement of local calcium signals from camera-based imaging. *Cell Calcium*. 56:147-156, 2014

## Contact

- GitHub: [FLIKA Repository](https://github.com/flika-org/flika)
- Issues: Use the GitHub issue tracker

---

**Version**: 1.0.0  
**Author**: George  
**Last Updated**: 2024
