# Advanced Beam Splitter Plugin for FLIKA

## Overview

The Advanced Beam Splitter plugin is a comprehensive tool for aligning and processing dual-channel TIRF (Total Internal Reflection Fluorescence) microscopy images acquired through beam splitter systems. This plugin significantly extends the capabilities of the original BeamSplitter plugin with advanced image processing features essential for quantitative microscopy analysis.

## Key Features

### 1. **Geometric Transformations**
- **XY Translation**: Fine-tune pixel-level alignment between channels
- **Rotation Correction**: Correct for angular misalignment (±180°)
- **Scale/Magnification Correction**: Adjust for channel-specific magnification differences (0.5x to 2x)
- **Live Preview**: Real-time RGB overlay showing alignment results

### 2. **Background Subtraction**
Three methods for removing background signal:
- **Rolling Ball**: Simulates ImageJ's rolling ball algorithm for uneven illumination
- **Gaussian**: Blur-based background estimation and subtraction
- **Manual**: Percentile-based background removal

### 3. **Photobleaching Correction**
Essential for time-lapse imaging:
- **Exponential Fitting**: Fits and corrects exponential decay typical of photobleaching
- **Histogram Matching**: Matches intensity distributions across time points

### 4. **Intensity Normalization**
Automatically normalizes intensity ranges between channels for:
- Accurate co-localization analysis
- Consistent FRET measurements
- Quantitative intensity comparisons

### 5. **Auto-Alignment**
- Uses phase cross-correlation for automatic sub-pixel alignment
- Ideal for standard bead-based calibration
- Press 'A' key or click button to activate

### 6. **Undo/Revert Functionality**
- Always maintains original images in memory
- Press 'U' key or click button to revert to original state
- Safe experimentation with different parameters

## Installation

### Method 1: FLIKA Plugin Manager
1. Open FLIKA
2. Go to **Plugins → Plugin Manager**
3. Search for "Advanced Beam Splitter"
4. Click **Download** and **Install**

### Method 2: Manual Installation
1. Copy the plugin files to your FLIKA plugins directory:
   - `advanced_beam_splitter.py`
   - `__init__.py`
   - `info.xml`
2. Restart FLIKA
3. Plugin will appear under **Plugins → Advanced Beam Splitter**

## Dependencies

Required Python packages (usually auto-installed with FLIKA):
- `numpy` - Array operations
- `scipy` - Image transformations and optimization
- `pyqtgraph` - GUI controls
- `scikit-image` - Advanced image processing

## Usage Guide

### Quick Start

1. **Load Your Images**
   - Open both channel windows in FLIKA
   - One will be the reference (typically shown in red)
   - The other will be aligned to match (typically shown in green)

2. **Launch the Plugin**
   - Go to **Plugins → Advanced Beam Splitter**
   - The GUI window will open

3. **Select Windows**
   - **Reference Channel (Red)**: Select the window to align TO
   - **Align Channel (Green)**: Select the window to transform

4. **Coarse Alignment**
   - Click **Auto-Align (A)** for automatic initial alignment
   - OR use arrow keys for manual adjustment
   - Preview window shows real-time RGB overlay

5. **Fine Tuning**
   - Use keyboard shortcuts (see below) for precise control
   - Adjust rotation if channels are not parallel
   - Adjust scale if one channel has different magnification

6. **Apply Processing** (Optional)
   - Select background subtraction method if needed
   - Enable photobleaching correction for time-lapse data
   - Enable intensity normalization for quantitative analysis

7. **Apply**
   - Press **Enter** or click **OK** to generate aligned image
   - New window opens with transformed channel

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **↑** | Move up by 1 pixel |
| **↓** | Move down by 1 pixel |
| **←** | Move left by 1 pixel |
| **→** | Move right by 1 pixel |
| **R** | Rotate counterclockwise by 0.5° |
| **T** | Rotate clockwise by 0.5° |
| **+** or **=** | Increase scale by 0.01 |
| **-** | Decrease scale by 0.01 |
| **A** | Auto-align using cross-correlation |
| **U** | Revert to original images |
| **Enter** | Apply transformations and close |

### Typical Workflow for TIRF Beam Splitter Alignment

#### Initial Calibration (Using Fluorescent Beads)

1. Acquire images of multi-color fluorescent beads (e.g., TetraSpeck, 100nm)
2. Load both channel images in FLIKA
3. Launch Advanced Beam Splitter
4. Click Auto-Align - this usually works well for beads
5. Fine-tune manually if needed (beads should perfectly overlap)
6. Note the transformation parameters (x, y, rotation, scale)
7. Apply and save the aligned bead image for reference

#### Experimental Data Alignment

1. Load your experimental dual-channel images
2. Launch Advanced Beam Splitter
3. Input the transformation parameters from calibration, OR
4. Use Auto-Align if your sample has clear overlapping features
5. Enable background subtraction:
   - Use **rolling_ball** for uneven illumination (radius ≈ 50 pixels)
   - Use **gaussian** for smooth backgrounds (sigma ≈ 20 pixels)
   - Use **manual** for simple threshold-based removal (percentile ≈ 5)
6. For time-lapse data, enable photobleaching correction:
   - **Exponential** works best for uniform photobleaching
   - **Histogram** better for complex or non-uniform bleaching
7. Enable intensity normalization for quantitative co-localization
8. Press Enter to generate final aligned image

### Parameter Guidelines

#### **Background Subtraction Radius/Sigma**
- **Rolling Ball Radius**: 
  - Small features (5-20 pixels): radius = 50-100
  - Large features (>50 pixels): radius = 100-200
  - Should be larger than the largest feature
  
- **Gaussian Sigma**:
  - High frequency noise: sigma = 5-10
  - Uneven illumination: sigma = 20-50
  - Depends on spatial scale of background variation

- **Manual Percentile**:
  - Background-heavy images: 5-10
  - Signal-heavy images: 1-3

#### **Scale Factor**
- Typically very close to 1.0 (0.95-1.05)
- Can be measured from bead calibration images
- Accounts for slight differences in optical path magnification

#### **Rotation**
- Usually small angles (<5°) unless mechanical misalignment
- Can be measured from calibration patterns or edges

## Advanced Features

### ROI-Based Analysis
After alignment, you can:
1. Create ROIs in the aligned image
2. Use FLIKA's built-in ROI tools to measure intensities
3. ROIs will correspond accurately across both channels

### Batch Processing
For processing multiple image pairs with the same parameters:

```python
from plugins.advanced_beam_splitter import advanced_beam_splitter

# Define parameters from calibration
params = {
    'x_shift': 5,
    'y_shift': -3,
    'rotation': 1.2,
    'scale_factor': 0.998,
    'background_method': 'rolling_ball',
    'background_radius': 50,
    'photobleach_correction': 'exponential',
    'normalize_intensity': True
}

# Process multiple pairs
for red_win, green_win in window_pairs:
    result = advanced_beam_splitter(
        red_win, green_win,
        params['x_shift'], params['y_shift'],
        params['rotation'], params['scale_factor'],
        params['background_method'], params['background_radius'],
        params['photobleach_correction'], params['normalize_intensity']
    )
```

### Integration with Other FLIKA Tools
After alignment, use:
- **ROI Manager**: For multi-region analysis
- **Intensity Profile**: For line scan analysis  
- **Co-localization**: Pearson/Manders coefficient calculation
- **Tracking**: For following particles/features over time

## Troubleshooting

### Problem: Auto-align doesn't work
**Solution**: 
- Ensure both images have clear overlapping features
- Try manual coarse alignment first, then use auto-align for fine-tuning
- Check that images have sufficient contrast

### Problem: Preview window is blank or shows artifacts
**Solution**:
- Check that both windows are valid and contain image data
- Ensure x/y shifts aren't pushing image completely out of bounds
- Try resetting all parameters to default values

### Problem: Background subtraction removes too much signal
**Solution**:
- Reduce the radius/sigma parameter
- Try a different background subtraction method
- For manual method, use a lower percentile value

### Problem: Photobleaching correction makes images worse
**Solution**:
- Verify this is actually a time-lapse (3D) stack
- Try the alternative correction method
- Some samples may have non-uniform bleaching that's difficult to correct
- Consider disabling if not critically needed

### Problem: Memory errors with large stacks
**Solution**:
- Process individual frames or subsets
- Reduce image bit depth if possible (16-bit → 8-bit)
- Close unnecessary windows in FLIKA
- Increase available RAM

## Technical Details

### Transformation Order
Transformations are applied in this order:
1. Scaling
2. Rotation
3. Translation

This order minimizes artifacts and provides intuitive control.

### Coordinate System
- Origin (0,0) is at top-left corner
- Positive x-shift moves right
- Positive y-shift moves down
- Positive rotation is counterclockwise

### Interpolation
- Uses 3rd-order spline interpolation for smooth results
- Minimal interpolation artifacts for typical microscopy data

### Background Subtraction Algorithms
- **Rolling Ball**: Morphological opening with disk structuring element
- **Gaussian**: High-pass filtering by subtracting Gaussian-blurred image
- **Manual**: Simple percentile-based threshold subtraction

### Photobleaching Models
- **Exponential**: Fits I(t) = A × exp(-t/τ) + C
- **Histogram**: Matches cumulative distribution functions

## Best Practices

1. **Always calibrate with fluorescent beads first**
   - Use the same imaging settings as your experiments
   - Calibrate at the beginning of each imaging session
   - Save calibration images for documentation

2. **Use consistent imaging parameters**
   - Keep laser powers, exposure times constant
   - Use same objective, dichroics, filters
   - Maintain focus and prevent stage drift

3. **Background subtraction**
   - Apply conservatively to avoid removing real signal
   - Validate on control regions without signal
   - Consider acquiring background-only images

4. **Photobleaching**
   - Minimize during acquisition (lower power, anti-fade)
   - Only correct if it affects your measurements
   - Validate correction on control samples

5. **Documentation**
   - Record all transformation parameters
   - Save aligned images separately from raw data
   - Keep notes on which corrections were applied

## Citations and References

If you use this plugin in your research, please cite:

- **FLIKA**: Ellefsen, K. L., et al. "Applications of FLIKA, a Python-based image processing and analysis platform, for studying local events of cellular calcium signaling." *Biochimica et Biophysica Acta (BBA)-Molecular Cell Research* 1866.7 (2019): 1171-1179.

- **For photobleaching correction methods**: Miura, K. "Bleach correction ImageJ plugin for compensating the photobleaching of time-lapse sequences." *F1000Research* 9 (2020): 1494.

- **For background correction**: Peng, T., et al. "A BaSiC tool for background and shading correction of optical microscopy images." *Nature Communications* 8.1 (2017): 14836.

## Support and Feedback

For questions, bug reports, or feature requests:
- Open an issue on the FLIKA GitHub repository
- Contact the FLIKA development team
- Check the FLIKA documentation and forums

## Version History

### Version 2.0.0 (Current)
- Complete rewrite with advanced features
- Added rotation and scaling corrections
- Multiple background subtraction methods
- Photobleaching correction
- Intensity normalization
- Auto-alignment capability
- Comprehensive keyboard shortcuts
- Revert/undo functionality
- Improved preview performance

### Version 1.0 (Original BeamSplitter)
- Basic XY translation alignment
- Simple RGB overlay preview
- Arrow key controls

## License

This plugin is distributed under the same license as FLIKA (open source).

---

**Happy Imaging!**

For more information about TIRF microscopy and dual-channel imaging, consult:
- Axelrod, D. "Total internal reflection fluorescence microscopy in cell biology." *Traffic* 2.11 (2001): 764-774.
- Mattheyses, A. L., et al. "Imaging with total internal reflection fluorescence microscopy for the cell biologist." *Journal of Cell Science* 123.21 (2010): 3621-3628.
