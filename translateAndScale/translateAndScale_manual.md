# Translate and Scale Plugin

A FLIKA plugin for aligning PIEZO1 single-molecule localization data to predefined micropattern templates. This tool provides both automated pattern detection and manual refinement capabilities for spatial alignment, rotation, and scaling of point data to match standardized template geometries.

## Overview

The `translateAndScale` plugin is designed to standardize the position and orientation of single-molecule localization microscopy (SMLM) data from cells grown on micropatterned substrates. By aligning experimental data to template patterns, it enables:

- Direct comparison between different recordings
- Standardized spatial analysis across experiments  
- Batch processing of multiple datasets
- Integration with downstream overlay and analysis plugins

This plugin is typically used as the first step in a multi-recording analysis workflow, preceding the `overlayMultipleRecordings` plugin.

## Background

### Micropattern Cell Culture

Micropatterning techniques control cell shape and adhesion by restricting where cells can attach to a substrate. Common micropattern geometries include:

- **Disc**: Circular patterns promoting radial symmetry
- **Square**: Four-fold symmetric patterns
- **Crossbow**: Asymmetric patterns for studying directional responses
- **Y-shape**: Three-fold symmetric branching patterns
- **H-shape**: Dumbbell-shaped patterns

These geometries allow researchers to study how cell shape and mechanical constraints affect PIEZO1 localization and mechanotransduction.

### PIEZO1 Mechanobiology

PIEZO1 channels transduce mechanical forces into calcium signals. Their spatial organization within cells is thought to be important for mechanosensing. By constraining cell geometry with micropatterns and using super-resolution imaging to visualize PIEZO1 localization, researchers can:

- Identify preferential localization regions
- Study relationship between cell shape and channel distribution
- Compare patterns across different cell types or conditions

## Installation

### Prerequisites

- Python 3.7+
- FLIKA installed and functional
- Required Python packages:
  - numpy
  - pandas
  - scipy
  - scikit-image
  - matplotlib
  - pyqtgraph
  - qtpy
  - tqdm

### Installation Steps

1. **Copy plugin to FLIKA directory:**
```bash
cp -r translateAndScale ~/.FLIKA/plugins/
```

2. **Restart FLIKA** or reload plugins via Plugin Manager

3. **Verify installation**: Check Plugins menu for "Translate and Scale"

## Data Requirements

### Input Files

The plugin requires two types of input:

#### 1. Image Stack (TIFF)
- **Format**: Multi-frame TIFF or single-frame image
- **Content**: Fluorescence microscopy data showing micropattern
- **Purpose**: Visual reference for pattern detection
- **Notes**: Plugin creates max projection if multi-frame

#### 2. Localization Data (CSV)
- **Format**: CSV file with comma-separated values
- **Required columns:**
  - `x` - x coordinates in pixels
  - `y` - y coordinates in pixels
- **Source**: Typically from particle tracking analysis (files with `_tracksRG` tag)
- **Optional columns**: Any additional data (preserved in output)

### Typical Data Pipeline

```
Raw microscopy
    ↓ (thunderSTORM or similar)
Localizations (*_locs.csv)
    ↓ (Single particle tracking)
Tracked data (*_tracksRG.csv)
    ↓ (THIS PLUGIN)
Aligned data (*_transform.csv, *_transform.tif)
    ↓ (overlayMultipleRecordings)
Multi-recording analysis
```

## Usage

### Starting the Plugin

```python
import flika
flika.start_flika()

# Open from menu
Plugins → Translate and Scale
```

Or programmatically:
```python
from flika.plugins.translateAndScale import TranslateAndScale
aligner = TranslateAndScale()
```

### Workflow

#### Step 1: Load Image

1. Open TIFF image in FLIKA showing the micropattern
2. Make it the current window
3. Select it in the plugin's "Window" dropdown

#### Step 2: Select Micropattern Template

Choose the appropriate template from the "Shape" dropdown:
- Disc
- Square  
- Crossbow
- Y-shape
- H-shape

The template determines what pattern the plugin will try to detect and align to.

#### Step 3: Start Automatic Alignment

Click **"Start Align"** button

The plugin will:
1. Convert image to binary using Otsu thresholding
2. Detect pattern center using center of mass
3. Estimate pattern size by comparing to template
4. Calculate rotation angle using phase correlation
5. Display an interactive ROI overlay showing detected alignment

A Region of Interest (ROI) appears on the image:
- **Position**: Detected pattern center
- **Size**: Detected pattern dimensions
- **Rotation**: Detected orientation angle

#### Step 4: Manual Refinement (Optional)

Adjust the ROI manually if automatic detection is imperfect:

**Position**: Drag ROI to reposition

**Rotation**: 
- Drag the rotation handle (typically on edge/corner)
- Or use arrow keys for fine adjustment

**Size**: 
- Drag size handles to adjust scale
- Note: Resizing only affects rescaling if "Rescale data" is checked

#### Step 5: Load Point Data

1. Click **"Load Data"** button
2. Select the CSV file containing localization/tracking data
3. Points are overlaid on the image (green dots)

A new window opens showing both:
- The original/max-projected image
- Localization points as green scatter plot

#### Step 6: Transform Data

Click **"Transform Data"** button

The plugin:
1. Rotates image and points to align with template (0° reference)
2. Centers pattern on the image
3. Optionally rescales if "Rescale data" is checked
4. Updates visualization with transformed data

#### Step 7: Save Results

Click **"Save Data"** button

Creates two output files:
- `*_transform.tif` - Aligned image
- `*_transform.csv` - Aligned coordinates
- `*_align.txt` - Alignment parameters

### Interactive ROI Controls

The alignment ROI provides visual feedback and manual control:

**ROI Display Elements:**
- Pattern outline matching selected template shape
- Center crosshair
- Rotation handle for angular adjustment
- Size handles for scaling

**ROI Information:**
- Current center position (x, y)
- Current rotation angle
- Current size (pattern scale)

## Template Patterns

### Built-in Templates

Each template is a binary 2D array defining the expected pattern:

#### Disc
- **Shape**: Circular
- **Dimensions**: 21×21 pixels
- **Symmetry**: Radially symmetric
- **Use case**: Isotropic cell spreading

#### Square  
- **Shape**: Square/rectangular
- **Dimensions**: 20×20 pixels
- **Symmetry**: 4-fold rotational
- **Use case**: Four-way symmetric constraints

#### Crossbow
- **Shape**: Asymmetric arrow-like pattern
- **Dimensions**: 20×26 pixels
- **Symmetry**: Mirror symmetric along one axis
- **Use case**: Directional studies

#### Y-Shape
- **Shape**: Three-armed branching
- **Dimensions**: 22×23 pixels
- **Symmetry**: 3-fold rotational
- **Use case**: Branching geometry studies

#### H-Shape
- **Shape**: Dumbbell/H-shaped
- **Dimensions**: 20×26 pixels
- **Symmetry**: 2-fold rotational
- **Use case**: Bi-polar cell organization

## Key Features

### Automatic Pattern Detection

The plugin uses computer vision techniques to detect patterns:

1. **Image Preprocessing:**
```python
# Apply Gaussian blur to reduce noise
img_blurred = gaussian_filter(img, sigma=3)

# Otsu thresholding for binarization
threshold = threshold_otsu(img_blurred)
binary = img_blurred > threshold
```

2. **Center Detection:**
```python
# Find center of mass of binary pattern
center_y, center_x = center_of_mass(binary)
```

3. **Size Estimation:**
```python
# Compare binary pattern size to template
# Rescale template to match pattern dimensions
size_ratio = detected_size / template_size
```

4. **Rotation Detection:**
```python
# Phase cross-correlation between binary image and template
# Determines shift and rotation
shift, error, phasediff = phase_cross_correlation(binary, template)
```

### Manual Alignment Controls

When automatic detection fails or needs refinement:

- **Drag ROI**: Reposition pattern center
- **Rotate handle**: Adjust orientation
- **Size handles**: Change scale
- **Keyboard**: Fine position adjustments

### Data Transformation

The transformation pipeline:

```python
# 1. Calculate rotation needed
target_angle = 0  # Standard orientation
rotation_angle = target_angle - detected_angle

# 2. Rotate points around their center
center = data[['x', 'y']].mean()
x_rot, y_rot = rotate_around_point(data['x'], data['y'], 
                                    rotation_angle, origin=center)

# 3. Rotate image
img_rotated = rotate(img, -rotation_angle, reshape=False)

# 4. Crop and center on pattern
img_centered = crop_to_pattern(img_rotated, center, crop_size=400)

# 5. Center points on image
img_center = img_centered.shape[0] / 2
x_final = x_rot + img_center - center[0]
y_final = y_rot + img_center - center[1]

# 6. Optionally rescale
if rescale_enabled:
    scale_factor = target_size / detected_size
    img_scaled = rescale(img_centered, scale_factor)
    x_final = x_final * scale_factor
    y_final = y_final * scale_factor
```

### Rescaling Option

**When to use rescaling:**
- Micropattern sizes vary between experiments
- Need to normalize to standard size for comparison
- Quantitative analysis requires size standardization

**How it works:**
1. Check "Rescale data" checkbox
2. Set "Target size" value (pixels)
3. Plugin calculates: `scale_factor = target_size / detected_size`
4. Both image and coordinates are scaled

**Note:** In early versions (as mentioned in email thread), rescaling was added later after initial testing revealed size variations between recordings.

## Control Panel Interface

### Main Controls
- **Window selector**: Choose image window
- **Shape selector**: Choose micropattern template
- **Start Align**: Initiate automatic detection
- **Load Data**: Load localization CSV file
- **Transform Data**: Apply transformation
- **Save Data**: Export aligned results
- **Rescale data**: Enable/disable size normalization
- **Target size**: Set standard size for rescaling (pixels)

## Output Files

### Alignment Parameters (`*_align.txt`)

CSV file containing alignment metadata:

| Parameter | Description |
|-----------|-------------|
| center_x | X coordinate of pattern center |
| center_y | Y coordinate of pattern center |
| angle | Rotation angle applied (degrees) |
| size | Detected pattern size (pixels) |
| pattern | Template name (disc, square, etc.) |
| file | Original image filename |

**Example:**
```csv
,center_x,center_y,angle,size,pattern,file
0,256.3,251.7,-12.4,245.8,square,experiment1.tif
```

### Transformed Image (`*_transform.tif`)

TIFF image with:
- Pattern centered on image
- Rotated to standard orientation (0°)
- Optionally rescaled to target size
- Ready for overlay with other aligned images

### Transformed Coordinates (`*_transform.csv`)

CSV file with all original columns plus:

| Column | Description |
|--------|-------------|
| x_transformed | Aligned x coordinates |
| y_transformed | Aligned y coordinates |

**Original columns preserved:**
- All tracking data (Rg, track_length, etc.)
- Original x, y coordinates
- Custom analysis parameters

## Practical Tips

### Achieving Good Alignment

1. **Image Quality:**
   - Ensure micropattern is clearly visible
   - Good signal-to-noise ratio helps detection
   - Max projection often better than single frames

2. **Template Selection:**
   - Choose the correct pattern template
   - Incorrect template leads to poor alignment
   - If unsure, test multiple templates

3. **Manual Refinement:**
   - Automatic detection is starting point
   - Always visually verify alignment
   - Fine-tune rotation for best match
   - Center pattern carefully

4. **Rescaling Decisions:**
   - Enable if pattern sizes vary significantly
   - Leave disabled if sizes are consistent
   - Choose target size based on typical pattern size

### Common Adjustments

**Pattern not detected:**
- Manually position ROI over pattern
- Adjust manually before clicking "Start Align"
- Check image contrast and quality

**Rotation is off:**
- Drag rotation handle to adjust
- Use small increments for precision
- Verify against pattern features

**Size mismatch:**
- Enable "Rescale data"
- Set appropriate target size
- Verify scaling doesn't distort data

### Batch Processing

For multiple files:

```python
import glob
import os

# Get all relevant files
image_files = glob.glob('/data/*.tif')
data_files = glob.glob('/data/*_tracksRG.csv')

for img_file, data_file in zip(image_files, data_files):
    # 1. Open image
    window = flika.open_file(img_file)
    
    # 2. Set window in plugin
    # (done via GUI)
    
    # 3. Select appropriate template
    # (done via GUI)
    
    # 4. Start alignment
    aligner.startAlign()
    
    # 5. Verify visually and adjust if needed
    # (manual step)
    
    # 6. Load data
    # Select data_file via GUI
    aligner.loadData()
    
    # 7. Transform
    aligner.transformData()
    
    # 8. Save
    aligner.saveTransformedData()
```

**Note:** Full automation requires additional scripting; GUI steps can be automated with PyQt signals.

## Troubleshooting

### Common Issues

**Problem**: ROI appears at wrong location
- **Cause**: Pattern detection failed
- **Solution**: Manually drag ROI to correct position before transforming

**Problem**: Points appear rotated incorrectly
- **Cause**: Rotation detection error or wrong template
- **Solution**: 
  1. Verify correct template selected
  2. Manually adjust rotation handle
  3. Check that image shows clear pattern

**Problem**: Transformed image is mostly black
- **Cause**: Cropping removed too much of image
- **Solution**: 
  1. Verify pattern is well-centered before transform
  2. Check pattern detection location
  3. May need to adjust crop size in code (default: 400 pixels)

**Problem**: Rescaling creates tiny or huge results
- **Cause**: Incorrect target size or size detection
- **Solution**:
  1. Disable rescaling temporarily
  2. Check detected size in alignment file
  3. Set target size appropriately (typically 200-400)

**Problem**: Error loading data file
- **Cause**: Missing required columns or wrong file format
- **Solution**:
  1. Verify file has `x` and `y` columns
  2. Check for proper CSV formatting
  3. File should be tracking output (`*_tracksRG.csv`)

### Size Variation Issues

As noted in the development email thread, early users encountered issues when micropattern sizes varied significantly between recordings. The rescaling feature was added to address this:

**Before rescaling was added:**
- Larger patterns would be displayed much bigger than smaller ones
- Direct comparison was difficult
- Heatmaps had different scales

**After rescaling feature:**
- All patterns normalized to same size
- Direct spatial comparison possible
- Consistent analysis scales

## Detailed Examples

### Example 1: Basic Alignment Workflow

```python
# 1. Start FLIKA
import flika
flika.start_flika()

# 2. Open image
img_window = flika.open_file('cell1_pattern.tif')

# 3. Open plugin
# Plugins → Translate and Scale

# 4. Setup
# - Select img_window in "Window" dropdown
# - Select "square" from "Shape" dropdown

# 5. Auto-detect
# Click "Start Align"
# - ROI appears showing detected pattern

# 6. Manual adjustment
# - Visually verify alignment
# - Drag rotation handle if needed
# - Reposition ROI if center is off

# 7. Load points
# Click "Load Data"
# - Select "cell1_tracksRG.csv"
# - Green points appear on image

# 8. Transform
# Click "Transform Data"
# - New window shows aligned image and points

# 9. Save
# Click "Save Data"
# Output files:
# - cell1_tracksRG_transform.tif
# - cell1_tracksRG_transform.csv
# - cell1_tracksRG_align.txt
```

### Example 2: Rescaling for Size Normalization

```python
# For a set of recordings with variable micropattern sizes

# First recording (size ~400 pixels):
# 1. Open image and data
# 2. Start alignment  
# 3. Check "Rescale data"
# 4. Set "Target size" = 400
# 5. Transform and save

# Second recording (size ~300 pixels):
# 1. Open image and data
# 2. Start alignment
# 3. Check "Rescale data"
# 4. Set "Target size" = 400  # Same as first
# 5. Transform and save

# Result: Both recordings now have patterns of same size (400 px)
# Can be directly compared in overlayMultipleRecordings
```

### Example 3: Manual Alignment for Difficult Patterns

```python
# When automatic detection fails:

# 1. Load image
# 2. Select appropriate template
# 3. Click "Start Align"
# 4. ROI appears (possibly in wrong place)

# Manual adjustment:
# 5. Drag ROI to correct position over pattern
# 6. Rotate ROI to match pattern orientation
#    - Identify distinctive feature (e.g., pattern asymmetry)
#    - Align feature to expected orientation
# 7. Adjust size handles if needed

# 8. Continue with Load Data → Transform Data → Save Data
```

### Example 4: Crossbow Pattern Alignment

```python
# Crossbow patterns are asymmetric - orientation matters!

# 1. Open image showing crossbow-patterned cell
# 2. Select "crossbow" template
# 3. Start alignment
# 4. Verify orientation:
#    - Arrow should point in consistent direction
#    - If pointing wrong way, rotate 180°
# 5. The crossbow template has specific orientation expectations
# 6. Transform and save
# 7. All crossbow recordings should align to same orientation
```

## Technical Details

### Coordinate Transformation Mathematics

**Rotation transformation:**

Given point (x, y), rotation angle θ, and origin (ox, oy):

```
x' = ox + (x - ox)cos(θ) + (y - oy)sin(θ)
y' = oy - (x - ox)sin(θ) + (y - oy)cos(θ)
```

**Implemented in code:**
```python
def rotate_around_point(x, y, angle, origin=(0,0)):
    radians = angle * math.pi / 180
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    dx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    dy = offset_y - sin_rad * adjusted_x + cos_rad * adjusted_y
    return dx, dy
```

Note the negative sign in dy calculation - this accounts for inverted y-axis in image coordinates.

### Image Processing Pipeline

1. **Max Projection** (if multi-frame):
```python
if len(img.shape) > 2:
    img = np.max(img, axis=0)
```

2. **Gaussian Filtering**:
```python
img_smooth = gaussian_filter(img, sigma=3)
```

3. **Binary Conversion**:
```python
threshold = threshold_otsu(img_smooth)
binary = img_smooth > threshold
```

4. **Morphological Operations** (optional):
```python
binary = binary_closing(binary)
binary = binary_fill_holes(binary)
```

5. **Pattern Matching**:
```python
# Phase correlation for alignment
shift, error, phasediff = phase_cross_correlation(
    binary, template, upsample_factor=10
)
```

### ROI Implementation

The plugin uses PyQtGraph's ROI (Region of Interest) functionality:

```python
# Create custom ROI class
class TemplateROI(pg.ROI):
    def __init__(self, pos, size, pattern):
        super().__init__(pos, size)
        self.pattern = pattern
        self.addRotateHandle([1, 1], [0.5, 0.5])
        self.addScaleHandle([0, 0], [1, 1])
        # Additional handles for resizing
        
    def getAngle(self):
        # Returns current rotation angle
        
    def getCenter(self):
        # Returns center position
        
    def getSize(self):
        # Returns pattern size
```

### File I/O

**Reading CSV:**
```python
data = pd.read_csv(filename)
# Preserves all columns
# Requires at minimum: 'x', 'y'
```

**Writing CSV:**
```python
data.to_csv(output_filename, index=None)
# index=None prevents row numbers from being saved
```

**Writing TIFF:**
```python
import skimage.io as skio
skio.imsave(filename, img_array)
```

## Integration with Analysis Pipeline

### Upstream Tools

**Single-Molecule Localization:**
- thunderSTORM (ImageJ/Fiji)
- pynsight (FLIKA plugin)
- Custom localization software

**Particle Tracking:**
- Custom FLIKA plugins
- TrackMate (ImageJ/Fiji)
- Python tracking libraries (trackpy, etc.)

### Downstream Tools

**Multi-Recording Overlay:**
- overlayMultipleRecordings (FLIKA plugin)

**Statistical Analysis:**
- Custom Python scripts
- R statistical packages
- MATLAB analysis tools

### Complete Workflow

```
Microscopy Acquisition
    ↓
Raw TIFF stacks
    ↓ (thunderSTORM)
Localizations (*_locs.csv)
    ↓ (Particle tracking)
Tracked particles (*_tracksRG.csv)
    ↓ (translateAndScale - THIS PLUGIN)
Aligned data (*_transform.csv, *_transform.tif)
    ↓ (overlayMultipleRecordings)
Combined heatmaps and analysis
    ↓ (Statistical analysis)
Publication figures and results
```

## Advanced Features

### Custom Templates

To add a new micropattern template:

1. Create binary array defining pattern:
```python
customShape = np.array([
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0]
])
```

2. Add to template dictionary in code:
```python
self.templateDict = {
    'Disc': discShape,
    'Square': squareShape,
    'Custom': customShape  # Add here
}
```

3. Update GUI selector:
```python
self.items['shape'].setItems(['Disc', 'Square', 'Custom'])
```

### Alignment Quality Metrics

To quantify alignment quality, you can:

1. **Calculate overlap between binary image and template:**
```python
overlap = np.sum(binary & template) / np.sum(template)
```

2. **Measure centroid distance:**
```python
detected_center = center_of_mass(binary)
expected_center = (img.shape[0]/2, img.shape[1]/2)
distance = np.linalg.norm(np.array(detected_center) - 
                          np.array(expected_center))
```

3. **Assess rotation accuracy:**
```python
# Compare pattern orientation to template
rotation_error = abs(detected_angle - expected_angle)
```

## Development History

Based on email correspondence between developers and users:

**v1.0 (February 2024):**
- Initial release
- Automatic pattern detection
- Manual ROI adjustment
- Transform and save functionality
- Templates: disc, square, crossbow, Y-shape, H-shape

**v1.1 (March 2024):**
- Added rescaling feature
- Addressed issues with variable pattern sizes
- Improved memory handling for large files
- Fixed scaling bugs for large datasets

**Key development insight** (from email thread): "I was originally going to include scaling but didn't as all the recordings I saw looked to be roughly the same size." - This was later added when users discovered size variation across experiments.

## Performance Considerations

### Memory Usage

Typical memory requirements:
- **Small datasets** (<10,000 points): Minimal (~50 MB)
- **Medium datasets** (10,000-100,000 points): Moderate (~200 MB)
- **Large datasets** (>100,000 points): Significant (>500 MB)

**Memory-saving tips:**
1. Close unnecessary windows
2. Process files one at a time
3. Use subsampling for initial alignment testing

### Processing Speed

Alignment speed depends on:
- Image size (larger = slower)
- Point count (more = slower)
- Pattern complexity

Typical processing times:
- Pattern detection: 1-5 seconds
- Data transformation: <1 second for most datasets
- Image rotation: 1-3 seconds

## Citation

When using this plugin, please cite:

```
Bertaccini, G. A., Casanellas, I., Evans, E. L., Nourse, J. L., Dickinson, G. D., 
Liu, G., Seal, S., Ly, A. T., Holt, J. R., Wijerathne, T. D., Yan, S., Hui, E. E., 
Lacroix, J. J., Panicker, M. M., Upadhyayula, S., Parker, I., & Pathak, M. M. (2025). 
Visualizing PIEZO1 localization and activity in hiPSC-derived single cells and organoids 
with HaloTag technology. Nature Communications, 16(1), 5556. 
https://doi.org/10.1038/s41467-025-59150-1
```

Original bioRxiv preprint:
```
Bertaccini, G. A. et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, Cellular 
and Tissue Imaging. bioRxiv, 2023.12.22.573117. 
https://doi.org/10.1101/2023.12.22.573117
```

## Contributing

To contribute to this plugin:

1. Fork the repository
2. Create a feature branch
3. Test with various micropattern geometries
4. Submit pull request with description and test cases

## Support and Contact

For assistance:

- **GitHub Issues**: Bug reports and feature requests
- **FLIKA Documentation**: https://flika-org.github.io/
- **Contact**: George Dickinson (george.dickinson@gmail.com)
- **Lab Website**: Pathak Lab, UC Irvine

## License

MIT License - See LICENSE file for details

## Acknowledgments

Developed by George Dickinson for the Pathak Laboratory at UC Irvine. This work was supported by research into PIEZO1 mechanotransduction using micropatterned human induced pluripotent stem cells.

Special thanks to Alan Ly, Gabriella Bertaccini, and Ignasi Casanellas for extensive testing and feedback during development.

## Related Resources

### FLIKA Resources
- FLIKA GitHub: https://github.com/flika-org/flika
- FLIKA Documentation: https://flika-org.github.io/
- Plugin Development Guide: https://flika-org.github.io/plugins.html

### Microscopy Analysis
- thunderSTORM: http://zitmen.github.io/thunderstorm/
- Single-molecule localization overview: Nature Methods reviews
- Phase correlation: scipy.signal.phase_cross_correlation documentation

### PIEZO1 Research
- Pathak Lab: https://pathaklab.bio.uci.edu/
- PIEZO1 structure (PDB): 6BPZ, 5Z10
- Mechanobiology reviews: Annual Review of Cell and Developmental Biology

## Appendix: Template Array Visualizations

### Disc Template

```
. . . . . . . . . . 1 . . . . . . . . . .
. . . . . . 1 1 1 1 1 1 1 1 1 . . . . . .
. . . . 1 1 1 1 1 1 1 1 1 1 1 1 1 . . . .
. . . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
. . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . .
. . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 .
. . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . .
. . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . .
. . . 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 . . .
. . . . 1 1 1 1 1 1 1 1 1 1 1 1 1 . . . .
. . . . . . 1 1 1 1 1 1 1 1 1 . . . . . .
. . . . . . . . . . 1 . . . . . . . . . .
```

### Square Template

```
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```

Note: Crossbow, Y-shape, and H-shape templates are more complex asymmetric patterns - refer to the code arrays for exact specifications.