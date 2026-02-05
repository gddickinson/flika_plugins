# Overlay Multiple Recordings Plugin

A FLIKA plugin for aligning, comparing, and analyzing multiple PIEZO1 recordings. This tool enables precise spatial alignment of different recordings and generates combined visualizations and analyses.

## Features

### Alignment
- Interactive point/pattern alignment
- Translation controls (X, Y)
- Rotation adjustment
- Scale transformation
- Real-time visualization

### Analysis
- Heat map generation
- Density analysis
- Point distribution visualization
- Multi-recording comparisons

### Data Management
- Batch processing
- Position tracking
- Transform saving
- Coordinate system normalization

## Installation

1. Copy to FLIKA plugins directory:
```bash
cp -r overlayMultipleRecordings ~/.FLIKA/plugins/
```

2. Dependencies:
```python
import numpy as np
import pandas as pd
from scipy.ndimage import rotate
from skimage.registration import phase_cross_correlation
```

## Usage

### Basic Operation

1. Load reference image:
```python
import flika
flika.start_flika()
window = open_file('reference.tif')
```

2. Initialize plugin:
```python
from flika.plugins.overlayMultipleRecordings import OverlayMultipleRecordings
overlay = OverlayMultipleRecordings()
```

3. Load data and align:
```python
# Select data folder containing transform files
overlay.loadData()
```

### Control Panel

#### Movement Controls
- Left/Right translation
- Up/Down translation
- Clockwise/Counter-clockwise rotation
- Step size adjustment
- Rotation angle control

#### File Management
- File selection dropdown
- Data loading interface
- Transform saving

## Key Functions

### Data Loading
```python
# Load transform files from directory
overlay.loadData()

# Update display
overlay.update()
```

### Point Manipulation
```python
# Move points
overlay.move(shiftValue, direction)  # direction: 'l', 'r', 'u', 'd'

# Rotate points
overlay.rotatePoints(angle)
```

### Visualization
```python
# Plot all points
overlay.plotDataPoints()

# Plot selected data
overlay.plotSelectedDataPoints()

# Generate heatmap
overlay.plotHeatMap()
```

## GUI Components

### Main Window
- Control panel
- Visualization area
- File selector

### Control Panel
- Movement controls
- Transform controls
- Parameter adjustments

### Visualization Options
- Point display
- Heatmap
- Selected data highlighting

## Workflow

1. Data Preparation:
   - Organize transform files
   - Prepare reference image
   - Set analysis parameters

2. Alignment Process:
   - Load reference data
   - Select files for alignment
   - Apply transformations
   - Fine-tune positioning

3. Analysis:
   - Generate heatmaps
   - Compare distributions
   - Export results

## Parameters

### Movement Controls
```python
settings = {
    'stepSize': 1,
    'multiply': 1,
    'degrees': 1,
    'heatmapBins': 100
}
```

### Visualization
```python
display = {
    'pointSize': 2,
    'selectedColor': 'red',
    'defaultColor': 'green'
}
```

## Output Files

- `*_transform.csv`: Transformed coordinates
- `*_newPos_transform.csv`: Updated positions
- Heatmap visualizations

## Tips

1. Data Organization:
   - Use consistent file naming
   - Keep transform files together
   - Maintain folder structure

2. Alignment:
   - Start with coarse adjustments
   - Refine with small steps
   - Verify alignment visually

3. Analysis:
   - Check point distributions
   - Use appropriate bin sizes
   - Validate transformations

## Troubleshooting

Common issues and solutions:

1. Alignment Problems:
   - Check reference coordinates
   - Verify transform files
   - Reset position and try again

2. Visualization Issues:
   - Update point display
   - Adjust heatmap parameters
   - Refresh view

3. File Management:
   - Check file paths
   - Verify file formats
   - Update file permissions

## Best Practices

1. Before Starting:
   - Organize data files
   - Set up reference system
   - Plan alignment strategy

2. During Alignment:
   - Save progress regularly
   - Verify transformations
   - Document parameters

3. After Processing:
   - Validate results
   - Export transformed data
   - Back up alignments

## Contributing

To contribute:
1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## Citation

When using this plugin, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Support

For assistance:
- Open GitHub issue
- Check documentation
- Contact authors via paper

## License

MIT License - See LICENSE file for details