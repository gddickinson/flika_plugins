# Translate and Scale Plugin

A FLIKA plugin for aligning PIEZO1 localizations to template patterns. Provides both automated and manual tools for spatial alignment, rotation, and scaling of point data to match predefined micropattern templates.

## Features

### Template Matching
- Multiple predefined templates:
  - Disc
  - Square
  - Crossbow
  - Y-shape
  - H-shape
- Automatic pattern detection
- Manual fine-tuning

### Transformation Tools
- Automatic alignment
- Manual position adjustment
- Rotation control
- Scale manipulation

### Analysis
- Phase correlation alignment
- Center of mass detection
- Binary pattern matching
- Transform validation

## Installation

1. Copy to FLIKA plugins directory:
```bash
cp -r translateAndScale ~/.FLIKA/plugins/
```

2. Dependencies:
```python
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, gaussian_filter
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate, rescale
```

## Usage

### Basic Operation

1. Load microscopy data:
```python
import flika
flika.start_flika()
window = open_file('data.tif')
```

2. Initialize plugin:
```python
from flika.plugins.translateAndScale import TranslateAndScale
transform = TranslateAndScale()
```

3. Start alignment:
```python
# Select template and start alignment
transform.startAlign()
```

### Template Selection

Available templates:
```python
templates = {
    'Disc': Standard circular pattern
    'Square': Square grid pattern
    'Crossbow': Crossbow shape
    'Y-shape': Y-shaped pattern
    'H-shape': H-shaped pattern
}
```

### Alignment Process

1. Automatic Detection:
- Image thresholding
- Pattern detection
- Initial alignment
- Phase correlation

2. Manual Refinement:
- Position adjustment
- Rotation control
- Scale modification
- Visual verification

## Key Functions

### Pattern Detection
```python
# Auto-detect micropattern
transform.startAlign()
```

### Transform Data
```python
# Apply transformation
transform.transformData()

# Save results
transform.saveTransformedData()
```

## GUI Components

### Main Window
- Template selector
- Control buttons
- Parameter display

### Control Panel
- Position controls
- Rotation adjustment
- Scale modification

### Display Parameters
- Center position
- Current angle
- Pattern size

## Parameters

### Alignment Settings
```python
settings = {
    'target_size': 400,
    'target_angle': 0,
    'gaussian_sigma': 3,
    'binary_threshold': 'otsu'
}
```

### Output Options
```python
export = {
    'save_transformed_image': True,
    'save_coordinates': True,
    'export_alignment': True
}
```

## Output Files

- `*_transform.tif`: Transformed image
- `*_transform.csv`: Transformed coordinates
- `*_align.txt`: Alignment parameters

## Workflow

### 1. Template Selection
- Choose appropriate pattern
- Set initial parameters
- Start alignment

### 2. Automatic Alignment
- Pattern detection
- Initial positioning
- Rotation estimation
- Scale adjustment

### 3. Manual Refinement
- Fine-tune position
- Adjust rotation
- Modify scale
- Verify alignment

### 4. Export Results
- Save transformed image
- Export coordinates
- Store alignment parameters

## Tips

1. Pattern Selection:
   - Match template to experiment
   - Consider pattern orientation
   - Account for scale differences

2. Alignment:
   - Start with auto-alignment
   - Verify pattern detection
   - Fine-tune manually if needed
   - Save intermediate results

3. Validation:
   - Check alignment visually
   - Verify transformations
   - Compare with reference

## Troubleshooting

Common issues and solutions:

1. Pattern Detection:
   - Adjust threshold
   - Check image contrast
   - Verify pattern visibility

2. Alignment:
   - Reset and retry
   - Use manual controls
   - Check parameter ranges

3. Export:
   - Verify file paths
   - Check permissions
   - Validate output format

## Implementation Details

### Pattern Detection
```python
# Preprocessing steps
1. Gaussian filtering
2. Otsu thresholding
3. Binary operations
4. Center detection

# Alignment steps
1. Scale estimation
2. Position correction
3. Rotation detection
4. Fine adjustment
```

### Coordinate Transform
```python
# Transform sequence
1. Rotation around center
2. Position adjustment
3. Scale normalization
4. Coordinate update
```

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