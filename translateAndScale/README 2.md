# TranslateAndScale - FLIKA Plugin

A FLIKA plugin for aligning and transforming microscopy data with micropatterns. This plugin provides tools for automated pattern detection, alignment, and transformation of localization data, particularly useful for analyzing protein dynamics in micropatterned cells.

## Features

### Pattern Recognition
- Automated micropattern detection
- Support for multiple pattern templates:
  - Disc
  - Square
  - Crossbow
  - Y-shape
  - H-shape
- Automatic center detection and alignment
- Phase correlation for rotation estimation

### Data Transformation
- Interactive ROI manipulation
- Rotation and translation controls
- Optional rescaling functionality
- Coordinate transformation for localization data
- Batch processing capabilities

### Visualization
- Real-time visualization of transformations
- Interactive parameter display
- Point cloud overlay
- Template matching preview

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - pandas
  - scipy
  - scikit-image
  - PyQt5
  - pyqtgraph
  - tqdm
  - numba (optional, for performance)

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/translateAndScale.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your microscopy image stack
3. Navigate to `Plugins > TranslateAndScale`
4. Select:
   - Image Window
   - Data File (CSV format)
   - Pattern Template
   - Rescaling options

### Alignment Workflow
1. Click 'Restart Alignment' to begin
   - Automatic pattern detection
   - Initial alignment estimation
2. Fine-tune alignment if needed:
   - Drag ROI to adjust position
   - Use handles to adjust rotation
   - Modify size as needed
3. Click 'Save Alignment' to store transformation parameters

### Data Transformation
1. Load localization data (CSV format)
2. Adjust transformation parameters
3. Click 'Transform Data' to preview
4. Use 'Save Transform' to export:
   - Transformed image (TIFF)
   - Transformed coordinates (CSV)

## Input Data Format
- Image data: Standard microscopy formats
- Localization data: CSV files with columns:
  - x, y: Coordinates
  - Additional columns preserved in output

## Output Files
- `*_align.txt`: Alignment parameters
- `*_transform.tiff`: Transformed image
- `*_transform.csv`: Transformed coordinates

## Implementation Details

### Automatic Detection
- Gaussian filtering for noise reduction
- Otsu thresholding for segmentation
- Binary operations for pattern enhancement
- Phase correlation for rotation estimation

### ROI Manipulation
- Template-based ROI visualization
- Interactive handles for transformation
- Real-time parameter updates
- Center point tracking

## Version History

Current Version: 2024.01.17

## Author

George Dickinson (george.dickinson@gmail.com)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Best results with high-contrast micropatterns
- Automatic detection may need manual refinement
- Preprocessing heavy images may improve pattern detection
- Keep original data backups before transformation
- Performance depends on image and dataset size