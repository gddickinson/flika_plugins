# DipoleAnalysis FLIKA Plugin

A plugin for [FLIKA](https://github.com/flika-org/flika) that enables analysis of two-color dipole images, with capabilities for cluster detection, orientation analysis, and angle measurements.

## Overview

The DipoleAnalysis plugin provides a three-step workflow for analyzing two-color dipole images:
1. Filter clusters using dipole image mask
2. Extract and analyze clusters from two-color images
3. Rotate and analyze cluster orientations

## Features

- Multi-step image processing workflow
- Automated cluster detection and analysis
- Gaussian blur and threshold adjustments
- Angle measurements
- Orientation analysis
- Batch processing capabilities
- Result visualization and export

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- Dependencies:
  - NumPy
  - SciPy
  - scikit-image
  - matplotlib
  - pandas
  - PyQt5

## Installation

1. Ensure you have FLIKA installed
2. Install required dependencies:
```bash
pip install numpy scipy scikit-image matplotlib pandas
```
3. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

### Step 1: Filter Clusters by Dipole Image

1. Select dipole image window
2. Select two-color image window
3. Configure parameters:
   - Gaussian blur value
   - Threshold value
4. Click "Filter Clusters by Dipole Image"
5. Results are saved to specified output folder

### Step 2: Extract Clusters

1. Select processed two-color image
2. Configure extraction parameters:
   - Gaussian blur value
   - Threshold value
   - Minimum cluster area
3. Click "Extract Clusters from 2-Color Image"
4. Individual clusters are saved as separate files

### Step 3: Cluster Orientation Analysis

1. Load extracted clusters
2. Click "Rotate Clusters - Get Positions"
3. Plugin analyzes:
   - Cluster orientations
   - Origami angles
   - Helical domain angles
4. Results are saved with visualization plots

## Parameters

### Step 1
- **Gaussian Blur**: Smoothing factor for initial processing
- **Threshold**: Initial segmentation threshold

### Step 2
- **Gaussian Blur**: Smoothing for cluster detection
- **Threshold**: Cluster segmentation threshold
- **Minimum Cluster Area**: Size filter for clusters

### Output Options
- Separate result folders for each processing step
- CSV export of measurements
- Visualization plots
- Rotated and annotated images

## File Formats

### Input
- TIFF images:
  - Dipole image
  - Two-color fluorescence image

### Output
- Processed TIFF images
- CSV files with measurements
- PNG plots with annotations
- Analysis summary files

## Tips

1. **Image Preparation**:
   - Ensure images are properly aligned
   - Remove any artifacts before processing
   - Use consistent imaging parameters

2. **Parameter Optimization**:
   - Start with default values
   - Adjust based on image quality
   - Validate results visually

3. **Result Analysis**:
   - Check orientation measurements
   - Verify cluster detection
   - Review angle calculations

## Known Issues

- May require manual adjustment of thresholds for optimal results
- Processing time increases with image size
- Requires sufficient RAM for large datasets

## Future Improvements

- [ ] Automated parameter optimization
- [ ] Batch processing interface
- [ ] Additional measurement options
- [ ] 3D analysis capabilities
- [ ] Result visualization enhancements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson

## Version History

- 2019.05.21: Initial release

## Acknowledgments

- Built on the FLIKA plugin template
- Thanks to the FLIKA development team
- Uses scikit-image for advanced image processing

## Citation

If you use this software in your research, please cite:
[Citation information to be added]
