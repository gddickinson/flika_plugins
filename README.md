# FLIKA Plugins Collection

A collection of Python plugins developed at UCI to extend [FLIKA](https://github.com/flika-org/flika), an interactive image processing program for biologists.

## Overview

These plugins were developed for research at:
- [Parker Lab](https://parkerlab.bio.uci.edu/)
- [Pathak Lab](https://www.pathaklab-uci.com/)

## Installation

1. Download the desired plugin folder
2. Move it to your `~/.FLIKA/plugins` directory
3. Restart FLIKA
4. The plugin will appear in FLIKA's plugins menu

For detailed plugin installation instructions, visit the [FLIKA documentation](http://flika-org.github.io/plugins.html).

## Plugins

### Image Analysis
- **annotator** - Add annotations to image stacks
- **detect_puffs** - Calcium puff detection and analysis
- **dipoleAnalysis** - Analysis tools for dipole images
- **fft_Chunker** - FFT analysis on time-series data
- **openCV** - OpenCV-based image processing tools
- **overlay** - Image overlay and comparison tools

### ROI Tools
- **centerSurroundROI** - Create linked center and surround ROIs
- **roiExtras** - Additional ROI analysis capabilities
- **linescan** - Line scan analysis tools

### File Management & Loading
- **fileLoader** - Extended file format support including ND2
- **lightSheet_tiffLoader** - Specialized light sheet TIFF loading
- **packageManager** - Plugin and dependency management

### Specialized Analysis
- **pynsight** - Single molecule localization
- **quantimus** - Quantitative image analysis
- **rodentTracker** - Animal tracking and behavior analysis
- **locsAndTracksPlotter** - Particle tracking visualization and analysis
  - Compatible with:
    - pynsight
    - [thunderStorm](https://zitmen.github.io/thunderstorm/)
    - [trackpy](http://soft-matter.github.io/trackpy/v0.6.1/)

### Light Sheet Tools
- **light_sheet_analyzer** - Light sheet microscopy analysis
- **onTheFly_Reconstruction** - Real-time image reconstruction

### Simulation
- **simulateMEPP** - Miniature end-plate potential simulation
- **simulatePuff** - Calcium puff simulation

### Data Analysis
- **GlobalAnalysisPlugin** - Global data analysis tools
- **puffMapper** - Spatial analysis of calcium puffs
- **scaledAverageSubtract** - Background subtraction tools
- **synapse_plugin** - Synaptic activity analysis
- **volumeSlider** - 3D volume visualization and analysis

### Specialized Applications
- **neuroLab_207** - Neuroscience lab-specific tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

### Primary Contributors
- George Dickinson
- Kyle Ellefsen
- Brett Settle

### Acknowledgments
Several plugins were adapted or extended from original code by Kyle Ellefsen and Brett Settle.

## Support

For issues specific to these plugins, please use the GitHub issues system.
For general FLIKA support, visit the [FLIKA documentation](http://flika-org.github.io/).

## Version History

Updates and changes are tracked in individual plugin directories.

## Citing

If you use these plugins in your research, please cite both FLIKA and the specific plugin used.

## Contact

- **George Dickinson**
- Institution: University of California, Irvine
- Lab: Ian Parker Lab, McGaugh Hall
- Date: August 23, 2020

---
*Note: Each plugin directory contains detailed documentation specific to that tool.*