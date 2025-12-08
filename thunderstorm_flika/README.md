# ThunderSTORM for FLIKA

A comprehensive FLIKA plugin implementing thunderSTORM functionality for Single Molecule Localization Microscopy (SMLM) data analysis.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![FLIKA](https://img.shields.io/badge/FLIKA-0.2.25%2B-green)
![License](https://img.shields.io/badge/license-GPL--3.0-orange)

## Overview

ThunderSTORM for FLIKA brings comprehensive SMLM analysis capabilities to FLIKA, providing a complete Python implementation of the popular ImageJ thunderSTORM plugin. Analyze PALM, STORM, and other super-resolution microscopy data with state-of-the-art algorithms.

### Key Features

- **Complete Analysis Pipeline**: From raw images to super-resolution reconstruction
- **Multiple Algorithms**: Various filtering, detection, and fitting methods
- **Advanced PSF Fitting**: LSQ, WLSQ, MLE, Radial Symmetry, Centroid
- **Post-Processing Suite**: Quality filtering, merging, drift correction
- **Flexible Rendering**: Gaussian, Histogram, ASH, Scatter visualization
- **Simulation Tools**: Generate test data with ground truth
- **User-Friendly GUI**: Intuitive interface with tabbed organization

## Installation

### Prerequisites

```bash
# Install required Python packages
pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile
```

### Installation Steps

1. **Install the plugin files:**
   ```bash
   # Copy the entire thunderstorm_flika folder to FLIKA's plugin directory
   cp -r thunderstorm_flika ~/.FLIKA/plugins/
   ```

2. **Install the thunderstorm_python package:**
   ```bash
   # Copy the thunderstorm_python package to the plugin directory
   cp -r thunderstorm_python ~/.FLIKA/plugins/thunderstorm_flika/
   ```

3. **Verify installation:**
   - Restart FLIKA
   - Check for "ThunderSTORM" in the Plugins menu
   - If successful, you'll see a startup message in the console

### Directory Structure

After installation, your plugin directory should look like:

```
~/.FLIKA/plugins/thunderstorm_flika/
├── __init__.py
├── info.xml
├── about.html
├── README.md
└── thunderstorm_python/
    ├── __init__.py
    ├── filters.py
    ├── detection.py
    ├── fitting.py
    ├── postprocessing.py
    ├── visualization.py
    ├── simulation.py
    ├── utils.py
    └── pipeline.py
```

## Quick Start

### 1. Quick Analysis (Fastest Way)

```python
# In FLIKA:
# 1. Open your SMLM movie
# 2. Go to: Plugins → ThunderSTORM → Quick Analysis
# 3. View the super-resolution image!
```

### 2. Custom Analysis

```python
# In FLIKA:
# 1. Open your SMLM data
# 2. Go to: Plugins → ThunderSTORM → Run Analysis
# 3. Configure parameters in tabs:
#    - Filtering: wavelet, gaussian, DoG, etc.
#    - Detection: local maximum, non-maximum suppression
#    - Fitting: LSQ, WLSQ, MLE, Radial Symmetry
#    - Camera/Render: pixel size, rendering options
# 4. Click "Run" to analyze
# 5. Save localizations as CSV
```

## Available Tools

### 1. Run Analysis
**Menu:** `Plugins → ThunderSTORM → Run Analysis`

Complete SMLM analysis pipeline with full parameter control:
- **Filtering**: Wavelet (recommended), Gaussian, DoG, Lowered Gaussian, Median
- **Detection**: Local Maximum, Non-Maximum Suppression, Centroid
- **PSF Fitting**: Gaussian LSQ/WLSQ/MLE, Radial Symmetry, Centroid
- **Rendering**: Gaussian, Histogram, ASH, Scatter

### 2. Quick Analysis
**Menu:** `Plugins → ThunderSTORM → Quick Analysis`

Fast analysis with optimized default parameters. Great for quick inspection.

### 3. Post-Processing
**Menu:** `Plugins → ThunderSTORM → Post-Processing`

Refine localization data:
- **Quality Filtering**: Filter by intensity, uncertainty, sigma ranges
- **Density Filtering**: Remove isolated localizations
- **Molecule Merging**: Combine reappearing molecules (blinking correction)

### 4. Drift Correction
**Menu:** `Plugins → ThunderSTORM → Drift Correction`

Correct sample drift:
- **Cross-Correlation**: Estimate drift from reconstructions
- **Fiducial Markers**: Track fiducial markers

### 5. Rendering
**Menu:** `Plugins → ThunderSTORM → Rendering`

Create super-resolution images:
- **Gaussian**: High-quality rendering (each molecule as 2D Gaussian)
- **Histogram**: Fast rendering with optional jittering
- **ASH**: Average Shifted Histogram (good speed/quality balance)
- **Scatter**: Simple scatter plot

### 6. Simulate Data
**Menu:** `Plugins → ThunderSTORM → Simulate Data`

Generate synthetic SMLM data:
- Multiple test patterns (Siemens star, grid, circles, random)
- Realistic photon noise and PSF simulation
- Blinking dynamics
- Ground truth export for validation

## Typical Workflow

```
1. Load Data
   ↓
2. Run Analysis (or Quick Analysis)
   ↓
3. Inspect Results
   ↓
4. Drift Correction (if needed)
   ↓
5. Post-Processing (quality filtering, merging)
   ↓
6. Re-Render (with desired rendering method)
   ↓
7. Export Localizations & Images
```

## Parameter Guide

### Recommended Starting Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Filter | Wavelet | Works well for most data |
| Wavelet Scale | 2 | Standard scale |
| Detector | Local Maximum | Robust detection |
| Threshold | `std(Wave.F1)` | Adaptive threshold |
| Fitter | Gaussian LSQ | Good speed/quality |
| Fit Radius | 3 pixels | Adjust for spot size |
| Initial Sigma | 1.3 | Typical PSF width |
| Pixel Size | 100 nm | Check your camera specs |

### Threshold Expressions

The detection threshold supports mathematical expressions:
- `std(Wave.F1)` - Standard deviation (recommended)
- `2*std(Wave.F1)` - 2× standard deviation (more stringent)
- `mean(Wave.F1) + 3*std(Wave.F1)` - Mean + 3σ
- `100` - Fixed threshold value

### PSF Fitting Methods Comparison

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Gaussian LSQ | Fast | Good | General purpose |
| Gaussian WLSQ | Fast | Very Good | Low SNR data |
| Gaussian MLE | Slow | Excellent | Publication quality |
| Radial Symmetry | Very Fast | Good | Quick preview |
| Centroid | Very Fast | Fair | Simple analysis |

## Output Format

### Localization CSV Columns

| Column | Description | Units |
|--------|-------------|-------|
| `x [nm]`, `y [nm]` | Molecular positions | nm |
| `frame` | Frame number | - |
| `intensity` | Fitted intensity | photons |
| `background` | Local background | photons |
| `sigma_x`, `sigma_y` | PSF widths | pixels |
| `uncertainty` | Localization precision | nm |
| `chi_squared` | Goodness of fit | - |

## Tips & Best Practices

### Parameter Selection
- **Start with defaults**: Wavelet + Local Maximum + LSQ works for most data
- **Tune threshold**: Increase if too many false positives, decrease if missing molecules
- **Check PSF sigma**: Should be 1.0-1.6 pixels
- **Fit radius**: 3-5 pixels is usually sufficient

### Quality Control
- Mean intensity should be reasonable for your fluorophore
- Mean uncertainty <30 nm indicates good precision
- Sigma distribution should be narrow around expected PSF width
- Compare before/after post-processing to avoid over-filtering

### Performance
- **Quick preview**: Use Radial Symmetry fitter
- **Final analysis**: Use MLE for publication quality
- **Large datasets**: Use LSQ for speed/quality balance

## Troubleshooting

### Plugin Not Loading
- Check that `thunderstorm_python` package is in plugin directory
- Verify all dependencies are installed: `pip install numpy scipy scikit-image matplotlib pandas pywavelets`
- Check FLIKA console for error messages
- Ensure FLIKA version ≥ 0.2.25

### No Molecules Detected
- Lower the detection threshold
- Try different filter types (wavelet usually works best)
- Check that data is suitable for SMLM analysis
- Verify image has single molecules (not diffuse signal)

### PSF Fitting Fails
- Adjust `initial_sigma` to match your PSF size
- Increase `fit_radius` for bright, wide spots
- Try Radial Symmetry fitter for difficult data
- Check that detected spots are actual molecules

### Memory Issues
- Process smaller stacks at a time
- Use Radial Symmetry or Centroid for faster processing
- Close other applications to free memory

## Example Usage

### Python Script Example

```python
from flika import start_flika, global_vars as g
from flika.process.file_ import open_file

# Start FLIKA
start_flika()

# Open data
window = open_file('path/to/your/smlm_data.tif')

# Run analysis programmatically
from thunderstorm_flika import ThunderSTORM_RunAnalysis

analysis = ThunderSTORM_RunAnalysis()
analysis.show()  # Opens GUI
# Or set parameters and run:
# analysis.process()
```

### Batch Processing

```python
from pathlib import Path
from flika import global_vars as g
from flika.process.file_ import open_file
from thunderstorm_flika import quick_analysis_simple

# Process multiple files
data_dir = Path('data/')
for tif_file in data_dir.glob('*.tif'):
    print(f"Processing {tif_file.name}...")
    
    # Open file
    window = open_file(str(tif_file))
    
    # Quick analysis
    quick_analysis_simple()
    
    # Results are in the new window
    # Save as needed
```

## Scientific Reference

This plugin implements algorithms from:

> Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., & Hagen, G. M. (2014).  
> **ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis and super-resolution imaging.**  
> *Bioinformatics*, 30(16), 2389-2390.  
> DOI: [10.1093/bioinformatics/btu202](https://doi.org/10.1093/bioinformatics/btu202)

### Related Publications

The algorithms are based on methods from:
- **Wavelet filtering**: Izeddin et al. (2012)
- **MLE fitting**: Smith et al. (2010)
- **Radial symmetry**: Parthasarathy (2012)
- **Drift correction**: Mlodzianoski et al. (2011)

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your_username/thunderstorm_flika.git
cd thunderstorm_flika

# Install in development mode
ln -s $(pwd) ~/.FLIKA/plugins/thunderstorm_flika

# Make changes and test in FLIKA
```

## Support

- **Issues**: [GitHub Issues](https://github.com/your_username/thunderstorm_flika/issues)
- **Email**: george@research.edu
- **Documentation**: See `about.html` for detailed documentation

## License

This plugin is distributed under the GNU General Public License v3.0, consistent with the original thunderSTORM plugin.

## Acknowledgments

- Original thunderSTORM ImageJ plugin by Martin Ovesný et al.
- FLIKA development team
- All cited algorithm authors

---

**ThunderSTORM for FLIKA** - Bringing super-resolution to FLIKA  
Version 1.0.0 | 2024
