# SPT Batch Analysis Plugin for FLIKA

**Version:** 2025.08.26  
**Author:** George Dickinson  
**Platform:** FLIKA (FLuorescence Image analysis Kit)

A comprehensive FLIKA plugin for automated batch analysis of Single Particle Tracking (SPT) data from fluorescence microscopy experiments. Designed for high-throughput analysis of mechanosensitive channel dynamics, membrane protein trafficking, and cellular mechanotransduction studies.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Analysis Methods](#analysis-methods)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

The SPT Batch Analysis plugin integrates particle detection, linking, feature calculation, and advanced statistical analysis into a unified workflow optimized for batch processing. Originally developed for analyzing PIEZO1 mechanosensitive channel dynamics in TIRF microscopy experiments, the plugin is applicable to any single-molecule or particle tracking study.

### Design Philosophy

- **Modular Architecture**: Individual analysis components can be enabled/disabled independently
- **Batch Processing**: Automated processing of multiple datasets with comprehensive error handling
- **Reproducibility**: JSON configuration files and metadata export ensure analysis reproducibility
- **Flexibility**: Multiple particle detection and linking algorithms with extensive parameter control
- **Publication-Ready**: Generates formatted visualizations and comprehensive statistical outputs

---

## Key Features

### 🔬 Particle Detection
- **U-Track Integration**: Statistical significance-based particle detection
- **Background Estimation**: Automatic background and noise level calculation
- **Adaptive Thresholding**: Alpha-level significance testing for robust detection
- **Quality Filtering**: Minimum intensity thresholds and PSF-based filtering

### 🔗 Particle Linking
Three complementary algorithms for trajectory reconstruction:

1. **Built-in Method**: Fast gap-closing algorithm optimized for sparse particles
2. **Trackpy**: Crocker & Grier algorithm with adaptive search radius
3. **U-Track Mixed Motion**: LAP-based linking with heterogeneous motion models (PMMS)

### 📊 Feature Analysis
Over 45 quantitative features including:
- **Geometric**: Radius of gyration, asymmetry, skewness, kurtosis, fractal dimension
- **Kinematic**: Velocities, mean squared displacement, directional persistence
- **Spatial**: Nearest neighbor analysis, local density, clustering metrics
- **Temporal**: Lag displacements, autocorrelation, dwell times

### 🤖 Machine Learning
- **SVM Classification**: Automated mobility classification (mobile/confined/trapped)
- **Box-Cox Preprocessing**: Feature normalization for improved classification
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Custom Training**: Use your own labeled datasets

### 📈 Advanced Analysis
- **Geometric Analysis Suite**: Alternative Rg calculations and scaled metrics
- **Autocorrelation Analysis**: Directional persistence following Gorelik & Gautreau
- **Background Subtraction**: ROI-based intensity correction
- **Data Enhancement**: Missing point interpolation and intensity calculation

---

## Installation

### Prerequisites

- FLIKA (version 0.2.25 or higher)
- Python 3.7+
- Required packages (automatically installed):
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - scikit-image
  - tqdm
  - matplotlib

### Installation Methods

#### Method 1: FLIKA Plugin Manager (Recommended)

```python
# In FLIKA:
Plugins → Plugin Manager → Available Plugins
# Search for "SPT Batch Analysis" and click Install
```

#### Method 2: Manual Installation

```bash
# Clone or download the repository
cd ~/.FLIKA/plugins/
git clone https://github.com/username/spt_batch_analysis.git

# Or download and extract:
# https://github.com/username/spt_batch_analysis/archive/master.zip
```

#### Method 3: Direct Import

```python
# Place the plugin folder in your FLIKA plugins directory
# Windows: C:\Users\<username>\.FLIKA\plugins\
# Mac/Linux: ~/.FLIKA/plugins/
```

### Verify Installation

```python
# Launch FLIKA
# Check: Plugins → SPT Batch Analysis → Launch SPT Analysis
# Should open the plugin interface
```

---

## Quick Start

### Basic Workflow

1. **Organize Your Data**
   ```
   experiment_folder/
   ├── condition1/
   │   ├── image001.tif
   │   ├── image002.tif
   │   └── ...
   └── condition2/
       ├── image001.tif
       └── ...
   ```

2. **Launch Plugin**
   - Open FLIKA
   - Navigate to: `Plugins → SPT Batch Analysis → Launch SPT Analysis`

3. **Configure Analysis**
   - **Files Tab**: Select input directory and file pattern (`*.tif`)
   - **Parameters Tab**: Choose linking method (start with "Built-in")
   - **Analysis Steps Tab**: Enable desired analyses
   - Click **Start Analysis**

4. **Review Results**
   - Results saved in same directory as input files
   - Check `_enhanced_analysis.csv` for complete results
   - View `_autocorr_plot.png` for directional persistence

### Example: PIEZO1 Channel Analysis

```python
# Recommended settings for PIEZO1 TIRF microscopy:
{
  "pixel_size": 108,           # nm per pixel
  "frame_length": 0.1,         # seconds per frame
  "min_track_segments": 4,     # minimum track length
  "max_gap_frames": 36,        # allow blinking
  "max_link_distance": 3,      # pixels
  "enable_nearest_neighbors": true,
  "enable_velocity_analysis": true,
  "enable_diffusion_analysis": true,
  "enable_straightness_analysis": true
}
```

---

## Detailed Usage

### File Organization

The plugin expects files organized with the following structure:

```
data/
├── experiment1/
│   ├── sample.tif              # Raw microscopy image
│   ├── sample_locs.csv         # Optional: pre-detected particles
│   └── sample_ROI.txt          # Optional: background ROI
├── experiment2/
│   └── ...
└── config.json                 # Optional: analysis configuration
```

#### Supported File Types

- **Images**: `.tif`, `.tiff` (multi-frame stacks)
- **Localizations**: `.csv` with ThunderSTORM-compatible format
- **ROI Files**: `.txt` with coordinates for background measurement
- **Configuration**: `.json` with analysis parameters

### Interface Tabs

#### 1. Files Tab
Select input files and configure batch processing:
- **Input Directory**: Root folder containing your data
- **File Pattern**: Glob pattern for file matching (e.g., `*.tif`, `*_locs.csv`)
- **Recursive Search**: Enable to search subdirectories
- **Experiment Name**: Auto-detected from folder structure or manual entry

#### 2. Detection Tab
Configure U-Track particle detection (optional):
- **Enable Detection**: Turn on to detect particles from raw images
- **PSF Sigma**: Expected point spread function width (pixels)
- **Alpha Threshold**: Statistical significance level (0.001-0.5)
- **Minimum Intensity**: Absolute intensity cutoff
- **Output Format**: Choose localization file format

> **Note**: Skip detection if using pre-detected localizations from ThunderSTORM or similar tools.

#### 3. Parameters Tab
Choose and configure particle linking:

**Built-in Method:**
```python
max_link_distance: 3      # Maximum linking distance (pixels)
max_gap_frames: 36        # Maximum allowed gap (frames)
min_track_segments: 4     # Minimum detections per track
```

**Trackpy Method:**
```python
search_distance: 5        # Search radius (pixels)
memory: 10                # Gap closing memory
adaptive_stop: 1.5        # Adaptive termination threshold
adaptive_step: 0.95       # Adaptive search reduction
```

**U-Track Mixed Motion:**
```python
motion_model: "brownian_confined"  # Motion type
regime_sensitivity: 0.3            # Regime change sensitivity
diffusion_scaling: [0.5, 2.0]     # Diffusion coefficient range
```

#### 4. Analysis Steps Tab
Enable desired analysis components:
- ☑ **Nearest Neighbors**: Spatial clustering analysis
- ☑ **Velocity Analysis**: Instantaneous and mean velocities
- ☑ **Diffusion Analysis**: MSD and diffusion coefficients
- ☑ **SVM Classification**: Machine learning mobility classification
- ☑ **Background Subtraction**: ROI-based intensity correction
- ☑ **Localization Error**: Precision metrics
- ☑ **Straightness Analysis**: Path directionality

#### 5. Geometric Analysis Tab
Alternative geometric analysis methods:
- **Simple Rg**: Direct radius of gyration calculation
- **Scaled Rg**: Normalized metrics for mobility classification
- **Linear Motion**: Detect predominantly linear trajectories

#### 6. Autocorrelation Tab
Directional persistence analysis:
- **Max Lag**: Maximum time lag for autocorrelation
- **Lag Step**: Interval between lag points
- **Plot Options**: Customize visualization appearance
- **Live Analysis**: Analyze currently loaded tracks

#### 7. Export Control Tab
Customize output files:
- **Column Selection**: Choose which features to export
- **Export Formats**: Enable multiple file types
- **Metadata Export**: Save analysis parameters
- **Intermediate Files**: Keep tracks.csv and features.csv

#### 8. Progress Tab
Monitor analysis execution:
- Real-time progress updates
- Error logging
- Processing statistics
- Performance metrics

---

## Analysis Methods

### Particle Detection

The plugin uses U-Track's detection algorithm based on:

1. **Background Estimation**: Gaussian fitting to image histogram
2. **Noise Characterization**: Standard deviation calculation
3. **Significance Testing**: Alpha-level hypothesis testing
4. **Spot Fitting**: 2D Gaussian PSF fitting to localizations

**Key Parameters:**
- `psfSigma`: Expected PSF width (typical: 1.0-2.0 pixels)
- `alphaLocMax`: Significance threshold (lower = stricter)
- `minIntensity`: Absolute intensity filter

### Particle Linking

#### Built-in Method
Fast nearest-neighbor linking with gap closing:
- Links particles within `max_link_distance`
- Closes gaps up to `max_gap_frames`
- Optimized for sparse, well-separated particles

#### Trackpy
Implements Crocker & Grier (1996) algorithm:
- Adaptive search radius
- Memory-based gap closing
- Optimized for dense particle fields
- Supports subnet tracking

#### U-Track Mixed Motion
LAP-based linking with motion models:
- Supports heterogeneous motion (PMMS)
- Handles diffusion coefficient switching
- Advanced gap closing with cost matrices
- Best for complex cellular dynamics

### Feature Calculation

#### Geometric Features
- **Radius of Gyration**: `sqrt(mean((x-x_mean)² + (y-y_mean)²))`
- **Asymmetry**: Ratio of eigenvalues of position covariance
- **Kurtosis**: Fourth moment of displacement distribution
- **Fractal Dimension**: Box-counting dimension
- **Net Displacement**: Euclidean distance from start to end

#### Kinematic Features
- **Instantaneous Velocity**: Frame-to-frame displacement / time
- **Mean Velocity**: Average over entire track
- **MSD**: Mean squared displacement vs. time lag
- **Directional Persistence**: Autocorrelation of velocity vectors

#### Spatial Features
- **Nearest Neighbors**: Count within multiple radii
- **Local Density**: Particles per unit area
- **Cluster Membership**: Spatial clustering metrics

### SVM Classification

Machine learning-based mobility classification:

1. **Feature Selection**: NetDispl, Straight, Asymmetry, Rg, Kurtosis, FracDim
2. **Preprocessing**: Box-Cox transformation for normality
3. **Training**: SVM with RBF kernel and grid search
4. **Classification**: Mobile / Confined / Trapped

**Training Data Format:**
```csv
NetDispl,Straight,Asymmetry,radiusGyration,Kurtosis,fracDimension,Elected_Label
12.5,0.85,1.2,3.4,2.1,1.8,mobile
3.2,0.45,0.9,1.1,3.5,1.2,confined
0.8,0.15,0.6,0.5,5.2,1.05,trapped
```

### Autocorrelation Analysis

Directional persistence following Gorelik & Gautreau (2014):

1. Calculate velocity vectors for each track
2. Compute autocorrelation: `C(Δt) = <v(t) · v(t+Δt)> / <v²>`
3. Average across all tracks
4. Fit exponential decay: `C(Δt) = exp(-Δt/τ)`

**Interpretation:**
- **τ (persistence time)**: How long directionality is maintained
- **Rapid decay**: Random/confined motion
- **Slow decay**: Persistent/directed motion

---

## Output Files

### File Naming Convention

All output files use the input filename as prefix:

```
input_file.tif → input_file_[suffix].csv
```

### Output File Types

#### 1. Detection Results (`_locsID.csv`)
Particle localizations from U-Track detection:
```csv
frame,x [nm],y [nm],intensity [photon],id
1,5432.1,3210.5,850.3,1
1,6543.2,4321.6,920.7,2
2,5445.3,3215.8,835.2,1
...
```

#### 2. Basic Tracks (`_tracks.csv`)
Core trajectory information:
```csv
Experiment,TrackID,frame,x [nm],y [nm],intensity [photon]
Exp1,1,1,5432.1,3210.5,850.3
Exp1,1,2,5445.3,3215.8,835.2
...
```

#### 3. Features Only (`_features.csv`)
Analysis features without raw positions:
- All geometric features
- All kinematic features
- Classification results (if enabled)
- Spatial statistics (if enabled)

#### 4. Complete Analysis (`_enhanced_analysis.csv`)
Comprehensive results including:
- All track positions
- All calculated features
- Metadata columns
- Analysis timestamps

**Column Categories:**
- **Metadata**: Experiment, TrackID, timestamp
- **Positions**: x [nm], y [nm], frame
- **Geometric**: Rg, asymmetry, NetDispl, straightness, etc.
- **Kinematic**: velocities, MSD, diffusion coefficients
- **Spatial**: nearest neighbors, local density
- **Classification**: SVM predictions, probabilities

#### 5. Autocorrelation Data
- `_autocorr_average.csv`: Mean autocorrelation curve with SEM
- `_autocorr_individual.csv`: Per-track autocorrelation data

#### 6. Autocorrelation Plot (`_autocorr_plot.png`)
Publication-ready visualization with:
- Mean decay curve with error bars
- Exponential fit (if applicable)
- Persistence time annotation
- Formatted axes and legend

#### 7. Analysis Metadata (`_analysis_metadata.json`)
Complete parameter documentation:
```json
{
  "analysis_timestamp": "2025-01-20T15:30:45",
  "plugin_version": "2025.08.26",
  "parameters": { ... },
  "enabled_analyses": [ ... ],
  "output_columns": [ ... ],
  "linking_method": "built-in",
  "statistics": {
    "total_tracks": 1523,
    "total_detections": 45690,
    "processing_time": 125.3
  }
}
```

---

## Configuration

### Configuration Files

The plugin supports JSON configuration files for reproducible analysis:

#### Example Configuration

```json
{
  "pixel_size": 108,
  "frame_length": 0.1,
  "min_track_segments": 4,
  "max_gap_frames": 36,
  "max_link_distance": 3,
  "nn_radii": [3, 5, 10, 20, 30],
  "rg_mobility_threshold": 2.11,
  "experiment_name": "PIEZO1_WT_Control",
  
  "enable_nearest_neighbors": true,
  "enable_svm_classification": false,
  "enable_velocity_analysis": true,
  "enable_diffusion_analysis": true,
  "enable_background_subtraction": false,
  "enable_localization_error": true,
  "enable_straightness_analysis": true,
  "save_intermediate": true
}
```

### Expert Configurations

Pre-optimized settings for common experimental types:

#### Fast Membrane Proteins
```json
{
  "pixel_size": 108,
  "frame_length": 0.01,
  "max_gap_frames": 10,
  "max_link_distance": 5,
  "min_track_segments": 5,
  "nn_radii": [2, 3, 5, 8, 12],
  "rg_mobility_threshold": 2.5
}
```

#### Slow/Confined Proteins
```json
{
  "pixel_size": 108,
  "frame_length": 0.1,
  "max_gap_frames": 5,
  "max_link_distance": 2,
  "min_track_segments": 10,
  "nn_radii": [1, 2, 3, 5, 8],
  "rg_mobility_threshold": 1.8
}
```

#### Vesicle Trafficking
```json
{
  "pixel_size": 160,
  "frame_length": 0.05,
  "max_gap_frames": 20,
  "max_link_distance": 8,
  "min_track_segments": 3,
  "rg_mobility_threshold": 3.0
}
```

#### Single Molecule Tracking
```json
{
  "pixel_size": 120,
  "frame_length": 0.02,
  "max_gap_frames": 3,
  "max_link_distance": 4,
  "min_track_segments": 6,
  "enable_background_subtraction": true,
  "enable_localization_error": true
}
```

### Loading Configurations

```python
# In plugin interface:
# Files Tab → Load Configuration → Select .json file
# Or drag-and-drop JSON file onto plugin window
```

### Saving Configurations

```python
# Configure parameters in interface
# Files Tab → Save Configuration → Choose location
# Automatically includes all current settings
```

---

## Troubleshooting

### Common Issues

#### 1. Recursion Error During Linking

**Symptom**: `RecursionError: maximum recursion depth exceeded`

**Solution**:
```json
{
  "max_gap_frames": 10,
  "max_link_distance": 2,
  "min_track_segments": 5
}
```

**Explanation**: Reduce gap closing parameters to limit search complexity.

#### 2. Memory Error with Large Datasets

**Symptom**: `MemoryError` or system freeze

**Solution**:
```json
{
  "nn_radii": [3, 10],
  "enable_svm_classification": false,
  "enable_velocity_analysis": false,
  "save_intermediate": false
}
```

**Explanation**: Reduce memory-intensive analyses or process fewer files per batch.

#### 3. Poor Particle Linking

**Symptom**: Fragmented tracks, incorrect connections

**Solution**:
```json
{
  "max_gap_frames": 5,
  "max_link_distance": 1.5,
  "min_track_segments": 8
}
```

**Explanation**: Use stricter linking criteria for cleaner trajectories.

#### 4. SVM Classification Fails

**Symptom**: `ValueError: could not convert string to float` or classification errors

**Solution**:
- Verify training data format matches requirements
- Ensure `Elected_Label` column contains only: "mobile", "confined", "trapped"
- Check for missing values or non-numeric features
- Disable SVM if training data unavailable

#### 5. Detection Produces No Results

**Symptom**: Empty localization files

**Solutions**:
- Increase `alpha_threshold` (less stringent, e.g., 0.1)
- Decrease `min_intensity` threshold
- Verify PSF sigma matches your microscope setup
- Check image bit depth and scaling

#### 6. Autocorrelation Analysis Fails

**Symptom**: No output plots or errors during Phase 2

**Solution**:
- Ensure tracks are long enough (min 10 frames recommended)
- Check that tracks were successfully generated in Phase 1
- Verify output directory has write permissions

### Performance Optimization

#### For Speed
```json
{
  "min_track_segments": 3,
  "max_gap_frames": 10,
  "nn_radii": [3, 5, 10],
  "enable_nearest_neighbors": true,
  "enable_svm_classification": false,
  "enable_velocity_analysis": false,
  "enable_diffusion_analysis": false
}
```

#### For Quality
```json
{
  "min_track_segments": 10,
  "max_gap_frames": 50,
  "nn_radii": [1, 2, 3, 5, 8, 12, 20, 30],
  "enable_svm_classification": true,
  "save_intermediate": true
}
```

### Debug Logging

Enable detailed logging for troubleshooting:

```python
# In config.py or through plugin interface:
{
  "logging": {
    "console_level": "DEBUG",
    "file_level": "DEBUG",
    "log_to_file": true
  }
}

# Check logs in: ~/.FLIKA/logs/pytrack/
```

### Getting Help

1. Check Progress Tab for error messages
2. Review log files in `~/.FLIKA/logs/`
3. Verify input file formats match specifications
4. Test with example data before processing full datasets
5. Report issues with:
   - Plugin version
   - Python/FLIKA version
   - Input file examples
   - Full error messages
   - Configuration used

---

## Citation

If you use this plugin in your research, please cite:

```bibtex
@software{spt_batch_analysis,
  author = {Dickinson, George},
  title = {SPT Batch Analysis Plugin for FLIKA},
  year = {2025},
  version = {2025.08.26},
  url = {https://github.com/username/spt_batch_analysis}
}
```

### Related Publications

This plugin was developed for and used in studies of:
- PIEZO1 mechanosensitive channel dynamics
- Mechanotransduction in neural stem cells
- Cell edge dynamics and protein localization

**Key Methodologies:**
- U-Track detection: Jaqaman et al. (2008) Nature Methods
- U-Track linking: Jaqaman et al. (2008) Nature Methods
- Autocorrelation: Gorelik & Gautreau (2014) Methods Cell Biol
- SVM classification: Golan & Sherman (2017) Biophys J
- Trackpy: Crocker & Grier (1996) J Colloid Interface Sci

---

## Development

### Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Code Structure

```
spt_batch_analysis/
├── __init__.py              # Main plugin interface
├── config.py                # Configuration management
├── logging_setup.py         # Logging infrastructure
├── utrack_linking.py        # Particle linking algorithms
├── detection/               # Particle detection modules
├── analysis/                # Feature calculation
├── classification/          # Machine learning
├── visualization/           # Plotting utilities
└── utils/                   # Helper functions
```

### Testing

```python
# Run test suite:
python -m pytest tests/

# Run specific test:
python -m pytest tests/test_linking.py -v

# Generate coverage report:
python -m pytest --cov=spt_batch_analysis tests/
```

### Dependencies

**Core Requirements:**
- numpy >= 1.19.0
- pandas >= 1.2.0
- scipy >= 1.6.0
- scikit-learn >= 0.24.0
- scikit-image >= 0.18.0

**Optional Requirements:**
- trackpy >= 0.5.0 (for trackpy linking)
- tqdm >= 4.60.0 (for progress bars)
- matplotlib >= 3.3.0 (for visualization)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2025 George Dickinson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgments

- **Dr. Medha Pathak's Lab** at UC Irvine for supporting development
- **FLIKA Development Team** for the excellent image analysis platform
- **U-Track Team** for particle detection and linking algorithms
- **Trackpy Contributors** for the Crocker-Grier implementation
- **Scientific Community** for methodology development and validation

---

## Contact

**George Dickinson**  
Researcher, Dr. Medha Pathak's Lab  
UC Irvine  

For questions, bug reports, or feature requests:
- GitHub Issues: [link to repository]
- Email: [your email]

---

## Changelog

### Version 2025.08.26 (Current)
- Comprehensive batch processing with error handling
- Multiple particle linking algorithms (Built-in, Trackpy, U-Track)
- SVM classification with custom training support
- Autocorrelation analysis (Gorelik & Gautreau methodology)
- Geometric analysis suite
- Export control with selective column output
- Metadata export for reproducibility
- Enhanced logging and debugging

### Version 2025.01.20
- Initial public release
- U-Track detection integration
- Basic feature calculation
- Nearest neighbor analysis
- Configuration file support

---

## Appendix

### A. Feature Definitions

| Feature | Formula | Units | Interpretation |
|---------|---------|-------|----------------|
| Radius of Gyration | `sqrt(mean((x-x̄)² + (y-ȳ)²))` | nm | Spatial extent of track |
| Net Displacement | `sqrt((x_end - x_start)² + (y_end - y_start)²)` | nm | Start-to-end distance |
| Straightness | `NetDispl / PathLength` | - | 0=circular, 1=straight |
| Asymmetry | `λ_major / λ_minor` | - | Elongation of track |
| Fractal Dimension | Box-counting method | - | Path complexity |
| Mean Velocity | `PathLength / TotalTime` | nm/s | Average speed |
| Diffusion Coefficient | `slope(MSD) / 4` | μm²/s | Mobility measure |

### B. Parameter Recommendations by Imaging Setup

| Setup | Pixel Size | Frame Rate | Max Link Dist | Max Gap |
|-------|------------|------------|---------------|---------|
| TIRF Super-res | 100-120 nm | 10-100 Hz | 2-4 pixels | 3-10 frames |
| TIRF Standard | 150-200 nm | 1-10 Hz | 3-6 pixels | 10-36 frames |
| Confocal | 60-100 nm | 0.5-5 Hz | 3-8 pixels | 5-20 frames |
| Widefield | 100-200 nm | 1-30 Hz | 2-5 pixels | 5-15 frames |

### C. Recommended Analysis Combinations

**For Spatial Analysis:**
- Enable: Nearest Neighbors, Localization Error
- Optional: Background Subtraction
- Disable: Velocity, Diffusion, SVM

**For Motion Analysis:**
- Enable: Velocity, Diffusion, Straightness, Autocorrelation
- Optional: SVM Classification
- Disable: Nearest Neighbors

**For Classification:**
- Enable: All geometric features, SVM Classification
- Require: Training data with labeled examples
- Optional: Velocity, Straightness

**For Publication:**
- Enable: All analyses
- Save intermediate files
- Export metadata
- Generate all visualizations

---

**Last Updated:** December 8, 2025  
**Documentation Version:** 1.0
