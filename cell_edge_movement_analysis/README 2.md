# Cell Edge Movement Analysis - FLIKA Plugin

A comprehensive FLIKA plugin for analyzing the relationship between cell edge movement and PIEZO1 mechanosensitive channel protein intensity using advanced image processing techniques.

## 🎯 Overview

This plugin provides a complete workflow for:
- Loading and preprocessing TIRF microscopy images and cell masks
- Interactive mask editing with professional drawing tools
- Analyzing cell edge movement using vertical displacement tracking
- Correlating movement with PIEZO1 protein intensity
- Real-time visualization of analysis progress
- Comprehensive statistical analysis and export

## 📦 Installation

### Method 1: FLIKA Plugin Directory

1. Copy the entire `cell_edge_movement_analysis` folder to your FLIKA plugins directory:
   ```
   ~/.FLIKA/plugins/cell_edge_movement_analysis/
   ```

2. Directory structure:
   ```
   cell_edge_movement_analysis/
   ├── __init__.py          # Main plugin file
   ├── analysis_core.py     # Core analysis functions
   ├── info.xml             # Plugin metadata
   ├── about.html           # Plugin information
   └── README.md            # This file
   ```

3. Restart FLIKA

4. The plugin will appear in: **Plugins → Cell Analysis → Edge Movement Analysis**

### Method 2: Manual Launch

From FLIKA's Python console:
```python
import sys
sys.path.append('/path/to/cell_edge_movement_analysis')
from cell_edge_movement_analysis import launch_plugin
launch_plugin()
```

## 🔧 Requirements

### Core Dependencies (Required)
- **FLIKA**: 0.2.25 or higher
- **Python**: 3.7+
- **NumPy**: For numerical operations
- **Pandas**: For data management
- **SciPy**: For statistical analysis
- **scikit-image**: For image processing
- **QtPy**: For GUI (included with FLIKA)
- **PyQtGraph**: For visualization (included with FLIKA)

### Optional Dependencies (Enhanced Features)
- **Matplotlib**: For enhanced plotting and figure export
  ```bash
  pip install matplotlib
  ```

## 🚀 Quick Start Guide

### 1. Prepare Your Data

Load your data into FLIKA:
1. Open your PIEZO1 TIRF microscopy image stack
2. Open your corresponding cell mask stack
3. Verify both stacks have the same dimensions (frames × height × width)

### 2. Launch the Plugin

**Plugins → Cell Analysis → Edge Movement Analysis**

### 3. Setup Tab - Select Windows

1. Choose your PIEZO1 signal window from the dropdown
2. Choose your cell mask window from the dropdown
3. Verify the window information shows compatible dimensions

### 4. Edit Masks (Optional)

Use the integrated mask editor:
- **Draw**: Click and drag to add to mask
- **Erase**: Remove parts of mask
- **Fill**: Fill entire frame with mask
- **Clear**: Remove entire mask from frame
- **Brush Size**: Adjust drawing width (1-50 pixels)
- **Copy to Next**: Duplicate current mask to next frame
- **Undo/Redo**: Fix mistakes

**Save your modifications** when complete!

### 5. Configuration Tab - Set Parameters

#### Sampling Parameters
- **Number of points** (default: 12): How many points to sample along the cell edge
- **Sampling depth** (default: 200 px): How deep into the cell to sample
- **Sampling width** (default: 75 px): Width of each sampling region
- **Minimum coverage** (default: 0.8): Minimum fraction of sampling region inside cell
- **Try 180° rotation**: Attempt rotated placement if initial fails
- **Exclude endpoints**: Skip the very edges of the cell

#### Movement Detection
- **Movement threshold** (default: 0.1 px): Minimum displacement to classify as moving
- **Min valid pixels** (default: 5): Minimum valid displacements for scoring
- **Temporal direction**: 
  - *future*: Intensity at N predicts movement N→N+1
  - *past*: Movement N-1→N correlates with intensity at N

#### Output Options
- Save detailed CSV with point-by-point data
- Save summary JSON with statistics
- Save correlation plots
- Save frame-by-frame visualizations

**Save/Load configurations** for reproducibility!

### 6. Run Analysis Tab - Execute Analysis

1. Set output directory (or use default)
2. Click **▶️ Start Analysis**
3. Monitor progress in the log window
4. Analysis runs in background thread - FLIKA remains responsive

### 7. Results Tab - Visualize & Export

- **Frame Navigation**: Use slider or buttons to navigate
- **Overlays**: Toggle cell edge, sampling regions, movement vectors
- **Frame Info**: View current frame statistics
- **Summary Stats**: Overall analysis results
- **Export**: Save results CSV and generate plots

## 📊 Analysis Method

### Vertical Displacement X-Axis Full

The plugin implements a sophisticated edge tracking method:

1. **Edge Detection**: Detect cell boundaries from binary masks using contour finding
2. **X-Position Processing**: Process every x-pixel position across the cell
3. **Uppermost Edge Tracking**: Find minimum y-coordinate at each x-position
4. **Displacement Calculation**: Calculate vertical displacement between frames
   - Negative displacement = Cell extending (moving up)
   - Positive displacement = Cell retracting (moving down)
5. **Movement Classification**: 
   - *Extending*: movement_score < -threshold
   - *Retracting*: movement_score > threshold
   - *Stable*: |movement_score| ≤ threshold
6. **Intensity Sampling**: Sample PIEZO1 intensity at regular intervals along edge
7. **Correlation Analysis**: Correlate local movement with local intensity

## 🎨 GUI Organization

### Tab 1: Setup & Mask Editor
- Window selection dropdowns
- Window compatibility checking
- Interactive mask editing tools
- Brush size control
- Undo/redo functionality
- Save modified masks

### Tab 2: Analysis Configuration
- Sampling parameters
- Movement detection settings
- Temporal direction selection
- Output options
- Save/load configuration profiles
- Reset to defaults

### Tab 3: Run Analysis
- Output directory selection
- Start/stop analysis buttons
- Real-time progress bar
- Detailed analysis log
- Background processing

### Tab 4: Results & Visualization
- Frame navigation slider
- Overlay controls
- Current frame statistics
- Summary statistics display
- Export results and plots

## 📈 Understanding Results

### Movement Classification

- **Extending**: Cell edge moving outward (up in image coordinates)
  - Indicates protrusive activity
  - Movement score < -threshold (default: -0.1)
  
- **Retracting**: Cell edge moving inward (down in image coordinates)
  - Indicates retraction
  - Movement score > threshold (default: 0.1)
  
- **Stable**: Cell edge relatively stationary
  - Minimal movement detected
  - |movement_score| ≤ threshold

### Output Files

#### detailed_movement_results.csv
Point-by-point data for every frame:
- Frame index
- Transition index
- Sampling point coordinates
- Local movement scores
- Intensity values
- Validity flags
- Point status
- Frame-level movement type

#### movement_summary_statistics.json
Comprehensive analysis summary:
- Total frames and transitions processed
- Valid measurement percentages
- Movement distribution (extending/retracting/stable)
- Correlation analysis (R², p-value, slope)
- Configuration parameters used

#### Visualization Plots (if enabled)
- `movement_intensity_correlation.png`: Scatter plot with regression
- `movement_summary.png`: Movement scores over time
- `frame_*_plots/`: Individual frame visualizations

## 🔬 Temporal Directions Explained

### Future Direction (Default)
```
Frame N intensity → Predicts → Movement from N to N+1
```
**Use when**: Testing if current PIEZO1 localization influences subsequent cell movement

**Biological interpretation**: Higher PIEZO1 intensity at time N may predict extension at N+1

### Past Direction
```
Movement from N-1 to N → Correlates with → Frame N intensity
```
**Use when**: Testing if recent cell movement affects current PIEZO1 localization

**Biological interpretation**: Extension from N-1 to N may recruit PIEZO1 at time N

## 💡 Tips & Best Practices

### Data Preparation
- Ensure masks accurately represent cell boundaries
- Use the mask editor to fix any errors before analysis
- Verify PIEZO1 signal quality and background levels
- Check for photobleaching and correct if necessary

### Configuration
- Start with default parameters for initial analysis
- Adjust `n_points` based on cell size (larger cells → more points)
- Increase `depth` if you need to sample deeper into the cell
- Lower `min_cell_coverage` if sampling regions frequently fail
- Use `exclude_endpoints` to avoid edge artifacts

### Performance
- Analysis runs in background - you can continue using FLIKA
- Large stacks (>100 frames) may take several minutes
- Disable frame plot saving for faster processing
- Results are cached - navigate frames instantly after analysis

### Interpretation
- Check correlation p-value for statistical significance (typically p < 0.05)
- R² indicates variance explained (0-1 scale)
- Positive slope: Higher intensity → more retraction
- Negative slope: Higher intensity → more extension

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No windows in dropdown | No FLIKA windows open | Open image stacks first, then refresh |
| Dimension mismatch warning | Image and mask sizes differ | Ensure same dimensions, or proceed with caution |
| Few valid sampling points | Incorrect mask or tight parameters | Edit masks, lower min_cell_coverage |
| No movement detected | Stable cell or threshold too high | Lower movement_threshold, check masks |
| Analysis crashes | Missing dependencies | Install required packages |
| Slow performance | Large dataset | Disable frame plots, reduce n_points |

### Error Messages

**"No cell edge detected"**
- Mask is empty or all black
- Solution: Check mask window, use mask editor

**"Missing windows"**
- Windows not selected in Setup tab
- Solution: Select both PIEZO1 and mask windows

**"Matplotlib not available"**
- Matplotlib not installed
- Solution: `pip install matplotlib`

## 🔍 Research Applications

### Ideal For
- **Mechanobiology**: PIEZO1 localization during cell spreading/migration
- **Cell Migration**: Edge dynamics and molecular recruitme

nt
- **Live Cell Imaging**: Real-time dynamics of membrane proteins
- **Drug Studies**: Effects of compounds on edge movement
- **Comparative Analysis**: Different cell types or conditions

### Compatible With
- TIRF microscopy data
- Confocal microscopy (if edge is visible)
- Any fluorescence microscopy showing membrane proteins
- Binary masks from any segmentation method

## 📚 Citation

When using this plugin in publications:

```
Cell Edge Movement Analysis Plugin for FLIKA
George Dickinson
UC Irvine, Dr. Medha Pathak's Lab
Version 1.0.0, 2025
```

Also cite FLIKA:
```
Ellefsen, K., Settle, B., Parker, I. & Smith, I. F. 
An algorithm for automated detection, localization and measurement 
of local calcium signals from camera-based imaging. 
Cell Calcium 56:147-156, 2014
```

## 🤝 Contributing

### Bug Reports
- Provide sample data if possible
- Include error messages and screenshots
- Describe steps to reproduce

### Feature Requests
- Describe the desired functionality
- Explain the scientific use case
- Suggest implementation approach if possible

### Code Contributions
The plugin uses a modular architecture:
- `__init__.py`: GUI and main plugin class
- `analysis_core.py`: Core analysis functions
- Easy to extend with new analysis methods or visualizations

## 📄 License

This plugin is distributed under the MIT License. Free for academic and commercial use.

## 🙏 Acknowledgments

Developed in **Dr. Medha Pathak's Lab** at **UC Irvine**

Special thanks to the FLIKA development team for creating an excellent platform for biological image analysis.

---

**Transform your cell edge dynamics research with advanced quantitative analysis! 🔬📊🎯**
