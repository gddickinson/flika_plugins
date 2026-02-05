# Tracking Results Plotter - FLIKA Plugin

A comprehensive FLIKA plugin for visualizing and analyzing particle tracking results with professional-grade tools for overlay, filtering, and statistical analysis.

## 🎯 Features

- **📍 Interactive Overlays**: Overlay tracks and particles onto TIFF stacks with real-time updates
- **🎨 Smart Coloring**: Color-code tracks and points by any numerical property 
- **🔍 Advanced Filtering**: Multi-condition filtering with real-time preview
- **📈 Comprehensive Plotting**: Time traces, flower plots, histograms, scatter plots
- **📊 Track Statistics**: Automatic calculation and export of track summary statistics
- **🔄 Flexible Data Handling**: Automatic column detection for various CSV formats

## 📦 Installation

### 1. Download Plugin Files
Save the following files to your FLIKA plugins directory:
```
~/.FLIKA/plugins/tracking_results_plotter/
├── __init__.py
├── info.xml
├── about.html
└── README.md
```

### 2. Install Optional Dependencies
For enhanced plotting and statistical features:
```bash
pip install matplotlib scipy seaborn
```

### 3. Restart FLIKA
The plugin will appear in: **Plugins → Tracking Analysis → Launch Results Plotter**

## 📋 Data Format Requirements

The plugin automatically detects common column names:

### Required Columns
- **Track ID**: `track_number`, `track_id`, `id`, `particle`
- **Frame**: `frame`, `Frame`, `t`, `time`
- **X Coordinate**: `x`, `X`, `x_coord`, `position_x`
- **Y Coordinate**: `y`, `Y`, `y_coord`, `position_y`

### Optional Columns (Enhances Functionality)
- **Intensity**: `intensity`, `Intensity`, `signal`, `amp`
- **Experiment**: `Experiment`, `experiment`, `condition`, `sample`
- **Analysis Results**: Any numerical properties for filtering/coloring

### Example CSV Header
```csv
track_number,frame,x,y,intensity,radius_gyration,velocity,SVM,Experiment
1,0,125.2,136.1,1002,2.1,0.8,Mobile,Control
1,1,126.1,136.9,998,2.1,0.9,Mobile,Control
...
```

## 🚀 Quick Start Guide

### 1. Load Your Data
1. Launch: **Plugins → Tracking Analysis → Launch Results Plotter**
2. **Data Tab**: Click "Load CSV File" and select your tracking results
3. Verify column detection in the data information panel

### 2. Set Target Window
1. Open your TIFF stack in FLIKA
2. **Data Tab**: Select window in "Target Window" dropdown
3. Click "Set Active Window"

### 3. Configure Display
1. **Display Tab**: Adjust track/point appearance
2. Enable color coding by selecting a property
3. Choose colormap and size parameters

### 4. Apply Filters (Optional)
1. **Filters Tab**: Add filter conditions
   - Example: `velocity > 2.0` to show fast tracks
   - Example: `track_length > 50` for long tracks
2. Click "Apply Filters"

### 5. Visualize Results
1. Click **"Plot Overlays"** to display tracks and points
2. Navigate through frames to see overlays update
3. Use **Plots Tab** for additional analysis

## 📊 Analysis Features

### Track Overlays
- Connected line segments showing particle trajectories
- Color coding by any numerical property
- Customizable line width and colors
- Real-time updates during frame navigation

### Point Overlays  
- Frame-specific particle display
- Size and color customization
- Property-based coloring

### Statistical Analysis
- **Track Statistics**: Length, position variance, intensity stats
- **Histograms**: Property distributions with statistics overlay
- **Scatter Plots**: Correlation analysis between properties
- **Time Traces**: Intensity and position over time
- **Flower Plots**: Centered trajectory visualization

### Filtering System
- Multi-condition filtering (>, >=, <, <=, ==, !=)
- Real-time preview of filter effects
- Easy filter management interface

## 🎨 GUI Organization

### Tabbed Interface
- **📁 Data**: File loading, window selection, data preview
- **🎨 Display**: Appearance settings, color coding options
- **🔍 Filters**: Multi-condition filtering with preview
- **📊 Analysis**: Statistics calculation and track selection
- **📈 Plots**: Time traces, flower plots, histograms, scatter plots

### Control Buttons
- **Plot Overlays**: Display tracks and points on active window
- **Clear Overlays**: Remove all overlays from window
- **Update Display**: Apply current settings to visualization

## 💡 Usage Tips

### Data Preparation
- Ensure consistent track numbering across frames
- Include experiment/condition labels for comparative analysis
- Validate data integrity before visualization
- Use descriptive column names for better auto-detection

### Visualization Best Practices
- Use color coding to reveal patterns in data
- Filter data to focus on specific populations
- Combine multiple plot types for comprehensive analysis
- Export statistics for quantitative comparisons

### Performance Optimization
- Apply filters to reduce displayed track count
- Clear overlays before applying new settings
- Close unused plot windows to free memory
- Use auto-update sparingly with large datasets

## 🔧 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Data not loading | Missing required columns | Check column names match expected formats |
| Overlays not appearing | No active window set | Select FLIKA window in Data tab |
| Plots not generating | Matplotlib not installed | `pip install matplotlib` |
| Performance issues | Too many tracks displayed | Apply filters to reduce track count |

### Error Messages
- **"Missing required columns"**: Check CSV format and column names
- **"No window open"**: Open TIFF stack in FLIKA first
- **"No data loaded"**: Load CSV file before plotting
- **"Matplotlib not available"**: Install matplotlib for plotting features

## 🔬 Research Applications

### Ideal For
- **Single Particle Tracking**: Comprehensive SPT result visualization
- **Protein Dynamics**: Membrane protein mobility analysis
- **Diffusion Studies**: Flower plots and statistical analysis
- **Quality Control**: Track validation and data assessment
- **Publication Figures**: High-quality plots for manuscripts
- **Comparative Analysis**: Multi-condition experimental comparisons

### Compatible With
- SPT Batch Analysis plugin results
- TrackMate output (with proper formatting)
- Custom tracking algorithm results
- Any CSV with position and time data

## 📈 System Requirements

- **FLIKA**: Version 0.2.25 or higher
- **Python**: 3.7+ (included with FLIKA)
- **Memory**: 4GB+ RAM (8GB+ recommended for large datasets)
- **Dependencies**: NumPy, Pandas, QtPy (included with FLIKA)
- **Optional**: Matplotlib, SciPy, Seaborn (for enhanced features)

## 🤝 Contributing

Contributions welcome! The plugin uses a modular architecture:

- `TrackingDataManager`: Data loading and validation
- `TrackOverlay`: ROI creation and management  
- `TrackingResultsPlotter`: Main GUI and coordination

### Adding New Features
1. **New Plot Types**: Extend plotting methods in main class
2. **Custom Filters**: Add filter operators in `_apply_filters`
3. **Analysis Tools**: Implement in analysis tab methods
4. **Export Formats**: Extend export functionality

## 📄 License

This plugin is distributed under the MIT license. Free for academic and commercial use.

## 📚 Citation

When using this plugin in research:

```
Tracking Results Plotter: Comprehensive particle tracking visualization plugin for FLIKA
```

Also cite the base FLIKA platform:
```
FLIKA: A plugin-based image processing and analysis platform for biological research
```

## 📞 Support

- **Issues**: Report bugs with sample data and error messages
- **Feature Requests**: Suggest new visualization capabilities  
- **Documentation**: Help improve user guides and examples
- **Code**: Submit pull requests following modular architecture

---

**Transform your particle tracking data into insights with professional-grade visualization tools! 📊🎯📈**