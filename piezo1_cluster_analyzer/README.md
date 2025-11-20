# PIEZO1 Cluster and Spatial Distribution Analyzer
### For the Medha Pathak Lab, UC Irvine

## Overview

This FLIKA plugin provides comprehensive spatial and temporal analysis of PIEZO1 mechanosensitive channel localization and clustering dynamics. It's specifically designed for analyzing TIRF microscopy data of fluorescently-labeled PIEZO1 proteins.

## Key Features

### Spatial Analysis
- **Automatic Puncta Detection**: Multiple thresholding methods (Otsu, adaptive, manual)
- **Cluster Detection**: DBSCAN-based clustering algorithm
- **Spatial Statistics**: 
  - Nearest neighbor analysis
  - Ripley's K-function for pattern classification
  - Density mapping and hotspot detection
  - Spatial distribution classification (clustered, random, or dispersed)

### Temporal Analysis
- Track puncta and cluster dynamics over time
- Measure clustering coefficient evolution
- Quantify temporal trends in spatial organization
- Analyze formation and dissolution of clusters

### Regional Analysis
- Compare spatial patterns across different cellular regions
- Useful for analyzing leading edge vs. cell body
- Identify spatial heterogeneity

### Visualization
- Comprehensive multi-panel analysis plots
- Density heatmaps
- Cluster overlay visualizations
- Temporal evolution plots
- Regional comparison plots

### Export Capabilities
- Puncta coordinates (CSV)
- Cluster properties (CSV)
- Temporal dynamics (CSV)
- Summary statistics (JSON)
- Publication-ready figures

## Installation

1. Copy the `piezo1_cluster_analyzer` folder to your FLIKA plugins directory:
   - Windows: `C:\Users\YourName\.FLIKA\plugins\`
   - Mac: `~/.FLIKA/plugins/`
   - Linux: `~/.FLIKA/plugins/`

2. The folder should contain:
   - `piezo1_cluster_analyzer.py`
   - `__init__.py`
   - `info.xml`
   - `about.html`
   - `README.md`

3. Restart FLIKA

4. The plugin will appear in the FLIKA menu under "Plugins" → "PIEZO1 Cluster Analyzer"

## Required Dependencies

The plugin requires the following Python packages (usually already included with FLIKA):
- numpy
- scipy
- scikit-learn
- scikit-image
- pandas
- matplotlib
- pyqtgraph
- qtpy

If any are missing, install via:
```bash
pip install numpy scipy scikit-learn scikit-image pandas matplotlib --break-system-packages
```

## Usage

### Basic Workflow

1. **Load Your Data**
   - Open your PIEZO1 fluorescence image in FLIKA (2D or 3D time series)
   - The image should show puncta/spots representing PIEZO1 channels

2. **Open the Plugin**
   - Go to Plugins → PIEZO1 Cluster Analyzer
   - The analysis GUI will appear

3. **Configure Detection Parameters**
   - **Threshold Method**: Choose detection method
     - `otsu`: Automatic global thresholding (recommended for most cases)
     - `adaptive`: Local adaptive thresholding (better for uneven illumination)
     - `manual`: Uses 70th percentile as threshold
   - **Min/Max Puncta Size**: Filter puncta by size (in pixels)
     - Typical values: Min=3, Max=100
     - Adjust based on your pixel size and expected PIEZO1 cluster size

4. **Configure Clustering Parameters**
   - **Cluster Distance**: Maximum distance between puncta to form cluster (pixels)
     - Start with 20 pixels and adjust based on results
     - Larger values = more permissive clustering
   - **Min Cluster Size**: Minimum number of puncta to define a cluster
     - Typical value: 3-5 puncta

5. **Temporal Analysis** (for time series)
   - Check "Analyze Temporal" to track dynamics over time
   - Set frame range if you want to analyze a subset

6. **Run Analysis**
   - Click "Run Analysis" button
   - Status bar will show progress
   - Results are stored in memory

7. **Visualize Results**
   - Click "Visualize Results" to see comprehensive analysis plots
   - The plot window includes 9 panels:
     1. Spatial distribution with clusters
     2. Density heatmap
     3. Nearest neighbor distance distribution
     4. Cluster size distribution
     5. Statistics summary
     6. Temporal evolution (if time series)
     7. Clustering coefficient dynamics
     8. Cluster size dynamics
     9. Ripley's K evolution or regional comparison

8. **Regional Analysis** (optional)
   - Click "Analyze Regions" to compare spatial patterns across cellular regions
   - Divides image into 4 quadrants by default
   - Useful for comparing leading edge vs. trailing edge

9. **Export Data**
   - Click "Export Data"
   - Choose filename and location
   - Exports multiple files:
     - `*_puncta.csv`: All puncta coordinates and properties
     - `*_clusters.csv`: Cluster properties
     - `*_temporal.csv`: Time series data (if applicable)
     - `*_summary.json`: Overall statistics

## Understanding the Results

### Key Metrics

**Nearest Neighbor (NN) Distance**
- Mean distance between each punctum and its closest neighbor
- Lower values = more clustering
- NN Distance Ratio compares to random distribution
  - Ratio < 1: More clustered than random
  - Ratio ≈ 1: Random distribution
  - Ratio > 1: More dispersed than random

**Clustering Coefficient**
- Proportion of puncta pairs within the cluster distance
- Range: 0 (all far apart) to 1 (all close together)
- Higher values indicate more clustering

**Ripley's K-function (L-value)**
- Statistical test for spatial pattern
- L > 0: Clustered
- L ≈ 0: Random
- L < 0: Regular/dispersed
- Values > 2 or < -2 are statistically significant

**Distribution Type**
- Automatic classification based on Ripley's L:
  - "Highly Clustered" (L > 2)
  - "Clustered" (0 < L < 2)
  - "Random" (-2 < L < 0)
  - "Regular/Dispersed" (L < -2)

**Fraction Clustered**
- Proportion of puncta that belong to clusters
- Range: 0 (no clustering) to 1 (all puncta in clusters)

### Temporal Metrics (Time Series)

- **Puncta Count Dynamics**: Track formation/disappearance of PIEZO1 puncta
- **Cluster Formation**: Watch clusters form and dissolve over time
- **Clustering Trends**: Positive slope = increasing clustering over time

## Tips and Best Practices

### Image Preparation
1. **Background Subtraction**: Apply background subtraction before analysis if needed
2. **Bleaching Correction**: For time series, correct photobleaching if significant
3. **Image Quality**: Ensure good signal-to-noise ratio for accurate detection

### Parameter Tuning
1. **Start Conservative**: Begin with smaller puncta size and larger cluster distance
2. **Iterate**: Run analysis, visualize, adjust parameters, repeat
3. **Validate**: Manually check a few frames to ensure detection is accurate
4. **Document**: Record your final parameters for reproducibility

### Biological Interpretation
1. **Context Matters**: Interpret results in the context of your experimental conditions
2. **Controls**: Compare to control conditions or unstimulated cells
3. **Statistics**: Use multiple cells/experiments for statistical analysis
4. **Time Course**: For mechanotransduction studies, relate clustering to stimulation timing

## Research Applications

This plugin is designed for studying:

1. **PIEZO1 Localization Dynamics**
   - How PIEZO1 distribution changes during mechanotransduction
   - Cluster formation in response to mechanical stimulation
   - Subcellular targeting to leading edges or focal adhesions

2. **Neural Stem Cell Mechanobiology**
   - PIEZO1 organization during differentiation
   - Matrix stiffness effects on PIEZO1 clustering
   - Correlation with calcium flicker activity

3. **Wound Healing and Migration**
   - PIEZO1 enrichment at leading edge
   - Dynamic reorganization during cell movement
   - Differences between leader and follower cells

4. **Membrane Protein Organization**
   - General tool for analyzing membrane protein clustering
   - Can be adapted for other mechanosensitive channels
   - Applicable to various punctate signals

## Troubleshooting

### No Puncta Detected
- Lower the min puncta size
- Try different threshold method
- Check if image needs preprocessing (background subtraction)
- Verify image has good contrast

### Too Many False Positives
- Increase min puncta size
- Reduce max puncta size
- Use adaptive thresholding for uneven illumination

### No Clusters Detected
- Increase cluster distance parameter
- Reduce min cluster size
- Verify that puncta are actually clustering (check spatial plot)

### Analysis Too Slow
- Reduce frame range for initial testing
- Use temporal binning in preprocessing
- Process smaller regions of interest

### Memory Issues
- Process fewer frames at a time
- Reduce image size if possible
- Close other FLIKA windows

## Citing This Plugin

If you use this plugin in your research, please cite:

1. **FLIKA**: Ellefsen, K., Settle, B., Parker, I. & Smith, I. An algorithm for automated detection, localization and measurement of local calcium signals from camera-based imaging. Cell Calcium 56:147-156, 2014

2. **Pathak Lab Research**: Cite relevant papers on PIEZO1 localization dynamics from the Pathak lab

## Support and Contact

For questions, bug reports, or feature requests:
- Contact the Pathak Lab: https://www.pathaklab-uci.com/
- FLIKA documentation: https://flika-org.github.io/

## Version History

**v1.0.0** (2025)
- Initial release
- Core spatial analysis features
- Temporal tracking
- Regional analysis
- Export capabilities

## License

This plugin is provided as-is for research use by the Medha Pathak Lab and collaborators.

---

**Developed by Claude for the Medha Pathak Lab, UC Irvine**
