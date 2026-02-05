# Flicker/Transient Detector
### For the Medha Pathak Lab, UC Irvine

## Overview

This FLIKA plugin provides automated detection and comprehensive characterization of transient intensity increases in fluorescence time series, specifically designed for analyzing **Ca²⁺ flickers** and **PIEZO1-mediated calcium signals**.

## Why This Plugin?

Your lab's research on PIEZO1 and Ca²⁺ flickers requires quantitative analysis of these rapid, localized events. This plugin automates the tedious process of manually identifying and measuring flickers, providing:

- **Objective, reproducible detection**
- **Comprehensive event characterization**
- **Statistical analysis across multiple ROIs**
- **Temporal pattern analysis**
- **Publication-ready visualizations**

## Key Features

### Detection Algorithms

**Four complementary methods for robust detection:**

1. **Threshold Method** (Recommended for most cases)
   - Simple ΔF/F₀ threshold crossing
   - Intuitive and fast
   - Best for: Clean signals with clear baseline

2. **Derivative Method**
   - Detects rapid intensity changes
   - Sensitive to sharp onset
   - Best for: Fast, sharp transients

3. **Z-score Method**
   - Statistical significance-based
   - Handles variable baselines
   - Best for: Noisy data with drift

4. **Wavelet Method**
   - Multi-scale analysis
   - Most sophisticated
   - Best for: Very noisy data or complex signals

### Event Characterization

For each detected event, the plugin measures:

- **Amplitude**: Peak height above baseline
- **ΔF/F₀**: Fractional fluorescence change (standard metric)
- **Duration**: Event length in seconds
- **Rise Time**: Time from 10% to 90% of peak
- **Decay Tau**: Exponential decay time constant
- **Area Under Curve (AUC)**: Total signal

### Analysis Capabilities

- **ROI-based Analysis**: Analyze multiple ROIs simultaneously
- **Regional Analysis**: Divide frame into regions to compare (e.g., leading edge vs. cell body)
- **Temporal Patterns**: Inter-event intervals, clustering, frequency
- **Statistical Comparisons**: Compare across ROIs, conditions, time periods
- **Raster Plots**: Visualize event timing across all ROIs

### Visualization & Export

- 9-panel comprehensive analysis dashboard
- Raster plots showing temporal distribution
- Event property distributions
- Statistical summaries
- Complete data export (CSV, JSON)

## Installation

1. Copy the `flicker_detector` folder to your FLIKA plugins directory:
   - Windows: `C:\Users\YourName\.FLIKA\plugins\`
   - Mac: `~/.FLIKA/plugins/`
   - Linux: `~/.FLIKA/plugins/`

2. Restart FLIKA

3. Access via: Plugins → Flicker/Transient Detector

## Required Dependencies

Standard scientific Python packages (usually included with FLIKA):
- numpy, scipy, pandas
- matplotlib, pyqtgraph
- scikit-learn (optional, for advanced features)

Install if needed:
```bash
pip install numpy scipy pandas matplotlib --break-system-packages
```

## Quick Start Guide

### Basic Workflow

1. **Prepare Your Data**
   - Load a calcium imaging time series in FLIKA
   - Should be 3D (time, x, y)
   - Apply preprocessing if needed (background subtraction, bleaching correction)

2. **Draw ROIs (Optional)**
   - Draw ROIs over regions of interest (cells, subcellular regions)
   - The plugin will analyze each ROI separately
   - Or use whole-frame analysis

3. **Open the Plugin**
   - Plugins → Flicker/Transient Detector
   - GUI will appear

4. **Set Detection Parameters**
   - **Detection Method**: Start with 'threshold' (easiest)
   - **Threshold**: 2.0 is a good starting point for ΔF/F₀
   - **Min Duration**: 3 frames (adjust based on expected flicker duration)
   - **Max Duration**: 100 frames (prevent detecting long slow changes)
   - **Frame Rate**: Enter your acquisition rate in Hz

5. **Run Detection**
   - Click "Detect Events"
   - Status bar shows progress
   - Results stored in memory

6. **Visualize Results**
   - Click "Visualize Results"
   - 9-panel analysis window appears
   - Review detected events on traces

7. **Fine-tune (if needed)**
   - Adjust threshold up (fewer events) or down (more events)
   - Change min/max duration filters
   - Try different detection method
   - Re-run detection

8. **Export Data**
   - Click "Export Data"
   - Saves multiple files:
     - `*_events.csv`: All event properties
     - `*_traces.csv`: Intensity traces
     - `*_summary.json`: Overall statistics
     - `*_roi_stats.csv`: Per-ROI statistics

## Parameter Guide

### Detection Method

**Threshold** (Recommended first try)
- Detects when signal exceeds baseline by threshold amount
- Threshold = ΔF/F₀ value (2.0 = 200% increase)
- Most intuitive, works well for clean data

**Derivative**
- Detects rapid increases (steep slopes)
- Threshold = rate of change
- Good for detecting onset of sharp events

**Z-score**
- Statistical significance (threshold = # of standard deviations)
- Threshold = 3.0 means 3 SD above mean
- Best for data with variable baselines

**Wavelet**
- Multi-scale pattern matching
- Threshold = wavelet coefficient magnitude
- Most sophisticated, best for noisy data

### Duration Filters

**Min Duration**: Minimum event length in frames
- Too low: Detect noise spikes
- Too high: Miss brief events
- Typical: 2-5 frames (0.2-0.5 s at 10 Hz)

**Max Duration**: Maximum event length in frames
- Prevents detecting slow baseline changes
- Typical: 50-200 frames (5-20 s at 10 Hz)

### Baseline Calculation

**Percentile** (Default)
- Uses low percentile of trace (e.g., 10th percentile)
- Robust to events
- Best for most cases

**Mean/Median**
- Simple average
- Affected by many events

**Rolling**
- Time-varying baseline
- Good for drifting baselines

### Smoothing

**Smoothing Sigma**: Gaussian filter parameter
- 0 = no smoothing
- 1.0 = mild smoothing (recommended)
- >2.0 = heavy smoothing (may miss brief events)

## Understanding the Output

### Main Analysis Window

**Panel 1: Example Traces**
- Shows first 3 ROI traces with detected events
- Red shading = detected events
- Verify detection quality visually

**Panel 2: Amplitude Distribution**
- Histogram of event amplitudes
- Check for biological variation
- Outliers may indicate artifacts

**Panel 3: Duration Distribution**
- Histogram of event durations
- Should match expected flicker duration
- Multiple peaks may indicate event types

**Panel 4: ΔF/F₀ Distribution**
- Standard metric for calcium imaging
- Compare across experiments
- Typical Ca²⁺ flickers: 0.5-5.0 ΔF/F₀

**Panel 5: Kinetics Scatter**
- Rise time vs. decay time
- Reveals event subpopulations
- Fast rise + slow decay = typical Ca²⁺ response

**Panel 6: Temporal Distribution**
- When events occur during recording
- Check for time-dependent changes
- Cluster = bursts of activity

**Panel 7: ROI Comparison**
- Event frequency per ROI
- Identify "hotspots"
- Statistical comparison

**Panel 8: Statistics Summary**
- Key numbers at a glance
- Copy for papers/presentations

**Panel 9: Properties Correlation**
- Amplitude vs. Duration
- Color = ΔF/F₀
- Check for relationships

### Raster Plot

- Each row = one ROI
- Each bar = one event
- Horizontal axis = time
- Shows temporal patterns across cell

### Key Metrics Explained

**Frequency (Hz)**: Events per second
- Higher = more active
- Compare across conditions

**Amplitude**: Peak height above baseline
- In same units as image intensity
- May vary with laser power, gain

**ΔF/F₀**: Normalized amplitude
- Independent of absolute brightness
- Standard for comparing across experiments
- Typical flickers: 0.5-5.0

**Duration**: Event width
- Biological property of event
- Ca²⁺ flickers typically 0.1-2 s

**Rise Time**: Onset speed
- Fast (~50-200 ms for Ca²⁺ flickers)
- Reflects channel opening

**Decay Tau**: Recovery speed
- Exponential time constant
- Reflects calcium clearance
- Typical: 0.5-2 s for flickers

**Inter-Event Interval (IEI)**: Time between events
- Mean IEI = 1 / frequency
- CV (coefficient of variation) indicates regularity
- Low CV = regular firing, High CV = irregular

**AUC**: Integrated signal
- Total calcium influx
- Useful for comparing event "strength"

## Research Applications

### 1. PIEZO1-Mediated Ca²⁺ Flickers

**Question**: How does mechanical stimulation affect flicker frequency?

**Workflow**:
1. Record baseline (no stimulation)
2. Apply mechanical stimulus (stretch, poking)
3. Record stimulated period
4. Analyze both periods separately
5. Compare frequencies, amplitudes

**Expected**: Increased frequency and/or amplitude with PIEZO1 activation

### 2. Spatial Distribution of Flickers

**Question**: Are flickers enriched at the leading edge?

**Workflow**:
1. Draw ROIs at leading edge and cell body
2. Run detection
3. Compare frequencies between ROIs
4. Use raster plot to visualize spatial patterns

**Expected**: Higher frequency at leading edge if PIEZO1 is enriched there

### 3. Drug Effects on Flicker Properties

**Question**: How does a PIEZO1 inhibitor affect flickers?

**Workflow**:
1. Analyze control cells
2. Analyze drug-treated cells
3. Compare event properties (especially frequency, amplitude)
4. Statistical test (t-test, ANOVA)

**Expected**: Decreased frequency/amplitude with inhibitor

### 4. Flicker Clustering Analysis

**Question**: Do flickers occur in bursts?

**Workflow**:
1. Analyze inter-event intervals
2. Check IEI distribution
3. Look for multiple peaks (indicates clustering)
4. Use raster plot to visualize bursts

**Interpretation**:
- Regular IEI = periodic firing
- Short IEI = bursts
- Exponential IEI = Poisson (random) process

### 5. Developmental Changes

**Question**: How do flicker properties change during differentiation?

**Workflow**:
1. Image cells at multiple time points
2. Analyze each time point
3. Track changes in frequency, amplitude, kinetics
4. Correlate with differentiation markers

**Expected**: Changes in flicker properties during fate transitions

### 6. Correlation with Cell Behavior

**Question**: Do flickers predict migration?

**Workflow**:
1. Detect flickers in migrating cells
2. Measure migration speed separately
3. Correlate flicker frequency with migration speed
4. Use regional analysis to relate edge flickers to protrusions

## Tips and Best Practices

### Data Quality

1. **Good Signal-to-Noise**
   - Essential for accurate detection
   - Use appropriate laser power and gain
   - Average frames if too noisy (reduces temporal resolution)

2. **Photobleaching Correction**
   - Correct bleaching before analysis
   - Otherwise baseline drifts down, affecting detection
   - Use FLIKA's bleaching correction or other tools

3. **Background Subtraction**
   - Remove background fluorescence
   - Improves ΔF/F₀ calculation
   - Apply before flicker detection

### Parameter Optimization

1. **Start Conservative**
   - Higher threshold = fewer false positives
   - Gradually lower until you get expected events
   - Check visually on traces

2. **Use Multiple ROIs for Validation**
   - Draw ROI on cell (should have events)
   - Draw ROI on background (should have few/no events)
   - If background has many events, threshold too low

3. **Verify with Manual Inspection**
   - Look at example traces
   - Are shaded regions really events?
   - Are some events missed?
   - Adjust parameters accordingly

4. **Method Selection**
   - Threshold: Start here, works 80% of the time
   - Z-score: If baseline varies a lot
   - Derivative: If events have sharp onset
   - Wavelet: If other methods fail due to noise

### Statistical Analysis

1. **N Numbers**
   - Analyze multiple cells per condition
   - Report both # of cells and # of events
   - Use per-cell averages for statistics

2. **Appropriate Tests**
   - Compare frequencies: t-test or Mann-Whitney
   - Compare amplitudes: Kolmogorov-Smirnov test
   - Multiple conditions: ANOVA

3. **Reporting**
   - Report detection parameters used
   - Show example traces with detected events
   - Include distributions, not just means
   - Report both mean and median (median more robust)

### Common Issues

**Too Many Events Detected**
- Threshold too low
- Check if detecting noise
- Increase threshold or min_duration

**Too Few Events Detected**
- Threshold too high
- Try different detection method
- Lower threshold gradually

**Events Look Wrong**
- Wrong baseline calculation
- Try different baseline method
- Check for photobleaching

**No Events Detected**
- No events present, or threshold way too high
- Check signal-to-noise
- Try zscore method with lower threshold (2.0)

**Detector Picks Up Slow Changes**
- Max duration too large
- Reduce max_duration parameter
- Use derivative method

## Combining with Other Analyses

### With PIEZO1 Cluster Analyzer

1. Run flicker detector on calcium channel
2. Run cluster analyzer on PIEZO1 channel
3. Compare spatial distributions
4. Do flickers occur where PIEZO1 clusters?

### With Kymograph

1. Create kymograph along cell edge
2. Detect events in kymograph
3. Analyze spatiotemporal patterns
4. Relate to membrane dynamics

### With Tracking

1. Track PIEZO1 puncta
2. Detect calcium flickers
3. Correlate puncta position with flicker location
4. Do flickers follow puncta movement?

## Example Analysis Script

For batch processing multiple movies:

```python
# In FLIKA/Python environment
from flika.process.file_ import open_file
from plugins.flicker_detector.flicker_transient_detector import flicker_transient_detector

# List of files
files = ['cell1.tif', 'cell2.tif', 'cell3.tif']

results_all = []

for filename in files:
    # Open movie
    win = open_file(filename)
    
    # Run detection
    results = flicker_transient_detector(
        win, 
        method='threshold',
        threshold=2.0,
        min_duration=3,
        framerate=10.0
    )
    
    results['filename'] = filename
    results_all.append(results)
    
    # Export
    flicker_transient_detector.export_data(filename.replace('.tif', ''))
    
# Compare across cells
frequencies = [r['frequency_hz'] for r in results_all]
print(f"Mean frequency: {np.mean(frequencies):.3f} ± {np.std(frequencies):.3f} Hz")
```

## Troubleshooting

### Plugin Won't Load
- Check all files are in correct directory
- Restart FLIKA completely
- Check FLIKA console for error messages

### Detection Fails
- Verify image is 3D time series
- Check framerate is set
- Try simplest method (threshold) first

### Results Look Wrong
- Visually inspect traces
- Start with very conservative parameters
- Compare to manual detection on subset

### Export Fails
- Check write permissions
- Ensure filename has no special characters
- Try different directory

## Updates and Support

For questions, issues, or feature requests:
- Contact Pathak Lab: https://www.pathaklab-uci.com/
- FLIKA documentation: https://flika-org.github.io/

## Citation

If you use this plugin in your research:

1. **FLIKA**: Ellefsen, K., Settle, B., Parker, I. & Smith, I. An algorithm for automated detection, localization and measurement of local calcium signals from camera-based imaging. Cell Calcium 56:147-156, 2014

2. **Your PIEZO1 work**: Cite relevant Pathak lab papers

## Version History

**v1.0.0** (2025)
- Initial release
- Four detection methods
- Comprehensive event characterization
- ROI and regional analysis
- Statistical analysis and export
- Raster plots

---

**Created by Claude for the Medha Pathak Lab, UC Irvine**

*Designed to advance understanding of PIEZO1 mechanotransduction and calcium signaling*
