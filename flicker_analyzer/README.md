# Flicker Analyzer
### Automated Ca²⁺ Flicker and Transient Event Detection
### For the Medha Pathak Lab, UC Irvine

## Overview

This FLIKA plugin provides **ROI-free, puncta-free detection** of transient fluorescence events (flickers) using sophisticated temporal signal analysis. Unlike traditional approaches that require manual ROI placement or puncta detection, this plugin automatically analyzes every pixel's temporal behavior to identify when and where brief intensity changes occur.

## Key Innovation

**No ROI Drawing, No Puncta Detection Required!**

Traditional workflow:
1. Manually draw ROIs around suspected flicker sites
2. Extract intensity traces
3. Manually identify peaks
4. **Problem**: Subjective, time-consuming, misses unexpected locations

**This plugin's workflow:**
1. Load your fluorescence movie
2. Click "Detect Flickers"
3. **Done!** Automatically finds all flickers everywhere

The plugin analyzes the temporal characteristics of every pixel to identify statistically significant transient events.

## Perfect For

- **Ca²⁺ flickers** - Local calcium release events
- **PIEZO1-mediated calcium transients** - Mechanically-activated events  
- **Spontaneous activity mapping** - Where does activity occur?
- **Mechanotransduction events** - Brief responses to mechanical stimuli
- **Any localized, transient fluorescence changes**

## Detection Methods

### 1. Z-Score Analysis (Recommended) ⭐

**How it works:** For each pixel, calculates a rolling baseline and standard deviation. Identifies timepoints where intensity significantly exceeds baseline (z-score > threshold).

**Best for:** General-purpose flicker detection, robust to noise

**Key parameter:** Threshold (typical: 2.5-4.0)
- 2.5 = more sensitive, may include noise
- 3.0 = balanced (recommended)
- 4.0 = very conservative, only strong flickers

### 2. Temporal Derivative

**How it works:** Detects rapid increases in intensity (dF/dt). Identifies the rising phase of transients.

**Best for:** Fast, sharp transients; photolysis experiments

**Key parameter:** Threshold (% change, typical: 10-50%)

### 3. Wavelet Detection

**How it works:** Uses continuous wavelet transform (Mexican Hat wavelet) to identify brief events at multiple timescales.

**Best for:** Detecting transients of varying durations, excellent time localization

**Key parameter:** Threshold (wavelet coefficient, typical: 5-15)

### 4. Variance-Based

**How it works:** Identifies pixels with high temporal variance (variability over time).

**Best for:** Finding regions of high activity; exploratory analysis

**Key parameter:** Threshold (variance value, dataset-dependent)

## Installation

### Standard Installation

1. Copy the `flicker_analyzer` folder to your FLIKA plugins directory:
   - Windows: `C:\Users\YourName\.FLIKA\plugins\`
   - Mac: `~/.FLIKA/plugins/`
   - Linux: `~/.FLIKA/plugins/`

2. Ensure the folder contains:
   - `flicker_analyzer.py`
   - `__init__.py`
   - `info.xml`
   - `about.html`
   - `README.md`

3. Restart FLIKA

### Dependencies

Required packages (usually included with FLIKA):
- numpy
- scipy
- matplotlib
- pandas
- pyqtgraph
- **PyWavelets** (for wavelet method)

If PyWavelets is missing:
```bash
pip install PyWavelets --break-system-packages
```

## Usage Guide

### Basic Workflow

1. **Load Your Movie**
   - Open a time-series fluorescence movie in FLIKA
   - Should show transient events (Ca²⁺ flickers, etc.)
   - Works on raw data - preprocessing not strictly required

2. **Open the Plugin**
   - Go to Plugins → Flicker Analyzer
   - The analysis GUI will open

3. **Choose Detection Method**
   - **Start with Z-Score** (recommended for most cases)
   - Try other methods if results are unsatisfactory

4. **Set Threshold**
   - **Z-Score**: Start with 3.0
   - **Derivative**: Start with 20% 
   - **Wavelet**: Start with 10
   - **Variance**: Dataset-dependent, try 50-100

5. **Configure Duration Criteria**
   - **Min Duration**: Shortest real flicker (2-3 frames typical)
   - **Max Duration**: Longest real flicker (20-50 frames typical)
   - These filter out artifacts and slow trends

6. **Set Smoothing (Optional)**
   - **Spatial Smoothing**: 0.5-2.0 reduces noise, 0 for none
   - **Temporal Smoothing**: 0 for none, 5-11 for noisy data
   - Start with defaults, adjust if needed

7. **Run Detection**
   - Click "Detect Flickers"
   - Status bar shows progress
   - Wait for completion message

8. **Visualize Results**
   - Click "Visualize Results"
   - Comprehensive 9-panel figure appears

9. **Review and Adjust**
   - If too many false positives: increase threshold
   - If missing flickers: decrease threshold
   - Re-run with adjusted parameters

10. **Export Data**
    - Click "Export Data"
    - Saves multiple files for further analysis

### Parameter Recommendations by Application

**Ca²⁺ Flickers (Fast, Local):**
- Method: Z-Score or Derivative
- Threshold: 3.0 (z-score) or 25% (derivative)
- Min Duration: 2-3 frames
- Max Duration: 20 frames
- Spatial Smoothing: 1.0
- Temporal Smoothing: 0 (preserve fast kinetics)

**PIEZO1-Mediated Calcium:**
- Method: Z-Score
- Threshold: 2.5-3.0
- Min Duration: 3-5 frames
- Max Duration: 30 frames
- Spatial Smoothing: 1.5
- Temporal Smoothing: 0-5

**Slow Calcium Waves:**
- Method: Derivative or Wavelet
- Threshold: 15% (derivative)
- Min Duration: 10 frames
- Max Duration: 100 frames
- Spatial Smoothing: 2.0
- Temporal Smoothing: 7

**Noisy Data:**
- Increase spatial smoothing (1.5-2.5)
- Use temporal smoothing (5-11, must be odd)
- Use Z-Score method (most robust)
- Increase threshold (3.5-4.0)

## Understanding the Output

### Visualization Panels

1. **Frequency Map** (top-left)
   - Heatmap showing where flickers occur most often
   - Cyan dots = individual event locations
   - Hot colors = high frequency regions
   - Identifies "hotspots" of activity

2. **Amplitude Map** (top-center)
   - Average flicker intensity at each location
   - Shows strength of events, not just frequency
   - Useful for comparing flicker magnitude across cell

3. **Temporal Activity** (top-right)
   - Number of active flickers at each frame
   - Shows when flickers occur over time
   - Peaks indicate bursts of activity

4. **Raster Plot** (middle-left)
   - Each row = one flicker event
   - Horizontal line shows duration
   - Visualize all events simultaneously
   - See temporal patterns

5. **Duration Distribution** (middle-center)
   - Histogram of flicker durations
   - Red line = mean duration
   - Characterizes flicker kinetics

6. **Amplitude Distribution** (middle-right)
   - Histogram of flicker amplitudes
   - Red line = mean amplitude
   - Shows intensity range of events

7. **Size Distribution** (bottom-left)
   - Histogram of spatial sizes
   - How large are flickers?
   - Can distinguish point sources from diffuse events

8. **Statistics Summary** (bottom-center)
   - Comprehensive numerical statistics
   - Count, frequency, duration, amplitude
   - Spatial and temporal metrics
   - Copy for publications

9. **Amplitude Over Time** (bottom-right)
   - Scatter plot of amplitude vs. time
   - Detect temporal trends
   - Are flickers getting stronger/weaker?

### Key Metrics Explained

**Total Flickers**
- Count of all detected events
- Compare between conditions (control vs. treatment)

**Frequency (per pixel per frame)**
- Normalized activity rate
- Accounts for image size and duration
- Use for quantitative comparisons

**Mean Duration**
- Average flicker lifetime in frames
- Convert to seconds using frame rate
- Characterizes kinetics

**Mean Amplitude**
- Average intensity increase during flickers
- In arbitrary fluorescence units
- Compare across cells (normalize first)

**Mean Active Flickers Per Frame**
- How many events occur simultaneously
- High value = coordinated activity

**Max Simultaneous Flickers**
- Peak number of concurrent events
- Indicates activity bursts

**Inter-Flicker Interval (IFI)**
- Time between flickers at same location
- Characterizes firing pattern
- Regular vs. random activity

**Activity Center**
- X,Y coordinates of activity centroid
- Where is activity concentrated?
- Useful for asymmetric distributions

### Additional Windows

Click "Create Frequency/Amplitude Windows" to generate:
- Separate FLIKA windows with frequency and amplitude maps
- Can overlay on original movie
- Apply further FLIKA processing

## Advanced Features

### Spatial Clustering

The plugin automatically groups nearby active pixels into single events. This:
- Defines flicker spatial extent
- Calculates flicker size
- Prevents over-counting of large events

**Merge Distance** parameter controls this (default: 10 pixels)

### Baseline Correction

Z-Score and Variance methods use **rolling baseline** calculation:
- Adapts to slow intensity changes
- Handles photobleaching automatically
- **Baseline Window** sets adaptation rate

Large window (100+ frames) = slow adaptation
Small window (20-30 frames) = fast adaptation

### Temporal Filtering

Optional smoothing helps with noisy data:
- **Spatial Gaussian**: Reduces pixel noise
- **Savitzky-Golay**: Smooths temporal traces

Trade-off: Improves detection vs. Reduces temporal resolution

## Workflow Examples

### Example 1: Compare Control vs. Stimulated

```
Goal: Does mechanical stimulation increase Ca2+ flicker frequency?

1. Record 5 min baseline, then apply stretch, record 5 min
2. Split into two movies (pre and post)
3. Run Flicker Analyzer on both with SAME parameters
4. Compare statistics:
   - Flicker count
   - Frequency per pixel per frame
   - Mean amplitude
5. Statistical test (t-test, Mann-Whitney)
6. Plot frequency maps side-by-side
```

### Example 2: Identify Active Regions

```
Goal: Where are PIEZO1 flickers concentrated?

1. Run Flicker Analyzer
2. Examine frequency map
3. Create frequency map window
4. Threshold to identify top 10% of pixels
5. Overlay on PIEZO1 localization image
6. Quantify colocalization
```

### Example 3: Temporal Pattern Analysis

```
Goal: Are flickers periodic or random?

1. Run Flicker Analyzer
2. Export temporal activity data
3. Load CSV in analysis software
4. Compute autocorrelation
5. Detect periodicities (FFT)
6. Compare to random simulation
```

### Example 4: High-Throughput Screening

```
Goal: Test 20 drug conditions on flicker frequency

1. Acquire movies for all conditions
2. Batch process:
   - Write script to loop through movies
   - Call flicker_analyzer() programmatically
   - Save summary statistics
3. Compile results into spreadsheet
4. Statistical analysis and plotting
```

## Troubleshooting

### No Flickers Detected

**Symptoms:** Analysis completes but reports 0 flickers

**Solutions:**
1. Lower threshold (try 2.0 for z-score)
2. Reduce min duration (try 1-2 frames)
3. Increase spatial smoothing (reduces noise threshold)
4. Check if movie actually has transients (visual inspection)
5. Try different detection method

### Too Many False Positives

**Symptoms:** Thousands of flickers, including obvious noise

**Solutions:**
1. Increase threshold (try 3.5-4.0 for z-score)
2. Increase min duration (filter brief noise spikes)
3. Increase min flicker size (filter single noisy pixels)
4. Add spatial smoothing
5. Consider preprocessing (background subtraction)

### Flickers Incorrectly Split

**Symptoms:** One flicker detected as multiple events

**Solutions:**
1. Increase merge distance (20-30 pixels)
2. Reduce spatial smoothing (may be spreading signal)
3. Adjust min duration

### Flickers Incorrectly Merged

**Symptoms:** Multiple distinct flickers counted as one

**Solutions:**
1. Decrease merge distance (5-10 pixels)
2. Increase spatial resolution (if possible)
3. Check max duration (may be too long)

### Very Slow Processing

**Symptoms:** Analysis takes many minutes

**Causes:**
- Large movies (high resolution or many frames)
- Wavelet method (slower than others)
- No spatial smoothing (processes every pixel)

**Solutions:**
1. Use Z-Score or Derivative (faster)
2. Spatially bin movie first (Process → Bin)
3. Analyze subset of frames initially
4. Add moderate spatial smoothing (reduces effective pixels)

### Memory Errors

**Symptoms:** Plugin crashes or FLIKA freezes

**Solutions:**
1. Close other FLIKA windows
2. Reduce movie size (crop or bin)
3. Process shorter time segments
4. Restart FLIKA

## Validation and Quality Control

### Visual Inspection

Always validate automated detection:
1. Play movie, note obvious flickers
2. Check if detected in output
3. Review frequency map - reasonable?
4. Examine few events in raster plot

### Parameter Sensitivity

Test parameter robustness:
1. Run with threshold ± 0.5
2. Count should not change drastically
3. If very sensitive, results may be unreliable

### Positive/Negative Controls

- **Positive control:** Movie with obvious flickers
- **Negative control:** Baseline movie (no stimulus)
- Detection should show clear difference

### Comparison to Manual Analysis

Occasionally validate against manual ROI-based analysis:
1. Draw ROIs at known flicker sites
2. Extract traces and count manually
3. Compare to automated detection
4. Should agree within ~10-20%

## Data Analysis Tips

### Comparing Conditions

When comparing experimental conditions:
1. **Use identical parameters** across all movies
2. **Normalize** if intensity varies (careful with normalization method)
3. **Multiple cells** - analyze several, report mean ± SEM
4. **Statistical tests** - appropriate for your design (t-test, ANOVA, etc.)
5. **Report** - Mean flicker frequency ± SEM (n=X cells)

### Correlation Analysis

Correlate flicker activity with other measurements:
- PIEZO1 intensity at flicker sites
- Cell shape/migration parameters  
- Mechanical stimulation timing
- Drug application

Export flicker coordinates and use custom scripts for correlation.

### Time-Course Experiments

For experiments over hours:
1. Correct photobleaching first (if severe)
2. Analyze in time bins (e.g., every 10 min)
3. Track changes in frequency, amplitude over time
4. Use temporal activity data

### Spatial Analysis

Beyond built-in maps:
- Divide cell into regions (leading edge vs. body)
- Compare frequency in each region
- Correlate with cytoskeletal features
- Radial analysis from cell center

## Publication Guidelines

### Methods Section

Example text:

"Ca²⁺ flickers were detected using an automated pixel-wise temporal analysis plugin for FLIKA. Each pixel's fluorescence trace was analyzed using a z-score method with a rolling baseline (50 frames). Transient events with z-score > 3.0, duration between 2-30 frames, and spatial extent > 4 pixels were classified as flickers. Frequency maps were generated by counting events at each location and spatially smoothing (σ=2). Flicker amplitude, duration, and spatial properties were quantified automatically."

### Reporting Statistics

Report these key metrics:
- Number of cells analyzed
- Total flickers per cell (mean ± SEM)
- Flicker frequency (events/μm²/s or events/pixel/frame)
- Mean flicker duration (s)
- Mean flicker amplitude (ΔF/F or arbitrary units)
- Statistical test used and p-values

### Figures

Effective figure panels:
1. **Representative frequency map** - shows spatial distribution
2. **Quantification** - bar plot of flicker frequency by condition
3. **Duration distribution** - histogram comparing conditions
4. **Temporal activity** - shows response to stimulus over time

## Programmatic Use

For batch processing or custom analysis:

```python
from flika.plugins.flicker_analyzer import flicker_analyzer

# Assuming window is loaded
results = flicker_analyzer(
    window=g.win,
    method='zscore',
    threshold=3.0,
    min_duration=2,
    max_duration=30
)

# Access results
print(f"Detected {results['n_flickers']} flickers")
print(f"Mean duration: {results['mean_duration']} frames")

# Get event list
events = flicker_analyzer.flicker_events
for event in events[:10]:  # First 10 events
    print(f"Frame {event['start_frame']}: "
          f"Duration {event['duration']}, "
          f"Amplitude {event['amplitude']:.2f}")
```

## Integration with Other Tools

### With Your Existing Plugins

**Cluster Analyzer** (detect PIEZO1 clustering):
- Run both on same movie
- Correlate flicker locations with PIEZO1 clusters
- Question: Do flickers occur at PIEZO1-rich regions?

**Kymograph Analyzer** (spatiotemporal dynamics):
- Draw line across cell
- Create kymograph
- Use to validate flicker timing/propagation

**Puncta Tracker** (if you add it):
- Track PIEZO1 puncta
- Overlay flicker frequency map
- Do tracked puncta colocalize with flicker sites?

### With Standard FLIKA Tools

**Process Menu:**
- Background subtraction before analysis
- Photobleaching correction
- Spatial/temporal filtering

**ROI Tools:**
- Define cell boundaries
- Analyze flickers only within cell
- Compare subcellular regions

## Future Enhancements

Potential additions based on feedback:
- Wave propagation detection and characterization
- Colocalization with other channels automatically
- Machine learning classification of event types
- 3D support for confocal z-stacks
- Real-time detection during acquisition

## Tips for Best Results

1. **Start Simple:** Use Z-Score with default parameters first
2. **Iterate:** Adjust parameters based on initial results  
3. **Validate:** Visually check a few detections match reality
4. **Consistent:** Use same parameters across compared conditions
5. **Document:** Record all parameters used for reproducibility
6. **Export:** Save raw data for additional analysis later

## FAQ

**Q: Do I need to background subtract first?**
A: Not strictly necessary - z-score method handles varying baselines. But can improve results for very uneven illumination.

**Q: What if my flickers are at edge of detection?**
A: Lower threshold and use stricter duration criteria. Better to detect marginal events and filter than miss them entirely.

**Q: Can I analyze just part of the cell?**
A: Yes - crop the movie to your region of interest first, then analyze.

**Q: How do I convert to ΔF/F?**
A: Run analysis on raw data first. For quantification, normalize movie to ΔF/F, then re-analyze or convert amplitudes post-hoc.

**Q: My movie has drift - will this work?**
A: Moderate drift is OK. For severe drift, use image registration first (e.g., StackReg).

**Q: Can I detect calcium waves vs. flickers?**
A: Yes! Waves have longer duration and larger size. Increase max duration (100+ frames) and merge distance.

## Support

For questions, issues, or feature requests:
- Contact: Pathak Lab, UC Irvine
- Email: [through lab website]
- GitHub: [if you set one up]

## Citing

If you use this plugin, please cite:

1. **FLIKA**: Ellefsen, K., et al. Cell Calcium 56:147-156, 2014
2. **Your relevant PIEZO1/flicker paper**

## Version History

**v1.0.0** (2025)
- Initial release
- Four detection methods (z-score, derivative, wavelet, variance)
- Comprehensive visualization and export
- No ROI or puncta detection required

---

## Summary

This plugin revolutionizes flicker analysis by:
- ✓ Removing subjective ROI placement
- ✓ Eliminating difficult puncta detection step
- ✓ Automatically finding ALL flickers EVERYWHERE
- ✓ Providing rigorous statistical detection
- ✓ Generating publication-quality outputs

Perfect for your Ca²⁺ flicker and PIEZO1 mechanotransduction research!

**Developed by Claude for the Medha Pathak Lab, UC Irvine**
