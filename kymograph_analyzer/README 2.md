# Kymograph Analyzer Plugin for FLIKA

## Overview

The Kymograph Analyzer plugin is designed specifically for studying spatiotemporal dynamics in TIRF microscopy, with a particular focus on analyzing **Ca²⁺ flickers**, **PIEZO1 channel activity**, and **membrane protein dynamics** in single cells.

## Perfect For Studying:

- **Ca²⁺ Flickers**: Localized calcium signals at the plasma membrane
- **PIEZO1 Activity**: Mechanosensitive channel dynamics and localization
- **Membrane Protein Movements**: Tracking proteins along membranes over time
- **Vesicle Trafficking**: Exocytosis and endocytosis events
- **Wave Propagation**: Calcium waves, actin waves, or protein recruitment waves
- **Protein Recruitment Dynamics**: Temporal analysis of protein assembly
- **Temporal Correlations**: Analyzing relationships between spatially separated events

## Key Features

### 1. **Flexible ROI Drawing**
- **Line ROI**: Draw straight lines across your region of interest
- **Polyline ROI**: Create complex paths following membrane contours
- **Freehand ROI**: Trace irregular paths
- **Adjustable Width**: Average intensity across a band (1-50 pixels)

### 2. **Advanced Processing**
- **Temporal Binning**: Reduce temporal noise by averaging frames
- **Normalization**: Scale intensity to [0, 1] for comparisons
- **Detrending**: Remove slow baseline drifts
- **Gaussian Smoothing**: Reduce noise while preserving features

### 3. **Automatic Event Detection**
Three detection methods:
- **Peaks**: Detect sharp Ca²⁺ flickers or transient events
- **Threshold**: Simple amplitude-based detection
- **Z-score**: Statistical deviation-based detection

Configurable parameters:
- Detection threshold (sigma or z-score)
- Minimum distance between events
- Minimum event duration

### 4. **Quantitative Analysis**
- **Event Properties**: Amplitude, duration, width, timing
- **Velocity Measurements**: Wave propagation speeds
- **Statistical Profiling**: Mean, std, temporal/spatial averages
- **Dwell Time Analysis**: How long events persist at locations

### 5. **Comprehensive Visualization**
- Kymograph display with proper scaling
- Spatial intensity profiles (averaged over time)
- Temporal intensity profiles (averaged over space)
- Event scatter plots with amplitude mapping
- Intensity distribution histograms
- Statistical summaries

### 6. **Data Export**
- Kymograph arrays (.npy format)
- Event tables (.csv with all detected events)
- Statistical summaries (.csv)
- Publication-quality figures

## Installation

1. **Copy files to FLIKA plugins directory:**
   ```
   <FLIKA plugins>/kymograph_analyzer/
   ├── __init__.py
   ├── kymograph_analyzer.py
   ├── info.xml
   └── about.html
   ```

2. **Required dependencies** (usually included with FLIKA):
   - numpy
   - scipy
   - matplotlib
   - pandas
   - pyqtgraph

3. **Restart FLIKA**

4. **Access plugin:**
   `Plugins → Kymograph Analyzer`

## Quick Start Guide

### For Analyzing Ca²⁺ Flickers (PIEZO1 Studies)

1. **Open your TIRF time-lapse movie**
   - Must be a 3D stack (time, x, y)
   - Ensure adequate frame rate for your Ca²⁺ dynamics

2. **Launch the plugin**
   - `Plugins → Kymograph Analyzer`

3. **Draw a line ROI**
   - Click "Draw Line ROI" button
   - Draw a line along the membrane where you see Ca²⁺ flickers
   - Adjust line width (5-10 pixels works well for subcellular features)

4. **Set processing options**
   - **Normalize**: ON (for consistent comparison)
   - **Detrend**: ON (if you see slow baseline drift)
   - **Gaussian Smoothing**: 0.5-1.0 (reduces noise)
   - **Temporal Binning**: 1 (unless very noisy)

5. **Generate Kymograph**
   - Click "Generate Kymograph"
   - New window shows time (vertical) vs. space (horizontal)
   - Ca²⁺ flickers appear as bright spots/streaks

6. **Detect Ca²⁺ Events**
   - Set **Detection Method**: "peaks"
   - Set **Threshold**: 2-3 (STD above baseline)
   - Set **Min Distance**: 3-5 frames (depends on frame rate)
   - Set **Min Duration**: 2-4 frames (transient vs sustained)
   - Click "Detect Events"

7. **Analyze and Visualize**
   - Click "Full Analysis & Plot"
   - Examine event distribution
   - Check temporal and spatial profiles
   - Note propagation patterns

8. **Export Results**
   - Click "Export Data"
   - Choose filename
   - Saves kymograph, events table, and statistics

## Detailed Usage

### Understanding Kymographs

A kymograph is a 2D representation where:
- **Vertical axis (Y)** = Time (frames)
- **Horizontal axis (X)** = Spatial position along ROI
- **Intensity/Color** = Fluorescence intensity

**Interpreting patterns:**
- **Vertical lines**: Stationary events at fixed positions
- **Diagonal lines**: Moving events/waves
- **Bright spots**: Transient localized events (Ca²⁺ flickers!)
- **Horizontal lines**: Simultaneous events across space

### ROI Selection Tips

**For Ca²⁺ Flickers:**
- Draw line along plasma membrane
- Width: 5-10 pixels (captures membrane region)
- Multiple lines for different membrane regions

**For Protein Tracking:**
- Follow protein movement trajectory
- Polyline ROI for curved paths
- Width: 3-5 pixels for single molecules

**For Wave Analysis:**
- Draw perpendicular to wave direction
- Longer ROI captures more spatial extent
- Width depends on wave width

### Parameter Guidelines

#### Line Width
- **1-3 pixels**: Single molecule tracking
- **5-10 pixels**: Subcellular structures (Ca²⁺ flickers, adhesions)
- **10-20 pixels**: Larger features (stress fibers, broad waves)

#### Temporal Binning
- **1 (no binning)**: Fast dynamics, good signal
- **2-3**: Moderate noise reduction
- **4-10**: Very noisy data, slower dynamics

#### Gaussian Smoothing (Sigma)
- **0**: No smoothing
- **0.5-1.0**: Slight noise reduction
- **1.5-3.0**: Moderate smoothing
- **>3**: Heavy smoothing (may lose features)

#### Event Detection Threshold
- **1.5-2.0 STD**: Sensitive (more false positives)
- **2.0-3.0 STD**: Balanced
- **3.0-5.0 STD**: Selective (high confidence only)

#### Minimum Duration
- **2-3 frames**: Transient flickers
- **4-6 frames**: Brief events
- **>10 frames**: Sustained events

### Advanced Applications

#### 1. Analyzing PIEZO1 Ca²⁺ Flickers

Based on Ellefsen et al. (2019), PIEZO1 generates localized Ca²⁺ flickers:

```
Workflow:
1. TIRF imaging of Ca²⁺ indicator (GCaMP, Cal-520, etc.)
2. Draw line ROI along cell membrane
3. Generate kymograph (width=7, normalize=True)
4. Detect events (method='peaks', threshold=2.5, min_duration=2)
5. Analyze spatial distribution of flickers
6. Measure flicker properties (amplitude, frequency, duration)
```

**Key Measurements:**
- Flicker frequency per unit length
- Flicker amplitude distribution
- Spatial clustering of flickers
- Temporal correlations between flickers

#### 2. Measuring Protein Wave Velocities

For actin waves, protein recruitment waves:

```
Workflow:
1. Draw line perpendicular to wave direction
2. Generate kymograph
3. Diagonal features = moving waves
4. Measure velocity: slope = distance/time
```

**Velocity Calculation:**
- Identify diagonal streak in kymograph
- Measure spatial displacement (Δx)
- Measure time taken (Δt)
- Velocity = Δx / Δt (pixels/frame)
- Convert to physical units using pixel size and frame interval

#### 3. Vesicle Trafficking Analysis

For exocytosis, endocytosis:

```
Workflow:
1. Label vesicles (pHluorin, Rab proteins, etc.)
2. Draw line along trafficking route
3. Generate kymograph
4. Detect events (appearing/disappearing spots)
5. Measure:
   - Dwell time (how long vesicle stays)
   - Movement velocity
   - Fusion events (bright flashes)
```

#### 4. Temporal Correlation Analysis

To see if events at different locations are correlated:

```
Workflow:
1. Generate kymograph
2. Select two spatial positions
3. Extract intensity traces over time
4. Calculate cross-correlation
5. Determine lag time between events
```

### Integration with Other FLIKA Tools

**After Kymograph Analysis:**

1. **ROI Analysis**
   - Use coordinates from detected events
   - Create ROIs at event locations
   - Measure intensities in original movie

2. **Tracking (if available)**
   - Use event positions as seeds
   - Track movements through space
   - Combine with temporal data from kymograph

3. **Statistical Analysis**
   - Export event tables
   - Use FLIKA's analysis tools
   - Create intensity vs. time plots

4. **Co-localization**
   - Generate kymographs for two channels
   - Detect events in each
   - Compare timing and spatial overlap

## Scientific Applications

### Published Use Cases

1. **PIEZO1 Ca²⁺ Signaling** (Pathak Lab)
   - Study of myosin-II mediated traction forces evoking localized Piezo1 Ca²⁺ flickers
   - Kymographs reveal spatial and temporal patterns of Ca²⁺ events

2. **Membrane Protein Dynamics**
   - Dwell time measurements
   - Endocytosis kinetics
   - Protein recruitment timing

3. **Intraflagellar Transport**
   - Tracking IFT particles in cilia
   - Velocity measurements
   - Directional analysis

4. **Actin Dynamics**
   - Retrograde flow measurements
   - Wave propagation
   - Assembly/disassembly kinetics

### Typical Measurements from Kymographs

**For Ca²⁺ Flickers:**
- Event frequency (events/µm/s)
- Flicker amplitude (ΔF/F₀ or absolute intensity)
- Flicker duration (ms or frames)
- Spatial clustering index
- Rise time and decay time

**For Protein Movements:**
- Velocity (µm/s)
- Directional persistence
- Pause frequency and duration
- Run length distribution

**For Waves:**
- Propagation velocity (µm/s)
- Wave amplitude
- Wave frequency
- Spatial extent

## Troubleshooting

### Issue: Kymograph looks noisy/grainy
**Solutions:**
- Increase line width (average over more pixels)
- Apply Gaussian smoothing (sigma=1-2)
- Use temporal binning (bin 2-3 frames)
- Check original image quality

### Issue: Events not detected
**Solutions:**
- Lower threshold (try 1.5-2.0 STD)
- Reduce minimum duration
- Check if detrending helps
- Verify ROI is in correct location

### Issue: Too many false positive events
**Solutions:**
- Increase threshold (try 3.0+ STD)
- Increase minimum duration
- Use 'zscore' method instead of 'peaks'
- Apply more smoothing

### Issue: Kymograph shows diagonal artifacts
**Causes:**
- Stage drift during acquisition
- Motion in the field of view

**Solutions:**
- Use drift correction before kymograph
- Draw ROI avoiding drifted regions
- Consider motion correction tools

### Issue: Events appear smeared in time
**Solutions:**
- Reduce temporal binning
- Check frame rate is adequate
- Reduce Gaussian smoothing

### Issue: Horizontal banding in kymograph
**Causes:**
- Photobleaching
- Laser power fluctuations
- Focus drift

**Solutions:**
- Enable detrending
- Apply bleach correction to original stack first
- Use BaSiC correction (separate plugin)

## Best Practices

### Data Acquisition

1. **Frame Rate**
   - Ca²⁺ flickers: ≥20 Hz (50 ms/frame)
   - Protein tracking: ≥10 Hz
   - Slow dynamics: 1-5 Hz

2. **Spatial Resolution**
   - Pixel size ≤ desired feature size
   - Nyquist sampling: ≥2 pixels per smallest feature

3. **Signal-to-Noise**
   - Maximize signal (laser power, exposure time)
   - Minimize noise (cooling, binning, gain)
   - Balance to avoid photobleaching

### Analysis Workflow

1. **Start Simple**
   - Generate basic kymograph first
   - Inspect visually for patterns
   - Then apply processing

2. **Optimize Parameters**
   - Try different thresholds
   - Compare detection methods
   - Validate on known events

3. **Validate Results**
   - Check detected events visually
   - Compare with manual counting
   - Test on control/negative data

4. **Document Everything**
   - Save all parameters used
   - Export raw kymographs
   - Keep analysis notes

### Statistical Rigor

1. **Multiple Cells**
   - Analyze ≥10-20 cells
   - Report mean ± SEM
   - Test for statistical significance

2. **Controls**
   - Include negative controls
   - Compare treatment vs control
   - Use paired analyses when appropriate

3. **Reproducibility**
   - Use consistent ROI selection criteria
   - Apply same parameters across datasets
   - Blind analysis when possible

## Example Workflows

### Workflow 1: Basic Ca²⁺ Flicker Analysis

```python
from plugins.kymograph_analyzer import kymograph_analyzer

# 1. Load your TIRF movie
# (assume it's already open as 'window')

# 2. Generate kymograph programmatically
kymo_window = kymograph_analyzer(
    window=my_tirf_movie,
    roi_type='line',
    width=7,
    temporal_binning=1,
    normalize=True,
    detrend=True,
    gaussian_sigma=1.0
)

# 3. Detect Ca²⁺ flickers
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)

print(f"Detected {len(events)} Ca²⁺ flickers")

# 4. Analyze
stats = kymograph_analyzer.calculate_statistics()
print(f"Mean intensity: {stats['mean_intensity']:.3f}")

# 5. Create plots
kymograph_analyzer.plot_analysis()

# 6. Export
kymograph_analyzer.export_data("piezo1_flickers_cell1")
```

### Workflow 2: Comparing Multiple Cells

```python
# Process multiple cells
results = []

for cell_num, window in enumerate(cell_windows):
    # Generate kymograph
    kymo = kymograph_analyzer(window, width=7, normalize=True)
    
    # Detect events
    events = kymograph_analyzer.detect_events(threshold=2.5)
    
    # Store results
    results.append({
        'cell': cell_num,
        'n_events': len(events),
        'mean_amplitude': np.mean([e['amplitude'] for e in events])
    })

# Analyze across cells
import pandas as pd
df = pd.DataFrame(results)
print(df.describe())
```

### Workflow 3: Wave Velocity Measurement

```python
# For measuring wave propagation

# 1. Draw line perpendicular to wave
# 2. Generate kymograph
kymo = kymograph_analyzer(
    window=wave_movie,
    width=10,
    temporal_binning=1,
    normalize=True
)

# 3. Measure velocities
velocities = kymograph_analyzer.measure_velocities()

print(f"Mean wave velocity: {velocities['mean_velocity']:.2f} pixels/frame")

# Convert to physical units
pixel_size = 0.16  # µm/pixel
frame_interval = 0.05  # seconds
velocity_um_s = velocities['mean_velocity'] * pixel_size / frame_interval

print(f"Wave velocity: {velocity_um_s:.2f} µm/s")
```

## Citations

If you use this plugin in your research, please cite:

1. **FLIKA**:
   Ellefsen, K. L., et al. (2019). "Applications of FLIKA, a Python-based image processing and analysis platform, for studying local events of cellular calcium signaling." *BBA-Molecular Cell Research*, 1866(7), 1171-1179.

2. **For PIEZO1 Ca²⁺ Flickers**:
   Ellefsen, K. L., et al. (2019). "Myosin-II mediated traction forces evoke localized Piezo1 Ca²⁺ flickers." *Communications Biology*, 2(1), 298.

3. **For Kymograph Methods**:
   Relevant papers depending on your application (see references in your field)

## Support

For questions, issues, or feature requests:
- FLIKA Documentation: https://flika-org.github.io/
- FLIKA Forums/GitHub
- Pathak Lab: https://www.faculty.uci.edu/profile/?facultyId=6245

## Version History

### Version 1.0.0 (Current)
- Initial release
- Core kymograph generation
- Multiple ROI types
- Event detection (3 methods)
- Velocity measurements
- Statistical analysis
- Comprehensive visualization
- Data export

## Future Enhancements

Potential additions:
- Multi-channel kymographs
- Correlation analysis between spatial positions
- Machine learning event detection
- 3D kymographs (space-space-time)
- Integration with particle tracking
- Automated ROI placement
- Batch processing multiple movies
- Advanced velocity analysis (radon transform)

---

**Designed specifically for studying PIEZO1-mediated Ca²⁺ signaling and membrane protein dynamics in the Medha Pathak Lab at UC Irvine!**

Happy analyzing! 🔬📊✨
