# Puncta Analyzer Plugin for FLIKA

## Overview

The **Puncta Analyzer** is a comprehensive FLIKA plugin designed specifically for single-cell fluorescence microscopy analysis. It's perfect for studying protein localization, dynamics, and activity in live cells using TIRF microscopy and other techniques.

**Developed for the Pathak Lab at UCI** to study PIEZO1 mechanotransduction, Ca²⁺ signaling, and neural stem cell biology.

## Perfect For Your Research

### PIEZO1 Studies
- **Single-particle tracking** of fluorescently-labeled PIEZO1 channels
- **Localization analysis** - where PIEZO1 localizes in the membrane
- **Dynamics quantification** - diffusion, clustering, mobility states
- **Colocalization** with other proteins (cytoskeleton, caveolin, etc.)

### Ca²⁺ Flicker Detection
- **Automated detection** of transient Ca²⁺ signals
- **Spatiotemporal analysis** - when and where flickers occur
- **Event characterization** - amplitude, duration, rise/decay times
- **Statistical analysis** of flicker properties

### Protein Puncta Analysis
- **Spot/puncta detection** in fluorescence images
- **Intensity quantification** over time
- **Morphological analysis** (size, shape, distribution)
- **Time-lapse tracking** of protein clusters

## Key Features

### 1. Multiple Detection Algorithms

Choose the best algorithm for your data:

- **LoG (Laplacian of Gaussian)**: Best for round, well-defined puncta
  - Excellent for PIEZO1-GFP/tdTomato puncta
  - Good for beads and calibration samples
  
- **DoG (Difference of Gaussian)**: More robust to background
  - Good for noisy TIRF images
  - Better background suppression
  
- **Threshold**: Simple and fast
  - Good for high SNR data
  - Quick initial analysis
  
- **Wavelet**: Best for very noisy data
  - Robust to uneven illumination
  - Good for challenging samples

### 2. Sub-Pixel Localization

- **2D Gaussian fitting** for each detected puncta
- **Sub-pixel accuracy** (~10-50 nm precision)
- **Quality metrics** (SNR, fit quality)
- Essential for super-resolution-like precision

### 3. Particle Tracking

- **Frame-to-frame tracking** across time series
- **Nearest neighbor + LAP** assignment
- **Gap closing** for temporary disappearances
- **Track quality filtering**

Enables analysis of:
- PIEZO1 diffusion and mobility
- Protein recruitment dynamics
- Clustering and unclustering events

### 4. Ca²⁺ Flicker Detection

Specifically designed for detecting transient calcium signals:

- **Baseline estimation** using running median
- **Event detection** above threshold
- **Automatic characterization**:
  - Peak amplitude
  - Duration
  - Rise time
  - Decay time
  - Spatial location

Perfect for analyzing Piezo1-induced Ca²⁺ flickers reported in your publications!

### 5. Comprehensive Output

- **CSV export** of all detections, tracks, and events
- **Statistics summary** with key metrics
- **Visualization** of results
- **Raw data** for custom analysis in Python/R/MATLAB

## Installation

### Quick Install

1. Copy these files to your FLIKA plugins directory:
   ```
   <FLIKA plugins>/puncta_analyzer/
   ├── __init__.py
   ├── puncta_analyzer.py
   ├── info.xml
   └── about.html
   ```

2. Restart FLIKA

3. Find under: **Plugins → Puncta Analyzer**

### Dependencies

Required packages (usually included with FLIKA):
```bash
pip install numpy scipy scikit-image pandas pyqtgraph
```

## Usage Guide

### Basic Workflow

1. **Load your image**
   - Open TIRF time-series in FLIKA
   - Can be 2D (single frame) or 3D (time series)

2. **Launch plugin**
   - Go to **Plugins → Puncta Analyzer**

3. **Configure detection**
   - Select detection method (try LoG first)
   - Adjust sigma (match your puncta size, typically 2-3 pixels)
   - Set threshold (start with 2.0, adjust to see good detections)

4. **Enable tracking** (for time series)
   - Check "Enable Tracking"
   - Set max linking distance (typically 5-10 pixels)

5. **Enable event detection** (for Ca²⁺ flickers)
   - Check "Detect Ca²⁺ Flickers"
   - Set event threshold (1.5-2x baseline)
   - Set minimum duration (2-5 frames)

6. **Run analysis**
   - Click OK
   - Results appear in status bar and console

7. **Export results**
   - Check "Export to CSV"
   - Find files in `~/FLIKA_analysis/<image_name>/`

### Parameter Guidelines

#### Detection Parameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| **Sigma** | 1.5-3 pixels | Size of expected puncta. Measure from your images! |
| **Threshold** | 1.5-3.0 | Higher = fewer, more confident detections |
| **Min Size** | 3-10 px² | Filter out noise |
| **Max Size** | 50-200 px² | Filter out artifacts |

#### Tracking Parameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| **Max Distance** | 5-15 pixels | How far particles can move between frames |

For fast-moving PIEZO1: use 10-15 pixels
For slow diffusion: use 3-5 pixels

#### Event Detection Parameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| **Event Threshold** | 1.3-2.0x | Fold-change above baseline for Ca²⁺ flickers |
| **Min Duration** | 2-5 frames | Minimum event length to avoid noise |

For transient flickers: 2-3 frames
For sustained signals: 5-10 frames

## Example Workflows

### Workflow 1: PIEZO1 Localization in Single Cells

**Goal**: Map where PIEZO1 localizes in neural stem cells

```
1. Load: PIEZO1-tdTomato TIRF image
2. Method: LoG
3. Sigma: 2.0 pixels
4. Threshold: 2.5
5. Tracking: OFF (single frame)
6. Export: ON
7. Analyze: detections.csv → plot spatial distribution
```

**Output**: 
- CSV with X,Y coordinates of all PIEZO1 puncta
- Intensity and size of each puncta
- Can map distribution, create density maps

### Workflow 2: PIEZO1 Diffusion and Mobility

**Goal**: Track PIEZO1 movement in live cells

```
1. Load: PIEZO1-tdTomato time-lapse (TIRF)
2. Method: LoG
3. Sigma: 2.0 pixels
4. Threshold: 2.0
5. Tracking: ON
6. Max Distance: 10 pixels
7. Export: ON
8. Analyze: tracks.csv → calculate MSD, diffusion coefficients
```

**Output**:
- tracks.csv with trajectory for each particle
- Can calculate:
  - Mean squared displacement (MSD)
  - Diffusion coefficients
  - Mobile vs. immobile fractions
  - Confinement analysis

### Workflow 3: Ca²⁺ Flicker Detection

**Goal**: Detect and characterize Piezo1-induced Ca²⁺ flickers

```
1. Load: R-GECO or GCaMP time-lapse
2. Method: LoG or DoG
3. Sigma: 2.5 pixels
4. Threshold: 1.5
5. Tracking: ON
6. Events: ON
7. Event Threshold: 1.5x
8. Event Duration: 3 frames
9. Export: ON
10. Analyze: events.csv → flicker statistics
```

**Output**:
- events.csv with all detected flickers
- Amplitude, duration, rise/decay times
- Spatial location of each flicker
- Can correlate with stimulation timing

### Workflow 4: PIEZO1-Cytoskeleton Colocalization

**Goal**: Study relationship between PIEZO1 and F-actin

```
1. Run Puncta Analyzer on PIEZO1 channel → export
2. Run Puncta Analyzer on F-actin channel → export
3. Compare detections.csv from both channels
4. Calculate colocalization:
   - Distance between nearest PIEZO1 and actin puncta
   - Pearson/Manders coefficients
   - Statistical significance
```

## Output Files

When you export results, the plugin creates:

```
~/FLIKA_analysis/<image_name>/
├── detections.csv       # All detected puncta
├── tracks.csv           # Particle trajectories (if tracking enabled)
├── events.csv           # Ca²⁺ flickers (if event detection enabled)
└── summary.txt          # Overall statistics
```

### detections.csv

Columns:
- `frame`: Time point
- `x`, `y`: Position (pixels)
- `amplitude`: Peak intensity above background
- `sigma`: Gaussian width (size)
- `intensity`: Integrated intensity
- `snr`: Signal-to-noise ratio
- `size`: Area (pixels²)

### tracks.csv

Columns:
- `track_id`: Unique track identifier
- `frame`: Time point
- `x`, `y`: Position
- `intensity`: Intensity at this time point
- All detection properties

### events.csv

Columns:
- `track_id`: Which particle had the flicker
- `start_frame`, `peak_frame`, `end_frame`: Timing
- `duration`: Length in frames
- `amplitude`: Peak intensity change
- `rise_time`, `decay_time`: Kinetics
- `x`, `y`: Spatial location

## Data Analysis Examples

### Python Analysis Script

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load results
tracks = pd.read_csv('~/FLIKA_analysis/my_image/tracks.csv')
events = pd.read_csv('~/FLIKA_analysis/my_image/events.csv')

# Calculate diffusion coefficient for each track
from scipy.stats import linregress

for track_id in tracks['track_id'].unique():
    track = tracks[tracks['track_id'] == track_id]
    
    # Calculate MSD
    time = track['frame'].values
    x = track['x'].values
    y = track['y'].values
    
    msd = []
    for dt in range(1, min(10, len(track))):
        dx = x[dt:] - x[:-dt]
        dy = y[dt:] - y[:-dt]
        msd.append(np.mean(dx**2 + dy**2))
    
    # Fit MSD = 4*D*t
    if len(msd) > 3:
        slope, _, _, _, _ = linregress(range(1, len(msd)+1), msd)
        D = slope / 4  # Diffusion coefficient (pixels²/frame)
        print(f"Track {track_id}: D = {D:.3f} pixels²/frame")

# Analyze Ca²⁺ flicker properties
print(f"\nCa²⁺ Flicker Statistics:")
print(f"Total events: {len(events)}")
print(f"Mean duration: {events['duration'].mean():.1f} frames")
print(f"Mean amplitude: {events['amplitude'].mean():.1f}")
print(f"Mean rise time: {events['rise_time'].mean():.1f} frames")

# Plot spatial distribution of flickers
plt.figure(figsize=(8, 6))
plt.scatter(events['x'], events['y'], c=events['amplitude'], 
            s=events['duration']*10, alpha=0.6, cmap='hot')
plt.colorbar(label='Amplitude')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.title('Spatial Distribution of Ca²⁺ Flickers')
plt.show()
```

## Tips for Best Results

### Image Acquisition

1. **Use appropriate frame rate**
   - Ca²⁺ flickers: 10-50 Hz
   - PIEZO1 tracking: 1-10 Hz

2. **Optimize SNR**
   - Higher laser power → better detection
   - But avoid photobleaching
   - Use anti-fade reagents

3. **Stable imaging**
   - Minimize drift
   - Use focus stabilization
   - Temperature control

### Detection Optimization

1. **Start conservative**
   - Higher threshold initially
   - Verify detections look good
   - Then lower threshold if needed

2. **Check different methods**
   - Try LoG first
   - If too noisy, try DoG or Wavelet
   - Compare results

3. **Validate on test regions**
   - Draw ROI on clear puncta
   - Check if algorithm detects them
   - Adjust parameters

### Tracking Optimization

1. **Frame rate matters**
   - Too slow → particles move too far → lost tracks
   - Too fast → unnecessary data, photobleaching
   - Optimize for your biology

2. **Max distance**
   - Estimate maximum displacement
   - Add 20% buffer
   - Too large → false assignments

3. **Quality control**
   - Check track lengths
   - Visual inspection
   - Filter short/questionable tracks

## Troubleshooting

### No Detections or Too Few

**Problem**: Algorithm doesn't detect any/many puncta

**Solutions**:
- Lower threshold (try 1.5, 1.0)
- Check if puncta visible in raw image
- Adjust sigma to match puncta size
- Try different detection method
- Check image contrast

### Too Many Detections (False Positives)

**Problem**: Detecting noise or background as puncta

**Solutions**:
- Increase threshold (try 3.0, 4.0)
- Increase min_size filter
- Try DoG or Wavelet (better background rejection)
- Pre-process image (background subtraction)
- Check SNR of real puncta

### Poor Tracking

**Problem**: Tracks are fragmented or wrong

**Solutions**:
- Increase max_distance if particles move far
- Decrease max_distance if false links
- Improve detection first (tracking can't fix bad detections)
- Check frame rate is appropriate
- Filter tracks by length afterwards

### Events Not Detected

**Problem**: Ca²⁺ flickers not showing up in events.csv

**Solutions**:
- Lower event_threshold (try 1.3x)
- Reduce min_duration (try 2 frames)
- Check if flickers visible in tracks
- Verify tracks are long enough
- Inspect intensity traces manually

### Memory Errors

**Problem**: Plugin crashes with large datasets

**Solutions**:
- Process smaller regions (crop spatially)
- Process shorter time windows
- Reduce image resolution if acceptable
- Close other programs
- Add more RAM

## Advanced Features

### Custom Analysis Pipeline

You can call the plugin programmatically:

```python
from plugins.puncta_analyzer import puncta_analyzer
import flika.global_vars as g

# Get current window
win = g.m.currentWindow

# Run analysis with specific parameters
result = puncta_analyzer(
    data_window=win,
    detection_method='log',
    sigma=2.0,
    threshold=2.5,
    min_size=5,
    max_size=100,
    do_tracking=True,
    max_distance=10.0,
    detect_events=True,
    event_threshold=1.5,
    event_duration=3,
    export_results=True
)

# Access results
detections = puncta_analyzer.detections
tracks = puncta_analyzer.tracks
events = puncta_analyzer.events
```

### Batch Processing

```python
import os
from flika.window import Window

# Process all TIFF files in directory
data_dir = "/path/to/your/tirf/images"
files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

for filename in files:
    print(f"Processing {filename}...")
    
    # Load image
    path = os.path.join(data_dir, filename)
    win = Window(path)
    
    # Run analysis
    puncta_analyzer(
        data_window=win,
        detection_method='log',
        sigma=2.0,
        threshold=2.0,
        min_size=5,
        max_size=100,
        do_tracking=True,
        max_distance=10.0,
        detect_events=True,
        event_threshold=1.5,
        event_duration=3,
        export_results=True
    )
    
    # Clean up
    win.close()

print("Batch processing complete!")
```

## Integration with Other Tools

### FLIKA Integration

Puncta Analyzer integrates seamlessly with other FLIKA features:

- **ROI Manager**: Create ROIs around detected puncta
- **Intensity Traces**: Extract traces from detected locations
- **Filters**: Pre-process images before detection
- **Other Plugins**: Use with beam splitter alignment, etc.

### External Tools

Export CSVs are compatible with:
- **MATLAB**: Easy import with `readtable()`
- **R**: Use `read.csv()`
- **Python/Pandas**: Standard format
- **GraphPad Prism**: For statistics
- **Excel**: For quick visualization

## Citations

If you use this plugin in your research, please cite:

1. **FLIKA**: 
   Ellefsen, K. L., et al. (2019). "Applications of FLIKA, a Python-based image processing and analysis platform, for studying local events of cellular calcium signaling." *BBA-Molecular Cell Research*, 1866(7), 1171-1179.

2. **Relevant Pathak Lab Papers**:
   - Pathak, M. M., et al. (2014). "Stretch-activated ion channel Piezo1 directs lineage choice in human neural stem cells." *PNAS*, 111(45), 16148-16153.
   - Ellefsen, K., et al. (2019). "Myosin-II mediated traction forces evoke localized Piezo1 Ca²⁺ flickers." *Communications Biology*, 2, 298.
   - Holt, J. R., et al. (2021). "Spatiotemporal dynamics of PIEZO1 localization controls keratinocyte migration during wound healing." *eLife*, 10, e65415.

## Support

### Getting Help

1. **Check documentation** (this file)
2. **FLIKA forums**: Community support
3. **Pathak Lab**: If you're at UCI!
4. **GitHub issues**: For bug reports

### Reporting Bugs

Please include:
- FLIKA version
- Plugin version
- Error message (if any)
- Example image (if possible)
- Parameters used

## Version History

### Version 1.0.0 (Current)
- Initial release
- Multi-algorithm detection (LoG, DoG, Threshold, Wavelet)
- Sub-pixel Gaussian fitting
- Particle tracking with LAP
- Ca²⁺ flicker detection
- CSV export
- Developed for Pathak Lab PIEZO1 research

## Future Enhancements

Potential features for future versions:
- 3D detection and tracking (Z-stacks)
- Machine learning-based detection
- Improved tracking algorithms (Hungarian, Bayesian)
- Interactive ROI editing
- Real-time visualization dashboard
- Integration with super-resolution methods
- FRET analysis capabilities

## Acknowledgments

- **Pathak Lab at UCI** for research inspiration and feedback
- **FLIKA Development Team** for the excellent platform
- **Scientific Community** for algorithm development

---

**Happy analyzing! May your puncta be plentiful and your Ca²⁺ flickers be frequent! 🔬✨**

For questions specific to PIEZO1 research, check out the Pathak Lab website: https://www.pathaklab-uci.com/
