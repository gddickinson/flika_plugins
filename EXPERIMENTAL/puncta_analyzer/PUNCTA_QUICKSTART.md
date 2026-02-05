# Puncta Analyzer - Quick Start Guide

## For the Pathak Lab at UCI 🔬

Perfect for analyzing **PIEZO1 localization**, **Ca²⁺ flickers**, and **protein dynamics** in your TIRF microscopy data!

## Installation (2 Minutes)

1. **Copy files to FLIKA plugins directory:**
   ```
   <FLIKA_plugins>/puncta_analyzer/
   ├── __init__.py
   ├── puncta_analyzer.py
   ├── info.xml
   └── about.html
   ```

2. **Restart FLIKA**

3. **Find it:** Plugins → Puncta Analyzer

## Quick Analysis (5 Minutes)

### Detect PIEZO1-tdTomato Puncta

1. Load your TIRF image in FLIKA
2. Launch: **Plugins → Puncta Analyzer**
3. Settings:
   - Detection Method: **LoG**
   - Sigma: **2.0 pixels**
   - Threshold: **2.0**
   - Enable Tracking: **✓** (if time series)
   - Export to CSV: **✓**
4. Click **OK**
5. Results appear in `~/FLIKA_analysis/<image_name>/`

### Detect Ca²⁺ Flickers

1. Load GCaMP or R-GECO time-lapse
2. Launch plugin
3. Settings:
   - Detection Method: **LoG** or **DoG**
   - Sigma: **2.5 pixels**
   - Threshold: **1.5**
   - Enable Tracking: **✓**
   - Detect Ca²⁺ Flickers: **✓**
   - Event Threshold: **1.5x**
   - Event Duration: **3 frames**
   - Export: **✓**
4. Click **OK**
5. Check `events.csv` for flicker properties!

## What You Get

### Output Files (in ~/FLIKA_analysis/<image_name>/)

| File | Contains |
|------|----------|
| **detections.csv** | All detected puncta with x, y, intensity, size |
| **tracks.csv** | Particle trajectories over time |
| **events.csv** | Ca²⁺ flickers with amplitude, duration, kinetics |
| **summary.txt** | Overall statistics |

## Key Features for Your Research

### ✅ PIEZO1 Localization
- Automatically detect PIEZO1-GFP/tdTomato puncta
- Get X,Y coordinates for spatial mapping
- Measure intensity and size distributions
- Perfect for: "Where does PIEZO1 localize?"

### ✅ Single-Particle Tracking
- Track PIEZO1 movement frame-by-frame
- Calculate diffusion coefficients
- Identify mobile vs. immobile fractions
- Perfect for: "How does PIEZO1 move in the membrane?"

### ✅ Ca²⁺ Flicker Detection
- Automatically detect transient Ca²⁺ signals
- Measure amplitude, duration, rise/decay
- Map spatial distribution
- Perfect for: "Myosin-II mediated traction forces evoke localized Piezo1 Ca²⁺ flickers"

### ✅ Colocalization Analysis
- Detect puncta in two channels
- Export coordinates for comparison
- Calculate distances and correlations
- Perfect for: "Does PIEZO1 colocalize with cytoskeleton?"

## Common Use Cases

### 1. Map PIEZO1 Distribution
```
Single TIRF frame → LoG detection → 
Export detections.csv → Plot X,Y heatmap
```

### 2. Measure PIEZO1 Diffusion
```
Time-lapse TIRF → LoG + Tracking → 
Export tracks.csv → Calculate MSD & D
```

### 3. Quantify Ca²⁺ Flickers
```
GCaMP time-lapse → LoG + Event Detection → 
Export events.csv → Analyze amplitude & frequency
```

### 4. Study PIEZO1 Clustering
```
TIRF image → LoG detection → 
Export detections.csv → Pair correlation analysis
```

## Parameter Cheat Sheet

| Measurement | Method | Sigma | Threshold | Tracking | Events |
|-------------|--------|-------|-----------|----------|--------|
| PIEZO1 puncta (single frame) | LoG | 2.0 | 2.0-2.5 | OFF | OFF |
| PIEZO1 tracking (time-lapse) | LoG | 2.0 | 2.0 | ON (10px) | OFF |
| Ca²⁺ flickers (GCaMP/R-GECO) | LoG/DoG | 2.5 | 1.5 | ON (8px) | ON (1.5x, 3 frames) |
| Noisy TIRF data | Wavelet | 2.0 | 2.0 | ON | OFF |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Too few detections | Lower threshold (try 1.5 or 1.0) |
| Too many false positives | Increase threshold (try 3.0) |
| Fragmented tracks | Increase max distance |
| No events detected | Lower event threshold (try 1.3x) |

## Quick Python Analysis

```python
import pandas as pd
import numpy as np

# Load results
tracks = pd.read_csv('~/FLIKA_analysis/my_image/tracks.csv')
events = pd.read_csv('~/FLIKA_analysis/my_image/events.csv')

# Ca²⁺ flicker statistics
print(f"Total flickers: {len(events)}")
print(f"Mean amplitude: {events['amplitude'].mean():.2f}")
print(f"Mean duration: {events['duration'].mean():.1f} frames")

# PIEZO1 tracking: calculate mean displacement
for track_id in tracks['track_id'].unique():
    track = tracks[tracks['track_id'] == track_id]
    displacements = np.sqrt(np.diff(track['x'])**2 + np.diff(track['y'])**2)
    print(f"Track {track_id}: mean displacement = {displacements.mean():.2f} px/frame")
```

## Integration with Your Workflow

### Before Analysis
- Use **Advanced Beam Splitter** plugin for dual-channel alignment
- Apply background subtraction if needed
- Consider bleach correction for long time-lapse

### After Analysis
- Import CSVs into MATLAB/Python/R
- Calculate derived metrics (MSD, correlation functions)
- Combine with mechanical stimulation data
- Correlate with cell fate decisions

## Keyboard Shortcut

No shortcuts yet, but launch from: **Plugins → Puncta Analyzer**

## Perfect For These Papers

This plugin helps with experiments like:
- "Spatiotemporal dynamics of PIEZO1 localization" (Holt et al., 2021)
- "Myosin-II mediated traction forces evoke localized Piezo1 Ca²⁺ flickers" (Ellefsen et al., 2019)
- "Single-particle tracking reveals heterogeneous PIEZO1 diffusion" (bioRxiv, 2024)

## Example Output

After running analysis on PIEZO1-tdTomato time-lapse:
```
=== Puncta Analysis Results ===
n_frames: 100
n_detections_total: 847
n_detections_per_frame: 8.5
n_tracks: 23
mean_track_length: 36.8
n_events: 0  (enable if detecting Ca²⁺)
```

## Need Help?

1. **Read full documentation**: PUNCTA_README.md (comprehensive guide)
2. **Pathak Lab members**: Ask colleagues who've used it
3. **FLIKA forums**: Community support
4. **Parameter optimization**: Start conservative, iterate

## Batch Processing

```python
# Process all TIFF files in directory
import os
from flika.window import Window
from plugins.puncta_analyzer import puncta_analyzer

data_dir = "/path/to/tirf/data"
for filename in os.listdir(data_dir):
    if filename.endswith('.tif'):
        win = Window(os.path.join(data_dir, filename))
        puncta_analyzer(win, 'log', 2.0, 2.0, 5, 100, 
                       True, 10.0, True, 1.5, 3, True)
        win.close()
```

## Citations

Don't forget to cite:
- **FLIKA**: Ellefsen et al., 2019
- **Your awesome Pathak Lab papers!**

---

**Ready to analyze?** Load an image and give it a try! 🚀

For questions: Check PUNCTA_README.md or ask the lab!

*Plugin created specifically for studying PIEZO1 mechanotransduction at UCI* 🐻
