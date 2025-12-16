# Calcium Noise Analysis Plugin for FLIKA

A comprehensive calcium imaging noise analysis toolkit integrating traditional signal processing algorithms, noise characterization, and event detection methods.

## Features

### Noise Characterization
- **Power Spectrum Analysis**: Identify Ca²⁺-active pixels via PSD (Swaminathan et al. 2020)
- **SD Fluctuation Analysis**: FLIKA-style running variance with shot noise correction
- **Noise Metrics**: CASCADE noise metric (ν = σ_ΔF/F × √framerate)
- **Anscombe Transform**: Variance-stabilizing transform for Poisson noise

### Signal Processing
- **Butterworth Bandpass Filter**: Isolate calcium transient frequencies (0.1-5 Hz)
- **Baseline Estimation**: Multiple F₀ methods (percentile, maximin, photobleaching)
- **ΔF/F Calculation**: Normalized fluorescence with division-safe epsilon

### Event Detection & ROI Identification
- **Calcium Spark Detection**: Dual-threshold approach with variance stabilization
- **Local Correlation Image**: CaImAn/Suite2p-style ROI detection
- **Template Matching**: Exponential waveform detection (GCaMP6f/6s)

## Installation

### Method 1: Direct Installation

1. Download or clone this repository
2. Copy the `calcium_noise_analysis` folder to your FLIKA plugins directory:
   - **Windows**: `C:\Users\[YourUsername]\.FLIKA\plugins\`
   - **macOS/Linux**: `~/.FLIKA/plugins/`
3. Restart FLIKA

### Method 2: Using the Plugin Manager

1. Open FLIKA
2. Go to **Plugins → Plugin Manager**
3. Click **Install from GitHub**
4. Enter the repository URL
5. Click **Install**

## Quick Start

### Basic Workflow

```python
from flika import start_flika
from flika.process.file_ import open_file
import flika.plugins.calcium_noise_analysis as ca

# Start FLIKA
start_flika()

# Open your calcium imaging movie
window = open_file('path/to/your/movie.tif')

# Compute power spectrum map
ca.power_spectrum_map(low_freq_min=0.1, low_freq_max=5.0, high_freq_cutoff=50.0)

# Estimate baseline F₀
ca.estimate_baseline_f0(method='percentile', window_frames=1000)

# Compute ΔF/F
ca.compute_dff(f0_window=100, epsilon=1.0)

# Detect calcium sparks
ca.detect_calcium_sparks(intensity_thresh=2.0, peak_thresh=3.8, min_size=40)
```

## Usage Examples

### Example 1: Noise Characterization Pipeline

```python
# 1. Characterize noise level
ca.compute_noise_metric(baseline_frames=100)
# Output: Shows ν value per pixel (typical: 1-9)

# 2. Identify active regions
ca.power_spectrum_map(low_freq_min=0.1, low_freq_max=5.0)
# Output: PSM showing Ca²⁺-active pixels

# 3. Visualize fluctuations
ca.fluctuation_analysis(spatial_sigma=1.0, highpass_cutoff=0.5, window_size=30)
# Output: SD map highlighting transients
```

### Example 2: Event Detection Workflow

```python
# 1. Estimate baseline
ca.estimate_baseline_f0(method='percentile', window_frames=1000, percentile=8)
# Output: F₀ baseline movie

# 2. Compute ΔF/F
ca.compute_dff(f0_window=100, baseline_percentile=8, epsilon=1.0)
# Output: Normalized ΔF/F movie

# 3. Detect events
ca.detect_calcium_sparks(intensity_thresh=2.0, peak_thresh=3.8, min_size=40)
# Output: Labeled spark regions
```

### Example 3: Preprocessing for CASCADE

```python
# 1. Bandpass filter
ca.butterworth_bandpass(low_freq=0.1, high_freq=5.0, order=2)

# 2. Estimate baseline
ca.estimate_baseline_f0(method='maximin', window_frames=300)

# 3. Compute ΔF/F
ca.compute_dff(f0_window=100, epsilon=1.0)

# 4. Check noise level
ca.compute_noise_metric(baseline_frames=100)
# Use ν value to select appropriate CASCADE model
```

## Operations Reference

### Power Spectrum Map

```python
power_spectrum_map(low_freq_min=0.1, low_freq_max=5.0, high_freq_cutoff=50.0, 
                   nperseg=256, keepSourceWindow=False)
```

Identifies Ca²⁺-active pixels by comparing power in calcium frequency band (0.1-5 Hz) to high-frequency shot noise (>50 Hz).

**Parameters:**
- `low_freq_min`: Lower calcium frequency (Hz), default: 0.1
- `low_freq_max`: Upper calcium frequency (Hz), default: 5.0
- `high_freq_cutoff`: Shot noise threshold (Hz), default: 50.0
- `nperseg`: Welch segment length (power of 2), default: 256

### Fluctuation Analysis

```python
fluctuation_analysis(spatial_sigma=1.0, highpass_cutoff=0.5, window_size=30,
                     apply_shot_noise=False, shot_noise_factor=0.0, keepSourceWindow=False)
```

FLIKA-style SD fluctuation analysis highlighting transient local signals.

**Parameters:**
- `spatial_sigma`: Gaussian blur sigma, default: 1.0
- `highpass_cutoff`: High-pass filter frequency (Hz), default: 0.5
- `window_size`: Running variance window (frames), default: 30
- `shot_noise_factor`: Shot noise correction factor, default: 0.0

### Anscombe Transform

```python
anscombe_transform(inverse=False, keepSourceWindow=False)
```

Variance-stabilizing transform for Poisson noise: y = 2√(x + 3/8)

### Estimate Baseline F₀

```python
estimate_baseline_f0(method='percentile', window_frames=1000, percentile=8, keepSourceWindow=False)
```

Estimate baseline fluorescence using percentile, maximin, or photobleaching methods.

**Methods:**
- `'percentile'`: Sliding percentile (robust to transients)
- `'maximin'`: Suite2p max(min()) method
- `'photobleaching'`: Double exponential fit

### Compute ΔF/F

```python
compute_dff(f0_window=100, baseline_percentile=8, epsilon=1.0, keepSourceWindow=False)
```

Calculate normalized fluorescence: ΔF/F = (F - F₀) / max(F₀, ε)

### Detect Calcium Sparks

```python
detect_calcium_sparks(intensity_thresh=2.0, peak_thresh=3.8, min_size=40,
                      median_filter_size=3, uniform_filter_size=3, keepSourceWindow=False)
```

Dual-threshold spark detection with variance stabilization.

**Thresholds:**
- `intensity_thresh`: Boundary threshold (σ), default: 2.0
- `peak_thresh`: Peak confirmation (σ), default: 3.8
- `min_size`: Minimum event size (pixels), default: 40

### Local Correlation Image

```python
local_correlation_image(neighborhood=1, keepSourceWindow=False)
```

Compute local correlation for ROI detection (1 = 3×3, 2 = 5×5).

### Compute Noise Metric

```python
compute_noise_metric(baseline_frames=100, keepSourceWindow=False)
```

Calculate CASCADE noise metric: ν = σ_ΔF/F × √framerate

**Interpretation:**
- ν ≈ 1-2: Low noise (excellent quality)
- ν ≈ 3-5: Moderate noise (typical)
- ν ≈ 6-9: High noise (challenging)

### Butterworth Bandpass

```python
butterworth_bandpass(low_freq=0.1, high_freq=5.0, order=2, keepSourceWindow=False)
```

Bandpass filter isolating calcium transient frequencies.

## Module Architecture

The plugin uses a modular design for easy testing and extension:

```
calcium_noise_analysis/
├── calcium_noise_analysis.py    # Main plugin with BaseProcess classes
├── power_spectrum_module.py     # PSD analysis and bandpass filtering
├── fluctuation_module.py        # SD fluctuation analysis
├── shot_noise_module.py         # Anscombe transform and utilities
├── baseline_module.py           # F₀ estimation and ΔF/F
├── event_detection_module.py   # Spark detection algorithms
├── correlation_module.py        # Correlation image computation
├── info.xml                     # Plugin metadata
├── about.html                   # Documentation
└── README.md                    # This file
```

### Unit Testing

Each module includes built-in unit tests. Run tests with:

```bash
python power_spectrum_module.py
python fluctuation_module.py
python shot_noise_module.py
python baseline_module.py
python event_detection_module.py
python correlation_module.py
```

## TIRF Microscopy Applications

This plugin is optimized for TIRF microscopy workflows:

- **Camera Noise Handling**: Anscombe transform for low-photon imaging
- **Spark Detection**: Optimized for CICR (Ca²⁺-induced Ca²⁺ release)
- **Photobleaching Correction**: Essential for long recordings
- **Spatial Filtering**: Matched to TIRF point spread function
- **Noise Quantification**: CASCADE metric for quality assessment

## Requirements

- FLIKA >= 0.2.25
- NumPy
- SciPy
- Python 3.6+

## Performance Notes

- All operations work on both single images and multi-frame stacks
- Processing is frame-by-frame to minimize memory usage
- For very large stacks (>10GB), consider processing in chunks
- Correlation image computation can be slow; consider downsampling first
- Default framerate (30 Hz) used if metadata unavailable

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Plugin doesn't appear | Check installation path; restart FLIKA |
| Framerate not detected | Uses 30 Hz default; set Window.framerate manually |
| No sparks detected | Lower `peak_thresh`; verify ΔF/F is positive |
| Correlation image zeros | Need >100 frames; check for flat pixels |
| High noise metric (ν>10) | Consider binning or temporal filtering first |
| Photobleaching fit fails | Automatically falls back to percentile method |

## Research Foundation

This plugin implements methods from:

- **CASCADE** (Rupprecht et al., Nat. Methods 2021) - Noise-matched training
- **Swaminathan et al. 2020** (J. Physiol.) - Power spectral density analysis
- **Lock & Parker 2020** - SD fluctuation analysis (FLIKA)
- **Suite2p** (Pachitariu et al., bioRxiv 2017) - Maximin baseline
- **CaImAn** (Giovannucci et al., eLife 2019) - Correlation images

## Citation

When using FLIKA, please cite:

> Ellefsen, K., Settle, B., Parker, I. & Smith, I. An algorithm for automated detection, localization and measurement of local calcium signals from camera-based imaging. *Cell Calcium*. 56:147-156, 2014

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This plugin is distributed under the MIT license.

## Support

- **GitHub**: [FLIKA Repository](https://github.com/flika-org/flika)
- **Documentation**: [FLIKA Docs](https://flika-org.github.io/)
- **Issues**: Use GitHub issue tracker

---

**Version**: 1.0.0  
**Author**: George  
**Last Updated**: 2024
