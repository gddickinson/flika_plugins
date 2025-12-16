# Kymograph Analyzer - Example Scripts

## Example 1: Basic Ca²⁺ Flicker Analysis

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import flika.global_vars as g

# Get current window (your TIRF Ca²⁺ movie)
window = g.m.currentWindow

# Generate kymograph with standard settings
kymo_window = kymograph_analyzer(
    window=window,
    roi_type='line',
    width=7,
    temporal_binning=1,
    normalize=True,
    detrend=True,
    gaussian_sigma=1.0
)

# Detect Ca²⁺ flickers
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)

print(f"Detected {len(events)} Ca²⁺ flickers")

# Get statistics
stats = kymograph_analyzer.calculate_statistics()
print(f"Mean intensity: {stats['mean_intensity']:.3f}")
print(f"Intensity range: {stats['min_intensity']:.3f} - {stats['max_intensity']:.3f}")

# Create comprehensive analysis plot
kymograph_analyzer.plot_analysis()

# Export results
kymograph_analyzer.export_data("flicker_analysis_cell1")
```

## Example 2: Batch Processing Multiple Cells

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import pandas as pd

# List of your cell windows
cell_windows = [...]  # Your list of FLIKA windows

# Store results for all cells
all_results = []

# Process each cell
for i, cell_window in enumerate(cell_windows):
    print(f"Processing cell {i+1}/{len(cell_windows)}...")
    
    # Generate kymograph
    kymo = kymograph_analyzer(
        window=cell_window,
        width=7,
        normalize=True,
        detrend=True,
        gaussian_sigma=1.0
    )
    
    # Detect events
    events = kymograph_analyzer.detect_events(
        method='peaks',
        threshold=2.5,
        min_distance=3,
        min_duration=2
    )
    
    # Calculate statistics
    stats = kymograph_analyzer.calculate_statistics()
    
    # Store results
    all_results.append({
        'cell_id': i,
        'n_flickers': len(events),
        'mean_amplitude': np.mean([e['amplitude'] for e in events]) if events else 0,
        'mean_intensity': stats['mean_intensity'],
        'std_intensity': stats['std_intensity']
    })
    
    # Export individual cell data
    kymograph_analyzer.export_data(f"cell_{i:02d}_analysis")

# Create summary dataframe
df = pd.DataFrame(all_results)
print("\nSummary across all cells:")
print(df.describe())

# Save summary
df.to_csv("all_cells_summary.csv", index=False)
```

## Example 3: Comparing Control vs Treatment

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

def analyze_cell_group(windows, group_name):
    """Analyze a group of cells and return results."""
    results = []
    
    for i, window in enumerate(windows):
        kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)
        events = kymograph_analyzer.detect_events(
            method='peaks',
            threshold=2.5,
            min_distance=3,
            min_duration=2
        )
        
        if events:
            results.append({
                'group': group_name,
                'cell': i,
                'n_events': len(events),
                'mean_amplitude': np.mean([e['amplitude'] for e in events]),
                'mean_width': np.mean([e.get('width', np.nan) for e in events])
            })
    
    return pd.DataFrame(results)

# Analyze control group
control_windows = [...]  # Your control cell windows
control_df = analyze_cell_group(control_windows, 'Control')

# Analyze treatment group
treatment_windows = [...]  # Your treatment cell windows
treatment_df = analyze_cell_group(treatment_windows, 'Treatment')

# Combine results
all_data = pd.concat([control_df, treatment_df])

# Statistical comparison
control_events = control_df['n_events'].values
treatment_events = treatment_df['n_events'].values

t_stat, p_value = scipy_stats.ttest_ind(control_events, treatment_events)

print(f"Control: {control_events.mean():.2f} ± {control_events.std():.2f} events")
print(f"Treatment: {treatment_events.mean():.2f} ± {treatment_events.std():.2f} events")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")

# Save results
all_data.to_csv("control_vs_treatment.csv", index=False)
```

## Example 4: Analyzing Wave Velocities

```python
from plugins.kymograph_analyzer import kymograph_analyzer

# Generate kymograph
kymo = kymograph_analyzer(
    window=wave_movie,
    width=10,
    normalize=True
)

# Measure velocities
velocities = kymograph_analyzer.measure_velocities()

# Convert to physical units
pixel_size = 0.16  # µm/pixel (your microscope calibration)
frame_interval = 0.05  # seconds between frames

velocity_um_s = velocities['mean_velocity'] * pixel_size / frame_interval

print(f"Wave velocity: {velocity_um_s:.2f} µm/s")
print(f"Median velocity: {velocities['median_velocity'] * pixel_size / frame_interval:.2f} µm/s")
```

## Example 5: Flicker Frequency Analysis

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np

# Generate kymograph
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)

# Detect events
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)

# Calculate flicker frequency
n_events = len(events)
n_frames = kymo.shape[0]
frame_interval = 0.05  # seconds
total_time = n_frames * frame_interval  # seconds

# Spatial extent (convert pixels to µm)
pixel_size = 0.16  # µm/pixel
spatial_length = kymo.shape[1] * pixel_size  # µm

# Flicker frequency
frequency_per_um_per_s = n_events / (spatial_length * total_time)

print(f"Total flickers: {n_events}")
print(f"Total time: {total_time:.1f} s")
print(f"Spatial length: {spatial_length:.1f} µm")
print(f"Flicker frequency: {frequency_per_um_per_s:.3f} events/µm/s")
```

## Example 6: Spatial Distribution Analysis

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import matplotlib.pyplot as plt

# Generate kymograph
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)

# Detect events
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)

# Extract positions
positions = [e['position'] for e in events]

# Create histogram of spatial distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(positions, bins=20, alpha=0.7, color='blue')
plt.xlabel('Position along ROI (pixels)')
plt.ylabel('Number of events')
plt.title('Spatial Distribution of Ca²⁺ Flickers')
plt.grid(True, alpha=0.3)

# Cumulative distribution
plt.subplot(1, 2, 2)
sorted_positions = np.sort(positions)
cumulative = np.arange(1, len(sorted_positions) + 1) / len(sorted_positions)
plt.plot(sorted_positions, cumulative)
plt.xlabel('Position along ROI (pixels)')
plt.ylabel('Cumulative probability')
plt.title('Cumulative Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spatial_distribution.pdf')
plt.show()

# Test for clustering (Kolmogorov-Smirnov test)
from scipy.stats import kstest
# Compare to uniform distribution
ks_stat, ks_pvalue = kstest(np.array(positions) / kymo.shape[1], 'uniform')
print(f"K-S test for uniformity: p = {ks_pvalue:.4f}")
if ks_pvalue < 0.05:
    print("Events are NOT uniformly distributed (clustered or patterned)")
else:
    print("Events appear uniformly distributed")
```

## Example 7: Temporal Correlation Between Locations

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Generate kymograph
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)

# Select two spatial positions to compare
pos1 = kymo.shape[1] // 3  # Position at 1/3 along ROI
pos2 = 2 * kymo.shape[1] // 3  # Position at 2/3 along ROI

# Extract temporal traces
trace1 = kymo[:, pos1]
trace2 = kymo[:, pos2]

# Calculate cross-correlation
correlation = correlate(trace1, trace2, mode='full')
lags = np.arange(-len(trace1) + 1, len(trace1))

# Find peak correlation and lag
max_corr_idx = np.argmax(correlation)
max_lag = lags[max_corr_idx]
max_corr = correlation[max_corr_idx] / (np.std(trace1) * np.std(trace2) * len(trace1))

# Convert lag to time
frame_interval = 0.05  # seconds
time_lag = max_lag * frame_interval

print(f"Peak correlation: {max_corr:.3f}")
print(f"Time lag: {time_lag:.3f} seconds")

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(trace1)
plt.title(f'Position 1 (pixel {pos1})')
plt.xlabel('Time (frames)')
plt.ylabel('Intensity')

plt.subplot(1, 3, 2)
plt.plot(trace2)
plt.title(f'Position 2 (pixel {pos2})')
plt.xlabel('Time (frames)')
plt.ylabel('Intensity')

plt.subplot(1, 3, 3)
plt.plot(lags * frame_interval, correlation)
plt.axvline(time_lag, color='r', linestyle='--', label=f'Peak lag: {time_lag:.3f} s')
plt.xlabel('Time lag (seconds)')
plt.ylabel('Cross-correlation')
plt.title('Temporal Correlation')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temporal_correlation.pdf')
plt.show()
```

## Example 8: Event Properties Distribution

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import matplotlib.pyplot as plt

# Generate kymograph and detect events
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)

if not events:
    print("No events detected")
else:
    # Extract properties
    amplitudes = [e['amplitude'] for e in events]
    widths = [e.get('width', np.nan) for e in events if 'width' in e]
    times = [e['time'] for e in events]
    positions = [e['position'] for e in events]
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Amplitude distribution
    axes[0, 0].hist(amplitudes, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.mean(amplitudes), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(amplitudes):.3f}')
    axes[0, 0].set_xlabel('Amplitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Flicker Amplitude Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Width/duration distribution
    if widths:
        axes[0, 1].hist(widths, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].axvline(np.nanmean(widths), color='r', linestyle='--',
                          label=f'Mean: {np.nanmean(widths):.2f} frames')
        axes[0, 1].set_xlabel('Width (frames)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Flicker Duration Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Temporal distribution
    axes[1, 0].hist(times, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Time (frames)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Temporal Distribution of Events')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Amplitude vs Time
    scatter = axes[1, 1].scatter(times, amplitudes, c=positions, cmap='viridis', 
                                alpha=0.6, s=50)
    axes[1, 1].set_xlabel('Time (frames)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Amplitude Over Time')
    plt.colorbar(scatter, ax=axes[1, 1], label='Position')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('event_properties.pdf')
    plt.show()
    
    # Print summary statistics
    print(f"Event Summary (n={len(events)}):")
    print(f"Amplitude: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f}")
    if widths:
        print(f"Duration: {np.nanmean(widths):.2f} ± {np.nanstd(widths):.2f} frames")
```

## Example 9: Comparing Different Detection Thresholds

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import matplotlib.pyplot as plt

# Generate kymograph once
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)

# Try different thresholds
thresholds = [1.5, 2.0, 2.5, 3.0, 3.5]
results = []

for thresh in thresholds:
    events = kymograph_analyzer.detect_events(
        method='peaks',
        threshold=thresh,
        min_distance=3,
        min_duration=2
    )
    results.append({
        'threshold': thresh,
        'n_events': len(events),
        'mean_amplitude': np.mean([e['amplitude'] for e in events]) if events else 0
    })

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(thresholds, [r['n_events'] for r in results], 'o-', linewidth=2)
axes[0].set_xlabel('Threshold (STD)')
axes[0].set_ylabel('Number of Events Detected')
axes[0].set_title('Detection Sensitivity')
axes[0].grid(True, alpha=0.3)

axes[1].plot(thresholds, [r['mean_amplitude'] for r in results], 's-', 
            linewidth=2, color='green')
axes[1].set_xlabel('Threshold (STD)')
axes[1].set_ylabel('Mean Event Amplitude')
axes[1].set_title('Mean Amplitude of Detected Events')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('threshold_optimization.pdf')
plt.show()

# Print recommendations
optimal_thresh = thresholds[len(thresholds)//2]  # Middle value as starting point
print(f"Suggested starting threshold: {optimal_thresh}")
```

## Example 10: Advanced - Creating Publication Figure

```python
from plugins.kymograph_analyzer import kymograph_analyzer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 0.5

# Generate analysis
kymo = kymograph_analyzer(window, width=7, normalize=True, detrend=True)
events = kymograph_analyzer.detect_events(
    method='peaks',
    threshold=2.5,
    min_distance=3,
    min_duration=2
)
stats = kymograph_analyzer.calculate_statistics()

# Create figure with custom layout
fig = plt.figure(figsize=(7, 5))  # Nature/Science single column width
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])

# Kymograph
ax1 = plt.subplot(gs[0, 0])
im = ax1.imshow(kymo, aspect='auto', cmap='viridis', interpolation='nearest')
ax1.set_xlabel('Position (pixels)')
ax1.set_ylabel('Time (frames)')
ax1.set_title('A. Kymograph', loc='left', fontweight='bold')
cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Intensity', rotation=270, labelpad=15)

# Overlay detected events
if events:
    times = [e['time'] for e in events]
    positions = [e['position'] for e in events]
    ax1.scatter(positions, times, c='red', s=20, alpha=0.5, marker='x')

# Spatial profile
ax2 = plt.subplot(gs[0, 1])
spatial_profile = stats['temporal_mean']
ax2.plot(spatial_profile, np.arange(len(spatial_profile)), 'k-', linewidth=1)
ax2.set_xlabel('Intensity')
ax2.set_ylabel('Position (pixels)')
ax2.set_title('B. Spatial\nProfile', loc='left', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.invert_yaxis()

# Event distribution
ax3 = plt.subplot(gs[1, 0])
if events:
    positions = [e['position'] for e in events]
    ax3.hist(positions, bins=20, alpha=0.7, color='blue', edgecolor='black')
ax3.set_xlabel('Position (pixels)')
ax3.set_ylabel('Count')
ax3.set_title('C. Event Distribution', loc='left', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Statistics
ax4 = plt.subplot(gs[1, 1])
ax4.axis('off')
stats_text = f"""D. Statistics

Events: {len(events)}

Mean Int: {stats['mean_intensity']:.3f}
Std Int: {stats['std_intensity']:.3f}

Freq: {len(events)/kymo.shape[0]:.2f}
events/frame
"""
ax4.text(0.1, 0.5, stats_text, fontsize=7, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Tips for Using These Examples

1. **Copy and modify** - Use these as templates for your specific needs
2. **Adjust parameters** - Threshold, width, etc. depend on your data
3. **Save your settings** - Once optimized, use same parameters for all cells
4. **Batch processing** - Process multiple cells overnight
5. **Statistical rigor** - Always analyze ≥10-20 cells per condition

## Common Modifications

### Change pixel size / frame rate:
```python
pixel_size = 0.16  # µm/pixel - CHECK YOUR MICROSCOPE!
frame_interval = 0.05  # seconds - CHECK YOUR ACQUISITION SETTINGS!
```

### Adjust detection sensitivity:
```python
# More sensitive (more events, more false positives)
events = kymograph_analyzer.detect_events(threshold=1.5)

# Less sensitive (fewer events, high confidence)
events = kymograph_analyzer.detect_events(threshold=3.5)
```

### Process subset of frames:
```python
# Take only first 100 frames
subset_window = Window(original_window.image[:100], name="Subset")
kymo = kymograph_analyzer(subset_window, ...)
```

---

**Happy coding!** 🐍📊
