from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import zscore
from scipy import ndimage
import pywt  # PyWavelets for wavelet analysis
from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


class FlickerAnalyzer(BaseProcess_noPriorWindow):
    """
    Automated Flicker/Transient Event Detector using Temporal Signal Analysis
    
    Detects Ca2+ flickers, PIEZO1 activity transients, and other brief intensity 
    events WITHOUT requiring ROI placement or puncta detection. Instead, uses 
    pixel-wise temporal analysis to automatically identify when and where 
    transient events occur.
    
    Key Innovation: ROI-free, puncta-free detection
    - Analyzes every pixel's temporal behavior
    - Statistical detection of significant transients
    - Automatic spatial clustering of simultaneous events
    - Works on raw fluorescence movies
    
    Perfect for:
    - Ca2+ flickers and sparks
    - PIEZO1-mediated calcium transients
    - Spontaneous activity mapping
    - Mechanotransduction events
    - Any brief, localized fluorescence changes
    
    Methods Available:
    1. Temporal Derivative: Detects rapid intensity changes
    2. Z-Score Analysis: Statistical outlier detection
    3. Wavelet Detection: Multi-scale transient detection
    4. Variance-Based: Finds temporally variable regions
    
    Output:
    - Flicker event catalog (time, location, amplitude, duration)
    - Frequency maps showing where flickers occur
    - Amplitude maps showing flicker intensity
    - Temporal raster plots
    - Statistical summaries
    """
    
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.source_window = None
        self.flicker_events = []
        self.frequency_map = None
        self.amplitude_map = None
        self.temporal_activity = None
        self.analysis_results = {}
        
    def __call__(self, window, method='zscore', threshold=3.0, 
                 min_duration=2, max_duration=30, spatial_smoothing=1.0,
                 temporal_smoothing=0, baseline_window=50, 
                 min_flicker_size=4, merge_distance=10):
        """
        Detect flicker events using temporal analysis.
        
        Parameters:
        -----------
        window : Window
            Source fluorescence movie (must be 3D time series)
        method : str
            Detection method:
            - 'zscore': Statistical outlier detection (recommended)
            - 'derivative': Temporal derivative threshold
            - 'wavelet': Wavelet-based detection
            - 'variance': High temporal variance regions
        threshold : float
            Detection threshold (method-dependent)
            - zscore: z-score threshold (typically 2-4)
            - derivative: % change threshold
            - wavelet: coefficient threshold
            - variance: variance threshold
        min_duration : int
            Minimum flicker duration in frames
        max_duration : int
            Maximum flicker duration in frames
        spatial_smoothing : float
            Gaussian smoothing sigma for spatial filtering (reduces noise)
        temporal_smoothing : int
            Savitzky-Golay filter window for temporal smoothing (0=none)
        baseline_window : int
            Number of frames for baseline calculation (rolling)
        min_flicker_size : int
            Minimum spatial size of flicker in pixels
        merge_distance : float
            Merge events within this distance (pixels) and time
            
        Returns:
        --------
        dict : Analysis results with flicker events and statistics
        """
        if window is None:
            g.m.statusBar().showMessage("Error: No window selected")
            return None
            
        if window.image.ndim != 3:
            g.m.statusBar().showMessage("Error: Image must be a 3D time series")
            return None
            
        g.m.statusBar().showMessage(f"Analyzing flickers using {method} method...")
        t_start = time()
        
        self.source_window = window
        image = window.imageArray()
        n_frames, height, width = image.shape
        
        # Apply spatial smoothing to reduce noise
        if spatial_smoothing > 0:
            image_smooth = np.zeros_like(image)
            for t in range(n_frames):
                image_smooth[t] = gaussian_filter(image[t], sigma=spatial_smoothing)
            image = image_smooth
            
        # Detect transients using selected method
        g.m.statusBar().showMessage(f"Detecting transients with {method} method...")
        
        if method == 'zscore':
            transient_map = self.detect_zscore(
                image, threshold, baseline_window, temporal_smoothing
            )
        elif method == 'derivative':
            transient_map = self.detect_derivative(
                image, threshold, temporal_smoothing
            )
        elif method == 'wavelet':
            transient_map = self.detect_wavelet(
                image, threshold
            )
        elif method == 'variance':
            transient_map = self.detect_variance(
                image, threshold, baseline_window
            )
        else:
            g.m.statusBar().showMessage(f"Unknown method: {method}")
            return None
            
        # Identify individual flicker events
        g.m.statusBar().showMessage("Identifying flicker events...")
        self.flicker_events = self.identify_flicker_events(
            transient_map, image, min_duration, max_duration,
            min_flicker_size, merge_distance
        )
        
        if len(self.flicker_events) == 0:
            g.m.statusBar().showMessage("No flickers detected. Try adjusting parameters.")
            return None
            
        # Create summary maps
        g.m.statusBar().showMessage("Creating summary maps...")
        self.create_summary_maps(image.shape)
        
        # Calculate statistics
        self.analysis_results = self.calculate_statistics(image.shape)
        
        elapsed = time() - t_start
        g.m.statusBar().showMessage(
            f"Analysis complete: {len(self.flicker_events)} flickers detected "
            f"({elapsed:.2f} s)"
        )
        
        return self.analysis_results
        
    def detect_zscore(self, image, threshold, baseline_window, temporal_smoothing):
        """
        Detect transients using z-score analysis.
        
        For each pixel, calculate z-score relative to rolling baseline.
        Transients are pixels with z-score > threshold.
        """
        n_frames, height, width = image.shape
        transient_map = np.zeros_like(image, dtype=bool)
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                trace = image[:, y, x].astype(float)
                
                # Optional temporal smoothing
                if temporal_smoothing > 0:
                    if temporal_smoothing % 2 == 0:
                        temporal_smoothing += 1  # Must be odd
                    trace = savgol_filter(trace, temporal_smoothing, 2)
                
                # Calculate rolling baseline (mean) and std
                trace_padded = np.pad(trace, baseline_window//2, mode='edge')
                
                baseline = np.zeros(n_frames)
                baseline_std = np.zeros(n_frames)
                
                for t in range(n_frames):
                    window_start = t
                    window_end = t + baseline_window
                    window_data = trace_padded[window_start:window_end]
                    
                    baseline[t] = np.mean(window_data)
                    baseline_std[t] = np.std(window_data)
                
                # Calculate z-score
                baseline_std[baseline_std == 0] = 1  # Avoid division by zero
                z_scores = (trace - baseline) / baseline_std
                
                # Mark significant transients
                transient_map[:, y, x] = z_scores > threshold
                
        return transient_map
        
    def detect_derivative(self, image, threshold, temporal_smoothing):
        """
        Detect transients using temporal derivative.
        
        Identifies rapid increases in intensity (dF/dt).
        """
        n_frames, height, width = image.shape
        transient_map = np.zeros_like(image, dtype=bool)
        
        # Calculate temporal derivative
        derivative = np.diff(image, axis=0)
        
        # Normalize by baseline intensity to get % change
        baseline = image[:-1]
        baseline[baseline == 0] = 1
        percent_change = (derivative / baseline) * 100
        
        # Apply temporal smoothing if requested
        if temporal_smoothing > 0:
            if temporal_smoothing % 2 == 0:
                temporal_smoothing += 1
            for y in range(height):
                for x in range(width):
                    percent_change[:, y, x] = savgol_filter(
                        percent_change[:, y, x], temporal_smoothing, 2
                    )
        
        # Detect significant increases
        transient_frames = percent_change > threshold
        
        # Add back first frame (lost in diff)
        transient_map[1:] = transient_frames
        
        return transient_map
        
    def detect_wavelet(self, image, threshold):
        """
        Detect transients using wavelet analysis.
        
        Uses continuous wavelet transform to identify brief events.
        """
        n_frames, height, width = image.shape
        transient_map = np.zeros_like(image, dtype=bool)
        
        # Use Mexican Hat wavelet (good for transient detection)
        widths = np.arange(2, 10)  # Scales for transients of 2-10 frames
        
        for y in range(height):
            for x in range(width):
                trace = image[:, y, x].astype(float)
                
                # Continuous wavelet transform
                cwt_matrix = pywt.cwt(trace, widths, 'mexh')[0]
                
                # Find maximum wavelet coefficient at each timepoint
                max_coef = np.max(np.abs(cwt_matrix), axis=0)
                
                # Threshold wavelet coefficients
                transient_map[:, y, x] = max_coef > threshold
                
        return transient_map
        
    def detect_variance(self, image, threshold, window):
        """
        Detect regions with high temporal variance.
        
        Identifies pixels that vary significantly over time.
        """
        n_frames, height, width = image.shape
        
        # Calculate rolling variance
        variance_map = np.zeros((n_frames, height, width))
        
        trace_padded = np.pad(image, ((window//2, window//2), (0, 0), (0, 0)), 
                             mode='edge')
        
        for t in range(n_frames):
            window_start = t
            window_end = t + window
            window_data = trace_padded[window_start:window_end]
            variance_map[t] = np.var(window_data, axis=0)
            
        # Threshold based on variance
        transient_map = variance_map > threshold
        
        return transient_map
        
    def identify_flicker_events(self, transient_map, image, min_duration, 
                                max_duration, min_size, merge_distance):
        """
        Identify individual flicker events from transient map.
        
        Groups spatially and temporally connected transient pixels.
        """
        n_frames, height, width = transient_map.shape
        events = []
        
        # Process each frame
        for t in range(n_frames):
            # Find spatial clusters of active pixels at this timepoint
            labeled_frame, n_objects = label(transient_map[t])
            
            for obj_id in range(1, n_objects + 1):
                # Get pixels in this cluster
                mask = labeled_frame == obj_id
                
                # Check minimum size
                if np.sum(mask) < min_size:
                    continue
                    
                # Get centroid
                coords = np.where(mask)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                
                # Get intensity statistics
                intensities = image[t, mask]
                amplitude = np.mean(intensities)
                peak_intensity = np.max(intensities)
                
                # Check duration by looking forward in time
                duration = 1
                for dt in range(1, max_duration):
                    if t + dt >= n_frames:
                        break
                    # Check if transient continues near this location
                    y_min = max(0, int(centroid_y - merge_distance))
                    y_max = min(height, int(centroid_y + merge_distance))
                    x_min = max(0, int(centroid_x - merge_distance))
                    x_max = min(width, int(centroid_x + merge_distance))
                    
                    if np.any(transient_map[t + dt, y_min:y_max, x_min:x_max]):
                        duration += 1
                    else:
                        break
                        
                # Check duration criteria
                if duration < min_duration or duration > max_duration:
                    continue
                    
                # Create event
                event = {
                    'start_frame': t,
                    'duration': duration,
                    'centroid_x': centroid_x,
                    'centroid_y': centroid_y,
                    'amplitude': amplitude,
                    'peak_intensity': peak_intensity,
                    'size': np.sum(mask),
                    'end_frame': t + duration - 1
                }
                
                # Check if this overlaps with a recently detected event
                # (to avoid duplicate detection of same flicker)
                is_duplicate = False
                for prev_event in events[-20:]:  # Check recent events
                    if (abs(prev_event['centroid_x'] - centroid_x) < merge_distance and
                        abs(prev_event['centroid_y'] - centroid_y) < merge_distance and
                        abs(prev_event['start_frame'] - t) < duration):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    events.append(event)
                    
        return events
        
    def create_summary_maps(self, image_shape):
        """
        Create frequency and amplitude maps from detected events.
        """
        n_frames, height, width = image_shape
        
        # Frequency map: how many flickers at each location
        self.frequency_map = np.zeros((height, width))
        
        # Amplitude map: average amplitude of flickers at each location
        self.amplitude_map = np.zeros((height, width))
        amplitude_counts = np.zeros((height, width))
        
        # Temporal activity: number of active flickers at each frame
        self.temporal_activity = np.zeros(n_frames)
        
        for event in self.flicker_events:
            x, y = int(event['centroid_x']), int(event['centroid_y'])
            
            # Ensure coordinates are in bounds
            if 0 <= x < width and 0 <= y < height:
                self.frequency_map[y, x] += 1
                self.amplitude_map[y, x] += event['amplitude']
                amplitude_counts[y, x] += 1
                
            # Temporal activity
            for t in range(event['start_frame'], event['end_frame'] + 1):
                if t < n_frames:
                    self.temporal_activity[t] += 1
                    
        # Average amplitude
        amplitude_counts[amplitude_counts == 0] = 1  # Avoid division by zero
        self.amplitude_map = self.amplitude_map / amplitude_counts
        
        # Smooth maps for visualization
        self.frequency_map = gaussian_filter(self.frequency_map, sigma=2)
        self.amplitude_map = gaussian_filter(self.amplitude_map, sigma=2)
        
    def calculate_statistics(self, image_shape):
        """
        Calculate comprehensive statistics.
        """
        n_frames, height, width = image_shape
        
        stats = {}
        
        # Basic counts
        stats['n_flickers'] = len(self.flicker_events)
        stats['total_frames'] = n_frames
        
        # Temporal statistics
        durations = [e['duration'] for e in self.flicker_events]
        stats['mean_duration'] = np.mean(durations) if durations else 0
        stats['median_duration'] = np.median(durations) if durations else 0
        stats['std_duration'] = np.std(durations) if durations else 0
        
        # Amplitude statistics
        amplitudes = [e['amplitude'] for e in self.flicker_events]
        stats['mean_amplitude'] = np.mean(amplitudes) if amplitudes else 0
        stats['median_amplitude'] = np.median(amplitudes) if amplitudes else 0
        stats['std_amplitude'] = np.std(amplitudes) if amplitudes else 0
        
        # Size statistics
        sizes = [e['size'] for e in self.flicker_events]
        stats['mean_size'] = np.mean(sizes) if sizes else 0
        stats['median_size'] = np.median(sizes) if sizes else 0
        
        # Frequency statistics
        area_pixels = height * width
        stats['frequency_per_pixel_per_frame'] = len(self.flicker_events) / (area_pixels * n_frames)
        stats['mean_active_flickers_per_frame'] = np.mean(self.temporal_activity)
        stats['max_simultaneous_flickers'] = np.max(self.temporal_activity)
        
        # Spatial statistics
        if len(self.flicker_events) > 0:
            coords = np.array([[e['centroid_x'], e['centroid_y']] 
                              for e in self.flicker_events])
            
            # Spatial spread
            stats['spatial_extent_x'] = np.max(coords[:, 0]) - np.min(coords[:, 0])
            stats['spatial_extent_y'] = np.max(coords[:, 1]) - np.min(coords[:, 1])
            
            # Centroid of all activity
            stats['activity_center_x'] = np.mean(coords[:, 0])
            stats['activity_center_y'] = np.mean(coords[:, 1])
            
        # Inter-flicker intervals (at same location)
        stats['mean_inter_flicker_interval'] = self.calculate_inter_flicker_intervals()
        
        return stats
        
    def calculate_inter_flicker_intervals(self, distance_threshold=10):
        """
        Calculate inter-flicker intervals at similar locations.
        """
        if len(self.flicker_events) < 2:
            return 0
            
        intervals = []
        
        # Sort events by location and time
        events_sorted = sorted(self.flicker_events, 
                              key=lambda e: (e['centroid_x'], e['centroid_y'], 
                                           e['start_frame']))
        
        # Find events at similar locations
        for i, event1 in enumerate(events_sorted[:-1]):
            for event2 in events_sorted[i+1:]:
                # Check if at similar location
                dx = abs(event1['centroid_x'] - event2['centroid_x'])
                dy = abs(event1['centroid_y'] - event2['centroid_y'])
                
                if dx < distance_threshold and dy < distance_threshold:
                    # Calculate interval
                    interval = event2['start_frame'] - event1['end_frame']
                    if interval > 0:
                        intervals.append(interval)
                    break  # Only take next event at this location
                    
        return np.mean(intervals) if intervals else 0
        
    def plot_analysis(self):
        """
        Create comprehensive visualization of flicker analysis.
        """
        if not self.flicker_events:
            g.m.statusBar().showMessage("No flicker events to plot")
            return
            
        # Create figure with multiple subplots
        fig = Figure(figsize=(16, 12))
        
        image_shape = self.source_window.image.shape
        
        # 1. Frequency map
        ax1 = fig.add_subplot(3, 3, 1)
        im1 = ax1.imshow(self.frequency_map, cmap='hot', interpolation='bilinear')
        ax1.set_title(f'Flicker Frequency Map\n(Total: {len(self.flicker_events)} events)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        fig.colorbar(im1, ax=ax1, label='Number of Flickers')
        
        # Overlay event locations
        x_coords = [e['centroid_x'] for e in self.flicker_events]
        y_coords = [e['centroid_y'] for e in self.flicker_events]
        ax1.scatter(x_coords, y_coords, c='cyan', s=5, alpha=0.3, marker='o')
        
        # 2. Amplitude map
        ax2 = fig.add_subplot(3, 3, 2)
        im2 = ax2.imshow(self.amplitude_map, cmap='viridis', interpolation='bilinear')
        ax2.set_title('Average Flicker Amplitude Map')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        fig.colorbar(im2, ax=ax2, label='Mean Amplitude')
        
        # 3. Temporal activity
        ax3 = fig.add_subplot(3, 3, 3)
        frames = np.arange(len(self.temporal_activity))
        ax3.plot(frames, self.temporal_activity, 'b-', linewidth=1)
        ax3.fill_between(frames, self.temporal_activity, alpha=0.3)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Number of Active Flickers')
        ax3.set_title('Temporal Activity Profile')
        ax3.grid(True, alpha=0.3)
        
        # 4. Flicker raster plot
        ax4 = fig.add_subplot(3, 3, 4)
        for i, event in enumerate(self.flicker_events[:500]):  # Limit to 500 for visibility
            ax4.plot([event['start_frame'], event['end_frame']], 
                    [i, i], 'b-', linewidth=0.5)
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Event Number')
        ax4.set_title('Flicker Raster Plot\n(showing first 500 events)')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Duration distribution
        ax5 = fig.add_subplot(3, 3, 5)
        durations = [e['duration'] for e in self.flicker_events]
        ax5.hist(durations, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(self.analysis_results['mean_duration'], 
                   color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {self.analysis_results['mean_duration']:.1f} frames")
        ax5.set_xlabel('Duration (frames)')
        ax5.set_ylabel('Count')
        ax5.set_title('Flicker Duration Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Amplitude distribution
        ax6 = fig.add_subplot(3, 3, 6)
        amplitudes = [e['amplitude'] for e in self.flicker_events]
        ax6.hist(amplitudes, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(self.analysis_results['mean_amplitude'], 
                   color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {self.analysis_results['mean_amplitude']:.1f}")
        ax6.set_xlabel('Amplitude (intensity)')
        ax6.set_ylabel('Count')
        ax6.set_title('Flicker Amplitude Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Size distribution
        ax7 = fig.add_subplot(3, 3, 7)
        sizes = [e['size'] for e in self.flicker_events]
        ax7.hist(sizes, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax7.axvline(self.analysis_results['mean_size'], 
                   color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {self.analysis_results['mean_size']:.1f} px")
        ax7.set_xlabel('Size (pixels)')
        ax7.set_ylabel('Count')
        ax7.set_title('Flicker Size Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Statistics summary
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.axis('off')
        
        stats_text = "FLICKER STATISTICS\n" + "="*50 + "\n\n"
        stats_text += f"Total Flickers: {self.analysis_results['n_flickers']}\n"
        stats_text += f"Total Frames: {self.analysis_results['total_frames']}\n"
        stats_text += f"Frequency: {self.analysis_results['frequency_per_pixel_per_frame']:.2e} /px/frame\n\n"
        
        stats_text += "DURATION:\n"
        stats_text += f"  Mean: {self.analysis_results['mean_duration']:.2f} frames\n"
        stats_text += f"  Median: {self.analysis_results['median_duration']:.2f} frames\n"
        stats_text += f"  Std: {self.analysis_results['std_duration']:.2f} frames\n\n"
        
        stats_text += "AMPLITUDE:\n"
        stats_text += f"  Mean: {self.analysis_results['mean_amplitude']:.2f}\n"
        stats_text += f"  Median: {self.analysis_results['median_amplitude']:.2f}\n"
        stats_text += f"  Std: {self.analysis_results['std_amplitude']:.2f}\n\n"
        
        stats_text += "SPATIAL:\n"
        stats_text += f"  Mean Size: {self.analysis_results['mean_size']:.1f} pixels\n"
        stats_text += f"  Activity Center: ({self.analysis_results.get('activity_center_x', 0):.1f}, "
        stats_text += f"{self.analysis_results.get('activity_center_y', 0):.1f})\n\n"
        
        stats_text += "TEMPORAL:\n"
        stats_text += f"  Mean Active/Frame: {self.analysis_results['mean_active_flickers_per_frame']:.2f}\n"
        stats_text += f"  Max Simultaneous: {int(self.analysis_results['max_simultaneous_flickers'])}\n"
        stats_text += f"  Mean IFI: {self.analysis_results['mean_inter_flicker_interval']:.1f} frames\n"
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 9. Spatial-temporal scatter
        ax9 = fig.add_subplot(3, 3, 9)
        times = [e['start_frame'] for e in self.flicker_events]
        amplitudes = [e['amplitude'] for e in self.flicker_events]
        ax9.scatter(times, amplitudes, c='red', s=10, alpha=0.5)
        ax9.set_xlabel('Frame')
        ax9.set_ylabel('Amplitude')
        ax9.set_title('Flicker Amplitude Over Time')
        ax9.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Display figure
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("Flicker Analysis Results")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_gui)
        button_layout.addWidget(export_btn)
        
        create_windows_btn = QPushButton("Create Frequency/Amplitude Windows")
        create_windows_btn.clicked.connect(self.create_map_windows)
        button_layout.addWidget(create_windows_btn)
        
        layout.addLayout(button_layout)
        
        self.plot_window.setLayout(layout)
        self.plot_window.resize(1600, 1200)
        self.plot_window.show()
        
        g.m.statusBar().showMessage("Flicker analysis visualization created")
        
    def create_map_windows(self):
        """
        Create separate FLIKA windows for frequency and amplitude maps.
        """
        if self.frequency_map is None:
            g.m.statusBar().showMessage("No maps to create")
            return
            
        # Create frequency map window
        freq_win = Window(self.frequency_map, name="Flicker_Frequency_Map")
        
        # Create amplitude map window
        amp_win = Window(self.amplitude_map, name="Flicker_Amplitude_Map")
        
        g.m.statusBar().showMessage("Created frequency and amplitude map windows")
        
    def export_data(self, base_filename):
        """
        Export flicker analysis results.
        """
        import json
        
        # Export flicker events
        if self.flicker_events:
            df_events = pd.DataFrame(self.flicker_events)
            df_events.to_csv(f"{base_filename}_flickers.csv", index=False)
            
        # Export temporal activity
        df_temporal = pd.DataFrame({
            'frame': np.arange(len(self.temporal_activity)),
            'active_flickers': self.temporal_activity
        })
        df_temporal.to_csv(f"{base_filename}_temporal.csv", index=False)
        
        # Export frequency map
        np.save(f"{base_filename}_frequency_map.npy", self.frequency_map)
        
        # Export amplitude map
        np.save(f"{base_filename}_amplitude_map.npy", self.amplitude_map)
        
        # Export summary statistics as JSON
        summary = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                  for k, v in self.analysis_results.items()}
        
        with open(f"{base_filename}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        g.m.statusBar().showMessage(f"Data exported to {base_filename}_*")
        
    def gui(self):
        """
        Create graphical user interface.
        """
        self.gui_reset()
        
        # Window selector
        window_selector = WindowSelector()
        
        # Detection method
        self.method_combo = ComboBox()
        self.method_combo.addItems(['zscore', 'derivative', 'wavelet', 'variance'])
        
        # Threshold
        self.threshold_spin = pg.SpinBox(int=False, step=0.5, bounds=[0.5, 20], 
                                        value=3.0, decimals=1)
        
        # Duration parameters
        self.min_duration_spin = pg.SpinBox(int=True, step=1, bounds=[1, 50], value=2)
        self.max_duration_spin = pg.SpinBox(int=True, step=5, bounds=[5, 200], value=30)
        
        # Smoothing parameters
        self.spatial_smooth_spin = pg.SpinBox(int=False, step=0.5, bounds=[0, 10], 
                                             value=1.0, decimals=1)
        self.temporal_smooth_spin = pg.SpinBox(int=True, step=2, bounds=[0, 21], value=0)
        
        # Baseline window
        self.baseline_spin = pg.SpinBox(int=True, step=10, bounds=[10, 500], value=50)
        
        # Spatial parameters
        self.min_size_spin = pg.SpinBox(int=True, step=1, bounds=[1, 50], value=4)
        self.merge_distance_spin = pg.SpinBox(int=False, step=1, bounds=[1, 50], 
                                             value=10, decimals=0)
        
        # Analysis buttons
        self.analyze_button = QPushButton('Detect Flickers')
        self.analyze_button.clicked.connect(self.analyze_from_gui)
        
        self.plot_button = QPushButton('Visualize Results')
        self.plot_button.clicked.connect(self.plot_analysis)
        
        self.export_button = QPushButton('Export Data')
        self.export_button.clicked.connect(self.export_gui)
        
        # Build items list
        self.items.append({'name': 'window', 'string': 'Source Window',
                          'object': window_selector})
        self.items.append({'name': '', 'string': '--- Detection Method ---',
                          'object': None})
        self.items.append({'name': 'method', 'string': 'Detection Method',
                          'object': self.method_combo})
        self.items.append({'name': 'threshold', 'string': 'Threshold',
                          'object': self.threshold_spin})
        self.items.append({'name': '', 'string': '--- Duration Criteria ---',
                          'object': None})
        self.items.append({'name': 'min_duration', 'string': 'Min Duration (frames)',
                          'object': self.min_duration_spin})
        self.items.append({'name': 'max_duration', 'string': 'Max Duration (frames)',
                          'object': self.max_duration_spin})
        self.items.append({'name': '', 'string': '--- Smoothing ---',
                          'object': None})
        self.items.append({'name': 'spatial_smoothing', 'string': 'Spatial Smoothing (sigma)',
                          'object': self.spatial_smooth_spin})
        self.items.append({'name': 'temporal_smoothing', 'string': 'Temporal Smoothing (window, 0=none)',
                          'object': self.temporal_smooth_spin})
        self.items.append({'name': 'baseline_window', 'string': 'Baseline Window (frames)',
                          'object': self.baseline_spin})
        self.items.append({'name': '', 'string': '--- Spatial Parameters ---',
                          'object': None})
        self.items.append({'name': 'min_flicker_size', 'string': 'Min Flicker Size (pixels)',
                          'object': self.min_size_spin})
        self.items.append({'name': 'merge_distance', 'string': 'Merge Distance (pixels)',
                          'object': self.merge_distance_spin})
        self.items.append({'name': '', 'string': '--- Actions ---',
                          'object': None})
        self.items.append({'name': 'analyze', 'string': '',
                          'object': self.analyze_button})
        self.items.append({'name': 'plot', 'string': '',
                          'object': self.plot_button})
        self.items.append({'name': 'export', 'string': '',
                          'object': self.export_button})
        
        super().gui()
        
    def analyze_from_gui(self):
        """Run analysis from GUI parameters."""
        window = self.getValue('window')
        method = self.getValue('method')
        threshold = self.getValue('threshold')
        min_duration = self.getValue('min_duration')
        max_duration = self.getValue('max_duration')
        spatial_smoothing = self.getValue('spatial_smoothing')
        temporal_smoothing = self.getValue('temporal_smoothing')
        baseline_window = self.getValue('baseline_window')
        min_flicker_size = self.getValue('min_flicker_size')
        merge_distance = self.getValue('merge_distance')
        
        self(window, method, threshold, min_duration, max_duration,
             spatial_smoothing, temporal_smoothing, baseline_window,
             min_flicker_size, merge_distance)
             
    def export_gui(self):
        """Export data via file dialog."""
        filename, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Flicker Data", "",
            "All Files (*)"
        )
        if filename:
            self.export_data(filename)


# Create plugin instance
flicker_analyzer = FlickerAnalyzer()
