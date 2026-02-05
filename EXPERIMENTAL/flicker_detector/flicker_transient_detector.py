from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import zscore
from scipy.optimize import curve_fit
from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import makeROI
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import makeROI


class FlickerTransientDetector(BaseProcess_noPriorWindow):
    """
    Advanced Flicker and Transient Event Detector
    
    Automated detection and characterization of transient intensity increases
    such as Ca2+ flickers, PIEZO1-mediated calcium signals, and other rapid events.
    
    Designed for the Medha Pathak Lab at UCI.
    
    Perfect for analyzing:
    - Ca2+ flickers and sparks
    - PIEZO1-mediated calcium transients
    - Localized signaling events
    - Membrane potential changes
    - Protein recruitment events
    - Any transient intensity increases
    
    Features:
    - Multiple detection algorithms (threshold, derivative, wavelet, z-score)
    - Comprehensive event characterization (amplitude, duration, kinetics)
    - Regional and ROI-based analysis
    - Temporal pattern analysis (frequency, clustering)
    - Statistical comparisons
    - Raster plots and event maps
    - Publication-quality visualizations
    - Complete data export
    
    Detection Methods:
    1. Threshold: Simple ΔF/F₀ crossing (fast, intuitive)
    2. Derivative: Detects rapid changes (good for sharp events)
    3. Z-score: Statistical significance (handles variable baselines)
    4. Wavelet: Sophisticated multi-scale detection (best for noisy data)
    
    Input: Time series (single ROI trace or full movie)
    Output: Detected events with properties, statistics, and visualizations
    """
    
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.source_window = None
        self.traces = []  # List of intensity traces
        self.roi_names = []  # Names of ROIs
        self.detected_events = []  # List of detected events per trace
        self.event_properties = []  # Detailed properties of each event
        self.analysis_results = {}
        
    def __call__(self, window, method='threshold', threshold=2.0, 
                 min_duration=3, max_duration=100, smooth_sigma=1.0,
                 baseline_method='percentile', baseline_percentile=10,
                 framerate=None, use_rois=True, analyze_regions=False):
        """
        Detect flickers/transients in time series data.
        
        Parameters:
        -----------
        window : Window
            Source window (must be time series)
        method : str
            Detection method ('threshold', 'derivative', 'zscore', 'wavelet')
        threshold : float
            Detection threshold (meaning varies by method)
            - 'threshold': ΔF/F₀ threshold (e.g., 2.0 = 200% increase)
            - 'derivative': Minimum rate of change
            - 'zscore': Z-score threshold (e.g., 3.0 = 3 std above mean)
            - 'wavelet': Coefficient threshold
        min_duration : int
            Minimum event duration in frames
        max_duration : int
            Maximum event duration in frames
        smooth_sigma : float
            Gaussian smoothing sigma (0 = no smoothing)
        baseline_method : str
            Method for baseline calculation ('percentile', 'mean', 'median', 'rolling')
        baseline_percentile : float
            Percentile for baseline (if using percentile method)
        framerate : float
            Acquisition rate in Hz (None = use window framerate or assume 1 Hz)
        use_rois : bool
            Use ROIs if present (True) or analyze full frame (False)
        analyze_regions : bool
            Divide frame into regions for spatial analysis
            
        Returns:
        --------
        dict : Analysis results with detected events and statistics
        """
        if window is None:
            g.m.statusBar().showMessage("Error: No window selected")
            return None
            
        if window.image.ndim != 3:
            g.m.statusBar().showMessage("Error: Image must be a time series (3D)")
            return None
            
        g.m.statusBar().showMessage("Detecting flickers/transients...")
        t_start = time()
        
        self.source_window = window
        image = window.imageArray()
        n_frames = image.shape[0]
        
        # Get framerate
        if framerate is None:
            if hasattr(window, 'framerate') and window.framerate > 0:
                framerate = window.framerate
            else:
                framerate = 1.0  # Default 1 Hz
                g.m.statusBar().showMessage(f"Warning: No framerate found, assuming {framerate} Hz")
        
        self.framerate = framerate
        
        # Extract traces
        if use_rois and hasattr(window, 'rois') and len(window.rois) > 0:
            # Use ROI traces
            self.traces = []
            self.roi_names = []
            for i, roi in enumerate(window.rois):
                trace = roi.getTrace()
                if trace is not None and len(trace) == n_frames:
                    self.traces.append(trace)
                    self.roi_names.append(f"ROI_{i+1}")
                    
            if len(self.traces) == 0:
                g.m.statusBar().showMessage("Error: Could not extract ROI traces")
                return None
                
        elif analyze_regions:
            # Divide frame into regions
            self.traces, self.roi_names = self.extract_regional_traces(image)
            
        else:
            # Use whole frame average
            trace = np.mean(image, axis=(1, 2))
            self.traces = [trace]
            self.roi_names = ['Whole Frame']
            
        # Detect events in each trace
        self.detected_events = []
        self.event_properties = []
        
        for i, trace in enumerate(self.traces):
            # Calculate baseline
            baseline = self.calculate_baseline(trace, baseline_method, baseline_percentile)
            
            # Smooth if requested
            if smooth_sigma > 0:
                trace_smooth = gaussian_filter1d(trace, sigma=smooth_sigma)
            else:
                trace_smooth = trace.copy()
                
            # Detect events
            events = self.detect_events(
                trace_smooth, baseline, method, threshold, 
                min_duration, max_duration
            )
            
            # Characterize each event
            props = []
            for event in events:
                prop = self.characterize_event(
                    trace, baseline, event['start'], event['end'], framerate
                )
                prop['roi_name'] = self.roi_names[i]
                prop['roi_index'] = i
                props.append(prop)
                
            self.detected_events.append(events)
            self.event_properties.extend(props)
            
        # Compile analysis results
        self.analysis_results = self.compile_results()
        
        elapsed = time() - t_start
        total_events = sum(len(events) for events in self.detected_events)
        g.m.statusBar().showMessage(
            f"Detection complete: {total_events} events in {len(self.traces)} traces ({elapsed:.2f} s)"
        )
        
        return self.analysis_results
        
    def extract_regional_traces(self, image, n_regions=9):
        """
        Divide frame into rectangular regions and extract average traces.
        """
        rows = int(np.sqrt(n_regions))
        cols = int(np.ceil(n_regions / rows))
        
        n_frames, height, width = image.shape
        
        row_edges = np.linspace(0, height, rows + 1, dtype=int)
        col_edges = np.linspace(0, width, cols + 1, dtype=int)
        
        traces = []
        names = []
        
        for i in range(rows):
            for j in range(cols):
                y_start, y_end = row_edges[i], row_edges[i+1]
                x_start, x_end = col_edges[j], col_edges[j+1]
                
                region = image[:, y_start:y_end, x_start:x_end]
                trace = np.mean(region, axis=(1, 2))
                
                traces.append(trace)
                names.append(f"Region_{i}_{j}")
                
        return traces, names
        
    def calculate_baseline(self, trace, method='percentile', percentile=10):
        """
        Calculate baseline fluorescence.
        """
        if method == 'percentile':
            baseline = np.percentile(trace, percentile)
        elif method == 'mean':
            baseline = np.mean(trace)
        elif method == 'median':
            baseline = np.median(trace)
        elif method == 'rolling':
            # Rolling minimum with large window
            window = max(100, len(trace) // 10)
            baseline = pd.Series(trace).rolling(window, center=True, min_periods=1).min().values
        else:
            baseline = np.percentile(trace, 10)
            
        return baseline
        
    def detect_events(self, trace, baseline, method, threshold, 
                     min_duration, max_duration):
        """
        Detect events using specified method.
        """
        if method == 'threshold':
            events = self.detect_threshold(trace, baseline, threshold, min_duration, max_duration)
        elif method == 'derivative':
            events = self.detect_derivative(trace, baseline, threshold, min_duration, max_duration)
        elif method == 'zscore':
            events = self.detect_zscore(trace, threshold, min_duration, max_duration)
        elif method == 'wavelet':
            events = self.detect_wavelet(trace, baseline, threshold, min_duration, max_duration)
        else:
            events = self.detect_threshold(trace, baseline, threshold, min_duration, max_duration)
            
        return events
        
    def detect_threshold(self, trace, baseline, threshold, min_duration, max_duration):
        """
        Simple threshold crossing detection (ΔF/F₀ method).
        """
        # Calculate ΔF/F₀
        if isinstance(baseline, np.ndarray):
            # Rolling baseline
            df_f0 = (trace - baseline) / baseline
        else:
            # Single baseline value
            df_f0 = (trace - baseline) / baseline
            
        # Find regions above threshold
        above_threshold = df_f0 > threshold
        
        # Find event boundaries
        events = []
        in_event = False
        start_idx = 0
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                # Start of event
                start_idx = i
                in_event = True
            elif not above_threshold[i] and in_event:
                # End of event
                duration = i - start_idx
                if min_duration <= duration <= max_duration:
                    events.append({
                        'start': start_idx,
                        'end': i,
                        'duration': duration
                    })
                in_event = False
                
        # Check if last event extends to end
        if in_event:
            duration = len(trace) - start_idx
            if min_duration <= duration <= max_duration:
                events.append({
                    'start': start_idx,
                    'end': len(trace),
                    'duration': duration
                })
                
        return events
        
    def detect_derivative(self, trace, baseline, threshold, min_duration, max_duration):
        """
        Detect rapid increases using first derivative.
        """
        # Calculate first derivative
        derivative = np.gradient(trace)
        
        # Find peaks in derivative (rapid increases)
        peaks, properties = find_peaks(
            derivative, 
            height=threshold,
            distance=min_duration
        )
        
        # For each peak, find event boundaries
        events = []
        for peak_idx in peaks:
            # Search backward for start
            start_idx = peak_idx
            for i in range(peak_idx - 1, max(0, peak_idx - max_duration), -1):
                if trace[i] <= baseline or derivative[i] < 0:
                    start_idx = i
                    break
                    
            # Search forward for end
            end_idx = peak_idx
            for i in range(peak_idx + 1, min(len(trace), peak_idx + max_duration)):
                if trace[i] <= baseline:
                    end_idx = i
                    break
                    
            duration = end_idx - start_idx
            if min_duration <= duration <= max_duration:
                events.append({
                    'start': start_idx,
                    'end': end_idx,
                    'duration': duration
                })
                
        return events
        
    def detect_zscore(self, trace, threshold, min_duration, max_duration):
        """
        Detect events using z-score (statistical significance).
        """
        # Calculate z-scores
        z_scores = zscore(trace)
        
        # Find regions above threshold
        above_threshold = z_scores > threshold
        
        # Find event boundaries (same as threshold method)
        events = []
        in_event = False
        start_idx = 0
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                start_idx = i
                in_event = True
            elif not above_threshold[i] and in_event:
                duration = i - start_idx
                if min_duration <= duration <= max_duration:
                    events.append({
                        'start': start_idx,
                        'end': i,
                        'duration': duration
                    })
                in_event = False
                
        if in_event:
            duration = len(trace) - start_idx
            if min_duration <= duration <= max_duration:
                events.append({
                    'start': start_idx,
                    'end': len(trace),
                    'duration': duration
                })
                
        return events
        
    def detect_wavelet(self, trace, baseline, threshold, min_duration, max_duration):
        """
        Wavelet-based detection for multi-scale analysis.
        Uses continuous wavelet transform with Ricker (Mexican hat) wavelet.
        """
        from scipy import signal
        
        # Use Ricker wavelet for event detection
        # Scale range based on expected durations
        scales = np.arange(min_duration, min(max_duration, len(trace) // 4))
        
        if len(scales) == 0:
            return []
            
        # Compute continuous wavelet transform
        coefficients, frequencies = signal.cwt(trace, signal.ricker, scales)
        
        # Find peaks in wavelet coefficients
        max_coeffs = np.max(np.abs(coefficients), axis=0)
        
        peaks, _ = find_peaks(max_coeffs, height=threshold * np.std(max_coeffs))
        
        # For each peak, determine event boundaries
        events = []
        for peak_idx in peaks:
            # Find the scale with maximum coefficient
            scale_idx = np.argmax(np.abs(coefficients[:, peak_idx]))
            scale = scales[scale_idx]
            
            # Event boundaries based on scale
            start_idx = max(0, peak_idx - scale // 2)
            end_idx = min(len(trace), peak_idx + scale // 2)
            
            # Refine boundaries using trace values
            while start_idx > 0 and trace[start_idx] > baseline:
                start_idx -= 1
            while end_idx < len(trace) and trace[end_idx] > baseline:
                end_idx += 1
                
            duration = end_idx - start_idx
            if min_duration <= duration <= max_duration:
                events.append({
                    'start': start_idx,
                    'end': end_idx,
                    'duration': duration
                })
                
        return events
        
    def characterize_event(self, trace, baseline, start, end, framerate):
        """
        Extract detailed properties of a single event.
        """
        event_trace = trace[start:end+1]
        
        if isinstance(baseline, np.ndarray):
            baseline_value = np.mean(baseline[start:end+1])
        else:
            baseline_value = baseline
            
        # Basic properties
        peak_idx = np.argmax(event_trace)
        peak_value = event_trace[peak_idx]
        amplitude = peak_value - baseline_value
        duration_frames = end - start
        duration_time = duration_frames / framerate
        
        # ΔF/F₀
        df_f0 = amplitude / baseline_value if baseline_value > 0 else 0
        
        # Rise time (10% to 90% of peak)
        rise_start = baseline_value + 0.1 * amplitude
        rise_end = baseline_value + 0.9 * amplitude
        
        rise_indices = np.where((event_trace >= rise_start) & (event_trace <= rise_end))[0]
        if len(rise_indices) > 1:
            rise_time_frames = rise_indices[-1] - rise_indices[0]
            rise_time = rise_time_frames / framerate
        else:
            rise_time = 0
            
        # Decay time (try exponential fit)
        decay_trace = event_trace[peak_idx:]
        if len(decay_trace) > 3:
            try:
                # Exponential decay: y = A * exp(-t/tau) + C
                t_decay = np.arange(len(decay_trace)) / framerate
                
                def exp_decay(t, A, tau, C):
                    return A * np.exp(-t / tau) + C
                    
                popt, _ = curve_fit(
                    exp_decay, t_decay, decay_trace,
                    p0=[amplitude, 0.1, baseline_value],
                    maxfev=1000,
                    bounds=([0, 0.01, 0], [np.inf, 10, np.inf])
                )
                decay_tau = popt[1]
            except:
                # Fallback: 90% to 10% decay
                decay_start = baseline_value + 0.9 * amplitude
                decay_end = baseline_value + 0.1 * amplitude
                decay_indices = np.where((decay_trace <= decay_start) & (decay_trace >= decay_end))[0]
                if len(decay_indices) > 1:
                    decay_tau = (decay_indices[-1] - decay_indices[0]) / framerate
                else:
                    decay_tau = 0
        else:
            decay_tau = 0
            
        # Area under curve (integral)
        auc = np.trapz(event_trace - baseline_value) / framerate
        
        return {
            'start_frame': start,
            'end_frame': end,
            'peak_frame': start + peak_idx,
            'duration_frames': duration_frames,
            'duration_time': duration_time,
            'baseline': baseline_value,
            'peak_intensity': peak_value,
            'amplitude': amplitude,
            'df_f0': df_f0,
            'rise_time': rise_time,
            'decay_tau': decay_tau,
            'auc': auc
        }
        
    def compile_results(self):
        """
        Compile comprehensive analysis results.
        """
        results = {}
        
        # Overall statistics
        total_events = len(self.event_properties)
        results['total_events'] = total_events
        results['n_traces'] = len(self.traces)
        results['n_frames'] = len(self.traces[0]) if self.traces else 0
        results['duration_total'] = results['n_frames'] / self.framerate
        results['framerate'] = self.framerate
        
        if total_events == 0:
            return results
            
        # Event statistics
        df = pd.DataFrame(self.event_properties)
        
        for prop in ['duration_time', 'amplitude', 'df_f0', 'rise_time', 'decay_tau', 'auc']:
            if prop in df.columns:
                results[f'{prop}_mean'] = df[prop].mean()
                results[f'{prop}_median'] = df[prop].median()
                results[f'{prop}_std'] = df[prop].std()
                results[f'{prop}_min'] = df[prop].min()
                results[f'{prop}_max'] = df[prop].max()
                
        # Frequency analysis
        results['frequency_hz'] = total_events / results['duration_total']
        
        # Per-ROI statistics
        roi_stats = []
        for roi_name in self.roi_names:
            roi_events = df[df['roi_name'] == roi_name]
            if len(roi_events) > 0:
                roi_stats.append({
                    'roi_name': roi_name,
                    'n_events': len(roi_events),
                    'frequency': len(roi_events) / results['duration_total'],
                    'mean_amplitude': roi_events['amplitude'].mean(),
                    'mean_duration': roi_events['duration_time'].mean()
                })
                
        results['roi_statistics'] = roi_stats
        
        # Temporal pattern analysis
        if total_events > 1:
            # Inter-event intervals
            all_event_times = []
            for roi_idx, events in enumerate(self.detected_events):
                event_times = [e['start'] / self.framerate for e in events]
                all_event_times.extend(event_times)
                
            all_event_times = sorted(all_event_times)
            if len(all_event_times) > 1:
                intervals = np.diff(all_event_times)
                results['iei_mean'] = np.mean(intervals)
                results['iei_median'] = np.median(intervals)
                results['iei_std'] = np.std(intervals)
                results['iei_cv'] = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                
        return results
        
    def create_raster_plot(self):
        """
        Create raster plot showing events across all ROIs over time.
        """
        fig = Figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.traces)))
        
        for roi_idx, events in enumerate(self.detected_events):
            for event in events:
                start_time = event['start'] / self.framerate
                duration = event['duration'] / self.framerate
                
                # Plot as horizontal bar
                ax.barh(roi_idx, duration, left=start_time, height=0.8,
                       color=colors[roi_idx], alpha=0.7, edgecolor='black', linewidth=0.5)
                
        ax.set_yticks(range(len(self.roi_names)))
        ax.set_yticklabels(self.roi_names)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('ROI / Region')
        ax.set_title(f'Event Raster Plot (n={len(self.event_properties)} events)')
        ax.grid(True, alpha=0.3, axis='x')
        
        return fig
        
    def plot_analysis(self):
        """
        Create comprehensive visualization of detected events.
        """
        if not self.event_properties:
            g.m.statusBar().showMessage("No events detected to visualize")
            return
            
        # Create multi-panel figure
        fig = Figure(figsize=(16, 12))
        
        df = pd.DataFrame(self.event_properties)
        
        # 1. Example traces with detected events
        ax1 = fig.add_subplot(3, 3, 1)
        for i, (trace, events) in enumerate(zip(self.traces[:3], self.detected_events[:3])):
            time = np.arange(len(trace)) / self.framerate
            ax1.plot(time, trace, alpha=0.7, label=self.roi_names[i])
            
            # Mark events
            for event in events:
                ax1.axvspan(event['start'] / self.framerate, 
                           event['end'] / self.framerate,
                           alpha=0.3, color='red')
                           
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title('Example Traces with Detected Events')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Amplitude distribution
        ax2 = fig.add_subplot(3, 3, 2)
        ax2.hist(df['amplitude'], bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(df['amplitude'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {df['amplitude'].mean():.2f}")
        ax2.set_xlabel('Amplitude (a.u.)')
        ax2.set_ylabel('Count')
        ax2.set_title('Event Amplitude Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Duration distribution
        ax3 = fig.add_subplot(3, 3, 3)
        ax3.hist(df['duration_time'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(df['duration_time'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {df['duration_time'].mean():.2f} s")
        ax3.set_xlabel('Duration (s)')
        ax3.set_ylabel('Count')
        ax3.set_title('Event Duration Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ΔF/F₀ distribution
        ax4 = fig.add_subplot(3, 3, 4)
        ax4.hist(df['df_f0'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(df['df_f0'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {df['df_f0'].mean():.2f}")
        ax4.set_xlabel('ΔF/F₀')
        ax4.set_ylabel('Count')
        ax4.set_title('Event ΔF/F₀ Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Rise time vs decay tau
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.scatter(df['rise_time'], df['decay_tau'], alpha=0.6, s=30)
        ax5.set_xlabel('Rise Time (s)')
        ax5.set_ylabel('Decay Tau (s)')
        ax5.set_title('Event Kinetics')
        ax5.grid(True, alpha=0.3)
        
        # 6. Temporal distribution (event frequency over time)
        ax6 = fig.add_subplot(3, 3, 6)
        all_times = df['start_frame'] / self.framerate
        n_bins = min(50, len(all_times) // 5 + 1)
        ax6.hist(all_times, bins=n_bins, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Number of Events')
        ax6.set_title('Temporal Distribution of Events')
        ax6.grid(True, alpha=0.3)
        
        # 7. Per-ROI comparison
        ax7 = fig.add_subplot(3, 3, 7)
        roi_stats = self.analysis_results.get('roi_statistics', [])
        if roi_stats:
            roi_names_plot = [r['roi_name'] for r in roi_stats]
            frequencies = [r['frequency'] for r in roi_stats]
            
            bars = ax7.bar(range(len(roi_names_plot)), frequencies, alpha=0.7, color='cyan')
            ax7.set_xticks(range(len(roi_names_plot)))
            ax7.set_xticklabels(roi_names_plot, rotation=45, ha='right', fontsize=8)
            ax7.set_ylabel('Frequency (events/s)')
            ax7.set_title('Event Frequency by ROI')
            ax7.grid(True, alpha=0.3, axis='y')
            
        # 8. Statistics summary
        ax8 = fig.add_subplot(3, 3, 8)
        ax8.axis('off')
        
        stats_text = "EVENT STATISTICS\n" + "="*40 + "\n\n"
        stats_text += f"Total Events: {self.analysis_results['total_events']}\n"
        stats_text += f"Recording Duration: {self.analysis_results['duration_total']:.1f} s\n"
        stats_text += f"Frequency: {self.analysis_results['frequency_hz']:.3f} Hz\n\n"
        
        stats_text += f"Amplitude:\n"
        stats_text += f"  Mean: {self.analysis_results.get('amplitude_mean', 0):.2f}\n"
        stats_text += f"  Median: {self.analysis_results.get('amplitude_median', 0):.2f}\n\n"
        
        stats_text += f"Duration:\n"
        stats_text += f"  Mean: {self.analysis_results.get('duration_time_mean', 0):.3f} s\n"
        stats_text += f"  Median: {self.analysis_results.get('duration_time_median', 0):.3f} s\n\n"
        
        stats_text += f"ΔF/F₀:\n"
        stats_text += f"  Mean: {self.analysis_results.get('df_f0_mean', 0):.2f}\n"
        stats_text += f"  Median: {self.analysis_results.get('df_f0_median', 0):.2f}\n\n"
        
        if 'iei_mean' in self.analysis_results:
            stats_text += f"Inter-Event Interval:\n"
            stats_text += f"  Mean: {self.analysis_results['iei_mean']:.3f} s\n"
            stats_text += f"  CV: {self.analysis_results['iei_cv']:.2f}\n"
            
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 9. Amplitude vs Duration scatter
        ax9 = fig.add_subplot(3, 3, 9)
        scatter = ax9.scatter(df['duration_time'], df['amplitude'], 
                            c=df['df_f0'], cmap='viridis', alpha=0.6, s=30)
        ax9.set_xlabel('Duration (s)')
        ax9.set_ylabel('Amplitude (a.u.)')
        ax9.set_title('Event Properties')
        cbar = fig.colorbar(scatter, ax=ax9, label='ΔF/F₀')
        ax9.grid(True, alpha=0.3)
        
        fig.tight_layout()
        
        # Display main analysis
        self.analysis_window = QWidget()
        self.analysis_window.setWindowTitle("Flicker/Transient Detection Results")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        raster_btn = QPushButton("Show Raster Plot")
        raster_btn.clicked.connect(self.show_raster)
        button_layout.addWidget(raster_btn)
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_gui)
        button_layout.addWidget(export_btn)
        
        layout.addLayout(button_layout)
        
        self.analysis_window.setLayout(layout)
        self.analysis_window.resize(1600, 1200)
        self.analysis_window.show()
        
        g.m.statusBar().showMessage("Analysis visualization created")
        
    def show_raster(self):
        """
        Display raster plot in separate window.
        """
        fig = self.create_raster_plot()
        
        self.raster_window = QWidget()
        self.raster_window.setWindowTitle("Event Raster Plot")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        self.raster_window.setLayout(layout)
        self.raster_window.resize(1200, 800)
        self.raster_window.show()
        
    def export_data(self, base_filename):
        """
        Export all analysis results to files.
        """
        import json
        
        # Export event properties
        if self.event_properties:
            df = pd.DataFrame(self.event_properties)
            df.to_csv(f"{base_filename}_events.csv", index=False)
            
        # Export traces
        if self.traces:
            trace_data = {}
            for i, (trace, name) in enumerate(zip(self.traces, self.roi_names)):
                trace_data[name] = trace.tolist()
            
            df_traces = pd.DataFrame(trace_data)
            df_traces.to_csv(f"{base_filename}_traces.csv", index=False)
            
        # Export summary statistics
        summary = {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                  for k, v in self.analysis_results.items() 
                  if k != 'roi_statistics'}
                  
        # Add ROI statistics separately
        if 'roi_statistics' in self.analysis_results:
            df_roi = pd.DataFrame(self.analysis_results['roi_statistics'])
            df_roi.to_csv(f"{base_filename}_roi_stats.csv", index=False)
            
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
        self.method_combo.addItems(['threshold', 'derivative', 'zscore', 'wavelet'])
        
        # Threshold
        self.threshold_spin = pg.SpinBox(int=False, step=0.1, bounds=[0.1, 100],
                                        value=2.0, decimals=2)
        
        # Duration filters
        self.min_duration_spin = pg.SpinBox(int=True, step=1, bounds=[1, 1000], value=3)
        self.max_duration_spin = pg.SpinBox(int=True, step=5, bounds=[5, 10000], value=100)
        
        # Smoothing
        self.smooth_spin = pg.SpinBox(int=False, step=0.1, bounds=[0, 10],
                                     value=1.0, decimals=1)
        
        # Baseline method
        self.baseline_combo = ComboBox()
        self.baseline_combo.addItems(['percentile', 'mean', 'median', 'rolling'])
        
        self.baseline_percentile_spin = pg.SpinBox(int=False, step=1, bounds=[1, 50],
                                                   value=10, decimals=1)
        
        # Framerate
        self.framerate_spin = pg.SpinBox(int=False, step=1, bounds=[0.1, 1000],
                                        value=10.0, decimals=2)
        
        # Analysis options
        self.use_rois_check = CheckBox()
        self.use_rois_check.setChecked(True)
        
        self.analyze_regions_check = CheckBox()
        self.analyze_regions_check.setChecked(False)
        
        # Buttons
        self.detect_button = QPushButton('Detect Events')
        self.detect_button.clicked.connect(self.detect_from_gui)
        
        self.plot_button = QPushButton('Visualize Results')
        self.plot_button.clicked.connect(self.plot_analysis)
        
        self.raster_button = QPushButton('Show Raster Plot')
        self.raster_button.clicked.connect(self.show_raster)
        
        self.export_button = QPushButton('Export Data')
        self.export_button.clicked.connect(self.export_gui)
        
        # Build items list
        self.items.append({'name': 'window', 'string': 'Source Window',
                          'object': window_selector})
        self.items.append({'name': '', 'string': '--- Detection Parameters ---',
                          'object': None})
        self.items.append({'name': 'method', 'string': 'Detection Method',
                          'object': self.method_combo})
        self.items.append({'name': 'threshold', 'string': 'Threshold',
                          'object': self.threshold_spin})
        self.items.append({'name': 'min_duration', 'string': 'Min Duration (frames)',
                          'object': self.min_duration_spin})
        self.items.append({'name': 'max_duration', 'string': 'Max Duration (frames)',
                          'object': self.max_duration_spin})
        self.items.append({'name': 'smooth_sigma', 'string': 'Smoothing Sigma',
                          'object': self.smooth_spin})
        self.items.append({'name': '', 'string': '--- Baseline ---',
                          'object': None})
        self.items.append({'name': 'baseline_method', 'string': 'Baseline Method',
                          'object': self.baseline_combo})
        self.items.append({'name': 'baseline_percentile', 'string': 'Baseline Percentile',
                          'object': self.baseline_percentile_spin})
        self.items.append({'name': '', 'string': '--- Analysis Options ---',
                          'object': None})
        self.items.append({'name': 'framerate', 'string': 'Frame Rate (Hz)',
                          'object': self.framerate_spin})
        self.items.append({'name': 'use_rois', 'string': 'Use ROIs',
                          'object': self.use_rois_check})
        self.items.append({'name': 'analyze_regions', 'string': 'Analyze Regions',
                          'object': self.analyze_regions_check})
        self.items.append({'name': '', 'string': '--- Actions ---',
                          'object': None})
        self.items.append({'name': 'detect', 'string': '',
                          'object': self.detect_button})
        self.items.append({'name': 'plot', 'string': '',
                          'object': self.plot_button})
        self.items.append({'name': 'raster', 'string': '',
                          'object': self.raster_button})
        self.items.append({'name': 'export', 'string': '',
                          'object': self.export_button})
        
        super().gui()
        
    def detect_from_gui(self):
        """Run detection from GUI parameters."""
        window = self.getValue('window')
        method = self.getValue('method')
        threshold = self.getValue('threshold')
        min_duration = self.getValue('min_duration')
        max_duration = self.getValue('max_duration')
        smooth_sigma = self.getValue('smooth_sigma')
        baseline_method = self.getValue('baseline_method')
        baseline_percentile = self.getValue('baseline_percentile')
        framerate = self.getValue('framerate')
        use_rois = self.getValue('use_rois')
        analyze_regions = self.getValue('analyze_regions')
        
        self(window, method, threshold, min_duration, max_duration, 
             smooth_sigma, baseline_method, baseline_percentile, 
             framerate, use_rois, analyze_regions)
             
    def export_gui(self):
        """Export data via file dialog."""
        filename, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Event Data", "",
            "All Files (*)"
        )
        if filename:
            self.export_data(filename)


# Create plugin instance
flicker_transient_detector = FlickerTransientDetector()
