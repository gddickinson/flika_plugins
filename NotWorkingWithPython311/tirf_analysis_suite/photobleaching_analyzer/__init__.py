# photobleaching_analyzer/__init__.py
"""
Photobleaching Step Counter Plugin for FLIKA
Counts photobleaching steps to determine protein oligomerization states
"""

import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from flika.roi import ROI_rectangle
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class PhotobleachingAnalyzer(BaseProcess):
    """
    Analyze photobleaching steps to determine oligomerization states
    """
    
    def __init__(self):
        super().__init__()
        self.traces = []
        self.step_counts = []
        self.results_text = None
        
    def get_init_settings_dict(self):
        return {
            'roi_size': 7,
            'smoothing_window': 3,
            'step_threshold': 0.15,
            'min_step_duration': 5,
            'noise_threshold': 2.0,
            'fit_exponentials': True,
            'analyze_all_rois': True
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['roi_size'] = int(self.roi_size.value())
        params['smoothing_window'] = int(self.smoothing_window.value())
        params['step_threshold'] = self.step_threshold.value()
        params['min_step_duration'] = int(self.min_step_duration.value())
        params['noise_threshold'] = self.noise_threshold.value()
        params['fit_exponentials'] = self.fit_exponentials.isChecked()
        params['analyze_all_rois'] = self.analyze_all_rois.isChecked()
        return params
    
    def get_name(self):
        return 'Photobleaching Analyzer'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Photobleaching Analyzer'
    
    def setupGUI(self):
        super().setupGUI()
        self.roi_size.setRange(3, 21)
        self.smoothing_window.setRange(1, 15)
        self.step_threshold.setRange(0.05, 0.5)
        self.min_step_duration.setRange(2, 50)
        self.noise_threshold.setRange(1.0, 5.0)
        
        # Add analysis button
        self.analyze_button = QPushButton("Analyze Photobleaching")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout().addWidget(self.analyze_button)
        
        # Add results display
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setReadOnly(True)
        self.layout().addWidget(QLabel("Analysis Results:"))
        self.layout().addWidget(self.results_text)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.layout().addWidget(self.export_button)
    
    def exponential_decay(self, t, a, tau, offset):
        """Single exponential decay function"""
        return a * np.exp(-t / tau) + offset
    
    def multi_exponential(self, t, *params):
        """Multi-exponential decay function"""
        # params = [a1, tau1, a2, tau2, ..., offset]
        n_exp = (len(params) - 1) // 2
        result = np.zeros_like(t)
        
        for i in range(n_exp):
            a = params[2*i]
            tau = params[2*i + 1]
            result += a * np.exp(-t / tau)
        
        return result + params[-1]  # Add offset
    
    def detect_steps(self, trace, threshold, min_duration):
        """Detect photobleaching steps using edge detection and clustering"""
        # Smooth the trace
        smoothed = ndimage.uniform_filter1d(trace.astype(float), size=3)
        
        # Calculate derivative to find step edges
        derivative = np.gradient(smoothed)
        
        # Find significant negative steps (photobleaching events)
        step_candidates = np.where(derivative < -threshold * np.std(derivative))[0]
        
        if len(step_candidates) == 0:
            return [], []
        
        # Group nearby step candidates
        step_groups = []
        current_group = [step_candidates[0]]
        
        for i in range(1, len(step_candidates)):
            if step_candidates[i] - step_candidates[i-1] <= min_duration:
                current_group.append(step_candidates[i])
            else:
                step_groups.append(current_group)
                current_group = [step_candidates[i]]
        step_groups.append(current_group)
        
        # Take the center of each group as the step position
        step_positions = [int(np.mean(group)) for group in step_groups]
        
        # Calculate step amplitudes
        step_amplitudes = []
        for pos in step_positions:
            # Compare before and after the step
            before_window = max(0, pos - min_duration)
            after_window = min(len(smoothed), pos + min_duration)
            
            before_mean = np.mean(smoothed[before_window:pos])
            after_mean = np.mean(smoothed[pos:after_window])
            
            step_amplitudes.append(before_mean - after_mean)
        
        return step_positions, step_amplitudes
    
    def count_photobleaching_steps(self, trace, params):
        """Count photobleaching steps using multiple methods"""
        smoothed_trace = ndimage.uniform_filter1d(
            trace.astype(float), 
            size=params['smoothing_window']
        )
        
        # Method 1: Edge detection
        step_positions, step_amplitudes = self.detect_steps(
            smoothed_trace,
            params['step_threshold'],
            params['min_step_duration']
        )
        
        # Method 2: Plateau detection using clustering
        # Find plateaus in the smoothed trace
        plateau_levels = []
        current_level = smoothed_trace[0]
        plateau_levels.append(current_level)
        
        for i in range(1, len(smoothed_trace)):
            # Check if we've moved to a significantly different level
            if abs(smoothed_trace[i] - current_level) > params['step_threshold'] * np.std(smoothed_trace):
                # Find the new plateau level
                window_start = max(0, i - params['min_step_duration'])
                window_end = min(len(smoothed_trace), i + params['min_step_duration'])
                new_level = np.mean(smoothed_trace[window_start:window_end])
                
                if abs(new_level - current_level) > params['step_threshold'] * np.std(smoothed_trace):
                    plateau_levels.append(new_level)
                    current_level = new_level
        
        # Count unique plateaus (number of steps = plateaus - 1)
        if len(plateau_levels) > 1:
            # Use clustering to find distinct intensity levels
            plateau_array = np.array(plateau_levels).reshape(-1, 1)
            n_clusters = min(len(plateau_levels), 6)  # Max 6 steps
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(plateau_array)
            n_distinct_levels = len(np.unique(clusters))
            step_count_clustering = max(0, n_distinct_levels - 1)
        else:
            step_count_clustering = 0
        
        # Final step count (take maximum of methods)
        final_step_count = max(len(step_positions), step_count_clustering)
        
        return {
            'step_count': final_step_count,
            'step_positions': step_positions,
            'step_amplitudes': step_amplitudes,
            'smoothed_trace': smoothed_trace,
            'plateau_levels': plateau_levels
        }
    
    def fit_photobleaching_kinetics(self, trace, step_info):
        """Fit exponential decay models to photobleaching kinetics"""
        t = np.arange(len(trace))
        
        # Try fitting single exponential
        try:
            # Initial guess
            a0 = trace[0] - trace[-1]
            tau0 = len(trace) / 3
            offset0 = trace[-1]
            
            popt_single, _ = curve_fit(
                self.exponential_decay,
                t, trace,
                p0=[a0, tau0, offset0],
                maxfev=1000
            )
            
            single_fit = self.exponential_decay(t, *popt_single)
            single_r2 = 1 - np.sum((trace - single_fit)**2) / np.sum((trace - np.mean(trace))**2)
            
        except:
            popt_single = None
            single_r2 = 0
        
        # Try fitting multi-exponential based on step count
        multi_fit_results = []
        if step_info['step_count'] > 1:
            for n_exp in range(2, min(step_info['step_count'] + 1, 4)):
                try:
                    # Initial guess for multi-exponential
                    initial_params = []
                    for i in range(n_exp):
                        initial_params.extend([a0/n_exp, tau0*(i+1)])
                    initial_params.append(offset0)
                    
                    popt_multi, _ = curve_fit(
                        self.multi_exponential,
                        t, trace,
                        p0=initial_params,
                        maxfev=2000
                    )
                    
                    multi_fit = self.multi_exponential(t, *popt_multi)
                    multi_r2 = 1 - np.sum((trace - multi_fit)**2) / np.sum((trace - np.mean(trace))**2)
                    
                    multi_fit_results.append({
                        'n_exp': n_exp,
                        'params': popt_multi,
                        'r2': multi_r2,
                        'fit': multi_fit
                    })
                except:
                    continue
        
        return {
            'single_exp': {'params': popt_single, 'r2': single_r2},
            'multi_exp': multi_fit_results
        }
    
    def run_analysis(self):
        """Run photobleaching analysis on ROIs"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        
        # Get ROIs to analyze
        if params['analyze_all_rois'] and hasattr(g.win, 'rois') and g.win.rois:
            rois_to_analyze = g.win.rois
        else:
            # Create ROIs automatically or use current ROI
            if not hasattr(g.win, 'rois') or not g.win.rois:
                g.alert("No ROIs found. Please create ROIs first or enable automatic ROI creation.")
                return
            rois_to_analyze = [g.win.currentROI] if g.win.currentROI else g.win.rois[:1]
        
        results = []
        self.traces = []
        
        g.m.statusBar().showMessage("Analyzing photobleaching...")
        
        for i, roi in enumerate(rois_to_analyze):
            # Get trace from ROI
            trace = roi.getTrace()
            
            if trace is None or len(trace) < params['min_step_duration'] * 2:
                continue
            
            self.traces.append(trace)
            
            # Analyze photobleaching steps
            step_info = self.count_photobleaching_steps(trace, params)
            
            # Fit kinetics if requested
            kinetics = None
            if params['fit_exponentials']:
                kinetics = self.fit_photobleaching_kinetics(trace, step_info)
            
            # Calculate additional metrics
            initial_intensity = np.mean(trace[:5])
            final_intensity = np.mean(trace[-5:])
            total_bleaching = initial_intensity - final_intensity
            snr = initial_intensity / np.std(trace[:10])
            
            result = {
                'roi_id': i,
                'step_count': step_info['step_count'],
                'initial_intensity': initial_intensity,
                'final_intensity': final_intensity,
                'total_bleaching': total_bleaching,
                'snr': snr,
                'step_positions': step_info['step_positions'],
                'step_amplitudes': step_info['step_amplitudes'],
                'kinetics': kinetics
            }
            
            results.append(result)
        
        self.results = results
        self.display_results()
        
        g.m.statusBar().showMessage(f"Analysis complete! Analyzed {len(results)} ROIs", 3000)
    
    def display_results(self):
        """Display analysis results"""
        if not self.results:
            return
        
        # Calculate statistics
        step_counts = [r['step_count'] for r in self.results]
        step_count_dist = {i: step_counts.count(i) for i in range(max(step_counts) + 1)}
        
        # Format results text
        results_text = "=== Photobleaching Analysis Results ===\n\n"
        results_text += f"Total ROIs analyzed: {len(self.results)}\n\n"
        
        results_text += "Step Count Distribution:\n"
        for steps, count in step_count_dist.items():
            percentage = (count / len(self.results)) * 100
            results_text += f"  {steps} steps: {count} ROIs ({percentage:.1f}%)\n"
        
        results_text += f"\nAverage steps per ROI: {np.mean(step_counts):.2f}\n"
        results_text += f"Median steps per ROI: {np.median(step_counts):.1f}\n\n"
        
        # Individual ROI results
        results_text += "Individual ROI Results:\n"
        for i, result in enumerate(self.results):
            results_text += f"ROI {i+1}: {result['step_count']} steps, "
            results_text += f"SNR: {result['snr']:.1f}, "
            results_text += f"Total bleaching: {result['total_bleaching']:.0f}\n"
        
        self.results_text.setText(results_text)
        
        # Create visualization
        self.create_visualization()
    
    def create_visualization(self):
        """Create visualization of photobleaching analysis"""
        if not self.results or not self.traces:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Photobleaching Analysis Results')
        
        # Plot 1: Step count histogram
        step_counts = [r['step_count'] for r in self.results]
        axes[0,0].hist(step_counts, bins=range(max(step_counts) + 2), alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Number of Steps')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Step Count Distribution')
        
        # Plot 2: Example traces with step detection
        axes[0,1].set_title('Example Photobleaching Traces')
        for i, (trace, result) in enumerate(zip(self.traces[:5], self.results[:5])):
            t = np.arange(len(trace))
            axes[0,1].plot(t, trace + i*500, label=f'ROI {i+1} ({result["step_count"]} steps)')
            
            # Mark detected steps
            for step_pos in result['step_positions']:
                axes[0,1].axvline(step_pos, color='red', alpha=0.5, linestyle='--')
        
        axes[0,1].set_xlabel('Frame')
        axes[0,1].set_ylabel('Intensity (offset)')
        axes[0,1].legend()
        
        # Plot 3: SNR vs Step Count
        snrs = [r['snr'] for r in self.results]
        axes[1,0].scatter(step_counts, snrs, alpha=0.7)
        axes[1,0].set_xlabel('Number of Steps')
        axes[1,0].set_ylabel('Signal-to-Noise Ratio')
        axes[1,0].set_title('SNR vs Step Count')
        
        # Plot 4: Intensity characteristics
        initial_intensities = [r['initial_intensity'] for r in self.results]
        total_bleaching = [r['total_bleaching'] for r in self.results]
        axes[1,1].scatter(initial_intensities, total_bleaching, alpha=0.7)
        axes[1,1].set_xlabel('Initial Intensity')
        axes[1,1].set_ylabel('Total Bleaching')
        axes[1,1].set_title('Bleaching vs Initial Intensity')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self):
        """Export results to CSV"""
        if not hasattr(self, 'results') or not self.results:
            g.alert("No results to export! Run analysis first.")
            return
        
        # Prepare data for export
        export_data = []
        
        for result in self.results:
            row = {
                'roi_id': result['roi_id'],
                'step_count': result['step_count'],
                'initial_intensity': result['initial_intensity'],
                'final_intensity': result['final_intensity'],
                'total_bleaching': result['total_bleaching'],
                'snr': result['snr'],
                'step_positions': ';'.join(map(str, result['step_positions'])),
                'step_amplitudes': ';'.join(map(str, result['step_amplitudes']))
            }
            export_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(export_data)
        filename = f"{g.win.name}_photobleaching_analysis.csv"
        df.to_csv(filename, index=False)
        
        g.alert(f"Results exported to {filename}")
    
    def process(self):
        """Process method required by BaseProcess"""
        self.run_analysis()
        return None

# Register the plugin
PhotobleachingAnalyzer.menu_path = 'Plugins>TIRF Analysis>Photobleaching Analyzer'