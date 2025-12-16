# frap_analyzer/__init__.py
"""
FRAP (Fluorescence Recovery After Photobleaching) Analyzer Plugin for FLIKA
Analyzes recovery kinetics to determine protein mobility and binding dynamics
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from flika.roi import ROI_rectangle, makeROI
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTabWidget, QWidget

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class FRAPAnalyzer(BaseProcess):
    """
    Comprehensive FRAP analysis with multiple recovery models
    """
    
    def __init__(self):
        super().__init__()
        self.frap_roi = None
        self.control_roi = None
        self.background_roi = None
        self.recovery_data = None
        self.fit_results = {}
        self.results_tabs = None
        
    def get_init_settings_dict(self):
        return {
            'bleach_frame': 10,
            'prebleach_frames': 5,
            'recovery_model': ['single_exponential', 'double_exponential', 'anomalous_diffusion', 'reaction_dominant'],
            'normalize_method': ['full_scale', 'prebleach_average', 'plateau_method'],
            'background_correction': True,
            'photobleaching_correction': True,
            'roi_size': 20,
            'auto_detect_bleach': True,
            'fit_start_frame': 0,
            'confidence_interval': 95
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['bleach_frame'] = int(self.bleach_frame.value())
        params['prebleach_frames'] = int(self.prebleach_frames.value())
        params['recovery_model'] = self.recovery_model.currentText()
        params['normalize_method'] = self.normalize_method.currentText()
        params['background_correction'] = self.background_correction.isChecked()
        params['photobleaching_correction'] = self.photobleaching_correction.isChecked()
        params['roi_size'] = int(self.roi_size.value())
        params['auto_detect_bleach'] = self.auto_detect_bleach.isChecked()
        params['fit_start_frame'] = int(self.fit_start_frame.value())
        params['confidence_interval'] = self.confidence_interval.value()
        return params
    
    def get_name(self):
        return 'FRAP Analyzer'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>FRAP Analyzer'
    
    def setupGUI(self):
        super().setupGUI()
        self.bleach_frame.setRange(1, 1000)
        self.prebleach_frames.setRange(1, 50)
        self.roi_size.setRange(5, 100)
        self.fit_start_frame.setRange(0, 100)
        self.confidence_interval.setRange(90, 99)
        
        # Add ROI setup buttons
        roi_layout = QHBoxLayout()
        
        self.frap_roi_button = QPushButton("Select FRAP ROI")
        self.frap_roi_button.clicked.connect(self.select_frap_roi)
        roi_layout.addWidget(self.frap_roi_button)
        
        self.control_roi_button = QPushButton("Select Control ROI")
        self.control_roi_button.clicked.connect(self.select_control_roi)
        roi_layout.addWidget(self.control_roi_button)
        
        self.bg_roi_button = QPushButton("Select Background ROI")
        self.bg_roi_button.clicked.connect(self.select_background_roi)
        roi_layout.addWidget(self.bg_roi_button)
        
        self.layout().addLayout(roi_layout)
        
        # Add analysis button
        self.analyze_button = QPushButton("Analyze FRAP Recovery")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout().addWidget(self.analyze_button)
        
        # Add results tabs
        self.results_tabs = QTabWidget()
        self.layout().addWidget(self.results_tabs)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.layout().addWidget(self.export_button)
    
    def select_frap_roi(self):
        """Select FRAP ROI"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        size = params['roi_size']
        
        # Create ROI at center of image
        center_x = g.win.image.shape[2] // 2
        center_y = g.win.image.shape[1] // 2
        
        self.frap_roi = makeROI('rectangle', 
                               pos=[center_x - size//2, center_y - size//2],
                               size=[size, size],
                               window=g.win)
        
        g.alert("FRAP ROI created. Move it to the bleached area.")
    
    def select_control_roi(self):
        """Select control ROI for photobleaching correction"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        size = params['roi_size']
        
        # Create ROI offset from center
        center_x = g.win.image.shape[2] // 2 + size + 10
        center_y = g.win.image.shape[1] // 2
        
        self.control_roi = makeROI('rectangle',
                                  pos=[center_x - size//2, center_y - size//2],
                                  size=[size, size],
                                  window=g.win)
        
        g.alert("Control ROI created. Move it to an unbleached area with similar intensity.")
    
    def select_background_roi(self):
        """Select background ROI"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        size = params['roi_size']
        
        # Create ROI in corner
        self.background_roi = makeROI('rectangle',
                                     pos=[10, 10],
                                     size=[size, size],
                                     window=g.win)
        
        g.alert("Background ROI created. Move it to a cell-free area.")
    
    def auto_detect_bleach_frame(self, frap_trace):
        """Automatically detect the bleaching frame"""
        # Look for the largest intensity drop
        diff = np.diff(frap_trace)
        bleach_frame = np.argmin(diff) + 1
        
        # Validate - should be a significant drop
        intensity_drop = frap_trace[bleach_frame-1] - frap_trace[bleach_frame]
        relative_drop = intensity_drop / frap_trace[bleach_frame-1]
        
        if relative_drop < 0.1:  # Less than 10% drop
            g.alert("Warning: Automatic bleach detection found small intensity drop. Check bleach frame.")
        
        return bleach_frame
    
    def extract_roi_traces(self, image_stack, roi):
        """Extract intensity trace from ROI"""
        if roi is None:
            return None
        
        trace = []
        for frame_idx in range(image_stack.shape[0]):
            frame = image_stack[frame_idx]
            
            # Get ROI mask
            mask = roi.getMask()
            if len(mask) > 0:
                roi_pixels = frame[mask[0], mask[1]]
                trace.append(np.mean(roi_pixels))
            else:
                trace.append(0)
        
        return np.array(trace)
    
    def normalize_frap_data(self, frap_trace, control_trace, bg_trace, bleach_frame, method, prebleach_frames):
        """Normalize FRAP data using specified method"""
        # Background subtraction
        if bg_trace is not None:
            frap_corrected = frap_trace - np.mean(bg_trace)
            if control_trace is not None:
                control_corrected = control_trace - np.mean(bg_trace)
            else:
                control_corrected = None
        else:
            frap_corrected = frap_trace.copy()
            control_corrected = control_trace.copy() if control_trace is not None else None
        
        # Photobleaching correction using control ROI
        if control_corrected is not None:
            # Normalize control trace to its prebleach value
            control_prebleach = np.mean(control_corrected[:bleach_frame])
            control_normalized = control_corrected / control_prebleach
            
            # Apply correction
            frap_corrected = frap_corrected / control_normalized
        
        # Normalization
        prebleach_intensity = np.mean(frap_corrected[max(0, bleach_frame - prebleach_frames):bleach_frame])
        
        if method == 'full_scale':
            # Normalize to 0-1 scale
            postbleach_min = np.min(frap_corrected[bleach_frame:bleach_frame+5])
            normalized = (frap_corrected - postbleach_min) / (prebleach_intensity - postbleach_min)
            
        elif method == 'prebleach_average':
            # Normalize to prebleach average
            normalized = frap_corrected / prebleach_intensity
            
        elif method == 'plateau_method':
            # Use recovery plateau for normalization
            recovery_data = frap_corrected[bleach_frame:]
            if len(recovery_data) > 20:
                plateau_value = np.mean(recovery_data[-10:])  # Last 10 points
                postbleach_min = np.min(recovery_data[:5])    # First 5 points after bleach
                normalized = (frap_corrected - postbleach_min) / (plateau_value - postbleach_min)
            else:
                # Fallback to prebleach method
                normalized = frap_corrected / prebleach_intensity
        
        return normalized, prebleach_intensity
    
    def single_exponential_model(self, t, A, tau, plateau):
        """Single exponential recovery model"""
        return plateau - A * np.exp(-t / tau)
    
    def double_exponential_model(self, t, A1, tau1, A2, tau2, plateau):
        """Double exponential recovery model"""
        return plateau - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)
    
    def anomalous_diffusion_model(self, t, A, D, alpha, plateau):
        """Anomalous diffusion model (stretched exponential)"""
        return plateau - A * np.exp(-(t / D) ** alpha)
    
    def reaction_dominant_model(self, t, A, kon, koff, plateau):
        """Reaction-dominant recovery model"""
        k_total = kon + koff
        return plateau - A * (koff / k_total + (kon / k_total) * np.exp(-k_total * t))
    
    def fit_recovery_model(self, time, intensity, model_name):
        """Fit recovery model to data"""
        try:
            if model_name == 'single_exponential':
                # Initial guess
                A0 = 1 - np.min(intensity)
                tau0 = len(time) * 0.3
                plateau0 = np.mean(intensity[-5:]) if len(intensity) > 5 else np.max(intensity)
                
                popt, pcov = curve_fit(
                    self.single_exponential_model,
                    time, intensity,
                    p0=[A0, tau0, plateau0],
                    maxfev=5000,
                    bounds=([0, 1, 0], [2, len(time)*2, 2])
                )
                
                # Calculate metrics
                fitted_curve = self.single_exponential_model(time, *popt)
                half_time = popt[1] * np.log(2)
                mobile_fraction = popt[2] - (popt[2] - intensity[0] + popt[0])
                
                return {
                    'model': 'single_exponential',
                    'params': {'A': popt[0], 'tau': popt[1], 'plateau': popt[2]},
                    'fitted_curve': fitted_curve,
                    'half_time': half_time,
                    'mobile_fraction': mobile_fraction,
                    'tau': popt[1],
                    'covariance': pcov,
                    'r_squared': 1 - np.sum((intensity - fitted_curve)**2) / np.sum((intensity - np.mean(intensity))**2)
                }
                
            elif model_name == 'double_exponential':
                # Initial guess
                A1_0 = (1 - np.min(intensity)) * 0.7
                A2_0 = (1 - np.min(intensity)) * 0.3
                tau1_0 = len(time) * 0.1
                tau2_0 = len(time) * 0.5
                plateau0 = np.mean(intensity[-5:]) if len(intensity) > 5 else np.max(intensity)
                
                popt, pcov = curve_fit(
                    self.double_exponential_model,
                    time, intensity,
                    p0=[A1_0, tau1_0, A2_0, tau2_0, plateau0],
                    maxfev=5000,
                    bounds=([0, 1, 0, 1, 0], [2, len(time), 2, len(time)*2, 2])
                )
                
                fitted_curve = self.double_exponential_model(time, *popt)
                
                return {
                    'model': 'double_exponential',
                    'params': {'A1': popt[0], 'tau1': popt[1], 'A2': popt[2], 'tau2': popt[3], 'plateau': popt[4]},
                    'fitted_curve': fitted_curve,
                    'tau_fast': min(popt[1], popt[3]),
                    'tau_slow': max(popt[1], popt[3]),
                    'covariance': pcov,
                    'r_squared': 1 - np.sum((intensity - fitted_curve)**2) / np.sum((intensity - np.mean(intensity))**2)
                }
                
            elif model_name == 'anomalous_diffusion':
                # Initial guess
                A0 = 1 - np.min(intensity)
                D0 = len(time) * 0.3
                alpha0 = 0.8
                plateau0 = np.mean(intensity[-5:]) if len(intensity) > 5 else np.max(intensity)
                
                popt, pcov = curve_fit(
                    self.anomalous_diffusion_model,
                    time, intensity,
                    p0=[A0, D0, alpha0, plateau0],
                    maxfev=5000,
                    bounds=([0, 1, 0.1, 0], [2, len(time)*2, 2, 2])
                )
                
                fitted_curve = self.anomalous_diffusion_model(time, *popt)
                
                return {
                    'model': 'anomalous_diffusion',
                    'params': {'A': popt[0], 'D': popt[1], 'alpha': popt[2], 'plateau': popt[3]},
                    'fitted_curve': fitted_curve,
                    'diffusion_coeff': popt[1],
                    'anomalous_exponent': popt[2],
                    'covariance': pcov,
                    'r_squared': 1 - np.sum((intensity - fitted_curve)**2) / np.sum((intensity - np.mean(intensity))**2)
                }
        
        except Exception as e:
            print(f"Fitting failed for {model_name}: {e}")
            return None
        
        return None
    
    def run_analysis(self):
        """Run comprehensive FRAP analysis"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        if self.frap_roi is None:
            g.alert("Please select FRAP ROI first!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Analyzing FRAP recovery...")
        
        # Extract traces from ROIs
        frap_trace = self.extract_roi_traces(image_stack, self.frap_roi)
        control_trace = self.extract_roi_traces(image_stack, self.control_roi) if self.control_roi else None
        bg_trace = self.extract_roi_traces(image_stack, self.background_roi) if self.background_roi else None
        
        # Auto-detect bleach frame if requested
        if params['auto_detect_bleach']:
            bleach_frame = self.auto_detect_bleach_frame(frap_trace)
            print(f"Auto-detected bleach frame: {bleach_frame}")
        else:
            bleach_frame = params['bleach_frame']
        
        # Normalize data
        normalized_trace, prebleach_intensity = self.normalize_frap_data(
            frap_trace, control_trace, bg_trace, bleach_frame,
            params['normalize_method'], params['prebleach_frames']
        )
        
        # Extract recovery phase
        recovery_start = bleach_frame + params['fit_start_frame']
        recovery_time = np.arange(len(normalized_trace) - recovery_start)
        recovery_intensity = normalized_trace[recovery_start:]
        
        # Fit recovery model
        fit_result = self.fit_recovery_model(recovery_time, recovery_intensity, params['recovery_model'])
        
        # Store results
        self.recovery_data = {
            'raw_frap_trace': frap_trace,
            'control_trace': control_trace,
            'bg_trace': bg_trace,
            'normalized_trace': normalized_trace,
            'bleach_frame': bleach_frame,
            'recovery_start': recovery_start,
            'recovery_time': recovery_time,
            'recovery_intensity': recovery_intensity,
            'prebleach_intensity': prebleach_intensity
        }
        
        self.fit_results = fit_result if fit_result else {}
        self.params = params
        
        # Display results
        self.display_results()
        
        g.m.statusBar().showMessage("FRAP analysis complete!", 3000)
    
    def display_results(self):
        """Display comprehensive results"""
        # Clear existing tabs
        for i in range(self.results_tabs.count()):
            self.results_tabs.removeTab(0)
        
        # Tab 1: Summary
        self.create_summary_tab()
        
        # Tab 2: Plots
        self.create_plots_tab()
    
    def create_summary_tab(self):
        """Create summary results tab"""
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        if not self.fit_results:
            summary_text.setText("No fit results available. Check your data and parameters.")
            summary_layout.addWidget(summary_text)
            summary_widget.setLayout(summary_layout)
            self.results_tabs.addTab(summary_widget, "Summary")
            return
        
        # Format results
        summary = f"""=== FRAP Analysis Results ===

Experimental Parameters:
  Bleach frame: {self.recovery_data['bleach_frame']}
  Recovery frames analyzed: {len(self.recovery_data['recovery_intensity'])}
  Normalization method: {self.params['normalize_method']}
  Recovery model: {self.params['recovery_model']}

Prebleach Statistics:
  Average prebleach intensity: {self.recovery_data['prebleach_intensity']:.1f}
  Intensity immediately after bleach: {self.recovery_data['normalized_trace'][self.recovery_data['bleach_frame']]:.3f}
  Bleach depth: {1 - self.recovery_data['normalized_trace'][self.recovery_data['bleach_frame']]:.3f}

Recovery Analysis:
  Model: {self.fit_results.get('model', 'Unknown')}
  R-squared: {self.fit_results.get('r_squared', 0):.4f}
"""
        
        # Add model-specific parameters
        if 'params' in self.fit_results:
            summary += "\nFitted Parameters:\n"
            for param, value in self.fit_results['params'].items():
                summary += f"  {param}: {value:.4f}\n"
        
        # Add derived metrics
        if 'half_time' in self.fit_results:
            summary += f"\nDerived Metrics:\n"
            summary += f"  Half-time of recovery: {self.fit_results['half_time']:.2f} frames\n"
        
        if 'mobile_fraction' in self.fit_results:
            summary += f"  Mobile fraction: {self.fit_results['mobile_fraction']:.3f}\n"
            summary += f"  Immobile fraction: {1 - self.fit_results['mobile_fraction']:.3f}\n"
        
        if 'tau' in self.fit_results:
            summary += f"  Recovery time constant (τ): {self.fit_results['tau']:.2f} frames\n"
        
        if 'tau_fast' in self.fit_results and 'tau_slow' in self.fit_results:
            summary += f"  Fast time constant: {self.fit_results['tau_fast']:.2f} frames\n"
            summary += f"  Slow time constant: {self.fit_results['tau_slow']:.2f} frames\n"
        
        if 'diffusion_coeff' in self.fit_results:
            summary += f"  Diffusion coefficient: {self.fit_results['diffusion_coeff']:.4f}\n"
        
        if 'anomalous_exponent' in self.fit_results:
            summary += f"  Anomalous exponent (α): {self.fit_results['anomalous_exponent']:.3f}\n"
        
        summary_text.setText(summary)
        summary_layout.addWidget(summary_text)
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")
    
    def create_plots_tab(self):
        """Create plots tab"""
        if not self.recovery_data:
            return
        
        # Create comprehensive plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('FRAP Analysis Results')
        
        # Plot 1: Raw traces
        time_axis = np.arange(len(self.recovery_data['raw_frap_trace']))
        
        axes[0,0].plot(time_axis, self.recovery_data['raw_frap_trace'], 'b-', label='FRAP ROI', linewidth=2)
        if self.recovery_data['control_trace'] is not None:
            axes[0,0].plot(time_axis, self.recovery_data['control_trace'], 'g-', label='Control ROI', alpha=0.7)
        if self.recovery_data['bg_trace'] is not None:
            axes[0,0].plot(time_axis, self.recovery_data['bg_trace'], 'k-', label='Background', alpha=0.5)
        
        # Mark bleach frame
        axes[0,0].axvline(self.recovery_data['bleach_frame'], color='red', linestyle='--', 
                         label=f'Bleach frame ({self.recovery_data["bleach_frame"]})')
        
        axes[0,0].set_xlabel('Frame')
        axes[0,0].set_ylabel('Raw Intensity')
        axes[0,0].set_title('Raw Intensity Traces')
        axes[0,0].legend()
        
        # Plot 2: Normalized recovery curve with fit
        recovery_time_full = np.arange(len(self.recovery_data['normalized_trace']) - self.recovery_data['bleach_frame'])
        recovery_data_full = self.recovery_data['normalized_trace'][self.recovery_data['bleach_frame']:]
        
        axes[0,1].plot(recovery_time_full, recovery_data_full, 'bo', markersize=4, alpha=0.7, label='Data')
        
        if 'fitted_curve' in self.fit_results:
            fit_time = self.recovery_data['recovery_time']
            fitted_curve = self.fit_results['fitted_curve']
            axes[0,1].plot(fit_time, fitted_curve, 'r-', linewidth=2, label=f'Fit (R² = {self.fit_results.get("r_squared", 0):.3f})')
        
        axes[0,1].set_xlabel('Time after bleach (frames)')
        axes[0,1].set_ylabel('Normalized Intensity')
        axes[0,1].set_title('Recovery Curve and Fit')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        if 'fitted_curve' in self.fit_results:
            residuals = self.recovery_data['recovery_intensity'] - self.fit_results['fitted_curve']
            axes[1,0].plot(self.recovery_data['recovery_time'], residuals, 'bo', markersize=3)
            axes[1,0].axhline(y=0, color='red', linestyle='--')
            axes[1,0].set_xlabel('Time after bleach (frames)')
            axes[1,0].set_ylabel('Residuals')
            axes[1,0].set_title('Fit Residuals')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Recovery metrics visualization
        if 'half_time' in self.fit_results:
            # Show half-time on recovery curve
            half_time = self.fit_results['half_time']
            recovery_data_plot = self.recovery_data['recovery_intensity']
            time_plot = self.recovery_data['recovery_time']
            
            axes[1,1].plot(time_plot, recovery_data_plot, 'bo', markersize=4, alpha=0.7)
            if 'fitted_curve' in self.fit_results:
                axes[1,1].plot(time_plot, self.fit_results['fitted_curve'], 'r-', linewidth=2)
            
            # Mark half-time
            if half_time < len(time_plot):
                half_intensity = self.fit_results['fitted_curve'][int(half_time)] if 'fitted_curve' in self.fit_results else recovery_data_plot[int(half_time)]
                axes[1,1].axvline(half_time, color='orange', linestyle='--', label=f'Half-time ({half_time:.1f})')
                axes[1,1].axhline(half_intensity, color='orange', linestyle=':', alpha=0.5)
            
            axes[1,1].set_xlabel('Time after bleach (frames)')
            axes[1,1].set_ylabel('Normalized Intensity')
            axes[1,1].set_title('Recovery Metrics')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self):
        """Export FRAP analysis results"""
        if not self.recovery_data:
            g.alert("No results to export! Run analysis first.")
            return
        
        base_name = g.win.name
        
        # Export recovery curve data
        recovery_df = pd.DataFrame({
            'time_frames': self.recovery_data['recovery_time'],
            'normalized_intensity': self.recovery_data['recovery_intensity'],
            'fitted_curve': self.fit_results.get('fitted_curve', [None] * len(self.recovery_data['recovery_time']))
        })
        
        recovery_filename = f"{base_name}_frap_recovery.csv"
        recovery_df.to_csv(recovery_filename, index=False)
        
        # Export summary results
        summary_data = {
            'parameter': [],
            'value': []
        }
        
        # Add fit parameters
        if 'params' in self.fit_results:
            for param, value in self.fit_results['params'].items():
                summary_data['parameter'].append(param)
                summary_data['value'].append(value)
        
        # Add derived metrics
        metrics = ['half_time', 'mobile_fraction', 'tau', 'r_squared', 'tau_fast', 'tau_slow', 
                  'diffusion_coeff', 'anomalous_exponent']
        
        for metric in metrics:
            if metric in self.fit_results:
                summary_data['parameter'].append(metric)
                summary_data['value'].append(self.fit_results[metric])
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{base_name}_frap_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        g.alert(f"Results exported to {recovery_filename} and {summary_filename}")
    
    def process(self):
        """Process method required by BaseProcess"""
        self.run_analysis()
        return None

# Register the plugin
FRAPAnalyzer.menu_path = 'Plugins>TIRF Analysis>FRAP Analyzer'