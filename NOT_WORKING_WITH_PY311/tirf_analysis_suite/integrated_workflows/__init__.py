# integrated_workflows.py
"""
Integrated TIRF Analysis Workflows
Demonstrates how to combine multiple plugins for complete analysis pipelines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from flika import global_vars as g
from flika.window import Window
from qtpy.QtWidgets import QProgressDialog, QApplication

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class TIRFWorkflowManager:
    """Manages integrated analysis workflows combining multiple plugins"""
    
    def __init__(self):
        self.analysis_log = []
        self.results = {}
        
    def log_analysis_step(self, step_name, parameters=None, results=None):
        """Log an analysis step for reproducibility"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'parameters': parameters or {},
            'results': results or {}
        }
        self.analysis_log.append(log_entry)
        print(f"Workflow step: {step_name}")
    
    def single_molecule_complete_workflow(self):
        """Complete single molecule analysis workflow"""
        
        if g.win is None:
            g.alert("No image loaded! Please load a TIRF image stack first.")
            return
        
        original_window = g.win
        
        # Progress tracking
        progress = QProgressDialog("Running single molecule analysis workflow...", "Cancel", 0, 5)
        progress.setWindowTitle("TIRF Workflow")
        progress.show()
        QApplication.processEvents()
        
        try:
            # Step 1: Background Correction
            progress.setLabelText("Step 1/5: Background correction...")
            progress.setValue(1)
            QApplication.processEvents()
            
            from tirf_background_corrector import TIRFBackgroundCorrector
            bg_corrector = TIRFBackgroundCorrector()
            
            # Set optimal parameters for single molecule imaging
            bg_params = {
                'correction_method': 'rolling_ball',
                'rolling_ball_radius': 30,
                'flatfield_correction': True,
                'normalize_intensity': True
            }
            
            # Apply background correction
            corrected_window = bg_corrector.process()
            self.log_analysis_step("Background Correction", bg_params)
            
            if corrected_window:
                corrected_window.setAsCurrentWindow()
            
            # Step 2: Single Molecule Tracking  
            progress.setLabelText("Step 2/5: Single molecule tracking...")
            progress.setValue(2)
            QApplication.processEvents()
            
            from single_molecule_tracker import SingleMoleculeTracker
            tracker = SingleMoleculeTracker()
            
            tracking_params = {
                'detection_threshold': 3.5,
                'max_displacement': 5.0,
                'min_track_length': 5,
                'gaussian_fit': True,
                'subpixel_accuracy': True
            }
            
            # Run tracking
            tracker.run_tracking()
            
            n_tracks = len(tracker.tracks) if hasattr(tracker, 'tracks') else 0
            self.log_analysis_step("Single Molecule Tracking", tracking_params, 
                                 {'n_tracks': n_tracks})
            
            # Step 3: Track Analysis
            progress.setLabelText("Step 3/5: Analyzing track properties...")
            progress.setValue(3)
            QApplication.processEvents()
            
            if hasattr(tracker, 'tracks') and tracker.tracks:
                track_stats = self.analyze_track_statistics(tracker.tracks)
                self.results['track_statistics'] = track_stats
                self.log_analysis_step("Track Analysis", {}, track_stats)
            
            # Step 4: Diffusion Analysis
            progress.setLabelText("Step 4/5: Diffusion analysis...")
            progress.setValue(4)
            QApplication.processEvents()
            
            if hasattr(tracker, 'tracks') and tracker.tracks:
                diffusion_results = self.calculate_diffusion_coefficients(tracker.tracks)
                self.results['diffusion_analysis'] = diffusion_results
                self.log_analysis_step("Diffusion Analysis", {}, diffusion_results)
            
            # Step 5: Generate Report
            progress.setLabelText("Step 5/5: Generating analysis report...")
            progress.setValue(5)
            QApplication.processEvents()
            
            self.generate_single_molecule_report(original_window.name)
            
            progress.close()
            g.alert("Single molecule analysis workflow completed!")
            
        except Exception as e:
            progress.close()
            g.alert(f"Workflow failed: {str(e)}")
    
    def photobleaching_oligomerization_workflow(self):
        """Complete photobleaching analysis workflow for oligomerization studies"""
        
        if g.win is None:
            g.alert("No image loaded! Please load a photobleaching dataset first.")
            return
        
        original_window = g.win
        
        progress = QProgressDialog("Running photobleaching analysis workflow...", "Cancel", 0, 4)
        progress.setWindowTitle("Photobleaching Workflow")
        progress.show()
        QApplication.processEvents()
        
        try:
            # Step 1: Image Quality Assessment
            progress.setLabelText("Step 1/4: Assessing image quality...")
            progress.setValue(1)
            QApplication.processEvents()
            
            quality_metrics = self.assess_image_quality(g.win.image)
            self.log_analysis_step("Image Quality Assessment", {}, quality_metrics)
            
            if quality_metrics['snr'] < 3:
                g.alert("Warning: Low signal-to-noise ratio detected. Results may be unreliable.")
            
            # Step 2: ROI Selection and Validation
            progress.setLabelText("Step 2/4: Selecting analysis ROIs...")
            progress.setValue(2)
            QApplication.processEvents()
            
            # Auto-detect bright spots for ROI placement
            roi_positions = self.auto_detect_photobleaching_spots()
            
            if len(roi_positions) < 10:
                g.alert("Warning: Few suitable spots detected. Consider manual ROI selection.")
            
            # Step 3: Photobleaching Analysis
            progress.setLabelText("Step 3/4: Analyzing photobleaching steps...")
            progress.setValue(3)
            QApplication.processEvents()
            
            from photobleaching_analyzer import PhotobleachingAnalyzer
            pb_analyzer = PhotobleachingAnalyzer()
            
            pb_params = {
                'step_threshold': 0.2,
                'min_step_duration': 3,
                'smoothing_window': 3,
                'fit_exponentials': True
            }
            
            pb_analyzer.run_analysis()
            
            if hasattr(pb_analyzer, 'results'):
                oligomer_stats = self.analyze_oligomerization_distribution(pb_analyzer.results)
                self.results['oligomerization'] = oligomer_stats
                self.log_analysis_step("Photobleaching Analysis", pb_params, oligomer_stats)
            
            # Step 4: Statistical Analysis and Report
            progress.setLabelText("Step 4/4: Generating oligomerization report...")
            progress.setValue(4)
            QApplication.processEvents()
            
            self.generate_oligomerization_report(original_window.name)
            
            progress.close()
            g.alert("Photobleaching oligomerization workflow completed!")
            
        except Exception as e:
            progress.close()
            g.alert(f"Workflow failed: {str(e)}")
    
    def membrane_dynamics_workflow(self):
        """Complete membrane dynamics analysis workflow"""
        
        if g.win is None:
            g.alert("No image loaded! Please load a membrane dynamics dataset first.")
            return
        
        original_window = g.win
        
        progress = QProgressDialog("Running membrane dynamics workflow...", "Cancel", 0, 5)
        progress.setWindowTitle("Membrane Dynamics Workflow")
        progress.show()
        QApplication.processEvents()
        
        try:
            # Step 1: Preprocessing for Edge Detection
            progress.setLabelText("Step 1/5: Preprocessing for edge detection...")
            progress.setValue(1)
            QApplication.processEvents()
            
            from tirf_background_corrector import TIRFBackgroundCorrector
            bg_corrector = TIRFBackgroundCorrector()
            
            # Gentle background correction to preserve edges
            bg_params = {
                'correction_method': 'gaussian_high_pass',
                'gaussian_sigma': 20,
                'normalize_intensity': False
            }
            
            corrected_window = bg_corrector.process()
            if corrected_window:
                corrected_window.setAsCurrentWindow()
            
            self.log_analysis_step("Preprocessing", bg_params)
            
            # Step 2: Edge Detection and Validation
            progress.setLabelText("Step 2/5: Detecting cell edges...")
            progress.setValue(2)
            QApplication.processEvents()
            
            from membrane_dynamics_analyzer import MembraneDynamicsAnalyzer
            membrane_analyzer = MembraneDynamicsAnalyzer()
            
            membrane_params = {
                'edge_detection_method': 'canny',
                'gaussian_sigma': 2.0,
                'canny_low_threshold': 0.1,
                'canny_high_threshold': 0.2,
                'edge_smoothing': 5
            }
            
            # Step 3: Velocity Field Calculation
            progress.setLabelText("Step 3/5: Calculating velocity fields...")
            progress.setValue(3)
            QApplication.processEvents()
            
            membrane_analyzer.run_analysis()
            
            if hasattr(membrane_analyzer, 'edge_velocities'):
                velocity_stats = self.analyze_velocity_statistics(membrane_analyzer.edge_velocities)
                self.results['velocity_analysis'] = velocity_stats
                self.log_analysis_step("Velocity Analysis", {}, velocity_stats)
            
            # Step 4: Protrusion/Retraction Analysis
            progress.setLabelText("Step 4/5: Analyzing protrusion events...")
            progress.setValue(4)
            QApplication.processEvents()
            
            if hasattr(membrane_analyzer, 'events'):
                event_stats = self.analyze_membrane_events(membrane_analyzer.events)
                self.results['membrane_events'] = event_stats
                self.log_analysis_step("Event Analysis", {}, event_stats)
            
            # Step 5: Generate Report
            progress.setLabelText("Step 5/5: Generating membrane dynamics report...")
            progress.setValue(5)
            QApplication.processEvents()
            
            self.generate_membrane_report(original_window.name)
            
            progress.close()
            g.alert("Membrane dynamics workflow completed!")
            
        except Exception as e:
            progress.close()
            g.alert(f"Workflow failed: {str(e)}")
    
    def colocalization_workflow(self, channel2_window=None):
        """Complete colocalization analysis workflow"""
        
        if g.win is None:
            g.alert("No image loaded! Please load channel 1 first.")
            return
        
        if channel2_window is None:
            g.alert("Please specify channel 2 window for colocalization analysis.")
            return
        
        original_window = g.win
        
        progress = QProgressDialog("Running colocalization workflow...", "Cancel", 0, 5)
        progress.setWindowTitle("Colocalization Workflow")
        progress.show()
        QApplication.processEvents()
        
        try:
            # Step 1: Channel Registration Check
            progress.setLabelText("Step 1/5: Checking channel registration...")
            progress.setValue(1)
            QApplication.processEvents()
            
            registration_check = self.check_channel_registration(g.win.image, channel2_window.image)
            self.log_analysis_step("Registration Check", {}, registration_check)
            
            if registration_check['drift'] > 2:
                g.alert("Warning: Significant drift detected between channels. Consider image registration.")
            
            # Step 2: Optimize Detection Parameters
            progress.setLabelText("Step 2/5: Optimizing detection parameters...")
            progress.setValue(2)
            QApplication.processEvents()
            
            optimal_params = self.optimize_colocalization_parameters(g.win.image, channel2_window.image)
            self.log_analysis_step("Parameter Optimization", optimal_params)
            
            # Step 3: Colocalization Analysis
            progress.setLabelText("Step 3/5: Running colocalization analysis...")
            progress.setValue(3)
            QApplication.processEvents()
            
            from colocalization_analyzer import ColocalizationAnalyzer
            coloc_analyzer = ColocalizationAnalyzer()
            
            # Set optimized parameters
            coloc_params = {
                'detection_threshold_ch1': optimal_params['threshold_ch1'],
                'detection_threshold_ch2': optimal_params['threshold_ch2'],
                'colocalization_distance': 2.0,
                'randomization_test': True,
                'n_randomizations': 100
            }
            
            coloc_analyzer.run_analysis()
            
            # Step 4: Statistical Validation
            progress.setLabelText("Step 4/5: Statistical validation...")
            progress.setValue(4)
            QApplication.processEvents()
            
            if hasattr(coloc_analyzer, 'results'):
                statistical_summary = self.validate_colocalization_statistics(coloc_analyzer.results)
                self.results['colocalization'] = statistical_summary
                self.log_analysis_step("Statistical Validation", {}, statistical_summary)
            
            # Step 5: Generate Report
            progress.setLabelText("Step 5/5: Generating colocalization report...")
            progress.setValue(5)
            QApplication.processEvents()
            
            self.generate_colocalization_report(original_window.name, channel2_window.name)
            
            progress.close()
            g.alert("Colocalization workflow completed!")
            
        except Exception as e:
            progress.close()
            g.alert(f"Workflow failed: {str(e)}")
    
    # Analysis helper functions
    
    def analyze_track_statistics(self, tracks):
        """Analyze statistical properties of tracks"""
        if not tracks:
            return {}
        
        track_lengths = [len(track['frames']) for track in tracks]
        track_displacements = []
        
        for track in tracks:
            if len(track['positions']) > 1:
                positions = np.array(track['positions'])
                displacement = np.sqrt(np.sum((positions[-1] - positions[0])**2))
                track_displacements.append(displacement)
        
        return {
            'total_tracks': len(tracks),
            'mean_track_length': np.mean(track_lengths),
            'median_track_length': np.median(track_lengths),
            'mean_displacement': np.mean(track_displacements) if track_displacements else 0,
            'track_length_distribution': track_lengths,
            'displacement_distribution': track_displacements
        }
    
    def calculate_diffusion_coefficients(self, tracks):
        """Calculate diffusion coefficients from tracks"""
        diffusion_coeffs = []
        
        for track in tracks:
            if len(track['positions']) < 4:  # Need minimum points for MSD
                continue
            
            positions = np.array(track['positions'])
            
            # Calculate mean squared displacement (MSD)
            msd_values = []
            time_lags = []
            
            max_lag = min(10, len(positions) // 2)  # Use up to 10 time points
            
            for lag in range(1, max_lag):
                displacements = []
                for i in range(len(positions) - lag):
                    dx = positions[i + lag][0] - positions[i][0]
                    dy = positions[i + lag][1] - positions[i][1]
                    displacements.append(dx**2 + dy**2)
                
                if displacements:
                    msd_values.append(np.mean(displacements))
                    time_lags.append(lag)
            
            # Fit linear slope to get diffusion coefficient
            if len(msd_values) > 2:
                # D = slope / 4 (for 2D diffusion)
                slope = np.polyfit(time_lags, msd_values, 1)[0]
                D = slope / 4  # pixels²/frame
                diffusion_coeffs.append(D)
        
        return {
            'diffusion_coefficients': diffusion_coeffs,
            'mean_diffusion_coeff': np.mean(diffusion_coeffs) if diffusion_coeffs else 0,
            'median_diffusion_coeff': np.median(diffusion_coeffs) if diffusion_coeffs else 0
        }
    
    def assess_image_quality(self, image_stack):
        """Assess image quality metrics"""
        # Calculate SNR
        signal = np.mean(image_stack)
        noise = np.std(image_stack)
        snr = signal / noise if noise > 0 else 0
        
        # Check for saturation
        max_value = np.max(image_stack)
        saturation_level = np.sum(image_stack >= 0.98 * max_value) / image_stack.size
        
        # Temporal stability
        frame_means = np.mean(image_stack, axis=(1, 2))
        temporal_cv = np.std(frame_means) / np.mean(frame_means) if np.mean(frame_means) > 0 else 0
        
        return {
            'snr': snr,
            'saturation_level': saturation_level,
            'temporal_cv': temporal_cv,
            'mean_intensity': signal,
            'noise_level': noise
        }
    
    def auto_detect_photobleaching_spots(self):
        """Automatically detect suitable spots for photobleaching analysis"""
        # Simple implementation - detect local maxima
        first_frame = g.win.image[0]
        
        from scipy import ndimage
        from skimage import filters
        
        # Smooth and find local maxima
        smoothed = ndimage.gaussian_filter(first_frame.astype(float), sigma=1.5)
        threshold = np.mean(smoothed) + 3 * np.std(smoothed)
        
        local_maxima = filters.peak_local_maxima(
            smoothed, min_distance=10, threshold_abs=threshold, indices=False
        )
        
        # Get coordinates
        positions = np.where(local_maxima)
        return list(zip(positions[1], positions[0]))  # (x, y) format
    
    def analyze_oligomerization_distribution(self, photobleaching_results):
        """Analyze the distribution of oligomerization states"""
        if not photobleaching_results:
            return {}
        
        step_counts = [result['step_count'] for result in photobleaching_results]
        
        # Count distribution
        from collections import Counter
        step_distribution = Counter(step_counts)
        
        # Statistical analysis
        mode_steps = max(step_distribution, key=step_distribution.get)
        
        return {
            'step_distribution': dict(step_distribution),
            'total_spots': len(step_counts),
            'mean_steps': np.mean(step_counts),
            'median_steps': np.median(step_counts),
            'mode_steps': mode_steps,
            'step_counts': step_counts
        }
    
    def check_channel_registration(self, ch1_image, ch2_image):
        """Check registration between two channels"""
        # Simple cross-correlation based drift detection
        frame1_ch1 = ch1_image[0] if len(ch1_image.shape) > 2 else ch1_image
        frame1_ch2 = ch2_image[0] if len(ch2_image.shape) > 2 else ch2_image
        
        # Calculate cross-correlation
        from scipy.signal import correlate2d
        
        correlation = correlate2d(frame1_ch1, frame1_ch2, mode='same')
        max_pos = np.unravel_index(np.argmax(correlation), correlation.shape)
        center = (correlation.shape[0] // 2, correlation.shape[1] // 2)
        
        drift = np.sqrt((max_pos[0] - center[0])**2 + (max_pos[1] - center[1])**2)
        
        return {
            'drift': drift,
            'drift_x': max_pos[1] - center[1],
            'drift_y': max_pos[0] - center[0],
            'correlation_max': np.max(correlation)
        }
    
    def optimize_colocalization_parameters(self, ch1_image, ch2_image):
        """Optimize detection parameters for colocalization analysis"""
        # Simple optimization based on image statistics
        ch1_stats = {
            'mean': np.mean(ch1_image),
            'std': np.std(ch1_image)
        }
        
        ch2_stats = {
            'mean': np.mean(ch2_image),
            'std': np.std(ch2_image)
        }
        
        # Set thresholds based on image statistics
        threshold_ch1 = 3.0  # Conservative default
        threshold_ch2 = 3.0
        
        # Adjust based on SNR
        snr_ch1 = ch1_stats['mean'] / ch1_stats['std']
        snr_ch2 = ch2_stats['mean'] / ch2_stats['std']
        
        if snr_ch1 > 10:
            threshold_ch1 = 4.0  # Higher threshold for high SNR
        elif snr_ch1 < 5:
            threshold_ch1 = 2.5  # Lower threshold for low SNR
        
        if snr_ch2 > 10:
            threshold_ch2 = 4.0
        elif snr_ch2 < 5:
            threshold_ch2 = 2.5
        
        return {
            'threshold_ch1': threshold_ch1,
            'threshold_ch2': threshold_ch2,
            'snr_ch1': snr_ch1,
            'snr_ch2': snr_ch2
        }
    
    # Report generation functions
    
    def generate_single_molecule_report(self, dataset_name):
        """Generate comprehensive single molecule analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_single_molecule_report_{timestamp}.md"
        
        report = f"""# Single Molecule Analysis Report
**Dataset:** {dataset_name}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Software:** FLIKA TIRF Analysis Suite v1.0

## Analysis Summary
"""
        
        if 'track_statistics' in self.results:
            stats = self.results['track_statistics']
            report += f"""
### Tracking Results
- **Total tracks detected:** {stats['total_tracks']}
- **Mean track length:** {stats['mean_track_length']:.1f} frames
- **Median track length:** {stats['median_track_length']:.1f} frames
- **Mean displacement:** {stats['mean_displacement']:.2f} pixels
"""
        
        if 'diffusion_analysis' in self.results:
            diff = self.results['diffusion_analysis']
            report += f"""
### Diffusion Analysis
- **Mean diffusion coefficient:** {diff['mean_diffusion_coeff']:.4f} pixels²/frame
- **Median diffusion coefficient:** {diff['median_diffusion_coeff']:.4f} pixels²/frame
- **Number of analyzed tracks:** {len(diff['diffusion_coefficients'])}
"""
        
        # Add analysis log
        report += "\n## Analysis Steps\n"
        for step in self.analysis_log:
            report += f"- **{step['step']}** at {step['timestamp']}\n"
        
        # Save report
        try:
            with open(filename, 'w') as f:
                f.write(report)
            g.alert(f"Single molecule report saved: {filename}")
        except Exception as e:
            g.alert(f"Error saving report: {str(e)}")
    
    def generate_oligomerization_report(self, dataset_name):
        """Generate oligomerization analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_oligomerization_report_{timestamp}.md"
        
        report = f"""# Photobleaching Oligomerization Report
**Dataset:** {dataset_name}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Oligomerization Analysis
"""
        
        if 'oligomerization' in self.results:
            oligo = self.results['oligomerization']
            report += f"""
### Step Count Distribution
- **Total spots analyzed:** {oligo['total_spots']}
- **Mean steps per spot:** {oligo['mean_steps']:.2f}
- **Median steps per spot:** {oligo['median_steps']:.1f}
- **Most probable oligomerization state:** {oligo['mode_steps']}

### Distribution Details:
"""
            for steps, count in oligo['step_distribution'].items():
                percentage = (count / oligo['total_spots']) * 100
                report += f"- **{steps} steps:** {count} spots ({percentage:.1f}%)\n"
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            g.alert(f"Oligomerization report saved: {filename}")
        except Exception as e:
            g.alert(f"Error saving report: {str(e)}")
    
    def generate_membrane_report(self, dataset_name):
        """Generate membrane dynamics report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_membrane_report_{timestamp}.md"
        
        report = f"""# Membrane Dynamics Analysis Report
**Dataset:** {dataset_name}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Membrane Dynamics Summary
[Add specific results from membrane analysis]
"""
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            g.alert(f"Membrane dynamics report saved: {filename}")
        except Exception as e:
            g.alert(f"Error saving report: {str(e)}")
    
    def generate_colocalization_report(self, ch1_name, ch2_name):
        """Generate colocalization analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ch1_name}_{ch2_name}_colocalization_report_{timestamp}.md"
        
        report = f"""# Colocalization Analysis Report
**Channel 1:** {ch1_name}
**Channel 2:** {ch2_name}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Colocalization Summary
[Add specific results from colocalization analysis]
"""
        
        try:
            with open(filename, 'w') as f:
                f.write(report)
            g.alert(f"Colocalization report saved: {filename}")
        except Exception as e:
            g.alert(f"Error saving report: {str(e)}")

# Global workflow manager instance
workflow_manager = TIRFWorkflowManager()

# Menu integration functions
def run_single_molecule_workflow():
    """Run complete single molecule analysis workflow"""
    workflow_manager.single_molecule_complete_workflow()

def run_photobleaching_workflow():
    """Run complete photobleaching oligomerization workflow"""
    workflow_manager.photobleaching_oligomerization_workflow()

def run_membrane_dynamics_workflow():
    """Run complete membrane dynamics workflow"""
    workflow_manager.membrane_dynamics_workflow()

def run_colocalization_workflow():
    """Run complete colocalization workflow (requires channel 2 selection)"""
    # This would need additional GUI for channel selection
    g.alert("Please use the Colocalization Analyzer plugin directly and select Channel 2 window.")

def show_workflow_help():
    """Show help for integrated workflows"""
    help_text = """
TIRF Analysis Integrated Workflows

These workflows combine multiple plugins for complete analysis pipelines:

1. Single Molecule Workflow:
   • Background correction
   • Single molecule tracking  
   • Diffusion analysis
   • Statistical summary

2. Photobleaching Workflow:
   • Image quality assessment
   • Automatic ROI detection
   • Step counting analysis
   • Oligomerization statistics

3. Membrane Dynamics Workflow:
   • Edge-preserving preprocessing
   • Edge detection and tracking
   • Velocity field analysis
   • Protrusion/retraction events

4. Colocalization Workflow:
   • Channel registration check
   • Parameter optimization
   • Multi-method colocalization
   • Statistical validation

Each workflow generates a comprehensive analysis report.
"""
    g.alert(help_text)

# Register workflow functions in menu
run_single_molecule_workflow.menu_path = 'Plugins>TIRF Analysis>Integrated Workflows>Single Molecule Complete Analysis'
run_photobleaching_workflow.menu_path = 'Plugins>TIRF Analysis>Integrated Workflows>Photobleaching Oligomerization Study'
run_membrane_dynamics_workflow.menu_path = 'Plugins>TIRF Analysis>Integrated Workflows>Membrane Dynamics Analysis'
run_colocalization_workflow.menu_path = 'Plugins>TIRF Analysis>Integrated Workflows>Multi-Channel Colocalization'
show_workflow_help.menu_path = 'Plugins>TIRF Analysis>Integrated Workflows>Workflow Help'