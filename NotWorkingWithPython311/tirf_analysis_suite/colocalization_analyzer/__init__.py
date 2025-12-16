# colocalization_analyzer/__init__.py
"""
Colocalization Analyzer Plugin for FLIKA
Advanced colocalization analysis for multi-channel TIRF microscopy
"""

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, WindowSelector
from flika.roi import makeROI
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTabWidget, QWidget

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class ColocalizationAnalyzer(BaseProcess):
    """
    Comprehensive colocalization analysis for multi-channel imaging
    """
    
    def __init__(self):
        super().__init__()
        self.channel1_spots = []
        self.channel2_spots = []
        self.colocalized_pairs = []
        self.results_tabs = None
        
    def get_init_settings_dict(self):
        return {
            'channel2_window': None,
            'detection_threshold_ch1': 3.0,
            'detection_threshold_ch2': 3.0,
            'colocalization_distance': 2.0,
            'min_spot_intensity': 100,
            'gaussian_fit': True,
            'temporal_analysis': True,
            'pearson_correlation': True,
            'manders_coefficients': True,
            'randomization_test': True,
            'n_randomizations': 100
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['channel2_window'] = self.channel2_window.value()
        params['detection_threshold_ch1'] = self.detection_threshold_ch1.value()
        params['detection_threshold_ch2'] = self.detection_threshold_ch2.value()
        params['colocalization_distance'] = self.colocalization_distance.value()
        params['min_spot_intensity'] = self.min_spot_intensity.value()
        params['gaussian_fit'] = self.gaussian_fit.isChecked()
        params['temporal_analysis'] = self.temporal_analysis.isChecked()
        params['pearson_correlation'] = self.pearson_correlation.isChecked()
        params['manders_coefficients'] = self.manders_coefficients.isChecked()
        params['randomization_test'] = self.randomization_test.isChecked()
        params['n_randomizations'] = int(self.n_randomizations.value())
        return params
    
    def get_name(self):
        return 'Colocalization Analyzer'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Colocalization Analyzer'
    
    def setupGUI(self):
        super().setupGUI()
        self.detection_threshold_ch1.setRange(1.0, 10.0)
        self.detection_threshold_ch2.setRange(1.0, 10.0)
        self.colocalization_distance.setRange(0.5, 10.0)
        self.min_spot_intensity.setRange(10, 10000)
        self.n_randomizations.setRange(10, 1000)
        
        # Add analysis button
        self.analyze_button = QPushButton("Run Colocalization Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout().addWidget(self.analyze_button)
        
        # Add results tabs
        self.results_tabs = QTabWidget()
        self.layout().addWidget(self.results_tabs)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.layout().addWidget(self.export_button)
    
    def detect_spots_in_image(self, image, threshold, min_intensity):
        """Detect fluorescent spots in an image"""
        # Gaussian filtering for noise reduction
        filtered = ndimage.gaussian_filter(image.astype(float), sigma=1.0)
        
        # Local maxima detection
        local_maxima = ndimage.maximum_filter(filtered, size=3) == filtered
        
        # Threshold detection
        above_threshold = filtered > (threshold * np.std(filtered) + np.mean(filtered))
        above_min_intensity = filtered > min_intensity
        
        # Combine conditions
        spots = local_maxima & above_threshold & above_min_intensity
        
        # Get spot coordinates and properties
        spot_coords = np.where(spots)
        spot_positions = list(zip(spot_coords[1], spot_coords[0]))  # (x, y)
        spot_intensities = [image[y, x] for x, y in spot_positions]
        
        return spot_positions, spot_intensities
    
    def gaussian_2d_fit(self, image, x, y, window_size=5):
        """Fit 2D Gaussian for subpixel localization"""
        try:
            from scipy.optimize import curve_fit
            
            # Extract window
            x_int, y_int = int(x), int(y)
            x_min = max(0, x_int - window_size)
            x_max = min(image.shape[1], x_int + window_size + 1)
            y_min = max(0, y_int - window_size)
            y_max = min(image.shape[0], y_int + window_size + 1)
            
            window = image[y_min:y_max, x_min:x_max]
            
            # Create coordinate grids
            x_coords, y_coords = np.meshgrid(
                np.arange(x_min, x_max),
                np.arange(y_min, y_max)
            )
            
            def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
                x, y = coords
                return (amplitude * np.exp(-(((x-x0)/sigma_x)**2 + ((y-y0)/sigma_y)**2)/2) + offset).ravel()
            
            # Initial guess
            amplitude = np.max(window) - np.min(window)
            offset = np.min(window)
            initial_guess = [amplitude, x, y, 2.0, 2.0, offset]
            
            # Fit
            popt, _ = curve_fit(
                gaussian_2d,
                (x_coords, y_coords),
                window.ravel(),
                p0=initial_guess,
                maxfev=1000
            )
            
            return popt[1], popt[2], popt[0]  # x, y, amplitude
            
        except:
            return x, y, image[int(y), int(x)]
    
    def find_colocalized_spots(self, spots1, spots2, max_distance):
        """Find colocalized spots between two channels"""
        if len(spots1) == 0 or len(spots2) == 0:
            return []
        
        # Convert to numpy arrays
        coords1 = np.array([[x, y] for (x, y), _ in spots1])
        coords2 = np.array([[x, y] for (x, y), _ in spots2])
        
        # Calculate distance matrix
        distances = np.sqrt(((coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Find pairs within distance threshold
        colocalized = []
        used_ch2 = set()
        
        for i in range(len(coords1)):
            # Find closest spot in channel 2
            closest_distances = distances[i, :]
            valid_indices = [j for j in range(len(closest_distances)) 
                           if closest_distances[j] <= max_distance and j not in used_ch2]
            
            if valid_indices:
                closest_idx = min(valid_indices, key=lambda j: closest_distances[j])
                distance = closest_distances[closest_idx]
                
                colocalized.append({
                    'ch1_idx': i,
                    'ch2_idx': closest_idx,
                    'ch1_pos': coords1[i],
                    'ch2_pos': coords2[closest_idx],
                    'ch1_intensity': spots1[i][1],
                    'ch2_intensity': spots2[closest_idx][1],
                    'distance': distance
                })
                
                used_ch2.add(closest_idx)
        
        return colocalized
    
    def calculate_pearson_correlation(self, image1, image2, mask=None):
        """Calculate Pearson correlation coefficient"""
        if mask is not None:
            pixels1 = image1[mask]
            pixels2 = image2[mask]
        else:
            pixels1 = image1.flatten()
            pixels2 = image2.flatten()
        
        # Remove NaN and infinite values
        valid = np.isfinite(pixels1) & np.isfinite(pixels2)
        pixels1 = pixels1[valid]
        pixels2 = pixels2[valid]
        
        if len(pixels1) < 2:
            return 0, 1
        
        correlation, p_value = stats.pearsonr(pixels1, pixels2)
        return correlation, p_value
    
    def calculate_manders_coefficients(self, image1, image2, threshold1=None, threshold2=None):
        """Calculate Manders' colocalization coefficients"""
        if threshold1 is None:
            threshold1 = np.mean(image1) + 2 * np.std(image1)
        if threshold2 is None:
            threshold2 = np.mean(image2) + 2 * np.std(image2)
        
        # Manders' coefficients
        # M1: fraction of ch1 signal that colocalizes with ch2 signal above threshold
        # M2: fraction of ch2 signal that colocalizes with ch1 signal above threshold
        
        ch1_above_threshold = image1 > threshold1
        ch2_above_threshold = image2 > threshold2
        
        if np.sum(ch1_above_threshold) == 0:
            M1 = 0
        else:
            M1 = np.sum(image1[ch1_above_threshold & ch2_above_threshold]) / np.sum(image1[ch1_above_threshold])
        
        if np.sum(ch2_above_threshold) == 0:
            M2 = 0
        else:
            M2 = np.sum(image2[ch2_above_threshold & ch1_above_threshold]) / np.sum(image2[ch2_above_threshold])
        
        return M1, M2
    
    def randomization_test(self, spots1, spots2, max_distance, image_shape, n_randomizations=100):
        """Perform randomization test for colocalization significance"""
        # Observed colocalization
        observed_coloc = len(self.find_colocalized_spots(spots1, spots2, max_distance))
        
        # Randomization
        random_coloc_counts = []
        
        for _ in range(n_randomizations):
            # Randomize positions of channel 2 spots
            random_spots2 = []
            for _, intensity in spots2:
                random_x = np.random.uniform(0, image_shape[1])
                random_y = np.random.uniform(0, image_shape[0])
                random_spots2.append(((random_x, random_y), intensity))
            
            # Count colocalization in randomized data
            random_coloc = len(self.find_colocalized_spots(spots1, random_spots2, max_distance))
            random_coloc_counts.append(random_coloc)
        
        # Calculate p-value
        p_value = np.mean(np.array(random_coloc_counts) >= observed_coloc)
        
        return {
            'observed': observed_coloc,
            'expected_mean': np.mean(random_coloc_counts),
            'expected_std': np.std(random_coloc_counts),
            'p_value': p_value,
            'random_counts': random_coloc_counts
        }
    
    def run_analysis(self):
        """Run comprehensive colocalization analysis"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        
        # Get channel 2 window
        if params['channel2_window'] is None:
            g.alert("Please select Channel 2 window!")
            return
        
        channel1_stack = g.win.image
        channel2_stack = params['channel2_window'].image
        
        if channel1_stack.shape != channel2_stack.shape:
            g.alert("Channel 1 and Channel 2 must have the same dimensions!")
            return
        
        g.m.statusBar().showMessage("Running colocalization analysis...")
        
        # Initialize results storage
        frame_results = []
        all_ch1_spots = []
        all_ch2_spots = []
        all_colocalized = []
        
        # Analyze each frame
        for frame_idx in range(channel1_stack.shape[0]):
            ch1_frame = channel1_stack[frame_idx]
            ch2_frame = channel2_stack[frame_idx]
            
            # Detect spots in both channels
            ch1_positions, ch1_intensities = self.detect_spots_in_image(
                ch1_frame, params['detection_threshold_ch1'], params['min_spot_intensity']
            )
            ch2_positions, ch2_intensities = self.detect_spots_in_image(
                ch2_frame, params['detection_threshold_ch2'], params['min_spot_intensity']
            )
            
            # Combine positions and intensities
            ch1_spots = list(zip(ch1_positions, ch1_intensities))
            ch2_spots = list(zip(ch2_positions, ch2_intensities))
            
            # Refine with Gaussian fitting if requested
            if params['gaussian_fit']:
                refined_ch1 = []
                for (x, y), intensity in ch1_spots:
                    x_ref, y_ref, amp_ref = self.gaussian_2d_fit(ch1_frame, x, y)
                    refined_ch1.append(((x_ref, y_ref), amp_ref))
                ch1_spots = refined_ch1
                
                refined_ch2 = []
                for (x, y), intensity in ch2_spots:
                    x_ref, y_ref, amp_ref = self.gaussian_2d_fit(ch2_frame, x, y)
                    refined_ch2.append(((x_ref, y_ref), amp_ref))
                ch2_spots = refined_ch2
            
            # Find colocalized spots
            colocalized = self.find_colocalized_spots(
                ch1_spots, ch2_spots, params['colocalization_distance']
            )
            
            # Calculate correlation metrics
            correlation, correlation_p = self.calculate_pearson_correlation(ch1_frame, ch2_frame)
            
            # Calculate Manders' coefficients
            M1, M2 = self.calculate_manders_coefficients(ch1_frame, ch2_frame)
            
            # Store frame results
            frame_result = {
                'frame': frame_idx,
                'ch1_spots': len(ch1_spots),
                'ch2_spots': len(ch2_spots),
                'colocalized_spots': len(colocalized),
                'colocalization_fraction_ch1': len(colocalized) / max(1, len(ch1_spots)),
                'colocalization_fraction_ch2': len(colocalized) / max(1, len(ch2_spots)),
                'pearson_correlation': correlation,
                'correlation_p_value': correlation_p,
                'manders_M1': M1,
                'manders_M2': M2
            }
            
            frame_results.append(frame_result)
            all_ch1_spots.extend(ch1_spots)
            all_ch2_spots.extend(ch2_spots)
            all_colocalized.extend(colocalized)
        
        # Perform randomization test on combined data
        randomization_result = None
        if params['randomization_test']:
            g.m.statusBar().showMessage("Performing randomization test...")
            randomization_result = self.randomization_test(
                all_ch1_spots, all_ch2_spots, params['colocalization_distance'],
                channel1_stack[0].shape, params['n_randomizations']
            )
        
        # Store results
        self.frame_results = frame_results
        self.all_ch1_spots = all_ch1_spots
        self.all_ch2_spots = all_ch2_spots
        self.all_colocalized = all_colocalized
        self.randomization_result = randomization_result
        
        # Display results
        self.display_results()
        
        g.m.statusBar().showMessage("Colocalization analysis complete!", 3000)
    
    def display_results(self):
        """Display comprehensive results in tabs"""
        # Clear existing tabs
        for i in range(self.results_tabs.count()):
            self.results_tabs.removeTab(0)
        
        # Tab 1: Summary Statistics
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        # Calculate summary statistics
        total_ch1 = len(self.all_ch1_spots)
        total_ch2 = len(self.all_ch2_spots)
        total_colocalized = len(self.all_colocalized)
        
        summary = f"""=== Colocalization Analysis Summary ===

Total Spots Detected:
  Channel 1: {total_ch1}
  Channel 2: {total_ch2}
  Colocalized: {total_colocalized}

Colocalization Fractions:
  Ch1 colocalized: {total_colocalized/max(1,total_ch1):.3f}
  Ch2 colocalized: {total_colocalized/max(1,total_ch2):.3f}

Frame-by-frame Statistics:
  Average Ch1 spots/frame: {np.mean([r['ch1_spots'] for r in self.frame_results]):.1f}
  Average Ch2 spots/frame: {np.mean([r['ch2_spots'] for r in self.frame_results]):.1f}
  Average colocalized/frame: {np.mean([r['colocalized_spots'] for r in self.frame_results]):.1f}

Correlation Analysis:
  Mean Pearson correlation: {np.mean([r['pearson_correlation'] for r in self.frame_results]):.3f}
  Mean Manders M1: {np.mean([r['manders_M1'] for r in self.frame_results]):.3f}
  Mean Manders M2: {np.mean([r['manders_M2'] for r in self.frame_results]):.3f}
"""
        
        if self.randomization_result:
            summary += f"""
Randomization Test:
  Observed colocalization: {self.randomization_result['observed']}
  Expected (random): {self.randomization_result['expected_mean']:.1f} ± {self.randomization_result['expected_std']:.1f}
  P-value: {self.randomization_result['p_value']:.4f}
  Significance: {'Yes' if self.randomization_result['p_value'] < 0.05 else 'No'} (p < 0.05)
"""
        
        summary_text.setText(summary)
        summary_layout.addWidget(summary_text)
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")
        
        # Tab 2: Create visualizations
        self.create_visualizations()
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Colocalization Analysis Results')
        
        # Plot 1: Spot counts over time
        frames = [r['frame'] for r in self.frame_results]
        ch1_counts = [r['ch1_spots'] for r in self.frame_results]
        ch2_counts = [r['ch2_spots'] for r in self.frame_results]
        coloc_counts = [r['colocalized_spots'] for r in self.frame_results]
        
        axes[0,0].plot(frames, ch1_counts, 'r-', label='Channel 1', alpha=0.7)
        axes[0,0].plot(frames, ch2_counts, 'g-', label='Channel 2', alpha=0.7)
        axes[0,0].plot(frames, coloc_counts, 'b-', label='Colocalized', linewidth=2)
        axes[0,0].set_xlabel('Frame')
        axes[0,0].set_ylabel('Spot Count')
        axes[0,0].set_title('Spot Counts Over Time')
        axes[0,0].legend()
        
        # Plot 2: Colocalization fractions
        ch1_fractions = [r['colocalization_fraction_ch1'] for r in self.frame_results]
        ch2_fractions = [r['colocalization_fraction_ch2'] for r in self.frame_results]
        
        axes[0,1].plot(frames, ch1_fractions, 'r-', label='Ch1 fraction', alpha=0.7)
        axes[0,1].plot(frames, ch2_fractions, 'g-', label='Ch2 fraction', alpha=0.7)
        axes[0,1].set_xlabel('Frame')
        axes[0,1].set_ylabel('Colocalization Fraction')
        axes[0,1].set_title('Colocalization Fractions')
        axes[0,1].legend()
        axes[0,1].set_ylim(0, 1)
        
        # Plot 3: Pearson correlation over time
        correlations = [r['pearson_correlation'] for r in self.frame_results]
        axes[0,2].plot(frames, correlations, 'purple', linewidth=2)
        axes[0,2].set_xlabel('Frame')
        axes[0,2].set_ylabel('Pearson Correlation')
        axes[0,2].set_title('Pearson Correlation Over Time')
        axes[0,2].set_ylim(-1, 1)
        axes[0,2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot 4: Distance distribution
        if self.all_colocalized:
            distances = [pair['distance'] for pair in self.all_colocalized]
            axes[1,0].hist(distances, bins=20, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Distance (pixels)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Colocalization Distance Distribution')
        
        # Plot 5: Intensity correlation scatter
        if self.all_colocalized:
            ch1_intensities = [pair['ch1_intensity'] for pair in self.all_colocalized]
            ch2_intensities = [pair['ch2_intensity'] for pair in self.all_colocalized]
            axes[1,1].scatter(ch1_intensities, ch2_intensities, alpha=0.6)
            axes[1,1].set_xlabel('Channel 1 Intensity')
            axes[1,1].set_ylabel('Channel 2 Intensity')
            axes[1,1].set_title('Colocalized Spot Intensities')
        
        # Plot 6: Randomization test results
        if self.randomization_result:
            random_counts = self.randomization_result['random_counts']
            observed = self.randomization_result['observed']
            
            axes[1,2].hist(random_counts, bins=20, alpha=0.7, color='gray', label='Random')
            axes[1,2].axvline(observed, color='red', linewidth=3, label=f'Observed ({observed})')
            axes[1,2].set_xlabel('Colocalized Spots')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].set_title('Randomization Test')
            axes[1,2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self):
        """Export detailed results to CSV files"""
        if not hasattr(self, 'frame_results'):
            g.alert("No results to export! Run analysis first.")
            return
        
        base_name = g.win.name
        
        # Export frame-by-frame results
        frame_df = pd.DataFrame(self.frame_results)
        frame_filename = f"{base_name}_colocalization_frames.csv"
        frame_df.to_csv(frame_filename, index=False)
        
        # Export colocalized spot details
        if self.all_colocalized:
            coloc_data = []
            for pair in self.all_colocalized:
                coloc_data.append({
                    'ch1_x': pair['ch1_pos'][0],
                    'ch1_y': pair['ch1_pos'][1],
                    'ch2_x': pair['ch2_pos'][0],
                    'ch2_y': pair['ch2_pos'][1],
                    'ch1_intensity': pair['ch1_intensity'],
                    'ch2_intensity': pair['ch2_intensity'],
                    'distance': pair['distance']
                })
            
            coloc_df = pd.DataFrame(coloc_data)
            coloc_filename = f"{base_name}_colocalized_spots.csv"
            coloc_df.to_csv(coloc_filename, index=False)
        
        g.alert(f"Results exported to {frame_filename} and related files")
    
    def process(self):
        """Process method required by BaseProcess"""
        self.run_analysis()
        return None

# Register the plugin
ColocalizationAnalyzer.menu_path = 'Plugins>TIRF Analysis>Colocalization Analyzer'