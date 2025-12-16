# cluster_analyzer/__init__.py
"""
Protein Cluster Analyzer Plugin for FLIKA
Detects and analyzes protein clusters and aggregates in TIRF microscopy
"""

import numpy as np
import pandas as pd
from scipy import ndimage, spatial
from scipy.optimize import curve_fit
from sklearn.cluster import DBSCAN, KMeans
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from flika.roi import makeROI
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTabWidget, QWidget

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class ClusterAnalyzer(BaseProcess):
    """
    Comprehensive protein cluster detection and analysis
    """
    
    def __init__(self):
        super().__init__()
        self.detected_clusters = []
        self.cluster_properties = []
        self.density_maps = []
        self.results_tabs = None
        
    def get_init_settings_dict(self):
        return {
            'detection_method': ['threshold_watershed', 'dbscan_clustering', 'local_maxima', 'gradient_flow'],
            'intensity_threshold': 3.0,
            'min_cluster_size': 5,
            'max_cluster_size': 1000,
            'dbscan_eps': 3.0,
            'dbscan_min_samples': 5,
            'watershed_sigma': 2.0,
            'local_maxima_distance': 5,
            'cluster_merging_distance': 10.0,
            'analyze_temporal_dynamics': True,
            'calculate_density_maps': True,
            'fit_cluster_shapes': True,
            'background_subtraction': True
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['detection_method'] = self.detection_method.currentText()
        params['intensity_threshold'] = self.intensity_threshold.value()
        params['min_cluster_size'] = int(self.min_cluster_size.value())
        params['max_cluster_size'] = int(self.max_cluster_size.value())
        params['dbscan_eps'] = self.dbscan_eps.value()
        params['dbscan_min_samples'] = int(self.dbscan_min_samples.value())
        params['watershed_sigma'] = self.watershed_sigma.value()
        params['local_maxima_distance'] = int(self.local_maxima_distance.value())
        params['cluster_merging_distance'] = self.cluster_merging_distance.value()
        params['analyze_temporal_dynamics'] = self.analyze_temporal_dynamics.isChecked()
        params['calculate_density_maps'] = self.calculate_density_maps.isChecked()
        params['fit_cluster_shapes'] = self.fit_cluster_shapes.isChecked()
        params['background_subtraction'] = self.background_subtraction.isChecked()
        return params
    
    def get_name(self):
        return 'Cluster Analyzer'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Cluster Analyzer'
    
    def setupGUI(self):
        super().setupGUI()
        self.intensity_threshold.setRange(1.0, 20.0)
        self.min_cluster_size.setRange(2, 100)
        self.max_cluster_size.setRange(10, 10000)
        self.dbscan_eps.setRange(1.0, 20.0)
        self.dbscan_min_samples.setRange(2, 50)
        self.watershed_sigma.setRange(0.5, 10.0)
        self.local_maxima_distance.setRange(2, 50)
        self.cluster_merging_distance.setRange(1.0, 50.0)
        
        # Add analysis button
        self.analyze_button = QPushButton("Analyze Clusters")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout().addWidget(self.analyze_button)
        
        # Add results tabs
        self.results_tabs = QTabWidget()
        self.layout().addWidget(self.results_tabs)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.layout().addWidget(self.export_button)
    
    def preprocess_image(self, image, background_subtraction=True):
        """Preprocess image for cluster detection"""
        processed = image.astype(float)
        
        if background_subtraction:
            # Rolling ball background subtraction
            background = ndimage.grey_opening(processed, size=(50, 50))
            processed = processed - background
            processed = np.maximum(processed, 0)  # Remove negative values
        
        # Gaussian smoothing
        processed = ndimage.gaussian_filter(processed, sigma=1.0)
        
        return processed
    
    def detect_clusters_threshold_watershed(self, image, threshold, sigma, min_size, max_size):
        """Detect clusters using threshold and watershed segmentation"""
        # Threshold image
        threshold_value = threshold * np.std(image) + np.mean(image)
        binary = image > threshold_value
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_size)
        
        # Distance transform and watershed
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima as seeds
        local_maxima = filters.peak_local_maxima(distance, min_distance=int(sigma*2), 
                                                threshold_abs=sigma, indices=False)
        markers = ndimage.label(local_maxima)[0]
        
        # Watershed segmentation
        labels = segmentation.watershed(-distance, markers, mask=binary)
        
        return labels
    
    def detect_clusters_dbscan(self, image, eps, min_samples, threshold, min_size, max_size):
        """Detect clusters using DBSCAN clustering on bright pixels"""
        # Find bright pixels
        threshold_value = threshold * np.std(image) + np.mean(image)
        bright_pixels = np.where(image > threshold_value)
        
        if len(bright_pixels[0]) == 0:
            return np.zeros_like(image, dtype=int)
        
        # Prepare coordinates for clustering
        coords = np.column_stack([bright_pixels[0], bright_pixels[1]])
        
        # Weight by intensity
        intensities = image[bright_pixels]
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Create labeled image
        labels = np.zeros_like(image, dtype=int)
        
        current_label = 1
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_coords = coords[cluster_mask]
            
            # Check cluster size
            if len(cluster_coords) < min_size or len(cluster_coords) > max_size:
                continue
            
            # Assign label
            for coord in cluster_coords:
                labels[coord[0], coord[1]] = current_label
            
            current_label += 1
        
        return labels
    
    def detect_clusters_local_maxima(self, image, min_distance, threshold, min_size, max_size):
        """Detect clusters using local maxima detection"""
        # Find local maxima
        threshold_value = threshold * np.std(image) + np.mean(image)
        local_maxima = filters.peak_local_maxima(image, min_distance=min_distance,
                                                threshold_abs=threshold_value, indices=False)
        
        # Label connected components around maxima
        labels = ndimage.label(local_maxima)[0]
        
        # Expand regions using watershed
        distance = ndimage.distance_transform_edt(image > threshold_value)
        expanded_labels = segmentation.watershed(-image, labels, mask=image > threshold_value)
        
        return expanded_labels
    
    def detect_clusters_gradient_flow(self, image, threshold, min_size, max_size):
        """Detect clusters using gradient flow segmentation"""
        # Calculate gradient
        gradient = np.gradient(ndimage.gaussian_filter(image, sigma=1))
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        
        # Find regions with low gradient (cluster centers)
        low_gradient = gradient_magnitude < np.percentile(gradient_magnitude, 20)
        
        # Threshold on intensity
        threshold_value = threshold * np.std(image) + np.mean(image)
        high_intensity = image > threshold_value
        
        # Combine conditions
        cluster_seeds = low_gradient & high_intensity
        
        # Label and expand
        labels = ndimage.label(cluster_seeds)[0]
        
        # Watershed expansion
        expanded_labels = segmentation.watershed(-image, labels, mask=high_intensity)
        
        return expanded_labels
    
    def analyze_cluster_properties(self, image, labels):
        """Analyze properties of detected clusters"""
        properties = []
        
        # Get region properties
        regions = measure.regionprops(labels, intensity_image=image)
        
        for region in regions:
            # Basic properties
            area = region.area
            centroid = region.centroid
            intensity_mean = region.intensity_mean
            intensity_max = region.intensity_max
            intensity_min = region.intensity_min
            intensity_sum = region.intensity_sum
            
            # Geometric properties
            major_axis_length = region.major_axis_length
            minor_axis_length = region.minor_axis_length
            eccentricity = region.eccentricity
            orientation = region.orientation
            
            # Calculate additional metrics
            # Cluster intensity profile
            coords = region.coords
            intensities = image[coords[:, 0], coords[:, 1]]
            
            # Compactness (area / perimeter^2)
            perimeter = region.perimeter
            compactness = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            # Intensity weighted centroid
            weighted_centroid = np.average(coords, axis=0, weights=intensities)
            
            # Distance from geometric to intensity centroid
            centroid_shift = np.linalg.norm(np.array(centroid) - weighted_centroid)
            
            # Cluster radius (equivalent circle)
            equivalent_radius = np.sqrt(area / np.pi)
            
            # Intensity variance within cluster
            intensity_variance = np.var(intensities)
            intensity_cv = np.std(intensities) / intensity_mean if intensity_mean > 0 else 0
            
            properties.append({
                'label': region.label,
                'area': area,
                'centroid_y': centroid[0],
                'centroid_x': centroid[1],
                'weighted_centroid_y': weighted_centroid[0],
                'weighted_centroid_x': weighted_centroid[1],
                'centroid_shift': centroid_shift,
                'intensity_mean': intensity_mean,
                'intensity_max': intensity_max,
                'intensity_min': intensity_min,
                'intensity_sum': intensity_sum,
                'intensity_variance': intensity_variance,
                'intensity_cv': intensity_cv,
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length,
                'equivalent_radius': equivalent_radius,
                'eccentricity': eccentricity,
                'orientation': orientation,
                'compactness': compactness,
                'perimeter': perimeter
            })
        
        return properties
    
    def fit_cluster_shapes(self, image, labels, properties):
        """Fit elliptical shapes to clusters"""
        fitted_shapes = []
        
        for prop in properties:
            label_id = prop['label']
            
            # Get cluster pixels
            cluster_mask = labels == label_id
            coords = np.where(cluster_mask)
            
            if len(coords[0]) < 5:  # Need at least 5 points for ellipse fitting
                continue
            
            # Intensity-weighted coordinates
            intensities = image[coords]
            
            try:
                # Fit ellipse using method of moments
                y_coords = coords[0]
                x_coords = coords[1]
                
                # Weight by intensity
                weights = intensities / np.sum(intensities)
                
                # Calculate weighted moments
                mean_x = np.sum(x_coords * weights)
                mean_y = np.sum(y_coords * weights)
                
                # Second moments
                m20 = np.sum(weights * (x_coords - mean_x)**2)
                m02 = np.sum(weights * (y_coords - mean_y)**2)
                m11 = np.sum(weights * (x_coords - mean_x) * (y_coords - mean_y))
                
                # Ellipse parameters
                if m20 > 0 and m02 > 0:
                    # Semi-axes lengths
                    lambda1 = (m20 + m02 + np.sqrt((m20 - m02)**2 + 4*m11**2)) / 2
                    lambda2 = (m20 + m02 - np.sqrt((m20 - m02)**2 + 4*m11**2)) / 2
                    
                    major_axis = 2 * np.sqrt(lambda1)
                    minor_axis = 2 * np.sqrt(lambda2)
                    
                    # Orientation
                    if m11 == 0 and m20 >= m02:
                        angle = 0
                    elif m11 == 0 and m20 < m02:
                        angle = np.pi/2
                    else:
                        angle = 0.5 * np.arctan(2*m11 / (m20 - m02))
                    
                    fitted_shapes.append({
                        'label': label_id,
                        'center_x': mean_x,
                        'center_y': mean_y,
                        'major_axis': major_axis,
                        'minor_axis': minor_axis,
                        'angle': angle,
                        'eccentricity': np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
                    })
                
            except Exception as e:
                print(f"Shape fitting failed for cluster {label_id}: {e}")
                continue
        
        return fitted_shapes
    
    def calculate_density_map(self, image, bandwidth=10):
        """Calculate local protein density map"""
        # Gaussian kernel density estimation
        density_map = ndimage.gaussian_filter(image, sigma=bandwidth)
        
        # Normalize
        density_map = density_map / np.max(density_map)
        
        return density_map
    
    def analyze_cluster_temporal_dynamics(self, cluster_stack_properties):
        """Analyze temporal dynamics of clusters across frames"""
        if len(cluster_stack_properties) < 2:
            return {}
        
        # Track cluster formation/dissolution
        cluster_counts = [len(frame_props) for frame_props in cluster_stack_properties]
        
        # Average cluster properties over time
        all_areas = []
        all_intensities = []
        all_compactness = []
        
        for frame_props in cluster_stack_properties:
            frame_areas = [p['area'] for p in frame_props]
            frame_intensities = [p['intensity_mean'] for p in frame_props]
            frame_compactness = [p['compactness'] for p in frame_props]
            
            all_areas.extend(frame_areas)
            all_intensities.extend(frame_intensities)
            all_compactness.extend(frame_compactness)
        
        temporal_stats = {
            'cluster_counts': cluster_counts,
            'mean_cluster_count': np.mean(cluster_counts),
            'cluster_count_variance': np.var(cluster_counts),
            'mean_cluster_area': np.mean(all_areas) if all_areas else 0,
            'mean_cluster_intensity': np.mean(all_intensities) if all_intensities else 0,
            'mean_cluster_compactness': np.mean(all_compactness) if all_compactness else 0,
            'area_cv': np.std(all_areas) / np.mean(all_areas) if all_areas and np.mean(all_areas) > 0 else 0,
            'intensity_cv': np.std(all_intensities) / np.mean(all_intensities) if all_intensities and np.mean(all_intensities) > 0 else 0
        }
        
        return temporal_stats
    
    def run_analysis(self):
        """Run comprehensive cluster analysis"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Analyzing protein clusters...")
        
        # Initialize storage
        all_cluster_labels = []
        all_cluster_properties = []
        all_fitted_shapes = []
        all_density_maps = []
        
        # Process each frame
        for frame_idx in range(image_stack.shape[0]):
            frame = image_stack[frame_idx]
            
            # Preprocess
            processed_frame = self.preprocess_image(frame, params['background_subtraction'])
            
            # Detect clusters
            if params['detection_method'] == 'threshold_watershed':
                labels = self.detect_clusters_threshold_watershed(
                    processed_frame, params['intensity_threshold'], params['watershed_sigma'],
                    params['min_cluster_size'], params['max_cluster_size']
                )
            elif params['detection_method'] == 'dbscan_clustering':
                labels = self.detect_clusters_dbscan(
                    processed_frame, params['dbscan_eps'], params['dbscan_min_samples'],
                    params['intensity_threshold'], params['min_cluster_size'], params['max_cluster_size']
                )
            elif params['detection_method'] == 'local_maxima':
                labels = self.detect_clusters_local_maxima(
                    processed_frame, params['local_maxima_distance'], params['intensity_threshold'],
                    params['min_cluster_size'], params['max_cluster_size']
                )
            elif params['detection_method'] == 'gradient_flow':
                labels = self.detect_clusters_gradient_flow(
                    processed_frame, params['intensity_threshold'],
                    params['min_cluster_size'], params['max_cluster_size']
                )
            
            # Analyze cluster properties
            properties = self.analyze_cluster_properties(processed_frame, labels)
            
            # Fit cluster shapes if requested
            fitted_shapes = []
            if params['fit_cluster_shapes']:
                fitted_shapes = self.fit_cluster_shapes(processed_frame, labels, properties)
            
            # Calculate density map if requested
            density_map = None
            if params['calculate_density_maps']:
                density_map = self.calculate_density_map(processed_frame)
            
            # Store results
            all_cluster_labels.append(labels)
            all_cluster_properties.append(properties)
            all_fitted_shapes.append(fitted_shapes)
            all_density_maps.append(density_map)
            
            # Update progress
            if frame_idx % 10 == 0:
                progress = int((frame_idx / image_stack.shape[0]) * 100)
                g.m.statusBar().showMessage(f"Processing frame {frame_idx+1}/{image_stack.shape[0]} ({progress}%)")
        
        # Analyze temporal dynamics if requested
        temporal_stats = {}
        if params['analyze_temporal_dynamics']:
            temporal_stats = self.analyze_cluster_temporal_dynamics(all_cluster_properties)
        
        # Store results
        self.detected_clusters = all_cluster_labels
        self.cluster_properties = all_cluster_properties
        self.fitted_shapes = all_fitted_shapes
        self.density_maps = all_density_maps
        self.temporal_stats = temporal_stats
        self.params = params
        
        # Display results
        self.display_results()
        
        g.m.statusBar().showMessage("Cluster analysis complete!", 3000)
    
    def display_results(self):
        """Display comprehensive results"""
        # Clear existing tabs
        for i in range(self.results_tabs.count()):
            self.results_tabs.removeTab(0)
        
        # Tab 1: Summary
        self.create_summary_tab()
        
        # Tab 2: Visualizations
        self.create_visualization_tab()
    
    def create_summary_tab(self):
        """Create summary statistics tab"""
        summary_widget = QWidget()
        summary_layout = QVBoxLayout()
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        # Calculate overall statistics
        total_clusters = sum(len(frame_props) for frame_props in self.cluster_properties)
        n_frames = len(self.cluster_properties)
        
        if total_clusters == 0:
            summary_text.setText("No clusters detected. Try adjusting the detection parameters.")
            summary_layout.addWidget(summary_text)
            summary_widget.setLayout(summary_layout)
            self.results_tabs.addTab(summary_widget, "Summary")
            return
        
        # Collect all cluster properties
        all_areas = []
        all_intensities = []
        all_compactness = []
        all_eccentricity = []
        
        for frame_props in self.cluster_properties:
            all_areas.extend([p['area'] for p in frame_props])
            all_intensities.extend([p['intensity_mean'] for p in frame_props])
            all_compactness.extend([p['compactness'] for p in frame_props])
            all_eccentricity.extend([p['eccentricity'] for p in frame_props])
        
        summary = f"""=== Cluster Analysis Summary ===

Detection Parameters:
  Method: {self.params['detection_method']}
  Intensity threshold: {self.params['intensity_threshold']}
  Min cluster size: {self.params['min_cluster_size']} pixels
  Max cluster size: {self.params['max_cluster_size']} pixels

Overall Statistics:
  Total frames analyzed: {n_frames}
  Total clusters detected: {total_clusters}
  Average clusters per frame: {total_clusters/n_frames:.1f}

Cluster Size Statistics:
  Mean area: {np.mean(all_areas):.1f} pixels
  Median area: {np.median(all_areas):.1f} pixels
  Area range: {np.min(all_areas):.0f} - {np.max(all_areas):.0f} pixels
  Area CV: {np.std(all_areas)/np.mean(all_areas):.3f}

Intensity Statistics:
  Mean intensity: {np.mean(all_intensities):.1f}
  Intensity range: {np.min(all_intensities):.1f} - {np.max(all_intensities):.1f}
  Intensity CV: {np.std(all_intensities)/np.mean(all_intensities):.3f}

Shape Statistics:
  Mean compactness: {np.mean(all_compactness):.3f}
  Mean eccentricity: {np.mean(all_eccentricity):.3f}
  Compactness CV: {np.std(all_compactness)/np.mean(all_compactness):.3f}
"""
        
        # Add temporal statistics if available
        if hasattr(self, 'temporal_stats') and self.temporal_stats:
            summary += f"""
Temporal Dynamics:
  Mean clusters per frame: {self.temporal_stats['mean_cluster_count']:.1f}
  Cluster count variance: {self.temporal_stats['cluster_count_variance']:.1f}
  Cluster count stability: {1 - np.sqrt(self.temporal_stats['cluster_count_variance'])/self.temporal_stats['mean_cluster_count']:.3f}
"""
        
        summary_text.setText(summary)
        summary_layout.addWidget(summary_text)
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")
    
    def create_visualization_tab(self):
        """Create comprehensive visualizations"""
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Cluster Analysis Results')
        
        # Plot 1: First frame with detected clusters
        if g.win is not None and self.detected_clusters:
            first_frame = g.win.image[0]
            first_labels = self.detected_clusters[0]
            
            axes[0,0].imshow(first_frame, cmap='gray', alpha=0.7)
            axes[0,0].imshow(first_labels, cmap='viridis', alpha=0.5)
            axes[0,0].set_title('Detected Clusters (Frame 1)')
            axes[0,0].set_xlabel('X (pixels)')
            axes[0,0].set_ylabel('Y (pixels)')
        
        # Plot 2: Cluster count over time
        if hasattr(self, 'temporal_stats') and 'cluster_counts' in self.temporal_stats:
            cluster_counts = self.temporal_stats['cluster_counts']
            frames = range(len(cluster_counts))
            
            axes[0,1].plot(frames, cluster_counts, 'bo-', linewidth=2, markersize=4)
            axes[0,1].set_xlabel('Frame')
            axes[0,1].set_ylabel('Number of Clusters')
            axes[0,1].set_title('Cluster Count Over Time')
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Cluster size distribution
        all_areas = []
        for frame_props in self.cluster_properties:
            all_areas.extend([p['area'] for p in frame_props])
        
        if all_areas:
            axes[0,2].hist(all_areas, bins=30, alpha=0.7, edgecolor='black')
            axes[0,2].set_xlabel('Cluster Area (pixels)')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].set_title('Cluster Size Distribution')
        
        # Plot 4: Intensity vs Area scatter
        all_intensities = []
        areas_for_intensity = []
        for frame_props in self.cluster_properties:
            all_intensities.extend([p['intensity_mean'] for p in frame_props])
            areas_for_intensity.extend([p['area'] for p in frame_props])
        
        if all_intensities and areas_for_intensity:
            axes[1,0].scatter(areas_for_intensity, all_intensities, alpha=0.6, s=20)
            axes[1,0].set_xlabel('Cluster Area (pixels)')
            axes[1,0].set_ylabel('Mean Intensity')
            axes[1,0].set_title('Intensity vs Size')
        
        # Plot 5: Compactness distribution
        all_compactness = []
        for frame_props in self.cluster_properties:
            all_compactness.extend([p['compactness'] for p in frame_props])
        
        if all_compactness:
            axes[1,1].hist(all_compactness, bins=30, alpha=0.7, edgecolor='black')
            axes[1,1].set_xlabel('Compactness')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Cluster Shape Distribution')
        
        # Plot 6: Density map (average)
        if self.density_maps and any(dm is not None for dm in self.density_maps):
            valid_density_maps = [dm for dm in self.density_maps if dm is not None]
            if valid_density_maps:
                avg_density = np.mean(valid_density_maps, axis=0)
                im = axes[1,2].imshow(avg_density, cmap='hot', alpha=0.8)
                axes[1,2].set_title('Average Protein Density Map')
                plt.colorbar(im, ax=axes[1,2])
        
        plt.tight_layout()
        plt.show()
        
        # Create cluster overlay visualization
        self.create_cluster_overlay()
    
    def create_cluster_overlay(self):
        """Create overlay showing cluster detection results"""
        if not self.detected_clusters or g.win is None:
            return
        
        # Create RGB overlay for first frame
        first_frame = g.win.image[0]
        first_labels = self.detected_clusters[0]
        
        # Normalize original image
        normalized_frame = (first_frame - np.min(first_frame)) / (np.max(first_frame) - np.min(first_frame))
        
        # Create colored overlay
        overlay = np.zeros((*first_frame.shape, 3))
        overlay[:, :, 0] = normalized_frame  # Red channel = original image
        
        # Add cluster labels in green/blue
        unique_labels = np.unique(first_labels)
        for label_id in unique_labels:
            if label_id == 0:  # Background
                continue
            
            mask = first_labels == label_id
            overlay[mask, 1] = 0.7  # Green channel
            overlay[mask, 2] = 0.3  # Blue channel
        
        # Display overlay
        Window(overlay, name=f"{g.win.name}_cluster_overlay")
    
    def export_results(self):
        """Export cluster analysis results"""
        if not self.cluster_properties:
            g.alert("No results to export! Run analysis first.")
            return
        
        base_name = g.win.name
        
        # Export individual cluster data
        all_cluster_data = []
        
        for frame_idx, frame_props in enumerate(self.cluster_properties):
            for cluster in frame_props:
                cluster_data = cluster.copy()
                cluster_data['frame'] = frame_idx
                all_cluster_data.append(cluster_data)
        
        if all_cluster_data:
            clusters_df = pd.DataFrame(all_cluster_data)
            clusters_filename = f"{base_name}_clusters_detailed.csv"
            clusters_df.to_csv(clusters_filename, index=False)
        
        # Export summary statistics per frame
        frame_summary = []
        for frame_idx, frame_props in enumerate(self.cluster_properties):
            if frame_props:
                summary = {
                    'frame': frame_idx,
                    'cluster_count': len(frame_props),
                    'mean_area': np.mean([p['area'] for p in frame_props]),
                    'mean_intensity': np.mean([p['intensity_mean'] for p in frame_props]),
                    'mean_compactness': np.mean([p['compactness'] for p in frame_props]),
                    'total_cluster_area': np.sum([p['area'] for p in frame_props])
                }
            else:
                summary = {
                    'frame': frame_idx,
                    'cluster_count': 0,
                    'mean_area': 0,
                    'mean_intensity': 0,
                    'mean_compactness': 0,
                    'total_cluster_area': 0
                }
            frame_summary.append(summary)
        
        summary_df = pd.DataFrame(frame_summary)
        summary_filename = f"{base_name}_cluster_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        g.alert(f"Results exported to {clusters_filename} and {summary_filename}")
    
    def process(self):
        """Process method required by BaseProcess"""
        self.run_analysis()
        return None

# Register the plugin
ClusterAnalyzer.menu_path = 'Plugins>TIRF Analysis>Cluster Analyzer'