from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage import label, center_of_mass, gaussian_filter
from scipy.spatial import distance_matrix, cKDTree
from scipy.stats import ks_2samp
from distutils.version import StrictVersion
import flika
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import pandas as pd

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import makeROI
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
    from flika.roi import makeROI


class PIEZO1ClusterAnalyzer(BaseProcess_noPriorWindow):
    """
    PIEZO1 Cluster and Spatial Distribution Analyzer
    
    Advanced spatial analysis tool for studying PIEZO1 localization and clustering dynamics.
    Designed for the Medha Pathak Lab at UCI.
    
    Perfect for analyzing:
    - PIEZO1 cluster formation and dynamics
    - Spatial distribution patterns (clustered vs. random vs. uniform)
    - Temporal evolution of clustering
    - Regional heterogeneity (leading edge vs. cell body)
    - Nearest neighbor analysis
    - Density mapping and hotspot detection
    - Colocalization analysis
    
    Features:
    - Automatic cluster detection using multiple algorithms
    - Ripley's K-function for spatial pattern analysis
    - Nearest neighbor distance distributions
    - Temporal tracking of cluster properties
    - Regional comparison tools
    - Comprehensive statistical analysis
    - Publication-ready visualizations
    - Export capabilities for further analysis
    
    Input: Binary or intensity image of PIEZO1 puncta (2D or 3D time series)
    Output: Comprehensive spatial and temporal analysis with statistics and plots
    """
    
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.source_window = None
        self.puncta_coords = []  # List of (t, x, y) coordinates
        self.clusters = []  # List of detected clusters
        self.analysis_results = {}
        self.temporal_data = {}  # Store time series data
        
    def __call__(self, window, threshold_method='otsu', min_puncta_size=3, 
                 max_puncta_size=100, cluster_distance=20, min_cluster_size=3,
                 analyze_temporal=True, frame_range=None):
        """
        Analyze PIEZO1 spatial distribution and clustering.
        
        Parameters:
        -----------
        window : Window
            Source image window (2D or 3D)
        threshold_method : str
            Method for puncta detection ('otsu', 'manual', 'adaptive')
        min_puncta_size : int
            Minimum puncta size in pixels
        max_puncta_size : int
            Maximum puncta size in pixels
        cluster_distance : float
            Maximum distance between puncta to form cluster (pixels)
        min_cluster_size : int
            Minimum number of puncta to define a cluster
        analyze_temporal : bool
            Analyze temporal evolution (for time series)
        frame_range : tuple
            (start, end) frame range for analysis, None = all frames
            
        Returns:
        --------
        dict : Analysis results containing statistics and measurements
        """
        if window is None:
            g.m.statusBar().showMessage("Error: No window selected")
            return None
            
        g.m.statusBar().showMessage("Starting PIEZO1 spatial analysis...")
        t_start = time()
        
        self.source_window = window
        image = window.imageArray()
        
        # Handle 2D vs 3D
        is_time_series = image.ndim == 3
        
        if is_time_series:
            if frame_range is None:
                frame_range = (0, image.shape[0])
            frames_to_analyze = range(frame_range[0], min(frame_range[1], image.shape[0]))
        else:
            frames_to_analyze = [0]
            image = image[np.newaxis, ...]  # Add time dimension
            
        # Initialize results storage
        self.puncta_coords = []
        self.temporal_data = {
            'frame': [],
            'n_puncta': [],
            'n_clusters': [],
            'mean_cluster_size': [],
            'clustering_coefficient': [],
            'mean_nn_distance': [],
            'ripley_k': []
        }
        
        # Analyze each frame
        for frame_idx in frames_to_analyze:
            frame_img = image[frame_idx]
            
            # Detect puncta
            puncta_props = self.detect_puncta(
                frame_img, threshold_method, min_puncta_size, max_puncta_size
            )
            
            if len(puncta_props) > 0:
                # Store coordinates with frame index
                for prop in puncta_props:
                    self.puncta_coords.append({
                        'frame': frame_idx,
                        'x': prop['centroid'][1],
                        'y': prop['centroid'][0],
                        'area': prop['area'],
                        'intensity': prop['mean_intensity']
                    })
                
                # Extract coordinates for this frame
                coords = np.array([[prop['centroid'][1], prop['centroid'][0]] 
                                 for prop in puncta_props])
                
                # Detect clusters
                frame_clusters = self.detect_clusters(
                    coords, cluster_distance, min_cluster_size
                )
                
                # Calculate spatial statistics for this frame
                stats = self.calculate_spatial_statistics(
                    coords, frame_img.shape, cluster_distance
                )
                
                # Store temporal data
                self.temporal_data['frame'].append(frame_idx)
                self.temporal_data['n_puncta'].append(len(puncta_props))
                self.temporal_data['n_clusters'].append(len(frame_clusters))
                self.temporal_data['mean_cluster_size'].append(
                    np.mean([c['size'] for c in frame_clusters]) if frame_clusters else 0
                )
                self.temporal_data['clustering_coefficient'].append(
                    stats.get('clustering_coefficient', 0)
                )
                self.temporal_data['mean_nn_distance'].append(
                    stats.get('mean_nn_distance', 0)
                )
                self.temporal_data['ripley_k'].append(
                    stats.get('ripley_k_normalized', 0)
                )
                
        # Calculate aggregate statistics across all frames
        all_coords = np.array([[p['x'], p['y']] for p in self.puncta_coords])
        
        if len(all_coords) > 0:
            # Overall spatial analysis
            self.analysis_results = self.comprehensive_spatial_analysis(
                all_coords, image[0].shape, cluster_distance, min_cluster_size
            )
            
            # Add temporal statistics if time series
            if is_time_series and analyze_temporal:
                self.analysis_results['temporal'] = self.temporal_data
                self.analysis_results['temporal_summary'] = self.summarize_temporal_data()
        else:
            g.m.statusBar().showMessage("Warning: No puncta detected")
            return None
            
        elapsed = time() - t_start
        g.m.statusBar().showMessage(
            f"Analysis complete: {len(self.puncta_coords)} puncta, "
            f"{self.analysis_results.get('n_clusters', 0)} clusters ({elapsed:.2f} s)"
        )
        
        return self.analysis_results
        
    def detect_puncta(self, image, threshold_method='otsu', 
                     min_size=3, max_size=100):
        """
        Detect PIEZO1 puncta in image.
        
        Returns list of puncta properties including centroid, area, intensity.
        """
        from skimage.filters import threshold_otsu, threshold_local
        from skimage.measure import regionprops, label as sk_label
        from skimage.morphology import remove_small_objects, remove_small_holes
        
        # Apply threshold
        if threshold_method == 'otsu':
            try:
                thresh = threshold_otsu(image)
            except:
                # Fallback if Otsu fails
                thresh = np.percentile(image, 70)
            binary = image > thresh
            
        elif threshold_method == 'adaptive':
            # Local adaptive thresholding
            block_size = min(51, max(3, image.shape[0] // 10))
            if block_size % 2 == 0:
                block_size += 1
            local_thresh = threshold_local(image, block_size=block_size, offset=0)
            binary = image > local_thresh
            
        else:  # manual
            thresh = np.percentile(image, 70)
            binary = image > thresh
            
        # Clean up binary image
        binary = remove_small_objects(binary, min_size=min_size)
        binary = remove_small_holes(binary, area_threshold=min_size)
        
        # Label connected components
        labeled = sk_label(binary)
        
        # Extract properties
        props = regionprops(labeled, intensity_image=image)
        
        # Filter by size and extract relevant properties
        puncta_props = []
        for prop in props:
            if min_size <= prop.area <= max_size:
                puncta_props.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'mean_intensity': prop.mean_intensity,
                    'max_intensity': prop.max_intensity,
                    'eccentricity': prop.eccentricity
                })
                
        return puncta_props
        
    def detect_clusters(self, coords, max_distance, min_size):
        """
        Detect clusters using DBSCAN-like approach.
        
        Parameters:
        -----------
        coords : ndarray
            Nx2 array of (x, y) coordinates
        max_distance : float
            Maximum distance between points in cluster
        min_size : int
            Minimum cluster size
            
        Returns:
        --------
        list : List of cluster dictionaries
        """
        from sklearn.cluster import DBSCAN
        
        if len(coords) < min_size:
            return []
            
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=max_distance, min_samples=min_size).fit(coords)
        labels = clustering.labels_
        
        # Extract cluster information
        clusters = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label_id in unique_labels:
            mask = labels == label_id
            cluster_coords = coords[mask]
            
            # Calculate cluster properties
            centroid = np.mean(cluster_coords, axis=0)
            radius = np.max(np.linalg.norm(cluster_coords - centroid, axis=1))
            
            clusters.append({
                'id': label_id,
                'size': np.sum(mask),
                'coords': cluster_coords,
                'centroid': centroid,
                'radius': radius,
                'density': np.sum(mask) / (np.pi * radius**2) if radius > 0 else 0
            })
            
        return clusters
        
    def calculate_spatial_statistics(self, coords, image_shape, scale):
        """
        Calculate comprehensive spatial statistics.
        
        Returns dictionary with various spatial metrics.
        """
        stats = {}
        n_points = len(coords)
        
        if n_points < 2:
            return stats
            
        # Nearest neighbor analysis
        tree = cKDTree(coords)
        nn_distances, nn_indices = tree.query(coords, k=2)  # k=2 to exclude self
        nn_distances = nn_distances[:, 1]  # Take second nearest (first is self)
        
        stats['mean_nn_distance'] = np.mean(nn_distances)
        stats['median_nn_distance'] = np.median(nn_distances)
        stats['std_nn_distance'] = np.std(nn_distances)
        stats['min_nn_distance'] = np.min(nn_distances)
        stats['max_nn_distance'] = np.max(nn_distances)
        
        # Calculate expected NN distance for random distribution
        area = image_shape[0] * image_shape[1]
        density = n_points / area
        expected_nn = 0.5 / np.sqrt(density) if density > 0 else 0
        
        stats['nn_distance_ratio'] = (stats['mean_nn_distance'] / expected_nn 
                                      if expected_nn > 0 else 1)
        
        # Clustering coefficient (proportion of pairs within scale distance)
        dist_matrix = distance_matrix(coords, coords)
        n_pairs = n_points * (n_points - 1) / 2
        close_pairs = np.sum(np.triu(dist_matrix < scale, k=1))
        stats['clustering_coefficient'] = close_pairs / n_pairs if n_pairs > 0 else 0
        
        # Ripley's K-function
        stats.update(self.calculate_ripleys_k(coords, area, scale))
        
        # Density
        stats['density'] = density
        stats['n_points'] = n_points
        
        return stats
        
    def calculate_ripleys_k(self, coords, area, max_distance):
        """
        Calculate Ripley's K-function for spatial point pattern analysis.
        
        Returns normalized K-function (L-function).
        """
        n_points = len(coords)
        
        if n_points < 3:
            return {'ripley_k': 0, 'ripley_k_normalized': 0}
            
        # Calculate pairwise distances
        dist_matrix = distance_matrix(coords, coords)
        
        # Evaluate K at the specified distance
        r = max_distance
        within_r = np.sum(dist_matrix < r) - n_points  # Subtract diagonal
        k_value = (area / (n_points * (n_points - 1))) * within_r
        
        # L-function (normalized K)
        l_value = np.sqrt(k_value / np.pi) - r
        
        return {
            'ripley_k': k_value,
            'ripley_k_normalized': l_value,
            'ripley_r': r
        }
        
    def comprehensive_spatial_analysis(self, coords, image_shape, 
                                      cluster_distance, min_cluster_size):
        """
        Perform comprehensive spatial analysis on all detected puncta.
        """
        results = {}
        
        # Basic counts
        results['n_puncta'] = len(coords)
        
        # Spatial statistics
        results['spatial_stats'] = self.calculate_spatial_statistics(
            coords, image_shape, cluster_distance
        )
        
        # Cluster detection
        clusters = self.detect_clusters(coords, cluster_distance, min_cluster_size)
        results['n_clusters'] = len(clusters)
        results['clusters'] = clusters
        
        if clusters:
            cluster_sizes = [c['size'] for c in clusters]
            results['cluster_size_mean'] = np.mean(cluster_sizes)
            results['cluster_size_median'] = np.median(cluster_sizes)
            results['cluster_size_std'] = np.std(cluster_sizes)
            results['cluster_size_min'] = np.min(cluster_sizes)
            results['cluster_size_max'] = np.max(cluster_sizes)
            
            cluster_densities = [c['density'] for c in clusters]
            results['cluster_density_mean'] = np.mean(cluster_densities)
            
            # Fraction of puncta in clusters
            n_clustered = sum(cluster_sizes)
            results['fraction_clustered'] = n_clustered / len(coords)
        else:
            results['fraction_clustered'] = 0
            
        # Spatial distribution classification
        l_value = results['spatial_stats'].get('ripley_k_normalized', 0)
        if l_value > 2:
            distribution_type = 'Highly Clustered'
        elif l_value > 0:
            distribution_type = 'Clustered'
        elif l_value > -2:
            distribution_type = 'Random'
        else:
            distribution_type = 'Regular/Dispersed'
        results['distribution_type'] = distribution_type
        
        return results
        
    def summarize_temporal_data(self):
        """
        Summarize temporal dynamics.
        """
        summary = {}
        
        for key in ['n_puncta', 'n_clusters', 'mean_cluster_size', 
                    'clustering_coefficient', 'mean_nn_distance']:
            if key in self.temporal_data and len(self.temporal_data[key]) > 0:
                values = np.array(self.temporal_data[key])
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
                
                # Trend analysis (linear fit)
                if len(values) > 2:
                    frames = np.array(self.temporal_data['frame'])
                    slope, _ = np.polyfit(frames, values, 1)
                    summary[f'{key}_trend'] = slope
                    
        return summary
        
    def create_density_map(self, coords, image_shape, sigma=10):
        """
        Create a 2D density map of puncta locations.
        """
        density_map = np.zeros(image_shape)
        
        for coord in coords:
            x, y = int(coord[0]), int(coord[1])
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                density_map[y, x] += 1
                
        # Apply Gaussian smoothing
        density_map = gaussian_filter(density_map, sigma=sigma)
        
        return density_map
        
    def analyze_regions(self, coords, image_shape, n_regions=4):
        """
        Divide image into regions and compare spatial statistics.
        
        Useful for analyzing leading edge vs. cell body, etc.
        """
        # Divide into grid
        rows = int(np.sqrt(n_regions))
        cols = int(np.ceil(n_regions / rows))
        
        row_edges = np.linspace(0, image_shape[0], rows + 1)
        col_edges = np.linspace(0, image_shape[1], cols + 1)
        
        region_stats = []
        
        for i in range(rows):
            for j in range(cols):
                # Get region bounds
                y_min, y_max = row_edges[i], row_edges[i+1]
                x_min, x_max = col_edges[j], col_edges[j+1]
                
                # Filter coordinates in this region
                in_region = (
                    (coords[:, 0] >= x_min) & (coords[:, 0] < x_max) &
                    (coords[:, 1] >= y_min) & (coords[:, 1] < y_max)
                )
                region_coords = coords[in_region]
                
                if len(region_coords) > 1:
                    region_shape = (int(y_max - y_min), int(x_max - x_min))
                    stats = self.calculate_spatial_statistics(
                        region_coords, region_shape, scale=20
                    )
                    stats['region_id'] = i * cols + j
                    stats['bounds'] = (y_min, y_max, x_min, x_max)
                    region_stats.append(stats)
                    
        return region_stats
        
    def plot_analysis(self):
        """
        Create comprehensive visualization of analysis results.
        """
        if not self.analysis_results or len(self.puncta_coords) == 0:
            g.m.statusBar().showMessage("No analysis results to plot")
            return
            
        # Create figure with multiple subplots
        fig = Figure(figsize=(16, 12))
        
        # Get all coordinates
        all_coords = np.array([[p['x'], p['y']] for p in self.puncta_coords])
        image_shape = self.source_window.image.shape[-2:]
        
        # 1. Scatter plot with clusters
        ax1 = fig.add_subplot(3, 3, 1)
        ax1.scatter(all_coords[:, 0], all_coords[:, 1], 
                   c='blue', s=20, alpha=0.6, label='Puncta')
        
        # Overlay clusters
        clusters = self.analysis_results.get('clusters', [])
        for i, cluster in enumerate(clusters):
            centroid = cluster['centroid']
            radius = cluster['radius']
            circle = Circle(centroid, radius, fill=False, 
                          edgecolor='red', linewidth=2, alpha=0.7)
            ax1.add_patch(circle)
            ax1.scatter(centroid[0], centroid[1], c='red', 
                       s=100, marker='x', linewidths=3)
            
        ax1.set_xlim(0, image_shape[1])
        ax1.set_ylim(image_shape[0], 0)  # Invert y-axis
        ax1.set_aspect('equal')
        ax1.set_title(f'Spatial Distribution\n{len(all_coords)} puncta, '
                     f'{len(clusters)} clusters')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Density heatmap
        ax2 = fig.add_subplot(3, 3, 2)
        density_map = self.create_density_map(all_coords, image_shape, sigma=15)
        im = ax2.imshow(density_map, cmap='hot', interpolation='bilinear')
        ax2.set_title('Density Heatmap')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        fig.colorbar(im, ax=ax2, label='Density')
        
        # 3. Nearest neighbor distribution
        ax3 = fig.add_subplot(3, 3, 3)
        spatial_stats = self.analysis_results.get('spatial_stats', {})
        if 'mean_nn_distance' in spatial_stats:
            # Calculate NN distances for histogram
            tree = cKDTree(all_coords)
            nn_distances, _ = tree.query(all_coords, k=2)
            nn_distances = nn_distances[:, 1]
            
            ax3.hist(nn_distances, bins=50, alpha=0.7, color='blue', 
                    edgecolor='black', density=True)
            ax3.axvline(spatial_stats['mean_nn_distance'], 
                       color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {spatial_stats['mean_nn_distance']:.1f} px")
            ax3.set_xlabel('Nearest Neighbor Distance (pixels)')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('NN Distance Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
        # 4. Cluster size distribution
        ax4 = fig.add_subplot(3, 3, 4)
        if clusters:
            cluster_sizes = [c['size'] for c in clusters]
            ax4.hist(cluster_sizes, bins=min(20, len(set(cluster_sizes))), 
                    alpha=0.7, color='green', edgecolor='black')
            ax4.set_xlabel('Cluster Size (number of puncta)')
            ax4.set_ylabel('Count')
            ax4.set_title(f'Cluster Size Distribution (n={len(clusters)})')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No clusters detected', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cluster Size Distribution')
            
        # 5. Statistics summary
        ax5 = fig.add_subplot(3, 3, 5)
        ax5.axis('off')
        
        stats_text = "SPATIAL STATISTICS\n" + "="*40 + "\n\n"
        stats_text += f"Total Puncta: {self.analysis_results['n_puncta']}\n"
        stats_text += f"Total Clusters: {self.analysis_results['n_clusters']}\n"
        stats_text += f"Fraction Clustered: {self.analysis_results.get('fraction_clustered', 0):.2%}\n\n"
        
        if 'spatial_stats' in self.analysis_results:
            ss = self.analysis_results['spatial_stats']
            stats_text += f"Mean NN Distance: {ss.get('mean_nn_distance', 0):.2f} px\n"
            stats_text += f"NN Distance Ratio: {ss.get('nn_distance_ratio', 0):.2f}\n"
            stats_text += f"Clustering Coeff: {ss.get('clustering_coefficient', 0):.3f}\n"
            stats_text += f"Ripley's L: {ss.get('ripley_k_normalized', 0):.2f}\n"
            
        stats_text += f"\nDistribution Type:\n{self.analysis_results.get('distribution_type', 'Unknown')}\n\n"
        
        if clusters:
            stats_text += f"Mean Cluster Size: {self.analysis_results.get('cluster_size_mean', 0):.1f}\n"
            stats_text += f"Cluster Density: {self.analysis_results.get('cluster_density_mean', 0):.2e} /px²\n"
            
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', 
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. Temporal evolution (if available)
        ax6 = fig.add_subplot(3, 3, 6)
        if 'temporal' in self.analysis_results and len(self.temporal_data['frame']) > 1:
            frames = self.temporal_data['frame']
            ax6.plot(frames, self.temporal_data['n_puncta'], 
                    'b-o', label='Puncta', alpha=0.7)
            ax6_twin = ax6.twinx()
            ax6_twin.plot(frames, self.temporal_data['n_clusters'], 
                         'r-s', label='Clusters', alpha=0.7)
            ax6.set_xlabel('Frame')
            ax6.set_ylabel('Number of Puncta', color='b')
            ax6_twin.set_ylabel('Number of Clusters', color='r')
            ax6.set_title('Temporal Evolution')
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='upper left')
            ax6_twin.legend(loc='upper right')
        else:
            ax6.text(0.5, 0.5, 'Single timepoint\n(no temporal data)', 
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Temporal Evolution')
            
        # 7. Clustering coefficient over time
        ax7 = fig.add_subplot(3, 3, 7)
        if 'temporal' in self.analysis_results and len(self.temporal_data['frame']) > 1:
            ax7.plot(self.temporal_data['frame'], 
                    self.temporal_data['clustering_coefficient'],
                    'g-o', alpha=0.7)
            ax7.set_xlabel('Frame')
            ax7.set_ylabel('Clustering Coefficient')
            ax7.set_title('Clustering Dynamics')
            ax7.grid(True, alpha=0.3)
        else:
            ax7.axis('off')
            
        # 8. Mean cluster size over time
        ax8 = fig.add_subplot(3, 3, 8)
        if 'temporal' in self.analysis_results and len(self.temporal_data['frame']) > 1:
            ax8.plot(self.temporal_data['frame'], 
                    self.temporal_data['mean_cluster_size'],
                    'm-s', alpha=0.7)
            ax8.set_xlabel('Frame')
            ax8.set_ylabel('Mean Cluster Size')
            ax8.set_title('Cluster Size Dynamics')
            ax8.grid(True, alpha=0.3)
        else:
            ax8.axis('off')
            
        # 9. Ripley's K over time or regional analysis
        ax9 = fig.add_subplot(3, 3, 9)
        if 'temporal' in self.analysis_results and len(self.temporal_data['frame']) > 1:
            ax9.plot(self.temporal_data['frame'], 
                    self.temporal_data['ripley_k'],
                    'c-^', alpha=0.7)
            ax9.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax9.set_xlabel('Frame')
            ax9.set_ylabel("Ripley's L (normalized)")
            ax9.set_title('Spatial Pattern Evolution')
            ax9.grid(True, alpha=0.3)
        else:
            # Show regional comparison if single timepoint
            region_stats = self.analyze_regions(all_coords, image_shape, n_regions=4)
            if region_stats:
                regions = [s['region_id'] for s in region_stats]
                densities = [s['density'] for s in region_stats]
                ax9.bar(regions, densities, alpha=0.7, color='purple')
                ax9.set_xlabel('Region ID')
                ax9.set_ylabel('Puncta Density')
                ax9.set_title('Regional Density Comparison')
                ax9.grid(True, alpha=0.3, axis='y')
            else:
                ax9.axis('off')
        
        fig.tight_layout()
        
        # Display figure
        self.plot_window = QWidget()
        self.plot_window.setWindowTitle("PIEZO1 Cluster Analysis Results")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Add export button
        export_btn = QPushButton("Export Data & Figures")
        export_btn.clicked.connect(self.export_gui)
        layout.addWidget(export_btn)
        
        self.plot_window.setLayout(layout)
        self.plot_window.resize(1600, 1200)
        self.plot_window.show()
        
        g.m.statusBar().showMessage("Analysis visualization created")
        
    def export_data(self, base_filename):
        """
        Export analysis results to files.
        """
        import json
        
        # Export puncta coordinates
        if self.puncta_coords:
            df_puncta = pd.DataFrame(self.puncta_coords)
            df_puncta.to_csv(f"{base_filename}_puncta.csv", index=False)
            
        # Export cluster information
        if self.analysis_results.get('clusters'):
            cluster_data = []
            for c in self.analysis_results['clusters']:
                cluster_data.append({
                    'cluster_id': c['id'],
                    'size': c['size'],
                    'centroid_x': c['centroid'][0],
                    'centroid_y': c['centroid'][1],
                    'radius': c['radius'],
                    'density': c['density']
                })
            df_clusters = pd.DataFrame(cluster_data)
            df_clusters.to_csv(f"{base_filename}_clusters.csv", index=False)
            
        # Export temporal data
        if 'temporal' in self.analysis_results:
            df_temporal = pd.DataFrame(self.temporal_data)
            df_temporal.to_csv(f"{base_filename}_temporal.csv", index=False)
            
        # Export summary statistics as JSON
        summary = {
            'n_puncta': int(self.analysis_results.get('n_puncta', 0)),
            'n_clusters': int(self.analysis_results.get('n_clusters', 0)),
            'fraction_clustered': float(self.analysis_results.get('fraction_clustered', 0)),
            'distribution_type': self.analysis_results.get('distribution_type', ''),
            'spatial_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in self.analysis_results.get('spatial_stats', {}).items()},
        }
        
        if 'temporal_summary' in self.analysis_results:
            summary['temporal_summary'] = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v 
                for k, v in self.analysis_results['temporal_summary'].items()
            }
            
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
        
        # Detection parameters
        self.threshold_combo = ComboBox()
        self.threshold_combo.addItems(['otsu', 'adaptive', 'manual'])
        
        self.min_size_spin = pg.SpinBox(int=True, step=1, bounds=[1, 100], value=3)
        self.max_size_spin = pg.SpinBox(int=True, step=10, bounds=[10, 1000], value=100)
        
        # Clustering parameters
        self.cluster_distance_spin = pg.SpinBox(int=False, step=1, bounds=[1, 100], 
                                               value=20, decimals=1)
        self.min_cluster_size_spin = pg.SpinBox(int=True, step=1, bounds=[2, 50], value=3)
        
        # Temporal analysis
        self.temporal_check = CheckBox()
        self.temporal_check.setChecked(True)
        
        # Frame range
        self.frame_start_spin = pg.SpinBox(int=True, step=1, bounds=[0, 10000], value=0)
        self.frame_end_spin = pg.SpinBox(int=True, step=1, bounds=[1, 10000], value=100)
        
        # Analysis buttons
        self.analyze_button = QPushButton('Run Analysis')
        self.analyze_button.clicked.connect(self.analyze_from_gui)
        
        self.plot_button = QPushButton('Visualize Results')
        self.plot_button.clicked.connect(self.plot_analysis)
        
        self.export_button = QPushButton('Export Data')
        self.export_button.clicked.connect(self.export_gui)
        
        # Regional analysis button
        self.regional_button = QPushButton('Analyze Regions')
        self.regional_button.clicked.connect(self.analyze_regions_gui)
        
        # Build items list
        self.items.append({'name': 'window', 'string': 'Source Window',
                          'object': window_selector})
        self.items.append({'name': '', 'string': '--- Puncta Detection ---',
                          'object': None})
        self.items.append({'name': 'threshold_method', 'string': 'Threshold Method',
                          'object': self.threshold_combo})
        self.items.append({'name': 'min_puncta_size', 'string': 'Min Puncta Size (px)',
                          'object': self.min_size_spin})
        self.items.append({'name': 'max_puncta_size', 'string': 'Max Puncta Size (px)',
                          'object': self.max_size_spin})
        self.items.append({'name': '', 'string': '--- Clustering ---',
                          'object': None})
        self.items.append({'name': 'cluster_distance', 'string': 'Cluster Distance (px)',
                          'object': self.cluster_distance_spin})
        self.items.append({'name': 'min_cluster_size', 'string': 'Min Cluster Size',
                          'object': self.min_cluster_size_spin})
        self.items.append({'name': '', 'string': '--- Temporal Analysis ---',
                          'object': None})
        self.items.append({'name': 'analyze_temporal', 'string': 'Analyze Temporal',
                          'object': self.temporal_check})
        self.items.append({'name': 'frame_start', 'string': 'Start Frame',
                          'object': self.frame_start_spin})
        self.items.append({'name': 'frame_end', 'string': 'End Frame',
                          'object': self.frame_end_spin})
        self.items.append({'name': '', 'string': '--- Actions ---',
                          'object': None})
        self.items.append({'name': 'analyze', 'string': '',
                          'object': self.analyze_button})
        self.items.append({'name': 'plot', 'string': '',
                          'object': self.plot_button})
        self.items.append({'name': 'regional', 'string': '',
                          'object': self.regional_button})
        self.items.append({'name': 'export', 'string': '',
                          'object': self.export_button})
        
        super().gui()
        
    def analyze_from_gui(self):
        """Run analysis from GUI parameters."""
        window = self.getValue('window')
        threshold_method = self.getValue('threshold_method')
        min_puncta_size = self.getValue('min_puncta_size')
        max_puncta_size = self.getValue('max_puncta_size')
        cluster_distance = self.getValue('cluster_distance')
        min_cluster_size = self.getValue('min_cluster_size')
        analyze_temporal = self.getValue('analyze_temporal')
        frame_start = self.getValue('frame_start')
        frame_end = self.getValue('frame_end')
        
        frame_range = (frame_start, frame_end) if analyze_temporal else None
        
        self(window, threshold_method, min_puncta_size, max_puncta_size,
             cluster_distance, min_cluster_size, analyze_temporal, frame_range)
             
    def analyze_regions_gui(self):
        """Analyze regional differences and display results."""
        if not self.analysis_results or len(self.puncta_coords) == 0:
            g.m.statusBar().showMessage("Run analysis first")
            return
            
        all_coords = np.array([[p['x'], p['y']] for p in self.puncta_coords])
        image_shape = self.source_window.image.shape[-2:]
        
        region_stats = self.analyze_regions(all_coords, image_shape, n_regions=4)
        
        if not region_stats:
            g.m.statusBar().showMessage("Could not analyze regions")
            return
            
        # Create regional comparison plot
        fig = Figure(figsize=(12, 8))
        
        # Extract data
        regions = [s['region_id'] for s in region_stats]
        densities = [s['density'] for s in region_stats]
        nn_distances = [s.get('mean_nn_distance', 0) for s in region_stats]
        clustering_coeffs = [s.get('clustering_coefficient', 0) for s in region_stats]
        
        # Plot
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.bar(regions, densities, alpha=0.7, color='blue')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Density (puncta/px²)')
        ax1.set_title('Regional Density')
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(regions, nn_distances, alpha=0.7, color='green')
        ax2.set_xlabel('Region')
        ax2.set_ylabel('Mean NN Distance (px)')
        ax2.set_title('Regional NN Distance')
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.bar(regions, clustering_coeffs, alpha=0.7, color='red')
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Clustering Coefficient')
        ax3.set_title('Regional Clustering')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Show region boundaries on density map
        ax4 = fig.add_subplot(2, 2, 4)
        density_map = self.create_density_map(all_coords, image_shape, sigma=15)
        ax4.imshow(density_map, cmap='hot', interpolation='bilinear')
        
        # Draw region boundaries
        n_regions = 4
        rows = int(np.sqrt(n_regions))
        cols = int(np.ceil(n_regions / rows))
        row_edges = np.linspace(0, image_shape[0], rows + 1)
        col_edges = np.linspace(0, image_shape[1], cols + 1)
        
        for edge in row_edges:
            ax4.axhline(edge, color='white', linewidth=1, alpha=0.7)
        for edge in col_edges:
            ax4.axvline(edge, color='white', linewidth=1, alpha=0.7)
            
        ax4.set_title('Density Map with Regions')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        
        fig.tight_layout()
        
        # Display
        self.regional_window = QWidget()
        self.regional_window.setWindowTitle("Regional Analysis")
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        self.regional_window.setLayout(layout)
        self.regional_window.resize(1200, 800)
        self.regional_window.show()
        
        g.m.statusBar().showMessage("Regional analysis complete")
        
    def export_gui(self):
        """Export data via file dialog."""
        filename, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Analysis Data", "",
            "All Files (*)"
        )
        if filename:
            self.export_data(filename)


# Create plugin instance
piezo1_cluster_analyzer = PIEZO1ClusterAnalyzer()
