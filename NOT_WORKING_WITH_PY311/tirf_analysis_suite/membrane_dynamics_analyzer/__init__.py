# membrane_dynamics_analyzer/__init__.py
"""
Membrane Dynamics Analyzer Plugin for FLIKA
Analyzes cell edge movement and membrane dynamics in TIRF microscopy
"""

import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.interpolate import interp1d, UnivariateSpline
from skimage import filters, morphology, measure, segmentation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from flika.roi import makeROI, ROI_freehand
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTabWidget, QWidget

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class MembraneDynamicsAnalyzer(BaseProcess):
    """
    Analyze membrane dynamics and cell edge movement in TIRF microscopy
    """
    
    def __init__(self):
        super().__init__()
        self.cell_edges = []
        self.edge_velocities = []
        self.protrusion_retractions = []
        self.results_tabs = None
        
    def get_init_settings_dict(self):
        return {
            'edge_detection_method': ['canny', 'gradient', 'threshold'],
            'gaussian_sigma': 2.0,
            'canny_low_threshold': 0.1,
            'canny_high_threshold': 0.2,
            'threshold_value': 0.3,
            'edge_smoothing': 5,
            'temporal_smoothing': 3,
            'protrusion_threshold': 0.5,
            'min_protrusion_duration': 3,
            'spatial_sampling': 2,
            'velocity_window': 5
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['edge_detection_method'] = self.edge_detection_method.currentText()
        params['gaussian_sigma'] = self.gaussian_sigma.value()
        params['canny_low_threshold'] = self.canny_low_threshold.value()
        params['canny_high_threshold'] = self.canny_high_threshold.value()
        params['threshold_value'] = self.threshold_value.value()
        params['edge_smoothing'] = int(self.edge_smoothing.value())
        params['temporal_smoothing'] = int(self.temporal_smoothing.value())
        params['protrusion_threshold'] = self.protrusion_threshold.value()
        params['min_protrusion_duration'] = int(self.min_protrusion_duration.value())
        params['spatial_sampling'] = int(self.spatial_sampling.value())
        params['velocity_window'] = int(self.velocity_window.value())
        return params
    
    def get_name(self):
        return 'Membrane Dynamics Analyzer'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Membrane Dynamics Analyzer'
    
    def setupGUI(self):
        super().setupGUI()
        self.gaussian_sigma.setRange(0.5, 10.0)
        self.canny_low_threshold.setRange(0.01, 1.0)
        self.canny_high_threshold.setRange(0.01, 1.0)
        self.threshold_value.setRange(0.1, 1.0)
        self.edge_smoothing.setRange(1, 20)
        self.temporal_smoothing.setRange(1, 10)
        self.protrusion_threshold.setRange(0.1, 5.0)
        self.min_protrusion_duration.setRange(2, 20)
        self.spatial_sampling.setRange(1, 10)
        self.velocity_window.setRange(2, 20)
        
        # Add analysis button
        self.analyze_button = QPushButton("Analyze Membrane Dynamics")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.layout().addWidget(self.analyze_button)
        
        # Add results tabs
        self.results_tabs = QTabWidget()
        self.layout().addWidget(self.results_tabs)
        
        # Add export button
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.layout().addWidget(self.export_button)
    
    def detect_cell_edge(self, image, method, params):
        """Detect cell edge using specified method"""
        # Preprocess image
        smoothed = ndimage.gaussian_filter(image.astype(float), sigma=params['gaussian_sigma'])
        
        if method == 'canny':
            # Canny edge detection
            edges = filters.canny(
                smoothed,
                sigma=params['gaussian_sigma'],
                low_threshold=params['canny_low_threshold'],
                high_threshold=params['canny_high_threshold']
            )
            
        elif method == 'gradient':
            # Gradient-based edge detection
            grad_x = ndimage.sobel(smoothed, axis=1)
            grad_y = ndimage.sobel(smoothed, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Threshold gradient
            threshold = params['threshold_value'] * np.max(gradient_magnitude)
            edges = gradient_magnitude > threshold
            
        elif method == 'threshold':
            # Simple thresholding
            threshold = params['threshold_value'] * np.max(smoothed)
            binary = smoothed > threshold
            
            # Find edges of binary mask
            edges = binary ^ ndimage.binary_erosion(binary)
        
        # Clean up edges
        edges = morphology.binary_closing(edges, morphology.disk(2))
        edges = morphology.skeletonize(edges)
        
        return edges
    
    def extract_edge_contour(self, edge_image, smoothing=5):
        """Extract smooth contour from edge image"""
        # Find contours
        contours = measure.find_contours(edge_image, 0.5)
        
        if not contours:
            return None
        
        # Take the longest contour (main cell edge)
        longest_contour = max(contours, key=len)
        
        # Smooth the contour
        if len(longest_contour) > smoothing * 2:
            # Parametric smoothing
            t = np.arange(len(longest_contour))
            
            # Smooth x and y coordinates separately
            try:
                spline_y = UnivariateSpline(t, longest_contour[:, 0], s=smoothing)
                spline_x = UnivariateSpline(t, longest_contour[:, 1], s=smoothing)
                
                smoothed_contour = np.column_stack([spline_y(t), spline_x(t)])
            except:
                # Fallback to moving average
                window = min(smoothing, len(longest_contour) // 4)
                if window > 1:
                    smoothed_y = ndimage.uniform_filter1d(longest_contour[:, 0], size=window, mode='nearest')
                    smoothed_x = ndimage.uniform_filter1d(longest_contour[:, 1], size=window, mode='nearest')
                    smoothed_contour = np.column_stack([smoothed_y, smoothed_x])
                else:
                    smoothed_contour = longest_contour
        else:
            smoothed_contour = longest_contour
        
        return smoothed_contour
    
    def sample_contour_points(self, contour, n_points):
        """Sample points along contour for consistent analysis"""
        if contour is None or len(contour) < 2:
            return None
        
        # Calculate cumulative arc length
        diff = np.diff(contour, axis=0)
        arc_lengths = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        total_length = arc_lengths[-1]
        
        # Add zero at beginning
        arc_lengths = np.concatenate([[0], arc_lengths])
        
        # Sample points at regular intervals
        sample_lengths = np.linspace(0, total_length, n_points)
        
        # Interpolate to get sampled points
        interp_y = interp1d(arc_lengths, contour[:, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
        interp_x = interp1d(arc_lengths, contour[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
        
        sampled_points = np.column_stack([interp_y(sample_lengths), interp_x(sample_lengths)])
        
        return sampled_points, sample_lengths
    
    def calculate_edge_velocity(self, edge_sequence, dt=1):
        """Calculate edge velocity from sequence of edge positions"""
        if len(edge_sequence) < 2:
            return None
        
        velocities = []
        
        for i in range(len(edge_sequence) - 1):
            current_edge = edge_sequence[i]
            next_edge = edge_sequence[i + 1]
            
            if current_edge is None or next_edge is None:
                velocities.append(None)
                continue
            
            # Calculate displacement for each point
            displacements = next_edge - current_edge
            speeds = np.sqrt(np.sum(displacements**2, axis=1)) / dt
            
            velocities.append(speeds)
        
        return velocities
    
    def detect_protrusions_retractions(self, edge_velocities, normal_vectors, threshold, min_duration):
        """Detect protrusion and retraction events"""
        if not edge_velocities or not normal_vectors:
            return []
        
        events = []
        
        # For each point along the edge
        for point_idx in range(len(edge_velocities[0]) if edge_velocities[0] is not None else 0):
            point_velocities = []
            
            # Extract velocity time series for this point
            for frame_velocities in edge_velocities:
                if frame_velocities is not None and point_idx < len(frame_velocities):
                    # Project velocity onto normal direction
                    normal = normal_vectors[point_idx] if point_idx < len(normal_vectors) else [0, 1]
                    velocity_normal = frame_velocities[point_idx] * np.sign(np.dot([0, 1], normal))
                    point_velocities.append(velocity_normal)
                else:
                    point_velocities.append(0)
            
            # Detect events
            point_velocities = np.array(point_velocities)
            
            # Protrusions (positive velocity above threshold)
            protrusion_mask = point_velocities > threshold
            # Retractions (negative velocity below -threshold)
            retraction_mask = point_velocities < -threshold
            
            # Find continuous events
            for mask, event_type in [(protrusion_mask, 'protrusion'), (retraction_mask, 'retraction')]:
                # Find start and end of events
                event_starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
                event_ends = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
                
                # Handle edge cases
                if mask[0]:
                    event_starts = np.concatenate([[0], event_starts])
                if mask[-1]:
                    event_ends = np.concatenate([event_ends, [len(mask)]])
                
                # Match starts and ends
                for start, end in zip(event_starts, event_ends):
                    duration = end - start
                    if duration >= min_duration:
                        max_velocity = np.max(np.abs(point_velocities[start:end]))
                        
                        events.append({
                            'point_index': point_idx,
                            'type': event_type,
                            'start_frame': start,
                            'end_frame': end,
                            'duration': duration,
                            'max_velocity': max_velocity,
                            'mean_velocity': np.mean(np.abs(point_velocities[start:end]))
                        })
        
        return events
    
    def calculate_edge_normals(self, contour):
        """Calculate normal vectors for edge contour"""
        if contour is None or len(contour) < 3:
            return None
        
        # Calculate tangent vectors
        tangents = np.gradient(contour, axis=0)
        
        # Calculate normal vectors (perpendicular to tangents)
        normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
        
        # Normalize
        norms = np.sqrt(np.sum(normals**2, axis=1))
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms[:, np.newaxis]
        
        return normals
    
    def run_analysis(self):
        """Run comprehensive membrane dynamics analysis"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Analyzing membrane dynamics...")
        
        # Initialize storage
        all_edges = []
        all_sampled_edges = []
        
        # Calculate number of sample points
        n_sample_points = image_stack.shape[1] // params['spatial_sampling']
        
        # Process each frame
        for frame_idx in range(image_stack.shape[0]):
            frame = image_stack[frame_idx]
            
            # Detect cell edge
            edge_image = self.detect_cell_edge(frame, params['edge_detection_method'], params)
            
            # Extract contour
            contour = self.extract_edge_contour(edge_image, params['edge_smoothing'])
            
            if contour is not None:
                # Sample points along contour
                sampled_points, arc_lengths = self.sample_contour_points(contour, n_sample_points)
                all_sampled_edges.append(sampled_points)
            else:
                all_sampled_edges.append(None)
            
            all_edges.append(contour)
            
            # Update progress
            if frame_idx % 10 == 0:
                progress = int((frame_idx / image_stack.shape[0]) * 100)
                g.m.statusBar().showMessage(f"Processing frame {frame_idx+1}/{image_stack.shape[0]} ({progress}%)")
        
        # Apply temporal smoothing to edge positions
        if params['temporal_smoothing'] > 1:
            smoothed_edges = self.apply_temporal_smoothing(all_sampled_edges, params['temporal_smoothing'])
        else:
            smoothed_edges = all_sampled_edges
        
        # Calculate edge velocities
        edge_velocities = self.calculate_edge_velocity(smoothed_edges)
        
        # Calculate normal vectors (using first valid edge)
        first_valid_edge = next((edge for edge in all_sampled_edges if edge is not None), None)
        normal_vectors = self.calculate_edge_normals(first_valid_edge) if first_valid_edge is not None else None
        
        # Detect protrusion/retraction events
        events = self.detect_protrusions_retractions(
            edge_velocities, normal_vectors, 
            params['protrusion_threshold'], params['min_protrusion_duration']
        )
        
        # Store results
        self.all_edges = all_edges
        self.all_sampled_edges = all_sampled_edges
        self.smoothed_edges = smoothed_edges
        self.edge_velocities = edge_velocities
        self.normal_vectors = normal_vectors
        self.events = events
        self.params = params
        
        # Display results
        self.display_results()
        
        g.m.statusBar().showMessage("Membrane dynamics analysis complete!", 3000)
    
    def apply_temporal_smoothing(self, edge_sequence, window_size):
        """Apply temporal smoothing to edge sequence"""
        smoothed_sequence = []
        
        for i in range(len(edge_sequence)):
            if edge_sequence[i] is None:
                smoothed_sequence.append(None)
                continue
            
            # Collect valid edges in window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(edge_sequence), i + window_size // 2 + 1)
            
            valid_edges = [edge for edge in edge_sequence[start_idx:end_idx] if edge is not None]
            
            if valid_edges:
                # Average the edges
                mean_edge = np.mean(valid_edges, axis=0)
                smoothed_sequence.append(mean_edge)
            else:
                smoothed_sequence.append(edge_sequence[i])
        
        return smoothed_sequence
    
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
        
        # Calculate statistics
        n_frames = len(self.all_edges)
        n_valid_edges = sum(1 for edge in self.all_edges if edge is not None)
        
        # Event statistics
        protrusions = [e for e in self.events if e['type'] == 'protrusion']
        retractions = [e for e in self.events if e['type'] == 'retraction']
        
        # Velocity statistics
        all_velocities = []
        for frame_velocities in self.edge_velocities:
            if frame_velocities is not None:
                all_velocities.extend(frame_velocities)
        
        summary = f"""=== Membrane Dynamics Analysis Summary ===

Frames Analyzed: {n_frames}
Valid Edges Detected: {n_valid_edges} ({n_valid_edges/n_frames*100:.1f}%)

Edge Velocity Statistics:
  Mean velocity: {np.mean(all_velocities):.3f} pixels/frame
  Std velocity: {np.std(all_velocities):.3f} pixels/frame
  Max velocity: {np.max(all_velocities):.3f} pixels/frame

Membrane Events:
  Total protrusions: {len(protrusions)}
  Total retractions: {len(retractions)}
  
  Average protrusion duration: {np.mean([e['duration'] for e in protrusions]):.1f} frames
  Average retraction duration: {np.mean([e['duration'] for e in retractions]):.1f} frames
  
  Max protrusion velocity: {np.max([e['max_velocity'] for e in protrusions]) if protrusions else 0:.3f}
  Max retraction velocity: {np.max([e['max_velocity'] for e in retractions]) if retractions else 0:.3f}

Analysis Parameters:
  Edge detection method: {self.params['edge_detection_method']}
  Spatial sampling: {self.params['spatial_sampling']} pixels
  Temporal smoothing: {self.params['temporal_smoothing']} frames
  Protrusion threshold: {self.params['protrusion_threshold']} pixels/frame
"""
        
        summary_text.setText(summary)
        summary_layout.addWidget(summary_text)
        summary_widget.setLayout(summary_layout)
        self.results_tabs.addTab(summary_widget, "Summary")
    
    def create_visualization_tab(self):
        """Create visualizations"""
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Membrane Dynamics Analysis')
        
        # Plot 1: Edge overlay on first frame
        if g.win is not None and self.all_edges[0] is not None:
            first_frame = g.win.image[0]
            axes[0,0].imshow(first_frame, cmap='gray', alpha=0.7)
            
            edge = self.all_edges[0]
            axes[0,0].plot(edge[:, 1], edge[:, 0], 'r-', linewidth=2, label='Detected Edge')
            axes[0,0].set_title('Edge Detection (Frame 1)')
            axes[0,0].legend()
        
        # Plot 2: Velocity over time
        if self.edge_velocities:
            frame_mean_velocities = []
            for frame_vel in self.edge_velocities:
                if frame_vel is not None:
                    frame_mean_velocities.append(np.mean(frame_vel))
                else:
                    frame_mean_velocities.append(0)
            
            frames = range(len(frame_mean_velocities))
            axes[0,1].plot(frames, frame_mean_velocities, 'b-', linewidth=2)
            axes[0,1].set_xlabel('Frame')
            axes[0,1].set_ylabel('Mean Edge Velocity (pixels/frame)')
            axes[0,1].set_title('Edge Velocity Over Time')
        
        # Plot 3: Event timeline
        if self.events:
            y_positions = []
            colors = []
            durations = []
            
            for event in self.events:
                y_positions.append(event['point_index'])
                colors.append('red' if event['type'] == 'protrusion' else 'blue')
                durations.append(event['duration'])
                
                # Draw event as horizontal line
                axes[1,0].barh(event['point_index'], event['duration'], 
                              left=event['start_frame'], height=1,
                              color='red' if event['type'] == 'protrusion' else 'blue',
                              alpha=0.7)
            
            axes[1,0].set_xlabel('Frame')
            axes[1,0].set_ylabel('Edge Position')
            axes[1,0].set_title('Protrusion/Retraction Events')
            
            # Add legend
            import matplotlib.patches as mpatches
            red_patch = mpatches.Patch(color='red', label='Protrusions')
            blue_patch = mpatches.Patch(color='blue', label='Retractions')
            axes[1,0].legend(handles=[red_patch, blue_patch])
        
        # Plot 4: Event duration histogram
        if self.events:
            protrusion_durations = [e['duration'] for e in self.events if e['type'] == 'protrusion']
            retraction_durations = [e['duration'] for e in self.events if e['type'] == 'retraction']
            
            if protrusion_durations:
                axes[1,1].hist(protrusion_durations, alpha=0.7, color='red', label='Protrusions', bins=10)
            if retraction_durations:
                axes[1,1].hist(retraction_durations, alpha=0.7, color='blue', label='Retractions', bins=10)
            
            axes[1,1].set_xlabel('Duration (frames)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Event Duration Distribution')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self):
        """Export results to CSV files"""
        if not hasattr(self, 'events'):
            g.alert("No results to export! Run analysis first.")
            return
        
        base_name = g.win.name
        
        # Export events
        if self.events:
            events_df = pd.DataFrame(self.events)
            events_filename = f"{base_name}_membrane_events.csv"
            events_df.to_csv(events_filename, index=False)
        
        # Export velocity data
        velocity_data = []
        for frame_idx, frame_velocities in enumerate(self.edge_velocities):
            if frame_velocities is not None:
                for point_idx, velocity in enumerate(frame_velocities):
                    velocity_data.append({
                        'frame': frame_idx,
                        'point_index': point_idx,
                        'velocity': velocity
                    })
        
        if velocity_data:
            velocity_df = pd.DataFrame(velocity_data)
            velocity_filename = f"{base_name}_edge_velocities.csv"
            velocity_df.to_csv(velocity_filename, index=False)
        
        g.alert(f"Results exported to CSV files")
    
    def process(self):
        """Process method required by BaseProcess"""
        self.run_analysis()
        return None

# Register the plugin
MembraneDynamicsAnalyzer.menu_path = 'Plugins>TIRF Analysis>Membrane Dynamics Analyzer'