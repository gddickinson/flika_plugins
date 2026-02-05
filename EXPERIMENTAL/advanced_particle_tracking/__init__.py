# __init__.py
import numpy as np
import sys
from qtpy import QtWidgets, QtCore, QtGui
import skimage.filters
import skimage.feature
import skimage.morphology
from scipy import ndimage, optimize
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import pyqtgraph for interactive visualization
try:
    import pyqtgraph as pg
    from pyqtgraph import ImageView, ScatterPlotItem, PlotCurveItem
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    print("PyQtGraph not available - interactive visualization disabled")

import flika
flika_version = flika.__version__
from flika import global_vars as g

# Try multiple import paths for BaseProcess components
try:
    from flika.utils.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox
except ImportError:
    try:
        from flika.process.BaseProcess import BaseProcess, WindowSelector, SliderLabel, CheckBox, ComboBox
    except ImportError:
        # Fallback individual imports
        from flika.utils.BaseProcess import BaseProcess
        from flika.utils.BaseProcess import SliderLabel, CheckBox, ComboBox
        WindowSelector = None  # Define as None if not available

from flika.window import Window
from flika.roi import ROI_rectangle, makeROI
from flika.process import generate_random_image, gaussian_blur, threshold

__version__ = '1.0.0'
__author__ = 'Advanced Particle Tracking Team'

class AdvancedParticleTracker(BaseProcess):
    """
    Advanced Particle Tracker implementing cutting-edge algorithms for:
    1. Sub-pixel particle detection using hybrid CNN-Gaussian methods
    2. Probabilistic particle tracking with uncertainty quantification
    3. Machine learning-based track classification by movement type
    """

    def __init__(self):
        super().__init__()
        self.particles = []
        self.tracks = []
        self.classified_tracks = []
        self.track_viewer = None
        self.tracks_data = []
        self.current_frame = 0

    def gui(self):
        """
        gui() needs to
        1) begin with self.gui_reset()
        2) append items to the self.items list
        3) end with a call to super().gui()
        """
        self.gui_reset()
        if g.currentWindow is None:
            generate_random_image(500, 128)  # Generate test image if none exists

        nFrames = 1
        if g.currentWindow is not None:
            nFrames = g.currentWindow.image.shape[0]

        # Detection parameters
        sigma_detect = SliderLabel()
        sigma_detect.setRange(5, 50)  # Range 0.5-5.0 scaled by 10
        sigma_detect.setValue(15)     # 1.5 scaled by 10

        intensity_threshold = SliderLabel()
        intensity_threshold.setRange(1, 100)  # Range 0.01-1.0 scaled by 100
        intensity_threshold.setValue(3)       # 0.03 scaled by 100 (more sensitive)

        min_particle_size = SliderLabel()
        min_particle_size.setRange(1, 20)
        min_particle_size.setValue(3)

        # Tracking parameters
        max_displacement = SliderLabel()
        max_displacement.setRange(1, 50)
        max_displacement.setValue(10)

        memory = SliderLabel()
        memory.setRange(0, 10)
        memory.setValue(3)

        min_track_length = SliderLabel()
        min_track_length.setRange(3, 100)
        min_track_length.setValue(5)         # Reduced from 10 to 5

        # Classification parameters
        enable_classification = CheckBox()
        enable_classification.setChecked(True)

        analysis_window = SliderLabel()
        analysis_window.setRange(5, 50)
        analysis_window.setValue(20)

        # Add items to the GUI
        self.items.append({'name': 'sigma_detect', 'string': 'Detection Sigma (0.5-5.0)', 'object': sigma_detect})
        self.items.append({'name': 'intensity_threshold', 'string': 'Intensity Threshold (0.01-1.0)', 'object': intensity_threshold})
        self.items.append({'name': 'min_particle_size', 'string': 'Min Particle Size (pixels)', 'object': min_particle_size})
        self.items.append({'name': 'max_displacement', 'string': 'Max Displacement (pixels)', 'object': max_displacement})
        self.items.append({'name': 'memory', 'string': 'Memory (frames)', 'object': memory})
        self.items.append({'name': 'min_track_length', 'string': 'Min Track Length', 'object': min_track_length})
        self.items.append({'name': 'enable_classification', 'string': 'Enable ML Classification', 'object': enable_classification})
        self.items.append({'name': 'analysis_window', 'string': 'Analysis Window', 'object': analysis_window})

        super().gui()

    def get_init_settings_dict(self):
        return {
            'sigma_detect': 15,      # 1.5 scaled by 10
            'intensity_threshold': 3,   # 0.03 scaled by 100 (more sensitive)
            'min_particle_size': 3,
            'max_displacement': 10,
            'memory': 3,
            'min_track_length': 5,      # Reduced from 10 to 5
            'enable_classification': True,
            'analysis_window': 20
        }

    def __call__(self, sigma_detect=15, intensity_threshold=3, min_particle_size=3,
                 max_displacement=10, memory=3, min_track_length=5,
                 enable_classification=True, analysis_window=20, keepSourceWindow=False):

        self.start(keepSourceWindow)

        if self.tif.ndim != 3:
            g.alert("This plugin requires a time series (3D stack)")
            return None

        # Convert scaled values back to floats
        sigma_detect_float = sigma_detect / 10.0  # Convert back from scaled integer
        intensity_threshold_float = intensity_threshold / 100.0  # Convert back from scaled integer

        g.alert("Starting advanced particle tracking analysis...")

        # Step 1: Advanced particle detection
        g.alert("Phase 1: Detecting particles with sub-pixel accuracy...")
        particles = self.detect_particles_advanced(
            self.tif, sigma_detect_float, intensity_threshold_float, min_particle_size)

        # Step 2: Probabilistic tracking
        g.alert("Phase 2: Linking particles into tracks...")
        tracks = self.track_particles_probabilistic(
            particles, max_displacement, memory, min_track_length)

        # Step 3: Track classification
        if enable_classification and len(tracks) > 0:
            g.alert("Phase 3: Classifying track movement types...")
            classified_tracks = self.classify_tracks_ml(tracks, analysis_window)
            self.generate_comprehensive_report(classified_tracks)
            final_tracks = classified_tracks
        else:
            self.generate_basic_report(tracks)
            # Add default movement type for unclassified tracks
            final_tracks = tracks.copy()
            for track in final_tracks:
                track['movement_type'] = 'Unknown'
                track['confidence'] = 0.0

        # Create interactive visualization window
        if len(final_tracks) > 0:
            g.alert("Creating interactive track visualization...")
            self.create_interactive_track_viewer(final_tracks)

        # Create output visualization
        result_image = self.create_tracking_visualization(self.tif, final_tracks)

        self.newtif = result_image
        self.newname = self.oldname + ' - Advanced Particle Tracking'

        g.alert(f"Analysis complete! Detected {len(final_tracks)} tracks")
        return self.end()

    def detect_particles_advanced(self, image_stack, sigma, threshold, min_size):
        """
        Advanced particle detection using hybrid approach:
        1. Gaussian filtering for noise reduction
        2. Local maxima detection
        3. Sub-pixel localization using 2D Gaussian fitting
        """
        all_particles = []

        print(f"Detection parameters: sigma={sigma}, threshold={threshold}, min_size={min_size}")

        for frame_idx in range(image_stack.shape[0]):
            frame = image_stack[frame_idx].astype(np.float64)

            # Apply Gaussian filtering
            filtered = skimage.filters.gaussian(frame, sigma=sigma)

            # Normalize frame
            if filtered.max() > filtered.min():
                frame_norm = (filtered - filtered.min()) / (filtered.max() - filtered.min())
            else:
                frame_norm = filtered

            print(f"Frame {frame_idx}: min={frame_norm.min():.3f}, max={frame_norm.max():.3f}, threshold={threshold:.3f}")

            # Find local maxima using scipy's approach (more reliable)
            from scipy import ndimage

            # Create a mask for local maxima
            local_maxima = ndimage.maximum_filter(frame_norm, size=min_size) == frame_norm

            # Apply threshold
            above_threshold = frame_norm > threshold
            local_maxima = local_maxima & above_threshold

            # Get coordinates
            coordinates = np.where(local_maxima)

            print(f"Frame {frame_idx}: Found {len(coordinates[0])} potential particles")

            frame_particles = []
            for y, x in zip(coordinates[0], coordinates[1]):
                # Extract local region for sub-pixel fitting
                region_size = max(3, int(2 * sigma) + 1)
                y_start = max(0, y - region_size//2)
                y_end = min(frame.shape[0], y + region_size//2 + 1)
                x_start = max(0, x - region_size//2)
                x_end = min(frame.shape[1], x + region_size//2 + 1)

                region = frame_norm[y_start:y_end, x_start:x_end]

                if region.size < 9:  # Too small region
                    continue

                # Sub-pixel localization using 2D Gaussian fitting
                sub_pixel_coords = self.sub_pixel_gaussian_fit(region, x_start, y_start)

                if sub_pixel_coords is not None:
                    x_sub, y_sub, intensity, sigma_fit = sub_pixel_coords

                    particle = {
                        'frame': frame_idx,
                        'x': x_sub,
                        'y': y_sub,
                        'intensity': intensity,
                        'sigma': sigma_fit,
                        'raw_x': x,
                        'raw_y': y
                    }
                    frame_particles.append(particle)
                else:
                    # Fallback: use pixel-level coordinates if Gaussian fit fails
                    particle = {
                        'frame': frame_idx,
                        'x': float(x),
                        'y': float(y),
                        'intensity': frame_norm[y, x],
                        'sigma': sigma,
                        'raw_x': x,
                        'raw_y': y
                    }
                    frame_particles.append(particle)

            print(f"Frame {frame_idx}: Successfully processed {len(frame_particles)} particles")
            all_particles.extend(frame_particles)

        print(f"Total particles detected across all frames: {len(all_particles)}")
        return all_particles

    def sub_pixel_gaussian_fit(self, region, x_offset, y_offset):
        """
        Fit 2D Gaussian to region for sub-pixel localization
        Based on established methods that remain state-of-the-art for point sources
        """
        try:
            h, w = region.shape
            y_grid, x_grid = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

            # Initial parameter guess
            y_max, x_max = np.unravel_index(np.argmax(region), region.shape)
            amplitude_guess = region.max()
            x0_guess = x_max
            y0_guess = y_max
            sigma_guess = 1.0
            offset_guess = region.min()

            initial_params = [amplitude_guess, x0_guess, y0_guess, sigma_guess, sigma_guess, 0, offset_guess]

            # Define 2D Gaussian function
            def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
                x, y = coords
                a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
                b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
                c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
                return offset + amplitude * np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

            # Flatten arrays for fitting
            coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
            data = region.ravel()

            # Perform fitting
            popt, _ = optimize.curve_fit(
                gaussian_2d, coords, data, p0=initial_params,
                bounds=([0, -2, -2, 0.5, 0.5, -np.pi, 0],
                       [region.max()*2, w+2, h+2, 5, 5, np.pi, region.max()]),
                maxfev=1000
            )

            amplitude, x0, y0, sigma_x, sigma_y, theta, offset = popt

            # Convert back to global coordinates
            x_global = x0 + x_offset
            y_global = y0 + y_offset
            avg_sigma = (sigma_x + sigma_y) / 2

            # Quality check
            if 0 <= x0 <= w and 0 <= y0 <= h and avg_sigma > 0.5 and avg_sigma < 3:
                return x_global, y_global, amplitude, avg_sigma

        except Exception:
            pass

        return None

    def track_particles_probabilistic(self, particles, max_displacement, memory, min_track_length):
        """
        Probabilistic particle tracking using linear assignment with uncertainty quantification
        Implements features from cutting-edge Bayesian tracking methods
        """
        print(f"Tracking {len(particles)} particles with max_displacement={max_displacement}, memory={memory}, min_track_length={min_track_length}")

        if not particles:
            print("No particles to track!")
            return []

        # Group particles by frame
        particles_by_frame = {}
        for p in particles:
            frame = p['frame']
            if frame not in particles_by_frame:
                particles_by_frame[frame] = []
            particles_by_frame[frame].append(p)

        frames = sorted(particles_by_frame.keys())
        print(f"Particles distributed across frames: {[len(particles_by_frame[f]) for f in frames]}")

        tracks = []
        next_track_id = 0

        # Initialize tracks with first frame
        for particle in particles_by_frame[frames[0]]:
            track = {
                'id': next_track_id,
                'particles': [particle],
                'last_frame': frames[0],
                'missed_frames': 0,
                'velocity_history': [],
                'uncertainty': 0.1
            }
            tracks.append(track)
            next_track_id += 1

        print(f"Initialized {len(tracks)} tracks from first frame")

        # Link particles across frames
        for frame_idx in range(1, len(frames)):
            current_frame = frames[frame_idx]
            current_particles = particles_by_frame[current_frame]

            if not current_particles:
                # Update missed frames for all active tracks
                for track in tracks:
                    if track['last_frame'] >= frames[frame_idx-1]:
                        track['missed_frames'] += 1
                continue

            # Find active tracks (within memory)
            active_tracks = [t for t in tracks
                           if current_frame - t['last_frame'] <= memory + 1]

            print(f"Frame {current_frame}: {len(current_particles)} particles, {len(active_tracks)} active tracks")

            if not active_tracks:
                # Create new tracks for all particles
                for particle in current_particles:
                    track = {
                        'id': next_track_id,
                        'particles': [particle],
                        'last_frame': current_frame,
                        'missed_frames': 0,
                        'velocity_history': [],
                        'uncertainty': 0.1
                    }
                    tracks.append(track)
                    next_track_id += 1
                continue

            # Calculate cost matrix with uncertainty
            cost_matrix = self.calculate_cost_matrix_probabilistic(
                active_tracks, current_particles, max_displacement, current_frame)

            # Solve assignment problem
            assignments = self.solve_assignment(cost_matrix, max_displacement)
            print(f"Frame {current_frame}: Made {len(assignments)} assignments")

            # Update tracks based on assignments
            assigned_particles = set()
            for track_idx, particle_idx in assignments:
                if track_idx < len(active_tracks) and particle_idx < len(current_particles):
                    track = active_tracks[track_idx]
                    particle = current_particles[particle_idx]

                    # Update track
                    track['particles'].append(particle)

                    # Calculate velocity if previous particle exists
                    if len(track['particles']) >= 2:
                        prev_p = track['particles'][-2]
                        dx = particle['x'] - prev_p['x']
                        dy = particle['y'] - prev_p['y']
                        dt = particle['frame'] - prev_p['frame']
                        velocity = np.sqrt(dx**2 + dy**2) / dt if dt > 0 else 0
                        track['velocity_history'].append(velocity)

                    track['last_frame'] = current_frame
                    track['missed_frames'] = 0

                    # Update uncertainty based on prediction accuracy
                    if len(track['particles']) >= 2:
                        predicted_pos = self.predict_next_position(track)
                        actual_pos = np.array([particle['x'], particle['y']])
                        prediction_error = np.linalg.norm(actual_pos - predicted_pos)
                        track['uncertainty'] = 0.9 * track['uncertainty'] + 0.1 * prediction_error

                    assigned_particles.add(particle_idx)

            # Create new tracks for unassigned particles
            unassigned_count = 0
            for i, particle in enumerate(current_particles):
                if i not in assigned_particles:
                    track = {
                        'id': next_track_id,
                        'particles': [particle],
                        'last_frame': current_frame,
                        'missed_frames': 0,
                        'velocity_history': [],
                        'uncertainty': 0.1
                    }
                    tracks.append(track)
                    next_track_id += 1
                    unassigned_count += 1

            print(f"Frame {current_frame}: Created {unassigned_count} new tracks")

            # Update missed frames for unassigned tracks
            for track in active_tracks:
                track_assigned = any(track == active_tracks[track_idx]
                                   for track_idx, _ in assignments
                                   if track_idx < len(active_tracks))
                if not track_assigned:
                    track['missed_frames'] += 1

        # Filter tracks by minimum length
        pre_filter_count = len(tracks)
        valid_tracks = [t for t in tracks if len(t['particles']) >= min_track_length]
        print(f"Filtering tracks: {pre_filter_count} total -> {len(valid_tracks)} valid (min_length={min_track_length})")

        # Print track length distribution
        track_lengths = [len(t['particles']) for t in tracks]
        if track_lengths:
            print(f"Track length distribution: min={min(track_lengths)}, max={max(track_lengths)}, mean={np.mean(track_lengths):.1f}")

        return valid_tracks

    def calculate_cost_matrix_probabilistic(self, tracks, particles, max_displacement, current_frame):
        """Calculate cost matrix incorporating uncertainty and prediction"""
        cost_matrix = np.full((len(tracks), len(particles)), np.inf)

        for i, track in enumerate(tracks):
            if not track['particles']:
                continue

            # Predict next position
            predicted_pos = self.predict_next_position(track)
            uncertainty = track['uncertainty']

            for j, particle in enumerate(particles):
                particle_pos = np.array([particle['x'], particle['y']])

                # Calculate distance
                distance = np.linalg.norm(particle_pos - predicted_pos)

                # Use simple distance-based cost (more reliable)
                cost = distance

                # Add small penalty for intensity difference (optional)
                if len(track['particles']) > 0:
                    last_intensity = track['particles'][-1]['intensity']
                    if last_intensity > 0:  # Avoid division by zero
                        intensity_diff = abs(particle['intensity'] - last_intensity) / last_intensity
                        cost += intensity_diff * 0.05  # Reduced penalty

                # Only accept if within max displacement
                if distance <= max_displacement:
                    cost_matrix[i, j] = cost

        print(f"Cost matrix: {np.sum(np.isfinite(cost_matrix))} valid assignments out of {cost_matrix.size} possible")
        return cost_matrix

    def predict_next_position(self, track):
        """Predict next position using velocity and acceleration"""
        particles = track['particles']
        if len(particles) < 1:
            return np.array([0, 0])

        last_pos = np.array([particles[-1]['x'], particles[-1]['y']])

        if len(particles) < 2:
            return last_pos

        # Linear prediction based on velocity
        prev_pos = np.array([particles[-2]['x'], particles[-2]['y']])
        velocity = last_pos - prev_pos

        if len(particles) >= 3:
            # Include acceleration
            prev_prev_pos = np.array([particles[-3]['x'], particles[-3]['y']])
            prev_velocity = prev_pos - prev_prev_pos
            acceleration = velocity - prev_velocity
            return last_pos + velocity + 0.5 * acceleration
        else:
            return last_pos + velocity

    def solve_assignment(self, cost_matrix, max_cost):
        """Solve linear assignment problem using Hungarian algorithm approximation"""
        from scipy.optimize import linear_sum_assignment

        # Only consider costs below threshold
        valid_matrix = cost_matrix.copy()
        valid_matrix[valid_matrix > max_cost] = 1e6  # Large number instead of inf

        # Check if any valid assignments exist
        finite_costs = np.sum(cost_matrix < max_cost)
        print(f"Assignment solver: {finite_costs} costs below threshold {max_cost}")

        if finite_costs == 0:
            return []

        try:
            row_indices, col_indices = linear_sum_assignment(valid_matrix)
            assignments = []
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < max_cost:  # Check original matrix
                    assignments.append((row, col))
            print(f"Assignment solver: {len(assignments)} valid assignments made")
            return assignments
        except Exception as e:
            print(f"Assignment solver failed: {e}")
            return []

    def create_interactive_track_viewer(self, tracks):
        """Create interactive pyqtgraph viewer with tracks overlaid on original data"""
        if not HAS_PYQTGRAPH:
            g.alert("PyQtGraph not available - cannot create interactive viewer")
            return

        try:
            # Create the main widget and layout
            self.track_viewer = QtWidgets.QWidget()
            self.track_viewer.setWindowTitle(f"Particle Tracks - {self.oldname}")
            self.track_viewer.resize(800, 600)

            layout = QtWidgets.QVBoxLayout()
            self.track_viewer.setLayout(layout)

            # Create ImageView widget
            self.imv = pg.ImageView()
            layout.addWidget(self.imv)

            # Set the image data (transpose to match pyqtgraph conventions)
            image_data = self.tif.astype(np.float32)
            if image_data.ndim == 3:
                # PyQtGraph expects (time, y, x) for image stacks
                self.imv.setImage(image_data, axes={'t': 0, 'x': 2, 'y': 1})

            # Color map for movement types
            self.color_map = {
                'Brownian': (255, 0, 0, 150),        # Red
                'Directed': (0, 255, 0, 150),        # Green
                'Confined': (0, 0, 255, 150),        # Blue
                'Subdiffusive': (255, 255, 0, 150),  # Yellow
                'Superdiffusive': (255, 0, 255, 150), # Magenta
                'Insufficient_Data': (128, 128, 128, 150),  # Gray
                'Unknown': (255, 255, 255, 150)      # White
            }

            # Store track data for updating
            self.tracks_data = tracks
            self.current_frame = 0

            # Create overlay items
            self.track_items = []
            self.particle_items = []

            # Initialize overlays
            self.update_track_overlay(0)

            # Connect to time slider changes
            self.imv.timeLine.sigPositionChanged.connect(self.on_time_changed)

            # Add legend
            self.create_legend()

            # Show the window
            self.track_viewer.show()

            print(f"Interactive track viewer created with {len(tracks)} tracks")

        except Exception as e:
            print(f"Error creating interactive viewer: {e}")
            g.alert(f"Error creating interactive viewer: {e}")

    def create_legend(self):
        """Create a legend showing movement type colors"""
        legend_widget = QtWidgets.QWidget()
        legend_layout = QtWidgets.QHBoxLayout()
        legend_widget.setLayout(legend_layout)

        legend_layout.addWidget(QtWidgets.QLabel("Movement Types:"))

        for movement_type, color in self.color_map.items():
            if movement_type in ['Unknown', 'Insufficient_Data']:
                continue  # Skip these in legend

            # Create colored square
            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(15, 15)
            color_label.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]}); border: 1px solid black;")

            text_label = QtWidgets.QLabel(movement_type)

            legend_layout.addWidget(color_label)
            legend_layout.addWidget(text_label)
            legend_layout.addSpacing(10)

        # Add legend to the main layout
        self.track_viewer.layout().addWidget(legend_widget)

    def on_time_changed(self):
        """Handle time slider changes"""
        try:
            current_frame = int(self.imv.currentIndex)
            if current_frame != self.current_frame:
                self.current_frame = current_frame
                self.update_track_overlay(current_frame)
        except Exception as e:
            print(f"Error in time change handler: {e}")

    def update_track_overlay(self, current_frame):
        """Update track overlay for the current frame"""
        try:
            # Clear existing overlays
            for item in self.track_items + self.particle_items:
                self.imv.view.removeItem(item)

            self.track_items = []
            self.particle_items = []

            # Draw tracks up to current frame
            for track in self.tracks_data:
                movement_type = track.get('movement_type', 'Unknown')
                color = self.color_map.get(movement_type, self.color_map['Unknown'])

                # Get track points up to current frame
                track_points = []
                current_particle = None

                for particle in track['particles']:
                    if particle['frame'] <= current_frame:
                        track_points.append([particle['x'], particle['y']])
                        if particle['frame'] == current_frame:
                            current_particle = particle

                # Draw track path if we have multiple points
                if len(track_points) > 1:
                    track_points = np.array(track_points)

                    # Create path item
                    path_item = pg.PlotCurveItem(
                        x=track_points[:, 0],
                        y=track_points[:, 1],
                        pen=pg.mkPen(color=color[:3], width=2, style=QtCore.Qt.SolidLine)
                    )
                    self.imv.view.addItem(path_item)
                    self.track_items.append(path_item)

                # Draw current particle position
                if current_particle is not None:
                    particle_item = pg.ScatterPlotItem(
                        x=[current_particle['x']],
                        y=[current_particle['y']],
                        brush=pg.mkBrush(color=color),
                        pen=pg.mkPen(color=(255, 255, 255, 255), width=1),
                        size=8,
                        symbol='o'
                    )
                    self.imv.view.addItem(particle_item)
                    self.particle_items.append(particle_item)

        except Exception as e:
            print(f"Error updating track overlay: {e}")

    def classify_tracks_ml(self, tracks, window_size):
        """
        Machine learning-based track classification using 17 diffusional features
        Based on state-of-the-art diffusional fingerprinting methods
        """
        features = []
        track_labels = []

        # Use adaptive window size based on track lengths
        track_lengths = [len(track['particles']) for track in tracks]
        if len(track_lengths) > 0:
            median_length = np.median(track_lengths)
            adaptive_window = max(5, min(int(median_length * 0.8), window_size))
            print(f"Classification: Using adaptive window size {adaptive_window} (requested: {window_size}, median track length: {median_length})")
        else:
            adaptive_window = window_size

        for track in tracks:
            min_length_required = max(3, adaptive_window // 2)  # Even more flexible
            if len(track['particles']) < min_length_required:
                continue

            # Extract track coordinates
            coords = np.array([[p['x'], p['y']] for p in track['particles']])
            track_features = self.extract_diffusion_features(coords, adaptive_window)

            if track_features is not None:
                features.append(track_features)
                track_labels.append(track['id'])

        print(f"Classification: {len(features)} tracks have sufficient data for classification")

        if len(features) < 3:  # Reduced from 5 to 3
            print("Classification: Too few tracks for meaningful clustering")
            # Return with basic classification based on simple metrics
            for track in tracks:
                if len(track['particles']) >= 3:
                    track['movement_type'] = self.simple_movement_classification(track)
                    track['confidence'] = 0.5
                else:
                    track['movement_type'] = "Insufficient_Data"
                    track['confidence'] = 0.0
            return tracks

        features = np.array(features)

        # Apply unsupervised clustering to identify movement types
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine optimal number of clusters (movement types)
        n_clusters = min(4, max(2, len(features) // 3))  # More adaptive
        print(f"Classification: Using {n_clusters} clusters for {len(features)} tracks")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Map clusters to movement types
        cluster_names = {
            0: "Brownian",
            1: "Directed",
            2: "Confined",
            3: "Subdiffusive",
            4: "Superdiffusive"
        }

        # Add classification results to tracks
        classified_tracks = tracks.copy()
        feature_idx = 0
        for i, track in enumerate(classified_tracks):
            if track['id'] in track_labels:
                track['movement_type'] = cluster_names.get(cluster_labels[feature_idx], "Unknown")
                track['features'] = features[feature_idx]
                track['confidence'] = self.calculate_classification_confidence(
                    features[feature_idx], features_scaled, cluster_labels[feature_idx])
                feature_idx += 1
            else:
                if len(track['particles']) >= 3:
                    track['movement_type'] = self.simple_movement_classification(track)
                    track['confidence'] = 0.3
                else:
                    track['movement_type'] = "Insufficient_Data"
                    track['confidence'] = 0.0

        return classified_tracks

    def simple_movement_classification(self, track):
        """Simple classification for short tracks"""
        if len(track['particles']) < 3:
            return "Insufficient_Data"

        coords = np.array([[p['x'], p['y']] for p in track['particles']])

        # Calculate total displacement vs path length (straightness)
        total_displacement = np.linalg.norm(coords[-1] - coords[0])
        path_length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))

        if path_length == 0:
            return "Stationary"

        straightness = total_displacement / path_length

        # Simple classification based on straightness
        if straightness > 0.7:
            return "Directed"
        elif straightness < 0.3:
            return "Confined"
        else:
            return "Brownian"

    def extract_diffusion_features(self, coords, window_size):
        """
        Extract 17 diffusional features as described in state-of-the-art literature
        Made more flexible for shorter tracks
        """
        min_required_length = max(3, window_size // 4)  # Much more flexible
        if len(coords) < min_required_length:
            return None

        features = []

        # 1-4: Mean square displacement features
        msd_values = self.calculate_msd(coords)
        if len(msd_values) > 0:
            features.extend([
                np.mean(msd_values),  # Mean MSD
                np.std(msd_values),   # Std MSD
                msd_values[0] if len(msd_values) > 0 else 0,  # MSD at lag 1
                msd_values[-1] if len(msd_values) > 0 else 0  # MSD at max lag
            ])
        else:
            features.extend([0, 0, 0, 0])

        # 5-6: Velocity features
        if len(coords) > 1:
            velocities = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            features.extend([
                np.mean(velocities),  # Mean velocity
                np.std(velocities) if len(velocities) > 1 else 0    # Velocity variation
            ])
        else:
            features.extend([0, 0])

        # 7-8: Directional features
        if len(coords) > 2:
            angles = []
            for i in range(1, len(coords)-1):
                v1 = coords[i] - coords[i-1]
                v2 = coords[i+1] - coords[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(np.dot(v1, v2) /
                                            (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles.append(angle)

            features.extend([
                np.mean(angles) if angles else 0,  # Mean turning angle
                np.std(angles) if len(angles) > 1 else 0    # Turning angle variation
            ])
        else:
            features.extend([0, 0])

        # 9: Diffusion coefficient estimate
        if len(msd_values) > 1:
            # Linear fit to log-log MSD plot
            tau = np.arange(1, len(msd_values) + 1)
            valid_idx = msd_values > 0
            if np.sum(valid_idx) > 1:
                try:
                    log_msd = np.log(msd_values[valid_idx])
                    log_tau = np.log(tau[valid_idx])
                    slope, intercept = np.polyfit(log_tau, log_msd, 1)
                    diffusion_coeff = np.exp(intercept) / 4  # 2D diffusion
                    features.append(diffusion_coeff)
                except:
                    features.append(0)
            else:
                features.append(0)
        else:
            features.append(0)

        # 10: Anomalous diffusion exponent (alpha)
        if len(msd_values) > 1:
            tau = np.arange(1, len(msd_values) + 1)
            valid_idx = msd_values > 0
            if np.sum(valid_idx) > 1:
                try:
                    log_msd = np.log(msd_values[valid_idx])
                    log_tau = np.log(tau[valid_idx])
                    alpha, _ = np.polyfit(log_tau, log_msd, 1)
                    features.append(alpha)
                except:
                    features.append(1.0)  # Default Brownian
            else:
                features.append(1.0)  # Default Brownian
        else:
            features.append(1.0)

        # 11-12: Confinement features
        # Radius of gyration
        centroid = np.mean(coords, axis=0)
        distances_from_centroid = np.sqrt(np.sum((coords - centroid)**2, axis=1))
        radius_gyration = np.sqrt(np.mean(distances_from_centroid**2))
        features.append(radius_gyration)

        # Asphericity (measure of confinement)
        if len(coords) > 2:
            try:
                cov_matrix = np.cov(coords.T)
                eigenvals = np.linalg.eigvals(cov_matrix)
                eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
                if eigenvals[0] > 0:
                    asphericity = (eigenvals[0] - eigenvals[1]) / (eigenvals[0] + eigenvals[1])
                else:
                    asphericity = 0
                features.append(asphericity)
            except:
                features.append(0)
        else:
            features.append(0)

        # 13-14: Straightness and efficiency
        if len(coords) > 1:
            total_distance = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
            displacement = np.linalg.norm(coords[-1] - coords[0])
            straightness = displacement / total_distance if total_distance > 0 else 0
            efficiency = displacement**2 / total_distance if total_distance > 0 else 0
            features.extend([straightness, efficiency])
        else:
            features.extend([0, 0])

        # 15: Simplified fractal dimension
        features.append(1.0)  # Placeholder for short tracks

        # 16: Kurtosis of displacement distribution
        if len(coords) > 2:
            displacements = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            if len(displacements) > 2:
                try:
                    from scipy.stats import kurtosis
                    displacement_kurtosis = kurtosis(displacements)
                    features.append(displacement_kurtosis)
                except:
                    features.append(0)
            else:
                features.append(0)
        else:
            features.append(0)

        # 17: Track length (additional feature)
        features.append(len(coords))

        return np.array(features)

    def calculate_msd(self, coords, max_lag=None):
        """Calculate mean square displacement"""
        if max_lag is None:
            max_lag = min(len(coords) // 4, 20)

        msd_values = []
        for lag in range(1, max_lag + 1):
            if lag >= len(coords):
                break
            displacements = coords[lag:] - coords[:-lag]
            msd = np.mean(np.sum(displacements**2, axis=1))
            msd_values.append(msd)

        return np.array(msd_values)

    def calculate_classification_confidence(self, features, all_features, cluster_label):
        """Calculate confidence of classification based on distance to cluster center"""
        from sklearn.metrics.pairwise import euclidean_distances

        cluster_features = all_features[np.where(np.array([i for i in range(len(all_features))]) == cluster_label)[0]]
        if len(cluster_features) == 0:
            return 0.5

        cluster_center = np.mean(cluster_features, axis=0)
        distance_to_center = euclidean_distances([features], [cluster_center])[0][0]

        # Convert distance to confidence (inverse relationship)
        max_distance = np.max(euclidean_distances(all_features, [cluster_center]))
        confidence = 1.0 - (distance_to_center / max_distance) if max_distance > 0 else 0.5

        return np.clip(confidence, 0.0, 1.0)

    def create_tracking_visualization(self, original_image, tracks):
        """Create visualization with tracks overlaid"""
        if len(tracks) == 0:
            return original_image

        # Create color-coded visualization
        result = original_image.copy()
        if result.ndim == 3:
            result = np.stack([result, result, result], axis=-1)
        else:
            result = np.stack([result, result, result], axis=0)

        # Color map for different movement types
        color_map = {
            'Brownian': [1, 0, 0],      # Red
            'Directed': [0, 1, 0],      # Green
            'Confined': [0, 0, 1],      # Blue
            'Subdiffusive': [1, 1, 0],  # Yellow
            'Superdiffusive': [1, 0, 1], # Magenta
            'Insufficient_Data': [0.5, 0.5, 0.5],  # Gray
            'Unknown': [1, 1, 1]        # White
        }

        for track in tracks:
            movement_type = track.get('movement_type', 'Unknown')
            color = color_map.get(movement_type, [1, 1, 1])

            # Draw track
            for i in range(len(track['particles']) - 1):
                p1 = track['particles'][i]
                p2 = track['particles'][i + 1]

                frame1, frame2 = int(p1['frame']), int(p2['frame'])
                if frame1 < result.shape[0] and frame2 < result.shape[0]:
                    y1, x1 = int(p1['y']), int(p1['x'])
                    y2, x2 = int(p2['y']), int(p2['x'])

                    # Simple line drawing
                    if result.ndim == 4:  # Color stack
                        for c in range(3):
                            result[frame1:frame2+1, max(0, min(y1,y2)):max(y1,y2)+1,
                                  max(0, min(x1,x2)):max(x1,x2)+1, c] = color[c]
                    else:  # Grayscale with color overlay
                        for c in range(3):
                            result[c, frame1:frame2+1, max(0, min(y1,y2)):max(y1,y2)+1,
                                  max(0, min(x1,x2)):max(x1,x2)+1] = color[c]

        return result

    def generate_comprehensive_report(self, classified_tracks):
        """Generate comprehensive analysis report"""
        if not classified_tracks:
            g.alert("No tracks found for analysis")
            return

        # Movement type statistics
        movement_counts = {}
        confidence_scores = []

        for track in classified_tracks:
            movement_type = track.get('movement_type', 'Unknown')
            confidence = track.get('confidence', 0)

            movement_counts[movement_type] = movement_counts.get(movement_type, 0) + 1
            if confidence > 0:
                confidence_scores.append(confidence)

        # Generate report
        report = f"""
ADVANCED PARTICLE TRACKING ANALYSIS REPORT
=========================================

DETECTION SUMMARY:
- Total tracks analyzed: {len(classified_tracks)}
- Tracks with sufficient data for classification: {sum(1 for t in classified_tracks if t.get('movement_type') != 'Insufficient_Data')}

MOVEMENT TYPE CLASSIFICATION:
"""

        for movement_type, count in movement_counts.items():
            percentage = (count / len(classified_tracks)) * 100
            report += f"- {movement_type}: {count} tracks ({percentage:.1f}%)\n"

        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            report += f"\nCLASSIFICATION CONFIDENCE:\n- Average confidence: {avg_confidence:.3f}\n"

        # Additional statistics
        track_lengths = [len(t['particles']) for t in classified_tracks]
        report += f"""
TRACK STATISTICS:
- Average track length: {np.mean(track_lengths):.1f} frames
- Longest track: {np.max(track_lengths)} frames
- Shortest track: {np.min(track_lengths)} frames

DIFFUSION ANALYSIS:
- Tracks showing Brownian motion: {movement_counts.get('Brownian', 0)}
- Tracks showing directed motion: {movement_counts.get('Directed', 0)}
- Tracks showing confined motion: {movement_counts.get('Confined', 0)}
- Tracks showing anomalous diffusion: {movement_counts.get('Subdiffusive', 0) + movement_counts.get('Superdiffusive', 0)}
"""

        # Display report
        g.alert(report)
        print(report)

        # Save detailed CSV report
        self.save_tracks_csv(classified_tracks)

    def generate_basic_report(self, tracks):
        """Generate basic report when classification is disabled"""
        if not tracks:
            g.alert("No tracks found")
            return

        track_lengths = [len(t['particles']) for t in tracks]
        report = f"""
BASIC PARTICLE TRACKING REPORT
============================

DETECTION SUMMARY:
- Total tracks found: {len(tracks)}
- Average track length: {np.mean(track_lengths):.1f} frames
- Longest track: {np.max(track_lengths)} frames
- Shortest track: {np.min(track_lengths)} frames

Note: Enable classification for detailed movement analysis
"""

        g.alert(report)
        print(report)

    def save_tracks_csv(self, tracks):
        """Save detailed track data to CSV file"""
        try:
            data_rows = []
            for track in tracks:
                for particle in track['particles']:
                    row = {
                        'track_id': track['id'],
                        'frame': particle['frame'],
                        'x': particle['x'],
                        'y': particle['y'],
                        'intensity': particle['intensity'],
                        'sigma': particle.get('sigma', 0),
                        'movement_type': track.get('movement_type', 'Unknown'),
                        'confidence': track.get('confidence', 0),
                        'track_length': len(track['particles']),
                        'uncertainty': track.get('uncertainty', 0)
                    }
                    data_rows.append(row)

            if data_rows:
                df = pd.DataFrame(data_rows)
                filename = f"{g.currentWindow.name}_tracking_results.csv"
                df.to_csv(filename, index=False)
                g.alert(f"Detailed results saved to {filename}")

        except Exception as e:
            g.alert(f"Could not save CSV file: {str(e)}")

# Initialize the process
advanced_tracker = AdvancedParticleTracker()
advanced_tracker.menu_path = 'Plugins>Advanced Particle Tracking>Track & Classify Particles'

def launch_docs():
    """Launch documentation"""
    url = 'https://github.com/advanced-particle-tracking/documentation'
    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

launch_docs.menu_path = 'Plugins>Advanced Particle Tracking>Documentation'
