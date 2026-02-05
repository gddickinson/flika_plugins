# single_molecule_tracker/__init__.py
"""
Single Molecule Tracker Plugin for FLIKA
Tracks individual fluorescent spots over time with subpixel accuracy
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from flika import global_vars as g
from flika.window import Window
from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox
from flika.roi import makeROI
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem

__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Suite'

class SingleMoleculeTracker(BaseProcess):
    """
    Track individual fluorescent molecules over time using advanced algorithms
    """
    
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.spots_per_frame = []
        
    def get_init_settings_dict(self):
        return {
            'detection_threshold': 3.0,
            'min_spot_size': 2,
            'max_spot_size': 8,
            'max_displacement': 5.0,
            'min_track_length': 3,
            'fit_gaussian': True,
            'subpixel_accuracy': True
        }
    
    def get_params_dict(self):
        params = super().get_params_dict()
        params['detection_threshold'] = self.detection_threshold.value()
        params['min_spot_size'] = int(self.min_spot_size.value())
        params['max_spot_size'] = int(self.max_spot_size.value())
        params['max_displacement'] = self.max_displacement.value()
        params['min_track_length'] = int(self.min_track_length.value())
        params['fit_gaussian'] = self.fit_gaussian.isChecked()
        params['subpixel_accuracy'] = self.subpixel_accuracy.isChecked()
        return params
    
    def get_name(self):
        return 'Single Molecule Tracker'
    
    def get_menu_path(self):
        return 'Plugins>TIRF Analysis>Single Molecule Tracker'
    
    def setupGUI(self):
        super().setupGUI()
        self.detection_threshold.setRange(1.0, 10.0)
        self.min_spot_size.setRange(1, 20)
        self.max_spot_size.setRange(1, 50)
        self.max_displacement.setRange(0.5, 20.0)
        self.min_track_length.setRange(2, 100)
        
        # Add custom buttons
        button_layout = QHBoxLayout()
        
        self.track_button = QPushButton("Start Tracking")
        self.track_button.clicked.connect(self.run_tracking)
        button_layout.addWidget(self.track_button)
        
        self.export_button = QPushButton("Export Tracks")
        self.export_button.clicked.connect(self.export_tracks)
        button_layout.addWidget(self.export_button)
        
        self.layout().addLayout(button_layout)
    
    def gaussian_2d(self, coords, amplitude, x0, y0, sigma_x, sigma_y, offset):
        """2D Gaussian function for subpixel fitting"""
        x, y = coords
        return (amplitude * np.exp(-(((x-x0)/sigma_x)**2 + ((y-y0)/sigma_y)**2)/2) + offset).ravel()
    
    def detect_spots(self, frame, threshold, min_size, max_size):
        """Detect fluorescent spots in a single frame"""
        # Apply gaussian filter for noise reduction
        filtered = ndimage.gaussian_filter(frame.astype(float), sigma=1.0)
        
        # Local maxima detection
        local_maxima = ndimage.maximum_filter(filtered, size=3) == filtered
        
        # Threshold detection
        above_threshold = filtered > (threshold * np.std(filtered) + np.mean(filtered))
        
        # Combine conditions
        spots = local_maxima & above_threshold
        
        # Label connected components
        labeled, num_spots = ndimage.label(spots)
        
        spot_coords = []
        spot_intensities = []
        
        for i in range(1, num_spots + 1):
            spot_pixels = np.where(labeled == i)
            
            if len(spot_pixels[0]) < min_size or len(spot_pixels[0]) > max_size:
                continue
                
            # Center of mass
            y_center = np.mean(spot_pixels[0])
            x_center = np.mean(spot_pixels[1])
            
            # Total intensity
            intensity = np.sum(frame[spot_pixels])
            
            spot_coords.append([x_center, y_center])
            spot_intensities.append(intensity)
        
        return np.array(spot_coords), np.array(spot_intensities)
    
    def fit_gaussian_subpixel(self, frame, x, y, window_size=5):
        """Fit 2D Gaussian for subpixel accuracy"""
        try:
            # Extract window around spot
            x_int, y_int = int(x), int(y)
            x_min = max(0, x_int - window_size)
            x_max = min(frame.shape[1], x_int + window_size + 1)
            y_min = max(0, y_int - window_size)
            y_max = min(frame.shape[0], y_int + window_size + 1)
            
            window = frame[y_min:y_max, x_min:x_max]
            
            # Create coordinate grids
            x_coords, y_coords = np.meshgrid(
                np.arange(x_min, x_max),
                np.arange(y_min, y_max)
            )
            
            # Initial guess
            amplitude = np.max(window) - np.min(window)
            offset = np.min(window)
            
            initial_guess = [amplitude, x, y, 2.0, 2.0, offset]
            
            # Fit
            popt, _ = curve_fit(
                self.gaussian_2d,
                (x_coords, y_coords),
                window.ravel(),
                p0=initial_guess,
                maxfev=1000
            )
            
            return popt[1], popt[2], popt[0]  # x, y, amplitude
            
        except:
            return x, y, frame[int(y), int(x)]
    
    def link_spots(self, spots_frame1, spots_frame2, max_displacement):
        """Link spots between consecutive frames"""
        if len(spots_frame1) == 0 or len(spots_frame2) == 0:
            return []
        
        # Calculate distance matrix
        nbrs = NearestNeighbors(n_neighbors=1).fit(spots_frame2)
        distances, indices = nbrs.kneighbors(spots_frame1)
        
        links = []
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if dist <= max_displacement:
                links.append((i, idx))
        
        return links
    
    def run_tracking(self):
        """Main tracking algorithm"""
        if g.win is None:
            g.alert("No window open!")
            return
        
        params = self.get_params_dict()
        image_stack = g.win.image
        
        g.m.statusBar().showMessage("Detecting spots...")
        
        # Detect spots in all frames
        all_spots = []
        all_intensities = []
        
        for frame_idx in range(image_stack.shape[0]):
            frame = image_stack[frame_idx]
            spots, intensities = self.detect_spots(
                frame, 
                params['detection_threshold'],
                params['min_spot_size'],
                params['max_spot_size']
            )
            
            # Subpixel refinement if requested
            if params['subpixel_accuracy'] and params['fit_gaussian']:
                refined_spots = []
                refined_intensities = []
                
                for (x, y), intensity in zip(spots, intensities):
                    x_ref, y_ref, amp = self.fit_gaussian_subpixel(frame, x, y)
                    refined_spots.append([x_ref, y_ref])
                    refined_intensities.append(amp)
                
                spots = np.array(refined_spots) if refined_spots else spots
                intensities = np.array(refined_intensities) if refined_intensities else intensities
            
            all_spots.append(spots)
            all_intensities.append(intensities)
            self.spots_per_frame.append(len(spots))
        
        g.m.statusBar().showMessage("Linking tracks...")
        
        # Link spots into tracks
        tracks = []
        track_id = 0
        
        # Initialize tracks from first frame
        for i, (pos, intensity) in enumerate(zip(all_spots[0], all_intensities[0])):
            tracks.append({
                'track_id': track_id,
                'frames': [0],
                'positions': [pos],
                'intensities': [intensity]
            })
            track_id += 1
        
        # Link subsequent frames
        for frame_idx in range(1, len(all_spots)):
            current_spots = all_spots[frame_idx]
            current_intensities = all_intensities[frame_idx]
            
            # Get positions of active tracks
            active_tracks = [t for t in tracks if t['frames'][-1] == frame_idx - 1]
            
            if len(active_tracks) == 0:
                # Start new tracks
                for pos, intensity in zip(current_spots, current_intensities):
                    tracks.append({
                        'track_id': track_id,
                        'frames': [frame_idx],
                        'positions': [pos],
                        'intensities': [intensity]
                    })
                    track_id += 1
                continue
            
            # Link spots
            previous_positions = [t['positions'][-1] for t in active_tracks]
            links = self.link_spots(
                np.array(previous_positions),
                current_spots,
                params['max_displacement']
            )
            
            # Update linked tracks
            linked_track_indices = set()
            linked_spot_indices = set()
            
            for track_idx, spot_idx in links:
                tracks[active_tracks[track_idx]['track_id']]['frames'].append(frame_idx)
                tracks[active_tracks[track_idx]['track_id']]['positions'].append(current_spots[spot_idx])
                tracks[active_tracks[track_idx]['track_id']]['intensities'].append(current_intensities[spot_idx])
                linked_track_indices.add(track_idx)
                linked_spot_indices.add(spot_idx)
            
            # Start new tracks for unlinked spots
            for spot_idx, (pos, intensity) in enumerate(zip(current_spots, current_intensities)):
                if spot_idx not in linked_spot_indices:
                    tracks.append({
                        'track_id': track_id,
                        'frames': [frame_idx],
                        'positions': [pos],
                        'intensities': [intensity]
                    })
                    track_id += 1
        
        # Filter tracks by minimum length
        self.tracks = [t for t in tracks if len(t['frames']) >= params['min_track_length']]
        
        g.m.statusBar().showMessage(f"Tracking complete! Found {len(self.tracks)} tracks", 3000)
        
        # Visualize results
        self.visualize_tracks()
    
    def visualize_tracks(self):
        """Create visualization of tracks"""
        if not self.tracks:
            return
        
        # Create track overlay
        track_image = np.zeros_like(g.win.image[0], dtype=float)
        
        for track in self.tracks:
            positions = np.array(track['positions'])
            if len(positions) > 1:
                # Draw track as line
                for i in range(len(positions) - 1):
                    x1, y1 = positions[i].astype(int)
                    x2, y2 = positions[i + 1].astype(int)
                    
                    # Simple line drawing
                    if 0 <= x1 < track_image.shape[1] and 0 <= y1 < track_image.shape[0]:
                        track_image[y1, x1] = track['track_id'] + 1
                    if 0 <= x2 < track_image.shape[1] and 0 <= y2 < track_image.shape[0]:
                        track_image[y2, x2] = track['track_id'] + 1
        
        # Create new window with tracks
        Window(track_image, name=f"{g.win.name}_tracks")
        
        # Print summary statistics
        track_lengths = [len(t['frames']) for t in self.tracks]
        total_spots = sum(self.spots_per_frame)
        
        print(f"\n=== Tracking Results ===")
        print(f"Total tracks found: {len(self.tracks)}")
        print(f"Average track length: {np.mean(track_lengths):.1f} frames")
        print(f"Median track length: {np.median(track_lengths):.1f} frames")
        print(f"Total spots detected: {total_spots}")
        print(f"Average spots per frame: {np.mean(self.spots_per_frame):.1f}")
    
    def export_tracks(self):
        """Export track data to CSV"""
        if not self.tracks:
            g.alert("No tracks to export! Run tracking first.")
            return
        
        # Prepare data for export
        export_data = []
        
        for track in self.tracks:
            for frame, pos, intensity in zip(track['frames'], track['positions'], track['intensities']):
                export_data.append({
                    'track_id': track['track_id'],
                    'frame': frame,
                    'x': pos[0],
                    'y': pos[1],
                    'intensity': intensity
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(export_data)
        filename = f"{g.win.name}_tracks.csv"
        df.to_csv(filename, index=False)
        
        g.alert(f"Tracks exported to {filename}")
        print(f"Exported {len(self.tracks)} tracks to {filename}")

    def process(self):
        """Process method required by BaseProcess"""
        self.run_tracking()
        return None

# Register the plugin
SingleMoleculeTracker.menu_path = 'Plugins>TIRF Analysis>Single Molecule Tracker'