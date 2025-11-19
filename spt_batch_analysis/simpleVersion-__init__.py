#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIKA SPT Batch Analysis Plugin - Simplified Version
Comprehensive single particle tracking analysis pipeline

Created by: Assistant
Based on scripts by: George
"""

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
import pandas as pd
from tqdm import tqdm
import os, glob, sys
import json
import math
from pathlib import Path

# FLIKA imports
import flika
from flika import global_vars as g
from flika.window import Window
from flika.process.file_ import open_file
from flika.utils.misc import open_file_gui, save_file_gui

# Scientific computing
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import stats, spatial
import skimage.io as skio

# Qt imports
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                           QCheckBox, QComboBox, QTextEdit, QProgressBar,
                           QGroupBox, QGridLayout, QFormLayout, QFileDialog,
                           QListWidget, QFrame, QApplication)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont


class SPTAnalysisParameters:
    """Centralized parameter management for SPT analysis"""

    def __init__(self):
        # File processing parameters
        self.pixel_size = 108  # nm per pixel
        self.frame_length = 1  # seconds per frame
        self.min_track_segments = 4

        # Linking parameters
        self.max_gap_frames = 36
        self.max_link_distance = 3  # pixels

        # Feature calculation parameters
        self.nn_radii = [3, 5, 10, 20, 30]  # pixels
        self.rg_mobility_threshold = 2.11

        # Classification parameters
        self.training_data_path = ""
        self.experiment_name = ""

        # Output options
        self.save_intermediate = True

    def to_dict(self):
        """Convert parameters to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def from_dict(self, param_dict):
        """Load parameters from dictionary"""
        for key, value in param_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


class Points:
    """Optimized Points class for particle linking"""

    def __init__(self, txy_pts):
        self.frames = np.unique(txy_pts[:, 0]).astype(int)
        self.txy_pts = txy_pts
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []
        self.intensities = []
        self.tracks = []
        self.recursiveFailure = False

    def link_pts(self, maxFramesSkipped, maxDistance):
        """Link points across frames"""
        try:
            print('Linking points...')
            self._prepare_frame_data()

            tracks = []
            for frame in self.frames:
                for pt_idx in np.where(self.pts_remaining[frame])[0]:
                    self.pts_remaining[frame][pt_idx] = False
                    abs_pt_idx = self.pts_idx_by_frame[frame][pt_idx]
                    track = [abs_pt_idx]
                    track = self._extend_track(track, maxFramesSkipped, maxDistance)
                    tracks.append(track)

            self.tracks = tracks
            print(f'Linking complete. Created {len(tracks)} tracks.')

        except Exception as e:
            print(f"Error in linking: {e}")
            self.recursiveFailure = True

    def _prepare_frame_data(self):
        """Prepare frame-by-frame data structures"""
        self.pts_by_frame = []
        self.pts_remaining = []
        self.pts_idx_by_frame = []

        for frame in np.arange(0, np.max(self.frames) + 1):
            indices = np.where(self.txy_pts[:, 0] == frame)[0]
            pos = self.txy_pts[indices, 1:]
            self.pts_by_frame.append(pos)
            self.pts_remaining.append(np.ones(pos.shape[0], dtype=bool))
            self.pts_idx_by_frame.append(indices)

    def _extend_track(self, track, maxFramesSkipped, maxDistance, depth=0):
        """Extend track with recursion depth limiting"""
        if depth >= 1000:  # Prevent stack overflow
            self.recursiveFailure = True
            return track

        pt = self.txy_pts[track[-1]]

        for dt in np.arange(1, maxFramesSkipped + 2):
            frame = int(pt[0]) + dt
            if frame >= len(self.pts_remaining):
                return track

            candidates = self.pts_remaining[frame]
            nCandidates = np.count_nonzero(candidates)

            if nCandidates == 0:
                continue

            distances = np.sqrt(np.sum(
                (self.pts_by_frame[frame][candidates] - pt[1:]) ** 2, 1))

            if any(distances < maxDistance):
                next_pt_idx = np.where(candidates)[0][np.argmin(distances)]
                abs_next_pt_idx = self.pts_idx_by_frame[frame][next_pt_idx]
                track.append(abs_next_pt_idx)
                self.pts_remaining[frame][next_pt_idx] = False
                track = self._extend_track(track, maxFramesSkipped, maxDistance, depth + 1)
                return track

        return track

    def getIntensities(self, dataArray):
        """Extract intensities from image data"""
        print("Extracting intensities...")
        n, w, h = dataArray.shape
        self.intensities = []

        for point in tqdm(self.txy_pts, desc="Processing intensities"):
            frame = int(round(point[0]))
            x = int(round(point[1]))
            y = int(round(point[2]))

            # 3x3 pixel region
            xMin, xMax = max(0, x-1), min(w, x+2)
            yMin, yMax = max(0, y-1), min(h, y+2)

            intensity = np.mean(dataArray[frame][xMin:xMax, yMin:yMax])
            self.intensities.append(intensity)


class FeatureCalculator:
    """Feature calculation methods"""

    @staticmethod
    def radius_gyration_asymmetry(trackDF):
        """Calculate radius of gyration and asymmetry features"""
        points_array = np.array(trackDF[['x', 'y']].dropna())
        center = points_array.mean(0)
        normed_points = points_array - center[None, :]

        rg_tensor = np.einsum('im,in->mn', normed_points, normed_points) / len(points_array)
        eig_values, eig_vectors = np.linalg.eig(rg_tensor)

        radius_gyration = np.sqrt(np.sum(eig_values))

        asymmetry_num = (eig_values[0] - eig_values[1]) ** 2
        asymmetry_den = 2 * (eig_values[0] + eig_values[1]) ** 2
        asymmetry = -math.log(1 - (asymmetry_num / asymmetry_den)) if asymmetry_den > 0 else 0

        # Projection for skewness/kurtosis
        maxcol = list(eig_values).index(max(eig_values))
        dom_eig_vect = eig_vectors[:, maxcol]

        points_a = points_array[:-1]
        points_b = points_array[1:]
        ba = points_b - points_a
        proj = np.dot(ba, dom_eig_vect) / np.power(np.linalg.norm(dom_eig_vect), 2)

        skewness = stats.skew(proj) if len(proj) > 0 else 0
        kurtosis = stats.kurtosis(proj) if len(proj) > 0 else 0

        return radius_gyration, asymmetry, skewness, kurtosis

    @staticmethod
    def fractal_dimension(points_array):
        """Calculate fractal dimension"""
        if len(points_array) < 3:
            return 1.0

        try:
            total_path_length = np.sum(np.sqrt(np.sum((points_array[1:, :] - points_array[:-1, :]) ** 2, axis=1)))
            stepCount = len(points_array)

            if len(points_array) < 3:
                return 1.0

            candidates = points_array[spatial.ConvexHull(points_array).vertices]
            dist_mat = spatial.distance_matrix(candidates, candidates)
            largestDistance = np.max(dist_mat)

            if total_path_length > 0 and largestDistance > 0:
                fractal_dim = math.log(stepCount) / math.log(stepCount * largestDistance / total_path_length)
                return fractal_dim
            else:
                return 1.0
        except:
            return 1.0

    @staticmethod
    def net_displacement_efficiency(points_array):
        """Calculate net displacement and efficiency"""
        if len(points_array) < 2:
            return 0, 0

        net_displacement = np.linalg.norm(points_array[0] - points_array[-1])

        if len(points_array) < 3:
            return net_displacement, 1.0

        points_a = points_array[1:, :]
        points_b = points_array[:-1, :]
        dist_ab_SumSquared = sum((np.linalg.norm(points_a - points_b, axis=1)) ** 2)

        efficiency = (net_displacement ** 2) / ((len(points_array) - 1) * dist_ab_SumSquared) if dist_ab_SumSquared > 0 else 0
        return net_displacement, efficiency


class SPTBatchAnalysis(QWidget):
    """Main FLIKA plugin class for SPT batch analysis"""

    def __init__(self):
        super().__init__()
        self.parameters = SPTAnalysisParameters()
        self.setupUI()

    def setupUI(self):
        """Create the main user interface"""
        self.setWindowTitle("SPT Batch Analysis")
        self.setMinimumSize(600, 500)

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout(file_group)

        # Directory selection
        file_layout.addWidget(QLabel("Input Directory:"), 0, 0)
        self.dir_path_edit = QLabel("No directory selected")
        self.dir_path_edit.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        file_layout.addWidget(self.dir_path_edit, 0, 1)

        self.select_dir_btn = QPushButton("Select Directory")
        self.select_dir_btn.clicked.connect(self.select_directory)
        file_layout.addWidget(self.select_dir_btn, 0, 2)

        # File pattern
        file_layout.addWidget(QLabel("File Pattern:"), 1, 0)
        self.file_pattern_edit = QComboBox()
        self.file_pattern_edit.setEditable(True)
        self.file_pattern_edit.addItems(['**/*.tif', '**/*_bin*.tif', '**/*_crop*.tif'])
        file_layout.addWidget(self.file_pattern_edit, 1, 1)

        self.refresh_files_btn = QPushButton("Refresh")
        self.refresh_files_btn.clicked.connect(self.refresh_file_list)
        file_layout.addWidget(self.refresh_files_btn, 1, 2)

        main_layout.addWidget(file_group)

        # File list
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(100)
        main_layout.addWidget(self.file_list_widget)

        self.file_count_label = QLabel("0 files selected")
        main_layout.addWidget(self.file_count_label)

        # Parameters section
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.001, 1000.0)
        self.pixel_size_spin.setValue(self.parameters.pixel_size)
        self.pixel_size_spin.setSuffix(" nm")
        params_layout.addRow("Pixel Size:", self.pixel_size_spin)

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(self.parameters.max_gap_frames)
        params_layout.addRow("Max Gap Frames:", self.max_gap_spin)

        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(0.1, 100.0)
        self.max_dist_spin.setValue(self.parameters.max_link_distance)
        self.max_dist_spin.setSuffix(" pixels")
        params_layout.addRow("Max Link Distance:", self.max_dist_spin)

        self.min_segments_spin = QSpinBox()
        self.min_segments_spin.setRange(1, 1000)
        self.min_segments_spin.setValue(self.parameters.min_track_segments)
        params_layout.addRow("Min Track Segments:", self.min_segments_spin)

        main_layout.addWidget(params_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to start analysis")
        progress_layout.addWidget(self.status_label)

        main_layout.addWidget(progress_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        button_layout.addWidget(self.start_btn)

        self.save_params_btn = QPushButton("Save Parameters")
        self.save_params_btn.clicked.connect(self.save_parameters)
        button_layout.addWidget(self.save_params_btn)

        self.load_params_btn = QPushButton("Load Parameters")
        self.load_params_btn.clicked.connect(self.load_parameters)
        button_layout.addWidget(self.load_params_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        main_layout.addLayout(button_layout)

    def select_directory(self):
        """Select input directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if directory:
            self.dir_path_edit.setText(directory)
            self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list"""
        directory = self.dir_path_edit.text()
        if directory == "No directory selected":
            return

        pattern = self.file_pattern_edit.currentText()
        file_list = glob.glob(os.path.join(directory, pattern), recursive=True)

        self.file_list_widget.clear()
        for file_path in sorted(file_list):
            self.file_list_widget.addItem(os.path.basename(file_path))

        self.file_count_label.setText(f"{len(file_list)} files selected")
        self.file_paths = file_list  # Store full paths

    def update_parameters(self):
        """Update parameters from GUI"""
        self.parameters.pixel_size = self.pixel_size_spin.value()
        self.parameters.max_gap_frames = self.max_gap_spin.value()
        self.parameters.max_link_distance = self.max_dist_spin.value()
        self.parameters.min_track_segments = self.min_segments_spin.value()

    def start_analysis(self):
        """Start the batch analysis"""
        if not hasattr(self, 'file_paths') or not self.file_paths:
            g.alert("No files selected for analysis")
            return

        self.update_parameters()

        # Disable start button
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        try:
            total_files = len(self.file_paths)

            for i, file_path in enumerate(self.file_paths):
                self.status_label.setText(f"Processing {os.path.basename(file_path)}")
                QApplication.processEvents()  # Update GUI

                # Process single file
                success = self.process_file(file_path)

                if not success:
                    self.status_label.setText(f"Error processing {os.path.basename(file_path)}")
                    continue

                progress = int((i + 1) / total_files * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()  # Update GUI

            self.status_label.setText("Analysis completed successfully!")
            g.alert("Batch analysis completed!")

        except Exception as e:
            self.status_label.setText(f"Analysis failed: {str(e)}")
            g.alert(f"Analysis failed: {str(e)}")

        finally:
            self.start_btn.setEnabled(True)

    def process_file(self, file_path):
        """Process a single file"""
        try:
            # Load localization data
            data = self.load_localization_data(file_path)
            if data is None:
                return False

            # Link particles
            points = self.link_particles(data, file_path)
            if points is None or points.recursiveFailure:
                return False

            # Calculate features and save
            tracks_df = self.calculate_features(points)
            if tracks_df is None:
                return False

            # Save results
            output_path = file_path.replace('.tif', '_analysis_results.csv')
            tracks_df.to_csv(output_path, index=False)

            return True

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def load_localization_data(self, file_path):
        """Load localization data from CSV"""
        try:
            locs_file = file_path.replace('.tif', '_locsID.csv')
            if not os.path.exists(locs_file):
                locs_file = file_path.replace('.tif', '_locs.csv')

            if not os.path.exists(locs_file):
                print(f"No localization file found for {file_path}")
                return None

            df = pd.read_csv(locs_file)
            df['frame'] = df['frame'].astype(int) - 1  # Zero-based indexing
            df['x'] = df['x [nm]'] / self.parameters.pixel_size
            df['y'] = df['y [nm]'] / self.parameters.pixel_size

            return df[['frame', 'x', 'y']].to_numpy()

        except Exception as e:
            print(f"Error loading localization data: {e}")
            return None

    def link_particles(self, txy_pts, file_path):
        """Link particles across frames"""
        try:
            points = Points(txy_pts)
            points.link_pts(self.parameters.max_gap_frames, self.parameters.max_link_distance)

            # Load image for intensity extraction
            if os.path.exists(file_path):
                A = skio.imread(file_path, plugin='tifffile')
                points.getIntensities(A)

            return points

        except Exception as e:
            print(f"Error linking particles: {e}")
            return None

    def calculate_features(self, points):
        """Calculate track features"""
        try:
            # Convert tracks to DataFrame
            tracks_data = []

            for track_idx, track in enumerate(points.tracks):
                if len(track) < self.parameters.min_track_segments:
                    continue

                for pt_idx in track:
                    pt = points.txy_pts[pt_idx]
                    intensity = points.intensities[pt_idx] if pt_idx < len(points.intensities) else 0

                    tracks_data.append({
                        'track_number': track_idx,
                        'frame': int(pt[0]),
                        'x': pt[1],
                        'y': pt[2],
                        'intensity': intensity
                    })

            if not tracks_data:
                print("No valid tracks found")
                return None

            tracks_df = pd.DataFrame(tracks_data)
            tracks_df['n_segments'] = tracks_df.groupby('track_number')['track_number'].transform('count')

            # Calculate features for each track
            feature_data = []
            track_numbers = tracks_df['track_number'].unique()

            for track_num in track_numbers:
                track_data = tracks_df[tracks_df['track_number'] == track_num]
                points_array = track_data[['x', 'y']].to_numpy()

                try:
                    # Calculate features
                    rg, asymmetry, skewness, kurtosis = FeatureCalculator.radius_gyration_asymmetry(track_data)
                    fractal_dim = FeatureCalculator.fractal_dimension(points_array)
                    net_disp, efficiency = FeatureCalculator.net_displacement_efficiency(points_array)

                    feature_data.append({
                        'track_number': track_num,
                        'radius_gyration': rg,
                        'asymmetry': asymmetry,
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'fracDimension': fractal_dim,
                        'netDispl': net_disp,
                        'efficiency': efficiency
                    })

                except Exception as e:
                    print(f"Error calculating features for track {track_num}: {e}")
                    continue

            # Merge features
            if feature_data:
                features_df = pd.DataFrame(feature_data)
                tracks_df = tracks_df.merge(features_df, on='track_number', how='left')

            return tracks_df

        except Exception as e:
            print(f"Error calculating features: {e}")
            return None

    def save_parameters(self):
        """Save parameters to file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json)")
        if file_path:
            self.update_parameters()
            with open(file_path, 'w') as f:
                json.dump(self.parameters.to_dict(), f, indent=2)
            g.alert("Parameters saved successfully")

    def load_parameters(self):
        """Load parameters from file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    param_dict = json.load(f)
                self.parameters.from_dict(param_dict)
                self.update_gui_from_parameters()
                g.alert("Parameters loaded successfully")
            except Exception as e:
                g.alert(f"Error loading parameters: {e}")

    def update_gui_from_parameters(self):
        """Update GUI from parameters"""
        self.pixel_size_spin.setValue(self.parameters.pixel_size)
        self.max_gap_spin.setValue(self.parameters.max_gap_frames)
        self.max_dist_spin.setValue(self.parameters.max_link_distance)
        self.min_segments_spin.setValue(self.parameters.min_track_segments)


# Plugin instance management
spt_batch_analysis_instance = None

def launch_spt_analysis():
    """Launch the SPT batch analysis plugin"""
    global spt_batch_analysis_instance

    if spt_batch_analysis_instance is None or not spt_batch_analysis_instance.isVisible():
        spt_batch_analysis_instance = SPTBatchAnalysis()

    spt_batch_analysis_instance.show()
    spt_batch_analysis_instance.raise_()
    spt_batch_analysis_instance.activateWindow()

def launch_docs():
    """Launch documentation"""
    from qtpy.QtCore import QUrl
    from qtpy.QtGui import QDesktopServices
    url = 'https://github.com/flika-org/flika_plugin_template'
    QDesktopServices.openUrl(QUrl(url))
