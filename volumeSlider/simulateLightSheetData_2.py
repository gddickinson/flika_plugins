import sys
import numpy as np
import logging
import tifffile
import os
from scipy.ndimage import rotate
from typing import Tuple, List, Optional, Any

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QFileDialog, QMessageBox, QCheckBox, QDockWidget, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QDialog, QGridLayout, QTabWidget, QTextEdit)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QVector3D, QImage
import traceback

from matplotlib import pyplot as plt
from skimage.feature import blob_log

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.spatial import cKDTree
from scipy.ndimage import center_of_mass
from PyQt5.QtCore import pyqtSignal


#from data_generator import DataGenerator
#from volume_processor import VolumeProcessor

class VolumeProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def apply_threshold(self, data, threshold):
        try:
            return np.where(data > threshold, data, 0)
        except Exception as e:
            self.logger.error(f"Error in applying threshold: {str(e)}")
            return data

    def apply_gaussian_filter(self, data, sigma):
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(data, sigma)
        except ImportError:
            self.logger.error("SciPy not installed. Cannot apply Gaussian filter.")
            return data
        except Exception as e:
            self.logger.error(f"Error in applying Gaussian filter: {str(e)}")
            return data

    def calculate_statistics(self, data):
        try:
            stats = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
            self.logger.info(f"Statistics calculated: {stats}")
            return stats
        except Exception as e:
            self.logger.error(f"Error in calculating statistics: {str(e)}")
            return {}


class DataGenerator:
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.initLogging()

    def initLogging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def generate_multi_channel_volume(self, size=(100, 100, 30), num_channels=2, num_blobs=30,
                                      intensity_range=(0.5, 1.0), sigma_range=(2, 6), noise_level=0.02):
        volume = np.zeros((num_channels, *size))
        for c in range(num_channels):
            for _ in range(num_blobs):
                x, y, z = [np.random.randint(0, s) for s in size]
                sigma = np.random.uniform(*sigma_range)
                intensity = np.random.uniform(*intensity_range)

                x_grid, y_grid, z_grid = np.ogrid[-x:size[0]-x, -y:size[1]-y, -z:size[2]-z]
                blob = np.exp(-(x_grid*x_grid + y_grid*y_grid + z_grid*z_grid) / (2*sigma*sigma))
                volume[c] += intensity * blob

            volume[c] += np.random.normal(0, noise_level, size)
            volume[c] = np.clip(volume[c], 0, 1)
        return volume

    def generate_multi_channel_time_series(self, num_volumes, num_channels=2, size=(30, 100, 100),
                                           num_blobs=30, intensity_range=(0.5, 1.0), sigma_range=(2, 6),
                                           noise_level=0.02, movement_speed=1.0):
        z, y, x = size
        time_series = np.zeros((num_volumes, num_channels, z, y, x))
        blob_positions = np.random.rand(num_channels, num_blobs, 3) * np.array([z, y, x])
        blob_velocities = np.random.randn(num_channels, num_blobs, 3) * movement_speed

        for t in range(num_volumes):
            for c in range(num_channels):
                volume = np.zeros((z, y, x))
                for i in range(num_blobs):
                    bz, by, bx = blob_positions[c, i]
                    sigma = np.random.uniform(*sigma_range)
                    intensity = np.random.uniform(*intensity_range)

                    zz, yy, xx = np.ogrid[-bz:z-bz, -by:y-by, -bx:x-bx]
                    blob = np.exp(-(zz*zz + yy*yy + xx*xx) / (2*sigma*sigma))
                    volume += intensity * blob

                    # Update blob position
                    blob_positions[c, i] += blob_velocities[c, i]
                    blob_positions[c, i] %= [z, y, x]  # Wrap around the volume

                volume += np.random.normal(0, noise_level, (z, y, x))
                time_series[t, c] = np.clip(volume, 0, 1)

        self.data = time_series

        self.logger.debug(f"Volume shape: {volume.shape}")
        self.logger.debug(f"Blob shape: {blob.shape}")
        self.logger.debug(f"Time series shape: {time_series.shape}")

        print(f"Volume shape: {volume.shape}")
        print(f"Blob shape: {blob.shape}")
        print(f"Time series shape: {time_series.shape}")

        return time_series


    def generate_volume(self, size: Tuple[int, int, int] = (100, 100, 30),
                        num_blobs: int = 30,
                        intensity_range: Tuple[float, float] = (0.8, 1.0),
                        sigma_range: Tuple[float, float] = (2, 6),
                        noise_level: float = 0.02) -> np.ndarray:
        """Generate a single volume of data."""
        volume = np.zeros(size)
        for _ in range(num_blobs):
            x, y, z = [np.random.randint(0, s) for s in size]
            sigma = np.random.uniform(*sigma_range)
            intensity = np.random.uniform(*intensity_range)

            x_grid, y_grid, z_grid = np.ogrid[-x:size[0]-x, -y:size[1]-y, -z:size[2]-z]
            blob = np.exp(-(x_grid*x_grid + y_grid*y_grid + z_grid*z_grid) / (2*sigma*sigma))
            volume += intensity * blob

        volume += np.random.normal(0, noise_level, size)
        volume = np.clip(volume, 0, 1)
        return volume

    def generate_time_series(self, num_volumes: int,
                             size: Tuple[int, int, int] = (100, 100, 30),
                             num_blobs: int = 30,
                             intensity_range: Tuple[float, float] = (0.8, 1.0),
                             sigma_range: Tuple[float, float] = (2, 6),
                             noise_level: float = 0.02,
                             movement_speed: float = 1.0) -> np.ndarray:
        """Generate a time series of volumes with moving blobs."""
        time_series = np.zeros((num_volumes, *size))
        blob_positions = np.random.rand(num_blobs, 3) * np.array(size)
        blob_velocities = np.random.randn(num_blobs, 3) * movement_speed

        for t in range(num_volumes):
            volume = np.zeros(size)
            for i in range(num_blobs):
                x, y, z = blob_positions[i]
                sigma = np.random.uniform(*sigma_range)
                intensity = np.random.uniform(*intensity_range)

                x_grid, y_grid, z_grid = np.ogrid[-x:size[0]-x, -y:size[1]-y, -z:size[2]-z]
                blob = np.exp(-(x_grid*x_grid + y_grid*y_grid + z_grid*z_grid) / (2*sigma*sigma))
                volume += intensity * blob

                # Update blob position
                blob_positions[i] += blob_velocities[i]
                blob_positions[i] %= size  # Wrap around the volume

            volume += np.random.normal(0, noise_level, size)
            time_series[t] = np.clip(volume, 0, 1)

        self.data = time_series
        return time_series



    def generate_structured_multi_channel_time_series(self, num_volumes, num_channels=2, size=(30, 100, 100),
                                                      num_blobs=30, intensity_range=(0.5, 1.0), sigma_range=(2, 6),
                                                      noise_level=0.02, movement_speed=1.0, channel_ranges=None):
        z, y, x = size
        time_series = np.zeros((num_volumes, num_channels, z, y, x))

        if channel_ranges is None:
            channel_ranges = [((0, x), (0, y), (0, z)) for _ in range(num_channels)]

        blob_positions = []
        blob_velocities = []

        for c in range(num_channels):
            x_range, y_range, z_range = channel_ranges[c]
            channel_blobs = np.random.rand(num_blobs, 3)
            channel_blobs[:, 0] = channel_blobs[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
            channel_blobs[:, 1] = channel_blobs[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
            channel_blobs[:, 2] = channel_blobs[:, 2] * (z_range[1] - z_range[0]) + z_range[0]
            blob_positions.append(channel_blobs)
            blob_velocities.append(np.random.randn(num_blobs, 3) * movement_speed)

        for t in range(num_volumes):
            for c in range(num_channels):
                volume = np.zeros((z, y, x))
                x_range, y_range, z_range = channel_ranges[c]
                for i in range(num_blobs):
                    bx, by, bz = blob_positions[c][i]
                    sigma = np.random.uniform(*sigma_range)
                    intensity = np.random.uniform(*intensity_range)

                    xx, yy, zz = np.ogrid[max(0, int(bx-3*sigma)):min(x, int(bx+3*sigma)),
                                          max(0, int(by-3*sigma)):min(y, int(by+3*sigma)),
                                          max(0, int(bz-3*sigma)):min(z, int(bz+3*sigma))]
                    blob = np.exp(-((xx-bx)**2 + (yy-by)**2 + (zz-bz)**2) / (2*sigma*sigma))
                    volume[zz, yy, xx] += intensity * blob

                    # Update blob position
                    blob_positions[c][i] += blob_velocities[c][i]
                    blob_positions[c][i][0] = np.clip(blob_positions[c][i][0], x_range[0], x_range[1])
                    blob_positions[c][i][1] = np.clip(blob_positions[c][i][1], y_range[0], y_range[1])
                    blob_positions[c][i][2] = np.clip(blob_positions[c][i][2], z_range[0], z_range[1])

                volume += np.random.normal(0, noise_level, (z, y, x))
                time_series[t, c] = np.clip(volume, 0, 1)

        self.data = time_series
        return time_series

    def simulate_angular_recording(self, angle: float) -> np.ndarray:
        """Simulate an angular recording by rotating the volume."""
        if self.data is None:
            raise ValueError("No data to rotate. Generate data first.")
        rotated_data = rotate(self.data, angle, axes=(1, 2), reshape=False, mode='constant', cval=0)
        return rotated_data

    def correct_angular_recording(self, angle: float) -> np.ndarray:
        """Correct an angular recording by rotating the volume back."""
        if self.data is None:
            raise ValueError("No data to correct. Generate data first.")
        corrected_data = rotate(self.data, -angle, axes=(1, 2), reshape=False, mode='constant', cval=0)
        return corrected_data

    def save_tiff(self, filename: str):
        """Save the data as a TIFF stack."""
        if self.data is None:
            raise ValueError("No data to save. Generate data first.")
        tifffile.imwrite(filename, self.data)

    def save_numpy(self, filename: str):
        """Save the data as a numpy array."""
        if self.data is None:
            raise ValueError("No data to save. Generate data first.")
        np.save(filename, self.data)

    def load_tiff(self, filename: str):
        """Load data from a TIFF stack."""
        self.data = tifffile.imread(filename)
        return self.data

    def load_numpy(self, filename: str):
        """Load data from a numpy array file."""
        self.data = np.load(filename)
        return self.data

    def apply_gaussian_filter(self, sigma: float):
        """Apply a Gaussian filter to the data."""
        from scipy.ndimage import gaussian_filter
        if self.data is None:
            raise ValueError("No data to filter. Generate or load data first.")
        self.data = gaussian_filter(self.data, sigma)
        return self.data

    def adjust_intensity(self, gamma: float):
        """Adjust the intensity of the data using gamma correction."""
        if self.data is None:
            raise ValueError("No data to adjust. Generate or load data first.")
        self.data = np.power(self.data, gamma)
        return self.data

    def get_metadata(self) -> dict:
        """Return metadata about the current dataset."""
        if self.data is None:
            return {}
        self.metadata.update({
            'shape': self.data.shape,
            'dtype': str(self.data.dtype),
            'min': float(np.min(self.data)),
            'max': float(np.max(self.data)),
            'mean': float(np.mean(self.data)),
            'std': float(np.std(self.data))
        })
        return self.metadata

    def set_metadata(self, key: str, value: Any):
        """Set a metadata value."""
        self.metadata[key] = value

class BlobAnalyzer:
    def __init__(self, blobs):
        self.blobs = blobs
        self.channels = np.unique(blobs[:, 4]).astype(int)
        self.time_points = np.unique(blobs[:, 5]).astype(int)

    def calculate_nearest_neighbor_distances(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]

        # All blobs
        all_distances = self._calculate_nn_distances(time_blobs[:, :3])

        # Within channels
        within_channel_distances = {ch: self._calculate_nn_distances(time_blobs[time_blobs[:, 4] == ch][:, :3])
                                    for ch in self.channels}

        # Between channels
        between_channel_distances = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1][:, :3]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2][:, :3]
                    between_channel_distances[(ch1, ch2)] = self._calculate_cross_distances(blobs1, blobs2)

        return all_distances, within_channel_distances, between_channel_distances

    def _calculate_nn_distances(self, points):
        if len(points) < 2:
            return np.array([])
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)
        return distances[:, 1]  # Exclude self-distance

    def _calculate_cross_distances(self, points1, points2):
        if len(points1) == 0 or len(points2) == 0:
            return np.array([])
        tree = cKDTree(points2)
        distances, _ = tree.query(points1)
        return distances

    def calculate_blob_density(self, volume_size, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        total_volume = np.prod(volume_size)
        channel_densities = {ch: np.sum(time_blobs[:, 4] == ch) / total_volume for ch in self.channels}
        overall_density = len(time_blobs) / total_volume
        return overall_density, channel_densities

    def calculate_colocalization(self, distance_threshold, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        colocalization = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1][:, :3]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2][:, :3]
                    if len(blobs1) > 0 and len(blobs2) > 0:
                        tree = cKDTree(blobs2)
                        distances, _ = tree.query(blobs1)
                        colocalization[(ch1, ch2)] = np.mean(distances < distance_threshold)
                    else:
                        colocalization[(ch1, ch2)] = 0
        return colocalization

    def calculate_blob_sizes(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        return {ch: time_blobs[time_blobs[:, 4] == ch][:, 3] for ch in self.channels}

    def calculate_blob_intensities(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        return {ch: time_blobs[time_blobs[:, 4] == ch][:, 6] for ch in self.channels}  # Column 6 is now intensity

    def calculate_advanced_colocalization(self, time_point):
        time_blobs = self.blobs[self.blobs[:, 5] == time_point]
        results = {}
        for ch1 in self.channels:
            for ch2 in self.channels:
                if ch1 < ch2:
                    blobs1 = time_blobs[time_blobs[:, 4] == ch1]
                    blobs2 = time_blobs[time_blobs[:, 4] == ch2]
                    if len(blobs1) > 0 and len(blobs2) > 0:
                        # Use KDTree for efficient nearest neighbor search
                        tree1 = cKDTree(blobs1[:, :3])  # x, y, z coordinates
                        tree2 = cKDTree(blobs2[:, :3])

                        # Calculate distance threshold based on mean blob size
                        distance_threshold = np.mean(np.concatenate([blobs1[:, 3], blobs2[:, 3]]))

                        # Find nearest neighbors
                        distances, indices = tree1.query(blobs2[:, :3])

                        # Calculate intensity correlations for nearby blobs
                        nearby_mask = distances < distance_threshold
                        intensities1 = blobs1[indices[nearby_mask], 6]
                        intensities2 = blobs2[nearby_mask, 6]

                        if len(intensities1) > 1 and len(intensities2) > 1:
                            pearson = np.corrcoef(intensities1, intensities2)[0, 1]
                        else:
                            pearson = np.nan

                        # Calculate Manders' coefficients
                        m1 = np.sum(intensities1) / np.sum(blobs1[:, 6])
                        m2 = np.sum(intensities2) / np.sum(blobs2[:, 6])

                        results[(ch1, ch2)] = {'pearson': pearson, 'manders_m1': m1, 'manders_m2': m2}
                    else:
                        results[(ch1, ch2)] = {'pearson': np.nan, 'manders_m1': np.nan, 'manders_m2': np.nan}
        return results

class BlobAnalysisDialog(QDialog):
    def __init__(self, blob_analyzer, parent=None):
        super().__init__(parent)
        self.blob_analyzer = blob_analyzer
        self.setWindowTitle("Blob Analysis Results")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Add time point selection
        self.timeComboBox = QComboBox()
        self.timeComboBox.addItems([str(t) for t in self.blob_analyzer.time_points])
        self.timeComboBox.currentIndexChanged.connect(self.updateAnalysis)
        layout.addWidget(self.timeComboBox)

        self.tabWidget = QTabWidget()
        layout.addWidget(self.tabWidget)

        # Add a close button
        closeButton = QPushButton("Close")
        closeButton.clicked.connect(self.close)
        layout.addWidget(closeButton)

        self.setLayout(layout)

        self.updateAnalysis()

    def updateAnalysis(self):
        self.tabWidget.clear()
        time_point = int(self.timeComboBox.currentText())

        self.addDistanceAnalysisTab(time_point)
        self.addDensityAnalysisTab(time_point)
        self.addColocalizationTab(time_point)
        self.addBlobSizeTab(time_point)
        self.addIntensityAnalysisTab(time_point)
        self.add3DVisualizationTab(time_point)
        self.addStatsTab(time_point)

    def addDistanceAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        all_distances, within_channel_distances, between_channel_distances = self.blob_analyzer.calculate_nearest_neighbor_distances(time_point)

        plt.figure(figsize=(10, 6))
        plt.hist(all_distances, bins=50, alpha=0.5, label='All Blobs')
        for ch, distances in within_channel_distances.items():
            plt.hist(distances, bins=50, alpha=0.5, label=f'Channel {ch}')
        plt.xlabel('Nearest Neighbor Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Nearest Neighbor Distance Distribution (Time: {time_point})')
        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Distance Analysis")

    def addDensityAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        overall_density, channel_densities = self.blob_analyzer.calculate_blob_density((30, 100, 100), time_point)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append(f"Time Point: {time_point}")
        textEdit.append(f"Overall Blob Density: {overall_density:.6f} blobs/unit^3")
        for ch, density in channel_densities.items():
            textEdit.append(f"Channel {ch} Density: {density:.6f} blobs/unit^3")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Density Analysis")

    def addBlobSizeTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        blob_sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        plt.figure(figsize=(10, 6))
        for ch, sizes in blob_sizes.items():
            plt.hist(sizes, bins=50, alpha=0.5, label=f'Channel {ch}')
        plt.xlabel('Blob Size')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Blob Size Distribution (Time: {time_point})')
        canvas = FigureCanvas(plt.gcf())
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Blob Size Analysis")

    def addStatsTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)

        all_distances, within_channel_distances, _ = self.blob_analyzer.calculate_nearest_neighbor_distances(time_point)
        overall_density, channel_densities = self.blob_analyzer.calculate_blob_density((30, 100, 100), time_point)
        blob_sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        textEdit.append(f"Statistics for Time Point: {time_point}")
        textEdit.append("\nOverall Statistics:")
        time_blobs = self.blob_analyzer.blobs[self.blob_analyzer.blobs[:, 5] == time_point]
        textEdit.append(f"Total number of blobs: {len(time_blobs)}")
        textEdit.append(f"Overall blob density: {overall_density:.6f} blobs/unit^3")
        if len(all_distances) > 0:
            textEdit.append(f"Mean nearest neighbor distance: {np.mean(all_distances):.2f}")
            textEdit.append(f"Median nearest neighbor distance: {np.median(all_distances):.2f}")
        else:
            textEdit.append("Not enough blobs to calculate nearest neighbor distances.")

        for ch in self.blob_analyzer.channels:
            textEdit.append(f"\nChannel {ch} Statistics:")
            channel_blobs = time_blobs[time_blobs[:, 4] == ch]
            textEdit.append(f"Number of blobs: {len(channel_blobs)}")
            textEdit.append(f"Blob density: {channel_densities[ch]:.6f} blobs/unit^3")
            if len(blob_sizes[ch]) > 0:
                textEdit.append(f"Mean blob size: {np.mean(blob_sizes[ch]):.2f}")
                textEdit.append(f"Median blob size: {np.median(blob_sizes[ch]):.2f}")
            else:
                textEdit.append("No blobs detected in this channel.")
            if ch in within_channel_distances and len(within_channel_distances[ch]) > 0:
                textEdit.append(f"Mean nearest neighbor distance: {np.mean(within_channel_distances[ch]):.2f}")
                textEdit.append(f"Median nearest neighbor distance: {np.median(within_channel_distances[ch]):.2f}")
            else:
                textEdit.append("Not enough blobs to calculate nearest neighbor distances.")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Statistics")

    def addIntensityAnalysisTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        intensities = self.blob_analyzer.calculate_blob_intensities(time_point)
        sizes = self.blob_analyzer.calculate_blob_sizes(time_point)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for ch in self.blob_analyzer.channels:
            ax1.hist(intensities[ch], bins=50, alpha=0.5, label=f'Channel {ch}')
        ax1.set_xlabel('Intensity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Blob Intensity Distribution')
        ax1.legend()

        for ch in self.blob_analyzer.channels:
            ax2.scatter(sizes[ch], intensities[ch], alpha=0.5, label=f'Channel {ch}')
        ax2.set_xlabel('Blob Size')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Blob Size vs Intensity')
        ax2.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Intensity Analysis")

    def add3DVisualizationTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        time_blobs = self.blob_analyzer.blobs[self.blob_analyzer.blobs[:, 5] == time_point]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        for ch in self.blob_analyzer.channels:
            channel_blobs = time_blobs[time_blobs[:, 4] == ch]
            ax.scatter(channel_blobs[:, 0], channel_blobs[:, 1], channel_blobs[:, 2],
                       s=channel_blobs[:, 3]*10, alpha=0.5, label=f'Channel {ch}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Blob Positions (Time: {time_point})')
        ax.legend()

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "3D Visualization")

    def addColocalizationTab(self, time_point):
        tab = QWidget()
        layout = QVBoxLayout()

        basic_coloc = self.blob_analyzer.calculate_colocalization(distance_threshold=5, time_point=time_point)
        advanced_coloc = self.blob_analyzer.calculate_advanced_colocalization(time_point)

        textEdit = QTextEdit()
        textEdit.setReadOnly(True)
        textEdit.append(f"Time Point: {time_point}")
        textEdit.append("\nBasic Colocalization:")
        for (ch1, ch2), coloc in basic_coloc.items():
            textEdit.append(f"Channels {ch1} and {ch2}: {coloc:.2%}")

        textEdit.append("\nAdvanced Colocalization:")
        for (ch1, ch2), results in advanced_coloc.items():
            textEdit.append(f"Channels {ch1} and {ch2}:")
            pearson = results['pearson']
            textEdit.append(f"  Pearson's coefficient: {pearson:.4f}" if not np.isnan(pearson) else "  Pearson's coefficient: N/A")
            textEdit.append(f"  Manders' M1: {results['manders_m1']:.4f}")
            textEdit.append(f"  Manders' M2: {results['manders_m2']:.4f}")

        layout.addWidget(textEdit)
        tab.setLayout(layout)
        self.tabWidget.addTab(tab, "Colocalization")



class BlobResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blob Detection Results")
        self.layout = QVBoxLayout(self)
        self.table = QTableWidget()
        self.layout.addWidget(self.table)

    def update_results(self, blobs):
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["X", "Y", "Z", "Size", "Channel", "Time"])
        self.table.setRowCount(len(blobs))

        for i, blob in enumerate(blobs):
            y, x, z, r, channel, t = blob
            self.table.setItem(i, 0, QTableWidgetItem(f"{x:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{y:.2f}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{z:.2f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{r:.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{int(channel)}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{int(t)}"))

class ROI3D(gl.GLMeshItem):
    sigRegionChanged = pyqtSignal(object)

    def __init__(self, size=(10, 10, 10), color=(1, 1, 1, 0.3)):
        verts, faces = self.create_cube(size)
        super().__init__(vertexes=verts, faces=faces, smooth=False, drawEdges=True, edgeColor=color)
        self.size = size
        self.setColor(color)

    @staticmethod
    def create_cube(size):
        x, y, z = size
        verts = np.array([
            [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
            [0, 0, z], [x, 0, z], [x, y, z], [0, y, z]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 4, 5],
            [1, 2, 5], [2, 5, 6], [2, 3, 6], [3, 6, 7],
            [3, 0, 7], [0, 4, 7], [4, 5, 6], [4, 6, 7]
        ])
        return verts, faces

    def setPosition(self, pos):
        self.resetTransform()
        self.translate(*pos)
        self.sigRegionChanged.emit(self)

class TimeSeriesDialog(QDialog):
    def __init__(self, blob_analyzer, parent=None):
        super().__init__(parent)
        self.blob_analyzer = blob_analyzer
        self.setWindowTitle("Time Series Analysis")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        plot_widget = pg.PlotWidget()
        layout.addWidget(plot_widget)

        time_points = np.unique(self.blob_analyzer.blobs[:, 5])
        channels = np.unique(self.blob_analyzer.blobs[:, 4])

        for channel in channels:
            blob_counts = [np.sum((self.blob_analyzer.blobs[:, 5] == t) & (self.blob_analyzer.blobs[:, 4] == channel))
                           for t in time_points]
            plot_widget.plot(time_points, blob_counts, pen=(int(channel), len(channels)), name=f'Channel {int(channel)}')

        plot_widget.setLabel('left', "Number of Blobs")
        plot_widget.setLabel('bottom', "Time Point")
        plot_widget.addLegend()

        self.setLayout(layout)

class LightsheetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initLogging()
        self.volume_processor = VolumeProcessor()
        self.data = None
        self.data_generator = DataGenerator()

        self.channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # RGB for up to 3 channels
        self.initUI()

        self.generateData()

        self.playbackTimer = QTimer(self)
        self.playbackTimer.timeout.connect(self.advanceTimePoint)
        self.blob_results_dialog = BlobResultsDialog(self)

        self.showBlobResultsButton.setVisible(False)

    def initLogging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def initUI(self):
        self.setWindowTitle('Lightsheet Microscopy Viewer')
        self.setGeometry(100, 100, 1600, 900)

        # Remove central widget
        self.setCentralWidget(None)

        # Create docks

        self.createViewDocks()
        self.createDataGenerationDock()
        self.create3DVisualizationDock()
        self.createVisualizationControlDock()
        self.createPlaybackControlDock()

        # blob visualization dock
        self.createBlobVisualizationDock()

        # Create menu bar
        self.createMenuBar()

        # Organize docks
        self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)

        # Stack XY, XZ, and YZ views vertically on the left
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXY)
        self.splitDockWidget(self.dockXY, self.dockXZ, Qt.Vertical)
        self.splitDockWidget(self.dockXZ, self.dockYZ, Qt.Vertical)

        # Add 3D view to the right of the 2D views
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock3D)

        # Add control docks to the far right
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockDataGeneration)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockVisualizationControl)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockPlaybackControl)

        # Set the 3D view and control docks side by side
        self.splitDockWidget(self.dock3D, self.dockDataGeneration, Qt.Horizontal)
        self.tabifyDockWidget(self.dockDataGeneration, self.dockVisualizationControl)
        self.tabifyDockWidget(self.dockVisualizationControl, self.dockPlaybackControl)

        # Add blob visualization dock below the 3D view
        self.splitDockWidget(self.dock3D, self.dockBlobVisualization, Qt.Vertical)


        # Adjust dock sizes
        self.resizeDocks([self.dockXY, self.dockXZ, self.dockYZ], [200, 200, 200], Qt.Vertical)
        self.resizeDocks([self.dock3D, self.dockDataGeneration], [800, 300], Qt.Horizontal)

    def display_blob_results(self, blobs):
        self.blob_results_dialog.update_results(blobs)
        self.blob_results_dialog.show()

    def createViewDocks(self):
        # XY View
        self.dockXY = QDockWidget("XY View", self)
        self.imageViewXY = pg.ImageView()
        self.imageViewXY.ui.roiBtn.hide()
        self.imageViewXY.ui.menuBtn.hide()
        self.imageViewXY.setPredefinedGradient('viridis')
        self.imageViewXY.timeLine.sigPositionChanged.connect(self.updateMarkersFromSliders)
        self.dockXY.setWidget(self.imageViewXY)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXY)

        # XZ View
        self.dockXZ = QDockWidget("XZ View", self)
        self.imageViewXZ = pg.ImageView()
        self.imageViewXZ.ui.roiBtn.hide()
        self.imageViewXZ.ui.menuBtn.hide()
        self.imageViewXZ.setPredefinedGradient('viridis')
        self.imageViewXZ.timeLine.sigPositionChanged.connect(self.updateMarkersFromSliders)
        self.dockXZ.setWidget(self.imageViewXZ)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXZ)

        # YZ View
        self.dockYZ = QDockWidget("YZ View", self)
        self.imageViewYZ = pg.ImageView()
        self.imageViewYZ.ui.roiBtn.hide()
        self.imageViewYZ.ui.menuBtn.hide()
        self.imageViewYZ.setPredefinedGradient('viridis')
        self.imageViewYZ.timeLine.sigPositionChanged.connect(self.updateMarkersFromSliders)
        self.dockYZ.setWidget(self.imageViewYZ)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockYZ)


        # Set size policies for the image views
        for view in [self.imageViewXY, self.imageViewXZ, self.imageViewYZ]:
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.setMinimumSize(200, 200)

    def updateMarkersFromSliders(self):
        self.updateSliceMarkers()
        self.create3DVisualization()  # This might be heavy, consider optimizing if performance is an issue

    def updateSliceMarkers(self):
        if not hasattr(self, 'data') or self.data is None:
            return

        # Remove old slice marker items
        for item in self.slice_marker_items:
            try:
                self.blobGLView.removeItem(item)
            except:
                pass

        for item in self.main_slice_marker_items:
            try:
                self.glView.removeItem(item)
            except:
                pass

        self.slice_marker_items.clear()
        self.main_slice_marker_items.clear()


        if self.showSliceMarkersCheck.isChecked():
            _, _, depth, height, width = self.data.shape  # Assuming shape is (t, c, z, y, x)

            z_slice = int(self.imageViewXY.currentIndex)
            y_slice = int(self.imageViewXZ.currentIndex)
            x_slice = int(self.imageViewYZ.currentIndex)

            # Create new markers
            x_marker = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, height, 0], [x_slice, height, depth], [x_slice, 0, depth]]),
                                              color=(1, 0, 0, 1), width=2, mode='line_strip')
            y_marker = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [width, y_slice, 0], [width, y_slice, depth], [0, y_slice, depth]]),
                                              color=(0, 1, 0, 1), width=2, mode='line_strip')
            z_marker = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [width, 0, z_slice], [width, height, z_slice], [0, height, z_slice]]),
                                              color=(0, 0, 1, 1), width=2, mode='line_strip')

            # Add new markers
            self.glView.addItem(x_marker)
            self.glView.addItem(y_marker)
            self.glView.addItem(z_marker)


            # Create and add new markers for blob visualization view
            x_marker_vis = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, height, 0], [x_slice, height, depth], [x_slice, 0, depth]]),
                                             color=(1, 0, 0, 1), width=2, mode='line_strip')
            y_marker_vis = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [width, y_slice, 0], [width, y_slice, depth], [0, y_slice, depth]]),
                                             color=(0, 1, 0, 1), width=2, mode='line_strip')
            z_marker_vis = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [width, 0, z_slice], [width, height, z_slice], [0, height, z_slice]]),
                                             color=(0, 0, 1, 1), width=2, mode='line_strip')

            self.blobGLView.addItem(x_marker_vis)
            self.blobGLView.addItem(y_marker_vis)
            self.blobGLView.addItem(z_marker_vis)

            self.glView.update()
            self.blobGLView.update()

            self.slice_marker_items.extend([x_marker_vis, y_marker_vis, z_marker_vis])
            self.main_slice_marker_items.extend([x_marker, y_marker, z_marker])

            self.logger.debug(f"Slice positions - X: {x_slice}, Y: {y_slice}, Z: {z_slice}")

    def createDataGenerationDock(self):
        self.dockDataGeneration = QDockWidget("Data Generation", self)
        dataGenWidget = QWidget()
        layout = QVBoxLayout(dataGenWidget)

        layout.addWidget(QLabel("Number of Volumes:"))
        self.numVolumesSpinBox = QSpinBox()
        self.numVolumesSpinBox.setRange(1, 100)
        self.numVolumesSpinBox.setValue(10)
        layout.addWidget(self.numVolumesSpinBox)

        layout.addWidget(QLabel("Number of Blobs:"))
        self.numBlobsSpinBox = QSpinBox()
        self.numBlobsSpinBox.setRange(1, 100)
        self.numBlobsSpinBox.setValue(30)
        layout.addWidget(self.numBlobsSpinBox)

        layout.addWidget(QLabel("Noise Level:"))
        self.noiseLevelSpinBox = QDoubleSpinBox()
        self.noiseLevelSpinBox.setRange(0, 1)
        self.noiseLevelSpinBox.setSingleStep(0.01)
        self.noiseLevelSpinBox.setValue(0.02)
        layout.addWidget(self.noiseLevelSpinBox)

        layout.addWidget(QLabel("Movement Speed:"))
        self.movementSpeedSpinBox = QDoubleSpinBox()
        self.movementSpeedSpinBox.setRange(0, 10)
        self.movementSpeedSpinBox.setSingleStep(0.1)
        self.movementSpeedSpinBox.setValue(0.5)
        layout.addWidget(self.movementSpeedSpinBox)

        # Add checkbox for structured data
        self.structuredDataCheck = QCheckBox("Generate Structured Data")
        self.structuredDataCheck.setChecked(False)
        layout.addWidget(self.structuredDataCheck)

        # Add options for channel ranges
        self.channelRangeWidgets = []
        for i in range(2):  # Assuming 2 channels for now
            channelLayout = QGridLayout()
            channelLayout.addWidget(QLabel(f"Channel {i+1} Range:"), 0, 0, 1, 6)

            channelLayout.addWidget(QLabel("X min"), 1, 0)
            xMinSpin = QSpinBox()
            channelLayout.addWidget(xMinSpin, 1, 1)

            channelLayout.addWidget(QLabel("X max"), 1, 2)
            xMaxSpin = QSpinBox()
            channelLayout.addWidget(xMaxSpin, 1, 3)

            channelLayout.addWidget(QLabel("Y min"), 2, 0)
            yMinSpin = QSpinBox()
            channelLayout.addWidget(yMinSpin, 2, 1)

            channelLayout.addWidget(QLabel("Y max"), 2, 2)
            yMaxSpin = QSpinBox()
            channelLayout.addWidget(yMaxSpin, 2, 3)

            channelLayout.addWidget(QLabel("Z min"), 3, 0)
            zMinSpin = QSpinBox()
            channelLayout.addWidget(zMinSpin, 3, 1)

            channelLayout.addWidget(QLabel("Z max"), 3, 2)
            zMaxSpin = QSpinBox()
            channelLayout.addWidget(zMaxSpin, 3, 3)

            for spin in [xMinSpin, xMaxSpin, yMinSpin, yMaxSpin, zMinSpin, zMaxSpin]:
                spin.setRange(0, 100)

            self.channelRangeWidgets.append((xMinSpin, xMaxSpin, yMinSpin, yMaxSpin, zMinSpin, zMaxSpin))
            layout.addLayout(channelLayout)

        self.generateButton = QPushButton("Generate New Data")
        self.generateButton.clicked.connect(self.generateData)
        layout.addWidget(self.generateButton)

        self.saveButton = QPushButton("Save Data")
        self.saveButton.clicked.connect(self.saveData)
        layout.addWidget(self.saveButton)

        self.loadButton = QPushButton("Load Data")
        self.loadButton.clicked.connect(self.loadData)
        layout.addWidget(self.loadButton)

        layout.addStretch(1)  # This pushes everything up
        self.dockDataGeneration.setWidget(dataGenWidget)

    def createVisualizationControlDock(self):
        self.dockVisualizationControl = QDockWidget("Visualization Control", self)
        visControlWidget = QWidget()
        layout = QVBoxLayout(visControlWidget)

        # Channel controls
        self.channelControls = []
        for i in range(3):  # Assume max 3 channels for now
            channelLayout = QHBoxLayout()
            channelLayout.addWidget(QLabel(f"Channel {i+1}:"))
            visibilityCheck = QCheckBox("Visible")
            visibilityCheck.setChecked(True)
            visibilityCheck.stateChanged.connect(self.updateChannelVisibility)
            channelLayout.addWidget(visibilityCheck)
            opacitySlider = QSlider(Qt.Horizontal)
            opacitySlider.setRange(0, 100)
            opacitySlider.setValue(100)
            opacitySlider.valueChanged.connect(self.updateChannelOpacity)
            channelLayout.addWidget(opacitySlider)
            self.channelControls.append((visibilityCheck, opacitySlider))
            layout.addLayout(channelLayout)


        layout.addWidget(QLabel("Threshold:"))
        self.thresholdSpinBox = QDoubleSpinBox()
        self.thresholdSpinBox.setRange(0, 1)
        self.thresholdSpinBox.setSingleStep(0.05)
        self.thresholdSpinBox.setValue(0.5)
        self.thresholdSpinBox.valueChanged.connect(self.updateThreshold)
        layout.addWidget(self.thresholdSpinBox)

        layout.addWidget(QLabel("3D Rendering Mode:"))
        self.renderModeCombo = QComboBox()
        self.renderModeCombo.addItems(["Points", "Surface", "Wireframe"])
        self.renderModeCombo.currentTextChanged.connect(self.updateRenderMode)
        layout.addWidget(self.renderModeCombo)

        layout.addWidget(QLabel("Color Map:"))
        self.colorMapCombo = QComboBox()
        self.colorMapCombo.addItems(["Viridis", "Plasma", "Inferno", "Magma", "Grayscale"])
        self.colorMapCombo.currentTextChanged.connect(self.updateColorMap)
        layout.addWidget(self.colorMapCombo)

        self.showSliceMarkersCheck = QCheckBox("Show Slice Markers")
        self.showSliceMarkersCheck.stateChanged.connect(self.toggleSliceMarkers)
        layout.addWidget(self.showSliceMarkersCheck)

        layout.addWidget(QLabel("Clip Plane:"))
        self.clipSlider = QSlider(Qt.Horizontal)
        self.clipSlider.setMinimum(0)
        self.clipSlider.setMaximum(100)
        self.clipSlider.setValue(100)
        self.clipSlider.valueChanged.connect(self.updateClipPlane)
        layout.addWidget(self.clipSlider)

        # Add Blob Detection controls
        layout.addWidget(QLabel("Blob Detection:"))

        blobLayout = QGridLayout()

        blobLayout.addWidget(QLabel("Max Sigma:"), 0, 0)
        self.maxSigmaSpinBox = QDoubleSpinBox()
        self.maxSigmaSpinBox.setRange(1, 100)
        self.maxSigmaSpinBox.setValue(30)
        blobLayout.addWidget(self.maxSigmaSpinBox, 0, 1)

        blobLayout.addWidget(QLabel("Num Sigma:"), 1, 0)
        self.numSigmaSpinBox = QSpinBox()
        self.numSigmaSpinBox.setRange(1, 20)
        self.numSigmaSpinBox.setValue(10)
        blobLayout.addWidget(self.numSigmaSpinBox, 1, 1)

        blobLayout.addWidget(QLabel("Threshold:"), 2, 0)
        self.blobThresholdSpinBox = QDoubleSpinBox()
        self.blobThresholdSpinBox.setRange(0, 1)
        self.blobThresholdSpinBox.setSingleStep(0.01)
        self.blobThresholdSpinBox.setValue(0.1)
        blobLayout.addWidget(self.blobThresholdSpinBox, 2, 1)

        # Add checkbox for showing all blobs
        self.showAllBlobsCheck = QCheckBox("Show All Blobs")
        self.showAllBlobsCheck.setChecked(False)
        self.showAllBlobsCheck.stateChanged.connect(self.updateBlobVisualization)
        layout.addWidget(self.showAllBlobsCheck)

        layout.addLayout(blobLayout)

        # Add Blob Detection button
        self.blobDetectionButton = QPushButton("Detect Blobs")
        self.blobDetectionButton.clicked.connect(self.detect_blobs)
        layout.addWidget(self.blobDetectionButton)

        # Add button to show/hide blob results
        self.showBlobResultsButton = QPushButton("Show Blob Results")
        self.showBlobResultsButton.clicked.connect(self.toggleBlobResults)
        layout.addWidget(self.showBlobResultsButton)

        # Add Blob Analysis button
        self.blobAnalysisButton = QPushButton("Analyze Blobs")
        self.blobAnalysisButton.clicked.connect(self.analyzeBlobsasdkjfb )
        layout.addWidget(self.blobAnalysisButton)

        # Time Series Analysis button
        self.timeSeriesButton = QPushButton("Time Series Analysis")
        self.timeSeriesButton.clicked.connect(self.showTimeSeriesAnalysis)
        layout.addWidget(self.timeSeriesButton)

        layout.addStretch(1)  # This pushes everything up
        self.dockVisualizationControl.setWidget(visControlWidget)

    def createBlobVisualizationDock(self):
        self.dockBlobVisualization = QDockWidget("Blob Visualization", self)
        self.blobGLView = gl.GLViewWidget()
        self.dockBlobVisualization.setWidget(self.blobGLView)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dockBlobVisualization)

        # Add a grid to the view
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        self.blobGLView.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        self.blobGLView.addItem(gy)
        gz = gl.GLGridItem()
        self.blobGLView.addItem(gz)

        # Initialize empty lists to store blob and slice marker items
        self.blob_items = []
        self.slice_marker_items = []

    def create3DVisualizationDock(self):
        # 3D View
        self.dock3D = QDockWidget("3D View", self)
        self.glView = gl.GLViewWidget()
        self.dock3D.setWidget(self.glView)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock3D)

        # Add a grid to the view
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        self.glView.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        self.glView.addItem(gy)
        gz = gl.GLGridItem()
        self.glView.addItem(gz)

        # Initialize empty lists to store data and slice marker items
        self.data_items = []
        self.main_slice_marker_items = []


    def visualize_blobs(self, blobs):
        # Remove old blob visualizations
        for item in self.blob_items:
            self.blobGLView.removeItem(item)
        self.blob_items.clear()

        current_time = self.timeSlider.value()

        # Define colors for each channel (you can adjust these)
        channel_colors = [
            (1, 0, 0, 0.5),  # Red for channel 0
            (0, 1, 0, 0.5),  # Green for channel 1
            (0, 0, 1, 0.5),  # Blue for channel 2
            (1, 1, 0, 0.5),  # Yellow for channel 3 (if needed)
            (1, 0, 1, 0.5),  # Magenta for channel 4 (if needed)
        ]

        # Add new blob visualizations
        for blob in blobs:
            y, x, z, r, channel, t, intensity = blob
            mesh = gl.MeshData.sphere(rows=10, cols=20, radius=r)

            # Get color based on channel
            base_color = channel_colors[int(channel) % len(channel_colors)]

            # Adjust color based on whether it's a current or past blob
            if t == current_time:
                color = base_color
            else:
                color = tuple(c * 0.5 for c in base_color[:3]) + (base_color[3] * 0.5,)  # Dimmed color for past blobs

            ## Optionally, adjust color based on intensity
            #alpha = min(1.0, intensity / 255.0)  # Assuming intensity is in 0-255 range
            #color = (*base_color[:3], alpha)


            # Add to blob visualization view
            blob_item_vis = gl.GLMeshItem(meshdata=mesh, smooth=True, color=color, shader='shaded')
            blob_item_vis.translate(x, z, y)  # Swapped y and z
            self.blobGLView.addItem(blob_item_vis)
            self.blob_items.append(blob_item_vis)

        self.blobGLView.update()

    def advanceTimePoint(self):
        current_time = self.timeSlider.value()
        if current_time < self.timeSlider.maximum():
            self.timeSlider.setValue(current_time + 1)
        elif self.loopCheckBox.isChecked():
            self.timeSlider.setValue(0)
        else:
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")

    def updateTimePoint(self, value):
        if self.data is not None:
            self.currentTimePoint = value
            self.updateViews()
            self.create3DVisualization()
            self.updateBlobVisualization()
        else:
            self.logger.warning("No data available to update time point")

    def createPlaybackControlDock(self):
        self.dockPlaybackControl = QDockWidget("Playback Control", self)
        playbackControlWidget = QWidget()
        layout = QVBoxLayout(playbackControlWidget)

        layout.addWidget(QLabel("Time:"))
        self.timeSlider = QSlider(Qt.Horizontal)
        self.timeSlider.setMinimum(0)
        #self.timeSlider.setMaximum(self.data.shape[0] - 1)  # Assuming first dimension is time
        self.timeSlider.valueChanged.connect(self.updateTimePoint)
        layout.addWidget(self.timeSlider)

        playbackLayout = QHBoxLayout()
        self.playPauseButton = QPushButton("Play")
        self.playPauseButton.clicked.connect(self.togglePlayback)
        playbackLayout.addWidget(self.playPauseButton)

        self.speedLabel = QLabel("Speed:")
        playbackLayout.addWidget(self.speedLabel)

        self.speedSpinBox = QDoubleSpinBox()
        self.speedSpinBox.setRange(0.1, 10)
        self.speedSpinBox.setSingleStep(0.1)
        self.speedSpinBox.setValue(1)
        self.speedSpinBox.valueChanged.connect(self.updatePlaybackSpeed)
        playbackLayout.addWidget(self.speedSpinBox)

        self.loopCheckBox = QCheckBox("Loop")
        playbackLayout.addWidget(self.loopCheckBox)

        layout.addLayout(playbackLayout)

        layout.addStretch(1)  # This pushes everything up
        self.dockPlaybackControl.setWidget(playbackControlWidget)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Adjust dock sizes to maintain aspect ratio if needed
        width = self.width()
        left_width = width // 3
        right_width = width - left_width

        # Resize the 2D view docks
        self.resizeDocks([self.dockXY, self.dockXZ, self.dockYZ], [left_width] * 3, Qt.Horizontal)

        # Resize the 3D view and control docks
        control_width = right_width // 4  # Allocate 1/4 of right side to controls
        self.resizeDocks([self.dock3D], [right_width - control_width], Qt.Horizontal)
        self.resizeDocks([self.dockDataGeneration, self.dockVisualizationControl, self.dockPlaybackControl],
                         [control_width] * 3, Qt.Horizontal)

    def createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')

        loadAction = fileMenu.addAction('&Load Data')
        loadAction.triggered.connect(self.loadData)

        saveAction = fileMenu.addAction('&Save Data')
        saveAction.triggered.connect(self.saveData)

        quitAction = fileMenu.addAction('&Quit')
        quitAction.triggered.connect(self.close)

        viewMenu = menuBar.addMenu('&View')

        for dock in [self.dockXY, self.dockXZ, self.dockYZ, self.dock3D,
                     self.dockDataGeneration, self.dockVisualizationControl, self.dockPlaybackControl]:
            viewMenu.addAction(dock.toggleViewAction())


        analysisMenu = menuBar.addMenu('&Analysis')
        timeSeriesAction = analysisMenu.addAction('Time Series Analysis')
        timeSeriesAction.triggered.connect(self.showTimeSeriesAnalysis)


    def generateData(self):
        try:
            num_volumes = self.numVolumesSpinBox.value()
            num_blobs = self.numBlobsSpinBox.value()
            noise_level = self.noiseLevelSpinBox.value()
            movement_speed = self.movementSpeedSpinBox.value()
            num_channels = 2  # You can make this configurable if needed

            # Ensure consistent dimensions
            size = (30, 100, 100)  # (z, y, x)

            if self.structuredDataCheck.isChecked():
                channel_ranges = []
                for widgets in self.channelRangeWidgets:
                    xMin, xMax, yMin, yMax, zMin, zMax = [w.value() for w in widgets]
                    channel_ranges.append(((xMin, xMax), (yMin, yMax), (zMin, zMax)))

                # If we have fewer channel ranges than channels, add default ranges
                while len(channel_ranges) < num_channels:
                    channel_ranges.append(((0, size[2]), (0, size[1]), (0, size[0])))

                self.data = self.data_generator.generate_structured_multi_channel_time_series(
                    num_volumes=num_volumes,
                    num_channels=num_channels,
                    size=size,
                    num_blobs=num_blobs,
                    intensity_range=(0.8, 1.0),
                    sigma_range=(2, 6),
                    noise_level=noise_level,
                    movement_speed=movement_speed,
                    channel_ranges=channel_ranges
                )
            else:
                self.data = self.data_generator.generate_multi_channel_time_series(
                    num_volumes=num_volumes,
                    num_channels=num_channels,
                    size=size,
                    num_blobs=num_blobs,
                    intensity_range=(0.8, 1.0),
                    sigma_range=(2, 6),
                    noise_level=noise_level,
                    movement_speed=movement_speed
                )

            self.logger.info(f"Generated data shape: {self.data.shape}")
            self.logger.info(f"Data min: {self.data.min()}, max: {self.data.max()}, mean: {self.data.mean()}")
            self.logger.info(f"Generated  {num_blobs*num_channels*num_volumes} blobs")
            #self.visualize_data_distribution()  # Call this to visualize the data distribution
            self.updateUIForNewData()

            self.timeSlider.setMaximum(num_volumes - 1)
            self.updateViews()
            self.create3DVisualization()
            self.logger.info("Data generated and visualized successfully")
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error args: {e.args}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to generate data: {str(e)}")

    def updateViews(self):
        if self.data is None:
            self.logger.warning("No data to update views")
            return
        t = self.timeSlider.value()

        self.logger.debug(f"Updating views for time point {t}")
        self.logger.debug(f"Data shape: {self.data.shape}")

        # Assuming data shape is (t, c, z, y, x)
        num_channels, depth, height, width = self.data.shape[1:]

        # Prepare 3D RGB images for each view
        combined_xy = np.zeros((depth, height, width, 3))
        combined_xz = np.zeros((height, depth, width, 3))
        combined_yz = np.zeros((height, depth, width, 3))

        for c in range(min(num_channels, 3)):  # Limit to 3 channels for RGB
            if self.channelControls[c][0].isChecked():
                opacity = self.channelControls[c][1].value() / 100
                channel_data = self.data[t, c]

                self.logger.debug(f"Channel {c} data shape: {channel_data.shape}")

                # Normalize channel data
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)

                combined_xy[:, :, :, c] = channel_data * opacity
                combined_xz[:, :, :, c] = np.transpose(channel_data, (1, 0, 2)) * opacity
                combined_yz[:, :, :, c] = np.transpose(channel_data, (2, 0, 1)) * opacity

        self.logger.debug(f"Combined XY shape: {combined_xy.shape}")
        self.logger.debug(f"Combined XZ shape: {combined_xz.shape}")
        self.logger.debug(f"Combined YZ shape: {combined_yz.shape}")

        try:
            self.imageViewXY.setImage(combined_xy)
            self.imageViewXY.setCurrentIndex(depth // 2)

            self.imageViewXZ.setImage(combined_xz)
            self.imageViewXZ.setCurrentIndex(height // 2)

            self.imageViewYZ.setImage(combined_yz)
            self.imageViewYZ.setCurrentIndex(width // 2)

        except Exception as e:
            self.logger.error(f"Error setting images: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error(f"Error args: {e.args}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        self.updateSliceMarkers()

    def create3DVisualization(self):
        try:
            for item in (self.data_items):
                self.glView.removeItem(item)
        except:
            pass

        self.data_items.clear()

        try:
            #self.glView.clear()
            t = self.timeSlider.value()
            threshold = self.thresholdSpinBox.value()

            num_channels, depth, height, width = self.data.shape[1:]

            self.logger.debug(f"3D Visualization - Time: {t}, Threshold: {threshold}")
            self.logger.debug(f"Data shape: {self.data.shape}")

            total_points = 0
            for c in range(min(num_channels, 3)):  # Limit to 3 channels for RGB
                if self.channelControls[c][0].isChecked():
                    opacity = self.channelControls[c][1].value() / 100
                    volume_data = self.data[t, c]

                    # Get 3D coordinates of all points above threshold
                    z, y, x = np.where(volume_data > threshold)
                    pos = np.column_stack((x, y, z))

                    # debugging info
                    self.logger.debug(f"Channel {c} - Data shape: {volume_data.shape}")
                    self.logger.debug(f"Channel {c} - Data range: {volume_data.min():.3f} to {volume_data.max():.3f}")
                    self.logger.debug(f"Channel {c} - Threshold: {threshold}")
                    self.logger.debug(f"Channel {c} - Points above threshold: {len(pos)}")
                    self.logger.debug(f"Channel {c} - X range: {x.min()} to {x.max()}")
                    self.logger.debug(f"Channel {c} - Y range: {y.min()} to {y.max()}")
                    self.logger.debug(f"Channel {c} - Z range: {z.min()} to {z.max()}")


                    if len(pos) > 0:
                        colors = np.tile(self.channel_colors[c], (len(pos), 1))
                        colors[:, 3] = opacity * (volume_data[z, y, x] - volume_data.min()) / (volume_data.max() - volume_data.min())

                        scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=2)
                        self.glView.addItem(scatter)
                        self.data_items.append(scatter)

                        total_points += len(pos)

            self.glView.update()
            self.logger.debug(f"3D visualization created successfully. Total points plotted: {total_points}")
        except Exception as e:
            self.logger.error(f"Error in 3D visualization: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to create 3D visualization: {str(e)}")


    def visualize_data_distribution(self):
        t = self.timeSlider.value()
        num_channels, depth, height, width = self.data.shape[1:]

        for c in range(min(num_channels, 3)):
            volume_data = self.data[t, c]
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(np.max(volume_data, axis=0))
            plt.title(f'Channel {c} - XY Max Projection')
            plt.subplot(132)
            plt.imshow(np.max(volume_data, axis=1))
            plt.title(f'Channel {c} - XZ Max Projection')
            plt.subplot(133)
            plt.imshow(np.max(volume_data, axis=2))
            plt.title(f'Channel {c} - YZ Max Projection')
            plt.colorbar()
            plt.show()

        plt.figure()
        plt.hist(self.data[t].ravel(), bins=100)
        plt.title('Data Histogram')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

    def updateChannelVisibility(self):
        self.updateViews()
        self.create3DVisualization()

    def updateChannelOpacity(self):
        self.updateViews()
        self.create3DVisualization()

    def getColorMap(self):
        cmap_name = self.colorMapCombo.currentText().lower()
        if cmap_name == "grayscale":
            return pg.ColorMap(pos=[0.0, 1.0], color=[(0, 0, 0, 255), (255, 255, 255, 255)])
        else:
            return pg.colormap.get(cmap_name)

    def triangulate_points(self, points):
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices

    def updateRenderMode(self):
        self.create3DVisualization()

    def updateColorMap(self):
        self.create3DVisualization()

    def toggleSliceMarkers(self, state):
        if state == Qt.Checked:
            self.updateSliceMarkers()
        else:
            for attr in ['x_marker', 'y_marker', 'z_marker']:
                if hasattr(self, attr):
                    self.glView.removeItem(getattr(self, attr))
                    delattr(self, attr)
        self.glView.update()

    def updateClipPlane(self, value):
        try:
            clip_pos = (value / 100) * 30
            mask = self.scatter.pos[:, 2] <= clip_pos
            self.scatter.setData(pos=self.scatter.pos[mask],
                                 color=self.scatter.color[mask])
            self.logger.info(f"Clip plane updated to position {clip_pos}")
        except Exception as e:
            self.logger.error(f"Error updating clip plane: {str(e)}")


    def updateThreshold(self, value):
        self.create3DVisualization()

    def saveData(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "TIFF Files (*.tiff);;NumPy Files (*.npy)")
            if filename:
                if filename.endswith('.tiff'):
                    self.data_generator.save_tiff(filename)
                elif filename.endswith('.npy'):
                    self.data_generator.save_numpy(filename)
                self.logger.info(f"Data saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def loadData(self):
        try:
            filename, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "TIFF Files (*.tiff);;NumPy Files (*.npy)")
            if filename:
                if filename.endswith('.tiff'):
                    self.data = self.data_generator.load_tiff(filename)
                elif filename.endswith('.npy'):
                    self.data = self.data_generator.load_numpy(filename)
                self.timeSlider.setMaximum(self.data.shape[0] - 1)
                self.updateViews()
                self.create3DVisualization()
                self.logger.info(f"Data loaded from {filename}")

                # Stop playback when loading new data
                if self.playbackTimer.isActive():
                    self.playbackTimer.stop()
                    self.playPauseButton.setText("Play")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")

    def togglePlayback(self):
        if not hasattr(self, 'playbackTimer'):
            self.playbackTimer = QTimer(self)
            self.playbackTimer.timeout.connect(self.advanceTimePoint)

        if self.playbackTimer.isActive():
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")
        else:
            self.playbackTimer.start(int(1000 / self.speedSpinBox.value()))
            self.playPauseButton.setText("Pause")

    def updatePlaybackSpeed(self, value):
        if hasattr(self, 'playbackTimer') and self.playbackTimer.isActive():
            self.playbackTimer.setInterval(int(1000 / value))

    def updateUIForNewData(self):
        if self.data is not None:
            self.timeSlider.setMaximum(self.data.shape[0] - 1)
            self.updateViews()
            self.create3DVisualization()
        else:
            self.logger.warning("No data available to update UI")

    def closeEvent(self, event):
        # Stop the playback timer
        if hasattr(self, 'playbackTimer'):
            self.playbackTimer.stop()

        # Perform any other cleanup operations here
        # For example, you might want to save application settings

        # Log the application closure
        self.logger.info("Application closed")

        # Accept the event to allow the window to close
        event.accept()

        # Call the base class implementation
        super().closeEvent(event)

    def safeClose(self):
        self.close()  # This will trigger the closeEvent

    def exportProcessedData(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Processed Data", "", "TIFF Files (*.tiff);;NumPy Files (*.npy)")
        if filename:
            if filename.endswith('.tiff'):
                tifffile.imwrite(filename, self.data)
            elif filename.endswith('.npy'):
                np.save(filename, self.data)

    def exportScreenshot(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "", "PNG Files (*.png)")
        if filename:
            screen = QApplication.primaryScreen()
            screenshot = screen.grabWindow(self.winId())
            screenshot.save(filename, 'png')

    def exportVideo(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MP4 Files (*.mp4)")
        if filename:
            import imageio
            writer = imageio.get_writer(filename, fps=10)
            for t in range(self.data.shape[0]):
                self.timeSlider.setValue(t)
                QApplication.processEvents()
                screen = QApplication.primaryScreen()
                screenshot = screen.grabWindow(self.winId())
                writer.append_data(screenshot.toImage().convertToFormat(QImage.Format_RGB888).bits().asstring(screenshot.width() * screenshot.height() * 3))
            writer.close()

    def detect_blobs(self):
        if self.data is None:
            self.logger.warning("No data available for blob detection")
            return

        max_time =  self.timeSlider.maximum()+1
        num_channels = self.data.shape[1]

        # Get blob detection parameters from UI
        max_sigma = self.maxSigmaSpinBox.value()
        num_sigma = self.numSigmaSpinBox.value()
        threshold = self.blobThresholdSpinBox.value()

        all_blobs = []
        for channel in range(num_channels):
            for t in range(max_time):
                # Get the current 3D volume for this channel
                volume = self.data[t, channel]

                # Detect blobs
                blobs = blob_log(volume, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)

                # Calculate intensity for each blob
                for blob in blobs:
                    y, x, z, r = blob
                    y, x, z = int(y), int(x), int(z)
                    r = int(r)

                    # Define a small region around the blob center
                    y_min, y_max = max(0, y-r), min(volume.shape[0], y+r+1)
                    x_min, x_max = max(0, x-r), min(volume.shape[1], x+r+1)
                    z_min, z_max = max(0, z-r), min(volume.shape[2], z+r+1)

                    # Extract the region
                    region = volume[y_min:y_max, x_min:x_max, z_min:z_max]

                    # Calculate the intensity (you can use different measures here)
                    intensity = np.mean(region)  # or np.max(region), np.sum(region), etc.

                    # Add blob information including intensity
                    all_blobs.append([y, x, z, r, channel, t, intensity])


        # Combine blobs from all channels
        all_blobs = np.vstack(all_blobs)

        self.logger.info(f"Detected {len(all_blobs)} blobs across all channels")

        # Store all detected blobs
        self.all_detected_blobs = all_blobs

        # Display results
        self.display_blob_results(all_blobs)

        # Visualize blobs
        self.updateBlobVisualization()

        # Show the blob results button and update its text
        self.showBlobResultsButton.setVisible(True)
        self.showBlobResultsButton.setText("Show Blob Results")

        return all_blobs

    def display_blob_results(self, blobs):
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("Blob Detection Results")
        layout = QVBoxLayout(result_dialog)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels(["X", "Y", "Z", "Size", "Channel", "Time", "Intensity"])
        table.setRowCount(len(blobs))

        for i, blob in enumerate(blobs):
            y, x, z, r, channel, t, intensity = blob
            table.setItem(i, 0, QTableWidgetItem(f"{x:.2f}"))
            table.setItem(i, 1, QTableWidgetItem(f"{y:.2f}"))
            table.setItem(i, 2, QTableWidgetItem(f"{z:.2f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{r:.2f}"))
            table.setItem(i, 4, QTableWidgetItem(f"{int(channel)}"))
            table.setItem(i, 5, QTableWidgetItem(f"{int(t)}"))
            table.setItem(i, 6, QTableWidgetItem(f"{intensity:.2f}"))

        layout.addWidget(table)

        close_button = QPushButton("Close")
        close_button.clicked.connect(result_dialog.close)
        layout.addWidget(close_button)

        result_dialog.exec_()

    def updateBlobVisualization(self):
        if not hasattr(self, 'all_detected_blobs') or self.all_detected_blobs is None:
            return

        current_time = self.timeSlider.value()

        if self.showAllBlobsCheck.isChecked():
            blobs_to_show = self.all_detected_blobs
        else:
            blobs_to_show = self.all_detected_blobs[self.all_detected_blobs[:, 5] == current_time]

        self.visualize_blobs(blobs_to_show)

    def toggleBlobResults(self):
        if self.blob_results_dialog.isVisible():
            self.blob_results_dialog.hide()
            self.showBlobResultsButton.setText("Show Blob Results")
        else:
            if hasattr(self, 'all_detected_blobs'):
                self.blob_results_dialog.update_results(self.all_detected_blobs)
            self.blob_results_dialog.show()
            self.showBlobResultsButton.setText("Hide Blob Results")

    def clearDetectedBlobs(self):
        if hasattr(self, 'all_detected_blobs'):
            del self.all_detected_blobs
        self.updateBlobVisualization()
        self.blob_results_dialog.hide()
        self.showBlobResultsButton.setVisible(False)

    def analyzeBlobsasdkjfb(self):
        if hasattr(self, 'all_detected_blobs') and self.all_detected_blobs is not None:
            blob_analyzer = BlobAnalyzer(self.all_detected_blobs)
            analysis_dialog = BlobAnalysisDialog(blob_analyzer, self)
            analysis_dialog.setWindowTitle("Blob Analysis Results")
            analysis_dialog.setGeometry(100, 100, 800, 600)
            analysis_dialog.show()
        else:
            QMessageBox.warning(self, "No Blobs Detected", "Please detect blobs before running analysis.")

    def showTimeSeriesAnalysis(self):
        if hasattr(self, 'all_detected_blobs') and self.all_detected_blobs is not None:
            dialog = TimeSeriesDialog(BlobAnalyzer(self.all_detected_blobs), self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "No Data", "Please detect blobs first.")

def main():
    app = QApplication(sys.argv)
    viewer = LightsheetViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
