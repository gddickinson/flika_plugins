import sys
import numpy as np
import logging
import tifffile
import os
from scipy.ndimage import rotate
from typing import Tuple, List, Optional, Any

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QFileDialog, QMessageBox, QCheckBox, QDockWidget, QSizePolicy)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor


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



class LightsheetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initLogging()
        self.volume_processor = VolumeProcessor()
        self.data_generator = DataGenerator()
        self.playbackTimer = QTimer(self)
        self.playbackTimer.timeout.connect(self.advanceTimePoint)
        self.initUI()
        self.generateData()

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
        self.createVisualizationControlDock()
        self.createPlaybackControlDock()

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

        # Adjust dock sizes
        self.resizeDocks([self.dockXY, self.dockXZ, self.dockYZ], [200, 200, 200], Qt.Vertical)
        self.resizeDocks([self.dock3D, self.dockDataGeneration], [800, 300], Qt.Horizontal)


    def createViewDocks(self):
        # XY View
        self.dockXY = QDockWidget("XY View", self)
        self.imageViewXY = pg.ImageView()
        self.imageViewXY.ui.roiBtn.hide()
        self.imageViewXY.ui.menuBtn.hide()
        self.imageViewXY.timeLine.sigPositionChanged.connect(self.updateSliceMarkers)
        self.dockXY.setWidget(self.imageViewXY)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXY)

        # XZ View
        self.dockXZ = QDockWidget("XZ View", self)
        self.imageViewXZ = pg.ImageView()
        self.imageViewXZ.ui.roiBtn.hide()
        self.imageViewXZ.ui.menuBtn.hide()
        self.imageViewXZ.timeLine.sigPositionChanged.connect(self.updateSliceMarkers)
        self.dockXZ.setWidget(self.imageViewXZ)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockXZ)

        # YZ View
        self.dockYZ = QDockWidget("YZ View", self)
        self.imageViewYZ = pg.ImageView()
        self.imageViewYZ.ui.roiBtn.hide()
        self.imageViewYZ.ui.menuBtn.hide()
        self.imageViewYZ.timeLine.sigPositionChanged.connect(self.updateSliceMarkers)
        self.dockYZ.setWidget(self.imageViewYZ)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dockYZ)

        # 3D View
        self.dock3D = QDockWidget("3D View", self)
        self.glView = gl.GLViewWidget()
        self.dock3D.setWidget(self.glView)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock3D)

        # Set size policies for the image views
        for view in [self.imageViewXY, self.imageViewXZ, self.imageViewYZ, self.glView]:
            view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            view.setMinimumSize(200, 200)

    def updateSliceMarkers(self):
        if not hasattr(self, 'data') or self.data is None:
            return

        # Remove old markers
        for attr in ['x_marker', 'y_marker', 'z_marker']:
            if hasattr(self, attr):
                self.glView.removeItem(getattr(self, attr))
                delattr(self, attr)

        if self.showSliceMarkersCheck.isChecked():
            x_slice = self.imageViewXY.currentIndex
            y_slice = self.imageViewXZ.currentIndex
            z_slice = self.imageViewYZ.currentIndex

            # Create new markers
            self.x_marker = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, 100, 0], [x_slice, 100, 30], [x_slice, 0, 30]]),
                                              color=(1, 0, 0, 1), width=2, mode='line_strip')
            self.y_marker = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [100, y_slice, 0], [100, y_slice, 30], [0, y_slice, 30]]),
                                              color=(0, 1, 0, 1), width=2, mode='line_strip')
            self.z_marker = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [100, 0, z_slice], [100, 100, z_slice], [0, 100, z_slice]]),
                                              color=(0, 0, 1, 1), width=2, mode='line_strip')

            # Add new markers
            self.glView.addItem(self.x_marker)
            self.glView.addItem(self.y_marker)
            self.glView.addItem(self.z_marker)


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

        layout.addStretch(1)  # This pushes everything up
        self.dockVisualizationControl.setWidget(visControlWidget)

    def createPlaybackControlDock(self):
        self.dockPlaybackControl = QDockWidget("Playback Control", self)
        playbackControlWidget = QWidget()
        layout = QVBoxLayout(playbackControlWidget)

        layout.addWidget(QLabel("Time:"))
        self.timeSlider = QSlider(Qt.Horizontal)
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(0)
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

    def generateData(self):
        try:
            num_volumes = self.numVolumesSpinBox.value()
            num_blobs = self.numBlobsSpinBox.value()
            noise_level = self.noiseLevelSpinBox.value()
            movement_speed = self.movementSpeedSpinBox.value()

            self.data = self.data_generator.generate_time_series(
                num_volumes=num_volumes,
                size=(100, 100, 30),
                num_blobs=num_blobs,
                intensity_range=(0.8, 1.0),
                sigma_range=(2, 6),
                noise_level=noise_level,
                movement_speed=movement_speed
            )
            self.timeSlider.setMaximum(num_volumes - 1)
            self.updateViews()
            self.create3DVisualization()
            self.logger.info("Data generated and visualized successfully")
            self.logger.info(f"Data shape: {self.data.shape}")
            self.logger.info(f"Data metadata: {self.data_generator.get_metadata()}")
        except Exception as e:
            self.logger.error(f"Error in data generation: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to generate data: {str(e)}")

    def updateViews(self):
        if self.data is None:
            return
        t = self.timeSlider.value()
        self.imageViewXY.setImage(self.data[t])
        self.imageViewXZ.setImage(self.data[t].transpose(1, 2, 0))
        self.imageViewYZ.setImage(self.data[t].transpose(2, 1, 0))

        self.updateSliceMarkers()

    def create3DVisualization(self):
        try:
            self.glView.clear()
            t = self.timeSlider.value()
            threshold = self.thresholdSpinBox.value()

            volume_data = self.data[t] if self.data.ndim == 4 else self.data

            # Create color map
            cmap = self.getColorMap()

            if self.renderModeCombo.currentText() == "Points":
                bright_points = np.where(volume_data > threshold)
                pos = np.array(bright_points).T.astype(float)

                if len(pos) == 0:
                    self.logger.warning("No points above threshold for 3D visualization")
                    return

                colors = cmap.map(volume_data[bright_points] / np.max(volume_data))

                self.scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=2)
                self.glView.addItem(self.scatter)

            elif self.renderModeCombo.currentText() in ["Surface", "Wireframe"]:
                x, y, z = np.where(volume_data > threshold)
                if len(x) == 0:
                    self.logger.warning("No points above threshold for 3D visualization")
                    return

                verts = np.vstack([x, y, z]).T
                faces = np.array(self.triangulate_points(verts))

                mesh_data = gl.MeshData(vertexes=verts, faces=faces)
                colors = cmap.map(volume_data[volume_data > threshold] / np.max(volume_data))
                mesh_data.setVertexColors(colors)

                mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=True, drawEdges=self.renderModeCombo.currentText() == "Wireframe")
                self.glView.addItem(mesh)

            # Add coordinate axes
            ax = gl.GLAxisItem()
            ax.setSize(100, 100, 30)
            self.glView.addItem(ax)

            # Set up the view
            self.glView.setCameraPosition(distance=200, elevation=30, azimuth=45)
            self.glView.setBackgroundColor('k')

            # Update slice markers
            self.updateSliceMarkers()

            self.logger.info("3D visualization created successfully")
        except Exception as e:
            self.logger.error(f"Error in 3D visualization: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to create 3D visualization: {str(e)}")

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

    def addSliceMarkers(self):
        x_slice = self.imageViewXY.currentIndex
        y_slice = self.imageViewXZ.currentIndex
        z_slice = self.imageViewYZ.currentIndex

        x_marker = gl.GLLinePlotItem(pos=np.array([[x_slice, 0, 0], [x_slice, 100, 0], [x_slice, 100, 30], [x_slice, 0, 30]]), color=(1, 0, 0, 1), width=2, mode='line_strip')
        y_marker = gl.GLLinePlotItem(pos=np.array([[0, y_slice, 0], [100, y_slice, 0], [100, y_slice, 30], [0, y_slice, 30]]), color=(0, 1, 0, 1), width=2, mode='line_strip')
        z_marker = gl.GLLinePlotItem(pos=np.array([[0, 0, z_slice], [100, 0, z_slice], [100, 100, z_slice], [0, 100, z_slice]]), color=(0, 0, 1, 1), width=2, mode='line_strip')

        self.glView.addItem(x_marker)
        self.glView.addItem(y_marker)
        self.glView.addItem(z_marker)

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

    def updateTimePoint(self, value):
        self.updateViews()
        self.create3DVisualization()
        if value == self.timeSlider.maximum() and not self.loopCheckBox.isChecked():
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")

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
        if self.playbackTimer.isActive():
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")
        else:
            self.playbackTimer.start(int(1000 / self.speedSpinBox.value()))
            self.playPauseButton.setText("Pause")

    def updatePlaybackSpeed(self, value):
        if self.playbackTimer.isActive():
            self.playbackTimer.setInterval(int(1000 / value))

    def advanceTimePoint(self):
        current_time = self.timeSlider.value()
        if current_time < self.timeSlider.maximum():
            self.timeSlider.setValue(current_time + 1)
        elif self.loopCheckBox.isChecked():
            self.timeSlider.setValue(0)
        else:
            self.playbackTimer.stop()
            self.playPauseButton.setText("Play")

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

def main():
    app = QApplication(sys.argv)
    viewer = LightsheetViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
