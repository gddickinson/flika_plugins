import sys
import numpy as np
import logging
import tifffile
import os
from scipy.ndimage import rotate
from typing import Tuple, List, Optional, Any, Dict

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
                             QComboBox, QFileDialog, QMessageBox, QCheckBox, QDockWidget, QSizePolicy, QTableWidget,
                             QTableWidgetItem, QDialog, QGridLayout, QTabWidget, QTextEdit, QAction, QFormLayout)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5.QtCore import QTimer, QEvent
from PyQt5.QtGui import QColor, QVector3D, QImage, QMouseEvent, QWheelEvent
import traceback

from matplotlib import pyplot as plt
from skimage.feature import blob_log

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from scipy.spatial import cKDTree

from scipy.ndimage import distance_transform_edt, center_of_mass, binary_dilation, gaussian_filter
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import pyqtSlot
from scipy.spatial.transform import Rotation

#from data_generator import DataGenerator
#from volume_processor import VolumeProcessor

def line_3d(x0, y0, z0, x1, y1, z1):
    """Generate coordinates of a 3D line using Bresenham's algorithm."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1
    dm = max(dx, dy, dz)
    x, y, z = x0, y0, z0

    x_coords, y_coords, z_coords = [], [], []

    for _ in range(dm + 1):
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

        if x == x1 and y == y1 and z == z1:
            break

        if dx >= dy and dx >= dz:
            x += sx
            dy += dy
            dz += dz
            if dy >= dx:
                y += sy
                dy -= dx
            if dz >= dx:
                z += sz
                dz -= dx
        elif dy >= dx and dy >= dz:
            y += sy
            dx += dx
            dz += dz
            if dx >= dy:
                x += sx
                dx -= dy
            if dz >= dy:
                z += sz
                dz -= dy
        else:
            z += sz
            dx += dx
            dy += dy
            if dx >= dz:
                x += sx
                dx -= dz
            if dy >= dz:
                y += sy
                dy -= dz

    return np.array(x_coords), np.array(y_coords), np.array(z_coords)

class BiologicalSimulator:
    def __init__(self, size, num_time_points):
        self.size = size
        self.num_time_points = num_time_points
        self.logger = logging.getLogger(__name__)
        self.call_count = 0

    def simulate_protein_diffusion(self, D, initial_concentration):
        self.call_count += 1
        #print(f"Debug: Call number {self.call_count} to simulate_protein_diffusion")
        #print(f"Debug: Called from: {traceback.extract_stack()[-2][2]}")  # Print the caller's function name
        # ... rest of the method remains the same
        #print("Debug: Entering simulate_protein_diffusion method")
        #print(f"Debug: D = {D}")
        #print(f"Debug: self.size = {self.size}")
        #print(f"Debug: initial_concentration.shape = {initial_concentration.shape}")

        try:
            self.logger.info(f"Starting protein diffusion simulation with D={D}")
            self.logger.info(f"Simulator size: {self.size}")
            self.logger.info(f"Initial concentration shape: {initial_concentration.shape}")

            #print("Debug: Before shape check")
            if initial_concentration.shape != self.size:
                self.logger.warning(f"Initial concentration shape {initial_concentration.shape} does not match simulator size {self.size}")
                raise ValueError(f"Initial concentration shape {initial_concentration.shape} does not match simulator size {self.size}")

            #print("Debug: After shape check")
            result = np.zeros((self.num_time_points, *self.size))
            result[0] = initial_concentration

            #print("Debug: Before diffusion loop")
            # Assuming 1 second per time step
            for t in range(1, self.num_time_points):
                result[t] = gaussian_filter(result[t-1], sigma=np.sqrt(2*D))

            #print("Debug: After diffusion loop")
            self.logger.info("Protein diffusion simulation completed successfully")
            self.logger.info(f"Result shape: {result.shape}")
            return result

        except ValueError as e:
            # Log expected errors as warnings
            self.logger.warning(f"Expected error in protein diffusion simulation: {str(e)}")
            raise

        except Exception as e:
            #print(f"Debug: Exception caught: {str(e)}")
            self.logger.error(f"Error in protein diffusion simulation: {str(e)}")
            raise

    def simulate_active_transport(self, velocity: Tuple[float, float, float], cargo_concentration: np.ndarray) -> np.ndarray:
        """
        Simulate active transport of cargo along cytoskeleton.

        Args:
        velocity (Tuple[float, float, float]): Velocity vector for transport
        cargo_concentration (np.ndarray): Initial cargo concentration

        Returns:
        np.ndarray: Time series of cargo concentrations
        """
        try:
            result = np.zeros((self.num_time_points, *self.size))
            result[0] = cargo_concentration
            for t in range(1, self.num_time_points):
                result[t] = np.roll(result[t-1], shift=(int(velocity[0]), int(velocity[1]), int(velocity[2])), axis=(0,1,2))
            return result
        except Exception as e:
            self.logger.error(f"Error in active transport simulation: {str(e)}")
            raise



    def generate_cellular_structure(self, structure_type: str) -> np.ndarray:
        """
        Generate a 3D representation of a cellular structure.

        Args:
        structure_type (str): Type of cellular structure ('nucleus', 'mitochondria', 'actin', 'lysosomes')

        Returns:
        np.ndarray: 3D array representing the cellular structure
        """
        try:
            if structure_type == 'nucleus':
                return self._generate_nucleus()
            elif structure_type == 'mitochondria':
                return self._generate_mitochondria()
            elif structure_type == 'actin':
                return self._generate_actin()
            elif structure_type == 'lysosomes':
                return self._generate_lysosomes()
            else:
                raise ValueError(f"Unknown structure type: {structure_type}")
        except Exception as e:
            self.logger.error(f"Error generating cellular structure: {str(e)}")
            raise

    def generate_cell_membrane(self, center, radius, thickness=1):
        """
        Generate a cell plasma membrane.

        Args:
        center (tuple): The (z, y, x) coordinates of the cell center
        radius (float): The radius of the cell
        thickness (float): The thickness of the membrane

        Returns:
        np.ndarray: 3D array representing the cell membrane
        """
        try:
            self.logger.info(f"Generating cell membrane at {center} with radius {radius}")

            z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
            dist_from_center = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)

            membrane = (dist_from_center >= radius - thickness/2) & (dist_from_center <= radius + thickness/2)

            self.logger.info("Cell membrane generated successfully")
            return membrane.astype(float)

        except Exception as e:
            self.logger.error(f"Error in cell membrane generation: {str(e)}")
            raise


    def generate_nucleus(self, cell_interior, soma_center, nucleus_radius, pixel_size=(1,1,1)):
        try:
            self.logger.info(f"Generating cell nucleus at {soma_center} with radius {nucleus_radius}")

            # Ensure nucleus_radius is at least 1 pixel and not larger than 1/3 of the soma
            soma_radius = min(self.size) // 4
            nucleus_radius = min(max(1, nucleus_radius), soma_radius // 3)

            # Find a suitable center for the nucleus
            new_center = self.find_suitable_center(cell_interior, soma_center, nucleus_radius)

            if new_center is None:
                self.logger.warning("Unable to find suitable location for nucleus.")
                return np.zeros_like(cell_interior), soma_center

            # Create a spherical nucleus
            z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
            dist_from_center = np.sqrt(
                ((z - new_center[0]) * pixel_size[0])**2 +
                ((y - new_center[1]) * pixel_size[1])**2 +
                ((x - new_center[2]) * pixel_size[2])**2
            )

            # Create nucleus only within the cell interior
            nucleus = (dist_from_center <= nucleus_radius) & (cell_interior > 0)

            actual_volume = np.sum(nucleus)
            self.logger.info(f"Cell nucleus generated successfully. Nucleus volume: {actual_volume}")
            return nucleus.astype(float), new_center

        except Exception as e:
            self.logger.error(f"Error in cell nucleus generation: {str(e)}")
            raise

    def find_suitable_center(self, cell_shape, soma_center, nucleus_radius):
        z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
        dist_from_center = np.sqrt(
            (z - soma_center[0])**2 + (y - soma_center[1])**2 + (x - soma_center[2])**2
        )

        # Create a priority map that favors locations closer to the desired center
        priority_map = np.where(cell_shape, -dist_from_center, -np.inf)

        search_radius = 0
        max_search_radius = max(self.size)
        while search_radius < max_search_radius:
            potential_centers = np.argwhere(
                (dist_from_center <= search_radius) & (cell_shape > 0)
            )
            if len(potential_centers) > 0:
                # Sort potential centers by priority
                priorities = priority_map[tuple(potential_centers.T)]
                sorted_centers = potential_centers[np.argsort(priorities)]

                for center in sorted_centers:
                    # Check if the chosen center can accommodate the nucleus
                    z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
                    dist_from_center = np.sqrt(
                        (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
                    )
                    if np.sum((dist_from_center <= nucleus_radius) & (cell_shape > 0)) >= 7:
                        return center

            search_radius += 1

        # If no suitable center is found, return the center of mass of the cell shape
        if np.sum(cell_shape) > 0:
            return np.array(center_of_mass(cell_shape)).astype(int)
        else:
            return None

    def generate_er(self, cell_shape, soma_center, nucleus_radius, er_density=0.1, pixel_size=(1,1,1)):
        try:
            self.logger.info(f"Generating ER with density {er_density}")

            z, y, x = np.ogrid[:self.size[0], :self.size[1], :self.size[2]]
            dist_from_center = np.sqrt(
                ((z - soma_center[0]) * pixel_size[0])**2 +
                ((y - soma_center[1]) * pixel_size[1])**2 +
                ((x - soma_center[2]) * pixel_size[2])**2
            )

            # Create a mask for the cell
            cell_mask = cell_shape > 0
            nucleus_mask = dist_from_center <= nucleus_radius
            cytoplasm_mask = cell_mask & ~nucleus_mask

            self.logger.info(f"Cell volume: {np.sum(cell_mask)}, Nucleus volume: {np.sum(nucleus_mask)}, Cytoplasm volume: {np.sum(cytoplasm_mask)}")

            # Initialize ER
            er = np.zeros(self.size, dtype=bool)

            # Generate ER throughout the cytoplasm
            er = cytoplasm_mask & (np.random.rand(*self.size) < er_density)

            # Ensure higher ER density near the nucleus
            near_nucleus = (dist_from_center > nucleus_radius) & (dist_from_center <= nucleus_radius * 1.5)
            er |= near_nucleus & cytoplasm_mask & (np.random.rand(*self.size) < er_density * 2)

            self.logger.info(f"ER generated successfully. ER volume: {np.sum(er)}")
            return er.astype(float)

        except Exception as e:
            self.logger.error(f"Error in ER generation: {str(e)}")
            raise

    def generate_cell_shape(self, cell_type, size, pixel_size=(1,1,1), membrane_thickness=1, soma_center=None, **kwargs):
        try:
            self.logger.info(f"Generating {cell_type} cell shape")

            z, y, x = np.ogrid[:size[0], :size[1], :size[2]]
            if soma_center is None:
                soma_center = np.array(size) // 2

            if cell_type == 'spherical':
                radius = kwargs.get('cell_radius', min(size) // 4)
                self.logger.info(f"Generating spherical cell with radius {radius}")
                dist_from_center = np.sqrt(
                    ((z - soma_center[0]) * pixel_size[0])**2 +
                    ((y - soma_center[1]) * pixel_size[1])**2 +
                    ((x - soma_center[2]) * pixel_size[2])**2
                )
                cell_interior = dist_from_center <= (radius - membrane_thickness)
                cell_membrane = (dist_from_center <= radius) & (dist_from_center > (radius - membrane_thickness))
                cell_shape = cell_interior | cell_membrane

            elif cell_type == 'neuron':
                soma_radius = kwargs.get('soma_radius', min(size) // 8)
                axon_length = kwargs.get('axon_length', size[2] // 2)
                axon_width = kwargs.get('axon_width', size[1] // 20)
                num_dendrites = kwargs.get('num_dendrites', 5)
                dendrite_length = kwargs.get('dendrite_length', size[1] // 4)

                self.logger.info(f"Generating neuron with soma radius {soma_radius}")

                # Create soma (cell body)
                dist_from_soma_center = np.sqrt(
                    ((z - soma_center[0]) * pixel_size[0])**2 +
                    ((y - soma_center[1]) * pixel_size[1])**2 +
                    ((x - soma_center[2]) * pixel_size[2])**2
                )
                soma = dist_from_soma_center <= soma_radius

                # Create axon
                axon_start = soma_center[2] + soma_radius
                axon = (x >= axon_start) & (x < axon_start + axon_length) & \
                       (np.abs(y - soma_center[1]) <= axon_width // 2) & \
                       (np.abs(z - soma_center[0]) <= axon_width // 2)

                # Create axon terminals
                axon_end = axon_start + axon_length
                terminals = np.zeros_like(soma, dtype=bool)
                for _ in range(3):  # Create 3 terminals
                    terminal_center = (
                        soma_center[0] + np.random.randint(-axon_width, axon_width),
                        soma_center[1] + np.random.randint(-axon_width, axon_width),
                        axon_end + np.random.randint(0, size[2]//10)
                    )
                    terminal_radius = axon_width // 2
                    dist_from_terminal = np.sqrt(
                        ((z - terminal_center[0]) * pixel_size[0])**2 +
                        ((y - terminal_center[1]) * pixel_size[1])**2 +
                        ((x - terminal_center[2]) * pixel_size[2])**2
                    )
                    terminals |= dist_from_terminal <= terminal_radius

                # Create dendrites
                dendrites = np.zeros_like(soma, dtype=bool)
                for _ in range(num_dendrites):
                    angle = np.random.uniform(0, 2*np.pi)
                    end_point = (
                        int(soma_center[0] + dendrite_length * np.sin(angle) * np.cos(angle)),
                        int(soma_center[1] + dendrite_length * np.sin(angle)),
                        int(soma_center[2] - dendrite_length * np.cos(angle))
                    )
                    rr, cc, zz = line_3d(soma_center[0], soma_center[1], soma_center[2],
                                         end_point[0], end_point[1], end_point[2])
                    dendrites[rr, cc, zz] = True

                dendrites = binary_dilation(dendrites, iterations=2)

                # Combine all parts to create the cell interior
                cell_interior = soma | axon | terminals | dendrites

                # Create the membrane by dilating the cell interior
                cell_shape = binary_dilation(cell_interior, iterations=membrane_thickness)
                cell_membrane = cell_shape & ~cell_interior


            elif cell_type == 'epithelial':
                height = size[0] // 3
                outer_shape = z < height
                inner_shape = z < (height - membrane_thickness)
                cell_shape = outer_shape ^ inner_shape

            elif cell_type == 'muscle':
                outer_shape = ((y - soma_center[1])*pixel_size[1])**2 + ((z - soma_center[0])*pixel_size[0])**2 <= (min(size)//4)**2
                inner_shape = ((y - soma_center[1])*pixel_size[1])**2 + ((z - soma_center[0])*pixel_size[0])**2 <= (min(size)//4 - membrane_thickness)**2
                cell_shape = outer_shape ^ inner_shape

            else:
                raise ValueError(f"Unknown cell type: {cell_type}")

            non_zero_coords = np.argwhere(cell_shape > 0)
            self.logger.info(f"Non-zero cell shape coordinates: min={non_zero_coords.min(axis=0)}, max={non_zero_coords.max(axis=0)}")

            # Visualize the cell shape
            #plt.imshow(cell_shape[:, :, cell_shape.shape[2]//2])
            #plt.title("Cell Shape (Middle Slice)")
            #plt.colorbar()
            #plt.savefig("cell_shape.png")
            #plt.close()

            self.logger.info(f"{cell_type} cell shape generated successfully")
            return cell_shape.astype(float), cell_interior.astype(float), cell_membrane.astype(float)


        except Exception as e:
            self.logger.error(f"Error in cell shape generation: {str(e)}")
            raise

    def _generate_nucleus(self) -> np.ndarray:
        # Implement nucleus generation here
        pass

    def _generate_mitochondria(self) -> np.ndarray:
        # Implement mitochondria generation here
        pass

    def _generate_actin(self) -> np.ndarray:
        # Implement actin network generation here
        pass

    def _generate_lysosomes(self) -> np.ndarray:
        # Implement lysosome generation here
        pass

    def simulate_calcium_signal(self, signal_type: str, params: Dict) -> np.ndarray:
        """
        Simulate calcium signaling events.

        Args:
        signal_type (str): Type of calcium signal ('blip', 'puff', 'wave')
        params (Dict): Parameters for the specific signal type

        Returns:
        np.ndarray: Time series of calcium concentrations
        """
        try:
            if signal_type == 'blip':
                return self._simulate_calcium_blip(params)
            elif signal_type == 'puff':
                return self._simulate_calcium_puff(params)
            elif signal_type == 'wave':
                return self._simulate_calcium_wave(params)
            else:
                raise ValueError(f"Unknown calcium signal type: {signal_type}")
        except Exception as e:
            self.logger.error(f"Error in calcium signal simulation: {str(e)}")
            raise

    def _simulate_calcium_blip(self, params: Dict) -> np.ndarray:
        # Implement calcium blip simulation here
        pass

    def _simulate_calcium_puff(self, params: Dict) -> np.ndarray:
        # Implement calcium puff simulation here
        pass

    def _simulate_calcium_wave(self, params: Dict) -> np.ndarray:
        # Implement calcium wave simulation here
        pass


class BiologicalSimulationWidget(QWidget):
    simulationRequested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout(self)  # Use self as the parent for the layout

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_protein_tab()
        self.create_structure_tab()
        self.create_calcium_tab()

        # Simulate Button
        self.simulate_button = QPushButton("Simulate")
        self.simulate_button.clicked.connect(self.requestSimulation)
        main_layout.addWidget(self.simulate_button)

    def create_protein_tab(self):
        protein_tab = QWidget()
        layout = QVBoxLayout(protein_tab)

        # Protein Diffusion
        diffusion_layout = QHBoxLayout()
        diffusion_layout.addWidget(QLabel("Protein Diffusion:"))
        self.diffusion_checkbox = QCheckBox()
        diffusion_layout.addWidget(self.diffusion_checkbox)
        self.diffusion_coefficient = QDoubleSpinBox()
        self.diffusion_coefficient.setRange(0, 100)
        self.diffusion_coefficient.setValue(1)
        diffusion_layout.addWidget(self.diffusion_coefficient)
        layout.addLayout(diffusion_layout)

        # Active Transport
        transport_layout = QHBoxLayout()
        transport_layout.addWidget(QLabel("Active Transport:"))
        self.transport_checkbox = QCheckBox()
        transport_layout.addWidget(self.transport_checkbox)
        self.transport_velocity = QDoubleSpinBox()
        self.transport_velocity.setRange(-10, 10)
        self.transport_velocity.setValue(1)
        transport_layout.addWidget(self.transport_velocity)
        layout.addLayout(transport_layout)

        self.tab_widget.addTab(protein_tab, "Protein Dynamics")

    def create_structure_tab(self):
        structure_tab = QWidget()
        main_layout = QVBoxLayout(structure_tab)

        # Cellular Structures
        structure_layout = QHBoxLayout()
        structure_layout.addWidget(QLabel("Cellular Structure:"))
        self.structure_combo = QComboBox()
        self.structure_combo.addItems(['None', 'Cell Membrane', 'Nucleus', 'Cell Membrane + Nucleus', 'Cell Membrane + Nucleus + ER'])
        structure_layout.addWidget(self.structure_combo)
        main_layout.addLayout(structure_layout)

        # Create a form layout for the rest of the options
        form_layout = QFormLayout()

        # Cell type selection
        self.cell_type_combo = QComboBox()
        self.cell_type_combo.addItems(['spherical', 'neuron', 'epithelial', 'muscle'])
        form_layout.addRow("Cell Type:", self.cell_type_combo)

        # Pixel size inputs
        self.pixel_size_x = QDoubleSpinBox()
        self.pixel_size_y = QDoubleSpinBox()
        self.pixel_size_z = QDoubleSpinBox()
        for spinbox in [self.pixel_size_x, self.pixel_size_y, self.pixel_size_z]:
            spinbox.setRange(0.1, 10)
            spinbox.setSingleStep(0.1)
            spinbox.setValue(1)
        pixel_size_layout = QHBoxLayout()
        pixel_size_layout.addWidget(self.pixel_size_x)
        pixel_size_layout.addWidget(self.pixel_size_y)
        pixel_size_layout.addWidget(self.pixel_size_z)
        form_layout.addRow("Pixel Size (x, y, z):", pixel_size_layout)

        # Cell Membrane options
        self.membrane_options = QWidget()
        membrane_layout = QFormLayout(self.membrane_options)
        self.cell_radius = QSpinBox()
        self.cell_radius.setRange(5, 50)
        self.cell_radius.setValue(20)
        membrane_layout.addRow("Cell Radius:", self.cell_radius)
        self.membrane_thickness = QSpinBox()
        self.membrane_thickness.setRange(1, 5)
        self.membrane_thickness.setValue(1)
        membrane_layout.addRow("Membrane Thickness:", self.membrane_thickness)
        form_layout.addRow(self.membrane_options)

        # Nucleus options
        self.nucleus_options = QWidget()
        nucleus_layout = QFormLayout(self.nucleus_options)
        self.nucleus_radius = QSpinBox()
        self.nucleus_radius.setRange(1, 10)
        self.nucleus_radius.setValue(3)
        nucleus_layout.addRow("Nucleus Radius:", self.nucleus_radius)
        self.nucleus_thickness = QSpinBox()
        self.nucleus_thickness.setRange(1, 3)
        self.nucleus_thickness.setValue(1)
        nucleus_layout.addRow("Nucleus Thickness:", self.nucleus_thickness)
        form_layout.addRow(self.nucleus_options)

        # ER options
        self.er_options = QWidget()
        er_layout = QFormLayout(self.er_options)
        self.er_density = QDoubleSpinBox()
        self.er_density.setRange(0.05, 0.2)
        self.er_density.setSingleStep(0.01)
        self.er_density.setValue(0.1)
        er_layout.addRow("ER Density:", self.er_density)
        form_layout.addRow(self.er_options)
        main_layout.addLayout(form_layout)

        # Neuron-specific options
        self.neuron_options = QWidget()
        neuron_layout = QFormLayout(self.neuron_options)

        self.soma_radius = QSpinBox()
        self.soma_radius.setRange(1, 20)
        self.soma_radius.setValue(5)
        neuron_layout.addRow("Soma Radius:", self.soma_radius)

        self.axon_length = QSpinBox()
        self.axon_length.setRange(10, 100)
        self.axon_length.setValue(50)
        neuron_layout.addRow("Axon Length:", self.axon_length)

        self.axon_width = QSpinBox()
        self.axon_width.setRange(1, 10)
        self.axon_width.setValue(2)
        neuron_layout.addRow("Axon Width:", self.axon_width)

        self.num_dendrites = QSpinBox()
        self.num_dendrites.setRange(1, 10)
        self.num_dendrites.setValue(5)
        neuron_layout.addRow("Number of Dendrites:", self.num_dendrites)

        self.dendrite_length = QSpinBox()
        self.dendrite_length.setRange(5, 50)
        self.dendrite_length.setValue(25)
        neuron_layout.addRow("Dendrite Length:", self.dendrite_length)

        form_layout.addRow(self.neuron_options)

        self.structure_combo.currentTextChanged.connect(self.toggle_structure_options)

        self.tab_widget.addTab(structure_tab, "Cellular Structures")

    def toggle_structure_options(self, structure):
        self.membrane_options.setVisible('Cell Membrane' in structure)
        self.nucleus_options.setVisible('Nucleus' in structure)
        self.er_options.setVisible('ER' in structure)
        self.neuron_options.setVisible('neuron' in structure.lower())

    def create_calcium_tab(self):
        calcium_tab = QWidget()
        layout = QVBoxLayout(calcium_tab)

        # Calcium Signaling
        calcium_layout = QHBoxLayout()
        calcium_layout.addWidget(QLabel("Calcium Signal:"))
        self.calcium_combo = QComboBox()
        self.calcium_combo.addItems(['None', 'Blip', 'Puff', 'Wave'])
        calcium_layout.addWidget(self.calcium_combo)
        layout.addLayout(calcium_layout)

        # Add more calcium signal-related controls here
        # For example:
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Signal Intensity:"))
        self.calcium_intensity = QDoubleSpinBox()
        self.calcium_intensity.setRange(0, 1)
        self.calcium_intensity.setSingleStep(0.1)
        self.calcium_intensity.setValue(0.5)
        intensity_layout.addWidget(self.calcium_intensity)
        layout.addLayout(intensity_layout)

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Signal Duration:"))
        self.calcium_duration = QSpinBox()
        self.calcium_duration.setRange(1, 100)
        self.calcium_duration.setValue(10)
        duration_layout.addWidget(self.calcium_duration)
        layout.addLayout(duration_layout)

        self.tab_widget.addTab(calcium_tab, "Calcium Signaling")

    def requestSimulation(self):
        params = {
            'protein_diffusion': {
                'enabled': self.diffusion_checkbox.isChecked(),
                'coefficient': self.diffusion_coefficient.value()
            },
            'active_transport': {
                'enabled': self.transport_checkbox.isChecked(),
                'velocity': self.transport_velocity.value()
            },
            'cellular_structure': self.structure_combo.currentText(),
            'cell_radius': self.cell_radius.value(),
            'membrane_thickness': self.membrane_thickness.value(),
            'nucleus_radius': self.nucleus_radius.value(),
            'nucleus_thickness': self.nucleus_thickness.value(),
            'er_density': self.er_density.value(),
            'cell_type': self.cell_type_combo.currentText(),
            'pixel_size': (self.pixel_size_x.value(), self.pixel_size_y.value(), self.pixel_size_z.value()),
            'soma_radius': self.soma_radius.value(),
            'axon_length': self.axon_length.value(),
            'axon_width': self.axon_width.value(),
            'num_dendrites': self.num_dendrites.value(),
            'dendrite_length': self.dendrite_length.value(),

            'calcium_signal': {
                'type': self.calcium_combo.currentText(),
                'intensity': self.calcium_intensity.value(),
                'duration': self.calcium_duration.value()
            }
        }
        self.simulationRequested.emit(params)

    def toggle_structure_options(self, structure):
        self.membrane_options.setVisible(structure in ['Cell Membrane', 'Cell Membrane + Nucleus'])
        self.nucleus_options.setVisible(structure in ['Nucleus', 'Cell Membrane + Nucleus'])


class BiologicalSimulationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Biological Simulation")
        self.simulation_widget = BiologicalSimulationWidget()
        self.setCentralWidget(self.simulation_widget)
        self.resize(400, 300)  # Set an initial size for the window


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

    def _generate_single_volume(
        self,
        size: Tuple[int, int, int],
        blob_positions: np.ndarray,
        blob_velocities: np.ndarray,
        intensity_range: Tuple[float, float],
        sigma_range: Tuple[float, float],
        noise_level: float
    ) -> np.ndarray:
        z, y, x = size
        volume = np.zeros((z, y, x))

        for i, (bz, by, bx) in enumerate(blob_positions):
            sigma = np.random.uniform(*sigma_range)
            intensity = np.random.uniform(*intensity_range)

            zz, yy, xx = np.ogrid[
                max(0, int(bz-3*sigma)):min(z, int(bz+3*sigma)),
                max(0, int(by-3*sigma)):min(y, int(by+3*sigma)),
                max(0, int(bx-3*sigma)):min(x, int(bx+3*sigma))
            ]
            blob = np.exp(-((zz-bz)**2 + (yy-by)**2 + (xx-bx)**2) / (2*sigma*sigma))
            volume[zz, yy, xx] += intensity * blob

        volume += np.random.normal(0, noise_level, (z, y, x))
        return np.clip(volume, 0, 1)

    def generate_multi_channel_time_series(
        self,
        num_volumes: int,
        num_channels: int = 2,
        size: Tuple[int, int, int] = (30, 100, 100),
        num_blobs: int = 30,
        intensity_range: Tuple[float, float] = (0.5, 1.0),
        sigma_range: Tuple[float, float] = (2, 6),
        noise_level: float = 0.02,
        movement_speed: float = 1.0
    ) -> np.ndarray:
        z, y, x = size
        time_series = np.zeros((num_volumes, num_channels, z, y, x))
        blob_positions = np.random.rand(num_channels, num_blobs, 3) * np.array([z, y, x])
        blob_velocities = np.random.randn(num_channels, num_blobs, 3) * movement_speed

        for t in range(num_volumes):
            for c in range(num_channels):
                volume = self._generate_single_volume(
                    size, blob_positions[c], blob_velocities[c],
                    intensity_range, sigma_range, noise_level
                )
                time_series[t, c] = volume

                # Update blob positions
                blob_positions[c] += blob_velocities[c]
                blob_positions[c] %= [z, y, x]  # Wrap around the volume

        self.data = time_series
        self.logger.info(f"Generated time series with shape: {time_series.shape}")
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

        plt.close()

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

        plt.close()

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

        plt.close()

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

        plt.close()

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

######################################################################################################
######################################################################################################
'''                                   MAIN LIGHTSHEETVIEWER CLASS                                  '''
######################################################################################################
######################################################################################################

class LightsheetViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initLogging()
        self.volume_processor = VolumeProcessor()
        self.data = None
        self.lastPos = None
        self.data_generator = DataGenerator()

        self.channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # RGB for up to 3 channels
        self.biological_simulator = BiologicalSimulator(size=(30, 100, 100), num_time_points=10)
        self.biological_simulation_window = None  # We'll create this lazily

        self.setupChannelControlsWidget()

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


        self.connectViewEvents()

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

        self.check_view_state()  # Check the state after initialization


    def runBiologicalSimulation(self, params):
        try:
            # Run protein diffusion simulation if enabled
            if params['protein_diffusion']['enabled']:
                diffusion_coefficient = params['protein_diffusion']['coefficient']

                # Ensure the biological simulator has the correct number of time points
                if self.data is not None:
                    self.biological_simulator.num_time_points = self.data.shape[0]
                else:
                    self.logger.warning("No existing data. Creating new data set for protein diffusion.")
                    self.biological_simulator.num_time_points = 10  # or any default value

                initial_concentration = np.zeros(self.biological_simulator.size)
                initial_concentration[self.biological_simulator.size[0]//2,
                                      self.biological_simulator.size[1]//2,
                                      self.biological_simulator.size[2]//2] = 1.0  # Point source in the center

                diffusion_data = self.biological_simulator.simulate_protein_diffusion(
                    diffusion_coefficient,
                    initial_concentration
                )

                # Add the diffusion data as a new channel
                if self.data is None:
                    self.data = diffusion_data[:, np.newaxis, :, :, :]  # Add channel dimension
                else:
                    self.data = np.concatenate((self.data, diffusion_data[:, np.newaxis, :, :, :]), axis=1)

                self.logger.info(f"Added protein diffusion data. New data shape: {self.data.shape}")

                # Update UI elements for the new channel
                self.updateUIForNewData()

            # Run active transport simulation if enabled
            if params['active_transport']['enabled']:
                # We haven't implemented this yet, so let's just log it
                self.logger.info("Active transport simulation not yet implemented")

            # Generate cellular structures
            if params['cellular_structure'] != 'None':
                #center = (self.biological_simulator.size[0] // 2,
                #          self.biological_simulator.size[1] // 2,
                #          self.biological_simulator.size[2] // 2)

                # Define soma_center as the center of the volume
                soma_center = tuple(s // 2 for s in self.biological_simulator.size)
                pixel_size = params['pixel_size']
                cell_shape, cell_interior, cell_membrane = self.biological_simulator.generate_cell_shape(
                    params['cell_type'],
                    self.biological_simulator.size,
                    pixel_size,
                    membrane_thickness=params['membrane_thickness'],
                    cell_radius=params['cell_radius'],
                    soma_radius=params['soma_radius'],
                    axon_length=params['axon_length'],
                    axon_width=params['axon_width'],
                    num_dendrites=params['num_dendrites'],
                    dendrite_length=params['dendrite_length']
                )


                nucleus_radius = max(5, min(params['nucleus_radius'], params['soma_radius'] // 2))

                if 'Cell Membrane' in params['cellular_structure']:
                    membrane_timeseries = np.repeat(cell_membrane[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
                    self.data = membrane_timeseries
                    self.logger.info(f"Added cell membrane data. Data shape: {self.data.shape}")

                if 'Nucleus' in params['cellular_structure']:
                    nucleus_data, nucleus_center = self.biological_simulator.generate_nucleus(
                        cell_interior,
                        soma_center,
                        nucleus_radius,
                        pixel_size=pixel_size
                    )
                    self.logger.info(f"Nucleus data shape: {nucleus_data.shape}, min: {nucleus_data.min()}, max: {nucleus_data.max()}, non-zero elements: {np.count_nonzero(nucleus_data)}")
                    self.logger.info(f"Nucleus center: {nucleus_center}")
                    nucleus_timeseries = np.repeat(nucleus_data[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
                    if self.data is None:
                        self.data = nucleus_timeseries
                    else:
                        self.data = np.concatenate((self.data, nucleus_timeseries), axis=1)
                    self.logger.info(f"Added cell nucleus data. New data shape: {self.data.shape}")

                if 'ER' in params['cellular_structure']:
                    self.logger.info("Starting ER generation")
                    er_data = self.biological_simulator.generate_er(
                        cell_shape,
                        soma_center,
                        nucleus_radius,
                        params['er_density'],
                        pixel_size
                    )
                    self.logger.info(f"ER data shape: {er_data.shape}, min: {er_data.min()}, max: {er_data.max()}, non-zero elements: {np.count_nonzero(er_data)}")
                    if np.sum(er_data) == 0:
                        self.logger.warning("Generated ER has zero volume. ER will not be visible.")
                    er_timeseries = np.repeat(er_data[np.newaxis, np.newaxis, :, :, :], self.biological_simulator.num_time_points, axis=0)
                    if self.data is None:
                        self.data = er_timeseries
                    else:
                        self.data = np.concatenate((self.data, er_timeseries), axis=1)
                    self.logger.info(f"Added ER data. New data shape: {self.data.shape}")

                self.updateUIForNewData()
                self.updateViews()
                self.create3DVisualization()

            # Simulate calcium signal if selected
            if params['calcium_signal']['type'] != 'None':
                # We haven't implemented this yet, so let's just log it
                self.logger.info(f"Calcium signal simulation for {params['calcium_signal']['type']} not yet implemented")

            self.updateViews()
            self.create3DVisualization()

        except Exception as e:
            self.logger.error(f"Error in biological simulation: {str(e)}")
            QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation: {str(e)}")

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

        # Set up channel controls widget
        self.setupChannelControlsWidget()
        layout.addWidget(self.channelControlsWidget)


        layout.addWidget(QLabel("Threshold:"))
        self.thresholdSpinBox = QDoubleSpinBox()
        self.thresholdSpinBox.setRange(0, 1)
        self.thresholdSpinBox.setSingleStep(0.1)
        self.thresholdSpinBox.setValue(0)
        self.thresholdSpinBox.valueChanged.connect(self.updateThreshold)
        layout.addWidget(self.thresholdSpinBox)

        layout.addWidget(QLabel("Point Size"))
        self.scatterPointSizeSpinBox = QDoubleSpinBox()
        self.scatterPointSizeSpinBox.setRange(0, 100)
        self.scatterPointSizeSpinBox.setSingleStep(1)
        self.scatterPointSizeSpinBox.setValue(2)
        self.scatterPointSizeSpinBox.valueChanged.connect(self.updateScatterPointSize)
        layout.addWidget(self.scatterPointSizeSpinBox)


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

        # Add checkbox for synchronizing views
        self.syncViewsCheck = QCheckBox("Synchronize 3D Views")
        self.syncViewsCheck.setChecked(False)
        layout.addWidget(self.syncViewsCheck)

        # Add button for auto-scaling
        self.autoScaleButton = QPushButton("Auto Scale Views")
        self.autoScaleButton.clicked.connect(self.autoScaleViews)
        layout.addWidget(self.autoScaleButton)

        # Add buttons for orienting views
        self.topDownButton = QPushButton("Top-Down View")
        self.topDownButton.clicked.connect(self.setTopDownView)
        layout.addWidget(self.topDownButton)

        self.sideViewButton = QPushButton("Side View (XZ)")
        self.sideViewButton.clicked.connect(self.setSideView)
        layout.addWidget(self.sideViewButton)

        self.frontViewButton = QPushButton("Front View (YZ)")
        self.frontViewButton.clicked.connect(self.setFrontView)
        layout.addWidget(self.frontViewButton)

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
        self.blobThresholdSpinBox.setValue(0.5)
        blobLayout.addWidget(self.blobThresholdSpinBox, 2, 1)
        self.blobThresholdSpinBox.valueChanged.connect(self.updateBlobThreshold)

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

        # Connect mouse events
        self.blobGLView.mousePressEvent = self.on3DViewMousePress
        self.blobGLView.mouseReleaseEvent = self.on3DViewMouseRelease
        self.blobGLView.mouseMoveEvent = self.on3DViewMouseMove
        self.blobGLView.wheelEvent = self.on3DViewWheel
        self.logger.debug(f"Blob visualization dock created. blobGLView: {self.blobGLView}")

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

        self.glView.opts['fov'] = 60
        self.glView.opts['elevation'] = 30
        self.glView.opts['azimuth'] = 45

        # Connect mouse events
        self.glView.mousePressEvent = self.on3DViewMousePress
        self.glView.mouseReleaseEvent = self.on3DViewMouseRelease
        self.glView.mouseMoveEvent = self.on3DViewMouseMove
        self.glView.wheelEvent = self.on3DViewWheel
        self.logger.debug(f"3D visualization dock created. glView: {self.glView}")


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
            blob_item_vis.translate(z, x, y)  # Swapped y and z
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

        # Add a new menu item for the Biological Simulation window
        viewMenu = menuBar.addMenu('&View')
        self.showBioSimAction = QAction('Biological Simulation', self, checkable=True)
        self.showBioSimAction.triggered.connect(self.toggleBiologicalSimulationWindow)
        viewMenu.addAction(self.showBioSimAction)


    def toggleBiologicalSimulationWindow(self, checked):
        if checked:
            if self.biological_simulation_window is None:
                self.biological_simulation_window = BiologicalSimulationWindow(self)
                self.biological_simulation_window.simulation_widget.simulationRequested.connect(self.runBiologicalSimulation)
            self.biological_simulation_window.show()
        else:
            if self.biological_simulation_window:
                self.biological_simulation_window.hide()

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
            self.autoScaleViews()  # Add this line
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
        threshold = self.thresholdSpinBox.value()

        self.logger.debug(f"Updating views for time point {t}")
        self.logger.debug(f"Data shape: {self.data.shape}")

        num_channels = self.data.shape[1]
        depth, height, width = self.data.shape[2:]

        # Prepare 3D RGB images for each view
        combined_xy = np.zeros((depth, height, width, 3))
        combined_xz = np.zeros((height, depth, width, 3))
        combined_yz = np.zeros((height, depth, width, 3))

        # Define colors for each channel (RGB)
        channel_colors = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),  # Red, Green, Blue
            (1, 1, 0), (1, 0, 1), (0, 1, 1),  # Yellow, Magenta, Cyan
            (0.5, 0.5, 0.5), (1, 0.5, 0),     # Gray, Orange
        ]

        for c in range(num_channels):
            if c < len(self.channelControls) and self.channelControls[c][0].isChecked():
                opacity = self.channelControls[c][1].value() / 100
                channel_data = self.data[t, c]

                self.logger.debug(f"Channel {c} data shape: {channel_data.shape}")
                self.logger.debug(f"Channel {c} data range: {channel_data.min()} to {channel_data.max()}")
                self.logger.debug(f"Channel {c} non-zero elements: {np.count_nonzero(channel_data)}")

                # Normalize channel data
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)

                channel_data[channel_data < threshold] = 0

                # Apply color to the channel
                color = channel_colors[c % len(channel_colors)]
                colored_data = channel_data[:, :, :, np.newaxis] * color

                combined_xy += colored_data * opacity
                combined_xz += np.transpose(colored_data, (1, 0, 2, 3)) * opacity
                combined_yz += np.transpose(colored_data, (2, 0, 1, 3)) * opacity

        # Clip values to [0, 1] range
        combined_xy = np.clip(combined_xy, 0, 1)
        combined_xz = np.clip(combined_xz, 0, 1)
        combined_yz = np.clip(combined_yz, 0, 1)

        self.imageViewXY.setImage(combined_xy, autoLevels=False, levels=[0, 1])
        self.imageViewXZ.setImage(combined_xz, autoLevels=False, levels=[0, 1])
        self.imageViewYZ.setImage(combined_yz, autoLevels=False, levels=[0, 1])

        self.updateSliceMarkers()

    def create3DVisualization(self):
        try:
            for item in self.data_items:
                self.glView.removeItem(item)
            self.data_items.clear()

            t = self.timeSlider.value()
            threshold = self.thresholdSpinBox.value()
            self.logger.debug(f"Current threshold: {threshold}")

            num_channels = self.data.shape[1]

            channel_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # RGBA

            for c in range(num_channels):
                if c < len(self.channelControls) and self.channelControls[c][0].isChecked():
                    opacity = self.channelControls[c][1].value() / 100
                    volume_data = self.data[t, c]

                    self.logger.debug(f"3D Channel {c} data shape: {volume_data.shape}")
                    self.logger.debug(f"3D Channel {c} data range: {volume_data.min()} to {volume_data.max()}")
                    self.logger.debug(f"3D Channel {c} non-zero elements: {np.count_nonzero(volume_data)}")

                    # Get 3D coordinates of all points above threshold
                    z, y, x = np.where(volume_data > threshold)
                    pos = np.column_stack((x, y, z))

                    if len(pos) > 0:
                        colors = np.tile(channel_colors[c], (len(pos), 1))
                        colors[:, 3] = opacity * (volume_data[z, y, x] - volume_data.min()) / (volume_data.max() - volume_data.min())

                        scatter = gl.GLScatterPlotItem(pos=pos, color=colors, size=self.scatterPointSizeSpinBox.value())
                        self.glView.addItem(scatter)
                        self.data_items.append(scatter)

            self.glView.update()

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

    def updateScatterPointSize(self, value):
        self.updateViews()
        self.create3DVisualization()

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
        # if value >= 1:
        #     self.logger.warning("Array compute error")
        #     return

        if self.data is not None:
            self.updateViews()
            self.create3DVisualization()
        else:
            self.logger.warning("No data available to update display threshold")

    def updateBlobThreshold(self, value):
       self.filter_blobs()

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
                self.autoScaleViews()  # Add this line
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

            # Update channel controls
            num_channels = self.data.shape[1]

            # Clear existing channel controls
            for i in reversed(range(self.channelControlsLayout.count())):
                self.channelControlsLayout.itemAt(i).widget().setParent(None)
            self.channelControls.clear()

            for i in range(num_channels):
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

                channelWidget = QWidget()
                channelWidget.setLayout(channelLayout)
                self.channelControlsLayout.addWidget(channelWidget)
                self.logger.debug(f"Created control for channel {i}")

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

        # Make sure to close the biological simulation window when the main window is closed
        if self.biological_simulation_window:
            self.biological_simulation_window.close()

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
        #threshold = self.blobThresholdSpinBox.value()

        all_blobs = []
        for channel in range(num_channels):
            for t in range(max_time):
                # Get the current 3D volume for this channel
                volume = self.data[t, channel]

                # Detect blobs
                blobs = blob_log(volume, max_sigma=max_sigma, num_sigma=num_sigma)

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

        # Convert to numpy array
        all_blobs = np.array(all_blobs)

        # Store the original blob data
        self.original_blobs = all_blobs

        # Filter blobs based on threshold -> creates self.all_detected_blobs
        self.filter_blobs()

        self.logger.info(f"Detected {len(all_blobs)} blobs across all channels")

        # Store all detected blobs
        #self.all_detected_blobs = all_blobs

        # Display results
        self.display_blob_results(self.all_detected_blobs)

        # Visualize blobs
        self.updateBlobVisualization()

        # Show the blob results button and update its text
        self.showBlobResultsButton.setVisible(True)
        self.showBlobResultsButton.setText("Show Blob Results")

        return all_blobs


    def filter_blobs(self):
        if not hasattr(self, 'original_blobs'):
            return

        threshold = self.blobThresholdSpinBox.value()
        self.all_detected_blobs = self.original_blobs[self.original_blobs[:, 6] > threshold]

        # Update visualization
        self.updateBlobVisualization()


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

        # Clear previous visualizations
        for item in self.blob_items:
            self.blobGLView.removeItem(item)
        self.blob_items.clear()

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

    def autoScaleViews(self):
        if self.data is None:
            return

        # Get the bounds of the data
        z, y, x = self.data.shape[2:]  # Assuming shape is (t, c, z, y, x)
        center = QVector3D(x/2, y/2, z/2)

        # Calculate the diagonal of the bounding box
        diagonal = np.sqrt(x**2 + y**2 + z**2)

        # Set the camera position for both views
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(pos=center, distance=diagonal*1.2, elevation=30, azimuth=45)
            view.opts['center'] = center

        self.glView.update()
        self.blobGLView.update()



    def connectViewEvents(self):
        for view in [self.glView, self.blobGLView]:
            if view is not None:
                view.installEventFilter(self)
        self.logger.debug("View events connected")

    def eventFilter(self, source, event):
        if source in [self.glView, self.blobGLView]:
            if event.type() == QEvent.MouseButtonPress:
                self.on3DViewMousePress(event, source)
                return True
            elif event.type() == QEvent.MouseButtonRelease:
                self.on3DViewMouseRelease(event, source)
                return True
            elif event.type() == QEvent.MouseMove:
                self.on3DViewMouseMove(event, source)
                return True
            elif event.type() == QEvent.Wheel:
                self.on3DViewWheel(event, source)
                return True
        return super().eventFilter(source, event)

    @pyqtSlot(QEvent)
    def on3DViewMousePress(self, event, source):
        self.lastPos = event.pos()
        self.logger.debug(f"Mouse press event on {source}")

    @pyqtSlot(QEvent)
    def on3DViewMouseRelease(self, event, source):
        self.lastPos = None
        self.logger.debug(f"Mouse release event on {source}")

    @pyqtSlot(QEvent)
    def on3DViewMouseMove(self, event, source):
        if self.lastPos is None:
            return

        diff = event.pos() - self.lastPos
        self.lastPos = event.pos()

        self.logger.debug(f"Mouse move event on {source}")

        if event.buttons() == Qt.LeftButton:
            self.rotate3DViews(diff.x(), diff.y(), source)
        elif event.buttons() == Qt.MidButton:
            self.pan3DViews(diff.x(), diff.y(), source)

    @pyqtSlot(QEvent)
    def on3DViewWheel(self, event, source):
        delta = event.angleDelta().y()
        self.logger.debug(f"Wheel event on {source}")
        self.zoom3DViews(delta, source)

    def rotate3DViews(self, dx, dy, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'opts'):
                view.opts['elevation'] -= dy * 0.5
                view.opts['azimuth'] += dx * 0.5
                view.update()
            else:
                self.logger.error(f"Invalid view object: {view}")

    def pan3DViews(self, dx, dy, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'pan'):
                view.pan(dx, dy, 0, relative='view')
            else:
                self.logger.error(f"Invalid view object for panning: {view}")

    def zoom3DViews(self, delta, active_view):
        views_to_update = [active_view]
        if self.syncViewsCheck.isChecked():
            views_to_update = [self.glView, self.blobGLView]

        for view in views_to_update:
            if view is not None and hasattr(view, 'opts'):
                view.opts['fov'] *= 0.999**delta
                view.update()
            else:
                self.logger.error(f"Invalid view object for zooming: {view}")

    def check_view_state(self):
        self.logger.debug(f"glView state: {self.glView}, has opts: {hasattr(self.glView, 'opts')}")
        self.logger.debug(f"blobGLView state: {self.blobGLView}, has opts: {hasattr(self.blobGLView, 'opts')}")
        self.logger.debug(f"Sync checked: {self.syncViewsCheck.isChecked()}")


    def setTopDownView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=90, azimuth=0)
            view.update()

    def setSideView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=0, azimuth=0)
            view.update()

    def setFrontView(self):
        for view in [self.glView, self.blobGLView]:
            view.setCameraPosition(elevation=0, azimuth=90)
            view.update()

    def setupChannelControlsWidget(self):
        self.channelControlsWidget = QWidget()
        self.channelControlsLayout = QVBoxLayout(self.channelControlsWidget)
        self.channelControls = []

##############################################################################

def main():
    app = QApplication(sys.argv)
    viewer = LightsheetViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
