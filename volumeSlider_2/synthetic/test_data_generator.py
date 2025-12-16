#!/usr/bin/env python3
"""
Synthetic Test Data Generator
============================

Generates realistic synthetic fluorescence microscopy data for testing
the Volume Slider plugin. Includes various biological patterns and noise models.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import scipy.ndimage as ndi
from scipy import stats
from qtpy.QtCore import QObject, Signal


class PatternType(Enum):
    """Types of synthetic patterns to generate."""
    CALCIUM_PUFFS = "calcium_puffs"
    CALCIUM_WAVES = "calcium_waves"
    NEURON_DENDRITES = "neuron_dendrites"
    CELL_MIGRATION = "cell_migration"
    VESICLE_TRANSPORT = "vesicle_transport"
    TISSUE_STRUCTURE = "tissue_structure"
    RANDOM_BLOBS = "random_blobs"
    LIGHTSHEET_BEADS = "lightsheet_beads"


@dataclass
class GenerationParameters:
    """Parameters for synthetic data generation."""

    # Volume dimensions
    volume_shape: Tuple[int, int, int] = (50, 256, 256)  # Z, Y, X
    n_timepoints: int = 100

    # Pattern parameters
    pattern_type: PatternType = PatternType.CALCIUM_PUFFS
    n_patterns: int = 20
    pattern_intensity_range: Tuple[float, float] = (0.5, 2.0)
    pattern_size_range: Tuple[float, float] = (2.0, 8.0)

    # Temporal dynamics
    temporal_dynamics: bool = True
    dynamics_speed: float = 1.0
    dynamics_variability: float = 0.3

    # Noise parameters
    photon_noise: bool = True
    gaussian_noise_level: float = 0.05
    baseline_level: float = 0.1

    # Optical parameters
    psf_size: Tuple[float, float, float] = (2.0, 1.0, 1.0)  # Z, Y, X PSF sigma
    photobleaching: bool = True
    photobleaching_rate: float = 0.001

    # Output parameters
    output_dtype: np.dtype = np.float32
    normalize_output: bool = True


class TestDataGenerator(QObject):
    """
    Comprehensive synthetic data generator for fluorescence microscopy.

    Features:
    - Multiple biological patterns (Ca²⁺ puffs, waves, neurons, etc.)
    - Realistic noise models (photon, Gaussian, baseline)
    - Temporal dynamics with customizable speeds
    - Point spread function simulation
    - Photobleaching effects
    - Customizable parameters via GUI
    """

    # Signals for progress reporting
    progress_updated = Signal(int, str)  # percentage, status message
    generation_completed = Signal(np.ndarray, dict)  # data, metadata
    generation_failed = Signal(str)  # error message

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.current_parameters: Optional[GenerationParameters] = None
        self._is_generating = False

    def generate_data(self, params: GenerationParameters) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic data with given parameters.

        Args:
            params: Generation parameters

        Returns:
            Tuple of (data_array, metadata_dict)
        """
        try:
            self._is_generating = True
            self.current_parameters = params

            self.logger.info(f"Starting data generation: {params.pattern_type.value}, "
                           f"shape={params.volume_shape}, timepoints={params.n_timepoints}")

            self.progress_updated.emit(0, "Initializing generation...")

            # Initialize volume
            total_shape = (params.volume_shape[0], params.n_timepoints,
                          params.volume_shape[1], params.volume_shape[2])
            data = np.zeros(total_shape, dtype=np.float64)

            # Add baseline
            if params.baseline_level > 0:
                data += params.baseline_level
                self.progress_updated.emit(10, "Adding baseline...")

            # Generate pattern-specific structures
            self.progress_updated.emit(20, f"Generating {params.pattern_type.value} patterns...")
            data = self._add_patterns(data, params)

            # Add temporal dynamics
            if params.temporal_dynamics:
                self.progress_updated.emit(50, "Adding temporal dynamics...")
                data = self._add_temporal_dynamics(data, params)

            # Apply point spread function
            self.progress_updated.emit(70, "Applying point spread function...")
            data = self._apply_psf(data, params)

            # Add photobleaching
            if params.photobleaching:
                self.progress_updated.emit(80, "Applying photobleaching...")
                data = self._apply_photobleaching(data, params)

            # Add noise
            self.progress_updated.emit(90, "Adding noise...")
            data = self._add_noise(data, params)

            # Final processing
            self.progress_updated.emit(95, "Final processing...")
            if params.normalize_output:
                data = self._normalize_data(data)

            # Convert to requested dtype
            data = data.astype(params.output_dtype)

            # Generate metadata
            metadata = self._generate_metadata(params, data)

            self.progress_updated.emit(100, "Generation complete!")
            self.logger.info("Data generation completed successfully")

            self._is_generating = False
            return data, metadata

        except Exception as e:
            self._is_generating = False
            error_msg = f"Data generation failed: {str(e)}"
            self.logger.error(error_msg)
            self.generation_failed.emit(error_msg)
            raise

    def _add_patterns(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Add pattern-specific structures to the data."""

        pattern_generators = {
            PatternType.CALCIUM_PUFFS: self._generate_calcium_puffs,
            PatternType.CALCIUM_WAVES: self._generate_calcium_waves,
            PatternType.NEURON_DENDRITES: self._generate_neuron_dendrites,
            PatternType.CELL_MIGRATION: self._generate_cell_migration,
            PatternType.VESICLE_TRANSPORT: self._generate_vesicle_transport,
            PatternType.TISSUE_STRUCTURE: self._generate_tissue_structure,
            PatternType.RANDOM_BLOBS: self._generate_random_blobs,
            PatternType.LIGHTSHEET_BEADS: self._generate_lightsheet_beads
        }

        generator = pattern_generators.get(params.pattern_type)
        if generator:
            return generator(data, params)
        else:
            self.logger.warning(f"Unknown pattern type: {params.pattern_type}")
            return data

    def _generate_calcium_puffs(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate calcium puff patterns."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns):
            # Random location and properties
            z_center = np.random.randint(5, z_size - 5)
            y_center = np.random.randint(20, y_size - 20)
            x_center = np.random.randint(20, x_size - 20)

            intensity = np.random.uniform(*params.pattern_intensity_range)
            size = np.random.uniform(*params.pattern_size_range)

            # Temporal properties
            onset_time = np.random.randint(5, t_size - 20)
            duration = np.random.randint(5, 15)

            # Create 3D Gaussian puff
            zz, yy, xx = np.mgrid[0:z_size, 0:y_size, 0:x_size]
            gaussian_3d = np.exp(-((zz - z_center)**2 / (2 * size**2) +
                                 (yy - y_center)**2 / (2 * size**2) +
                                 (xx - x_center)**2 / (2 * size**2)))

            # Temporal profile (fast rise, exponential decay)
            for t in range(max(0, onset_time), min(t_size, onset_time + duration * 3)):
                if t < onset_time + duration:
                    # Rising phase
                    temporal_weight = (t - onset_time) / duration
                else:
                    # Decay phase
                    temporal_weight = np.exp(-(t - onset_time - duration) / duration)

                data[:, t, :, :] += intensity * temporal_weight * gaussian_3d

        return data

    def _generate_calcium_waves(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate calcium wave patterns."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns // 2):  # Fewer waves than puffs
            # Wave origin
            y_origin = np.random.randint(20, y_size - 20)
            x_origin = np.random.randint(20, x_size - 20)
            z_plane = np.random.randint(10, z_size - 10)

            wave_speed = np.random.uniform(0.5, 2.0)  # pixels per timepoint
            intensity = np.random.uniform(*params.pattern_intensity_range)
            onset_time = np.random.randint(5, t_size // 2)

            # Create expanding wave
            yy, xx = np.mgrid[0:y_size, 0:x_size]

            for t in range(onset_time, t_size):
                wave_radius = wave_speed * (t - onset_time)

                # Distance from origin
                distance = np.sqrt((yy - y_origin)**2 + (xx - x_origin)**2)

                # Wave profile (ring with decay)
                wave_thickness = 5.0
                wave_profile = np.exp(-((distance - wave_radius)**2 / (2 * wave_thickness**2)))

                # Decay over time
                time_decay = np.exp(-(t - onset_time) / 20.0)

                data[z_plane, t, :, :] += intensity * wave_profile * time_decay

        return data

    def _generate_neuron_dendrites(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate neuron dendrite-like structures."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns // 3):  # Fewer, larger structures
            # Starting point
            z_start = np.random.randint(5, z_size - 5)
            y_start = np.random.randint(20, y_size - 20)
            x_start = np.random.randint(20, x_size - 20)

            intensity = np.random.uniform(*params.pattern_intensity_range)

            # Create branching structure
            current_points = [(z_start, y_start, x_start)]

            for step in range(20):  # Growth steps
                new_points = []

                for z, y, x in current_points:
                    # Random walk with bias
                    for branch in range(np.random.randint(1, 3)):
                        dz = np.random.randint(-1, 2)
                        dy = np.random.randint(-2, 3)
                        dx = np.random.randint(-2, 3)

                        new_z = np.clip(z + dz, 0, z_size - 1)
                        new_y = np.clip(y + dy, 0, y_size - 1)
                        new_x = np.clip(x + dx, 0, x_size - 1)

                        new_points.append((new_z, new_y, new_x))

                        # Add to all timepoints with some variability
                        for t in range(t_size):
                            temporal_intensity = intensity * (0.8 + 0.4 * np.random.random())
                            data[new_z, t, new_y, new_x] += temporal_intensity

                current_points = new_points
                if len(current_points) > 50:  # Prevent explosion
                    break

        return data

    def _generate_cell_migration(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate migrating cell-like objects."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns // 4):  # Few migrating objects
            # Initial position
            z_pos = np.random.randint(5, z_size - 5)
            y_pos = np.random.uniform(30, y_size - 30)
            x_pos = np.random.uniform(30, x_size - 30)

            # Migration direction and speed
            y_velocity = np.random.uniform(-0.5, 0.5)
            x_velocity = np.random.uniform(-0.5, 0.5)

            intensity = np.random.uniform(*params.pattern_intensity_range)
            cell_size = np.random.uniform(8, 15)

            for t in range(t_size):
                # Update position
                y_pos += y_velocity + np.random.normal(0, 0.1)
                x_pos += x_velocity + np.random.normal(0, 0.1)

                # Boundary conditions
                y_pos = np.clip(y_pos, cell_size, y_size - cell_size)
                x_pos = np.clip(x_pos, cell_size, x_size - cell_size)

                # Create cell body
                yy, xx = np.mgrid[0:y_size, 0:x_size]
                cell_profile = np.exp(-((yy - y_pos)**2 + (xx - x_pos)**2) / (2 * cell_size**2))

                data[z_pos, t, :, :] += intensity * cell_profile

        return data

    def _generate_vesicle_transport(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate vesicle transport along tracks."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns):
            # Track parameters
            z_track = np.random.randint(2, z_size - 2)
            y_start = np.random.randint(10, y_size - 10)
            x_start = np.random.randint(10, x_size - 10)
            y_end = np.random.randint(10, y_size - 10)
            x_end = np.random.randint(10, x_size - 10)

            vesicle_size = np.random.uniform(1, 3)
            intensity = np.random.uniform(*params.pattern_intensity_range)

            # Transport timing
            start_time = np.random.randint(0, t_size // 3)
            transport_duration = np.random.randint(20, t_size - start_time)

            for t in range(start_time, min(t_size, start_time + transport_duration)):
                # Interpolate position along track
                progress = (t - start_time) / transport_duration
                y_pos = y_start + progress * (y_end - y_start)
                x_pos = x_start + progress * (x_end - x_start)

                # Add some randomness
                y_pos += np.random.normal(0, 0.5)
                x_pos += np.random.normal(0, 0.5)

                # Create vesicle
                yy, xx = np.mgrid[max(0, int(y_pos-5)):min(y_size, int(y_pos+6)),
                                 max(0, int(x_pos-5)):min(x_size, int(x_pos+6))]

                if yy.size > 0 and xx.size > 0:
                    vesicle = np.exp(-((yy - y_pos)**2 + (xx - x_pos)**2) / (2 * vesicle_size**2))
                    y_slice = slice(max(0, int(y_pos-5)), min(y_size, int(y_pos+6)))
                    x_slice = slice(max(0, int(x_pos-5)), min(x_size, int(x_pos+6)))
                    data[z_track, t, y_slice, x_slice] += intensity * vesicle

        return data

    def _generate_tissue_structure(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate tissue-like structural patterns."""
        z_size, t_size, y_size, x_size = data.shape

        # Create background tissue structure
        tissue_intensity = params.pattern_intensity_range[0] * 0.3

        # Generate Perlin-like noise for tissue texture
        for z in range(z_size):
            tissue_slice = self._generate_perlin_noise_2d(y_size, x_size, scale=20)
            tissue_slice = (tissue_slice + 1) / 2  # Normalize to [0, 1]

            for t in range(t_size):
                data[z, t, :, :] += tissue_intensity * tissue_slice

        # Add vessel-like structures
        for i in range(params.n_patterns // 2):
            self._add_vessel_structure(data, params)

        return data

    def _generate_random_blobs(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate simple random blob patterns."""
        z_size, t_size, y_size, x_size = data.shape

        for i in range(params.n_patterns):
            z_center = np.random.randint(2, z_size - 2)
            y_center = np.random.randint(10, y_size - 10)
            x_center = np.random.randint(10, x_size - 10)

            intensity = np.random.uniform(*params.pattern_intensity_range)
            size = np.random.uniform(*params.pattern_size_range)

            # 3D Gaussian blob
            zz, yy, xx = np.mgrid[0:z_size, 0:y_size, 0:x_size]
            blob = intensity * np.exp(-((zz - z_center)**2 / (2 * size**2) +
                                      (yy - y_center)**2 / (2 * size**2) +
                                      (xx - x_center)**2 / (2 * size**2)))

            # Add to all timepoints
            for t in range(t_size):
                data[:, t, :, :] += blob

        return data

    def _generate_lightsheet_beads(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Generate fluorescent bead patterns for lightsheet calibration."""
        z_size, t_size, y_size, x_size = data.shape

        # Regular grid of beads with some randomness
        bead_spacing = 20
        bead_intensity = params.pattern_intensity_range[1]
        bead_size = 2.0

        for z in range(bead_spacing//2, z_size, bead_spacing//3):
            for y in range(bead_spacing, y_size, bead_spacing):
                for x in range(bead_spacing, x_size, bead_spacing):
                    # Add some randomness to position
                    z_pos = z + np.random.randint(-3, 4)
                    y_pos = y + np.random.randint(-5, 6)
                    x_pos = x + np.random.randint(-5, 6)

                    z_pos = np.clip(z_pos, 1, z_size - 2)
                    y_pos = np.clip(y_pos, 5, y_size - 6)
                    x_pos = np.clip(x_pos, 5, x_size - 6)

                    # Create bead
                    zz, yy, xx = np.mgrid[0:z_size, 0:y_size, 0:x_size]
                    bead = bead_intensity * np.exp(-((zz - z_pos)**2 / (2 * bead_size**2) +
                                                   (yy - y_pos)**2 / (2 * bead_size**2) +
                                                   (xx - x_pos)**2 / (2 * bead_size**2)))

                    # Add to all timepoints with slight variation
                    for t in range(t_size):
                        variation = 1.0 + np.random.normal(0, 0.1)
                        data[:, t, :, :] += variation * bead

        return data

    def _add_temporal_dynamics(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Add temporal dynamics to static patterns."""
        z_size, t_size, y_size, x_size = data.shape

        # Create temporal modulation
        for z in range(z_size):
            for y in range(0, y_size, 10):  # Sample every 10th pixel for efficiency
                for x in range(0, x_size, 10):
                    if data[z, 0, y, x] > params.baseline_level:  # Only modulate active regions
                        # Random temporal profile
                        frequency = np.random.uniform(0.01, 0.1) * params.dynamics_speed
                        phase = np.random.uniform(0, 2 * np.pi)
                        amplitude = params.dynamics_variability

                        # Apply to local region
                        for dy in range(-5, 6):
                            for dx in range(-5, 6):
                                if 0 <= y+dy < y_size and 0 <= x+dx < x_size:
                                    modulation = 1 + amplitude * np.sin(
                                        2 * np.pi * frequency * np.arange(t_size) + phase
                                    )
                                    data[z, :, y+dy, x+dx] *= modulation

        return data

    def _apply_psf(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Apply point spread function blurring."""
        if all(s <= 0 for s in params.psf_size):
            return data

        # Apply 3D Gaussian blur to each timepoint
        for t in range(data.shape[1]):
            data[:, t, :, :] = ndi.gaussian_filter(
                data[:, t, :, :],
                sigma=params.psf_size
            )

        return data

    def _apply_photobleaching(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Apply exponential photobleaching."""
        t_size = data.shape[1]

        # Exponential decay over time
        bleach_curve = np.exp(-params.photobleaching_rate * np.arange(t_size))

        for t in range(t_size):
            data[:, t, :, :] *= bleach_curve[t]

        return data

    def _add_noise(self, data: np.ndarray, params: GenerationParameters) -> np.ndarray:
        """Add realistic noise to the data."""
        # Gaussian noise
        if params.gaussian_noise_level > 0:
            noise = np.random.normal(0, params.gaussian_noise_level, data.shape)
            data += noise

        # Photon noise (Poisson statistics)
        if params.photon_noise:
            # Scale data to reasonable photon counts
            photon_scale = 1000
            data_photons = data * photon_scale
            data_photons = np.maximum(data_photons, 0)  # Ensure non-negative

            # Apply Poisson noise
            data_noisy = np.random.poisson(data_photons)
            data = data_noisy.astype(np.float64) / photon_scale

        return data

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to reasonable range."""
        # Remove negative values
        data = np.maximum(data, 0)

        # Normalize to [0, 1] range
        data_max = np.percentile(data, 99.9)  # Use percentile to avoid outliers
        if data_max > 0:
            data = data / data_max

        return data

    def _generate_metadata(self, params: GenerationParameters, data: np.ndarray) -> Dict[str, Any]:
        """Generate comprehensive metadata for the synthetic data."""
        return {
            'data_type': 'synthetic',
            'generator_version': '2.0.0',
            'generation_timestamp': np.datetime64('now').astype(str),
            'pattern_type': params.pattern_type.value,
            'parameters': {
                'volume_shape': params.volume_shape,
                'n_timepoints': params.n_timepoints,
                'n_patterns': params.n_patterns,
                'pattern_intensity_range': params.pattern_intensity_range,
                'pattern_size_range': params.pattern_size_range,
                'temporal_dynamics': params.temporal_dynamics,
                'dynamics_speed': params.dynamics_speed,
                'photon_noise': params.photon_noise,
                'gaussian_noise_level': params.gaussian_noise_level,
                'baseline_level': params.baseline_level,
                'psf_size': params.psf_size,
                'photobleaching': params.photobleaching,
                'photobleaching_rate': params.photobleaching_rate
            },
            'data_properties': {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min_value': float(np.min(data)),
                'max_value': float(np.max(data)),
                'mean_value': float(np.mean(data)),
                'std_value': float(np.std(data))
            }
        }

    def _generate_perlin_noise_2d(self, height: int, width: int, scale: float = 10.0) -> np.ndarray:
        """Generate 2D Perlin-like noise."""
        # Simple implementation of Perlin-like noise
        # For production, consider using dedicated noise libraries

        # Create random gradients
        gradient_x = np.random.randn(height // scale + 2, width // scale + 2)
        gradient_y = np.random.randn(height // scale + 2, width // scale + 2)

        # Interpolate to full size
        from scipy.interpolate import RectBivariateSpline

        x_grad = np.linspace(0, height // scale + 1, height // scale + 2)
        y_grad = np.linspace(0, width // scale + 1, width // scale + 2)

        spline_x = RectBivariateSpline(x_grad, y_grad, gradient_x, kx=1, ky=1)
        spline_y = RectBivariateSpline(x_grad, y_grad, gradient_y, kx=1, ky=1)

        x_full = np.linspace(0, height // scale + 1, height)
        y_full = np.linspace(0, width // scale + 1, width)

        noise = spline_x(x_full, y_full) + spline_y(x_full, y_full)

        return noise

    def _add_vessel_structure(self, data: np.ndarray, params: GenerationParameters):
        """Add vessel-like tubular structures."""
        z_size, t_size, y_size, x_size = data.shape

        # Create random vessel path
        n_points = np.random.randint(5, 15)
        z_path = np.random.randint(2, z_size - 2, n_points)
        y_path = np.random.randint(10, y_size - 10, n_points)
        x_path = np.random.randint(10, x_size - 10, n_points)

        # Smooth the path
        from scipy.interpolate import interp1d
        t_points = np.linspace(0, 1, n_points)
        t_smooth = np.linspace(0, 1, n_points * 5)

        f_z = interp1d(t_points, z_path, kind='cubic')
        f_y = interp1d(t_points, y_path, kind='cubic')
        f_x = interp1d(t_points, x_path, kind='cubic')

        z_smooth = f_z(t_smooth)
        y_smooth = f_y(t_smooth)
        x_smooth = f_x(t_smooth)

        # Draw vessel
        vessel_radius = np.random.uniform(1, 3)
        vessel_intensity = params.pattern_intensity_range[0] * 0.5

        for i in range(len(z_smooth)):
            z_pos = int(np.clip(z_smooth[i], 0, z_size - 1))
            y_pos = int(np.clip(y_smooth[i], 0, y_size - 1))
            x_pos = int(np.clip(x_smooth[i], 0, x_size - 1))

            # Create vessel cross-section
            yy, xx = np.mgrid[max(0, y_pos-5):min(y_size, y_pos+6),
                             max(0, x_pos-5):min(x_size, x_pos+6)]

            if yy.size > 0 and xx.size > 0:
                vessel_cross = np.exp(-((yy - y_pos)**2 + (xx - x_pos)**2) /
                                    (2 * vessel_radius**2))

                for t in range(t_size):
                    y_slice = slice(max(0, y_pos-5), min(y_size, y_pos+6))
                    x_slice = slice(max(0, x_pos-5), min(x_size, x_pos+6))
                    data[z_pos, t, y_slice, x_slice] += vessel_intensity * vessel_cross

    def get_default_parameters(self) -> GenerationParameters:
        """Get default generation parameters."""
        return GenerationParameters()

    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern types."""
        return [pattern.value for pattern in PatternType]

    def is_generating(self) -> bool:
        """Check if data generation is in progress."""
        return self._is_generating


# Convenience functions
def generate_calcium_puffs_data(shape: Tuple[int, int, int] = (30, 100, 128, 128)) -> Tuple[np.ndarray, Dict]:
    """Quick function to generate calcium puffs test data."""
    generator = TestDataGenerator()
    params = GenerationParameters(
        volume_shape=shape[:3] if len(shape) == 4 else shape,
        n_timepoints=shape[1] if len(shape) == 4 else 50,
        pattern_type=PatternType.CALCIUM_PUFFS,
        n_patterns=25
    )
    return generator.generate_data(params)


def generate_lightsheet_beads_data(shape: Tuple[int, int, int] = (50, 10, 256, 256)) -> Tuple[np.ndarray, Dict]:
    """Quick function to generate lightsheet bead calibration data."""
    generator = TestDataGenerator()
    params = GenerationParameters(
        volume_shape=shape[:3] if len(shape) == 4 else shape,
        n_timepoints=shape[1] if len(shape) == 4 else 10,
        pattern_type=PatternType.LIGHTSHEET_BEADS,
        photon_noise=True,
        gaussian_noise_level=0.02,
        temporal_dynamics=False
    )
    return generator.generate_data(params)
