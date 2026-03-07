# pytrack/config.py
"""
Configuration management for PyTrack

This module handles all configuration parameters for detection, tracking,
classification, and visualization. Parameters can be loaded from YAML files,
modified programmatically, and validated for consistency.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """Configuration parameters for particle detection."""

    # Method selection
    method: str = "wavelet"  # "wavelet", "blob_log", "blob_dog", "blob_doh"

    # Wavelet detection parameters (à trous method)
    wavelet_scales: list = field(default_factory=lambda: [1, 2, 3])
    wavelet_threshold_factor: float = 3.0  # Factor * std for thresholding
    wavelet_min_area: int = 4  # Minimum spot area in pixels

    # Blob detection parameters
    blob_min_sigma: float = 1.0
    blob_max_sigma: float = 5.0
    blob_num_sigma: int = 10
    blob_threshold: float = 0.01
    blob_overlap: float = 0.5

    # Sub-pixel localization
    subpixel_localization: bool = True
    gaussian_fit_window: int = 5  # Window size for Gaussian fitting

    # Intensity measurement
    intensity_radius: float = 2.0  # Radius for intensity measurement
    background_estimation: str = "local"  # "local", "global", "none"
    background_radius: float = 5.0

    # Pre-processing
    gaussian_filter_sigma: float = 0.0  # 0 = no filtering

    # Quality control
    min_intensity: float = 0.0
    max_intensity: float = np.inf
    min_spot_size: float = 1.0
    max_spot_size: float = 20.0



@dataclass
class TrackingConfig:
    """Configuration parameters for particle tracking with multiple linking methods."""

    # Linking method selection
    linking_method: str = "utrack_lap"  # Options: utrack_lap, lap_package, scipy_lap, greedy, enhanced_greedy, hybrid, experimental_deep, experimental_graph, experimental_mcmc

    # Frame-to-frame linking parameters
    max_linking_distance: float = 5.0  # Maximum distance for linking
    linking_distance_auto: bool = True  # Auto-determine distance cutoff
    linking_percentile: float = 90.0  # Percentile for auto distance

    # Gap closing parameters
    max_gap_closing_distance: float = 8.0
    max_gap_frames: int = 5  # Maximum frames to bridge
    gap_closing_auto: bool = True
    gap_percentile: float = 95.0

    # Merging and splitting parameters
    max_merge_split_distance: float = 3.0
    enable_merging: bool = True
    enable_splitting: bool = True
    merge_split_auto: bool = True

    # Alternative cost penalties
    alternative_cost_factor: float = 1.05  # Factor for birth/death costs

    # Motion models
    motion_model: str = "brownian"  # "brownian", "linear", "adaptive"
    diffusion_radius: float = 2.0  # For Brownian motion model

    # Track quality
    min_track_length: int = 3  # Minimum track length to keep

    # LAP solver settings
    lap_solver: str = "scipy"  # "scipy", "lapjv", "lap_package"

    # Intensity-based costs (for merging/splitting)
    use_intensity_costs: bool = True
    intensity_weight: float = 0.1

    # Velocity-based costs (for adaptive motion)
    use_velocity_costs: bool = False
    velocity_weight: float = 0.1
    velocity_memory_frames: int = 3  # Frames to remember for velocity calculation

    # Advanced linking options
    linking_max_iterations: int = 1  # For iterative methods
    linking_convergence_threshold: float = 0.01  # For iterative methods

    # Method-specific parameters
    greedy_optimization_rounds: int = 2  # For enhanced greedy
    hybrid_method_timeout: float = 10.0  # Timeout for each method in hybrid

    # Experimental method parameters
    experimental_param_1: float = 1.0  # Generic parameter for experimental methods
    experimental_param_2: int = 10     # Generic parameter for experimental methods

    # Debug and performance options
    enable_linking_debug: bool = False  # Enable detailed debug output
    benchmark_methods: bool = False     # Test all methods and compare
    save_linking_costs: bool = False    # Save cost matrices for analysis


@dataclass
class ClassificationConfig:
    """Configuration parameters for track classification."""

    # Motion classification
    classify_motion: bool = True
    motion_classification_method: str = "msd"  # "msd", "velocity", "combined"

    # MSD-based classification
    msd_fitting_points: int = 10  # Number of points for MSD fitting
    msd_min_points: int = 5
    diffusion_threshold: float = 0.1  # D coefficient threshold

    # Velocity-based classification
    velocity_window: int = 5  # Window for velocity calculation
    directed_velocity_threshold: float = 0.5  # pixels/frame

    # Track quality scoring
    quality_scoring: bool = True
    quality_method: str = "combined"  # "length", "gaps", "intensity", "combined"

    # Quality thresholds
    min_quality_score: float = 0.3
    gap_penalty: float = 0.1
    intensity_consistency_weight: float = 0.2


@dataclass
class VisualizationConfig:
    """Configuration parameters for visualization."""

    # Display settings
    colormap: str = "viridis"  # Colormap for tracks
    track_line_width: float = 2.0
    particle_size: float = 3.0
    particle_opacity: float = 0.7

    # Track coloring
    color_by: str = "track_id"  # "track_id", "frame", "velocity", "classification"
    track_colors: list = field(default_factory=lambda: [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"
    ])

    # Time display
    show_tails: bool = True
    tail_length: int = 10  # Number of previous positions to show
    fade_tails: bool = True

    # Overlay settings
    show_detections: bool = True
    show_tracks: bool = True
    show_ids: bool = False
    show_frame_number: bool = True

    # Performance
    max_tracks_display: int = 1000  # Limit for performance
    update_rate: float = 30.0  # FPS for playback


@dataclass
class LoggingConfig:
    """Configuration parameters for logging."""

    # Logging levels
    console_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    file_level: str = "DEBUG"

    # Log files
    log_to_file: bool = True
    log_directory: str = "logs"
    log_filename: str = "pytrack.log"
    max_log_size: str = "10MB"
    backup_count: int = 5

    # Log format
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # Module-specific logging
    detection_debug: bool = False
    tracking_debug: bool = False
    classification_debug: bool = False


class PyTrackConfig:
    """Main configuration class for PyTrack."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML configuration file
        """
        # Initialize with defaults
        self.detection = DetectionConfig()
        self.tracking = TrackingConfig()
        self.classification = ClassificationConfig()
        self.visualization = VisualizationConfig()
        self.logging = LoggingConfig()

        # Load from file if provided
        if config_file is not None:
            self.load_from_file(config_file)

    def load_from_file(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)

            # Update configuration sections
            if 'detection' in config_data:
                self._update_config(self.detection, config_data['detection'])
            if 'tracking' in config_data:
                self._update_config(self.tracking, config_data['tracking'])
            if 'classification' in config_data:
                self._update_config(self.classification, config_data['classification'])
            if 'visualization' in config_data:
                self._update_config(self.visualization, config_data['visualization'])
            if 'logging' in config_data:
                self._update_config(self.logging, config_data['logging'])

            logger.info(f"Configuration loaded from {config_file}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def save_to_file(self, config_file: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_file: Path to save configuration
        """
        config_data = {
            'detection': self._config_to_dict(self.detection),
            'tracking': self._config_to_dict(self.tracking),
            'classification': self._config_to_dict(self.classification),
            'visualization': self._config_to_dict(self.visualization),
            'logging': self._config_to_dict(self.logging)
        }

        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    def validate(self) -> bool:
        """
        Validate configuration parameters.

        Returns:
            True if configuration is valid
        """
        valid = True

        # Validate detection parameters
        if self.detection.max_intensity <= self.detection.min_intensity:
            logger.error("max_intensity must be greater than min_intensity")
            valid = False

        if self.detection.max_spot_size <= self.detection.min_spot_size:
            logger.error("max_spot_size must be greater than min_spot_size")
            valid = False

        # Validate tracking parameters
        if self.tracking.max_gap_frames < 1:
            logger.error("max_gap_frames must be at least 1")
            valid = False

        if self.tracking.min_track_length < 2:
            logger.error("min_track_length must be at least 2")
            valid = False

        # Validate classification parameters
        if self.classification.msd_min_points > self.classification.msd_fitting_points:
            logger.error("msd_min_points cannot exceed msd_fitting_points")
            valid = False

        return valid

    def _update_config(self, config_obj: Any, config_dict: Dict[str, Any]) -> None:
        """Update configuration object with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def _config_to_dict(self, config_obj: Any) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        if hasattr(config_obj, '__dict__'):
            return {k: v for k, v in config_obj.__dict__.items() if not k.startswith('_')}
        else:
            return {}

    def get_section(self, section: str):
        """Get configuration section by name."""
        return getattr(self, section, None)

    def update_parameter(self, section: str, parameter: str, value: Any) -> None:
        """
        Update a specific parameter.

        Args:
            section: Configuration section name
            parameter: Parameter name
            value: New value
        """
        config_section = self.get_section(section)
        if config_section is not None:
            if hasattr(config_section, parameter):
                setattr(config_section, parameter, value)
                logger.debug(f"Updated {section}.{parameter} = {value}")
            else:
                logger.warning(f"Parameter {parameter} not found in section {section}")
        else:
            logger.warning(f"Configuration section {section} not found")


def create_default_config() -> str:
    """
    Create a default configuration file template.

    Returns:
        YAML string with default configuration
    """
    config = PyTrackConfig()

    default_yaml = f"""# PyTrack Default Configuration File
# This file contains all configurable parameters for PyTrack modules

detection:
  # Detection method: "wavelet", "blob_log", "blob_dog", "blob_doh"
  method: "{config.detection.method}"

  # Wavelet detection parameters
  wavelet_scales: {config.detection.wavelet_scales}
  wavelet_threshold_factor: {config.detection.wavelet_threshold_factor}
  wavelet_min_area: {config.detection.wavelet_min_area}

  # Blob detection parameters
  blob_min_sigma: {config.detection.blob_min_sigma}
  blob_max_sigma: {config.detection.blob_max_sigma}
  blob_num_sigma: {config.detection.blob_num_sigma}
  blob_threshold: {config.detection.blob_threshold}
  blob_overlap: {config.detection.blob_overlap}

  # Sub-pixel localization
  subpixel_localization: {config.detection.subpixel_localization}
  gaussian_fit_window: {config.detection.gaussian_fit_window}

  # Quality control
  min_intensity: {config.detection.min_intensity}
  max_intensity: {config.detection.max_intensity}
  min_spot_size: {config.detection.min_spot_size}
  max_spot_size: {config.detection.max_spot_size}

tracking:
  # Linking parameters
  max_linking_distance: {config.tracking.max_linking_distance}
  linking_distance_auto: {config.tracking.linking_distance_auto}

  # Gap closing parameters
  max_gap_closing_distance: {config.tracking.max_gap_closing_distance}
  max_gap_frames: {config.tracking.max_gap_frames}

  # Merging and splitting
  enable_merging: {config.tracking.enable_merging}
  enable_splitting: {config.tracking.enable_splitting}

  # Motion model
  motion_model: "{config.tracking.motion_model}"

  # Track quality
  min_track_length: {config.tracking.min_track_length}

classification:
  # Motion classification
  classify_motion: {config.classification.classify_motion}
  motion_classification_method: "{config.classification.motion_classification_method}"

  # Quality scoring
  quality_scoring: {config.classification.quality_scoring}
  min_quality_score: {config.classification.min_quality_score}

visualization:
  # Display settings
  colormap: "{config.visualization.colormap}"
  track_line_width: {config.visualization.track_line_width}
  color_by: "{config.visualization.color_by}"

  # Performance
  max_tracks_display: {config.visualization.max_tracks_display}
  update_rate: {config.visualization.update_rate}

logging:
  # Logging levels
  console_level: "{config.logging.console_level}"
  file_level: "{config.logging.file_level}"

  # Log files
  log_to_file: {config.logging.log_to_file}
  log_directory: "{config.logging.log_directory}"
  log_filename: "{config.logging.log_filename}"
"""
    return default_yaml


# Global configuration instance
_global_config = None

def get_config() -> PyTrackConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = PyTrackConfig()
    return _global_config

def set_config(config: PyTrackConfig) -> None:
    """Set the global configuration instance."""
    global _global_config
    _global_config = config

def load_config(config_file: Union[str, Path]) -> PyTrackConfig:
    """Load configuration from file and set as global."""
    config = PyTrackConfig(config_file)
    set_config(config)
    return config
