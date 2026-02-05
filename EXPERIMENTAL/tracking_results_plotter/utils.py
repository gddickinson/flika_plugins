# tracking_results_plotter/utils.py
"""
Utility functions and helper classes for the Tracking Results Plotter plugin.

This module provides common functionality, data validation, export utilities,
and mathematical functions used throughout the plugin.
"""

import numpy as np
import pandas as pd
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy import stats
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

class ColorManager:
    """
    Manages color schemes and colormaps for track and point visualization.
    Provides consistent color mapping across different visualization modes.
    """
    
    def __init__(self):
        self.default_colors = {
            'track': '#1f77b4',  # Blue
            'point': '#ff7f0e',  # Orange
            'start': '#2ca02c',  # Green
            'end': '#d62728',    # Red
            'selected': '#ff1493', # Deep pink
            'background': '#ffffff'
        }
        
        self.colormaps = {
            'viridis': self._create_viridis_colors(),
            'plasma': self._create_plasma_colors(),
            'hot': self._create_hot_colors(),
            'cool': self._create_cool_colors(),
            'rainbow': self._create_rainbow_colors()
        }
        
        self.property_ranges = {}  # Store min/max for normalization
    
    def _create_viridis_colors(self) -> List[str]:
        """Create viridis-like colormap."""
        return ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', 
                '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825']
    
    def _create_plasma_colors(self) -> List[str]:
        """Create plasma-like colormap."""
        return ['#0c0786', '#40039a', '#6a00a7', '#8f0da4', '#b93289',
                '#db5c68', '#f48849', '#fdb42f', '#fcce25']
    
    def _create_hot_colors(self) -> List[str]:
        """Create hot colormap."""
        return ['#000000', '#330000', '#660000', '#990000', '#cc0000',
                '#ff0000', '#ff3300', '#ff6600', '#ff9900', '#ffcc00', '#ffff00']
    
    def _create_cool_colors(self) -> List[str]:
        """Create cool colormap."""
        return ['#00ffff', '#19e6ff', '#33ccff', '#4db3ff', '#6699ff',
                '#8080ff', '#9966ff', '#b34dff', '#cc33ff', '#e619ff', '#ff00ff']
    
    def _create_rainbow_colors(self) -> List[str]:
        """Create rainbow colormap."""
        return ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00',
                '#00ff80', '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff']
    
    def get_color_for_value(self, value: float, property_name: str, 
                           colormap: str = 'viridis') -> str:
        """
        Get color for a specific value based on property range and colormap.
        
        Args:
            value: The value to map to color
            property_name: Name of the property (for range normalization)
            colormap: Name of colormap to use
            
        Returns:
            Hex color string
        """
        if np.isnan(value):
            return self.default_colors['background']
        
        # Get or set property range
        if property_name not in self.property_ranges:
            logger.warning(f"Property range not set for {property_name}. Using 0-1.")
            self.property_ranges[property_name] = (0, 1)
        
        min_val, max_val = self.property_ranges[property_name]
        
        # Normalize value to 0-1 range
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = np.clip((value - min_val) / (max_val - min_val), 0, 1)
        
        # Get color from colormap
        colors = self.colormaps.get(colormap, self.colormaps['viridis'])
        color_index = int(normalized * (len(colors) - 1))
        
        return colors[color_index]
    
    def set_property_range(self, property_name: str, min_val: float, max_val: float):
        """Set the value range for a property for color normalization."""
        self.property_ranges[property_name] = (min_val, max_val)
    
    def get_default_color(self, element_type: str) -> str:
        """Get default color for different elements."""
        return self.default_colors.get(element_type, '#000000')


class DataValidator:
    """
    Validates tracking data and provides detailed error reporting.
    Ensures data integrity and compatibility with visualization tools.
    """
    
    def __init__(self):
        self.required_columns = ['track_id', 'frame', 'x', 'y']
        self.validation_results = {}
        
    def validate_dataframe(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Comprehensive validation of tracking dataframe.
        
        Args:
            df: DataFrame to validate
            column_mapping: Mapping of required columns to actual column names
            
        Returns:
            Dictionary with validation results and suggestions
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'statistics': {}
        }
        
        # Check required columns
        missing_columns = []
        for req_col in self.required_columns:
            if req_col not in column_mapping:
                missing_columns.append(req_col)
        
        if missing_columns:
            results['is_valid'] = False
            results['errors'].append(f"Missing required column mappings: {missing_columns}")
            return results
        
        # Validate data types and ranges
        self._validate_track_ids(df, column_mapping, results)
        self._validate_frames(df, column_mapping, results)
        self._validate_coordinates(df, column_mapping, results)
        self._validate_tracks_continuity(df, column_mapping, results)
        
        # Calculate statistics
        self._calculate_data_statistics(df, column_mapping, results)
        
        return results
    
    def _validate_track_ids(self, df: pd.DataFrame, mapping: Dict[str, str], results: Dict):
        """Validate track ID column."""
        track_col = mapping['track_id']
        
        # Check for missing values
        if df[track_col].isna().any():
            results['warnings'].append("Track IDs contain missing values")
        
        # Check data type
        if not pd.api.types.is_integer_dtype(df[track_col]):
            try:
                df[track_col] = pd.to_numeric(df[track_col], errors='coerce')
                results['warnings'].append("Track IDs converted to numeric")
            except:
                results['errors'].append("Track IDs cannot be converted to numeric")
                results['is_valid'] = False
        
        # Check for negative IDs
        if (df[track_col] < 0).any():
            results['warnings'].append("Track IDs contain negative values")
    
    def _validate_frames(self, df: pd.DataFrame, mapping: Dict[str, str], results: Dict):
        """Validate frame column."""
        frame_col = mapping['frame']
        
        # Check for missing values
        if df[frame_col].isna().any():
            results['warnings'].append("Frame numbers contain missing values")
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(df[frame_col]):
            try:
                df[frame_col] = pd.to_numeric(df[frame_col], errors='coerce')
                results['warnings'].append("Frame numbers converted to numeric")
            except:
                results['errors'].append("Frame numbers cannot be converted to numeric")
                results['is_valid'] = False
        
        # Check for negative frames
        if (df[frame_col] < 0).any():
            results['warnings'].append("Frame numbers contain negative values")
        
        # Check frame continuity
        frame_range = df[frame_col].max() - df[frame_col].min() + 1
        unique_frames = df[frame_col].nunique()
        if unique_frames < frame_range * 0.8:
            results['warnings'].append("Large gaps detected in frame sequence")
    
    def _validate_coordinates(self, df: pd.DataFrame, mapping: Dict[str, str], results: Dict):
        """Validate coordinate columns."""
        for coord in ['x', 'y']:
            coord_col = mapping[coord]
            
            # Check for missing values
            if df[coord_col].isna().any():
                results['warnings'].append(f"{coord.upper()} coordinates contain missing values")
            
            # Check data type
            if not pd.api.types.is_numeric_dtype(df[coord_col]):
                try:
                    df[coord_col] = pd.to_numeric(df[coord_col], errors='coerce')
                    results['warnings'].append(f"{coord.upper()} coordinates converted to numeric")
                except:
                    results['errors'].append(f"{coord.upper()} coordinates cannot be converted to numeric")
                    results['is_valid'] = False
            
            # Check for reasonable coordinate ranges
            coord_range = df[coord_col].max() - df[coord_col].min()
            if coord_range > 100000:
                results['warnings'].append(f"{coord.upper()} coordinate range is very large ({coord_range:.1f})")
            elif coord_range < 1:
                results['warnings'].append(f"{coord.upper()} coordinate range is very small ({coord_range:.3f})")
    
    def _validate_tracks_continuity(self, df: pd.DataFrame, mapping: Dict[str, str], results: Dict):
        """Check track continuity and detect gaps."""
        track_col = mapping['track_id']
        frame_col = mapping['frame']
        
        discontinuous_tracks = []
        
        for track_id in df[track_col].unique():
            track_data = df[df[track_col] == track_id][frame_col].sort_values()
            
            if len(track_data) < 2:
                continue
            
            # Check for frame gaps
            frame_diffs = track_data.diff().dropna()
            max_gap = frame_diffs.max()
            
            if max_gap > 5:  # Arbitrary threshold for large gaps
                discontinuous_tracks.append(track_id)
        
        if discontinuous_tracks:
            results['warnings'].append(f"Tracks with large frame gaps: {len(discontinuous_tracks)} tracks")
            if len(discontinuous_tracks) <= 10:
                results['suggestions'].append(f"Discontinuous track IDs: {discontinuous_tracks}")
    
    def _calculate_data_statistics(self, df: pd.DataFrame, mapping: Dict[str, str], results: Dict):
        """Calculate useful data statistics."""
        stats = {}
        
        # Basic counts
        stats['total_points'] = len(df)
        stats['unique_tracks'] = df[mapping['track_id']].nunique()
        stats['frame_range'] = (df[mapping['frame']].min(), df[mapping['frame']].max())
        
        # Track length statistics
        track_lengths = df.groupby(mapping['track_id']).size()
        stats['track_length_stats'] = {
            'mean': track_lengths.mean(),
            'median': track_lengths.median(),
            'min': track_lengths.min(),
            'max': track_lengths.max(),
            'std': track_lengths.std()
        }
        
        # Coordinate statistics
        for coord in ['x', 'y']:
            coord_col = mapping[coord]
            stats[f'{coord}_range'] = (df[coord_col].min(), df[coord_col].max())
            stats[f'{coord}_std'] = df[coord_col].std()
        
        # Intensity statistics if available
        if 'intensity' in mapping and mapping['intensity'] in df.columns:
            intensity_col = mapping['intensity']
            stats['intensity_stats'] = {
                'mean': df[intensity_col].mean(),
                'median': df[intensity_col].median(),
                'min': df[intensity_col].min(),
                'max': df[intensity_col].max(),
                'std': df[intensity_col].std()
            }
        
        results['statistics'] = stats


class ExportManager:
    """
    Handles export of data, statistics, and visualizations to various formats.
    Provides consistent export functionality across the plugin.
    """
    
    def __init__(self):
        self.supported_formats = {
            'data': ['.csv', '.xlsx', '.json'],
            'plots': ['.png', '.pdf', '.svg', '.jpg'],
            'statistics': ['.csv', '.xlsx', '.json', '.txt']
        }
    
    def export_data(self, data: pd.DataFrame, filepath: str, 
                   include_metadata: bool = True) -> bool:
        """
        Export DataFrame to specified format.
        
        Args:
            data: DataFrame to export
            filepath: Output file path
            include_metadata: Whether to include metadata in export
            
        Returns:
            Success status
        """
        try:
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext == '.csv':
                data.to_csv(filepath, index=False)
            elif file_ext == '.xlsx':
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    data.to_excel(writer, sheet_name='Data', index=False)
                    if include_metadata:
                        self._add_metadata_sheet(writer, data)
            elif file_ext == '.json':
                data.to_json(filepath, orient='records', indent=2)
            else:
                logger.error(f"Unsupported data export format: {file_ext}")
                return False
            
            logger.info(f"Data exported successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False
    
    def export_statistics(self, stats_dict: Dict, filepath: str) -> bool:
        """
        Export statistics dictionary to specified format.
        
        Args:
            stats_dict: Dictionary containing statistics
            filepath: Output file path
            
        Returns:
            Success status
        """
        try:
            file_ext = Path(filepath).suffix.lower()
            
            if file_ext == '.json':
                with open(filepath, 'w') as f:
                    json.dump(stats_dict, f, indent=2, default=str)
            elif file_ext == '.txt':
                self._export_stats_as_text(stats_dict, filepath)
            elif file_ext in ['.csv', '.xlsx']:
                # Convert nested dict to flat DataFrame
                flat_stats = self._flatten_statistics(stats_dict)
                stats_df = pd.DataFrame([flat_stats])
                
                if file_ext == '.csv':
                    stats_df.to_csv(filepath, index=False)
                else:
                    stats_df.to_excel(filepath, index=False)
            else:
                logger.error(f"Unsupported statistics export format: {file_ext}")
                return False
            
            logger.info(f"Statistics exported successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting statistics: {str(e)}")
            return False
    
    def export_plot(self, figure, filepath: str, dpi: int = 300, 
                   bbox_inches: str = 'tight') -> bool:
        """
        Export matplotlib figure to file.
        
        Args:
            figure: Matplotlib figure object
            filepath: Output file path
            dpi: Resolution for raster formats
            bbox_inches: Bounding box mode
            
        Returns:
            Success status
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for plot export")
            return False
        
        try:
            figure.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
            logger.info(f"Plot exported successfully to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting plot: {str(e)}")
            return False
    
    def _add_metadata_sheet(self, writer, data: pd.DataFrame):
        """Add metadata sheet to Excel file."""
        metadata = {
            'Export Information': [
                f"Export Date: {pd.Timestamp.now()}",
                f"Total Rows: {len(data)}",
                f"Total Columns: {len(data.columns)}",
                f"Data Types: {dict(data.dtypes)}",
                f"Memory Usage: {data.memory_usage(deep=True).sum()} bytes"
            ]
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    
    def _export_stats_as_text(self, stats_dict: Dict, filepath: str):
        """Export statistics as formatted text file."""
        with open(filepath, 'w') as f:
            f.write("Tracking Results Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in stats_dict.items():
                f.write(f"{key}:\n")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"  {value}\n")
                f.write("\n")
    
    def _flatten_statistics(self, stats_dict: Dict) -> Dict:
        """Flatten nested statistics dictionary."""
        flat_dict = {}
        
        for key, value in stats_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_dict[f"{key}_{subkey}"] = subvalue
            else:
                flat_dict[key] = value
        
        return flat_dict


class MathUtils:
    """
    Mathematical utility functions for track analysis and visualization.
    Provides common calculations used in particle tracking analysis.
    """
    
    @staticmethod
    def calculate_displacement(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    @staticmethod
    def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate angle between two points in radians."""
        return np.arctan2(y2 - y1, x2 - x1)
    
    @staticmethod
    def calculate_track_velocity(track_data: pd.DataFrame, x_col: str, y_col: str, 
                               frame_col: str, time_interval: float = 1.0) -> np.ndarray:
        """
        Calculate instantaneous velocities for a track.
        
        Args:
            track_data: DataFrame containing single track data
            x_col, y_col: Column names for coordinates
            frame_col: Column name for frame numbers
            time_interval: Time between frames
            
        Returns:
            Array of velocities
        """
        track_data = track_data.sort_values(frame_col)
        
        x_vals = track_data[x_col].values
        y_vals = track_data[y_col].values
        
        if len(x_vals) < 2:
            return np.array([])
        
        # Calculate displacements
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        
        # Calculate velocities
        displacements = np.sqrt(dx**2 + dy**2)
        velocities = displacements / time_interval
        
        return velocities
    
    @staticmethod
    def calculate_radius_of_gyration(x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """
        Calculate radius of gyration for a set of coordinates.
        
        Args:
            x_coords, y_coords: Arrays of coordinates
            
        Returns:
            Radius of gyration
        """
        if len(x_coords) < 2:
            return np.nan
        
        # Calculate center of mass
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)
        
        # Calculate radius of gyration
        distances_squared = (x_coords - x_center)**2 + (y_coords - y_center)**2
        rg = np.sqrt(np.mean(distances_squared))
        
        return rg
    
    @staticmethod
    def calculate_track_straightness(track_data: pd.DataFrame, x_col: str, y_col: str) -> float:
        """
        Calculate track straightness (end-to-end distance / contour length).
        
        Args:
            track_data: DataFrame containing track data
            x_col, y_col: Column names for coordinates
            
        Returns:
            Straightness value (0-1)
        """
        x_vals = track_data[x_col].values
        y_vals = track_data[y_col].values
        
        if len(x_vals) < 2:
            return np.nan
        
        # End-to-end distance
        end_to_end = MathUtils.calculate_displacement(x_vals[0], y_vals[0], 
                                                     x_vals[-1], y_vals[-1])
        
        # Contour length
        contour_length = 0
        for i in range(1, len(x_vals)):
            contour_length += MathUtils.calculate_displacement(x_vals[i-1], y_vals[i-1],
                                                             x_vals[i], y_vals[i])
        
        if contour_length == 0:
            return np.nan
        
        return end_to_end / contour_length
    
    @staticmethod
    def calculate_turning_angles(track_data: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray:
        """
        Calculate turning angles for a track.
        
        Args:
            track_data: DataFrame containing track data
            x_col, y_col: Column names for coordinates
            
        Returns:
            Array of turning angles in radians
        """
        x_vals = track_data[x_col].values
        y_vals = track_data[y_col].values
        
        if len(x_vals) < 3:
            return np.array([])
        
        angles = []
        for i in range(1, len(x_vals) - 1):
            # Calculate angles between consecutive segments
            angle1 = MathUtils.calculate_angle(x_vals[i-1], y_vals[i-1], 
                                             x_vals[i], y_vals[i])
            angle2 = MathUtils.calculate_angle(x_vals[i], y_vals[i], 
                                             x_vals[i+1], y_vals[i+1])
            
            # Calculate turning angle
            turning_angle = angle2 - angle1
            
            # Normalize to [-π, π]
            while turning_angle > np.pi:
                turning_angle -= 2 * np.pi
            while turning_angle < -np.pi:
                turning_angle += 2 * np.pi
            
            angles.append(turning_angle)
        
        return np.array(angles)


class ConfigManager:
    """
    Manages plugin configuration settings and user preferences.
    Provides persistent storage of user settings and defaults.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Use FLIKA's user directory
            config_dir = os.path.expanduser("~/.FLIKA/plugins/tracking_results_plotter")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        
        # Default configuration
        self.default_config = {
            'display': {
                'point_size': 3,
                'line_width': 2,
                'point_color': 'red',
                'line_color': 'blue',
                'colormap': 'viridis',
                'show_tracks': True,
                'show_points': True,
                'auto_update': True
            },
            'analysis': {
                'min_track_length': 5,
                'time_interval': 1.0,
                'distance_units': 'pixels',
                'time_units': 'frames'
            },
            'export': {
                'default_format': 'csv',
                'include_metadata': True,
                'plot_dpi': 300,
                'plot_format': 'png'
            },
            'ui': {
                'window_size': [800, 600],
                'remember_window_position': True,
                'tab_position': 0
            }
        }
        
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults (in case new options were added)
                config = self.default_config.copy()
                for section, settings in loaded_config.items():
                    if section in config:
                        config[section].update(settings)
                    else:
                        config[section] = settings
                
                return config
            else:
                return self.default_config.copy()
                
        except Exception as e:
            logger.warning(f"Error loading config, using defaults: {str(e)}")
            return self.default_config.copy()
    
    def save_config(self) -> bool:
        """Save current configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            return False
    
    def get(self, section: str, key: str, default=None):
        """Get configuration value."""
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value):
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.default_config.copy()


def generate_sample_data(n_tracks: int = 10, n_frames: int = 100, 
                        image_size: Tuple[int, int] = (512, 512),
                        noise_level: float = 0.5) -> pd.DataFrame:
    """
    Generate sample tracking data for testing and demonstration.
    
    Args:
        n_tracks: Number of tracks to generate
        n_frames: Number of frames
        image_size: Size of the image (width, height)
        noise_level: Amount of random noise to add
        
    Returns:
        DataFrame with sample tracking data
    """
    np.random.seed(42)  # For reproducible results
    
    data = []
    
    for track_id in range(1, n_tracks + 1):
        # Random starting position
        start_x = np.random.uniform(50, image_size[0] - 50)
        start_y = np.random.uniform(50, image_size[1] - 50)
        
        # Random motion parameters
        diffusion_coef = np.random.uniform(0.1, 2.0)
        drift_x = np.random.uniform(-0.1, 0.1)
        drift_y = np.random.uniform(-0.1, 0.1)
        
        # Random track length (some tracks appear/disappear)
        track_length = np.random.randint(n_frames // 4, n_frames)
        start_frame = np.random.randint(0, n_frames - track_length)
        
        # Generate trajectory
        x, y = start_x, start_y
        base_intensity = np.random.uniform(500, 2000)
        
        for frame in range(start_frame, start_frame + track_length):
            # Random walk with drift
            x += drift_x + np.random.normal(0, diffusion_coef)
            y += drift_y + np.random.normal(0, diffusion_coef)
            
            # Add some noise to coordinates
            x_noisy = x + np.random.normal(0, noise_level)
            y_noisy = y + np.random.normal(0, noise_level)
            
            # Varying intensity with photobleaching
            intensity = base_intensity * np.exp(-frame * 0.001) + np.random.normal(0, 50)
            
            # Some example analysis results
            velocity = diffusion_coef + np.random.normal(0, 0.1)
            svm_class = np.random.choice(['Mobile', 'Intermediate', 'Trapped'], 
                                       p=[0.4, 0.3, 0.3])
            
            data.append({
                'track_number': track_id,
                'frame': frame,
                'x': x_noisy,
                'y': y_noisy,
                'intensity': max(0, intensity),
                'velocity': max(0, velocity),
                'SVM': svm_class,
                'Experiment': f'Condition_{track_id % 3 + 1}'
            })
            
            # Keep particles within image bounds
            x = np.clip(x, 10, image_size[0] - 10)
            y = np.clip(y, 10, image_size[1] - 10)
    
    df = pd.DataFrame(data)
    
    # Add some additional analysis columns
    df['track_length'] = df.groupby('track_number')['track_number'].transform('count')
    df['radius_gyration'] = np.random.uniform(1, 5, len(df))
    df['asymmetry'] = np.random.uniform(0, 1, len(df))
    
    return df


def create_example_csv(filepath: str, **kwargs):
    """Create an example CSV file with sample tracking data."""
    sample_data = generate_sample_data(**kwargs)
    sample_data.to_csv(filepath, index=False)
    logger.info(f"Example CSV created at {filepath}")


# Plugin utilities for common operations
def get_flika_plugin_dir() -> Path:
    """Get the FLIKA plugins directory."""
    return Path.home() / ".FLIKA" / "plugins"

def get_plugin_dir() -> Path:
    """Get this plugin's directory."""
    return get_flika_plugin_dir() / "tracking_results_plotter"

def setup_logging(log_level: str = "INFO"):
    """Setup logging for the plugin."""
    log_dir = get_plugin_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "tracking_plotter.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Initialize utilities
color_manager = ColorManager()
data_validator = DataValidator()
export_manager = ExportManager()
config_manager = ConfigManager()

# Setup logging
setup_logging()