"""
Utilities Module
================

Helper functions for:
- Data I/O (CSV, HDF5, etc.)
- Coordinate transformations
- Statistics and analysis
- File format conversions
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_localizations_csv(filepath):
    """Load localizations from CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
        
    Returns
    -------
    localizations : dict
        Dictionary with localization data
    """
    df = pd.read_csv(filepath)
    
    localizations = {}
    
    # Common column mappings
    column_map = {
        'x': ['x', 'x [nm]', 'x_nm', 'X'],
        'y': ['y', 'y [nm]', 'y_nm', 'Y'],
        'z': ['z', 'z [nm]', 'z_nm', 'Z'],
        'frame': ['frame', 'Frame', 'frame_number'],
        'intensity': ['intensity', 'Intensity', 'photons', 'N'],
        'background': ['background', 'Background', 'bg', 'offset'],
        'sigma_x': ['sigma_x', 'sigma', 'sigma [nm]', 'PSF_sigma'],
        'sigma_y': ['sigma_y', 'sigma', 'sigma [nm]', 'PSF_sigma'],
        'uncertainty': ['uncertainty', 'Uncertainty', 'precision', 'sigma_xy']
    }
    
    for key, possible_cols in column_map.items():
        for col in possible_cols:
            if col in df.columns:
                localizations[key] = df[col].values
                break
    
    return localizations


def save_localizations_csv(localizations, filepath, include_metadata=True):
    """Save localizations to CSV file.
    
    Parameters
    ----------
    localizations : dict
        Dictionary with localization data
    filepath : str or Path
        Output filepath
    include_metadata : bool
        Include additional metadata columns
    """
    # Convert to DataFrame
    data = {}
    
    # Core columns
    if 'x' in localizations:
        data['x [nm]'] = localizations['x']
    if 'y' in localizations:
        data['y [nm]'] = localizations['y']
    if 'z' in localizations:
        data['z [nm]'] = localizations['z']
    if 'frame' in localizations:
        data['frame'] = localizations['frame']
        
    # Optional columns
    if include_metadata:
        for key in ['intensity', 'background', 'sigma_x', 'sigma_y', 
                   'uncertainty', 'chi_squared']:
            if key in localizations:
                data[key] = localizations[key]
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


def load_image_stack(filepath):
    """Load image stack from various formats.
    
    Parameters
    ----------
    filepath : str or Path
        Path to image file
        
    Returns
    -------
    images : ndarray
        Image stack (n_frames, height, width)
    metadata : dict
        Image metadata
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.tif' or suffix == '.tiff':
        from tifffile import imread
        images = imread(str(filepath))
        
        # Ensure 3D
        if images.ndim == 2:
            images = images[np.newaxis, ...]
            
        metadata = {'filename': str(filepath)}
        
    elif suffix == '.npy':
        images = np.load(str(filepath))
        metadata = {'filename': str(filepath)}
        
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    return images, metadata


def save_image_stack(images, filepath):
    """Save image stack to file.
    
    Parameters
    ----------
    images : ndarray
        Image stack (n_frames, height, width)
    filepath : str or Path
        Output filepath
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.tif' or suffix == '.tiff':
        from tifffile import imwrite
        imwrite(str(filepath), images.astype(np.float32))
        
    elif suffix == '.npy':
        np.save(str(filepath), images)
        
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def compute_nearest_neighbor_distances(localizations):
    """Compute nearest neighbor distances.
    
    Parameters
    ----------
    localizations : dict
        Localization data with 'x', 'y'
        
    Returns
    -------
    distances : ndarray
        Nearest neighbor distance for each localization
    """
    from scipy.spatial import cKDTree
    
    positions = np.column_stack([localizations['x'], localizations['y']])
    tree = cKDTree(positions)
    
    # Query for 2 nearest (self + nearest)
    distances, indices = tree.query(positions, k=2)
    
    # Return distance to nearest neighbor (not self)
    return distances[:, 1]


def compute_ripley_k(localizations, radii, area=None):
    """Compute Ripley's K function for spatial statistics.
    
    Parameters
    ----------
    localizations : dict
        Localization data with 'x', 'y'
    radii : ndarray
        Radii to evaluate
    area : float, optional
        Total area (computed from data if not provided)
        
    Returns
    -------
    K : ndarray
        Ripley's K values
    L : ndarray
        Linearized K (L = sqrt(K/pi) - r)
    """
    from scipy.spatial import cKDTree
    
    positions = np.column_stack([localizations['x'], localizations['y']])
    n = len(positions)
    
    if area is None:
        x_range = localizations['x'].max() - localizations['x'].min()
        y_range = localizations['y'].max() - localizations['y'].min()
        area = x_range * y_range
    
    tree = cKDTree(positions)
    
    K = np.zeros(len(radii))
    
    for i, r in enumerate(radii):
        # Count pairs within distance r
        pairs = tree.query_pairs(r)
        n_pairs = len(pairs)
        
        # Ripley's K
        K[i] = (area * n_pairs) / (n * (n - 1))
    
    # Linearized L function
    L = np.sqrt(K / np.pi) - radii
    
    return K, L


def convert_coordinates(localizations, pixel_size_from, pixel_size_to=None,
                       origin=(0, 0)):
    """Convert coordinate system.
    
    Parameters
    ----------
    localizations : dict
        Localization data
    pixel_size_from : float
        Original pixel size (nm)
    pixel_size_to : float, optional
        Target pixel size (nm)
    origin : tuple
        Origin offset (x, y) in nm
        
    Returns
    -------
    converted : dict
        Converted localizations
    """
    converted = localizations.copy()
    
    # Translate
    converted['x'] = localizations['x'] - origin[0]
    converted['y'] = localizations['y'] - origin[1]
    
    # Scale
    if pixel_size_to is not None and pixel_size_to != pixel_size_from:
        scale = pixel_size_from / pixel_size_to
        converted['x'] *= scale
        converted['y'] *= scale
        
        if 'sigma_x' in converted:
            converted['sigma_x'] *= scale
        if 'sigma_y' in converted:
            converted['sigma_y'] *= scale
        if 'uncertainty' in converted:
            converted['uncertainty'] *= scale
    
    return converted


def compute_localization_density(localizations, pixel_size=100):
    """Compute spatial density map.
    
    Parameters
    ----------
    localizations : dict
        Localization data
    pixel_size : float
        Pixel size for density map (nm)
        
    Returns
    -------
    density_map : ndarray
        2D density map
    extent : tuple
        (x_min, x_max, y_min, y_max) in nm
    """
    x = localizations['x']
    y = localizations['y']
    
    # Determine bounds
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Create bins
    x_bins = np.arange(x_min, x_max + pixel_size, pixel_size)
    y_bins = np.arange(y_min, y_max + pixel_size, pixel_size)
    
    # Compute 2D histogram
    density_map, _, _ = np.histogram2d(y, x, bins=[y_bins, x_bins])
    
    extent = (x_min, x_max, y_min, y_max)
    
    return density_map, extent


def filter_by_roi(localizations, roi):
    """Filter localizations within region of interest.
    
    Parameters
    ----------
    localizations : dict
        Localization data
    roi : tuple
        (x_min, x_max, y_min, y_max) in nm
        
    Returns
    -------
    filtered : dict
        Filtered localizations
    """
    x_min, x_max, y_min, y_max = roi
    
    mask = ((localizations['x'] >= x_min) & (localizations['x'] <= x_max) &
            (localizations['y'] >= y_min) & (localizations['y'] <= y_max))
    
    filtered = {}
    for key, value in localizations.items():
        if isinstance(value, np.ndarray) and len(value) == len(mask):
            filtered[key] = value[mask]
        else:
            filtered[key] = value
    
    return filtered


def compute_statistics(localizations):
    """Compute summary statistics.
    
    Parameters
    ----------
    localizations : dict
        Localization data
        
    Returns
    -------
    stats : dict
        Summary statistics
    """
    stats = {
        'n_localizations': len(localizations['x']),
    }
    
    if 'frame' in localizations:
        stats['n_frames'] = len(np.unique(localizations['frame']))
        stats['mean_localizations_per_frame'] = stats['n_localizations'] / stats['n_frames']
    
    if 'intensity' in localizations:
        stats['mean_intensity'] = np.mean(localizations['intensity'])
        stats['median_intensity'] = np.median(localizations['intensity'])
        stats['std_intensity'] = np.std(localizations['intensity'])
    
    if 'sigma_x' in localizations:
        stats['mean_sigma_x'] = np.mean(localizations['sigma_x'])
        stats['median_sigma_x'] = np.median(localizations['sigma_x'])
    
    if 'uncertainty' in localizations:
        stats['mean_uncertainty'] = np.mean(localizations['uncertainty'])
        stats['median_uncertainty'] = np.median(localizations['uncertainty'])
    
    # Spatial extent
    stats['x_min'] = np.min(localizations['x'])
    stats['x_max'] = np.max(localizations['x'])
    stats['y_min'] = np.min(localizations['y'])
    stats['y_max'] = np.max(localizations['y'])
    stats['x_range'] = stats['x_max'] - stats['x_min']
    stats['y_range'] = stats['y_max'] - stats['y_min']
    
    # Nearest neighbor
    nn_dist = compute_nearest_neighbor_distances(localizations)
    stats['mean_nn_distance'] = np.mean(nn_dist)
    stats['median_nn_distance'] = np.median(nn_dist)
    
    return stats


def merge_localization_datasets(datasets):
    """Merge multiple localization datasets.
    
    Parameters
    ----------
    datasets : list of dict
        List of localization dictionaries
        
    Returns
    -------
    merged : dict
        Merged dataset
    """
    merged = {}
    
    # Get all keys
    all_keys = set()
    for dataset in datasets:
        all_keys.update(dataset.keys())
    
    # Merge each key
    for key in all_keys:
        arrays = [d[key] for d in datasets if key in d]
        
        if arrays and isinstance(arrays[0], np.ndarray):
            merged[key] = np.concatenate(arrays)
        else:
            merged[key] = arrays[0] if arrays else None
    
    return merged


class LocalizationTable:
    """Class for managing localization data with pandas-like interface."""
    
    def __init__(self, localizations=None):
        self.data = localizations if localizations is not None else {}
        
    def __len__(self):
        if 'x' in self.data:
            return len(self.data['x'])
        return 0
        
    def filter(self, condition):
        """Filter localizations by condition.
        
        Parameters
        ----------
        condition : callable or ndarray
            Boolean function or array
            
        Returns
        -------
        filtered : LocalizationTable
            Filtered table
        """
        if callable(condition):
            mask = condition(self.data)
        else:
            mask = condition
            
        filtered_data = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                filtered_data[key] = value[mask]
            else:
                filtered_data[key] = value
                
        return LocalizationTable(filtered_data)
        
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.data)
        
    def save(self, filepath, format='csv'):
        """Save to file."""
        if format == 'csv':
            save_localizations_csv(self.data, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    @classmethod
    def load(cls, filepath, format='csv'):
        """Load from file."""
        if format == 'csv':
            data = load_localizations_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return cls(data)
