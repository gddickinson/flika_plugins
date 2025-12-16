#!/usr/bin/env python3
"""
Utility functions for u-track Python port

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.

u-track is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

u-track is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with u-track.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import time
import numpy as np
import scipy.io
from scipy import stats, ndimage
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Union, Optional, Tuple, List, Dict, Any
import os
import warnings
from dataclasses import dataclass


def progress_text(fraction: float, message: str, bar_length: int = 50) -> None:
    """
    Display progress information with a progress bar.

    Equivalent to MATLAB's progressText function.

    Args:
        fraction: Completion fraction (0.0 to 1.0)
        message: Message to display
        bar_length: Length of the progress bar in characters
    """
    # Ensure fraction is between 0 and 1
    fraction = max(0.0, min(1.0, fraction))

    # Calculate number of filled characters
    filled_length = int(bar_length * fraction)

    # Create progress bar
    bar = '█' * filled_length + '░' * (bar_length - filled_length)

    # Calculate percentage
    percentage = fraction * 100

    # Format the progress message
    progress_msg = f'\r{message}: |{bar}| {percentage:.1f}% Complete'

    # Print without newline, flush immediately
    sys.stdout.write(progress_msg)
    sys.stdout.flush()

    # Add newline if complete
    if fraction >= 1.0:
        print()


# ==============================================================================
# MATLAB-Compatible Statistical Functions
# ==============================================================================

def normcdf(x: np.ndarray, mu: float = 0, sigma: float = 1) -> np.ndarray:
    """
    MATLAB-compatible normal cumulative distribution function.

    Args:
        x: Input values
        mu: Mean (default: 0)
        sigma: Standard deviation (default: 1)

    Returns:
        CDF values
    """
    return stats.norm.cdf(x, loc=mu, scale=sigma)


def prctile(data: np.ndarray, percentiles: Union[float, List[float]],
           axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    MATLAB-compatible percentile function.

    Args:
        data: Input data
        percentiles: Percentile(s) to compute (0-100)
        axis: Axis along which to compute percentiles

    Returns:
        Percentile value(s)
    """
    return np.percentile(data, percentiles, axis=axis)


def tcdf(x: np.ndarray, nu: float) -> np.ndarray:
    """
    MATLAB-compatible Student's t cumulative distribution function.

    Args:
        x: Input values
        nu: Degrees of freedom

    Returns:
        CDF values
    """
    return stats.t.cdf(x, df=nu)


def fcdf(x: np.ndarray, dfn: float, dfd: float) -> np.ndarray:
    """
    MATLAB-compatible F cumulative distribution function.

    Args:
        x: Input values
        dfn: Degrees of freedom numerator
        dfd: Degrees of freedom denominator

    Returns:
        CDF values
    """
    return stats.f.cdf(x, dfn=dfn, dfd=dfd)


# ==============================================================================
# Coordinate System Utilities
# ==============================================================================

def matlab_to_python_coords(coords: np.ndarray) -> np.ndarray:
    """
    Convert MATLAB 1-indexed coordinates to Python 0-indexed coordinates.

    Args:
        coords: MATLAB coordinates (1-indexed)

    Returns:
        Python coordinates (0-indexed)
    """
    return coords - 1


def python_to_matlab_coords(coords: np.ndarray) -> np.ndarray:
    """
    Convert Python 0-indexed coordinates to MATLAB 1-indexed coordinates.

    Args:
        coords: Python coordinates (0-indexed)

    Returns:
        MATLAB coordinates (1-indexed)
    """
    return coords + 1


def matrix_to_image_coords(row: int, col: int, image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert matrix coordinates (row, col) to image coordinates (x, y).

    Args:
        row: Matrix row index
        col: Matrix column index
        image_shape: Shape of image (height, width)

    Returns:
        Tuple of (x, y) in image coordinates
    """
    # In image coordinates: x is horizontal (columns), y is vertical (rows)
    x = col
    y = row
    return x, y


def image_to_matrix_coords(x: int, y: int, image_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert image coordinates (x, y) to matrix coordinates (row, col).

    Args:
        x: Image x coordinate (horizontal)
        y: Image y coordinate (vertical)
        image_shape: Shape of image (height, width)

    Returns:
        Tuple of (row, col) in matrix coordinates
    """
    # In matrix coordinates: row is vertical (y), col is horizontal (x)
    row = y
    col = x
    return row, col


# ==============================================================================
# Enhanced Distance and Clustering Functions
# ==============================================================================

def create_distance_matrix(points1: np.ndarray, points2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create distance matrix between two sets of points.

    Equivalent to MATLAB's createDistanceMatrix function or pdist functionality.

    Args:
        points1: First set of points, shape (N, D) where N is number of points, D is dimensionality
        points2: Second set of points, shape (M, D). If None, compute pairwise distances within points1

    Returns:
        Distance matrix of shape (N, M) if points2 is provided, or (N, N) if points2 is None

    Examples:
        >>> points = np.array([[0, 0], [1, 0], [0, 1]])
        >>> dist_matrix = create_distance_matrix(points)
        >>> print(dist_matrix.shape)
        (3, 3)

        >>> points1 = np.array([[0, 0], [1, 0]])
        >>> points2 = np.array([[0, 1], [1, 1]])
        >>> dist_matrix = create_distance_matrix(points1, points2)
        >>> print(dist_matrix.shape)
        (2, 2)
    """
    # Ensure points1 is a 2D array
    points1 = np.atleast_2d(points1)

    if points2 is None:
        # Compute pairwise distances within points1
        points2 = points1
    else:
        # Ensure points2 is a 2D array
        points2 = np.atleast_2d(points2)

    # Validate dimensions
    if points1.shape[1] != points2.shape[1]:
        raise ValueError("Points must have the same dimensionality")

    # Calculate distance matrix using scipy's cdist (Euclidean distance by default)
    distance_matrix = cdist(points1, points2, metric='euclidean')

    return distance_matrix


def find_overlap_psfs_enhanced(positions: np.ndarray, amplitudes: np.ndarray,
                              psf_sigma: float, overlap_threshold: float = 3.0) -> List[List[int]]:
    """
    Enhanced PSF overlap detection using proper clustering.

    Args:
        positions: Array of [x, y] positions
        amplitudes: Array of amplitudes
        psf_sigma: PSF sigma for overlap calculation
        overlap_threshold: Distance threshold in units of PSF sigma

    Returns:
        List of clusters, each containing indices of overlapping PSFs
    """
    if len(positions) == 0:
        return []

    if len(positions) == 1:
        return [[0]]

    # Calculate pairwise distances
    distances = squareform(pdist(positions))

    # Use hierarchical clustering
    threshold_distance = overlap_threshold * psf_sigma

    # Perform clustering using complete linkage
    condensed_distances = pdist(positions)
    linkage_matrix = linkage(condensed_distances, method='complete')
    cluster_labels = fcluster(linkage_matrix, threshold_distance, criterion='distance')

    # Group indices by cluster
    clusters = []
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0].tolist()
        clusters.append(cluster_indices)

    return clusters


def merge_close_features(positions: np.ndarray, amplitudes: np.ndarray,
                        merge_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge features that are closer than a specified distance.

    Args:
        positions: Array of [x, y] positions
        amplitudes: Array of amplitudes
        merge_distance: Distance threshold for merging

    Returns:
        Tuple of (merged_positions, merged_amplitudes)
    """
    if len(positions) <= 1:
        return positions.copy(), amplitudes.copy()

    # Find clusters of close features
    clusters = find_overlap_psfs_enhanced(positions, amplitudes, merge_distance/3, 3.0)

    merged_pos = []
    merged_amp = []

    for cluster in clusters:
        if len(cluster) == 1:
            # Single feature, keep as is
            merged_pos.append(positions[cluster[0]])
            merged_amp.append(amplitudes[cluster[0]])
        else:
            # Multiple features, merge by amplitude-weighted average
            cluster_pos = positions[cluster]
            cluster_amp = amplitudes[cluster]

            # Weight by amplitude
            total_amp = np.sum(cluster_amp)
            if total_amp > 0:
                weighted_pos = np.average(cluster_pos, axis=0, weights=cluster_amp)
            else:
                weighted_pos = np.mean(cluster_pos, axis=0)

            merged_pos.append(weighted_pos)
            merged_amp.append(total_amp)

    return np.array(merged_pos), np.array(merged_amp)


# ==============================================================================
# Data Structure Utilities
# ==============================================================================

@dataclass
class MovieInfoFrame:
    """Enhanced MovieInfo structure for a single frame"""
    xCoord: np.ndarray = None
    yCoord: np.ndarray = None
    amp: np.ndarray = None
    sigma: np.ndarray = None

    def __post_init__(self):
        # Initialize empty arrays if None
        if self.xCoord is None:
            self.xCoord = np.zeros((0, 2))
        if self.yCoord is None:
            self.yCoord = np.zeros((0, 2))
        if self.amp is None:
            self.amp = np.zeros((0, 2))
        if self.sigma is None:
            self.sigma = np.zeros((0, 2))

    def is_empty(self) -> bool:
        """Check if frame has no detections"""
        return len(self.xCoord) == 0

    def num_features(self) -> int:
        """Get number of features in this frame"""
        return len(self.xCoord)

    def get_positions(self) -> np.ndarray:
        """Get positions as [x, y] array"""
        if self.is_empty():
            return np.zeros((0, 2))
        return np.column_stack([self.xCoord[:, 0], self.yCoord[:, 0]])

    def get_uncertainties(self) -> np.ndarray:
        """Get position uncertainties as [dx, dy] array"""
        if self.is_empty():
            return np.zeros((0, 2))
        return np.column_stack([self.xCoord[:, 1], self.yCoord[:, 1]])


def validate_movie_info(movie_info: List) -> List[str]:
    """
    Validate MovieInfo structure and return list of issues found.

    Args:
        movie_info: List of MovieInfo structures

    Returns:
        List of validation error messages
    """
    issues = []

    if not isinstance(movie_info, list):
        issues.append("MovieInfo must be a list")
        return issues

    for i, frame in enumerate(movie_info):
        if frame is None:
            continue

        # Check required fields
        required_fields = ['xCoord', 'yCoord', 'amp']
        for field in required_fields:
            if not hasattr(frame, field):
                issues.append(f"Frame {i}: Missing required field '{field}'")
                continue

            field_value = getattr(frame, field)
            if field_value is not None and not isinstance(field_value, np.ndarray):
                issues.append(f"Frame {i}: Field '{field}' must be numpy array or None")

        # Check coordinate consistency
        if (hasattr(frame, 'xCoord') and hasattr(frame, 'yCoord') and
            frame.xCoord is not None and frame.yCoord is not None):
            if frame.xCoord.shape != frame.yCoord.shape:
                issues.append(f"Frame {i}: xCoord and yCoord shapes don't match")

        # Check amplitude consistency
        if (hasattr(frame, 'xCoord') and hasattr(frame, 'amp') and
            frame.xCoord is not None and frame.amp is not None):
            if len(frame.xCoord) != len(frame.amp):
                issues.append(f"Frame {i}: Number of coordinates and amplitudes don't match")

    return issues


def create_empty_movie_info(num_frames: int) -> List[MovieInfoFrame]:
    """
    Create an empty MovieInfo structure.

    Args:
        num_frames: Number of frames

    Returns:
        List of empty MovieInfoFrame objects
    """
    return [MovieInfoFrame() for _ in range(num_frames)]


class SimpleClass:
    """Simple class for creating objects with attributes."""
    pass


# ==============================================================================
# File I/O Utilities
# ==============================================================================

def save_movie_info_matlab(movie_info: List, filename: str,
                          additional_data: Optional[Dict] = None) -> None:
    """
    Save MovieInfo in MATLAB-compatible format.

    Args:
        movie_info: List of MovieInfo structures
        filename: Output filename (.mat)
        additional_data: Additional data to save
    """
    # Convert to MATLAB-compatible format
    matlab_movie_info = []

    for frame in movie_info:
        if frame is None or (hasattr(frame, 'is_empty') and frame.is_empty()):
            # Empty frame
            matlab_frame = {
                'xCoord': np.array([]).reshape(0, 2),
                'yCoord': np.array([]).reshape(0, 2),
                'amp': np.array([]).reshape(0, 2),
                'sigma': np.array([]).reshape(0, 2)
            }
        else:
            # Frame with data
            matlab_frame = {}
            if hasattr(frame, 'xCoord'):
                matlab_frame['xCoord'] = frame.xCoord if frame.xCoord is not None else np.array([]).reshape(0, 2)
            if hasattr(frame, 'yCoord'):
                matlab_frame['yCoord'] = frame.yCoord if frame.yCoord is not None else np.array([]).reshape(0, 2)
            if hasattr(frame, 'amp'):
                matlab_frame['amp'] = frame.amp if frame.amp is not None else np.array([]).reshape(0, 2)
            if hasattr(frame, 'sigma'):
                matlab_frame['sigma'] = frame.sigma if frame.sigma is not None else np.array([]).reshape(0, 2)

        matlab_movie_info.append(matlab_frame)

    # Prepare save data
    save_data = {'movieInfo': matlab_movie_info}
    if additional_data:
        save_data.update(additional_data)

    # Save to .mat file
    scipy.io.savemat(filename, save_data)


def load_movie_info_matlab(filename: str) -> Tuple[List, Dict]:
    """
    Load MovieInfo from MATLAB .mat file.

    Args:
        filename: Input filename (.mat)

    Returns:
        Tuple of (movie_info_list, additional_data_dict)
    """
    # Load .mat file
    mat_data = scipy.io.loadmat(filename)

    # Extract MovieInfo
    movie_info = []
    additional_data = {}

    if 'movieInfo' in mat_data:
        matlab_movie_info = mat_data['movieInfo']

        # Convert from MATLAB format
        if matlab_movie_info.size > 0:
            for i in range(len(matlab_movie_info)):
                frame_data = matlab_movie_info[i] if matlab_movie_info.ndim > 0 else matlab_movie_info

                frame = MovieInfoFrame()
                if isinstance(frame_data, np.ndarray) and frame_data.dtype.names:
                    # Structured array
                    for field in frame_data.dtype.names:
                        if hasattr(frame, field):
                            setattr(frame, field, frame_data[field])

                movie_info.append(frame)

    # Extract additional data
    for key, value in mat_data.items():
        if key not in ['movieInfo', '__header__', '__version__', '__globals__']:
            additional_data[key] = value

    return movie_info, additional_data


# ==============================================================================
# Enhanced NaN Handling
# ==============================================================================

def robust_nanmean(data: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    Calculate mean while ignoring NaN values, similar to MATLAB's nanmean.

    Args:
        data: Input data array
        axis: Axis along which to compute mean

    Returns:
        Mean value(s) with NaN values ignored
    """
    if data.size == 0:
        return np.nan

    # Handle all-NaN case
    if np.all(np.isnan(data)):
        if axis is None:
            return np.nan
        else:
            result_shape = list(data.shape)
            result_shape.pop(axis)
            return np.full(result_shape, np.nan)

    return np.nanmean(data, axis=axis)


def robust_nanstd(data: np.ndarray, axis: Optional[int] = None, ddof: int = 0) -> Union[float, np.ndarray]:
    """
    Calculate standard deviation while ignoring NaN values, similar to MATLAB's nanstd.

    Args:
        data: Input data array
        axis: Axis along which to compute standard deviation
        ddof: Delta degrees of freedom

    Returns:
        Standard deviation value(s) with NaN values ignored
    """
    if data.size == 0:
        return np.nan

    # Handle all-NaN case
    if np.all(np.isnan(data)):
        if axis is None:
            return np.nan
        else:
            result_shape = list(data.shape)
            result_shape.pop(axis)
            return np.full(result_shape, np.nan)

    return np.nanstd(data, axis=axis, ddof=ddof)


def fill_nan_with_interpolation(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Fill NaN values using interpolation.

    Args:
        data: Input data with potential NaN values
        method: Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
        Data with NaN values filled
    """
    if not np.any(np.isnan(data)):
        return data.copy()

    result = data.copy()

    if data.ndim == 1:
        # 1D case
        valid_mask = ~np.isnan(data)
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            result[~valid_mask] = np.interp(
                np.where(~valid_mask)[0],
                valid_indices,
                data[valid_mask]
            )
    else:
        # 2D case - interpolate along each axis
        from scipy import ndimage

        # Use distance transform for nearest neighbor interpolation
        if method == 'nearest':
            invalid_mask = np.isnan(data)
            if np.any(invalid_mask):
                # Find nearest valid pixel
                distance, indices = ndimage.distance_transform_edt(
                    invalid_mask, return_indices=True
                )
                result[invalid_mask] = data[tuple(indices[:, invalid_mask])]

    return result


# ==============================================================================
# Original Utility Functions (preserved)
# ==============================================================================

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string

    Examples:
        >>> format_time(65.5)
        '1m 5.5s'
        >>> format_time(3661)
        '1h 1m 1s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.0f}s"


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Raises:
        OSError: If directory cannot be created
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create directory {directory_path}: {e}")


def validate_image_dimensions(image: np.ndarray) -> Tuple[int, int]:
    """
    Validate image dimensions and return size.

    Args:
        image: Input image array

    Returns:
        Tuple of (height, width)

    Raises:
        ValueError: If image is not 2D or 3D
    """
    if image.ndim == 2:
        return image.shape
    elif image.ndim == 3:
        return image.shape[:2]
    else:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")


def find_unique_rows(array: np.ndarray, return_index: bool = False, return_inverse: bool = False) -> Union[np.ndarray, Tuple]:
    """
    Find unique rows in a 2D array, similar to MATLAB's unique function with 'rows' option.

    Args:
        array: Input 2D array
        return_index: If True, return indices of unique rows
        return_inverse: If True, return inverse indices

    Returns:
        Unique rows, and optionally indices

    Examples:
        >>> arr = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        >>> unique_rows = find_unique_rows(arr)
        >>> print(unique_rows)
        [[1 2]
         [3 4]
         [5 6]]
    """
    # Convert rows to a structured array for unique operation
    dtype = np.dtype(','.join([array.dtype.str] * array.shape[1]))
    structured_array = array.view(dtype).flatten()

    # Find unique values
    if return_index or return_inverse:
        unique_structured, indices, inverse = np.unique(
            structured_array, return_index=True, return_inverse=True
        )
        unique_rows = array[indices]

        result = [unique_rows]
        if return_index:
            result.append(indices)
        if return_inverse:
            result.append(inverse)
        return tuple(result) if len(result) > 1 else result[0]
    else:
        unique_structured = np.unique(structured_array)
        # Get the indices of unique rows
        unique_indices = []
        for unique_val in unique_structured:
            idx = np.where(structured_array == unique_val)[0][0]
            unique_indices.append(idx)
        return array[unique_indices]


def matlab_mod(x: Union[int, float, np.ndarray], y: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    MATLAB-style modulo operation.

    In MATLAB, mod(x, y) has the same sign as y, whereas Python's % operator
    has the same sign as x for negative numbers.

    Args:
        x: Dividend
        y: Divisor

    Returns:
        Modulo result with MATLAB behavior

    Examples:
        >>> matlab_mod(-1, 3)
        2
        >>> matlab_mod([-1, -2, 1, 2], 3)
        array([2, 1, 1, 2])
    """
    return ((x % y) + y) % y


def print_memory_usage() -> None:
    """
    Print current memory usage (if psutil is available).
    Useful for debugging memory issues in large image processing tasks.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        print("psutil not available for memory monitoring")


class Timer:
    """
    Simple context manager for timing code execution.

    Examples:
        >>> with Timer("Image processing"):
        ...     # Some time-consuming operation
        ...     time.sleep(1)
        Image processing took 1.0s

        >>> timer = Timer()
        >>> timer.start()
        >>> # Some operation
        >>> elapsed = timer.stop()
        >>> print(f"Operation took {elapsed:.2f} seconds")
    """

    def __init__(self, description: Optional[str] = None):
        self.description = description
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        elapsed = self.stop()
        if self.description:
            print(f"{self.description} took {format_time(elapsed)}")

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        self.end_time = time.time()
        return self.end_time - self.start_time

    def elapsed(self) -> float:
        """Get elapsed time without stopping the timer."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        return time.time() - self.start_time


def validate_psf_sigma(psf_sigma: float, min_sigma: float = 0.1, max_sigma: float = 10.0) -> float:
    """
    Validate and clamp PSF sigma to reasonable bounds.

    Args:
        psf_sigma: Input PSF sigma value
        min_sigma: Minimum allowed sigma
        max_sigma: Maximum allowed sigma

    Returns:
        Validated PSF sigma

    Raises:
        ValueError: If sigma is not positive
    """
    if psf_sigma <= 0:
        raise ValueError("PSF sigma must be positive")

    if psf_sigma < min_sigma:
        print(f"Warning: PSF sigma {psf_sigma} is very small, clamping to {min_sigma}")
        return min_sigma
    elif psf_sigma > max_sigma:
        print(f"Warning: PSF sigma {psf_sigma} is very large, clamping to {max_sigma}")
        return max_sigma

    return psf_sigma


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default_value: float = 0.0) -> np.ndarray:
    """
    Safely divide two arrays, handling division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero

    Returns:
        Result of division with safe handling of zero denominators
    """
    result = np.full_like(numerator, default_value, dtype=float)
    valid_mask = denominator != 0
    result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    return result

def safe_extract_coordinates(frame_info):
    """
    Safely extract coordinates from frame info, handling various data formats.

    Args:
        frame_info: MovieInfo frame object

    Returns:
        Tuple of (x_coords, y_coords, amplitudes, uncertainties)
    """
    x_coords = []
    y_coords = []
    amplitudes = []
    uncertainties = []

    if not hasattr(frame_info, 'xCoord') or frame_info.xCoord is None:
        return np.array([]), np.array([]), np.array([]), []

    x_data = frame_info.xCoord
    y_data = frame_info.yCoord if hasattr(frame_info, 'yCoord') and frame_info.yCoord is not None else None
    amp_data = frame_info.amp if hasattr(frame_info, 'amp') and frame_info.amp is not None else None

    try:
        # Handle different coordinate formats
        if x_data.ndim == 2 and x_data.shape[1] >= 1:
            x_coords = x_data[:, 0]
            x_uncertainties = x_data[:, 1] if x_data.shape[1] >= 2 else np.full(len(x_coords), 0.1)
        elif x_data.ndim == 1:
            x_coords = x_data
            x_uncertainties = np.full(len(x_coords), 0.1)
        elif x_data.ndim == 0:  # Single detection
            x_coords = np.array([float(x_data)])
            x_uncertainties = np.array([0.1])
        else:
            return np.array([]), np.array([]), np.array([]), []

        # Handle y coordinates
        if y_data is not None:
            if y_data.ndim == 2 and y_data.shape[1] >= 1:
                y_coords = y_data[:, 0]
                y_uncertainties = y_data[:, 1] if y_data.shape[1] >= 2 else np.full(len(y_coords), 0.1)
            elif y_data.ndim == 1:
                y_coords = y_data
                y_uncertainties = np.full(len(y_coords), 0.1)
            elif y_data.ndim == 0:  # Single detection
                y_coords = np.array([float(y_data)])
                y_uncertainties = np.array([0.1])
            else:
                y_coords = np.arange(len(x_coords))
                y_uncertainties = np.full(len(x_coords), 0.1)
        else:
            y_coords = np.arange(len(x_coords))
            y_uncertainties = np.full(len(x_coords), 0.1)

        # Handle amplitudes
        if amp_data is not None:
            if amp_data.ndim == 2 and amp_data.shape[1] >= 1:
                amplitudes = amp_data[:, 0]
            elif amp_data.ndim == 1:
                amplitudes = amp_data
            elif amp_data.ndim == 0:  # Single detection
                amplitudes = np.array([float(amp_data)])
            else:
                amplitudes = np.ones(len(x_coords))
        else:
            amplitudes = np.ones(len(x_coords))

        # Ensure all arrays have the same length
        min_length = min(len(x_coords), len(y_coords), len(amplitudes))
        if min_length > 0:
            x_coords = x_coords[:min_length]
            y_coords = y_coords[:min_length]
            amplitudes = amplitudes[:min_length]
            x_uncertainties = x_uncertainties[:min_length]
            y_uncertainties = y_uncertainties[:min_length]
            uncertainties = list(zip(x_uncertainties, y_uncertainties))

    except Exception as e:
        print(f"Warning: Error extracting coordinates: {e}")
        return np.array([]), np.array([]), np.array([]), []

    return x_coords, y_coords, amplitudes, uncertainties

# ==============================================================================
# Testing Functions
# ==============================================================================

def test_coordinate_conversion():
    """Test coordinate conversion functions."""
    print("Testing coordinate conversion...")

    # Test MATLAB <-> Python conversion
    matlab_coords = np.array([[1, 1], [10, 20], [100, 200]])
    python_coords = matlab_to_python_coords(matlab_coords)
    back_to_matlab = python_to_matlab_coords(python_coords)

    assert np.allclose(matlab_coords, back_to_matlab), "Coordinate conversion failed"
    print("✓ Coordinate conversion tests passed")


def test_statistical_functions():
    """Test statistical functions."""
    print("Testing statistical functions...")

    # Test normcdf
    x = np.array([-2, -1, 0, 1, 2])
    result = normcdf(x)
    expected = stats.norm.cdf(x)
    assert np.allclose(result, expected), "normcdf failed"

    # Test prctile
    data = np.random.randn(100)
    result = prctile(data, 50)
    expected = np.percentile(data, 50)
    assert np.isclose(result, expected), "prctile failed"

    print("✓ Statistical function tests passed")


# Example usage and testing
if __name__ == "__main__":
    # Test progress_text
    print("Testing progress_text:")
    for i in range(11):
        progress_text(i / 10, "Processing images")
        time.sleep(0.1)

    print("\nTesting create_distance_matrix:")
    # Test distance matrix
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    dist_matrix = create_distance_matrix(points)
    print("Points:")
    print(points)
    print("Distance matrix:")
    print(dist_matrix)

    print("\nTesting Timer:")
    # Test timer
    with Timer("Sleep test"):
        time.sleep(0.5)

    print("\nTesting unique rows:")
    # Test unique rows
    test_array = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4]])
    unique_rows, indices = find_unique_rows(test_array, return_index=True)
    print("Original array:")
    print(test_array)
    print("Unique rows:")
    print(unique_rows)
    print("Indices:")
    print(indices)

    # Run new tests
    print("\nTesting new utilities:")
    test_coordinate_conversion()
    test_statistical_functions()
    print("\nAll tests passed!")
