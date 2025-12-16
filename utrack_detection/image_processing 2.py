#!/usr/bin/env python3
"""
Image processing functions for u-track Python port.

This module contains image processing utilities for feature detection,
including Gaussian filtering, background estimation, and local maxima detection.

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.

u-track is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy as np
import warnings
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter
from typing import Tuple, Optional, Union, List
import matplotlib.pyplot as plt

# Import utilities
try:
    # Try relative imports first (when run as module)
    from .utils import robust_nanmean, robust_nanstd, fill_nan_with_interpolation
except ImportError:
    # Fallback to direct imports (when run as script)
    from utils import robust_nanmean, robust_nanstd, fill_nan_with_interpolation


def filter_gauss_2d(image: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """
    Apply 2D Gaussian filter to an image.

    This function applies a 2D Gaussian filter to smooth the input image.
    Equivalent to MATLAB's imgaussfilt or filterGauss2D.

    Args:
        image: Input 2D image array
        sigma: Standard deviation of the Gaussian kernel
        truncate: Truncate the filter at this many standard deviations

    Returns:
        Filtered image of the same size as input

    Raises:
        ValueError: If image is not 2D or sigma is non-positive
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")

    if sigma <= 0:
        return image.copy()  # Return copy if no filtering needed

    # Handle NaN values - preserve them in the output
    nan_mask = np.isnan(image)
    if np.any(nan_mask):
        # Create a copy and fill NaNs with zeros for filtering
        image_filtered = image.copy()
        image_filtered[nan_mask] = 0

        # Apply Gaussian filter
        filtered = gaussian_filter(image_filtered, sigma=sigma, truncate=truncate, mode='constant')

        # Create a mask image and filter it to normalize the result
        mask = (~nan_mask).astype(float)
        mask_filtered = gaussian_filter(mask, sigma=sigma, truncate=truncate, mode='constant')

        # Avoid division by zero
        mask_filtered[mask_filtered == 0] = np.nan
        filtered = filtered / mask_filtered

        # Restore NaN locations
        filtered[nan_mask] = np.nan

        return filtered
    else:
        # No NaNs, apply filter directly
        return gaussian_filter(image, sigma=sigma, truncate=truncate, mode='constant')


def locmax_2d(image: np.ndarray, window_size: Union[int, List[int]],
              threshold_rel: float = 0.0, threshold_abs: Optional[float] = None,
              exclude_border: bool = True) -> np.ndarray:
    """
    Enhanced 2D local maxima detection matching MATLAB locmax2d more closely.

    This function identifies local maxima by comparing each pixel to its neighborhood.
    Equivalent to MATLAB's locmax2d function with improved robustness.

    Args:
        image: Input 2D image
        window_size: Size of neighborhood window. Can be int or [height, width]
        threshold_rel: Relative threshold (fraction of image range)
        threshold_abs: Absolute threshold value (optional)
        exclude_border: Whether to exclude border pixels from detection

    Returns:
        Binary image where True indicates local maxima locations
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")

    # Handle window size parameter
    if isinstance(window_size, int):
        window_h = window_w = window_size
    elif isinstance(window_size, (list, tuple)) and len(window_size) == 2:
        window_h, window_w = window_size
    else:
        raise ValueError("window_size must be int or [height, width]")

    # Ensure odd window sizes for symmetric neighborhoods
    if window_h % 2 == 0:
        window_h += 1
    if window_w % 2 == 0:
        window_w += 1

    # Create a copy to work with, handling NaNs
    img_work = image.copy()
    nan_mask = np.isnan(img_work)

    # Replace NaNs with very negative values so they won't be maxima
    img_work[nan_mask] = -np.inf

    # Calculate thresholds
    if threshold_abs is None:
        # Use relative threshold
        img_range = np.nanmax(image) - np.nanmin(image)
        threshold_abs = np.nanmin(image) + threshold_rel * img_range

    # Apply threshold - pixels below threshold cannot be maxima
    below_threshold = img_work < threshold_abs
    img_work[below_threshold] = -np.inf

    # Find local maxima using maximum filter
    # A pixel is a local maximum if it equals the maximum in its neighborhood
    max_filtered = maximum_filter(img_work, size=(window_h, window_w), mode='constant', cval=-np.inf)

    # Create binary mask of local maxima
    local_maxima = (img_work == max_filtered) & (img_work > -np.inf)

    # Exclude borders if requested
    if exclude_border:
        border_h = window_h // 2
        border_w = window_w // 2

        # Create border mask
        border_mask = np.ones_like(local_maxima, dtype=bool)
        if border_h > 0:
            border_mask[border_h:-border_h, :] = False
        if border_w > 0:
            border_mask[:, border_w:-border_w] = False

        # Remove border maxima
        local_maxima[border_mask] = False

    # Exclude NaN locations
    local_maxima[nan_mask] = False

    return local_maxima


def locmax2d(image: np.ndarray, mask_size: Union[int, List[int]] = 3,
            threshold: float = 1) -> np.ndarray:
    """
    MATLAB-compatible locmax2d function for enhanced compatibility.

    This is a wrapper around locmax_2d that matches MATLAB calling conventions.

    Args:
        image: Input 2D image
        mask_size: Size of neighborhood mask (int or [height, width])
        threshold: Threshold type (1 for relative threshold matching original)

    Returns:
        Binary image with local maxima marked as True
    """
    if threshold == 1:
        # MATLAB-style: threshold=1 means no thresholding, find all local maxima
        return locmax_2d(image, mask_size, threshold_rel=0.0, exclude_border=True)
    else:
        # Use threshold as absolute value
        return locmax_2d(image, mask_size, threshold_abs=threshold, exclude_border=True)


def spatial_move_ave_bg(image_stack: np.ndarray, size_x: int, size_y: int,
                       window_size: int = 50, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial moving average background using robust statistics.

    This function estimates background intensity by dividing the image into overlapping
    windows and calculating robust mean and standard deviation for each region.

    Args:
        image_stack: 3D array (x, y, frames) or 2D array (x, y) for single image
        size_x: Image size in x dimension
        size_y: Image size in y dimension
        window_size: Size of the sliding window for background estimation
        overlap: Overlap fraction between adjacent windows (0-1)

    Returns:
        Tuple of (bg_mean, bg_std) arrays of same x,y size as input
    """

    # Handle both 2D and 3D inputs
    if image_stack.ndim == 2:
        # Single image
        image = image_stack
    elif image_stack.ndim == 3:
        # Multiple images - take mean across frames, ignoring NaNs
        image = robust_nanmean(image_stack, axis=2)
    else:
        raise ValueError("Image stack must be 2D or 3D")

    # Initialize output arrays
    bg_mean = np.full((size_x, size_y), np.nan)
    bg_std = np.full((size_x, size_y), np.nan)

    # Calculate step size based on overlap
    step_size = int(window_size * (1 - overlap))
    step_size = max(step_size, 1)  # Ensure at least 1 pixel step

    # Create coordinate grids for interpolation
    x_coords = np.arange(size_x)
    y_coords = np.arange(size_y)

    # Lists to store window centers and statistics
    window_centers_x = []
    window_centers_y = []
    window_means = []
    window_stds = []

    # Slide window across the image
    for x_start in range(0, size_x, step_size):
        for y_start in range(0, size_y, step_size):

            # Define window boundaries
            x_end = min(x_start + window_size, size_x)
            y_end = min(y_start + window_size, size_y)

            # Extract window
            window = image[x_start:x_end, y_start:y_end]

            # Skip windows that are too small or contain only NaNs
            if window.size < 10 or np.all(np.isnan(window)):
                continue

            # Calculate robust statistics for this window
            window_data = window.ravel()
            window_data = window_data[~np.isnan(window_data)]

            if len(window_data) < 10:  # Need minimum data points
                continue

            # Use robust statistics (remove outliers that might be features)
            mean_val, std_val = _robust_bg_stats(window_data)

            # Store window center and statistics
            center_x = x_start + (x_end - x_start) // 2
            center_y = y_start + (y_end - y_start) // 2

            window_centers_x.append(center_x)
            window_centers_y.append(center_y)
            window_means.append(mean_val)
            window_stds.append(std_val)

    # Interpolate background values across entire image
    if len(window_centers_x) > 0:
        # Create meshgrid for interpolation
        xx, yy = np.meshgrid(y_coords, x_coords)

        # Convert window centers to arrays
        centers_x = np.array(window_centers_x)
        centers_y = np.array(window_centers_y)
        means = np.array(window_means)
        stds = np.array(window_stds)

        # Use nearest neighbor interpolation for now
        # (could use scipy.interpolate for smoother results)
        for i in range(size_x):
            for j in range(size_y):
                # Find nearest window center
                distances = np.sqrt((centers_x - i)**2 + (centers_y - j)**2)
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    bg_mean[i, j] = means[nearest_idx]
                    bg_std[i, j] = stds[nearest_idx]

    # Fill any remaining NaNs with global statistics
    if np.any(np.isnan(bg_mean)) and not np.all(np.isnan(image)):
        global_data = image.ravel()
        global_data = global_data[~np.isnan(global_data)]
        if len(global_data) > 0:
            global_mean, global_std = _robust_bg_stats(global_data)
            bg_mean[np.isnan(bg_mean)] = global_mean
            bg_std[np.isnan(bg_std)] = global_std

    return bg_mean, bg_std


def _robust_bg_stats(data: np.ndarray, k: float = 3.0, max_iter: int = 3) -> Tuple[float, float]:
    """
    Calculate robust mean and standard deviation by iteratively removing outliers.

    Args:
        data: 1D array of intensity values
        k: Number of standard deviations for outlier threshold
        max_iter: Maximum number of iterations

    Returns:
        Tuple of (robust_mean, robust_std)
    """
    data = data.copy()

    for _ in range(max_iter):
        if len(data) < 10:
            break

        mean_val = np.mean(data)
        std_val = np.std(data)

        if std_val == 0:
            break

        # Remove outliers (likely bright features)
        outlier_threshold = mean_val + k * std_val
        data = data[data <= outlier_threshold]

    if len(data) == 0:
        return 0.0, 0.0

    return np.mean(data), np.std(data)


def find_local_maxima_positions(image: np.ndarray, window_size: Union[int, List[int]],
                               threshold_rel: float = 0.0, threshold_abs: Optional[float] = None,
                               exclude_border: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find positions and amplitudes of local maxima in a 2D image.

    This is a convenience function that returns the coordinates and amplitudes
    of local maxima rather than just a binary mask.

    Args:
        image: Input 2D image
        window_size: Size of neighborhood window
        threshold_rel: Relative threshold (fraction of image range)
        threshold_abs: Absolute threshold value (optional)
        exclude_border: Whether to exclude border pixels

    Returns:
        Tuple of (x_coords, y_coords, amplitudes) where each is a 1D array
    """

    # Get binary mask of local maxima
    maxima_mask = locmax_2d(image, window_size, threshold_rel, threshold_abs, exclude_border)

    # Extract coordinates and amplitudes
    y_coords, x_coords = np.where(maxima_mask)  # Note: np.where returns (row, col) = (y, x)
    amplitudes = image[y_coords, x_coords]

    return x_coords, y_coords, amplitudes


def centroid_2d(image: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid (center of mass) of a 2D image.

    This is equivalent to the MATLAB centroid2D function.

    Args:
        image: 2D image array

    Returns:
        Centroid coordinates [x, y] in image coordinate system (1-indexed like MATLAB)
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2D image")

    # Handle NaN values
    valid_mask = ~np.isnan(image)
    if not np.any(valid_mask):
        return np.array([np.nan, np.nan])

    # Get image dimensions
    rows, cols = image.shape

    # Create coordinate grids (1-indexed to match MATLAB)
    y_coords, x_coords = np.mgrid[1:rows+1, 1:cols+1]

    # Only use valid (non-NaN) pixels
    image_valid = image[valid_mask]
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]

    # Calculate total intensity
    total_intensity = np.sum(image_valid)

    if total_intensity == 0:
        # If no intensity, return geometric center
        centroid_x = (cols + 1) / 2
        centroid_y = (rows + 1) / 2
    else:
        # Calculate weighted centroid
        centroid_x = np.sum(x_valid * image_valid) / total_intensity
        centroid_y = np.sum(y_valid * image_valid) / total_intensity

    return np.array([centroid_x, centroid_y])


def adaptive_threshold_otsu(image: np.ndarray) -> float:
    """
    Calculate Otsu's adaptive threshold for an image.

    Args:
        image: Input 2D image

    Returns:
        Optimal threshold value
    """
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(image[~np.isnan(image)])
    except ImportError:
        # Fallback implementation
        return _otsu_threshold_simple(image)


def _otsu_threshold_simple(image: np.ndarray) -> float:
    """
    Simple implementation of Otsu's thresholding.

    Args:
        image: Input 2D image

    Returns:
        Optimal threshold value
    """
    # Remove NaN values
    valid_data = image[~np.isnan(image)]
    if len(valid_data) == 0:
        return 0.0

    # Create histogram
    hist, bin_edges = np.histogram(valid_data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate class probabilities and means
    total_pixels = len(valid_data)

    max_variance = 0
    optimal_threshold = 0

    for i in range(1, len(hist) - 1):
        # Background class
        w0 = np.sum(hist[:i]) / total_pixels
        if w0 == 0:
            continue
        mu0 = np.sum(hist[:i] * bin_centers[:i]) / np.sum(hist[:i])

        # Foreground class
        w1 = np.sum(hist[i:]) / total_pixels
        if w1 == 0:
            continue
        mu1 = np.sum(hist[i:] * bin_centers[i:]) / np.sum(hist[i:])

        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bin_centers[i]

    return optimal_threshold


def enhance_contrast_adaptive(image: np.ndarray, clip_limit: float = 0.03,
                             grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Args:
        image: Input 2D image
        clip_limit: Clipping limit for histogram equalization
        grid_size: Size of the local region for adaptive equalization

    Returns:
        Enhanced image
    """
    try:
        from skimage import exposure
        return exposure.equalize_adapthist(image, clip_limit=clip_limit, kernel_size=grid_size)
    except ImportError:
        # Fallback: simple contrast enhancement
        return _simple_contrast_enhancement(image)


def _simple_contrast_enhancement(image: np.ndarray) -> np.ndarray:
    """
    Simple contrast enhancement as fallback.

    Args:
        image: Input 2D image

    Returns:
        Enhanced image
    """
    # Remove NaN values for calculation
    valid_mask = ~np.isnan(image)
    if not np.any(valid_mask):
        return image.copy()

    valid_data = image[valid_mask]

    # Calculate percentiles for robust contrast enhancement
    p2, p98 = np.percentile(valid_data, (2, 98))

    # Stretch contrast
    enhanced = image.copy()
    enhanced[valid_mask] = np.clip((valid_data - p2) / (p98 - p2), 0, 1)

    return enhanced


def visualize_local_maxima(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray,
                          title: str = "Local Maxima Detection", figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Visualize detected local maxima overlaid on the original image.

    Args:
        image: Original 2D image
        x_coords: X coordinates of detected maxima
        y_coords: Y coordinates of detected maxima
        title: Plot title
        figsize: Figure size (width, height)
    """

    plt.figure(figsize=figsize)

    # Display image
    plt.imshow(image, cmap='gray', origin='lower')
    plt.colorbar(label='Intensity')

    # Overlay detected maxima
    if len(x_coords) > 0:
        plt.scatter(x_coords, y_coords, c='red', s=30, marker='o',
                   facecolors='none', edgecolors='red', linewidths=1.5)

    plt.title(f"{title} ({len(x_coords)} maxima detected)")
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.show()


def create_detection_summary_image(image: np.ndarray, detections: List,
                                  detection_types: List[str] = None) -> np.ndarray:
    """
    Create a 3-channel summary image showing original image and different detection types.

    Args:
        image: Original 2D image
        detections: List of detection coordinate arrays
        detection_types: List of detection type names

    Returns:
        3-channel RGB image with detections overlaid
    """
    # Normalize image
    img_norm = image.astype(float)
    if img_norm.max() > img_norm.min():
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())

    # Create 3-channel image
    rgb_image = np.stack([img_norm, img_norm, img_norm], axis=2)

    # Color map for different detection types
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    # Overlay detections
    for i, detection_coords in enumerate(detections):
        if len(detection_coords) > 0 and len(detection_coords[0]) >= 2:
            color = colors[i % len(colors)]
            y_coords = detection_coords[:, 1].astype(int)  # Row coordinates
            x_coords = detection_coords[:, 0].astype(int)  # Column coordinates

            # Ensure coordinates are within image bounds
            valid_mask = ((y_coords >= 0) & (y_coords < image.shape[0]) &
                         (x_coords >= 0) & (x_coords < image.shape[1]))

            if np.any(valid_mask):
                y_valid = y_coords[valid_mask]
                x_valid = x_coords[valid_mask]

                # Set detection pixels to color
                for channel in range(3):
                    rgb_image[y_valid, x_valid, channel] = color[channel]

    return rgb_image


def test_image_processing_functions():
    """
    Test function to verify the image processing functions work correctly.
    """

    print("Testing image processing functions...")

    # Create a test image with some peaks
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Add some Gaussian peaks
    peak1 = 100 * np.exp(-((X - 3)**2 + (Y - 2)**2) / 2)
    peak2 = 80 * np.exp(-((X + 2)**2 + (Y + 3)**2) / 4)
    peak3 = 60 * np.exp(-((X - 1)**2 + (Y + 1)**2) / 1)

    # Add noise
    noise = 5 * np.random.randn(*X.shape)
    test_image = peak1 + peak2 + peak3 + noise + 20  # Add baseline

    print(f"Test image shape: {test_image.shape}")
    print(f"Test image range: {np.min(test_image):.2f} to {np.max(test_image):.2f}")

    # Test Gaussian filtering
    print("\nTesting Gaussian filtering...")
    filtered = filter_gauss_2d(test_image, sigma=1.0)
    print(f"Filtered image range: {np.min(filtered):.2f} to {np.max(filtered):.2f}")

    # Test background estimation
    print("\nTesting background estimation...")
    bg_mean, bg_std = spatial_move_ave_bg(test_image, test_image.shape[0], test_image.shape[1])
    print(f"Background mean range: {np.min(bg_mean):.2f} to {np.max(bg_mean):.2f}")
    print(f"Background std range: {np.min(bg_std):.2f} to {np.max(bg_std):.2f}")

    # Test local maxima detection
    print("\nTesting local maxima detection...")
    x_max, y_max, amp_max = find_local_maxima_positions(filtered, window_size=5, threshold_rel=0.3)
    print(f"Found {len(x_max)} local maxima")

    if len(x_max) > 0:
        print("Maxima positions and amplitudes:")
        for i, (x, y, amp) in enumerate(zip(x_max, y_max, amp_max)):
            print(f"  Peak {i+1}: ({x}, {y}) with amplitude {amp:.2f}")

    # Test enhanced locmax2d
    print("\nTesting enhanced locmax2d...")
    maxima_mask = locmax2d(filtered, mask_size=5, threshold=1)
    num_maxima = np.sum(maxima_mask)
    print(f"Enhanced locmax2d found {num_maxima} maxima")

    # Test centroid calculation
    print("\nTesting centroid calculation...")
    # Create a small test region around the first peak
    if len(x_max) > 0:
        x_center, y_center = int(x_max[0]), int(y_max[0])
        x_start = max(0, x_center - 10)
        x_end = min(test_image.shape[1], x_center + 10)
        y_start = max(0, y_center - 10)
        y_end = min(test_image.shape[0], y_center + 10)

        test_region = filtered[y_start:y_end, x_start:x_end]
        centroid = centroid_2d(test_region)
        print(f"Centroid of test region: ({centroid[0]:.2f}, {centroid[1]:.2f})")

    # Visualize results (optional - uncomment to see plots)
    # visualize_local_maxima(filtered, x_max, y_max, "Test Local Maxima")

    print("\nImage processing functions test completed successfully!")

def normalize_image(image: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize image intensities using various methods.

    This function provides multiple normalization approaches commonly used
    in feature detection preprocessing.

    Args:
        image: Input 2D image array
        method: Normalization method. Options:
               'zscore' - Z-score normalization (mean=0, std=1)
               'minmax' - Min-max normalization to [0,1] range
               'percentile' - Percentile-based normalization (1st-99th percentile)
               'robust' - Robust normalization using median and MAD
               'clahe' - Contrast Limited Adaptive Histogram Equalization

    Returns:
        Normalized image of same shape as input
    """
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")

    # Handle NaN values
    valid_mask = ~np.isnan(image)
    if not np.any(valid_mask):
        return image.copy()

    result = image.copy()
    valid_data = image[valid_mask]

    if method == 'zscore':
        # Z-score normalization
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        if std_val > 0:
            result[valid_mask] = (valid_data - mean_val) / std_val
        else:
            result[valid_mask] = valid_data - mean_val

    elif method == 'minmax':
        # Min-max normalization to [0,1]
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        if max_val > min_val:
            result[valid_mask] = (valid_data - min_val) / (max_val - min_val)
        else:
            result[valid_mask] = 0

    elif method == 'percentile':
        # Percentile-based normalization (robust to outliers)
        p1, p99 = np.percentile(valid_data, [1, 99])
        if p99 > p1:
            clipped_data = np.clip(valid_data, p1, p99)
            result[valid_mask] = (clipped_data - p1) / (p99 - p1)
        else:
            result[valid_mask] = 0

    elif method == 'robust':
        # Robust normalization using median and MAD
        median_val = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median_val))
        if mad > 0:
            result[valid_mask] = (valid_data - median_val) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std for normal distribution
        else:
            result[valid_mask] = valid_data - median_val

    elif method == 'clahe':
        # Use the existing adaptive histogram equalization
        result = enhance_contrast_adaptive(image)

    else:
        # Unknown method - return copy
        warnings.warn(f"Unknown normalization method '{method}', returning original image")

    return result

if __name__ == "__main__":
    test_image_processing_functions()
