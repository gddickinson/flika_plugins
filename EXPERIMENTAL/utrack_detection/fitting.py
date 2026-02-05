#!/usr/bin/env python3
"""
Fitting module for u-track Python port.

This module provides functions for Gaussian fitting and robust statistical estimation,
equivalent to the MATLAB functions used in detectSubResFeatures2D_StandAlone.

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.
"""

import numpy as np
import scipy.optimize
from scipy import stats
from typing import Optional, Tuple, List, Union
import warnings

# Handle OptimizeWarning import
try:
    from scipy.optimize import OptimizeWarning
except ImportError:
    # Create a dummy class if OptimizeWarning doesn't exist
    class OptimizeWarning(UserWarning):
        pass


def gauss_2d_function(coords: np.ndarray, x0: float, y0: float, amplitude: float,
                     sigma: float, background: float) -> np.ndarray:
    """
    2D Gaussian function for fitting.

    Args:
        coords: Array of shape (N, 2) with [x, y] coordinates
        x0, y0: Center coordinates
        amplitude: Peak amplitude above background
        sigma: Standard deviation (assuming isotropic Gaussian)
        background: Background level

    Returns:
        1D array of function values at the given coordinates
    """
    x, y = coords[:, 0], coords[:, 1]

    # Calculate 2D Gaussian
    gauss = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + background

    return gauss


def gauss_2d_function_grid(xy_tuple: Tuple[np.ndarray, np.ndarray], x0: float, y0: float,
                          amplitude: float, sigma: float, background: float) -> np.ndarray:
    """
    2D Gaussian function for fitting on a grid (alternative interface).

    Args:
        xy_tuple: Tuple of (X, Y) meshgrid arrays
        x0, y0: Center coordinates
        amplitude: Peak amplitude above background
        sigma: Standard deviation
        background: Background level

    Returns:
        Flattened array of function values
    """
    X, Y = xy_tuple

    # Calculate 2D Gaussian
    gauss = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) + background

    return gauss.ravel()


def gauss_fit_nd(image: np.ndarray, coordinates: Optional[np.ndarray] = None,
                fit_parameters: List[str] = None, init_guess: List[float] = None,
                bounds: Optional[Tuple] = None) -> Optional[np.ndarray]:
    """
    N-dimensional Gaussian fitting function.

    This function fits a 2D Gaussian to image data, equivalent to the MATLAB GaussFitND function.

    Args:
        image: 2D image array to fit
        coordinates: Optional coordinate array. If None, uses image pixel coordinates
        fit_parameters: List of parameter names to fit ['X1', 'X2', 'A', 'Sxy', 'B']
        init_guess: Initial parameter guess [x0, y0, amplitude, sigma, background]
        bounds: Optional bounds for parameters

    Returns:
        Array of fitted parameters [x0, y0, amplitude, sigma, background] or None if fit fails
    """

    if image is None or image.size == 0:
        return None

    # Handle default parameters
    if fit_parameters is None:
        fit_parameters = ['X1', 'X2', 'A', 'Sxy', 'B']

    # Get image dimensions
    if image.ndim == 1:
        # Assume square image and reshape
        size = int(np.sqrt(len(image)))
        if size * size == len(image):
            image = image.reshape(size, size)
        else:
            return None

    rows, cols = image.shape

    # Create coordinate arrays if not provided
    if coordinates is None:
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        coordinates = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    else:
        # If coordinates provided, create meshgrid
        if coordinates.shape[1] == 2:
            x_coords = coordinates[:, 0].reshape(rows, cols)
            y_coords = coordinates[:, 1].reshape(rows, cols)
        else:
            return None

    # Flatten image for fitting
    image_flat = image.ravel()

    # Remove NaN values
    valid_mask = ~np.isnan(image_flat)
    if not np.any(valid_mask):
        return None

    image_flat = image_flat[valid_mask]
    coordinates_valid = coordinates[valid_mask]

    # Set default initial guess if not provided
    if init_guess is None:
        # Estimate initial parameters
        background_est = np.percentile(image_flat, 10)  # 10th percentile as background
        amplitude_est = np.max(image_flat) - background_est

        # Find peak location
        peak_idx = np.argmax(image_flat)
        x0_est = coordinates_valid[peak_idx, 0]
        y0_est = coordinates_valid[peak_idx, 1]

        # Estimate sigma from image size
        sigma_est = min(rows, cols) / 6.0

        init_guess = [x0_est, y0_est, amplitude_est, sigma_est, background_est]

    # Ensure init_guess has correct length
    if len(init_guess) < 5:
        init_guess.extend([1.0] * (5 - len(init_guess)))

    # Set up parameter bounds
    if bounds is None:
        # Default bounds
        x_bounds = (-0.5, cols + 0.5)
        y_bounds = (-0.5, rows + 0.5)
        amp_bounds = (0, np.max(image_flat) * 2)
        sigma_bounds = (0.1, max(rows, cols))
        bg_bounds = (np.min(image_flat) - abs(np.min(image_flat)),
                    np.max(image_flat) + abs(np.max(image_flat)))

        bounds = ([x_bounds[0], y_bounds[0], amp_bounds[0], sigma_bounds[0], bg_bounds[0]],
                 [x_bounds[1], y_bounds[1], amp_bounds[1], sigma_bounds[1], bg_bounds[1]])

    try:
        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Perform the fit
            popt, pcov = scipy.optimize.curve_fit(
                gauss_2d_function,
                coordinates_valid,
                image_flat,
                p0=init_guess,
                bounds=bounds,
                maxfev=2000,
                method='trf'  # Trust Region Reflective algorithm
            )

        # Validate the fit
        if np.any(np.isnan(popt)) or np.any(np.isinf(popt)):
            return None

        # Check if sigma is reasonable
        if popt[3] <= 0 or popt[3] > max(rows, cols):
            return None

        # Check if center is within reasonable bounds
        if (popt[0] < -rows or popt[0] > 2*cols or
            popt[1] < -cols or popt[1] > 2*rows):
            return None

        return popt

    except (RuntimeError, ValueError, OptimizeWarning):
        # Fit failed
        return None


def robust_mean(data: np.ndarray, weights: Optional[np.ndarray] = None,
               k: float = 3.0, max_iter: int = 0, return_inliers: bool = False) -> Union[float, Tuple]:
    """
    Calculate robust mean using iterative outlier removal.

    This function computes a robust estimate of the mean by iteratively removing
    outliers based on the median absolute deviation (MAD) or standard deviation.

    Args:
        data: 1D array of data values
        weights: Optional weights for each data point
        k: Number of standard deviations for outlier threshold (default: 3.0)
        max_iter: Maximum number of iterations (0 = automatic, default: 0)
        return_inliers: Whether to return inlier indices (default: False)

    Returns:
        If return_inliers is False: robust mean value
        If return_inliers is True: tuple of (robust_mean, robust_std, inlier_mask)
    """

    if data is None or len(data) == 0:
        if return_inliers:
            return np.nan, np.nan, np.array([], dtype=bool)
        return np.nan

    # Convert to numpy array and remove NaN values
    data = np.asarray(data, dtype=float)
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        if return_inliers:
            return np.nan, np.nan, valid_mask
        return np.nan

    data = data[valid_mask]

    if weights is not None:
        weights = np.asarray(weights)[valid_mask]
        if len(weights) != len(data):
            weights = None

    # Set maximum iterations if not specified
    if max_iter <= 0:
        max_iter = min(10, len(data))

    # Initialize variables
    current_data = data.copy()
    current_weights = weights.copy() if weights is not None else None
    inlier_mask = np.ones(len(data), dtype=bool)

    for iteration in range(max_iter):
        if len(current_data) < 3:  # Need at least 3 points
            break

        # Calculate current statistics
        if current_weights is not None:
            # Weighted statistics
            current_mean = np.average(current_data, weights=current_weights)
            variance = np.average((current_data - current_mean)**2, weights=current_weights)
            current_std = np.sqrt(variance)
        else:
            # Unweighted statistics
            current_mean = np.mean(current_data)
            current_std = np.std(current_data, ddof=1)

        if current_std == 0:  # All points are identical
            break

        # Calculate z-scores
        z_scores = np.abs(current_data - current_mean) / current_std

        # Find inliers
        current_inliers = z_scores <= k

        # Check if any outliers were found
        if np.all(current_inliers):
            break  # No more outliers to remove

        # Update data and weights for next iteration
        new_inlier_indices = np.where(inlier_mask)[0][current_inliers]
        inlier_mask = np.zeros(len(data), dtype=bool)
        inlier_mask[new_inlier_indices] = True

        current_data = data[inlier_mask]
        if weights is not None:
            current_weights = weights[inlier_mask]

    # Calculate final robust statistics
    if len(current_data) == 0:
        robust_mean_val = np.nan
        robust_std_val = np.nan
    else:
        if current_weights is not None:
            robust_mean_val = np.average(current_data, weights=current_weights)
            variance = np.average((current_data - robust_mean_val)**2, weights=current_weights)
            robust_std_val = np.sqrt(variance)
        else:
            robust_mean_val = np.mean(current_data)
            robust_std_val = np.std(current_data, ddof=1)

    # Extend inlier mask to original size (including NaN values)
    full_inlier_mask = np.zeros(len(valid_mask), dtype=bool)
    full_inlier_mask[valid_mask] = inlier_mask

    if return_inliers:
        return robust_mean_val, robust_std_val, full_inlier_mask
    else:
        return robust_mean_val


def estimate_noise_std(image: np.ndarray, method: str = 'mad') -> float:
    """
    Estimate noise standard deviation from an image.

    Args:
        image: 2D image array
        method: Method to use ('mad' for median absolute deviation, 'std' for standard deviation)

    Returns:
        Estimated noise standard deviation
    """

    if image is None or image.size == 0:
        return 0.0

    # Remove NaN values
    valid_data = image[~np.isnan(image)]

    if len(valid_data) == 0:
        return 0.0

    if method.lower() == 'mad':
        # Use median absolute deviation
        median_val = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median_val))
        # Convert MAD to standard deviation estimate (for normal distribution)
        noise_std = mad * 1.4826
    else:
        # Use robust standard deviation
        noise_std = robust_mean(valid_data, return_inliers=False)
        # Get the robust std from the full function
        _, noise_std, _ = robust_mean(valid_data, return_inliers=True)

    return noise_std


def fit_gaussian_mixture_1d(data: np.ndarray, num_components: int = 2,
                           max_iter: int = 100) -> Optional[dict]:
    """
    Fit a 1D Gaussian mixture model to data.

    Args:
        data: 1D array of data values
        num_components: Number of Gaussian components
        max_iter: Maximum number of EM iterations

    Returns:
        Dictionary with fitted parameters or None if fit fails
    """

    try:
        from sklearn.mixture import GaussianMixture

        if len(data) < num_components * 3:  # Need enough data points
            return None

        # Reshape data for sklearn
        data_reshaped = data.reshape(-1, 1)

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=num_components, max_iter=max_iter,
                             random_state=42, covariance_type='full')
        gmm.fit(data_reshaped)

        if gmm.converged_:
            return {
                'means': gmm.means_.flatten(),
                'covariances': gmm.covariances_.flatten(),
                'weights': gmm.weights_,
                'aic': gmm.aic(data_reshaped),
                'bic': gmm.bic(data_reshaped)
            }
        else:
            return None

    except ImportError:
        # sklearn not available, use simple approach
        warnings.warn("sklearn not available, using simplified mixture fitting")
        return None
    except Exception:
        return None


# Additional utility functions for fitting

def goodness_of_fit_2d_gaussian(image: np.ndarray, fitted_params: np.ndarray,
                                coordinates: Optional[np.ndarray] = None) -> dict:
    """
    Calculate goodness of fit metrics for 2D Gaussian fit.

    Args:
        image: Original image data
        fitted_params: Fitted Gaussian parameters [x0, y0, amplitude, sigma, background]
        coordinates: Coordinate array (optional)

    Returns:
        Dictionary with goodness of fit metrics
    """

    if coordinates is None:
        rows, cols = image.shape
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        coordinates = np.column_stack([x_coords.ravel(), y_coords.ravel()])

    # Calculate fitted values
    fitted_values = gauss_2d_function(coordinates, *fitted_params)
    fitted_image = fitted_values.reshape(image.shape)

    # Remove NaN values for calculations
    valid_mask = ~np.isnan(image.ravel())
    observed = image.ravel()[valid_mask]
    predicted = fitted_values[valid_mask]

    # Calculate metrics
    residuals = observed - predicted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(residuals**2))
    chi_squared = ss_res / np.var(observed) if np.var(observed) > 0 else np.inf

    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'chi_squared': chi_squared,
        'residuals': residuals,
        'fitted_image': fitted_image
    }
