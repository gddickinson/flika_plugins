#!/usr/bin/env python3
"""
Detection module for u-track Python port.

This module provides the main feature detection functions using mixture model fitting,
equivalent to the MATLAB detectSubResFeatures2D_V2 function.

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.
"""

import numpy as np
import scipy.optimize
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import matplotlib.pyplot as plt

# Import helper modules
try:
    # Try relative imports first (when run as module)
    from .utils import create_distance_matrix
    from .fitting import gauss_fit_nd
except ImportError:
    # Fallback to direct imports (when run as script)
    from utils import create_distance_matrix
    from fitting import gauss_fit_nd


@dataclass
class ClusterFeature:
    """Structure representing a cluster of features"""
    numMaxima: int = 0
    maximaPos: np.ndarray = field(default_factory=lambda: np.array([]))
    maximaAmp: np.ndarray = field(default_factory=lambda: np.array([]))
    pixels: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ClusterMMF:
    """Structure for MMF cluster results"""
    position: np.ndarray = field(default_factory=lambda: np.array([]))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([]))
    bgAmp: np.ndarray = field(default_factory=lambda: np.array([]))
    numDegFree: int = 0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class DetectedFeatures:
    """Structure for final detected features"""
    xCoord: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    yCoord: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    amp: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def centroid_2d(image: np.ndarray) -> np.ndarray:
    """
    Calculate the centroid (center of mass) of a 2D image.

    This is equivalent to the MATLAB centroid2D function.

    Args:
        image: 2D image array

    Returns:
        Centroid coordinates [x, y] in image coordinate system
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


def find_overlap_psfs_2d(cands: List, num_pixels_x: int, num_pixels_y: int,
                        psf_sigma: float) -> Tuple[List[ClusterFeature], int]:
    """
    Find overlapping PSFs and group them into clusters.

    Args:
        cands: List of candidate features
        num_pixels_x: Number of pixels in x direction
        num_pixels_y: Number of pixels in y direction
        psf_sigma: PSF sigma

    Returns:
        Tuple of (clusters, error_flag)
    """
    clusters = []
    err_flag = 0

    if not cands:
        return clusters, err_flag

    try:
        # Get valid candidates
        valid_cands = [c for c in cands if hasattr(c, 'status') and c.status == 1]

        if not valid_cands:
            return clusters, err_flag

        # Extract positions and amplitudes
        positions = np.array([c.Lmax for c in valid_cands])
        amplitudes = np.array([c.amp for c in valid_cands])

        # Create clusters based on PSF overlap (simplified approach)
        # In practice, this should check for overlapping PSF regions
        cluster_radius = 3 * psf_sigma  # PSFs within this distance are considered overlapping

        unassigned = list(range(len(valid_cands)))
        cluster_id = 0

        while unassigned:
            # Start new cluster with first unassigned candidate
            seed_idx = unassigned[0]
            current_cluster = [seed_idx]
            unassigned.remove(seed_idx)

            # Find all candidates within cluster radius
            changed = True
            while changed:
                changed = False
                cluster_positions = positions[current_cluster]

                for i in list(unassigned):
                    distances = np.sqrt(np.sum((cluster_positions - positions[i])**2, axis=1))
                    if np.any(distances < cluster_radius):
                        current_cluster.append(i)
                        unassigned.remove(i)
                        changed = True

            # Create cluster structure
            cluster_indices = np.array(current_cluster)
            cluster_pos = positions[cluster_indices]
            cluster_amp = amplitudes[cluster_indices]

            # Generate pixel coordinates for this cluster
            min_x = max(0, int(np.min(cluster_pos[:, 0]) - 3 * psf_sigma))
            max_x = min(num_pixels_x, int(np.max(cluster_pos[:, 0]) + 3 * psf_sigma) + 1)
            min_y = max(0, int(np.min(cluster_pos[:, 1]) - 3 * psf_sigma))
            max_y = min(num_pixels_y, int(np.max(cluster_pos[:, 1]) + 3 * psf_sigma) + 1)

            # Create pixel grid
            x_coords, y_coords = np.meshgrid(range(min_x, max_x), range(min_y, max_y))
            pixels = np.column_stack([
                x_coords.ravel(),
                y_coords.ravel(),
                y_coords.ravel() * num_pixels_x + x_coords.ravel()  # Linear index
            ])

            # Add matrix coordinates (1-indexed to match MATLAB)
            cluster_pos_with_idx = np.column_stack([
                cluster_pos,
                (cluster_pos[:, 1].astype(int) * num_pixels_x +
                 cluster_pos[:, 0].astype(int))
            ])

            cluster = ClusterFeature(
                numMaxima=len(cluster_indices),
                maximaPos=cluster_pos_with_idx,
                maximaAmp=cluster_amp,
                pixels=pixels
            )
            clusters.append(cluster)
            cluster_id += 1

    except Exception:
        err_flag = 1

    return clusters, err_flag


def mmf_init_guess_lower_upper_bounds(maxima_pos: np.ndarray, maxima_amp: np.ndarray,
                                           bg_amp: float, psf_sigma: float,
                                           cluster_pixels: np.ndarray, first_fit: bool,
                                           var_sigma: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed version of initialize guess and bounds for mixture model fitting.
    """
    num_maxima = len(maxima_pos)

    # Build parameter vector: [x1, y1, amp1, [sigma1], x2, y2, amp2, [sigma2], ..., bg]
    params_per_max = 3 + int(var_sigma)
    total_params = num_maxima * params_per_max + 1

    x0 = np.zeros(total_params)
    lb = np.zeros(total_params)
    ub = np.zeros(total_params)

    # Safety check for empty cluster_pixels
    if len(cluster_pixels) == 0:
        print(f"    MMF Debug: cluster_pixels is empty! Creating minimal bounds.")
        # Return minimal valid bounds instead of empty lists
        x0 = np.array([maxima_pos[0, 0], maxima_pos[0, 1], maxima_amp[0], bg_amp])
        lb = np.array([maxima_pos[0, 0] - 5, maxima_pos[0, 1] - 5, 0, 0])
        ub = np.array([maxima_pos[0, 0] + 5, maxima_pos[0, 1] + 5, maxima_amp[0] * 2, bg_amp * 2])
        return x0, lb, ub

    # Get pixel bounds safely
    min_x, max_x = np.min(cluster_pixels[:, 0]), np.max(cluster_pixels[:, 0])
    min_y, max_y = np.min(cluster_pixels[:, 1]), np.max(cluster_pixels[:, 1])

    # Set parameters for each maximum
    for i in range(num_maxima):
        idx_start = i * params_per_max

        # Position parameters
        x0[idx_start:idx_start+2] = maxima_pos[i]
        lb[idx_start:idx_start+2] = [min_x - 0.5, min_y - 0.5]
        ub[idx_start:idx_start+2] = [max_x + 0.5, max_y + 0.5]

        # Amplitude parameter
        x0[idx_start+2] = maxima_amp[i]
        lb[idx_start+2] = 0
        ub[idx_start+2] = 2 * maxima_amp[i] if maxima_amp[i] > 0 else 1.0

        # Sigma parameter (if variable)
        if var_sigma:
            x0[idx_start+3] = psf_sigma
            lb[idx_start+3] = 0.1
            ub[idx_start+3] = 5 * psf_sigma

    # Background parameter
    x0[-1] = bg_amp
    lb[-1] = 0
    ub[-1] = 2 * bg_amp if bg_amp > 0 else 1.0

    return x0, lb, ub


def fit_n_gaussians_2d(params: np.ndarray, image_data: np.ndarray,
                      cluster_pixels: np.ndarray, psf_sigma: float) -> np.ndarray:
    """
    Fit N 2D Gaussians with fixed sigma to image data.

    Args:
        params: Parameter vector [x1, y1, amp1, x2, y2, amp2, ..., bg]
        image_data: Flattened image data
        cluster_pixels: Pixel coordinates
        psf_sigma: Fixed PSF sigma

    Returns:
        Residuals (model - data)
    """
    num_pixels = len(image_data)
    num_gaussians = (len(params) - 1) // 3

    # Initialize model
    model = np.zeros(num_pixels)

    # Add each Gaussian
    for i in range(num_gaussians):
        idx_start = i * 3
        x_center = params[idx_start]
        y_center = params[idx_start + 1]
        amplitude = params[idx_start + 2]

        # Calculate Gaussian contribution
        x_coords = cluster_pixels[:, 0]
        y_coords = cluster_pixels[:, 1]

        gaussian = amplitude * np.exp(
            -((x_coords - x_center)**2 + (y_coords - y_center)**2) / (2 * psf_sigma**2)
        )
        model += gaussian

    # Add background
    model += params[-1]

    # Return residuals
    return model - image_data


def fit_n_gaussians_2d_var_sigma(params: np.ndarray, image_data: np.ndarray,
                                 cluster_pixels: np.ndarray) -> np.ndarray:
    """
    Fit N 2D Gaussians with variable sigma to image data.

    Args:
        params: Parameter vector [x1, y1, amp1, sigma1, x2, y2, amp2, sigma2, ..., bg]
        image_data: Flattened image data
        cluster_pixels: Pixel coordinates

    Returns:
        Residuals (model - data)
    """
    num_pixels = len(image_data)
    num_gaussians = (len(params) - 1) // 4

    # Initialize model
    model = np.zeros(num_pixels)

    # Add each Gaussian
    for i in range(num_gaussians):
        idx_start = i * 4
        x_center = params[idx_start]
        y_center = params[idx_start + 1]
        amplitude = params[idx_start + 2]
        sigma = params[idx_start + 3]

        # Calculate Gaussian contribution
        x_coords = cluster_pixels[:, 0]
        y_coords = cluster_pixels[:, 1]

        gaussian = amplitude * np.exp(
            -((x_coords - x_center)**2 + (y_coords - y_center)**2) / (2 * sigma**2)
        )
        model += gaussian

    # Add background
    model += params[-1]

    # Return residuals
    return model - image_data


def mmf_dist_pv(maxima_pos: np.ndarray, var_cov_mat: np.ndarray,
               num_maxima: int, num_deg_free: int, var_sigma: bool = False) -> np.ndarray:
    """
    Calculate p-values for distances between maxima using proper error propagation.

    This function properly uses the full variance-covariance matrix to calculate
    the uncertainty in distances between maxima pairs.

    Args:
        maxima_pos: Positions with uncertainties [x, y, dx, dy]
        var_cov_mat: Full variance-covariance matrix of all fitted parameters
        num_maxima: Number of maxima
        num_deg_free: Degrees of freedom
        var_sigma: Whether sigma was variable in the fit

    Returns:
        Matrix of p-values for distances between maxima pairs
    """
    p_values = np.ones((num_maxima, num_maxima))

    # Number of parameters per maximum (3 for fixed sigma, 4 for variable sigma)
    params_per_max = 3 + int(var_sigma)

    for i in range(num_maxima):
        for j in range(i + 1, num_maxima):

            # Calculate distance
            x1, y1 = maxima_pos[i, 0], maxima_pos[i, 1]
            x2, y2 = maxima_pos[j, 0], maxima_pos[j, 1]
            dx = x1 - x2
            dy = y1 - y2
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0:  # Avoid division by zero

                # Calculate uncertainty in distance using error propagation
                # Extract relevant indices in the covariance matrix
                # Parameters are ordered as: [x1, y1, amp1, [sigma1], x2, y2, amp2, [sigma2], ..., bg]
                i_x_idx = i * params_per_max       # x position of maximum i
                i_y_idx = i * params_per_max + 1   # y position of maximum i
                j_x_idx = j * params_per_max       # x position of maximum j
                j_y_idx = j * params_per_max + 1   # y position of maximum j

                # Check bounds
                if (max(i_x_idx, i_y_idx, j_x_idx, j_y_idx) >= var_cov_mat.shape[0]):
                    # If indices are out of bounds, use simplified calculation
                    dx_unc = np.sqrt(maxima_pos[i, 2]**2 + maxima_pos[j, 2]**2)
                    dy_unc = np.sqrt(maxima_pos[i, 3]**2 + maxima_pos[j, 3]**2)
                    dist_var = dx_unc**2 + dy_unc**2
                else:
                    # Extract 4x4 submatrix for the two maxima positions
                    indices = [i_x_idx, i_y_idx, j_x_idx, j_y_idx]
                    cov_sub = var_cov_mat[np.ix_(indices, indices)]

                    # Calculate gradient of distance function
                    # d = sqrt((x1-x2)^2 + (y1-y2)^2)
                    # ∂d/∂x1 = (x1-x2)/d, ∂d/∂y1 = (y1-y2)/d
                    # ∂d/∂x2 = -(x1-x2)/d, ∂d/∂y2 = -(y1-y2)/d
                    gradient = np.array([
                        dx / distance,      # ∂d/∂x1
                        dy / distance,      # ∂d/∂y1
                        -dx / distance,     # ∂d/∂x2
                        -dy / distance      # ∂d/∂y2
                    ])

                    # Calculate variance in distance using error propagation
                    # var(d) = ∇d^T * Cov * ∇d
                    dist_var = gradient.T @ cov_sub @ gradient

                # Calculate standard deviation
                if dist_var > 0:
                    dist_std = np.sqrt(dist_var)

                    # Perform t-test for distance significance
                    # H0: distance = 0, H1: distance ≠ 0
                    t_stat = distance / dist_std

                    # Two-tailed test
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), num_deg_free))

                else:
                    # If variance calculation fails, assume distance is significant
                    p_value = 0.0
            else:
                # If distance is zero, it's definitely not significant
                p_value = 1.0

            # Store p-value in symmetric matrix
            p_values[i, j] = p_value
            p_values[j, i] = p_value

    return p_values


def detect_sub_res_features_2d_v2(
    image: np.ndarray,
    cands: List,
    psf_sigma: float,
    test_alpha: Dict[str, float],
    visual: bool = False,
    do_mmf: bool = True,
    bit_depth: Union[bool, int] = True,  # Can be bool or int for bit depth
    save_results: bool = False,
    bg_noise_sigma: Optional[float] = None,
    var_sigma: bool = False
) -> DetectedFeatures:
    """
    Determine positions and intensity amplitudes of sub-resolution features using mixture model fitting.

    This is the Python equivalent of detectSubResFeatures2D_V2.m

    Args:
        image: Image being analyzed
        cands: Candidates structure from initial detection
        psf_sigma: Standard deviation of PSF (pixels)
        test_alpha: Alpha values for statistical tests
        visual: Whether to show results
        do_mmf: Whether to do mixture-model fitting
        bit_depth: Camera bit depth (or boolean for processing)
        save_results: Whether to save results
        bg_noise_sigma: Background noise sigma (optional)
        var_sigma: Whether to estimate sigma from fit

    Returns:
        DetectedFeatures structure
    """

    # Initialize outputs
    detected_features = DetectedFeatures()
    clusters_mmf = []
    image_n3 = None
    err_flag = 0

    # Input validation
    if image is None or psf_sigma <= 0:
        print('--detectSubResFeatures2D_V2: Invalid input arguments!')
        return detected_features

    # Handle bit depth parameter
    if isinstance(bit_depth, bool):
        if bit_depth:
            actual_bit_depth = 14  # Default
        else:
            actual_bit_depth = 8  # Assume 8-bit if False
    else:
        actual_bit_depth = bit_depth

    # Store the intensity scaling factor for later use
    intensity_scale_factor = 2**actual_bit_depth - 1

    # Set default test alpha values
    default_alpha = {'alphaR': 0.05, 'alphaA': 0.05, 'alphaD': 0.05, 'alphaF': 0.05}
    for key, default_val in default_alpha.items():
        if key not in test_alpha:
            test_alpha[key] = default_val

    # Get image dimensions
    num_pixels_y, num_pixels_x = image.shape

    # Extract test alpha values
    alpha_r = test_alpha['alphaR']
    alpha_a = test_alpha['alphaA']
    alpha_d = test_alpha['alphaD']
    alpha_f = test_alpha['alphaF']

    # Normalize image for processing (but keep track of original scale)
    image = image.astype(float) / intensity_scale_factor

    # Get background amplitude information from candidates
    if cands:
        bg_amps = [c.IBkg for c in cands if hasattr(c, 'status') and c.status == 1]
        if bg_amps:
            bg_amp_max = max(bg_amps)
            bg_amp_ave = np.mean(bg_amps)
        else:
            bg_amp_ave = 0.1
    else:
        bg_amp_ave = 0.1

    # Scale background amplitude to normalized range
    bg_amp_ave = bg_amp_ave / intensity_scale_factor

    # Determine signal overlap
    clusters, err_flag = find_overlap_psfs_2d(cands, num_pixels_x, num_pixels_y, psf_sigma)
    if err_flag:
        print('--detectSubResFeatures2D_V2: Could not place signals in clusters!')
        return detected_features

    if not clusters:
        return detected_features

    # Initialize vector indicating whether clusters should be retained
    keep_cluster = np.ones(len(clusters), dtype=bool)

    # Set optimization options (scipy equivalent)
    options = {
        'max_nfev': 10000,
        'xtol': 1e-6,
        'ftol': 1e-6
    }

    # Reserve memory for clustersMMF
    num_clusters = len(clusters)
    clusters_mmf = [ClusterMMF() for _ in range(num_clusters)]

    # Go over all clusters
    for i in range(num_clusters):

        # Get initial guess of positions and amplitudes
        num_maxima_t = clusters[i].numMaxima
        maxima_pos_t = clusters[i].maximaPos[:, :2].copy()
        maxima_amp_t = clusters[i].maximaAmp.copy() / intensity_scale_factor  # Normalize amplitudes
        bg_amp_t = bg_amp_ave
        cluster_pixels = clusters[i].pixels[:, :2]

        # Remove superfluous maxima
        if num_maxima_t > 1:

            # Calculate distances between all features
            dist_between_max = create_distance_matrix(maxima_pos_t, maxima_pos_t)

            # Find minimum distance for each maximum
            dist_between_max_sort = np.sort(dist_between_max, axis=1)
            dist_between_max_sort = dist_between_max_sort[:, 1:]  # Remove self-distance
            min_dist_between_max = dist_between_max_sort[:, 0]

            # Find minimum minimum distance
            min_min_dist_between_max = np.min(min_dist_between_max)

            # Remove maxima that are too close
            while min_min_dist_between_max <= (2 * psf_sigma):

                # Find maxima involved
                ij_max = np.where(dist_between_max_sort[:, 0] == min_min_dist_between_max)[0]

                # Determine which has smaller average distance
                ave_dist_ij = np.mean(dist_between_max_sort[ij_max, :], axis=1)
                max_2_remove = ij_max[np.argmin(ave_dist_ij)]
                max_2_keep = np.setdiff1d(np.arange(num_maxima_t), max_2_remove)

                # Remove from cluster
                num_maxima_t -= 1
                maxima_pos_t = maxima_pos_t[max_2_keep, :]
                maxima_amp_t = maxima_amp_t[max_2_keep]

                # Repeat calculation
                if num_maxima_t > 1:
                    dist_between_max = create_distance_matrix(maxima_pos_t, maxima_pos_t)
                    dist_between_max_sort = np.sort(dist_between_max, axis=1)
                    dist_between_max_sort = dist_between_max_sort[:, 1:]
                    min_dist_between_max = dist_between_max_sort[:, 0]
                    min_min_dist_between_max = np.min(min_dist_between_max)
                else:
                    min_min_dist_between_max = 3 * psf_sigma

        # Attempt first fit and iterative mixture-model fitting
        image_c = image.ravel()[clusters[i].pixels[:, 2].astype(int)]

        first_fit = True
        fit = True

        # Variables to store best fit
        num_maxima = num_maxima_t
        solution = None
        residuals = None
        residual_var = None
        jac_mat = None
        var_cov_mat = None

        while fit:

            # Collect initial guesses and bounds
            x0, lb, ub = mmf_init_guess_lower_upper_bounds(
                maxima_pos_t, maxima_amp_t, bg_amp_t, psf_sigma,
                cluster_pixels, first_fit, var_sigma
            )

            # DEBUG: Add debug output after bounds calculation
            print(f"    MMF Debug: Bounds calculated for {len(x0) if isinstance(x0, (list, np.ndarray)) and len(x0) > 0 else 0} parameters")
            if not isinstance(x0, (list, np.ndarray)) or len(x0) == 0:
                print(f"    MMF Debug: NO CANDIDATES after bounds calculation!")
                keep_cluster[i] = False
                break

            print(f"    MMF Debug: Sample initial guess: {x0[:6] if len(x0) >= 6 else x0}")
            if isinstance(lb, (list, np.ndarray)) and len(lb) > 0:
                print(f"    MMF Debug: Sample bounds: lb={lb[:3]}, ub={ub[:3]}")

            # Calculate degrees of freedom
            num_deg_free = len(cluster_pixels) - (3 + int(var_sigma)) * num_maxima_t - 1 #IS THIS RIGHT???
            num_deg_free_t = len(cluster_pixels) - (3 + int(var_sigma)) * num_maxima_t - 1

            print(f"    MMF Debug: Starting optimization with {num_maxima_t} maxima, {len(cluster_pixels)} pixels")
            successful_fits = 0
            failed_fits = 0

            try:
                print(f"    MMF Debug: Fitting candidate 1/{num_maxima_t} - initial guess: {x0}")

                # Perform fitting
                if var_sigma:
                    result = scipy.optimize.least_squares(
                        fit_n_gaussians_2d_var_sigma,
                        x0,
                        bounds=(lb, ub),
                        args=(image_c, cluster_pixels),
                        **options
                    )
                else:
                    result = scipy.optimize.least_squares(
                        fit_n_gaussians_2d,
                        x0,
                        bounds=(lb, ub),
                        args=(image_c, cluster_pixels, psf_sigma),
                        **options
                    )

                solution_t = result.x
                residuals_t = -result.fun  # Negative to match MATLAB convention
                jac_mat_t = result.jac

                # DEBUG: Check fitting results
                if hasattr(result, 'success'):
                    if result.success:
                        successful_fits += 1
                        print(f"    MMF Debug: Fitting SUCCESS - fitted params: {result.x[:6] if len(result.x) >= 6 else result.x}")
                        print(f"    MMF Debug: Cost: {result.cost if hasattr(result, 'cost') else 'N/A'}")
                    else:
                        failed_fits += 1
                        print(f"    MMF Debug: Fitting FAILED - {getattr(result, 'message', 'No message')}")
                        keep_cluster[i] = False
                        break
                else:
                    failed_fits += 1
                    print(f"    MMF Debug: Fitting - No success attribute in result")

                print(f"    MMF Debug: Fitting complete - {successful_fits} successful, {failed_fits} failed")

                if jac_mat_t is not None:
                    residual_var_t = np.sum(residuals_t**2) / num_deg_free_t

                    # Check if first fit
                    if first_fit:
                        first_fit = False
                    else:
                        # F-test for improvement
                        test_stat = residual_var_t / residual_var
                        p_value = stats.f.cdf(test_stat, num_deg_free, num_deg_free_t)

                        if p_value >= alpha_r:
                            fit = False
                            continue

                    # Update variables if fit is accepted
                    num_maxima = num_maxima_t
                    num_deg_free = num_deg_free_t
                    solution = solution_t
                    residuals = residuals_t
                    residual_var = residual_var_t
                    jac_mat = jac_mat_t

                    # Calculate variance-covariance matrix
                    try:
                        var_cov_mat = residual_var * np.linalg.inv(jac_mat.T @ jac_mat)
                        stand_dev_vec = np.sqrt(np.diag(var_cov_mat))

                        if np.all(np.isreal(stand_dev_vec)):

                            # Extract parameters
                            bg_amp = np.array([solution[-1], stand_dev_vec[-1]])
                            # SCALE BACKGROUND BACK TO ORIGINAL UNITS
                            bg_amp = bg_amp * intensity_scale_factor

                            solution_params = solution[:-1]
                            stand_dev_params = stand_dev_vec[:-1]

                            # Reshape into matrices
                            params_per_max = 3 + int(var_sigma)
                            solution_mat = solution_params.reshape(num_maxima, params_per_max)
                            stand_dev_mat = stand_dev_params.reshape(num_maxima, params_per_max)

                            # Extract positions and amplitudes
                            maxima_pos = np.column_stack([
                                solution_mat[:, :2],
                                stand_dev_mat[:, :2]
                            ])

                            # SCALE AMPLITUDES BACK TO ORIGINAL UNITS
                            maxima_amp = np.column_stack([
                                solution_mat[:, 2] * intensity_scale_factor,  # Scale amplitude back
                                stand_dev_mat[:, 2] * intensity_scale_factor  # Scale uncertainty back
                            ])

                            print(f"    MMF Debug: Extracted {len(maxima_pos)} features from fit")
                            print(f"    MMF Debug: Sample amplitude (scaled): {maxima_amp[0, 0]:.2f}")

                            # Prepare for next iteration if doing MMF
                            if do_mmf:
                                num_maxima_t = num_maxima + 1
                                # Use normalized amplitudes for next iteration guess
                                maxima_amp_t = np.append(maxima_amp[:, 0] / intensity_scale_factor,
                                                       np.mean(maxima_amp[:, 0]) / intensity_scale_factor)

                                # Find pixel with maximum residual
                                indx_max_res = np.argmax(residuals)
                                coord = cluster_pixels[indx_max_res, :]
                                maxima_pos_t = np.vstack([maxima_pos[:, :2], coord])
                                bg_amp_t = bg_amp[0] / intensity_scale_factor  # Normalize for next iteration
                            else:
                                fit = False
                        else:
                            print(f"    MMF Debug: Complex standard deviations, rejecting fit")
                            fit = False
                            keep_cluster[i] = False
                    except np.linalg.LinAlgError:
                        print(f"    MMF Debug: LinAlg error in covariance calculation")
                        fit = False
                        keep_cluster[i] = False
                else:
                    print(f"    MMF Debug: No Jacobian returned")
                    fit = False
                    keep_cluster[i] = False

            except Exception as e:
                print(f"    MMF Debug: Fitting ERROR - {e}")
                fit = False
                keep_cluster[i] = False

        # Continue with amplitude and distance tests if cluster is retained
        if keep_cluster[i] and solution is not None:

            print(f"    MMF Debug: Starting validation tests for {num_maxima} features")

            # Amplitude significance test (use normalized residual_var for test)
            test_stat = (maxima_amp[:, 0] / intensity_scale_factor) / np.sqrt((maxima_amp[:, 1] / intensity_scale_factor)**2 + residual_var)
            p_value = 1 - stats.t.cdf(test_stat, num_deg_free)

            # Remove insignificant amplitudes
            while np.max(p_value) > alpha_a and num_maxima > 1:

                # Remove maximum with largest p-value
                indx_bad = np.argmax(p_value)
                indx_keep = np.setdiff1d(np.arange(num_maxima), indx_bad)

                maxima_pos = maxima_pos[indx_keep, :]
                maxima_amp = maxima_amp[indx_keep, :]
                num_maxima -= 1

                print(f"    MMF Debug: Removed 1 feature due to amplitude test, {num_maxima} remaining")

                # Refit with remaining maxima
                x0, lb, ub = mmf_init_guess_lower_upper_bounds(
                    maxima_pos[:, :2], maxima_amp[:, 0] / intensity_scale_factor, bg_amp[0] / intensity_scale_factor,
                    psf_sigma, cluster_pixels, True, var_sigma
                )

                num_deg_free = len(cluster_pixels) - (3 + int(var_sigma)) * num_maxima - 1

                try:
                    if var_sigma:
                        result = scipy.optimize.least_squares(
                            fit_n_gaussians_2d_var_sigma,
                            x0,
                            bounds=(lb, ub),
                            args=(image_c, cluster_pixels),
                            **options
                        )
                    else:
                        result = scipy.optimize.least_squares(
                            fit_n_gaussians_2d,
                            x0,
                            bounds=(lb, ub),
                            args=(image_c, cluster_pixels, psf_sigma),
                            **options
                        )

                    solution = result.x
                    residuals = -result.fun
                    jac_mat = result.jac
                    residual_var = np.sum(residuals**2) / num_deg_free

                    var_cov_mat = residual_var * np.linalg.inv(jac_mat.T @ jac_mat)
                    stand_dev_vec = np.sqrt(np.diag(var_cov_mat))

                    if np.all(np.isreal(stand_dev_vec)):
                        # Extract parameters (same as above)
                        bg_amp = np.array([solution[-1], stand_dev_vec[-1]]) * intensity_scale_factor
                        solution_params = solution[:-1]
                        stand_dev_params = stand_dev_vec[:-1]

                        params_per_max = 3 + int(var_sigma)
                        solution_mat = solution_params.reshape(num_maxima, params_per_max)
                        stand_dev_mat = stand_dev_params.reshape(num_maxima, params_per_max)

                        maxima_pos = np.column_stack([
                            solution_mat[:, :2],
                            stand_dev_mat[:, :2]
                        ])
                        maxima_amp = np.column_stack([
                            solution_mat[:, 2] * intensity_scale_factor,
                            stand_dev_mat[:, 2] * intensity_scale_factor
                        ])

                        # Recalculate amplitude test
                        test_stat = (maxima_amp[:, 0] / intensity_scale_factor) / np.sqrt((maxima_amp[:, 1] / intensity_scale_factor)**2 + residual_var)
                        p_value = 1 - stats.t.cdf(test_stat, num_deg_free)
                    else:
                        keep_cluster[i] = False
                        break

                except (np.linalg.LinAlgError, Exception):
                    keep_cluster[i] = False
                    break

            # Distance significance test
            if num_maxima > 1 and keep_cluster[i]:
                p_value_dist = mmf_dist_pv(maxima_pos, var_cov_mat, num_maxima, num_deg_free, var_sigma)

                while np.max(p_value_dist) > alpha_d:

                    # Find pair with maximum p-value
                    indx1, indx2 = np.unravel_index(np.argmax(p_value_dist), p_value_dist.shape)

                    # Remove maximum with smaller amplitude
                    if maxima_amp[indx1, 0] < maxima_amp[indx2, 0]:
                        indx_bad = indx1
                    else:
                        indx_bad = indx2

                    indx_keep = np.setdiff1d(np.arange(num_maxima), indx_bad)
                    maxima_pos = maxima_pos[indx_keep, :]
                    maxima_amp = maxima_amp[indx_keep, :]
                    num_maxima -= 1

                    print(f"    MMF Debug: Removed 1 feature due to distance test, {num_maxima} remaining")

                    if num_maxima <= 1:
                        break

                    # Similar refit process as in amplitude test
                    # Refit and recalculate distance test
                    try:
                        x0, lb, ub = mmf_init_guess_lower_upper_bounds(
                            maxima_pos[:, :2], maxima_amp[:, 0] / intensity_scale_factor, bg_amp[0] / intensity_scale_factor,
                            psf_sigma, cluster_pixels, True, var_sigma
                        )

                        num_deg_free = len(cluster_pixels) - (3 + int(var_sigma)) * num_maxima - 1

                        if var_sigma:
                            result = scipy.optimize.least_squares(
                                fit_n_gaussians_2d_var_sigma,
                                x0,
                                bounds=(lb, ub),
                                args=(image_c, cluster_pixels),
                                **options
                            )
                        else:
                            result = scipy.optimize.least_squares(
                                fit_n_gaussians_2d,
                                x0,
                                bounds=(lb, ub),
                                args=(image_c, cluster_pixels, psf_sigma),
                                **options
                            )

                        solution = result.x
                        residuals = -result.fun
                        jac_mat = result.jac
                        residual_var = np.sum(residuals**2) / num_deg_free

                        var_cov_mat = residual_var * np.linalg.inv(jac_mat.T @ jac_mat)
                        stand_dev_vec = np.sqrt(np.diag(var_cov_mat))

                        if np.all(np.isreal(stand_dev_vec)):
                            # Extract parameters
                            bg_amp = np.array([solution[-1], stand_dev_vec[-1]]) * intensity_scale_factor
                            solution_params = solution[:-1]
                            stand_dev_params = stand_dev_vec[:-1]

                            params_per_max = 3 + int(var_sigma)
                            solution_mat = solution_params.reshape(num_maxima, params_per_max)
                            stand_dev_mat = stand_dev_params.reshape(num_maxima, params_per_max)

                            maxima_pos = np.column_stack([
                                solution_mat[:, :2],
                                stand_dev_mat[:, :2]
                            ])
                            maxima_amp = np.column_stack([
                                solution_mat[:, 2] * intensity_scale_factor,
                                stand_dev_mat[:, 2] * intensity_scale_factor
                            ])

                            # Recalculate distance test
                            p_value_dist = mmf_dist_pv(maxima_pos, var_cov_mat, num_maxima, num_deg_free, var_sigma)
                        else:
                            keep_cluster[i] = False
                            break

                    except (np.linalg.LinAlgError, Exception):
                        keep_cluster[i] = False
                        break

            # Store final results
            if keep_cluster[i]:
                print(f"    MMF Debug: Final cluster has {num_maxima} validated features")
                print(f"    MMF Debug: Sample final amplitude: {maxima_amp[0, 0]:.2f}")
                clusters_mmf[i].position = maxima_pos
                clusters_mmf[i].amplitude = maxima_amp
                clusters_mmf[i].bgAmp = bg_amp
                clusters_mmf[i].numDegFree = num_deg_free
                clusters_mmf[i].residuals = residuals

                if var_sigma and solution is not None:
                    params_per_max = 4
                    solution_mat = solution[:-1].reshape(num_maxima, params_per_max)
                    stand_dev_mat = stand_dev_vec[:-1].reshape(num_maxima, params_per_max)
                    clusters_mmf[i].sigma = np.column_stack([
                        solution_mat[:, 3],
                        stand_dev_mat[:, 3]
                    ])
        else:
            print(f"    MMF Debug: Cluster {i} rejected - no valid solution")

    # Final processing - retain only significant clusters
    indx = np.where(keep_cluster)[0]

    print(f"    MMF Debug: {len(indx)} clusters passed all tests")

    if len(indx) > 0:
        clusters_mmf = [clusters_mmf[i] for i in indx]

        # Store information in detected_features structure
        if clusters_mmf:
            all_positions = np.vstack([c.position for c in clusters_mmf if c.position.size > 0])
            all_amplitudes = np.vstack([c.amplitude for c in clusters_mmf if c.amplitude.size > 0])
            sigma_arrays = [np.atleast_2d(c.sigma) for c in clusters_mmf if hasattr(c, 'sigma') and hasattr(c.sigma, 'size') and c.sigma.size > 0]
            all_sigmas = np.vstack(sigma_arrays) if sigma_arrays else np.array([[1.0, 1.0]])

            if len(all_positions) > 0:
                detected_features.xCoord = all_positions[:, [0, 2]]  # x and dx
                detected_features.yCoord = all_positions[:, [1, 3]]  # y and dy
                detected_features.amp = all_amplitudes

                if len(all_sigmas) > 0:
                    detected_features.sigma = all_sigmas

    final_feature_count = len(detected_features.xCoord) if detected_features.xCoord.size > 0 else 0
    print(f"    MMF Debug: About to return {final_feature_count} features")
    if final_feature_count > 0:
        print(f"    MMF Debug: Sample final feature: x={detected_features.xCoord[0]}, y={detected_features.yCoord[0]}")
        print(f"    MMF Debug: Sample final amplitude: {detected_features.amp[0, 0]:.2f}")
    else:
        print(f"    MMF Debug: No features survived final processing")

    # Visualization
    if visual:
        image_norm = image / np.percentile(image, 99.9)
        image_norm = np.clip(image_norm, 0, 1)
        image_n3 = np.stack([image_norm, image_norm, image_norm], axis=2)

        # Add visualization markers (simplified)
        plt.figure(figsize=(10, 8))
        plt.imshow(image_n3)
        plt.title('Detected Features')
        plt.show()

    return detected_features


def centroid_sub_res_features_2d(
    image: np.ndarray,
    cands: List,
    psf_sigma: float,
    visual: bool = False,
    bit_depth: int = 16,
    save_results: bool = False
) -> DetectedFeatures:
    """
    Determine positions and intensity amplitudes of sub-resolution features using centroid calculation.

    This is the Python equivalent of centroidSubResFeatures2D.m, providing a faithful
    implementation of the MATLAB version.

    Args:
        image: Image being analyzed
        cands: Candidate features from initial detection
        psf_sigma: Standard deviation of PSF (pixels)
        visual: Whether to show results
        bit_depth: Camera bit depth (default: 16)
        save_results: Whether to save results (not implemented)

    Returns:
        DetectedFeatures structure with centroid calculations
    """

    # Initialize outputs
    detected_features = DetectedFeatures()
    image_n3 = None
    err_flag = 0

    # Input validation
    if image is None:
        print('--centroidSubResFeatures2D: Image input is required!')
        return detected_features

    if cands is None:
        print('--centroidSubResFeatures2D: Candidates input is required!')
        return detected_features

    if psf_sigma <= 0:
        print('--centroidSubResFeatures2D: PSF sigma must be positive!')
        return detected_features

    # Check bit depth
    if bit_depth <= 0 or bit_depth != int(bit_depth):
        print('--centroidSubResFeatures2D: Variable "bitDepth" should be a positive integer!')
        bit_depth = 16  # Use default

    # Get number of pixels in each direction
    num_pixels_y, num_pixels_x = image.shape

    # Store original scale factor
    intensity_scale_factor = 2**bit_depth - 1

    # Divide image by bit depth, to normalize it between 0 and 1
    image = image.astype(float) / intensity_scale_factor

    # Get local maxima information from cands
    valid_cands = [c for c in cands if hasattr(c, 'status') and c.status == 1]

    if not valid_cands:
        print('--centroidSubResFeatures2D: No valid candidates found!')
        return detected_features

    # Extract information from valid candidates
    loc_max_pos = np.array([c.Lmax for c in valid_cands])
    loc_max_amp = np.array([c.amp for c in valid_cands])
    bg_amp = np.array([c.IBkg for c in valid_cands])

    num_loc_max = len(valid_cands)

    # Get half the PSF range in pixels
    psf_half_range = int(round(2 * psf_sigma))

    # Initialize arrays for results
    x_coords = np.zeros((num_loc_max, 2))  # [coordinate, uncertainty]
    y_coords = np.zeros((num_loc_max, 2))
    amplitudes = np.zeros((num_loc_max, 2))

    # Go over all local maxima
    for i in range(num_loc_max):

        # Get position (convert from 1-indexed MATLAB to 0-indexed Python)
        mid_pixel = loc_max_pos[i] - 1  # Convert to 0-indexed

        # Get part of image relevant for this local maximum
        min_coord_y = max(int(mid_pixel[1] - psf_half_range), 0)
        max_coord_y = min(int(mid_pixel[1] + psf_half_range), num_pixels_y - 1)
        min_coord_x = max(int(mid_pixel[0] - psf_half_range), 0)
        max_coord_x = min(int(mid_pixel[0] + psf_half_range), num_pixels_x - 1)

        # Extract image region
        image_loc_max = image[min_coord_y:max_coord_y+1, min_coord_x:max_coord_x+1]

        # Calculate centroid in small image
        try:
            ce = centroid_2d(image_loc_max)

            # Check if centroid calculation was successful
            if np.any(np.isnan(ce)):
                # If centroid failed, use original position
                ce = np.array([psf_half_range + 1, psf_half_range + 1])

        except Exception:
            # If centroid calculation fails, use original position
            ce = np.array([psf_half_range + 1, psf_half_range + 1])

        # Shift to coordinates in overall image
        # Note: centroid_2d returns [x, y] in image coordinates (1-indexed)
        # We need to convert back to 0-indexed and add the offset
        ce_global = (ce - 1) + np.array([min_coord_x, min_coord_y])

        # Store information in detected_features structure
        # Convert coordinates to image coordinate system (1-indexed for consistency with MATLAB)
        x_coords[i, :] = [ce_global[0] + 1, 0.2]  # +1 to convert back to 1-indexed
        y_coords[i, :] = [ce_global[1] + 1, 0.2]  # +1 to convert back to 1-indexed
        # Keep amplitudes in original intensity units (no scaling back needed as we're using original candidate amplitudes)
        amplitudes[i, :] = [loc_max_amp[i] - bg_amp[i], 0.0]  # Amplitude above background

    # Store results
    detected_features.xCoord = x_coords
    detected_features.yCoord = y_coords
    detected_features.amp = amplitudes
    detected_features.sigma = np.array([])  # Empty array for consistency with MMF version

    # Visualization
    if visual:
        # Make 3 layers out of original image (normalized)
        image_norm = image / np.max(image)
        image_n3 = np.stack([image_norm, image_norm, image_norm], axis=2)

        # Place zeros in pixels of maxima from cands (convert to 0-indexed)
        pos_l_indices = ((loc_max_pos[:, 1] - 1).astype(int), (loc_max_pos[:, 0] - 1).astype(int))
        for j in range(3):
            image_n3[pos_l_indices[0], pos_l_indices[1], j] = 0

        # Place zeros in pixels of maxima from centroid calculation (convert to 0-indexed)
        pos_c_indices = ((y_coords[:, 0] - 1).astype(int), (x_coords[:, 0] - 1).astype(int))
        for j in range(3):
            # Make sure indices are within bounds
            valid_indices = ((pos_c_indices[0] >= 0) & (pos_c_indices[0] < num_pixels_y) &
                           (pos_c_indices[1] >= 0) & (pos_c_indices[1] < num_pixels_x))
            if np.any(valid_indices):
                image_n3[pos_c_indices[0][valid_indices], pos_c_indices[1][valid_indices], j] = 0

        # Label maxima from cands in blue
        image_n3[pos_l_indices[0], pos_l_indices[1], 2] = 1

        # Label maxima from centroid calculation in red
        # A maximum from centroid calculation that falls in the same pixel
        # as that from cands will appear in magenta
        valid_indices = ((pos_c_indices[0] >= 0) & (pos_c_indices[0] < num_pixels_y) &
                        (pos_c_indices[1] >= 0) & (pos_c_indices[1] < num_pixels_x))
        if np.any(valid_indices):
            image_n3[pos_c_indices[0][valid_indices], pos_c_indices[1][valid_indices], 0] = 1

        # Plot image
        plt.figure(figsize=(10, 8))
        plt.imshow(image_n3, origin='upper')
        plt.title('Centroid Detection Results')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')

        # Add legend
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='blue', label='Original candidates')
        red_patch = mpatches.Patch(color='red', label='Centroid results')
        magenta_patch = mpatches.Patch(color='magenta', label='Overlapping')
        plt.legend(handles=[blue_patch, red_patch, magenta_patch])

        plt.show()

    return detected_features
