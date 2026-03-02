"""
PSF Fitting Module with Numba Optimization
===========================================

Implements PSF fitting methods:
- 2D Integrated Gaussian (LSQ - Least Squares)
- 2D Integrated Gaussian (WLSQ - Weighted Least Squares)
- 2D Integrated Gaussian (MLE - Maximum Likelihood Estimation)
- Phasor/Radial Symmetry
- Centroid

Uses pixel-integrated Gaussian PSF model (erf-based) and true
Levenberg-Marquardt optimization matching ImageJ thunderSTORM.

This version uses Numba JIT compilation for 10-50x speedup!

Author: George K (with Claude)
Date: 2025-12-08
Updated: 2025-12-11 (added WLSQ and MLE fitters)
Updated: 2026-03-02 (integrated Gaussian PSF + true LM optimizer)
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

# Try to import Numba - fall back to regular Python if not available
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("✓ Numba available - using optimized fitting")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠ Numba not available - using standard fitting (slower)")
    print("  Install with: pip install numba")

    # Create dummy decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# ============================================================================
# CORE NUMBA-OPTIMIZED FUNCTIONS
# ============================================================================

@njit(fastmath=True)
def gaussian_2d(x, y, x0, y0, amplitude, sigma_x, sigma_y, background, theta=0.0):
    """
    Evaluate 2D Gaussian at positions (x, y)

    Numba-optimized for speed!
    """
    if theta == 0.0:
        # Faster path for axis-aligned Gaussians
        dx = (x - x0) / sigma_x
        dy = (y - y0) / sigma_y
        return amplitude * np.exp(-0.5 * (dx*dx + dy*dy)) + background
    else:
        # Rotated Gaussian
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx = x - x0
        dy = y - y0

        xr = cos_theta * dx + sin_theta * dy
        yr = -sin_theta * dx + cos_theta * dy

        return amplitude * np.exp(-0.5 * ((xr/sigma_x)**2 + (yr/sigma_y)**2)) + background


@njit(fastmath=True)
def extract_roi(image, row, col, radius):
    """Extract region of interest around a position"""
    height, width = image.shape

    r0 = max(0, row - radius)
    r1 = min(height, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(width, col + radius + 1)

    roi = image[r0:r1, c0:c1].copy()

    return roi, r0, r1, c0, c1


# ============================================================================
# INTEGRATED GAUSSIAN PSF MODEL (erf-based)
# ============================================================================
# Matches ImageJ thunderSTORM: pixel value = integral of Gaussian over pixel area
# Parameters: [x0, y0, sigma, intensity, offset]
#   x0, y0  = sub-pixel center position
#   sigma   = PSF standard deviation (symmetric)
#   intensity = total molecule intensity (photons under the PSF)
#   offset  = local background per pixel

SQRT2 = math.sqrt(2.0)
TWO_SQRT2PI = 2.0 * math.sqrt(2.0 * math.pi)


@njit(fastmath=True)
def integrated_gaussian_value(jj, ii, x0, y0, sigma, intensity, offset):
    """Compute integrated Gaussian PSF value for pixel (jj, ii).

    Parameters
    ----------
    jj : float
        Pixel column coordinate (x)
    ii : float
        Pixel row coordinate (y)
    x0, y0 : float
        Sub-pixel PSF center
    sigma : float
        PSF standard deviation
    intensity : float
        Total molecule intensity
    offset : float
        Background per pixel

    Returns
    -------
    value : float
        Expected pixel value
    """
    s2 = SQRT2 * sigma
    ex = 0.5 * (math.erf((jj - x0 + 0.5) / s2) - math.erf((jj - x0 - 0.5) / s2))
    ey = 0.5 * (math.erf((ii - y0 + 0.5) / s2) - math.erf((ii - y0 - 0.5) / s2))
    return offset + intensity * ex * ey


@njit(fastmath=True)
def integrated_gaussian_jacobian(jj, ii, x0, y0, sigma, intensity, offset):
    """Compute Jacobian of integrated Gaussian for pixel (jj, ii).

    Returns partial derivatives with respect to [x0, y0, sigma, intensity, offset].

    Parameters
    ----------
    jj, ii : float
        Pixel coordinates
    x0, y0, sigma, intensity, offset : float
        Model parameters

    Returns
    -------
    jac : array of 5 floats
        [dmodel/dx0, dmodel/dy0, dmodel/dsigma, dmodel/dintensity, dmodel/doffset]
    """
    jac = np.zeros(5)
    s2 = SQRT2 * sigma

    # erf terms for x and y
    erf_xp = math.erf((jj - x0 + 0.5) / s2)
    erf_xm = math.erf((jj - x0 - 0.5) / s2)
    erf_yp = math.erf((ii - y0 + 0.5) / s2)
    erf_ym = math.erf((ii - y0 - 0.5) / s2)

    ex = 0.5 * (erf_xp - erf_xm)
    ey = 0.5 * (erf_yp - erf_ym)

    # Gaussian terms for derivatives of erf
    # d/du erf(u) = 2/sqrt(pi) * exp(-u^2)
    # We need d(erf((jj - x0 + 0.5)/(sqrt(2)*sigma)))/dx0 = -1/(sqrt(2)*sigma) * 2/sqrt(pi) * exp(...)
    inv_s2 = 1.0 / s2
    two_over_sqrtpi = 2.0 / math.sqrt(math.pi)

    ux_p = (jj - x0 + 0.5) * inv_s2
    ux_m = (jj - x0 - 0.5) * inv_s2
    uy_p = (ii - y0 + 0.5) * inv_s2
    uy_m = (ii - y0 - 0.5) * inv_s2

    gx_p = two_over_sqrtpi * math.exp(-ux_p * ux_p)
    gx_m = two_over_sqrtpi * math.exp(-ux_m * ux_m)
    gy_p = two_over_sqrtpi * math.exp(-uy_p * uy_p)
    gy_m = two_over_sqrtpi * math.exp(-uy_m * uy_m)

    # d(ex)/dx0 = 0.5 * (-inv_s2 * gx_p - (-inv_s2 * gx_m))
    #           = 0.5 * inv_s2 * (gx_m - gx_p)
    dex_dx0 = 0.5 * inv_s2 * (gx_m - gx_p)
    dey_dy0 = 0.5 * inv_s2 * (gy_m - gy_p)

    # d(ex)/dsigma = 0.5 * (d/dsigma erf_xp - d/dsigma erf_xm)
    # d/dsigma erf(u) where u = (jj-x0+0.5)/(sqrt(2)*sigma)
    # du/dsigma = -(jj-x0+0.5)/(sqrt(2)*sigma^2) = -u/sigma
    dex_dsigma = 0.5 * (-ux_p * gx_p + ux_m * gx_m) / sigma
    dey_dsigma = 0.5 * (-uy_p * gy_p + uy_m * gy_m) / sigma

    # jac[0] = dmodel/dx0 = intensity * dex_dx0 * ey
    jac[0] = intensity * dex_dx0 * ey

    # jac[1] = dmodel/dy0 = intensity * ex * dey_dy0
    jac[1] = intensity * ex * dey_dy0

    # jac[2] = dmodel/dsigma = intensity * (dex_dsigma * ey + ex * dey_dsigma)
    jac[2] = intensity * (dex_dsigma * ey + ex * dey_dsigma)

    # jac[3] = dmodel/dintensity = ex * ey
    jac[3] = ex * ey

    # jac[4] = dmodel/doffset = 1.0
    jac[4] = 1.0

    return jac


# ============================================================================
# 5x5 LINEAR SOLVER (Gaussian elimination with partial pivoting)
# ============================================================================

@njit(fastmath=True)
def solve_5x5(A, b):
    """Solve 5x5 linear system A*x = b via Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : ndarray (5, 5)
        Coefficient matrix (modified in-place)
    b : ndarray (5,)
        Right-hand side (modified in-place)

    Returns
    -------
    x : ndarray (5,)
        Solution vector. All zeros if system is singular.
    """
    n = 5
    x = np.zeros(n)

    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_val = abs(A[k, k])
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        if max_val < 1e-30:
            # Singular matrix
            return x

        # Swap rows
        if max_row != k:
            for j in range(k, n):
                A[k, j], A[max_row, j] = A[max_row, j], A[k, j]
            b[k], b[max_row] = b[max_row], b[k]

        # Eliminate below
        pivot = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            A[i, k] = 0.0
            b[i] -= factor * b[k]

    # Back substitution
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-30:
            x[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]

    return x


@njit(parallel=True, fastmath=True)
def fit_gaussian_lsq_batch_numba(image, positions, fit_radius=3,
                                  initial_sigma=1.3, max_iterations=50):
    """
    Fit integrated Gaussian PSF to multiple positions using Least Squares
    with true Levenberg-Marquardt optimization.

    Uses erf-based pixel-integrated Gaussian model matching ImageJ thunderSTORM.
    5 parameters: [x0, y0, sigma, intensity, offset]

    Parameters
    ----------
    image : ndarray (2D)
        Image to fit
    positions : ndarray (N, 2)
        Array of (row, col) positions
    fit_radius : int
        Fitting radius in pixels
    initial_sigma : float
        Initial guess for sigma
    max_iterations : int
        Maximum Levenberg-Marquardt iterations

    Returns
    -------
    results : ndarray (N, 8)
        [x, y, intensity, sigma, sigma, background, chi_sq, success]
    """
    n_fits = len(positions)
    results = np.zeros((n_fits, 8))

    for i in prange(n_fits):
        row = int(positions[i, 0])
        col = int(positions[i, 1])

        # Extract ROI
        r0 = max(0, row - fit_radius)
        r1 = min(image.shape[0], row + fit_radius + 1)
        c0 = max(0, col - fit_radius)
        c1 = min(image.shape[1], col + fit_radius + 1)

        if r1 - r0 < 3 or c1 - c0 < 3:
            results[i, 7] = 0.0
            continue

        roi = image[r0:r1, c0:c1]
        ny = roi.shape[0]
        nx = roi.shape[1]
        n_pixels = ny * nx

        # Initial parameter estimates: [x0, y0, sigma, intensity, offset]
        offset = np.min(roi)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma = initial_sigma
        intensity = peak * 2.0 * math.pi * sigma * sigma  # Approx total intensity

        # Levenberg-Marquardt
        lambda_lm = 1.0

        # Compute initial chi-squared
        chi_sq = 0.0
        for ii in range(ny):
            for jj in range(nx):
                model_val = integrated_gaussian_value(float(jj), float(ii), x0, y0, sigma, intensity, offset)
                r = roi[ii, jj] - model_val
                chi_sq += r * r

        for iteration in range(max_iterations):
            # Build Hessian approximation H = J^T * J and gradient g = J^T * r
            H = np.zeros((5, 5))
            g = np.zeros(5)

            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    model_val = integrated_gaussian_value(fj, fi, x0, y0, sigma, intensity, offset)
                    residual = roi[ii, jj] - model_val
                    J = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    for p in range(5):
                        g[p] += J[p] * residual
                        for q in range(5):
                            H[p][q] += J[p] * J[q]

            # Augment diagonal: H_aug = H + lambda * diag(H)
            H_aug = H.copy()
            for p in range(5):
                H_aug[p][p] *= (1.0 + lambda_lm)

            # Solve for parameter update
            g_copy = g.copy()
            dp = solve_5x5(H_aug, g_copy)

            # Candidate parameters
            x0_new = x0 + dp[0]
            y0_new = y0 + dp[1]
            sigma_new = sigma + dp[2]
            intensity_new = intensity + dp[3]
            offset_new = offset + dp[4]

            # Constrain
            sigma_new = max(0.5, min(sigma_new, 10.0))
            intensity_new = max(1.0, intensity_new)
            offset_new = max(0.0, offset_new)

            # Compute new chi-squared
            chi_sq_new = 0.0
            for ii in range(ny):
                for jj in range(nx):
                    model_val = integrated_gaussian_value(float(jj), float(ii), x0_new, y0_new, sigma_new, intensity_new, offset_new)
                    r = roi[ii, jj] - model_val
                    chi_sq_new += r * r

            if chi_sq_new < chi_sq:
                # Accept step
                rel_change = abs(chi_sq - chi_sq_new) / max(chi_sq, 1e-30)
                x0 = x0_new
                y0 = y0_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                chi_sq = chi_sq_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-6:
                    break
            else:
                # Reject step
                lambda_lm *= 10.0
                if lambda_lm > 1e10:
                    break

        # Store results: [x, y, intensity, sigma, sigma, background, chi_sq, success]
        results[i, 0] = x0 + c0   # x in image coordinates
        results[i, 1] = y0 + r0   # y in image coordinates
        results[i, 2] = intensity  # total intensity
        results[i, 3] = sigma      # sigma
        results[i, 4] = sigma      # sigma (duplicate for backward compat)
        results[i, 5] = offset     # background
        results[i, 6] = chi_sq / n_pixels
        results[i, 7] = 1.0        # success

    return results


@njit(parallel=True, fastmath=True)
def fit_gaussian_wlsq_batch_numba(image, positions, fit_radius=3,
                                   initial_sigma=1.3, max_iterations=50):
    """
    Fit integrated Gaussian PSF using Weighted Least Squares
    with true Levenberg-Marquardt optimization.

    Weights: w = 1/max(data, 1) (Poisson variance model).

    Parameters
    ----------
    image : ndarray (2D)
        Image to fit
    positions : ndarray (N, 2)
        Array of (row, col) positions
    fit_radius : int
        Fitting radius in pixels
    initial_sigma : float
        Initial guess for sigma
    max_iterations : int
        Maximum optimization iterations

    Returns
    -------
    results : ndarray (N, 8)
        [x, y, intensity, sigma, sigma, background, chi_sq, success]
    """
    n_fits = len(positions)
    results = np.zeros((n_fits, 8))

    for i in prange(n_fits):
        row = int(positions[i, 0])
        col = int(positions[i, 1])

        r0 = max(0, row - fit_radius)
        r1 = min(image.shape[0], row + fit_radius + 1)
        c0 = max(0, col - fit_radius)
        c1 = min(image.shape[1], col + fit_radius + 1)

        if r1 - r0 < 3 or c1 - c0 < 3:
            results[i, 7] = 0.0
            continue

        roi = image[r0:r1, c0:c1]
        ny = roi.shape[0]
        nx = roi.shape[1]
        n_pixels = ny * nx

        # Initial parameters
        offset = np.min(roi)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma = initial_sigma
        intensity = peak * 2.0 * math.pi * sigma * sigma

        lambda_lm = 1.0

        # Compute initial weighted chi-squared
        chi_sq = 0.0
        for ii in range(ny):
            for jj in range(nx):
                model_val = integrated_gaussian_value(float(jj), float(ii), x0, y0, sigma, intensity, offset)
                data = roi[ii, jj]
                r = data - model_val
                w = 1.0 / max(data, 1.0)
                chi_sq += w * r * r

        for iteration in range(max_iterations):
            H = np.zeros((5, 5))
            g = np.zeros(5)

            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    model_val = integrated_gaussian_value(fj, fi, x0, y0, sigma, intensity, offset)
                    data = roi[ii, jj]
                    residual = data - model_val
                    w = 1.0 / max(data, 1.0)
                    J = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    for p in range(5):
                        g[p] += w * J[p] * residual
                        for q in range(5):
                            H[p][q] += w * J[p] * J[q]

            H_aug = H.copy()
            for p in range(5):
                H_aug[p][p] *= (1.0 + lambda_lm)

            g_copy = g.copy()
            dp = solve_5x5(H_aug, g_copy)

            x0_new = x0 + dp[0]
            y0_new = y0 + dp[1]
            sigma_new = max(0.5, min(sigma + dp[2], 10.0))
            intensity_new = max(1.0, intensity + dp[3])
            offset_new = max(0.0, offset + dp[4])

            chi_sq_new = 0.0
            for ii in range(ny):
                for jj in range(nx):
                    model_val = integrated_gaussian_value(float(jj), float(ii), x0_new, y0_new, sigma_new, intensity_new, offset_new)
                    data = roi[ii, jj]
                    r = data - model_val
                    w = 1.0 / max(data, 1.0)
                    chi_sq_new += w * r * r

            if chi_sq_new < chi_sq:
                rel_change = abs(chi_sq - chi_sq_new) / max(chi_sq, 1e-30)
                x0 = x0_new
                y0 = y0_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                chi_sq = chi_sq_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-6:
                    break
            else:
                lambda_lm *= 10.0
                if lambda_lm > 1e10:
                    break

        results[i, 0] = x0 + c0
        results[i, 1] = y0 + r0
        results[i, 2] = intensity
        results[i, 3] = sigma
        results[i, 4] = sigma
        results[i, 5] = offset
        results[i, 6] = chi_sq / n_pixels
        results[i, 7] = 1.0

    return results


@njit(parallel=True, fastmath=True)
def fit_gaussian_mle_batch_numba(image, positions, fit_radius=3,
                                  initial_sigma=1.3, max_iterations=50):
    """
    Fit integrated Gaussian PSF using Maximum Likelihood Estimation
    with true Levenberg-Marquardt optimization.

    Poisson noise model: -log L = sum(model - data*log(model))

    Parameters
    ----------
    image : ndarray (2D)
        Image to fit
    positions : ndarray (N, 2)
        Array of (row, col) positions
    fit_radius : int
        Fitting radius in pixels
    initial_sigma : float
        Initial guess for sigma
    max_iterations : int
        Maximum optimization iterations

    Returns
    -------
    results : ndarray (N, 8)
        [x, y, intensity, sigma, sigma, background, neg_log_likelihood, success]
    """
    n_fits = len(positions)
    results = np.zeros((n_fits, 8))

    for i in prange(n_fits):
        row = int(positions[i, 0])
        col = int(positions[i, 1])

        r0 = max(0, row - fit_radius)
        r1 = min(image.shape[0], row + fit_radius + 1)
        c0 = max(0, col - fit_radius)
        c1 = min(image.shape[1], col + fit_radius + 1)

        if r1 - r0 < 3 or c1 - c0 < 3:
            results[i, 7] = 0.0
            continue

        roi = image[r0:r1, c0:c1]
        ny = roi.shape[0]
        nx = roi.shape[1]
        n_pixels = ny * nx

        # Initial parameters
        offset = max(np.min(roi), 0.1)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma = initial_sigma
        intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

        lambda_lm = 1.0

        # Initial negative log-likelihood
        neg_ll = 0.0
        for ii in range(ny):
            for jj in range(nx):
                model_val = integrated_gaussian_value(float(jj), float(ii), x0, y0, sigma, intensity, offset)
                model_val = max(model_val, 1e-10)
                data = roi[ii, jj]
                neg_ll += model_val - data * math.log(model_val)

        for iteration in range(max_iterations):
            # For MLE with Poisson noise, the Hessian approximation uses
            # H = J^T * diag(1/model) * J and g = J^T * (1 - data/model)
            # But we use the Fisher information approximation:
            # H[p][q] = sum(J_p * J_q / model), g[p] = sum((1 - data/model) * J_p)
            H = np.zeros((5, 5))
            g = np.zeros(5)

            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    model_val = integrated_gaussian_value(fj, fi, x0, y0, sigma, intensity, offset)
                    model_val = max(model_val, 1e-10)
                    data = roi[ii, jj]
                    J = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    # Weight = 1/model (Fisher information)
                    w = 1.0 / model_val
                    factor = data / model_val - 1.0  # negative gradient direction

                    for p in range(5):
                        g[p] += factor * J[p]
                        for q in range(5):
                            H[p][q] += w * J[p] * J[q]

            H_aug = H.copy()
            for p in range(5):
                H_aug[p][p] *= (1.0 + lambda_lm)

            g_copy = g.copy()
            dp = solve_5x5(H_aug, g_copy)

            x0_new = x0 + dp[0]
            y0_new = y0 + dp[1]
            sigma_new = max(0.5, min(sigma + dp[2], 10.0))
            intensity_new = max(1.0, intensity + dp[3])
            offset_new = max(0.1, offset + dp[4])

            neg_ll_new = 0.0
            for ii in range(ny):
                for jj in range(nx):
                    model_val = integrated_gaussian_value(float(jj), float(ii), x0_new, y0_new, sigma_new, intensity_new, offset_new)
                    model_val = max(model_val, 1e-10)
                    data = roi[ii, jj]
                    neg_ll_new += model_val - data * math.log(model_val)

            if neg_ll_new < neg_ll:
                rel_change = abs(neg_ll - neg_ll_new) / max(abs(neg_ll), 1e-30)
                x0 = x0_new
                y0 = y0_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                neg_ll = neg_ll_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-6:
                    break
            else:
                lambda_lm *= 10.0
                if lambda_lm > 1e10:
                    break

        results[i, 0] = x0 + c0
        results[i, 1] = y0 + r0
        results[i, 2] = intensity
        results[i, 3] = sigma
        results[i, 4] = sigma
        results[i, 5] = offset
        results[i, 6] = neg_ll / n_pixels
        results[i, 7] = 1.0

    return results


def compute_localization_precision(intensity, background, sigma, pixel_size,
                                    fitting_method='lsq', is_emccd=False):
    """
    Compute localization precision using Thompson-Mortensen-Quan formula.

    Base variance (Thompson et al., 2002):
        σ² = (s² + a²/12) / N + (8π·(s² + a²/12)²·b²) / (a²·N²)

    Corrections:
    - Mortensen et al. (2010): multiply by 16/9 for least-squares fitting
    - EMCCD excess noise: multiply by F=2 (Ulbrich & Bhatt, 2015)

    Parameters
    ----------
    intensity : float
        Total molecule intensity (photons). For integrated Gaussian model
        this is the total intensity parameter, NOT peak amplitude.
    background : float
        Background per pixel
    sigma : float
        PSF standard deviation (same units as pixel_size)
    pixel_size : float
        Pixel size (set to 1.0 for pixel-unit computation)
    fitting_method : str
        Fitting method: 'lsq', 'wlsq', or 'mle'
    is_emccd : bool
        Whether camera is EMCCD (applies F=2 excess noise factor)

    Returns
    -------
    precision : float
        Localization precision (sqrt of variance)
    """
    if intensity <= 0:
        return np.nan

    s = sigma
    a = pixel_size
    N = intensity
    b = background

    # sa2 = s^2 + a^2/12  (effective variance including pixelation)
    sa2 = s * s + a * a / 12.0

    # Base variance (Thompson et al., 2002)
    # Note: using sa2 in numerator of term2 as per Mortensen formulation
    term1 = sa2 / N
    term2 = (8.0 * np.pi * sa2 * sa2 * b * b) / (a * a * N * N)
    variance = term1 + term2

    # Mortensen correction for least-squares fitting (Mortensen 2010, Eq. 6)
    if fitting_method in ('lsq', 'wlsq'):
        variance *= 16.0 / 9.0

    # EMCCD excess noise factor F=2
    if is_emccd:
        variance *= 2.0

    return np.sqrt(variance)


# ============================================================================
# FITTING RESULT CLASS
# ============================================================================

@dataclass
class FitResult:
    """Container for fitting results.

    Note: With the integrated Gaussian model, 'intensity' is the total
    molecule intensity (photons under the PSF), NOT peak amplitude.
    """
    x: np.ndarray
    y: np.ndarray
    intensity: np.ndarray
    background: np.ndarray
    sigma_x: np.ndarray
    sigma_y: np.ndarray
    chi_squared: np.ndarray
    uncertainty: np.ndarray
    success: np.ndarray

    def __len__(self):
        return len(self.x)

    def to_array(self):
        """Convert to dictionary of arrays"""
        return {
            'x': self.x,
            'y': self.y,
            'intensity': self.intensity,
            'background': self.background,
            'sigma_x': self.sigma_x,
            'sigma_y': self.sigma_y,
            'chi_squared': self.chi_squared,
            'uncertainty': self.uncertainty
        }


# ============================================================================
# FITTER CLASSES
# ============================================================================

class BaseFitter:
    """Base class for PSF fitters"""

    def __init__(self, name="BaseFitter"):
        self.name = name
        self.pixel_size = 100.0  # nm
        self.photons_per_adu = 1.0
        self.baseline = 0.0
        self.em_gain = 1.0
        self.is_emccd = False

    def set_camera_params(self, pixel_size=100.0, photons_per_adu=1.0,
                         baseline=0.0, em_gain=1.0, is_emccd=False):
        """Set camera calibration parameters"""
        self.pixel_size = pixel_size
        self.photons_per_adu = photons_per_adu
        self.baseline = baseline
        self.em_gain = em_gain
        self.is_emccd = is_emccd

    def fit(self, image, positions, fit_radius=3):
        """Fit PSF to detected positions

        Parameters
        ----------
        image : ndarray
            2D image
        positions : ndarray
            Array of (row, col) positions
        fit_radius : int
            Fitting radius in pixels

        Returns
        -------
        result : FitResult
            Fitting results
        """
        raise NotImplementedError


class GaussianLSQFitter(BaseFitter):
    """
    Integrated Gaussian PSF fitting using Least Squares
    with true Levenberg-Marquardt optimization.

    NUMBA-OPTIMIZED for 10-50x speedup!

    Parameters
    ----------
    initial_sigma : float
        Initial guess for sigma
    elliptical : bool
        Fit elliptical Gaussian (sigma_x != sigma_y) — not yet supported
    integrated : bool
        Ignored (always uses integrated Gaussian model)
    """

    def __init__(self, initial_sigma=1.3, elliptical=False, integrated=False):
        super().__init__("GaussianLSQ")
        self.initial_sigma = initial_sigma
        self.elliptical = elliptical

        if NUMBA_AVAILABLE:
            self.name += " (Numba)"

    def fit(self, image, positions, fit_radius=3):
        """Fit integrated Gaussian to all positions"""

        if len(positions) == 0:
            return FitResult(
                x=np.array([]),
                y=np.array([]),
                intensity=np.array([]),
                background=np.array([]),
                sigma_x=np.array([]),
                sigma_y=np.array([]),
                chi_squared=np.array([]),
                uncertainty=np.array([]),
                success=np.array([])
            )

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        # Call Numba-optimized integrated Gaussian LSQ fitter
        results = fit_gaussian_lsq_batch_numba(
            image, positions, fit_radius, self.initial_sigma
        )

        # Results: [x, y, intensity, sigma, sigma, background, chi_sq, success]
        x = results[:, 0]
        y = results[:, 1]
        intensity = results[:, 2]
        sigma_x = results[:, 3]
        sigma_y = results[:, 4]
        background = results[:, 5]
        chi_squared = results[:, 6]
        success = results[:, 7]

        mask = success > 0

        # Compute uncertainties with Mortensen correction
        uncertainty = np.zeros(len(x))
        for i in range(len(x)):
            if mask[i]:
                uncertainty[i] = compute_localization_precision(
                    intensity[i], background[i], sigma_x[i],
                    1.0, fitting_method='lsq', is_emccd=self.is_emccd
                )

        return FitResult(
            x=x[mask],
            y=y[mask],
            intensity=intensity[mask],
            background=background[mask],
            sigma_x=sigma_x[mask],
            sigma_y=sigma_y[mask],
            chi_squared=chi_squared[mask],
            uncertainty=uncertainty[mask],
            success=success[mask]
        )


class GaussianWLSQFitter(BaseFitter):
    """
    Integrated Gaussian PSF fitting using Weighted Least Squares
    with true Levenberg-Marquardt optimization.

    NUMBA-OPTIMIZED for 10-50x speedup!

    Parameters
    ----------
    initial_sigma : float
        Initial guess for sigma
    elliptical : bool
        Fit elliptical Gaussian — not yet supported
    integrated : bool
        Ignored (always uses integrated Gaussian model)
    """

    def __init__(self, initial_sigma=1.3, elliptical=False, integrated=False):
        super().__init__("GaussianWLSQ")
        self.initial_sigma = initial_sigma
        self.elliptical = elliptical

        if NUMBA_AVAILABLE:
            self.name += " (Numba)"

    def fit(self, image, positions, fit_radius=3):
        """Fit integrated Gaussian using weighted least squares"""

        if len(positions) == 0:
            return FitResult(
                x=np.array([]),
                y=np.array([]),
                intensity=np.array([]),
                background=np.array([]),
                sigma_x=np.array([]),
                sigma_y=np.array([]),
                chi_squared=np.array([]),
                uncertainty=np.array([]),
                success=np.array([])
            )

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        results = fit_gaussian_wlsq_batch_numba(
            image, positions, fit_radius, self.initial_sigma
        )

        x = results[:, 0]
        y = results[:, 1]
        intensity = results[:, 2]
        sigma_x = results[:, 3]
        sigma_y = results[:, 4]
        background = results[:, 5]
        chi_squared = results[:, 6]
        success = results[:, 7]

        mask = success > 0

        uncertainty = np.zeros(len(x))
        for i in range(len(x)):
            if mask[i]:
                uncertainty[i] = compute_localization_precision(
                    intensity[i], background[i], sigma_x[i],
                    1.0, fitting_method='wlsq', is_emccd=self.is_emccd
                )

        return FitResult(
            x=x[mask],
            y=y[mask],
            intensity=intensity[mask],
            background=background[mask],
            sigma_x=sigma_x[mask],
            sigma_y=sigma_y[mask],
            chi_squared=chi_squared[mask],
            uncertainty=uncertainty[mask],
            success=success[mask]
        )


class GaussianMLEFitter(BaseFitter):
    """
    Integrated Gaussian PSF fitting using Maximum Likelihood Estimation
    with true Levenberg-Marquardt optimization.

    Assumes Poisson noise model — optimal for SMLM data.

    NUMBA-OPTIMIZED for 10-50x speedup!

    Parameters
    ----------
    initial_sigma : float
        Initial guess for sigma
    elliptical : bool
        Fit elliptical Gaussian — not yet supported
    """

    def __init__(self, initial_sigma=1.3, elliptical=False):
        super().__init__("GaussianMLE")
        self.initial_sigma = initial_sigma
        self.elliptical = elliptical

        if NUMBA_AVAILABLE:
            self.name += " (Numba)"

    def fit(self, image, positions, fit_radius=3):
        """Fit integrated Gaussian using maximum likelihood estimation"""

        if len(positions) == 0:
            return FitResult(
                x=np.array([]),
                y=np.array([]),
                intensity=np.array([]),
                background=np.array([]),
                sigma_x=np.array([]),
                sigma_y=np.array([]),
                chi_squared=np.array([]),
                uncertainty=np.array([]),
                success=np.array([])
            )

        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        results = fit_gaussian_mle_batch_numba(
            image, positions, fit_radius, self.initial_sigma
        )

        x = results[:, 0]
        y = results[:, 1]
        intensity = results[:, 2]
        sigma_x = results[:, 3]
        sigma_y = results[:, 4]
        background = results[:, 5]
        chi_squared = results[:, 6]
        success = results[:, 7]

        mask = success > 0

        # MLE: no Mortensen correction (it's for LSQ only)
        uncertainty = np.zeros(len(x))
        for i in range(len(x)):
            if mask[i]:
                uncertainty[i] = compute_localization_precision(
                    intensity[i], background[i], sigma_x[i],
                    1.0, fitting_method='mle', is_emccd=self.is_emccd
                )

        return FitResult(
            x=x[mask],
            y=y[mask],
            intensity=intensity[mask],
            background=background[mask],
            sigma_x=sigma_x[mask],
            sigma_y=sigma_y[mask],
            chi_squared=chi_squared[mask],
            uncertainty=uncertainty[mask],
            success=success[mask]
        )


class CentroidFitter(BaseFitter):
    """Simple weighted centroid fitting (fastest, least accurate)"""

    def __init__(self):
        super().__init__("Centroid")

    def fit(self, image, positions, fit_radius=3):
        """Fit using weighted centroid"""

        results = {
            'x': [],
            'y': [],
            'intensity': [],
            'background': [],
            'sigma_x': [],
            'sigma_y': []
        }

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract ROI
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            roi = image[r0:r1, c0:c1]

            # Weighted centroid
            total = roi.sum()
            if total > 0:
                rows, cols = np.mgrid[r0:r1, c0:c1]
                row_center = (rows * roi).sum() / total
                col_center = (cols * roi).sum() / total

                results['y'].append(row_center)  # Note: y is row
                results['x'].append(col_center)  # x is col
                results['intensity'].append(np.max(roi))
                results['background'].append(np.min(roi))
                results['sigma_x'].append(1.0)
                results['sigma_y'].append(1.0)

        return FitResult(
            x=np.array(results['x']),
            y=np.array(results['y']),
            intensity=np.array(results['intensity']),
            background=np.array(results['background']),
            sigma_x=np.array(results['sigma_x']),
            sigma_y=np.array(results['sigma_y']),
            chi_squared=np.zeros(len(results['x'])),
            uncertainty=np.ones(len(results['x'])),
            success=np.ones(len(results['x']))
        )


class RadialSymmetryFitter(BaseFitter):
    """
    Radial symmetry / phasor localization
    Very fast, good for high SNR data
    """

    def __init__(self):
        super().__init__("RadialSymmetry")

    def fit(self, image, positions, fit_radius=3):
        """Fit using radial symmetry method"""

        results = {
            'x': [],
            'y': [],
            'intensity': [],
            'background': [],
            'sigma_x': [],
            'sigma_y': []
        }

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract ROI
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            roi = image[r0:r1, c0:c1]

            # Compute gradients
            grad_y, grad_x = np.gradient(roi.astype(float))

            # Radial symmetry center
            # Simple version: weighted by gradient magnitude
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            if grad_mag.sum() > 0:
                rows, cols = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]

                row_center = (rows * grad_mag).sum() / grad_mag.sum()
                col_center = (cols * grad_mag).sum() / grad_mag.sum()

                results['y'].append(row_center + r0)
                results['x'].append(col_center + c0)
                results['intensity'].append(np.max(roi))
                results['background'].append(np.min(roi))
                results['sigma_x'].append(1.5)
                results['sigma_y'].append(1.5)

        return FitResult(
            x=np.array(results['x']),
            y=np.array(results['y']),
            intensity=np.array(results['intensity']),
            background=np.array(results['background']),
            sigma_x=np.array(results['sigma_x']),
            sigma_y=np.array(results['sigma_y']),
            chi_squared=np.zeros(len(results['x'])),
            uncertainty=np.ones(len(results['x'])),
            success=np.ones(len(results['x']))
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_fitter(fitter_type, **kwargs):
    """Factory function to create fitters

    Parameters
    ----------
    fitter_type : str
        Type: 'gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle', 'radial_symmetry', 'centroid'
    **kwargs : dict
        Fitter-specific parameters

    Returns
    -------
    fitter : BaseFitter
        Configured fitter object
    """
    fitter_map = {
        'gaussian_lsq': GaussianLSQFitter,
        'gaussian_wlsq': GaussianWLSQFitter,
        'gaussian_mle': GaussianMLEFitter,
        'centroid': CentroidFitter,
        'radial_symmetry': RadialSymmetryFitter,
    }

    fitter_type = fitter_type.lower()
    if fitter_type not in fitter_map:
        available = ', '.join(fitter_map.keys())
        raise ValueError(f"Unknown fitter type: {fitter_type}. Available: {available}")

    return fitter_map[fitter_type](**kwargs)


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_fitters():
    """Benchmark different fitters"""
    import time

    print("="*70)
    print("FITTING PERFORMANCE BENCHMARK")
    print("="*70)

    # Create test data
    image = np.random.randn(256, 256) * 10 + 100

    # Add some Gaussians
    n_molecules = 100
    positions = np.random.randint(10, 246, (n_molecules, 2))

    for pos in positions:
        y, x = np.mgrid[-5:6, -5:6]
        gaussian = 1000 * np.exp(-(x**2 + y**2) / (2 * 1.5**2))

        r0 = int(pos[0]) - 5
        r1 = int(pos[0]) + 6
        c0 = int(pos[1]) - 5
        c1 = int(pos[1]) + 6

        if r0 >= 0 and r1 < 256 and c0 >= 0 and c1 < 256:
            image[r0:r1, c0:c1] += gaussian

    print(f"\nTest image: {image.shape}")
    print(f"Fitting {n_molecules} molecules")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()

    # Test Gaussian LSQ
    print("1. Gaussian LSQ Fitter")
    fitter = GaussianLSQFitter(integrated=False)

    start = time.time()
    result = fitter.fit(image, positions, fit_radius=3)
    elapsed = time.time() - start

    print(f"   Time: {elapsed*1000:.1f} ms")
    print(f"   Per molecule: {elapsed*1000/n_molecules:.2f} ms")
    print(f"   Found: {len(result)} localizations")

    # Test Integrated Gaussian
    print("\n2. Integrated Gaussian LSQ Fitter")
    fitter2 = GaussianLSQFitter(integrated=True)

    start = time.time()
    result2 = fitter2.fit(image, positions, fit_radius=3)
    elapsed2 = time.time() - start

    print(f"   Time: {elapsed2*1000:.1f} ms")
    print(f"   Per molecule: {elapsed2*1000/n_molecules:.2f} ms")
    print(f"   Found: {len(result2)} localizations")

    # Test Centroid
    print("\n3. Centroid Fitter (baseline)")
    fitter3 = CentroidFitter()

    start = time.time()
    result3 = fitter3.fit(image, positions, fit_radius=3)
    elapsed3 = time.time() - start

    print(f"   Time: {elapsed3*1000:.1f} ms")
    print(f"   Per molecule: {elapsed3*1000/n_molecules:.2f} ms")
    print(f"   Speedup vs LSQ: {elapsed/elapsed3:.1f}x")

    print("\n" + "="*70)


if __name__ == "__main__":
    print(__doc__)
    benchmark_fitters()
