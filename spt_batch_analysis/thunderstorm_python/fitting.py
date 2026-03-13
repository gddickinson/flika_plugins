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
from scipy.special import erf
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

@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
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
TWO_OVER_SQRTPI = 2.0 / math.sqrt(math.pi)


@njit(fastmath=True, cache=True)
def _erf_approx(x):
    """Fast erf approximation using Abramowitz & Stegun formula 7.1.26.

    Maximum error: |epsilon(x)| <= 1.5e-7.
    This avoids scipy.special.erf dependency for Numba JIT.
    """
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


@njit(fastmath=True, cache=True)
def _erf_array(arr):
    """Apply erf approximation to a flat array."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = _erf_approx(arr[i])
    return out


# ============================================================================
# NUMBA-COMPILED MFA GRADIENT FUNCTIONS
# ============================================================================

@njit(fastmath=True, cache=True)
def _mfa_wlsq_gradient_numba(params, roi_flat, grid_ii, grid_jj,
                               n_emitters, n_params):
    """Numba-compiled WLSQ gradient for MFA.

    WLSQ = sum((data - model)^2 * weight), weight = 1/data.
    Returns (objective_value, gradient_vector).
    """
    grad = np.zeros(n_params)
    n_pixels = len(roi_flat)

    # Build model
    offset = params[-1]
    if offset < 1e-10:
        offset = 1e-10
    model = np.full(n_pixels, offset)
    sqrt2 = 1.4142135623730951  # math.sqrt(2)

    # Pre-allocate per-emitter arrays
    ex_arr = np.empty((n_emitters, n_pixels))
    ey_arr = np.empty((n_emitters, n_pixels))
    gx_p_arr = np.empty((n_emitters, n_pixels))
    gx_m_arr = np.empty((n_emitters, n_pixels))
    gy_p_arr = np.empty((n_emitters, n_pixels))
    gy_m_arr = np.empty((n_emitters, n_pixels))
    sig_arr = np.empty(n_emitters)
    inten_arr = np.empty(n_emitters)

    two_over_sqrtpi = 1.1283791670955126  # 2.0 / sqrt(pi)

    for k in range(n_emitters):
        base = k * 4
        x0 = params[base]
        y0 = params[base + 1]
        sig = abs(params[base + 2])
        if sig < 0.1:
            sig = 0.1
        inten = params[base + 3]
        if inten < 1e-10:
            inten = 1e-10
        s2 = sqrt2 * sig

        sig_arr[k] = sig
        inten_arr[k] = inten

        for i in range(n_pixels):
            ux_p = (grid_jj[i] - x0 + 0.5) / s2
            ux_m = (grid_jj[i] - x0 - 0.5) / s2
            uy_p = (grid_ii[i] - y0 + 0.5) / s2
            uy_m = (grid_ii[i] - y0 - 0.5) / s2

            erf_xp = _erf_approx(ux_p)
            erf_xm = _erf_approx(ux_m)
            erf_yp = _erf_approx(uy_p)
            erf_ym = _erf_approx(uy_m)

            ex = 0.5 * (erf_xp - erf_xm)
            ey = 0.5 * (erf_yp - erf_ym)
            model[i] += inten * ex * ey

            gx_p = two_over_sqrtpi * math.exp(-ux_p * ux_p)
            gx_m = two_over_sqrtpi * math.exp(-ux_m * ux_m)
            gy_p = two_over_sqrtpi * math.exp(-uy_p * uy_p)
            gy_m = two_over_sqrtpi * math.exp(-uy_m * uy_m)

            ex_arr[k, i] = ex
            ey_arr[k, i] = ey
            gx_p_arr[k, i] = gx_p
            gx_m_arr[k, i] = gx_m
            gy_p_arr[k, i] = gy_p
            gy_m_arr[k, i] = gy_m

    # Clip model
    for i in range(n_pixels):
        if model[i] < 1e-10:
            model[i] = 1e-10

    # Compute data-based weights (matching ImageJ)
    max_data = roi_flat[0]
    for i in range(1, n_pixels):
        if roi_flat[i] > max_data:
            max_data = roi_flat[i]
    if max_data < 1e-10:
        max_data = 1e-10
    min_weight = 1.0 / max_data
    max_weight = 1000.0 * min_weight

    # Compute objective and gradient common factor
    wlsq = 0.0
    common = np.empty(n_pixels)
    for i in range(n_pixels):
        d = roi_flat[i]
        if d < 1e-10:
            d = 1e-10
        w = 1.0 / d
        if w > max_weight:
            w = max_weight
        elif w < min_weight:
            w = min_weight
        residual = roi_flat[i] - model[i]
        wlsq += residual * residual * w
        common[i] = -2.0 * residual * w

    # Gradient w.r.t. offset
    grad_offset = 0.0
    for i in range(n_pixels):
        grad_offset += common[i]
    grad[-1] = grad_offset

    # Gradient w.r.t. each emitter's parameters
    for k in range(n_emitters):
        base = k * 4
        sig = sig_arr[k]
        inten = inten_arr[k]
        inv_s2 = 1.0 / (sqrt2 * sig)

        g_x0 = 0.0
        g_y0 = 0.0
        g_sig = 0.0
        g_int = 0.0

        for i in range(n_pixels):
            ex = ex_arr[k, i]
            ey = ey_arr[k, i]
            c = common[i]

            dex_dx0 = 0.5 * inv_s2 * (gx_m_arr[k, i] - gx_p_arr[k, i])
            dey_dy0 = 0.5 * inv_s2 * (gy_m_arr[k, i] - gy_p_arr[k, i])

            ux_p = (grid_jj[i] - params[base] + 0.5) / (sqrt2 * sig)
            ux_m = (grid_jj[i] - params[base] - 0.5) / (sqrt2 * sig)
            uy_p = (grid_ii[i] - params[base + 1] + 0.5) / (sqrt2 * sig)
            uy_m = (grid_ii[i] - params[base + 1] - 0.5) / (sqrt2 * sig)

            dex_dsig = 0.5 * (-ux_p * gx_p_arr[k, i] + ux_m * gx_m_arr[k, i]) / sig
            dey_dsig = 0.5 * (-uy_p * gy_p_arr[k, i] + uy_m * gy_m_arr[k, i]) / sig

            g_x0 += c * inten * dex_dx0 * ey
            g_y0 += c * inten * ex * dey_dy0
            g_sig += c * inten * (dex_dsig * ey + ex * dey_dsig)
            g_int += c * ex * ey

        grad[base] = g_x0
        grad[base + 1] = g_y0
        grad[base + 2] = g_sig
        grad[base + 3] = g_int

    return wlsq, grad


@njit(fastmath=True, cache=True)
def _mfa_nll_gradient_numba(params, roi_flat, grid_ii, grid_jj,
                             n_emitters, n_params):
    """Numba-compiled Poisson NLL gradient for MFA.

    Returns (nll_value, gradient_vector).
    """
    grad = np.zeros(n_params)
    n_pixels = len(roi_flat)

    offset = params[-1]
    if offset < 1e-10:
        offset = 1e-10
    model = np.full(n_pixels, offset)
    sqrt2 = 1.4142135623730951
    two_over_sqrtpi = 1.1283791670955126

    ex_arr = np.empty((n_emitters, n_pixels))
    ey_arr = np.empty((n_emitters, n_pixels))
    gx_p_arr = np.empty((n_emitters, n_pixels))
    gx_m_arr = np.empty((n_emitters, n_pixels))
    gy_p_arr = np.empty((n_emitters, n_pixels))
    gy_m_arr = np.empty((n_emitters, n_pixels))
    sig_arr = np.empty(n_emitters)
    inten_arr = np.empty(n_emitters)

    for k in range(n_emitters):
        base = k * 4
        x0 = params[base]
        y0 = params[base + 1]
        sig = abs(params[base + 2])
        if sig < 0.1:
            sig = 0.1
        inten = params[base + 3]
        if inten < 1e-10:
            inten = 1e-10
        s2 = sqrt2 * sig

        sig_arr[k] = sig
        inten_arr[k] = inten

        for i in range(n_pixels):
            ux_p = (grid_jj[i] - x0 + 0.5) / s2
            ux_m = (grid_jj[i] - x0 - 0.5) / s2
            uy_p = (grid_ii[i] - y0 + 0.5) / s2
            uy_m = (grid_ii[i] - y0 - 0.5) / s2

            erf_xp = _erf_approx(ux_p)
            erf_xm = _erf_approx(ux_m)
            erf_yp = _erf_approx(uy_p)
            erf_ym = _erf_approx(uy_m)

            ex = 0.5 * (erf_xp - erf_xm)
            ey = 0.5 * (erf_yp - erf_ym)
            model[i] += inten * ex * ey

            gx_p = two_over_sqrtpi * math.exp(-ux_p * ux_p)
            gx_m = two_over_sqrtpi * math.exp(-ux_m * ux_m)
            gy_p = two_over_sqrtpi * math.exp(-uy_p * uy_p)
            gy_m = two_over_sqrtpi * math.exp(-uy_m * uy_m)

            ex_arr[k, i] = ex
            ey_arr[k, i] = ey
            gx_p_arr[k, i] = gx_p
            gx_m_arr[k, i] = gx_m
            gy_p_arr[k, i] = gy_p
            gy_m_arr[k, i] = gy_m

    # Clip model
    for i in range(n_pixels):
        if model[i] < 1e-10:
            model[i] = 1e-10

    # NLL = sum(model - data * log(model))
    nll = 0.0
    ratio = np.empty(n_pixels)
    for i in range(n_pixels):
        nll += model[i] - roi_flat[i] * math.log(model[i])
        ratio[i] = 1.0 - roi_flat[i] / model[i]

    # Gradient w.r.t. offset
    grad_offset = 0.0
    for i in range(n_pixels):
        grad_offset += ratio[i]
    grad[-1] = grad_offset

    for k in range(n_emitters):
        base = k * 4
        sig = sig_arr[k]
        inten = inten_arr[k]
        inv_s2 = 1.0 / (sqrt2 * sig)

        g_x0 = 0.0
        g_y0 = 0.0
        g_sig = 0.0
        g_int = 0.0

        for i in range(n_pixels):
            ex = ex_arr[k, i]
            ey = ey_arr[k, i]
            r = ratio[i]

            dex_dx0 = 0.5 * inv_s2 * (gx_m_arr[k, i] - gx_p_arr[k, i])
            dey_dy0 = 0.5 * inv_s2 * (gy_m_arr[k, i] - gy_p_arr[k, i])

            ux_p = (grid_jj[i] - params[base] + 0.5) / (sqrt2 * sig)
            ux_m = (grid_jj[i] - params[base] - 0.5) / (sqrt2 * sig)
            uy_p = (grid_ii[i] - params[base + 1] + 0.5) / (sqrt2 * sig)
            uy_m = (grid_ii[i] - params[base + 1] - 0.5) / (sqrt2 * sig)

            dex_dsig = 0.5 * (-ux_p * gx_p_arr[k, i] + ux_m * gx_m_arr[k, i]) / sig
            dey_dsig = 0.5 * (-uy_p * gy_p_arr[k, i] + uy_m * gy_m_arr[k, i]) / sig

            g_x0 += r * inten * dex_dx0 * ey
            g_y0 += r * inten * ex * dey_dy0
            g_sig += r * inten * (dex_dsig * ey + ex * dey_dsig)
            g_int += r * ex * ey

        grad[base] = g_x0
        grad[base + 1] = g_y0
        grad[base + 2] = g_sig
        grad[base + 3] = g_int

    return nll, grad


@njit(fastmath=True, cache=True)
def _mfa_compute_rss_numba(params, roi_flat, grid_ii, grid_jj, n_emitters):
    """Numba-compiled RSS computation for F-test (weight=1/data)."""
    n_pixels = len(roi_flat)
    offset = params[-1]
    if offset < 1e-10:
        offset = 1e-10
    model = np.full(n_pixels, offset)
    sqrt2 = 1.4142135623730951

    for k in range(n_emitters):
        base = k * 4
        x0 = params[base]
        y0 = params[base + 1]
        sig = abs(params[base + 2])
        if sig < 0.1:
            sig = 0.1
        inten = params[base + 3]
        if inten < 1e-10:
            inten = 1e-10
        s2 = sqrt2 * sig

        for i in range(n_pixels):
            ux_p = (grid_jj[i] - x0 + 0.5) / s2
            ux_m = (grid_jj[i] - x0 - 0.5) / s2
            uy_p = (grid_ii[i] - y0 + 0.5) / s2
            uy_m = (grid_ii[i] - y0 - 0.5) / s2
            ex = 0.5 * (_erf_approx(ux_p) - _erf_approx(ux_m))
            ey = 0.5 * (_erf_approx(uy_p) - _erf_approx(uy_m))
            model[i] += inten * ex * ey

    # Clip model
    for i in range(n_pixels):
        if model[i] < 1e-10:
            model[i] = 1e-10

    # Weight by 1/data with clamping
    max_data = roi_flat[0]
    for i in range(1, n_pixels):
        if roi_flat[i] > max_data:
            max_data = roi_flat[i]
    if max_data < 1e-10:
        max_data = 1e-10
    min_weight = 1.0 / max_data
    max_weight = 1000.0 * min_weight

    rss = 0.0
    for i in range(n_pixels):
        d = roi_flat[i]
        if d < 1e-10:
            d = 1e-10
        w = 1.0 / d
        if w > max_weight:
            w = max_weight
        elif w < min_weight:
            w = min_weight
        residual = roi_flat[i] - model[i]
        rss += residual * residual * w

    return rss


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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


@njit(parallel=True, fastmath=True, cache=True)
def fit_gaussian_lsq_batch_numba(image, positions, fit_radius=3,
                                  initial_sigma=1.3, max_iterations=1001):
    """
    Fit integrated Gaussian PSF to multiple positions using Least Squares
    with true Levenberg-Marquardt optimization.

    Uses erf-based pixel-integrated Gaussian model matching ImageJ thunderSTORM.
    5 parameters: [x0, y0, sigma, intensity, offset]

    Positivity of sigma, intensity, and offset is enforced via squared
    parameterization: the optimizer works in sqrt-space for these three
    parameters, ensuring they remain non-negative without hard clipping.

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
        offset = max(np.min(roi), 0.01)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma = initial_sigma
        intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

        # Squared parameterization: optimizer works on p = [x0, y0, sqrt(sigma), sqrt(intensity), sqrt(offset)]
        # Physical params = [x0, y0, p2^2, p3^2, p4^2]
        p2 = math.sqrt(sigma)
        p3 = math.sqrt(intensity)
        p4 = math.sqrt(offset)

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
            # Jacobian w.r.t. transformed params: chain rule for squared params
            H = np.zeros((5, 5))
            g = np.zeros(5)

            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    model_val = integrated_gaussian_value(fj, fi, x0, y0, sigma, intensity, offset)
                    residual = roi[ii, jj] - model_val
                    J_orig = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    # Transform Jacobian: dmodel/dp_k = dmodel/dtheta_k * dtheta_k/dp_k
                    # For k=0,1 (x0,y0): dtheta/dp = 1
                    # For k=2 (sigma): dtheta/dp = 2*p2
                    # For k=3 (intensity): dtheta/dp = 2*p3
                    # For k=4 (offset): dtheta/dp = 2*p4
                    J = np.zeros(5)
                    J[0] = J_orig[0]
                    J[1] = J_orig[1]
                    J[2] = J_orig[2] * 2.0 * p2
                    J[3] = J_orig[3] * 2.0 * p3
                    J[4] = J_orig[4] * 2.0 * p4

                    for p in range(5):
                        g[p] += J[p] * residual
                        for q in range(5):
                            H[p][q] += J[p] * J[q]

            # Augment diagonal: H_aug = H + lambda * diag(H)
            H_aug = H.copy()
            for p in range(5):
                H_aug[p][p] *= (1.0 + lambda_lm)

            # Solve for parameter update in transformed space
            g_copy = g.copy()
            dp = solve_5x5(H_aug, g_copy)

            # Candidate transformed parameters
            x0_new = x0 + dp[0]
            y0_new = y0 + dp[1]
            p2_new = p2 + dp[2]
            p3_new = p3 + dp[3]
            p4_new = p4 + dp[4]

            # Back-transform to physical params (always non-negative)
            sigma_new = p2_new * p2_new
            intensity_new = p3_new * p3_new
            offset_new = p4_new * p4_new

            # Sanity bounds on sigma
            sigma_new = min(sigma_new, 10.0)
            if sigma_new < 0.1:
                sigma_new = 0.1
                p2_new = math.sqrt(sigma_new)

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
                p2 = p2_new
                p3 = p3_new
                p4 = p4_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                chi_sq = chi_sq_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-8:
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
        results[i, 6] = chi_sq
        # Mark as failed if sigma hit the cap (diverged fit)
        results[i, 7] = 0.0 if sigma >= 9.9 else 1.0

    return results


@njit(parallel=True, fastmath=True, cache=True)
def fit_gaussian_wlsq_batch_numba(image, positions, fit_radius=3,
                                   initial_sigma=1.3, max_iterations=1001):
    """
    Fit integrated Gaussian PSF using Weighted Least Squares
    with true Levenberg-Marquardt optimization.

    Weights: w = 1/max(data, 1) (Poisson variance model).
    Squared parameterization for positivity enforcement on sigma, intensity, offset.

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
        offset = max(np.min(roi), 0.01)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma = initial_sigma
        intensity = max(peak * 2.0 * math.pi * sigma * sigma, 1.0)

        # Squared parameterization
        p2 = math.sqrt(sigma)
        p3 = math.sqrt(intensity)
        p4 = math.sqrt(offset)

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
                    J_orig = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    # Transform Jacobian for squared parameterization
                    J = np.zeros(5)
                    J[0] = J_orig[0]
                    J[1] = J_orig[1]
                    J[2] = J_orig[2] * 2.0 * p2
                    J[3] = J_orig[3] * 2.0 * p3
                    J[4] = J_orig[4] * 2.0 * p4

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
            p2_new = p2 + dp[2]
            p3_new = p3 + dp[3]
            p4_new = p4 + dp[4]

            sigma_new = p2_new * p2_new
            intensity_new = p3_new * p3_new
            offset_new = p4_new * p4_new

            sigma_new = min(sigma_new, 10.0)
            if sigma_new < 0.1:
                sigma_new = 0.1
                p2_new = math.sqrt(sigma_new)

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
                p2 = p2_new
                p3 = p3_new
                p4 = p4_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                chi_sq = chi_sq_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-8:
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
        results[i, 6] = chi_sq
        # Mark as failed if sigma hit the cap (diverged fit)
        results[i, 7] = 0.0 if sigma >= 9.9 else 1.0

    return results


@njit(parallel=True, fastmath=True, cache=True)
def fit_gaussian_mle_batch_numba(image, positions, fit_radius=3,
                                  initial_sigma=1.3, max_iterations=5000):
    """
    Fit integrated Gaussian PSF using Maximum Likelihood Estimation
    with true Levenberg-Marquardt optimization.

    Poisson noise model: -log L = sum(model - data*log(model))
    Squared parameterization for positivity enforcement on sigma, intensity, offset.
    Tighter convergence tolerance (1e-8) and higher max iterations (1000)
    matching original ThunderSTORM's MLE convergence requirements.

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

        # Squared parameterization
        p2 = math.sqrt(sigma)
        p3 = math.sqrt(intensity)
        p4 = math.sqrt(offset)

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
            # Fisher information Hessian with squared parameterization
            H = np.zeros((5, 5))
            g = np.zeros(5)

            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    model_val = integrated_gaussian_value(fj, fi, x0, y0, sigma, intensity, offset)
                    model_val = max(model_val, 1e-10)
                    data = roi[ii, jj]
                    J_orig = integrated_gaussian_jacobian(fj, fi, x0, y0, sigma, intensity, offset)

                    # Transform Jacobian for squared parameterization
                    J = np.zeros(5)
                    J[0] = J_orig[0]
                    J[1] = J_orig[1]
                    J[2] = J_orig[2] * 2.0 * p2
                    J[3] = J_orig[3] * 2.0 * p3
                    J[4] = J_orig[4] * 2.0 * p4

                    # Weight = 1/model (Fisher information)
                    w = 1.0 / model_val
                    factor = data / model_val - 1.0

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
            p2_new = p2 + dp[2]
            p3_new = p3 + dp[3]
            p4_new = p4 + dp[4]

            sigma_new = p2_new * p2_new
            intensity_new = p3_new * p3_new
            offset_new = p4_new * p4_new

            sigma_new = min(sigma_new, 10.0)
            if sigma_new < 0.1:
                sigma_new = 0.1
                p2_new = math.sqrt(sigma_new)
            offset_new = max(offset_new, 1e-10)

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
                p2 = p2_new
                p3 = p3_new
                p4 = p4_new
                sigma = sigma_new
                intensity = intensity_new
                offset = offset_new
                neg_ll = neg_ll_new
                lambda_lm /= 10.0
                lambda_lm = max(lambda_lm, 1e-7)

                if rel_change < 1e-8:
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
        results[i, 6] = neg_ll
        # Mark as failed if sigma hit the cap (diverged fit)
        results[i, 7] = 0.0 if sigma >= 9.9 else 1.0

    return results


def compute_localization_precision(intensity, background, sigma, pixel_size,
                                    fitting_method='lsq', is_emccd=False):
    """
    Compute localization precision using Thompson formula.

    Matches ThunderSTORM's MoleculeDescriptor.uncertaintyXY():
        sqrt(sa2/N * (F + 4*tau))

    where sa2 = s^2 + a^2/12, tau = 2*pi*sa2*b^2/(N*a^2),
    F = 2 for EMCCD (excess noise factor), 1 otherwise.

    Empirically validated: this formula matches ImageJ ThunderSTORM
    output within ~1% for all fitting methods (LSQ, WLSQ, MLE).

    References:
    - Thompson et al. 2002 (base formula)
    - Quan et al. 2010 (EMCCD excess noise factor F=2)

    Parameters
    ----------
    intensity : float
        Total molecule intensity (photons)
    background : float
        Background standard deviation per pixel (photons)
    sigma : float
        PSF standard deviation (same units as pixel_size)
    pixel_size : float
        Pixel size (set to 1.0 for pixel-unit computation)
    fitting_method : str
        Fitting method: 'lsq', 'wlsq', or 'mle' (same formula for all)
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

    # EMCCD excess noise factor (Quan et al. 2010)
    F = 2.0 if is_emccd else 1.0

    # sa2 = s^2 + a^2/12  (effective PSF variance including pixelation)
    sa2 = s * s + a * a / 12.0

    # tau = 2*pi*sa2*b^2 / (N*a^2)
    # b is background noise standard deviation per pixel
    tau = 2.0 * np.pi * sa2 * b * b / (N * a * a)

    # Thompson formula: var = sa2/N * (F + 4*tau)
    variance = sa2 / N * (F + 4.0 * tau)

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

    # FWHM = 2 * sqrt(2 * ln2) * sigma ≈ 2.3548 * sigma
    FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ≈ 0.4247

    def __init__(self, name="BaseFitter"):
        self.name = name
        self.pixel_size = 100.0  # nm
        self.photons_per_adu = 1.0
        self.baseline = 0.0
        self.em_gain = 1.0
        self.is_emccd = False
        # Sigma range constraint (in pixels).
        # Set via set_sigma_range() or set_fwhm_range().
        # None means no constraint.
        self.sigma_range = None  # (min_sigma, max_sigma) in pixels

    def set_camera_params(self, pixel_size=100.0, photons_per_adu=1.0,
                         baseline=0.0, em_gain=1.0, is_emccd=False):
        """Set camera calibration parameters"""
        self.pixel_size = pixel_size
        self.photons_per_adu = photons_per_adu
        self.baseline = baseline
        self.em_gain = em_gain
        self.is_emccd = is_emccd

    def set_sigma_range(self, min_sigma=None, max_sigma=None):
        """Set allowed sigma range in pixels for fit rejection.

        Fits with sigma outside this range are discarded.  Matches
        thunderSTORM's FWHM range validation.

        Parameters
        ----------
        min_sigma, max_sigma : float or None
            Bounds in pixels.  None disables that bound.
        """
        if min_sigma is None and max_sigma is None:
            self.sigma_range = None
        else:
            self.sigma_range = (
                min_sigma if min_sigma is not None else 0.0,
                max_sigma if max_sigma is not None else float('inf')
            )

    def set_fwhm_range(self, min_fwhm_nm=None, max_fwhm_nm=None):
        """Set allowed FWHM range in nanometers for fit rejection.

        Convenience wrapper that converts FWHM (nm) to sigma (pixels)
        using the current pixel_size.

        Parameters
        ----------
        min_fwhm_nm, max_fwhm_nm : float or None
            Bounds in nanometers.
        """
        min_sigma = None
        max_sigma = None
        if min_fwhm_nm is not None:
            min_sigma = (min_fwhm_nm * self.FWHM_TO_SIGMA) / self.pixel_size
        if max_fwhm_nm is not None:
            max_sigma = (max_fwhm_nm * self.FWHM_TO_SIGMA) / self.pixel_size
        self.set_sigma_range(min_sigma, max_sigma)

    def _apply_sigma_filter(self, result):
        """Filter a FitResult by the configured sigma range.

        Parameters
        ----------
        result : FitResult
            Raw fitting results

        Returns
        -------
        filtered : FitResult
            Results with out-of-range sigma values removed
        """
        if self.sigma_range is None or len(result.x) == 0:
            return result

        min_s, max_s = self.sigma_range
        # Use average of sigma_x and sigma_y for symmetric/elliptical
        sigma_avg = (result.sigma_x + result.sigma_y) / 2.0
        mask = (sigma_avg >= min_s) & (sigma_avg <= max_s)

        return FitResult(
            x=result.x[mask],
            y=result.y[mask],
            intensity=result.intensity[mask],
            background=result.background[mask],
            sigma_x=result.sigma_x[mask],
            sigma_y=result.sigma_y[mask],
            chi_squared=result.chi_squared[mask],
            uncertainty=result.uncertainty[mask],
            success=result.success[mask]
        )

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

        result = FitResult(
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
        return self._apply_sigma_filter(result)


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

        result = FitResult(
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
        return self._apply_sigma_filter(result)


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

        result = FitResult(
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
        return self._apply_sigma_filter(result)


class CentroidFitter(BaseFitter):
    """Simple weighted centroid fitting (fastest, least accurate)"""

    def __init__(self, **kwargs):
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


@njit(fastmath=True, cache=True)
def _radial_symmetry_uniform_filter_3x3(arr, ny, nx):
    """3x3 box filter with 'nearest' boundary, matching scipy uniform_filter(size=3, mode='nearest')."""
    out = np.empty((ny, nx))
    for i in range(ny):
        for j in range(nx):
            s = 0.0
            for di in range(-1, 2):
                ii = i + di
                if ii < 0:
                    ii = 0
                elif ii >= ny:
                    ii = ny - 1
                for dj in range(-1, 2):
                    jj = j + dj
                    if jj < 0:
                        jj = 0
                    elif jj >= nx:
                        jj = nx - 1
                    s += arr[ii, jj]
            out[i, j] = s / 9.0
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _radial_symmetry_batch_numba(image, positions, fit_radius):
    """Numba-compiled batch radial symmetry fitting.

    Returns ndarray (N, 9): [x, y, intensity, background, sigma, chi2, uncertainty, success, _unused]
    """
    n_fits = len(positions)
    results = np.zeros((n_fits, 9))
    img_h = image.shape[0]
    img_w = image.shape[1]

    for idx in prange(n_fits):
        row = int(positions[idx, 0])
        col = int(positions[idx, 1])

        r0 = max(0, row - fit_radius)
        r1 = min(img_h, row + fit_radius + 1)
        c0 = max(0, col - fit_radius)
        c1 = min(img_w, col + fit_radius + 1)

        ny = r1 - r0
        nx = c1 - c0
        if ny < 3 or nx < 3:
            continue

        # Copy ROI to float
        roi = np.empty((ny, nx))
        for i in range(ny):
            for j in range(nx):
                roi[i, j] = float(image[r0 + i, c0 + j])

        # Diagonal gradients
        gny = ny - 1
        gnx = nx - 1
        dIdu = np.empty((gny, gnx))
        dIdv = np.empty((gny, gnx))
        for i in range(gny):
            for j in range(gnx):
                dIdu[i, j] = roi[i, j + 1] - roi[i + 1, j]
                dIdv[i, j] = roi[i, j] - roi[i + 1, j + 1]

        # Smooth with 3x3 box filter
        if gny >= 3 and gnx >= 3:
            dIdu = _radial_symmetry_uniform_filter_3x3(dIdu, gny, gnx)
            dIdv = _radial_symmetry_uniform_filter_3x3(dIdv, gny, gnx)

        # Compute slopes, intercepts, weights, and WLS sums in one pass
        # First pass: gradient magnitudes and centroid
        total_gm = 0.0
        gm_x_sum = 0.0
        gm_y_sum = 0.0
        for i in range(gny):
            for j in range(gnx):
                gm = math.sqrt(dIdu[i, j] ** 2 + dIdv[i, j] ** 2)
                row_g = float(i) + 0.5
                col_g = float(j) + 0.5
                total_gm += gm
                gm_x_sum += col_g * gm
                gm_y_sum += row_g * gm

        if total_gm < 1e-30:
            continue
        x_centroid = gm_x_sum / total_gm
        y_centroid = gm_y_sum / total_gm

        # Second pass: WLS sums
        sw = 0.0
        smw = 0.0
        smmw = 0.0
        sbw = 0.0
        smbw = 0.0
        n_valid = 0

        for i in range(gny):
            for j in range(gnx):
                denom = dIdu[i, j] - dIdv[i, j]
                if abs(denom) <= 1e-6:
                    continue
                n_valid += 1

                row_g = float(i) + 0.5
                col_g = float(j) + 0.5
                m_val = -(dIdu[i, j] + dIdv[i, j]) / denom
                b_val = row_g - m_val * col_g

                gm = math.sqrt(dIdu[i, j] ** 2 + dIdv[i, j] ** 2)
                dist = math.sqrt((col_g - x_centroid) ** 2 + (row_g - y_centroid) ** 2)
                if dist < 1e-10:
                    dist = 1e-10
                w = gm / dist

                w_line = w / (m_val * m_val + 1.0)
                sw += w_line
                smw += m_val * w_line
                smmw += m_val * m_val * w_line
                sbw += b_val * w_line
                smbw += m_val * b_val * w_line

        if n_valid < 3:
            continue

        det_val = smw * smw - smmw * sw
        if abs(det_val) < 1e-30:
            continue

        xc = (smbw * sw - smw * sbw) / det_val
        yc = (smbw * smw - smmw * sbw) / det_val

        # Peak intensity and background
        peak_val = roi[0, 0]
        bg_val = roi[0, 0]
        for i in range(ny):
            for j in range(nx):
                if roi[i, j] > peak_val:
                    peak_val = roi[i, j]
                if roi[i, j] < bg_val:
                    bg_val = roi[i, j]

        # Sigma estimate from intensity-weighted second moment
        total_w = 0.0
        r2_sum = 0.0
        for i in range(ny):
            for j in range(nx):
                v = roi[i, j] - bg_val
                if v > 0:
                    r2 = (float(j) - xc) ** 2 + (float(i) - yc) ** 2
                    r2_sum += v * r2
                    total_w += v

        if total_w > 0:
            sigma_est = math.sqrt(r2_sum / total_w)
            if sigma_est < 0.5:
                sigma_est = 0.5
            elif sigma_est > fit_radius:
                sigma_est = float(fit_radius)
        else:
            sigma_est = 1.5

        # Chi-squared: sum of squared residuals vs Gaussian model
        N_photons = peak_val - bg_val
        chi2 = 0.0
        if N_photons > 0 and sigma_est > 0:
            inv_2s2 = 1.0 / (2.0 * sigma_est * sigma_est)
            for i in range(ny):
                for j in range(nx):
                    model_val = bg_val + N_photons * math.exp(-((float(j) - xc) ** 2 + (float(i) - yc) ** 2) * inv_2s2)
                    diff = roi[i, j] - model_val
                    chi2 += diff * diff

        # Thompson uncertainty
        sa2 = sigma_est * sigma_est + 1.0 / 12.0
        if N_photons > 0:
            term1 = sa2 / N_photons
            term2 = (8.0 * math.pi * sa2 * sa2 * bg_val * bg_val) / (N_photons * N_photons)
            unc = math.sqrt(term1 + term2)
        else:
            unc = 1e10

        results[idx, 0] = xc + float(c0)
        results[idx, 1] = yc + float(r0)
        results[idx, 2] = peak_val
        results[idx, 3] = bg_val
        results[idx, 4] = sigma_est
        results[idx, 5] = chi2
        results[idx, 6] = unc
        results[idx, 7] = 1.0  # success

    return results


class RadialSymmetryFitter(BaseFitter):
    """
    Radial symmetry localization (Parthasarathy 2012).

    Computes image gradients along 45-degree rotated coordinates,
    fits gradient lines via weighted least squares to find the point
    of maximal radial symmetry.  Model-free, non-iterative, very fast.

    Uses Numba JIT compilation for ~5-10x speedup.
    """

    def __init__(self, **kwargs):
        super().__init__("RadialSymmetry")

    def fit(self, image, positions, fit_radius=3):
        """Fit using radial symmetry method (Parthasarathy 2012)."""
        if len(positions) == 0:
            return FitResult(
                x=np.array([]), y=np.array([]),
                intensity=np.array([]), background=np.array([]),
                sigma_x=np.array([]), sigma_y=np.array([]),
                chi_squared=np.array([]), uncertainty=np.array([]),
                success=np.array([])
            )

        img = image.astype(np.float64) if image.dtype != np.float64 else image
        pos = positions.astype(np.float64) if positions.dtype != np.float64 else positions

        raw = _radial_symmetry_batch_numba(img, pos, fit_radius)

        # Filter to successful fits
        mask = raw[:, 7] > 0.5
        good = raw[mask]

        return FitResult(
            x=good[:, 0].copy(),
            y=good[:, 1].copy(),
            intensity=good[:, 2].copy(),
            background=good[:, 3].copy(),
            sigma_x=good[:, 4].copy(),
            sigma_y=good[:, 4].copy(),
            chi_squared=good[:, 5].copy(),
            uncertainty=good[:, 6].copy(),
            success=np.ones(len(good))
        )


# ============================================================================
# PHASOR-BASED LOCALIZATION (Martens et al. 2018)
# ============================================================================

class PhasorFitter(BaseFitter):
    """Phasor-based localization (Martens et al. 2018, J. Chem. Phys.).

    Converts ROI around PSF to phase vectors by computing first Fourier
    coefficients in x and y.  Phasor angles yield lateral position.
    Model-free, non-iterative, extremely fast.
    """

    def __init__(self, **kwargs):
        super().__init__("Phasor")

    def fit(self, image, positions, fit_radius=3):
        """Fit using phasor (first Fourier coefficient) method."""
        results = {
            'x': [], 'y': [], 'intensity': [], 'background': [],
            'sigma_x': [], 'sigma_y': []
        }

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            roi = image[r0:r1, c0:c1].astype(float)

            ny, nx = roi.shape
            if ny < 2 or nx < 2:
                continue

            # Compute first Fourier coefficient along each axis
            # Phasor x: sum of roi * exp(-2*pi*i*col_index/nx) -> angle gives x
            # Phasor y: sum of roi * exp(-2*pi*i*row_index/ny) -> angle gives y
            col_indices = np.arange(nx)
            row_indices = np.arange(ny)

            # Sum along rows to get 1D projection for x
            proj_x = roi.sum(axis=0)
            phase_x = np.sum(proj_x * np.exp(-2j * np.pi * col_indices / nx))
            angle_x = np.angle(phase_x)
            if angle_x < 0:
                angle_x += 2 * np.pi
            xc = angle_x * nx / (2 * np.pi)

            # Sum along cols to get 1D projection for y
            proj_y = roi.sum(axis=1)
            phase_y = np.sum(proj_y * np.exp(-2j * np.pi * row_indices / ny))
            angle_y = np.angle(phase_y)
            if angle_y < 0:
                angle_y += 2 * np.pi
            yc = angle_y * ny / (2 * np.pi)

            results['x'].append(xc + c0)
            results['y'].append(yc + r0)
            results['intensity'].append(float(np.sum(roi) - np.min(roi) * ny * nx))
            results['background'].append(float(np.min(roi)))
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
# ELLIPTICAL GAUSSIAN PSF (7 parameters)
# ============================================================================

@njit(fastmath=True, cache=True)
def integrated_elliptical_gaussian_value(jj, ii, x0, y0, sigma1, sigma2, intensity, offset):
    """Compute integrated elliptical Gaussian PSF value for pixel (jj, ii).

    Parameters: x0, y0, sigma1 (x-width), sigma2 (y-width), intensity, offset.
    No rotation angle — aligned to pixel axes (sufficient for astigmatism).
    """
    s2x = math.sqrt(2.0) * sigma1
    s2y = math.sqrt(2.0) * sigma2
    ex = 0.5 * (math.erf((jj - x0 + 0.5) / s2x) - math.erf((jj - x0 - 0.5) / s2x))
    ey = 0.5 * (math.erf((ii - y0 + 0.5) / s2y) - math.erf((ii - y0 - 0.5) / s2y))
    return offset + intensity * ex * ey


@njit(fastmath=True, cache=True)
def integrated_elliptical_gaussian_jacobian(jj, ii, x0, y0, sigma1, sigma2, intensity, offset):
    """Jacobian of elliptical integrated Gaussian w.r.t. [x0, y0, sigma1, sigma2, intensity, offset]."""
    jac = np.zeros(6)
    s2x = math.sqrt(2.0) * sigma1
    s2y = math.sqrt(2.0) * sigma2

    erf_xp = math.erf((jj - x0 + 0.5) / s2x)
    erf_xm = math.erf((jj - x0 - 0.5) / s2x)
    erf_yp = math.erf((ii - y0 + 0.5) / s2y)
    erf_ym = math.erf((ii - y0 - 0.5) / s2y)

    ex = 0.5 * (erf_xp - erf_xm)
    ey = 0.5 * (erf_yp - erf_ym)

    two_over_sqrtpi = 2.0 / math.sqrt(math.pi)
    inv_s2x = 1.0 / s2x
    inv_s2y = 1.0 / s2y

    ux_p = (jj - x0 + 0.5) * inv_s2x
    ux_m = (jj - x0 - 0.5) * inv_s2x
    uy_p = (ii - y0 + 0.5) * inv_s2y
    uy_m = (ii - y0 - 0.5) * inv_s2y

    gx_p = two_over_sqrtpi * math.exp(-ux_p * ux_p)
    gx_m = two_over_sqrtpi * math.exp(-ux_m * ux_m)
    gy_p = two_over_sqrtpi * math.exp(-uy_p * uy_p)
    gy_m = two_over_sqrtpi * math.exp(-uy_m * uy_m)

    dex_dx0 = 0.5 * inv_s2x * (gx_m - gx_p)
    dey_dy0 = 0.5 * inv_s2y * (gy_m - gy_p)
    dex_dsigma1 = 0.5 * (-ux_p * gx_p + ux_m * gx_m) / sigma1
    dey_dsigma2 = 0.5 * (-uy_p * gy_p + uy_m * gy_m) / sigma2

    jac[0] = intensity * dex_dx0 * ey           # d/dx0
    jac[1] = intensity * ex * dey_dy0           # d/dy0
    jac[2] = intensity * dex_dsigma1 * ey       # d/dsigma1
    jac[3] = intensity * ex * dey_dsigma2       # d/dsigma2
    jac[4] = ex * ey                            # d/dintensity
    jac[5] = 1.0                                # d/doffset
    return jac


@njit(fastmath=True, cache=True)
def solve_6x6(A, b):
    """Solve 6x6 linear system via Gaussian elimination with partial pivoting."""
    n = 6
    x = np.zeros(n)
    for k in range(n):
        max_val = abs(A[k, k])
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i
        if max_val < 1e-30:
            return x
        if max_row != k:
            for j in range(k, n):
                A[k, j], A[max_row, j] = A[max_row, j], A[k, j]
            b[k], b[max_row] = b[max_row], b[k]
        pivot = A[k, k]
        for i in range(k + 1, n):
            factor = A[i, k] / pivot
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]
            A[i, k] = 0.0
            b[i] -= factor * b[k]
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-30:
            x[i] = 0.0
        else:
            s = b[i]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x[i] = s / A[i, i]
    return x


@njit(parallel=True, fastmath=True, cache=True)
def fit_elliptical_gaussian_mle_batch(image, positions, fit_radius=3,
                                       initial_sigma=1.3, max_iterations=1000):
    """Fit elliptical integrated Gaussian PSF using MLE with LM.

    6 parameters: [x0, y0, sigma1, sigma2, intensity, offset].
    Squared parameterization for sigma1, sigma2, intensity, offset.

    Returns
    -------
    results : ndarray (N, 9)
        [x, y, intensity, sigma1, sigma2, background, neg_ll, success, 0]
    """
    n_fits = len(positions)
    results = np.zeros((n_fits, 9))

    for i in prange(n_fits):
        row = int(positions[i, 0])
        col = int(positions[i, 1])
        r0 = max(0, row - fit_radius)
        r1 = min(image.shape[0], row + fit_radius + 1)
        c0 = max(0, col - fit_radius)
        c1 = min(image.shape[1], col + fit_radius + 1)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue

        roi = image[r0:r1, c0:c1]
        ny = roi.shape[0]
        nx = roi.shape[1]
        n_pixels = ny * nx

        offset = max(np.min(roi), 0.1)
        peak = np.max(roi) - offset
        x0 = float(col - c0)
        y0 = float(row - r0)
        sigma1 = initial_sigma
        sigma2 = initial_sigma
        intensity = max(peak * 2.0 * math.pi * sigma1 * sigma2, 1.0)

        # Squared parameterization for positivity
        p_s1 = math.sqrt(sigma1)
        p_s2 = math.sqrt(sigma2)
        p_I = math.sqrt(intensity)
        p_bg = math.sqrt(offset)

        lambda_lm = 1.0

        neg_ll = 0.0
        for ii in range(ny):
            for jj in range(nx):
                mv = integrated_elliptical_gaussian_value(float(jj), float(ii), x0, y0, sigma1, sigma2, intensity, offset)
                mv = max(mv, 1e-10)
                neg_ll += mv - roi[ii, jj] * math.log(mv)

        for iteration in range(max_iterations):
            H = np.zeros((6, 6))
            g = np.zeros(6)
            for ii in range(ny):
                for jj in range(nx):
                    fj = float(jj)
                    fi = float(ii)
                    mv = integrated_elliptical_gaussian_value(fj, fi, x0, y0, sigma1, sigma2, intensity, offset)
                    mv = max(mv, 1e-10)
                    data = roi[ii, jj]
                    J_orig = integrated_elliptical_gaussian_jacobian(fj, fi, x0, y0, sigma1, sigma2, intensity, offset)
                    # Transform: params 2,3,4,5 are squared
                    J = np.zeros(6)
                    J[0] = J_orig[0]
                    J[1] = J_orig[1]
                    J[2] = J_orig[2] * 2.0 * p_s1
                    J[3] = J_orig[3] * 2.0 * p_s2
                    J[4] = J_orig[4] * 2.0 * p_I
                    J[5] = J_orig[5] * 2.0 * p_bg
                    w = 1.0 / mv
                    factor = data / mv - 1.0
                    for p in range(6):
                        g[p] += factor * J[p]
                        for q in range(6):
                            H[p][q] += w * J[p] * J[q]

            H_aug = H.copy()
            for p in range(6):
                H_aug[p][p] *= (1.0 + lambda_lm)
            dp = solve_6x6(H_aug, g.copy())

            x0_n = x0 + dp[0]
            y0_n = y0 + dp[1]
            ps1_n = p_s1 + dp[2]
            ps2_n = p_s2 + dp[3]
            pI_n = p_I + dp[4]
            pbg_n = p_bg + dp[5]

            s1_n = min(ps1_n * ps1_n, 10.0)
            s2_n = min(ps2_n * ps2_n, 10.0)
            I_n = pI_n * pI_n
            bg_n = max(pbg_n * pbg_n, 1e-10)
            if s1_n < 0.1:
                s1_n = 0.1
                ps1_n = math.sqrt(s1_n)
            if s2_n < 0.1:
                s2_n = 0.1
                ps2_n = math.sqrt(s2_n)

            neg_ll_n = 0.0
            for ii in range(ny):
                for jj in range(nx):
                    mv = integrated_elliptical_gaussian_value(float(jj), float(ii), x0_n, y0_n, s1_n, s2_n, I_n, bg_n)
                    mv = max(mv, 1e-10)
                    neg_ll_n += mv - roi[ii, jj] * math.log(mv)

            if neg_ll_n < neg_ll:
                rel = abs(neg_ll - neg_ll_n) / max(abs(neg_ll), 1e-30)
                x0, y0 = x0_n, y0_n
                p_s1, p_s2, p_I, p_bg = ps1_n, ps2_n, pI_n, pbg_n
                sigma1, sigma2, intensity, offset = s1_n, s2_n, I_n, bg_n
                neg_ll = neg_ll_n
                lambda_lm = max(lambda_lm / 10.0, 1e-7)
                if rel < 1e-8:
                    break
            else:
                lambda_lm *= 10.0
                if lambda_lm > 1e10:
                    break

        results[i, 0] = x0 + c0
        results[i, 1] = y0 + r0
        results[i, 2] = intensity
        results[i, 3] = sigma1
        results[i, 4] = sigma2
        results[i, 5] = offset
        results[i, 6] = neg_ll
        results[i, 7] = 1.0

    return results


class EllipticalGaussianMLEFitter(BaseFitter):
    """Elliptical Gaussian PSF fitting using MLE with Levenberg-Marquardt.

    Fits independent sigma_x and sigma_y, enabling astigmatism-based 3D
    localization when combined with a defocus calibration curve.

    Parameters
    ----------
    initial_sigma : float
        Initial guess for sigma (used for both axes).
    """

    def __init__(self, initial_sigma=1.3):
        super().__init__("EllipticalGaussianMLE")
        self.initial_sigma = initial_sigma
        if NUMBA_AVAILABLE:
            self.name += " (Numba)"

    def fit(self, image, positions, fit_radius=3):
        if len(positions) == 0:
            return FitResult(
                x=np.array([]), y=np.array([]), intensity=np.array([]),
                background=np.array([]), sigma_x=np.array([]),
                sigma_y=np.array([]), chi_squared=np.array([]),
                uncertainty=np.array([]), success=np.array([])
            )
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        results = fit_elliptical_gaussian_mle_batch(
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
                sigma_eff = np.sqrt(sigma_x[i] * sigma_y[i])
                uncertainty[i] = compute_localization_precision(
                    intensity[i], background[i], sigma_eff,
                    1.0, fitting_method='mle', is_emccd=self.is_emccd
                )

        result = FitResult(
            x=x[mask], y=y[mask], intensity=intensity[mask],
            background=background[mask], sigma_x=sigma_x[mask],
            sigma_y=sigma_y[mask], chi_squared=chi_squared[mask],
            uncertainty=uncertainty[mask], success=success[mask]
        )
        return self._apply_sigma_filter(result)


# ============================================================================
# MULTI-EMITTER FITTING (MFA)
# ============================================================================

class MultiEmitterFitter(BaseFitter):
    """Multi-emitter analysis: fits 1..N overlapping PSFs per region.

    Uses chi-squared log-likelihood ratio test for model selection:
    the fit starts with N=1 and increases the number of emitters until
    the improvement is not statistically significant.

    Performance optimizations:
    - L-BFGS-B optimizer (gradient-based, ~3-5x faster than Nelder-Mead)
    - Fast N=1 pre-check: skips MFA when single emitter fit is good
    - Pre-computed chi2 critical values (avoids per-detection scipy call)
    - Cached pixel grids when ROI size is constant
    - Numba JIT NLL when available (additional ~3-5x speedup)

    Parameters
    ----------
    base_fitter_type : str
        Underlying fitter ('gaussian_mle' or 'gaussian_lsq').
    initial_sigma : float
        Initial PSF sigma guess.
    max_emitters : int
        Maximum number of emitters per region.
    p_value_threshold : float
        P-value threshold for accepting more complex model (default 1e-6,
        matching ImageJ ThunderSTORM).
    """

    def __init__(self, base_fitter_type='gaussian_mle', initial_sigma=1.3,
                 max_emitters=5, p_value_threshold=1e-6,
                 keep_same_intensity=True, fixed_intensity=False,
                 intensity_range=(500, 2500),
                 mfa_fitting_method='wlsq',
                 mfa_model_selection_iterations=50,
                 mfa_enable_final_refit=True):
        super().__init__("MultiEmitterMLE")
        self.initial_sigma = initial_sigma
        self.max_emitters = max_emitters
        self.p_value_threshold = p_value_threshold
        self.base_fitter_type = base_fitter_type
        self.keep_same_intensity = keep_same_intensity
        self.fixed_intensity = fixed_intensity
        self.intensity_range = intensity_range

        # MFA fitting method: 'wlsq' (default, matches ImageJ) or 'mle'
        # MLE has higher statistical power which causes over-splitting.
        self.mfa_fitting_method = mfa_fitting_method.lower()

        # Model selection iterations: ImageJ uses 50 during model comparison,
        # then does a final unlimited re-fit of the winning model.
        self.mfa_model_selection_iterations = mfa_model_selection_iterations

        # Final re-fit: after model selection, re-fit the winning model with
        # no iteration limit for better parameter estimates.
        self.mfa_enable_final_refit = mfa_enable_final_refit

        # Pre-compute F-test critical values for model comparison
        from scipy.stats import f as f_dist
        self._f_critical = {}
        # For a 7x7 ROI = 49 pixels. Cache for common ROI sizes.
        # ImageJ DoF: base=5, sameI → each extra emitter adds 3 params (x,y,sigma)
        # Not sameI → each extra emitter adds 4 params (x,y,sigma,intensity)
        std_n_pixels = (2 * 3 + 1) ** 2  # default fit_radius=3 → 49 pixels
        params_per_extra = 3 if keep_same_intensity else 4
        for n_em in range(2, max_emitters + 1):
            # DoF for model N: 5 + (n-1)*params_per_extra if sameI,
            # else 4*n + 1
            if keep_same_intensity:
                p_prev = 3 * (n_em - 1) + 2
                p_full = 3 * n_em + 2
            else:
                p_prev = 4 * (n_em - 1) + 1
                p_full = 4 * n_em + 1
            delta_p = p_full - p_prev  # = params_per_extra
            dof2 = std_n_pixels - p_full
            if dof2 > 0:
                self._f_critical[(delta_p, dof2)] = f_dist.ppf(
                    1.0 - p_value_threshold, delta_p, dof2)

        # Grid cache for constant ROI sizes
        self._grid_cache = {}

    def _get_pixel_grids(self, ny, nx):
        """Get pixel coordinate grids, using cache for repeated sizes."""
        key = (ny, nx)
        if key not in self._grid_cache:
            jj = np.arange(nx, dtype=np.float64)
            ii = np.arange(ny, dtype=np.float64)
            grid_jj, grid_ii = np.meshgrid(jj, ii)
            self._grid_cache[key] = (grid_ii.ravel(), grid_jj.ravel())
        return self._grid_cache[key]

    @staticmethod
    def _nll_vectorized(params, roi_flat, grid_ii, grid_jj, n_emitters):
        """Vectorized Poisson negative log-likelihood for N emitters.

        All pixel operations are numpy array ops — no Python loops over pixels.
        """
        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)
        sqrt2 = math.sqrt(2.0)

        for k in range(n_emitters):
            base = k * 4
            x0 = params[base]
            y0 = params[base + 1]
            sig = max(abs(params[base + 2]), 0.1)
            inten = max(params[base + 3], 1e-10)
            s2 = sqrt2 * sig
            ex = 0.5 * (erf((grid_jj - x0 + 0.5) / s2) - erf((grid_jj - x0 - 0.5) / s2))
            ey = 0.5 * (erf((grid_ii - y0 + 0.5) / s2) - erf((grid_ii - y0 - 0.5) / s2))
            model += inten * ex * ey

        np.clip(model, 1e-10, None, out=model)
        return np.sum(model - roi_flat * np.log(model))

    @staticmethod
    def _nll_gradient(params, roi_flat, grid_ii, grid_jj, n_emitters):
        """Analytical gradient of Poisson NLL for L-BFGS-B optimizer.

        Returns both NLL value and gradient vector for efficiency.
        """
        n_params = 4 * n_emitters + 1
        grad = np.zeros(n_params)

        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)
        sqrt2 = math.sqrt(2.0)
        two_over_sqrtpi = 2.0 / math.sqrt(math.pi)

        # Store per-emitter erf terms for gradient computation
        ex_all = []
        ey_all = []
        gx_p_all = []
        gx_m_all = []
        gy_p_all = []
        gy_m_all = []
        sig_all = []
        inten_all = []

        for k in range(n_emitters):
            base = k * 4
            x0 = params[base]
            y0 = params[base + 1]
            sig = max(abs(params[base + 2]), 0.1)
            inten = max(params[base + 3], 1e-10)
            s2 = sqrt2 * sig

            ux_p = (grid_jj - x0 + 0.5) / s2
            ux_m = (grid_jj - x0 - 0.5) / s2
            uy_p = (grid_ii - y0 + 0.5) / s2
            uy_m = (grid_ii - y0 - 0.5) / s2

            erf_xp = erf(ux_p)
            erf_xm = erf(ux_m)
            erf_yp = erf(uy_p)
            erf_ym = erf(uy_m)

            ex = 0.5 * (erf_xp - erf_xm)
            ey = 0.5 * (erf_yp - erf_ym)
            model += inten * ex * ey

            # Gaussian derivative terms
            gx_p = two_over_sqrtpi * np.exp(-ux_p * ux_p)
            gx_m = two_over_sqrtpi * np.exp(-ux_m * ux_m)
            gy_p = two_over_sqrtpi * np.exp(-uy_p * uy_p)
            gy_m = two_over_sqrtpi * np.exp(-uy_m * uy_m)

            ex_all.append(ex)
            ey_all.append(ey)
            gx_p_all.append(gx_p)
            gx_m_all.append(gx_m)
            gy_p_all.append(gy_p)
            gy_m_all.append(gy_m)
            sig_all.append(sig)
            inten_all.append(inten)

        np.clip(model, 1e-10, None, out=model)
        nll = np.sum(model - roi_flat * np.log(model))

        # Common factor: (1 - data/model) for each pixel
        ratio = 1.0 - roi_flat / model  # shape (n_pixels,)

        # Gradient w.r.t. offset
        grad[-1] = np.sum(ratio)

        # Gradient w.r.t. each emitter's parameters
        for k in range(n_emitters):
            base = k * 4
            ex = ex_all[k]
            ey = ey_all[k]
            sig = sig_all[k]
            inten = inten_all[k]
            inv_s2 = 1.0 / (sqrt2 * sig)

            dex_dx0 = 0.5 * inv_s2 * (gx_m_all[k] - gx_p_all[k])
            dey_dy0 = 0.5 * inv_s2 * (gy_m_all[k] - gy_p_all[k])

            # d(ex)/dsigma
            ux_p = (grid_jj - params[base] + 0.5) / (sqrt2 * sig)
            ux_m = (grid_jj - params[base] - 0.5) / (sqrt2 * sig)
            uy_p = (grid_ii - params[base + 1] + 0.5) / (sqrt2 * sig)
            uy_m = (grid_ii - params[base + 1] - 0.5) / (sqrt2 * sig)
            dex_dsig = 0.5 * (-ux_p * gx_p_all[k] + ux_m * gx_m_all[k]) / sig
            dey_dsig = 0.5 * (-uy_p * gy_p_all[k] + uy_m * gy_m_all[k]) / sig

            # Chain rule: d(nll)/d(param) = sum( ratio * d(model)/d(param) )
            grad[base] = np.sum(ratio * inten * dex_dx0 * ey)      # x0
            grad[base + 1] = np.sum(ratio * inten * ex * dey_dy0)  # y0
            grad[base + 2] = np.sum(ratio * inten * (dex_dsig * ey + ex * dey_dsig))  # sigma
            grad[base + 3] = np.sum(ratio * ex * ey)               # intensity

        return nll, grad

    @staticmethod
    def _wlsq_vectorized(params, roi_flat, grid_ii, grid_jj, n_emitters):
        """Weighted least squares objective matching ImageJ ThunderSTORM.

        WLSQ = sum((data - model)^2 * weight), where weight = 1/data.
        ImageJ uses data-based weights with min/max clamping.
        """
        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)
        sqrt2 = math.sqrt(2.0)

        for k in range(n_emitters):
            base = k * 4
            x0 = params[base]
            y0 = params[base + 1]
            sig = max(abs(params[base + 2]), 0.1)
            inten = max(params[base + 3], 1e-10)
            s2 = sqrt2 * sig
            ex = 0.5 * (erf((grid_jj - x0 + 0.5) / s2) - erf((grid_jj - x0 - 0.5) / s2))
            ey = 0.5 * (erf((grid_ii - y0 + 0.5) / s2) - erf((grid_ii - y0 - 0.5) / s2))
            model += inten * ex * ey

        np.clip(model, 1e-10, None, out=model)
        # Weight by 1/data (matching ImageJ), with clamping
        min_weight = 1.0 / max(np.max(roi_flat), 1e-10)
        max_weight = 1000.0 * min_weight
        weights = 1.0 / np.clip(roi_flat, 1e-10, None)
        weights = np.clip(weights, min_weight, max_weight)
        return float(np.sum((roi_flat - model) ** 2 * weights))

    @staticmethod
    def _wlsq_gradient(params, roi_flat, grid_ii, grid_jj, n_emitters):
        """Analytical gradient of WLSQ for L-BFGS-B, matching ImageJ.

        WLSQ = sum((d - m)^2 * w), where w = 1/d (data-based weights).
        d(WLSQ)/d(param) = sum( -2 * (d - m) * w * dm/dparam )

        Returns both WLSQ value and gradient vector.
        """
        n_params = 4 * n_emitters + 1
        grad = np.zeros(n_params)

        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)
        sqrt2 = math.sqrt(2.0)
        two_over_sqrtpi = 2.0 / math.sqrt(math.pi)

        ex_all = []
        ey_all = []
        gx_p_all = []
        gx_m_all = []
        gy_p_all = []
        gy_m_all = []
        sig_all = []
        inten_all = []

        for k in range(n_emitters):
            base = k * 4
            x0 = params[base]
            y0 = params[base + 1]
            sig = max(abs(params[base + 2]), 0.1)
            inten = max(params[base + 3], 1e-10)
            s2 = sqrt2 * sig

            ux_p = (grid_jj - x0 + 0.5) / s2
            ux_m = (grid_jj - x0 - 0.5) / s2
            uy_p = (grid_ii - y0 + 0.5) / s2
            uy_m = (grid_ii - y0 - 0.5) / s2

            erf_xp = erf(ux_p)
            erf_xm = erf(ux_m)
            erf_yp = erf(uy_p)
            erf_ym = erf(uy_m)

            ex = 0.5 * (erf_xp - erf_xm)
            ey = 0.5 * (erf_yp - erf_ym)
            model += inten * ex * ey

            gx_p = two_over_sqrtpi * np.exp(-ux_p * ux_p)
            gx_m = two_over_sqrtpi * np.exp(-ux_m * ux_m)
            gy_p = two_over_sqrtpi * np.exp(-uy_p * uy_p)
            gy_m = two_over_sqrtpi * np.exp(-uy_m * uy_m)

            ex_all.append(ex)
            ey_all.append(ey)
            gx_p_all.append(gx_p)
            gx_m_all.append(gx_m)
            gy_p_all.append(gy_p)
            gy_m_all.append(gy_m)
            sig_all.append(sig)
            inten_all.append(inten)

        np.clip(model, 1e-10, None, out=model)

        # Compute data-based weights (matching ImageJ)
        min_weight = 1.0 / max(np.max(roi_flat), 1e-10)
        max_weight = 1000.0 * min_weight
        weights = 1.0 / np.clip(roi_flat, 1e-10, None)
        weights = np.clip(weights, min_weight, max_weight)

        residual = roi_flat - model
        wlsq = float(np.sum(residual ** 2 * weights))

        # Common gradient factor: -2 * (data - model) * weight
        common = -2.0 * residual * weights

        # Gradient w.r.t. offset
        grad[-1] = np.sum(common)

        for k in range(n_emitters):
            base = k * 4
            ex = ex_all[k]
            ey = ey_all[k]
            sig = sig_all[k]
            inten = inten_all[k]
            inv_s2 = 1.0 / (sqrt2 * sig)

            dex_dx0 = 0.5 * inv_s2 * (gx_m_all[k] - gx_p_all[k])
            dey_dy0 = 0.5 * inv_s2 * (gy_m_all[k] - gy_p_all[k])

            ux_p = (grid_jj - params[base] + 0.5) / (sqrt2 * sig)
            ux_m = (grid_jj - params[base] - 0.5) / (sqrt2 * sig)
            uy_p = (grid_ii - params[base + 1] + 0.5) / (sqrt2 * sig)
            uy_m = (grid_ii - params[base + 1] - 0.5) / (sqrt2 * sig)
            dex_dsig = 0.5 * (-ux_p * gx_p_all[k] + ux_m * gx_m_all[k]) / sig
            dey_dsig = 0.5 * (-uy_p * gy_p_all[k] + uy_m * gy_m_all[k]) / sig

            grad[base] = np.sum(common * inten * dex_dx0 * ey)
            grad[base + 1] = np.sum(common * inten * ex * dey_dy0)
            grad[base + 2] = np.sum(common * inten * (dex_dsig * ey + ex * dey_dsig))
            grad[base + 3] = np.sum(common * ex * ey)

        return wlsq, grad

    def _fit_n_emitters(self, roi, n_emitters, grid_ii, grid_jj,
                        initial_positions=None, maxiter=50,
                        p0_override=None):
        """Fit N emitters in a single ROI using MLE or WLSQ with L-BFGS-B.

        The fitting method is controlled by self.mfa_fitting_method:
        - 'wlsq': Weighted least squares (Pearson chi-squared), matches ImageJ
        - 'mle': Maximum likelihood estimation (Poisson NLL)

        Parameters
        ----------
        maxiter : int
            Maximum optimizer iterations. ImageJ uses 50 for model selection
            and unlimited for the final re-fit of the winning model.
        p0_override : ndarray, optional
            If provided, use as initial parameters instead of computing from
            initial_positions. Used for the final re-fit.

        Returns (params_array, objective_value, n_params, rss).
        """
        from scipy.optimize import minimize

        ny, nx = roi.shape
        roi_flat = roi.ravel().astype(np.float64)

        # Total params: 4*N + 1 (shared offset)
        n_params = 4 * n_emitters + 1

        # Intensity bounds: if fixed_intensity is enabled, constrain range
        if self.fixed_intensity and self.intensity_range:
            i_lo, i_hi = float(self.intensity_range[0]), float(self.intensity_range[1])
        else:
            i_lo, i_hi = 1.0, None  # unconstrained upper bound

        # Build bounds (always needed)
        bounds = []
        for k in range(n_emitters):
            bounds.extend([
                (-1.0, nx + 1.0),           # x0
                (-1.0, ny + 1.0),           # y0
                (0.3, 5.0 * self.initial_sigma),  # sigma
                (i_lo, i_hi),               # intensity
            ])
        bounds.append((0.01, None))  # offset

        # Initial guesses
        if p0_override is not None:
            p0 = p0_override.copy()
        else:
            bg_init = max(float(np.percentile(roi, 10)), 0.1)
            peak = float(np.max(roi)) - bg_init
            p0 = []
            if initial_positions is not None and len(initial_positions) >= n_emitters:
                for k in range(n_emitters):
                    x_init = initial_positions[k][1]  # col
                    y_init = initial_positions[k][0]  # row
                    p0.extend([x_init, y_init, self.initial_sigma,
                               max(peak / n_emitters, 1.0)])
            else:
                for k in range(n_emitters):
                    p0.extend([
                        nx / 2.0 + (k - n_emitters / 2.0) * 1.5,
                        ny / 2.0,
                        self.initial_sigma,
                        max(peak / n_emitters, 1.0)
                    ])
            p0.append(bg_init)
            p0 = np.array(p0, dtype=np.float64)

        # Select objective and gradient based on fitting method
        use_wlsq = (self.mfa_fitting_method == 'wlsq')

        # "Keep same intensity" constraint (matching ImageJ's fixParams):
        # Force all emitter intensities to equal the first emitter's intensity
        fix_intensity = (self.keep_same_intensity and n_emitters > 1)

        def _fix_params(params):
            """Enforce shared intensity across all emitters (ImageJ fixParams)."""
            if fix_intensity:
                I0 = params[3]  # first emitter's intensity
                for k in range(1, n_emitters):
                    params[k * 4 + 3] = I0
            return params

        # Use Numba-compiled gradient functions when available
        if NUMBA_AVAILABLE:
            if use_wlsq:
                def obj_and_grad(params):
                    params = _fix_params(params)
                    return _mfa_wlsq_gradient_numba(
                        params, roi_flat, grid_ii, grid_jj,
                        n_emitters, n_params)
            else:
                def obj_and_grad(params):
                    params = _fix_params(params)
                    return _mfa_nll_gradient_numba(
                        params, roi_flat, grid_ii, grid_jj,
                        n_emitters, n_params)
        else:
            # Fallback to scipy-based gradient (slower)
            if use_wlsq:
                def obj_and_grad(params):
                    params = _fix_params(params)
                    return self._wlsq_gradient(params, roi_flat, grid_ii,
                                               grid_jj, n_emitters)
            else:
                def obj_and_grad(params):
                    params = _fix_params(params)
                    return self._nll_gradient(params, roi_flat, grid_ii,
                                             grid_jj, n_emitters)

        # Degrees of freedom: with sameI, intensity counts as 1 param
        # (matching ImageJ's getDoF: baseDof + (n-1)*(baseDof - 2))
        # Base PSF has 5 params (x, y, sigma, intensity, offset).
        # With sameI: total = 5 + (n-1)*3 = 3n + 2
        if fix_intensity:
            effective_n_params = 3 * n_emitters + 2
        else:
            effective_n_params = n_params

        try:
            result = minimize(obj_and_grad, p0, method='L-BFGS-B',
                              jac=True, bounds=bounds,
                              options={'maxiter': maxiter, 'ftol': 1e-8,
                                       'gtol': 1e-5})
            final_params = _fix_params(result.x.copy())
            if NUMBA_AVAILABLE:
                rss = _mfa_compute_rss_numba(final_params, roi_flat,
                                              grid_ii, grid_jj, n_emitters)
            else:
                rss = self._compute_rss(final_params, roi_flat, grid_ii,
                                         grid_jj, n_emitters)
            return final_params, result.fun, effective_n_params, rss
        except Exception:
            final_params = _fix_params(p0.copy())
            val, _ = obj_and_grad(final_params)
            if NUMBA_AVAILABLE:
                rss = _mfa_compute_rss_numba(final_params, roi_flat,
                                              grid_ii, grid_jj, n_emitters)
            else:
                rss = self._compute_rss(final_params, roi_flat, grid_ii,
                                         grid_jj, n_emitters)
            return final_params, val, effective_n_params, rss

    @staticmethod
    def _compute_rss(params, roi_flat, grid_ii, grid_jj, n_emitters):
        """Compute weighted sum of squared residuals for F-test.

        Uses weights = 1/data (observed values), matching ImageJ ThunderSTORM's
        getChiSquared() method. This differs from Pearson chi-squared (1/model).
        """
        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)
        sqrt2 = math.sqrt(2.0)
        for k in range(n_emitters):
            base = k * 4
            x0, y0 = params[base], params[base + 1]
            sig = max(abs(params[base + 2]), 0.1)
            inten = max(params[base + 3], 1e-10)
            s2 = sqrt2 * sig
            ex = 0.5 * (erf((grid_jj - x0 + 0.5) / s2) - erf((grid_jj - x0 - 0.5) / s2))
            ey = 0.5 * (erf((grid_ii - y0 + 0.5) / s2) - erf((grid_ii - y0 - 0.5) / s2))
            model += inten * ex * ey
        np.clip(model, 1e-10, None, out=model)
        # Weight by 1/data (matching ImageJ), with clamping for stability
        min_weight = 1.0 / max(np.max(roi_flat), 1e-10)
        max_weight = 1000.0 * min_weight
        weights = 1.0 / np.clip(roi_flat, 1e-10, None)
        weights = np.clip(weights, min_weight, max_weight)
        return float(np.sum((roi_flat - model) ** 2 * weights))

    def _quick_goodness(self, roi, params, grid_ii, grid_jj):
        """Fast goodness-of-fit check for single-emitter pre-screening.

        Computes a normalized chi-squared-like statistic. For Poisson data,
        the Pearson chi-squared = sum((obs - model)^2 / model) should be
        approximately n_pixels if the model is good.

        Returns reduced chi-squared (should be ~1 for a good fit).
        """
        roi_flat = roi.ravel().astype(np.float64)
        offset = max(params[-1], 1e-10)
        model = np.full_like(roi_flat, offset)

        x0, y0 = params[0], params[1]
        sig = max(abs(params[2]), 0.1)
        inten = max(params[3], 1e-10)
        s2 = math.sqrt(2.0) * sig
        ex = 0.5 * (erf((grid_jj - x0 + 0.5) / s2) - erf((grid_jj - x0 - 0.5) / s2))
        ey = 0.5 * (erf((grid_ii - y0 + 0.5) / s2) - erf((grid_ii - y0 - 0.5) / s2))
        model += inten * ex * ey

        np.clip(model, 1e-10, None, out=model)
        pearson = np.sum((roi_flat - model) ** 2 / model)
        n_pixels = len(roi_flat)
        dof = n_pixels - 5  # 5 params for single emitter
        if dof <= 0:
            return 1.0
        return pearson / dof

    def fit(self, image, positions, fit_radius=3):
        """Run multi-emitter analysis on each detection.

        Matches ImageJ ThunderSTORM: always tries N=1 through N=maxN,
        using F-test with data-weighted chi-squared for model selection.
        """
        all_x, all_y, all_intensity, all_bg = [], [], [], []
        all_sx, all_sy, all_chi2, all_unc, all_success = [], [], [], [], []

        # Pre-compute the standard ROI pixel grids (most ROIs have the same size)
        std_size = 2 * fit_radius + 1
        grid_ii, grid_jj = self._get_pixel_grids(std_size, std_size)

        # Boundary exclusion: reject emitters too close to ROI edge
        # ImageJ uses: maxX = size/2 - sigma/2, same for maxY
        max_offset = (std_size / 2.0) - self.initial_sigma / 2.0

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue

            roi = image[r0:r1, c0:c1].astype(np.float64)
            ny, nx = roi.shape
            n_pixels = ny * nx

            # Use cached grids when ROI matches standard size
            if ny == std_size and nx == std_size:
                gi, gj = grid_ii, grid_jj
            else:
                gi, gj = self._get_pixel_grids(ny, nx)

            # Fit N=1 (model selection phase uses limited iterations)
            ms_iters = self.mfa_model_selection_iterations
            init_pos = [np.array([row - r0, col - c0])]
            best_params, best_nll, best_n_params, best_rss = \
                self._fit_n_emitters(
                    roi, 1, gi, gj, initial_positions=init_pos,
                    maxiter=ms_iters)
            best_n = 1

            # Try increasing N (matching ImageJ: always try all N)
            if self.max_emitters > 1:
                prev_params = best_params
                for n_em in range(2, self.max_emitters + 1):
                    # Build initial positions: previous results + residual peak
                    init_pos_n = []
                    for k in range(n_em - 1):
                        base = k * 4
                        init_pos_n.append(np.array([prev_params[base + 1],
                                                    prev_params[base]]))
                    # Build model image from previous fit to find residual peak
                    model_img = np.full((ny, nx), max(prev_params[-1], 0.1))
                    sqrt2 = math.sqrt(2.0)
                    jj_1d = np.arange(nx, dtype=np.float64)
                    ii_1d = np.arange(ny, dtype=np.float64)
                    for k in range(n_em - 1):
                        base = k * 4
                        x0, y0 = prev_params[base], prev_params[base + 1]
                        sig = max(abs(prev_params[base + 2]), 0.1)
                        inten = max(prev_params[base + 3], 0.0)
                        s2 = sqrt2 * sig
                        ex = 0.5 * (erf((jj_1d - x0 + 0.5) / s2) -
                                    erf((jj_1d - x0 - 0.5) / s2))
                        ey = 0.5 * (erf((ii_1d - y0 + 0.5) / s2) -
                                    erf((ii_1d - y0 - 0.5) / s2))
                        model_img += inten * np.outer(ey, ex)
                    residual = roi - model_img
                    peak_idx = np.unravel_index(np.argmax(residual), (ny, nx))
                    init_pos_n.append(np.array([float(peak_idx[0]),
                                                float(peak_idx[1])]))

                    params_n, nll_n, n_params_n, rss_n = \
                        self._fit_n_emitters(
                            roi, n_em, gi, gj,
                            initial_positions=init_pos_n)

                    # F-test for model comparison (matching ImageJ ThunderSTORM)
                    # ImageJ computes: pValue = 1 - F.cdf(F_stat, delta_p, dof2)
                    # and accepts if pValue < pValueThr
                    delta_p = n_params_n - best_n_params  # = 4
                    dof2 = n_pixels - n_params_n
                    if dof2 > 0 and best_rss > rss_n and delta_p > 0:
                        f_stat = ((best_rss - rss_n) / delta_p) / \
                                 (rss_n / dof2)
                        # Look up or compute critical value
                        key = (delta_p, dof2)
                        if key in self._f_critical:
                            f_crit = self._f_critical[key]
                        else:
                            from scipy.stats import f as f_dist
                            f_crit = f_dist.ppf(
                                1.0 - self.p_value_threshold,
                                delta_p, dof2)
                            self._f_critical[key] = f_crit
                        if f_stat > f_crit:
                            best_params = params_n
                            best_nll = nll_n
                            best_n_params = n_params_n
                            best_rss = rss_n
                            best_n = n_em
                            prev_params = params_n
                        else:
                            break
                    else:
                        break

            # Final re-fit of winning model with no iteration limit
            # (matching ImageJ: model selection uses limited iters, final fit unlimited)
            if self.mfa_enable_final_refit:
                best_params, best_nll, _, _ = self._fit_n_emitters(
                    roi, best_n, gi, gj,
                    maxiter=1000, p0_override=best_params)

            # Extract results for each emitter in winning model
            offset = max(best_params[-1], 0.0)

            # Boundary exclusion: reject emitters too far from ROI center
            # (matching ImageJ's eliminateBadFits)
            cx, cy = nx / 2.0, ny / 2.0
            for k in range(best_n):
                base = k * 4
                x0_local = best_params[base]
                y0_local = best_params[base + 1]
                # ImageJ checks abs(x - center) <= maxOffset
                if abs(x0_local - cx) > max_offset or abs(y0_local - cy) > max_offset:
                    continue

                x0 = x0_local + c0
                y0 = y0_local + r0
                sigma = max(abs(best_params[base + 2]), 0.1)
                inten = max(best_params[base + 3], 0.0)

                all_x.append(x0)
                all_y.append(y0)
                all_intensity.append(inten)
                all_bg.append(offset)
                all_sx.append(sigma)
                all_sy.append(sigma)
                all_chi2.append(best_nll)
                all_unc.append(compute_localization_precision(
                    inten, offset, sigma, 1.0,
                    fitting_method=self.mfa_fitting_method,
                    is_emccd=self.is_emccd
                ))
                all_success.append(1.0)

        return FitResult(
            x=np.array(all_x), y=np.array(all_y),
            intensity=np.array(all_intensity),
            background=np.array(all_bg),
            sigma_x=np.array(all_sx), sigma_y=np.array(all_sy),
            chi_squared=np.array(all_chi2),
            uncertainty=np.array(all_unc),
            success=np.array(all_success)
        )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_fitter(fitter_type, **kwargs):
    """Factory function to create fitters

    Parameters
    ----------
    fitter_type : str
        Type: 'gaussian_lsq', 'gaussian_wlsq', 'gaussian_mle',
        'elliptical_gaussian_mle', 'multi_emitter',
        'radial_symmetry', 'phasor', 'centroid'
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
        'elliptical_gaussian_mle': EllipticalGaussianMLEFitter,
        'multi_emitter': MultiEmitterFitter,
        'centroid': CentroidFitter,
        'radial_symmetry': RadialSymmetryFitter,
        'phasor': PhasorFitter,
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
