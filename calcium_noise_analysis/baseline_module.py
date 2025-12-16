"""
Baseline Estimation Module
===========================
F₀ baseline estimation and ΔF/F calculation for calcium imaging analysis.

Includes multiple baseline estimation methods:
- Sliding percentile (robust to transients)
- Maximin filter (Suite2p method)
- Photobleaching correction (exponential fit)

Author: George
"""

import numpy as np
from scipy.ndimage import percentile_filter, minimum_filter, maximum_filter
from scipy.optimize import curve_fit


def estimate_f0_percentile(trace, window=1000, percentile=8):
    """
    Sliding percentile baseline estimation—robust to transients.
    
    Uses a sliding window to compute the specified percentile at each timepoint.
    The 8th percentile over 15-30s windows is standard for calcium imaging.
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        window (int): Window size in frames (typically 15-30s worth)
        percentile (int): Percentile value (0-100), typical: 8
    
    Returns:
        ndarray: Estimated baseline F₀ (same length as input)
    
    Example:
        >>> import numpy as np
        >>> # Create trace with baseline and transients
        >>> t = np.arange(1000)
        >>> baseline = 100 + 0.01 * t  # Slow drift
        >>> transients = 50 * (np.random.rand(1000) > 0.95)
        >>> trace = baseline + transients
        >>> f0 = estimate_f0_percentile(trace, window=300)
        >>> print(f"Baseline follows trend: {np.corrcoef(baseline, f0)[0,1] > 0.9}")
    
    Notes:
        - Robust to brief calcium transients (won't pull baseline up)
        - Tracks slow baseline drift and photobleaching
        - Window size should be longer than typical transient duration
    """
    # Apply percentile filter with specified window and percentile
    f0 = percentile_filter(trace, percentile, size=window, mode='nearest')
    return f0


def estimate_f0_maximin(trace, window=300):
    """
    Suite2p maximin method: maximum of sliding minimum.
    
    Computes sliding minimum, then sliding maximum of that result.
    This provides a fast, robust baseline estimate that follows slow drift.
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        window (int): Window size in frames
    
    Returns:
        ndarray: Estimated baseline F₀ (same length as input)
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000) * 10 + 100
        >>> f0 = estimate_f0_maximin(trace, window=300)
        >>> print(f"F₀ tracks baseline: {abs(f0.mean() - 100) < 5}")
    
    Notes:
        - Fast computation (just two 1D filters)
        - Less sensitive to parameter choice than percentile
        - Default method in Suite2p pipeline
    """
    # First pass: sliding minimum
    min_filtered = minimum_filter(trace, size=window, mode='nearest')
    
    # Second pass: sliding maximum of the minimum
    f0 = maximum_filter(min_filtered, size=window, mode='nearest')
    
    return f0


def photobleaching_baseline(trace, max_iterations=10000):
    """
    Double exponential fit for photobleaching correction.
    
    Fits a double exponential decay model to estimate the photobleaching trend:
    F(t) = a1*exp(-t/τ1) + a2*exp(-t/τ2) + c
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        max_iterations (int): Maximum iterations for curve fitting
    
    Returns:
        ndarray: Estimated photobleaching baseline
    
    Example:
        >>> import numpy as np
        >>> # Create trace with exponential decay
        >>> t = np.arange(1000)
        >>> bleach = 200 * np.exp(-t/500) + 100
        >>> trace = bleach + np.random.randn(1000) * 5
        >>> f0 = photobleaching_baseline(trace)
        >>> print(f"Baseline follows bleaching: {np.corrcoef(bleach, f0)[0,1] > 0.95}")
    
    Notes:
        - Best for long recordings with significant photobleaching
        - May fail to converge for short traces or traces without bleaching
        - Use try-except and fall back to percentile method if fitting fails
    """
    def double_exp(t, a1, tau1, a2, tau2, c):
        """Double exponential decay function."""
        return a1 * np.exp(-t/tau1) + a2 * np.exp(-t/tau2) + c
    
    t = np.arange(len(trace))
    
    # Initial parameter guesses
    p0 = [
        trace[0] * 0.5,  # a1
        len(trace) / 2,   # tau1
        trace[0] * 0.3,   # a2
        len(trace) / 10,  # tau2
        trace[-1]         # c (final baseline)
    ]
    
    # Fit double exponential
    try:
        popt, _ = curve_fit(
            double_exp, t, trace, 
            p0=p0, 
            maxfev=max_iterations,
            bounds=([0, 1, 0, 1, 0],  # Lower bounds
                   [np.inf, len(trace)*2, np.inf, len(trace)*2, np.inf])  # Upper bounds
        )
        baseline = double_exp(t, *popt)
    except (RuntimeError, ValueError) as e:
        # If fitting fails, fall back to percentile method
        print(f"Photobleaching fit failed: {e}, using percentile method")
        baseline = estimate_f0_percentile(trace, window=len(trace)//10)
    
    return baseline


def compute_dff(trace, f0, epsilon=1.0):
    """
    Calculate ΔF/F (normalized fluorescence change).
    
    Formula: ΔF/F = (F - F₀) / max(F₀, ε)
    
    The epsilon parameter prevents division artifacts when F₀ is near zero.
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array)
        f0 (ndarray): Baseline fluorescence (same shape as trace)
        epsilon (float): Minimum baseline value to prevent division errors
    
    Returns:
        ndarray: ΔF/F values
    
    Example:
        >>> import numpy as np
        >>> trace = np.array([100, 120, 150, 130, 100])
        >>> f0 = np.array([100, 100, 100, 100, 100])
        >>> dff = compute_dff(trace, f0, epsilon=1.0)
        >>> print(dff)  # [0.0, 0.2, 0.5, 0.3, 0.0]
    
    Notes:
        - F₀ should be estimated using one of the baseline methods
        - Epsilon prevents division by zero for very dim pixels
        - Typical epsilon values: 0.1-10 depending on intensity scale
    """
    # Ensure F₀ is at least epsilon to avoid division by zero
    f0_safe = np.maximum(f0, epsilon)
    
    # Compute ΔF/F
    dff = (trace - f0) / f0_safe
    
    return dff


def compute_dff_stack(image_stack, method='percentile', window=1000, 
                      percentile=8, epsilon=1.0):
    """
    Compute ΔF/F for entire image stack (all pixels).
    
    Convenience function that estimates F₀ and computes ΔF/F for each pixel.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        method (str): Baseline estimation method ('percentile', 'maximin', 'photobleaching')
        window (int): Window size for percentile/maximin methods
        percentile (int): Percentile value for percentile method
        epsilon (float): Minimum baseline value
    
    Returns:
        ndarray: ΔF/F stack (T, H, W)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64) * 10 + 100
        >>> dff_stack = compute_dff_stack(stack, method='percentile')
        >>> print(dff_stack.shape)
        (500, 64, 64)
    """
    T, H, W = image_stack.shape
    dff_stack = np.zeros_like(image_stack, dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            trace = image_stack[:, i, j].astype(np.float64)
            
            # Estimate baseline
            if method == 'percentile':
                f0 = estimate_f0_percentile(trace, window=window, percentile=percentile)
            elif method == 'maximin':
                f0 = estimate_f0_maximin(trace, window=window)
            elif method == 'photobleaching':
                f0 = photobleaching_baseline(trace)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Compute ΔF/F
            dff_stack[:, i, j] = compute_dff(trace, f0, epsilon=epsilon)
    
    return dff_stack


def correct_photobleaching(trace, method='exponential'):
    """
    Correct for photobleaching using multiplicative correction.
    
    Fits photobleaching curve, then applies correction: F_corrected = F / baseline * baseline[0]
    This preserves the relative dynamics while removing the decay trend.
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        method (str): Correction method ('exponential' or 'linear')
    
    Returns:
        ndarray: Photobleaching-corrected trace
    
    Example:
        >>> import numpy as np
        >>> t = np.arange(1000)
        >>> bleach = np.exp(-t/500)
        >>> signal = 1 + 0.5 * np.sin(2 * np.pi * t / 100)
        >>> trace = 100 * bleach * signal
        >>> corrected = correct_photobleaching(trace)
        >>> # Signal preserved, bleaching removed
    """
    if method == 'exponential':
        baseline = photobleaching_baseline(trace)
    elif method == 'linear':
        # Simple linear detrending
        t = np.arange(len(trace))
        coeffs = np.polyfit(t, trace, deg=1)
        baseline = np.polyval(coeffs, t)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Multiplicative correction
    baseline_safe = np.maximum(baseline, 1e-10)
    corrected = (trace / baseline_safe) * baseline[0]
    
    return corrected


def estimate_f0_from_histogram(trace, percentile=10):
    """
    Estimate constant F₀ from histogram (global percentile).
    
    Simple method that computes a single F₀ value from the entire trace.
    Useful for short recordings where sliding window doesn't make sense.
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        percentile (int): Percentile value (0-100), typical: 5-10
    
    Returns:
        float: Estimated baseline F₀
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000) * 10 + 100
        >>> f0 = estimate_f0_from_histogram(trace, percentile=10)
        >>> print(f"F₀: {f0:.2f}, should be near 100")
    """
    f0 = np.percentile(trace, percentile)
    return f0


if __name__ == '__main__':
    """Unit tests for baseline module."""
    print("Testing baseline_module...")
    
    # Test 1: Percentile baseline
    print("\n1. Testing estimate_f0_percentile...")
    np.random.seed(42)
    t = np.arange(1000)
    baseline = 100 + 0.02 * t  # Slow drift
    transients = 50 * (np.random.rand(1000) > 0.97)  # Sparse transients
    trace = baseline + transients + np.random.randn(1000) * 2
    
    f0 = estimate_f0_percentile(trace, window=300, percentile=8)
    correlation = np.corrcoef(baseline, f0)[0, 1]
    print(f"   Baseline-F₀ correlation: {correlation:.3f}")
    print(f"   F₀ follows drift: {correlation > 0.95}")
    assert correlation > 0.95, "F₀ doesn't track baseline"
    assert np.mean(f0) < np.mean(trace), "F₀ should be below mean"
    print("   ✓ Percentile baseline working correctly")
    
    # Test 2: Maximin baseline
    print("\n2. Testing estimate_f0_maximin...")
    f0_maximin = estimate_f0_maximin(trace, window=300)
    print(f"   F₀ shape: {f0_maximin.shape}")
    print(f"   F₀ mean: {f0_maximin.mean():.2f}")
    print(f"   Trace mean: {trace.mean():.2f}")
    assert f0_maximin.shape == trace.shape, "Shape mismatch"
    assert f0_maximin.mean() < trace.mean(), "F₀ should be below mean"
    print("   ✓ Maximin baseline working correctly")
    
    # Test 3: Photobleaching baseline
    print("\n3. Testing photobleaching_baseline...")
    # Create trace with exponential decay
    bleach = 200 * np.exp(-t/400) + 50
    trace_bleach = bleach + transients + np.random.randn(1000) * 2
    
    f0_bleach = photobleaching_baseline(trace_bleach)
    correlation_bleach = np.corrcoef(bleach, f0_bleach)[0, 1]
    print(f"   Bleaching-F₀ correlation: {correlation_bleach:.3f}")
    print(f"   F₀ follows bleaching: {correlation_bleach > 0.90}")
    assert correlation_bleach > 0.90, "F₀ doesn't track photobleaching"
    print("   ✓ Photobleaching baseline working correctly")
    
    # Test 4: ΔF/F computation
    print("\n4. Testing compute_dff...")
    trace_simple = np.array([100, 120, 150, 130, 100], dtype=np.float64)
    f0_simple = np.array([100, 100, 100, 100, 100], dtype=np.float64)
    dff = compute_dff(trace_simple, f0_simple, epsilon=1.0)
    expected = np.array([0.0, 0.2, 0.5, 0.3, 0.0])
    print(f"   ΔF/F: {dff}")
    print(f"   Expected: {expected}")
    assert np.allclose(dff, expected), "ΔF/F calculation incorrect"
    print("   ✓ ΔF/F computation working correctly")
    
    # Test 5: ΔF/F stack computation
    print("\n5. Testing compute_dff_stack...")
    T, H, W = 200, 16, 16
    stack = np.random.randn(T, H, W) * 10 + 100
    
    dff_stack = compute_dff_stack(stack, method='percentile', window=100)
    print(f"   Input shape: {stack.shape}")
    print(f"   Output shape: {dff_stack.shape}")
    print(f"   Output mean: {dff_stack.mean():.4f} (should be near 0)")
    assert dff_stack.shape == stack.shape, "Shape mismatch"
    assert abs(dff_stack.mean()) < 0.1, "ΔF/F mean should be near 0"
    print("   ✓ ΔF/F stack computation working correctly")
    
    # Test 6: Photobleaching correction
    print("\n6. Testing correct_photobleaching...")
    bleach_curve = np.exp(-t/500)
    signal = 1 + 0.5 * np.sin(2 * np.pi * t / 100)
    trace_pb = 100 * bleach_curve * signal
    
    corrected = correct_photobleaching(trace_pb)
    # Check that trend is removed (linear fit slope should be near 0)
    t_norm = (t - t.mean()) / t.std()
    slope = np.polyfit(t_norm, corrected, deg=1)[0]
    print(f"   Original trend slope: {np.polyfit(t_norm, trace_pb, deg=1)[0]:.2f}")
    print(f"   Corrected trend slope: {slope:.2f} (should be near 0)")
    assert abs(slope) < 5, "Photobleaching not corrected"
    print("   ✓ Photobleaching correction working correctly")
    
    # Test 7: Histogram baseline
    print("\n7. Testing estimate_f0_from_histogram...")
    trace_hist = np.random.randn(1000) * 10 + 100
    f0_hist = estimate_f0_from_histogram(trace_hist, percentile=10)
    print(f"   F₀ from histogram: {f0_hist:.2f}")
    print(f"   True baseline: 100")
    assert 90 < f0_hist < 110, "Histogram baseline incorrect"
    print("   ✓ Histogram baseline working correctly")
    
    print("\n✅ All baseline module tests passed!")
