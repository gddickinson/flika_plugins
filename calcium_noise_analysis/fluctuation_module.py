"""
Fluctuation Analysis Module
============================
FLIKA-style SD fluctuation analysis for detecting local Ca²⁺ signals.
Highlights transient local signals through running variance calculation
with shot noise correction.

Based on Lock & Parker 2020 methodology.

Author: George
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt


def fluctuation_analysis(image_stack, fs, spatial_sigma=1.0, 
                         highpass_cutoff=0.5, window_size=30, 
                         shot_noise_factor=None):
    """
    FLIKA-style SD fluctuation analysis for local Ca²⁺ signals.
    
    Algorithm:
    1. Apply spatial Gaussian blur to reduce pixel-level noise
    2. Apply temporal high-pass filter to remove baseline drift
    3. Compute windowed variance (running standard deviation)
    4. Optionally subtract predicted shot noise variance (proportional to mean intensity)
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency in Hz
        spatial_sigma (float): Sigma for Gaussian spatial smoothing
        highpass_cutoff (float): High-pass filter cutoff frequency in Hz
        window_size (int): Size of running variance window in frames
        shot_noise_factor (float): Factor for shot noise correction (None to disable)
                                   Typical value: ~0.001-0.01 depending on camera
    
    Returns:
        ndarray: Standard deviation map stack (T, H, W) showing fluctuations
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64) + 100
        >>> fs = 30.0
        >>> sd_stack = fluctuation_analysis(stack, fs, spatial_sigma=1.0)
        >>> print(sd_stack.shape)
        (500, 64, 64)
    
    Notes:
        - High-pass filtering removes slow baseline drift
        - Shot noise correction is optional but recommended for low-light imaging
        - Window size should be ~1 second worth of frames (e.g., 30 frames at 30 Hz)
    """
    T, H, W = image_stack.shape
    
    # Step 1: Spatial smoothing
    print("Applying spatial smoothing...")
    blurred = np.array([
        gaussian_filter(frame.astype(np.float64), spatial_sigma) 
        for frame in image_stack
    ])
    
    # Step 2: Temporal high-pass filter (remove baseline drift)
    print("Applying temporal high-pass filter...")
    nyquist = fs / 2.0
    cutoff_norm = highpass_cutoff / nyquist
    cutoff_norm = max(0.001, min(cutoff_norm, 0.999))
    
    b, a = butter(2, cutoff_norm, btype='high')
    
    # Apply filter to each pixel's time series
    filtered = np.zeros_like(blurred)
    for i in range(H):
        for j in range(W):
            filtered[:, i, j] = filtfilt(b, a, blurred[:, i, j])
    
    # Step 3: Running variance with shot noise correction
    print("Computing running variance...")
    half_win = window_size // 2
    var_stack = np.zeros_like(filtered)
    
    for t in range(half_win, T - half_win):
        window = filtered[t-half_win:t+half_win+1]
        var_stack[t] = np.var(window, axis=0)
        
        # Optional shot noise correction
        if shot_noise_factor is not None:
            mean_intensity = np.mean(blurred[t-half_win:t+half_win+1], axis=0)
            var_stack[t] -= shot_noise_factor * mean_intensity
            var_stack[t] = np.maximum(var_stack[t], 0)  # Ensure non-negative
    
    # Return standard deviation (square root of variance)
    return np.sqrt(var_stack)


def compute_local_variance_map(image_stack, window_size=30):
    """
    Compute local variance map (temporal variance for each pixel).
    
    Simplified version of fluctuation_analysis without filtering.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        window_size (int): Size of running variance window in frames
    
    Returns:
        ndarray: Variance map stack (T, H, W)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> var_map = compute_local_variance_map(stack, window_size=30)
        >>> print(var_map.shape)
        (500, 64, 64)
    """
    T, H, W = image_stack.shape
    half_win = window_size // 2
    var_stack = np.zeros_like(image_stack, dtype=np.float64)
    
    for t in range(half_win, T - half_win):
        window = image_stack[t-half_win:t+half_win+1]
        var_stack[t] = np.var(window, axis=0)
    
    return var_stack


def temporal_highpass_filter(image_stack, fs, cutoff=0.5, order=2):
    """
    Apply temporal high-pass filter to remove baseline drift.
    
    Filters each pixel's time series independently.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency in Hz
        cutoff (float): High-pass cutoff frequency in Hz
        order (int): Filter order
    
    Returns:
        ndarray: High-pass filtered image stack (T, H, W)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64) + 100
        >>> fs = 30.0
        >>> filtered = temporal_highpass_filter(stack, fs, cutoff=0.5)
        >>> print(filtered.shape)
        (500, 64, 64)
    """
    T, H, W = image_stack.shape
    
    # Design high-pass filter
    nyquist = fs / 2.0
    cutoff_norm = cutoff / nyquist
    cutoff_norm = max(0.001, min(cutoff_norm, 0.999))
    
    b, a = butter(order, cutoff_norm, btype='high')
    
    # Apply filter to each pixel
    filtered = np.zeros_like(image_stack, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            filtered[:, i, j] = filtfilt(b, a, image_stack[:, i, j].astype(np.float64))
    
    return filtered


def shot_noise_corrected_variance(image_stack, mean_stack, shot_noise_factor):
    """
    Compute variance with shot noise correction.
    
    Shot noise variance is proportional to mean intensity (Poisson statistics).
    Subtracts predicted shot noise variance from observed variance.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        mean_stack (ndarray): Mean intensity stack (T, H, W) or (H, W)
        shot_noise_factor (float): Proportionality factor for shot noise
    
    Returns:
        ndarray: Shot-noise corrected variance map
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.poisson(100, (500, 64, 64))
        >>> mean = np.mean(stack, axis=0)
        >>> var_corrected = shot_noise_corrected_variance(stack, mean, shot_noise_factor=0.01)
        >>> print(var_corrected.shape)
        (64, 64)
    """
    # Compute variance
    variance = np.var(image_stack, axis=0)
    
    # Compute mean if not provided
    if mean_stack.ndim == 2:
        mean_intensity = mean_stack
    else:
        mean_intensity = np.mean(mean_stack, axis=0)
    
    # Subtract shot noise variance
    corrected_variance = variance - shot_noise_factor * mean_intensity
    corrected_variance = np.maximum(corrected_variance, 0)  # Ensure non-negative
    
    return corrected_variance


def moving_average_filter(trace, window_size):
    """
    Apply moving average filter to 1D trace.
    
    Parameters:
        trace (ndarray): Input time series (1D)
        window_size (int): Window size for moving average
    
    Returns:
        ndarray: Smoothed trace
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000)
        >>> smoothed = moving_average_filter(trace, window_size=10)
        >>> print(smoothed.shape)
        (1000,)
    """
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(trace, kernel, mode='same')
    return smoothed


if __name__ == '__main__':
    """Unit tests for fluctuation module."""
    print("Testing fluctuation_module...")
    
    # Test 1: Full fluctuation analysis
    print("\n1. Testing fluctuation_analysis...")
    np.random.seed(42)
    T, H, W = 200, 32, 32
    fs = 30.0
    
    # Create synthetic data with local fluctuations
    stack = np.random.randn(T, H, W) * 5 + 100
    
    # Add local calcium transient
    t = np.arange(T)
    transient = 20 * np.exp(-(t - 100)**2 / 200)
    stack[:, 15:20, 15:20] += transient[:, np.newaxis, np.newaxis]
    
    sd_stack = fluctuation_analysis(
        stack, fs, 
        spatial_sigma=1.0,
        highpass_cutoff=0.5,
        window_size=30
    )
    
    print(f"   Input shape: {stack.shape}")
    print(f"   Output shape: {sd_stack.shape}")
    print(f"   Output range: [{sd_stack.min():.4f}, {sd_stack.max():.4f}]")
    print(f"   Signal region max: {sd_stack[:, 15:20, 15:20].max():.4f}")
    print(f"   Background max: {sd_stack[:, 0:10, 0:10].max():.4f}")
    assert sd_stack.shape == stack.shape, "Shape mismatch"
    assert sd_stack[:, 15:20, 15:20].max() > sd_stack[:, 0:10, 0:10].max(), "Fluctuation not detected"
    print("   ✓ Fluctuation analysis working correctly")
    
    # Test 2: Local variance map
    print("\n2. Testing compute_local_variance_map...")
    var_map = compute_local_variance_map(stack, window_size=30)
    print(f"   Input shape: {stack.shape}")
    print(f"   Output shape: {var_map.shape}")
    print(f"   Output range: [{var_map.min():.4f}, {var_map.max():.4f}]")
    assert var_map.shape == stack.shape, "Shape mismatch"
    print("   ✓ Local variance map computed correctly")
    
    # Test 3: Temporal high-pass filter
    print("\n3. Testing temporal_highpass_filter...")
    stack_with_drift = stack.copy()
    stack_with_drift += np.linspace(0, 50, T)[:, np.newaxis, np.newaxis]
    
    filtered = temporal_highpass_filter(stack_with_drift, fs, cutoff=0.5)
    print(f"   Input mean: {stack_with_drift.mean():.4f}")
    print(f"   Output mean: {filtered.mean():.4f}")
    print(f"   Input trend: {stack_with_drift[:, 16, 16].mean():.4f}")
    print(f"   Output trend: {filtered[:, 16, 16].mean():.4f}")
    assert abs(filtered.mean()) < 10, "DC component not removed"
    print("   ✓ High-pass filter working correctly")
    
    # Test 4: Shot noise corrected variance
    print("\n4. Testing shot_noise_corrected_variance...")
    poisson_stack = np.random.poisson(100, (T, H, W)).astype(np.float64)
    mean = np.mean(poisson_stack, axis=0)
    var_corrected = shot_noise_corrected_variance(
        poisson_stack, mean, shot_noise_factor=0.01
    )
    print(f"   Input shape: {poisson_stack.shape}")
    print(f"   Output shape: {var_corrected.shape}")
    print(f"   Mean variance: {var_corrected.mean():.4f}")
    assert var_corrected.shape == (H, W), "Shape mismatch"
    assert np.all(var_corrected >= 0), "Negative variance found"
    print("   ✓ Shot noise correction working correctly")
    
    # Test 5: Moving average filter
    print("\n5. Testing moving_average_filter...")
    trace = np.random.randn(1000) * 10
    smoothed = moving_average_filter(trace, window_size=10)
    print(f"   Input std: {trace.std():.4f}")
    print(f"   Output std: {smoothed.std():.4f}")
    assert smoothed.std() < trace.std(), "Smoothing not working"
    print("   ✓ Moving average filter working correctly")
    
    print("\n✅ All fluctuation module tests passed!")
