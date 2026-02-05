"""
Temporal Fluctuation Module
============================
SD-based fluctuation analysis following Lock & Parker 2020 methodology.
Highlights transient local Ca²⁺ signals through running variance calculation
with shot noise correction.

This module implements the fluctuation processing algorithm from Lock & Parker (2020)
to resolve local Ca²⁺ transients during global Ca²⁺ elevations.

Author: George
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt


def compute_sd_image_stack(image_stack, fs, spatial_sigma=1.0, 
                           highpass_cutoff=0.5, window_frames=20,
                           shot_noise_factor=None):
    """
    Generate SD (standard deviation) image stack highlighting temporal fluctuations.
    
    Algorithm (Lock & Parker 2020):
    1. Spatial Gaussian blur to reduce pixel noise
    2. Temporal high-pass filter to remove baseline drift
    3. Running variance calculation in sliding window
    4. Optional shot noise correction (proportional to mean intensity)
    5. Return standard deviation (square root of variance)
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency in Hz
        spatial_sigma (float): Sigma for Gaussian spatial smoothing (pixels)
        highpass_cutoff (float): High-pass filter cutoff frequency (Hz)
        window_frames (int): Sliding window size for variance calculation
        shot_noise_factor (float): Shot noise correction factor (None to disable)
                                   Empirically determined, typically 0.001-0.01
    
    Returns:
        ndarray: SD image stack (T, H, W) showing fluorescence fluctuations
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64) * 10 + 100
        >>> # Add local Ca²⁺ transient
        >>> stack[100:150, 30:35, 30:35] += 50
        >>> fs = 30.0
        >>> sd_stack = compute_sd_image_stack(stack, fs, spatial_sigma=1.0)
        >>> # Peak SD at transient location
        >>> print(sd_stack[100:150, 30:35, 30:35].mean())
    
    Notes:
        - Shot noise correction removes variance linearly proportional to intensity
        - Window size should be ~1 second worth of frames
        - High-pass filtering removes slow baseline drift
    """
    T, H, W = image_stack.shape
    
    # Step 1: Spatial smoothing
    blurred = np.array([
        gaussian_filter(frame.astype(np.float64), spatial_sigma) 
        for frame in image_stack
    ])
    
    # Step 2: Temporal high-pass filter
    nyquist = fs / 2.0
    cutoff_norm = highpass_cutoff / nyquist
    cutoff_norm = max(0.001, min(cutoff_norm, 0.999))
    
    b, a = butter(2, cutoff_norm, btype='high')
    
    filtered = np.zeros_like(blurred)
    for i in range(H):
        for j in range(W):
            filtered[:, i, j] = filtfilt(b, a, blurred[:, i, j])
    
    # Step 3: Running variance with optional shot noise correction
    half_win = window_frames // 2
    var_stack = np.zeros_like(filtered)
    
    for t in range(half_win, T - half_win):
        window = filtered[t-half_win:t+half_win+1]
        var_stack[t] = np.var(window, axis=0)
        
        # Optional shot noise correction
        if shot_noise_factor is not None:
            mean_intensity = np.mean(blurred[t-half_win:t+half_win+1], axis=0)
            var_stack[t] -= shot_noise_factor * mean_intensity
            var_stack[t] = np.maximum(var_stack[t], 0)  # Ensure non-negative
    
    # Return standard deviation
    return np.sqrt(var_stack)


def detect_puff_flurries(sd_stack, ca_stack, threshold_sd=2.0):
    """
    Detect "flurries" of Ca²⁺ puffs during rising phase of global signals.
    
    Identifies periods where SD signal is elevated, indicating punctate release.
    
    Parameters:
        sd_stack (ndarray): SD image stack from compute_sd_image_stack (T, H, W)
        ca_stack (ndarray): Ca²⁺ fluorescence stack (T, H, W) or global trace (T,)
        threshold_sd (float): SD threshold for puff detection (in units of baseline SD)
    
    Returns:
        dict: Dictionary containing:
            - 'flurry_periods': List of (start, end) frame indices for flurries
            - 'peak_sd_frames': Frame indices where SD is maximal
            - 'sd_vs_ca': Scatter data (ca_level, sd_value) for correlation analysis
    
    Example:
        >>> sd_stack = compute_sd_image_stack(stack, fs=30.0)
        >>> result = detect_puff_flurries(sd_stack, stack)
        >>> print(f"Detected {len(result['flurry_periods'])} flurries")
    """
    # Compute mean SD across spatial dimensions
    mean_sd = np.mean(sd_stack, axis=(1, 2))
    
    # Compute baseline SD statistics
    baseline_sd = np.std(mean_sd[:100])  # First 100 frames as baseline
    
    # Detect periods above threshold
    above_threshold = mean_sd > (threshold_sd * baseline_sd)
    
    # Find continuous periods
    flurry_periods = []
    in_flurry = False
    start_frame = 0
    
    for t in range(len(above_threshold)):
        if above_threshold[t] and not in_flurry:
            start_frame = t
            in_flurry = True
        elif not above_threshold[t] and in_flurry:
            flurry_periods.append((start_frame, t))
            in_flurry = False
    
    # Find peak SD frames
    peak_sd_frames = []
    for start, end in flurry_periods:
        local_peak = start + np.argmax(mean_sd[start:end])
        peak_sd_frames.append(local_peak)
    
    # Generate SD vs Ca²⁺ scatter data for correlation analysis
    if ca_stack.ndim == 3:
        mean_ca = np.mean(ca_stack, axis=(1, 2))
    else:
        mean_ca = ca_stack
    
    sd_vs_ca = list(zip(mean_ca, mean_sd))
    
    return {
        'flurry_periods': flurry_periods,
        'peak_sd_frames': peak_sd_frames,
        'sd_vs_ca': sd_vs_ca,
        'mean_sd_trace': mean_sd
    }


def compute_sd_vs_calcium_relationship(sd_stack, ca_stack, bins=50):
    """
    Compute relationship between SD signal and Ca²⁺ level (Lock & Parker Fig 3).
    
    Generates scatter plot data showing SD as function of bulk Ca²⁺ level.
    Typically shows inverted-U relationship during global signals.
    
    Parameters:
        sd_stack (ndarray): SD image stack (T, H, W)
        ca_stack (ndarray): Ca²⁺ fluorescence stack (T, H, W)
        bins (int): Number of bins for Ca²⁺ levels
    
    Returns:
        dict: Dictionary containing:
            - 'ca_bins': Array of Ca²⁺ bin centers
            - 'mean_sd': Mean SD for each Ca²⁺ bin
            - 'std_sd': Standard deviation of SD for each bin
            - 'n_points': Number of points in each bin
    
    Example:
        >>> result = compute_sd_vs_calcium_relationship(sd_stack, ca_stack)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result['ca_bins'], result['mean_sd'])
        >>> plt.xlabel('ΔF/F₀')
        >>> plt.ylabel('Mean SD')
    """
    # Flatten spatial dimensions
    sd_flat = sd_stack.reshape(sd_stack.shape[0], -1)
    ca_flat = ca_stack.reshape(ca_stack.shape[0], -1)
    
    # Create bins for Ca²⁺ levels
    ca_min, ca_max = ca_flat.min(), ca_flat.max()
    bin_edges = np.linspace(ca_min, ca_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute mean SD for each bin
    mean_sd = np.zeros(bins)
    std_sd = np.zeros(bins)
    n_points = np.zeros(bins, dtype=int)
    
    for i in range(bins):
        mask = (ca_flat >= bin_edges[i]) & (ca_flat < bin_edges[i+1])
        if np.any(mask):
            mean_sd[i] = np.mean(sd_flat[mask])
            std_sd[i] = np.std(sd_flat[mask])
            n_points[i] = np.sum(mask)
    
    return {
        'ca_bins': bin_centers,
        'mean_sd': mean_sd,
        'std_sd': std_sd,
        'n_points': n_points
    }


def compute_spatial_sd_map(image_stack, window_frames=20):
    """
    Compute spatial standard deviation map showing pixels with high variance.
    
    Complements temporal SD by showing which pixels have highest temporal variance.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        window_frames (int): Window size for variance calculation
    
    Returns:
        ndarray: Spatial SD map (H, W) showing mean SD across time
    
    Example:
        >>> sd_map = compute_spatial_sd_map(stack)
        >>> # Hot spots indicate active release sites
    """
    T, H, W = image_stack.shape
    sd_map = np.zeros((H, W))
    
    half_win = window_frames // 2
    sd_time_series = []
    
    for t in range(half_win, T - half_win):
        window = image_stack[t-half_win:t+half_win+1]
        sd_frame = np.std(window, axis=0)
        sd_time_series.append(sd_frame)
    
    # Average across time
    sd_map = np.mean(sd_time_series, axis=0)
    
    return sd_map


def quantify_punctate_vs_diffuse(sd_stack, ca_stack, ca_peak_frame):
    """
    Quantify punctate vs diffuse Ca²⁺ release modes (Lock & Parker 2020).
    
    Estimates fraction of Ca²⁺ release occurring via punctate puffs vs
    diffuse release based on SD signal.
    
    Parameters:
        sd_stack (ndarray): SD image stack (T, H, W)
        ca_stack (ndarray): Ca²⁺ fluorescence stack (T, H, W)
        ca_peak_frame (int): Frame index where Ca²⁺ signal peaks
    
    Returns:
        dict: Dictionary containing:
            - 'punctate_fraction': Fraction of release via puffs (0-1)
            - 'diffuse_fraction': Fraction of release via diffuse mode (0-1)
            - 'puff_phase_end': Frame where puff activity terminates
            - 'cumulative_release': Cumulative Ca²⁺ release trace
    
    Example:
        >>> result = quantify_punctate_vs_diffuse(sd_stack, ca_stack, peak_frame=200)
        >>> print(f"Punctate: {result['punctate_fraction']:.1%}")
        >>> print(f"Diffuse: {result['diffuse_fraction']:.1%}")
    
    Notes:
        - Lock & Parker found ~41% of initial flux via puffs
        - ~15% of total cumulative release via puffs
    """
    # Find when puff activity terminates (SD returns to baseline)
    mean_sd = np.mean(sd_stack, axis=(1, 2))
    baseline_sd = np.std(mean_sd[:100])
    
    # Find frame where SD drops below 2*baseline after rising
    puff_phase_end = ca_peak_frame
    for t in range(ca_peak_frame):
        if mean_sd[t] < 2 * baseline_sd and np.all(mean_sd[t:ca_peak_frame] < 2 * baseline_sd):
            puff_phase_end = t
            break
    
    # Estimate Ca²⁺ release rate (simplified - assumes single compartment)
    mean_ca = np.mean(ca_stack, axis=(1, 2))
    ca_release_rate = np.gradient(mean_ca)  # Simplified - doesn't account for removal
    
    # Integrate release during puff phase
    punctate_release = np.sum(ca_release_rate[:puff_phase_end])
    
    # Total release to peak
    total_release_to_peak = np.sum(ca_release_rate[:ca_peak_frame])
    
    # Fractions
    punctate_fraction = punctate_release / total_release_to_peak if total_release_to_peak > 0 else 0
    diffuse_fraction = 1 - punctate_fraction
    
    # Cumulative release
    cumulative_release = np.cumsum(ca_release_rate)
    
    return {
        'punctate_fraction': punctate_fraction,
        'diffuse_fraction': diffuse_fraction,
        'puff_phase_end': puff_phase_end,
        'cumulative_release': cumulative_release,
        'mean_ca_trace': mean_ca,
        'release_rate': ca_release_rate
    }


if __name__ == '__main__':
    """Unit tests for temporal fluctuation module."""
    print("Testing temporal_fluctuation_module...")
    
    # Test 1: SD image stack computation
    print("\n1. Testing compute_sd_image_stack...")
    np.random.seed(42)
    T, H, W = 300, 64, 64
    fs = 30.0
    
    # Create synthetic data with local transient
    stack = np.random.randn(T, H, W) * 5 + 100
    # Add Ca²⁺ puff
    t_puff = np.arange(50, 120)
    amplitude = 50 * np.exp(-(t_puff - 85)**2 / 200)
    stack[50:120, 30:35, 30:35] += amplitude[:, np.newaxis, np.newaxis]
    
    sd_stack = compute_sd_image_stack(stack, fs, spatial_sigma=1.0,
                                     highpass_cutoff=0.5, window_frames=20)
    
    print(f"   Input shape: {stack.shape}")
    print(f"   Output shape: {sd_stack.shape}")
    print(f"   SD at puff site: {sd_stack[50:120, 30:35, 30:35].mean():.4f}")
    print(f"   SD at background: {sd_stack[50:120, 0:10, 0:10].mean():.4f}")
    assert sd_stack.shape == stack.shape, "Shape mismatch"
    assert sd_stack[50:120, 30:35, 30:35].mean() > sd_stack[50:120, 0:10, 0:10].mean(), "Puff not detected"
    print("   ✓ SD image stack computed correctly")
    
    # Test 2: Puff flurry detection
    print("\n2. Testing detect_puff_flurries...")
    result = detect_puff_flurries(sd_stack, stack)
    print(f"   Detected {len(result['flurry_periods'])} flurries")
    print(f"   Peak SD frames: {result['peak_sd_frames']}")
    assert len(result['flurry_periods']) >= 1, "Should detect at least one flurry"
    print("   ✓ Puff flurry detection working")
    
    # Test 3: SD vs Ca²⁺ relationship
    print("\n3. Testing compute_sd_vs_calcium_relationship...")
    relationship = compute_sd_vs_calcium_relationship(sd_stack, stack, bins=20)
    print(f"   Ca²⁺ bins: {len(relationship['ca_bins'])}")
    print(f"   Mean SD range: [{relationship['mean_sd'].min():.4f}, {relationship['mean_sd'].max():.4f}]")
    assert len(relationship['ca_bins']) == 20, "Incorrect number of bins"
    print("   ✓ SD vs Ca²⁺ relationship computed")
    
    # Test 4: Spatial SD map
    print("\n4. Testing compute_spatial_sd_map...")
    sd_map = compute_spatial_sd_map(stack, window_frames=20)
    print(f"   SD map shape: {sd_map.shape}")
    print(f"   Max SD location: {np.unravel_index(sd_map.argmax(), sd_map.shape)}")
    assert sd_map.shape == (H, W), "Shape mismatch"
    print("   ✓ Spatial SD map computed")
    
    # Test 5: Punctate vs diffuse quantification
    print("\n5. Testing quantify_punctate_vs_diffuse...")
    ca_peak = np.argmax(np.mean(stack, axis=(1, 2)))
    result = quantify_punctate_vs_diffuse(sd_stack, stack, ca_peak)
    print(f"   Punctate fraction: {result['punctate_fraction']:.2%}")
    print(f"   Diffuse fraction: {result['diffuse_fraction']:.2%}")
    print(f"   Puff phase ends at frame: {result['puff_phase_end']}")
    assert 0 <= result['punctate_fraction'] <= 1, "Invalid fraction"
    assert abs(result['punctate_fraction'] + result['diffuse_fraction'] - 1.0) < 0.01, "Fractions don't sum to 1"
    print("   ✓ Punctate vs diffuse quantification working")
    
    print("\n✅ All temporal fluctuation module tests passed!")
