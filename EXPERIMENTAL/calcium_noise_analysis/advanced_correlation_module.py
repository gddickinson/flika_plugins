"""
Advanced Correlation Analysis Module
====================================
Implements advanced correlation techniques from Swaminathan et al. 2020:
- Cross-correlation maps (CRM)
- Ring correlation analysis
- Nearest-neighbor correlations
- Spatial-temporal correlation analysis

Author: George
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def cross_correlate_traces(trace1, trace2, max_lag=50):
    """
    Compute cross-correlation between two time series.
    
    Parameters:
        trace1 (ndarray): First time series (1D array)
        trace2 (ndarray): Second time series (1D array)
        max_lag (int): Maximum lag to compute (in frames)
    
    Returns:
        ndarray: Cross-correlation values at each lag
    
    Example:
        >>> import numpy as np
        >>> trace1 = np.sin(2 * np.pi * 1.0 * np.arange(1000) / 30.0)
        >>> trace2 = trace1.copy()
        >>> corr = cross_correlate_traces(trace1, trace2, max_lag=50)
        >>> print(f"Peak correlation: {corr[0]:.3f}")
    """
    # Normalize traces (zero mean, unit variance)
    trace1_norm = (trace1 - np.mean(trace1)) / (np.std(trace1) + 1e-10)
    trace2_norm = (trace2 - np.mean(trace2)) / (np.std(trace2) + 1e-10)
    
    # Compute cross-correlation using numpy correlate
    full_corr = np.correlate(trace1_norm, trace2_norm, mode='same')
    
    # Extract lags from -max_lag to +max_lag
    center = len(full_corr) // 2
    corr = full_corr[center:center+max_lag+1] / len(trace1)
    
    return corr


def nearest_neighbor_correlation(image_stack, max_lag=50):
    """
    Compute nearest-neighbor cross-correlations for each pixel.
    
    For each pixel, cross-correlates its time trace with the 8 surrounding pixels.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        max_lag (int): Maximum lag for cross-correlation (frames)
    
    Returns:
        ndarray: Mean correlation values (H, W, max_lag+1)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> corr = nearest_neighbor_correlation(stack, max_lag=50)
        >>> print(corr.shape)
        (64, 64, 51)
    
    Notes:
        - Edge pixels use available neighbors
        - Correlation is averaged over all 8 neighbors
    """
    T, H, W = image_stack.shape
    correlations = np.zeros((H, W, max_lag + 1))
    
    # Neighbor offsets (8-connected)
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for i in range(H):
        for j in range(W):
            center_trace = image_stack[:, i, j]
            neighbor_corrs = []
            
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                
                # Check bounds
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_trace = image_stack[:, ni, nj]
                    corr = cross_correlate_traces(center_trace, neighbor_trace, max_lag)
                    neighbor_corrs.append(corr)
            
            # Average over all valid neighbors
            if neighbor_corrs:
                correlations[i, j, :] = np.mean(neighbor_corrs, axis=0)
    
    return correlations


def compute_correlation_map(image_stack, max_lag=50):
    """
    Generate correlation map (CRM) by computing ξ values.
    
    ξ(p) = 2 * Σ(ρ(n)) from n=1 to 25 - Σ(ρ(n)) from n=1 to 50
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        max_lag (int): Maximum lag for correlation (frames)
    
    Returns:
        ndarray: Correlation map (H, W) with ξ values
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> crm = compute_correlation_map(stack, max_lag=50)
        >>> print(crm.shape)
        (64, 64)
    """
    # Compute nearest-neighbor correlations
    nn_corr = nearest_neighbor_correlation(image_stack, max_lag=max_lag)
    
    # Compute ξ using equation from paper
    # ξ(p) = 2*Σ(ρ(n), n=1..25) - Σ(ρ(n), n=1..50)
    sum_25 = np.sum(nn_corr[:, :, 1:26], axis=2)
    sum_50 = np.sum(nn_corr[:, :, 1:51] if max_lag >= 50 else nn_corr[:, :, 1:], axis=2)
    
    xi = 2 * sum_25 - sum_50
    
    return xi


def ring_correlation(image_stack, center_pixel, length_scales, max_lag=50):
    """
    Compute ring correlation at different spatial length scales.
    
    Cross-correlates center pixel with pixels at increasing distances.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        center_pixel (tuple): (row, col) coordinates of center pixel
        length_scales (list): List of length scales (in pixels) to compute
        max_lag (int): Maximum lag for cross-correlation
    
    Returns:
        ndarray: Mean ξ values at each length scale (len(length_scales),)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> xi_values = ring_correlation(stack, (32, 32), [3, 5, 9, 15], max_lag=50)
        >>> print(xi_values)
    
    Notes:
        - Length scale ℓ defines ℓ×ℓ square region
        - Correlates center with peripheral pixels in ring
    """
    T, H, W = image_stack.shape
    cy, cx = center_pixel
    center_trace = image_stack[:, cy, cx]
    
    xi_values = np.zeros(len(length_scales))
    
    for idx, ell in enumerate(length_scales):
        # Define peripheral pixels (ring at distance ℓ)
        half_ell = ell // 2
        peripheral_corrs = []
        
        # Top and bottom rows
        for j in range(cx - half_ell, cx + half_ell + 1):
            for i in [cy - half_ell, cy + half_ell]:
                if 0 <= i < H and 0 <= j < W and not (i == cy and j == cx):
                    trace = image_stack[:, i, j]
                    corr = cross_correlate_traces(center_trace, trace, max_lag)
                    peripheral_corrs.append(corr)
        
        # Left and right columns (excluding corners already counted)
        for i in range(cy - half_ell + 1, cy + half_ell):
            for j in [cx - half_ell, cx + half_ell]:
                if 0 <= i < H and 0 <= j < W:
                    trace = image_stack[:, i, j]
                    corr = cross_correlate_traces(center_trace, trace, max_lag)
                    peripheral_corrs.append(corr)
        
        # Compute ξ for this length scale
        if peripheral_corrs:
            mean_corr = np.mean(peripheral_corrs, axis=0)
            # ξ = 2*Σ(ρ(n), n=1..25) - Σ(ρ(n), n=1..50)
            sum_25 = np.sum(mean_corr[1:26])
            sum_50 = np.sum(mean_corr[1:51] if max_lag >= 50 else mean_corr[1:])
            xi_values[idx] = 2 * sum_25 - sum_50
    
    return xi_values


def spatial_extent_correlation(image_stack, active_sites, max_length=21):
    """
    Determine spatial extent of correlations at active sites.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        active_sites (list): List of (row, col) coordinates of active sites
        max_length (int): Maximum length scale to test (pixels)
    
    Returns:
        dict: Dictionary with 'length_scales' and 'xi_values' arrays
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> sites = [(32, 32), (48, 48)]
        >>> result = spatial_extent_correlation(stack, sites, max_length=21)
        >>> print(f"Correlation extends to: {result['half_max_distance']:.1f} pixels")
    """
    length_scales = list(range(3, max_length + 1, 2))  # Odd numbers: 3, 5, 7, ...
    all_xi = []
    
    for site in active_sites:
        xi_values = ring_correlation(image_stack, site, length_scales, max_lag=50)
        all_xi.append(xi_values)
    
    # Average over all sites
    mean_xi = np.mean(all_xi, axis=0)
    
    # Find half-maximum distance
    max_xi = np.max(mean_xi)
    half_max = max_xi / 2
    
    # Interpolate to find distance
    half_max_idx = np.where(mean_xi < half_max)[0]
    if len(half_max_idx) > 0:
        half_max_distance = length_scales[half_max_idx[0]]
    else:
        half_max_distance = length_scales[-1]
    
    return {
        'length_scales': np.array(length_scales),
        'xi_values': mean_xi,
        'half_max_distance': half_max_distance,
        'individual_xi': np.array(all_xi)
    }


def exponential_fit_correlation(correlation_curve, fs):
    """
    Fit exponential decay to correlation curve to extract time constant.
    
    Fits: ρ(t) = A * exp(-t/τ)
    
    Parameters:
        correlation_curve (ndarray): Correlation values at each lag
        fs (float): Sampling frequency (Hz)
    
    Returns:
        dict: Dictionary with 'tau' (time constant in seconds) and 'amplitude'
    
    Example:
        >>> import numpy as np
        >>> lags = np.arange(50)
        >>> corr = 0.5 * np.exp(-lags / 10.0)
        >>> result = exponential_fit_correlation(corr, fs=30.0)
        >>> print(f"Time constant: {result['tau']*1000:.1f} ms")
    """
    from scipy.optimize import curve_fit
    
    # Remove lag 0 (self-correlation)
    corr = correlation_curve[1:]
    lags = np.arange(1, len(correlation_curve))
    
    # Time in seconds
    time = lags / fs
    
    # Exponential decay function
    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)
    
    try:
        # Initial guess
        p0 = [corr[0], 0.05]  # A ~ first value, tau ~ 50ms
        
        # Fit
        popt, pcov = curve_fit(exp_decay, time, corr, p0=p0, 
                               bounds=([0, 0.001], [np.inf, 1.0]))
        
        amplitude, tau = popt
        
        # Compute R² goodness of fit
        residuals = corr - exp_decay(time, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((corr - np.mean(corr))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'tau': tau,
            'amplitude': amplitude,
            'r_squared': r_squared,
            'fit_curve': exp_decay(time, *popt)
        }
    
    except RuntimeError:
        print("Exponential fit failed")
        return {
            'tau': np.nan,
            'amplitude': np.nan,
            'r_squared': np.nan,
            'fit_curve': np.full_like(corr, np.nan)
        }


def correlation_time_constant_map(image_stack, fs, threshold=0.1):
    """
    Generate map of correlation time constants across image.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency (Hz)
        threshold (float): Minimum correlation amplitude to attempt fit
    
    Returns:
        ndarray: Map of time constants in seconds (H, W)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(500, 64, 64)
        >>> tau_map = correlation_time_constant_map(stack, fs=30.0)
        >>> print(tau_map.shape)
        (64, 64)
    """
    nn_corr = nearest_neighbor_correlation(image_stack, max_lag=50)
    H, W = nn_corr.shape[:2]
    
    tau_map = np.zeros((H, W))
    
    for i in range(H):
        for j in range(W):
            corr_curve = nn_corr[i, j, :]
            
            # Only fit if correlation is substantial
            if np.max(corr_curve[1:]) > threshold:
                result = exponential_fit_correlation(corr_curve, fs)
                tau_map[i, j] = result['tau']
            else:
                tau_map[i, j] = np.nan
    
    return tau_map


if __name__ == '__main__':
    """Unit tests for advanced correlation module."""
    print("Testing advanced_correlation_module...")
    
    # Test 1: Cross-correlation between traces
    print("\n1. Testing cross_correlate_traces...")
    t = np.arange(1000) / 30.0
    trace1 = np.sin(2 * np.pi * 1.0 * t)
    trace2 = trace1 + np.random.randn(1000) * 0.1
    
    corr = cross_correlate_traces(trace1, trace2, max_lag=50)
    print(f"   Correlation shape: {corr.shape}")
    print(f"   Peak correlation: {corr[0]:.3f}")
    assert corr.shape == (51,), "Correlation shape incorrect"
    assert corr[0] > 0.9, "Peak correlation should be high"
    print("   ✓ Cross-correlation working correctly")
    
    # Test 2: Nearest-neighbor correlation
    print("\n2. Testing nearest_neighbor_correlation...")
    T, H, W = 500, 32, 32
    stack = np.random.randn(T, H, W)
    
    # Add correlated signal to a region
    signal = np.sin(2 * np.pi * 1.0 * np.arange(T) / 30.0)
    stack[:, 15:20, 15:20] += signal[:, np.newaxis, np.newaxis]
    
    nn_corr = nearest_neighbor_correlation(stack, max_lag=50)
    print(f"   NN correlation shape: {nn_corr.shape}")
    print(f"   Signal region correlation: {nn_corr[17, 17, 0]:.3f}")
    print(f"   Background correlation: {nn_corr[5, 5, 0]:.3f}")
    assert nn_corr.shape == (H, W, 51), "Shape mismatch"
    assert nn_corr[17, 17, 0] > nn_corr[5, 5, 0], "Signal region should have higher correlation"
    print("   ✓ Nearest-neighbor correlation working correctly")
    
    # Test 3: Correlation map
    print("\n3. Testing compute_correlation_map...")
    crm = compute_correlation_map(stack, max_lag=50)
    print(f"   CRM shape: {crm.shape}")
    print(f"   CRM range: [{crm.min():.3f}, {crm.max():.3f}]")
    print(f"   Signal region ξ: {crm[17, 17]:.3f}")
    print(f"   Background ξ: {crm[5, 5]:.3f}")
    assert crm.shape == (H, W), "CRM shape mismatch"
    assert crm[17, 17] > crm[5, 5], "Signal region should have higher ξ"
    print("   ✓ Correlation map working correctly")
    
    # Test 4: Ring correlation
    print("\n4. Testing ring_correlation...")
    length_scales = [3, 5, 9, 15]
    xi_values = ring_correlation(stack, (17, 17), length_scales, max_lag=50)
    print(f"   Ring correlation values: {xi_values}")
    print(f"   Length scales: {length_scales}")
    assert len(xi_values) == len(length_scales), "Length mismatch"
    print("   ✓ Ring correlation working correctly")
    
    # Test 5: Spatial extent
    print("\n5. Testing spatial_extent_correlation...")
    active_sites = [(17, 17), (18, 18)]
    result = spatial_extent_correlation(stack, active_sites, max_length=21)
    print(f"   Half-max distance: {result['half_max_distance']:.1f} pixels")
    print(f"   Number of sites: {len(active_sites)}")
    assert 'half_max_distance' in result, "Missing half_max_distance"
    print("   ✓ Spatial extent analysis working correctly")
    
    # Test 6: Exponential fit
    print("\n6. Testing exponential_fit_correlation...")
    lags = np.arange(51)
    true_tau = 0.05  # 50 ms
    corr_test = 0.5 * np.exp(-lags / (true_tau * 30.0))  # 30 Hz sampling
    
    result = exponential_fit_correlation(corr_test, fs=30.0)
    print(f"   Fitted tau: {result['tau']*1000:.1f} ms")
    print(f"   True tau: {true_tau*1000:.1f} ms")
    print(f"   R²: {result['r_squared']:.3f}")
    assert abs(result['tau'] - true_tau) < 0.01, "Tau estimation inaccurate"
    assert result['r_squared'] > 0.98, "Fit quality poor"
    print("   ✓ Exponential fitting working correctly")
    
    print("\n✅ All advanced correlation module tests passed!")
