"""
Power Spectrum Map and Correlation Map Module
==============================================
Implements PSM and CRM generation from Swaminathan et al. 2020:
- Power Spectrum Maps (PSM) showing excess power ratio η
- Correlation Maps (CRM) showing correlation value ξ
- Mean and maximum maps
- Active site identification

Author: George
"""

import numpy as np
from scipy.signal import welch


def compute_excess_power_ratio(image_stack, fs, roi_size=3, 
                               low_freq_range=(0.1, 5.0), 
                               high_freq_range=(50, 62),
                               nperseg=256):
    """
    Compute excess power ratio η for each ROI in image stack.
    
    η = (P_LFR - P_HFR) / P_HFR
    
    where P_LFR is mean power in low frequency range (calcium signals)
    and P_HFR is mean power in high frequency range (shot noise)
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency (Hz)
        roi_size (int): Size of ROI for spatial averaging (pixels)
        low_freq_range (tuple): (min, max) frequency for calcium signals (Hz)
        high_freq_range (tuple): (min, max) frequency for shot noise (Hz)
        nperseg (int): Segment length for Welch's method
    
    Returns:
        ndarray: Map of η values (H//roi_size, W//roi_size)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(1000, 128, 128)
        >>> eta_map = compute_excess_power_ratio(stack, fs=30.0, roi_size=3)
        >>> print(eta_map.shape)
    
    Notes:
        - η normalizes for fluorescence intensity variations
        - Higher η indicates more calcium activity
        - Shot noise provides normalization baseline
    """
    T, H, W = image_stack.shape
    
    # Determine output dimensions after ROI sub-sampling
    H_out = H // roi_size
    W_out = W // roi_size
    
    eta_map = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            # Extract ROI
            row_start = i * roi_size
            row_end = row_start + roi_size
            col_start = j * roi_size
            col_end = col_start + roi_size
            
            roi = image_stack[:, row_start:row_end, col_start:col_end]
            
            # Average over ROI
            trace = np.mean(roi, axis=(1, 2))
            
            # Compute power spectrum
            freqs, psd = welch(trace, fs=fs, nperseg=nperseg, 
                              noverlap=nperseg//2, nfft=1024)
            
            # Define frequency masks
            low_mask = (freqs >= low_freq_range[0]) & (freqs <= low_freq_range[1])
            high_mask = (freqs >= high_freq_range[0]) & (freqs <= high_freq_range[1])
            
            # Compute mean power in each range
            if np.any(low_mask) and np.any(high_mask):
                P_LFR = np.mean(psd[low_mask])
                P_HFR = np.mean(psd[high_mask])
                
                # Compute η
                if P_HFR > 0:
                    eta_map[i, j] = (P_LFR - P_HFR) / P_HFR
                else:
                    eta_map[i, j] = 0
            else:
                eta_map[i, j] = 0
    
    return eta_map


def compute_power_spectrum_map(image_stack, fs, subsection_length=1024, 
                               roi_size=3, **kwargs):
    """
    Generate Power Spectrum Map (PSM) from image stack.
    
    Divides stack into time subsections and computes η for each.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency (Hz)
        subsection_length (int): Length of each time subsection (frames)
        roi_size (int): Size of spatial ROI averaging (pixels)
        **kwargs: Additional arguments for compute_excess_power_ratio
    
    Returns:
        list: List of PSMs, one per time subsection
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(5000, 128, 128)
        >>> psm_list = compute_power_spectrum_map(stack, fs=125.0, subsection_length=1024)
        >>> print(f"Generated {len(psm_list)} PSMs")
    """
    T = image_stack.shape[0]
    n_subsections = T // subsection_length
    
    psm_list = []
    
    for i in range(n_subsections):
        start = i * subsection_length
        end = start + subsection_length
        
        subsection = image_stack[start:end]
        
        # Compute η map for this subsection
        eta_map = compute_excess_power_ratio(subsection, fs, roi_size=roi_size, **kwargs)
        
        psm_list.append(eta_map)
    
    return psm_list


def mean_power_spectrum_map(psm_list):
    """
    Compute mean PSM from list of PSMs.
    
    Parameters:
        psm_list (list): List of PSM arrays
    
    Returns:
        ndarray: Mean PSM
    
    Example:
        >>> import numpy as np
        >>> psm_list = [np.random.randn(42, 42) for _ in range(5)]
        >>> mean_psm = mean_power_spectrum_map(psm_list)
        >>> print(mean_psm.shape)
    """
    return np.mean(psm_list, axis=0)


def maximum_power_spectrum_map(psm_list):
    """
    Compute maximum PSM from list of PSMs.
    
    Takes pixel-wise maximum across all time subsections.
    Emphasizes sites with large but infrequent events.
    
    Parameters:
        psm_list (list): List of PSM arrays
    
    Returns:
        ndarray: Maximum PSM
    
    Example:
        >>> import numpy as np
        >>> psm_list = [np.random.randn(42, 42) for _ in range(5)]
        >>> max_psm = maximum_power_spectrum_map(psm_list)
        >>> print(max_psm.shape)
    """
    return np.max(psm_list, axis=0)


def identify_hotspots(psm, threshold=2.0, min_size=1):
    """
    Identify hot spots in PSM where η exceeds threshold.
    
    Parameters:
        psm (ndarray): Power spectrum map (H, W)
        threshold (float): Minimum η value for hotspot
        min_size (int): Minimum number of contiguous pixels
    
    Returns:
        list: List of (row, col) coordinates of hotspot centers
    
    Example:
        >>> import numpy as np
        >>> psm = np.random.randn(42, 42)
        >>> psm[20, 20] = 5.0
        >>> psm[30, 30] = 4.0
        >>> hotspots = identify_hotspots(psm, threshold=2.0)
        >>> print(f"Found {len(hotspots)} hotspots")
    """
    from scipy.ndimage import label, center_of_mass
    
    # Threshold map
    binary = psm > threshold
    
    # Label connected components
    labeled, n_features = label(binary)
    
    hotspots = []
    
    for i in range(1, n_features + 1):
        region = labeled == i
        size = np.sum(region)
        
        if size >= min_size:
            # Find center of mass
            center = center_of_mass(region)
            hotspots.append((int(center[0]), int(center[1])))
    
    return hotspots


def compute_correlation_map_from_stack(image_stack, max_lag=50, subsection_length=500):
    """
    Generate Correlation Map (CRM) from image stack.
    
    Uses nearest-neighbor cross-correlations.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        max_lag (int): Maximum lag for correlation (frames)
        subsection_length (int): Length of time subsections (frames)
    
    Returns:
        list: List of CRMs, one per time subsection
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(5000, 128, 128)
        >>> crm_list = compute_correlation_map_from_stack(stack, max_lag=50)
        >>> print(f"Generated {len(crm_list)} CRMs")
    """
    from . import advanced_correlation_module as acm
    
    T = image_stack.shape[0]
    n_subsections = T // subsection_length
    
    crm_list = []
    
    for i in range(n_subsections):
        start = i * subsection_length
        end = start + subsection_length
        
        subsection = image_stack[start:end]
        
        # Compute correlation map
        crm = acm.compute_correlation_map(subsection, max_lag=max_lag)
        
        crm_list.append(crm)
    
    return crm_list


def mean_correlation_map(crm_list):
    """
    Compute mean CRM from list of CRMs.
    
    Parameters:
        crm_list (list): List of CRM arrays
    
    Returns:
        ndarray: Mean CRM
    """
    return np.mean(crm_list, axis=0)


def maximum_correlation_map(crm_list):
    """
    Compute maximum CRM from list of CRMs.
    
    Parameters:
        crm_list (list): List of CRM arrays
    
    Returns:
        ndarray: Maximum CRM
    """
    return np.max(crm_list, axis=0)


def compare_psm_crm(psm, crm, roi_size=3):
    """
    Compare PSM and CRM to identify consistent active sites.
    
    Sites should show both high η and high ξ values.
    
    Parameters:
        psm (ndarray): Power spectrum map
        crm (ndarray): Correlation map  
        roi_size (int): ROI size used in PSM (for alignment)
    
    Returns:
        dict: Dictionary with comparison metrics
    
    Example:
        >>> import numpy as np
        >>> psm = np.random.randn(42, 42)
        >>> crm = np.random.randn(128, 128)
        >>> result = compare_psm_crm(psm, crm, roi_size=3)
        >>> print(f"Correlation between PSM and CRM: {result['correlation']:.3f}")
    """
    # Downsample CRM to match PSM resolution
    H_psm, W_psm = psm.shape
    H_crm, W_crm = crm.shape
    
    crm_downsampled = np.zeros((H_psm, W_psm))
    
    for i in range(H_psm):
        for j in range(W_psm):
            row_start = i * roi_size
            row_end = min(row_start + roi_size, H_crm)
            col_start = j * roi_size
            col_end = min(col_start + roi_size, W_crm)
            
            crm_downsampled[i, j] = np.mean(crm[row_start:row_end, col_start:col_end])
    
    # Normalize both maps
    psm_norm = (psm - np.mean(psm)) / (np.std(psm) + 1e-10)
    crm_norm = (crm_downsampled - np.mean(crm_downsampled)) / (np.std(crm_downsampled) + 1e-10)
    
    # Compute correlation
    correlation = np.corrcoef(psm_norm.flatten(), crm_norm.flatten())[0, 1]
    
    # Identify sites high in both
    psm_thresh = np.percentile(psm, 90)
    crm_thresh = np.percentile(crm_downsampled, 90)
    
    active_both = (psm > psm_thresh) & (crm_downsampled > crm_thresh)
    n_active = np.sum(active_both)
    
    return {
        'correlation': correlation,
        'n_active_both': n_active,
        'psm_downsampled': psm,
        'crm_downsampled': crm_downsampled,
        'active_sites': active_both
    }


if __name__ == '__main__':
    """Unit tests for PSM/CRM module."""
    print("Testing psm_crm_module...")
    
    # Test 1: Excess power ratio
    print("\n1. Testing compute_excess_power_ratio...")
    np.random.seed(42)
    T, H, W = 1000, 64, 64
    fs = 125.0
    stack = np.random.randn(T, H, W)
    
    # Add calcium-like signal
    t = np.arange(T) / fs
    signal = np.sin(2 * np.pi * 1.0 * t)
    stack[:, 30:35, 30:35] += signal[:, np.newaxis, np.newaxis] * 5
    
    eta_map = compute_excess_power_ratio(stack, fs, roi_size=3)
    print(f"   η map shape: {eta_map.shape}")
    print(f"   η range: [{eta_map.min():.3f}, {eta_map.max():.3f}]")
    print(f"   Signal region η: {eta_map[10, 10]:.3f}")
    print(f"   Background η: {eta_map[0, 0]:.3f}")
    assert eta_map.shape == (H//3, W//3), "Shape mismatch"
    assert eta_map[10, 10] > eta_map[0, 0], "Signal region should have higher η"
    print("   ✓ Excess power ratio working correctly")
    
    # Test 2: Power spectrum map
    print("\n2. Testing compute_power_spectrum_map...")
    stack_long = np.random.randn(5000, 64, 64)
    psm_list = compute_power_spectrum_map(stack_long, fs=125.0, subsection_length=1024, roi_size=3)
    print(f"   Generated {len(psm_list)} PSMs")
    print(f"   Each PSM shape: {psm_list[0].shape}")
    assert len(psm_list) == 4, "Should generate 4 PSMs from 5000 frames"
    print("   ✓ Power spectrum map generation working correctly")
    
    # Test 3: Mean and max PSM
    print("\n3. Testing mean and maximum PSMs...")
    mean_psm = mean_power_spectrum_map(psm_list)
    max_psm = maximum_power_spectrum_map(psm_list)
    print(f"   Mean PSM shape: {mean_psm.shape}")
    print(f"   Max PSM shape: {max_psm.shape}")
    assert mean_psm.shape == psm_list[0].shape, "Shape mismatch"
    assert np.all(max_psm >= mean_psm), "Max should be >= mean everywhere"
    print("   ✓ Mean and maximum PSMs working correctly")
    
    # Test 4: Hotspot identification
    print("\n4. Testing identify_hotspots...")
    # Create synthetic PSM with hotspots
    test_psm = np.random.randn(42, 42) * 0.5
    test_psm[20:23, 20:23] = 5.0
    test_psm[35:38, 35:38] = 4.0
    
    hotspots = identify_hotspots(test_psm, threshold=2.0)
    print(f"   Found {len(hotspots)} hotspots")
    print(f"   Hotspot locations: {hotspots}")
    assert len(hotspots) >= 1, "Should find at least 1 hotspot"
    print("   ✓ Hotspot identification working correctly")
    
    # Test 5: PSM vs CRM comparison
    print("\n5. Testing compare_psm_crm...")
    psm_test = np.random.randn(42, 42)
    crm_test = np.random.randn(128, 128)
    
    # Make them somewhat correlated
    for i in range(42):
        for j in range(42):
            crm_test[i*3:(i+1)*3, j*3:(j+1)*3] = psm_test[i, j]
    
    result = compare_psm_crm(psm_test, crm_test, roi_size=3)
    print(f"   Correlation: {result['correlation']:.3f}")
    print(f"   Active sites in both: {result['n_active_both']}")
    assert 'correlation' in result, "Missing correlation"
    print("   ✓ PSM/CRM comparison working correctly")
    
    print("\n✅ All PSM/CRM module tests passed!")
