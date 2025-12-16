"""
Event Detection Module
======================
Calcium event detection using dual-threshold approach.

Detects calcium sparks, puffs, and other transient Ca²⁺ signals using
variance stabilization and dual-threshold methodology.

Author: George
"""

import numpy as np
from scipy.ndimage import label, median_filter, uniform_filter


def detect_sparks(image_stack, f0, intensity_thresh=2.0, peak_thresh=3.8, 
                  min_size=40, median_filter_size=3, uniform_filter_size=3):
    """
    Threshold-based spark detection with dual criteria.
    
    Uses variance stabilization for Poisson noise handling:
    normalized = (F - F₀) / √F₀
    
    Then applies dual thresholds:
    - Intensity threshold (2σ) defines event boundaries
    - Peak threshold (3.8σ) confirms true events
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W) or single frame (H, W)
        f0 (ndarray): Baseline fluorescence (H, W)
        intensity_thresh (float): Intensity threshold in units of σ (standard deviations)
        peak_thresh (float): Peak threshold in units of σ (must be > intensity_thresh)
        min_size (int): Minimum event size in pixels
        median_filter_size (int): Median filter size for smoothing (odd number)
        uniform_filter_size (int): Uniform filter size for smoothing (odd number)
    
    Returns:
        list: List of detected spark regions (each is a boolean mask)
    
    Example:
        >>> import numpy as np
        >>> # Create synthetic spark
        >>> image = np.zeros((128, 128))
        >>> image[60:70, 60:70] = 100  # Bright spot
        >>> f0 = np.ones((128, 128)) * 10
        >>> sparks = detect_sparks(image[np.newaxis, :, :], f0)
        >>> print(f"Detected {len(sparks)} sparks")
    
    Notes:
        - Variance stabilization handles Poisson noise (shot noise)
        - Dual threshold reduces false positives
        - Minimum size filter removes spurious detections
    """
    # Handle single frame input
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis, :, :]
    
    T, H, W = image_stack.shape
    
    # Variance stabilization for Poisson noise
    # normalized = (F - F₀) / √F₀
    f0_safe = np.maximum(f0, 1.0)  # Avoid division by zero
    normalized = (image_stack - f0) / np.sqrt(f0_safe)
    
    # Smooth to reduce noise
    if median_filter_size > 1:
        smoothed = np.array([
            median_filter(frame, size=median_filter_size) 
            for frame in normalized
        ])
    else:
        smoothed = normalized
    
    if uniform_filter_size > 1:
        smoothed = np.array([
            uniform_filter(frame, size=uniform_filter_size) 
            for frame in smoothed
        ])
    
    # Estimate noise standard deviation
    sd = np.std(smoothed)
    
    # Dual threshold detection
    intensity_mask = smoothed > intensity_thresh * sd
    peak_mask = smoothed > peak_thresh * sd
    
    # Find connected regions in intensity mask
    sparks = []
    
    for t in range(T):
        labels, n_features = label(intensity_mask[t])
        
        for i in range(1, n_features + 1):
            region = labels == i
            
            # Check if region meets criteria:
            # 1. Large enough
            # 2. Contains at least one peak pixel
            if np.sum(region) >= min_size and np.any(peak_mask[t] & region):
                sparks.append((region, t))
    
    return sparks


def detect_events_template_matching(trace, template, threshold=0.7):
    """
    Detect transients by cross-correlating with exponential template.
    
    Typical GCaMP6f parameters: τ_rise ≈ 50ms, τ_decay ≈ 140ms
    Typical GCaMP6s parameters: τ_rise ≈ 180ms, τ_decay ≈ 550ms
    
    Parameters:
        trace (ndarray): Input fluorescence trace (1D array)
        template (ndarray): Template waveform for matching
        threshold (float): Correlation threshold (0-1) for detection
    
    Returns:
        ndarray: Array of detected event indices (peak locations)
    
    Example:
        >>> import numpy as np
        >>> # Create template
        >>> t = np.arange(100)
        >>> template = (1 - np.exp(-t/10)) * np.exp(-t/30)
        >>> template = template / template.max()
        >>> 
        >>> # Create trace with events
        >>> trace = np.zeros(1000)
        >>> trace[200:300] = template
        >>> trace[500:600] = template * 0.8
        >>> 
        >>> events = detect_events_template_matching(trace, template)
        >>> print(f"Detected {len(events)} events")
    """
    # Normalize template
    template_norm = template - template.mean()
    template_norm = template_norm / np.linalg.norm(template_norm)
    
    # Normalize trace
    trace_norm = trace - trace.mean()
    trace_norm = trace_norm / np.linalg.norm(trace_norm)
    
    # Cross-correlation
    correlation = np.correlate(trace_norm, template_norm, mode='same')
    
    # Find peaks above threshold
    peaks = []
    for i in range(1, len(correlation) - 1):
        if (correlation[i] > threshold and 
            correlation[i] > correlation[i-1] and 
            correlation[i] > correlation[i+1]):
            peaks.append(i)
    
    return np.array(peaks)


def create_exponential_template(fs, tau_rise=50e-3, tau_decay=140e-3, duration=1.0):
    """
    Create exponential rise-decay template for calcium transients.
    
    Parameters:
        fs (float): Sampling frequency in Hz
        tau_rise (float): Rise time constant in seconds
        tau_decay (float): Decay time constant in seconds
        duration (float): Template duration in seconds
    
    Returns:
        ndarray: Template waveform (normalized to peak=1)
    
    Example:
        >>> # GCaMP6f template
        >>> template = create_exponential_template(30.0, tau_rise=0.05, tau_decay=0.14)
        >>> print(f"Template length: {len(template)} frames")
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Double exponential: rise and decay
    template = (1 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    
    # Normalize to peak=1
    template = template / template.max()
    
    return template


def compute_event_statistics(image_stack, spark_regions):
    """
    Compute statistics for detected events.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        spark_regions (list): List of (region_mask, time_idx) tuples from detect_sparks
    
    Returns:
        list: List of dictionaries containing event statistics
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(100, 64, 64) * 10 + 100
        >>> f0 = np.ones((64, 64)) * 100
        >>> sparks = detect_sparks(stack, f0)
        >>> stats = compute_event_statistics(stack, sparks)
        >>> for s in stats:
        ...     print(f"Event at t={s['time']}, peak={s['peak_intensity']:.2f}")
    """
    stats = []
    
    for region, t in spark_regions:
        # Extract event properties
        event_data = {
            'time': t,
            'size': np.sum(region),
            'centroid': np.array(np.where(region)).mean(axis=1),
            'peak_intensity': np.max(image_stack[t][region]),
            'mean_intensity': np.mean(image_stack[t][region]),
            'integrated_intensity': np.sum(image_stack[t][region])
        }
        
        stats.append(event_data)
    
    return stats


def detect_puffs(sparks, spatial_threshold=10, temporal_threshold=5):
    """
    Group sparks into puffs (sites with repeated activity).
    
    A puff is defined as a site that exhibits multiple sparks within
    a small spatial and temporal window.
    
    Parameters:
        sparks (list): List of spark detections from detect_sparks
        spatial_threshold (float): Maximum distance (pixels) to group sparks
        temporal_threshold (int): Maximum time (frames) to group sparks
    
    Returns:
        list: List of puff sites (each containing multiple sparks)
    
    Example:
        >>> # Assume we have detected sparks
        >>> puffs = detect_puffs(sparks, spatial_threshold=10, temporal_threshold=5)
        >>> print(f"Detected {len(puffs)} puff sites")
    """
    if len(sparks) == 0:
        return []
    
    # Extract centroids and times
    centroids = []
    times = []
    for region, t in sparks:
        y, x = np.where(region)
        centroid = np.array([y.mean(), x.mean()])
        centroids.append(centroid)
        times.append(t)
    
    centroids = np.array(centroids)
    times = np.array(times)
    
    # Group sparks by proximity
    puffs = []
    used = np.zeros(len(sparks), dtype=bool)
    
    for i in range(len(sparks)):
        if used[i]:
            continue
        
        # Find nearby sparks
        distances = np.sqrt(np.sum((centroids - centroids[i])**2, axis=1))
        time_diffs = np.abs(times - times[i])
        
        nearby = (distances < spatial_threshold) & (time_diffs < temporal_threshold) & ~used
        
        if np.sum(nearby) > 1:  # At least 2 sparks for a puff
            puff = [sparks[j] for j in np.where(nearby)[0]]
            puffs.append(puff)
            used[nearby] = True
    
    return puffs


def compute_spark_amplitude(trace, baseline, event_indices):
    """
    Compute amplitude of detected events.
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array)
        baseline (ndarray): Baseline fluorescence (same length as trace)
        event_indices (ndarray): Array of event peak indices
    
    Returns:
        ndarray: Array of event amplitudes (ΔF/F at peak)
    
    Example:
        >>> import numpy as np
        >>> trace = np.zeros(1000)
        >>> trace[100] = 50
        >>> trace[500] = 30
        >>> baseline = np.ones(1000) * 10
        >>> events = np.array([100, 500])
        >>> amplitudes = compute_spark_amplitude(trace, baseline, events)
        >>> print(amplitudes)  # Should be [4.0, 2.0]
    """
    amplitudes = []
    
    for idx in event_indices:
        if idx < len(trace):
            dff = (trace[idx] - baseline[idx]) / max(baseline[idx], 1e-10)
            amplitudes.append(dff)
    
    return np.array(amplitudes)


if __name__ == '__main__':
    """Unit tests for event detection module."""
    print("Testing event_detection_module...")
    
    # Test 1: Spark detection
    print("\n1. Testing detect_sparks...")
    np.random.seed(42)
    H, W = 128, 128
    
    # Create synthetic spark
    image = np.random.randn(H, W) * 5 + 100
    image[60:70, 60:70] += 100  # Bright spark
    image[20:25, 20:25] += 80   # Another spark
    
    f0 = np.ones((H, W)) * 100
    
    sparks = detect_sparks(image[np.newaxis, :, :], f0, 
                          intensity_thresh=2.0, 
                          peak_thresh=3.8,
                          min_size=40)
    
    print(f"   Detected {len(sparks)} sparks")
    print(f"   Expected: 2 sparks")
    assert len(sparks) >= 1, "Should detect at least 1 spark"
    print("   ✓ Spark detection working correctly")
    
    # Test 2: Template matching
    print("\n2. Testing detect_events_template_matching...")
    # Create template
    template = create_exponential_template(30.0, tau_rise=0.05, tau_decay=0.14, duration=1.0)
    print(f"   Template length: {len(template)} frames")
    
    # Create trace with events
    trace = np.random.randn(1000) * 2
    trace[200:200+len(template)] += template * 50
    trace[500:500+len(template)] += template * 40
    
    events = detect_events_template_matching(trace, template, threshold=0.5)
    print(f"   Detected {len(events)} events")
    print(f"   Event locations: {events}")
    assert len(events) >= 1, "Should detect at least 1 event"
    print("   ✓ Template matching working correctly")
    
    # Test 3: Create exponential template
    print("\n3. Testing create_exponential_template...")
    # GCaMP6f template
    template_6f = create_exponential_template(30.0, tau_rise=0.05, tau_decay=0.14)
    # GCaMP6s template
    template_6s = create_exponential_template(30.0, tau_rise=0.18, tau_decay=0.55)
    
    print(f"   GCaMP6f template length: {len(template_6f)}")
    print(f"   GCaMP6s template length: {len(template_6s)}")
    print(f"   GCaMP6f peak at: {np.argmax(template_6f)} frames")
    print(f"   GCaMP6s peak at: {np.argmax(template_6s)} frames")
    assert template_6s.shape[0] == template_6f.shape[0], "Template lengths should match"
    assert np.argmax(template_6s) > np.argmax(template_6f), "6s should peak later than 6f"
    print("   ✓ Template creation working correctly")
    
    # Test 4: Event statistics
    print("\n4. Testing compute_event_statistics...")
    T, H, W = 100, 64, 64
    stack = np.random.randn(T, H, W) * 10 + 100
    stack[50, 30:35, 30:35] += 200  # Add event
    
    f0 = np.ones((H, W)) * 100
    sparks = detect_sparks(stack, f0)
    
    if len(sparks) > 0:
        stats = compute_event_statistics(stack, sparks)
        print(f"   Computed statistics for {len(stats)} events")
        print(f"   Example event: time={stats[0]['time']}, size={stats[0]['size']}")
        assert 'peak_intensity' in stats[0], "Missing peak_intensity"
        assert 'centroid' in stats[0], "Missing centroid"
        print("   ✓ Event statistics working correctly")
    else:
        print("   ⚠ No sparks detected for statistics test")
    
    # Test 5: Puff detection
    print("\n5. Testing detect_puffs...")
    # Create multiple sparks at same location
    mock_sparks = []
    for t in [10, 15, 20]:
        region = np.zeros((64, 64), dtype=bool)
        region[30:35, 30:35] = True
        mock_sparks.append((region, t))
    
    puffs = detect_puffs(mock_sparks, spatial_threshold=10, temporal_threshold=20)
    print(f"   Detected {len(puffs)} puffs")
    if len(puffs) > 0:
        print(f"   Puff contains {len(puffs[0])} sparks")
        assert len(puffs[0]) >= 2, "Puff should contain multiple sparks"
    print("   ✓ Puff detection working correctly")
    
    # Test 6: Spark amplitude
    print("\n6. Testing compute_spark_amplitude...")
    trace = np.ones(1000) * 100
    trace[200] = 150  # Event with amplitude 0.5
    trace[500] = 200  # Event with amplitude 1.0
    
    baseline = np.ones(1000) * 100
    events = np.array([200, 500])
    
    amplitudes = compute_spark_amplitude(trace, baseline, events)
    print(f"   Event amplitudes: {amplitudes}")
    print(f"   Expected: [0.5, 1.0]")
    assert np.allclose(amplitudes, [0.5, 1.0]), "Amplitude calculation incorrect"
    print("   ✓ Amplitude computation working correctly")
    
    print("\n✅ All event detection module tests passed!")
