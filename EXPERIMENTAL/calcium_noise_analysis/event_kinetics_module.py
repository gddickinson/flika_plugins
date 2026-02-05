"""
Event Kinetics Module
=====================
Detailed kinetic analysis of individual Ca²⁺ events (puffs, sparks, waves).
Extracts rise times, decay times, amplitudes, durations, and kinetic parameters.

Author: George
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def extract_event_kinetics(trace, fs, event_indices, baseline=None):
    """
    Extract comprehensive kinetic parameters from detected events.
    
    For each event, measures:
    - Peak amplitude
    - Rise time (10-90%)
    - Decay time (exponential fit)
    - Half-width (FWHM)
    - Rise rate
    - Decay rate
    - Event duration
    - Area under curve
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array)
        fs (float): Sampling frequency (Hz)
        event_indices (ndarray): Array of event peak frame indices
        baseline (ndarray): Baseline fluorescence (same length as trace)
                          If None, uses mean of first 10% of trace
    
    Returns:
        list: List of dictionaries, one per event, containing kinetic parameters
    
    Example:
        >>> trace = np.zeros(1000)
        >>> # Add exponential event
        >>> t = np.arange(100)
        >>> trace[200:300] = 100 * np.exp(-t/20)
        >>> events = np.array([200])
        >>> kinetics = extract_event_kinetics(trace, fs=30.0, event_indices=events)
        >>> print(kinetics[0]['decay_tau'])  # ~20 frames = 667 ms at 30 Hz
    """
    if baseline is None:
        baseline = np.mean(trace[:len(trace)//10])
    elif np.isscalar(baseline):
        baseline = np.ones_like(trace) * baseline
    
    results = []
    dt = 1.0 / fs  # Time per frame in seconds
    
    for peak_idx in event_indices:
        # Safety checks
        if peak_idx < 10 or peak_idx >= len(trace) - 10:
            continue
        
        # Find event boundaries
        start_idx = _find_event_start(trace, peak_idx, baseline)
        end_idx = _find_event_end(trace, peak_idx, baseline)
        
        if start_idx is None or end_idx is None:
            continue
        
        # Extract event segment
        event_trace = trace[start_idx:end_idx+1]
        event_baseline = baseline[start_idx:end_idx+1] if not np.isscalar(baseline) else baseline
        
        # Compute ΔF/F
        dff = (event_trace - event_baseline) / np.maximum(event_baseline, 1.0)
        
        # Peak amplitude
        peak_amplitude = np.max(dff)
        peak_relative_idx = np.argmax(dff)
        
        # Rise time (10-90%)
        rise_time, rise_rate = _compute_rise_time(dff[:peak_relative_idx+1], dt)
        
        # Decay time (exponential fit)
        decay_tau, decay_half_time = _compute_decay_time(dff[peak_relative_idx:], dt)
        
        # Half-width (FWHM)
        fwhm = _compute_fwhm(dff, dt)
        
        # Event duration (time above 10% of peak)
        duration = _compute_duration(dff, dt, threshold=0.1)
        
        # Area under curve
        auc = np.trapz(dff, dx=dt)
        
        # Compile results
        result = {
            'peak_idx': peak_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'peak_amplitude': peak_amplitude,
            'rise_time_10_90': rise_time,
            'rise_rate': rise_rate,
            'decay_tau': decay_tau,
            'decay_half_time': decay_half_time,
            'fwhm': fwhm,
            'duration': duration,
            'auc': auc,
            'event_trace': event_trace,
            'dff_trace': dff
        }
        
        results.append(result)
    
    return results


def _find_event_start(trace, peak_idx, baseline, threshold_factor=0.1):
    """Find event start by searching backward from peak."""
    if np.isscalar(baseline):
        threshold = baseline + threshold_factor * (trace[peak_idx] - baseline)
    else:
        threshold = baseline[peak_idx] + threshold_factor * (trace[peak_idx] - baseline[peak_idx])
    
    for i in range(peak_idx, max(0, peak_idx - 100), -1):
        if trace[i] < threshold:
            return i
    return max(0, peak_idx - 50)  # Default if not found


def _find_event_end(trace, peak_idx, baseline, threshold_factor=0.1):
    """Find event end by searching forward from peak."""
    if np.isscalar(baseline):
        threshold = baseline + threshold_factor * (trace[peak_idx] - baseline)
    else:
        threshold = baseline[peak_idx] + threshold_factor * (trace[peak_idx] - baseline[peak_idx])
    
    for i in range(peak_idx, min(len(trace), peak_idx + 100)):
        if trace[i] < threshold:
            return i
    return min(len(trace) - 1, peak_idx + 50)  # Default if not found


def _compute_rise_time(rising_phase, dt):
    """Compute 10-90% rise time."""
    if len(rising_phase) < 3:
        return None, None
    
    peak = rising_phase[-1]
    threshold_10 = 0.1 * peak
    threshold_90 = 0.9 * peak
    
    # Find crossing points
    idx_10 = np.where(rising_phase >= threshold_10)[0]
    idx_90 = np.where(rising_phase >= threshold_90)[0]
    
    if len(idx_10) == 0 or len(idx_90) == 0:
        return None, None
    
    t_10 = idx_10[0]
    t_90 = idx_90[0]
    
    rise_time = (t_90 - t_10) * dt
    rise_rate = (threshold_90 - threshold_10) / rise_time if rise_time > 0 else None
    
    return rise_time, rise_rate


def _compute_decay_time(decay_phase, dt):
    """Fit exponential to decay phase and extract tau."""
    if len(decay_phase) < 5:
        return None, None
    
    def exp_decay(t, A, tau, C):
        return A * np.exp(-t / tau) + C
    
    t = np.arange(len(decay_phase)) * dt
    y = decay_phase
    
    try:
        # Initial guess
        A0 = y[0] - y[-1]
        tau0 = len(decay_phase) * dt / 3
        C0 = y[-1]
        
        popt, _ = curve_fit(exp_decay, t, y, p0=[A0, tau0, C0], maxfev=1000)
        tau = popt[1]
        half_time = tau * np.log(2)
        
        return tau, half_time
    except:
        # Fallback: use 1/e time
        target = y[0] / np.e
        idx = np.where(y <= target)[0]
        if len(idx) > 0:
            tau = idx[0] * dt
            return tau, tau * np.log(2)
        return None, None


def _compute_fwhm(trace, dt):
    """Compute full-width at half-maximum."""
    peak = np.max(trace)
    half_max = peak / 2
    
    above_half = trace >= half_max
    if not np.any(above_half):
        return None
    
    indices = np.where(above_half)[0]
    fwhm = (indices[-1] - indices[0] + 1) * dt
    
    return fwhm


def _compute_duration(trace, dt, threshold=0.1):
    """Compute duration above threshold fraction of peak."""
    peak = np.max(trace)
    threshold_val = threshold * peak
    
    above_threshold = trace >= threshold_val
    if not np.any(above_threshold):
        return None
    
    indices = np.where(above_threshold)[0]
    duration = (indices[-1] - indices[0] + 1) * dt
    
    return duration


def compute_population_statistics(kinetics_list):
    """
    Compute population statistics across multiple events.
    
    Parameters:
        kinetics_list (list): List of kinetics dictionaries from extract_event_kinetics
    
    Returns:
        dict: Dictionary of mean, std, median for each kinetic parameter
    
    Example:
        >>> kinetics = extract_event_kinetics(trace, fs=30.0, event_indices=events)
        >>> stats = compute_population_statistics(kinetics)
        >>> print(f"Mean rise time: {stats['rise_time_10_90']['mean']:.3f} s")
    """
    if len(kinetics_list) == 0:
        return {}
    
    # Extract each parameter
    parameters = ['peak_amplitude', 'rise_time_10_90', 'rise_rate', 
                  'decay_tau', 'decay_half_time', 'fwhm', 'duration', 'auc']
    
    stats = {}
    for param in parameters:
        values = [k[param] for k in kinetics_list if k.get(param) is not None]
        
        if len(values) > 0:
            stats[param] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
    
    return stats


def classify_events_by_kinetics(kinetics_list, 
                                fast_decay_threshold=0.2,
                                slow_decay_threshold=0.5):
    """
    Classify events into categories based on kinetic properties.
    
    Categories:
    - 'fast': Rapid rise and decay (blips)
    - 'slow': Slower kinetics (puffs)
    - 'sustained': Long-lasting events (waves)
    
    Parameters:
        kinetics_list (list): List of kinetics dictionaries
        fast_decay_threshold (float): Threshold for fast decay (seconds)
        slow_decay_threshold (float): Threshold for slow decay (seconds)
    
    Returns:
        dict: Dictionary mapping event indices to categories
    
    Example:
        >>> kinetics = extract_event_kinetics(trace, fs=30.0, event_indices=events)
        >>> categories = classify_events_by_kinetics(kinetics)
        >>> print(categories)  # {0: 'fast', 1: 'slow', ...}
    """
    categories = {}
    
    for i, event in enumerate(kinetics_list):
        decay_tau = event.get('decay_tau')
        duration = event.get('duration')
        
        if decay_tau is None or duration is None:
            categories[i] = 'unknown'
            continue
        
        # Classify based on kinetics
        if decay_tau < fast_decay_threshold:
            categories[i] = 'fast'
        elif decay_tau < slow_decay_threshold:
            categories[i] = 'slow'
        else:
            categories[i] = 'sustained'
    
    return categories


def compute_inter_event_intervals(event_indices, fs):
    """
    Compute inter-event intervals (IEI) between consecutive events.
    
    Parameters:
        event_indices (ndarray): Array of event peak frame indices
        fs (float): Sampling frequency (Hz)
    
    Returns:
        dict: Dictionary containing:
            - 'iei': Array of inter-event intervals (seconds)
            - 'frequency': Mean event frequency (Hz)
            - 'cv': Coefficient of variation of IEI (regularity measure)
    
    Example:
        >>> events = np.array([100, 250, 420, 550])
        >>> iei_stats = compute_inter_event_intervals(events, fs=30.0)
        >>> print(f"Mean IEI: {np.mean(iei_stats['iei']):.3f} s")
    """
    if len(event_indices) < 2:
        return {'iei': np.array([]), 'frequency': 0, 'cv': None}
    
    # Sort events
    sorted_events = np.sort(event_indices)
    
    # Compute intervals in frames, convert to seconds
    intervals_frames = np.diff(sorted_events)
    intervals_sec = intervals_frames / fs
    
    # Compute statistics
    mean_iei = np.mean(intervals_sec)
    frequency = 1.0 / mean_iei if mean_iei > 0 else 0
    cv = np.std(intervals_sec) / mean_iei if mean_iei > 0 else None
    
    return {
        'iei': intervals_sec,
        'frequency': frequency,
        'cv': cv,
        'mean_iei': mean_iei,
        'std_iei': np.std(intervals_sec)
    }


def compute_amplitude_distribution(kinetics_list, bins=20):
    """
    Compute amplitude distribution histogram.
    
    Parameters:
        kinetics_list (list): List of kinetics dictionaries
        bins (int): Number of histogram bins
    
    Returns:
        dict: Dictionary containing:
            - 'bin_centers': Array of bin center values
            - 'counts': Array of event counts per bin
            - 'frequencies': Normalized frequencies (probability density)
    
    Example:
        >>> kinetics = extract_event_kinetics(trace, fs=30.0, event_indices=events)
        >>> amp_dist = compute_amplitude_distribution(kinetics, bins=20)
        >>> plt.bar(amp_dist['bin_centers'], amp_dist['counts'])
    """
    amplitudes = [k['peak_amplitude'] for k in kinetics_list if 'peak_amplitude' in k]
    
    if len(amplitudes) == 0:
        return {'bin_centers': np.array([]), 'counts': np.array([]), 'frequencies': np.array([])}
    
    counts, bin_edges = np.histogram(amplitudes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    frequencies = counts / np.sum(counts)
    
    return {
        'bin_centers': bin_centers,
        'counts': counts,
        'frequencies': frequencies,
        'amplitudes': np.array(amplitudes)
    }


if __name__ == '__main__':
    """Unit tests for event kinetics module."""
    print("Testing event_kinetics_module...")
    
    # Test 1: Extract event kinetics
    print("\n1. Testing extract_event_kinetics...")
    np.random.seed(42)
    fs = 30.0
    dt = 1.0 / fs
    
    # Create synthetic event
    t = np.arange(1000)
    trace = np.zeros(1000) + 100
    
    # Add exponential rise-decay event
    event_t = np.arange(150)
    rise = 1 - np.exp(-event_t[:50] / 10)
    decay = np.exp(-(event_t[50:] - 50) / 30)
    event = np.concatenate([rise, decay]) * 50
    trace[200:350] += event
    
    event_indices = np.array([225])  # Peak around frame 225
    kinetics = extract_event_kinetics(trace, fs, event_indices)
    
    print(f"   Number of events analyzed: {len(kinetics)}")
    if len(kinetics) > 0:
        print(f"   Peak amplitude: {kinetics[0]['peak_amplitude']:.4f}")
        print(f"   Rise time: {kinetics[0]['rise_time_10_90']:.4f} s")
        print(f"   Decay tau: {kinetics[0]['decay_tau']:.4f} s")
        print(f"   FWHM: {kinetics[0]['fwhm']:.4f} s")
        assert kinetics[0]['peak_amplitude'] > 0, "Invalid amplitude"
    print("   ✓ Event kinetics extraction working")
    
    # Test 2: Population statistics
    print("\n2. Testing compute_population_statistics...")
    # Create multiple events
    event_indices_multi = np.array([225, 500, 750])
    trace[500:600] += 30 * np.exp(-np.arange(100) / 20)
    trace[750:850] += 40 * np.exp(-np.arange(100) / 25)
    
    kinetics_multi = extract_event_kinetics(trace, fs, event_indices_multi)
    stats = compute_population_statistics(kinetics_multi)
    
    print(f"   Number of events: {len(kinetics_multi)}")
    if 'peak_amplitude' in stats:
        print(f"   Mean amplitude: {stats['peak_amplitude']['mean']:.4f}")
        print(f"   Std amplitude: {stats['peak_amplitude']['std']:.4f}")
    print("   ✓ Population statistics computed")
    
    # Test 3: Event classification
    print("\n3. Testing classify_events_by_kinetics...")
    categories = classify_events_by_kinetics(kinetics_multi)
    print(f"   Event categories: {categories}")
    assert len(categories) == len(kinetics_multi), "Classification mismatch"
    print("   ✓ Event classification working")
    
    # Test 4: Inter-event intervals
    print("\n4. Testing compute_inter_event_intervals...")
    iei_stats = compute_inter_event_intervals(event_indices_multi, fs)
    print(f"   Number of intervals: {len(iei_stats['iei'])}")
    print(f"   Mean IEI: {iei_stats['mean_iei']:.4f} s")
    print(f"   Frequency: {iei_stats['frequency']:.4f} Hz")
    assert len(iei_stats['iei']) == len(event_indices_multi) - 1, "IEI count mismatch"
    print("   ✓ Inter-event interval computation working")
    
    # Test 5: Amplitude distribution
    print("\n5. Testing compute_amplitude_distribution...")
    amp_dist = compute_amplitude_distribution(kinetics_multi, bins=10)
    print(f"   Number of bins: {len(amp_dist['bin_centers'])}")
    print(f"   Total count: {np.sum(amp_dist['counts'])}")
    assert np.sum(amp_dist['counts']) == len(kinetics_multi), "Count mismatch"
    print("   ✓ Amplitude distribution computed")
    
    print("\n✅ All event kinetics module tests passed!")
