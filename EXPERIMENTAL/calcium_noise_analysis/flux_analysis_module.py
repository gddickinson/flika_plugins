"""
Calcium Flux Analysis Module
=============================
Analyze Ca2+ release rates and cumulative liberation from fluorescence traces.
Based on Lock & Parker 2020 methodology.

Computes:
- Instantaneous Ca2+ release flux
- Cumulative Ca2+ release
- Relative contributions of punctate vs. diffuse release

Author: George
"""

import numpy as np
from scipy.optimize import curve_fit


def estimate_clearance_rate(decay_trace, method='exponential'):
    """
    Estimate Ca2+ clearance rate constant from decay phase.
    
    Fits exponential decay to estimate first-order clearance rate k.
    F(t) = F0 * exp(-k*t)
    
    Parameters:
        decay_trace (ndarray): Fluorescence trace during decay (1D array)
        method (str): Fitting method ('exponential' or 'linear_log')
    
    Returns:
        float: Clearance rate constant k (s⁻¹)
    
    Example:
        >>> import numpy as np
        >>> t = np.arange(100) / 30.0  # 30 Hz sampling
        >>> decay = 100 * np.exp(-0.3 * t) + 10
        >>> k = estimate_clearance_rate(decay)
        >>> print(f"Clearance rate: {k:.3f} s⁻¹")
    
    Notes:
        - Use tail-end of Ca2+ response when release has ceased
        - Assumes first-order clearance kinetics
        - Typical HEK values: 0.2-0.6 s⁻¹
    """
    if method == 'exponential':
        # Exponential fit
        def exp_decay(t, F0, k):
            return F0 * np.exp(-k * t)
        
        t = np.arange(len(decay_trace))
        
        try:
            # Initial guess
            p0 = [decay_trace[0], 0.3]
            
            # Fit
            popt, _ = curve_fit(exp_decay, t, decay_trace, p0=p0,
                               bounds=([0, 0], [np.inf, 10.0]))
            
            k = popt[1]
            
        except RuntimeError:
            print("Exponential fit failed, using linear log method")
            method = 'linear_log'
    
    if method == 'linear_log':
        # Linear fit to log-transformed data
        # log(F) = log(F0) - k*t
        
        t = np.arange(len(decay_trace))
        
        # Avoid log(0)
        trace_safe = np.maximum(decay_trace, 1e-10)
        log_trace = np.log(trace_safe)
        
        # Linear fit
        coeffs = np.polyfit(t, log_trace, deg=1)
        k = -coeffs[0]  # Negative of slope
    
    return max(k, 0)


def compute_release_flux(trace, clearance_rate, dt=1.0):
    """
    Calculate instantaneous Ca2+ release flux from fluorescence trace.
    
    Release flux = dF/dt + k*F(t)
    
    where k is the clearance rate constant.
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array, ΔF/F0 or F)
        clearance_rate (float): Clearance rate constant k (s⁻¹)
        dt (float): Time step (seconds)
    
    Returns:
        ndarray: Instantaneous release flux (same units as trace)
    
    Example:
        >>> import numpy as np
        >>> trace = np.array([0, 1, 2, 2.5, 2.8, 2.9, 2.85, 2.7])
        >>> k = 0.3
        >>> flux = compute_release_flux(trace, k, dt=1/30.0)
        >>> print(flux)
    
    Notes:
        - Positive flux = Ca2+ release
        - Negative flux = net removal (clearance > release)
        - Units match input trace units per second
    """
    # Compute derivative dF/dt
    dF_dt = np.gradient(trace, dt)
    
    # Compute clearance term k*F(t)
    clearance_flux = clearance_rate * trace
    
    # Total flux = change + clearance
    release_flux = dF_dt + clearance_flux
    
    return release_flux


def cumulative_release(release_flux, dt=1.0):
    """
    Calculate cumulative Ca2+ release by integrating flux.
    
    Parameters:
        release_flux (ndarray): Instantaneous release flux (1D array)
        dt (float): Time step (seconds)
    
    Returns:
        ndarray: Cumulative release over time
    
    Example:
        >>> import numpy as np
        >>> flux = np.array([0, 5, 10, 8, 5, 2, 1, 0])
        >>> cumul = cumulative_release(flux, dt=1/30.0)
        >>> print(cumul[-1])  # Total release
    """
    return np.cumsum(release_flux) * dt


def punctate_vs_diffuse_release(trace, puff_times, clearance_rate, 
                                baseline_end, peak_time, dt=1.0):
    """
    Estimate relative contributions of punctate vs diffuse Ca2+ release.
    
    Based on Lock & Parker 2020 Fig. 8 analysis.
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array)
        puff_times (ndarray): Times when puff activity present (boolean array)
        clearance_rate (float): Clearance rate constant (s⁻¹)
        baseline_end (int): Frame index where stimulation begins
        peak_time (int): Frame index of peak Ca2+ response
        dt (float): Time step (seconds)
    
    Returns:
        dict: Dictionary with punctate and diffuse contributions
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000) * 2 + 5
        >>> puff_times = np.zeros(1000, dtype=bool)
        >>> puff_times[100:300] = True  # Puff flurry during rising phase
        >>> result = punctate_vs_diffuse_release(trace, puff_times, k=0.3, 
        ...                                      baseline_end=100, peak_time=400)
        >>> print(f"Punctate: {result['punctate_percent']:.1f}%")
    
    Notes:
        - Punctate = Ca2+ released during puff flurry
        - Diffuse = Ca2+ released when puffs absent
        - Analysis from baseline to peak of response
    """
    # Compute release flux
    flux = compute_release_flux(trace, clearance_rate, dt=dt)
    
    # Analyze from baseline_end to peak_time
    analysis_flux = flux[baseline_end:peak_time+1]
    analysis_puff_times = puff_times[baseline_end:peak_time+1]
    
    # Separate punctate and diffuse components
    punctate_flux = np.where(analysis_puff_times, analysis_flux, 0)
    diffuse_flux = np.where(~analysis_puff_times, analysis_flux, 0)
    
    # Integrate to get total release
    total_punctate = np.sum(punctate_flux) * dt
    total_diffuse = np.sum(diffuse_flux) * dt
    total_release = total_punctate + total_diffuse
    
    # Calculate percentages
    if total_release > 0:
        punctate_percent = (total_punctate / total_release) * 100
        diffuse_percent = (total_diffuse / total_release) * 100
    else:
        punctate_percent = 0
        diffuse_percent = 0
    
    return {
        'total_punctate': total_punctate,
        'total_diffuse': total_diffuse,
        'total_release': total_release,
        'punctate_percent': punctate_percent,
        'diffuse_percent': diffuse_percent,
        'punctate_flux': punctate_flux,
        'diffuse_flux': diffuse_flux
    }


def analyze_release_kinetics(trace, baseline_frames, fs, 
                             detect_puffs_func=None):
    """
    Complete analysis of Ca2+ release kinetics from fluorescence trace.
    
    Performs:
    1. Clearance rate estimation from tail
    2. Release flux calculation
    3. Cumulative release
    4. Punctate vs diffuse separation (if puff detection provided)
    
    Parameters:
        trace (ndarray): Fluorescence trace (1D array, ΔF/F0)
        baseline_frames (int): Number of baseline frames before stimulation
        fs (float): Sampling frequency (Hz)
        detect_puffs_func (callable): Optional function to detect puff times
                                     Should return boolean array same length as trace
    
    Returns:
        dict: Dictionary with all analysis results
    
    Example:
        >>> import numpy as np
        >>> # Create synthetic response
        >>> t = np.arange(1000) / 30.0
        >>> trace = 5 * (1 - np.exp(-t/0.5)) * np.exp(-t/5.0)
        >>> result = analyze_release_kinetics(trace, baseline_frames=100, fs=30.0)
        >>> print(f"Peak release rate: {result['peak_flux']:.2f}")
    """
    dt = 1.0 / fs
    
    # Find peak
    peak_idx = np.argmax(trace)
    
    # Estimate clearance rate from tail (after peak)
    # Use frames from 80% down to 20% of peak on falling phase
    peak_val = trace[peak_idx]
    threshold_80 = peak_val * 0.8
    threshold_20 = peak_val * 0.2
    
    # Find indices
    after_peak = trace[peak_idx:]
    idx_80 = np.where(after_peak < threshold_80)[0]
    idx_20 = np.where(after_peak < threshold_20)[0]
    
    if len(idx_80) > 0 and len(idx_20) > 0:
        start_idx = peak_idx + idx_80[0]
        end_idx = peak_idx + idx_20[0]
        
        if end_idx > start_idx:
            decay_trace = trace[start_idx:end_idx]
            k = estimate_clearance_rate(decay_trace)
        else:
            k = 0.3  # Default
    else:
        k = 0.3  # Default
    
    # Compute release flux
    flux = compute_release_flux(trace, k, dt=dt)
    
    # Cumulative release
    cumul = cumulative_release(flux, dt=dt)
    
    # Peak flux
    peak_flux = np.max(flux)
    peak_flux_time = np.argmax(flux)
    
    results = {
        'clearance_rate': k,
        'release_flux': flux,
        'cumulative_release': cumul,
        'peak_flux': peak_flux,
        'peak_flux_time': peak_flux_time * dt,
        'total_release': cumul[-1],
        'dt': dt
    }
    
    # If puff detection provided, separate punctate vs diffuse
    if detect_puffs_func is not None:
        puff_times = detect_puffs_func(trace)
        
        if len(puff_times) == len(trace):
            punc_diff = punctate_vs_diffuse_release(
                trace, puff_times, k, 
                baseline_frames, peak_idx, dt=dt
            )
            results.update(punc_diff)
    
    return results


def compare_release_conditions(traces_dict, fs, **kwargs):
    """
    Compare Ca2+ release between different experimental conditions.
    
    Parameters:
        traces_dict (dict): Dictionary of {condition_name: trace_array}
        fs (float): Sampling frequency (Hz)
        **kwargs: Additional arguments for analyze_release_kinetics
    
    Returns:
        dict: Comparison results for all conditions
    
    Example:
        >>> import numpy as np
        >>> traces = {
        ...     'control': np.random.randn(1000),
        ...     'treatment': np.random.randn(1000)
        ... }
        >>> comparison = compare_release_conditions(traces, fs=30.0)
    """
    results = {}
    
    for condition, trace in traces_dict.items():
        results[condition] = analyze_release_kinetics(trace, fs=fs, **kwargs)
    
    # Add comparative metrics
    if len(results) >= 2:
        conditions = list(results.keys())
        
        # Compare peak fluxes
        peak_fluxes = {c: results[c]['peak_flux'] for c in conditions}
        
        # Compare total release
        total_releases = {c: results[c]['total_release'] for c in conditions}
        
        # Compare clearance rates
        clearance_rates = {c: results[c]['clearance_rate'] for c in conditions}
        
        results['comparison'] = {
            'peak_fluxes': peak_fluxes,
            'total_releases': total_releases,
            'clearance_rates': clearance_rates
        }
    
    return results


if __name__ == '__main__':
    """Unit tests for flux analysis module."""
    print("Testing flux_analysis_module...")
    
    # Test 1: Clearance rate estimation
    print("\n1. Testing estimate_clearance_rate...")
    t = np.arange(100) / 30.0
    true_k = 0.3
    decay = 100 * np.exp(-true_k * t) + 10
    
    k_est = estimate_clearance_rate(decay, method='exponential')
    print(f"   True k: {true_k:.3f} s⁻¹")
    print(f"   Estimated k: {k_est:.3f} s⁻¹")
    assert abs(k_est - true_k) < 0.05, "Clearance rate estimation inaccurate"
    print("   ✓ Clearance rate estimation working correctly")
    
    # Test 2: Release flux calculation
    print("\n2. Testing compute_release_flux...")
    # Create rising trace
    trace = np.array([0, 1, 2, 3, 4, 4.5, 4.8, 4.9, 4.85, 4.7, 4.4, 4.0])
    k = 0.3
    dt = 1/30.0
    
    flux = compute_release_flux(trace, k, dt=dt)
    print(f"   Flux shape: {flux.shape}")
    print(f"   Peak flux: {flux.max():.2f}")
    assert flux.shape == trace.shape, "Shape mismatch"
    assert flux.max() > 0, "Peak flux should be positive"
    print("   ✓ Release flux calculation working correctly")
    
    # Test 3: Cumulative release
    print("\n3. Testing cumulative_release...")
    cumul = cumulative_release(flux, dt=dt)
    print(f"   Cumulative shape: {cumul.shape}")
    print(f"   Total release: {cumul[-1]:.2f}")
    assert cumul.shape == flux.shape, "Shape mismatch"
    assert np.all(np.diff(cumul) >= -1e-10), "Cumulative should be monotonic increasing"
    print("   ✓ Cumulative release working correctly")
    
    # Test 4: Punctate vs diffuse
    print("\n4. Testing punctate_vs_diffuse_release...")
    trace_full = np.concatenate([np.zeros(100), trace, trace[::-1], np.zeros(50)])
    puff_times = np.zeros(len(trace_full), dtype=bool)
    puff_times[105:115] = True  # Puffs during rise
    
    result = punctate_vs_diffuse_release(
        trace_full, puff_times, k, 
        baseline_end=100, peak_time=107, dt=dt
    )
    
    print(f"   Punctate: {result['punctate_percent']:.1f}%")
    print(f"   Diffuse: {result['diffuse_percent']:.1f}%")
    print(f"   Total: {result['punctate_percent'] + result['diffuse_percent']:.1f}%")
    assert abs(result['punctate_percent'] + result['diffuse_percent'] - 100) < 1, "Percentages should sum to 100"
    print("   ✓ Punctate vs diffuse analysis working correctly")
    
    # Test 5: Complete release kinetics analysis
    print("\n5. Testing analyze_release_kinetics...")
    t = np.arange(1000) / 30.0
    synthetic_trace = 5 * (1 - np.exp(-t/0.5)) * np.exp(-t/3.0)
    
    result = analyze_release_kinetics(synthetic_trace, baseline_frames=0, fs=30.0)
    
    print(f"   Clearance rate: {result['clearance_rate']:.3f} s⁻¹")
    print(f"   Peak flux: {result['peak_flux']:.2f}")
    print(f"   Total release: {result['total_release']:.2f}")
    assert 'release_flux' in result, "Missing release_flux"
    assert 'cumulative_release' in result, "Missing cumulative_release"
    print("   ✓ Complete release kinetics analysis working correctly")
    
    print("\n✅ All flux analysis module tests passed!")
