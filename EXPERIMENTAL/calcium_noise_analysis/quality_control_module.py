"""
Quality Control Module
======================
Data quality assessment and signal-to-noise ratio (SNR) evaluation for
calcium imaging experiments.

Provides metrics for:
- Signal quality (SNR, noise floor)
- Photobleaching assessment
- Motion artifacts detection
- Data reliability scores

Author: George
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr


def compute_snr_metrics(image_stack, fs, signal_roi=None, noise_roi=None):
    """
    Compute comprehensive signal-to-noise ratio metrics.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency (Hz)
        signal_roi (tuple): (y_min, y_max, x_min, x_max) for signal region
                           If None, uses entire image
        noise_roi (tuple): (y_min, y_max, x_min, x_max) for noise estimation
                          If None, uses first 10% of frames as baseline
    
    Returns:
        dict: Dictionary containing:
            - 'temporal_snr': Ratio of signal variance to noise variance
            - 'spatial_snr': Mean signal divided by spatial noise std
            - 'peak_snr': Peak signal divided by baseline noise
            - 'psm_snr': SNR estimated from power spectrum
            - 'quality_score': Overall quality score (0-1)
    
    Example:
        >>> snr_metrics = compute_snr_metrics(stack, fs=30.0)
        >>> print(f"Temporal SNR: {snr_metrics['temporal_snr']:.2f}")
        >>> print(f"Quality score: {snr_metrics['quality_score']:.2%}")
    """
    T, H, W = image_stack.shape
    
    # Extract signal region
    if signal_roi is None:
        signal = np.mean(image_stack, axis=(1, 2))
    else:
        y1, y2, x1, x2 = signal_roi
        signal = np.mean(image_stack[:, y1:y2, x1:x2], axis=(1, 2))
    
    # Estimate noise
    if noise_roi is None:
        # Use first 10% as baseline noise
        noise_baseline = signal[:T//10]
    else:
        y1, y2, x1, x2 = noise_roi
        noise_baseline = np.mean(image_stack[:T//10, y1:y2, x1:x2], axis=(1, 2))
    
    # Temporal SNR
    signal_var = np.var(signal)
    noise_var = np.var(noise_baseline)
    temporal_snr = signal_var / noise_var if noise_var > 0 else 0
    
    # Spatial SNR
    mean_signal = np.mean(signal)
    spatial_noise_std = np.std(image_stack[:T//10], axis=0).mean()
    spatial_snr = mean_signal / spatial_noise_std if spatial_noise_std > 0 else 0
    
    # Peak SNR
    peak_signal = np.max(signal)
    baseline_noise_std = np.std(noise_baseline)
    peak_snr = (peak_signal - mean_signal) / baseline_noise_std if baseline_noise_std > 0 else 0
    
    # Power spectrum SNR
    psm_snr = _compute_psm_snr(signal, fs)
    
    # Overall quality score (weighted combination)
    quality_score = _compute_quality_score(temporal_snr, spatial_snr, peak_snr, psm_snr)
    
    return {
        'temporal_snr': temporal_snr,
        'spatial_snr': spatial_snr,
        'peak_snr': peak_snr,
        'psm_snr': psm_snr,
        'quality_score': quality_score,
        'signal_trace': signal,
        'noise_baseline': noise_baseline
    }


def _compute_psm_snr(trace, fs):
    """Estimate SNR from power spectrum (signal band vs noise band)."""
    freqs, psd = welch(trace, fs=fs, nperseg=min(256, len(trace)//4))
    
    # Signal band: 0.1-5 Hz (typical Ca²⁺ signals)
    signal_mask = (freqs >= 0.1) & (freqs <= 5.0)
    # Noise band: >50 Hz (shot noise)
    noise_mask = freqs > 50
    
    if np.any(signal_mask) and np.any(noise_mask):
        signal_power = np.mean(psd[signal_mask])
        noise_power = np.mean(psd[noise_mask])
        return signal_power / noise_power if noise_power > 0 else 0
    return 0


def _compute_quality_score(temporal_snr, spatial_snr, peak_snr, psm_snr):
    """Compute normalized quality score from SNR metrics."""
    # Normalize each metric (empirical thresholds)
    t_score = min(temporal_snr / 10.0, 1.0)
    s_score = min(spatial_snr / 50.0, 1.0)
    p_score = min(peak_snr / 20.0, 1.0)
    psm_score = min(psm_snr / 10.0, 1.0)
    
    # Weighted average
    weights = [0.3, 0.2, 0.3, 0.2]
    quality_score = np.average([t_score, s_score, p_score, psm_score], weights=weights)
    
    return quality_score


def assess_photobleaching(image_stack, method='exponential'):
    """
    Assess photobleaching in image stack.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        method (str): 'exponential' or 'linear'
    
    Returns:
        dict: Dictionary containing:
            - 'bleach_detected': Boolean indicating significant bleaching
            - 'bleach_rate': Rate of fluorescence decay (per frame)
            - 'bleach_half_time': Time to 50% fluorescence (frames)
            - 'bleach_percent': Percentage fluorescence loss
            - 'fitted_curve': Fitted bleaching curve
    
    Example:
        >>> bleach_info = assess_photobleaching(stack)
        >>> if bleach_info['bleach_detected']:
        >>>     print(f"Bleaching: {bleach_info['bleach_percent']:.1f}%")
    """
    T, H, W = image_stack.shape
    
    # Compute mean fluorescence over time
    mean_fluorescence = np.mean(image_stack, axis=(1, 2))
    
    t = np.arange(T)
    
    # Fit decay model
    if method == 'exponential':
        from scipy.optimize import curve_fit
        
        def exp_decay(t, F0, tau, C):
            return F0 * np.exp(-t / tau) + C
        
        try:
            F0_guess = mean_fluorescence[0]
            tau_guess = T / 2
            C_guess = mean_fluorescence[-1]
            
            popt, _ = curve_fit(exp_decay, t, mean_fluorescence, 
                               p0=[F0_guess, tau_guess, C_guess],
                               maxfev=5000)
            
            F0, tau, C = popt
            fitted_curve = exp_decay(t, F0, tau, C)
            
            bleach_rate = -1 / tau  # per frame
            bleach_half_time = tau * np.log(2)
            
        except:
            # Fallback to linear
            coeffs = np.polyfit(t, mean_fluorescence, deg=1)
            fitted_curve = np.polyval(coeffs, t)
            bleach_rate = coeffs[0] / mean_fluorescence[0]  # Fractional per frame
            bleach_half_time = 0.5 * mean_fluorescence[0] / abs(coeffs[0]) if coeffs[0] != 0 else np.inf
            
    else:  # Linear
        coeffs = np.polyfit(t, mean_fluorescence, deg=1)
        fitted_curve = np.polyval(coeffs, t)
        bleach_rate = coeffs[0] / mean_fluorescence[0]  # Fractional per frame
        bleach_half_time = 0.5 * mean_fluorescence[0] / abs(coeffs[0]) if coeffs[0] != 0 else np.inf
    
    # Assess significance
    bleach_percent = 100 * (1 - fitted_curve[-1] / fitted_curve[0])
    bleach_detected = abs(bleach_percent) > 10  # >10% change
    
    return {
        'bleach_detected': bleach_detected,
        'bleach_rate': bleach_rate,
        'bleach_half_time': bleach_half_time,
        'bleach_percent': bleach_percent,
        'fitted_curve': fitted_curve,
        'mean_fluorescence': mean_fluorescence
    }


def detect_motion_artifacts(image_stack, threshold=0.5):
    """
    Detect motion artifacts using frame-to-frame correlation.
    
    Sudden drops in correlation indicate movement or artifacts.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        threshold (float): Correlation threshold below which motion is flagged
    
    Returns:
        dict: Dictionary containing:
            - 'motion_detected': Boolean array for each frame
            - 'correlation_trace': Frame-to-frame correlation values
            - 'artifact_frames': List of frame indices with artifacts
            - 'quality_frames': Percentage of good frames
    
    Example:
        >>> motion_info = detect_motion_artifacts(stack, threshold=0.8)
        >>> print(f"Artifact frames: {len(motion_info['artifact_frames'])}")
        >>> print(f"Quality: {motion_info['quality_frames']:.1f}%")
    """
    T, H, W = image_stack.shape
    
    correlations = np.zeros(T - 1)
    
    for t in range(T - 1):
        frame1 = image_stack[t].flatten()
        frame2 = image_stack[t + 1].flatten()
        
        corr, _ = pearsonr(frame1, frame2)
        correlations[t] = corr
    
    # Detect artifacts
    motion_detected = correlations < threshold
    artifact_frames = np.where(motion_detected)[0].tolist()
    quality_frames = 100 * (1 - np.sum(motion_detected) / len(motion_detected))
    
    return {
        'motion_detected': motion_detected,
        'correlation_trace': correlations,
        'artifact_frames': artifact_frames,
        'quality_frames': quality_frames
    }


def compute_cascade_noise_metric(dff_stack, fs):
    """
    Compute CASCADE noise metric (ν) for quality assessment.
    
    ν = σ_ΔF/F × √framerate
    
    Typical values:
    - ν ≈ 1-2: Excellent quality
    - ν ≈ 3-5: Good quality  
    - ν ≈ 6-9: Moderate quality
    - ν > 10: Poor quality
    
    Parameters:
        dff_stack (ndarray): ΔF/F image stack (T, H, W)
        fs (float): Sampling frequency (Hz)
    
    Returns:
        dict: Dictionary containing:
            - 'nu': CASCADE noise metric ν
            - 'nu_map': Spatial map of ν values (H, W)
            - 'quality_category': 'excellent', 'good', 'moderate', or 'poor'
    
    Example:
        >>> cascade_metrics = compute_cascade_noise_metric(dff_stack, fs=30.0)
        >>> print(f"ν = {cascade_metrics['nu']:.2f}")
        >>> print(f"Quality: {cascade_metrics['quality_category']}")
    """
    T, H, W = dff_stack.shape
    
    # Compute baseline standard deviation (first 100 frames)
    baseline = dff_stack[:min(100, T)]
    sigma_dff = np.std(baseline, axis=0)
    
    # CASCADE noise metric
    nu_map = sigma_dff * np.sqrt(fs)
    nu = np.median(nu_map)  # Use median for robustness
    
    # Categorize quality
    if nu < 2:
        quality_category = 'excellent'
    elif nu < 5:
        quality_category = 'good'
    elif nu < 10:
        quality_category = 'moderate'
    else:
        quality_category = 'poor'
    
    return {
        'nu': nu,
        'nu_map': nu_map,
        'quality_category': quality_category,
        'sigma_dff': sigma_dff
    }


def generate_quality_report(image_stack, fs, dff_stack=None):
    """
    Generate comprehensive quality control report.
    
    Combines all QC metrics into single report.
    
    Parameters:
        image_stack (ndarray): Raw fluorescence stack (T, H, W)
        fs (float): Sampling frequency (Hz)
        dff_stack (ndarray): ΔF/F stack (T, H, W), optional
    
    Returns:
        dict: Comprehensive quality report with all metrics
    
    Example:
        >>> report = generate_quality_report(stack, fs=30.0, dff_stack=dff)
        >>> print(f"Overall quality: {report['overall_quality']}")
        >>> for warning in report['warnings']:
        >>>     print(f"⚠ {warning}")
    """
    report = {}
    warnings = []
    
    # SNR metrics
    snr_metrics = compute_snr_metrics(image_stack, fs)
    report['snr'] = snr_metrics
    
    if snr_metrics['quality_score'] < 0.3:
        warnings.append(f"Low SNR (quality score: {snr_metrics['quality_score']:.2f})")
    
    # Photobleaching
    bleach_info = assess_photobleaching(image_stack)
    report['photobleaching'] = bleach_info
    
    if bleach_info['bleach_detected']:
        warnings.append(f"Significant photobleaching ({bleach_info['bleach_percent']:.1f}%)")
    
    # Motion artifacts
    motion_info = detect_motion_artifacts(image_stack)
    report['motion'] = motion_info
    
    if motion_info['quality_frames'] < 90:
        warnings.append(f"Motion artifacts ({100-motion_info['quality_frames']:.1f}% of frames)")
    
    # CASCADE metric
    if dff_stack is not None:
        cascade_metrics = compute_cascade_noise_metric(dff_stack, fs)
        report['cascade'] = cascade_metrics
        
        if cascade_metrics['nu'] > 10:
            warnings.append(f"High CASCADE noise (ν={cascade_metrics['nu']:.1f})")
    
    # Overall quality assessment
    quality_factors = [snr_metrics['quality_score']]
    if bleach_info['bleach_detected']:
        quality_factors.append(0.7)
    if motion_info['quality_frames'] < 90:
        quality_factors.append(0.6)
    
    overall_quality = np.mean(quality_factors)
    
    if overall_quality >= 0.7:
        overall_category = 'good'
    elif overall_quality >= 0.5:
        overall_category = 'acceptable'
    else:
        overall_category = 'poor'
    
    report['overall_quality'] = overall_quality
    report['overall_category'] = overall_category
    report['warnings'] = warnings
    
    return report


if __name__ == '__main__':
    """Unit tests for quality control module."""
    print("Testing quality_control_module...")
    
    # Test 1: SNR metrics
    print("\n1. Testing compute_snr_metrics...")
    np.random.seed(42)
    T, H, W = 500, 64, 64
    fs = 30.0
    
    # Create clean signal
    stack = np.random.randn(T, H, W) * 5 + 100
    # Add Ca²⁺ transient
    signal = 50 * np.exp(-(np.arange(T) - 250)**2 / 1000)
    stack += signal[:, np.newaxis, np.newaxis]
    
    snr_metrics = compute_snr_metrics(stack, fs)
    print(f"   Temporal SNR: {snr_metrics['temporal_snr']:.2f}")
    print(f"   Spatial SNR: {snr_metrics['spatial_snr']:.2f}")
    print(f"   Quality score: {snr_metrics['quality_score']:.2%}")
    assert 0 <= snr_metrics['quality_score'] <= 1, "Invalid quality score"
    print("   ✓ SNR metrics computed")
    
    # Test 2: Photobleaching assessment
    print("\n2. Testing assess_photobleaching...")
    # Add exponential decay (bleaching)
    bleach = np.exp(-np.arange(T) / 200)
    stack_bleached = stack * bleach[:, np.newaxis, np.newaxis]
    
    bleach_info = assess_photobleaching(stack_bleached)
    print(f"   Bleaching detected: {bleach_info['bleach_detected']}")
    print(f"   Bleach percent: {bleach_info['bleach_percent']:.1f}%")
    print(f"   Half-time: {bleach_info['bleach_half_time']:.1f} frames")
    assert bleach_info['bleach_detected'], "Should detect bleaching"
    print("   ✓ Photobleaching assessment working")
    
    # Test 3: Motion artifact detection
    print("\n3. Testing detect_motion_artifacts...")
    # Add motion artifact (shift frame)
    stack_motion = stack.copy()
    stack_motion[250] = np.roll(stack_motion[250], shift=10, axis=0)
    
    motion_info = detect_motion_artifacts(stack_motion, threshold=0.95)
    print(f"   Artifact frames: {len(motion_info['artifact_frames'])}")
    print(f"   Quality frames: {motion_info['quality_frames']:.1f}%")
    assert len(motion_info['artifact_frames']) > 0, "Should detect motion"
    print("   ✓ Motion artifact detection working")
    
    # Test 4: CASCADE noise metric
    print("\n4. Testing compute_cascade_noise_metric...")
    dff_stack = (stack - 100) / 100  # Simplified ΔF/F
    
    cascade_metrics = compute_cascade_noise_metric(dff_stack, fs)
    print(f"   ν = {cascade_metrics['nu']:.2f}")
    print(f"   Quality category: {cascade_metrics['quality_category']}")
    assert cascade_metrics['nu'] > 0, "Invalid ν value"
    print("   ✓ CASCADE metric computed")
    
    # Test 5: Quality report
    print("\n5. Testing generate_quality_report...")
    report = generate_quality_report(stack, fs, dff_stack)
    print(f"   Overall quality: {report['overall_quality']:.2%}")
    print(f"   Overall category: {report['overall_category']}")
    print(f"   Warnings: {len(report['warnings'])}")
    assert 'overall_quality' in report, "Missing overall quality"
    print("   ✓ Quality report generated")
    
    print("\n✅ All quality control module tests passed!")
