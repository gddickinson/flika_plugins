"""
Oscillation Analysis Module
===========================
Detection and characterization of oscillatory Ca²⁺ signals including
frequency analysis, amplitude modulation, and regularity metrics.

Useful for analyzing Ca²⁺ oscillations in response to hormones,
neurotransmitters, or other stimuli.

Author: George
"""

import numpy as np
from scipy.signal import find_peaks, welch, hilbert, butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit


def detect_oscillations(trace, fs, min_peak_distance=None, peak_prominence=0.1):
    """
    Detect oscillatory Ca²⁺ signals and extract peak times.
    
    Parameters:
        trace (ndarray): Ca²⁺ fluorescence trace (1D array)
        fs (float): Sampling frequency (Hz)
        min_peak_distance (float): Minimum time between peaks (seconds)
                                   If None, auto-determined from dominant frequency
        peak_prominence (float): Minimum prominence for peak detection (ΔF/F units)
    
    Returns:
        dict: Dictionary containing:
            - 'peaks': Array of peak frame indices
            - 'peak_times': Array of peak times (seconds)
            - 'peak_amplitudes': Array of peak amplitudes
            - 'oscillating': Boolean indicating if oscillations detected
            - 'n_peaks': Number of peaks detected
    
    Example:
        >>> trace = np.sin(2 * np.pi * 0.1 * np.arange(1000) / 30.0) + np.random.randn(1000) * 0.1
        >>> osc = detect_oscillations(trace, fs=30.0)
        >>> print(f"Detected {osc['n_peaks']} peaks")
        >>> print(f"Oscillating: {osc['oscillating']}")
    """
    # Auto-determine peak distance if not provided
    if min_peak_distance is None:
        # Use power spectrum to estimate dominant frequency
        freqs, psd = welch(trace, fs=fs, nperseg=min(256, len(trace)//4))
        if len(freqs) > 1:
            # Dominant frequency in 0.01-2 Hz range (typical Ca²⁺ oscillations)
            valid_freqs = (freqs >= 0.01) & (freqs <= 2.0)
            if np.any(valid_freqs):
                dominant_freq = freqs[valid_freqs][np.argmax(psd[valid_freqs])]
                min_peak_distance = 0.5 / dominant_freq  # Half period
            else:
                min_peak_distance = 1.0  # Default 1 second
        else:
            min_peak_distance = 1.0
    
    min_peak_distance_frames = int(min_peak_distance * fs)
    
    # Find peaks
    peaks, properties = find_peaks(trace, 
                                   distance=min_peak_distance_frames,
                                   prominence=peak_prominence)
    
    peak_times = peaks / fs
    peak_amplitudes = trace[peaks]
    
    # Determine if oscillating (at least 3 peaks)
    oscillating = len(peaks) >= 3
    
    return {
        'peaks': peaks,
        'peak_times': peak_times,
        'peak_amplitudes': peak_amplitudes,
        'oscillating': oscillating,
        'n_peaks': len(peaks)
    }


def compute_oscillation_frequency(trace, fs, method='fft'):
    """
    Compute dominant oscillation frequency.
    
    Parameters:
        trace (ndarray): Ca²⁺ fluorescence trace
        fs (float): Sampling frequency (Hz)
        method (str): 'fft' or 'welch' for frequency estimation
    
    Returns:
        dict: Dictionary containing:
            - 'dominant_frequency': Dominant frequency (Hz)
            - 'dominant_period': Dominant period (seconds)
            - 'frequency_spectrum': (freqs, power) arrays
            - 'peak_frequency_snr': SNR at dominant frequency
    
    Example:
        >>> freq_info = compute_oscillation_frequency(trace, fs=30.0)
        >>> print(f"Dominant frequency: {freq_info['dominant_frequency']:.3f} Hz")
        >>> print(f"Period: {freq_info['dominant_period']:.2f} s")
    """
    if method == 'fft':
        # FFT method
        N = len(trace)
        yf = fft(trace)
        xf = fftfreq(N, 1/fs)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])
    else:
        # Welch method (more robust to noise)
        xf, power = welch(trace, fs=fs, nperseg=min(256, len(trace)//4))
    
    # Focus on physiological range (0.01-2 Hz for Ca²⁺)
    valid_range = (xf >= 0.01) & (xf <= 2.0)
    
    if np.any(valid_range):
        valid_freqs = xf[valid_range]
        valid_power = power[valid_range]
        
        # Find dominant frequency
        dominant_idx = np.argmax(valid_power)
        dominant_frequency = valid_freqs[dominant_idx]
        dominant_period = 1.0 / dominant_frequency if dominant_frequency > 0 else np.inf
        
        # Compute SNR (peak power vs mean power in other frequencies)
        peak_power = valid_power[dominant_idx]
        other_power = np.concatenate([valid_power[:dominant_idx], 
                                     valid_power[dominant_idx+1:]])
        mean_other_power = np.mean(other_power) if len(other_power) > 0 else 1e-10
        peak_snr = peak_power / mean_other_power
    else:
        dominant_frequency = 0
        dominant_period = np.inf
        peak_snr = 0
    
    return {
        'dominant_frequency': dominant_frequency,
        'dominant_period': dominant_period,
        'frequency_spectrum': (xf, power),
        'peak_frequency_snr': peak_snr
    }


def compute_amplitude_modulation(trace, fs, window_size=10.0):
    """
    Compute amplitude modulation envelope using Hilbert transform.
    
    Extracts the slowly-varying amplitude envelope of oscillations.
    
    Parameters:
        trace (ndarray): Ca²⁺ fluorescence trace
        fs (float): Sampling frequency (Hz)
        window_size (float): Smoothing window size (seconds)
    
    Returns:
        dict: Dictionary containing:
            - 'envelope': Amplitude envelope trace
            - 'modulation_index': Degree of amplitude modulation (0-1)
            - 'mean_amplitude': Mean oscillation amplitude
            - 'amplitude_cv': Coefficient of variation of amplitude
    
    Example:
        >>> am = compute_amplitude_modulation(trace, fs=30.0)
        >>> print(f"Modulation index: {am['modulation_index']:.2f}")
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(trace)
        >>> plt.plot(am['envelope'], 'r--', label='Envelope')
    """
    # Apply Hilbert transform
    analytic_signal = hilbert(trace)
    envelope = np.abs(analytic_signal)
    
    # Smooth envelope
    window_samples = int(window_size * fs)
    from scipy.ndimage import uniform_filter1d
    envelope_smooth = uniform_filter1d(envelope, size=window_samples, mode='nearest')
    
    # Compute modulation metrics
    mean_amplitude = np.mean(envelope_smooth)
    amplitude_std = np.std(envelope_smooth)
    amplitude_cv = amplitude_std / mean_amplitude if mean_amplitude > 0 else 0
    
    # Modulation index: normalized range
    modulation_index = (envelope_smooth.max() - envelope_smooth.min()) / \
                      (envelope_smooth.max() + envelope_smooth.min()) \
                      if (envelope_smooth.max() + envelope_smooth.min()) > 0 else 0
    
    return {
        'envelope': envelope_smooth,
        'modulation_index': modulation_index,
        'mean_amplitude': mean_amplitude,
        'amplitude_cv': amplitude_cv
    }


def compute_regularity_metrics(peak_times):
    """
    Compute metrics quantifying regularity of oscillations.
    
    Parameters:
        peak_times (ndarray): Array of peak times (seconds)
    
    Returns:
        dict: Dictionary containing:
            - 'iei': Inter-event intervals (seconds)
            - 'mean_iei': Mean inter-event interval
            - 'cv_iei': Coefficient of variation (irregularity measure)
            - 'regularity_index': Regularity score (0=irregular, 1=perfectly regular)
    
    Example:
        >>> osc = detect_oscillations(trace, fs=30.0)
        >>> regularity = compute_regularity_metrics(osc['peak_times'])
        >>> print(f"CV: {regularity['cv_iei']:.2f} (lower = more regular)")
        >>> print(f"Regularity index: {regularity['regularity_index']:.2f}")
    
    Notes:
        - CV < 0.2: Regular oscillations
        - CV 0.2-0.5: Moderately irregular
        - CV > 0.5: Highly irregular
    """
    if len(peak_times) < 2:
        return {
            'iei': np.array([]),
            'mean_iei': 0,
            'cv_iei': 0,
            'regularity_index': 0
        }
    
    # Compute inter-event intervals
    iei = np.diff(peak_times)
    mean_iei = np.mean(iei)
    std_iei = np.std(iei)
    cv_iei = std_iei / mean_iei if mean_iei > 0 else 0
    
    # Regularity index (inverse of CV, normalized)
    regularity_index = 1.0 / (1.0 + cv_iei)
    
    return {
        'iei': iei,
        'mean_iei': mean_iei,
        'cv_iei': cv_iei,
        'regularity_index': regularity_index
    }


def compute_phase_coherence(trace1, trace2, fs, freq_band=(0.01, 2.0)):
    """
    Compute phase coherence between two oscillating signals.
    
    Measures synchronization between Ca²⁺ oscillations in different regions.
    
    Parameters:
        trace1 (ndarray): First Ca²⁺ trace
        trace2 (ndarray): Second Ca²⁺ trace
        fs (float): Sampling frequency (Hz)
        freq_band (tuple): (low, high) frequency band to analyze (Hz)
    
    Returns:
        dict: Dictionary containing:
            - 'coherence': Magnitude-squared coherence (0-1)
            - 'phase_difference': Mean phase difference (radians)
            - 'phase_locking_value': PLV measuring phase synchrony (0-1)
            - 'coherence_spectrum': (freqs, coherence) arrays
    
    Example:
        >>> coherence = compute_phase_coherence(trace_roi1, trace_roi2, fs=30.0)
        >>> print(f"Coherence: {coherence['coherence']:.2f}")
        >>> print(f"Phase diff: {coherence['phase_difference']:.2f} rad")
    """
    from scipy.signal import coherence
    
    # Compute coherence spectrum
    freqs, Cxy = coherence(trace1, trace2, fs=fs, nperseg=min(256, len(trace1)//4))
    
    # Average coherence in frequency band
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if np.any(band_mask):
        coherence_value = np.mean(Cxy[band_mask])
    else:
        coherence_value = 0
    
    # Bandpass filter to frequency band
    nyquist = fs / 2
    low_norm = freq_band[0] / nyquist
    high_norm = freq_band[1] / nyquist
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(0.001, min(high_norm, 0.999))
    
    b, a = butter(2, [low_norm, high_norm], btype='band')
    filtered1 = filtfilt(b, a, trace1)
    filtered2 = filtfilt(b, a, trace2)
    
    # Extract phase using Hilbert transform
    phase1 = np.angle(hilbert(filtered1))
    phase2 = np.angle(hilbert(filtered2))
    
    # Phase difference
    phase_diff = phase1 - phase2
    mean_phase_diff = np.angle(np.mean(np.exp(1j * phase_diff)))
    
    # Phase locking value (PLV)
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return {
        'coherence': coherence_value,
        'phase_difference': mean_phase_diff,
        'phase_locking_value': plv,
        'coherence_spectrum': (freqs, Cxy)
    }


def fit_oscillation_envelope(trace, fs, peak_times):
    """
    Fit exponential envelope to oscillation decay.
    
    Useful for analyzing Ca²⁺ oscillations that decay over time.
    
    Parameters:
        trace (ndarray): Ca²⁺ fluorescence trace
        fs (float): Sampling frequency (Hz)
        peak_times (ndarray): Array of peak times (seconds)
    
    Returns:
        dict: Dictionary containing:
            - 'decay_tau': Exponential decay time constant (seconds)
            - 'initial_amplitude': Initial amplitude
            - 'steady_state_amplitude': Steady-state amplitude
            - 'fitted_envelope': Fitted exponential curve
    
    Example:
        >>> osc = detect_oscillations(trace, fs=30.0)
        >>> envelope_fit = fit_oscillation_envelope(trace, fs, osc['peak_times'])
        >>> print(f"Decay tau: {envelope_fit['decay_tau']:.1f} s")
    """
    if len(peak_times) < 3:
        return {
            'decay_tau': None,
            'initial_amplitude': None,
            'steady_state_amplitude': None,
            'fitted_envelope': None
        }
    
    # Get peak amplitudes at peak times
    peak_indices = (peak_times * fs).astype(int)
    peak_indices = peak_indices[peak_indices < len(trace)]
    peak_amplitudes = trace[peak_indices]
    
    # Fit exponential decay
    def exp_decay(t, A0, tau, C):
        return A0 * np.exp(-t / tau) + C
    
    try:
        # Initial guesses
        A0_guess = peak_amplitudes[0] - peak_amplitudes[-1]
        tau_guess = peak_times[-1] / 2
        C_guess = peak_amplitudes[-1]
        
        popt, _ = curve_fit(exp_decay, peak_times, peak_amplitudes,
                           p0=[A0_guess, tau_guess, C_guess],
                           maxfev=5000)
        
        A0, tau, C = popt
        
        # Generate fitted curve for entire trace
        t_full = np.arange(len(trace)) / fs
        fitted_envelope = exp_decay(t_full, A0, tau, C)
        
        return {
            'decay_tau': tau,
            'initial_amplitude': A0 + C,
            'steady_state_amplitude': C,
            'fitted_envelope': fitted_envelope
        }
    except:
        return {
            'decay_tau': None,
            'initial_amplitude': None,
            'steady_state_amplitude': None,
            'fitted_envelope': None
        }


def classify_oscillation_pattern(trace, fs):
    """
    Classify oscillation pattern into categories.
    
    Categories:
    - 'sustained': Regular oscillations throughout
    - 'transient': Oscillations that decay
    - 'irregular': Irregular or noisy oscillations
    - 'none': No clear oscillations
    
    Parameters:
        trace (ndarray): Ca²⁺ fluorescence trace
        fs (float): Sampling frequency (Hz)
    
    Returns:
        dict: Dictionary containing:
            - 'pattern': Pattern classification string
            - 'confidence': Confidence score (0-1)
            - 'features': Dictionary of classification features
    
    Example:
        >>> pattern = classify_oscillation_pattern(trace, fs=30.0)
        >>> print(f"Pattern: {pattern['pattern']} (confidence: {pattern['confidence']:.2f})")
    """
    # Detect oscillations
    osc = detect_oscillations(trace, fs)
    
    if not osc['oscillating']:
        return {
            'pattern': 'none',
            'confidence': 0.9,
            'features': {}
        }
    
    # Compute features
    freq_info = compute_oscillation_frequency(trace, fs)
    regularity = compute_regularity_metrics(osc['peak_times'])
    am = compute_amplitude_modulation(trace, fs)
    
    # Classification logic
    features = {
        'n_peaks': osc['n_peaks'],
        'frequency': freq_info['dominant_frequency'],
        'cv_iei': regularity['cv_iei'],
        'modulation_index': am['modulation_index']
    }
    
    # Sustained: >5 peaks, regular (CV < 0.3), low modulation
    if (osc['n_peaks'] >= 5 and 
        regularity['cv_iei'] < 0.3 and 
        am['modulation_index'] < 0.5):
        pattern = 'sustained'
        confidence = 0.8
    
    # Transient: moderate peaks, high modulation (amplitude decay)
    elif (osc['n_peaks'] >= 3 and 
          am['modulation_index'] > 0.6):
        pattern = 'transient'
        confidence = 0.7
    
    # Irregular: oscillating but high CV
    elif regularity['cv_iei'] > 0.5:
        pattern = 'irregular'
        confidence = 0.6
    
    else:
        pattern = 'unclassified'
        confidence = 0.5
    
    return {
        'pattern': pattern,
        'confidence': confidence,
        'features': features
    }


if __name__ == '__main__':
    """Unit tests for oscillation analysis module."""
    print("Testing oscillation_analysis_module...")
    
    # Test 1: Detect oscillations
    print("\n1. Testing detect_oscillations...")
    np.random.seed(42)
    fs = 30.0
    t = np.arange(1000) / fs
    
    # Create oscillatory signal
    freq = 0.15  # 0.15 Hz oscillations
    trace = 100 + 50 * np.sin(2 * np.pi * freq * t) + np.random.randn(1000) * 5
    
    osc = detect_oscillations(trace, fs)
    print(f"   Oscillating: {osc['oscillating']}")
    print(f"   Number of peaks: {osc['n_peaks']}")
    print(f"   Peak times: {osc['peak_times'][:5]}")  # First 5
    assert osc['oscillating'], "Should detect oscillations"
    assert osc['n_peaks'] >= 3, "Should detect multiple peaks"
    print("   ✓ Oscillation detection working")
    
    # Test 2: Frequency analysis
    print("\n2. Testing compute_oscillation_frequency...")
    freq_info = compute_oscillation_frequency(trace, fs, method='welch')
    print(f"   Dominant frequency: {freq_info['dominant_frequency']:.3f} Hz")
    print(f"   Dominant period: {freq_info['dominant_period']:.2f} s")
    print(f"   Frequency SNR: {freq_info['peak_frequency_snr']:.2f}")
    assert abs(freq_info['dominant_frequency'] - freq) < 0.05, "Frequency estimation error"
    print("   ✓ Frequency computation working")
    
    # Test 3: Amplitude modulation
    print("\n3. Testing compute_amplitude_modulation...")
    # Add amplitude modulation
    am_factor = 0.5 + 0.5 * np.sin(2 * np.pi * 0.02 * t)
    trace_am = 100 + 50 * np.sin(2 * np.pi * freq * t) * am_factor + np.random.randn(1000) * 5
    
    am = compute_amplitude_modulation(trace_am, fs, window_size=5.0)
    print(f"   Modulation index: {am['modulation_index']:.2f}")
    print(f"   Mean amplitude: {am['mean_amplitude']:.2f}")
    print(f"   Amplitude CV: {am['amplitude_cv']:.2f}")
    assert am['modulation_index'] > 0, "Should detect modulation"
    print("   ✓ Amplitude modulation analysis working")
    
    # Test 4: Regularity metrics
    print("\n4. Testing compute_regularity_metrics...")
    osc = detect_oscillations(trace, fs)
    regularity = compute_regularity_metrics(osc['peak_times'])
    print(f"   Mean IEI: {regularity['mean_iei']:.2f} s")
    print(f"   CV of IEI: {regularity['cv_iei']:.2f}")
    print(f"   Regularity index: {regularity['regularity_index']:.2f}")
    assert regularity['mean_iei'] > 0, "Invalid IEI"
    print("   ✓ Regularity metrics computed")
    
    # Test 5: Phase coherence
    print("\n5. Testing compute_phase_coherence...")
    # Create two coherent signals with phase shift
    phase_shift = np.pi / 4
    trace2 = 100 + 50 * np.sin(2 * np.pi * freq * t + phase_shift) + np.random.randn(1000) * 5
    
    coherence = compute_phase_coherence(trace, trace2, fs)
    print(f"   Coherence: {coherence['coherence']:.2f}")
    print(f"   Phase difference: {coherence['phase_difference']:.2f} rad")
    print(f"   PLV: {coherence['phase_locking_value']:.2f}")
    # PLV is more robust measure of phase synchrony
    assert coherence['phase_locking_value'] > 0.5, "Should detect phase locking"
    print("   ✓ Phase coherence computation working")
    
    # Test 6: Oscillation pattern classification
    print("\n6. Testing classify_oscillation_pattern...")
    pattern = classify_oscillation_pattern(trace, fs)
    print(f"   Pattern: {pattern['pattern']}")
    print(f"   Confidence: {pattern['confidence']:.2f}")
    print(f"   Features: {pattern['features']}")
    assert pattern['pattern'] in ['sustained', 'transient', 'irregular', 'none', 'unclassified'], \
           "Invalid pattern"
    print("   ✓ Pattern classification working")
    
    print("\n✅ All oscillation analysis module tests passed!")
