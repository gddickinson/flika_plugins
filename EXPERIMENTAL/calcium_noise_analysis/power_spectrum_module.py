"""
Power Spectrum Analysis Module
===============================
Power spectral density analysis for identifying Ca²⁺-active pixels
and Butterworth bandpass filtering for calcium transient isolation.

Based on Swaminathan et al. 2020 methodology.

Author: George
"""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt


def compute_power_spectrum_map(image_stack, fs, low_freq=(0.1, 5), high_freq=50, 
                                nperseg=256, noverlap=None, nfft=1024):
    """
    Generate 2D map identifying Ca²⁺-active pixels via spectral analysis.
    
    Separates calcium signals (0.1-5 Hz) from high-frequency shot noise (>50 Hz)
    by computing Welch periodograms for each pixel and comparing power in the
    two frequency bands.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        fs (float): Sampling frequency in Hz
        low_freq (tuple): (min, max) frequency range for calcium signals in Hz
        high_freq (float): Frequency above which to measure shot noise in Hz
        nperseg (int): Length of each segment for Welch's method (power of 2)
        noverlap (int): Number of points to overlap between segments (default: nperseg//2)
        nfft (int): Length of FFT used (default: 1024)
    
    Returns:
        ndarray: Power spectrum map (H, W) where higher values indicate Ca²⁺ activity
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(1000, 128, 128)
        >>> fs = 30.0  # 30 Hz frame rate
        >>> psm = compute_power_spectrum_map(stack, fs)
        >>> print(psm.shape)
        (128, 128)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    T, H, W = image_stack.shape
    psm = np.zeros((H, W), dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            # Compute power spectral density for this pixel's time series
            freqs, psd = welch(
                image_stack[:, i, j], 
                fs=fs, 
                nperseg=nperseg, 
                noverlap=noverlap,
                nfft=nfft
            )
            
            # Define frequency masks
            low_mask = (freqs >= low_freq[0]) & (freqs <= low_freq[1])
            high_mask = freqs > high_freq
            
            # Compute difference between low-freq (signal) and high-freq (noise) power
            if np.any(low_mask) and np.any(high_mask):
                psm[i, j] = np.mean(psd[low_mask]) - np.mean(psd[high_mask])
            else:
                psm[i, j] = 0.0
    
    return psm


def butterworth_bandpass(trace, fs, low=0.1, high=5.0, order=2):
    """
    Bandpass filter isolating calcium transient frequencies.
    
    Typical calcium signal frequencies are 0.1-5 Hz. This filter removes
    both low-frequency baseline drift and high-frequency shot noise.
    
    Uses second-order sections (SOS) for numerical stability.
    
    Parameters:
        trace (ndarray): Input time series (1D array)
        fs (float): Sampling frequency in Hz
        low (float): Low frequency cutoff in Hz
        high (float): High frequency cutoff in Hz
        order (int): Filter order (higher = steeper rolloff)
    
    Returns:
        ndarray: Bandpass filtered trace
    
    Example:
        >>> import numpy as np
        >>> trace = np.random.randn(1000)
        >>> fs = 30.0
        >>> filtered = butterworth_bandpass(trace, fs, low=0.1, high=5.0)
        >>> print(filtered.shape)
        (1000,)
    
    Notes:
        - Uses forward-backward filtering (sosfiltfilt) for zero phase shift
        - Cutoff frequencies are normalized to Nyquist frequency (fs/2)
        - Returns array of same shape as input
    """
    nyquist = fs / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist
    
    # Ensure normalized frequencies are in valid range (0, 1)
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(0.001, min(high_norm, 0.999))
    
    # Create second-order sections
    sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
    
    # Apply zero-phase filtering
    filtered = sosfiltfilt(sos, trace)
    
    return filtered


def compute_single_psd(trace, fs, nperseg=256, noverlap=None, nfft=1024):
    """
    Compute power spectral density for a single trace.
    
    Convenience function for analyzing individual time series.
    
    Parameters:
        trace (ndarray): Input time series (1D array)
        fs (float): Sampling frequency in Hz
        nperseg (int): Length of each segment for Welch's method
        noverlap (int): Number of points to overlap between segments
        nfft (int): Length of FFT used
    
    Returns:
        tuple: (frequencies, power_spectral_density)
            - frequencies (ndarray): Array of sample frequencies
            - power_spectral_density (ndarray): Power spectral density
    
    Example:
        >>> import numpy as np
        >>> trace = np.sin(2 * np.pi * 1.0 * np.arange(1000) / 30.0)
        >>> freqs, psd = compute_single_psd(trace, fs=30.0)
        >>> peak_freq = freqs[np.argmax(psd)]
        >>> print(f"Peak frequency: {peak_freq:.2f} Hz")
        Peak frequency: 1.00 Hz
    """
    if noverlap is None:
        noverlap = nperseg // 2
    
    freqs, psd = welch(
        trace, 
        fs=fs, 
        nperseg=nperseg, 
        noverlap=noverlap,
        nfft=nfft
    )
    
    return freqs, psd


def compute_signal_to_noise_ratio(trace, fs, signal_band=(0.1, 5.0), 
                                   noise_band=(50, None), nperseg=256):
    """
    Compute signal-to-noise ratio in frequency domain.
    
    Compares power in signal frequency band to power in noise frequency band.
    
    Parameters:
        trace (ndarray): Input time series (1D array)
        fs (float): Sampling frequency in Hz
        signal_band (tuple): (min, max) frequency range for signal in Hz
        noise_band (tuple): (min, max) frequency range for noise in Hz
                           (None for max means Nyquist frequency)
        nperseg (int): Length of each segment for Welch's method
    
    Returns:
        float: Signal-to-noise ratio (power in signal band / power in noise band)
    
    Example:
        >>> import numpy as np
        >>> # Create signal with 1 Hz component and noise
        >>> t = np.arange(1000) / 30.0
        >>> signal = 10 * np.sin(2 * np.pi * 1.0 * t)
        >>> noise = np.random.randn(1000)
        >>> trace = signal + noise
        >>> snr = compute_signal_to_noise_ratio(trace, fs=30.0)
        >>> print(f"SNR: {snr:.2f}")
    """
    freqs, psd = compute_single_psd(trace, fs, nperseg=nperseg)
    
    # Define frequency masks
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    
    if noise_band[1] is None:
        noise_mask = freqs >= noise_band[0]
    else:
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    
    # Compute average power in each band
    signal_power = np.mean(psd[signal_mask]) if np.any(signal_mask) else 0.0
    noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1.0
    
    # Return SNR (avoid division by zero)
    snr = signal_power / max(noise_power, 1e-10)
    
    return snr


if __name__ == '__main__':
    """Unit tests for power spectrum module."""
    print("Testing power_spectrum_module...")
    
    # Test 1: Power spectrum map computation
    print("\n1. Testing compute_power_spectrum_map...")
    np.random.seed(42)
    T, H, W = 500, 32, 32
    fs = 30.0
    
    # Create synthetic data with calcium-like signal
    t = np.arange(T) / fs
    calcium_signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz signal
    
    stack = np.random.randn(T, H, W) * 0.5
    stack[:, 15:20, 15:20] += calcium_signal[:, np.newaxis, np.newaxis]
    
    psm = compute_power_spectrum_map(stack, fs)
    print(f"   PSM shape: {psm.shape}")
    print(f"   PSM range: [{psm.min():.4f}, {psm.max():.4f}]")
    print(f"   Signal region mean: {psm[15:20, 15:20].mean():.4f}")
    print(f"   Background mean: {psm[0:10, 0:10].mean():.4f}")
    assert psm.shape == (H, W), "PSM shape mismatch"
    assert psm[15:20, 15:20].mean() > psm[0:10, 0:10].mean(), "Signal not detected"
    print("   ✓ Power spectrum map computed correctly")
    
    # Test 2: Butterworth bandpass
    print("\n2. Testing butterworth_bandpass...")
    trace = np.sin(2 * np.pi * 1.0 * t) + np.sin(2 * np.pi * 30.0 * t) + 5.0
    filtered = butterworth_bandpass(trace, fs, low=0.1, high=5.0, order=2)
    print(f"   Input shape: {trace.shape}")
    print(f"   Output shape: {filtered.shape}")
    print(f"   Input mean: {trace.mean():.4f}")
    print(f"   Output mean: {filtered.mean():.4f}")
    assert filtered.shape == trace.shape, "Shape mismatch"
    assert abs(filtered.mean()) < 0.5, "DC component not removed"
    print("   ✓ Bandpass filter working correctly")
    
    # Test 3: Single PSD computation
    print("\n3. Testing compute_single_psd...")
    trace = np.sin(2 * np.pi * 2.5 * t)
    freqs, psd = compute_single_psd(trace, fs)
    peak_freq = freqs[np.argmax(psd)]
    print(f"   Frequencies shape: {freqs.shape}")
    print(f"   PSD shape: {psd.shape}")
    print(f"   Peak frequency: {peak_freq:.2f} Hz")
    assert freqs.shape == psd.shape, "Shape mismatch"
    assert abs(peak_freq - 2.5) < 0.5, "Peak frequency incorrect"
    print("   ✓ PSD computation working correctly")
    
    # Test 4: Signal-to-noise ratio
    print("\n4. Testing compute_signal_to_noise_ratio...")
    signal = 10 * np.sin(2 * np.pi * 1.0 * t)
    noise = np.random.randn(T)
    trace_with_signal = signal + noise
    snr = compute_signal_to_noise_ratio(trace_with_signal, fs)
    print(f"   SNR: {snr:.2f}")
    assert snr > 1.0, "SNR should be > 1 for signal + noise"
    print("   ✓ SNR computation working correctly")
    
    print("\n✅ All power spectrum module tests passed!")
