"""
Fitting Module
==============
Specialized fitting functions for calcium imaging noise analysis:
- Lorentzian fits to power spectra
- Exponential fits to correlation curves
- Double exponential fits for photobleaching

Author: George
"""

import numpy as np
from scipy.optimize import curve_fit


def lorentzian(f, A, fc, C):
    """
    Lorentzian function for power spectrum fitting.
    
    S(f) = A / (1 + (f/fc)²) + C
    
    Parameters:
        f (ndarray): Frequency array
        A (float): Amplitude
        fc (float): Cutoff frequency
        C (float): Noise floor offset
    
    Returns:
        ndarray: Lorentzian values
    """
    return A / (1 + (f / fc)**2) + C


def fit_lorentzian_spectrum(freqs, psd, freq_range=(0.1, 20)):
    """
    Fit Lorentzian function to power spectrum.
    
    Extracts characteristic time constant from cutoff frequency:
    τ = 1 / (2π * fc)
    
    Parameters:
        freqs (ndarray): Frequency array (Hz)
        psd (ndarray): Power spectral density
        freq_range (tuple): (min, max) frequency range for fitting
    
    Returns:
        dict: Fit parameters and derived quantities
    
    Example:
        >>> import numpy as np
        >>> # Generate synthetic Lorentzian spectrum
        >>> freqs = np.logspace(-1, 2, 100)
        >>> true_fc = 3.7  # Hz
        >>> psd = 100 / (1 + (freqs/true_fc)**2) + 1
        >>> result = fit_lorentzian_spectrum(freqs, psd)
        >>> print(f"Fitted fc: {result['fc']:.2f} Hz")
        >>> print(f"Time constant: {result['tau']*1000:.1f} ms")
    
    Notes:
        - Cutoff frequency fc corresponds to half-power point
        - Time constant τ relates to mean event duration
        - Fits over specified frequency range to avoid noise
    """
    # Select frequency range
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]
    
    if len(freqs_fit) < 10:
        return {
            'fc': np.nan,
            'tau': np.nan,
            'amplitude': np.nan,
            'noise_floor': np.nan,
            'r_squared': np.nan,
            'fit_curve': np.full_like(freqs, np.nan)
        }
    
    try:
        # Initial parameter guess
        A_init = np.max(psd_fit) - np.min(psd_fit)
        fc_init = freqs_fit[len(freqs_fit)//2]  # Middle frequency
        C_init = np.min(psd_fit)
        
        p0 = [A_init, fc_init, C_init]
        
        # Fit
        popt, pcov = curve_fit(
            lorentzian, freqs_fit, psd_fit,
            p0=p0,
            bounds=([0, 0.01, 0], [np.inf, 100, np.inf]),
            maxfev=10000
        )
        
        A, fc, C = popt
        
        # Compute time constant
        tau = 1.0 / (2 * np.pi * fc)
        
        # Compute R²
        fit_values = lorentzian(freqs_fit, *popt)
        residuals = psd_fit - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((psd_fit - np.mean(psd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Full fit curve
        full_fit = lorentzian(freqs, *popt)
        
        return {
            'fc': fc,
            'tau': tau,
            'amplitude': A,
            'noise_floor': C,
            'r_squared': r_squared,
            'fit_curve': full_fit,
            'fit_params': popt,
            'fit_cov': pcov
        }
    
    except (RuntimeError, ValueError) as e:
        print(f"Lorentzian fit failed: {e}")
        return {
            'fc': np.nan,
            'tau': np.nan,
            'amplitude': np.nan,
            'noise_floor': np.nan,
            'r_squared': np.nan,
            'fit_curve': np.full_like(freqs, np.nan)
        }


def single_exponential(t, A, tau):
    """
    Single exponential decay function.
    
    f(t) = A * exp(-t/τ)
    
    Parameters:
        t (ndarray): Time array
        A (float): Amplitude
        tau (float): Time constant
    
    Returns:
        ndarray: Exponential values
    """
    return A * np.exp(-t / tau)


def fit_exponential_decay(t, trace, bounds=None):
    """
    Fit single exponential decay to time trace.
    
    Parameters:
        t (ndarray): Time array (seconds)
        trace (ndarray): Signal values
        bounds (tuple): Optional ((A_min, tau_min), (A_max, tau_max))
    
    Returns:
        dict: Fit parameters
    
    Example:
        >>> import numpy as np
        >>> t = np.arange(100) / 30.0
        >>> trace = 10 * np.exp(-t / 0.05)
        >>> result = fit_exponential_decay(t, trace)
        >>> print(f"Tau: {result['tau']*1000:.1f} ms")
    """
    if bounds is None:
        bounds = ([0, 0.001], [np.inf, 10.0])
    
    try:
        # Initial guess
        A_init = trace[0]
        tau_init = 0.05  # 50 ms default
        p0 = [A_init, tau_init]
        
        # Fit
        popt, pcov = curve_fit(
            single_exponential, t, trace,
            p0=p0,
            bounds=bounds
        )
        
        A, tau = popt
        
        # Compute R²
        fit_values = single_exponential(t, *popt)
        residuals = trace - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((trace - np.mean(trace))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'amplitude': A,
            'tau': tau,
            'r_squared': r_squared,
            'fit_curve': fit_values,
            'half_life': tau * np.log(2)
        }
    
    except (RuntimeError, ValueError) as e:
        print(f"Exponential fit failed: {e}")
        return {
            'amplitude': np.nan,
            'tau': np.nan,
            'r_squared': np.nan,
            'fit_curve': np.full_like(trace, np.nan),
            'half_life': np.nan
        }


def double_exponential(t, A1, tau1, A2, tau2, C):
    """
    Double exponential function.
    
    f(t) = A1*exp(-t/τ1) + A2*exp(-t/τ2) + C
    
    Useful for photobleaching and complex decay kinetics.
    """
    return A1 * np.exp(-t/tau1) + A2 * np.exp(-t/tau2) + C


def fit_double_exponential(t, trace):
    """
    Fit double exponential to time trace.
    
    Parameters:
        t (ndarray): Time array (seconds)
        trace (ndarray): Signal values
    
    Returns:
        dict: Fit parameters
    
    Example:
        >>> import numpy as np
        >>> t = np.arange(1000) / 30.0
        >>> trace = 10*np.exp(-t/0.1) + 5*np.exp(-t/1.0) + 2
        >>> result = fit_double_exponential(t, trace)
        >>> print(f"Fast tau: {result['tau1']*1000:.1f} ms")
        >>> print(f"Slow tau: {result['tau2']*1000:.1f} ms")
    """
    try:
        # Initial guess
        A1_init = trace[0] * 0.5
        tau1_init = (t[-1] - t[0]) / 10  # Fast component
        A2_init = trace[0] * 0.3
        tau2_init = (t[-1] - t[0]) / 2   # Slow component
        C_init = trace[-1]
        
        p0 = [A1_init, tau1_init, A2_init, tau2_init, C_init]
        
        # Fit with bounds
        popt, pcov = curve_fit(
            double_exponential, t, trace,
            p0=p0,
            bounds=([0, 0.001, 0, 0.001, 0],
                   [np.inf, 100, np.inf, 100, np.inf]),
            maxfev=20000
        )
        
        A1, tau1, A2, tau2, C = popt
        
        # Sort by time constant (fast first)
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
            A1, A2 = A2, A1
        
        # Compute R²
        fit_values = double_exponential(t, *popt)
        residuals = trace - fit_values
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((trace - np.mean(trace))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'A1': A1,
            'tau1': tau1,
            'A2': A2,
            'tau2': tau2,
            'offset': C,
            'r_squared': r_squared,
            'fit_curve': fit_values
        }
    
    except (RuntimeError, ValueError) as e:
        print(f"Double exponential fit failed: {e}")
        return {
            'A1': np.nan,
            'tau1': np.nan,
            'A2': np.nan,
            'tau2': np.nan,
            'offset': np.nan,
            'r_squared': np.nan,
            'fit_curve': np.full_like(trace, np.nan)
        }


def fit_power_law(f, psd, freq_range=(0.1, 20)):
    """
    Fit power law to spectrum: S(f) = A * f^(-α) + C
    
    Useful for identifying 1/f noise characteristics.
    
    Parameters:
        f (ndarray): Frequency array
        psd (ndarray): Power spectral density
        freq_range (tuple): Frequency range for fitting
    
    Returns:
        dict: Fit parameters including slope α
    
    Example:
        >>> import numpy as np
        >>> f = np.logspace(0, 2, 50)
        >>> psd = 100 * f**(-1.5) + 1
        >>> result = fit_power_law(f, psd)
        >>> print(f"Slope α: {result['alpha']:.2f}")
    """
    # Select frequency range
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    f_fit = f[mask]
    psd_fit = psd[mask]
    
    if len(f_fit) < 5:
        return {
            'alpha': np.nan,
            'amplitude': np.nan,
            'offset': np.nan,
            'r_squared': np.nan
        }
    
    # Fit in log-log space for numerical stability
    # log(S) = log(A) - α*log(f) + log(C)
    # Approximate by fitting: log(S - C_est) = log(A) - α*log(f)
    
    C_est = np.min(psd_fit)
    
    try:
        log_f = np.log(f_fit)
        log_psd = np.log(psd_fit - C_est + 1e-10)
        
        # Linear fit to log-log
        coeffs = np.polyfit(log_f, log_psd, deg=1)
        alpha = -coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Compute R²
        predicted = A * f_fit**(-alpha) + C_est
        residuals = psd_fit - predicted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((psd_fit - np.mean(psd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'alpha': alpha,
            'amplitude': A,
            'offset': C_est,
            'r_squared': r_squared,
            'fit_curve': A * f**(-alpha) + C_est
        }
    
    except (RuntimeError, ValueError) as e:
        print(f"Power law fit failed: {e}")
        return {
            'alpha': np.nan,
            'amplitude': np.nan,
            'offset': np.nan,
            'r_squared': np.nan
        }


if __name__ == '__main__':
    """Unit tests for fitting module."""
    print("Testing fitting_module...")
    
    # Test 1: Lorentzian fit
    print("\n1. Testing fit_lorentzian_spectrum...")
    freqs = np.logspace(-1, 2, 100)
    true_fc = 3.7
    true_tau = 1.0 / (2 * np.pi * true_fc)
    psd_true = 100 / (1 + (freqs/true_fc)**2) + 1
    
    # Add some noise
    psd_noisy = psd_true + np.random.randn(len(psd_true)) * 2
    
    result = fit_lorentzian_spectrum(freqs, psd_noisy, freq_range=(0.1, 20))
    
    print(f"   True fc: {true_fc:.2f} Hz")
    print(f"   Fitted fc: {result['fc']:.2f} Hz")
    print(f"   True tau: {true_tau*1000:.1f} ms")
    print(f"   Fitted tau: {result['tau']*1000:.1f} ms")
    print(f"   R²: {result['r_squared']:.3f}")
    
    assert abs(result['fc'] - true_fc) < 0.5, "fc estimation inaccurate"
    assert result['r_squared'] > 0.9, "Poor fit quality"
    print("   ✓ Lorentzian fitting working correctly")
    
    # Test 2: Single exponential fit
    print("\n2. Testing fit_exponential_decay...")
    t = np.arange(100) / 30.0
    true_tau = 0.05  # 50 ms
    trace = 10 * np.exp(-t / true_tau)
    
    result = fit_exponential_decay(t, trace)
    
    print(f"   True tau: {true_tau*1000:.1f} ms")
    print(f"   Fitted tau: {result['tau']*1000:.1f} ms")
    print(f"   R²: {result['r_squared']:.3f}")
    
    assert abs(result['tau'] - true_tau) < 0.005, "tau estimation inaccurate"
    assert result['r_squared'] > 0.99, "Poor fit quality"
    print("   ✓ Exponential fitting working correctly")
    
    # Test 3: Double exponential fit
    print("\n3. Testing fit_double_exponential...")
    t = np.arange(1000) / 30.0
    true_tau1 = 0.05  # 50 ms
    true_tau2 = 0.5   # 500 ms
    trace = 10*np.exp(-t/true_tau1) + 5*np.exp(-t/true_tau2) + 2
    
    result = fit_double_exponential(t, trace)
    
    print(f"   True fast tau: {true_tau1*1000:.1f} ms")
    print(f"   Fitted fast tau: {result['tau1']*1000:.1f} ms")
    print(f"   True slow tau: {true_tau2*1000:.1f} ms")
    print(f"   Fitted slow tau: {result['tau2']*1000:.1f} ms")
    print(f"   R²: {result['r_squared']:.3f}")
    
    assert abs(result['tau1'] - true_tau1) < 0.01, "Fast tau inaccurate"
    assert abs(result['tau2'] - true_tau2) < 0.1, "Slow tau inaccurate"
    assert result['r_squared'] > 0.98, "Poor fit quality"
    print("   ✓ Double exponential fitting working correctly")
    
    # Test 4: Power law fit
    print("\n4. Testing fit_power_law...")
    f = np.logspace(0, 2, 50)
    true_alpha = 1.5
    psd = 100 * f**(-true_alpha) + 1
    
    result = fit_power_law(f, psd, freq_range=(1, 50))
    
    print(f"   True α: {true_alpha:.2f}")
    print(f"   Fitted α: {result['alpha']:.2f}")
    print(f"   R²: {result['r_squared']:.3f}")
    
    assert abs(result['alpha'] - true_alpha) < 0.2, "Alpha estimation inaccurate"
    assert result['r_squared'] > 0.95, "Poor fit quality"
    print("   ✓ Power law fitting working correctly")
    
    print("\n✅ All fitting module tests passed!")
