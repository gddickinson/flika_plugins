"""
Shot Noise Analysis Module
===========================
Anscombe variance-stabilizing transform and related utilities for
handling Poisson-distributed photon shot noise in calcium imaging.

The Anscombe transform converts Poisson noise to approximately Gaussian
with unit variance, enabling standard denoising techniques.

Author: George
"""

import numpy as np


def anscombe_transform(image):
    """
    Apply Anscombe variance-stabilizing transform for Poisson noise.
    
    Transform: y = 2√(x + 3/8)
    
    Converts Poisson-distributed data (where variance = mean) to approximately
    Gaussian distribution with unit variance. This enables standard denoising
    techniques that assume Gaussian noise.
    
    Parameters:
        image (ndarray): Input image or stack with Poisson noise
    
    Returns:
        ndarray: Variance-stabilized image
    
    Example:
        >>> import numpy as np
        >>> poisson_data = np.random.poisson(100, (256, 256))
        >>> stabilized = anscombe_transform(poisson_data)
        >>> print(f"Original variance: {poisson_data.var():.2f}")
        >>> print(f"Stabilized variance: {stabilized.var():.2f}")
    
    Notes:
        - Input should be non-negative (Poisson data)
        - Output is approximately Gaussian with unit variance
        - Use inverse_anscombe() to transform back to original domain
    """
    # Ensure input is non-negative
    image = np.maximum(image, 0)
    
    # Apply Anscombe transform
    transformed = 2 * np.sqrt(image + 3/8)
    
    return transformed


def inverse_anscombe(image):
    """
    Apply inverse Anscombe transform.
    
    Inverse: x = (y/2)² - 3/8
    
    Converts variance-stabilized data back to Poisson domain.
    
    Parameters:
        image (ndarray): Variance-stabilized image (from anscombe_transform)
    
    Returns:
        ndarray: Image in original Poisson domain
    
    Example:
        >>> import numpy as np
        >>> original = np.random.poisson(100, (256, 256))
        >>> stabilized = anscombe_transform(original)
        >>> recovered = inverse_anscombe(stabilized)
        >>> print(f"Mean absolute error: {np.mean(np.abs(original - recovered)):.2f}")
    
    Notes:
        - Negative values are clipped to zero
        - Some information loss is expected due to transform non-invertibility
    """
    # Apply inverse transform
    recovered = (image / 2) ** 2 - 3/8
    
    # Ensure non-negative (clip small negative values from numerical error)
    recovered = np.maximum(recovered, 0)
    
    return recovered


def generalized_anscombe_transform(image, gain=1.0, readout_noise=0.0, offset=0.0):
    """
    Apply generalized Anscombe transform accounting for camera parameters.
    
    Transform: y = (2/α) √[max(αI + (3/8)α² + σ² - αμ, 0)]
    
    Where:
    - I is the input image
    - α (alpha) is the camera gain
    - σ² is the readout noise variance
    - μ is the offset
    
    Parameters:
        image (ndarray): Input image
        gain (float): Camera gain (electrons per ADU)
        readout_noise (float): Readout noise standard deviation (electrons)
        offset (float): Camera offset (ADU)
    
    Returns:
        ndarray: Variance-stabilized image
    
    Example:
        >>> import numpy as np
        >>> # Simulate camera data
        >>> true_photons = np.random.poisson(1000, (256, 256))
        >>> gain = 0.5  # 0.5 e-/ADU
        >>> readout = 5.0  # 5 e- readout noise
        >>> camera_data = true_photons / gain + np.random.randn(256, 256) * readout / gain
        >>> stabilized = generalized_anscombe_transform(camera_data, gain, readout)
    
    Notes:
        - Accounts for realistic camera noise model
        - Requires camera calibration (gain, readout noise)
        - More accurate than basic Anscombe for real camera data
    """
    alpha = gain
    sigma_sq = readout_noise ** 2
    mu = offset
    
    # Apply generalized Anscombe transform
    argument = alpha * image + (3/8) * alpha**2 + sigma_sq - alpha * mu
    argument = np.maximum(argument, 0)  # Ensure non-negative
    
    transformed = (2 / alpha) * np.sqrt(argument)
    
    return transformed


def estimate_poisson_noise_level(image):
    """
    Estimate noise level assuming Poisson statistics.
    
    For Poisson noise: σ = √μ
    
    Returns the mean noise standard deviation across the image.
    
    Parameters:
        image (ndarray): Input image with Poisson noise
    
    Returns:
        float: Estimated noise level (mean of local standard deviations)
    
    Example:
        >>> import numpy as np
        >>> poisson_data = np.random.poisson(100, (256, 256))
        >>> noise_level = estimate_poisson_noise_level(poisson_data)
        >>> print(f"Estimated noise level: {noise_level:.2f}")
        >>> print(f"Theoretical (√100): {np.sqrt(100):.2f}")
    """
    # For Poisson, noise level = sqrt(mean intensity)
    mean_intensity = np.mean(image)
    noise_level = np.sqrt(max(mean_intensity, 0))
    
    return noise_level


def compute_normalized_noise_variance(image, f0=None):
    """
    Compute normalized noise variance for variance stabilization check.
    
    For Poisson noise: var(F) / F₀ should be approximately constant
    
    Parameters:
        image (ndarray): Input image stack (T, H, W) or single image
        f0 (ndarray): Baseline fluorescence (optional, computed if not provided)
    
    Returns:
        ndarray: Normalized variance map (var/mean)
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.poisson(100, (100, 64, 64))
        >>> norm_var = compute_normalized_noise_variance(stack)
        >>> print(f"Mean normalized variance: {norm_var.mean():.2f}")
    """
    if image.ndim == 3:
        # Time series - compute variance and mean over time
        variance = np.var(image, axis=0)
        if f0 is None:
            f0 = np.mean(image, axis=0)
    else:
        # Single image - just return image itself
        variance = image
        if f0 is None:
            f0 = image
    
    # Avoid division by zero
    f0 = np.maximum(f0, 1e-10)
    
    normalized_variance = variance / f0
    
    return normalized_variance


def assess_noise_model(image_stack, plot=False):
    """
    Assess if noise follows Poisson statistics.
    
    Checks if variance scales linearly with mean intensity across pixels.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        plot (bool): If True, create diagnostic plot (requires matplotlib)
    
    Returns:
        dict: Dictionary containing:
            - 'slope': Slope of variance vs mean (should be ~1 for Poisson)
            - 'intercept': Intercept (readout noise contribution)
            - 'r_squared': Coefficient of determination
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.poisson(100, (100, 64, 64))
        >>> results = assess_noise_model(stack)
        >>> print(f"Slope (should be ~1): {results['slope']:.2f}")
        >>> print(f"R²: {results['r_squared']:.3f}")
    """
    # Compute mean and variance for each pixel
    mean_vals = np.mean(image_stack, axis=0).flatten()
    var_vals = np.var(image_stack, axis=0).flatten()
    
    # Remove pixels with very low intensity (unreliable)
    valid_mask = mean_vals > 10
    mean_vals = mean_vals[valid_mask]
    var_vals = var_vals[valid_mask]
    
    # Fit linear model: variance = slope * mean + intercept
    coeffs = np.polyfit(mean_vals, var_vals, deg=1)
    slope, intercept = coeffs
    
    # Compute R²
    predicted = slope * mean_vals + intercept
    ss_res = np.sum((var_vals - predicted) ** 2)
    ss_tot = np.sum((var_vals - np.mean(var_vals)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    results = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared
    }
    
    # Optional plotting
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6))
            plt.scatter(mean_vals, var_vals, alpha=0.3, s=1)
            plt.plot(mean_vals, predicted, 'r-', linewidth=2, 
                    label=f'Fit: var = {slope:.2f}*mean + {intercept:.2f}')
            plt.xlabel('Mean Intensity')
            plt.ylabel('Variance')
            plt.title(f'Noise Model Assessment (R² = {r_squared:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")
    
    return results


if __name__ == '__main__':
    """Unit tests for shot noise module."""
    print("Testing shot_noise_module...")
    
    # Test 1: Anscombe transform
    print("\n1. Testing anscombe_transform...")
    np.random.seed(42)
    poisson_data = np.random.poisson(100, (1000, 64, 64))
    
    # Original data: variance ≈ mean (Poisson property)
    orig_mean = poisson_data.mean()
    orig_var = poisson_data.var()
    print(f"   Original mean: {orig_mean:.2f}")
    print(f"   Original variance: {orig_var:.2f}")
    print(f"   Variance/Mean ratio: {orig_var/orig_mean:.2f} (should be ~1)")
    
    # Apply Anscombe transform
    stabilized = anscombe_transform(poisson_data)
    stab_var = stabilized.var()
    print(f"   Stabilized variance: {stab_var:.2f} (should be ~1)")
    assert 0.8 < stab_var < 1.2, "Variance not stabilized properly"
    print("   ✓ Anscombe transform working correctly")
    
    # Test 2: Inverse Anscombe
    print("\n2. Testing inverse_anscombe...")
    recovered = inverse_anscombe(stabilized)
    mae = np.mean(np.abs(poisson_data - recovered))
    rel_error = mae / poisson_data.mean()
    print(f"   Mean absolute error: {mae:.2f}")
    print(f"   Relative error: {rel_error:.4f}")
    assert rel_error < 0.1, "Inverse transform error too large"
    print("   ✓ Inverse Anscombe working correctly")
    
    # Test 3: Generalized Anscombe
    print("\n3. Testing generalized_anscombe_transform...")
    # Simulate realistic camera data
    true_photons = np.random.poisson(1000, (256, 256))
    gain = 0.5  # e-/ADU
    readout = 5.0  # e- readout noise
    camera_data = true_photons / gain + np.random.randn(256, 256) * readout / gain
    
    stabilized_gen = generalized_anscombe_transform(camera_data, gain, readout)
    print(f"   Camera data mean: {camera_data.mean():.2f}")
    print(f"   Stabilized mean: {stabilized_gen.mean():.2f}")
    print(f"   Stabilized variance: {stabilized_gen.var():.2f}")
    assert stabilized_gen.shape == camera_data.shape, "Shape mismatch"
    print("   ✓ Generalized Anscombe working correctly")
    
    # Test 4: Poisson noise level estimation
    print("\n4. Testing estimate_poisson_noise_level...")
    test_image = np.random.poisson(100, (256, 256))
    noise_level = estimate_poisson_noise_level(test_image)
    theoretical = np.sqrt(100)
    print(f"   Estimated noise level: {noise_level:.2f}")
    print(f"   Theoretical (√100): {theoretical:.2f}")
    assert abs(noise_level - theoretical) < 1.0, "Noise level estimation incorrect"
    print("   ✓ Noise level estimation working correctly")
    
    # Test 5: Normalized noise variance
    print("\n5. Testing compute_normalized_noise_variance...")
    stack = np.random.poisson(100, (100, 64, 64))
    norm_var = compute_normalized_noise_variance(stack)
    print(f"   Normalized variance shape: {norm_var.shape}")
    print(f"   Mean normalized variance: {norm_var.mean():.2f} (should be ~1)")
    assert norm_var.shape == (64, 64), "Shape mismatch"
    assert 0.8 < norm_var.mean() < 1.2, "Normalized variance incorrect"
    print("   ✓ Normalized variance computation working correctly")
    
    # Test 6: Noise model assessment
    print("\n6. Testing assess_noise_model...")
    stack = np.random.poisson(100, (100, 64, 64))
    results = assess_noise_model(stack, plot=False)
    print(f"   Slope: {results['slope']:.2f} (should be ~1 for Poisson)")
    print(f"   Intercept: {results['intercept']:.2f}")
    print(f"   R²: {results['r_squared']:.3f}")
    assert 0.8 < results['slope'] < 1.2, "Slope indicates non-Poisson noise"
    assert results['r_squared'] > 0.8, "Poor fit to linear model"
    print("   ✓ Noise model assessment working correctly")
    
    print("\n✅ All shot noise module tests passed!")
