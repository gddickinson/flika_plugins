"""
PSF Models for Synthetic Puncta Generation

Implements realistic Point Spread Function models for generating
synthetic PIEZO1-HaloTag puncta with sub-pixel localization ground truth.

Based on:
- Gibson & Lanni 3D PSF model
- Gaussian approximation for speed
- Realistic noise characteristics
"""

import numpy as np
from scipy.special import j1  # Bessel function
from typing import Tuple, Optional


class PSFModel:
    """Base class for PSF models."""
    
    def __init__(self, pixel_size_nm: float = 130.0):
        """
        Args:
            pixel_size_nm: Camera pixel size in nanometers
        """
        self.pixel_size_nm = pixel_size_nm
        self.pixel_size_um = pixel_size_nm / 1000.0
    
    def generate(self, 
                 positions: np.ndarray,
                 image_size: Tuple[int, int],
                 photon_counts: np.ndarray) -> np.ndarray:
        """Generate PSF image from emitter positions.
        
        Args:
            positions: Nx2 array of (x, y) positions in pixels (sub-pixel precision)
            image_size: (height, width) of output image
            photon_counts: N array of photon counts per emitter
            
        Returns:
            image: Generated PSF image
        """
        raise NotImplementedError


class Gaussian2DPSF(PSFModel):
    """
    2D Gaussian PSF approximation.
    
    Fast and sufficient for TIRF microscopy where particles are in a thin
    focal plane. Sigma is determined by wavelength and NA.
    """
    
    def __init__(self,
                 pixel_size_nm: float = 130.0,
                 wavelength_nm: float = 646.0,  # JF646
                 numerical_aperture: float = 1.49,
                 sigma_xy_nm: Optional[float] = None):
        """
        Args:
            pixel_size_nm: Camera pixel size
            wavelength_nm: Emission wavelength
            numerical_aperture: Objective NA
            sigma_xy_nm: Override sigma (otherwise computed from Abbe limit)
        """
        super().__init__(pixel_size_nm)
        
        self.wavelength_nm = wavelength_nm
        self.na = numerical_aperture
        
        if sigma_xy_nm is None:
            # Abbe diffraction limit: d = λ / (2 * NA)
            # Convert to Gaussian sigma: σ ≈ d / 2.355 (FWHM to sigma)
            fwhm_nm = wavelength_nm / (2.0 * numerical_aperture)
            self.sigma_nm = fwhm_nm / 2.355
        else:
            self.sigma_nm = sigma_xy_nm
        
        # Convert to pixels
        self.sigma_px = self.sigma_nm / pixel_size_nm
        
        print(f"Gaussian PSF: σ = {self.sigma_nm:.1f} nm ({self.sigma_px:.2f} px)")
    
    def generate(self,
                 positions: np.ndarray,
                 image_size: Tuple[int, int],
                 photon_counts: np.ndarray) -> np.ndarray:
        """Generate 2D Gaussian PSF image."""
        
        height, width = image_size
        image = np.zeros((height, width), dtype=np.float32)
        
        # Create coordinate grids
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        
        # Add each emitter
        for (x_pos, y_pos), photons in zip(positions, photon_counts):
            # Gaussian PSF centered at (x_pos, y_pos)
            dx = x_grid - x_pos
            dy = y_grid - y_pos
            
            # 2D Gaussian
            psf = np.exp(-(dx**2 + dy**2) / (2 * self.sigma_px**2))
            
            # Normalize to photon count
            psf = psf / psf.sum() * photons
            
            image += psf
        
        return image


class Airy2DPSF(PSFModel):
    """
    2D Airy disk PSF (more accurate than Gaussian).
    
    Based on Fraunhofer diffraction through circular aperture.
    Slower but more physically accurate.
    """
    
    def __init__(self,
                 pixel_size_nm: float = 130.0,
                 wavelength_nm: float = 646.0,
                 numerical_aperture: float = 1.49):
        super().__init__(pixel_size_nm)
        
        self.wavelength_nm = wavelength_nm
        self.na = numerical_aperture
        
        # Airy disk radius to first minimum
        self.airy_radius_nm = 0.61 * wavelength_nm / numerical_aperture
        self.airy_radius_px = self.airy_radius_nm / pixel_size_nm
        
        print(f"Airy PSF: r = {self.airy_radius_nm:.1f} nm ({self.airy_radius_px:.2f} px)")
    
    def generate(self,
                 positions: np.ndarray,
                 image_size: Tuple[int, int],
                 photon_counts: np.ndarray) -> np.ndarray:
        """Generate 2D Airy disk PSF image."""
        
        height, width = image_size
        image = np.zeros((height, width), dtype=np.float32)
        
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        
        for (x_pos, y_pos), photons in zip(positions, photon_counts):
            dx = x_grid - x_pos
            dy = y_grid - y_pos
            r = np.sqrt(dx**2 + dy**2)
            
            # Airy pattern: I(r) = I0 * (2*J1(kr) / kr)^2
            # where k = 2π * NA / λ
            k = 2 * np.pi * self.na / (self.wavelength_nm / self.pixel_size_nm)
            kr = k * r
            
            # Handle r=0 case
            psf = np.zeros_like(r)
            mask = kr > 1e-6
            psf[mask] = (2 * j1(kr[mask]) / kr[mask])**2
            psf[~mask] = 1.0  # Limit as kr -> 0
            
            # Normalize
            psf = psf / psf.sum() * photons
            
            image += psf
        
        return image


def add_noise(image: np.ndarray,
              baseline: float = 100.0,
              read_noise_std: float = 1.5,
              dark_current: float = 0.05) -> np.ndarray:
    """
    Add realistic camera noise to image.
    
    Args:
        image: Clean photon count image
        baseline: Camera baseline (offset)
        read_noise_std: Read noise standard deviation
        dark_current: Dark current electrons per pixel
        
    Returns:
        noisy_image: Image with Poisson + Gaussian noise
    """
    
    # Add dark current
    image_with_dark = image + dark_current
    
    # Poisson (shot) noise
    noisy = np.random.poisson(image_with_dark).astype(np.float32)
    
    # Gaussian (read) noise
    noisy += np.random.normal(0, read_noise_std, size=image.shape)
    
    # Add baseline
    noisy += baseline
    
    # Clip to valid range
    noisy = np.clip(noisy, 0, 65535)
    
    return noisy.astype(np.uint16)


def add_background(image: np.ndarray,
                   mean_bg: float = 500.0,
                   bg_std: float = 50.0) -> np.ndarray:
    """
    Add spatially varying background.
    
    Args:
        image: Input image
        mean_bg: Mean background level
        bg_std: Background spatial variation std
        
    Returns:
        image_with_bg: Image with added background
    """
    
    # Smooth random background
    from scipy.ndimage import gaussian_filter
    
    bg = np.random.normal(mean_bg, bg_std, size=image.shape)
    bg = gaussian_filter(bg, sigma=5.0)
    
    return image + bg


# Example usage and testing
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create PSF models
    gaussian_psf = Gaussian2DPSF(
        pixel_size_nm=130.0,
        wavelength_nm=646.0,
        numerical_aperture=1.49
    )
    
    airy_psf = Airy2DPSF(
        pixel_size_nm=130.0,
        wavelength_nm=646.0,
        numerical_aperture=1.49
    )
    
    # Test positions
    positions = np.array([
        [32.3, 32.7],  # Sub-pixel positions
        [48.1, 48.9],
        [64.5, 64.5]
    ])
    
    photon_counts = np.array([1000, 1500, 800])
    
    # Generate images
    gaussian_img = gaussian_psf.generate(positions, (100, 100), photon_counts)
    airy_img = airy_psf.generate(positions, (100, 100), photon_counts)
    
    # Add noise
    gaussian_noisy = add_noise(add_background(gaussian_img))
    airy_noisy = add_noise(add_background(airy_img))
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gaussian_img, cmap='hot')
    axes[0, 0].set_title('Gaussian PSF (clean)')
    
    axes[0, 1].imshow(gaussian_noisy, cmap='hot')
    axes[0, 1].set_title('Gaussian PSF (noisy)')
    
    axes[0, 2].imshow(gaussian_img[25:45, 25:45], cmap='hot')
    axes[0, 2].set_title('Zoom: Single PSF')
    
    axes[1, 0].imshow(airy_img, cmap='hot')
    axes[1, 0].set_title('Airy PSF (clean)')
    
    axes[1, 1].imshow(airy_noisy, cmap='hot')
    axes[1, 1].set_title('Airy PSF (noisy)')
    
    axes[1, 2].imshow(airy_img[25:45, 25:45], cmap='hot')
    axes[1, 2].set_title('Zoom: Single Airy')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('psf_comparison.png', dpi=150)
    print("Saved psf_comparison.png")
