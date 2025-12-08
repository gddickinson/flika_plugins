"""
PSF Fitting Module
==================

Implements sub-pixel localization methods:
- Gaussian PSF fitting (2D and 3D astigmatism)
- Least Squares (LSQ) fitting
- Weighted Least Squares (WLSQ) fitting
- Maximum Likelihood Estimation (MLE)
- Radial Symmetry method
- Centroid refinement
- Multiple emitter fitting
"""

import numpy as np
from scipy import optimize, ndimage
from scipy.special import erf


class LocalizationResult:
    """Container for localization results."""

    def __init__(self):
        self.x = []  # x position (col)
        self.y = []  # y position (row)
        self.z = []  # z position (if 3D)
        self.intensity = []  # Fitted intensity
        self.background = []  # Fitted background
        self.sigma_x = []  # PSF width in x
        self.sigma_y = []  # PSF width in y (for elliptical)
        self.frame = []  # Frame number
        self.uncertainty = []  # Localization uncertainty
        self.chi_squared = []  # Goodness of fit

    def add_localization(self, x, y, intensity, background, sigma_x,
                        sigma_y=None, frame=None, uncertainty=None,
                        chi_squared=None, z=None):
        """Add a localization."""
        self.x.append(x)
        self.y.append(y)
        self.intensity.append(intensity)
        self.background.append(background)
        self.sigma_x.append(sigma_x)
        self.sigma_y.append(sigma_y if sigma_y is not None else sigma_x)
        self.frame.append(frame)
        self.uncertainty.append(uncertainty)
        self.chi_squared.append(chi_squared)
        if z is not None:
            self.z.append(z)

    def to_array(self):
        """Convert to numpy array."""
        data = {
            'x': np.array(self.x),
            'y': np.array(self.y),
            'intensity': np.array(self.intensity),
            'background': np.array(self.background),
            'sigma_x': np.array(self.sigma_x),
            'sigma_y': np.array(self.sigma_y),
            'frame': np.array(self.frame),
            'uncertainty': np.array(self.uncertainty),
            'chi_squared': np.array(self.chi_squared)
        }
        if self.z:
            data['z'] = np.array(self.z)
        return data

    def __len__(self):
        return len(self.x)


class BaseFitter:
    """Base class for PSF fitters."""

    def __init__(self, name="BaseFitter"):
        self.name = name
        self.pixel_size = 100.0  # nm/pixel
        self.photons_per_adu = 1.0  # Conversion factor
        self.baseline = 0.0  # Camera baseline offset
        self.em_gain = 1.0  # EM gain for EMCCD

    def set_camera_params(self, pixel_size=None, photons_per_adu=None,
                         baseline=None, em_gain=None):
        """Set camera parameters."""
        if pixel_size is not None:
            self.pixel_size = pixel_size
        if photons_per_adu is not None:
            self.photons_per_adu = photons_per_adu
        if baseline is not None:
            self.baseline = baseline
        if em_gain is not None:
            self.em_gain = em_gain

    def fit(self, image, positions, fit_radius=3):
        """Fit PSF at given positions.

        Parameters
        ----------
        image : ndarray
            Image to fit
        positions : ndarray
            Array of (row, col) approximate positions
        fit_radius : int
            Radius of fitting region

        Returns
        -------
        result : LocalizationResult
            Fitted localizations
        """
        raise NotImplementedError


def gaussian_2d(xy, x0, y0, amplitude, sigma_x, sigma_y, background):
    """2D Gaussian function.

    Parameters
    ----------
    xy : tuple
        (x, y) coordinates
    x0, y0 : float
        Center position
    amplitude : float
        Peak amplitude
    sigma_x, sigma_y : float
        Standard deviations
    background : float
        Background offset

    Returns
    -------
    values : ndarray
        Gaussian values at xy positions
    """
    x, y = xy
    g = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                            ((y - y0) ** 2) / (2 * sigma_y ** 2)))
    return (g + background).ravel()


def integrated_gaussian_2d(xy, x0, y0, amplitude, sigma_x, sigma_y, background):
    """Integrated 2D Gaussian (accounts for pixel integration).

    More accurate for fitting to pixel data.
    """
    x, y = xy

    # Pixel boundaries
    x_left = x - 0.5
    x_right = x + 0.5
    y_bottom = y - 0.5
    y_top = y + 0.5

    # Integrated Gaussian using error function
    norm_x = np.sqrt(2) * sigma_x
    norm_y = np.sqrt(2) * sigma_y

    int_x = 0.5 * (erf((x_right - x0) / norm_x) - erf((x_left - x0) / norm_x))
    int_y = 0.5 * (erf((y_top - y0) / norm_y) - erf((y_bottom - y0) / norm_y))

    g = amplitude * int_x * int_y
    return (g + background).ravel()


class GaussianLSQFitter(BaseFitter):
    """Gaussian fitting using Least Squares.

    Parameters
    ----------
    integrated : bool
        Use integrated Gaussian model
    elliptical : bool
        Fit elliptical Gaussian (different sigma_x, sigma_y)
    initial_sigma : float
        Initial guess for sigma
    """

    def __init__(self, integrated=True, elliptical=False, initial_sigma=1.3):
        super().__init__("GaussianLSQ")
        self.integrated = integrated
        self.elliptical = elliptical
        self.initial_sigma = initial_sigma

    def fit(self, image, positions, fit_radius=3):
        """Fit Gaussian to positions using LSQ."""
        result = LocalizationResult()

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            region = image[r0:r1, c0:c1]

            # Create coordinate grids relative to region
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]

            # Initial parameters
            background_init = np.median(region)
            amplitude_init = region.max() - background_init
            x0_init = col - c0
            y0_init = row - r0
            sigma_init = self.initial_sigma

            if self.elliptical:
                p0 = [x0_init, y0_init, amplitude_init, sigma_init, sigma_init, background_init]
            else:
                p0 = [x0_init, y0_init, amplitude_init, sigma_init, sigma_init, background_init]

            # Choose model
            model = integrated_gaussian_2d if self.integrated else gaussian_2d

            try:
                # Fit
                popt, pcov = optimize.curve_fit(
                    model,
                    (x_coords, y_coords),
                    region.ravel(),
                    p0=p0,
                    maxfev=1000
                )

                # Convert back to image coordinates
                x_fit = popt[0] + c0
                y_fit = popt[1] + r0
                amplitude = popt[2]
                sigma_x = abs(popt[3])
                sigma_y = abs(popt[4])
                bg = popt[5]

                # Compute uncertainty (Thompson et al., 2002)
                uncertainty = self._compute_uncertainty(sigma_x, amplitude, bg, self.pixel_size)

                # Compute chi-squared
                fitted_data = model((x_coords, y_coords), *popt).reshape(region.shape)
                chi_sq = np.sum((region - fitted_data) ** 2)

                result.add_localization(
                    x=y_fit,  # Swap: y_fit contains row → now stored in x
                    y=x_fit,  # Swap: x_fit contains col → now stored in y
                    intensity=amplitude,
                    background=bg,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    uncertainty=uncertainty,
                    chi_squared=chi_sq
                )

            except (RuntimeError, ValueError):
                # Fitting failed, use approximate position
                result.add_localization(
                    x=row,  # Swap: row → x
                    y=col,  # Swap: col → y
                    intensity=0,
                    background=0,
                    sigma_x=self.initial_sigma,
                    sigma_y=self.initial_sigma,
                    uncertainty=np.inf,
                    chi_squared=np.inf
                )

        return result

    def _compute_uncertainty(self, sigma, intensity, background, pixel_size):
        """Compute localization uncertainty (Thompson et al., 2002).

        σ² = (σ_PSF² + a²/12) / N + (8πσ_PSF⁴b²) / (a²N²)

        where:
        σ_PSF = PSF standard deviation
        a = pixel size
        N = signal photons
        b = background photons per pixel
        """
        sigma_nm = sigma * pixel_size
        N = intensity * self.photons_per_adu
        b = background * self.photons_per_adu
        a = pixel_size

        if N <= 0:
            return np.inf

        term1 = (sigma_nm**2 + a**2/12) / N
        term2 = (8 * np.pi * sigma_nm**4 * b**2) / (a**2 * N**2)

        uncertainty = np.sqrt(term1 + term2)
        return uncertainty


class GaussianMLEFitter(BaseFitter):
    """Gaussian fitting using Maximum Likelihood Estimation.

    Uses Poisson likelihood for photon-counting noise.
    Based on Mortensen et al., 2010.

    Parameters
    ----------
    integrated : bool
        Use integrated Gaussian model
    elliptical : bool
        Fit elliptical Gaussian
    initial_sigma : float
        Initial guess for sigma
    """

    def __init__(self, integrated=True, elliptical=False, initial_sigma=1.3):
        super().__init__("GaussianMLE")
        self.integrated = integrated
        self.elliptical = elliptical
        self.initial_sigma = initial_sigma

    def fit(self, image, positions, fit_radius=3):
        """Fit Gaussian using MLE."""
        result = LocalizationResult()

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            region = image[r0:r1, c0:c1].astype(float)

            # Convert to photon counts
            region_photons = (region - self.baseline) * self.photons_per_adu
            region_photons = np.maximum(region_photons, 0.1)  # Avoid log(0)

            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:region.shape[0], 0:region.shape[1]]

            # Initial parameters
            background_init = np.median(region_photons)
            amplitude_init = region_photons.max() - background_init
            x0_init = col - c0
            y0_init = row - r0
            sigma_init = self.initial_sigma

            p0 = [x0_init, y0_init, amplitude_init, sigma_init, sigma_init, background_init]

            # Negative log-likelihood function
            model = integrated_gaussian_2d if self.integrated else gaussian_2d

            def neg_log_likelihood(params):
                predicted = model((x_coords, y_coords), *params).reshape(region.shape)
                predicted = np.maximum(predicted, 0.1)  # Avoid log(0)

                # Poisson log-likelihood: sum(data*log(model) - model)
                nll = -np.sum(region_photons * np.log(predicted) - predicted)
                return nll

            try:
                # Minimize negative log-likelihood
                res = optimize.minimize(
                    neg_log_likelihood,
                    p0,
                    method='L-BFGS-B',
                    bounds=[
                        (0, region.shape[1]),  # x0
                        (0, region.shape[0]),  # y0
                        (0, None),  # amplitude
                        (0.5, 5),  # sigma_x
                        (0.5, 5),  # sigma_y
                        (0, None)  # background
                    ]
                )

                popt = res.x

                # Convert back to image coordinates
                x_fit = popt[0] + c0
                y_fit = popt[1] + r0
                amplitude = popt[2]
                sigma_x = abs(popt[3])
                sigma_y = abs(popt[4])
                bg = popt[5]

                # Compute uncertainty (for MLE with photon noise)
                uncertainty = self._compute_mle_uncertainty(sigma_x, amplitude, bg)

                result.add_localization(
                    x=y_fit,  # Swap: y_fit contains row → now stored in x
                    y=x_fit,  # Swap: x_fit contains col → now stored in y
                    intensity=amplitude,
                    background=bg,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    uncertainty=uncertainty,
                    chi_squared=res.fun
                )

            except (RuntimeError, ValueError):
                # Fitting failed
                result.add_localization(
                    x=row,  # Swap: row → x
                    y=col,  # Swap: col → y
                    intensity=0,
                    background=0,
                    sigma_x=self.initial_sigma,
                    sigma_y=self.initial_sigma,
                    uncertainty=np.inf,
                    chi_squared=np.inf
                )

        return result

    def _compute_mle_uncertainty(self, sigma, intensity, background):
        """Compute MLE uncertainty (Mortensen et al., 2010)."""
        sigma_nm = sigma * self.pixel_size
        N = intensity
        b = background
        a = self.pixel_size

        if N <= 0:
            return np.inf

        # Simplified Cramér-Rao bound for MLE
        var = (sigma_nm**2 / N) * (16/9 + 8*np.pi*sigma_nm**2*b/(N*a**2))

        return np.sqrt(var)


class RadialSymmetryFitter(BaseFitter):
    """Radial symmetry method for fast subpixel localization.

    Based on Parthasarathy (2012) - non-iterative, very fast.

    Parameters
    ----------
    smoothing : bool
        Apply 3x3 smoothing to gradients
    """

    def __init__(self, smoothing=True):
        super().__init__("RadialSymmetry")
        self.smoothing = smoothing

    def fit(self, image, positions, fit_radius=3):
        """Fit using radial symmetry."""
        result = LocalizationResult()

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            region = image[r0:r1, c0:c1].astype(float)

            # Compute gradients
            gy, gx = np.gradient(region)

            # Optional smoothing of gradients
            if self.smoothing:
                gy = ndimage.uniform_filter(gy, 3)
                gx = ndimage.uniform_filter(gx, 3)

            # Compute gradient magnitude
            g_mag = np.sqrt(gx**2 + gy**2) + 1e-10

            # Normalize gradients
            gx_norm = gx / g_mag
            gy_norm = gy / g_mag

            # Create position grids
            y_grid, x_grid = np.mgrid[0:region.shape[0], 0:region.shape[1]]

            # Compute weights (gradient magnitude)
            weights = g_mag.ravel()

            # Solve for radial center using weighted least squares
            # Each gradient vector defines a line; find intersection
            x_flat = x_grid.ravel()
            y_flat = y_grid.ravel()
            gx_flat = gx_norm.ravel()
            gy_flat = gy_norm.ravel()

            # Perpendicular to gradient gives radial direction
            # Point on line: (x_i, y_i)
            # Direction perpendicular to gradient: (-gy, gx)

            # Weighted least squares solution
            denom = np.sum(weights * (gx_flat**2 + gy_flat**2))

            if denom > 0:
                x0 = np.sum(weights * (gx_flat * (gx_flat * x_flat + gy_flat * y_flat))) / denom
                y0 = np.sum(weights * (gy_flat * (gx_flat * x_flat + gy_flat * y_flat))) / denom

                # Convert to image coordinates
                x_fit = x0 + c0
                y_fit = y0 + r0

                # Estimate intensity and background
                intensity = region.max() - np.median(region)
                background = np.median(region)

                result.add_localization(
                    x=x_fit,
                    y=y_fit,
                    intensity=intensity,
                    background=background,
                    sigma_x=self.initial_sigma if hasattr(self, 'initial_sigma') else 1.3,
                    uncertainty=self.pixel_size  # Rough estimate
                )
            else:
                # Failed - use approximate position
                result.add_localization(
                    x=col,
                    y=row,
                    intensity=0,
                    background=0,
                    sigma_x=1.3,
                    uncertainty=np.inf
                )

        return result


class CentroidFitter(BaseFitter):
    """Simple centroid localization (fast but less accurate)."""

    def __init__(self):
        super().__init__("Centroid")

    def fit(self, image, positions, fit_radius=3):
        """Compute intensity-weighted centroid."""
        result = LocalizationResult()

        for pos in positions:
            row, col = int(pos[0]), int(pos[1])

            # Extract fitting region
            r0 = max(0, row - fit_radius)
            r1 = min(image.shape[0], row + fit_radius + 1)
            c0 = max(0, col - fit_radius)
            c1 = min(image.shape[1], col + fit_radius + 1)

            region = image[r0:r1, c0:c1].astype(float)

            # Subtract background
            background = np.percentile(region, 25)
            region_bg_sub = region - background
            region_bg_sub = np.maximum(region_bg_sub, 0)

            # Compute centroid
            total = region_bg_sub.sum()

            if total > 0:
                y_grid, x_grid = np.mgrid[r0:r1, c0:c1]
                x_fit = (x_grid * region_bg_sub).sum() / total
                y_fit = (y_grid * region_bg_sub).sum() / total
                intensity = region.max() - background
            else:
                x_fit = col
                y_fit = row
                intensity = 0

            result.add_localization(
                x=x_fit,
                y=y_fit,
                intensity=intensity,
                background=background,
                sigma_x=1.3,
                uncertainty=self.pixel_size
            )

        return result


def create_fitter(fitter_type, **kwargs):
    """Factory function to create fitters.

    Parameters
    ----------
    fitter_type : str
        Type of fitter: 'gaussian_lsq', 'gaussian_mle',
        'radial_symmetry', 'centroid'
    **kwargs : dict
        Fitter-specific parameters

    Returns
    -------
    fitter : BaseFitter
        Configured fitter object
    """
    fitter_map = {
        'gaussian_lsq': GaussianLSQFitter,
        'gaussian_mle': GaussianMLEFitter,
        'radial_symmetry': RadialSymmetryFitter,
        'centroid': CentroidFitter
    }

    fitter_type = fitter_type.lower()
    if fitter_type not in fitter_map:
        raise ValueError(f"Unknown fitter type: {fitter_type}")

    return fitter_map[fitter_type](**kwargs)
