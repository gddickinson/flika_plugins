"""
Image Filtering Module
======================

Implements all filtering methods from thunderSTORM for feature enhancement:
- Gaussian filters
- Wavelet filters (à trous algorithm)
- Difference of Gaussians (DoG)
- Difference of Averaging filters
- Lowered Gaussian (band-pass)
- Box filters
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d
import pywt


class BaseFilter:
    """Base class for all image filters.

    All filters export named variables via the `variables` dict after
    ``apply()`` is called.  At minimum every filter stores:
      - ``I``  – the original input image
      - ``F``  – the final filtered output

    Subclasses may add additional intermediate variables (e.g.
    ``G1``, ``G2`` for DoG) so that threshold expressions like
    ``std(DoG.G1)`` work correctly, matching the original thunderSTORM.
    """

    def __init__(self, name="BaseFilter"):
        self.name = name
        self.variables = {}  # populated after apply()

    def apply(self, image):
        """Apply filter to image.

        Parameters
        ----------
        image : ndarray
            2D image array

        Returns
        -------
        filtered : ndarray
            Filtered image
        """
        raise NotImplementedError

    def _store_variables(self, original, filtered, **extras):
        """Store standard + extra variables after filtering.

        Parameters
        ----------
        original : ndarray
            Original input image
        filtered : ndarray
            Final filtered result
        **extras : ndarray
            Additional named variables (e.g. G1=gauss1, G2=gauss2)
        """
        self.variables = {'I': original.astype(float), 'F': filtered, 'F1': filtered}
        self.variables.update(extras)


class GaussianFilter(BaseFilter):
    """Gaussian lowpass filter.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian kernel
    """

    def __init__(self, sigma=1.6):
        super().__init__("Gaussian")
        self.sigma = sigma

    def apply(self, image):
        """Apply Gaussian filter."""
        img = image.astype(float)
        filtered = ndimage.gaussian_filter(img, self.sigma)
        self._store_variables(image, filtered)
        return filtered


class WaveletFilter(BaseFilter):
    """Wavelet filter using B-spline à trous algorithm.

    Matches ThunderSTORM's CompoundWaveletFilter exactly:
    - Generates a B-spline kernel from order and scale parameters
    - Always uses exactly 2 planes (levels 1 and 2)
    - Output F = v1 - v2 (band-pass between planes)
    - F1 = image - v1 (fine detail, used by threshold expressions)

    Parameters
    ----------
    scale : float
        B-spline scale (controls kernel width, NOT decomposition levels)
    order : int
        B-spline order (3 for cubic B-spline)
    """

    def __init__(self, scale=2, order=3):
        super().__init__("Wavelet")
        self.scale = float(scale)
        self.order = int(order)

    @staticmethod
    def _bspline_N(k, t):
        """Recursive B-spline basis function N(k, t).

        Matches ThunderSTORM's BSpline.N() exactly.
        """
        if k <= 1:
            # Haar basis
            return 1.0 if (t >= 0.0 and t < 1.0) else 0.0
        else:
            Nt = WaveletFilter._bspline_N(k - 1, t)
            Nt_1 = WaveletFilter._bspline_N(k - 1, t - 1)
            return t / (k - 1) * Nt + (k - t) / (k - 1) * Nt_1

    @staticmethod
    def _create_bspline_kernel(order, scale):
        """Generate B-spline kernel matching ThunderSTORM's BSpline.create().

        BSpline.create(k, s, t) = normalize(N(k, t/s + k/2))
        where t = [-nSamples/2, ..., nSamples/2-1]
        and nSamples = order * scale - 1
        """
        n_samples = int(order * scale - 1)
        # Match Kotlin integer division: -nSamples/2
        half = n_samples // 2
        t = np.arange(-half, -half + n_samples, dtype=float)

        # Evaluate B-spline: N(k, t/s + k/2)
        kernel = np.array([
            WaveletFilter._bspline_N(order, ti / scale + order / 2.0)
            for ti in t
        ])

        # Normalize to sum = 1
        s = kernel.sum()
        if s > 0:
            kernel /= s

        return kernel

    def apply(self, image):
        """Apply wavelet filter matching ThunderSTORM's CompoundWaveletFilter.

        Uses exactly 2 planes:
        - Plane 1: convolve with base B-spline kernel → v1
        - Plane 2: convolve v1 with upsampled kernel (2x spacing) → v2
        - Output F = v1 - v2 (band-pass)
        - F1 = image - v1 (fine detail for threshold expressions)
        """
        img = image.astype(float)

        # Generate B-spline kernel
        h = self._create_bspline_kernel(self.order, self.scale)
        n_samples = len(h)

        # Plane 2 kernel: upsample by inserting 1 zero between elements
        # kernelSize_plane2 = 2 * (nSamples - 1) + 1
        h2_len = 2 * (n_samples - 1) + 1
        h2 = np.zeros(h2_len)
        for i in range(n_samples):
            h2[i * 2] = h[i]

        # Margin for padding (= w2.kernelSize / 2)
        margin = h2_len // 2

        # Pad image with duplicate (replicate) padding to match ThunderSTORM
        padded = np.pad(img, margin, mode='edge')

        # Plane 1: convolve padded image with base kernel (separable)
        temp = convolve1d(padded, h, axis=1, mode='nearest')
        v1 = convolve1d(temp, h, axis=0, mode='nearest')

        # Plane 2: convolve v1 with upsampled kernel (separable)
        temp = convolve1d(v1, h2, axis=1, mode='nearest')
        v2 = convolve1d(temp, h2, axis=0, mode='nearest')

        # Crop back to original size
        v1_crop = v1[margin:margin + img.shape[0], margin:margin + img.shape[1]]
        v2_crop = v2[margin:margin + img.shape[0], margin:margin + img.shape[1]]

        # Output: band-pass between planes
        filtered = v1_crop - v2_crop

        # F1 = image - v1 (fine detail, used by std(Wave.F1) threshold)
        # F2 = image - v2 (broader detail)
        self.F1 = img - v1_crop

        self._store_variables(image, filtered,
                              F1=self.F1,
                              F2=img - v2_crop)
        return filtered


class DifferenceOfGaussians(BaseFilter):
    """Difference of Gaussians (DoG) band-pass filter.

    Parameters
    ----------
    sigma1 : float
        Sigma for first Gaussian
    sigma2 : float
        Sigma for second Gaussian (should be > sigma1)
    """

    def __init__(self, sigma1=1.0, sigma2=1.6):
        super().__init__("DoG")
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply(self, image):
        """Apply DoG filter."""
        img = image.astype(float)
        gauss1 = ndimage.gaussian_filter(img, self.sigma1)
        gauss2 = ndimage.gaussian_filter(img, self.sigma2)
        filtered = gauss1 - gauss2
        self._store_variables(image, filtered, G1=gauss1, G2=gauss2)
        return filtered


class LoweredGaussian(BaseFilter):
    """Lowered Gaussian (zero-mean) filter.

    Matches original thunderSTORM: constructs a Gaussian kernel and
    subtracts its mean so that the kernel sums to zero.  This is a
    band-pass filter that removes both high-frequency noise and
    low-frequency background in a single convolution.

    Parameters
    ----------
    sigma : float
        Gaussian sigma
    """

    def __init__(self, sigma=1.6, **kwargs):
        super().__init__("LoweredGaussian")
        self.sigma = sigma

    def apply(self, image):
        """Apply lowered Gaussian filter (zero-mean kernel)."""
        img = image.astype(float)
        # Build explicit 2D Gaussian kernel
        radius = int(np.ceil(3.0 * self.sigma))
        size = 2 * radius + 1
        ax = np.arange(-radius, radius + 1, dtype=float)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * self.sigma**2))
        kernel /= kernel.sum()
        # Subtract mean → zero-sum kernel (matches thunderSTORM)
        kernel -= kernel.mean()
        filtered = convolve2d(img, kernel, mode='same', boundary='symm')
        self._store_variables(image, filtered)
        return filtered


class DifferenceOfAveraging(BaseFilter):
    """Difference of averaging (box) filters.

    Parameters
    ----------
    size1 : int
        Size of first averaging kernel
    size2 : int
        Size of second averaging kernel (should be > size1)
    """

    def __init__(self, size1=3, size2=5):
        super().__init__("DiffAvg")
        self.size1 = size1
        self.size2 = size2

    def apply(self, image):
        """Apply difference of averaging filters."""
        img = image.astype(float)
        avg1 = ndimage.uniform_filter(img, self.size1)
        avg2 = ndimage.uniform_filter(img, self.size2)
        filtered = avg1 - avg2
        self._store_variables(image, filtered, B1=avg1, B2=avg2)
        return filtered


class MedianFilter(BaseFilter):
    """Median filter for noise reduction.

    Supports BOX (square) and CROSS (plus-shaped) patterns, matching
    the original thunderSTORM median filter options.

    Parameters
    ----------
    size : int
        Size of median filter kernel
    pattern : str
        'box' for square neighbourhood, 'cross' for plus-shaped
        (4-connected) neighbourhood.  Default 'box'.
    """

    def __init__(self, size=3, pattern='box'):
        super().__init__("Median")
        self.size = size
        self.pattern = pattern.lower()

    def apply(self, image):
        """Apply median filter."""
        if self.pattern == 'cross':
            # Build cross-shaped footprint
            radius = self.size // 2
            fp = np.zeros((self.size, self.size), dtype=bool)
            fp[radius, :] = True   # horizontal bar
            fp[:, radius] = True   # vertical bar
            filtered = ndimage.median_filter(image, footprint=fp)
        else:
            filtered = ndimage.median_filter(image, self.size)
        self._store_variables(image, filtered.astype(float))
        return filtered


class BoxFilter(BaseFilter):
    """Simple averaging (box) filter.

    Parameters
    ----------
    size : int
        Size of box kernel
    """

    def __init__(self, size=3):
        super().__init__("Box")
        self.size = size

    def apply(self, image):
        """Apply box filter."""
        filtered = ndimage.uniform_filter(image.astype(float), self.size)
        self._store_variables(image, filtered)
        return filtered


class NoFilter(BaseFilter):
    """Pass-through filter (no filtering)."""

    def __init__(self):
        super().__init__("NoFilter")

    def apply(self, image):
        """Return image unchanged."""
        filtered = image.astype(float)
        self._store_variables(image, filtered)
        return filtered


def create_filter(filter_type, **kwargs):
    """Factory function to create filters.

    Parameters
    ----------
    filter_type : str
        Type of filter: 'gaussian', 'wavelet', 'dog', 'lowered_gaussian',
        'diff_avg', 'median', 'box', or 'none'
    **kwargs : dict
        Filter-specific parameters

    Returns
    -------
    filter : BaseFilter
        Configured filter object
    """
    filter_map = {
        'gaussian': GaussianFilter,
        'wavelet': WaveletFilter,
        'dog': DifferenceOfGaussians,
        'lowered_gaussian': LoweredGaussian,
        'diff_avg': DifferenceOfAveraging,
        'median': MedianFilter,
        'box': BoxFilter,
        'none': NoFilter
    }

    filter_type = filter_type.lower()
    if filter_type not in filter_map:
        raise ValueError(f"Unknown filter type: {filter_type}")

    return filter_map[filter_type](**kwargs)


def compute_threshold_expression(image, filtered_image, expression,
                                 wave_f1=None, filter_obj=None):
    """Evaluate threshold expression.

    ThunderSTORM allows threshold specification using expressions like:
    - 'std(Wave.F1)' - standard deviation of wavelet level 1
    - '2*std(Wave.F1)' - 2x standard deviation
    - 'mean(I1) + 3*std(I1)' - mean + 3*std of raw image
    - 'std(DoG.G1)' - std of first Gaussian in DoG filter

    Supports filter-prefixed variable names matching the original
    thunderSTORM convention:  ``<FilterName>.<VarName>`` is resolved
    from the filter object's ``variables`` dict.

    Parameters
    ----------
    image : ndarray
        Raw image
    filtered_image : ndarray
        Filtered image
    expression : str or float
        Threshold expression or value
    wave_f1 : ndarray, optional
        First wavelet detail coefficient (I - V1). When provided, this
        is used as Wave.F1 in threshold expressions instead of the
        final filtered output. This matches real thunderSTORM where
        Wave.F1 always refers to the first detail coefficient.
    filter_obj : BaseFilter, optional
        The filter instance that produced *filtered_image*.  If
        provided, its ``variables`` dict is used to resolve
        filter-prefixed names (e.g. ``DoG.G1``, ``Wave.F2``).

    Returns
    -------
    threshold : float
        Computed threshold value
    """
    if isinstance(expression, (int, float)):
        return float(expression)

    expression_str = str(expression)

    # Build namespace -------------------------------------------------------
    # Standard functions (matching thunderSTORM's formula parser)
    namespace = {
        'std': np.std,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'var': np.var,
        'sum': np.sum,
        'abs': np.abs,
    }

    # Default variables
    namespace['I'] = image.astype(float)
    namespace['I1'] = image.astype(float)   # legacy alias
    namespace['F'] = filtered_image

    # Inject filter-exported variables (un-prefixed) if available
    if filter_obj is not None and hasattr(filter_obj, 'variables'):
        namespace.update(filter_obj.variables)

    # wave_f1 takes priority for F1 (e.g. wavelet F1 computed separately
    # for threshold when using a non-wavelet filter)
    f1_data = wave_f1 if wave_f1 is not None else filtered_image
    namespace['F1'] = f1_data

    # Resolve filter-prefixed names:  "Wave.F1" → "F1",  "DoG.G1" → "G1"
    # We replace "<Name>.<Var>" with just "<Var>" since the un-prefixed
    # variable is already in the namespace from the filter export above.
    import re
    expression_parsed = re.sub(r'([A-Za-z]\w*)\.(\w+)', r'\2', expression_str)

    # thunderSTORM uses ^ for power; Python uses **
    expression_parsed = expression_parsed.replace('^', '**')

    try:
        threshold = eval(expression_parsed, {"__builtins__": {}}, namespace)
        return float(threshold)
    except Exception as e:
        raise ValueError(f"Invalid threshold expression '{expression}': {e}")
