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
from scipy.signal import convolve2d
import pywt


class BaseFilter:
    """Base class for all image filters."""

    def __init__(self, name="BaseFilter"):
        self.name = name

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
        return ndimage.gaussian_filter(image.astype(float), self.sigma)


class WaveletFilter(BaseFilter):
    """Wavelet filter using à trous (undecimated) algorithm.

    Implements B-spline wavelet transform as used in thunderSTORM.
    Based on Izeddin et al. (2012).

    Parameters
    ----------
    scale : int
        Wavelet scale (1-5 typically)
    order : int
        B-spline order (3 for cubic B-spline)
    """

    def __init__(self, scale=2, order=3):
        super().__init__("Wavelet")
        self.scale = scale
        self.order = order

    def apply(self, image):
        """Apply wavelet filter using à trous algorithm."""
        # Implement à trous algorithm for B-spline wavelets
        img = image.astype(float)

        # Generate B-spline filter coefficients
        if self.order == 3:
            # Cubic B-spline filter
            h = np.array([1/16, 1/4, 3/8, 1/4, 1/16])
        elif self.order == 1:
            # Linear B-spline
            h = np.array([1/4, 1/2, 1/4])
        else:
            # Default to cubic
            h = np.array([1/16, 1/4, 3/8, 1/4, 1/16])

        # Apply à trous algorithm
        approximation = img.copy()

        for j in range(self.scale):
            # Upsampling by inserting zeros
            h_upsampled = np.zeros(len(h) + (len(h) - 1) * (2**j - 1))
            indices = np.arange(len(h)) * 2**j
            h_upsampled[indices] = h

            # Convolve rows
            temp = np.zeros_like(img)
            for i in range(img.shape[0]):
                temp[i] = np.convolve(approximation[i], h_upsampled, mode='same')

            # Convolve columns
            approx_new = np.zeros_like(img)
            for j_col in range(img.shape[1]):
                approx_new[:, j_col] = np.convolve(temp[:, j_col], h_upsampled, mode='same')

            # Wavelet coefficient is difference
            wavelet = approximation - approx_new
            approximation = approx_new

        return wavelet


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
        return gauss1 - gauss2


class LoweredGaussian(BaseFilter):
    """Lowered Gaussian band-pass filter.

    Gaussian minus averaging filter.

    Parameters
    ----------
    sigma : float
        Gaussian sigma
    size : int
        Size of averaging kernel
    """

    def __init__(self, sigma=1.6, size=3):
        super().__init__("LoweredGaussian")
        self.sigma = sigma
        self.size = size

    def apply(self, image):
        """Apply lowered Gaussian filter."""
        img = image.astype(float)
        gauss = ndimage.gaussian_filter(img, self.sigma)
        avg = ndimage.uniform_filter(img, self.size)
        return gauss - avg


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
        return avg1 - avg2


class MedianFilter(BaseFilter):
    """Median filter for noise reduction.

    Parameters
    ----------
    size : int
        Size of median filter kernel
    """

    def __init__(self, size=3):
        super().__init__("Median")
        self.size = size

    def apply(self, image):
        """Apply median filter."""
        return ndimage.median_filter(image, self.size)


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
        return ndimage.uniform_filter(image.astype(float), self.size)


class NoFilter(BaseFilter):
    """Pass-through filter (no filtering)."""

    def __init__(self):
        super().__init__("NoFilter")

    def apply(self, image):
        """Return image unchanged."""
        return image.astype(float)


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


def compute_threshold_expression(image, filtered_image, expression):
    """Evaluate threshold expression.

    ThunderSTORM allows threshold specification using expressions like:
    - 'std(Wave.F1)' - standard deviation of wavelet level 1
    - '2*std(Wave.F1)' - 2x standard deviation
    - 'mean(I1) + 3*std(I1)' - mean + 3*std of raw image

    Parameters
    ----------
    image : ndarray
        Raw image
    filtered_image : ndarray
        Filtered image
    expression : str or float
        Threshold expression or value

    Returns
    -------
    threshold : float
        Computed threshold value
    """
    if isinstance(expression, (int, float)):
        return float(expression)

    # Replace Wave.F1 notation with F1 for compatibility
    # This allows expressions like 'std(Wave.F1)' to work
    expression_parsed = str(expression).replace('Wave.F1', 'F1')

    # Create namespace for expression evaluation
    namespace = {
        'std': np.std,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'min': np.min,
        'I1': image,  # Raw image
        'F1': filtered_image  # Filtered image (Wave.F1 in thunderSTORM)
    }

    try:
        threshold = eval(expression_parsed, {"__builtins__": {}}, namespace)
        return float(threshold)
    except Exception as e:
        raise ValueError(f"Invalid threshold expression '{expression}': {e}")
