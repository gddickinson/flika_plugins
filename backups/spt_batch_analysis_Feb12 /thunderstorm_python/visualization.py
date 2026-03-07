"""
Visualization Module
===================

Implements visualization methods for super-resolution reconstruction:
- Gaussian rendering
- Histogram with jittering
- Average shifted histogram
- Scatter plot
"""

import numpy as np
from scipy import ndimage


class BaseRenderer:
    """Base class for renderers."""

    def __init__(self, name="BaseRenderer"):
        self.name = name

    def render(self, localizations, pixel_size=10, image_size=None):
        """Render super-resolution image.

        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y'
        pixel_size : float
            Pixel size for rendering (nm)
        image_size : tuple, optional
            (width, height) of output image

        Returns
        -------
        image : ndarray
            Rendered image
        """
        raise NotImplementedError


class GaussianRenderer(BaseRenderer):
    """Gaussian rendering - each localization rendered as 2D Gaussian.

    Parameters
    ----------
    sigma : float or str
        Gaussian sigma in nm. Can be:
        - float: fixed sigma for all localizations
        - 'computed': use computed localization uncertainty
        - 'auto': automatically determine from data
    """

    def __init__(self, sigma=20.0):
        super().__init__("Gaussian")
        self.sigma = sigma

    def render(self, localizations, pixel_size=10, image_size=None):
        """Render using Gaussian splatting."""
        # Determine image size - use (height, width) = (rows, cols) convention for numpy arrays
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 10
            y_max = int(np.max(localizations['y']) / pixel_size) + 10
            image_size = (y_max, x_max)  # (height, width) = (rows, cols) for numpy

        image = np.zeros(image_size)

        # Determine sigma for each localization
        if isinstance(self.sigma, str):
            if self.sigma == 'computed' and 'uncertainty' in localizations:
                sigmas = localizations['uncertainty'] / pixel_size
            elif self.sigma == 'auto':
                # Use median sigma from fitted PSF
                if 'sigma_x' in localizations:
                    median_sigma_nm = np.median(localizations['sigma_x'])  # Already in nm
                    sigmas = np.full(len(localizations['x']), median_sigma_nm / pixel_size)
                else:
                    sigmas = np.full(len(localizations['x']), 2.0)  # Default to 2 pixels
            else:
                sigmas = np.full(len(localizations['x']), 2.0)
        else:
            sigmas = np.full(len(localizations['x']), self.sigma / pixel_size)

        # Render each localization
        for i in range(len(localizations['x'])):
            x = localizations['x'][i] / pixel_size
            y = localizations['y'][i] / pixel_size
            sigma = sigmas[i]

            # Determine region to render
            radius = int(3 * sigma)
            x0 = int(x) - radius
            x1 = int(x) + radius + 1
            y0 = int(y) - radius
            y1 = int(y) + radius + 1

            # Clip to image bounds (image is [rows, cols] = [y, x])
            y0_clip = max(0, y0)
            y1_clip = min(image_size[0], y1)  # image_size[0] is height (rows/y)
            x0_clip = max(0, x0)
            x1_clip = min(image_size[1], x1)  # image_size[1] is width (cols/x)

            if x0_clip >= x1_clip or y0_clip >= y1_clip:
                continue

            # Create Gaussian kernel
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))

            # Extract valid region
            gaussian_clip = gaussian[y0_clip-y0:y1_clip-y0, x0_clip-x0:x1_clip-x0]

            # Add to image
            image[y0_clip:y1_clip, x0_clip:x1_clip] += gaussian_clip

        return image


class HistogramRenderer(BaseRenderer):
    """Histogram rendering - count localizations per pixel.

    Parameters
    ----------
    jittering : bool
        Add random jitter to avoid discretization artifacts
    n_averages : int
        Number of jittered images to average (if jittering=True)
    """

    def __init__(self, jittering=False, n_averages=10):
        super().__init__("Histogram")
        self.jittering = jittering
        self.n_averages = n_averages

    def render(self, localizations, pixel_size=10, image_size=None):
        """Render as histogram."""
        # Determine image size - use (height, width) = (rows, cols) convention
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 10
            y_max = int(np.max(localizations['y']) / pixel_size) + 10
            image_size = (y_max, x_max)  # (height, width) = (rows, cols)

        if not self.jittering:
            # Simple histogram
            image = np.zeros(image_size)

            x_coords = (localizations['x'] / pixel_size).astype(int)
            y_coords = (localizations['y'] / pixel_size).astype(int)

            # Clip to bounds (image_size[0]=height/y, image_size[1]=width/x)
            valid = ((x_coords >= 0) & (x_coords < image_size[1]) &
                    (y_coords >= 0) & (y_coords < image_size[0]))

            x_coords = x_coords[valid]
            y_coords = y_coords[valid]

            # Accumulate
            for x, y in zip(x_coords, y_coords):
                image[y, x] += 1

        else:
            # Jittered histogram - average multiple random shifts
            image = np.zeros(image_size)

            for _ in range(self.n_averages):
                # Add random jitter
                jitter_x = np.random.uniform(-0.5, 0.5, len(localizations['x']))
                jitter_y = np.random.uniform(-0.5, 0.5, len(localizations['y']))

                x_coords = ((localizations['x'] / pixel_size) + jitter_x).astype(int)
                y_coords = ((localizations['y'] / pixel_size) + jitter_y).astype(int)

                # Clip to bounds (image_size[0]=height/y, image_size[1]=width/x)
                valid = ((x_coords >= 0) & (x_coords < image_size[1]) &
                        (y_coords >= 0) & (y_coords < image_size[0]))

                x_coords = x_coords[valid]
                y_coords = y_coords[valid]

                # Accumulate
                temp = np.zeros(image_size)
                for x, y in zip(x_coords, y_coords):
                    temp[y, x] += 1

                image += temp

            image /= self.n_averages

        return image


class AverageShiftedHistogram(BaseRenderer):
    """Average Shifted Histogram (ASH) rendering.

    Faster alternative to Gaussian rendering with similar quality.
    Based on Scott (1985).

    Parameters
    ----------
    n_shifts : int
        Number of shifts (2, 4, or 8)
    """

    def __init__(self, n_shifts=4):
        super().__init__("ASH")
        self.n_shifts = n_shifts

    def render(self, localizations, pixel_size=10, image_size=None):
        """Render using average shifted histogram."""
        # Determine image size - use (height, width) = (rows, cols) convention
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 10
            y_max = int(np.max(localizations['y']) / pixel_size) + 10
            image_size = (y_max, x_max)  # (height, width) = (rows, cols)

        # Generate shift patterns
        if self.n_shifts == 2:
            shifts = [(0, 0), (0.5, 0.5)]
        elif self.n_shifts == 4:
            shifts = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]
        elif self.n_shifts == 8:
            shifts = [(0, 0), (0.33, 0), (0.67, 0), (0, 0.33),
                     (0.33, 0.33), (0.67, 0.33), (0, 0.67), (0.33, 0.67)]
        else:
            shifts = [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]

        # Accumulate shifted histograms
        image = np.zeros(image_size)

        for shift_x, shift_y in shifts:
            x_coords = ((localizations['x'] / pixel_size) + shift_x).astype(int)
            y_coords = ((localizations['y'] / pixel_size) + shift_y).astype(int)

            # Clip to bounds (image_size[0]=height/y, image_size[1]=width/x)
            valid = ((x_coords >= 0) & (x_coords < image_size[1]) &
                    (y_coords >= 0) & (y_coords < image_size[0]))

            x_coords = x_coords[valid]
            y_coords = y_coords[valid]

            # Accumulate
            temp = np.zeros(image_size)
            for x, y in zip(x_coords, y_coords):
                temp[y, x] += 1

            # Smooth with box filter
            temp = ndimage.uniform_filter(temp, size=3)

            image += temp

        # Average
        image /= len(shifts)

        return image


class ScatterRenderer(BaseRenderer):
    """Simple scatter plot - mark localization positions."""

    def __init__(self):
        super().__init__("Scatter")

    def render(self, localizations, pixel_size=10, image_size=None):
        """Render as scatter plot."""
        # Determine image size - use (height, width) = (rows, cols) convention
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 10
            y_max = int(np.max(localizations['y']) / pixel_size) + 10
            image_size = (y_max, x_max)  # (height, width) = (rows, cols)

        image = np.zeros(image_size)

        x_coords = (localizations['x'] / pixel_size).astype(int)
        y_coords = (localizations['y'] / pixel_size).astype(int)

        # Clip to bounds (image_size[0]=height/y, image_size[1]=width/x)
        valid = ((x_coords >= 0) & (x_coords < image_size[1]) &
                (y_coords >= 0) & (y_coords < image_size[0]))

        x_coords = x_coords[valid]
        y_coords = y_coords[valid]

        # Mark positions
        image[y_coords, x_coords] = 1

        return image


def render_3d_projection(localizations, pixel_size=10, z_range=None,
                        n_slices=10, colorize=True):
    """Render 3D data as slices or projection.

    Parameters
    ----------
    localizations : dict
        Localization data with 'x', 'y', 'z'
    pixel_size : float
        Pixel size (nm)
    z_range : tuple, optional
        (z_min, z_max) for slicing
    n_slices : int
        Number of Z slices
    colorize : bool
        Color-code by Z position

    Returns
    -------
    slices : list of ndarray or ndarray
        List of 2D images (slices) or single RGB image if colorize=True
    """
    if 'z' not in localizations:
        raise ValueError("3D rendering requires 'z' coordinate")

    # Determine Z range
    if z_range is None:
        z_min = np.min(localizations['z'])
        z_max = np.max(localizations['z'])
    else:
        z_min, z_max = z_range

    z_step = (z_max - z_min) / n_slices

    # Render each slice
    slices = []
    renderer = HistogramRenderer()

    for i in range(n_slices):
        z_low = z_min + i * z_step
        z_high = z_min + (i + 1) * z_step

        # Filter localizations in this Z range
        mask = (localizations['z'] >= z_low) & (localizations['z'] < z_high)

        slice_locs = {}
        for key in ['x', 'y']:
            slice_locs[key] = localizations[key][mask]

        if len(slice_locs['x']) > 0:
            slice_img = renderer.render(slice_locs, pixel_size=pixel_size)
        else:
            # Empty slice
            slice_img = np.zeros((100, 100))

        slices.append(slice_img)

    if colorize:
        # Create RGB image with color-coded Z
        # Combine slices with different colors
        max_shape = max([s.shape for s in slices])
        rgb_image = np.zeros((max_shape[0], max_shape[1], 3))

        for i, slice_img in enumerate(slices):
            # Color based on Z position
            color_weight = i / len(slices)

            # Resize if needed
            if slice_img.shape != max_shape:
                padded = np.zeros(max_shape)
                padded[:slice_img.shape[0], :slice_img.shape[1]] = slice_img
                slice_img = padded

            # Add to RGB channels with color gradient
            rgb_image[:, :, 0] += slice_img * (1 - color_weight)  # Red for low Z
            rgb_image[:, :, 2] += slice_img * color_weight  # Blue for high Z
            rgb_image[:, :, 1] += slice_img * 0.5  # Green for mid Z

        return rgb_image

    return slices


def create_renderer(renderer_type, **kwargs):
    """Factory function to create renderers.

    Parameters
    ----------
    renderer_type : str
        Type of renderer: 'gaussian', 'histogram', 'ash', 'scatter'
    **kwargs : dict
        Renderer-specific parameters

    Returns
    -------
    renderer : BaseRenderer
        Configured renderer object
    """
    renderer_map = {
        'gaussian': GaussianRenderer,
        'histogram': HistogramRenderer,
        'ash': AverageShiftedHistogram,
        'scatter': ScatterRenderer
    }

    renderer_type = renderer_type.lower()
    if renderer_type not in renderer_map:
        raise ValueError(f"Unknown renderer type: {renderer_type}")

    return renderer_map[renderer_type](**kwargs)


def apply_colormap(image, cmap='hot'):
    """Apply colormap to grayscale image.

    Parameters
    ----------
    image : ndarray
        Grayscale image
    cmap : str
        Colormap name ('hot', 'viridis', 'gray', etc.)

    Returns
    -------
    colored : ndarray
        RGB image
    """
    # Normalize to [0, 1]
    img_norm = (image - image.min()) / (image.max() - image.min() + 1e-10)

    # Simple colormaps
    if cmap == 'hot':
        # Hot colormap: black -> red -> yellow -> white
        rgb = np.zeros((*image.shape, 3))
        rgb[:, :, 0] = np.clip(img_norm * 3, 0, 1)
        rgb[:, :, 1] = np.clip((img_norm - 0.33) * 3, 0, 1)
        rgb[:, :, 2] = np.clip((img_norm - 0.67) * 3, 0, 1)
    elif cmap == 'gray':
        rgb = np.stack([img_norm] * 3, axis=-1)
    elif cmap == 'viridis':
        # Approximation of viridis
        rgb = np.zeros((*image.shape, 3))
        rgb[:, :, 0] = 0.267 + 0.005 * img_norm
        rgb[:, :, 1] = 0.005 + 0.55 * img_norm
        rgb[:, :, 2] = 0.329 + 0.55 * img_norm
    else:
        # Default to grayscale
        rgb = np.stack([img_norm] * 3, axis=-1)

    return rgb
