"""
Molecule Detection Module
=========================

Implements methods for detecting approximate molecular positions:
- Local maximum detection
- Non-maximum suppression
- Centroid of connected components

Updated with improved edge/border handling to prevent edge artifacts.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max


def safe_border_width(fit_radius):
    """Determine safe border width based on fitting parameters.

    Parameters
    ----------
    fit_radius : int
        Fitting radius in pixels

    Returns
    -------
    border_width : int
        Recommended border exclusion width
    """
    return max(5, int(fit_radius * 1.5))  # At least 5 pixels, or 1.5x fit radius


class BaseDetector:
    """Base class for molecule detectors."""

    def __init__(self, name="BaseDetector"):
        self.name = name

    def detect(self, image, threshold):
        """Detect molecules in image.

        Parameters
        ----------
        image : ndarray
            Filtered image
        threshold : float
            Detection threshold

        Returns
        -------
        positions : ndarray
            Array of (row, col) positions, shape (N, 2)
        """
        raise NotImplementedError


class LocalMaximumDetector(BaseDetector):
    """Detect molecules as local maxima.

    Parameters
    ----------
    connectivity : str
        '4-neighbourhood' or '8-neighbourhood'
    min_distance : int
        Minimum distance between peaks
    exclude_border : int or bool
        Border exclusion width (int) or True for automatic
    """

    def __init__(self, connectivity='8-neighbourhood', min_distance=1, exclude_border=True):
        super().__init__("LocalMaximum")
        self.connectivity = connectivity
        self.min_distance = min_distance
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect local maxima above threshold."""
        # Determine border exclusion
        if self.exclude_border is True:
            border_pixels = max(3, self.min_distance)
        elif isinstance(self.exclude_border, int):
            border_pixels = self.exclude_border
        else:
            border_pixels = False

        # Use peak_local_max for robust detection
        coordinates = peak_local_max(
            image,
            min_distance=self.min_distance,
            threshold_abs=threshold,
            exclude_border=border_pixels if border_pixels else False
        )

        # Additional safety border filtering
        if border_pixels and len(coordinates) > 0:
            coordinates = remove_border_detections(
                coordinates,
                image.shape,
                border=border_pixels
            )

        return coordinates


class NonMaximumSuppression(BaseDetector):
    """Non-maximum suppression detector.

    Finds local maxima using morphological operations.

    Parameters
    ----------
    connectivity : int
        Connectivity for morphological operations (1 or 2)
    exclude_border : int or bool
        Border exclusion width
    """

    def __init__(self, connectivity=2, exclude_border=True):
        super().__init__("NonMaxSuppression")
        self.connectivity = connectivity
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect using non-maximum suppression."""
        # Threshold image
        binary = image > threshold

        # Find local maxima using morphological dilation
        struct = ndimage.generate_binary_structure(2, self.connectivity)
        dilated = ndimage.grey_dilation(image, footprint=struct)

        # Maxima are where image equals dilated image
        maxima = (image == dilated) & binary

        # Get coordinates
        coordinates = np.column_stack(np.where(maxima))

        # Apply border exclusion
        if self.exclude_border and len(coordinates) > 0:
            border_width = 3 if self.exclude_border is True else self.exclude_border
            coordinates = remove_border_detections(
                coordinates,
                image.shape,
                border=border_width
            )

        return coordinates


class CentroidDetector(BaseDetector):
    """Centroid of connected components detector.

    Segments image and finds centroids of connected regions.

    Parameters
    ----------
    connectivity : int
        Connectivity for connected components (1 or 2)
    min_area : int
        Minimum area of components to keep
    exclude_border : int or bool
        Border exclusion width
    """

    def __init__(self, connectivity=2, min_area=1, exclude_border=True):
        super().__init__("Centroid")
        self.connectivity = connectivity
        self.min_area = min_area
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Detect centroids of connected components."""
        # Threshold image
        binary = image > threshold

        # Label connected components
        labeled = measure.label(binary, connectivity=self.connectivity)

        # Get region properties
        regions = measure.regionprops(labeled, intensity_image=image)

        # Extract centroids of regions above min_area
        coordinates = []
        for region in regions:
            if region.area >= self.min_area:
                # Use weighted centroid (intensity-weighted)
                coordinates.append([region.weighted_centroid[0],
                                  region.weighted_centroid[1]])

        if not coordinates:
            return np.array([]).reshape(0, 2)

        coordinates = np.array(coordinates)

        # Apply border exclusion
        if self.exclude_border and len(coordinates) > 0:
            border_width = 3 if self.exclude_border is True else self.exclude_border
            coordinates = remove_border_detections(
                coordinates,
                image.shape,
                border=border_width
            )

        return coordinates


class GridDetector(BaseDetector):
    """Detect molecules on a regular grid.

    Useful for simulations or structured samples.

    Parameters
    ----------
    spacing : int
        Grid spacing in pixels
    exclude_border : int or bool
        Border exclusion width
    """

    def __init__(self, spacing=10, exclude_border=True):
        super().__init__("Grid")
        self.spacing = spacing
        self.exclude_border = exclude_border

    def detect(self, image, threshold):
        """Generate grid of positions."""
        rows, cols = image.shape

        # Determine border offset
        if self.exclude_border is True:
            border = self.spacing // 2
        elif isinstance(self.exclude_border, int):
            border = self.exclude_border
        else:
            border = 0

        # Create grid with border offset
        row_coords = np.arange(border, rows - border, self.spacing)
        col_coords = np.arange(border, cols - border, self.spacing)

        grid_rows, grid_cols = np.meshgrid(row_coords, col_coords, indexing='ij')
        coordinates = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])

        # Filter by threshold
        if len(coordinates) > 0:
            row_idx = coordinates[:, 0].astype(int)
            col_idx = coordinates[:, 1].astype(int)

            # Ensure indices are within bounds
            valid = ((row_idx >= 0) & (row_idx < rows) &
                    (col_idx >= 0) & (col_idx < cols))

            coordinates = coordinates[valid]
            row_idx = row_idx[valid]
            col_idx = col_idx[valid]

            if len(coordinates) > 0:
                intensities = image[row_idx, col_idx]
                mask = intensities > threshold
                coordinates = coordinates[mask]

        return coordinates


def create_detector(detector_type, **kwargs):
    """Factory function to create detectors.

    Parameters
    ----------
    detector_type : str
        Type of detector: 'local_maximum', 'non_maximum_suppression',
        'centroid', or 'grid'
    **kwargs : dict
        Detector-specific parameters

    Returns
    -------
    detector : BaseDetector
        Configured detector object
    """
    detector_map = {
        'local_maximum': LocalMaximumDetector,
        'non_maximum_suppression': NonMaximumSuppression,
        'centroid': CentroidDetector,
        'grid': GridDetector
    }

    detector_type = detector_type.lower()
    if detector_type not in detector_map:
        raise ValueError(f"Unknown detector type: {detector_type}")

    return detector_map[detector_type](**kwargs)


def refine_detections(detections, image, radius=3):
    """Refine detected positions to subpixel accuracy using centroid.

    Parameters
    ----------
    detections : ndarray
        Array of (row, col) positions
    image : ndarray
        Image for computing weighted centroid
    radius : int
        Radius around detection for centroid calculation

    Returns
    -------
    refined : ndarray
        Refined positions
    """
    if len(detections) == 0:
        return detections

    refined = []
    rows, cols = image.shape

    for pos in detections:
        row, col = int(pos[0]), int(pos[1])

        # Extract local region
        r0 = max(0, row - radius)
        r1 = min(rows, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(cols, col + radius + 1)

        region = image[r0:r1, c0:c1]

        # Compute weighted centroid
        total = region.sum()
        if total > 0:
            rows_grid, cols_grid = np.mgrid[r0:r1, c0:c1]
            row_center = (rows_grid * region).sum() / total
            col_center = (cols_grid * region).sum() / total
            refined.append([row_center, col_center])
        else:
            refined.append(pos)

    return np.array(refined)


def filter_detections_by_intensity(detections, image, min_intensity=None, max_intensity=None):
    """Filter detections based on intensity criteria.

    Parameters
    ----------
    detections : ndarray
        Array of (row, col) positions
    image : ndarray
        Image for intensity lookup
    min_intensity : float, optional
        Minimum intensity threshold
    max_intensity : float, optional
        Maximum intensity threshold

    Returns
    -------
    filtered : ndarray
        Filtered positions
    """
    if len(detections) == 0:
        return detections

    # Get intensities at detection positions
    rows = np.clip(detections[:, 0].astype(int), 0, image.shape[0]-1)
    cols = np.clip(detections[:, 1].astype(int), 0, image.shape[1]-1)
    intensities = image[rows, cols]

    # Apply filters
    mask = np.ones(len(detections), dtype=bool)

    if min_intensity is not None:
        mask &= (intensities >= min_intensity)

    if max_intensity is not None:
        mask &= (intensities <= max_intensity)

    return detections[mask]


def remove_border_detections(detections, image_shape, border=5):
    """Remove detections too close to image borders.

    This is critical for preventing edge artifacts in SMLM analysis,
    as PSF fitting requires complete data around each detection.

    CRITICAL: Coordinate system conventions:
    - detections is Nx2 array in (row, col) format from peak detection
    - detections[:, 0] = row coordinate (y-direction, vertical)
    - detections[:, 1] = col coordinate (x-direction, horizontal)
    - image_shape is (rows, cols) = (height, width)

    Parameters
    ----------
    detections : ndarray
        Array of (row, col) positions
    image_shape : tuple
        Shape of image (rows, cols) = (height, width)
    border : int
        Border width in pixels

    Returns
    -------
    filtered : ndarray
        Detections away from borders
    """
    if len(detections) == 0:
        return detections

    # Unpack dimensions explicitly
    # image_shape is (rows, cols) = (height, width)
    height, width = image_shape

    # detections[:, 0] is row (y), must be in [border, height-border)
    # detections[:, 1] is col (x), must be in [border, width-border)
    mask = ((detections[:, 0] >= border) &
            (detections[:, 0] < height - border) &
            (detections[:, 1] >= border) &
            (detections[:, 1] < width - border))

    return detections[mask]


def validate_detections_in_bounds(detections, image_shape, tolerance=0.5):
    """Validate that all detections are within image bounds.

    Removes any detections outside valid image area, with optional
    tolerance for subpixel positions near edges.

    Parameters
    ----------
    detections : ndarray
        Array of (row, col) positions (can be subpixel)
    image_shape : tuple
        Shape of image (rows, cols)
    tolerance : float
        Tolerance for positions outside bounds (default 0.5 pixels)

    Returns
    -------
    filtered : ndarray
        Detections within valid bounds
    """
    if len(detections) == 0:
        return detections

    rows, cols = image_shape

    # Allow small tolerance for subpixel precision
    mask = ((detections[:, 0] >= -tolerance) &
            (detections[:, 0] < rows + tolerance) &
            (detections[:, 1] >= -tolerance) &
            (detections[:, 1] < cols + tolerance))

    return detections[mask]
