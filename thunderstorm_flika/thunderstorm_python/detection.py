"""
Molecule Detection Module
=========================

Implements methods for detecting approximate molecular positions:
- Local maximum detection
- Non-maximum suppression  
- Centroid of connected components
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from skimage.feature import peak_local_max


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
    """
    
    def __init__(self, connectivity='8-neighbourhood', min_distance=1):
        super().__init__("LocalMaximum")
        self.connectivity = connectivity
        self.min_distance = min_distance
        
    def detect(self, image, threshold):
        """Detect local maxima above threshold."""
        # Use peak_local_max for robust detection
        footprint_size = 3 if self.connectivity == '8-neighbourhood' else None
        
        coordinates = peak_local_max(
            image,
            min_distance=self.min_distance,
            threshold_abs=threshold,
            exclude_border=True
        )
        
        return coordinates


class NonMaximumSuppression(BaseDetector):
    """Non-maximum suppression detector.
    
    Finds local maxima using morphological operations.
    
    Parameters
    ----------
    connectivity : int
        Connectivity for morphological operations (1 or 2)
    """
    
    def __init__(self, connectivity=2):
        super().__init__("NonMaxSuppression")
        self.connectivity = connectivity
        
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
    """
    
    def __init__(self, connectivity=2, min_area=1):
        super().__init__("Centroid")
        self.connectivity = connectivity
        self.min_area = min_area
        
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
        
        return np.array(coordinates) if coordinates else np.array([]).reshape(0, 2)


class GridDetector(BaseDetector):
    """Detect molecules on a regular grid.
    
    Useful for simulations or structured samples.
    
    Parameters
    ----------
    spacing : int
        Grid spacing in pixels
    """
    
    def __init__(self, spacing=10):
        super().__init__("Grid")
        self.spacing = spacing
        
    def detect(self, image, threshold):
        """Generate grid of positions."""
        rows, cols = image.shape
        
        # Create grid
        row_coords = np.arange(self.spacing//2, rows, self.spacing)
        col_coords = np.arange(self.spacing//2, cols, self.spacing)
        
        grid_rows, grid_cols = np.meshgrid(row_coords, col_coords, indexing='ij')
        coordinates = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])
        
        # Filter by threshold
        intensities = image[coordinates[:, 0].astype(int), 
                           coordinates[:, 1].astype(int)]
        mask = intensities > threshold
        
        return coordinates[mask]


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
    refined = []
    
    for pos in detections:
        row, col = int(pos[0]), int(pos[1])
        
        # Extract local region
        r0 = max(0, row - radius)
        r1 = min(image.shape[0], row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(image.shape[1], col + radius + 1)
        
        region = image[r0:r1, c0:c1]
        
        # Compute weighted centroid
        total = region.sum()
        if total > 0:
            rows, cols = np.mgrid[r0:r1, c0:c1]
            row_center = (rows * region).sum() / total
            col_center = (cols * region).sum() / total
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
    
    Parameters
    ----------
    detections : ndarray
        Array of (row, col) positions
    image_shape : tuple
        Shape of image (rows, cols)
    border : int
        Border width in pixels
        
    Returns
    -------
    filtered : ndarray
        Detections away from borders
    """
    if len(detections) == 0:
        return detections
        
    rows, cols = image_shape
    
    mask = ((detections[:, 0] >= border) & 
            (detections[:, 0] < rows - border) &
            (detections[:, 1] >= border) & 
            (detections[:, 1] < cols - border))
            
    return detections[mask]
