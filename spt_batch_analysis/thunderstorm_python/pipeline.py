"""
ThunderSTORM Pipeline
=====================

Main analysis pipeline integrating all components.
Provides high-level interface similar to thunderSTORM's "Run Analysis".

Updated with comprehensive edge/border handling to prevent artifacts.
"""

import numpy as np
from . import filters, detection, fitting, postprocessing, visualization, utils


class ThunderSTORM:
    """Main ThunderSTORM analysis pipeline.

    Parameters
    ----------
    filter_type : str
        Image filter type ('gaussian', 'wavelet', 'dog', etc.)
    filter_params : dict
        Filter parameters
    detector_type : str
        Detector type ('local_maximum', 'non_maximum_suppression', 'centroid')
    detector_params : dict
        Detector parameters
    fitter_type : str
        PSF fitter type ('gaussian_lsq', 'gaussian_mle', 'radial_symmetry', 'centroid')
    fitter_params : dict
        Fitter parameters
    threshold_expression : str or float
        Detection threshold expression
    pixel_size : float
        Camera pixel size in nm
    photons_per_adu : float
        Photoelectrons per A/D count
    baseline : float
        Camera baseline offset
    em_gain : float
        EM gain (for EMCCD cameras)
    border_exclusion : int or None
        Border exclusion width (if None, computed automatically)
    """

    def __init__(self,
                 filter_type='wavelet',
                 filter_params=None,
                 detector_type='local_maximum',
                 detector_params=None,
                 fitter_type='gaussian_lsq',
                 fitter_params=None,
                 threshold_expression='std(Wave.F1)',
                 pixel_size=100.0,
                 photons_per_adu=1.0,
                 baseline=100.0,
                 em_gain=1.0,
                 border_exclusion=None):

        # Create components
        self.filter = filters.create_filter(
            filter_type,
            **(filter_params if filter_params else {})
        )

        self.detector = detection.create_detector(
            detector_type,
            **(detector_params if detector_params else {})
        )

        self.fitter = fitting.create_fitter(
            fitter_type,
            **(fitter_params if fitter_params else {})
        )

        self.threshold_expression = threshold_expression

        # Camera parameters
        self.pixel_size = pixel_size
        self.fitter.set_camera_params(
            pixel_size=pixel_size,
            photons_per_adu=photons_per_adu,
            baseline=baseline,
            em_gain=em_gain
        )

        # Border handling
        self.border_exclusion = border_exclusion

        # Post-processing components (optional)
        self.drift_corrector = None
        self.merger = None
        self.localization_filter = None
        self.density_filter = None

        # Results
        self.localizations = None
        self.images = None

    def _get_border_width(self, fit_radius):
        """Determine border exclusion width.

        Parameters
        ----------
        fit_radius : int
            Fitting radius in pixels

        Returns
        -------
        border_width : int
            Border exclusion width
        """
        if self.border_exclusion is not None:
            return self.border_exclusion
        else:
            return detection.safe_border_width(fit_radius)

    def _diagnose_dimensions(self, image, positions, result, stage=""):
        """Diagnostic function to help identify dimension issues.

        Parameters
        ----------
        image : ndarray
            Current image
        positions : ndarray, optional
            Detection positions in (row, col) format
        result : dict, optional
            Localization results with 'x', 'y' keys
        stage : str
            Description of current processing stage
        """
        print(f"\n=== Dimension Diagnostic: {stage} ===")
        print(f"Image shape: {image.shape} (rows/height, cols/width)")

        if positions is not None and len(positions) > 0:
            print(f"Positions shape: {positions.shape}")
            print(f"Position range: row=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], " +
                  f"col=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")

        if result is not None and len(result.get('x', [])) > 0:
            print(f"Result count: {len(result['x'])}")
            print(f"Result range (pixels): x=[{result['x'].min():.1f}, {result['x'].max():.1f}], " +
                  f"y=[{result['y'].min():.1f}, {result['y'].max():.1f}]")

            # Check if results are within bounds
            height, width = image.shape
            x_out = np.sum((result['x'] < 0) | (result['x'] >= width))
            y_out = np.sum((result['y'] < 0) | (result['y'] >= height))
            if x_out > 0 or y_out > 0:
                print(f"WARNING: Out of bounds detections: x={x_out}, y={y_out}")

        print("=" * 50)

    def _empty_result(self, frame_number):
        """Create empty result dictionary for frames with no detections.

        Parameters
        ----------
        frame_number : int
            Frame number

        Returns
        -------
        result : dict
            Empty result dictionary
        """
        return {
            'x': np.array([]),
            'y': np.array([]),
            'intensity': np.array([]),
            'background': np.array([]),
            'sigma_x': np.array([]),
            'sigma_y': np.array([]),
            'frame': np.array([]),
            'uncertainty': np.array([]),
            'chi_squared': np.array([])
        }

    def _validate_localizations(self, localizations, image_shape):
        """Remove localizations outside valid image area.

        This is a critical safety check to ensure no particles are
        reported outside the actual image bounds, which can occur
        due to edge effects or fitting errors.

        CRITICAL: Coordinate system conventions:
        - image_shape is (rows, cols) = (height, width)
        - localizations['x'] corresponds to column/width dimension
        - localizations['y'] corresponds to row/height dimension
        - Therefore: x should be checked against cols, y against rows

        Parameters
        ----------
        localizations : dict
            Localization results with 'x', 'y' in pixels
        image_shape : tuple
            (rows, cols) = (height, width) of image

        Returns
        -------
        filtered : dict
            Validated localizations
        """
        if len(localizations['x']) == 0:
            return localizations

        # Unpack dimensions explicitly for clarity
        # image_shape is (rows, cols) = (height, width)
        height, width = image_shape

        # Validation bounds:
        # - x (column coordinate) must be in [0, width)
        # - y (row coordinate) must be in [0, height)
        valid = ((localizations['x'] >= 0) &
                 (localizations['x'] < width) &
                 (localizations['y'] >= 0) &
                 (localizations['y'] < height))

        # Count rejected localizations for diagnostics
        n_rejected = np.sum(~valid)
        if n_rejected > 0:
            # Diagnostic info about rejected localizations
            x_out = np.sum((localizations['x'] < 0) | (localizations['x'] >= width))
            y_out = np.sum((localizations['y'] < 0) | (localizations['y'] >= height))
            if n_rejected > 10:  # Only warn if significant number
                print(f"Warning: Rejected {n_rejected} out-of-bounds localizations " +
                      f"({x_out} in x, {y_out} in y)")

        # Apply mask to all arrays
        filtered = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray) and len(value) == len(valid):
                filtered[key] = value[valid]
            else:
                filtered[key] = value

        return filtered

    def analyze_frame(self, image, frame_number=0, fit_radius=3, debug=False):
        """Analyze single frame with comprehensive edge handling.

        Parameters
        ----------
        image : ndarray
            2D image to analyze
        frame_number : int
            Frame number
        fit_radius : int
            Fitting radius in pixels
        debug : bool
            Enable diagnostic output for debugging dimension issues

        Returns
        -------
        result : dict
            Localization results for this frame
        """
        if debug:
            print(f"\n{'='*60}")
            print(f"Analyzing frame {frame_number}")
            print(f"{'='*60}")
            print(f"Image shape: {image.shape} (height, width) = ({image.shape[0]}, {image.shape[1]})")
            print(f"Fit radius: {fit_radius}")
            border_width = self._get_border_width(fit_radius)
            print(f"Border exclusion: {border_width} pixels")

        # Step 1: Filter image
        filtered = self.filter.apply(image)

        # Step 2: Compute threshold
        threshold = filters.compute_threshold_expression(
            image, filtered, self.threshold_expression
        )

        # Step 3: Detect molecules
        positions = self.detector.detect(filtered, threshold)

        if debug and len(positions) > 0:
            print(f"\nAfter detection: {len(positions)} positions")
            print(f"Position range: row=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], " +
                  f"col=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")

        # Step 4: Remove border detections BEFORE fitting
        # This prevents trying to fit PSFs that extend beyond image bounds
        if len(positions) > 0:
            border_width = self._get_border_width(fit_radius)
            positions = detection.remove_border_detections(
                positions,
                image.shape,
                border=border_width
            )

            if debug:
                print(f"\nAfter border removal: {len(positions)} positions")
                if len(positions) > 0:
                    print(f"Position range: row=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], " +
                          f"col=[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")

        # Check if any detections remain
        if len(positions) == 0:
            return self._empty_result(frame_number)

        # Step 5: Fit PSF
        try:
            fit_result = self.fitter.fit(image, positions, fit_radius=fit_radius)
        except Exception as e:
            print(f"Warning: Fitting failed for frame {frame_number}: {e}")
            return self._empty_result(frame_number)

        # Step 6: Convert to dict format
        result = fit_result.to_array()

        if debug and len(result['x']) > 0:
            print(f"\nAfter fitting: {len(result['x'])} localizations")
            print(f"Localization range (pixels): x=[{result['x'].min():.1f}, {result['x'].max():.1f}], " +
                  f"y=[{result['y'].min():.1f}, {result['y'].max():.1f}]")

        # Step 7: Validate localizations are within image bounds (in pixels)
        # This catches any edge cases from the fitting process
        result = self._validate_localizations(result, image.shape)

        # Check if any localizations remain after validation
        if len(result['x']) == 0:
            return self._empty_result(frame_number)

        if debug and len(result['x']) > 0:
            print(f"\nAfter validation: {len(result['x'])} localizations")
            print(f"Localization range (pixels): x=[{result['x'].min():.1f}, {result['x'].max():.1f}], " +
                  f"y=[{result['y'].min():.1f}, {result['y'].max():.1f}]")

        # Step 8: Add frame number
        result['frame'] = np.full(len(result['x']), frame_number)

        # Step 9: Convert from pixels to nanometers
        result['x'] = result['x'] * self.pixel_size
        result['y'] = result['y'] * self.pixel_size

        # Also convert sigma and uncertainty to nm
        if 'sigma_x' in result and result['sigma_x'] is not None:
            result['sigma_x'] = result['sigma_x'] * self.pixel_size
        if 'sigma_y' in result and result['sigma_y'] is not None:
            result['sigma_y'] = result['sigma_y'] * self.pixel_size
        if 'uncertainty' in result and result['uncertainty'] is not None:
            result['uncertainty'] = result['uncertainty'] * self.pixel_size

        if debug:
            print(f"\nFinal result in nm: x=[{result['x'].min():.1f}, {result['x'].max():.1f}], " +
                  f"y=[{result['y'].min():.1f}, {result['y'].max():.1f}]")
            print(f"{'='*60}\n")

        return result

    def analyze_stack(self, images, fit_radius=3, show_progress=True):
        """Analyze image stack.

        Parameters
        ----------
        images : ndarray
            3D image stack (n_frames, height, width)
        fit_radius : int
            Fitting radius in pixels
        show_progress : bool
            Show progress bar

        Returns
        -------
        localizations : dict
            All localizations from all frames
        """
        # Ensure 3D
        if images.ndim == 2:
            images = images[np.newaxis, ...]

        self.images = images
        n_frames = images.shape[0]

        # Analyze each frame
        all_results = []

        if show_progress:
            try:
                from tqdm import tqdm
                frame_iter = tqdm(range(n_frames), desc='Analyzing frames')
            except ImportError:
                frame_iter = range(n_frames)
                print(f"Analyzing {n_frames} frames...")
        else:
            frame_iter = range(n_frames)

        for i in frame_iter:
            result = self.analyze_frame(images[i], frame_number=i, fit_radius=fit_radius)
            all_results.append(result)

        # Combine results
        self.localizations = self._combine_results(all_results)

        return self.localizations

    def _combine_results(self, results):
        """Combine results from multiple frames."""
        combined = {}

        # Get all keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())

        # Concatenate arrays
        for key in all_keys:
            arrays = [r[key] for r in results if key in r]
            if arrays and isinstance(arrays[0], np.ndarray):
                combined[key] = np.concatenate(arrays)
            else:
                combined[key] = None

        return combined

    def apply_drift_correction(self, method='cross_correlation', **kwargs):
        """Apply drift correction to localizations.

        Parameters
        ----------
        method : str
            'cross_correlation' or 'fiducial'
        **kwargs : dict
            Method-specific parameters

        Returns
        -------
        corrected : dict
            Drift-corrected localizations
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        self.drift_corrector = postprocessing.DriftCorrector(method=method, **kwargs)

        # Compute drift
        frames = np.arange(self.images.shape[0])

        if method == 'cross_correlation':
            self.drift_corrector.compute_drift_xcorr(
                self.localizations, frames, **kwargs
            )
        elif method == 'fiducial':
            self.drift_corrector.compute_drift_fiducial(
                self.localizations, **kwargs
            )
        else:
            raise ValueError(f"Unknown drift correction method: {method}")

        # Apply correction
        self.localizations = self.drift_corrector.apply_drift_correction(
            self.localizations
        )

        return self.localizations

    def merge_molecules(self, max_distance=50, max_frame_gap=1):
        """Merge reappearing molecules.

        Parameters
        ----------
        max_distance : float
            Maximum distance in nm
        max_frame_gap : int
            Maximum frame gap

        Returns
        -------
        merged : dict
            Merged localizations
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        self.merger = postprocessing.MolecularMerger(
            max_distance=max_distance,
            max_frame_gap=max_frame_gap
        )

        self.localizations = self.merger.merge(self.localizations)

        return self.localizations

    def filter_localizations(self, **filter_params):
        """Filter localizations by quality.

        Parameters
        ----------
        **filter_params : dict
            Filter parameters (min_intensity, max_intensity, etc.)

        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        self.localization_filter = postprocessing.LocalizationFilter(**filter_params)

        self.localizations = self.localization_filter.filter(self.localizations)

        return self.localizations

    def filter_by_density(self, radius=50, min_neighbors=3):
        """Filter by local density.

        Parameters
        ----------
        radius : float
            Search radius in nm
        min_neighbors : int
            Minimum number of neighbors

        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        self.density_filter = postprocessing.LocalDensityFilter(
            radius=radius,
            min_neighbors=min_neighbors
        )

        self.localizations = self.density_filter.filter(self.localizations)

        return self.localizations

    def render(self, renderer_type='gaussian', pixel_size=10, **renderer_params):
        """Render super-resolution image.

        Parameters
        ----------
        renderer_type : str
            Renderer type ('gaussian', 'histogram', 'ash', 'scatter')
        pixel_size : float
            Rendering pixel size in nm
        **renderer_params : dict
            Renderer-specific parameters

        Returns
        -------
        image : ndarray
            Rendered super-resolution image
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        renderer = visualization.create_renderer(renderer_type, **renderer_params)

        image = renderer.render(self.localizations, pixel_size=pixel_size)

        return image

    def get_statistics(self):
        """Get summary statistics of localizations.

        Returns
        -------
        stats : dict
            Summary statistics
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        return utils.compute_statistics(self.localizations)

    def save(self, filepath, format='csv'):
        """Save localizations to file.

        Parameters
        ----------
        filepath : str
            Output filepath
        format : str
            File format ('csv')
        """
        if self.localizations is None:
            raise ValueError("Must analyze images first")

        if format == 'csv':
            utils.save_localizations_csv(self.localizations, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(self, filepath, format='csv'):
        """Load localizations from file.

        Parameters
        ----------
        filepath : str
            Input filepath
        format : str
            File format ('csv')
        """
        if format == 'csv':
            self.localizations = utils.load_localizations_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return self.localizations


def create_default_pipeline(border_exclusion=None):
    """Create ThunderSTORM pipeline with default settings.

    These are the default settings that work well on many datasets.

    Parameters
    ----------
    border_exclusion : int, optional
        Border exclusion width in pixels (auto-computed if None)

    Returns
    -------
    pipeline : ThunderSTORM
        Configured pipeline
    """
    return ThunderSTORM(
        filter_type='wavelet',
        filter_params={'scale': 2, 'order': 3},
        detector_type='local_maximum',
        detector_params={'connectivity': '8-neighbourhood', 'min_distance': 1, 'exclude_border': True},
        fitter_type='gaussian_lsq',
        fitter_params={'integrated': True, 'elliptical': False, 'initial_sigma': 1.3},
        threshold_expression='std(Wave.F1)',
        pixel_size=100.0,  # nm
        photons_per_adu=1.0,
        baseline=100.0,
        em_gain=1.0,
        border_exclusion=border_exclusion
    )


def quick_analysis(images, border_exclusion=None, **kwargs):
    """Quick analysis with default parameters.

    Parameters
    ----------
    images : ndarray
        Image stack to analyze
    border_exclusion : int, optional
        Border exclusion width
    **kwargs : dict
        Additional parameters for pipeline

    Returns
    -------
    localizations : dict
        Detected localizations
    rendered : ndarray
        Rendered super-resolution image
    pipeline : ThunderSTORM
        Pipeline object
    """
    # Create pipeline
    pipeline = create_default_pipeline(border_exclusion=border_exclusion)

    # Update any custom parameters
    for key, value in kwargs.items():
        if hasattr(pipeline, key):
            setattr(pipeline, key, value)

    # Analyze
    localizations = pipeline.analyze_stack(images)

    # Render
    rendered = pipeline.render(renderer_type='gaussian', pixel_size=10)

    return localizations, rendered, pipeline
