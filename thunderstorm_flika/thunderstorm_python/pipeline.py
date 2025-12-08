"""
ThunderSTORM Pipeline
=====================

Main analysis pipeline integrating all components.
Provides high-level interface similar to thunderSTORM's "Run Analysis".
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
                 em_gain=1.0):
        
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
        
        # Post-processing components (optional)
        self.drift_corrector = None
        self.merger = None
        self.localization_filter = None
        self.density_filter = None
        
        # Results
        self.localizations = None
        self.images = None
        
    def analyze_frame(self, image, frame_number=0, fit_radius=3):
        """Analyze single frame.
        
        Parameters
        ----------
        image : ndarray
            2D image to analyze
        frame_number : int
            Frame number
        fit_radius : int
            Fitting radius in pixels
            
        Returns
        -------
        result : dict
            Localization results for this frame
        """
        # Step 1: Filter image
        filtered = self.filter.apply(image)
        
        # Step 2: Compute threshold
        threshold = filters.compute_threshold_expression(
            image, filtered, self.threshold_expression
        )
        
        # Step 3: Detect molecules
        positions = self.detector.detect(filtered, threshold)
        
        # Step 4: Fit PSF
        if len(positions) > 0:
            fit_result = self.fitter.fit(image, positions, fit_radius=fit_radius)
            
            # Convert to dict format
            result = fit_result.to_array()
            result['frame'] = np.full(len(fit_result), frame_number)
        else:
            # No detections
            result = {
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


def create_default_pipeline():
    """Create ThunderSTORM pipeline with default settings.
    
    These are the default settings that work well on many datasets.
    
    Returns
    -------
    pipeline : ThunderSTORM
        Configured pipeline
    """
    return ThunderSTORM(
        filter_type='wavelet',
        filter_params={'scale': 2, 'order': 3},
        detector_type='local_maximum',
        detector_params={'connectivity': '8-neighbourhood', 'min_distance': 1},
        fitter_type='gaussian_lsq',
        fitter_params={'integrated': True, 'elliptical': False, 'initial_sigma': 1.3},
        threshold_expression='std(Wave.F1)',
        pixel_size=100.0,  # nm
        photons_per_adu=1.0,
        baseline=100.0,
        em_gain=1.0
    )


def quick_analysis(images, **kwargs):
    """Quick analysis with default parameters.
    
    Parameters
    ----------
    images : ndarray
        Image stack to analyze
    **kwargs : dict
        Additional parameters for pipeline
        
    Returns
    -------
    localizations : dict
        Detected localizations
    rendered : ndarray
        Rendered super-resolution image
    """
    # Create pipeline
    pipeline = create_default_pipeline()
    
    # Update any custom parameters
    for key, value in kwargs.items():
        if hasattr(pipeline, key):
            setattr(pipeline, key, value)
    
    # Analyze
    localizations = pipeline.analyze_stack(images)
    
    # Render
    rendered = pipeline.render(renderer_type='gaussian', pixel_size=10)
    
    return localizations, rendered, pipeline
