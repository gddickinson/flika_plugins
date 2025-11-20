#!/usr/bin/env python3
"""
General gap closing parameters and tracking script - ENHANCED WITH CENTRALIZED LOGGING

Python port of u-track's scriptTrackGeneral.m

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.

u-track is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

u-track is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with u-track.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import os
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import time

# Import the centralized logging system
from .module_logger import (
    get_module_logger, PerformanceTimer, log_function_call,
    log_array_info, LoggingMixin
)

# Get logger for this module
logger = get_module_logger('track_general')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_len(obj):
    """Safely get length of object, handling numpy arrays and None values"""
    if obj is None:
        return 0
    try:
        return len(obj)
    except TypeError:
        return 1 if obj else 0

def safe_bool_check(obj):
    """Safely check if object is truthy, handling numpy arrays"""
    if obj is None:
        return False
    if hasattr(obj, '__len__'):
        try:
            return len(obj) > 0
        except (TypeError, ValueError):
            return obj is not None
    return bool(obj)

# =============================================================================
# ENHANCED PARAMETER CLASSES WITH LOGGING
# =============================================================================

@dataclass
class GapCloseParam(LoggingMixin):
    """Parameters for gap closing - ENHANCED WITH CENTRALIZED LOGGING"""
    time_window: int = 5  # maximum allowed time gap (in frames) between track segments
    merge_split: int = 1  # 1: merging and splitting, 2: only merging, 3: only splitting, 0: neither
    min_track_len: int = 2  # minimum length of track segments from linking for gap closing
    diagnostics: int = 0  # 1 to plot histogram of gap lengths, 0 otherwise
    tolerance: float = 0.05  # relative change tolerance for gap closing iteration

    def __post_init__(self):
        """Initialize parameters with logging"""
        # Initialize logging for this instance
        self._init_logging()

        self.log_info("=== GapCloseParam INITIALIZATION ===")

        params = {
            'time_window': self.time_window,
            'merge_split': self.merge_split,
            'min_track_len': self.min_track_len,
            'diagnostics': self.diagnostics,
            'tolerance': self.tolerance
        }

        self.log_parameters(params, "GapCloseParam values")

        # Validate parameters
        warnings = self._validate_parameters()

        if warnings:
            for warning in warnings:
                self.log_warning(f"Parameter validation: {warning}")
        else:
            self.log_info("All parameters are within expected ranges")

        self.log_info("GapCloseParam initialized successfully")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.log_info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.log_info(f"  {key}: {value}")

    def _validate_parameters(self) -> List[str]:
        """Validate parameter ranges"""
        warnings = []

        if not (1 <= self.time_window <= 50):
            warnings.append(f"time_window ({self.time_window}) outside recommended range [1, 50]")

        if not (0 <= self.merge_split <= 3):
            warnings.append(f"merge_split ({self.merge_split}) outside valid range [0, 3]")

        if not (1 <= self.min_track_len <= 100):
            warnings.append(f"min_track_len ({self.min_track_len}) outside recommended range [1, 100]")

        if not (0 <= self.diagnostics <= 1):
            warnings.append(f"diagnostics ({self.diagnostics}) should be 0 or 1")

        if not (0.001 <= self.tolerance <= 0.5):
            warnings.append(f"tolerance ({self.tolerance}) outside recommended range [0.001, 0.5]")

        return warnings

    def export_dict(self):
        """Export parameters as dict with logging"""
        params_dict = {
            'time_window': self.time_window,
            'merge_split': self.merge_split,
            'min_track_len': self.min_track_len,
            'diagnostics': self.diagnostics,
            'tolerance': self.tolerance
        }

        self.log_debug("Exporting GapCloseParam as dictionary")
        self.log_parameters(params_dict, "exported parameters")

        return params_dict


@dataclass
class CostMatrixParameters(LoggingMixin):
    """Unified parameters for cost matrix calculations - ENHANCED WITH CENTRALIZED LOGGING"""
    # Motion model
    linear_motion: int = 0  # 0: Brownian only, 1: Brownian+directed, 2: with switching

    # Search radius parameters
    min_search_radius: float = 6.0  # minimum allowed search radius
    max_search_radius: float = 6.0  # maximum allowed search radius

    # Brownian motion parameters
    brown_std_mult: Union[float, np.ndarray] = 3.0  # multiplication factor for search radius
    brown_scaling: List[float] = field(default_factory=lambda: [0.25, 0.01])  # time scaling powers
    time_reach_conf_b: int = 5  # time to reach confinement for Brownian motion

    # Linear motion parameters
    lin_std_mult: Union[float, np.ndarray] = field(default_factory=lambda: np.full(5, 3.0))
    lin_scaling: List[float] = field(default_factory=lambda: [1.0, 0.01])  # time scaling powers
    time_reach_conf_l: int = 5  # time to reach confinement for linear motion

    # Directional constraints
    max_angle_vv: float = 30.0  # maximum angle between track directions (degrees)

    # Density and local features
    use_local_density: int = 1  # expand search radius of isolated features
    nn_window: int = 5  # frames to look for nearest neighbor

    # Amplitude constraints (for merge/split)
    amp_ratio_limit: Optional[List[float]] = None  # intensity ratios [min, max]

    # Gap closing specific
    gap_penalty: float = 1.5  # penalty for temporary disappearance
    len_for_classify: int = 5  # minimum track length for motion classification
    res_limit: Optional[float] = None  # resolution limit
    gap_exclude_ms: int = 1  # flag to allow gaps to exclude merges and splits
    strategy_bd: int = -1  # strategy to calculate birth and death cost

    # Lifetime considerations
    lft_cdf: Optional[List[float]] = None  # lifetime cumulative density function

    # Kalman filter initialization
    kalman_init_param: Optional[Dict] = None  # Kalman filter initialization parameters

    # Diagnostics
    diagnostics: Optional[List[int]] = None  # frames for linking distance histograms

    def __post_init__(self):
        """Initialize parameters with comprehensive logging and validation"""
        # Initialize logging for this instance
        self._init_logging()

        self.log_info("=== CostMatrixParameters INITIALIZATION ===")

        # Convert and validate parameters
        self._convert_parameters()
        self._set_defaults()

        # Log all parameters
        params = self._collect_all_parameters()
        self.log_parameters(params, "CostMatrixParameters values")

        # Validate parameters
        warnings = self._validate_parameter_relationships()

        if warnings:
            for warning in warnings:
                self.log_warning(f"Parameter validation: {warning}")
        else:
            self.log_info("All parameter relationships are valid")

        self.log_info("CostMatrixParameters initialized successfully")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.log_info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.log_info(f"  {key}: {value}")

    def _convert_parameters(self):
        """Convert parameters to appropriate formats with logging"""
        self.log_debug("Converting parameters to appropriate formats")

        # Convert lin_std_mult to numpy array if it isn't already
        if not isinstance(self.lin_std_mult, np.ndarray):
            original_val = self.lin_std_mult
            self.lin_std_mult = np.full(5, float(self.lin_std_mult))
            self.log_debug(f"Converted lin_std_mult: {original_val} → array of shape {self.lin_std_mult.shape}")

        # Convert brown_std_mult to appropriate format
        if isinstance(self.brown_std_mult, (list, tuple)):
            original_val = self.brown_std_mult
            self.brown_std_mult = np.array(self.brown_std_mult)
            self.log_debug(f"Converted brown_std_mult: {original_val} → array of shape {self.brown_std_mult.shape}")

        log_array_info(self.logger, "lin_std_mult", self.lin_std_mult, "after conversion")
        log_array_info(self.logger, "brown_std_mult", self.brown_std_mult, "after conversion")

    def _set_defaults(self):
        """Set default values for optional parameters"""
        self.log_debug("Setting default values for optional parameters")

        if self.amp_ratio_limit is None:
            self.amp_ratio_limit = [0.7, 4.0]
            self.log_debug(f"Set default amp_ratio_limit: {self.amp_ratio_limit}")

        if self.res_limit is None:
            self.res_limit = 0.5
            self.log_debug(f"Set default res_limit: {self.res_limit}")

    def _collect_all_parameters(self):
        """Collect all parameters into a dictionary for logging"""
        return {
            'linear_motion': self.linear_motion,
            'min_search_radius': self.min_search_radius,
            'max_search_radius': self.max_search_radius,
            'brown_std_mult': self.brown_std_mult,
            'brown_scaling': self.brown_scaling,
            'time_reach_conf_b': self.time_reach_conf_b,
            'lin_std_mult': self.lin_std_mult,
            'lin_scaling': self.lin_scaling,
            'time_reach_conf_l': self.time_reach_conf_l,
            'max_angle_vv': self.max_angle_vv,
            'use_local_density': self.use_local_density,
            'nn_window': self.nn_window,
            'amp_ratio_limit': self.amp_ratio_limit,
            'gap_penalty': self.gap_penalty,
            'len_for_classify': self.len_for_classify,
            'res_limit': self.res_limit,
            'gap_exclude_ms': self.gap_exclude_ms,
            'strategy_bd': self.strategy_bd,
            'lft_cdf': self.lft_cdf,
            'kalman_init_param': self.kalman_init_param,
            'diagnostics': self.diagnostics
        }

    def _validate_parameter_relationships(self):
        """Validate logical relationships between parameters"""
        self.log_debug("Validating parameter relationships")
        warnings = []

        # Check search radius relationship
        if self.min_search_radius > self.max_search_radius:
            warning = f"min_search_radius ({self.min_search_radius}) > max_search_radius ({self.max_search_radius})"
            warnings.append(warning)
        else:
            self.log_debug(f"Search radius relationship valid: {self.min_search_radius} ≤ {self.max_search_radius}")

        # Check amplitude ratio limits
        if self.amp_ratio_limit and len(self.amp_ratio_limit) >= 2:
            if self.amp_ratio_limit[0] > self.amp_ratio_limit[1]:
                warning = f"amp_ratio_limit min ({self.amp_ratio_limit[0]}) > max ({self.amp_ratio_limit[1]})"
                warnings.append(warning)
            else:
                self.log_debug(f"Amplitude ratio limits valid: {self.amp_ratio_limit}")

        # Check array consistency
        if isinstance(self.brown_std_mult, np.ndarray):
            if self.brown_std_mult.size > 1:
                self.log_debug(f"brown_std_mult is time-varying array with {self.brown_std_mult.size} elements")
            else:
                self.log_debug(f"brown_std_mult is scalar: {self.brown_std_mult}")

        # Validate ranges
        range_checks = [
            ('linear_motion', self.linear_motion, 0, 2),
            ('min_search_radius', self.min_search_radius, 0.1, 100.0),
            ('max_search_radius', self.max_search_radius, 0.1, 100.0),
            ('max_angle_vv', self.max_angle_vv, 0.0, 180.0),
            ('use_local_density', self.use_local_density, 0, 1),
            ('nn_window', self.nn_window, 1, 20),
            ('gap_penalty', self.gap_penalty, 1.0, 10.0),
            ('len_for_classify', self.len_for_classify, 2, 50),
            ('time_reach_conf_b', self.time_reach_conf_b, 1, 20),
            ('time_reach_conf_l', self.time_reach_conf_l, 1, 20)
        ]

        for param_name, value, min_val, max_val in range_checks:
            if not (min_val <= value <= max_val):
                warning = f"{param_name} ({value}) outside recommended range [{min_val}, {max_val}]"
                warnings.append(warning)

        return warnings

    def export_dict(self):
        """Export parameters as dict with logging"""
        params_dict = self._collect_all_parameters()
        self.log_debug("Exporting CostMatrixParameters as dictionary")
        self.log_parameters(params_dict, "exported parameters")
        return params_dict


@dataclass
class LinkingParameters(LoggingMixin):
    """Parameters for frame-to-frame linking cost matrix (legacy - use CostMatrixParameters)"""
    linear_motion: int = 0
    min_search_radius: float = 6.0
    max_search_radius: float = 6.0
    brown_std_mult: float = 3.0
    use_local_density: int = 1
    nn_window: int = 5
    kalman_init_param: Optional[Dict] = None
    diagnostics: Optional[List[int]] = None

    def __post_init__(self):
        # Initialize logging for this instance
        self._init_logging()
        self.log_warning("Using legacy LinkingParameters - consider using CostMatrixParameters")
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('logger')}
        self.log_parameters(params, "LinkingParameters (legacy)")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.logger.info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")


@dataclass
class GapClosingParameters(LoggingMixin):
    """Parameters for gap closing cost matrix (legacy - use CostMatrixParameters)"""
    linear_motion: int = 0
    min_search_radius: float = 6.0
    max_search_radius: float = 6.0
    brown_std_mult: np.ndarray = field(default_factory=lambda: np.full(5, 3.0))
    brown_scaling: List[float] = field(default_factory=lambda: [0.25, 0.01])
    time_reach_conf_b: int = 5
    amp_ratio_limit: Optional[List[float]] = None
    len_for_classify: int = 5
    use_local_density: int = 0
    nn_window: int = 5
    lin_std_mult: np.ndarray = field(default_factory=lambda: np.full(5, 3.0))
    lin_scaling: List[float] = field(default_factory=lambda: [1.0, 0.01])
    time_reach_conf_l: int = 5
    max_angle_vv: float = 30.0
    gap_penalty: float = 1.5
    res_limit: Optional[float] = None
    gap_exclude_ms: int = 1
    strategy_bd: int = -1

    def __post_init__(self):
        # Initialize logging for this instance
        self._init_logging()
        self.log_warning("Using legacy GapClosingParameters - consider using CostMatrixParameters")
        params = {k: v for k, v in self.__dict__.items() if not k.startswith('logger')}
        self.log_parameters(params, "GapClosingParameters (legacy)")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.logger.info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")


# Create alias for consistency with other modules
GapCloseParameters = GapCloseParam


@dataclass
class CostMatrix(LoggingMixin):
    """Cost matrix configuration"""
    func_name: str
    parameters: Union[LinkingParameters, GapClosingParameters]

    def __post_init__(self):
        # Initialize logging for this instance
        self._init_logging()
        self.log_info("=== CostMatrix INITIALIZATION ===")
        self.log_info(f"Function name: {self.func_name}")
        self.log_info(f"Parameters type: {type(self.parameters).__name__}")

        if hasattr(self.parameters, 'export_dict'):
            self.parameters.export_dict()

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)


@dataclass
class KalmanFunctions(LoggingMixin):
    """Kalman filter function names"""
    reserve_mem: str = 'kalman_res_mem_lm'
    initialize: str = 'kalman_init_linear_motion'
    calc_gain: str = 'kalman_gain_linear_motion'
    time_reverse: str = 'kalman_reverse_linear_motion'

    def __post_init__(self):
        # Initialize logging for this instance
        self._init_logging()
        self.log_info("=== KalmanFunctions INITIALIZATION ===")
        params = {
            'reserve_mem': self.reserve_mem,
            'initialize': self.initialize,
            'calc_gain': self.calc_gain,
            'time_reverse': self.time_reverse
        }
        self.log_parameters(params, "KalmanFunctions")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.log_info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.log_info(f"  {key}: {value}")


@dataclass
class SaveResults(LoggingMixin):
    """Configuration for saving results"""
    directory: str
    filename: str

    def __post_init__(self):
        # Initialize logging for this instance
        self._init_logging()
        self.log_info("=== SaveResults INITIALIZATION ===")
        params = {'directory': self.directory, 'filename': self.filename}
        self.log_parameters(params, "SaveResults")

        # Validate directory
        if not os.path.exists(self.directory):
            self.log_warning(f"Save directory does not exist: {self.directory}")
        else:
            self.log_info(f"Save directory exists: {self.directory}")

    def _init_logging(self):
        """Initialize logging for dataclass instances"""
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            module_name = self.__class__.__name__.lower()

        self.logger = get_module_logger(module_name)

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.log_info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.log_info(f"  {key}: {value}")


# =============================================================================
# ENHANCED PARTICLE TRACKER WITH COMPREHENSIVE LOGGING
# =============================================================================

class ParticleTracker(LoggingMixin):
    """Fixed particle tracking class that ensures individual modules are used - ENHANCED WITH CENTRALIZED LOGGING"""

    def __init__(self):
        super().__init__()

        self.log_info("=== ParticleTracker INITIALIZATION ===")

        self.gap_close_param = None
        self.cost_matrices = []
        self.kalman_functions = None
        self.save_results = None
        self.verbose = True
        self.prob_dim = 2

        # Add tracking for module usage
        self.modules_used = {
            'linking': False,
            'gap_closing': False,
            'cost_matrices': False,
            'kalman_filters': False
        }

        initial_state = {
            'gap_close_param': self.gap_close_param,
            'cost_matrices_count': len(self.cost_matrices),
            'kalman_functions': self.kalman_functions,
            'save_results': self.save_results,
            'verbose': self.verbose,
            'prob_dim': self.prob_dim,
            'modules_used': self.modules_used
        }

        self.log_parameters(initial_state, "ParticleTracker initial state")
        self.log_info("ParticleTracker initialized successfully")

    def run_tracking(self, movie_info: List[Dict],
                    save_dir: str = None,
                    filename: str = None,
                    cost_matrices: Optional[List[Dict]] = None,
                    gap_close_param: Optional[GapCloseParam] = None,
                    kalman_functions: Optional[KalmanFunctions] = None) -> tuple:
        """
        FIXED: Run the complete tracking pipeline using individual modules - ENHANCED WITH CENTRALIZED LOGGING
        """
        log_function_call(self.logger, 'run_tracking',
                         (movie_info, save_dir, filename),
                         {'cost_matrices': cost_matrices is not None,
                          'gap_close_param': gap_close_param is not None,
                          'kalman_functions': kalman_functions is not None})

        self.log_info("=== STARTING FIXED PARTICLE TRACKING PIPELINE ===")

        # Log input parameters
        input_params = {
            'movie_info_frames': len(movie_info) if movie_info is not None else 0,
            'save_dir': save_dir,
            'filename': filename,
            'cost_matrices_provided': cost_matrices is not None,
            'gap_close_param_provided': gap_close_param is not None,
            'kalman_functions_provided': kalman_functions is not None
        }
        self.log_parameters(input_params, "run_tracking inputs")

        with self.time_operation("Complete tracking pipeline"):

            # Set parameters if provided, otherwise use defaults
            if cost_matrices is not None:
                self.log_info(f"Received {len(cost_matrices)} cost matrices")
                self.cost_matrices = cost_matrices
                for i, cm in enumerate(cost_matrices):
                    self.log_debug(f"Cost matrix {i}: {cm}")

            if gap_close_param is not None:
                self.log_info("Received gap close parameters")
                self.gap_close_param = gap_close_param
                if hasattr(gap_close_param, 'export_dict'):
                    gap_close_param.export_dict()

            if kalman_functions is not None:
                self.log_info("Received Kalman functions")
                self.kalman_functions = kalman_functions

            # Ensure we have parameters
            if not hasattr(self, 'gap_close_param') or self.gap_close_param is None:
                self.log_info("Setting up default parameters")
                self.setup_parameters(save_dir or "./results", filename or "tracks_test.pkl")

            # Import and verify individual modules
            try:
                self.log_info("Importing and verifying individual modules")

                with self.time_operation("Module imports"):
                    from cost_matrices import CostMatrixRandomDirectedSwitchingMotion
                    cost_matrix_calc = CostMatrixRandomDirectedSwitchingMotion()
                    self.modules_used['cost_matrices'] = True
                    self.log_info("✓ cost_matrices.py imported successfully")

                    from kalman_filters import KalmanFilterLinearMotion
                    kalman_filter = KalmanFilterLinearMotion()
                    self.modules_used['kalman_filters'] = True
                    self.log_info("✓ kalman_filters.py imported successfully")

                    from gap_closing import GapCloser
                    gap_closer = GapCloser()
                    self.modules_used['gap_closing'] = True
                    self.log_info("✓ gap_closing.py imported successfully")

                    from linking import FeatureLinker
                    feature_linker = FeatureLinker()
                    self.modules_used['linking'] = True
                    self.log_info("✓ linking.py imported successfully")

                    # Also import analysis module
                    from track_analysis import analyze_tracking_results, MotionAnalyzer
                    self.log_info("✓ track_analysis.py imported successfully")

                modules_status = {
                    'cost_matrix_calc': type(cost_matrix_calc).__name__,
                    'kalman_filter': type(kalman_filter).__name__,
                    'gap_closer': type(gap_closer).__name__,
                    'feature_linker': type(feature_linker).__name__
                }
                self.log_parameters(modules_status, "imported modules")

            except ImportError as e:
                self.log_error(f"Failed to import individual modules: {str(e)}")
                return [], {}, 1

            # Run the fixed tracking pipeline
            self.log_info("Calling main tracking pipeline")
            tracks_final, kalman_info_link, err_flag = self.track_close_gaps_kalman_sparse_fixed(
                movie_info, cost_matrix_calc, kalman_filter, gap_closer, feature_linker
            )

            # Save results if directory is specified
            if save_dir and filename:
                self.log_info(f"Saving results to {save_dir}/{filename}")
                self._save_results_with_module_info(tracks_final, kalman_info_link, err_flag, save_dir, filename)

        completion_status = {
            'tracks_final_count': safe_len(tracks_final),
            'kalman_info_keys': list(kalman_info_link.keys()) if isinstance(kalman_info_link, dict) else 'Not a dict',
            'err_flag': err_flag,
            'modules_used': self.modules_used
        }
        self.log_parameters(completion_status, "tracking completion status")

        self.log_info("=== TRACKING PIPELINE COMPLETED ===")

        return tracks_final, kalman_info_link, err_flag

    def track_close_gaps_kalman_sparse_fixed(self, movie_info: List[Dict],
                                           cost_matrix_calc, kalman_filter,
                                           gap_closer, feature_linker) -> tuple:
        """
        FIXED: Main tracking function that explicitly uses individual modules - ENHANCED WITH CENTRALIZED LOGGING
        """
        log_function_call(self.logger, 'track_close_gaps_kalman_sparse_fixed',
                         (movie_info, cost_matrix_calc, kalman_filter, gap_closer, feature_linker))

        self.log_info("=== STARTING FIXED TRACKING PIPELINE ===")

        pipeline_inputs = {
            'movie_info_frames': safe_len(movie_info),
            'cost_matrix_calc_type': type(cost_matrix_calc).__name__,
            'kalman_filter_type': type(kalman_filter).__name__,
            'gap_closer_type': type(gap_closer).__name__,
            'feature_linker_type': type(feature_linker).__name__,
            'prob_dim': self.prob_dim,
            'verbose': self.verbose
        }
        self.log_parameters(pipeline_inputs, "pipeline inputs")

        with self.time_operation("Fixed tracking pipeline"):

            try:
                # Validate movie_info structure
                self.log_info("Validating movie_info structure")
                movie_info = self._validate_movie_info(movie_info)
                self.log_info(f"Processing {safe_len(movie_info)} frames")

                # STEP 1: Frame-to-frame linking using linking.py
                self.log_info("=== STEP 1: FRAME-TO-FRAME LINKING ===")

                with self.time_operation("Frame-to-frame linking"):

                    # Extract parameters for linking
                    if len(self.cost_matrices) > 0:
                        if hasattr(self.cost_matrices[0], 'parameters'):
                            link_params = self.cost_matrices[0].parameters
                        else:
                            link_params = self.cost_matrices[0].get('parameters', {})
                    else:
                        self.log_warning("No cost matrices defined, using defaults")
                        link_params = {}

                    # Log linking parameters
                    if hasattr(link_params, 'export_dict'):
                        link_params_dict = link_params.export_dict()
                    elif hasattr(link_params, '__dict__'):
                        link_params_dict = link_params.__dict__
                    else:
                        link_params_dict = link_params

                    self.log_parameters(link_params_dict, "linking parameters")

                    # Call the individual linking module
                    self.log_info("Calling feature_linker.link_features_kalman_sparse")

                    tracks_feat_indx_link, tracks_coord_amp_link, kalman_info_link, nn_dist_linked_feat, \
                    linking_costs, link_err_flag, trackability_data = \
                        feature_linker.link_features_kalman_sparse(
                            movie_info,
                            cost_matrix_calc.cost_mat_random_directed_switching_motion_link,
                            link_params,
                            self.kalman_functions,
                            self.prob_dim,
                            None,  # kalman_info_prev
                            None,  # linking_costs_prev
                            self.verbose
                        )

                    # Log linking results
                    linking_results = {
                        'tracks_feat_indx_link_count': safe_len(tracks_feat_indx_link),
                        'tracks_coord_amp_link_count': safe_len(tracks_coord_amp_link),
                        'kalman_info_link_keys': list(kalman_info_link.keys()) if isinstance(kalman_info_link, dict) else 'Not a dict',
                        'link_err_flag': link_err_flag
                    }
                    self.log_parameters(linking_results, "linking results")

                if link_err_flag != 0:
                    self.log_error("Error in frame-to-frame linking")
                    return [], {}, 1

                self.log_info(f"Linking completed: {safe_len(tracks_feat_indx_link)} track segments")

                # STEP 2: Gap closing using gap_closing.py
                self.log_info("=== STEP 2: GAP CLOSING ===")

                with self.time_operation("Gap closing"):

                    # Extract parameters for gap closing
                    if len(self.cost_matrices) > 1:
                        if hasattr(self.cost_matrices[1], 'parameters'):
                            gap_params = self.cost_matrices[1].parameters
                        else:
                            gap_params = self.cost_matrices[1].get('parameters', {})
                    else:
                        gap_params = link_params  # Use same parameters

                    # Log gap closing parameters
                    if hasattr(gap_params, 'export_dict'):
                        gap_params_dict = gap_params.export_dict()
                    elif hasattr(gap_params, '__dict__'):
                        gap_params_dict = gap_params.__dict__
                    else:
                        gap_params_dict = gap_params

                    self.log_parameters(gap_params_dict, "gap closing parameters")

                    # Create gap close parameters dict
                    if hasattr(self.gap_close_param, '__dict__'):
                        gap_close_dict = self.gap_close_param.__dict__
                    else:
                        gap_close_dict = self.gap_close_param

                    self.log_parameters(gap_close_dict, "gap close settings")

                    # Call the individual gap closing module
                    self.log_info("Calling gap_closer.close_gaps")

                    tracks_final = gap_closer.close_gaps(
                        tracks_feat_indx_link,
                        tracks_coord_amp_link,
                        kalman_info_link,
                        nn_dist_linked_feat,
                        gap_close_dict,
                        {'parameters': gap_params},
                        self.prob_dim,
                        movie_info,
                        self.verbose
                    )

                    # Log gap closing results
                    gap_results = {
                        'tracks_final_count': safe_len(tracks_final),
                        'tracks_final_type': type(tracks_final).__name__
                    }
                    self.log_parameters(gap_results, "gap closing results")

                self.log_info(f"Gap closing completed: {safe_len(tracks_final)} final tracks")

                # STEP 3: Verify individual module usage
                self.log_info("=== STEP 3: MODULE USAGE VERIFICATION ===")

                verification_results = {
                    'linking_module_used': self.modules_used['linking'],
                    'gap_closing_module_used': self.modules_used['gap_closing'],
                    'cost_matrices_module_used': self.modules_used['cost_matrices'],
                    'kalman_filters_module_used': self.modules_used['kalman_filters'],
                    'tracks_from_modules': safe_len(tracks_final) > 0,
                    'kalman_info_from_modules': bool(kalman_info_link)
                }

                self.log_parameters(verification_results, "module verification")

                for module, used in verification_results.items():
                    if used:
                        self.log_info(f"✓ {module}: {used}")
                    else:
                        self.log_warning(f"✗ {module}: {used}")

                # Store verification in kalman_info for later retrieval
                if not isinstance(kalman_info_link, dict):
                    kalman_info_link = {}
                kalman_info_link['module_verification'] = verification_results

                err_flag = 0
                self.log_info("FIXED tracking pipeline completed successfully!")

            except ImportError as e:
                self.log_error(f"Import error in individual modules: {str(e)}")
                return [], {}, 1
            except Exception as e:
                self.log_error(f"Error during FIXED tracking: {str(e)}")
                self.log_exception("Full traceback:")
                return [], {}, 1

        self.log_info("=== FIXED TRACKING PIPELINE COMPLETED ===")
        return tracks_final, kalman_info_link, err_flag

    def _save_results_with_module_info(self, tracks_final, kalman_info_link, err_flag, save_dir, filename):
        """FIXED: Save tracking results with module verification info - ENHANCED WITH CENTRALIZED LOGGING"""
        log_function_call(self.logger, '_save_results_with_module_info',
                         (tracks_final, kalman_info_link, err_flag, save_dir, filename))

        self.log_info("=== SAVING RESULTS WITH MODULE INFO ===")

        save_inputs = {
            'tracks_final_count': safe_len(tracks_final),
            'kalman_info_type': type(kalman_info_link).__name__,
            'err_flag': err_flag,
            'save_dir': save_dir,
            'filename': filename
        }
        self.log_parameters(save_inputs, "save inputs")

        with self.time_operation("Saving results"):

            import pickle

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Prepare comprehensive data for saving
            results = {
                # Core tracking results from individual modules
                'tracks_final': tracks_final,
                'kalman_info_link': kalman_info_link,
                'err_flag': err_flag,

                # Parameters used
                'gap_close_param': self.gap_close_param,
                'cost_matrices': self.cost_matrices,
                'kalman_functions': self.kalman_functions,

                # Module verification
                'modules_used': self.modules_used.copy(),
                'module_verification': kalman_info_link.get('module_verification', {}) if isinstance(kalman_info_link, dict) else {},

                # Tracking metadata
                'tracking_metadata': {
                    'prob_dim': self.prob_dim,
                    'verbose': self.verbose,
                    'num_final_tracks': safe_len(tracks_final),
                    'tracking_method': 'individual_modules_verified',
                    'save_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            # Log the data being saved
            save_summary = {
                'tracks_final': f"{safe_len(results['tracks_final'])} tracks",
                'kalman_info_link': f"{len(results['kalman_info_link'])} keys" if isinstance(results['kalman_info_link'], dict) else str(type(results['kalman_info_link'])),
                'err_flag': results['err_flag'],
                'modules_used': results['modules_used'],
                'tracking_metadata': results['tracking_metadata']
            }
            self.log_parameters(save_summary, "data being saved")

            # Ensure .pkl extension
            if not filename.endswith('.pkl'):
                filename = filename.replace('.mat', '.pkl')
                if not filename.endswith('.pkl'):
                    filename += '.pkl'

            filepath = os.path.join(save_dir, filename)

            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

                self.log_info(f"Results saved successfully to: {filepath}")
                self.log_info(f"Saved {safe_len(tracks_final)} tracks with module verification")

                # Log module usage summary
                modules_summary = ", ".join([f"{k}={v}" for k, v in self.modules_used.items()])
                self.log_info(f"Module usage: {modules_summary}")

            except Exception as e:
                self.log_error(f"Error saving results: {str(e)}")

    def setup_parameters(self, save_dir: str = "./results", filename: str = "tracks_test.pkl"):
        """Setup tracking parameters with enhanced defaults - ENHANCED WITH CENTRALIZED LOGGING"""
        log_function_call(self.logger, 'setup_parameters', (save_dir, filename))

        self.log_info("=== PARAMETER SETUP ===")

        setup_inputs = {
            'save_dir': save_dir,
            'filename': filename,
            'existing_gap_close_param': self.gap_close_param is not None,
            'existing_cost_matrices': len(self.cost_matrices) if self.cost_matrices else 0,
            'existing_kalman_functions': self.kalman_functions is not None
        }
        self.log_parameters(setup_inputs, "setup inputs")

        with self.time_operation("Parameter setup"):

            # General gap closing parameters
            if not hasattr(self, 'gap_close_param') or self.gap_close_param is None:
                self.log_info("Creating default gap close parameters")
                self.gap_close_param = GapCloseParam(
                    time_window=5,
                    merge_split=1,
                    min_track_len=3,
                    tolerance=0.05
                )

            # Cost matrix for frame-to-frame linking
            if not hasattr(self, 'cost_matrices') or not self.cost_matrices:
                self.log_info("Creating default cost matrices")

                # Create enhanced linking parameters
                self.log_debug("Creating linking parameters")
                linking_params = CostMatrixParameters(
                    linear_motion=1,  # Mixed motion
                    min_search_radius=2.0,
                    max_search_radius=10.0,
                    brown_std_mult=3.0,
                    lin_std_mult=3.0,
                    use_local_density=1,
                    max_angle_vv=30.0,
                    brown_scaling=[0.25, 0.01],
                    lin_scaling=[1.0, 0.01],
                    time_reach_conf_b=5,
                    time_reach_conf_l=5,
                    res_limit=0.5
                )

                linking_cost_matrix = {
                    'func_name': 'cost_mat_random_directed_switching_motion_link',
                    'parameters': linking_params
                }

                # Create enhanced gap closing parameters
                self.log_debug("Creating gap closing parameters")

                # Ensure we have a valid time window value
                time_window_val = getattr(self.gap_close_param, 'time_window', 5)
                if not isinstance(time_window_val, int) or time_window_val < 1:
                    time_window_val = 5
                    self.log_warning(f"Invalid time_window, using default: {time_window_val}")

                gap_closing_params = CostMatrixParameters(
                    linear_motion=1,
                    min_search_radius=2.0,
                    max_search_radius=15.0,  # Larger for gap closing
                    brown_std_mult=np.linspace(3.0, 4.5, time_window_val),
                    lin_std_mult=np.linspace(3.0, 4.5, time_window_val),
                    gap_penalty=1.5,
                    use_local_density=1,
                    max_angle_vv=45.0,  # More permissive for gap closing
                    brown_scaling=[0.25, 0.01],
                    lin_scaling=[1.0, 0.01],
                    time_reach_conf_b=time_window_val,
                    time_reach_conf_l=time_window_val,
                    res_limit=0.5
                )

                gap_closing_cost_matrix = {
                    'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                    'parameters': gap_closing_params
                }

                self.cost_matrices = [linking_cost_matrix, gap_closing_cost_matrix]
                self.log_info(f"Created {len(self.cost_matrices)} cost matrices")

            # Kalman filter functions
            if not hasattr(self, 'kalman_functions') or self.kalman_functions is None:
                self.log_info("Creating default Kalman functions")
                self.kalman_functions = KalmanFunctions()

        # Log final parameter setup
        final_setup = {
            'gap_close_param_type': type(self.gap_close_param).__name__,
            'cost_matrices_count': len(self.cost_matrices),
            'kalman_functions_type': type(self.kalman_functions).__name__,
            'prob_dim': self.prob_dim,
            'verbose': self.verbose
        }
        self.log_parameters(final_setup, "final parameter setup")

        self.log_info("Parameters setup completed with individual module compatibility")

    def _validate_movie_info(self, movie_info: List[Dict]) -> List[Dict]:
        """Enhanced validation with detailed logging - ENHANCED WITH CENTRALIZED LOGGING"""
        log_function_call(self.logger, '_validate_movie_info', (movie_info,))

        self.log_info("=== MOVIE INFO VALIDATION ===")

        validation_inputs = {
            'input_frames': safe_len(movie_info),
            'prob_dim': self.prob_dim
        }
        self.log_parameters(validation_inputs, "validation inputs")

        with self.time_operation("Movie info validation"):

            validated_info = []
            total_detections = 0
            frame_stats = []

            for i, frame_info in enumerate(movie_info):
                validated_frame = frame_info.copy()

                # Log original frame structure for first few frames
                if i < 3:
                    frame_debug = {
                        'frame_index': i,
                        'original_keys': list(frame_info.keys()),
                        'has_num': 'num' in frame_info,
                        'has_x_coord': 'x_coord' in frame_info,
                        'has_y_coord': 'y_coord' in frame_info,
                        'has_all_coord': 'all_coord' in frame_info,
                        'has_amp': 'amp' in frame_info
                    }
                    self.log_debug(f"Frame {i} original structure: {frame_debug}")

                # Ensure required fields exist
                if 'num' not in validated_frame:
                    if 'x_coord' in validated_frame:
                        validated_frame['num'] = len(validated_frame['x_coord'])
                        self.log_debug(f"Frame {i}: Set num = {validated_frame['num']} from x_coord length")
                    else:
                        validated_frame['num'] = 0
                        self.log_debug(f"Frame {i}: Set num = 0 (no coordinates found)")

                frame_detections = validated_frame['num']
                total_detections += frame_detections

                frame_stats.append({
                    'frame': i,
                    'detections': frame_detections
                })

                # Create all_coord if it doesn't exist
                if 'all_coord' not in validated_frame and validated_frame['num'] > 0:
                    self.log_debug(f"Frame {i}: Creating all_coord field")
                    self._create_all_coord_field(validated_frame)
                elif 'all_coord' not in validated_frame:
                    validated_frame['all_coord'] = np.array([]).reshape(0, 2 * self.prob_dim)
                    self.log_debug(f"Frame {i}: Created empty all_coord field")

                # Ensure amp exists
                if 'amp' not in validated_frame and validated_frame['num'] > 0:
                    validated_frame['amp'] = np.ones((validated_frame['num'], 2))
                    self.log_debug(f"Frame {i}: Created default amp field with shape {validated_frame['amp'].shape}")
                elif 'amp' not in validated_frame:
                    validated_frame['amp'] = np.array([]).reshape(0, 2)
                    self.log_debug(f"Frame {i}: Created empty amp field")

                # Log validated frame structure for first few frames
                if i < 3:
                    validated_debug = {
                        'frame_index': i,
                        'final_keys': list(validated_frame.keys()),
                        'num': validated_frame['num'],
                        'all_coord_shape': validated_frame['all_coord'].shape,
                        'amp_shape': validated_frame['amp'].shape
                    }
                    self.log_debug(f"Frame {i} validated structure: {validated_debug}")

                validated_info.append(validated_frame)

        # Summary statistics
        validation_summary = {
            'total_frames': safe_len(validated_info),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / safe_len(validated_info) if validated_info else 0,
            'frames_with_detections': sum(1 for f in frame_stats if f['detections'] > 0),
            'max_detections_in_frame': max(f['detections'] for f in frame_stats) if frame_stats else 0,
            'min_detections_in_frame': min(f['detections'] for f in frame_stats) if frame_stats else 0
        }
        self.log_parameters(validation_summary, "validation summary")

        self.log_info(f"Validation completed: {safe_len(validated_info)} frames, {total_detections} total detections")
        return validated_info

    def _create_all_coord_field(self, validated_frame):
        """Create all_coord field from individual coordinate arrays - ENHANCED WITH CENTRALIZED LOGGING"""
        log_function_call(self.logger, '_create_all_coord_field', (validated_frame,))

        num_features = validated_frame['num']

        creation_inputs = {
            'num_features': num_features,
            'prob_dim': self.prob_dim,
            'available_fields': list(validated_frame.keys())
        }
        self.log_parameters(creation_inputs, "all_coord creation inputs")

        if self.prob_dim == 2:
            x_coord = validated_frame.get('x_coord', np.zeros((num_features, 2)))
            y_coord = validated_frame.get('y_coord', np.zeros((num_features, 2)))

            log_array_info(self.logger, "x_coord", x_coord, "original")
            log_array_info(self.logger, "y_coord", y_coord, "original")

            # Ensure proper shape
            if x_coord.ndim == 1:
                x_coord = np.column_stack([x_coord, np.ones(len(x_coord)) * 0.1])
                self.log_debug("Reshaped x_coord to 2D")
            if y_coord.ndim == 1:
                y_coord = np.column_stack([y_coord, np.ones(len(y_coord)) * 0.1])
                self.log_debug("Reshaped y_coord to 2D")

            validated_frame['all_coord'] = np.column_stack([
                x_coord[:, 0], x_coord[:, 1],
                y_coord[:, 0], y_coord[:, 1]
            ])

            log_array_info(self.logger, "all_coord", validated_frame['all_coord'], "created for 2D")

        elif self.prob_dim == 3:
            x_coord = validated_frame.get('x_coord', np.zeros((num_features, 2)))
            y_coord = validated_frame.get('y_coord', np.zeros((num_features, 2)))
            z_coord = validated_frame.get('z_coord', np.zeros((num_features, 2)))

            log_array_info(self.logger, "x_coord", x_coord, "original 3D")
            log_array_info(self.logger, "y_coord", y_coord, "original 3D")
            log_array_info(self.logger, "z_coord", z_coord, "original 3D")

            # Ensure proper shape
            if x_coord.ndim == 1:
                x_coord = np.column_stack([x_coord, np.ones(len(x_coord)) * 0.1])
            if y_coord.ndim == 1:
                y_coord = np.column_stack([y_coord, np.ones(len(y_coord)) * 0.1])
            if z_coord.ndim == 1:
                z_coord = np.column_stack([z_coord, np.ones(len(z_coord)) * 0.1])

            validated_frame['all_coord'] = np.column_stack([
                x_coord[:, 0], x_coord[:, 1],
                y_coord[:, 0], y_coord[:, 1],
                z_coord[:, 0], z_coord[:, 1]
            ])

            log_array_info(self.logger, "all_coord", validated_frame['all_coord'], "created for 3D")

    def track_close_gaps_kalman_sparse(self, movie_info: List[Dict]) -> tuple:
        """
        Main tracking function implementation - ENHANCED WITH CENTRALIZED LOGGING (Legacy method)
        """
        log_function_call(self.logger, 'track_close_gaps_kalman_sparse', (movie_info,))

        self.log_warning("Using legacy track_close_gaps_kalman_sparse method")

        legacy_inputs = {
            'movie_info_frames': safe_len(movie_info),
            'method': 'legacy_track_close_gaps_kalman_sparse'
        }
        self.log_parameters(legacy_inputs, "legacy method inputs")

        with self.time_operation("Legacy tracking method"):

            try:
                # Import required modules
                self.log_info("Importing modules for legacy method")

                try:
                    from cost_matrices import CostMatrixRandomDirectedSwitchingMotion
                    self.log_info("✓ Imported CostMatrixRandomDirectedSwitchingMotion")
                except ImportError:
                    self.log_warning("Could not import CostMatrixRandomDirectedSwitchingMotion, using basic implementation")
                    class CostMatrixRandomDirectedSwitchingMotion:
                        def cost_mat_random_directed_switching_motion_link(self, *args, **kwargs):
                            logger.error("Cost matrix function not implemented")
                            return np.array([]), np.array([]), {}, -5, 1

                try:
                    from kalman_filters import KalmanFilterLinearMotion
                    self.log_info("✓ Imported KalmanFilterLinearMotion")
                except ImportError:
                    self.log_warning("Could not import KalmanFilterLinearMotion")
                    KalmanFilterLinearMotion = None

                try:
                    from gap_closing import GapCloser
                    self.log_info("✓ Imported GapCloser")
                except ImportError:
                    self.log_warning("Could not import GapCloser")
                    class GapCloser:
                        def close_gaps(self, *args, **kwargs):
                            logger.error("Gap closing not implemented")
                            return []

                try:
                    from linking import FeatureLinker
                    self.log_info("✓ Imported FeatureLinker")
                except ImportError:
                    self.log_warning("Could not import FeatureLinker")
                    class FeatureLinker:
                        def link_features_kalman_sparse(self, *args, **kwargs):
                            logger.error("Feature linking not implemented")
                            return [], [], [], [], {}, 1, None

                # Initialize components
                cost_matrix_calc = CostMatrixRandomDirectedSwitchingMotion()
                kalman_filter = KalmanFilterLinearMotion() if KalmanFilterLinearMotion else None
                gap_closer = GapCloser()
                feature_linker = FeatureLinker()

                components = {
                    'cost_matrix_calc': type(cost_matrix_calc).__name__,
                    'kalman_filter': type(kalman_filter).__name__ if kalman_filter else 'None',
                    'gap_closer': type(gap_closer).__name__,
                    'feature_linker': type(feature_linker).__name__
                }
                self.log_parameters(components, "legacy components")

                # Validate movie_info structure
                movie_info = self._validate_movie_info(movie_info)

                # Perform frame-to-frame linking
                self.log_info("Performing frame-to-frame linking (legacy)")

                with self.time_operation("Legacy linking"):

                    # Extract parameters from cost matrices
                    if hasattr(self.cost_matrices[0], 'parameters'):
                        link_params = self.cost_matrices[0].parameters
                    else:
                        link_params = self.cost_matrices[0].get('parameters', {})

                    if hasattr(self.cost_matrices[1], 'parameters'):
                        gap_params = self.cost_matrices[1].parameters
                    else:
                        gap_params = self.cost_matrices[1].get('parameters', {})

                    tracks_feat_indx_link, tracks_coord_amp_link, kalman_info_link, nn_dist_linked_feat, \
                    linking_costs, link_err_flag, trackability_data = \
                        feature_linker.link_features_kalman_sparse(
                            movie_info,
                            cost_matrix_calc.cost_mat_random_directed_switching_motion_link,
                            link_params,
                            self.kalman_functions,
                            self.prob_dim,
                            None,  # kalman_info_prev
                            None,  # linking_costs_prev
                            self.verbose
                        )

                legacy_linking_results = {
                    'tracks_feat_indx_link_count': safe_len(tracks_feat_indx_link),
                    'link_err_flag': link_err_flag
                }
                self.log_parameters(legacy_linking_results, "legacy linking results")

                if link_err_flag != 0:
                    self.log_error("Error in frame-to-frame linking")
                    return [], {}, 1

                # Close gaps and handle merges/splits
                self.log_info("Closing gaps and handling merges/splits (legacy)")

                with self.time_operation("Legacy gap closing"):

                    # Create gap close parameters dict from object
                    if hasattr(self.gap_close_param, '__dict__'):
                        gap_close_dict = self.gap_close_param.__dict__
                    else:
                        gap_close_dict = self.gap_close_param

                    tracks_final = gap_closer.close_gaps(
                        tracks_feat_indx_link,
                        tracks_coord_amp_link,
                        kalman_info_link,
                        nn_dist_linked_feat,
                        gap_close_dict,
                        {'parameters': gap_params},  # Wrap parameters in dict format
                        self.prob_dim,
                        movie_info,
                        self.verbose
                    )

                err_flag = 0
                self.log_info("Legacy tracking completed successfully")

                legacy_final_results = {
                    'tracks_final_count': safe_len(tracks_final),
                    'err_flag': err_flag,
                    'method': 'legacy_track_close_gaps_kalman_sparse'
                }
                self.log_parameters(legacy_final_results, "legacy final results")

            except ImportError as e:
                self.log_error(f"Import error - make sure all modules are in the same directory: {str(e)}")
                tracks_final = []
                kalman_info_link = {}
                err_flag = 1
            except Exception as e:
                self.log_error(f"Error during tracking: {str(e)}")
                self.log_exception("Full traceback:")
                tracks_final = []
                kalman_info_link = {}
                err_flag = 1

        return tracks_final, kalman_info_link, err_flag

    def _save_results(self, tracks_final, kalman_info_link, err_flag, save_dir, filename):
        """Save tracking results - ENHANCED WITH CENTRALIZED LOGGING (Legacy method)"""
        log_function_call(self.logger, '_save_results',
                         (tracks_final, kalman_info_link, err_flag, save_dir, filename))

        self.log_info("=== LEGACY SAVE RESULTS ===")

        legacy_save_inputs = {
            'tracks_final_count': safe_len(tracks_final),
            'err_flag': err_flag,
            'save_dir': save_dir,
            'filename': filename,
            'method': 'legacy_save_results'
        }
        self.log_parameters(legacy_save_inputs, "legacy save inputs")

        with self.time_operation("Legacy save results"):

            import pickle

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Prepare data for saving
            results = {
                'tracks_final': tracks_final,
                'kalman_info_link': kalman_info_link,
                'err_flag': err_flag,
                'gap_close_param': self.gap_close_param,
                'cost_matrices': self.cost_matrices,
                'kalman_functions': self.kalman_functions
            }

            # Save as pickle file (ensure .pkl extension)
            if not filename.endswith('.pkl'):
                filename = filename.replace('.mat', '.pkl')
                if not filename.endswith('.pkl'):
                    filename += '.pkl'

            filepath = os.path.join(save_dir, filename)

            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f)

                self.log_info(f"Results saved to: {filepath}")

            except Exception as e:
                self.log_error(f"Error saving results: {str(e)}")


# =============================================================================
# MAIN FUNCTION AND UTILITIES WITH LOGGING
# =============================================================================

def main():
    """Main function demonstrating usage - ENHANCED WITH CENTRALIZED LOGGING"""
    logger.info("=== MAIN FUNCTION DEMONSTRATION ===")

    with PerformanceTimer(logger, "Main function demonstration"):

        # Example usage
        tracker = ParticleTracker()

        # Generate example data
        logger.info("Generating example data")

        movie_info = []
        num_frames = 10

        logger.info(f"Generating {num_frames} frames of example data")

        for frame_idx in range(num_frames):
            # Generate example detection data
            num_particles = np.random.randint(5, 15)

            # Random particle positions
            x_positions = np.random.rand(num_particles) * 100
            y_positions = np.random.rand(num_particles) * 100
            x_uncertainties = np.ones(num_particles) * 0.1
            y_uncertainties = np.ones(num_particles) * 0.1

            # Intensities/amplitudes
            amplitudes = np.random.rand(num_particles) * 1000 + 100
            amp_uncertainties = amplitudes * 0.1

            frame_info = {
                'x_coord': np.column_stack([x_positions, x_uncertainties]),
                'y_coord': np.column_stack([y_positions, y_uncertainties]),
                'amp': np.column_stack([amplitudes, amp_uncertainties]),
                'num': num_particles
            }

            if frame_idx < 3:  # Log first 3 frames
                frame_debug = {
                    'frame_index': frame_idx,
                    'num_particles': num_particles,
                    'x_range': f"{np.min(x_positions):.1f} - {np.max(x_positions):.1f}",
                    'y_range': f"{np.min(y_positions):.1f} - {np.max(y_positions):.1f}",
                    'amp_range': f"{np.min(amplitudes):.1f} - {np.max(amplitudes):.1f}"
                }
                logger.debug(f"Example frame {frame_idx}: {frame_debug}")

            movie_info.append(frame_info)

        # Set custom save directory
        save_dir = "/project/biophysics/jaqaman_lab/vegf_tsp1/slee/VEGFR2/2015/20150312_HMVECp6_AAL/AAL-NoVEGF/Stream_10min/TrackingPackage/tracks/"
        filename = "tracks_test.pkl"

        # For testing without the full directory structure, use current directory
        if not os.path.exists(save_dir):
            save_dir = "./results"
            logger.info(f"Using local directory: {save_dir}")

        save_config = {
            'save_dir': save_dir,
            'filename': filename,
            'directory_exists': os.path.exists(save_dir)
        }
        logger.info(f"Save configuration: {save_config}")

        # Run tracking
        logger.info("Running main tracking example")

        try:
            with PerformanceTimer(logger, "Main tracking execution"):
                tracks_final, kalman_info_link, err_flag = tracker.run_tracking(
                    movie_info, save_dir, filename
                )

            # Log final results
            final_results = {
                'tracks_final_count': safe_len(tracks_final),
                'kalman_info_keys': list(kalman_info_link.keys()) if isinstance(kalman_info_link, dict) else 'Not a dict',
                'err_flag': err_flag,
                'success': err_flag == 0
            }
            logger.info(f"Main tracking results: {final_results}")

            logger.info("Tracking completed!")
            logger.info(f"Error flag: {err_flag}")
            logger.info(f"Number of final tracks: {safe_len(tracks_final)}")

            # Display basic track information
            if tracks_final and safe_len(tracks_final) > 0:
                logger.info("Track summary:")
                for i, track in enumerate(tracks_final[:5]):  # Show first 5 tracks
                    seq_events = track.get('seq_of_events', [])
                    if len(seq_events) >= 2:
                        start_frame = seq_events[0][0]
                        end_frame = seq_events[-1][0]
                        track_length = end_frame - start_frame + 1
                        logger.info(f"Track {i+1}: frames {start_frame}-{end_frame} (length: {track_length})")

            logger.info("=== MAIN FUNCTION COMPLETED SUCCESSFULLY ===")

        except Exception as e:
            logger.error(f"Error running tracking: {str(e)}")
            logger.error("Make sure all module files are in the same directory")
            logger.exception("Full traceback:")


def load_movie_info_from_file(filepath: str) -> List[Dict]:
    """
    Load movie information from various file formats - ENHANCED WITH CENTRALIZED LOGGING
    """
    log_function_call(logger, 'load_movie_info_from_file', (filepath,))

    logger.info("=== LOADING MOVIE INFO FROM FILE ===")

    load_inputs = {
        'filepath': filepath,
        'file_exists': os.path.exists(filepath) if isinstance(filepath, str) else False
    }
    logger.info(f"Load inputs: {load_inputs}")

    with PerformanceTimer(logger, "Loading movie info from file"):

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        file_ext = os.path.splitext(filepath)[1].lower()
        logger.info(f"File extension detected: {file_ext}")

        if file_ext == '.mat':
            logger.info("Loading MATLAB .mat file")
            try:
                import scipy.io
                mat_data = scipy.io.loadmat(filepath)
                result = _convert_matlab_movie_info(mat_data)
                logger.info(f"Successfully loaded MAT file: {safe_len(result)} frames")
                return result
            except ImportError:
                raise ImportError("scipy is required to load .mat files")

        elif file_ext in ['.csv', '.txt']:
            logger.info("Loading CSV/text file")
            result = _load_csv_movie_info(filepath)
            logger.info(f"Successfully loaded CSV file: {safe_len(result)} frames")
            return result

        elif file_ext == '.pkl':
            logger.info("Loading pickle file")
            import pickle
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            logger.info(f"Successfully loaded PKL file: {safe_len(result)} frames")
            return result

        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


def _convert_matlab_movie_info(mat_data: Dict) -> List[Dict]:
    """Convert MATLAB movieInfo structure to Python format - ENHANCED WITH CENTRALIZED LOGGING"""
    log_function_call(logger, '_convert_matlab_movie_info', (mat_data,))

    logger.info("=== CONVERTING MATLAB MOVIE INFO ===")

    with PerformanceTimer(logger, "MATLAB movieInfo conversion"):

        matlab_keys = list(mat_data.keys())
        conversion_inputs = {
            'mat_data_keys': matlab_keys,
            'has_movieInfo': 'movieInfo' in mat_data
        }
        logger.info(f"MATLAB conversion inputs: {conversion_inputs}")

        movie_info = []

        if 'movieInfo' in mat_data:
            matlab_movie_info = mat_data['movieInfo']
            logger.info(f"Found movieInfo with {safe_len(matlab_movie_info)} frames")

            for i, frame_data in enumerate(matlab_movie_info):
                frame_info = {
                    'x_coord': frame_data.get('xCoord', np.array([])),
                    'y_coord': frame_data.get('yCoord', np.array([])),
                    'amp': frame_data.get('amp', np.array([])),
                    'num': frame_data.get('num', 0)
                }

                if i < 3:  # Log first 3 frames
                    frame_debug = {
                        'frame_index': i,
                        'x_coord_shape': frame_info['x_coord'].shape if hasattr(frame_info['x_coord'], 'shape') else 'No shape',
                        'y_coord_shape': frame_info['y_coord'].shape if hasattr(frame_info['y_coord'], 'shape') else 'No shape',
                        'amp_shape': frame_info['amp'].shape if hasattr(frame_info['amp'], 'shape') else 'No shape',
                        'num': frame_info['num']
                    }
                    logger.debug(f"MATLAB frame {i} conversion: {frame_debug}")

                movie_info.append(frame_info)

        conversion_results = {
            'converted_frames': safe_len(movie_info),
            'total_detections': sum(frame.get('num', 0) for frame in movie_info)
        }
        logger.info(f"MATLAB conversion results: {conversion_results}")

    return movie_info


def _load_csv_movie_info(filepath: str) -> List[Dict]:
    """Load movie info from CSV file - ENHANCED WITH CENTRALIZED LOGGING"""
    log_function_call(logger, '_load_csv_movie_info', (filepath,))

    logger.info("=== LOADING CSV MOVIE INFO ===")

    with PerformanceTimer(logger, "CSV movie info loading"):

        import pandas as pd

        # Load CSV file
        df = pd.read_csv(filepath)

        csv_info = {
            'csv_shape': df.shape,
            'csv_columns': list(df.columns),
            'has_frame_column': 'frame' in df.columns,
            'unique_frames': df['frame'].nunique() if 'frame' in df.columns else 'No frame column'
        }
        logger.info(f"CSV file info: {csv_info}")

        # Group by frame
        movie_info = []
        for frame_num in sorted(df['frame'].unique()):
            frame_data = df[df['frame'] == frame_num]

            frame_info = {
                'x_coord': np.column_stack([frame_data['x'].values,
                                          frame_data.get('x_err', np.ones(len(frame_data)) * 0.1).values]),
                'y_coord': np.column_stack([frame_data['y'].values,
                                          frame_data.get('y_err', np.ones(len(frame_data)) * 0.1).values]),
                'amp': np.column_stack([frame_data.get('intensity', np.ones(len(frame_data)) * 100).values,
                                      frame_data.get('int_err', np.ones(len(frame_data)) * 10).values]),
                'num': len(frame_data)
            }

            if len(movie_info) < 3:  # Log first 3 frames
                csv_frame_debug = {
                    'frame_number': frame_num,
                    'detections_in_frame': len(frame_data),
                    'x_coord_shape': frame_info['x_coord'].shape,
                    'y_coord_shape': frame_info['y_coord'].shape,
                    'amp_shape': frame_info['amp'].shape
                }
                logger.debug(f"CSV frame {frame_num}: {csv_frame_debug}")

            movie_info.append(frame_info)

        csv_results = {
            'loaded_frames': safe_len(movie_info),
            'total_detections': sum(frame['num'] for frame in movie_info)
        }
        logger.info(f"CSV loading results: {csv_results}")

    return movie_info


def test_basic_functionality():
    """Test basic functionality to verify imports and setup - ENHANCED WITH CENTRALIZED LOGGING"""
    logger.info("=== TESTING BASIC FUNCTIONALITY ===")

    with PerformanceTimer(logger, "Basic functionality test"):

        try:
            logger.info("Testing basic particle tracker functionality")

            # Test parameter creation
            logger.info("Creating parameters")
            cost_params = CostMatrixParameters()
            gap_params = GapCloseParameters()
            logger.info("Parameters created successfully")

            # Test tracker creation
            logger.info("Creating tracker")
            tracker = ParticleTracker()
            logger.info("Tracker created successfully")

            # Test parameter setup
            logger.info("Setting up parameters")
            tracker.setup_parameters()
            logger.info("Parameters setup successfully")

            logger.info("Basic functionality test passed!")
            return True

        except Exception as e:
            logger.error(f"Basic functionality test failed: {str(e)}")
            logger.exception("Full traceback:")
            return False


if __name__ == "__main__":
    logger.info("=== SCRIPT EXECUTION START ===")

    # Run basic test first
    if test_basic_functionality():
        logger.info("Running main tracking example")
        main()
    else:
        logger.error("Basic test failed. Please check your installation.")
        logger.error("Make sure all required Python files are in the same directory:")
        required_files = [
            "track_general.py", "cost_matrices.py", "kalman_filters.py",
            "linking.py", "gap_closing.py", "utils.py", "track_analysis.py",
            "visualization.py", "test_tracking.py", "module_logger.py"
        ]
        for f in required_files:
            logger.error(f"  - {f}")

    logger.info("=== SCRIPT EXECUTION COMPLETED ===")
