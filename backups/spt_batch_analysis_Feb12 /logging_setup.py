# pytrack/logging_setup.py
"""
Comprehensive logging setup for PyTrack

This module configures logging for all PyTrack components with support for:
- Console and file logging with different levels
- Rotating log files with size limits
- Performance monitoring and debugging
- Parameter flow tracking
- Error reporting and debugging assistance
"""

import logging
import logging.handlers
import sys
import time
import functools
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import traceback
import threading
from contextlib import contextmanager
import sys
from pathlib import Path

# Add the current directory to Python path so we can import our modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # Import configuration
    from .config import get_config
except:
    # Import configuration
    from config import get_config


class PyTrackFormatter(logging.Formatter):
    """Custom formatter for PyTrack with enhanced information."""

    def __init__(self, include_thread=False, include_function=False):
        self.include_thread = include_thread
        self.include_function = include_function
        super().__init__()

    def format(self, record):
        # Base format
        fmt = "%(asctime)s - %(name)s - %(levelname)s"

        # Add thread info if requested
        if self.include_thread:
            fmt += " - [%(thread)d]"

        # Add function info if requested
        if self.include_function and hasattr(record, 'funcName'):
            fmt += " - %(funcName)s"

        fmt += " - %(message)s"

        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class PerformanceLogger:
    """Logger for performance monitoring and timing."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._timers[name] = time.time()
        self.logger.debug(f"Timer '{name}' started")

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        if name not in self._timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0

        elapsed = time.time() - self._timers[name]
        del self._timers[name]
        self.logger.debug(f"Timer '{name}' stopped: {elapsed:.4f}s")
        return elapsed

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing code blocks."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)


class ParameterFlowLogger:
    """Logger for tracking parameter flow through the pipeline."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_parameters(self, module: str, function: str, parameters: Dict[str, Any]) -> None:
        """Log parameters being passed to a function."""
        if self.logger.isEnabledFor(logging.DEBUG):
            param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
            self.logger.debug(f"{module}.{function} called with: {param_str}")

    def log_results(self, module: str, function: str, results: Dict[str, Any]) -> None:
        """Log results returned from a function."""
        if self.logger.isEnabledFor(logging.DEBUG):
            result_str = ", ".join([f"{k}={v}" for k, v in results.items()])
            self.logger.debug(f"{module}.{function} returned: {result_str}")


class ErrorReporter:
    """Enhanced error reporting with context information."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def report_error(self, error: Exception, context: Dict[str, Any] = None,
                    suggestions: str = None) -> None:
        """
        Report an error with enhanced context information.

        Args:
            error: The exception that occurred
            context: Additional context information
            suggestions: Suggested solutions or debugging steps
        """
        error_msg = f"Error: {type(error).__name__}: {str(error)}"

        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            error_msg += f"\nContext: {context_str}"

        if suggestions:
            error_msg += f"\nSuggestions: {suggestions}"

        # Add stack trace
        error_msg += f"\nStack trace:\n{traceback.format_exc()}"

        self.logger.error(error_msg)

    def report_validation_error(self, parameter: str, value: Any,
                              expected: str, module: str = None) -> None:
        """Report parameter validation errors with helpful information."""
        msg = f"Invalid parameter: {parameter}={value}, expected {expected}"
        if module:
            msg = f"[{module}] {msg}"

        suggestions = self._get_parameter_suggestions(parameter, value, expected)
        if suggestions:
            msg += f"\nSuggestion: {suggestions}"

        self.logger.error(msg)

    def _get_parameter_suggestions(self, parameter: str, value: Any, expected: str) -> str:
        """Generate helpful suggestions for parameter errors."""
        suggestions = {
            'max_linking_distance': "Try values between 3-10 pixels based on particle movement",
            'wavelet_scales': "Use scales [1,2,3] for small particles, [2,3,4] for larger ones",
            'blob_threshold': "Lower values (0.001-0.01) detect more particles",
            'min_track_length': "Use values >= 3 for reliable tracking statistics"
        }
        return suggestions.get(parameter, "Check documentation for valid parameter ranges")


def setup_logging(config=None) -> Dict[str, logging.Logger]:
    """
    Set up comprehensive logging for PyTrack.

    Args:
        config: Logging configuration, uses global config if None

    Returns:
        Dictionary of configured loggers for each module
    """
    if config is None:
        config = get_config().logging

    # Create log directory
    log_dir = Path(config.log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers to avoid duplicates
    root_logger = logging.getLogger('pytrack')
    root_logger.handlers.clear()

    # Set root logger level to DEBUG to allow all messages through
    root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.console_level.upper()))
    console_formatter = PyTrackFormatter(include_function=True)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if config.log_to_file:
        log_file = log_dir / config.log_filename

        # Convert size string to bytes
        max_bytes = _parse_size_string(config.max_log_size)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, config.file_level.upper()))
        file_formatter = PyTrackFormatter(include_thread=True, include_function=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Create module-specific loggers
    modules = ['detection', 'tracking', 'classification', 'visualization', 'utils']
    loggers = {}

    for module in modules:
        logger = logging.getLogger(f'pytrack.{module}')
        logger.setLevel(logging.DEBUG)

        # Add debug handler for specific modules if requested
        debug_attr = f'{module}_debug'
        if hasattr(config, debug_attr) and getattr(config, debug_attr):
            debug_file = log_dir / f'{module}_debug.log'
            debug_handler = logging.handlers.RotatingFileHandler(
                debug_file,
                maxBytes=max_bytes // 4,  # Smaller debug files
                backupCount=2
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(file_formatter)
            logger.addHandler(debug_handler)

        loggers[module] = logger

    # Log setup completion
    root_logger.info("PyTrack logging initialized successfully")
    root_logger.info(f"Console level: {config.console_level}")
    root_logger.info(f"File level: {config.file_level}")
    root_logger.info(f"Log directory: {log_dir.absolute()}")

    return loggers


def _parse_size_string(size_str: str) -> int:
    """Convert size string like '10MB' to bytes."""
    size_str = size_str.upper()
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }

    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            return int(size_str[:-len(suffix)]) * multiplier

    # Default to MB if no suffix
    try:
        return int(size_str) * 1024**2
    except ValueError:
        return 10 * 1024**2  # Default 10MB


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module or function."""
    return logging.getLogger(f'pytrack.{name}')


def log_function_call(logger: logging.Logger = None, log_params: bool = True,
                     log_results: bool = False, log_timing: bool = False):
    """
    Decorator to automatically log function calls with parameters and timing.

    Args:
        logger: Logger to use, creates one from function module if None
        log_params: Whether to log function parameters
        log_results: Whether to log function results
        log_timing: Whether to log execution time
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__.split('.')[-1])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"

            # Log function entry
            if log_params and logger.isEnabledFor(logging.DEBUG):
                params = {}
                # Get parameter names
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Filter out large objects for logging
                for name, value in bound_args.arguments.items():
                    if isinstance(value, (int, float, str, bool, type(None))):
                        params[name] = value
                    elif hasattr(value, '__len__'):
                        params[name] = f"<{type(value).__name__} len={len(value)}>"
                    else:
                        params[name] = f"<{type(value).__name__}>"

                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                logger.debug(f"→ {func_name}({param_str})")

            # Execute function with timing
            start_time = time.time() if log_timing else None
            try:
                result = func(*args, **kwargs)

                # Log timing
                if log_timing:
                    elapsed = time.time() - start_time
                    logger.debug(f"← {func_name} completed in {elapsed:.4f}s")

                # Log results
                if log_results and logger.isEnabledFor(logging.DEBUG):
                    if isinstance(result, (int, float, str, bool, type(None))):
                        logger.debug(f"← {func_name} returned: {result}")
                    elif hasattr(result, '__len__'):
                        logger.debug(f"← {func_name} returned: <{type(result).__name__} len={len(result)}>")
                    else:
                        logger.debug(f"← {func_name} returned: <{type(result).__name__}>")

                return result

            except Exception as e:
                if log_timing:
                    elapsed = time.time() - start_time
                    logger.error(f"✗ {func_name} failed after {elapsed:.4f}s: {e}")
                else:
                    logger.error(f"✗ {func_name} failed: {e}")
                raise

        return wrapper
    return decorator


def create_progress_logger(logger: logging.Logger, total: int,
                          description: str = "Processing") -> Callable:
    """
    Create a progress logging function for long-running operations.

    Args:
        logger: Logger to use
        total: Total number of items to process
        description: Description of the operation

    Returns:
        Function to call with current progress
    """
    start_time = time.time()
    last_log_time = start_time

    def log_progress(current: int, message: str = None):
        nonlocal last_log_time

        now = time.time()
        # Log every 5 seconds or at 10% intervals
        if (now - last_log_time > 5.0 or
            current % max(1, total // 10) == 0 or
            current == total):

            progress = (current / total) * 100
            elapsed = now - start_time

            if current > 0:
                eta = (elapsed / current) * (total - current)
                eta_str = f", ETA: {eta:.1f}s"
            else:
                eta_str = ""

            log_msg = f"{description}: {current}/{total} ({progress:.1f}%)"
            if message:
                log_msg += f" - {message}"
            log_msg += f" [Elapsed: {elapsed:.1f}s{eta_str}]"

            logger.info(log_msg)
            last_log_time = now

    return log_progress


# Global instances for easy access
_performance_loggers: Dict[str, PerformanceLogger] = {}
_parameter_loggers: Dict[str, ParameterFlowLogger] = {}
_error_reporters: Dict[str, ErrorReporter] = {}


def get_performance_logger(module: str) -> PerformanceLogger:
    """Get performance logger for a module."""
    if module not in _performance_loggers:
        logger = get_logger(module)
        _performance_loggers[module] = PerformanceLogger(logger)
    return _performance_loggers[module]


def get_parameter_logger(module: str) -> ParameterFlowLogger:
    """Get parameter flow logger for a module."""
    if module not in _parameter_loggers:
        logger = get_logger(module)
        _parameter_loggers[module] = ParameterFlowLogger(logger)
    return _parameter_loggers[module]


def get_error_reporter(module: str) -> ErrorReporter:
    """Get error reporter for a module."""
    if module not in _error_reporters:
        logger = get_logger(module)
        _error_reporters[module] = ErrorReporter(logger)
    return _error_reporters[module]


# Convenience function to initialize all loggers
def initialize_logging():
    """Initialize all PyTrack logging components."""
    loggers = setup_logging()

    # Initialize specialized loggers for each module
    for module in loggers.keys():
        get_performance_logger(module)
        get_parameter_logger(module)
        get_error_reporter(module)

    return loggers
