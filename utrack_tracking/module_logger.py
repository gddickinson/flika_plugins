#!/usr/bin/env python3
"""
Centralized Logging System for U-Track Modules

This module provides a consistent logging framework for all U-Track modules.
Each module gets its own log file with timestamps, log levels, and proper formatting.

Author: U-Track Enhanced Version
Version: 1.0.0
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class MillisecondFormatter(logging.Formatter):
    """Custom formatter that includes milliseconds in timestamp"""

    def formatTime(self, record, datefmt=None):
        """Format timestamp with milliseconds"""
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime('%H:%M:%S', ct)
        # Add milliseconds
        s += '.%03d' % (record.msecs)
        return s


class UTrackLogger:
    """
    Centralized logging system for U-Track modules

    Features:
    - Individual log files for each module
    - Configurable log levels
    - Timestamp formatting
    - Console output control
    - Log file rotation
    - Performance timing utilities
    """

    _loggers: Dict[str, logging.Logger] = {}
    _log_directory: Optional[Path] = None
    _global_log_level = logging.DEBUG
    _console_output = False  # Set to True to also show in console
    _session_id = None

    @classmethod
    def setup_logging_directory(cls, log_dir: Optional[str] = None):
        """Set up the main logging directory"""
        if log_dir is None:
            # Default to a logs directory in the plugin folder
            plugin_dir = Path(__file__).parent
            log_dir = plugin_dir / "logs"
        else:
            log_dir = Path(log_dir)

        # Create directory if it doesn't exist
        log_dir.mkdir(parents=True, exist_ok=True)
        cls._log_directory = log_dir

        # Create session ID for this tracking session
        cls._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        return log_dir

    @classmethod
    def get_logger(cls, module_name: str,
                   log_level: Optional[int] = None,
                   console_output: Optional[bool] = None) -> logging.Logger:
        """
        Get a logger for a specific module

        Args:
            module_name: Name of the module (e.g., 'linking', 'gap_closing')
            log_level: Optional log level override
            console_output: Optional console output override

        Returns:
            Configured logger instance
        """

        # Ensure logging directory is set up
        if cls._log_directory is None:
            cls.setup_logging_directory()

        # Use provided settings or defaults
        if log_level is None:
            log_level = cls._global_log_level
        if console_output is None:
            console_output = cls._console_output

        # Return existing logger if already created
        logger_key = f"{module_name}_{log_level}_{console_output}"
        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        # Create new logger
        logger = logging.getLogger(f"utrack.{module_name}")
        logger.setLevel(log_level)

        # Clear any existing handlers
        logger.handlers.clear()

        # Create file handler
        log_filename = f"utrack_{module_name}_{cls._session_id}.log"
        log_filepath = cls._log_directory / log_filename

        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)

        # Create console handler if requested
        console_handler = None
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console

        # Create formatter with millisecond precision
        formatter = MillisecondFormatter(
            '%(asctime)s | %(levelname)8s | %(funcName)20s | %(message)s'
        )

        # Apply formatter to handlers
        file_handler.setFormatter(formatter)
        if console_handler:
            console_formatter = logging.Formatter(
                '%(levelname)s [%(name)s]: %(message)s'
            )
            console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        if console_handler:
            logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        # Store logger
        cls._loggers[logger_key] = logger

        # Log initial message
        logger.info(f"=== {module_name.upper()} MODULE LOGGING STARTED ===")
        logger.info(f"Log file: {log_filepath}")
        logger.info(f"Log level: {logging.getLevelName(log_level)}")
        logger.info(f"Session ID: {cls._session_id}")

        return logger

    @classmethod
    def set_global_log_level(cls, level: int):
        """Set global log level for all future loggers"""
        cls._global_log_level = level

    @classmethod
    def set_console_output(cls, enabled: bool):
        """Enable/disable console output for all future loggers"""
        cls._console_output = enabled

    @classmethod
    def get_log_directory(cls) -> Path:
        """Get the current log directory"""
        if cls._log_directory is None:
            cls.setup_logging_directory()
        return cls._log_directory

    @classmethod
    def get_session_id(cls) -> str:
        """Get the current session ID"""
        if cls._session_id is None:
            cls._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls._session_id

    @classmethod
    def list_log_files(cls) -> list:
        """List all log files in the log directory"""
        if cls._log_directory is None:
            return []
        return list(cls._log_directory.glob("utrack_*.log"))

    @classmethod
    def clear_old_logs(cls, days_to_keep: int = 7):
        """Clear log files older than specified days"""
        if cls._log_directory is None:
            return

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        removed_count = 0

        for log_file in cls._log_directory.glob("utrack_*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                removed_count += 1

        return removed_count


class PerformanceTimer:
    """
    Utility class for timing operations and logging performance
    """

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"TIMING: Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            self.logger.info(f"TIMING: {self.operation_name} completed in {duration:.4f} seconds")
        else:
            self.logger.error(f"TIMING: {self.operation_name} failed after {duration:.4f} seconds")

    def lap(self, lap_name: str):
        """Log an intermediate timing"""
        if self.start_time is None:
            self.logger.warning("TIMING: lap() called before timer started")
            return

        lap_time = time.time() - self.start_time
        self.logger.debug(f"TIMING: {self.operation_name} - {lap_name}: {lap_time:.4f} seconds")


class LoggingMixin:
    """
    Mixin class to add consistent logging to any class
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get module name from the class's module
        try:
            module_name = self.__class__.__module__.split('.')[-1]
            if module_name == '__main__':
                module_name = self.__class__.__name__.lower()
        except (AttributeError, IndexError):
            # Fallback to class name if module detection fails
            module_name = self.__class__.__name__.lower()

        self.logger = UTrackLogger.get_logger(module_name)
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

    def log_exception(self, message: str):
        """Log exception with traceback"""
        self.logger.exception(message)

    def log_parameters(self, params: Dict[str, Any], context: str = ""):
        """Log parameters in a structured way"""
        context_str = f" ({context})" if context else ""
        self.logger.info(f"PARAMETERS{context_str}:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def time_operation(self, operation_name: str) -> PerformanceTimer:
        """Create a performance timer for an operation"""
        return PerformanceTimer(self.logger, operation_name)


# Convenience functions for quick logging setup
def get_module_logger(module_name: str = None,
                     log_level: Optional[int] = None,
                     console_output: Optional[bool] = None) -> logging.Logger:
    """
    Convenience function to get a logger for the current module

    Args:
        module_name: Optional module name. If None, infers from call stack
        log_level: Optional log level override (e.g., logging.DEBUG, logging.INFO)
        console_output: Optional console output override

    Returns:
        Configured logger
    """
    if module_name is None:
        # Try to infer module name from the calling frame
        import inspect
        frame = inspect.currentframe().f_back
        module_name = frame.f_globals.get('__name__', 'unknown')
        if '.' in module_name:
            module_name = module_name.split('.')[-1]
        if module_name == '__main__':
            module_name = 'main'

    return UTrackLogger.get_logger(module_name, log_level, console_output)


def log_function_call(logger: logging.Logger, func_name: str, args: tuple = (), kwargs: dict = None):
    """Log a function call with its arguments"""
    if kwargs is None:
        kwargs = {}

    args_str = ', '.join(str(arg) for arg in args)
    kwargs_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"CALL: {func_name}({all_args})")


def log_array_info(logger: logging.Logger, array_name: str, array, context: str = ""):
    """Log information about numpy arrays and other data structures"""
    import numpy as np

    context_str = f" ({context})" if context else ""

    if array is None:
        logger.debug(f"ARRAY{context_str}: {array_name} = None")
    elif isinstance(array, np.ndarray):
        try:
            logger.debug(f"ARRAY{context_str}: {array_name} shape={array.shape}, dtype={array.dtype}, "
                        f"min={np.min(array):.3f}, max={np.max(array):.3f}, mean={np.mean(array):.3f}")
        except (ValueError, TypeError) as e:
            logger.debug(f"ARRAY{context_str}: {array_name} shape={array.shape}, dtype={array.dtype} (stats unavailable: {e})")
    elif isinstance(array, (list, tuple)):
        try:
            logger.debug(f"ARRAY{context_str}: {array_name} = {type(array).__name__} with {len(array)} elements")
            if len(array) <= 10:  # Show contents for small arrays
                logger.debug(f"ARRAY{context_str}: {array_name} contents = {array}")
        except Exception as e:
            logger.debug(f"ARRAY{context_str}: {array_name} = {type(array).__name__} (length unavailable: {e})")
    elif isinstance(array, (int, float, complex)):
        logger.debug(f"ARRAY{context_str}: {array_name} = {array} ({type(array).__name__})")
    else:
        try:
            if hasattr(array, '__len__'):
                logger.debug(f"ARRAY{context_str}: {array_name} = {type(array)} with {len(array)} elements")
            else:
                logger.debug(f"ARRAY{context_str}: {array_name} = {type(array)} = {str(array)[:100]}")
        except Exception as e:
            logger.debug(f"ARRAY{context_str}: {array_name} = {type(array)} (info unavailable: {e})")


# Example usage patterns for modules:
"""
# At the top of each module:
from .module_logger import get_module_logger, PerformanceTimer, log_function_call, log_array_info

# Get logger for this module
logger = get_module_logger()  # Auto-detects module name

# OR explicitly name it:
logger = get_module_logger('linking')

# Basic logging:
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")

# Function call logging:
def my_function(param1, param2, option=None):
    log_function_call(logger, 'my_function', (param1, param2), {'option': option})
    # ... function body ...

# Performance timing:
with PerformanceTimer(logger, "Cost matrix calculation"):
    # ... expensive operation ...
    pass

# Array logging:
log_array_info(logger, "movie_info", movie_info_array, "input data")

# Class-based logging (inherit from LoggingMixin):
class MyTracker(LoggingMixin):
    def __init__(self):
        super().__init__()  # Sets up self.logger

    def track_particles(self):
        self.log_info("Starting particle tracking")
        with self.time_operation("Particle tracking"):
            # ... tracking code ...
            pass
"""

# Configure logging on import
if __name__ != "__main__":
    # Set up logging directory when module is imported
    UTrackLogger.setup_logging_directory()
