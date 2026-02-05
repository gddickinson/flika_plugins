#!/usr/bin/env python3
"""
Logger Configuration Module
===========================

Comprehensive logging setup for the Volume Slider plugin.
Provides structured logging with multiple handlers, performance monitoring,
and integration with Qt applications.
"""

import logging
import logging.handlers
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
from contextlib import contextmanager
import json

# Qt imports for GUI logging integration
try:
    from qtpy.QtCore import QObject, Signal
    from qtpy.QtWidgets import QApplication
    HAS_QT = True
except ImportError:
    HAS_QT = False
    QObject = object
    Signal = lambda *args, **kwargs: None


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname

        return super().format(record)


class PerformanceFormatter(logging.Formatter):
    """Specialized formatter for performance monitoring logs."""

    def format(self, record):
        """Format performance records with timing information."""
        if hasattr(record, 'duration'):
            record.msg = f"{record.msg} [Duration: {record.duration:.3f}s]"

        if hasattr(record, 'memory_usage'):
            record.msg = f"{record.msg} [Memory: {record.memory_usage:.1f}MB]"

        return super().format(record)


class QtLogHandler(logging.Handler, QObject):
    """
    Custom logging handler that emits Qt signals for GUI integration.

    Allows GUI components to receive and display log messages in real-time.
    """

    # Qt signals for log messages
    log_message = Signal(str, int, str)  # message, level, timestamp

    def __init__(self):
        logging.Handler.__init__(self)
        if HAS_QT:
            QObject.__init__(self)
        self.setLevel(logging.INFO)  # Default to INFO for GUI

    def emit(self, record):
        """Emit log record as Qt signal."""
        if HAS_QT:
            try:
                message = self.format(record)
                timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
                self.log_message.emit(message, record.levelno, timestamp)
            except Exception:
                self.handleError(record)


class DataProcessingFilter(logging.Filter):
    """Custom filter for data processing related messages."""

    def filter(self, record):
        """Filter records related to data processing."""
        # Add context information for data processing operations
        if hasattr(record, 'operation'):
            record.msg = f"[{record.operation}] {record.msg}"

        # Add data shape information if available
        if hasattr(record, 'data_shape'):
            record.msg = f"{record.msg} (shape: {record.data_shape})"

        return True


class LogManager:
    """
    Central log management system for the Volume Slider plugin.

    Features:
    - Multiple output handlers (console, file, rotating file, GUI)
    - Performance monitoring integration
    - Structured logging with context
    - Log level management
    - Error aggregation and reporting
    """

    def __init__(self):
        self.root_logger = None
        self.qt_handler = None
        self.log_directory = None
        self.performance_logger = None
        self.error_count = 0
        self.warning_count = 0
        self.start_time = time.time()

    def setup_logging(self,
                     log_level: Union[str, int] = logging.INFO,
                     log_dir: Optional[Union[str, Path]] = None,
                     enable_console: bool = True,
                     enable_file: bool = True,
                     enable_gui: bool = True,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5) -> logging.Logger:
        """
        Setup comprehensive logging system.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files (default: user's temp directory)
            enable_console: Enable console output
            enable_file: Enable file logging
            enable_gui: Enable Qt GUI integration
            max_file_size: Maximum size per log file before rotation
            backup_count: Number of backup files to keep

        Returns:
            Configured root logger instance
        """
        # Convert string log level to integer
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Setup log directory
        if log_dir is None:
            log_dir = Path.home() / '.volume_slider' / 'logs'
        else:
            log_dir = Path(log_dir)

        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_directory = log_dir

        # Configure root logger
        self.root_logger = logging.getLogger('volume_slider')
        self.root_logger.setLevel(log_level)

        # Clear existing handlers
        self.root_logger.handlers.clear()

        # Setup formatters
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )

        colored_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        performance_formatter = PerformanceFormatter(
            fmt='%(asctime)s | PERF | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)

            # Use colored output if terminal supports it
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                console_handler.setFormatter(colored_formatter)
            else:
                console_handler.setFormatter(simple_formatter)

            self.root_logger.addHandler(console_handler)

        # File handler with rotation
        if enable_file:
            log_file = log_dir / f'volume_slider_{datetime.now().strftime("%Y%m%d")}.log'

            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # Always capture all levels in files
            file_handler.setFormatter(detailed_formatter)

            self.root_logger.addHandler(file_handler)

        # Error-only file handler
        if enable_file:
            error_log_file = log_dir / f'volume_slider_errors_{datetime.now().strftime("%Y%m%d")}.log'

            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)

            self.root_logger.addHandler(error_handler)

        # Qt GUI handler
        if enable_gui and HAS_QT:
            self.qt_handler = QtLogHandler()
            self.qt_handler.setFormatter(simple_formatter)
            self.root_logger.addHandler(self.qt_handler)

        # Performance logging setup
        self.performance_logger = logging.getLogger('volume_slider.performance')

        if enable_file:
            perf_log_file = log_dir / f'volume_slider_performance_{datetime.now().strftime("%Y%m%d")}.log'
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(performance_formatter)
            perf_handler.addFilter(DataProcessingFilter())

            self.performance_logger.addHandler(perf_handler)
            self.performance_logger.setLevel(logging.INFO)

        # Add custom error tracking
        self._setup_error_tracking()

        # Log startup information
        self.root_logger.info("="*60)
        self.root_logger.info("Volume Slider Plugin - Logging System Initialized")
        self.root_logger.info(f"Log Level: {logging.getLevelName(log_level)}")
        self.root_logger.info(f"Log Directory: {log_dir}")
        self.root_logger.info(f"Python Version: {sys.version}")
        if HAS_QT:
            app = QApplication.instance()
            if app:
                self.root_logger.info(f"Qt Application: {app.applicationName()} {app.applicationVersion()}")
        self.root_logger.info("="*60)

        return self.root_logger

    def _setup_error_tracking(self):
        """Setup automatic error counting and tracking."""

        class ErrorTrackingHandler(logging.Handler):
            def __init__(self, manager):
                super().__init__()
                self.manager = manager

            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    self.manager.error_count += 1
                elif record.levelno >= logging.WARNING:
                    self.manager.warning_count += 1

        error_tracker = ErrorTrackingHandler(self)
        error_tracker.setLevel(logging.WARNING)
        self.root_logger.addHandler(error_tracker)

    def get_qt_handler(self) -> Optional[QtLogHandler]:
        """Get the Qt logging handler for GUI integration."""
        return self.qt_handler

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'uptime_formatted': f"{uptime/3600:.1f} hours",
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'log_directory': str(self.log_directory) if self.log_directory else None,
            'log_level': logging.getLevelName(self.root_logger.level) if self.root_logger else None
        }

    def create_session_report(self) -> str:
        """Create a session summary report."""
        stats = self.get_log_statistics()

        report = f"""
Volume Slider Plugin - Session Report
=====================================
Session Duration: {stats['uptime_formatted']}
Errors Encountered: {stats['error_count']}
Warnings Generated: {stats['warning_count']}
Log Directory: {stats['log_directory']}
Log Level: {stats['log_level']}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        if not self.log_directory:
            return

        cutoff_time = time.time() - (days_to_keep * 24 * 3600)
        cleaned_count = 0

        for log_file in self.log_directory.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_count += 1
                except OSError as e:
                    self.root_logger.warning(f"Could not delete old log file {log_file}: {e}")

        if cleaned_count > 0:
            self.root_logger.info(f"Cleaned up {cleaned_count} old log files")


# Global log manager instance
_log_manager = LogManager()


def setup_logging(**kwargs) -> logging.Logger:
    """
    Setup logging system with default configuration.

    This is the main entry point for logging setup.
    """
    return _log_manager.setup_logging(**kwargs)


def get_logger(name: str = 'volume_slider') -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)


def get_performance_logger() -> logging.Logger:
    """Get the performance monitoring logger."""
    if _log_manager.performance_logger:
        return _log_manager.performance_logger
    else:
        return get_logger('volume_slider.performance')


def get_qt_handler() -> Optional[QtLogHandler]:
    """Get the Qt logging handler for GUI integration."""
    return _log_manager.get_qt_handler()


@contextmanager
def log_performance(operation_name: str,
                   logger: Optional[logging.Logger] = None,
                   log_memory: bool = False):
    """
    Context manager for performance logging.

    Usage:
        with log_performance("data_loading"):
            # ... expensive operation ...
            pass
    """
    if logger is None:
        logger = get_performance_logger()

    start_time = time.perf_counter()
    start_memory = None

    if log_memory:
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            log_memory = False

    logger.info(f"Started: {operation_name}")

    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration = end_time - start_time

        # Create log record with custom attributes
        record = logger.makeRecord(
            logger.name, logging.INFO, '', 0,
            f"Completed: {operation_name}", (), None
        )
        record.duration = duration
        record.operation = operation_name

        if log_memory and start_memory is not None:
            try:
                import psutil
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                record.memory_usage = end_memory - start_memory
            except ImportError:
                pass

        logger.handle(record)


def log_exception(logger: Optional[logging.Logger] = None,
                 message: str = "An exception occurred"):
    """
    Log the current exception with full traceback.

    Usage:
        try:
            # ... code that might raise exception ...
            pass
        except Exception:
            log_exception(logger, "Failed to process data")
    """
    if logger is None:
        logger = get_logger()

    logger.error(f"{message}: {traceback.format_exc()}")


def log_data_operation(operation: str,
                      data_shape: tuple = None,
                      logger: Optional[logging.Logger] = None):
    """
    Log a data processing operation with context.

    Args:
        operation: Description of the operation
        data_shape: Shape of the data being processed
        logger: Logger to use (default: main logger)
    """
    if logger is None:
        logger = get_logger()

    record = logger.makeRecord(
        logger.name, logging.INFO, '', 0, operation, (), None
    )
    record.operation = operation
    if data_shape:
        record.data_shape = data_shape

    logger.handle(record)


def get_log_statistics() -> Dict[str, Any]:
    """Get current logging statistics."""
    return _log_manager.get_log_statistics()


def create_session_report() -> str:
    """Create a session summary report."""
    return _log_manager.create_session_report()


def cleanup_old_logs(days_to_keep: int = 30):
    """Clean up old log files."""
    _log_manager.cleanup_old_logs(days_to_keep)


# Convenience functions for common logging patterns
def log_memory_usage(logger: Optional[logging.Logger] = None):
    """Log current memory usage if psutil is available."""
    if logger is None:
        logger = get_logger()

    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"Current memory usage: {memory_mb:.1f} MB")
    except ImportError:
        logger.debug("psutil not available for memory monitoring")


def log_system_info(logger: Optional[logging.Logger] = None):
    """Log system information for debugging."""
    if logger is None:
        logger = get_logger()

    import platform

    logger.info(f"System: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Architecture: {platform.machine()}")

    try:
        import numpy as np
        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        import scipy
        logger.info(f"SciPy: {scipy.__version__}")
    except ImportError:
        pass

    if HAS_QT:
        from qtpy import PYQT_VERSION
        logger.info(f"Qt: {PYQT_VERSION}")


def configure_debug_logging():
    """Quick setup for debug-level logging during development."""
    return setup_logging(
        log_level='DEBUG',
        enable_console=True,
        enable_file=True,
        enable_gui=False
    )


def configure_production_logging():
    """Setup for production use with minimal console output."""
    return setup_logging(
        log_level='INFO',
        enable_console=False,
        enable_file=True,
        enable_gui=True
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the logging system
    logger = setup_logging(log_level='DEBUG')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test performance logging
    with log_performance("test_operation", log_memory=True):
        time.sleep(0.1)  # Simulate work
        import numpy as np
        data = np.random.random((1000, 1000))  # Simulate memory usage

    # Test data operation logging
    log_data_operation("Loading test data", (100, 50, 256, 256))

    # Test system info logging
    log_system_info()

    # Print session report
    print(create_session_report())
