#!/usr/bin/env python3
"""
Error Handler and Logging Utilities
===================================

Centralized error handling, logging configuration, and debugging utilities
for the Volume Slider plugin.
"""

import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, TextIO
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings
from qtpy.QtWidgets import QMessageBox, QApplication
from qtpy.QtCore import QObject, Signal


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class ErrorInfo:
    """Information about an error."""
    error_type: str
    message: str
    traceback: Optional[str] = None
    timestamp: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = None
    user_action: Optional[str] = None


class ErrorHandler(QObject):
    """
    Centralized error handling system.

    Features:
    - Structured error logging
    - User-friendly error dialogs
    - Error recovery suggestions
    - Error reporting and analytics
    - Integration with Qt message boxes
    """

    # Signals for error events
    error_occurred = Signal(str, str, str)  # title, message, details
    warning_occurred = Signal(str, str)     # title, message
    info_message = Signal(str)              # message

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = logger or logging.getLogger(__name__)

        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.max_history = 100

        # User interaction
        self.show_dialogs = True
        self.auto_report = False

        # Recovery suggestions
        self.recovery_suggestions = {
            'FileNotFoundError': [
                "Check if the file path is correct",
                "Verify the file exists and is accessible",
                "Try selecting the file using the file browser"
            ],
            'MemoryError': [
                "Close other applications to free memory",
                "Try processing smaller data chunks",
                "Restart the application"
            ],
            'ValueError': [
                "Check input parameters are valid",
                "Verify data format is correct",
                "Try different parameter values"
            ],
            'ImportError': [
                "Check if required packages are installed",
                "Verify Python environment is correct",
                "Try reinstalling dependencies"
            ]
        }

    def handle_error(self, title: str, message: str, error: Optional[Exception] = None,
                    context: Optional[Dict[str, Any]] = None,
                    show_dialog: bool = True) -> None:
        """
        Handle an error with comprehensive logging and user notification.

        Args:
            title: Error title/category
            message: Human-readable error message
            error: Optional exception object
            context: Additional context information
            show_dialog: Whether to show error dialog to user
        """
        try:
            # Create error info
            error_info = ErrorInfo(
                error_type=type(error).__name__ if error else 'UnknownError',
                message=message,
                traceback=self._format_traceback(error) if error else None,
                timestamp=datetime.now(),
                context=context or {},
                user_action=title
            )

            # Add to history
            self._add_to_history(error_info)

            # Log the error
            self._log_error(error_info)

            # Show user dialog if requested
            if show_dialog and self.show_dialogs:
                self._show_error_dialog(title, message, error_info)

            # Emit signal
            self.error_occurred.emit(
                title,
                message,
                error_info.traceback or "No traceback available"
            )

        except Exception as handler_error:
            # Fallback - don't let error handler itself crash
            fallback_msg = f"Error in error handler: {str(handler_error)}"
            self.logger.critical(fallback_msg)
            print(f"CRITICAL ERROR: {fallback_msg}", file=sys.stderr)

    def handle_warning(self, title: str, message: str,
                      show_dialog: bool = False) -> None:
        """Handle a warning."""
        try:
            self.logger.warning(f"{title}: {message}")

            if show_dialog and self.show_dialogs:
                self._show_warning_dialog(title, message)

            self.warning_occurred.emit(title, message)

        except Exception as e:
            self.logger.error(f"Error in warning handler: {str(e)}")

    def handle_info(self, message: str, show_dialog: bool = False) -> None:
        """Handle an informational message."""
        try:
            self.logger.info(message)

            if show_dialog and self.show_dialogs:
                self._show_info_dialog(message)

            self.info_message.emit(message)

        except Exception as e:
            self.logger.error(f"Error in info handler: {str(e)}")

    def _format_traceback(self, error: Exception) -> str:
        """Format exception traceback."""
        if error is None:
            return "No exception provided"

        return ''.join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))

    def _add_to_history(self, error_info: ErrorInfo) -> None:
        """Add error to history with size limit."""
        self.error_history.append(error_info)

        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with full details."""
        log_message = f"{error_info.user_action}: {error_info.message}"

        if error_info.context:
            context_str = ", ".join(f"{k}={v}" for k, v in error_info.context.items())
            log_message += f" [Context: {context_str}]"

        self.logger.error(log_message)

        if error_info.traceback:
            self.logger.debug(f"Traceback: {error_info.traceback}")

    def _show_error_dialog(self, title: str, message: str, error_info: ErrorInfo) -> None:
        """Show error dialog to user."""
        try:
            # Create detailed message with recovery suggestions
            detailed_message = message

            # Add recovery suggestions if available
            suggestions = self.recovery_suggestions.get(error_info.error_type, [])
            if suggestions:
                detailed_message += "\n\nSuggested solutions:\n"
                detailed_message += "\n".join(f"• {suggestion}" for suggestion in suggestions)

            # Create message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle(f"Error: {title}")
            msg_box.setText(message)

            if len(detailed_message) > len(message):
                msg_box.setDetailedText(detailed_message)

            if error_info.traceback:
                msg_box.setDetailedText(
                    (msg_box.detailedText() + "\n\nTechnical Details:\n" +
                     error_info.traceback)
                )

            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()

        except Exception as e:
            # Fallback to simple print if Qt dialogs fail
            print(f"ERROR: {title} - {message}", file=sys.stderr)

    def _show_warning_dialog(self, title: str, message: str) -> None:
        """Show warning dialog to user."""
        try:
            QMessageBox.warning(None, f"Warning: {title}", message)
        except Exception as e:
            print(f"WARNING: {title} - {message}", file=sys.stderr)

    def _show_info_dialog(self, message: str) -> None:
        """Show info dialog to user."""
        try:
            QMessageBox.information(None, "Information", message)
        except Exception as e:
            print(f"INFO: {message}")

    def get_error_history(self) -> List[ErrorInfo]:
        """Get error history."""
        return self.error_history.copy()

    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()
        self.logger.info("Error history cleared")

    def export_error_log(self, filepath: str) -> None:
        """Export error history to file."""
        try:
            with open(filepath, 'w') as f:
                f.write(f"Error Log Export - {datetime.now().isoformat()}\n")
                f.write("=" * 50 + "\n\n")

                for i, error_info in enumerate(self.error_history):
                    f.write(f"Error #{i+1}:\n")
                    f.write(f"  Timestamp: {error_info.timestamp}\n")
                    f.write(f"  Type: {error_info.error_type}\n")
                    f.write(f"  Action: {error_info.user_action}\n")
                    f.write(f"  Message: {error_info.message}\n")

                    if error_info.context:
                        f.write(f"  Context: {error_info.context}\n")

                    if error_info.traceback:
                        f.write(f"  Traceback:\n{error_info.traceback}\n")

                    f.write("\n" + "-" * 40 + "\n\n")

            self.logger.info(f"Error log exported to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to export error log: {str(e)}")

    def set_dialog_mode(self, show_dialogs: bool) -> None:
        """Enable or disable error dialogs."""
        self.show_dialogs = show_dialogs
        self.logger.info(f"Error dialogs {'enabled' if show_dialogs else 'disabled'}")


def setup_logging(log_level: LogLevel = LogLevel.INFO,
                 log_file: Optional[str] = None,
                 console_output: bool = True,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup comprehensive logging for the plugin.

    Args:
        log_level: Minimum logging level
        log_file: Optional log file path
        console_output: Whether to log to console
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('volume_slider_plugin')
    logger.setLevel(log_level.value)

    # Clear existing handlers
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    formatter = logging.Formatter(format_string)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level.value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        try:
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level.value)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        except Exception as e:
            print(f"Failed to setup file logging: {str(e)}", file=sys.stderr)

    # Capture warnings
    logging.captureWarnings(True)

    logger.info("Logging system initialized")
    return logger


class DebugContext:
    """Context manager for debugging sections of code."""

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time

        if exc_type is None:
            self.logger.debug(f"Operation completed: {self.operation_name} "
                            f"(took {duration.total_seconds():.3f}s)")
        else:
            self.logger.error(f"Operation failed: {self.operation_name} "
                            f"(after {duration.total_seconds():.3f}s) - "
                            f"{exc_type.__name__}: {exc_val}")

        # Don't suppress exceptions
        return False


def log_performance(func: Callable) -> Callable:
    """Decorator to log function performance."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            duration = datetime.now() - start_time
            logger.debug(f"{func.__name__} completed in {duration.total_seconds():.3f}s")
            return result

        except Exception as e:
            duration = datetime.now() - start_time
            logger.error(f"{func.__name__} failed after {duration.total_seconds():.3f}s: {str(e)}")
            raise

    return wrapper


def safe_execute(func: Callable, error_handler: ErrorHandler,
                operation_name: str, *args, **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        error_handler: ErrorHandler instance
        operation_name: Name of operation for error reporting
        *args, **kwargs: Arguments for the function

    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)

    except Exception as e:
        error_handler.handle_error(
            title=operation_name,
            message=f"Operation '{operation_name}' failed: {str(e)}",
            error=e,
            context={'function': func.__name__, 'args': str(args)[:100]}
        )
        return None


class MemoryMonitor:
    """Monitor memory usage and warn about potential issues."""

    def __init__(self, logger: logging.Logger, warning_threshold_mb: float = 1000.0):
        self.logger = logger
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes

    def check_memory_usage(self, context: str = "") -> Dict[str, float]:
        """Check current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            memory_mb = memory_info.rss / 1024 / 1024

            if memory_mb > self.warning_threshold / 1024 / 1024:
                self.logger.warning(f"High memory usage: {memory_mb:.1f} MB {context}")

            return {
                'memory_mb': memory_mb,
                'memory_percent': process.memory_percent()
            }

        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return {}

        except Exception as e:
            self.logger.error(f"Memory monitoring failed: {str(e)}")
            return {}


def create_default_error_handler() -> ErrorHandler:
    """Create error handler with default configuration."""
    # Setup default logging
    log_file = Path.home() / ".flika" / "plugins" / "volume_slider" / "error.log"
    logger = setup_logging(
        log_level=LogLevel.INFO,
        log_file=str(log_file),
        console_output=True
    )

    return ErrorHandler(logger)


def handle_qt_exceptions():
    """Setup Qt exception handling."""
    def qt_exception_hook(exc_type, exc_value, exc_traceback):
        """Handle uncaught Qt exceptions."""
        logger = logging.getLogger('qt_exceptions')

        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"Uncaught Qt exception: {error_msg}")

        # Show critical error dialog
        if QApplication.instance():
            QMessageBox.critical(
                None,
                "Critical Error",
                f"An unhandled error occurred:\n\n{exc_type.__name__}: {exc_value}\n\n"
                "Please check the log files for details."
            )

        # Call original hook
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    # Install exception hook
    sys.excepthook = qt_exception_hook


class ValidationError(Exception):
    """Custom exception for data validation errors."""

    def __init__(self, message: str, field: Optional[str] = None,
                 value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ProcessingError(Exception):
    """Custom exception for data processing errors."""

    def __init__(self, message: str, operation: Optional[str] = None,
                 data_shape: Optional[tuple] = None):
        self.operation = operation
        self.data_shape = data_shape
        super().__init__(message)


class VisualizationError(Exception):
    """Custom exception for visualization errors."""

    def __init__(self, message: str, render_mode: Optional[str] = None,
                 data_points: Optional[int] = None):
        self.render_mode = render_mode
        self.data_points = data_points
        super().__init__(message)


def setup_error_handling() -> ErrorHandler:
    """Setup complete error handling system."""
    # Setup Qt exception handling
    handle_qt_exceptions()

    # Create error handler
    error_handler = create_default_error_handler()

    # Setup warning filters
    warnings.filterwarnings('always', category=UserWarning)
    warnings.filterwarnings('always', category=RuntimeWarning)

    return error_handler


if __name__ == "__main__":
    # Test the error handling system
    error_handler = setup_error_handling()

    # Test various error types
    try:
        raise ValueError("Test validation error")
    except ValueError as e:
        error_handler.handle_error("Test Error", "This is a test error", e)

    error_handler.handle_warning("Test Warning", "This is a test warning")
    error_handler.handle_info("Test completed successfully")

    # Test performance logging
    @log_performance
    def test_function():
        import time
        time.sleep(0.1)
        return "success"

    result = test_function()

    print(f"Error handler test completed. History contains {len(error_handler.get_error_history())} errors.")
