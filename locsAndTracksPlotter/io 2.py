#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input/Output Module for FLIKA Tracking Plugin

This module provides file dialog utilities and file selection widgets
for the tracking analysis pipeline.

Created on Fri Jun  2 14:53:24 2023
@author: george
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import flika
from flika import global_vars as g
from qtpy.QtCore import Signal
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QFileDialog

# Set up logging
logger = logging.getLogger(__name__)


def open_file_gui(prompt: str = "Open File",
                  directory: Optional[str] = None,
                  filetypes: str = '') -> Optional[str]:
    """
    File dialog for opening an existing file with robust error handling.

    Args:
        prompt: String to display at the top of the window
        directory: Initial directory to open (None for default)
        filetypes: Filter for file types, format: "Images (*.png);;Other (*.*)"

    Returns:
        The selected file path, or None if cancelled/error

    Example:
        >>> filename = open_file_gui("Select CSV file", filetypes="CSV (*.csv)")
    """
    try:
        logger.debug(f"Opening file dialog with prompt: '{prompt}'")

        filename = None

        # Determine initial directory
        if directory is None:
            try:
                filename = g.settings.get('filename', '')
                if filename:
                    directory = os.path.dirname(filename)
            except (AttributeError, KeyError) as e:
                logger.warning(f"Could not get default filename from settings: {e}")
                directory = None

        # Fallback to user's home directory if no directory found
        if directory is None or not os.path.exists(directory):
            directory = str(Path.home())
            logger.debug(f"Using fallback directory: {directory}")

        # Open file dialog
        if filename and os.path.exists(filename):
            result = QFileDialog.getOpenFileName(g.m, prompt, filename, filetypes)
        else:
            result = QFileDialog.getOpenFileName(g.m, prompt, directory, filetypes)

        # Handle tuple return value from PyQt
        if isinstance(result, tuple):
            filename, selected_filter = result
            # Add extension if not present and filter specified
            if selected_filter and '.' not in os.path.basename(filename):
                extension = selected_filter.rsplit('*.')[-1].rstrip(')')
                if extension and extension != '*':
                    filename += '.' + extension
        else:
            filename = result

        # Validate result
        if not filename or str(filename).strip() == '':
            logger.info("No file selected by user")
            if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
                g.m.statusBar().showMessage('No File Selected')
            return None

        # Convert to string and validate path
        filename = str(filename)
        if not os.path.exists(filename):
            logger.warning(f"Selected file does not exist: {filename}")
            return None

        logger.info(f"File selected: {filename}")
        return filename

    except Exception as e:
        logger.error(f"Error in file dialog: {e}")
        if hasattr(g, 'm') and hasattr(g.m, 'statusBar'):
            g.m.statusBar().showMessage(f'Error opening file dialog: {str(e)}')
        return None


class FileSelector(QWidget):
    """
    Widget combining a button and label for file selection.

    This widget provides a button that opens a file dialog when clicked,
    and displays the selected filename in a label.

    Signals:
        valueChanged: Emitted when a file is selected
    """

    valueChanged = Signal()

    def __init__(self, filetypes: str = '*.*', mainGUI=None):
        """
        Initialize the FileSelector widget.

        Args:
            filetypes: File type filter string
            mainGUI: Reference to main GUI for accessing settings
        """
        super().__init__()

        self.mainGUI = mainGUI
        self.filetypes = filetypes
        self.filename = ''
        self.columns = []

        # Get pixel size from main GUI if available
        try:
            if mainGUI and hasattr(mainGUI, 'trackPlotOptions'):
                self.pixelSize = mainGUI.trackPlotOptions.pixelSize_selector.value()
            else:
                self.pixelSize = 108  # Default value in nanometers
        except Exception as e:
            logger.warning(f"Could not get pixel size from mainGUI: {e}")
            self.pixelSize = 108

        self._setup_ui()
        logger.debug("FileSelector initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        try:
            # Create widgets
            self.button = QPushButton('Load Data')
            self.label = QLabel('None')

            # Set up layout
            self.layout = QHBoxLayout()
            self.layout.addWidget(self.button)
            self.layout.addWidget(self.label)
            self.setLayout(self.layout)

            # Connect signals
            self.button.clicked.connect(self._on_button_clicked)

            # Style the label to show it's a filename
            font = QFont()
            font.setFamily("Courier")
            self.label.setFont(font)

        except Exception as e:
            logger.error(f"Error setting up FileSelector UI: {e}")

    def _on_button_clicked(self) -> None:
        """Handle button click to open file dialog."""
        try:
            # Check if main window exists
            if not hasattr(g, 'win') or g.win is None:
                error_msg = 'Load tiff stack and set as current window first'
                logger.warning(error_msg)
                if hasattr(g, 'alert'):
                    g.alert(error_msg)
                return

            # Open file dialog
            prompt = 'Select data file to load'
            selected_file = open_file_gui(prompt, filetypes=self.filetypes)

            if selected_file:
                self.filename = selected_file
                # Update label with truncated filename
                display_name = os.path.basename(self.filename)
                if len(display_name) > 20:
                    display_name = '...' + display_name[-20:]
                else:
                    display_name = '...' + display_name

                self.label.setText(display_name)
                self.label.setToolTip(self.filename)  # Show full path on hover

                logger.info(f"File selected: {self.filename}")
                self.valueChanged.emit()

        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            self.label.setText('Error')

    def value(self) -> str:
        """
        Get the currently selected filename.

        Returns:
            The selected filename path
        """
        return self.filename

    def setValue(self, filename: Union[str, Path]) -> None:
        """
        Set the filename programmatically.

        Args:
            filename: Path to the file to set
        """
        try:
            self.filename = str(filename)
            if self.filename:
                display_name = os.path.basename(self.filename)
                if len(display_name) > 20:
                    display_name = '...' + display_name[-20:]
                else:
                    display_name = '...' + display_name
                self.label.setText(display_name)
                self.label.setToolTip(self.filename)
                logger.debug(f"Filename set to: {self.filename}")
            else:
                self.label.setText('None')
                self.label.setToolTip('')
        except Exception as e:
            logger.error(f"Error setting filename: {e}")
            self.label.setText('Error')


class FileSelector_overlay(QWidget):
    """
    Specialized file selector for overlay TIFF files.

    Similar to FileSelector but specifically designed for loading
    overlay images in TIFF format.

    Signals:
        valueChanged: Emitted when a file is selected
    """

    valueChanged = Signal()

    def __init__(self, filetypes: str = '*.*'):
        """
        Initialize the overlay file selector.

        Args:
            filetypes: File type filter string (default: all files)
        """
        super().__init__()

        self.filetypes = filetypes
        self.filename = ''

        self._setup_ui()
        logger.debug("FileSelector_overlay initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        try:
            # Create widgets
            self.button = QPushButton('Load Tiff')
            self.label = QLabel('None')

            # Set up layout
            self.layout = QHBoxLayout()
            self.layout.addWidget(self.button)
            self.layout.addWidget(self.label)
            self.setLayout(self.layout)

            # Connect signals
            self.button.clicked.connect(self._on_button_clicked)

            # Style the label
            font = QFont()
            font.setFamily("Courier")
            self.label.setFont(font)

        except Exception as e:
            logger.error(f"Error setting up FileSelector_overlay UI: {e}")

    def _on_button_clicked(self) -> None:
        """Handle button click to open file dialog."""
        try:
            prompt = 'Select TIFF file to overlay'
            selected_file = open_file_gui(prompt, filetypes=self.filetypes)

            if selected_file:
                self.filename = selected_file
                # Update label with truncated filename
                display_name = os.path.basename(self.filename)
                if len(display_name) > 20:
                    display_name = '...' + display_name[-20:]
                else:
                    display_name = '...' + display_name

                self.label.setText(display_name)
                self.label.setToolTip(self.filename)

                logger.info(f"Overlay file selected: {self.filename}")
                self.valueChanged.emit()

        except Exception as e:
            logger.error(f"Error in overlay file selection: {e}")
            self.label.setText('Error')

    def value(self) -> str:
        """
        Get the currently selected filename.

        Returns:
            The selected filename path
        """
        return self.filename

    def setValue(self, filename: Union[str, Path]) -> None:
        """
        Set the filename programmatically.

        Args:
            filename: Path to the file to set
        """
        try:
            self.filename = str(filename)
            if self.filename:
                display_name = os.path.basename(self.filename)
                if len(display_name) > 20:
                    display_name = '...' + display_name[-20:]
                else:
                    display_name = '...' + display_name
                self.label.setText(display_name)
                self.label.setToolTip(self.filename)
                logger.debug(f"Overlay filename set to: {self.filename}")
            else:
                self.label.setText('None')
                self.label.setToolTip('')
        except Exception as e:
            logger.error(f"Error setting overlay filename: {e}")
            self.label.setText('Error')


def validate_file_path(filepath: Union[str, Path]) -> bool:
    """
    Validate that a file path exists and is readable.

    Args:
        filepath: Path to validate

    Returns:
        True if file exists and is readable, False otherwise
    """
    try:
        path = Path(filepath)
        return path.exists() and path.is_file() and os.access(path, os.R_OK)
    except Exception as e:
        logger.error(f"Error validating file path {filepath}: {e}")
        return False


def get_file_extension(filepath: Union[str, Path]) -> str:
    """
    Get the file extension from a path.

    Args:
        filepath: Path to extract extension from

    Returns:
        File extension (without dot) or empty string if none
    """
    try:
        return Path(filepath).suffix.lstrip('.')
    except Exception as e:
        logger.error(f"Error getting file extension from {filepath}: {e}")
        return ''
