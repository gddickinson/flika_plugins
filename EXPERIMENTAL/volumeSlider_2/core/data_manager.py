#!/usr/bin/env python3
"""
Core Data Manager Module
========================

Centralized data management for the Volume Slider plugin.
Handles raw data, processed data, metadata, and data transformations.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from pathlib import Path
import json
from datetime import datetime
from qtpy.QtCore import QObject, Signal


class DataManager(QObject):
    """
    Centralized data management system for volume data.

    Features:
    - Raw and processed data storage
    - Metadata management
    - Data validation and type conversion
    - Memory-efficient volume access
    - Automatic data statistics calculation
    - Change notifications via signals
    """

    # Signals for data state changes
    data_changed = Signal()
    metadata_changed = Signal()
    processing_completed = Signal(str)  # Processing step name

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Data storage
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[np.ndarray] = None
        self._overlay_data: Optional[np.ndarray] = None

        # Metadata
        self._metadata: Dict[str, Any] = {}
        self._processing_history: List[Dict[str, Any]] = []

        # Data properties
        self._data_stats: Optional[Dict[str, Any]] = None
        self._volume_count: int = 0
        self._current_dtype = np.float32

        # Memory management
        self._memory_threshold = 2 * 1024**3  # 2GB threshold
        self._use_memory_mapping = False

        self.logger.info("DataManager initialized")

    @property
    def raw_data(self) -> Optional[np.ndarray]:
        """Get raw data."""
        return self._raw_data

    @property
    def processed_data(self) -> Optional[np.ndarray]:
        """Get processed data."""
        return self._processed_data if self._processed_data is not None else self._raw_data

    @property
    def has_data(self) -> bool:
        """Check if any data is loaded."""
        return self._raw_data is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return self._metadata.copy()

    @property
    def data_statistics(self) -> Optional[Dict[str, Any]]:
        """Get data statistics."""
        return self._data_stats.copy() if self._data_stats else None

    @property
    def processing_history(self) -> List[Dict[str, Any]]:
        """Get processing history."""
        return self._processing_history.copy()

    def set_data(self, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Set raw data with validation and metadata.

        Args:
            data: Input data array
            metadata: Optional metadata dictionary

        Returns:
            bool: Success status
        """
        try:
            # Validate input data
            if not isinstance(data, np.ndarray):
                raise TypeError("Data must be a numpy array")

            if data.size == 0:
                raise ValueError("Data array is empty")

            if data.ndim < 2 or data.ndim > 4:
                raise ValueError(f"Data must be 2D-4D, got {data.ndim}D")

            # Memory management
            memory_needed = data.nbytes
            if memory_needed > self._memory_threshold:
                self.logger.warning(f"Large dataset ({memory_needed/1024**3:.2f} GB), "
                                  "consider using memory mapping")
                self._use_memory_mapping = True

            # Store data
            self._raw_data = self._ensure_4d(data.astype(self._current_dtype))
            self._processed_data = None  # Reset processed data

            # Update metadata
            self._update_metadata(metadata or {})

            # Calculate statistics
            self._calculate_statistics()

            # Update volume count
            self._volume_count = self._raw_data.shape[1] if self._raw_data.ndim == 4 else 1

            # Log success
            self.logger.info(f"Data set successfully: shape={self._raw_data.shape}, "
                           f"dtype={self._raw_data.dtype}, size={memory_needed/1024**2:.1f}MB")

            # Emit signals
            self.data_changed.emit()
            self.metadata_changed.emit()

            return True

        except Exception as e:
            self.logger.error(f"Failed to set data: {str(e)}")
            return False

    def set_processed_data(self, data: np.ndarray, processing_step: str) -> bool:
        """
        Set processed data and record processing history.

        Args:
            data: Processed data array
            processing_step: Name of the processing step

        Returns:
            bool: Success status
        """
        try:
            if not isinstance(data, np.ndarray):
                raise TypeError("Processed data must be a numpy array")

            # Ensure compatibility with raw data
            if self._raw_data is not None and data.shape != self._raw_data.shape:
                self.logger.warning(f"Processed data shape {data.shape} differs from "
                                  f"raw data shape {self._raw_data.shape}")

            self._processed_data = data.astype(self._current_dtype)

            # Record processing history
            self._add_processing_step(processing_step)

            # Recalculate statistics for processed data
            self._calculate_statistics(use_processed=True)

            self.logger.info(f"Processed data set: step={processing_step}")

            # Emit signals
            self.data_changed.emit()
            self.processing_completed.emit(processing_step)

            return True

        except Exception as e:
            self.logger.error(f"Failed to set processed data: {str(e)}")
            return False

    def set_overlay_data(self, overlay_data: np.ndarray) -> bool:
        """
        Set overlay data for visualization.

        Args:
            overlay_data: Overlay data array

        Returns:
            bool: Success status
        """
        try:
            if not isinstance(overlay_data, np.ndarray):
                raise TypeError("Overlay data must be a numpy array")

            self._overlay_data = self._ensure_4d(overlay_data.astype(self._current_dtype))

            self.logger.info(f"Overlay data set: shape={self._overlay_data.shape}")
            self.data_changed.emit()

            return True

        except Exception as e:
            self.logger.error(f"Failed to set overlay data: {str(e)}")
            return False

    def get_current_volume_data(self, volume_index: int) -> Optional[np.ndarray]:
        """
        Get data for specific volume index.

        Args:
            volume_index: Volume index to retrieve

        Returns:
            numpy array or None if invalid index
        """
        try:
            active_data = self.processed_data
            if active_data is None:
                return None

            if active_data.ndim == 3:
                return active_data  # Single volume
            elif active_data.ndim == 4:
                if 0 <= volume_index < active_data.shape[1]:
                    return active_data[:, volume_index, :, :]
                else:
                    self.logger.warning(f"Invalid volume index {volume_index}, "
                                      f"max is {active_data.shape[1]-1}")
                    return None
            else:
                self.logger.error(f"Unexpected data dimensionality: {active_data.ndim}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting volume data: {str(e)}")
            return None

    def get_slice_data(self, axis: int, slice_index: int,
                      volume_index: int = 0) -> Optional[np.ndarray]:
        """
        Get 2D slice from volume data.

        Args:
            axis: Axis to slice along (0=Z, 1=Y, 2=X)
            slice_index: Index along the axis
            volume_index: Volume index for 4D data

        Returns:
            2D numpy array or None if invalid
        """
        try:
            volume_data = self.get_current_volume_data(volume_index)
            if volume_data is None:
                return None

            if axis == 0:  # Z axis (depth)
                if 0 <= slice_index < volume_data.shape[0]:
                    return volume_data[slice_index, :, :]
            elif axis == 1:  # Y axis
                if 0 <= slice_index < volume_data.shape[1]:
                    return volume_data[:, slice_index, :]
            elif axis == 2:  # X axis
                if 0 <= slice_index < volume_data.shape[2]:
                    return volume_data[:, :, slice_index]
            else:
                raise ValueError(f"Invalid axis {axis}, must be 0, 1, or 2")

            self.logger.warning(f"Invalid slice index {slice_index} for axis {axis}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting slice data: {str(e)}")
            return None

    def get_projection(self, axis: int, method: str = 'max',
                      volume_index: int = 0) -> Optional[np.ndarray]:
        """
        Get projection along specified axis.

        Args:
            axis: Axis to project along (0=Z, 1=Y, 2=X)
            method: Projection method ('max', 'mean', 'sum', 'std')
            volume_index: Volume index for 4D data

        Returns:
            2D projection array or None if invalid
        """
        try:
            volume_data = self.get_current_volume_data(volume_index)
            if volume_data is None:
                return None

            projection_methods = {
                'max': np.max,
                'mean': np.mean,
                'sum': np.sum,
                'std': np.std,
                'min': np.min
            }

            if method not in projection_methods:
                raise ValueError(f"Invalid projection method '{method}'. "
                               f"Available: {list(projection_methods.keys())}")

            func = projection_methods[method]
            projection = func(volume_data, axis=axis)

            return projection

        except Exception as e:
            self.logger.error(f"Error calculating projection: {str(e)}")
            return None

    def get_volume_count(self) -> int:
        """Get number of volumes in the dataset."""
        return self._volume_count

    def get_data_shape(self) -> Optional[Tuple[int, ...]]:
        """Get shape of current data."""
        active_data = self.processed_data
        return active_data.shape if active_data is not None else None

    def get_data_dtype(self) -> Optional[np.dtype]:
        """Get data type of current data."""
        active_data = self.processed_data
        return active_data.dtype if active_data is not None else None

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics in MB."""
        usage = {'raw': 0.0, 'processed': 0.0, 'overlay': 0.0, 'total': 0.0}

        if self._raw_data is not None:
            usage['raw'] = self._raw_data.nbytes / 1024**2

        if self._processed_data is not None:
            usage['processed'] = self._processed_data.nbytes / 1024**2

        if self._overlay_data is not None:
            usage['overlay'] = self._overlay_data.nbytes / 1024**2

        usage['total'] = usage['raw'] + usage['processed'] + usage['overlay']

        return usage

    def clear_data(self, data_type: str = 'all'):
        """
        Clear specified data from memory.

        Args:
            data_type: Type to clear ('all', 'raw', 'processed', 'overlay')
        """
        try:
            if data_type in ('all', 'raw'):
                self._raw_data = None
                self.logger.info("Raw data cleared")

            if data_type in ('all', 'processed'):
                self._processed_data = None
                self.logger.info("Processed data cleared")

            if data_type in ('all', 'overlay'):
                self._overlay_data = None
                self.logger.info("Overlay data cleared")

            if data_type == 'all':
                self._metadata.clear()
                self._processing_history.clear()
                self._data_stats = None
                self._volume_count = 0

            self.data_changed.emit()

        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")

    def save_processed_data(self, filepath: Union[str, Path]):
        """
        Save processed data to file.

        Args:
            filepath: Output file path
        """
        try:
            if self._processed_data is None:
                raise ValueError("No processed data to save")

            filepath = Path(filepath)

            if filepath.suffix.lower() in ['.npy']:
                np.save(filepath, self._processed_data)
            elif filepath.suffix.lower() in ['.tif', '.tiff']:
                import tifffile
                tifffile.imsave(filepath, self._processed_data)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")

            # Save metadata alongside
            metadata_path = filepath.with_suffix('.json')
            self._save_metadata(metadata_path)

            self.logger.info(f"Processed data saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save processed data: {str(e)}")
            raise

    def export_metadata(self, filepath: Union[str, Path]):
        """Export metadata to JSON file."""
        try:
            self._save_metadata(Path(filepath))
            self.logger.info(f"Metadata exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metadata: {str(e)}")
            raise

    def _ensure_4d(self, data: np.ndarray) -> np.ndarray:
        """Ensure data is 4D (Z, T, Y, X) format."""
        if data.ndim == 2:
            # (Y, X) -> (1, 1, Y, X)
            return data[np.newaxis, np.newaxis, :, :]
        elif data.ndim == 3:
            # Assume (Z, Y, X) -> (Z, 1, Y, X)
            return data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            return data
        else:
            raise ValueError(f"Cannot convert {data.ndim}D data to 4D")

    def _update_metadata(self, new_metadata: Dict[str, Any]):
        """Update metadata with data properties."""
        self._metadata.update(new_metadata)

        # Add automatic metadata
        if self._raw_data is not None:
            self._metadata.update({
                'data_shape': self._raw_data.shape,
                'data_dtype': str(self._raw_data.dtype),
                'data_size_mb': self._raw_data.nbytes / 1024**2,
                'volume_count': self._volume_count,
                'timestamp': datetime.now().isoformat(),
                'plugin_version': '2.0.0'
            })

    def _calculate_statistics(self, use_processed: bool = False):
        """Calculate comprehensive data statistics."""
        try:
            data = self._processed_data if use_processed and self._processed_data is not None else self._raw_data

            if data is None:
                return

            self._data_stats = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'percentile_1': float(np.percentile(data, 1)),
                'percentile_99': float(np.percentile(data, 99)),
                'non_zero_fraction': float(np.count_nonzero(data) / data.size),
                'data_range': float(np.max(data) - np.min(data))
            }

            self.logger.debug(f"Statistics calculated: mean={self._data_stats['mean']:.2f}, "
                            f"std={self._data_stats['std']:.2f}")

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            self._data_stats = None

    def _add_processing_step(self, step_name: str):
        """Add processing step to history."""
        step_info = {
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'data_shape': self._processed_data.shape if self._processed_data is not None else None,
            'data_dtype': str(self._processed_data.dtype) if self._processed_data is not None else None
        }

        self._processing_history.append(step_info)
        self._metadata['processing_history'] = self._processing_history

    def _save_metadata(self, filepath: Path):
        """Save metadata to JSON file."""
        try:
            # Prepare metadata for JSON serialization
            export_metadata = self._metadata.copy()
            export_metadata['data_statistics'] = self._data_stats
            export_metadata['memory_usage'] = self.get_memory_usage()

            with open(filepath, 'w') as f:
                json.dump(export_metadata, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save metadata: {str(e)}")
            raise


class DataValidator:
    """Utility class for data validation."""

    @staticmethod
    def validate_volume_data(data: np.ndarray) -> Tuple[bool, str]:
        """
        Validate volume data for compatibility.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not isinstance(data, np.ndarray):
                return False, "Data must be numpy array"

            if data.size == 0:
                return False, "Data array is empty"

            if data.ndim < 2 or data.ndim > 4:
                return False, f"Data must be 2D-4D, got {data.ndim}D"

            if not np.issubdtype(data.dtype, np.number):
                return False, f"Data must be numeric, got {data.dtype}"

            if np.any(np.isnan(data)):
                return False, "Data contains NaN values"

            if np.any(np.isinf(data)):
                return False, "Data contains infinite values"

            # Check reasonable size limits
            if data.nbytes > 8 * 1024**3:  # 8GB
                return False, f"Data too large ({data.nbytes/1024**3:.1f}GB), max 8GB"

            return True, "Data is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    @staticmethod
    def suggest_data_type_conversion(data: np.ndarray) -> np.dtype:
        """Suggest optimal data type for given data."""
        if data.dtype == np.float64:
            return np.float32  # Usually sufficient precision
        elif data.dtype in [np.int64, np.int32]:
            if np.min(data) >= 0 and np.max(data) <= 65535:
                return np.uint16
            else:
                return np.int32
        else:
            return data.dtype  # Keep original type
