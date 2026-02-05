#!/usr/bin/env python3
"""
Data Loaders Module
==================

Comprehensive data loading system for various microscopy file formats.
Handles TIFF stacks, NumPy arrays, HDF5 files, and specialized formats
with automatic metadata extraction and axis ordering detection.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from enum import Enum
import re
import json
from abc import ABC, abstractmethod

# Optional imports with fallbacks
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    tifffile = None

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    from nd2reader import ND2Reader
    HAS_ND2 = True
except ImportError:
    HAS_ND2 = False
    ND2Reader = None

try:
    import czifile
    HAS_CZI = True
except ImportError:
    HAS_CZI = False
    czifile = None

from qtpy.QtCore import QObject, Signal


class AxisOrder(Enum):
    """Standard axis ordering conventions."""
    TZCYX = "TZCYX"  # Time, Z, Channel, Y, X
    ZCYX = "ZCYX"    # Z, Channel, Y, X
    TZYX = "TZYX"    # Time, Z, Y, X
    ZYX = "ZYX"      # Z, Y, X
    CZYX = "CZYX"    # Channel, Z, Y, X
    YX = "YX"        # Y, X (single image)
    TYX = "TYX"      # Time, Y, X (2D time series)
    CYX = "CYX"      # Channel, Y, X (multichannel 2D)


@dataclass
class LoadedData:
    """Container for loaded data with metadata."""
    data: np.ndarray
    metadata: Dict[str, Any]
    original_shape: Tuple[int, ...]
    axis_order: str
    file_path: Union[str, Path]
    loader_type: str

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size_mb(self) -> float:
        return self.data.nbytes / (1024 ** 2)


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if this loader can handle the file."""
        pass

    @abstractmethod
    def load(self, filepath: Union[str, Path],
             progress_callback: Optional[Callable[[int, str], None]] = None) -> LoadedData:
        """Load data from file."""
        pass

    @abstractmethod
    def get_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get file information without loading data."""
        pass

    def _update_progress(self, callback: Optional[Callable], percent: int, message: str):
        """Helper to update progress if callback provided."""
        if callback:
            callback(percent, message)


class TiffLoader(BaseDataLoader):
    """Loader for TIFF files with comprehensive metadata extraction."""

    SUPPORTED_EXTENSIONS = {'.tif', '.tiff', '.tiff'}

    def __init__(self):
        super().__init__()
        if not HAS_TIFFFILE:
            raise ImportError("tifffile package required for TIFF loading")

    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if file is a supported TIFF."""
        path = Path(filepath)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS and path.exists()

    def load(self, filepath: Union[str, Path],
             progress_callback: Optional[Callable[[int, str], None]] = None) -> LoadedData:
        """Load TIFF file with automatic axis detection."""

        filepath = Path(filepath)
        self.logger.info(f"Loading TIFF file: {filepath}")

        self._update_progress(progress_callback, 0, "Opening TIFF file...")

        try:
            with tifffile.TiffFile(str(filepath)) as tif:
                # Extract metadata
                metadata = self._extract_tiff_metadata(tif)

                # Detect axis order
                axis_order = self._detect_axis_order(tif, metadata)

                self._update_progress(progress_callback, 20, "Reading image data...")

                # Load the actual data
                data = tif.asarray()
                original_shape = data.shape

                self._update_progress(progress_callback, 60, "Processing axis order...")

                # Standardize axis order if needed
                data, final_axis_order = self._standardize_axes(data, axis_order, metadata)

                self._update_progress(progress_callback, 80, "Finalizing metadata...")

                # Update metadata with processed information
                metadata.update({
                    'original_shape': original_shape,
                    'original_axis_order': axis_order,
                    'standardized_axis_order': final_axis_order,
                    'processed_shape': data.shape,
                    'file_size_bytes': filepath.stat().st_size,
                    'data_size_bytes': data.nbytes
                })

                self._update_progress(progress_callback, 100, "Loading complete")

                return LoadedData(
                    data=data,
                    metadata=metadata,
                    original_shape=original_shape,
                    axis_order=final_axis_order,
                    file_path=filepath,
                    loader_type="TIFF"
                )

        except Exception as e:
            self.logger.error(f"Failed to load TIFF file {filepath}: {str(e)}")
            raise RuntimeError(f"TIFF loading failed: {str(e)}")

    def get_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get TIFF file information without loading data."""
        filepath = Path(filepath)

        try:
            with tifffile.TiffFile(str(filepath)) as tif:
                metadata = self._extract_tiff_metadata(tif)
                axis_order = self._detect_axis_order(tif, metadata)

                # Get basic info without loading full data
                series = tif.series[0]

                info = {
                    'file_path': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024**2),
                    'shape': series.shape,
                    'dtype': series.dtype,
                    'axis_order': axis_order,
                    'estimated_memory_mb': (series.shape[0] if series.shape else 0) *
                                         np.dtype(series.dtype).itemsize / (1024**2),
                    'n_pages': len(tif.pages),
                    'metadata': metadata
                }

                return info

        except Exception as e:
            self.logger.error(f"Failed to get TIFF info for {filepath}: {str(e)}")
            return {'error': str(e)}

    def _extract_tiff_metadata(self, tif: 'tifffile.TiffFile') -> Dict[str, Any]:
        """Extract comprehensive metadata from TIFF file."""
        metadata = {
            'software': None,
            'datetime': None,
            'description': None,
            'pixel_size': {},
            'acquisition_params': {},
            'custom_tags': {}
        }

        try:
            # Extract from first page
            if tif.pages:
                page = tif.pages[0]
                tags = page.tags

                # Standard TIFF tags
                if 'Software' in tags:
                    metadata['software'] = tags['Software'].value

                if 'DateTime' in tags:
                    metadata['datetime'] = tags['DateTime'].value

                if 'ImageDescription' in tags:
                    desc = tags['ImageDescription'].value
                    metadata['description'] = desc

                    # Try to parse JSON in description
                    if desc and desc.strip().startswith('{'):
                        try:
                            json_meta = json.loads(desc)
                            metadata['acquisition_params'].update(json_meta)
                        except json.JSONDecodeError:
                            pass

                # Pixel size information
                if 'XResolution' in tags and 'YResolution' in tags:
                    x_res = tags['XResolution'].value
                    y_res = tags['YResolution'].value
                    if isinstance(x_res, (list, tuple)) and len(x_res) == 2:
                        metadata['pixel_size']['x'] = x_res[1] / x_res[0] if x_res[0] != 0 else None
                    if isinstance(y_res, (list, tuple)) and len(y_res) == 2:
                        metadata['pixel_size']['y'] = y_res[1] / y_res[0] if y_res[0] != 0 else None

                # MicroManager metadata
                if hasattr(page, 'micromanager_metadata'):
                    mm_meta = page.micromanager_metadata
                    if mm_meta:
                        metadata['acquisition_params']['micromanager'] = mm_meta

                # ImageJ metadata
                if hasattr(page, 'imagej_metadata'):
                    ij_meta = page.imagej_metadata
                    if ij_meta:
                        metadata['acquisition_params']['imagej'] = ij_meta

                # Custom tags
                for tag_name, tag in tags.items():
                    if tag_name not in ['Software', 'DateTime', 'ImageDescription',
                                      'XResolution', 'YResolution']:
                        try:
                            metadata['custom_tags'][tag_name] = tag.value
                        except:
                            pass  # Skip tags that can't be read

            # Series information
            if tif.series:
                series = tif.series[0]
                metadata['series_info'] = {
                    'shape': series.shape,
                    'dtype': str(series.dtype),
                    'axes': getattr(series, 'axes', 'unknown')
                }

        except Exception as e:
            self.logger.warning(f"Error extracting TIFF metadata: {str(e)}")

        return metadata

    def _detect_axis_order(self, tif: 'tifffile.TiffFile', metadata: Dict[str, Any]) -> str:
        """Detect axis order from TIFF file structure."""

        if not tif.series:
            return 'unknown'

        series = tif.series[0]
        shape = series.shape
        ndim = len(shape)

        # Try to get axes from tifffile
        if hasattr(series, 'axes') and series.axes:
            detected_axes = series.axes
            # Convert tifffile axis labels to our standard
            axis_mapping = {
                'T': 'T', 'Z': 'Z', 'C': 'C', 'Y': 'Y', 'X': 'X',
                'S': 'C',  # Sample -> Channel
                'I': 'T',  # Index -> Time
            }

            standard_axes = ''.join(axis_mapping.get(ax, ax) for ax in detected_axes)
            if self._is_valid_axis_order(standard_axes):
                return standard_axes

        # Fallback: guess based on shape and metadata
        return self._guess_axis_order_from_shape(shape, metadata)

    def _guess_axis_order_from_shape(self, shape: Tuple[int, ...], metadata: Dict[str, Any]) -> str:
        """Guess axis order based on array shape and metadata clues."""

        ndim = len(shape)

        if ndim == 2:
            return 'YX'

        elif ndim == 3:
            # Look for clues in metadata
            software = metadata.get('software', '').lower()
            description = metadata.get('description', '').lower()

            # Common patterns
            if 'micromanager' in software or 'micro-manager' in description:
                # MicroManager often uses ZYX for z-stacks, TYX for time series
                if shape[0] > 100:  # Likely time series if first dim is large
                    return 'TYX'
                else:
                    return 'ZYX'

            elif 'imagej' in software or 'fiji' in software:
                # ImageJ typically uses ZYX
                return 'ZYX'

            else:
                # Default guess based on shape
                if shape[0] > shape[1] and shape[0] > shape[2]:
                    return 'TYX'  # Time series
                else:
                    return 'ZYX'  # Z-stack

        elif ndim == 4:
            # 4D is often TZYX or CZYX
            if shape[1] < 10 and shape[0] > 10:
                return 'TCZYX'[:-2] + 'YX'  # CZYX if second dim small
            else:
                return 'TZYX'

        elif ndim == 5:
            return 'TCZYX'

        else:
            return 'unknown'

    def _is_valid_axis_order(self, axes: str) -> bool:
        """Check if axis order string is valid."""
        valid_chars = set('TZCYX')
        return all(c in valid_chars for c in axes) and len(set(axes)) == len(axes)

    def _standardize_axes(self, data: np.ndarray, axis_order: str,
                         metadata: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """Standardize data to consistent axis order (TZCYX or subset)."""

        if axis_order == 'unknown':
            # Can't standardize unknown axes
            return data, axis_order

        # Define target order priority: T, Z, C, Y, X
        target_order = 'TZCYX'

        # Find which axes are present
        present_axes = []
        for ax in target_order:
            if ax in axis_order:
                present_axes.append(ax)

        # If already in correct order, no change needed
        current_order = ''.join(ax for ax in axis_order if ax in target_order)
        target = ''.join(present_axes)

        if current_order == target:
            return data, current_order

        # Calculate permutation
        try:
            perm = []
            for target_ax in present_axes:
                source_idx = axis_order.index(target_ax)
                perm.append(source_idx)

            # Apply permutation
            standardized_data = np.transpose(data, perm)

            self.logger.info(f"Standardized axes from {axis_order} to {target}")
            return standardized_data, target

        except (ValueError, IndexError) as e:
            self.logger.warning(f"Could not standardize axes {axis_order}: {str(e)}")
            return data, axis_order


class NumpyLoader(BaseDataLoader):
    """Loader for NumPy array files (.npy, .npz)."""

    SUPPORTED_EXTENSIONS = {'.npy', '.npz'}

    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if file is a NumPy file."""
        path = Path(filepath)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS and path.exists()

    def load(self, filepath: Union[str, Path],
             progress_callback: Optional[Callable[[int, str], None]] = None) -> LoadedData:
        """Load NumPy file."""

        filepath = Path(filepath)
        self.logger.info(f"Loading NumPy file: {filepath}")

        self._update_progress(progress_callback, 0, "Loading NumPy file...")

        try:
            if filepath.suffix.lower() == '.npy':
                data = np.load(str(filepath))
                metadata = self._create_numpy_metadata(filepath, data)

            elif filepath.suffix.lower() == '.npz':
                npz_file = np.load(str(filepath))
                # Take the first array if multiple
                if len(npz_file.files) == 1:
                    data = npz_file[npz_file.files[0]]
                else:
                    # Return the largest array
                    largest_key = max(npz_file.files,
                                    key=lambda k: npz_file[k].size)
                    data = npz_file[largest_key]

                metadata = self._create_numpy_metadata(filepath, data)
                metadata['npz_files'] = list(npz_file.files)

            self._update_progress(progress_callback, 50, "Processing data...")

            # Guess axis order based on shape
            axis_order = self._guess_numpy_axis_order(data.shape)

            metadata.update({
                'axis_order': axis_order,
                'original_shape': data.shape
            })

            self._update_progress(progress_callback, 100, "Loading complete")

            return LoadedData(
                data=data,
                metadata=metadata,
                original_shape=data.shape,
                axis_order=axis_order,
                file_path=filepath,
                loader_type="NumPy"
            )

        except Exception as e:
            self.logger.error(f"Failed to load NumPy file {filepath}: {str(e)}")
            raise RuntimeError(f"NumPy loading failed: {str(e)}")

    def get_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get NumPy file information."""
        filepath = Path(filepath)

        try:
            if filepath.suffix.lower() == '.npy':
                # For .npy files, we can get shape/dtype without loading
                with open(filepath, 'rb') as f:
                    version = np.lib.format.read_magic(f)
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)

                info = {
                    'file_path': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024**2),
                    'shape': shape,
                    'dtype': dtype,
                    'estimated_memory_mb': np.prod(shape) * np.dtype(dtype).itemsize / (1024**2),
                    'fortran_order': fortran
                }

            else:  # .npz
                with np.load(str(filepath)) as npz_file:
                    arrays_info = {}
                    total_size = 0

                    for name in npz_file.files:
                        arr = npz_file[name]
                        arrays_info[name] = {
                            'shape': arr.shape,
                            'dtype': str(arr.dtype),
                            'size_mb': arr.nbytes / (1024**2)
                        }
                        total_size += arr.nbytes

                    info = {
                        'file_path': str(filepath),
                        'file_size_mb': filepath.stat().st_size / (1024**2),
                        'arrays': arrays_info,
                        'total_memory_mb': total_size / (1024**2)
                    }

            return info

        except Exception as e:
            self.logger.error(f"Failed to get NumPy info for {filepath}: {str(e)}")
            return {'error': str(e)}

    def _create_numpy_metadata(self, filepath: Path, data: np.ndarray) -> Dict[str, Any]:
        """Create metadata for NumPy array."""
        return {
            'file_path': str(filepath),
            'file_size_bytes': filepath.stat().st_size,
            'data_size_bytes': data.nbytes,
            'compression': 'npz' if filepath.suffix.lower() == '.npz' else None,
            'loader_info': {
                'loader_type': 'NumPy',
                'numpy_version': np.__version__
            }
        }

    def _guess_numpy_axis_order(self, shape: Tuple[int, ...]) -> str:
        """Guess axis order for NumPy array based on shape."""
        ndim = len(shape)

        if ndim == 2:
            return 'YX'
        elif ndim == 3:
            # Guess based on relative sizes
            if shape[0] > 50:  # Large first dimension likely time
                return 'TYX'
            else:
                return 'ZYX'
        elif ndim == 4:
            return 'TZYX'
        elif ndim == 5:
            return 'TCZYX'
        else:
            return 'unknown'


class HDF5Loader(BaseDataLoader):
    """Loader for HDF5 files."""

    SUPPORTED_EXTENSIONS = {'.h5', '.hdf5', '.hdf'}

    def __init__(self):
        super().__init__()
        if not HAS_H5PY:
            raise ImportError("h5py package required for HDF5 loading")

    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if file is an HDF5 file."""
        path = Path(filepath)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS and path.exists()

    def load(self, filepath: Union[str, Path],
             progress_callback: Optional[Callable[[int, str], None]] = None) -> LoadedData:
        """Load HDF5 file."""

        filepath = Path(filepath)
        self.logger.info(f"Loading HDF5 file: {filepath}")

        self._update_progress(progress_callback, 0, "Opening HDF5 file...")

        try:
            with h5py.File(str(filepath), 'r') as h5f:
                # Find the main dataset
                dataset_path = self._find_main_dataset(h5f)

                if not dataset_path:
                    raise ValueError("No suitable dataset found in HDF5 file")

                self._update_progress(progress_callback, 20, f"Loading dataset: {dataset_path}")

                dataset = h5f[dataset_path]
                data = dataset[...]  # Load all data

                self._update_progress(progress_callback, 60, "Extracting metadata...")

                # Extract metadata
                metadata = self._extract_h5_metadata(h5f, dataset, dataset_path)

                # Guess axis order
                axis_order = self._guess_h5_axis_order(data.shape, metadata)

                metadata.update({
                    'dataset_path': dataset_path,
                    'axis_order': axis_order,
                    'original_shape': data.shape
                })

                self._update_progress(progress_callback, 100, "Loading complete")

                return LoadedData(
                    data=data,
                    metadata=metadata,
                    original_shape=data.shape,
                    axis_order=axis_order,
                    file_path=filepath,
                    loader_type="HDF5"
                )

        except Exception as e:
            self.logger.error(f"Failed to load HDF5 file {filepath}: {str(e)}")
            raise RuntimeError(f"HDF5 loading failed: {str(e)}")

    def get_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get HDF5 file information."""
        filepath = Path(filepath)

        try:
            with h5py.File(str(filepath), 'r') as h5f:
                datasets_info = {}

                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets_info[name] = {
                            'shape': obj.shape,
                            'dtype': str(obj.dtype),
                            'size_mb': obj.size * obj.dtype.itemsize / (1024**2)
                        }

                h5f.visititems(collect_datasets)

                info = {
                    'file_path': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024**2),
                    'datasets': datasets_info
                }

                return info

        except Exception as e:
            self.logger.error(f"Failed to get HDF5 info for {filepath}: {str(e)}")
            return {'error': str(e)}

    def _find_main_dataset(self, h5f: h5py.File) -> Optional[str]:
        """Find the main dataset in HDF5 file."""
        datasets = []

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                datasets.append((name, obj.size))

        h5f.visititems(collect_datasets)

        if not datasets:
            return None

        # Return the largest dataset
        return max(datasets, key=lambda x: x[1])[0]

    def _extract_h5_metadata(self, h5f: h5py.File, dataset: h5py.Dataset,
                            dataset_path: str) -> Dict[str, Any]:
        """Extract metadata from HDF5 file."""
        metadata = {
            'file_attributes': dict(h5f.attrs),
            'dataset_attributes': dict(dataset.attrs),
            'dataset_path': dataset_path,
            'file_size_bytes': Path(h5f.filename).stat().st_size,
            'data_size_bytes': dataset.size * dataset.dtype.itemsize,
            'compression': dataset.compression,
            'chunks': dataset.chunks
        }

        return metadata

    def _guess_h5_axis_order(self, shape: Tuple[int, ...], metadata: Dict[str, Any]) -> str:
        """Guess axis order for HDF5 data."""
        # Look for axis information in attributes
        attrs = metadata.get('dataset_attributes', {})

        if 'axes' in attrs:
            return attrs['axes']

        # Fallback to shape-based guessing
        ndim = len(shape)

        if ndim == 2:
            return 'YX'
        elif ndim == 3:
            return 'ZYX'
        elif ndim == 4:
            return 'TZYX'
        elif ndim == 5:
            return 'TCZYX'
        else:
            return 'unknown'


class DataLoaderManager(QObject):
    """
    Central manager for all data loaders.

    Automatically selects appropriate loader based on file type
    and provides unified interface for loading various formats.
    """

    # Signals
    loading_started = Signal(str)  # filepath
    loading_progress = Signal(int, str)  # percent, message
    loading_completed = Signal(object)  # LoadedData
    loading_failed = Signal(str)  # error message

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize available loaders
        self.loaders: List[BaseDataLoader] = []
        self._register_loaders()

        self.logger.info(f"Registered {len(self.loaders)} data loaders")

    def _register_loaders(self):
        """Register all available data loaders."""

        # Always available
        self.loaders.append(NumpyLoader())

        # Conditional loaders based on available packages
        if HAS_TIFFFILE:
            self.loaders.append(TiffLoader())
            self.logger.info("TIFF loader registered")
        else:
            self.logger.warning("TIFF loader not available - install tifffile")

        if HAS_H5PY:
            self.loaders.append(HDF5Loader())
            self.logger.info("HDF5 loader registered")
        else:
            self.logger.warning("HDF5 loader not available - install h5py")

        # Additional specialized loaders can be added here

    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        extensions = set()

        for loader in self.loaders:
            if hasattr(loader, 'SUPPORTED_EXTENSIONS'):
                extensions.update(loader.SUPPORTED_EXTENSIONS)

        return sorted(extensions)

    def can_load(self, filepath: Union[str, Path]) -> bool:
        """Check if any loader can handle this file."""
        return any(loader.can_load(filepath) for loader in self.loaders)

    def find_loader(self, filepath: Union[str, Path]) -> Optional[BaseDataLoader]:
        """Find appropriate loader for file."""
        for loader in self.loaders:
            if loader.can_load(filepath):
                return loader
        return None

    def load_data(self, filepath: Union[str, Path],
                  loader_type: Optional[str] = None) -> LoadedData:
        """
        Load data from file using appropriate loader.

        Args:
            filepath: Path to file
            loader_type: Optional specific loader type to use

        Returns:
            LoadedData object
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Find loader
        if loader_type:
            loader = self._find_loader_by_type(loader_type)
            if not loader:
                raise ValueError(f"Loader type '{loader_type}' not available")
        else:
            loader = self.find_loader(filepath)
            if not loader:
                raise ValueError(f"No suitable loader found for {filepath}")

        self.logger.info(f"Loading {filepath} with {loader.__class__.__name__}")

        # Emit signals
        self.loading_started.emit(str(filepath))

        try:
            # Load with progress callback
            loaded_data = loader.load(
                filepath,
                progress_callback=self._emit_progress
            )

            self.loading_completed.emit(loaded_data)
            return loaded_data

        except Exception as e:
            error_msg = f"Loading failed: {str(e)}"
            self.logger.error(error_msg)
            self.loading_failed.emit(error_msg)
            raise

    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get file information without loading full data."""
        filepath = Path(filepath)

        loader = self.find_loader(filepath)
        if not loader:
            return {'error': f"No suitable loader for {filepath}"}

        try:
            return loader.get_info(filepath)
        except Exception as e:
            return {'error': str(e)}

    def _find_loader_by_type(self, loader_type: str) -> Optional[BaseDataLoader]:
        """Find loader by type name."""
        type_mapping = {
            'tiff': TiffLoader,
            'numpy': NumpyLoader,
            'hdf5': HDF5Loader
        }

        target_class = type_mapping.get(loader_type.lower())
        if not target_class:
            return None

        for loader in self.loaders:
            if isinstance(loader, target_class):
                return loader

        return None

    def _emit_progress(self, percent: int, message: str):
        """Emit progress signal."""
        self.loading_progress.emit(percent, message)

    def get_loader_info(self) -> Dict[str, Any]:
        """Get information about available loaders."""
        info = {
            'available_loaders': [],
            'supported_extensions': self.get_supported_extensions(),
            'total_loaders': len(self.loaders)
        }

        for loader in self.loaders:
            loader_info = {
                'name': loader.__class__.__name__,
                'extensions': getattr(loader, 'SUPPORTED_EXTENSIONS', [])
            }
            info['available_loaders'].append(loader_info)

        return info


# Convenience functions
def load_data_file(filepath: Union[str, Path],
                   progress_callback: Optional[Callable] = None) -> LoadedData:
    """Convenience function to load any supported data file."""
    manager = DataLoaderManager()
    return manager.load_data(filepath)


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to get file information."""
    manager = DataLoaderManager()
    return manager.get_file_info(filepath)


def get_supported_formats() -> List[str]:
    """Get list of supported file formats."""
    manager = DataLoaderManager()
    return manager.get_supported_extensions()
