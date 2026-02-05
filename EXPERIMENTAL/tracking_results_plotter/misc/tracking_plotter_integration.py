# tracking_results_plotter/integration.py
"""
Integration helpers for the Tracking Results Plotter plugin.

This module provides utilities for integrating with other FLIKA plugins,
analysis tools, and external software commonly used in particle tracking.
"""

import numpy as np
import pandas as pd
import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import warnings

try:
    import flika
    from flika import global_vars as g
    from flika.window import Window
    from flika.roi import ROI_Base
    from flika.process.file_ import open_file
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False

# Import plugin modules
try:
    from .utils import TrackingDataManager, ColorManager, ExportManager
    from .advanced_plots import AdvancedPlotter
    PLUGIN_MODULES_AVAILABLE = True
except ImportError:
    PLUGIN_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class SPTBatchAnalysisIntegration:
    """
    Integration with SPT Batch Analysis plugin for seamless workflow.
    Provides automatic data loading and analysis chaining.
    """
    
    def __init__(self):
        self.spt_plugin_available = self._check_spt_plugin()
        self.data_formats = {
            'tracks': ['_tracks.csv', '_enhanced_analysis.csv'],
            'features': ['_features.csv', '_analysis_results.csv'],
            'detection': ['_locsID.csv', '_detections.csv']
        }
    
    def _check_spt_plugin(self) -> bool:
        """Check if SPT Batch Analysis plugin is available."""
        try:
            # Try to find SPT plugin
            plugins_dir = Path.home() / ".FLIKA" / "plugins"
            spt_dir = plugins_dir / "spt_batch_analysis"
            
            return spt_dir.exists() and (spt_dir / "__init__.py").exists()
        except Exception:
            return False
    
    def find_spt_output_files(self, directory: Union[str, Path], 
                             experiment_pattern: str = "*") -> Dict[str, List[Path]]:
        """
        Find SPT Batch Analysis output files in a directory.
        
        Args:
            directory: Directory to search for SPT output files
            experiment_pattern: Pattern to match experiment names
            
        Returns:
            Dictionary mapping file types to lists of found files
        """
        directory = Path(directory)
        found_files = {data_type: [] for data_type in self.data_formats}
        
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return found_files
        
        # Search for files with SPT naming patterns
        for data_type, suffixes in self.data_formats.items():
            for suffix in suffixes:
                pattern = f"{experiment_pattern}{suffix}"
                files = list(directory.glob(pattern))
                found_files[data_type].extend(files)
        
        logger.info(f"Found SPT files in {directory}:")
        for data_type, files in found_files.items():
            logger.info(f"  {data_type}: {len(files)} files")
        
        return found_files
    
    def auto_load_spt_results(self, directory: Union[str, Path], 
                             data_type: str = 'tracks') -> Optional[pd.DataFrame]:
        """
        Automatically load SPT results from directory.
        
        Args:
            directory: Directory containing SPT output files
            data_type: Type of data to load ('tracks', 'features', 'detection')
            
        Returns:
            Combined DataFrame or None if no files found
        """
        found_files = self.find_spt_output_files(directory)
        
        if data_type not in found_files or not found_files[data_type]:
            logger.warning(f"No {data_type} files found in {directory}")
            return None
        
        # Load and combine all files of the specified type
        dataframes = []
        
        for filepath in found_files[data_type]:
            try:
                df = pd.read_csv(filepath)
                
                # Add source file information
                df['source_file'] = filepath.name
                df['experiment_base'] = filepath.stem.replace('_tracks', '').replace('_features', '').replace('_enhanced_analysis', '')
                
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} rows from {filepath.name}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {str(e)}")
        
        if not dataframes:
            return None
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(dataframes)} files into dataset with {len(combined_df)} rows")
        
        return combined_df
    
    def create_analysis_pipeline(self, data_directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a complete analysis pipeline from SPT output files.
        
        Args:
            data_directory: Directory containing SPT results
            
        Returns:
            Dictionary with loaded data and analysis results
        """
        results = {
            'tracks_data': None,
            'features_data': None,
            'detection_data': None,
            'analysis_summary': {},
            'pipeline_success': False
        }
        
        try:
            # Load different data types
            for data_type in ['tracks', 'features', 'detection']:
                data = self.auto_load_spt_results(data_directory, data_type)
                results[f'{data_type}_data'] = data
                
                if data is not None:
                    results['analysis_summary'][data_type] = {
                        'n_rows': len(data),
                        'n_experiments': data['experiment_base'].nunique() if 'experiment_base' in data.columns else 'unknown',
                        'columns': list(data.columns)
                    }
            
            results['pipeline_success'] = any(data is not None for data in 
                                            [results['tracks_data'], results['features_data']])
            
            logger.info(f"Analysis pipeline {'completed' if results['pipeline_success'] else 'failed'}")
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
        
        return results


class TrackMateIntegration:
    """
    Integration with TrackMate/ImageJ tracking results.
    Handles TrackMate XML and CSV export formats.
    """
    
    def __init__(self):
        self.trackmate_columns = {
            'TRACK_ID': 'track_id',
            'FRAME': 'frame', 
            'POSITION_X': 'x',
            'POSITION_Y': 'y',
            'TOTAL_INTENSITY': 'intensity',
            'MEAN_INTENSITY': 'mean_intensity'
        }
    
    def convert_trackmate_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Convert TrackMate CSV export to standard format.
        
        Args:
            filepath: Path to TrackMate CSV file
            
        Returns:
            Converted DataFrame
        """
        filepath = Path(filepath)
        
        try:
            # Load TrackMate data
            df = pd.read_csv(filepath)
            
            # Map TrackMate columns to standard names
            column_mapping = {}
            for tm_col, std_col in self.trackmate_columns.items():
                if tm_col in df.columns:
                    column_mapping[tm_col] = std_col
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['track_id', 'frame', 'x', 'y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Convert data types
            df['track_id'] = df['track_id'].astype(int)
            df['frame'] = df['frame'].astype(int)
            df['x'] = pd.to_numeric(df['x'])
            df['y'] = pd.to_numeric(df['y'])
            
            # Add source information
            df['source'] = 'TrackMate'
            df['source_file'] = filepath.name
            
            logger.info(f"Converted TrackMate data: {len(df)} points, {df['track_id'].nunique()} tracks")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting TrackMate data: {str(e)}")
            raise
    
    def parse_trackmate_xml(self, filepath: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Parse TrackMate XML file (requires xml parsing).
        
        Args:
            filepath: Path to TrackMate XML file
            
        Returns:
            Dictionary with spots and tracks DataFrames
        """
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("XML parsing not available")
        
        filepath = Path(filepath)
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Parse spots (detections)
            spots_data = []
            for spot in root.findall('.//Spot'):
                spot_data = {
                    'id': int(spot.get('ID', 0)),
                    'frame': int(spot.get('FRAME', 0)),
                    'x': float(spot.get('POSITION_X', 0)),
                    'y': float(spot.get('POSITION_Y', 0)),
                    'intensity': float(spot.get('TOTAL_INTENSITY', 0))
                }
                spots_data.append(spot_data)
            
            spots_df = pd.DataFrame(spots_data)
            
            # Parse tracks
            tracks_data = []
            for track in root.findall('.//Track'):
                track_id = int(track.get('TRACK_ID', 0))
                for edge in track.findall('.//Edge'):
                    source_id = int(edge.get('SPOT_SOURCE_ID', 0))
                    target_id = int(edge.get('SPOT_TARGET_ID', 0))
                    # Link spots to tracks here
            
            # Create tracks DataFrame by linking spots
            if len(spots_data) > 0:
                spots_df['track_id'] = spots_df['id']  # Simplified - would need proper linking
                tracks_df = spots_df.rename(columns={'id': 'spot_id'})
            else:
                tracks_df = pd.DataFrame()
            
            logger.info(f"Parsed TrackMate XML: {len(spots_df)} spots, {len(tracks_df)} track points")
            
            return {
                'spots': spots_df,
                'tracks': tracks_df
            }
            
        except Exception as e:
            logger.error(f"Error parsing TrackMate XML: {str(e)}")
            raise


class UTrackIntegration:
    """
    Integration with u-track MATLAB tracking software.
    Handles u-track output formats and conversions.
    """
    
    def __init__(self):
        self.utrack_formats = {
            'tracksFinal': 'matlab_struct',
            'movieInfo': 'matlab_struct',
            'csv_export': 'csv'
        }
    
    def convert_utrack_csv(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Convert u-track CSV export to standard format.
        
        Args:
            filepath: Path to u-track CSV file
            
        Returns:
            Converted DataFrame
        """
        filepath = Path(filepath)
        
        try:
            df = pd.read_csv(filepath)
            
            # u-track CSV typically has columns like:
            # xCoord, yCoord, frame, trackID, etc.
            utrack_mapping = {
                'xCoord': 'x',
                'yCoord': 'y', 
                'frame': 'frame',
                'trackID': 'track_id',
                'amp': 'intensity'
            }
            
            # Map columns
            column_mapping = {}
            for utrack_col, std_col in utrack_mapping.items():
                if utrack_col in df.columns:
                    column_mapping[utrack_col] = std_col
            
            df = df.rename(columns=column_mapping)
            
            # Add source information
            df['source'] = 'u-track'
            df['source_file'] = filepath.name
            
            logger.info(f"Converted u-track data: {len(df)} points")
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting u-track data: {str(e)}")
            raise
    
    def load_utrack_matlab(self, filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load u-track MATLAB .mat file (requires scipy).
        
        Args:
            filepath: Path to u-track .mat file
            
        Returns:
            Converted DataFrame or None if scipy not available
        """
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.warning("SciPy not available for MATLAB file loading")
            return None
        
        filepath = Path(filepath)
        
        try:
            # Load MATLAB file
            mat_data = loadmat(filepath)
            
            # u-track typically stores data in 'tracksFinal' structure
            if 'tracksFinal' in mat_data:
                tracks_final = mat_data['tracksFinal']
                
                # Convert MATLAB structure to DataFrame
                # This is a simplified conversion - actual u-track structure is complex
                tracks_data = []
                
                for track_idx, track in enumerate(tracks_final[0]):
                    if len(track) > 0:
                        coords = track[0]  # Coordinates matrix
                        
                        for frame_idx in range(coords.shape[0]):
                            if not np.isnan(coords[frame_idx, 0]):
                                tracks_data.append({
                                    'track_id': track_idx + 1,
                                    'frame': frame_idx,
                                    'x': coords[frame_idx, 0],
                                    'y': coords[frame_idx, 1],
                                    'intensity': coords[frame_idx, 3] if coords.shape[1] > 3 else np.nan
                                })
                
                df = pd.DataFrame(tracks_data)
                df['source'] = 'u-track'
                df['source_file'] = filepath.name
                
                logger.info(f"Loaded u-track MATLAB data: {len(df)} points, {df['track_id'].nunique()} tracks")
                
                return df
            
            else:
                logger.warning("No 'tracksFinal' found in MATLAB file")
                return None
                
        except Exception as e:
            logger.error(f"Error loading u-track MATLAB file: {str(e)}")
            return None


class MultiFormatLoader:
    """
    Universal loader for multiple tracking result formats.
    Automatically detects format and applies appropriate conversion.
    """
    
    def __init__(self):
        self.integrations = {
            'spt_batch': SPTBatchAnalysisIntegration(),
            'trackmate': TrackMateIntegration(),
            'utrack': UTrackIntegration()
        }
        
        self.format_signatures = {
            'spt_batch': ['track_number', 'radius_gyration', 'SVM'],
            'trackmate': ['TRACK_ID', 'POSITION_X', 'TOTAL_INTENSITY'],
            'utrack': ['xCoord', 'yCoord', 'trackID'],
            'generic': ['track_id', 'frame', 'x', 'y']
        }
    
    def detect_format(self, filepath: Union[str, Path]) -> str:
        """
        Detect the format of a tracking results file.
        
        Args:
            filepath: Path to tracking results file
            
        Returns:
            Detected format name
        """
        filepath = Path(filepath)
        
        try:
            if filepath.suffix.lower() == '.xml':
                return 'trackmate_xml'
            elif filepath.suffix.lower() == '.mat':
                return 'utrack_matlab'
            elif filepath.suffix.lower() == '.csv':
                # Read first few rows to check column names
                df_sample = pd.read_csv(filepath, nrows=5)
                columns = df_sample.columns.tolist()
                
                # Check for format signatures
                for format_name, signature_cols in self.format_signatures.items():
                    if any(col in columns for col in signature_cols):
                        matches = sum(1 for col in signature_cols if col in columns)
                        if matches >= len(signature_cols) * 0.6:  # 60% match threshold
                            return format_name
                
                return 'generic'
            else:
                return 'unknown'
                
        except Exception as e:
            logger.error(f"Error detecting format: {str(e)}")
            return 'unknown'
    
    def load_tracking_data(self, filepath: Union[str, Path], 
                          force_format: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load tracking data from file with automatic format detection.
        
        Args:
            filepath: Path to tracking results file
            force_format: Force specific format instead of auto-detection
            
        Returns:
            Loaded and standardized DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        # Detect or use forced format
        format_name = force_format or self.detect_format(filepath)
        logger.info(f"Loading {filepath.name} as {format_name} format")
        
        try:
            # Load based on detected format
            if format_name == 'spt_batch':
                df = pd.read_csv(filepath)
                
            elif format_name == 'trackmate':
                df = self.integrations['trackmate'].convert_trackmate_csv(filepath)
                
            elif format_name == 'trackmate_xml':
                xml_data = self.integrations['trackmate'].parse_trackmate_xml(filepath)
                df = xml_data.get('tracks')
                
            elif format_name == 'utrack':
                df = self.integrations['utrack'].convert_utrack_csv(filepath)
                
            elif format_name == 'utrack_matlab':
                df = self.integrations['utrack'].load_utrack_matlab(filepath)
                
            elif format_name == 'generic':
                df = pd.read_csv(filepath)
                
            else:
                logger.error(f"Unknown format: {format_name}")
                return None
            
            if df is not None:
                logger.info(f"Successfully loaded {len(df)} data points")
                
                # Add format information
                df['input_format'] = format_name
                df['input_file'] = filepath.name
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {format_name} format: {str(e)}")
            return None


class WorkflowBuilder:
    """
    Build automated analysis workflows combining multiple tools.
    """
    
    def __init__(self):
        self.loader = MultiFormatLoader()
        self.steps = []
        self.results = {}
    
    def add_step(self, step_name: str, function: Callable, **kwargs):
        """Add a step to the workflow."""
        self.steps.append({
            'name': step_name,
            'function': function,
            'kwargs': kwargs
        })
    
    def create_standard_workflow(self, input_file: Union[str, Path], 
                                output_dir: Union[str, Path]) -> 'WorkflowBuilder':
        """
        Create a standard analysis workflow.
        
        Args:
            input_file: Input tracking results file
            output_dir: Directory for output files
            
        Returns:
            Configured workflow builder
        """
        # Step 1: Load data
        self.add_step('load_data', self._load_data, 
                     input_file=input_file)
        
        # Step 2: Validate data
        self.add_step('validate_data', self._validate_data)
        
        # Step 3: Generate summary statistics
        self.add_step('calculate_statistics', self._calculate_statistics)
        
        # Step 4: Create visualizations
        self.add_step('create_plots', self._create_plots, 
                     output_dir=output_dir)
        
        # Step 5: Export results
        self.add_step('export_results', self._export_results,
                     output_dir=output_dir)
        
        return self
    
    def _load_data(self, input_file: Union[str, Path]) -> pd.DataFrame:
        """Load tracking data."""
        return self.loader.load_tracking_data(input_file)
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data."""
        if not PLUGIN_MODULES_AVAILABLE:
            return {'valid': True, 'warnings': ['Validation skipped - plugin modules not available']}
        
        from .utils import DataValidator
        
        validator = DataValidator()
        # Create basic column mapping
        mapping = {
            'track_id': 'track_id' if 'track_id' in data.columns else 'track_number',
            'frame': 'frame',
            'x': 'x',
            'y': 'y'
        }
        
        return validator.validate_dataframe(data, mapping)
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics."""
        stats = {}
        
        # Basic statistics
        stats['n_points'] = len(data)
        stats['n_tracks'] = data['track_id'].nunique() if 'track_id' in data.columns else data['track_number'].nunique()
        
        # Track length statistics
        track_col = 'track_id' if 'track_id' in data.columns else 'track_number'
        track_lengths = data.groupby(track_col).size()
        
        stats['track_lengths'] = {
            'mean': float(track_lengths.mean()),
            'median': float(track_lengths.median()),
            'min': int(track_lengths.min()),
            'max': int(track_lengths.max())
        }
        
        return stats
    
    def _create_plots(self, data: pd.DataFrame, output_dir: Union[str, Path]):
        """Create visualization plots."""
        if not PLUGIN_MODULES_AVAILABLE:
            logger.warning("Cannot create plots - plugin modules not available")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Cannot create plots - matplotlib not available")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create simple trajectory plot
        plt.figure(figsize=(10, 8))
        
        track_col = 'track_id' if 'track_id' in data.columns else 'track_number'
        unique_tracks = data[track_col].unique()[:10]  # First 10 tracks
        
        for track_id in unique_tracks:
            track_data = data[data[track_col] == track_id].sort_values('frame')
            plt.plot(track_data['x'], track_data['y'], alpha=0.7, linewidth=1)
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Track Trajectories')
        plt.grid(True, alpha=0.3)
        
        plot_file = output_dir / 'trajectories.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trajectory plot saved to: {plot_file}")
    
    def _export_results(self, data: pd.DataFrame, stats: Dict[str, Any], 
                       output_dir: Union[str, Path]):
        """Export analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export data
        data_file = output_dir / 'processed_data.csv'
        data.to_csv(data_file, index=False)
        
        # Export statistics
        stats_file = output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results exported to: {output_dir}")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete workflow.
        
        Returns:
            Dictionary with results from each step
        """
        logger.info(f"Running workflow with {len(self.steps)} steps")
        
        results = {}
        data = None
        
        for step in self.steps:
            step_name = step['name']
            function = step['function']
            kwargs = step['kwargs']
            
            try:
                logger.info(f"Executing step: {step_name}")
                
                # Pass data from previous steps
                if step_name == 'load_data':
                    result = function(**kwargs)
                    data = result
                elif step_name in ['validate_data', 'calculate_statistics']:
                    result = function(data, **kwargs)
                elif step_name == 'create_plots':
                    result = function(data, **kwargs)
                elif step_name == 'export_results':
                    stats = results.get('calculate_statistics', {})
                    result = function(data, stats, **kwargs)
                else:
                    result = function(data, **kwargs)
                
                results[step_name] = result
                logger.info(f"✓ Step {step_name} completed")
                
            except Exception as e:
                logger.error(f"✗ Step {step_name} failed: {str(e)}")
                results[step_name] = {'error': str(e)}
        
        logger.info("Workflow execution complete")
        return results


# Create global instances
spt_integration = SPTBatchAnalysisIntegration()
trackmate_integration = TrackMateIntegration()
utrack_integration = UTrackIntegration()
multi_loader = MultiFormatLoader()

def create_workflow(input_file: Union[str, Path], 
                   output_dir: Union[str, Path]) -> WorkflowBuilder:
    """
    Convenience function to create a standard analysis workflow.
    
    Args:
        input_file: Input tracking results file
        output_dir: Directory for output files
        
    Returns:
        Configured workflow builder
    """
    workflow = WorkflowBuilder()
    return workflow.create_standard_workflow(input_file, output_dir)