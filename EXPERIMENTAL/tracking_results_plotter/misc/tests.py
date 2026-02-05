# tracking_results_plotter/test_tracking_plotter.py
"""
Comprehensive test suite for the Tracking Results Plotter plugin.

This module provides unit tests, integration tests, and performance tests
to ensure the plugin works correctly across different scenarios and data formats.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Tuple, Optional

# Add plugin path for imports
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

# Import plugin modules
try:
    from utils import (TrackingDataManager, DataValidator, ColorManager, 
                      ExportManager, MathUtils, generate_sample_data, 
                      ConfigManager)
    from advanced_plots import AdvancedPlotter
    PLUGIN_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Plugin modules not available for testing: {e}")
    PLUGIN_MODULES_AVAILABLE = False

# Mock FLIKA if not available
try:
    import flika
    from flika import global_vars as g
    from flika.window import Window
    FLIKA_AVAILABLE = True
except ImportError:
    FLIKA_AVAILABLE = False
    
    # Create mock FLIKA modules
    class MockWindow:
        def __init__(self, image, name='test'):
            self.image = image
            self.name = name
            self.currentIndex = 0
    
    class MockGlobalVars:
        def __init__(self):
            self.win = None
            self.m = None
        
        def alert(self, message):
            print(f"FLIKA Alert: {message}")
    
    g = MockGlobalVars()

class TestTrackingDataManager(unittest.TestCase):
    """Test suite for TrackingDataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
            
        self.data_manager = TrackingDataManager()
        self.sample_data = generate_sample_data(n_tracks=5, n_frames=50)
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_valid_data(self):
        """Test loading valid tracking data."""
        success = self.data_manager.load_data(self.temp_file.name)
        
        self.assertTrue(success)
        self.assertIsNotNone(self.data_manager.data)
        self.assertEqual(len(self.data_manager.data), len(self.sample_data))
        
        # Check column mapping
        self.assertIn('track_id', self.data_manager.column_mapping)
        self.assertIn('frame', self.data_manager.column_mapping)
        self.assertIn('x', self.data_manager.column_mapping)
        self.assertIn('y', self.data_manager.column_mapping)
    
    def test_load_invalid_file(self):
        """Test loading non-existent file."""
        success = self.data_manager.load_data('nonexistent_file.csv')
        self.assertFalse(success)
    
    def test_column_detection(self):
        """Test automatic column detection."""
        self.data_manager.load_data(self.temp_file.name)
        
        # Check that required columns were detected
        required_columns = ['track_id', 'frame', 'x', 'y']
        for col in required_columns:
            self.assertIn(col, self.data_manager.column_mapping)
            mapped_col = self.data_manager.column_mapping[col]
            self.assertIn(mapped_col, self.data_manager.data.columns)
    
    def test_get_tracks(self):
        """Test track filtering functionality."""
        self.data_manager.load_data(self.temp_file.name)
        
        # Test getting all tracks
        all_tracks = self.data_manager.get_tracks()
        self.assertEqual(len(all_tracks), len(self.sample_data))
        
        # Test getting specific tracks
        track_ids = [1, 2]
        filtered_tracks = self.data_manager.get_tracks(track_ids)
        unique_tracks = filtered_tracks[self.data_manager.column_mapping['track_id']].unique()
        self.assertTrue(all(tid in track_ids for tid in unique_tracks))
    
    def test_get_track_summary(self):
        """Test track summary statistics."""
        self.data_manager.load_data(self.temp_file.name)
        
        summary = self.data_manager.get_track_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn('track_id', summary.columns)
        self.assertIn('n_points', summary.columns)
        self.assertIn('x_mean', summary.columns)
        self.assertIn('y_mean', summary.columns)
    
    def test_derived_columns(self):
        """Test creation of derived columns."""
        self.data_manager.load_data(self.temp_file.name)
        
        # Check derived columns were created
        derived_cols = ['track_length_frames', 'frame_in_track', 
                       'displacement_x', 'displacement_y', 'displacement_magnitude']
        
        for col in derived_cols:
            self.assertIn(col, self.data_manager.data.columns)
            self.assertIn(col, self.data_manager.numeric_columns)


class TestDataValidator(unittest.TestCase):
    """Test suite for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
            
        self.validator = DataValidator()
        self.valid_data = generate_sample_data(n_tracks=3, n_frames=20)
        self.column_mapping = {
            'track_id': 'track_number',
            'frame': 'frame',
            'x': 'x',
            'y': 'y'
        }
    
    def test_validate_valid_data(self):
        """Test validation of valid tracking data."""
        results = self.validator.validate_dataframe(self.valid_data, self.column_mapping)
        
        self.assertTrue(results['is_valid'])
        self.assertEqual(len(results['errors']), 0)
        self.assertIn('statistics', results)
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        incomplete_mapping = {
            'track_id': 'track_number',
            'frame': 'frame'
            # Missing x and y
        }
        
        results = self.validator.validate_dataframe(self.valid_data, incomplete_mapping)
        
        self.assertFalse(results['is_valid'])
        self.assertGreater(len(results['errors']), 0)
    
    def test_validate_data_types(self):
        """Test validation of data types."""
        # Create data with string track IDs
        invalid_data = self.valid_data.copy()
        invalid_data['track_number'] = invalid_data['track_number'].astype(str)
        
        results = self.validator.validate_dataframe(invalid_data, self.column_mapping)
        
        # Should still be valid (string IDs can be converted)
        self.assertTrue(results['is_valid'])
        self.assertGreater(len(results['warnings']), 0)
    
    def test_validate_coordinate_ranges(self):
        """Test validation of coordinate ranges."""
        # Create data with extreme coordinates
        extreme_data = self.valid_data.copy()
        extreme_data['x'] = extreme_data['x'] * 10000  # Very large coordinates
        
        results = self.validator.validate_dataframe(extreme_data, self.column_mapping)
        
        self.assertTrue(results['is_valid'])  # Valid but with warnings
        self.assertGreater(len(results['warnings']), 0)
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        results = self.validator.validate_dataframe(self.valid_data, self.column_mapping)
        
        stats = results['statistics']
        self.assertIn('total_points', stats)
        self.assertIn('unique_tracks', stats)
        self.assertIn('track_length_stats', stats)
        self.assertIn('x_range', stats)
        self.assertIn('y_range', stats)
        
        self.assertEqual(stats['total_points'], len(self.valid_data))
        self.assertEqual(stats['unique_tracks'], self.valid_data['track_number'].nunique())


class TestColorManager(unittest.TestCase):
    """Test suite for ColorManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
            
        self.color_manager = ColorManager()
    
    def test_color_initialization(self):
        """Test color manager initialization."""
        self.assertIn('track', self.color_manager.default_colors)
        self.assertIn('viridis', self.color_manager.colormaps)
        self.assertIsInstance(self.color_manager.colormaps['viridis'], list)
    
    def test_set_property_range(self):
        """Test setting property range for normalization."""
        self.color_manager.set_property_range('velocity', 0.0, 10.0)
        
        self.assertIn('velocity', self.color_manager.property_ranges)
        self.assertEqual(self.color_manager.property_ranges['velocity'], (0.0, 10.0))
    
    def test_get_color_for_value(self):
        """Test color mapping for values."""
        self.color_manager.set_property_range('test_prop', 0.0, 1.0)
        
        # Test valid value
        color = self.color_manager.get_color_for_value(0.5, 'test_prop', 'viridis')
        self.assertIsInstance(color, str)
        self.assertTrue(color.startswith('#') or color in ['red', 'blue', 'green'])
        
        # Test NaN value
        color_nan = self.color_manager.get_color_for_value(np.nan, 'test_prop', 'viridis')
        self.assertEqual(color_nan, self.color_manager.default_colors['background'])
    
    def test_get_default_color(self):
        """Test getting default colors."""
        track_color = self.color_manager.get_default_color('track')
        self.assertEqual(track_color, self.color_manager.default_colors['track'])
        
        unknown_color = self.color_manager.get_default_color('unknown')
        self.assertEqual(unknown_color, '#000000')


class TestMathUtils(unittest.TestCase):
    """Test suite for MathUtils class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
    
    def test_calculate_displacement(self):
        """Test displacement calculation."""
        displacement = MathUtils.calculate_displacement(0, 0, 3, 4)
        self.assertAlmostEqual(displacement, 5.0, places=5)
        
        displacement_zero = MathUtils.calculate_displacement(1, 1, 1, 1)
        self.assertEqual(displacement_zero, 0.0)
    
    def test_calculate_angle(self):
        """Test angle calculation."""
        # Test known angles
        angle_right = MathUtils.calculate_angle(0, 0, 1, 0)  # 0 radians
        self.assertAlmostEqual(angle_right, 0.0, places=5)
        
        angle_up = MathUtils.calculate_angle(0, 0, 0, 1)  # π/2 radians
        self.assertAlmostEqual(angle_up, np.pi/2, places=5)
    
    def test_calculate_radius_of_gyration(self):
        """Test radius of gyration calculation."""
        # Test simple case: square
        x_coords = np.array([0, 1, 1, 0])
        y_coords = np.array([0, 0, 1, 1])
        
        rg = MathUtils.calculate_radius_of_gyration(x_coords, y_coords)
        self.assertIsInstance(rg, float)
        self.assertGreater(rg, 0)
        
        # Test single point
        single_rg = MathUtils.calculate_radius_of_gyration(np.array([1]), np.array([1]))
        self.assertTrue(np.isnan(single_rg))
    
    def test_calculate_track_velocity(self):
        """Test velocity calculation."""
        # Create simple track data
        track_data = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 0, 0, 0],
            'frame': [0, 1, 2, 3]
        })
        
        velocities = MathUtils.calculate_track_velocity(track_data, 'x', 'y', 'frame')
        
        # Should be constant velocity of 1 unit/frame
        expected_velocities = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(velocities, expected_velocities)
    
    def test_calculate_track_straightness(self):
        """Test track straightness calculation."""
        # Perfectly straight track
        straight_track = pd.DataFrame({
            'x': [0, 1, 2, 3],
            'y': [0, 0, 0, 0]
        })
        
        straightness = MathUtils.calculate_track_straightness(straight_track, 'x', 'y')
        self.assertAlmostEqual(straightness, 1.0, places=5)
        
        # Single point track
        single_point = pd.DataFrame({'x': [1], 'y': [1]})
        straightness_single = MathUtils.calculate_track_straightness(single_point, 'x', 'y')
        self.assertTrue(np.isnan(straightness_single))
    
    def test_calculate_turning_angles(self):
        """Test turning angle calculation."""
        # L-shaped track (90-degree turn)
        l_track = pd.DataFrame({
            'x': [0, 1, 1],
            'y': [0, 0, 1]
        })
        
        angles = MathUtils.calculate_turning_angles(l_track, 'x', 'y')
        
        self.assertEqual(len(angles), 1)  # One turning angle for 3 points
        self.assertAlmostEqual(abs(angles[0]), np.pi/2, places=3)  # 90-degree turn


class TestExportManager(unittest.TestCase):
    """Test suite for ExportManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
            
        self.export_manager = ExportManager()
        self.sample_data = generate_sample_data(n_tracks=3, n_frames=20)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_export_data_csv(self):
        """Test CSV data export."""
        output_file = os.path.join(self.temp_dir, 'test_export.csv')
        
        success = self.export_manager.export_data(self.sample_data, output_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_file))
        
        # Verify exported data
        exported_data = pd.read_csv(output_file)
        self.assertEqual(len(exported_data), len(self.sample_data))
        self.assertEqual(list(exported_data.columns), list(self.sample_data.columns))
    
    def test_export_statistics_json(self):
        """Test JSON statistics export."""
        output_file = os.path.join(self.temp_dir, 'test_stats.json')
        
        stats = {
            'n_tracks': 5,
            'mean_velocity': 1.5,
            'nested_stats': {'mean': 10.0, 'std': 2.0}
        }
        
        success = self.export_manager.export_statistics(stats, output_file)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_file))
        
        # Verify exported stats
        import json
        with open(output_file, 'r') as f:
            exported_stats = json.load(f)
        
        self.assertEqual(exported_stats['n_tracks'], 5)
        self.assertEqual(exported_stats['mean_velocity'], 1.5)
    
    def test_export_unsupported_format(self):
        """Test export with unsupported format."""
        output_file = os.path.join(self.temp_dir, 'test_export.xyz')
        
        success = self.export_manager.export_data(self.sample_data, output_file)
        
        self.assertFalse(success)
        self.assertFalse(os.path.exists(output_file))


class TestAdvancedPlotter(unittest.TestCase):
    """Test suite for AdvancedPlotter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            self.matplotlib_available = True
        except ImportError:
            self.matplotlib_available = False
            
        if self.matplotlib_available:
            self.plotter = AdvancedPlotter()
            self.sample_data = generate_sample_data(n_tracks=5, n_frames=50)
            self.column_mapping = {
                'track_id': 'track_number',
                'frame': 'frame',
                'x': 'x',
                'y': 'y',
                'intensity': 'intensity'
            }
    
    def test_plotter_initialization(self):
        """Test plotter initialization."""
        if not self.matplotlib_available:
            self.skipTest("Matplotlib not available")
            
        self.assertIsNotNone(self.plotter.color_manager)
        self.assertIn('dpi', self.plotter.figure_style)
        self.assertIn('figsize', self.plotter.figure_style)
    
    def test_create_flower_plot_grid(self):
        """Test flower plot grid creation."""
        if not self.matplotlib_available:
            self.skipTest("Matplotlib not available")
        
        track_ids = [1, 2, 3]
        
        try:
            fig = self.plotter.create_flower_plot_grid(
                self.sample_data, self.column_mapping, track_ids, grid_size=(2, 2))
            
            self.assertIsNotNone(fig)
            # Clean up
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"Flower plot creation failed: {e}")
    
    def test_create_temporal_analysis_figure(self):
        """Test temporal analysis figure creation."""
        if not self.matplotlib_available:
            self.skipTest("Matplotlib not available")
        
        # Use first track ID
        track_id = self.sample_data['track_number'].iloc[0]
        
        try:
            fig = self.plotter.create_temporal_analysis_figure(
                self.sample_data, self.column_mapping, track_id)
            
            self.assertIsNotNone(fig)
            # Clean up
            import matplotlib.pyplot as plt
            plt.close(fig)
            
        except Exception as e:
            self.fail(f"Temporal analysis figure creation failed: {e}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete plugin workflow."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
            
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = generate_sample_data(n_tracks=10, n_frames=100)
        
        # Create sample CSV file
        self.csv_file = os.path.join(self.temp_dir, 'integration_test.csv')
        self.sample_data.to_csv(self.csv_file, index=False)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_workflow(self):
        """Test complete plugin workflow."""
        # 1. Load data
        data_manager = TrackingDataManager()
        success = data_manager.load_data(self.csv_file)
        self.assertTrue(success)
        
        # 2. Validate data
        validator = DataValidator()
        results = validator.validate_dataframe(
            data_manager.data, data_manager.column_mapping)
        self.assertTrue(results['is_valid'])
        
        # 3. Get track summary
        summary = data_manager.get_track_summary()
        self.assertGreater(len(summary), 0)
        
        # 4. Export results
        export_manager = ExportManager()
        output_file = os.path.join(self.temp_dir, 'workflow_output.csv')
        export_success = export_manager.export_data(summary, output_file)
        self.assertTrue(export_success)
        
        # 5. Verify exported file
        self.assertTrue(os.path.exists(output_file))
        exported_summary = pd.read_csv(output_file)
        self.assertEqual(len(exported_summary), len(summary))
    
    def test_error_handling(self):
        """Test error handling in workflow."""
        # Test with invalid file
        data_manager = TrackingDataManager()
        success = data_manager.load_data('nonexistent_file.csv')
        self.assertFalse(success)
        
        # Test with empty data
        empty_data = pd.DataFrame()
        validator = DataValidator()
        
        with self.assertRaises(Exception):
            validator.validate_dataframe(empty_data, {})


class TestPerformance(unittest.TestCase):
    """Performance tests for the plugin."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        if not PLUGIN_MODULES_AVAILABLE:
            self.skipTest("Plugin modules not available")
    
    def test_large_dataset_loading(self):
        """Test loading performance with large datasets."""
        import time
        
        # Create large dataset
        large_data = generate_sample_data(n_tracks=1000, n_frames=500)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        large_data.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        try:
            # Time the loading process
            start_time = time.time()
            
            data_manager = TrackingDataManager()
            success = data_manager.load_data(temp_file.name)
            
            end_time = time.time()
            loading_time = end_time - start_time
            
            self.assertTrue(success)
            self.assertLess(loading_time, 30.0)  # Should load within 30 seconds
            
            print(f"Large dataset loading time: {loading_time:.2f} seconds")
            print(f"Data size: {len(large_data)} rows")
            
        finally:
            os.unlink(temp_file.name)
    
    def test_memory_usage(self):
        """Test memory usage with large datasets."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process large dataset
        large_data = generate_sample_data(n_tracks=500, n_frames=1000)
        
        data_manager = TrackingDataManager()
        data_manager.data = large_data
        data_manager._process_data_types()
        data_manager._add_derived_columns()
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del data_manager
        del large_data
        gc.collect()
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 1000)  # Less than 1GB increase


def run_test_suite():
    """Run the complete test suite."""
    print("🧪 Running Tracking Results Plotter Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestTrackingDataManager,
        TestDataValidator,
        TestColorManager,
        TestMathUtils,
        TestExportManager,
        TestAdvancedPlotter,
        TestIntegration,
        TestPerformance
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Suite Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")
    
    return success


def run_specific_test(test_name: str):
    """Run a specific test class or method."""
    test_mapping = {
        'data': TestTrackingDataManager,
        'validator': TestDataValidator,
        'color': TestColorManager,
        'math': TestMathUtils,
        'export': TestExportManager,
        'plots': TestAdvancedPlotter,
        'integration': TestIntegration,
        'performance': TestPerformance
    }
    
    if test_name in test_mapping:
        test_class = test_mapping[test_name]
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {list(test_mapping.keys())}")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name == 'all':
            run_test_suite()
        else:
            run_specific_test(test_name)
    else:
        run_test_suite()