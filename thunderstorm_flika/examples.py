"""
ThunderSTORM for FLIKA - Example Usage Scripts
================================================

This file contains example scripts demonstrating how to use the
ThunderSTORM FLIKA plugin programmatically.
"""

import numpy as np
from pathlib import Path

# Import FLIKA
from flika import start_flika, global_vars as g
from flika.window import Window
from flika.process.file_ import open_file

# Import ThunderSTORM components
from thunderstorm_flika import (
    ThunderSTORM_RunAnalysis,
    ThunderSTORM_PostProcessing,
    ThunderSTORM_DriftCorrection,
    ThunderSTORM_Rendering,
    quick_analysis_simple,
    simulate_smlm_data
)


# ============================================================================
# Example 1: Quick Analysis
# ============================================================================

def example_1_quick_analysis():
    """
    Simplest way to analyze SMLM data.
    Uses default parameters for fast analysis.
    """
    print("\n" + "="*60)
    print("Example 1: Quick Analysis")
    print("="*60)
    
    # Start FLIKA if not already running
    # start_flika()
    
    # Open your SMLM data
    # Replace with your actual file path
    window = open_file('path/to/your/smlm_data.tif')
    
    # Run quick analysis
    result = quick_analysis_simple()
    
    print("Quick analysis complete!")
    print(f"Result window: {result.name}")


# ============================================================================
# Example 2: Custom Analysis with GUI
# ============================================================================

def example_2_custom_analysis_gui():
    """
    Open the full analysis GUI for customization.
    """
    print("\n" + "="*60)
    print("Example 2: Custom Analysis with GUI")
    print("="*60)
    
    # Open data
    window = open_file('path/to/your/smlm_data.tif')
    
    # Create and show analysis dialog
    analysis = ThunderSTORM_RunAnalysis()
    analysis.show()
    
    # User interacts with GUI, then clicks Run
    print("GUI opened. Configure parameters and click 'Run'.")


# ============================================================================
# Example 3: Programmatic Analysis (No GUI)
# ============================================================================

def example_3_programmatic_analysis():
    """
    Run analysis programmatically without GUI.
    Full control over all parameters.
    """
    print("\n" + "="*60)
    print("Example 3: Programmatic Analysis")
    print("="*60)
    
    # Import thunderSTORM directly
    from thunderstorm_python import ThunderSTORM
    
    # Load data
    window = open_file('path/to/your/smlm_data.tif')
    image_stack = window.image
    
    # Ensure 3D stack
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis, ...]
    
    # Create custom pipeline
    pipeline = ThunderSTORM(
        # Filtering
        filter_type='wavelet',
        filter_params={'scale': 2, 'order': 3},
        
        # Detection
        detector_type='local_maximum',
        detector_params={'connectivity': '8-neighbourhood'},
        
        # Fitting
        fitter_type='gaussian_mle',  # Use MLE for best quality
        fitter_params={
            'initial_sigma': 1.3,
            'integrated': True,
            'elliptical': False
        },
        
        # Threshold
        threshold_expression='2*std(Wave.F1)',
        
        # Camera parameters
        pixel_size=100.0,  # nm
        photons_per_adu=0.45,
        baseline=100.0,
        em_gain=1.0
    )
    
    # Analyze
    print(f"Analyzing {len(image_stack)} frames...")
    localizations = pipeline.analyze_stack(image_stack, fit_radius=4)
    
    print(f"Detected {len(localizations['x'])} molecules")
    
    # Post-processing
    print("Applying drift correction...")
    localizations = pipeline.apply_drift_correction(method='cross_correlation')
    
    print("Merging blinking molecules...")
    localizations = pipeline.merge_molecules(max_distance=50, max_frame_gap=2)
    
    print("Filtering by quality...")
    localizations = pipeline.filter_localizations(
        min_intensity=500,
        max_uncertainty=30
    )
    
    print(f"Final count: {len(localizations['x'])} molecules")
    
    # Render
    print("Rendering super-resolution image...")
    sr_image = pipeline.render(renderer_type='gaussian', pixel_size=10)
    
    # Create window
    result_window = Window(sr_image, name="SR_programmatic")
    
    # Save results
    pipeline.save('localizations_programmatic.csv')
    print("Saved localizations to: localizations_programmatic.csv")
    
    return result_window


# ============================================================================
# Example 4: Batch Processing
# ============================================================================

def example_4_batch_processing():
    """
    Process multiple SMLM movies automatically.
    """
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    from thunderstorm_python import ThunderSTORM, utils
    
    # Input and output directories
    data_dir = Path('data/smlm_movies/')
    output_dir = Path('results/')
    output_dir.mkdir(exist_ok=True)
    
    # Create pipeline with consistent parameters
    pipeline = ThunderSTORM(
        filter_type='wavelet',
        detector_type='local_maximum',
        fitter_type='gaussian_lsq',
        threshold_expression='std(Wave.F1)',
        pixel_size=100.0
    )
    
    # Process all TIFF files
    tif_files = list(data_dir.glob('*.tif'))
    print(f"Found {len(tif_files)} files to process")
    
    for i, tif_file in enumerate(tif_files, 1):
        print(f"\n[{i}/{len(tif_files)}] Processing {tif_file.name}...")
        
        try:
            # Load
            window = open_file(str(tif_file))
            image_stack = window.image
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]
            
            # Analyze
            localizations = pipeline.analyze_stack(image_stack, show_progress=False)
            print(f"  Detected {len(localizations['x'])} molecules")
            
            # Post-process
            localizations = pipeline.filter_localizations(min_intensity=300)
            print(f"  After filtering: {len(localizations['x'])} molecules")
            
            # Save localizations
            output_csv = output_dir / f"{tif_file.stem}_localizations.csv"
            pipeline.save(output_csv)
            
            # Render and save image
            sr_image = pipeline.render(renderer_type='gaussian', pixel_size=10)
            output_tif = output_dir / f"{tif_file.stem}_SR.tif"
            utils.save_image_stack(sr_image[np.newaxis, ...], output_tif)
            
            print(f"  Saved: {output_csv.name} and {output_tif.name}")
            
            # Close window
            window.close()
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\nBatch processing complete! Results in {output_dir}")


# ============================================================================
# Example 5: Simulation and Performance Evaluation
# ============================================================================

def example_5_simulation_and_evaluation():
    """
    Generate simulated data, analyze it, and evaluate performance.
    """
    print("\n" + "="*60)
    print("Example 5: Simulation and Performance Evaluation")
    print("="*60)
    
    from thunderstorm_python.simulation import (
        SMLMSimulator, PerformanceEvaluator, create_test_pattern
    )
    from thunderstorm_python import ThunderSTORM
    
    # Create simulator
    simulator = SMLMSimulator(
        image_size=(256, 256),
        pixel_size=100.0,
        psf_sigma=150.0,
        photons_per_molecule=1000,
        background_photons=15
    )
    
    # Create test pattern
    pattern = create_test_pattern('siemens_star', size=256)
    
    # Generate movie
    print("Generating simulated movie...")
    n_frames = 100
    movie, ground_truth = simulator.generate_movie(
        n_frames=n_frames,
        mask=pattern,
        blinking=True
    )
    
    print(f"Generated {n_frames} frames")
    
    # Analyze
    print("Analyzing simulated data...")
    pipeline = ThunderSTORM()
    localizations = pipeline.analyze_stack(movie, show_progress=False)
    
    print(f"Detected {len(localizations['x'])} molecules")
    
    # Evaluate performance
    print("Evaluating performance...")
    evaluator = PerformanceEvaluator(tolerance=100.0)  # 100 nm tolerance
    
    # Combine ground truth from all frames
    gt_combined = {
        'x': np.concatenate([gt['x'] for gt in ground_truth]),
        'y': np.concatenate([gt['y'] for gt in ground_truth])
    }
    
    metrics = evaluator.evaluate(localizations, gt_combined)
    
    # Print results
    print("\nPerformance Metrics:")
    print(f"  True Positives:  {metrics['n_true_positive']}")
    print(f"  False Positives: {metrics['n_false_positive']}")
    print(f"  False Negatives: {metrics['n_false_negative']}")
    print(f"  Recall:          {metrics['recall']:.3f}")
    print(f"  Precision:       {metrics['precision']:.3f}")
    print(f"  F1 Score:        {metrics['f1_score']:.3f}")
    print(f"  Jaccard Index:   {metrics['jaccard']:.3f}")
    print(f"  RMSE:            {metrics['rmse']:.2f} nm")
    
    # Create windows
    Window(movie[0], name="Simulated_frame0")
    sr_image = pipeline.render(renderer_type='gaussian', pixel_size=10)
    Window(sr_image, name="Simulated_SR")


# ============================================================================
# Example 6: Renderer Comparison
# ============================================================================

def example_6_renderer_comparison():
    """
    Compare different rendering methods on the same data.
    """
    print("\n" + "="*60)
    print("Example 6: Renderer Comparison")
    print("="*60)
    
    from thunderstorm_python import visualization, utils
    
    # Load localizations
    localizations = utils.load_localizations_csv('localizations.csv')
    print(f"Loaded {len(localizations['x'])} localizations")
    
    # Create different renderers
    renderers = {
        'Gaussian (σ=20nm)': visualization.GaussianRenderer(sigma=20),
        'Gaussian (computed)': visualization.GaussianRenderer(sigma='computed'),
        'Histogram': visualization.HistogramRenderer(jittering=False),
        'Jittered Histogram': visualization.HistogramRenderer(jittering=True, n_averages=10),
        'ASH (n=4)': visualization.AverageShiftedHistogram(n_shifts=4),
        'ASH (n=8)': visualization.AverageShiftedHistogram(n_shifts=8),
        'Scatter': visualization.ScatterRenderer()
    }
    
    # Render with each method
    pixel_size = 10  # nm
    
    for name, renderer in renderers.items():
        print(f"Rendering with {name}...")
        img = renderer.render(localizations, pixel_size=pixel_size)
        Window(img, name=f"Render_{name}")
    
    print("Comparison complete! Check the windows.")


# ============================================================================
# Example 7: 3D Analysis (if applicable)
# ============================================================================

def example_7_3d_analysis():
    """
    Analyze 3D SMLM data using astigmatism.
    Note: Requires calibration data.
    """
    print("\n" + "="*60)
    print("Example 7: 3D Analysis (Astigmatism)")
    print("="*60)
    
    from thunderstorm_python import ThunderSTORM
    
    # Load 3D data
    window = open_file('path/to/3d_smlm_data.tif')
    image_stack = window.image
    
    # Create pipeline for 3D
    pipeline = ThunderSTORM(
        filter_type='wavelet',
        detector_type='local_maximum',
        fitter_type='gaussian_lsq',
        fitter_params={
            'elliptical': True,  # Required for 3D
            'initial_sigma': 1.3
        },
        threshold_expression='std(Wave.F1)',
        pixel_size=100.0
    )
    
    # Analyze
    print("Analyzing 3D data...")
    localizations = pipeline.analyze_stack(image_stack)
    
    print(f"Detected {len(localizations['x'])} molecules")
    print(f"Sigma X range: {localizations['sigma_x'].min():.2f} - {localizations['sigma_x'].max():.2f}")
    print(f"Sigma Y range: {localizations['sigma_y'].min():.2f} - {localizations['sigma_y'].max():.2f}")
    
    # Note: For actual Z estimation, you need calibration data
    # See thunderSTORM documentation for calibration procedure
    
    print("\nNote: For Z position estimation, calibration is required.")
    print("See thunderSTORM documentation for calibration procedure.")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    """
    Run examples.
    
    Uncomment the example you want to run.
    """
    
    print("\nThunderSTORM for FLIKA - Example Scripts")
    print("=========================================\n")
    
    # Start FLIKA
    # start_flika()
    
    # Choose an example to run:
    
    # example_1_quick_analysis()
    # example_2_custom_analysis_gui()
    # example_3_programmatic_analysis()
    # example_4_batch_processing()
    # example_5_simulation_and_evaluation()
    # example_6_renderer_comparison()
    # example_7_3d_analysis()
    
    print("\nDone! Check the FLIKA windows for results.")
