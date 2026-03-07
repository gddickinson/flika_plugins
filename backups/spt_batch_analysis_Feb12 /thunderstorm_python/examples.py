"""
Comprehensive ThunderSTORM Python Example
==========================================

This script demonstrates all major features of the thunderSTORM Python package.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import thunderSTORM components
from thunderstorm_python import ThunderSTORM, create_default_pipeline, quick_analysis
from thunderstorm_python.simulation import SMLMSimulator, PerformanceEvaluator, create_test_pattern
from thunderstorm_python import visualization, utils


def example_1_quick_analysis():
    """Example 1: Quick analysis with simulated data."""
    print("\n" + "="*60)
    print("Example 1: Quick Analysis")
    print("="*60)
    
    # Create simulated data
    print("Generating simulated SMLM data...")
    simulator = SMLMSimulator(
        image_size=(128, 128),
        pixel_size=100.0,
        psf_sigma=150.0,
        photons_per_molecule=800,
        background_photons=20
    )
    
    # Create Siemens star pattern
    pattern = create_test_pattern('siemens_star', size=128)
    
    # Generate 100 frames
    movie, ground_truth = simulator.generate_movie(
        n_frames=100,
        mask=pattern,
        blinking=True
    )
    
    print(f"Generated {len(movie)} frames")
    
    # Quick analysis
    print("Running analysis...")
    localizations, sr_image, pipeline = quick_analysis(movie)
    
    print(f"Detected {len(localizations['x'])} molecules")
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(movie[0], cmap='gray')
    axes[0].set_title('Raw Frame 0')
    axes[0].axis('off')
    
    axes[1].imshow(np.mean(movie, axis=0), cmap='gray')
    axes[1].set_title('Average Projection')
    axes[1].axis('off')
    
    axes[2].imshow(sr_image, cmap='hot')
    axes[2].set_title('Super-Resolution')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('example1_quick_analysis.png', dpi=150)
    print("Saved: example1_quick_analysis.png")


def example_2_custom_pipeline():
    """Example 2: Custom analysis pipeline with all options."""
    print("\n" + "="*60)
    print("Example 2: Custom Pipeline")
    print("="*60)
    
    # Generate data
    simulator = SMLMSimulator(
        image_size=(256, 256),
        pixel_size=100.0
    )
    
    positions = simulator.generate_molecule_positions(n_molecules=500)
    movie, ground_truth = simulator.generate_movie(
        n_frames=200,
        molecule_positions=positions,
        blinking=True
    )
    
    # Create custom pipeline
    print("Creating custom pipeline...")
    pipeline = ThunderSTORM(
        filter_type='wavelet',
        filter_params={'scale': 2, 'order': 3},
        detector_type='local_maximum',
        detector_params={'connectivity': '8-neighbourhood'},
        fitter_type='gaussian_mle',
        fitter_params={'integrated': True, 'elliptical': False},
        threshold_expression='1.5*std(Wave.F1)',
        pixel_size=100.0,
        photons_per_adu=1.0,
        baseline=0.0
    )
    
    # Analyze
    print("Analyzing stack...")
    localizations = pipeline.analyze_stack(movie, fit_radius=3)
    print(f"Initial detections: {len(localizations['x'])}")
    
    # Post-processing
    print("Applying post-processing...")
    
    # 1. Drift correction
    localizations = pipeline.apply_drift_correction(method='cross_correlation')
    
    # 2. Merge reappearing molecules
    localizations = pipeline.merge_molecules(max_distance=50, max_frame_gap=2)
    print(f"After merging: {len(localizations['x'])}")
    
    # 3. Filter by quality
    localizations = pipeline.filter_localizations(
        min_intensity=300,
        max_uncertainty=50
    )
    print(f"After filtering: {len(localizations['x'])}")
    
    # 4. Remove isolated localizations
    localizations = pipeline.filter_by_density(radius=100, min_neighbors=3)
    print(f"After density filter: {len(localizations['x'])}")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"\nStatistics:")
    print(f"  Mean intensity: {stats['mean_intensity']:.1f} photons")
    print(f"  Mean uncertainty: {stats['mean_uncertainty']:.2f} nm")
    print(f"  Spatial extent: {stats['x_range']:.0f} x {stats['y_range']:.0f} nm")
    
    # Render
    sr_image = pipeline.render(renderer_type='gaussian', pixel_size=10)
    
    # Save
    pipeline.save('example2_localizations.csv')
    print("\nSaved: example2_localizations.csv")
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(movie[0], cmap='gray')
    plt.title('Raw Data')
    
    plt.subplot(122)
    plt.imshow(sr_image, cmap='hot')
    plt.title(f'Super-Resolution\n({len(localizations["x"])} molecules)')
    
    plt.tight_layout()
    plt.savefig('example2_custom_pipeline.png', dpi=150)
    print("Saved: example2_custom_pipeline.png")


def example_3_renderer_comparison():
    """Example 3: Compare different rendering methods."""
    print("\n" + "="*60)
    print("Example 3: Renderer Comparison")
    print("="*60)
    
    # Generate data
    simulator = SMLMSimulator()
    pattern = create_test_pattern('grid', size=128)
    movie, _ = simulator.generate_movie(n_frames=500, mask=pattern)
    
    # Analyze
    pipeline = create_default_pipeline()
    localizations = pipeline.analyze_stack(movie, show_progress=False)
    
    print(f"Detected {len(localizations['x'])} molecules")
    
    # Create different renderers
    renderers = {
        'Gaussian (σ=20nm)': visualization.GaussianRenderer(sigma=20),
        'Gaussian (σ=computed)': visualization.GaussianRenderer(sigma='computed'),
        'Histogram': visualization.HistogramRenderer(jittering=False),
        'Jittered Histogram': visualization.HistogramRenderer(jittering=True, n_averages=10),
        'ASH (n=2)': visualization.AverageShiftedHistogram(n_shifts=2),
        'ASH (n=8)': visualization.AverageShiftedHistogram(n_shifts=8),
    }
    
    # Render with each method
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, renderer) in enumerate(renderers.items()):
        print(f"Rendering with {name}...")
        img = renderer.render(localizations, pixel_size=10)
        axes[i].imshow(img, cmap='hot')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('example3_renderers.png', dpi=150)
    print("Saved: example3_renderers.png")


def example_4_performance_evaluation():
    """Example 4: Evaluate algorithm performance with ground truth."""
    print("\n" + "="*60)
    print("Example 4: Performance Evaluation")
    print("="*60)
    
    # Generate data with known ground truth
    simulator = SMLMSimulator(
        photons_per_molecule=1000,
        background_photons=15
    )
    
    positions = simulator.generate_molecule_positions(n_molecules=200)
    movie, ground_truth_list = simulator.generate_movie(
        n_frames=50,
        molecule_positions=positions,
        blinking=True
    )
    
    # Analyze
    pipeline = create_default_pipeline()
    localizations = pipeline.analyze_stack(movie, show_progress=False)
    
    # Evaluate performance
    evaluator = PerformanceEvaluator(tolerance=100.0)  # 100 nm tolerance
    
    # Aggregate ground truth
    gt_all = {
        'x': np.concatenate([gt['x'] for gt in ground_truth_list]),
        'y': np.concatenate([gt['y'] for gt in ground_truth_list])
    }
    
    metrics = evaluator.evaluate(localizations, gt_all)
    
    print(f"\nPerformance Metrics:")
    print(f"  True Positives:  {metrics['n_true_positive']}")
    print(f"  False Positives: {metrics['n_false_positive']}")
    print(f"  False Negatives: {metrics['n_false_negative']}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  F1 Score:  {metrics['f1_score']:.3f}")
    print(f"  Jaccard:   {metrics['jaccard']:.3f}")
    print(f"  RMSE:      {metrics['rmse']:.2f} nm")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth
    renderer = visualization.ScatterRenderer()
    gt_img = renderer.render(gt_all, pixel_size=10)
    axes[0].imshow(gt_img, cmap='gray')
    axes[0].set_title(f'Ground Truth\n({len(gt_all["x"])} molecules)')
    
    # Detected
    det_img = renderer.render(localizations, pixel_size=10)
    axes[1].imshow(det_img, cmap='gray')
    axes[1].set_title(f'Detected\n({len(localizations["x"])} molecules)')
    
    plt.tight_layout()
    plt.savefig('example4_performance.png', dpi=150)
    print("\nSaved: example4_performance.png")


def example_5_analysis_plots():
    """Example 5: Generate analysis plots."""
    print("\n" + "="*60)
    print("Example 5: Analysis Plots")
    print("="*60)
    
    # Generate and analyze data
    simulator = SMLMSimulator()
    movie, _ = simulator.generate_movie(n_frames=500)
    
    pipeline = create_default_pipeline()
    localizations = pipeline.analyze_stack(movie, show_progress=False)
    
    print(f"Analyzing {len(localizations['x'])} localizations...")
    
    # Create comprehensive analysis figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Localization precision histogram
    ax1 = plt.subplot(2, 3, 1)
    plt.hist(localizations['uncertainty'], bins=50, edgecolor='black')
    plt.xlabel('Localization Uncertainty (nm)')
    plt.ylabel('Count')
    plt.title('Localization Precision')
    
    # 2. Intensity histogram
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(localizations['intensity'], bins=50, edgecolor='black')
    plt.xlabel('Intensity (photons)')
    plt.ylabel('Count')
    plt.title('Photon Counts')
    
    # 3. PSF sigma histogram
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(localizations['sigma_x'], bins=50, edgecolor='black')
    plt.xlabel('PSF Sigma (pixels)')
    plt.ylabel('Count')
    plt.title('PSF Width Distribution')
    
    # 4. Intensity vs precision scatter
    ax4 = plt.subplot(2, 3, 4)
    plt.scatter(localizations['intensity'], localizations['uncertainty'], 
                alpha=0.3, s=1)
    plt.xlabel('Intensity (photons)')
    plt.ylabel('Uncertainty (nm)')
    plt.title('Intensity vs Precision')
    
    # 5. Localizations per frame
    ax5 = plt.subplot(2, 3, 5)
    frames, counts = np.unique(localizations['frame'], return_counts=True)
    plt.plot(frames, counts)
    plt.xlabel('Frame Number')
    plt.ylabel('Localizations')
    plt.title('Detections per Frame')
    
    # 6. Spatial density map
    ax6 = plt.subplot(2, 3, 6)
    density_map, extent = utils.compute_localization_density(
        localizations, pixel_size=100
    )
    plt.imshow(density_map, cmap='hot', extent=extent)
    plt.colorbar(label='Localizations per pixel')
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.title('Spatial Density')
    
    plt.tight_layout()
    plt.savefig('example5_analysis_plots.png', dpi=150)
    print("Saved: example5_analysis_plots.png")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("ThunderSTORM Python - Comprehensive Examples")
    print("="*60)
    
    # Create output directory
    output_dir = Path('thunderstorm_examples')
    output_dir.mkdir(exist_ok=True)
    
    # Run examples
    try:
        example_1_quick_analysis()
        example_2_custom_pipeline()
        example_3_renderer_comparison()
        example_4_performance_evaluation()
        example_5_analysis_plots()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
