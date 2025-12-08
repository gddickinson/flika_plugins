"""
Parallel Processing Module for ThunderSTORM
Significantly speeds up SMLM analysis by processing frames in parallel

Author: George K (with Claude)
Date: 2025-12-08
"""

import numpy as np
from typing import Optional, Dict, Any
import time
from functools import partial


def analyze_stack_parallel_joblib(pipeline, images, fit_radius=3, show_progress=True, n_jobs=-1):
    """
    Parallel version of analyze_stack using joblib (RECOMMENDED)

    Joblib is better than multiprocessing for NumPy-heavy operations because:
    - Efficient array serialization
    - Better memory management
    - Works well with NumPy's internal threading

    Parameters
    ----------
    pipeline : ThunderSTORM
        The analysis pipeline
    images : ndarray
        Image stack (frames, height, width)
    fit_radius : int
        Fitting radius in pixels
    show_progress : bool
        Show tqdm progress bar
    n_jobs : int
        Number of parallel jobs (-1 = all cores, -2 = all but one, >0 = specific count)
        NOTE: Cannot be 0!

    Returns
    -------
    localizations : dict
        Combined localization results
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError("Install joblib: pip install joblib")

    # Validate n_jobs
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0. Use -1 (all cores), -2 (all but one), or a positive integer.")

    # Ensure 3D
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    pipeline.images = images
    n_frames = images.shape[0]

    print(f"Processing {n_frames} frames using {n_jobs} parallel jobs...")
    start_time = time.time()

    # Create a wrapper function that includes all necessary data
    def analyze_single_frame(frame_idx, image, pipeline_state, fit_radius):
        """Analyze a single frame - this runs in parallel"""
        # Reconstruct the minimal pipeline components needed
        # Note: We pass pipeline_state to avoid pickling the whole pipeline
        result = pipeline.analyze_frame(image, frame_number=frame_idx, fit_radius=fit_radius)
        return result

    # Process frames in parallel
    if show_progress:
        try:
            from tqdm import tqdm

            # Use joblib with tqdm
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(analyze_single_frame)(i, images[i], None, fit_radius)
                for i in tqdm(range(n_frames), desc='Analyzing frames (parallel)')
            )
        except ImportError:
            all_results = Parallel(n_jobs=n_jobs)(
                delayed(analyze_single_frame)(i, images[i], None, fit_radius)
                for i in range(n_frames)
            )
    else:
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(analyze_single_frame)(i, images[i], None, fit_radius)
            for i in range(n_frames)
        )

    elapsed = time.time() - start_time
    print(f"Analysis complete in {elapsed:.1f}s ({n_frames/elapsed:.1f} frames/sec)")

    # Combine results (this is fast, no need to parallelize)
    pipeline.localizations = pipeline._combine_results(all_results)

    return pipeline.localizations


def analyze_stack_parallel_multiprocessing(pipeline, images, fit_radius=3, show_progress=True, n_processes=None):
    """
    Parallel version using multiprocessing (alternative to joblib)

    Use this if you don't want to install joblib, but joblib is generally better.

    Parameters
    ----------
    pipeline : ThunderSTORM
        The analysis pipeline
    images : ndarray
        Image stack
    fit_radius : int
        Fitting radius
    show_progress : bool
        Show progress
    n_processes : int, optional
        Number of processes (None = CPU count)

    Returns
    -------
    localizations : dict
    """
    from multiprocessing import Pool, cpu_count

    # Ensure 3D
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    pipeline.images = images
    n_frames = images.shape[0]

    if n_processes is None:
        n_processes = cpu_count()

    print(f"Processing {n_frames} frames using {n_processes} processes...")
    start_time = time.time()

    # Create partial function with fixed parameters
    analyze_func = partial(_analyze_frame_wrapper,
                          pipeline=pipeline,
                          fit_radius=fit_radius)

    # Process in parallel
    with Pool(processes=n_processes) as pool:
        if show_progress:
            try:
                from tqdm import tqdm
                all_results = list(tqdm(
                    pool.imap(analyze_func, enumerate(images)),
                    total=n_frames,
                    desc='Analyzing frames (parallel)'
                ))
            except ImportError:
                all_results = pool.map(analyze_func, enumerate(images))
        else:
            all_results = pool.map(analyze_func, enumerate(images))

    elapsed = time.time() - start_time
    print(f"Analysis complete in {elapsed:.1f}s ({n_frames/elapsed:.1f} frames/sec)")

    # Combine results
    pipeline.localizations = pipeline._combine_results(all_results)

    return pipeline.localizations


def _analyze_frame_wrapper(frame_data, pipeline, fit_radius):
    """Wrapper function for multiprocessing"""
    frame_idx, image = frame_data
    return pipeline.analyze_frame(image, frame_number=frame_idx, fit_radius=fit_radius)


def analyze_stack_batch(pipeline, images, fit_radius=3, show_progress=True, batch_size=100, n_jobs=-1):
    """
    Parallel processing with batching for very large datasets

    Processes frames in batches to manage memory efficiently.
    Useful for 10,000+ frame datasets.

    Parameters
    ----------
    pipeline : ThunderSTORM
        Analysis pipeline
    images : ndarray
        Image stack
    fit_radius : int
        Fitting radius
    show_progress : bool
        Show progress
    batch_size : int
        Number of frames per batch (adjust based on RAM)
    n_jobs : int
        Parallel jobs per batch

    Returns
    -------
    localizations : dict
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError("Install joblib: pip install joblib")

    # Ensure 3D
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    pipeline.images = images
    n_frames = images.shape[0]
    n_batches = (n_frames + batch_size - 1) // batch_size

    print(f"Processing {n_frames} frames in {n_batches} batches of {batch_size}...")
    start_time = time.time()

    all_results = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_frames)
        batch_images = images[start_idx:end_idx]

        if show_progress:
            print(f"  Batch {batch_idx+1}/{n_batches} (frames {start_idx}-{end_idx-1})...")

        # Process batch in parallel
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(pipeline.analyze_frame)(batch_images[i],
                                           frame_number=start_idx + i,
                                           fit_radius=fit_radius)
            for i in range(len(batch_images))
        )

        all_results.extend(batch_results)

    elapsed = time.time() - start_time
    print(f"Analysis complete in {elapsed:.1f}s ({n_frames/elapsed:.1f} frames/sec)")

    # Combine results
    pipeline.localizations = pipeline._combine_results(all_results)

    return pipeline.localizations


def profile_analysis(pipeline, images, fit_radius=3, n_frames_test=10):
    """
    Profile the analysis to identify bottlenecks

    Analyzes a subset of frames and reports timing for each step.

    Parameters
    ----------
    pipeline : ThunderSTORM
        Analysis pipeline
    images : ndarray
        Image stack
    fit_radius : int
        Fitting radius
    n_frames_test : int
        Number of frames to profile

    Returns
    -------
    profile_results : dict
        Timing information for each processing step
    """
    import time

    # Ensure 3D
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    n_frames_test = min(n_frames_test, images.shape[0])
    test_images = images[:n_frames_test]

    print(f"Profiling analysis on {n_frames_test} frames...")
    print("=" * 60)

    # Time individual components
    timings = {
        'filter': [],
        'detect': [],
        'fit': [],
        'total': []
    }

    for i in range(n_frames_test):
        image = test_images[i]

        # Filter
        t0 = time.time()
        filtered = pipeline.filter.filter(image)
        t1 = time.time()
        timings['filter'].append(t1 - t0)

        # Detect
        threshold = pipeline.compute_threshold(filtered)
        detections = pipeline.detector.detect(filtered, threshold)
        t2 = time.time()
        timings['detect'].append(t2 - t1)

        # Fit
        if len(detections[0]) > 0:
            fits = pipeline.fitter.fit(image, detections, fit_radius=fit_radius)
        t3 = time.time()
        timings['fit'].append(t3 - t2)

        timings['total'].append(t3 - t0)

    # Report statistics
    print(f"\nAverage timing per frame ({n_frames_test} frames):")
    print(f"  Filtering:  {np.mean(timings['filter'])*1000:.1f} ms ± {np.std(timings['filter'])*1000:.1f} ms")
    print(f"  Detection:  {np.mean(timings['detect'])*1000:.1f} ms ± {np.std(timings['detect'])*1000:.1f} ms")
    print(f"  Fitting:    {np.mean(timings['fit'])*1000:.1f} ms ± {np.std(timings['fit'])*1000:.1f} ms")
    print(f"  Total:      {np.mean(timings['total'])*1000:.1f} ms ± {np.std(timings['total'])*1000:.1f} ms")
    print(f"\nEstimated time for full dataset:")
    total_frames = images.shape[0]
    estimated = np.mean(timings['total']) * total_frames
    print(f"  Sequential: {estimated:.1f} s ({estimated/60:.1f} min)")

    # Estimate parallel speedup
    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    estimated_parallel = estimated / (n_cores * 0.7)  # 70% efficiency estimate
    print(f"  Parallel ({n_cores} cores): {estimated_parallel:.1f} s ({estimated_parallel/60:.1f} min)")
    print(f"  Expected speedup: {estimated/estimated_parallel:.1f}x")
    print("=" * 60)

    return timings


# Convenience function
def get_optimal_n_jobs():
    """
    Determine optimal number of parallel jobs

    Returns
    -------
    n_jobs : int
        Recommended number of jobs (-1 = all cores, -2 = all but one)
    """
    import multiprocessing
    n_cores = multiprocessing.cpu_count()

    if n_cores <= 2:
        return -1  # Use all cores
    else:
        return -2  # Leave one core free for system
