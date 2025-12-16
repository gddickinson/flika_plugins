"""
Correlation Analysis Module
============================
Local correlation image computation for neuron/ROI detection in calcium imaging.

Based on CaImAn and Suite2p correlation image methodology.

Author: George
"""

import numpy as np


def local_correlation_image(movie, neighborhood=1):
    """
    Compute local correlation image for neuron/ROI detection.
    
    For each pixel, computes average Pearson correlation with its spatial neighbors
    across time. High correlation indicates co-active regions (neurons, dendrites).
    
    Algorithm:
    1. For each pixel, extract its time series and neighbor time series
    2. Compute Pearson correlation between center and each neighbor
    3. Average correlations across all neighbors
    
    Parameters:
        movie (ndarray): Input movie (T, H, W)
        neighborhood (int): Radius of neighborhood (1 = 3x3, 2 = 5x5, etc.)
    
    Returns:
        ndarray: Local correlation image (H, W) with values in [-1, 1]
    
    Example:
        >>> import numpy as np
        >>> # Create synthetic data with a correlated region
        >>> T, H, W = 500, 64, 64
        >>> movie = np.random.randn(T, H, W)
        >>> # Add correlated signal to center region
        >>> signal = np.random.randn(T)
        >>> movie[:, 30:35, 30:35] += signal[:, np.newaxis, np.newaxis]
        >>> corr_img = local_correlation_image(movie, neighborhood=1)
        >>> print(f"Max correlation at center: {corr_img[32, 32]:.3f}")
    
    Notes:
        - Pixels with high correlation indicate potential ROIs
        - Computation can be slow for large movies
        - Edge pixels have fewer neighbors (use valid data only)
        - Typical neighborhood=1 (3x3) or neighborhood=2 (5x5)
    """
    T, H, W = movie.shape
    corr_img = np.zeros((H, W), dtype=np.float32)
    
    # Process each pixel (excluding edges)
    for y in range(neighborhood, H - neighborhood):
        for x in range(neighborhood, W - neighborhood):
            # Get center pixel time series
            center = movie[:, y, x]
            center_mean = np.mean(center)
            center_std = np.std(center)
            
            if center_std < 1e-10:  # Skip pixels with no variance
                continue
            
            # Compute correlation with all neighbors
            correlations = []
            
            for dy in range(-neighborhood, neighborhood + 1):
                for dx in range(-neighborhood, neighborhood + 1):
                    if dy == 0 and dx == 0:  # Skip center pixel
                        continue
                    
                    # Get neighbor time series
                    neighbor = movie[:, y + dy, x + dx]
                    neighbor_mean = np.mean(neighbor)
                    neighbor_std = np.std(neighbor)
                    
                    if neighbor_std < 1e-10:  # Skip neighbors with no variance
                        continue
                    
                    # Compute Pearson correlation
                    corr = np.sum((center - center_mean) * (neighbor - neighbor_mean))
                    corr /= (T * center_std * neighbor_std + 1e-10)
                    correlations.append(corr)
            
            # Average correlation across neighbors
            if len(correlations) > 0:
                corr_img[y, x] = np.mean(correlations)
    
    return corr_img


def local_correlation_image_efficient(movie, neighborhood=1):
    """
    Efficient implementation of local correlation image using vectorized operations.
    
    Much faster than the naive implementation for large movies.
    
    Parameters:
        movie (ndarray): Input movie (T, H, W)
        neighborhood (int): Radius of neighborhood
    
    Returns:
        ndarray: Local correlation image (H, W)
    
    Example:
        >>> import numpy as np
        >>> movie = np.random.randn(500, 64, 64)
        >>> corr_img = local_correlation_image_efficient(movie, neighborhood=1)
        >>> print(corr_img.shape)
        (64, 64)
    """
    T, H, W = movie.shape
    
    # Normalize movie (z-score for each pixel)
    movie_norm = movie - np.mean(movie, axis=0, keepdims=True)
    stds = np.std(movie, axis=0, keepdims=True)
    stds[stds < 1e-10] = 1.0  # Avoid division by zero
    movie_norm = movie_norm / stds
    
    corr_img = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    
    # Iterate over neighbor offsets
    for dy in range(-neighborhood, neighborhood + 1):
        for dx in range(-neighborhood, neighborhood + 1):
            if dy == 0 and dx == 0:  # Skip center
                continue
            
            # Determine valid overlap region
            y_start = max(0, -dy)
            y_end = min(H, H - dy)
            x_start = max(0, -dx)
            x_end = min(W, W - dx)
            
            # Center pixels
            center = movie_norm[:, y_start:y_end, x_start:x_end]
            
            # Neighbor pixels (shifted)
            neighbor = movie_norm[:, y_start+dy:y_end+dy, x_start+dx:x_end+dx]
            
            # Compute correlation (already normalized)
            corr = np.mean(center * neighbor, axis=0)
            
            # Accumulate
            corr_img[y_start:y_end, x_start:x_end] += corr
            count[y_start:y_end, x_start:x_end] += 1
    
    # Average over neighbors
    count[count == 0] = 1  # Avoid division by zero
    corr_img = corr_img / count
    
    return corr_img


def compute_max_correlation_image(movie, neighborhood=1):
    """
    Compute maximum correlation image (max correlation with any neighbor).
    
    Alternative to average correlation that may be more sensitive to
    detecting edges of ROIs.
    
    Parameters:
        movie (ndarray): Input movie (T, H, W)
        neighborhood (int): Radius of neighborhood
    
    Returns:
        ndarray: Maximum correlation image (H, W)
    
    Example:
        >>> import numpy as np
        >>> movie = np.random.randn(500, 64, 64)
        >>> max_corr = compute_max_correlation_image(movie)
        >>> print(max_corr.shape)
        (64, 64)
    """
    T, H, W = movie.shape
    
    # Normalize movie
    movie_norm = movie - np.mean(movie, axis=0, keepdims=True)
    stds = np.std(movie, axis=0, keepdims=True)
    stds[stds < 1e-10] = 1.0
    movie_norm = movie_norm / stds
    
    max_corr_img = np.full((H, W), -1.0, dtype=np.float32)
    
    # Iterate over neighbor offsets
    for dy in range(-neighborhood, neighborhood + 1):
        for dx in range(-neighborhood, neighborhood + 1):
            if dy == 0 and dx == 0:
                continue
            
            # Determine valid overlap region
            y_start = max(0, -dy)
            y_end = min(H, H - dy)
            x_start = max(0, -dx)
            x_end = min(W, W - dx)
            
            # Center and neighbor pixels
            center = movie_norm[:, y_start:y_end, x_start:x_end]
            neighbor = movie_norm[:, y_start+dy:y_end+dy, x_start+dx:x_end+dx]
            
            # Compute correlation
            corr = np.mean(center * neighbor, axis=0)
            
            # Update maximum
            max_corr_img[y_start:y_end, x_start:x_end] = np.maximum(
                max_corr_img[y_start:y_end, x_start:x_end], corr
            )
    
    return max_corr_img


def compute_temporal_correlation(trace1, trace2):
    """
    Compute Pearson correlation between two time series.
    
    Parameters:
        trace1 (ndarray): First time series (1D array)
        trace2 (ndarray): Second time series (1D array)
    
    Returns:
        float: Pearson correlation coefficient (-1 to 1)
    
    Example:
        >>> import numpy as np
        >>> trace1 = np.sin(np.arange(100) * 0.1)
        >>> trace2 = np.sin(np.arange(100) * 0.1 + 0.5)
        >>> corr = compute_temporal_correlation(trace1, trace2)
        >>> print(f"Correlation: {corr:.3f}")
    """
    # Remove means
    trace1_centered = trace1 - np.mean(trace1)
    trace2_centered = trace2 - np.mean(trace2)
    
    # Compute correlation
    numerator = np.sum(trace1_centered * trace2_centered)
    denominator = np.sqrt(np.sum(trace1_centered**2) * np.sum(trace2_centered**2))
    
    if denominator < 1e-10:
        return 0.0
    
    correlation = numerator / denominator
    
    return correlation


def compute_pairwise_correlations(traces):
    """
    Compute pairwise correlations between all traces.
    
    Parameters:
        traces (ndarray): Array of traces (N_cells, T)
    
    Returns:
        ndarray: Correlation matrix (N_cells, N_cells)
    
    Example:
        >>> import numpy as np
        >>> traces = np.random.randn(10, 1000)  # 10 cells, 1000 frames
        >>> corr_matrix = compute_pairwise_correlations(traces)
        >>> print(corr_matrix.shape)
        (10, 10)
    """
    N, T = traces.shape
    corr_matrix = np.zeros((N, N), dtype=np.float32)
    
    # Normalize traces (z-score)
    traces_norm = traces - np.mean(traces, axis=1, keepdims=True)
    stds = np.std(traces, axis=1, keepdims=True)
    stds[stds < 1e-10] = 1.0
    traces_norm = traces_norm / stds
    
    # Compute correlation matrix efficiently
    corr_matrix = np.dot(traces_norm, traces_norm.T) / T
    
    return corr_matrix


def local_correlation_3d(movie, neighborhood_spatial=1, neighborhood_temporal=1):
    """
    Compute 3D local correlation (spatial + temporal neighborhood).
    
    Considers both spatial and temporal neighbors for correlation calculation.
    Useful for detecting spatiotemporal patterns.
    
    Parameters:
        movie (ndarray): Input movie (T, H, W)
        neighborhood_spatial (int): Spatial neighborhood radius
        neighborhood_temporal (int): Temporal neighborhood radius (frames)
    
    Returns:
        ndarray: 3D local correlation map (T, H, W)
    
    Example:
        >>> import numpy as np
        >>> movie = np.random.randn(500, 64, 64)
        >>> corr_3d = local_correlation_3d(movie, neighborhood_spatial=1, 
        ...                                 neighborhood_temporal=2)
        >>> print(corr_3d.shape)
        (500, 64, 64)
    """
    T, H, W = movie.shape
    
    # Normalize movie
    movie_norm = movie - np.mean(movie)
    movie_norm = movie_norm / (np.std(movie) + 1e-10)
    
    corr_map = np.zeros((T, H, W), dtype=np.float32)
    
    # Process each time point (excluding edges)
    for t in range(neighborhood_temporal, T - neighborhood_temporal):
        # Process each pixel (excluding edges)
        for y in range(neighborhood_spatial, H - neighborhood_spatial):
            for x in range(neighborhood_spatial, W - neighborhood_spatial):
                # Get center voxel
                center = movie_norm[t, y, x]
                
                # Compute correlation with spatiotemporal neighbors
                correlations = []
                
                for dt in range(-neighborhood_temporal, neighborhood_temporal + 1):
                    for dy in range(-neighborhood_spatial, neighborhood_spatial + 1):
                        for dx in range(-neighborhood_spatial, neighborhood_spatial + 1):
                            if dt == 0 and dy == 0 and dx == 0:  # Skip center
                                continue
                            
                            neighbor = movie_norm[t + dt, y + dy, x + dx]
                            correlations.append(center * neighbor)
                
                # Average correlation
                if len(correlations) > 0:
                    corr_map[t, y, x] = np.mean(correlations)
    
    return corr_map


if __name__ == '__main__':
    """Unit tests for correlation module."""
    print("Testing correlation_module...")
    
    # Test 1: Local correlation image (basic)
    print("\n1. Testing local_correlation_image...")
    np.random.seed(42)
    T, H, W = 200, 32, 32
    
    # Create synthetic data with correlated region
    movie = np.random.randn(T, H, W) * 5
    signal = np.random.randn(T) * 20
    movie[:, 15:18, 15:18] += signal[:, np.newaxis, np.newaxis]
    
    corr_img = local_correlation_image(movie, neighborhood=1)
    
    print(f"   Correlation image shape: {corr_img.shape}")
    print(f"   Max correlation: {corr_img.max():.3f}")
    print(f"   Correlation at center: {corr_img[16, 16]:.3f}")
    print(f"   Correlation at edge: {corr_img[2, 2]:.3f}")
    assert corr_img.shape == (H, W), "Shape mismatch"
    assert corr_img[16, 16] > corr_img[2, 2], "Center should have higher correlation"
    print("   ✓ Local correlation image working correctly")
    
    # Test 2: Efficient correlation image
    print("\n2. Testing local_correlation_image_efficient...")
    corr_img_fast = local_correlation_image_efficient(movie, neighborhood=1)
    
    print(f"   Fast correlation shape: {corr_img_fast.shape}")
    print(f"   Fast max correlation: {corr_img_fast.max():.3f}")
    # Results should be similar (not identical due to different implementations)
    diff = np.abs(corr_img - corr_img_fast)
    print(f"   Mean difference from basic: {diff.mean():.4f}")
    assert diff.mean() < 0.2, "Efficient version differs too much"
    print("   ✓ Efficient correlation image working correctly")
    
    # Test 3: Maximum correlation image
    print("\n3. Testing compute_max_correlation_image...")
    max_corr = compute_max_correlation_image(movie, neighborhood=1)
    
    print(f"   Max correlation shape: {max_corr.shape}")
    print(f"   Max correlation value: {max_corr.max():.3f}")
    print(f"   Max at center: {max_corr[16, 16]:.3f}")
    assert max_corr.shape == (H, W), "Shape mismatch"
    assert max_corr.max() >= corr_img_fast.max(), "Max should be >= average"
    print("   ✓ Maximum correlation image working correctly")
    
    # Test 4: Temporal correlation
    print("\n4. Testing compute_temporal_correlation...")
    trace1 = np.sin(np.arange(100) * 0.1)
    trace2 = np.sin(np.arange(100) * 0.1 + 0.5)  # Phase shifted
    trace3 = np.random.randn(100)  # Uncorrelated
    
    corr_12 = compute_temporal_correlation(trace1, trace2)
    corr_13 = compute_temporal_correlation(trace1, trace3)
    
    print(f"   Correlation (sin, sin shifted): {corr_12:.3f}")
    print(f"   Correlation (sin, random): {corr_13:.3f}")
    assert corr_12 > 0.5, "Shifted sine waves should be correlated"
    assert abs(corr_13) < 0.3, "Random should be uncorrelated"
    print("   ✓ Temporal correlation working correctly")
    
    # Test 5: Pairwise correlations
    print("\n5. Testing compute_pairwise_correlations...")
    N_cells = 10
    T_frames = 500
    traces = np.random.randn(N_cells, T_frames)
    
    # Make some cells correlated
    traces[1] = traces[0] + np.random.randn(T_frames) * 0.3
    traces[2] = traces[0] + np.random.randn(T_frames) * 0.3
    
    corr_matrix = compute_pairwise_correlations(traces)
    
    print(f"   Correlation matrix shape: {corr_matrix.shape}")
    print(f"   Diagonal (self-correlation): {corr_matrix[0, 0]:.3f}")
    print(f"   Correlated pair: {corr_matrix[0, 1]:.3f}")
    print(f"   Uncorrelated pair: {corr_matrix[0, 5]:.3f}")
    assert corr_matrix.shape == (N_cells, N_cells), "Shape mismatch"
    assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal should be 1"
    assert corr_matrix[0, 1] > corr_matrix[0, 5], "Correlated pair should have higher correlation"
    print("   ✓ Pairwise correlations working correctly")
    
    # Test 6: 3D local correlation
    print("\n6. Testing local_correlation_3d...")
    T_small, H_small, W_small = 50, 16, 16
    movie_small = np.random.randn(T_small, H_small, W_small)
    
    corr_3d = local_correlation_3d(movie_small, 
                                   neighborhood_spatial=1, 
                                   neighborhood_temporal=1)
    
    print(f"   3D correlation shape: {corr_3d.shape}")
    print(f"   3D correlation range: [{corr_3d.min():.3f}, {corr_3d.max():.3f}]")
    assert corr_3d.shape == (T_small, H_small, W_small), "Shape mismatch"
    print("   ✓ 3D local correlation working correctly")
    
    print("\n✅ All correlation module tests passed!")
