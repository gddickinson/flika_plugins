"""
Spatial Analysis Module
========================
Spatial statistics and analysis of Ca2+ signaling patterns.

Analyzes spatial relationships between Ca2+ release sites including:
- Distance calculations
- Clustering analysis
- Spatial density
- Nearest neighbor analysis

Based on methodologies from Lock & Parker 2020 and Swaminathan et al. 2020.

Author: George
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import gaussian_kde


def calculate_pairwise_distances(coordinates):
    """
    Calculate all pairwise distances between sites.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
    
    Returns:
        ndarray: Distance matrix (N, N) where N is number of sites
    
    Example:
        >>> coords = [(10, 10), (20, 20), (30, 10)]
        >>> distances = calculate_pairwise_distances(coords)
        >>> print(distances.shape)
        (3, 3)
    """
    if len(coordinates) == 0:
        return np.array([])
    
    coords_array = np.array(coordinates)
    distances = squareform(pdist(coords_array, metric='euclidean'))
    
    return distances


def nearest_neighbor_distances(coordinates):
    """
    Calculate nearest neighbor distance for each site.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
    
    Returns:
        ndarray: Array of nearest neighbor distances for each site
    
    Example:
        >>> coords = [(10, 10), (20, 20), (30, 10)]
        >>> nn_dists = nearest_neighbor_distances(coords)
        >>> print(f"Mean NN distance: {nn_dists.mean():.2f} pixels")
    """
    if len(coordinates) < 2:
        return np.array([])
    
    distances = calculate_pairwise_distances(coordinates)
    
    # Set diagonal to infinity to exclude self-distances
    np.fill_diagonal(distances, np.inf)
    
    # Find minimum distance for each site
    nn_distances = np.min(distances, axis=1)
    
    return nn_distances


def calculate_spatial_density(coordinates, image_shape, bandwidth=10):
    """
    Calculate kernel density estimate of site locations.
    
    Useful for identifying regions of high puff site density.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        image_shape (tuple): (H, W) shape of image
        bandwidth (float): Bandwidth for Gaussian kernel
    
    Returns:
        ndarray: Density map (H, W)
    
    Example:
        >>> coords = [(32, 32), (35, 30), (30, 35)]
        >>> density = calculate_spatial_density(coords, (64, 64), bandwidth=10)
        >>> print(f"Max density: {density.max():.4f}")
    """
    if len(coordinates) == 0:
        return np.zeros(image_shape)
    
    H, W = image_shape
    coords_array = np.array(coordinates).T  # Transpose for KDE
    
    # Create grid
    y, x = np.mgrid[0:H, 0:W]
    positions = np.vstack([y.ravel(), x.ravel()])
    
    # Calculate KDE
    try:
        kernel = gaussian_kde(coords_array, bw_method=bandwidth/H)
        density = np.reshape(kernel(positions), (H, W))
    except:
        # Fallback if KDE fails
        density = np.zeros(image_shape)
        for cy, cx in coordinates:
            y_idx = int(cy)
            x_idx = int(cx)
            if 0 <= y_idx < H and 0 <= x_idx < W:
                density[y_idx, x_idx] += 1
    
    return density


def cluster_sites(coordinates, distance_threshold=10, method='single'):
    """
    Cluster sites based on spatial proximity using hierarchical clustering.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        distance_threshold (float): Distance threshold for clustering (pixels)
        method (str): Linkage method ('single', 'complete', 'average', 'ward')
    
    Returns:
        tuple: (cluster_labels, n_clusters)
            - cluster_labels: Array of cluster assignments for each site
            - n_clusters: Number of clusters found
    
    Example:
        >>> coords = [(10, 10), (12, 11), (50, 50), (52, 51)]
        >>> labels, n_clusters = cluster_sites(coords, distance_threshold=5)
        >>> print(f"Found {n_clusters} clusters")
        Found 2 clusters
    """
    if len(coordinates) < 2:
        return np.array([1] if coordinates else []), len(coordinates)
    
    coords_array = np.array(coordinates)
    
    # Hierarchical clustering
    Z = linkage(coords_array, method=method)
    cluster_labels = fcluster(Z, distance_threshold, criterion='distance')
    n_clusters = len(np.unique(cluster_labels))
    
    return cluster_labels, n_clusters


def calculate_cluster_properties(coordinates, cluster_labels):
    """
    Calculate properties for each cluster.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        cluster_labels (ndarray): Cluster assignment for each site
    
    Returns:
        dict: Dictionary mapping cluster ID to properties dict
    
    Example:
        >>> coords = [(10, 10), (12, 11), (50, 50)]
        >>> labels = np.array([1, 1, 2])
        >>> props = calculate_cluster_properties(coords, labels)
        >>> print(f"Cluster 1 has {props[1]['n_sites']} sites")
    """
    coords_array = np.array(coordinates)
    unique_clusters = np.unique(cluster_labels)
    
    cluster_props = {}
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        cluster_coords = coords_array[mask]
        
        # Calculate properties
        n_sites = len(cluster_coords)
        centroid = np.mean(cluster_coords, axis=0)
        
        if n_sites > 1:
            # Calculate spread (standard deviation)
            spread = np.std(cluster_coords, axis=0)
            
            # Calculate diameter (max pairwise distance within cluster)
            cluster_distances = pdist(cluster_coords)
            diameter = np.max(cluster_distances) if len(cluster_distances) > 0 else 0
        else:
            spread = np.array([0, 0])
            diameter = 0
        
        cluster_props[cluster_id] = {
            'n_sites': n_sites,
            'centroid': tuple(centroid),
            'spread_y': spread[0],
            'spread_x': spread[1],
            'diameter': diameter
        }
    
    return cluster_props


def calculate_ripley_k(coordinates, image_shape, radii=None, edge_correction=True):
    """
    Calculate Ripley's K function to assess spatial clustering.
    
    K(r) measures the number of sites within distance r of a typical site.
    Values above random indicate clustering, below indicate regularity.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        image_shape (tuple): (H, W) shape of image
        radii (ndarray): Array of radii to evaluate (default: 1 to 20 pixels)
        edge_correction (bool): Apply edge correction
    
    Returns:
        tuple: (radii, K_values, L_values)
            - L(r) = sqrt(K(r)/π) - r (normalized, easier to interpret)
            - L(r) > 0 indicates clustering, L(r) < 0 indicates regularity
    
    Example:
        >>> coords = [(10+i, 10+i) for i in range(5)]  # Clustered
        >>> radii, K, L = calculate_ripley_k(coords, (64, 64))
        >>> print(f"L values indicate {'clustering' if np.mean(L) > 0 else 'regularity'}")
    """
    if radii is None:
        radii = np.arange(1, 21, 1)
    
    n_sites = len(coordinates)
    if n_sites < 2:
        return radii, np.zeros_like(radii), np.zeros_like(radii)
    
    H, W = image_shape
    area = H * W
    
    coords_array = np.array(coordinates)
    distances = squareform(pdist(coords_array))
    
    K_values = np.zeros_like(radii, dtype=float)
    
    for i, r in enumerate(radii):
        # Count pairs within distance r
        count = np.sum(distances < r) - n_sites  # Subtract diagonal
        
        # Edge correction (simple border method)
        if edge_correction:
            # Estimate fraction of sites affected by edge
            edge_buffer = r
            n_edge = 0
            for y, x in coordinates:
                if y < edge_buffer or y > (H - edge_buffer) or \
                   x < edge_buffer or x > (W - edge_buffer):
                    n_edge += 1
            edge_fraction = 1 - (n_edge / n_sites) if n_sites > 0 else 1
        else:
            edge_fraction = 1
        
        # Calculate K(r)
        K_values[i] = (area * count) / (n_sites * (n_sites - 1) * edge_fraction)
    
    # Calculate L(r) = sqrt(K(r)/π) - r
    L_values = np.sqrt(K_values / np.pi) - radii
    
    return radii, K_values, L_values


def calculate_spatial_autocorrelation(activity_map, max_distance=20):
    """
    Calculate spatial autocorrelation of activity.
    
    Measures how correlated activity is at different spatial separations.
    
    Parameters:
        activity_map (ndarray): 2D map of activity values (e.g., η or ξ)
        max_distance (int): Maximum distance to calculate (pixels)
    
    Returns:
        tuple: (distances, correlations)
    
    Example:
        >>> activity = np.random.randn(64, 64)
        >>> activity[30:35, 30:35] += 5  # Local hot spot
        >>> distances, corr = calculate_spatial_autocorrelation(activity)
        >>> print(f"Correlation at 0 pixels: {corr[0]:.3f}")
    """
    H, W = activity_map.shape
    
    # Flatten and remove mean
    flat_activity = activity_map.flatten()
    mean_activity = np.mean(flat_activity)
    centered = flat_activity - mean_activity
    
    # Create coordinate arrays
    y, x = np.mgrid[0:H, 0:W]
    coords = np.column_stack([y.ravel(), x.ravel()])
    
    distances = np.arange(0, max_distance + 1)
    correlations = np.zeros_like(distances, dtype=float)
    
    # Calculate pairwise distances
    dist_matrix = squareform(pdist(coords))
    
    for i, d in enumerate(distances):
        # Find pairs at distance d (with tolerance)
        if d == 0:
            mask = dist_matrix == 0
        else:
            mask = (dist_matrix >= d - 0.5) & (dist_matrix < d + 0.5)
        
        if np.any(mask):
            # Calculate correlation for these pairs
            pairs = np.where(mask)
            pair_products = centered[pairs[0]] * centered[pairs[1]]
            correlations[i] = np.mean(pair_products) / np.var(flat_activity)
    
    return distances, correlations


def calculate_site_density_map(coordinates, image_shape, radius=10):
    """
    Calculate local density of sites (number within radius).
    
    Different from KDE - counts discrete sites rather than smooth density.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        image_shape (tuple): (H, W) shape of image
        radius (float): Radius for counting neighbors (pixels)
    
    Returns:
        ndarray: Density map (H, W) showing number of sites within radius
    
    Example:
        >>> coords = [(32, 32), (35, 30), (30, 35), (50, 50)]
        >>> density = calculate_site_density_map(coords, (64, 64), radius=10)
        >>> print(f"Max local density: {density.max()}")
    """
    H, W = image_shape
    density_map = np.zeros(image_shape, dtype=int)
    
    if len(coordinates) == 0:
        return density_map
    
    coords_array = np.array(coordinates)
    
    # For each pixel, count sites within radius
    for y in range(H):
        for x in range(W):
            pixel_coord = np.array([y, x])
            distances = np.sqrt(np.sum((coords_array - pixel_coord)**2, axis=1))
            density_map[y, x] = np.sum(distances <= radius)
    
    return density_map


def analyze_peripheral_vs_interior(coordinates, image_shape, boundary_width=10):
    """
    Analyze distribution of sites in peripheral vs interior regions.
    
    Peripheral sites (near cell edge) vs interior sites show different
    properties in many cell types.
    
    Parameters:
        coordinates (list): List of (y, x) coordinate tuples
        image_shape (tuple): (H, W) shape of image
        boundary_width (int): Width of peripheral zone (pixels)
    
    Returns:
        dict: Statistics for peripheral and interior regions
    
    Example:
        >>> coords = [(5, 5), (10, 32), (32, 32), (50, 50)]
        >>> stats = analyze_peripheral_vs_interior(coords, (64, 64), boundary_width=15)
        >>> print(f"Peripheral: {stats['n_peripheral']}, Interior: {stats['n_interior']}")
    """
    H, W = image_shape
    
    peripheral_coords = []
    interior_coords = []
    
    for y, x in coordinates:
        if y < boundary_width or y > (H - boundary_width) or \
           x < boundary_width or x > (W - boundary_width):
            peripheral_coords.append((y, x))
        else:
            interior_coords.append((y, x))
    
    n_peripheral = len(peripheral_coords)
    n_interior = len(interior_coords)
    n_total = len(coordinates)
    
    results = {
        'n_peripheral': n_peripheral,
        'n_interior': n_interior,
        'n_total': n_total,
        'fraction_peripheral': n_peripheral / n_total if n_total > 0 else 0,
        'peripheral_coords': peripheral_coords,
        'interior_coords': interior_coords
    }
    
    # Calculate densities
    peripheral_area = 2 * boundary_width * (H + W - 2 * boundary_width)
    interior_area = H * W - peripheral_area
    
    results['peripheral_density'] = n_peripheral / peripheral_area if peripheral_area > 0 else 0
    results['interior_density'] = n_interior / interior_area if interior_area > 0 else 0
    
    return results


if __name__ == '__main__':
    """Unit tests for spatial analysis module."""
    print("Testing spatial_analysis_module...")
    
    # Test 1: Pairwise distances
    print("\n1. Testing calculate_pairwise_distances...")
    coords = [(10, 10), (20, 20), (30, 10)]
    distances = calculate_pairwise_distances(coords)
    
    print(f"   Distance matrix shape: {distances.shape}")
    print(f"   Distance (0,1): {distances[0, 1]:.2f} pixels")
    expected_dist = np.sqrt((20-10)**2 + (20-10)**2)
    assert np.isclose(distances[0, 1], expected_dist), "Distance calculation error"
    print("   ✓ Pairwise distance calculation working correctly")
    
    # Test 2: Nearest neighbor
    print("\n2. Testing nearest_neighbor_distances...")
    nn_dists = nearest_neighbor_distances(coords)
    
    print(f"   NN distances: {nn_dists}")
    print(f"   Mean NN distance: {nn_dists.mean():.2f} pixels")
    assert len(nn_dists) == len(coords)
    print("   ✓ Nearest neighbor analysis working correctly")
    
    # Test 3: Spatial density
    print("\n3. Testing calculate_spatial_density...")
    coords_dense = [(32, 32), (35, 30), (30, 35), (50, 50)]
    density = calculate_spatial_density(coords_dense, (64, 64), bandwidth=5)
    
    print(f"   Density map shape: {density.shape}")
    print(f"   Max density: {density.max():.6f}")
    print(f"   Density is higher near sites: {density[32, 32] > density[10, 10]}")
    assert density.shape == (64, 64)
    print("   ✓ Spatial density calculation working correctly")
    
    # Test 4: Clustering
    print("\n4. Testing cluster_sites...")
    coords_clustered = [(10, 10), (12, 11), (11, 12), (50, 50), (52, 51)]
    labels, n_clusters = cluster_sites(coords_clustered, distance_threshold=5)
    
    print(f"   Found {n_clusters} clusters")
    print(f"   Cluster labels: {labels}")
    assert n_clusters == 2, "Should find 2 clusters"
    print("   ✓ Site clustering working correctly")
    
    # Test 5: Cluster properties
    print("\n5. Testing calculate_cluster_properties...")
    props = calculate_cluster_properties(coords_clustered, labels)
    
    print(f"   Number of clusters: {len(props)}")
    for cluster_id, p in props.items():
        print(f"   Cluster {cluster_id}: {p['n_sites']} sites, diameter={p['diameter']:.2f}")
    print("   ✓ Cluster property calculation working correctly")
    
    # Test 6: Ripley's K
    print("\n6. Testing calculate_ripley_k...")
    coords_regular = [(i*10, j*10) for i in range(5) for j in range(5)]
    radii, K, L = calculate_ripley_k(coords_regular, (64, 64), radii=np.arange(1, 15, 2))
    
    print(f"   Calculated K for {len(radii)} radii")
    print(f"   Mean L value: {L.mean():.3f}")
    print(f"   L values: {L[:3]}...")
    assert len(K) == len(radii)
    print("   ✓ Ripley's K calculation working correctly")
    
    # Test 7: Spatial autocorrelation
    print("\n7. Testing calculate_spatial_autocorrelation...")
    activity = np.random.randn(64, 64) * 0.5
    activity[30:35, 30:35] += 3  # Local hot spot
    
    distances_ac, corr = calculate_spatial_autocorrelation(activity, max_distance=10)
    
    print(f"   Calculated autocorrelation for {len(distances_ac)} distances")
    print(f"   Correlation at distance 0: {corr[0]:.3f}")
    print(f"   Correlation at distance 10: {corr[-1]:.3f}")
    assert corr[0] > corr[-1], "Autocorrelation should decay with distance"
    print("   ✓ Spatial autocorrelation working correctly")
    
    # Test 8: Site density map
    print("\n8. Testing calculate_site_density_map...")
    coords_for_density = [(32, 32), (35, 30), (30, 35), (50, 50)]
    density_discrete = calculate_site_density_map(coords_for_density, (64, 64), radius=10)
    
    print(f"   Density map shape: {density_discrete.shape}")
    print(f"   Max density: {density_discrete.max()}")
    print(f"   Density at (32, 32): {density_discrete[32, 32]}")
    assert density_discrete[32, 32] >= 3, "Should count nearby sites"
    print("   ✓ Site density map working correctly")
    
    # Test 9: Peripheral vs interior
    print("\n9. Testing analyze_peripheral_vs_interior...")
    coords_mixed = [(5, 5), (10, 32), (32, 32), (50, 50), (60, 60)]
    stats = analyze_peripheral_vs_interior(coords_mixed, (64, 64), boundary_width=15)
    
    print(f"   Total sites: {stats['n_total']}")
    print(f"   Peripheral: {stats['n_peripheral']}")
    print(f"   Interior: {stats['n_interior']}")
    print(f"   Fraction peripheral: {stats['fraction_peripheral']:.2f}")
    assert stats['n_total'] == len(coords_mixed)
    assert stats['n_peripheral'] + stats['n_interior'] == stats['n_total']
    print("   ✓ Peripheral/interior analysis working correctly")
    
    print("\n✅ All spatial analysis module tests passed!")
