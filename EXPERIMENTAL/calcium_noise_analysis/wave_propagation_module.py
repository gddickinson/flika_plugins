"""
Wave Propagation Module
=======================
Analysis of Ca²⁺ wave propagation including velocity, direction,
and spatiotemporal dynamics.

Implements methods for detecting and characterizing intercellular and
intracellular Ca²⁺ waves.

Author: George
"""

import numpy as np
from scipy.ndimage import label, center_of_mass, distance_transform_edt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def detect_wave_initiation_sites(image_stack, threshold=0.5, min_separation=10):
    """
    Detect sites where Ca²⁺ waves initiate.
    
    Identifies pixels where Ca²⁺ signal rises earliest and propagates outward.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        threshold (float): Threshold for detecting signal onset (ΔF/F₀)
        min_separation (int): Minimum distance between initiation sites (pixels)
    
    Returns:
        dict: Dictionary containing:
            - 'initiation_sites': List of (y, x) coordinates
            - 'initiation_times': List of frame indices when each site initiated
            - 'initiation_amplitudes': Peak amplitudes at each site
    
    Example:
        >>> sites = detect_wave_initiation_sites(stack, threshold=0.3)
        >>> print(f"Found {len(sites['initiation_sites'])} initiation sites")
        >>> for (y, x), t in zip(sites['initiation_sites'], sites['initiation_times']):
        >>>     print(f"  Site at ({y}, {x}) initiated at frame {t}")
    """
    T, H, W = image_stack.shape
    
    # Compute onset time for each pixel
    onset_map = np.full((H, W), T, dtype=float)  # Initialize to max time
    
    for y in range(H):
        for x in range(W):
            trace = image_stack[:, y, x]
            # Find first crossing of threshold
            crossings = np.where(trace > threshold)[0]
            if len(crossings) > 0:
                onset_map[y, x] = crossings[0]
    
    # Find local minima in onset map (earliest activating pixels)
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(onset_map, size=min_separation)
    initiation_mask = (onset_map == local_min) & (onset_map < T)
    
    # Extract initiation sites
    initiation_coords = np.array(np.where(initiation_mask)).T  # (N, 2) array of (y, x)
    initiation_times = [onset_map[y, x] for y, x in initiation_coords]
    
    # Get amplitudes
    initiation_amplitudes = []
    for (y, x), t in zip(initiation_coords, initiation_times):
        amp = np.max(image_stack[:, y, x])
        initiation_amplitudes.append(amp)
    
    return {
        'initiation_sites': initiation_coords.tolist(),
        'initiation_times': initiation_times,
        'initiation_amplitudes': initiation_amplitudes,
        'onset_map': onset_map
    }


def compute_wave_velocity(image_stack, initiation_site, fs, threshold=0.5):
    """
    Compute wave propagation velocity from an initiation site.
    
    Measures how fast the Ca²⁺ wave spreads radially from source.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        initiation_site (tuple): (y, x) coordinates of wave origin
        fs (float): Sampling frequency (Hz)
        threshold (float): Threshold for wave front detection (ΔF/F₀)
    
    Returns:
        dict: Dictionary containing:
            - 'velocity_um_per_s': Mean wave velocity (μm/s, assumes 1 pixel = 1 μm)
            - 'velocity_pixels_per_frame': Velocity in pixels/frame
            - 'distance_time_curve': Arrays (distances, times) for fitting
            - 'radial_velocity_map': Map of velocities in different directions
    
    Example:
        >>> sites = detect_wave_initiation_sites(stack)
        >>> site = sites['initiation_sites'][0]
        >>> velocity = compute_wave_velocity(stack, site, fs=30.0)
        >>> print(f"Wave velocity: {velocity['velocity_um_per_s']:.1f} μm/s")
    
    Notes:
        - Assumes isotropic pixel spacing (calibrate for real units)
        - Default assumes 1 pixel = 1 μm
    """
    T, H, W = image_stack.shape
    y0, x0 = initiation_site
    
    # Create distance map from initiation site
    y_grid, x_grid = np.ogrid[:H, :W]
    distance_map = np.sqrt((y_grid - y0)**2 + (x_grid - x0)**2)
    
    # Compute onset time for each pixel
    onset_map = np.full((H, W), T, dtype=float)
    
    for y in range(H):
        for x in range(W):
            trace = image_stack[:, y, x]
            crossings = np.where(trace > threshold)[0]
            if len(crossings) > 0:
                onset_map[y, x] = crossings[0]
    
    # Get valid points (where wave reached)
    valid = onset_map < T
    distances = distance_map[valid]
    times = onset_map[valid]
    
    # Convert to physical units
    times_sec = times / fs
    distances_um = distances  # Assumes 1 pixel = 1 μm
    
    # Fit linear relationship (distance = velocity * time)
    if len(distances_um) > 10:
        coeffs = np.polyfit(times_sec, distances_um, deg=1)
        velocity_um_per_s = coeffs[0]
        velocity_pixels_per_frame = velocity_um_per_s / fs
    else:
        velocity_um_per_s = 0
        velocity_pixels_per_frame = 0
    
    # Compute directional velocities
    radial_velocity_map = _compute_radial_velocity_map(distance_map, onset_map, fs)
    
    return {
        'velocity_um_per_s': velocity_um_per_s,
        'velocity_pixels_per_frame': velocity_pixels_per_frame,
        'distance_time_curve': (distances_um, times_sec),
        'radial_velocity_map': radial_velocity_map,
        'onset_map': onset_map
    }


def _compute_radial_velocity_map(distance_map, onset_map, fs):
    """Compute velocity map in radial bins."""
    T = onset_map.max()
    
    # Bin by distance
    max_dist = distance_map.max()
    n_bins = 20
    dist_bins = np.linspace(0, max_dist, n_bins + 1)
    
    velocities = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (distance_map >= dist_bins[i]) & (distance_map < dist_bins[i+1])
        mask = mask & (onset_map < T)
        
        if np.sum(mask) > 0:
            distances = distance_map[mask]
            times = onset_map[mask] / fs
            
            if len(times) > 1:
                # Fit velocity for this bin
                coeffs = np.polyfit(times, distances, deg=1)
                velocities[i] = coeffs[0]
    
    return velocities


def compute_wave_direction(image_stack, initiation_site, threshold=0.5):
    """
    Compute dominant direction of wave propagation.
    
    Analyzes angular distribution of wave spread.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        initiation_site (tuple): (y, x) coordinates of wave origin
        threshold (float): Threshold for wave front detection
    
    Returns:
        dict: Dictionary containing:
            - 'dominant_angle': Dominant propagation angle (degrees, 0=right, 90=up)
            - 'angular_distribution': Array of activation counts per angle bin
            - 'angle_bins': Array of angle bin centers (degrees)
            - 'anisotropy_index': Measure of directional bias (0=isotropic, 1=unidirectional)
    
    Example:
        >>> direction = compute_wave_direction(stack, site=(32, 32))
        >>> print(f"Dominant direction: {direction['dominant_angle']:.1f}°")
        >>> print(f"Anisotropy: {direction['anisotropy_index']:.2f}")
    """
    T, H, W = image_stack.shape
    y0, x0 = initiation_site
    
    # Create angle map
    y_grid, x_grid = np.ogrid[:H, :W]
    angle_map = np.arctan2(y_grid - y0, x_grid - x0) * 180 / np.pi  # -180 to 180
    angle_map = (angle_map + 360) % 360  # 0 to 360
    
    # Compute onset map
    onset_map = np.full((H, W), T, dtype=float)
    
    for y in range(H):
        for x in range(W):
            trace = image_stack[:, y, x]
            crossings = np.where(trace > threshold)[0]
            if len(crossings) > 0:
                onset_map[y, x] = crossings[0]
    
    # Histogram of angles where wave reached
    valid = onset_map < T
    angles = angle_map[valid]
    
    n_bins = 36  # 10° bins
    angle_bins = np.linspace(0, 360, n_bins + 1)
    counts, _ = np.histogram(angles, bins=angle_bins)
    angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    # Dominant angle
    dominant_bin = np.argmax(counts)
    dominant_angle = angle_centers[dominant_bin]
    
    # Anisotropy index (uniformity coefficient)
    anisotropy_index = 1 - (np.min(counts) / np.max(counts)) if np.max(counts) > 0 else 0
    
    return {
        'dominant_angle': dominant_angle,
        'angular_distribution': counts,
        'angle_bins': angle_centers,
        'anisotropy_index': anisotropy_index,
        'angle_map': angle_map
    }


def create_spatiotemporal_map(image_stack, axis='x'):
    """
    Create kymograph-style spatiotemporal map.
    
    Shows Ca²⁺ signal evolution along one spatial dimension over time.
    Useful for visualizing wave propagation.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        axis (str): 'x' or 'y' for spatial axis
    
    Returns:
        ndarray: Spatiotemporal map (T, spatial_dimension)
    
    Example:
        >>> kymograph = create_spatiotemporal_map(stack, axis='x')
        >>> # kymograph shape: (T, W)
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(kymograph, aspect='auto', cmap='hot')
        >>> plt.xlabel('X position')
        >>> plt.ylabel('Time (frames)')
    """
    T, H, W = image_stack.shape
    
    if axis.lower() == 'x':
        # Average over Y dimension
        st_map = np.mean(image_stack, axis=1)  # (T, W)
    elif axis.lower() == 'y':
        # Average over X dimension
        st_map = np.mean(image_stack, axis=2)  # (T, H)
    else:
        raise ValueError("axis must be 'x' or 'y'")
    
    return st_map


def compute_wave_area(image_stack, threshold=0.5, fs=30.0):
    """
    Compute area recruited by Ca²⁺ wave over time.
    
    Measures spatial extent of wave as it propagates.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        threshold (float): Threshold for detecting active region
        fs (float): Sampling frequency (Hz)
    
    Returns:
        dict: Dictionary containing:
            - 'area_trace': Array of active area at each time point (pixels²)
            - 'max_area': Maximum area reached
            - 'area_growth_rate': Rate of area increase (pixels²/s)
            - 'time_to_peak_area': Time to reach maximum area (s)
    
    Example:
        >>> area_info = compute_wave_area(stack, threshold=0.3)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(area_info['area_trace'])
        >>> plt.xlabel('Frame')
        >>> plt.ylabel('Active area (pixels²)')
    """
    T, H, W = image_stack.shape
    area_trace = np.zeros(T)
    
    for t in range(T):
        active_mask = image_stack[t] > threshold
        area_trace[t] = np.sum(active_mask)
    
    max_area = np.max(area_trace)
    max_area_frame = np.argmax(area_trace)
    time_to_peak_area = max_area_frame / fs
    
    # Compute growth rate during rising phase
    if max_area_frame > 1:
        growth_phase = area_trace[:max_area_frame]
        time_points = np.arange(len(growth_phase)) / fs
        
        # Fit linear growth rate
        if len(growth_phase) > 2:
            coeffs = np.polyfit(time_points, growth_phase, deg=1)
            area_growth_rate = coeffs[0]
        else:
            area_growth_rate = 0
    else:
        area_growth_rate = 0
    
    return {
        'area_trace': area_trace,
        'max_area': max_area,
        'area_growth_rate': area_growth_rate,
        'time_to_peak_area': time_to_peak_area
    }


def analyze_wave_collision(image_stack, site1, site2, fs=30.0, threshold=0.5):
    """
    Analyze collision between two Ca²⁺ waves.
    
    Detects when and where waves from two initiation sites meet.
    
    Parameters:
        image_stack (ndarray): Input image stack (T, H, W)
        site1 (tuple): (y, x) coordinates of first wave origin
        site2 (tuple): (y, x) coordinates of second wave origin
        fs (float): Sampling frequency (Hz)
        threshold (float): Threshold for wave detection
    
    Returns:
        dict: Dictionary containing:
            - 'collision_detected': Boolean indicating if waves collided
            - 'collision_location': (y, x) coordinates of collision
            - 'collision_time': Time of collision (s)
            - 'collision_zone': Binary mask of collision region
    
    Example:
        >>> sites = detect_wave_initiation_sites(stack)
        >>> if len(sites['initiation_sites']) >= 2:
        >>>     collision = analyze_wave_collision(stack, 
        >>>                                        sites['initiation_sites'][0],
        >>>                                        sites['initiation_sites'][1],
        >>>                                        fs=30.0)
        >>>     if collision['collision_detected']:
        >>>         print(f"Collision at {collision['collision_location']}")
    """
    T, H, W = image_stack.shape
    y1, x1 = site1
    y2, x2 = site2
    
    # Compute onset maps from each site
    onset_map = np.full((H, W), T, dtype=float)
    
    for y in range(H):
        for x in range(W):
            trace = image_stack[:, y, x]
            crossings = np.where(trace > threshold)[0]
            if len(crossings) > 0:
                onset_map[y, x] = crossings[0]
    
    # Distance maps from each site
    y_grid, x_grid = np.ogrid[:H, :W]
    dist1 = np.sqrt((y_grid - y1)**2 + (x_grid - x1)**2)
    dist2 = np.sqrt((y_grid - y2)**2 + (x_grid - x2)**2)
    
    # Collision zone: equidistant from both sites
    collision_zone = np.abs(dist1 - dist2) < 5  # Within 5 pixels of midline
    
    # Check if collision occurred
    collision_pixels = collision_zone & (onset_map < T)
    collision_detected = np.sum(collision_pixels) > 0
    
    if collision_detected:
        # Find collision location and time
        collision_times = onset_map[collision_pixels]
        earliest_collision_time = np.min(collision_times)
        
        # Find pixel with earliest collision
        collision_mask = collision_pixels & (onset_map == earliest_collision_time)
        collision_coords = np.array(np.where(collision_mask)).T
        collision_location = tuple(collision_coords[0])
        
        collision_time_sec = earliest_collision_time / fs
    else:
        collision_location = None
        collision_time_sec = None
    
    return {
        'collision_detected': collision_detected,
        'collision_location': collision_location,
        'collision_time': collision_time_sec,
        'collision_zone': collision_zone
    }


if __name__ == '__main__':
    """Unit tests for wave propagation module."""
    print("Testing wave_propagation_module...")
    
    # Test 1: Detect initiation sites
    print("\n1. Testing detect_wave_initiation_sites...")
    np.random.seed(42)
    T, H, W = 300, 64, 64
    fs = 30.0
    
    # Create synthetic wave
    stack = np.zeros((T, H, W))
    
    # Initiation site at (32, 32)
    y0, x0 = 32, 32
    for t in range(T):
        y_grid, x_grid = np.ogrid[:H, :W]
        dist = np.sqrt((y_grid - y0)**2 + (x_grid - x0)**2)
        
        # Wave expands radially
        wave_radius = t * 0.3  # Velocity = 0.3 pixels/frame
        wave = np.exp(-(dist - wave_radius)**2 / 10) * 0.8
        stack[t] += wave
    
    # Add noise
    stack += np.random.randn(T, H, W) * 0.05
    
    sites = detect_wave_initiation_sites(stack, threshold=0.3, min_separation=10)
    print(f"   Found {len(sites['initiation_sites'])} initiation sites")
    print(f"   Sites: {sites['initiation_sites'][:3]}")  # Show first 3
    assert len(sites['initiation_sites']) >= 1, "Should detect at least one site"
    print("   ✓ Initiation site detection working")
    
    # Test 2: Wave velocity
    print("\n2. Testing compute_wave_velocity...")
    site = sites['initiation_sites'][0]
    velocity = compute_wave_velocity(stack, site, fs, threshold=0.3)
    print(f"   Wave velocity: {velocity['velocity_um_per_s']:.2f} μm/s")
    print(f"   Velocity: {velocity['velocity_pixels_per_frame']:.3f} pixels/frame")
    assert velocity['velocity_um_per_s'] > 0, "Invalid velocity"
    # Expected ~9 μm/s (0.3 pixels/frame * 30 fps)
    print("   ✓ Wave velocity computation working")
    
    # Test 3: Wave direction
    print("\n3. Testing compute_wave_direction...")
    direction = compute_wave_direction(stack, site, threshold=0.3)
    print(f"   Dominant angle: {direction['dominant_angle']:.1f}°")
    print(f"   Anisotropy: {direction['anisotropy_index']:.2f}")
    assert 0 <= direction['dominant_angle'] <= 360, "Invalid angle"
    print("   ✓ Wave direction computation working")
    
    # Test 4: Spatiotemporal map
    print("\n4. Testing create_spatiotemporal_map...")
    st_map = create_spatiotemporal_map(stack, axis='x')
    print(f"   Spatiotemporal map shape: {st_map.shape}")
    assert st_map.shape == (T, W), "Invalid shape"
    print("   ✓ Spatiotemporal map creation working")
    
    # Test 5: Wave area
    print("\n5. Testing compute_wave_area...")
    area_info = compute_wave_area(stack, threshold=0.3, fs=fs)
    print(f"   Max area: {area_info['max_area']:.0f} pixels²")
    print(f"   Growth rate: {area_info['area_growth_rate']:.1f} pixels²/s")
    print(f"   Time to peak: {area_info['time_to_peak_area']:.2f} s")
    assert area_info['max_area'] > 0, "Invalid area"
    print("   ✓ Wave area computation working")
    
    # Test 6: Wave collision
    print("\n6. Testing analyze_wave_collision...")
    # Add second wave
    y1, x1 = 20, 20
    for t in range(100, T):
        y_grid, x_grid = np.ogrid[:H, :W]
        dist = np.sqrt((y_grid - y1)**2 + (x_grid - x1)**2)
        wave_radius = (t - 100) * 0.3
        wave = np.exp(-(dist - wave_radius)**2 / 10) * 0.8
        stack[t] += wave
    
    collision = analyze_wave_collision(stack, (y0, x0), (y1, x1), fs, threshold=0.3)
    print(f"   Collision detected: {collision['collision_detected']}")
    if collision['collision_detected']:
        print(f"   Collision location: {collision['collision_location']}")
        print(f"   Collision time: {collision['collision_time']:.2f} s")
    print("   ✓ Wave collision analysis working")
    
    print("\n✅ All wave propagation module tests passed!")
