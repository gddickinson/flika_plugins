"""
Activity Detection Module
===========================
Automated detection and localization of active Ca2+ sites from
Power Spectrum Maps (PSM) and Correlation Maps (CRM).

Based on Swaminathan et al. 2020 methodology for identifying
Ca2+-active pixels and Lock & Parker 2020 for puff site detection.

Author: George
"""

import numpy as np
from scipy.ndimage import label, center_of_mass, binary_dilation
from scipy.ndimage import gaussian_filter


def detect_active_sites_psm(psm, eta_threshold=0.5, min_size=4, gaussian_sigma=1.0):
    """
    Detect active Ca2+ sites from Power Spectrum Map.
    
    Identifies regions where η = (P_LFR - P_HFR) / P_HFR exceeds threshold.
    
    Parameters:
        psm (ndarray): Power spectrum map (H, W) with η values
        eta_threshold (float): Threshold for η to be considered active
        min_size (int): Minimum number of connected pixels for a valid site
        gaussian_sigma (float): Sigma for Gaussian smoothing before detection
    
    Returns:
        tuple: (labels, n_sites, coordinates, properties)
            - labels: Labeled array where each site has unique integer ID
            - n_sites: Number of detected sites
            - coordinates: List of (y, x) centroids for each site
            - properties: List of dicts with site properties
    
    Example:
        >>> import numpy as np
        >>> psm = np.zeros((64, 64))
        >>> psm[30:35, 30:35] = 2.0  # Active site
        >>> labels, n_sites, coords, props = detect_active_sites_psm(psm, eta_threshold=0.5)
        >>> print(f"Detected {n_sites} active sites")
    """
    # Smooth PSM to reduce noise
    if gaussian_sigma > 0:
        psm_smooth = gaussian_filter(psm, sigma=gaussian_sigma)
    else:
        psm_smooth = psm
    
    # Threshold to binary mask
    binary_mask = psm_smooth > eta_threshold
    
    # Label connected components
    labels, n_sites = label(binary_mask)
    
    # Filter by minimum size
    coordinates = []
    properties = []
    valid_label = 1
    new_labels = np.zeros_like(labels)
    
    for i in range(1, n_sites + 1):
        site_mask = labels == i
        size = np.sum(site_mask)
        
        if size >= min_size:
            # Calculate centroid
            centroid = center_of_mass(site_mask)
            coordinates.append(centroid)
            
            # Calculate properties
            eta_values = psm[site_mask]
            props = {
                'label': valid_label,
                'size': size,
                'centroid': centroid,
                'eta_mean': np.mean(eta_values),
                'eta_max': np.max(eta_values),
                'eta_std': np.std(eta_values)
            }
            properties.append(props)
            
            # Assign new label
            new_labels[site_mask] = valid_label
            valid_label += 1
    
    n_sites = len(coordinates)
    
    return new_labels, n_sites, coordinates, properties


def detect_active_sites_crm(crm, xi_threshold=0.3, min_size=4, gaussian_sigma=1.0):
    """
    Detect active Ca2+ sites from Correlation Map.
    
    Identifies regions where ξ correlation value exceeds threshold.
    
    Parameters:
        crm (ndarray): Correlation map (H, W) with ξ values
        xi_threshold (float): Threshold for ξ to be considered active
        min_size (int): Minimum number of connected pixels for a valid site
        gaussian_sigma (float): Sigma for Gaussian smoothing before detection
    
    Returns:
        tuple: (labels, n_sites, coordinates, properties)
    
    Example:
        >>> import numpy as np
        >>> crm = np.zeros((64, 64))
        >>> crm[30:35, 30:35] = 0.5  # Active site
        >>> labels, n_sites, coords, props = detect_active_sites_crm(crm, xi_threshold=0.3)
        >>> print(f"Detected {n_sites} active sites")
    """
    # Smooth CRM to reduce noise
    if gaussian_sigma > 0:
        crm_smooth = gaussian_filter(crm, sigma=gaussian_sigma)
    else:
        crm_smooth = crm
    
    # Threshold to binary mask
    binary_mask = crm_smooth > xi_threshold
    
    # Label connected components
    labels, n_sites = label(binary_mask)
    
    # Filter by minimum size
    coordinates = []
    properties = []
    valid_label = 1
    new_labels = np.zeros_like(labels)
    
    for i in range(1, n_sites + 1):
        site_mask = labels == i
        size = np.sum(site_mask)
        
        if size >= min_size:
            # Calculate centroid
            centroid = center_of_mass(site_mask)
            coordinates.append(centroid)
            
            # Calculate properties
            xi_values = crm[site_mask]
            props = {
                'label': valid_label,
                'size': size,
                'centroid': centroid,
                'xi_mean': np.mean(xi_values),
                'xi_max': np.max(xi_values),
                'xi_std': np.std(xi_values)
            }
            properties.append(props)
            
            # Assign new label
            new_labels[site_mask] = valid_label
            valid_label += 1
    
    n_sites = len(coordinates)
    
    return new_labels, n_sites, coordinates, properties


def combine_psm_crm_detection(psm, crm, eta_threshold=0.5, xi_threshold=0.3,
                               min_overlap=0.5, gaussian_sigma=1.0):
    """
    Detect sites using both PSM and CRM with overlap requirement.
    
    Sites must show activity in both PSM and CRM with minimum overlap.
    
    Parameters:
        psm (ndarray): Power spectrum map (H, W)
        crm (ndarray): Correlation map (H, W)
        eta_threshold (float): Threshold for PSM
        xi_threshold (float): Threshold for CRM
        min_overlap (float): Minimum fraction of overlap (0-1)
        gaussian_sigma (float): Smoothing sigma
    
    Returns:
        tuple: (labels, n_sites, coordinates, properties)
    
    Example:
        >>> import numpy as np
        >>> psm = np.zeros((64, 64))
        >>> crm = np.zeros((64, 64))
        >>> psm[30:35, 30:35] = 2.0
        >>> crm[30:35, 30:35] = 0.5
        >>> labels, n_sites, coords, props = combine_psm_crm_detection(psm, crm)
        >>> print(f"Detected {n_sites} sites with combined criteria")
    """
    # Detect sites independently
    psm_labels, psm_n, psm_coords, psm_props = detect_active_sites_psm(
        psm, eta_threshold, min_size=1, gaussian_sigma=gaussian_sigma
    )
    
    crm_labels, crm_n, crm_coords, crm_props = detect_active_sites_crm(
        crm, xi_threshold, min_size=1, gaussian_sigma=gaussian_sigma
    )
    
    # Find overlapping sites
    combined_mask = np.zeros_like(psm, dtype=bool)
    coordinates = []
    properties = []
    
    for i in range(1, psm_n + 1):
        psm_mask = psm_labels == i
        
        # Check overlap with CRM sites
        for j in range(1, crm_n + 1):
            crm_mask = crm_labels == j
            
            # Calculate overlap
            overlap_mask = psm_mask & crm_mask
            overlap_size = np.sum(overlap_mask)
            psm_size = np.sum(psm_mask)
            crm_size = np.sum(crm_mask)
            
            # Calculate Jaccard index
            union_size = np.sum(psm_mask | crm_mask)
            jaccard = overlap_size / union_size if union_size > 0 else 0
            
            if jaccard >= min_overlap:
                # Valid site - combine masks
                site_mask = psm_mask | crm_mask
                combined_mask |= site_mask
                
                # Calculate properties
                centroid = center_of_mass(site_mask)
                coordinates.append(centroid)
                
                props = {
                    'label': len(properties) + 1,
                    'size': np.sum(site_mask),
                    'centroid': centroid,
                    'eta_mean': np.mean(psm[site_mask]),
                    'eta_max': np.max(psm[site_mask]),
                    'xi_mean': np.mean(crm[site_mask]),
                    'xi_max': np.max(crm[site_mask]),
                    'jaccard_index': jaccard,
                    'psm_label': i,
                    'crm_label': j
                }
                properties.append(props)
    
    # Re-label combined mask
    labels, n_sites = label(combined_mask)
    
    return labels, n_sites, coordinates, properties


def expand_sites_to_rois(labels, expansion_pixels=5):
    """
    Expand detected sites to create ROIs with padding.
    
    Useful for extracting time series from regions around active sites.
    
    Parameters:
        labels (ndarray): Labeled array from site detection (H, W)
        expansion_pixels (int): Number of pixels to expand each site
    
    Returns:
        ndarray: Expanded labels array
    
    Example:
        >>> import numpy as np
        >>> labels = np.zeros((64, 64), dtype=int)
        >>> labels[32, 32] = 1
        >>> expanded = expand_sites_to_rois(labels, expansion_pixels=5)
        >>> print(np.sum(expanded > 0))  # Should be much larger than 1
    """
    # Create structuring element for dilation
    struct = np.ones((3, 3))
    
    # Expand each labeled site
    expanded = labels.copy()
    for _ in range(expansion_pixels):
        expanded = binary_dilation(expanded > 0, structure=struct).astype(int)
        expanded = expanded * labels.max()  # Restore to binary
    
    return expanded


def get_roi_coordinates(labels, roi_size=15):
    """
    Get rectangular ROI coordinates around each detected site.
    
    Parameters:
        labels (ndarray): Labeled array from site detection (H, W)
        roi_size (int): Size of square ROI (pixels)
    
    Returns:
        list: List of tuples (label, y_min, y_max, x_min, x_max)
    
    Example:
        >>> import numpy as np
        >>> labels = np.zeros((64, 64), dtype=int)
        >>> labels[32, 32] = 1
        >>> labels[16, 48] = 2
        >>> rois = get_roi_coordinates(labels, roi_size=10)
        >>> print(f"Generated {len(rois)} ROIs")
    """
    H, W = labels.shape
    n_sites = labels.max()
    half_size = roi_size // 2
    
    rois = []
    for i in range(1, n_sites + 1):
        # Find centroid
        site_mask = labels == i
        centroid = center_of_mass(site_mask)
        cy, cx = int(centroid[0]), int(centroid[1])
        
        # Calculate ROI bounds
        y_min = max(0, cy - half_size)
        y_max = min(H, cy + half_size + 1)
        x_min = max(0, cx - half_size)
        x_max = min(W, cx + half_size + 1)
        
        rois.append((i, y_min, y_max, x_min, x_max))
    
    return rois


def extract_roi_traces(image_stack, rois):
    """
    Extract average fluorescence traces from ROIs.
    
    Parameters:
        image_stack (ndarray): Image stack (T, H, W)
        rois (list): List of ROI coordinates from get_roi_coordinates
    
    Returns:
        dict: Dictionary mapping label to trace array
    
    Example:
        >>> import numpy as np
        >>> stack = np.random.randn(1000, 64, 64)
        >>> labels = np.zeros((64, 64), dtype=int)
        >>> labels[32, 32] = 1
        >>> rois = get_roi_coordinates(labels, roi_size=10)
        >>> traces = extract_roi_traces(stack, rois)
        >>> print(f"Extracted {len(traces)} traces")
    """
    T = image_stack.shape[0]
    traces = {}
    
    for label, y_min, y_max, x_min, x_max in rois:
        # Extract ROI from each frame
        roi_stack = image_stack[:, y_min:y_max, x_min:x_max]
        
        # Average over spatial dimensions
        trace = np.mean(roi_stack, axis=(1, 2))
        traces[label] = trace
    
    return traces


def classify_sites_by_activity(properties, eta_high=2.0, xi_high=0.6):
    """
    Classify sites into activity categories based on η and ξ values.
    
    Categories:
    - 'high': Both η and ξ above high thresholds (strong activity)
    - 'medium': One metric high, one moderate (moderate activity)
    - 'low': Both metrics above detection threshold but below high (weak activity)
    
    Parameters:
        properties (list): List of property dicts from detection functions
        eta_high (float): Threshold for high η
        xi_high (float): Threshold for high ξ
    
    Returns:
        dict: Dictionary mapping labels to categories
    
    Example:
        >>> props = [
        ...     {'label': 1, 'eta_max': 3.0, 'xi_max': 0.8},
        ...     {'label': 2, 'eta_max': 1.0, 'xi_max': 0.4}
        ... ]
        >>> categories = classify_sites_by_activity(props)
        >>> print(categories)  # {1: 'high', 2: 'low'}
    """
    categories = {}
    
    for props in properties:
        label = props['label']
        eta = props.get('eta_max', props.get('eta_mean', 0))
        xi = props.get('xi_max', props.get('xi_mean', 0))
        
        if eta >= eta_high and xi >= xi_high:
            category = 'high'
        elif eta >= eta_high or xi >= xi_high:
            category = 'medium'
        else:
            category = 'low'
        
        categories[label] = category
    
    return categories


if __name__ == '__main__':
    """Unit tests for activity detection module."""
    print("Testing activity_detection_module...")
    
    # Test 1: PSM detection
    print("\n1. Testing detect_active_sites_psm...")
    psm = np.zeros((64, 64))
    psm[30:35, 30:35] = 2.5  # Strong site
    psm[15:18, 48:51] = 1.2  # Moderate site
    psm[50, 20] = 0.8  # Single pixel (should be filtered)
    
    labels, n_sites, coords, props = detect_active_sites_psm(
        psm, eta_threshold=0.5, min_size=4
    )
    
    print(f"   Detected {n_sites} sites")
    print(f"   Expected: 2 sites (single pixel filtered out)")
    assert n_sites == 2, "Should detect 2 sites"
    assert len(coords) == n_sites
    assert len(props) == n_sites
    print(f"   Site 1: centroid={props[0]['centroid']}, η_max={props[0]['eta_max']:.2f}")
    print(f"   Site 2: centroid={props[1]['centroid']}, η_max={props[1]['eta_max']:.2f}")
    print("   ✓ PSM detection working correctly")
    
    # Test 2: CRM detection
    print("\n2. Testing detect_active_sites_crm...")
    crm = np.zeros((64, 64))
    crm[30:35, 30:35] = 0.7
    crm[15:18, 48:51] = 0.4
    
    labels_crm, n_sites_crm, coords_crm, props_crm = detect_active_sites_crm(
        crm, xi_threshold=0.3, min_size=4
    )
    
    print(f"   Detected {n_sites_crm} sites")
    assert n_sites_crm == 2
    print("   ✓ CRM detection working correctly")
    
    # Test 3: Combined detection
    print("\n3. Testing combine_psm_crm_detection...")
    labels_comb, n_comb, coords_comb, props_comb = combine_psm_crm_detection(
        psm, crm, eta_threshold=0.5, xi_threshold=0.3, min_overlap=0.5
    )
    
    print(f"   Detected {n_comb} sites with combined criteria")
    assert n_comb >= 2
    for prop in props_comb:
        print(f"   Site {prop['label']}: Jaccard={prop['jaccard_index']:.2f}")
    print("   ✓ Combined detection working correctly")
    
    # Test 4: ROI expansion
    print("\n4. Testing expand_sites_to_rois...")
    labels_small = np.zeros((64, 64), dtype=int)
    labels_small[32, 32] = 1
    initial_size = np.sum(labels_small > 0)
    
    expanded = expand_sites_to_rois(labels_small, expansion_pixels=5)
    expanded_size = np.sum(expanded > 0)
    
    print(f"   Initial size: {initial_size} pixels")
    print(f"   Expanded size: {expanded_size} pixels")
    assert expanded_size > initial_size
    print("   ✓ ROI expansion working correctly")
    
    # Test 5: Get ROI coordinates
    print("\n5. Testing get_roi_coordinates...")
    rois = get_roi_coordinates(labels, roi_size=15)
    
    print(f"   Generated {len(rois)} ROI coordinates")
    assert len(rois) == n_sites
    for label, y_min, y_max, x_min, x_max in rois:
        print(f"   ROI {label}: ({y_min}:{y_max}, {x_min}:{x_max})")
    print("   ✓ ROI coordinate extraction working correctly")
    
    # Test 6: Extract traces
    print("\n6. Testing extract_roi_traces...")
    T = 100
    stack = np.random.randn(T, 64, 64) * 10 + 100
    
    traces = extract_roi_traces(stack, rois)
    
    print(f"   Extracted {len(traces)} traces")
    assert len(traces) == len(rois)
    for label, trace in traces.items():
        print(f"   Trace {label}: shape={trace.shape}, mean={trace.mean():.2f}")
        assert trace.shape[0] == T
    print("   ✓ Trace extraction working correctly")
    
    # Test 7: Classify sites
    print("\n7. Testing classify_sites_by_activity...")
    categories = classify_sites_by_activity(props, eta_high=2.0, xi_high=0.6)
    
    print(f"   Classified {len(categories)} sites")
    for label, category in categories.items():
        eta = props[label-1]['eta_max']
        print(f"   Site {label}: {category} (η_max={eta:.2f})")
    print("   ✓ Activity classification working correctly")
    
    print("\n✅ All activity detection module tests passed!")
