"""
Improved Watershed Segmentation.

Improvements over baseline:
- H-maxima transform for better center detection
- Adaptive min_distance based on grain statistics
- Optional morphological operations
- Distance map thresholding
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_opening, binary_erosion
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima, ball


def watershed_h_maxima(distance_map, h=0.05, verbose=True):
    """
    Watershed segmentation using h-maxima transform.
    
    H-maxima suppresses local maxima whose height is less than h.
    This is more robust to noise than peak_local_max.
    
    Why it works better:
    - Real grain centers have high distance values
    - Noise creates small spurious peaks
    - H-maxima filters out peaks smaller than h
    
    Args:
        distance_map: 3D distance map
        h: Height threshold for h-maxima
        verbose: Print progress
    
    Returns:
        labels, num_grains, grain_mask
    """
    if verbose:
        print(f"Using h-maxima transform with h={h}")
    
    grain_mask = distance_map > 0
    
    # H-maxima transform
    markers_bool = h_maxima(distance_map, h=h)
    coordinates = np.array(np.where(markers_bool)).T
    
    num_grains = len(coordinates)
    if verbose:
        print(f"H-maxima found {num_grains} centers")
    
    # Create markers
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    # Watershed
    labels = watershed(-distance_map, markers, mask=grain_mask)
    labels = labels * grain_mask.astype(np.int32)
    
    return labels, num_grains, grain_mask


def watershed_adaptive(distance_map, grain_radius_estimate=5, verbose=True):
    """
    Watershed with adaptive min_distance.
    
    Adapts min_distance based on estimated grain radius.
    
    Args:
        distance_map: 3D distance map
        grain_radius_estimate: Estimated average grain radius in voxels
        verbose: Print progress
    
    Returns:
        labels, num_grains, grain_mask
    """
    # Adaptive min_distance: 80% of estimated radius
    min_distance = max(2, int(grain_radius_estimate * 0.8))
    
    if verbose:
        print(f"Adaptive min_distance: {min_distance}")
    
    grain_mask = distance_map > 0
    
    coordinates = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=grain_mask.astype(int),
        exclude_border=False
    )
    
    num_grains = len(coordinates)
    if verbose:
        print(f"Found {num_grains} centers")
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    labels = watershed(-distance_map, markers, mask=grain_mask)
    labels = labels * grain_mask.astype(np.int32)
    
    return labels, num_grains, grain_mask


def watershed_with_preprocessing(distance_map, threshold=0.05, opening_radius=1,
                                 min_distance=3, verbose=True):
    """
    Watershed with distance map preprocessing.
    
    Preprocessing steps:
    1. Threshold low values (remove noise)
    2. Morphological opening (remove small artifacts)
    
    Args:
        distance_map: 3D distance map
        threshold: Minimum distance value to keep
        opening_radius: Radius for morphological opening
        min_distance: For peak_local_max
        verbose: Print progress
    
    Returns:
        labels, num_grains, grain_mask
    """
    if verbose:
        print(f"Preprocessing: threshold={threshold}, opening_radius={opening_radius}")
    
    # Copy and threshold
    dist_processed = distance_map.copy()
    dist_processed[dist_processed < threshold] = 0
    
    # Morphological opening on mask
    grain_mask = dist_processed > 0
    if opening_radius > 0:
        struct = ball(opening_radius)
        grain_mask = binary_opening(grain_mask, structure=struct)
        dist_processed = dist_processed * grain_mask
    
    if verbose:
        print(f"After preprocessing: {grain_mask.sum()} voxels in mask")
    
    # Find centers
    coordinates = peak_local_max(
        dist_processed,
        min_distance=min_distance,
        labels=grain_mask.astype(int),
        exclude_border=False
    )
    
    num_grains = len(coordinates)
    if verbose:
        print(f"Found {num_grains} centers")
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    labels = watershed(-dist_processed, markers, mask=grain_mask)
    labels = labels * grain_mask.astype(np.int32)
    
    return labels, num_grains, grain_mask


def watershed_combined(distance_map, h=0.05, threshold=0.03, verbose=True):
    """
    Combined watershed with all improvements.
    
    - Threshold low values
    - H-maxima for robust center detection
    
    Recommended for best results.
    
    Args:
        distance_map: 3D distance map
        h: H-maxima height threshold
        threshold: Distance map threshold
        verbose: Print progress
    
    Returns:
        labels, num_grains, grain_mask
    """
    if verbose:
        print(f"Combined watershed: h={h}, threshold={threshold}")
    
    # Threshold
    dist_processed = distance_map.copy()
    dist_processed[dist_processed < threshold] = 0
    
    grain_mask = dist_processed > 0
    
    # H-maxima
    markers_bool = h_maxima(dist_processed, h=h)
    coordinates = np.array(np.where(markers_bool)).T
    
    num_grains = len(coordinates)
    if verbose:
        print(f"Found {num_grains} grain centers")
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    labels = watershed(-dist_processed, markers, mask=grain_mask)
    labels = labels * grain_mask.astype(np.int32)
    
    return labels, num_grains, grain_mask
