"""
Baseline Watershed Segmentation.
Copy of the original implementation from data_yukiko/watershed_segmentation.py
"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def watershed_segmentation(distance_map, min_distance=3, verbose=True):
    """
    Perform watershed segmentation on a distance map.
    
    Steps:
    1. Extract grain mask (distance > 0)
    2. Find local maxima (grain centers)
    3. Create markers from maxima
    4. Apply watershed algorithm
    5. Mask the result
    
    Args:
        distance_map: 3D distance map (center=1, edge=0)
        min_distance: Minimum distance between peaks
        verbose: Print progress messages
    
    Returns:
        labels: Labeled grains (0=background)
        num_grains: Number of grains detected
        grain_mask: Binary mask of all grains
    """
    if verbose:
        print(f"Input shape: {distance_map.shape}")
        print(f"Distance map range: [{distance_map.min():.4f}, {distance_map.max():.4f}]")
    
    # Step 1: Extract grain mask
    grain_mask = distance_map > 0
    if verbose:
        print(f"Grain mask voxels: {grain_mask.sum()}")
    
    # Step 2: Find local maxima
    coordinates = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=grain_mask.astype(int),
        exclude_border=False
    )
    
    num_grains = len(coordinates)
    if verbose:
        print(f"Found {num_grains} local maxima")
    
    # Step 3: Create markers
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    # Step 4: Watershed
    labels = watershed(-distance_map, markers, mask=grain_mask)
    
    # Step 5: Apply mask
    labels = labels * grain_mask.astype(np.int32)
    
    if verbose:
        print(f"Segmentation complete: {num_grains} grains")
    
    return labels, num_grains, grain_mask
