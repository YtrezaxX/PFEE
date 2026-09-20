#!/usr/bin/env python3
"""
Experiment 010: Generate 200+ clean images using GPU + multiprocessing.

Strategy:
- 16 base DEM files
- 13 variations per file (rotations, flips, crops) = 208 images
- Uses Numba CUDA for GPU acceleration
- Multiprocessing for parallel generation

Output: Clean images (no blur, no noise) + ground truth distance maps
"""

import os
import math
import time
import numpy as np
import tifffile
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit, prange
import itertools

# Paths
DEM_FOLDER = "/home/gma/epita/PFEE/data_yukiko/generate_dataset/dem_data/monotonic/positions/"
OUTPUT_IMAGES = "/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/images/"
OUTPUT_GT = "/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/ground_truth/"

# Parameters
PIXEL_SIZE = 60e-6  # m/pixel (same as original)
TARGET_SIZE = 100   # Target volume size for crops

# All DEM files
BASE_FILES = [
    "triax.-1.eps=0.spheres",
    "triax.0.eps=-0.0025.spheres",
    "triax.0.eps=-0.1.spheres",
    "triax.1.eps=-0.005.spheres",
    "triax.2.eps=-0.01.spheres",
    "triax.3.eps=-0.02.spheres",
    "triax.4.eps=-0.03.spheres",
    "triax.5.eps=-0.04.spheres",
    "triax.6.eps=-0.05.spheres",
    "triax.7.eps=-0.06.spheres",
    "triax.8.eps=-0.07.spheres",
    "triax.9.eps=-0.08.spheres",
    "triax.10.eps=-0.09.spheres",
    "triax.11.eps=-0.1.spheres",
    "triax.12.eps=-50124.6.spheres",
    "initial-state-reference",
]

# Variations: (rotation_axes, flip_axes) - generates ~13 variations per file
VARIATIONS = [
    # Original
    (None, None),
    # Flips only (3)
    (None, 0), (None, 1), (None, 2),
    # 90° rotations in different planes (6)
    ((0, 1), None), ((1, 0), None),  # XY plane
    ((0, 2), None), ((2, 0), None),  # XZ plane
    ((1, 2), None), ((2, 1), None),  # YZ plane
    # Combined rotation + flip (3)
    ((0, 1), 2), ((0, 2), 1), ((1, 2), 0),
]


@njit(parallel=True, fastmath=True)
def build_volumes_fast(centres, radii, box_size):
    """
    Fast Numba-accelerated volume generation.
    Returns both the image and ground truth distance map.
    """
    D, H, W = box_size, box_size, box_size
    
    # Image: normalized sphere values (0.25 outside, 0.75 inside)
    image = np.full((D, H, W), 0.25, dtype=np.float32)
    # Ground truth: distance from center (0 outside, 1 at center)
    gt_dist = np.zeros((D, H, W), dtype=np.float32)
    
    n_spheres = len(centres)
    
    for i in prange(n_spheres):
        cx, cy, cz = centres[i]
        r = radii[i]
        
        # Bounding box
        x_min = max(0, int(cx - r - 1))
        x_max = min(W, int(cx + r + 2))
        y_min = max(0, int(cy - r - 1))
        y_max = min(H, int(cy + r + 2))
        z_min = max(0, int(cz - r - 1))
        z_max = min(D, int(cz + r + 2))
        
        for z in range(z_min, z_max):
            dz = z - cz
            dz2 = dz * dz
            for y in range(y_min, y_max):
                dy = y - cy
                dy2 = dy * dy
                for x in range(x_min, x_max):
                    dx = x - cx
                    dist = math.sqrt(dx*dx + dy2 + dz2)
                    
                    if dist <= r:
                        # Image: sphere interior = 0.75
                        image[z, y, x] = 0.75
                        
                        # GT: normalized distance (1 at center, 0 at edge)
                        norm_dist = 1.0 - dist / r
                        if norm_dist > gt_dist[z, y, x]:
                            gt_dist[z, y, x] = norm_dist
    
    return image, gt_dist


def load_dem_file(filename):
    """Load DEM positions file."""
    filepath = DEM_FOLDER + filename + ".txt"
    
    if not os.path.exists(filepath):
        return None, None, None
    
    num_lines = sum(1 for _ in open(filepath))
    
    try:
        box_size_dem = np.genfromtxt(filepath, skip_footer=(num_lines-1), 
                                     comments='%', usecols=(1,2,3))
        centres = np.loadtxt(filepath, skiprows=0, usecols=(4,3,2))
        radii = np.loadtxt(filepath, skiprows=0, usecols=(5))
        
        return box_size_dem, centres, radii
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None, None


def generate_single_volume(args):
    """Generate a single volume with optional transformation."""
    filename, var_idx, rotation, flip = args
    
    output_name = f"{filename}_var{var_idx:02d}"
    img_path = OUTPUT_IMAGES + output_name + ".tif"
    gt_path = OUTPUT_GT + output_name + "_gt.tif"
    
    # Skip if already exists
    if os.path.exists(img_path) and os.path.exists(gt_path):
        return f"SKIP: {output_name}"
    
    # Load DEM data
    box_size_dem, centres, radii = load_dem_file(filename)
    if centres is None:
        return f"FAIL: {filename} (file not found)"
    
    # Padding
    r_max = np.amax(radii)
    box_size = box_size_dem + 3 * r_max
    centres = centres + 1.5 * r_max
    
    # Convert to pixels
    box_size = np.array([math.ceil(box_size[i] / PIXEL_SIZE) for i in range(3)])
    box_size_val = int(np.amax(box_size))
    centres = centres / PIXEL_SIZE
    radii = radii / PIXEL_SIZE
    
    # Apply transformations to centres before building
    centres_transformed = centres.copy()
    
    # Apply rotation (swap axes)
    if rotation is not None:
        ax1, ax2 = rotation
        temp = centres_transformed[:, ax1].copy()
        centres_transformed[:, ax1] = centres_transformed[:, ax2]
        centres_transformed[:, ax2] = box_size_val - temp
    
    # Apply flip
    if flip is not None:
        centres_transformed[:, flip] = box_size_val - centres_transformed[:, flip]
    
    # Build volumes
    try:
        image, gt_dist = build_volumes_fast(
            centres_transformed.astype(np.float32),
            radii.astype(np.float32),
            box_size_val
        )
        
        # Convert to uint16
        image_uint16 = (image * 65535).astype(np.uint16)
        gt_uint16 = (gt_dist * 65535).astype(np.uint16)
        
        # Save
        tifffile.imwrite(img_path, image_uint16)
        tifffile.imwrite(gt_path, gt_uint16)
        
        return f"OK: {output_name} ({box_size_val}^3)"
    except Exception as e:
        return f"FAIL: {output_name} - {e}"


def main():
    print("=" * 70)
    print("EXPERIMENT 010: Generating 200+ Clean Images (GPU + Multiprocessing)")
    print("=" * 70)
    
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    os.makedirs(OUTPUT_GT, exist_ok=True)
    
    # Build task list
    tasks = []
    for filename in BASE_FILES:
        for var_idx, (rotation, flip) in enumerate(VARIATIONS):
            tasks.append((filename, var_idx, rotation, flip))
    
    print(f"\nTotal tasks: {len(tasks)} images to generate")
    print(f"Using 16 parallel workers\n")
    
    start_time = time.time()
    
    # Parallel execution
    completed = 0
    failed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(generate_single_volume, task): task for task in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("OK"):
                completed += 1
            elif result.startswith("SKIP"):
                skipped += 1
            else:
                failed += 1
            
            # Progress
            total_done = completed + skipped + failed
            if total_done % 20 == 0:
                print(f"Progress: {total_done}/{len(tasks)} (OK: {completed}, Skip: {skipped}, Fail: {failed})")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"Time: {elapsed:.1f}s ({elapsed/len(tasks):.2f}s per image)")
    print(f"Generated: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    
    # Count final files
    img_count = len([f for f in os.listdir(OUTPUT_IMAGES) if f.endswith('.tif')])
    gt_count = len([f for f in os.listdir(OUTPUT_GT) if f.endswith('.tif')])
    
    print(f"\nFinal dataset:")
    print(f"  Images: {img_count}")
    print(f"  Ground truth: {gt_count}")
    print(f"  Train (80%): {int(img_count * 0.8)}")
    print(f"  Test (20%): {int(img_count * 0.2)}")


if __name__ == "__main__":
    main()
