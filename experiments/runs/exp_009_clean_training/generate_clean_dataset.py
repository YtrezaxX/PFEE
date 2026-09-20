"""
Generate CLEAN dataset (no blur, no noise) for training.

Original: gaussian=0.8, std_dev=0.3
Clean:    gaussian=0.0, std_dev=0.0
"""

import numpy as np
import os
import sys
import math
import tifffile
import scipy.ndimage
from multiprocessing import Pool

sys.path.insert(0, '/home/gma/epita/PFEE/data_yukiko/generate_dataset')

try:
    import spam.kalisphera
except ImportError:
    print("ERROR: spam not installed")
    sys.exit(1)

# Folders
folder_dem = "/home/gma/epita/PFEE/data_yukiko/generate_dataset/dem_data/monotonic/"
folder_output = "/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/images/"
folder_gt = "/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/ground_truth/"

os.makedirs(folder_output, exist_ok=True)
os.makedirs(folder_gt, exist_ok=True)

# Parameters - NO BLUR, NO NOISE
gaussian = 0.0
std_dev = 0.0
pixel_size = 60.e-6

# All 14 files
filenames = [
    "triax.-1.eps=0.spheres",
    "triax.0.eps=-0.0025.spheres",
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
]


def build_image(filename):
    """Generate clean image."""
    positions_file = folder_dem + "positions/" + filename + ".txt"
    
    if not os.path.exists(positions_file):
        print(f"  SKIP: {filename} (file not found)")
        return None
    
    output_path = folder_output + filename + "_clean.tif"
    if os.path.exists(output_path):
        print(f"  SKIP: {filename} (already exists)")
        return output_path
    
    num_lines = sum(1 for line in open(positions_file))
    
    # Load DEM data
    box_size_dem = np.genfromtxt(positions_file, skip_footer=(num_lines-1), comments='%', usecols=(1,2,3))
    centres = np.loadtxt(positions_file, skiprows=0, usecols=(4,3,2))
    radii = np.loadtxt(positions_file, skiprows=0, usecols=(5))
    
    # Padding
    r_max = np.amax(radii)
    box_size = box_size_dem + 3*r_max
    centres[:,:] = centres[:,:] + 1.5*r_max
    
    # Convert to pixels
    box_size = np.array([math.ceil(box_size[0]/pixel_size), 
                         math.ceil(box_size[1]/pixel_size), 
                         math.ceil(box_size[2]/pixel_size)])
    box_size = int(np.amax(box_size))
    
    centres = centres / pixel_size
    radii = radii / pixel_size
    
    print(f"  Generating {filename}: {box_size}^3 voxels")
    
    # Create image
    Box = np.zeros((box_size, box_size, box_size), dtype="<f8")
    spam.kalisphera.makeSphere(Box, centres, radii)
    
    Box[Box > 1.0] = 1.0
    Box[Box < 0.0] = 0.0
    
    # Transform to 0.25-0.75 range
    Box = Box * 0.5 + 0.25
    
    # NO blur, NO noise for clean images
    
    # Convert to uint16
    Box = np.clip(Box * 65536, 0, 65535)
    
    tifffile.imwrite(output_path, Box.astype('uint16'))
    return output_path


def generate_distance_map(filename):
    """Generate ground truth distance map."""
    positions_file = folder_dem + "positions/" + filename + ".txt"
    
    if not os.path.exists(positions_file):
        return None
    
    output_path = folder_gt + filename + "_ground_truth.tif"
    if os.path.exists(output_path):
        print(f"  SKIP GT: {filename} (already exists)")
        return output_path
    
    num_lines = sum(1 for line in open(positions_file))
    
    box_size_dem = np.genfromtxt(positions_file, skip_footer=(num_lines-1), comments='%', usecols=(1,2,3))
    centres = np.loadtxt(positions_file, skiprows=0, usecols=(4,3,2))
    radii = np.loadtxt(positions_file, skiprows=0, usecols=(5))
    
    r_max = np.amax(radii)
    box_size = box_size_dem + 3*r_max
    centres[:,:] = centres[:,:] + 1.5*r_max
    
    box_size = np.array([math.ceil(box_size[0]/pixel_size), 
                         math.ceil(box_size[1]/pixel_size), 
                         math.ceil(box_size[2]/pixel_size)])
    box_size = int(np.amax(box_size))
    
    centres = centres / pixel_size
    radii = radii / pixel_size
    
    print(f"  Generating GT {filename}")
    
    # Distance map
    Box_GT = np.zeros((box_size, box_size, box_size), dtype="<f8")
    
    z, y, x = np.ogrid[:box_size, :box_size, :box_size]
    
    for i in range(len(centres)):
        cx, cy, cz = centres[i]
        r = radii[i]
        
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        inside = dist <= r
        
        # Normalized distance: 1 at center, 0 at edge
        normalized_dist = np.where(inside, 1 - dist/r, 0)
        Box_GT = np.maximum(Box_GT, normalized_dist)
    
    Box_GT = np.clip(Box_GT * 65536, 0, 65535)
    tifffile.imwrite(output_path, Box_GT.astype('uint16'))
    return output_path


def main():
    print("=" * 60)
    print("GENERATING CLEAN TRAINING DATASET")
    print("gaussian=0.0, noise=0.0")
    print("=" * 60)
    
    print("\n1. Generating clean images...")
    for f in filenames:
        build_image(f)
    
    print("\n2. Generating ground truth distance maps...")
    for f in filenames:
        generate_distance_map(f)
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    
    # Count files
    img_count = len([f for f in os.listdir(folder_output) if f.endswith('.tif')])
    gt_count = len([f for f in os.listdir(folder_gt) if f.endswith('.tif')])
    print(f"Images: {img_count}")
    print(f"Ground truth: {gt_count}")


if __name__ == "__main__":
    main()
