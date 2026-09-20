"""
Generate clean images (without noise) for testing.

Original parameters:
- gaussian = 0.8 (blur)
- std_dev = 0.3 (30% noise)

We'll generate:
1. No noise, no blur (perfectly clean)
2. No noise, with blur (blur only)
3. Low noise (10%)
4. Original (30% noise) for comparison
"""

import numpy as np
import os
import sys
import math
import tifffile
import scipy.ndimage
from glob import glob

# Add spam to path
sys.path.insert(0, '/home/gma/epita/PFEE/data_yukiko/generate_dataset')

try:
    import spam.kalisphera
except ImportError:
    print("ERROR: spam not installed. Run: pip install spam")
    sys.exit(1)

# Folders
folder_dem = "/home/gma/epita/PFEE/data_yukiko/generate_dataset/dem_data/monotonic/"
folder_output = "/home/gma/epita/PFEE/experiments/runs/exp_008_clean_images/dataset/"

# Create output folder
os.makedirs(folder_output, exist_ok=True)

# Parameters
pixel_size = 60.e-6  # m/pixel

# Use first file for testing
filename = "triax.-1.eps=0.spheres"

# Noise configurations to test
configs = [
    {"name": "clean", "gaussian": 0.0, "std_dev": 0.0},
    {"name": "blur_only", "gaussian": 0.8, "std_dev": 0.0},
    {"name": "low_noise", "gaussian": 0.8, "std_dev": 0.1},
    {"name": "original", "gaussian": 0.8, "std_dev": 0.3},
]


def build_image(filename, gaussian, std_dev, output_name):
    """Generate image with specified noise parameters."""
    
    positions_file = folder_dem + "positions/" + filename + ".txt"
    
    if not os.path.exists(positions_file):
        print(f"ERROR: File not found: {positions_file}")
        return None
    
    num_lines = sum(1 for line in open(positions_file))
    
    # Load DEM data
    box_size_dem = np.genfromtxt(positions_file, skip_footer=(num_lines-1), comments='%', usecols=(1,2,3))
    centres = np.loadtxt(positions_file, skiprows=0, usecols=(4,3,2))  # x,y,z
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
    
    print(f"Generating {output_name}: box_size={box_size}, gaussian={gaussian}, noise={std_dev}")
    
    # Create image
    Box = np.zeros((box_size, box_size, box_size), dtype="<f8")
    spam.kalisphera.makeSphere(Box, centres, radii)
    
    # Clamp values
    Box[Box > 1.0] = 1.0
    Box[Box < 0.0] = 0.0
    
    # Transform to image with peaks at 0.25 (void) and 0.75 (particle)
    Box = Box * 0.5 + 0.25
    
    # Apply blur
    if gaussian > 0:
        Box = scipy.ndimage.gaussian_filter(Box, sigma=gaussian)
    
    # Apply noise
    if std_dev > 0:
        Box = np.random.normal(Box, scale=std_dev)
    
    # Convert to uint16
    Box = np.clip(Box * 65536, 0, 65535)
    
    output_path = folder_output + output_name + ".tif"
    tifffile.imwrite(output_path, Box.astype('uint16'))
    print(f"  Saved: {output_path}")
    
    return Box


def main():
    print("=" * 60)
    print("GENERATING CLEAN TEST IMAGES")
    print("=" * 60)
    
    for config in configs:
        build_image(
            filename,
            gaussian=config["gaussian"],
            std_dev=config["std_dev"],
            output_name=config["name"]
        )
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
