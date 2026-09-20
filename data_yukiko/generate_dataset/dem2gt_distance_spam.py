#!/usr/bin/python
# -*- coding: utf-8 -*-

# Ground truth distance map generator for DEM sphere assemblies

import numpy
import os
import math
import tifffile
from multiprocessing import Pool

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

folder       = "dem_data/monotonic/"
folderGT     = "../dataset/ground_truth/"

if not os.path.exists(folder):
    print(f"{bcolors.FAIL} No Input data detected, please download the data and put it in {folder} {bcolors.ENDC}")
    exit()

if not os.path.exists(folderGT):
    os.makedirs(folderGT)
    print(f"{bcolors.WARNING} No ground truth folder detected. Creating {folderGT}{bcolors.ENDC}")

# input
filenames = ["triax.-1.eps=0.spheres",
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

def generate_distance_map(box_shape, centres, radii):
    """
    Generate ground truth distance map where:
    - Center of each grain = 1.0
    - Edge of each grain approaches 0.0
    - Distance is normalized by radius for each grain
    """
    Box_GT = numpy.zeros(box_shape, dtype="<f8")
    
    # Create coordinate grids
    z, y, x = numpy.mgrid[0:box_shape[0], 0:box_shape[1], 0:box_shape[2]]
    
    for i, (center, radius) in enumerate(zip(centres, radii)):
        # Calculate distance from center of this sphere
        dist = numpy.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Normalize by radius: center=0, edge=radius
        # Then invert: center=1, edge=0
        normalized_dist = numpy.maximum(0, 1.0 - (dist / radius))
        
        # Take maximum value at each voxel (in case spheres overlap)
        Box_GT = numpy.maximum(Box_GT, normalized_dist)
    
    return Box_GT

def build_ground_truth(filename):
    import time
    
    num_lines = sum(1 for line in open(folder+"positions/"+filename+".txt"))
    pixel_size = 60.e-6  # m/pixel

    # Loading the DEM file
    box_size_dem = numpy.genfromtxt(folder+"positions/"+filename+".txt",
                                     skip_footer=(num_lines-1),
                                     comments='%',
                                     usecols=(1,2,3))
    centres = numpy.loadtxt(folder+"positions/"+filename+".txt",
                            skiprows=0,
                            usecols=(4,3,2))  # x,y,z
    radii = numpy.loadtxt(folder+"positions/"+filename+".txt",
                          skiprows=0,
                          usecols=(5))

    # Get maximum radius to pad our image (periodic boundaries...)
    r_max = numpy.amax(radii)
    box_size = box_size_dem + 3*r_max
    centres[:,:] = centres[:,:] + 1.5*r_max  # move the positions to the new center

    print(f'Processing {filename}: box size = {box_size}')

    # Turn the mm measures into pixels
    box_size = numpy.array([math.ceil(box_size[0]/pixel_size),
                            math.ceil(box_size[1]/pixel_size),
                            math.ceil(box_size[2]/pixel_size)])
    box_size = int(numpy.amax(box_size))
    print(f'Box size in pixels: {box_size}')

    centres = centres/pixel_size
    radii = radii/pixel_size

    # Generate ground truth distance map
    start_time = time.time()
    print(f"{bcolors.OKCYAN}Generating ground truth distance map...{bcolors.ENDC}")
    
    Box_GT = generate_distance_map((box_size, box_size, box_size), centres, radii)
    
    print(f"{bcolors.OKGREEN}Finished ground truth generation after {time.time() - start_time:.2f} s {bcolors.ENDC}")
    
    # Convert to 16bit image (0-65535 range)
    Box_GT = numpy.rint(Box_GT * 65535)
    
    output_file = folderGT + filename + "_ground_truth.tif"
    tifffile.imwrite(output_file, Box_GT.astype('uint16'))
    print(f"{bcolors.OKGREEN}Ground truth saved to {output_file}{bcolors.ENDC}\n")

print(f"{bcolors.OKCYAN}\t ------------------------------------------")
print("\t Ground Truth Distance Map Generator")
print(f"\t ------------------------------------------\n{bcolors.ENDC}")

# Execute the function with all files
with Pool(5) as p:
    p.map(build_ground_truth, filenames)

print(f"{bcolors.OKGREEN}DONE!{bcolors.ENDC}")