#!/usr/bin/python
# -*- coding: utf-8 -*-

# 2018-05-10 MW
# The objective of this small program is to turn states of an assembly of grain coming from
#   WOO (DEM) into images using the Kalisphera implementation in spam

# 2025-09-15 MW
# Update to the current version of spam

import numpy
import os, sys, time
import math
import tifffile
import scipy.ndimage
from multiprocessing import Pool
from multiprocessing import Pool
import numpy as np

import spam.kalisphera

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



filename = "triax.0.eps=-0.0025.spheres"
folder       = "dem_data/monotonic/"
folderOutput = "../dataset/truth/"
pixel_size  = 30.e-6 #m/pixel
RADIUS = 1.0


if not os.path.exists(folder) :
    print(f"{bcolors.FAIL} No Input data detected, please download the data and put it in {folder} {bcolors.ENDC}")
    pass

elif not os.path.exists(folderOutput) :
    print(f"{bcolors.WARNING} No output folder detected. Creating {folderOutput}{bcolors.ENDC}")
    os.makedirs(folderOutput)



def writeHeightMap(volume_shape, centres, radii):
    """
    Create a 3D numpy array where each voxel inside a sphere
    has value = (sphere radius - distance to its center),
    and 0 outside.
    """
    start = time.time()
    H, W, D = volume_shape, volume_shape, volume_shape
    heightmap = np.zeros(volume_shape, dtype=np.float32)


    # Precompute voxel coordinate grid
    z, y, x = np.mgrid[0:H, 0:W, 0:D]

    for c, r in zip(centres, radii):
        # Convert to individual coordinates
        cx, cy, cz = c
        r *= RADIUS

        # Compute distance heightmap for this sphere
        dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
        inside = dist <= r

        # Compute r - d for inside voxels
        val = np.where(inside, r - dist, 0)

        # Keep maximum value if overlapping
        heightmap = np.maximum(heightmap, val)

    # Write the heightmap
    tifffile.imwrite(folderOutput + "heightmap.tif", heightmap.astype('uint16') )
    print(f"{bcolors.OKGREEN}Finished writing the heightmap after ... ", time.time() - start, f" s {bcolors.ENDC}")




def build_image(filename) :
    num_lines = sum(1 for line in open(folder+"positions/"+filename+".txt"))

    # start of the program


    # loading the dem file
    box_size_dem = numpy.genfromtxt(folder+"positions/"+filename+".txt",skip_footer=(num_lines-1),comments='%',usecols=(1,2,3))
    centres   = numpy.loadtxt(folder+"positions/"+filename+".txt",skiprows=0,usecols=(4,3,2)) #! x,y,z
    radii     = numpy.loadtxt(folder+"positions/"+filename+".txt",skiprows=0,usecols=(5))

    # get maximum radius to pad our image (periodic boundaries...)
    r_max         = numpy.amax(radii)
    box_size      = box_size_dem + 3*r_max
    centres[:,:]  = centres[:,:] + 1.5*r_max # move the positions to the new center of the image

    print('new box size', box_size)

    # turn the mm measures into pixels
    box_size = numpy.array([math.ceil(box_size[0]/pixel_size), math.ceil(box_size[1]/pixel_size), math.ceil(box_size[2]/pixel_size)])
    box_size = int(numpy.amax(box_size))
    print('box size in pixels', box_size)

    centres = centres/pixel_size
    radii   = radii/pixel_size

    # Build .tif files
    writeHeightMap(box_size, centres, radii)
    print(f"Finished")
    # writeContactMap(box_size, centres, radii)



print(f"{bcolors.OKCYAN}\t ------------------------------------------")
print("\t Program to turn DEM assemblies into a heightmap and contact points")
print(f"\t ------------------------------------------\n{bcolors.ENDC}")
build_image(filename)

print(f"{bcolors.OKGREEN}DONE!")
