#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Generate all synthetic images to match ground truth files.
Modified version of dem2image_spam.py to generate ALL available volumes.
"""

import numpy
import os
import math
import tifffile
import scipy.ndimage
from multiprocessing import Pool
import spam.kalisphera

# Parameters
gaussian = 0.8
std_dev = 0.3
pixel_size = 60.e-6  # m/pixel

# Folders
folder = "/home/gma/epita/PFEE/data_yukiko/generate_dataset/dem_data/monotonic/"
folderOutput = "/home/gma/epita/PFEE/data_yukiko/dataset/default/"

if not os.path.exists(folderOutput):
    os.makedirs(folderOutput)

# ALL files that have ground truth
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
    output_file = folderOutput + filename + f"_image_gaussian={int(gaussian*10):02d}_noise={int(std_dev*100):02d}.tif"
    
    # Skip if already exists
    if os.path.exists(output_file):
        print(f"Skipping {filename} (already exists)")
        return
    
    num_lines = sum(1 for line in open(folder + "positions/" + filename + ".txt"))

    # Loading the DEM file
    box_size_dem = numpy.genfromtxt(
        folder + "positions/" + filename + ".txt",
        skip_footer=(num_lines - 1),
        comments='%',
        usecols=(1, 2, 3)
    )
    centres = numpy.loadtxt(
        folder + "positions/" + filename + ".txt",
        skiprows=0,
        usecols=(4, 3, 2)
    )  # x,y,z
    radii = numpy.loadtxt(
        folder + "positions/" + filename + ".txt",
        skiprows=0,
        usecols=(5)
    )

    # Get maximum radius to pad our image
    r_max = numpy.amax(radii)
    box_size = box_size_dem + 3 * r_max
    centres[:, :] = centres[:, :] + 1.5 * r_max

    print(f'Processing {filename}: box size = {box_size}')

    # Turn the mm measures into pixels
    box_size = numpy.array([
        math.ceil(box_size[0] / pixel_size),
        math.ceil(box_size[1] / pixel_size),
        math.ceil(box_size[2] / pixel_size)
    ])
    box_size = int(numpy.amax(box_size))
    print(f'Box size in pixels: {box_size}')

    centres = centres / pixel_size
    radii = radii / pixel_size

    # Create the big image
    Box = numpy.zeros((box_size, box_size, box_size), dtype="<f8")

    # Create the whole assembly
    spam.kalisphera.makeSphere(Box, centres, radii)
    print(f"Finished kalisphera for {filename}")

    # Normalize values
    Box[numpy.where(Box > 1.0)] = 1.0
    Box[numpy.where(Box < 0.0)] = 0.0

    # Transform to image
    Box = Box * 0.5
    Box = Box + 0.25

    # Apply blur
    if gaussian != 0:
        Box = scipy.ndimage.gaussian_filter(Box, sigma=gaussian)

    # Apply noise
    if std_dev != 0:
        Box = numpy.random.normal(Box, scale=std_dev)

    # Convert to 16bit
    Box = numpy.rint(Box * 65536)
    
    tifffile.imwrite(output_file, Box.astype('uint16'))
    print(f"Saved: {output_file}")


if __name__ == '__main__':
    print("=" * 50)
    print("Generating all synthetic images")
    print("=" * 50)
    
    # Execute with all files
    with Pool(4) as p:
        p.map(build_image, filenames)
    
    print("\nDONE!")
