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



# Parameters
gaussian    = 0.8
std_dev     = 0.3
pixel_size  = 30.e-6 #m/pixel

# FOlders
folder       = "dem_data/monotonic/"
folderOutput = "../dataset/default/"

if not os.path.exists(folder) :
    print(f"{bcolors.FAIL} No Input data detected, please download the data and put it in {folder} {bcolors.ENDC}")
    pass

elif not os.path.exists(folderOutput) :
    print(f"{bcolors.WARNING} No output folder detected. Creating {folderOutput}{bcolors.ENDC}")
    os.makedirs(folderOutput)

# input
filenames    = [ #"triax.-1.eps=0.spheres",
                "triax.0.eps=-0.0025.spheres",
                #"triax.1.eps=-0.005.spheres",
                "triax.2.eps=-0.01.spheres",
                #"triax.3.eps=-0.02.spheres",
                "triax.4.eps=-0.03.spheres",
                #"triax.5.eps=-0.04.spheres",
                "triax.6.eps=-0.05.spheres",
                #"triax.7.eps=-0.06.spheres",
                "triax.8.eps=-0.07.spheres",
                #"triax.9.eps=-0.08.spheres",
                #"triax.10.eps=-0.09.spheres",
                #"triax.11.eps=-0.1.spheres",
                #"triax.12.eps=-50124.6.spheres",
                ]




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

    # create the big image
    Box = numpy.zeros( (box_size, box_size, box_size), dtype="<f8")

    start_time = time.time()

    # create the whole assembly
    spam.kalisphera.makeSphere(Box, centres, radii)

    print(f"{bcolors.OKGREEN}finished kalisphera calculations after ... ", time.time() - start_time, f" s {bcolors.ENDC}")

    # some voxels at the contact might get a value higher than 1.0 due to the adding (and overlapping) of the spheres...
    # reset these values to 1.0
    Box[numpy.where(Box > 1.0)] = 1.0
    Box[numpy.where(Box < 0.0)] = 0.0

    # transform to an image with peaks at 0.25 (void) and 0.75 (particle)

    Box = Box * 0.5
    Box = Box + 0.25

    # apply blur
    if gaussian!=0:
        start_time = time.time()
        Box = scipy.ndimage.gaussian_filter(Box,sigma=gaussian)
        print(f"{bcolors.UNDERLINE}finished scipy gaussian calculations after ... ", time.time() - start_time, f" s {bcolors.ENDC}")
        

    # apply noise
    if std_dev!=0:
        Box = numpy.random.normal(Box,scale=std_dev)
        print("\nAdding noise with a std deviation of ", std_dev, "\n")

    # convert to 8bit image
    Box = numpy.rint(Box*65536)

    tifffile.imwrite(folderOutput + filename+"_image_gaussian=%02i"%(gaussian*10)+"_noise=%02i.tif"%(std_dev*100) , Box.astype('uint16') )



print(f"{bcolors.OKCYAN}\t ------------------------------------------")
print("\t Program to turn DEM assemblies into images")
print(f"\t ------------------------------------------\n{bcolors.ENDC}")

# Execute the function with all files
with Pool(5) as p:
    p.map(build_image, filenames)

print(f"{bcolors.OKGREEN}DONE!")
