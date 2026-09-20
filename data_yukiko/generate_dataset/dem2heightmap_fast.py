#!/usr/bin/env python3
"""
GPU-accelerated heightmap builder for sphere assemblies.
Uses per-sphere bounding boxes + Numba CUDA kernels.

Requirements:
  - numpy
  - tifffile
  - numba
  - (optional) CUDA-capable GPU + NVIDIA driver + numba.cuda
"""

import os
import math
import time
import numpy as np
import tifffile
from numba import cuda, njit, prange

# --- USER PARAMETERS (same defaults as your original script) ---
pixel_size = 10.e-6  # m / pixel
RADIUS = 1.0         # your code multiplied radii by this (kept for compatibility)
folder = "dem_data/monotonic/"
folderOutput = "../dataset/truth/"
filename = "triax.0.eps=-0.0025.spheres"

# GPU kernel launch configuration (tune if you want)
THREADS_PER_BLOCK = (8, 8, 4)  # x, y, z threads per block (swap as needed)


def ensure_dirs():
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Input folder not found: {folder}")
    if not os.path.exists(folderOutput):
        print("Creating output folder:", folderOutput)
        os.makedirs(folderOutput, exist_ok=True)


# ------------------------------
# CUDA kernel: compute local sphere contributions and atomic-max into global
# ------------------------------
@cuda.jit
def sphere_update_kernel(heightmap_flat, H, W, D, cx, cy, cz, r, xmin, ymin, zmin):
    """
    heightmap_flat: 1D flattened float32 device array with size H*W*D
    H,W,D: ints for full volume shape
    cx,cy,cz,r: floats (in pixels)
    xmin,ymin,zmin: integer offsets for the local bounding-box origin (in voxels)
    Each thread maps to a voxel in the bbox; compute val and atomic-max on global heightmap.
    """
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y
    bdz = cuda.blockDim.z

    # compute local voxel coordinates inside bbox
    lx = bx * bdx + tx
    ly = by * bdy + ty
    lz = bz * bdz + tz

    # global coords are bbox origin + local
    x = xmin + lx
    y = ymin + ly
    z = zmin + lz

    # check bounds (the launcher ensures grid covers bbox exactly, but double-check)
    if x >= 0 and x < W and y >= 0 and y < H and z >= 0 and z < D:
        # compute distance
        dx = (x + 0.0) - cx
        dy = (y + 0.0) - cy
        dz = (z + 0.0) - cz
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist <= r:
            val = r - dist  # float32
            idx = z * (H * W) + y * W + x  # flattened index (H=rows, W=cols, D=depth interpreted)
            # atomic max - Numba supports atomic.max for float32 on newer versions/drivers
            # if your Numba/CUDA doesn't support float atomic max, see note below for fallback.
            cuda.atomic.max(heightmap_flat, idx, val)


# ------------------------------
# CPU fallback kernel (fast bounding-box update using Numba njit parallel)
# ------------------------------
@njit(parallel=True, fastmath=True)
def update_bbox_cpu(heightmap, cx, cy, cz, r, xmin, ymin, zmin, xmax, ymax, zmax):
    """
    heightmap: 3D numpy array float32 (H, W, D)
    bbox coords inclusive: xmin..xmax-1 etc
    """
    H, W, D = heightmap.shape
    for z in prange(zmin, zmax):
        dz = (z - cz)
        dz2 = dz * dz
        for y in range(ymin, ymax):
            dy = (y - cy)
            dy2 = dy * dy
            for x in range(xmin, xmax):
                dx = (x - cx)
                dist = math.sqrt(dx*dx + dy2 + dz2)
                if dist <= r:
                    val = r - dist
                    if val > heightmap[z, y, x]:
                        heightmap[z, y, x] = val


# ------------------------------
# Helper: read positions file (matching your format)
# Expecting: index Sphere x y z radius  (or similar)
# We'll be flexible and parse last 4 numeric columns
# ------------------------------
def load_positions(path):
    """
    Returns centres (N,3) in meters, radii (N,) in meters
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # gather numeric tokens from the line (floats)
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except:
                    pass
            if len(nums) >= 4:
                # assume last 4 numbers are x,y,z,r  (if different ordering adjust here)
                x, y, z, r = nums[-4], nums[-3], nums[-2], nums[-1]
                data.append((x, y, z, r))
    arr = np.array(data, dtype=np.float64)
    centres = arr[:, 0:3]
    radii = arr[:, 3]
    return centres, radii


# ------------------------------
# Main builder function
# ------------------------------
def build_image_gpu(filename_basename):
    ensure_dirs()
    posfile = os.path.join(folder, "positions", filename_basename + ".txt")
    if not os.path.exists(posfile):
        raise FileNotFoundError(posfile)

    print("Loading positions:", posfile)
    centres_m, radii_m = load_positions(posfile)
    N = centres_m.shape[0]
    print("Found spheres:", N)

    # follow your original code: compute box size dem (first line in positions file maybe)
    # For simplicity we'll compute bounding box from centres + radii
    # convert all to pixels
    centres_px = centres_m / pixel_size
    radii_px = radii_m / pixel_size * RADIUS

    # compute box extents
    min_coords = np.min(centres_px - radii_px[:, None], axis=0)
    max_coords = np.max(centres_px + radii_px[:, None], axis=0)
    # pad a bit
    pad = int(math.ceil(np.max(radii_px) * 3))
    min_coords -= pad
    max_coords += pad

    # clamp to non-negative and compute integer domain size
    min_coords = np.floor(min_coords).astype(int)
    max_coords = np.ceil(max_coords).astype(int)
    box_size = np.maximum(max_coords - min_coords, 1)
    # We'll use a cubic box as in your original code (amax)
    box_size_val = int(np.amax(box_size))
    H = W = D = box_size_val
    print(f"Volume size (voxels): {H} x {W} x {D}")

    # shift centres so they sit in box coords (like your original + center shift)
    shift = -min_coords  # translate to start at zero
    centres_px_shifted = centres_px + shift

    # allocate device heightmap (flattened)
    total_vox = H * W * D
    heightmap = np.zeros(total_vox, dtype=np.float32)

    # check for CUDA device
    use_cuda = False
    try:
        device = cuda.get_current_device()
        use_cuda = True
        print("CUDA device detected:", device.name)
    except cuda.cudadrv.error.CudaSupportError:
        print("CUDA not available; using CPU fallback (still faster than full-volume loops).")
        use_cuda = False

    start_all = time.time()

    if use_cuda:
        # allocate on device
        d_heightmap = cuda.to_device(heightmap)

        # loop spheres and launch small kernels per bounding box
        for i in range(N):
            cx, cy, cz = centres_px_shifted[i]
            r = float(radii_px[i])

            # bounding box in integer voxels
            xmin = int(max(0, math.floor(cx - r)))
            xmax = int(min(W, math.ceil(cx + r) + 1))
            ymin = int(max(0, math.floor(cy - r)))
            ymax = int(min(H, math.ceil(cy + r) + 1))
            zmin = int(max(0, math.floor(cz - r)))
            zmax = int(min(D, math.ceil(cz + r) + 1))

            bx = xmax - xmin
            by = ymax - ymin
            bz = zmax - zmin
            if bx <= 0 or by <= 0 or bz <= 0:
                continue

            # compute grid dims for kernel launch
            tpx, tpy, tpz = THREADS_PER_BLOCK
            gdx = (bx + tpx - 1) // tpx
            gdy = (by + tpy - 1) // tpy
            gdz = (bz + tpz - 1) // tpz

            # launch kernel with args: flat array, H, W, D, cx,cy,cz, r, xmin,ymin,zmin
            sphere_update_kernel[(gdx, gdy, gdz), (tpx, tpy, tpz)](
                d_heightmap, H, W, D,
                float(cx), float(cy), float(cz), float(r),
                int(xmin), int(ymin), int(zmin)
            )

        # copy back
        d_heightmap.copy_to_host(heightmap)
    else:
        # CPU fallback: use 3D array view and numba-parallel bbox update
        height3 = np.zeros((D, H, W), dtype=np.float32)  # careful: z-major for indexing in update_bbox_cpu
        for i in range(N):
            cx, cy, cz = centres_px_shifted[i]
            r = float(radii_px[i])

            xmin = int(max(0, math.floor(cx - r)))
            xmax = int(min(W, math.ceil(cx + r) + 1))
            ymin = int(max(0, math.floor(cy - r)))
            ymax = int(min(H, math.ceil(cy + r) + 1))
            zmin = int(max(0, math.floor(cz - r)))
            zmax = int(min(D, math.ceil(cz + r) + 1))

            if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                continue

            # update in-place
            update_bbox_cpu(height3, cx, cy, cz, r, xmin, ymin, zmin, xmax, ymax, zmax)

        # flatten to same layout as device version (z major -> flattened idx calculation used before)
        # Our flattened index formula was: idx = z * (H*W) + y * W + x
        # height3 has shape (D, H, W) indexing [z,y,x]
        heightmap = height3.ravel()  # memory layout matches ravel order (C contiguous -> z fastest? keep consistent)

    elapsed = time.time() - start_all
    print(f"Finished sphere updates in {elapsed:.3f} s")

    # convert to 3D and save
    # reconstruct 3D array with shape (D, H, W) -> we want probably (H, W, D) as in your original
    height3 = heightmap.reshape((D, H, W))
    # reorder to (H, W, D)
    height_hwd = np.transpose(height3, (1, 2, 0))

    # scale/convert to uint16 as your original did
    # you might want to scale values to some unit; here we simply cast (clip to uint16 range)
    height_out = np.clip(height_hwd, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    outpath = os.path.join(folderOutput, "heightmap.tif")
    tifffile.imwrite(outpath, height_out)
    print("Wrote heightmap to", outpath)


# ------------------------------
# Main guard
# ------------------------------
if __name__ == "__main__":
    print("GPU Heightmap builder starting...")
    build_image_gpu(filename)
    print("DONE")

