import numpy as np
import tifffile
import os
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import time

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


def watershed_segmentation(distance_map, min_distance=3, verbose=True):
    start_time = time.time()
    
    # Step 1: Extract grain mask (thresholding > 0)
    if verbose:
        print(f"{bcolors.OKCYAN}Step 1: Extracting grain mask...{bcolors.ENDC}")
    grain_mask = distance_map > 0
    num_voxels = np.sum(grain_mask)
    if verbose:
        print(f"  Grain mask contains {num_voxels:,} voxels ({num_voxels / distance_map.size * 100:.2f}%)")
    
    # Step 2: Extract grain centers (local maxima)
    if verbose:
        print(f"{bcolors.OKCYAN}Step 2: Detecting grain centers (local maxima)...{bcolors.ENDC}")
    
    # Find local maxima with minimum distance constraint
    coordinates = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=grain_mask.astype(int),
        exclude_border=False
    )
    
    if verbose:
        print(f"  Found {len(coordinates)} local maxima (grain centers)")
    
    # Step 3: Label centers with arbitrary numbering
    if verbose:
        print(f"{bcolors.OKCYAN}Step 3: Labeling grain centers...{bcolors.ENDC}")
    
    # Create markers array (0 everywhere, numbered at centers)
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1  # Labels start at 1
    
    if verbose:
        print(f"  Created {len(coordinates)} labeled markers")
    
    # Step 4: Propagate labels by watershed using distance as priority
    if verbose:
        print(f"{bcolors.OKCYAN}Step 4: Propagating labels with watershed...{bcolors.ENDC}")
    
    # Watershed uses distance map inverted (lower values = higher priority)
    # Since our distance map has high values at centers, we negate it
    labels = watershed(-distance_map, markers, mask=grain_mask)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"  Watershed completed in {elapsed:.2f}s")
    
    # Step 5: Final masking (already done by watershed mask parameter)
    if verbose:
        print(f"{bcolors.OKCYAN}Step 5: Applying final mask...{bcolors.ENDC}")
    
    # Double-check masking
    labels = labels * grain_mask.astype(np.int32)
    
    num_grains = len(coordinates)
    
    if verbose:
        print(f"{bcolors.OKGREEN}Segmentation complete!{bcolors.ENDC}")
        print(f"  Total grains detected: {num_grains}")
        print(f"  Total time: {time.time() - start_time:.2f}s")
    
    return labels, num_grains, grain_mask


def process_distance_map_file(input_file, output_file=None, min_distance=3, verbose=True):
    if verbose:
        print(f"{bcolors.HEADER}{'='*60}")
        print(f"Watershed Segmentation Pipeline")
        print(f"{'='*60}{bcolors.ENDC}")
        print(f"Input file: {input_file}")
    
    # Load distance map
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    if verbose:
        print(f"{bcolors.OKCYAN}Loading distance map...{bcolors.ENDC}")
    
    distance_map = tifffile.imread(input_file)
    
    # Convert from uint16 to float (normalize back to 0-1 range)
    if distance_map.dtype == np.uint16:
        distance_map = distance_map.astype(np.float64) / 65535.0
    
    if verbose:
        print(f"  Shape: {distance_map.shape}")
        print(f"  Data type: {distance_map.dtype}")
        print(f"  Value range: [{distance_map.min():.4f}, {distance_map.max():.4f}]")
    
    # Perform segmentation
    labels, num_grains, grain_mask = watershed_segmentation(
        distance_map, 
        min_distance=min_distance,
        verbose=verbose
    )
    
    # Save results if output path provided
    if output_file:
        if verbose:
            print(f"{bcolors.OKCYAN}Saving segmentation results...{bcolors.ENDC}")
        
        # Save as 32-bit integer (to handle many labels)
        tifffile.imwrite(output_file, labels.astype(np.int32))
        
        if verbose:
            print(f"{bcolors.OKGREEN}Saved to: {output_file}{bcolors.ENDC}")
    
    return labels, num_grains, grain_mask


def batch_process(folder_gt="dataset/ground_truth/", folder_output="dataset/segmented/", 
                  min_distance=3, file_pattern="_ground_truth.tif"):
    # Create output folder if needed
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
        print(f"{bcolors.WARNING}Created output folder: {folder_output}{bcolors.ENDC}")
    
    # Find all ground truth files
    gt_files = [f for f in os.listdir(folder_gt) if f.endswith(file_pattern)]
    
    if not gt_files:
        print(f"{bcolors.FAIL}No ground truth files found in {folder_gt}{bcolors.ENDC}")
        return
    
    print(f"{bcolors.HEADER}Found {len(gt_files)} files to process{bcolors.ENDC}\n")
    
    # Process each file
    results = []
    for filename in gt_files:
        input_path = os.path.join(folder_gt, filename)
        output_filename = filename.replace(file_pattern, "_segmented.tif")
        output_path = os.path.join(folder_output, output_filename)
        
        try:
            labels, num_grains, _ = process_distance_map_file(
                input_path, 
                output_path,
                min_distance=min_distance,
                verbose=True
            )
            results.append((filename, num_grains, True))
        except Exception as e:
            print(f"{bcolors.FAIL}Error processing {filename}: {e}{bcolors.ENDC}")
            results.append((filename, 0, False))
        
        print()  # Blank line between files
    
    # Summary
    print(f"{bcolors.HEADER}{'='*60}")
    print("Batch Processing Summary")
    print(f"{'='*60}{bcolors.ENDC}")
    successful = sum(1 for _, _, success in results if success)
    print(f"Processed: {successful}/{len(results)} files successfully")
    print(f"\n{bcolors.OKGREEN}Batch processing complete!{bcolors.ENDC}")


if __name__ == "__main__":
    # Example usage
    print(f"{bcolors.HEADER}")
    print("="*60)
    print("Watershed Segmentation Pipeline for DEM Grain Analysis")
    print("="*60)
    print(f"{bcolors.ENDC}\n")
    
    # Process all ground truth files
    batch_process(
        folder_gt="dataset/ground_truth/",
        folder_output="dataset/segmented/",
        min_distance=3  # Adjust based on typical grain size
    )
