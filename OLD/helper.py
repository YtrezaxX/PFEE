import tifffile
import random


def read_voxel_region(tiff_path, region_size=100, random_offset_range=50):
    """
    Read a random voxel region from a TIFF file.
    
    Args:
        tiff_path (str): Path to the TIFF file
        region_size (int): Size of the cubic region to extract (default: 100)
        random_offset_range (int): Maximum random offset from center (default: 50)
    
    Returns:
        tuple: (region_array, start_coordinates)
            - region_array: numpy array of the extracted region
            - start_coordinates: tuple of (start_z, start_y, start_x)
    """
    with tifffile.TiffFile(tiff_path) as tif:
        shape = tif.series[0].shape
        
        if len(shape) == 3:
            depth, height, width = shape
        elif len(shape) == 2:
            height, width = shape
            depth = 1
        else:
            raise ValueError("Unexpected image dimensions!")
        
        center_z = depth // 2
        center_y = height // 2
        center_x = width // 2
        
        offset_z = random.randint(-random_offset_range, random_offset_range)
        offset_y = random.randint(-random_offset_range, random_offset_range)
        offset_x = random.randint(-random_offset_range, random_offset_range)
        
        start_z = max(0, min(center_z + offset_z - region_size//2, depth - region_size))
        start_y = max(0, min(center_y + offset_y - region_size//2, height - region_size))
        start_x = max(0, min(center_x + offset_x - region_size//2, width - region_size))
        
        end_z = start_z + region_size
        end_y = start_y + region_size
        end_x = start_x + region_size
        
        print(f"Reading region: Z[{start_z}:{end_z}], Y[{start_y}:{end_y}], X[{start_x}:{end_x}]")
        
        region = tif.asarray()[start_z:end_z, start_y:end_y, start_x:end_x]
        
        return region, (start_z, start_y, start_x)
