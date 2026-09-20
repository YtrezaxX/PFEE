"""
Dataset classes for grain segmentation experiments.
Based on the original implementation in data_yukiko/unet_training_pipeline.ipynb
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import random


class GrainDistanceDataset(Dataset):
    """
    Dataset for (image, distance_map) pairs.
    Extracts random 3D patches.
    """
    def __init__(self, image_files, distance_files, patch_size=64, num_patches_per_volume=10):
        """
        Args:
            image_files: List of paths to synthetic images
            distance_files: List of paths to distance maps (same order)
            patch_size: Size of 3D patches to extract
            num_patches_per_volume: Number of patches per volume
        """
        self.image_files = image_files
        self.distance_files = distance_files
        self.patch_size = patch_size
        self.num_patches = num_patches_per_volume
        
        # Pre-load all volumes
        print("Loading volumes...")
        self.images = []
        self.distances = []
        
        for img_path, dist_path in zip(image_files, distance_files):
            # Load image
            img = tifffile.imread(img_path).astype(np.float32)
            # Normalize to 0-1
            img = img / 65535.0 if img.max() > 1 else img
            self.images.append(img)
            
            # Load distance map
            dist = tifffile.imread(dist_path).astype(np.float32)
            dist = dist / 65535.0 if dist.max() > 1 else dist
            self.distances.append(dist)
        
        print(f"Loaded {len(self.images)} volumes")
    
    def __len__(self):
        return len(self.images) * self.num_patches
    
    def __getitem__(self, idx):
        # Determine which volume
        volume_idx = idx // self.num_patches
        
        img = self.images[volume_idx]
        dist = self.distances[volume_idx]
        
        # Extract random patch
        d, h, w = img.shape
        ps = self.patch_size
        
        # Random positions
        z = random.randint(0, d - ps)
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        
        # Extract patches
        img_patch = img[z:z+ps, y:y+ps, x:x+ps]
        dist_patch = dist[z:z+ps, y:y+ps, x:x+ps]
        
        # Convert to PyTorch tensors (add channel dimension)
        img_tensor = torch.from_numpy(img_patch[np.newaxis, ...])  # (1, D, H, W)
        dist_tensor = torch.from_numpy(dist_patch[np.newaxis, ...])  # (1, D, H, W)
        
        return img_tensor, dist_tensor


class GrainDistanceDatasetAugmented(Dataset):
    """
    Dataset with data augmentation for grain segmentation.
    Supports flips, rotations, and intensity augmentations.
    """
    def __init__(self, image_files, distance_files, patch_size=64, 
                 num_patches_per_volume=50, augment=True,
                 elastic_deform=False, intensity_shift=False):
        self.image_files = image_files
        self.distance_files = distance_files
        self.patch_size = patch_size
        self.num_patches = num_patches_per_volume
        self.augment = augment
        self.elastic_deform = elastic_deform
        self.intensity_shift = intensity_shift
        
        print("Loading volumes...")
        self.images = []
        self.distances = []
        
        for img_path, dist_path in zip(image_files, distance_files):
            img = tifffile.imread(img_path).astype(np.float32)
            img = img / 65535.0 if img.max() > 1 else img
            self.images.append(img)
            
            dist = tifffile.imread(dist_path).astype(np.float32)
            dist = dist / 65535.0 if dist.max() > 1 else dist
            self.distances.append(dist)
        
        print(f"Loaded {len(self.images)} volumes")
    
    def __len__(self):
        return len(self.images) * self.num_patches
    
    def _apply_augmentations(self, img_patch, dist_patch):
        """Apply augmentations to both image and distance map."""
        # Random flips
        if random.random() > 0.5:
            img_patch = np.flip(img_patch, axis=0).copy()
            dist_patch = np.flip(dist_patch, axis=0).copy()
        if random.random() > 0.5:
            img_patch = np.flip(img_patch, axis=1).copy()
            dist_patch = np.flip(dist_patch, axis=1).copy()
        if random.random() > 0.5:
            img_patch = np.flip(img_patch, axis=2).copy()
            dist_patch = np.flip(dist_patch, axis=2).copy()
        
        # Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            img_patch = np.rot90(img_patch, k=k, axes=(1, 2)).copy()
            dist_patch = np.rot90(dist_patch, k=k, axes=(1, 2)).copy()
        
        # Intensity shift (only on image, not distance map)
        if self.intensity_shift and random.random() > 0.5:
            shift = random.uniform(-0.15, 0.15)
            img_patch = np.clip(img_patch + shift, 0, 1)
        
        return img_patch, dist_patch
    
    def __getitem__(self, idx):
        volume_idx = idx // self.num_patches
        
        img = self.images[volume_idx]
        dist = self.distances[volume_idx]
        
        d, h, w = img.shape
        ps = self.patch_size
        
        z = random.randint(0, d - ps)
        y = random.randint(0, h - ps)
        x = random.randint(0, w - ps)
        
        img_patch = img[z:z+ps, y:y+ps, x:x+ps]
        dist_patch = dist[z:z+ps, y:y+ps, x:x+ps]
        
        if self.augment:
            img_patch, dist_patch = self._apply_augmentations(img_patch, dist_patch)
        
        img_tensor = torch.from_numpy(img_patch[np.newaxis, ...].astype(np.float32))
        dist_tensor = torch.from_numpy(dist_patch[np.newaxis, ...].astype(np.float32))
        
        return img_tensor, dist_tensor
