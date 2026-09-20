"""
Data augmentation transforms for 3D volumetric data.
"""

import numpy as np
from scipy import ndimage
import random


def random_flip_3d(image, mask):
    """Random flip along each axis."""
    if random.random() > 0.5:
        image = np.flip(image, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()
    if random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    return image, mask


def random_rotate_3d(image, mask):
    """Random 90-degree rotation."""
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
    return image, mask


def intensity_shift(image, max_shift=0.15):
    """Random intensity shift (only on image, not mask)."""
    shift = random.uniform(-max_shift, max_shift)
    return np.clip(image + shift, 0, 1), None


def gaussian_noise(image, sigma=0.02):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1), None


def elastic_deformation_3d(image, mask, sigma=10, alpha=200):
    """
    Apply elastic deformation to 3D volume.
    
    Args:
        image: 3D input image
        mask: 3D distance map
        sigma: Gaussian filter sigma (controls smoothness)
        alpha: Deformation magnitude
    
    Returns:
        Deformed image and mask
    """
    shape = image.shape
    
    # Generate random displacement fields
    dx = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dz = ndimage.gaussian_filter(
        (np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    # Create coordinate grids
    z, y, x = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing='ij'
    )
    
    # Apply displacement
    indices = (
        np.clip(z + dz, 0, shape[0] - 1).astype(int),
        np.clip(y + dy, 0, shape[1] - 1).astype(int),
        np.clip(x + dx, 0, shape[2] - 1).astype(int)
    )
    
    # Deform both image and mask
    deformed_image = image[indices]
    deformed_mask = mask[indices]
    
    return deformed_image, deformed_mask


def gaussian_blur_3d(image, sigma=0.5):
    """Apply Gaussian blur."""
    return ndimage.gaussian_filter(image, sigma=sigma), None


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for transform in self.transforms:
            result = transform(image, mask)
            if isinstance(result, tuple):
                if result[1] is not None:
                    image, mask = result
                else:
                    image = result[0]
            else:
                image = result
        return image, mask


def get_augmentation_transforms(config='basic'):
    """
    Get augmentation transforms based on config.
    
    Args:
        config: 'none', 'basic', 'elastic', 'intensity', 'full'
    
    Returns:
        Transform function
    """
    if config == 'none':
        return lambda img, mask: (img, mask)
    
    elif config == 'basic':
        # Flips and rotations only
        def transform(img, mask):
            img, mask = random_flip_3d(img, mask)
            img, mask = random_rotate_3d(img, mask)
            return img, mask
        return transform
    
    elif config == 'elastic':
        # Basic + elastic deformation
        def transform(img, mask):
            img, mask = random_flip_3d(img, mask)
            img, mask = random_rotate_3d(img, mask)
            if random.random() > 0.5:
                img, mask = elastic_deformation_3d(img, mask)
            return img, mask
        return transform
    
    elif config == 'intensity':
        # Basic + intensity augmentation
        def transform(img, mask):
            img, mask = random_flip_3d(img, mask)
            img, mask = random_rotate_3d(img, mask)
            if random.random() > 0.5:
                img, _ = intensity_shift(img)
            if random.random() > 0.5:
                img, _ = gaussian_noise(img)
            return img, mask
        return transform
    
    elif config == 'full':
        # All augmentations
        def transform(img, mask):
            img, mask = random_flip_3d(img, mask)
            img, mask = random_rotate_3d(img, mask)
            if random.random() > 0.3:
                img, mask = elastic_deformation_3d(img, mask, sigma=10, alpha=150)
            if random.random() > 0.5:
                img, _ = intensity_shift(img, max_shift=0.1)
            if random.random() > 0.5:
                img, _ = gaussian_noise(img, sigma=0.02)
            return img, mask
        return transform
    
    else:
        raise ValueError(f"Unknown augmentation config: {config}")
