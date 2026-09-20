"""
Normalization functions for input images.
"""

import numpy as np


def normalize_minmax(volume, target_min=0, target_max=1):
    """
    Min-max normalization to [target_min, target_max].
    """
    vol_min = volume.min()
    vol_max = volume.max()
    
    if vol_max - vol_min == 0:
        return np.zeros_like(volume) + target_min
    
    normalized = (volume - vol_min) / (vol_max - vol_min)
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized.astype(np.float32)


def normalize_percentile(volume, lower=1, upper=99, target_min=0, target_max=1):
    """
    Percentile-based normalization (robust to outliers).
    """
    p_lower = np.percentile(volume, lower)
    p_upper = np.percentile(volume, upper)
    
    volume = np.clip(volume, p_lower, p_upper)
    
    return normalize_minmax(volume, target_min, target_max)


def normalize_zscore(volume):
    """
    Z-score normalization (mean=0, std=1).
    """
    mean = volume.mean()
    std = volume.std()
    
    if std == 0:
        return np.zeros_like(volume)
    
    return ((volume - mean) / std).astype(np.float32)


def normalize_uint16(volume):
    """
    Normalize uint16 volume to [0, 1].
    """
    return (volume / 65535.0).astype(np.float32)
