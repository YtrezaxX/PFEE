"""
Evaluation metrics for grain segmentation experiments.
"""

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima


def compute_distance_metrics(pred, target):
    """
    Compute metrics for distance map prediction.
    
    Returns:
        dict with MSE, MAE, SSIM
    """
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    
    # Simple SSIM approximation
    mean_pred = np.mean(pred)
    mean_target = np.mean(target)
    var_pred = np.var(pred)
    var_target = np.var(target)
    covar = np.mean((pred - mean_pred) * (target - mean_target))
    
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mean_pred * mean_target + c1) * (2 * covar + c2)) / \
           ((mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'ssim': float(ssim)
    }


def watershed_segmentation(distance_map, min_distance=3, use_h_maxima=False, h=0.05):
    """
    Perform watershed segmentation on distance map.
    
    Args:
        distance_map: 3D distance map (center=1, edge=0)
        min_distance: Minimum distance between grain centers
        use_h_maxima: Use h-maxima transform instead of peak_local_max
        h: Height parameter for h-maxima (if used)
    
    Returns:
        labels: Segmented grain labels
        num_grains: Number of grains detected
    """
    # Extract grain mask
    grain_mask = distance_map > 0
    
    if use_h_maxima:
        # H-maxima transform for better center detection
        markers_bool = h_maxima(distance_map, h=h)
        coordinates = np.array(np.where(markers_bool)).T
    else:
        # Standard peak_local_max
        coordinates = peak_local_max(
            distance_map,
            min_distance=min_distance,
            labels=grain_mask.astype(int),
            exclude_border=False
        )
    
    # Create markers
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    # Watershed
    labels = watershed(-distance_map, markers, mask=grain_mask)
    labels = labels * grain_mask.astype(np.int32)
    
    num_grains = len(coordinates)
    
    return labels, num_grains


def grain_count_accuracy(pred_distance, gt_distance, min_distance=3):
    """
    Compute grain count accuracy.
    
    Args:
        pred_distance: Predicted distance map
        gt_distance: Ground truth distance map
        min_distance: For watershed
    
    Returns:
        dict with grain counts and accuracy
    """
    _, pred_count = watershed_segmentation(pred_distance, min_distance)
    _, gt_count = watershed_segmentation(gt_distance, min_distance)
    
    accuracy = pred_count / gt_count if gt_count > 0 else 0.0
    
    return {
        'pred_grains': pred_count,
        'gt_grains': gt_count,
        'grain_count_accuracy': float(accuracy),
        'grain_diff': pred_count - gt_count
    }


def compute_iou(pred_mask, gt_mask):
    """Compute Intersection over Union."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_dice(pred_mask, gt_mask):
    """Compute Dice coefficient."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2 * intersection / total)


def compute_metrics(pred_distance, gt_distance, threshold=0.1, min_distance=3):
    """
    Compute all metrics for evaluation.
    
    Args:
        pred_distance: Predicted distance map
        gt_distance: Ground truth distance map
        threshold: Threshold for binarization
        min_distance: For watershed
    
    Returns:
        dict with all metrics
    """
    # Distance map metrics
    dist_metrics = compute_distance_metrics(pred_distance, gt_distance)
    
    # Grain count metrics
    count_metrics = grain_count_accuracy(pred_distance, gt_distance, min_distance)
    
    # Mask-based metrics
    pred_mask = pred_distance > threshold
    gt_mask = gt_distance > threshold
    
    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)
    
    return {
        **dist_metrics,
        **count_metrics,
        'iou': iou,
        'dice': dice
    }
