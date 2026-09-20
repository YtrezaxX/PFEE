# PFEE - 3D Sand Grain Detection

Projet de Fin d'Etudes en Entreprise. The goal: find and separate individual sand grains
inside 3D tomography volumes, where grains touch each other and the boundaries between
them are barely visible.

## Result

| | Grain detection F1 |
|---|---|
| Original baseline | 74.1% |
| Target | > 95% |
| **This work (exp_010)** | **99.92%** |

Measured on **unseen data**: 52 test volumes from 4 DEM base files never used in training.
Out of 283,920 ground-truth grains, the model found 283,538 (52 false positives, 382 misses).

- Precision 99.98% / Recall 99.87%
- Dice 95.30% / IoU 91.03%
- Grain size ratio 1.03 (predicted grains match the real size, not shrunk)

Metric: each predicted grain is matched to the nearest ground-truth grain within 10 voxels.
Grain-level matching, not voxel accuracy, since the point is counting and separating grains.

## Approach

1. **Dataset generation** - DEM simulation outputs (sphere positions and contacts from
   triaxial compression) are converted into synthetic 3D volumes with exact ground truth.
   208 volumes from 16 DEM files x 13 variations, generated in 8.3 seconds.
2. **Segmentation** - 3D Attention U-Net (48 features) predicting a distance-transform-like
   map, trained with MSE (70%) + gradient loss (30%), mixed precision, 50 epochs.
3. **Grain separation** - watershed with h-maxima markers and a double threshold: a high
   threshold (0.2) to place the markers, a low one (0.08) for the mask. This is what fixed
   grains coming out too small.

## What moved the needle

| Change | Effect |
|---|---|
| Grain matching instead of grain count as metric | Correct measurement (97.4% real) |
| H-maxima instead of peak_local_max | +4% F1 |
| Masking threshold 0.0 to 0.2 (excludes background) | 97.4% to 98.45% |
| Double threshold watershed | size_ratio 0.52 to 1.03, F1 98.52% |
| Large clean dataset (208 vs 14 volumes) | 90.74% to 99.92% |

Dead ends, kept in the log for the record: Dice loss (85%), intensity-shift augmentation
(50%), and training on a small 14-image set, which generalised badly. Training on noisy
images and testing on clean ones (or the reverse) costs ~10 points of F1 - the domain shift
is real, so train on the distribution you will run on.

## Layout

```
experiments/
  EXPERIMENT_LOG.md     full journal of the 10 experiments, with parameters and results
  common/               datasets, losses, metrics, evaluation, visualisation
  models/               unet_baseline, unet_residual, unet_attention
  runs/exp_XXX/         one folder per experiment: train.py, results, evaluation scripts
data_yukiko/
  generate_dataset/     DEM output -> heightmap / synthetic volume converters
  GrainID_training_piepline.ipynb
OLD/                    first classical-CV attempts (Otsu, k-means, DBSCAN, adaptive threshold)
```

## Running it

```bash
uv sync                 # or: pip install -r requirements.txt
python experiments/runs/exp_010_large_clean/generate_large_dataset.py
python experiments/runs/exp_010_large_clean/train.py
python experiments/runs/exp_010_large_clean/compute_dice.py
```

Needs a CUDA GPU (developed on an RTX 5070 Ti, ~12 GB VRAM).
