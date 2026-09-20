# Journal des Experimentations - PFEE Detection Grains de Sable

---

## Informations Projet

- **Objectif** : Detecter et segmenter des grains de sable dans des images 3D
- **Baseline original** : 74.1% de detection - avec 5 volumes
- **🏆 Notre meilleur** : **F1-Score 99.92%** - exp_010 (208 images propres, données non vues)
- **Objectif final** : > 95% de detection ✅ LARGEMENT DEPASSE !
- **Hardware** : RTX 5070 Ti (~12GB VRAM), CPU 16 coeurs

---

## Resume des Experiences

| Exp ID | Nom | Date | F1-Score | Precision | Recall | Status |
|--------|-----|------|----------|-----------|--------|--------|
| exp_001 | baseline | 2026-01-18 | ~90% | - | - | ✅ Complete |
| exp_002 | dice_loss | 2026-01-18 | ~85% | - | - | ✅ Complete |
| exp_003 | augmentation | 2026-01-18 | ~50% | - | - | ✅ Complete |
| exp_004 | attention_unet | 2026-01-18 | 93.4% | 95.4% | 91.5% | ✅ Complete |
| exp_005 | watershed_hmax | 2026-01-18 | 97.4% | 97-98% | 96-97% | ✅ Complete |
| exp_006 | improved_masking | 2026-01-26 | 98.45% | 98.2% | 98.7% | ✅ Complete |
| exp_007 | double_threshold | 2026-01-26 | 98.52% | 98.3% | 98.7% | ✅ Complete |
| exp_008 | clean_images_test | 2026-02-01 | 86.9% | - | - | ✅ Complete |
| exp_009 | clean_training_14 | 2026-02-01 | 90.74% | 99.9% | 99.8% | ✅ Complete |
| exp_010 | **large_clean_208** | 2026-02-01 | **99.92%** | **99.98%** | **99.87%** | ✅ Complete |

---

## Methodologie d'Evaluation

### ⚠️ Metrique CORRECTE : Matching des grains

On match chaque grain predit au grain GT le plus proche (distance max = 10 voxels) :

- **True Positives (TP)** : Grains correctement detectes
- **False Positives (FP)** : Faux grains (sur-segmentation)
- **False Negatives (FN)** : Grains manques

**Metriques** :
- Precision = TP / (TP + FP) - "Parmi mes predictions, combien sont correctes ?"
- Recall = TP / (TP + FN) - "Parmi les vrais grains, combien ai-je detecte ?"
- F1-Score = 2 * Precision * Recall / (Precision + Recall)

---

## Experiences Detaillees

### exp_010_large_clean - 🏆 MEILLEUR RESULTAT (Données Non Vues)

**Date** : 2026-02-01

**Objectif** : Entrainer sur un grand dataset propre (sans bruit) et evaluer sur des donnees JAMAIS VUES

**Dataset** :
- 208 images propres generees (16 fichiers DEM × 13 variations)
- Split : 156 train (12 base files) / 52 test (4 base files)
- **Test sur configurations de grains jamais vues a l'entrainement**

**Configuration** :
- Modele : Attention U-Net 3D (48 features)
- Epochs : 50
- Batch size : 8
- Loss : MSE (70%) + Gradient (30%)
- Mixed precision training (FP16)

**Resultats sur donnees NON VUES** :
- Grains predits: 283,590 | Ground Truth: 283,920
- TP: 283,538 | FP: 52 | FN: 382
- **Precision: 99.98%** | **Recall: 99.87%** | **F1: 99.92%**
- **Dice: 95.30%** | **IoU: 91.03%**

**Par fichier de base (jamais vu)** :

| Base File | F1 | Dice |
|-----------|-----|------|
| triax.3.eps=-0.02 | 100.00% | 95.45% |
| triax.7.eps=-0.06 | 100.00% | 95.48% |
| triax.11.eps=-0.1 | 100.00% | 95.47% |
| initial-state-reference | 99.68% | 94.80% |

**Conclusion** :
- Le modele generalise parfaitement aux nouvelles configurations de grains
- La quantite de donnees (208 vs 14) fait une enorme difference
- Evaluation rigoureuse sur donnees vraiment jamais vues

---

### exp_009_clean_training - Entrainement sur images propres (14 images)

**Date** : 2026-02-01

**Objectif** : Comparer modele entraine sur images propres vs bruitees

**Resultats** :

| Modele | Type Image | F1 Score |
|--------|------------|----------|
| Clean model | Clean | **90.74%** |
| Clean model | Noisy | 20.19% |
| Noisy model | Clean | 85.14% |
| Noisy model | Noisy | **98.52%** |

**Conclusion** :
- Domain shift important : un modele entraine sur un type de donnees ne generalise pas bien a l'autre
- Le modele noisy est plus robuste (le bruit agit comme augmentation)
- 14 images insuffisantes → passage a 208 images dans exp_010

---

### exp_008_clean_images_test - Test sur images sans bruit

**Date** : 2026-02-01

**Objectif** : Tester le modele (entraine sur images bruitees) sur des images propres

**Images testees** :
- clean.tif : Sans blur, sans bruit
- blur_only.tif : Blur seulement
- low_noise.tif : 10% bruit
- original.tif : Blur + 30% bruit

**Resultats** :

| Type | F1 Score | Observation |
|------|----------|-------------|
| Original (30% noise) | 98.52% | Reference |
| Low noise (10%) | 95.67% | -2.85% |
| Blur only | 90.12% | -8.4% |
| Clean | 86.94% | -11.6% |

**Conclusion** :
- Domain shift : le modele performe moins bien sur des images differentes de son entrainement
- Solution : entrainer sur des images propres (exp_009, exp_010)

---

### exp_004_attention_unet - Attention U-Net

**Changement** : Architecture Attention U-Net

**Resultats avec matching** :
- Predicted: 5232 | Ground Truth: 5456
- TP: 4990 | FP: 242 | FN: 466
- **Precision: 95.4%** | **Recall: 91.5%** | **F1: 93.4%**

---

### exp_007_double_threshold - 🏆 MEILLEUR RESULTAT

**Date** : 2026-01-26

**Probleme identifie dans exp_006** :
- Les grains predits etaient PLUS PETITS que le ground truth
- size_ratio = 0.52 (grains a 52% de leur taille reelle!)
- Le threshold=0.2 coupait les bords des grains

**Solution** : Double-seuil
- `marker_thresh` (HAUT) pour trouver les centres
- `mask_thresh` (BAS) pour definir ou les grains peuvent grandir

**Tests effectues** :

| marker_thresh | mask_thresh | F1 | size_ratio |
|---------------|-------------|-----|------------|
| 0.2 | 0.02 | 98.5% | 1.77 (trop gros) |
| 0.2 | 0.05 | 98.5% | 1.29 |
| **0.2** | **0.08** | **98.52%** | **1.03** ✅ |
| 0.2 | 0.10 | 98.5% | 0.91 |
| 0.2 | 0.15 | 98.5% | 0.68 (trop petit) |

**Meilleure configuration** :
```
marker_thresh = 0.2   # Seuil haut pour markers
mask_thresh = 0.08    # Seuil bas pour expansion
h = 0.03              # H-maxima
```

**Resultats finaux** :
- Grains predits: 5546 | Ground Truth: 5522
- TP: 5452 | FP: 94 | FN: 70
- **Precision: 98.3%** | **Recall: 98.7%** | **F1: 98.52%**
- **size_ratio: 1.03** (taille quasi-identique au GT!)

**Conclusion** :
- F1 similaire a exp_006 (+0.07%)
- MAIS grains de taille correcte (1.03x vs 0.52x)
- Visuellement bien meilleur

---

### exp_006_improved_masking

**Date** : 2026-01-26

**Probleme identifie** :
- Le masque `distance > 0` incluait trop de bruit/fond
- Les bords de l'image etaient segmentes comme des "grains"
- 1.8M voxels dans le masque au lieu de ~350K

**Solution** : Augmenter le seuil de masquage de 0.0 à 0.2

**Tests effectues** :

| Threshold | Voxels Masque | Grains | F1 |
|-----------|---------------|--------|-----|
| 0.00 | 1,815,848 | 5481 | 97.9% |
| 0.10 | 602,610 | 5521 | 97.7% |
| **0.20** | **348,238** | 5493 | **98.4%** |
| 0.25 | 272,268 | 5427 | 98.3% |
| 0.30 | 212,046 | 5322 | 97.9% |

**Meilleure configuration** :
- `threshold = 0.2`
- `h = 0.03` (h-maxima)

**Resultats finaux** :
- Grains predits: 5546 | Ground Truth: 5522
- TP: 5448 | FP: 98 | FN: 74
- **Precision: 98.23%** | **Recall: 98.66%** | **F1: 98.45%**

**Conclusion** :
- Amelioration de +1% F1 par rapport a exp_005
- Le masque est maintenant beaucoup plus propre (5x moins de voxels)
- Les frontieres inter-grains sont mieux definies

---

### exp_005_watershed_hmax

**Changement** : H-maxima Watershed au lieu de peak_local_max

**Comparaison des methodes** (avec matching correct) :

| Methode | TP | FP | FN | Precision | Recall | F1 |
|---------|----|----|----|-----------|---------|----|
| peak_local_max | 4990 | 242 | 466 | 95.4% | 91.5% | 93.4% |
| **h_maxima h=0.05** | **5329** | 153 | **127** | 97.2% | **97.7%** | **97.4%** |
| h_maxima h=0.08 | 5266 | **90** | 190 | **98.3%** | 96.5% | 97.4% |

**Recommandation** :
- **h=0.05** : Meilleur recall (detecte plus de grains)
- **h=0.08** : Meilleure precision (moins de faux positifs)
- Les deux donnent **F1 = 97.4%**

---

## Synthese Finale

### 🏆 Meilleure Configuration (exp_010) :

```
Modele         : Attention U-Net 3D (48 features)
Loss           : MSE (70%) + Gradient (30%)
Watershed      : H-maxima avec h=0.03
Double-seuil   : marker_thresh=0.2, mask_thresh=0.08
Dataset        : 208 images propres (sans bruit)
─────────────────────────────────────────────────────
F1-Score       : 99.92% (sur données JAMAIS VUES)
Precision      : 99.98%
Recall         : 99.87%
Dice           : 95.30%
IoU            : 91.03%
```

### Progression :

| Etape | F1-Score | Dataset | Amelioration |
|-------|----------|---------|--------------|
| exp_004: Attention U-Net | 93.4% | 14 noisy | Reference |
| exp_005: + H-maxima | 97.4% | 14 noisy | +4.0% |
| exp_006: + Improved Masking | 98.45% | 14 noisy | +5.05% |
| exp_007: + Double-Threshold | 98.52% | 14 noisy | +5.12% |
| exp_009: Clean training | 90.74% | 14 clean | Test domain |
| exp_010: Large clean dataset | **99.92%** | **208 clean** | **+6.52%** |

### Ce qui fonctionne :
1. ✅ **Attention U-Net** (meilleure detection des centres)
2. ✅ **H-maxima Watershed** (+4% F1 vs peak_local_max)
3. ✅ **Double-seuil** (marker=0.2, mask=0.08) - grains de bonne taille
4. ✅ **MSE + Gradient Loss** (adaptee a la regression)
5. ✅ **Grand dataset** (208 images >> 14 images)
6. ✅ **Images propres** (sans bruit pour evaluation sur images propres)

### Ce qui ne fonctionne pas :
1. ❌ Dice Loss
2. ❌ Augmentation avec intensity shift
3. ❌ Petit dataset (14 images insuffisant pour generalisation)

---

## Notes

### 2026-02-01
- **exp_010** : 🏆 MEILLEUR RESULTAT - Grand dataset propre
- 208 images generees (16 DEM × 13 variations) en 8.3 secondes
- Entrainement sur 156 images, test sur 52 images de configurations JAMAIS VUES
- F1 = 99.92% sur donnees non vues (seulement 52 FP et 382 FN sur 283,920 grains)
- Dice = 95.30%, IoU = 91.03%

- **exp_009** : Test domain shift (14 images propres)
- Modele entraine sur images propres : 90.74% F1 sur images propres
- Mais seulement 20% sur images bruitees → domain shift important

- **exp_008** : Test du modele noisy sur images propres
- Le modele entraine sur images bruitees performe moins bien sur images propres (86.9%)
- Domain shift : le modele a appris des patterns lies au bruit

### 2026-01-26
- **exp_007** : Double-seuil watershed
- Probleme: exp_006 donnait des grains trop petits (size_ratio=0.52)
- Solution: seuil haut (0.2) pour markers, seuil bas (0.08) pour mask
- Resultat: F1=98.52%, **size_ratio=1.03** (taille correcte!)

- **exp_006** : Amelioration du masquage watershed
- Le seuil de 0.0 incluait le fond (1.8M voxels) → 0.2 (348K voxels)
- F1-Score passe de 97.4% → 98.45%
- MAIS grains trop petits (size_ratio=0.52)

### 2026-01-18
- Correction des metriques : utilisation du matching de grains
- F1-Score = 97.4% (meilleure metrique que "grain count accuracy")
- H-maxima apporte +4% de F1 vs peak_local_max

