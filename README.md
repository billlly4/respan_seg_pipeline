# ResPan — RESPAN Training & Analysis Pipeline

Personal scripts for training and running the RESPAN pipeline (CARE, SelfNet, nnU-Net)
on fluorescence microscopy data for dendritic spine analysis.

---

## Environments

| Environment | Python | Framework | Used for |
|---|---|---|---|
| `respandev` | 3.10 | TensorFlow 2.10 | CARE training & inference |
| `respaninternal` | 3.9 | PyTorch 2.5.1 | SelfNet + nnU-Net training & inference |

---

## Pipeline Overview

```
Raw images
    │
    ├── [optional] CARE        → denoise (requires paired low/high SNR training data)
    │
    ├── [optional] SelfNet     → restore axial resolution (self-supervised, no pairs needed)
    │
    └── nnU-Net                → segment spines / dendrites / soma
```

Training each model is independent. During analysis, RESPAN chains them automatically
based on which models you have trained.

**Important:** train nnU-Net on images that look the same as what you will feed it
during analysis. If you run CARE + SelfNet during analysis, train nnU-Net on
CARE + SelfNet restored images.

---

## 1. Train CARE (denoising)

Requires paired images: same field of view imaged fast/noisy and slow/clean.
Place them in `CARE_training/data/lowSNR/` and `CARE_training/data/highSNR/`.

```bash
# Test run (5 epochs, fast)
C:/Users/ARASH/Anaconda3/envs/respandev/python.exe  f:/ResPan/train_care.py

# Real training
C:/Users/ARASH/Anaconda3/envs/respandev/python.exe  f:/ResPan/train_care.py \
  --epochs 150 --steps_per_epoch 400 --batch_size 16 --num_patches 100
```

Key parameters: `--data`, `--output`, `--name`, `--patch_size`, `--epochs`, `--batch_size`

---

## 2. Train SelfNet (axial restoration)

Self-supervised — no paired data needed. Uses the internal difference between the
sharp XY view and the blurry XZ/YZ views of a 3D stack.

**Z-plane requirement:** `original_Z >= ceil(64 * scale)`
- scale = XY pixel size / Z step size (e.g. 0.1326 / 0.41 = 0.32)
- For scale=0.32: need at least 21 Z planes
- For scale=0.41: need at least 27 Z planes

Use multiple images to cover content diversity (cell bodies, dendrites, etc.).
All images must share the same voxel size.

```bash
# Test run (3 epochs)
C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_selfnet.py \
  --images \
    "f:/ResPan/selfnet_training/data/lifeact/220325_DIV16cx_EGFP_BFP2_Lifeact_LTP30min_0001_ch2.tif" \
    "f:/ResPan/selfnet_training/data/lifeact/140325_DIV17cx_BFP2_Lifeact_EGFP_LTPctrl_0003_ch2.tif" \
    "f:/ResPan/selfnet_training/data/lifeact/220325_DIV16cx_EGFP_BFP2_Lifeact_LTP30min_0005_ch2.tif" \
  --scale 0.32 --max_v 65535 --bg_threshold 1000 \
  --name selfnet_lifeact_test --epochs 3

# Real training
C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_selfnet.py \
  --images <image1.tif> <image2.tif> ... \
  --scale 0.32 --max_v 65535 --bg_threshold 1000 \
  --name selfnet_lifeact --epochs 40
```

Key parameters: `--images`, `--scale`, `--max_v`, `--bg_threshold`, `--epochs`

---

## 3. Train nnU-Net (segmentation)

Requires annotated images: place raw stacks in `training_data/imagesTr/` and
matching label masks in `training_data/labelsTr/`. Labels use:
`0=background, 1=spine, 2=dendrite, 3=soma`

Run the three steps in sequence:

```bash
PY=C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe

# Step 1 — set up dataset structure and dataset.json
$PY  f:/ResPan/train_nnunet.py  setup

# Step 2 — fingerprint + preprocess (nnU-Net auto-configures architecture)
$PY  f:/ResPan/train_nnunet.py  plan

# Step 3 — train 3d_fullres fold=all (~12-24h on GPU)
$PY  f:/ResPan/train_nnunet.py  train

# Or run all three steps in one go
$PY  f:/ResPan/train_nnunet.py  all
```

---

## 4. Run analysis

```bash
C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/run_respan.py
```

CARE, SelfNet, and nnU-Net are chained automatically. To enable restoration,
set these flags directly in `run_respan.py` inside `_make_settings()`:

```python
# Enable CARE denoising (per channel)
s.image_restore       = True
s.c2_restore          = True
s.c2_rest_model_path  = r'f:\ResPan\CARE_training\output\care_model'
s.c2_rest_type        = ['care']

# Enable SelfNet axial restoration
s.axial_restore       = True
s.selfnet_path        = r'f:\ResPan\selfnet_training\output\selfnet_lifeact'
```

Leave both as `False` to skip restoration and run nnU-Net directly on raw images.

---

## Data configuration

See `data_conf_requirements.md` for full rules on which images can be grouped
together for training each model, the Z-plane constraint for SelfNet, and
the chicken-and-egg problem if no paired data is available for CARE.
