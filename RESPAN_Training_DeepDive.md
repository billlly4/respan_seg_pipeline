# RESPAN Training: Complete Technical Deep-Dive

## Pipeline Overview and Order

The three models sit in sequence. CARE runs first (denoises the structural channel), SelfNet runs second (restores axial resolution, making the data nearly isotropic), and nnU-Net runs last (segments the isotropic/cleaned-up volume into spines, dendrites, soma). Each is independently optional — RESPAN's YAML controls which are active. But when all three are on: CARE → SelfNet → nnU-Net.

---

## 1. CARE — Supervised Fluorescence Restoration

**Framework:** CSBDeep/TensorFlow. Not PyTorch.

**Purpose:** Removes noise and improves SNR in your structural channel. Requires paired training data — a low-SNR image and its corresponding high-SNR ground truth, same field of view.

### Required Data Structure

```
your_training_data/
├── lowSNR/
│   ├── image001.tif
│   ├── image002.tif
│   └── ...
└── highSNR/
    ├── image001.tif
    ├── image002.tif
    └── ...
```

Filenames must match between folders. Each TIF is a 3D Z-stack (ZYX). The high-SNR images are typically acquired with longer exposure, laser averaging, or by averaging many fast acquisitions.

### How RESPAN Calls the Training

From `ImageAnalysis/ModelTraining.py`:

```python
train_care(
    inputdir,            # folder with lowSNR/ and highSNR/
    model_output,        # where to save
    model_name,          # subfolder name
    augmentation,        # bool
    image_type,          # "confocal" or "widefield"
    patch_size,          # (pz, py, px), e.g. (16, 64, 64)
    num_patches_per_img,
    epochs,
    unet_kern_size,      # 3 or 5
    unet_n_depth,        # U-Net depth, default 2
    batch_size,
    steps_per_epoch,
    pct_validation       # fraction held out
)
```

**Hardcoded values:**
- `initial_learning_rate = 0.0004`
- `probabilistic = True` — outputs a mean + variance prediction (Probabilistic CARE)
- **Z must be a multiple of 8** — the code auto-pads if it isn't, then crops after

### Architecture

Probabilistic U-Net (CSBDeep's implementation). With `unet_n_depth=2` you get an encoder with 2 downsampling stages. The probabilistic head outputs both a restored image mean and a per-pixel uncertainty map. During inference, only the mean is used. The `unet_kern_size` controls the 3D convolution kernel — 3 is faster, 5 captures more context but needs more VRAM.

### Augmentation (`augment_images_v1`)

Applied at data-prep time, before training starts. Produces rotated copies of each training pair:
- Always: 90°, 180°, 270° rotations → 4× your data
- If `image_type == "confocal"`: adds Poisson noise augmentation (mimics photon shot noise)
- Optional: Gaussian blur, intensity scaling

### Inference (what happens at analysis time)

From `Segmentation_and_Restoration.py:restore_image()`:

```python
restored = rest_model.predict(channel_image, axes='ZYX', n_tiles=settings.tiles_for_prediction)
```

Then clips output at `1.2 × original_max` to prevent restoration from hallucinating extremely bright voxels. The CSBDeep `n_tiles` parameter controls how the volume is split if it doesn't fit in VRAM — set in the YAML as `tiles_for_prediction`.

### Outputs During Training

- `training_evaluation.csv` — per-epoch loss history
- `training_history.png` — loss curves
- Validation PNGs — restored vs. ground-truth comparison images at each validation

---

## 2. SelfNet — Self-Supervised Axial Resolution Restoration

**Framework:** PyTorch (not TensorFlow).

**Purpose:** Restores the blurred Z-axis information in anisotropic data. After SelfNet runs, `input_resZ` is set equal to `input_resXY` — the volume is treated as isotropic from that point forward. This is what makes it self-supervised: you never need a ground-truth isotropic image. The network learns from the assumption that high-resolution XY planes describe what an axial view *should* look like if Z resolution matched XY resolution.

**Hardware requirement:** SelfNet inference requires **two GPUs** (cuda:0 and cuda:1). Training only uses cuda:0.

### The Scale Parameter

The single most important number. `scale = XY_pixel_size / Z_step_size`. For this project's data:

| Dataset | XY | Z | scale |
|---|---|---|---|
| Cortical overview | 82.5 nm | 200 nm | 0.4125 |
| Hippo | 82.5 nm | 200 nm | 0.4125 |
| LTP cortical | 102 nm | 200 nm | 0.51 |

This controls how much to compress XY when simulating what an axial slice looks like, and how much to stretch XZ/YZ when normalizing them to XY pixel dimensions.

### Training: 3 Phases in One Script

Training is called as a single script with CLI args. RESPAN calls it internally from `ModelTraining.py` as a subprocess.

**CLI:**
```bash
python SelfNet_Model_Training.py \
  --input_dir    /path/to/folder/with/tif/  \
  --model_path   /path/to/save/             \
  --model_name   my_model_name              \
  --min_v        0                          \
  --max_v        65535                      \
  --bg_threshold 1000                       \
  --scale        0.41                       \
  --xy_int       4                          \
  --xz_int       8                          \
  --batch_size   8                          \
  --epochs       60                         \
  --log_interval    1                       \
  --imshow_interval 100
```

**`--scale`** = XY_res / Z_res (e.g. 82.5 nm / 200 nm = 0.41)  
**`--bg_threshold`** = minimum pixel intensity for a patch to be used in training (discards background-only patches)  
**`--xy_int`** = use every Nth XY plane (default 4 — keeps 1 in 4 Z planes)  
**`--xz_int`** = use every Nth XZ slice (default 8)  

**Input data:** A single 3D TIF in `input_dir/`. The script picks the first `.tif` it finds. This should be a representative dendritic z-stack — same type of data you'll apply the model to.

---

#### Phase 1: Create Slices (`self_net_create_slices`)

Reads the 3D TIF (ZYX) and transposes it to (YXZ). Then creates four folders inside `training_slices/`:

| Folder | Contents | What it represents |
|---|---|---|
| `xy/` | One TIF per Z plane | High-resolution XY slices (the target / "sharp" domain) |
| `xy_lr/` | Same planes, downsampled then upsampled | XY degraded to simulate Z-like blur (input to generator) |
| `xz/` | Resliced XZ planes, scaled by `1/scale` | Actual low-res axial views of the dendrite |
| `yz/` | Resliced YZ planes, scaled by `1/scale` | Same, perpendicular axis |

The degradation of `xy_lr` is: resize from `(H, W)` to `(round(H × scale), W)`, then resize back to `(H, W)`. This mathematically mimics what a Z section looks like when the Z step is coarser than XY — the same spatial blurring pattern the network will see in real XZ planes.

Resizing uses GPU-accelerated PyTorch trilinear interpolation (`reslice_and_resize_torch`) for the XZ/YZ planes, and skimage order-3 spline for the XY_LR generation.

---

#### Phase 2: Generate Training Patches (`generate_training_data`)

Cuts all three image types into **64×64 patches with stride 64** (no overlap). Each patch is only kept if `max(patch) >= bg_threshold`. This discards patches that are pure background.

Sampling interval:
- `xy_int=4` → samples every 4th XY plane (keeps 1/4 of Z planes)
- `xz_int=8` → samples every 8th XZ plane

This is designed to roughly balance the number of XY vs. XZ patches in training, since you have far more XY planes than XZ planes in a typical stack.

All patches are stacked and saved as a compressed NumPy archive:
```
input_dir/train_data/train_data.npz
  → xy:    (N, 64, 64)  float32  — HR target
  → xy_lr: (N, 64, 64)  float32  — LR degraded input
  → xz:    (M, 64, 64)  float32  — real axial slices
```

---

#### Phase 3: Train (`train_self_net`)

A **CycleGAN + supervised deblurring** hybrid. Five networks are instantiated:

| Network | Type | Role |
|---|---|---|
| `netG_A` | Deblur_Net (ResNet generator) | Translates degraded XY_LR → synthetic XZ domain |
| `netG_B` | Deblur_Net | Translates real XZ → synthetic XY_LR domain |
| `deblur_net` | Deblur_Net | **The keeper**: translates synthetic XZ → restored XY |
| `netD_A` | PatchGAN (2 layers, ndf=64) | Discriminates real vs. fake in XZ domain |
| `netD_B` | PatchGAN (2 layers, ndf=64) | Discriminates real vs. fake in XY_LR domain |

The CycleGAN pair (netG_A, netG_B, netD_A, netD_B) is scaffolding — it teaches the generators to correctly model the XY↔XZ domain gap. `deblur_net` then uses what G_A learned to actually restore real axial slices. Only `deblur_net` is saved and used at inference time; the CycleGAN components are discarded.

**Loss weights:**

```python
lambda_cycle    = 1    # CycleGAN reconstruction (L1 + SSIM)
lambda_deblur   = 1    # Final deblur supervision (L1 + SSIM vs. real XY)
lambda_feedback = 0    # becomes 0.1 after epoch 2: feedback from deblur_net to G_A
lambda_gan      = 0.1  # adversarial loss weight (intentionally weak)
lambda_ssim     = 0.1  # SSIM component of reconstruction losses
```

GAN loss type: **LSGAN** (MSE on discriminator outputs). More stable than vanilla GAN.

Reconstruction loss: `L1_Charbonnier_loss` — a smooth L1 approximation: `sqrt(diff² + 1e-3)`. Less sensitive to outliers than standard L1.

**Training loop per batch:**
1. Generator step: update G_A + G_B via cycle loss + feedback + GAN loss
2. Discriminator step (every other batch): update D_A + D_B
3. Deblur step (every batch): update deblur_net on `G_A(xy_lr) → xy` supervision

**Learning rate schedule:** halved every 15 epochs (γ=0.5). Starting LR: 1e-4 for both generators and discriminators. Optimizer: Adam (β1=0.5, β2=0.999).

**Augmentation during training:** 8 modes of flip/rotate applied randomly per sample (identity, vertical flip, 180°+flip, 180°, 90°+flip, 90°, 270°, 270°+flip).

**Checkpoints:** Only `deblur_net.state_dict()` is saved:
```
<model_path>/<model_name>/saved_models/SelfNet_Epoch_N_Batch_M.pkl
```

**Preview images** saved at `intermediate_results/SelfNet_epochNNN_itersM_1.tif` — shows what XZ slice #1 looks like after deblurring at each checkpoint. Watch these during training to verify the network is learning.

**Architecture — Deblur_Net (ResNet generator):**
```
ReflectionPad(3) → Conv7(1→64) → LeakyReLU(0.2)
→ Conv3(64→64) → LeakyReLU   [×2 layers]
→ ResBlock(64)                [×6 blocks]
→ Conv3(64→64) → LeakyReLU   [×2 layers]
→ ReflectionPad(3) → Conv7(64→1)
```

Each ResBlock: `ReflectionPad→Conv3→LeakyReLU → ReflectionPad→Conv3` + skip connection.

All operations are 2D (slice-by-slice on single-channel patches). No 3D convolutions. Normalization: InstanceNorm2d.

**After SelfNet runs during inference:** `settings.input_resZ = settings.input_resXY`. From nnU-Net's point of view, the data is now isotropic — the rescaling step treats all three axes equally.

---

## 3. nnU-Net — 3D Spine Segmentation

**Framework:** nnU-Net v2, PyTorch. Self-configuring: automatically determines network depth, patch size, normalization, augmentation, and training schedule from your dataset's fingerprint.

**Labels:** background=0, spine=1, dendrite=2, soma=3, ignore=4.

### Required Data Structure

```
nnUNet_raw/
└── Dataset{ID}_{Name}/
    ├── imagesTr/
    │   ├── case001_0000.tif   # _0000 = channel index
    │   └── ...
    ├── labelsTr/
    │   ├── case001.tif        # integer label mask, same shape as image
    │   └── ...
    └── dataset.json
```

RESPAN's `generate_nnunet_json()` generates `dataset.json` automatically by scanning the imagesTr/labelsTr folders. It reads every label TIF, finds all unique intensity values, and maps them to class names (background=0, then 1, 2, 3, 4 in the order they appear).

### Training Workflow

RESPAN's `train_nnUNet()` sets three environment variables, then fires two commands:

```python
os.environ['nnUNet_raw']          = raw_dir
os.environ['nnUNet_preprocessed'] = preprocessed_dir
os.environ['nnUNet_results']      = results_dir
```

**Step 1 — Plan and preprocess:**
```bash
nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity
```

nnU-Net reads your dataset, computes the fingerprint (image shapes, voxel spacings, intensity statistics, label frequencies), and automatically designs:
- Network topology (number of encoder/decoder stages)
- Patch size and batch size (maximizes GPU utilization)
- Resampling strategy (what spacing to normalize all images to)
- Normalization scheme (z-score or min-max, with or without clipping)

The fingerprint is saved to `nnUNet_preprocessed/Dataset{ID}/dataset_fingerprint.json`. Plans go to `plans.json`.

**Step 2 — Train:**
```bash
nnUNetv2_train {dataset_id} 3d_fullres all
```

- `3d_fullres`: the main 3D full-resolution configuration (RESPAN always uses this; `3d_lowres` and `2d` configurations exist but are unused)
- `all`: trains on 100% of your data with no cross-validation holdout (as opposed to folds 0–4 for 5-fold CV)
- Expected time: **12–24 hours** on a single GPU with a reasonable dataset

**Training log location:**
```
nnUNet_results/
└── Dataset{ID}_SpineSegmentation/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_all/
            └── training_log_{YYYY_MM_DD_HH_MM_SS}.txt
```

### Model 1A Configuration (for reference)

From `plans.json` and `dataset_fingerprint.json`:

| Property | Value |
|---|---|
| Training images | 244 |
| Patch size | 32 × 256 × 256 |
| Network | 7-stage PlainConvUNet |
| Z stride strategy | (1,2,2) for early stages — Z downsampled more slowly than XY |
| Normalization spacing | [1,1,1] — all images rescaled to model resolution before training |
| Z range in training data | 15–191 planes, median 33 |
| XY range in training data | 184–1497 px, median 409 |
| Foreground intensity | mean 3132, median 1043, 99.5th %ile 21986 |

### Inference (what RESPAN does at analysis time)

From `Segmentation_and_Restoration.py:run_nnunet_predict()`:

```bash
python clean_launcher.py nnunet_predict_bat \
  -i <input_dir>   \
  -o <output_dir>  \
  -d {dataset_id}  \
  -c 3d_fullres    \
  -f all
```

`nnUNet_results` env var must point to the **parent directory of the model folder** (the directory containing `Dataset{ID}_...`).

Before this, `nnunet_create_labels()` rescales your data to the model's native resolution:
```python
scaling_factors = (input_resZ / model_resZ, input_resXY / model_resXY, input_resXY / model_resXY)
```
Applied via `skimage.transform.resize`. If Z < 5 planes after rescaling, it pads to 5.

**Tiling mode:** For large images, `patch_images_for_nnunet()` splits the volume into overlapping blocks, runs inference per tile, then `reassemble_patch_predictions()` fuses predictions by majority vote (most common label per voxel across all overlapping tiles).

---

## What You Need To Train Your Own Models

| | CARE | SelfNet | nnU-Net |
|---|---|---|---|
| **Why train your own?** | Your SNR is very different from what the pre-trained model expects | Your XY/Z ratio differs significantly from what the pre-trained model was trained on | You want to add a new cell type, label, or imaging modality |
| **Training data needed** | Paired low/high SNR stacks of the same FOV | One representative 3D TIF of your data | 3D stacks + manual segmentation masks |
| **How many images?** | 20–50 pairs typically sufficient | 1–3 representative volumes | Model 1A used 244 — more is better; ~50 minimum |
| **Training time** | Hours | Hours (60 epochs) | 12–24 hours |
| **GPU requirement** | 1 GPU (TF) | 1 GPU for training, 2 GPUs for inference | 1 GPU |
| **Key hyperparameter** | `patch_size`, `epochs`, `unet_n_depth` | `--scale` (= XY_res / Z_res) | Automatic — nnU-Net self-configures |
| **Output file used at inference** | CSBDeep model weights (h5) | `saved_models/SelfNet_Epoch_N_Batch_M.pkl` (deblur_net only) | `fold_all/checkpoint_best.pth` + `dataset.json` + `plans.json` + `dataset_fingerprint.json` |

### Note on Model 1B Failure

The Model 1B folder downloaded from Google Drive was missing `dataset.json`, `plans.json`, and `dataset_fingerprint.json` from the `nnUNetTrainer__nnUNetPlans__3d_fullres/` directory — only `fold_all/checkpoint_best.pth` was present. All four files are required for nnU-Net inference. Re-download from the source and verify all files are present before use.

### Note on Model 2 and Anisotropic Data

Model 2 was trained on isotropic 65×65×65 nm data. When applied to data with 200 nm Z steps, RESPAN must rescale the input to match Model 2's native resolution. This stretches the Z axis dramatically, distorting spine geometry beyond what the network was trained to recognize — resulting in near-zero spine detection (1 spine per dataset). **Do not use Model 2 for anisotropic data with Z steps > ~100 nm.**

---

## Practical Training Pathway — Custom nnU-Net for Your Own Spine Data

### Decision: which model to train first

Train **nnU-Net only**. CARE and SelfNet are restoration pre-processors — they only matter if raw image quality or Z anisotropy is clearly preventing segmentation. If Model 1A detects spines on your data at all, start by training a custom nnU-Net on your own annotations. Add CARE or SelfNet later only if a well-trained nnU-Net still clearly fails on specific stacks.

---

### How many annotations you need

| Target | Stacks | Notes |
|---|---|---|
| Minimum to start | 30 | Enough to see if training is converging |
| Thesis-grade | 50–60 | Reliable morphology classification |
| Strong / publishable | 80–120 | Across multiple cells and conditions |

Quality beats quantity. 35 carefully corrected stacks from one imaging condition outperform 80 sloppy annotations mixed across conditions. Do not change label meanings (integer values per class) partway through annotation — fix the convention once at the start and keep it.

---

### Step 1 — Run Model 1A to generate starting labels

Run RESPAN with Model 1A on each stack you intend to annotate. The output `labels.tif` gives you:

```
0 = background
1 = spine
2 = dendrite
3 = soma  (present only if soma is in field of view)
```

This is your starting point — you are correcting, not annotating from scratch. Typical correction time per stack: 5–15 minutes.

---

### Step 2 — Set up the case folder

For each stack, build this structure under `f:\ResPan\manual_fixing_labels\`:

```
manual_fixing_labels\
└── caseNNN\
    ├── imagesTr\
    │   └── caseNNN.tif          ← raw Ch0 structural TIF (uint16)
    ├── labelsTr\
    │   └── caseNNN.tif          ← Model 1A labels.tif (rename to match)
    └── labels_as_channels\      ← created by Script 1 (do not create manually)
```

Filenames in `imagesTr\` and `labelsTr\` **must be identical** — the scripts match them by filename.

---

### Step 3 — Run Script 1: Labels → Channels (in Fiji)

Script: `manual_fixing_labels\imageJ_plugins\1_Labels_to_Channels_for_Annotation.ijm`

1. Open Fiji → Run the macro
2. Select the `caseNNN\` folder when prompted
3. Set **MaxLabel** = number of label classes present (not counting background 0):
   - Labels 0,1,2 only (spine + dendrite, no soma) → **MaxLabel = 2**
   - Labels 0,1,2,3 (spine + dendrite + soma) → **MaxLabel = 3**
   - Labels 0,1,2,3,4 (spine + dendrite + soma + ignore) → **MaxLabel = 4**

Output: `labels_as_channels\caseNNN.tif` — a multi-channel composite:

```
Channel 1 = raw image (8-bit, reference only)
Channel 2 = spine mask (binary, 0 or 255)
Channel 3 = dendrite mask (binary, 0 or 255)
Channel 4 = soma mask (binary, 0 or 255)   — only if MaxLabel >= 3
```

---

### Step 4 — Annotate in Fiji

Open `labels_as_channels\caseNNN.tif`. Use **Image → Color → Channels Tool** to work on each channel independently.

**Channel 2 — spine (red):**
- Spot-check against raw. Add missed spines with pencil tool. Remove false positives by erasing.
- Do not repaint entire spines from scratch unless Model 1A clearly missed the class entirely.

**Channel 3 — dendrite (green): repaint this manually.**
- Erase all existing content: Edit → Fill with black on every slice.
- Repaint the full dendrite shaft slice by slice using pencil, circle brush ≈ 3–5 px diameter.
- Paint a solid continuous tube from one end of the visible dendrite to the other, in every Z plane where it appears.
- Do not paint onto spine necks — leave those to the spine class.
- Do not skip Z planes. Gaps in the dendrite label will fragment the 3D skeleton and corrupt spine density measurements.
- Include branch points but stop at the soma boundary.

**Channel 4 — soma (blue):**
- Erase all: Edit → Select All → Edit → Fill with black on all slices.
- Leave completely empty — soma is not needed for spine density analysis.

Save the composite back to `labels_as_channels\caseNNN.tif` (overwrite).

---

### Step 5 — Run Script 2: Channels → Labels (in Fiji)

Script: `manual_fixing_labels\imageJ_plugins\2_Channels_to_Labels_for_Training.ijm`

1. Open Fiji → Run the macro
2. Select the `caseNNN\` folder
3. Set **MaxLabel = 2** (spine + dendrite only, soma has been erased)

Output: `labelsTr_YYYY_MM_DD\caseNNN.tif` — integer label TIF with:
```
0 = background
1 = spine
2 = dendrite
```

This is the file you use as the training label for nnU-Net. Label priority in overlapping voxels: spine (1) overrides dendrite (2) — this is handled automatically by the script's subtract logic.

---

### Step 6 — Build the nnU-Net dataset

Copy corrected pairs into the training dataset:

```
f:\ResPan\nnUNet_training\nnUNet_raw\Dataset101_Spines\
├── imagesTr\
│   ├── case001_0000.tif    ← raw Ch0 TIF (uint16), _0000 suffix required
│   ├── case002_0000.tif
│   └── ...
├── labelsTr\
│   ├── case001.tif         ← corrected integer label TIF, no suffix
│   ├── case002.tif
│   └── ...
└── dataset.json            ← generated in Step 7
```

Naming rules:
- `case{NNN}_0000.tif` and `case{NNN}.tif` must share the same `case{NNN}` prefix
- Use zero-padded numbers: `case001`, `case002`, not `case1`, `case2`
- Both files must have **identical ZYX shape**

---

### Step 7 — Generate dataset.json

```bash
conda activate respaninternal
python
```

```python
import sys
sys.path.insert(0, r'f:\ResPan\cloned_respan\RESPAN\ImageAnalysis')
from ModelTraining import generate_nnunet_json

generate_nnunet_json(
    imagesTr_dir = r'f:\ResPan\nnUNet_training\nnUNet_raw\Dataset101_Spines\imagesTr',
    labelsTr_dir = r'f:\ResPan\nnUNet_training\nnUNet_raw\Dataset101_Spines\labelsTr',
    output_dir   = r'f:\ResPan\nnUNet_training\nnUNet_raw\Dataset101_Spines',
    dataset_name = 'Spines',
    dataset_id   = 101
)
```

After running, open `dataset.json` and verify the label dictionary shows `{"0": "background", "1": "spine", "2": "dendrite"}` and the file list matches your case count.

---

### Step 8 — Plan, preprocess, and train

```bash
conda activate respaninternal

set nnUNet_raw=f:\ResPan\nnUNet_training\nnUNet_raw
set nnUNet_preprocessed=f:\ResPan\nnUNet_training\nnUNet_preprocessed
set nnUNet_results=f:\ResPan\nnUNet_training\nnUNet_results

nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
```

Check the output: look at `dataset_fingerprint.json` in `nnUNet_preprocessed\Dataset101_Spines\`. Confirm median Z and XY shapes are what you expect. If the median Z is reported as < 20, your stacks are too thin and you need larger acquisitions.

```bash
nnUNetv2_train 101 3d_fullres all
```

Expected time: 12–24 hours. Monitor the training log:

```
f:\ResPan\nnUNet_training\nnUNet_results\Dataset101_Spines\
    nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\
        training_log_{date}.txt
```

Watch for `pseudo_dice` increasing toward 1.0 and both `train_loss` and `val_loss` decreasing. If `val_loss` starts increasing while `train_loss` keeps falling, you are overfitting — you need more training data.

---

### Step 9 — Point RESPAN to your new model

After training completes, the model is at:

```
f:\ResPan\nnUNet_training\nnUNet_results\Dataset101_Spines\
    nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\
        checkpoint_best.pth
```

In the RESPAN GUI, set the nnU-Net model path to `f:\ResPan\nnUNet_training\nnUNet_results\` (the results root, not the checkpoint itself). Set dataset ID to 101. Enable the YAML checkbox so voxel sizes are read correctly. Run analysis the same way as with Model 1A.

---

## Image Specs Checklist

### What to check before acquiring new data

The specs below are derived from Model 1A's training distribution and the nnU-Net patch size of 32×256×256. If you are training your own model, your training data should match your acquisition specs — but these are also the specs that will give you the best results with the pretrained Model 1A.

### nnU-Net (segmentation)

| Property | Minimum | Ideal | Notes |
|---|---|---|---|
| XY pixel size | 60–150 nm | 65–102 nm | Model 1A trained on 65 nm native; 82–102 nm is within range |
| Z step | 150–300 nm | 200 nm | Coarser than 300 nm collapses spine necks; finer is fine |
| Z planes | 32 | 50+ | Less than 32 means the full patch depth cannot be used |
| XY frame size | 256×256 px | 400×400+ | Below 256 on any axis gets padded, wasting training efficiency |
| Bit depth | 16-bit uint16 | 16-bit uint16 | 8-bit loses dynamic range; 32-bit float not supported |
| Saturated pixels | Avoid | Zero | Clips high-intensity signal; disrupts z-score normalization |
| Signal brightness | — | Foreground median ~1000/65535 | Model 1A training median foreground was 1043; match this range |
| Channel | Ch0 structural only | Ch0 structural only | nnU-Net sees one channel (_0000); do not pass the functional channel |
| Isotropy ratio | < 4× | < 3× | At 82.5 nm / 200 nm the ratio is 2.4× — acceptable |

**Do not mix in one training dataset:**
- Confocal and widefield (different PSF → different spine appearance)
- Z steps differing by more than 2× (e.g. 200 nm and 410 nm together)

**Can mix:**
- XY pixel sizes between 65 and 130 nm — nnU-Net resamples to median spacing
- Different cell types (cortical and hippocampal) if they share similar morphology and imaging conditions

### Are your current example stacks OK for training?

| Stack | Shape | Assessment |
|---|---|---|
| `140325_DIV17cx_LTP30min` | (15, 141, 108) uint16, 102 nm XY / 200 nm Z | **Not suitable for training.** Z=15 is below the 32-plane patch depth. XY 141×108 is below 256 in both axes. Max pixel = 65535 (saturated). Use for validation or result inspection only. |
| `C2-HippoDIV14_control` | (28, 165, 362) uint16, 81.3 nm XY / 200 nm Z | **Marginal and wrong channel.** Z=28 is close but below ideal. One XY axis (165 px) is below 256. More critically, this is Ch2 (functional), not Ch0 (structural) — wrong input for nnU-Net. |

**The correct stacks to use for training** are the large dendritic ROI stacks from the LIF files:
- `180626_cortical_overview` series 20–21: 82.5 nm XY — large overview stacks with full dendritic fields
- `280622no2_hippo_200K` series 3–6: 82.5 nm XY / 200 nm Z dendritic ROI stacks

Before adding any stack to training, verify dimensions:
```python
from tifffile import imread
img = imread("your_stack.tif")
print(img.shape)   # need (Z >= 32, Y >= 256, X >= 256)
print(img.max())   # should be well below 65535
```

---

### CARE (only if you decide you need it)

| Property | Requirement |
|---|---|
| Paired acquisition | Required — same FOV, same cells, same session |
| Low SNR image | Fast acquisition (short exposure or few averages) |
| High SNR image | Slow acquisition (long exposure or many averages) |
| Shape | Both images must have identical ZYX shape |
| Bit depth | 16-bit uint16 |
| Minimum pairs | 20 |
| Saturated pixels | Must avoid — CARE tries to restore clipped voxels and will produce artifacts |
| Z planes | Any; RESPAN auto-pads to multiple of 8 |
| Modality | Must match the data you will restore at analysis time |

Do not fabricate CARE pairs from unrelated images or different sessions. The network is learning the noise statistics of your specific microscope and acquisition settings.

---

### SelfNet (only if Z anisotropy is clearly hurting segmentation)

| Property | Requirement |
|---|---|
| Input | Single 3D TIF, same modality and resolution as your analysis data |
| Z planes | 50+ (need enough XZ slices after interval sampling to build patch dataset) |
| `--scale` | XY_pixel_size ÷ Z_step. Examples: 82.5 nm / 200 nm → **0.41** · 102 nm / 200 nm → **0.51** |
| Two GPUs | Required for inference (cuda:0 and cuda:1); training uses only cuda:0 |
| Stack content | Must contain clear dendritic structure with visible spines in XY — that is what the network learns to restore in XZ |

Your current small stacks (15 Z planes, 28 Z planes) are too thin for SelfNet training. The cortical overview stacks from the LIF are the right input.

**Inference note:** After SelfNet runs, RESPAN sets `input_resZ = input_resXY`. nnU-Net will then treat the restored volume as isotropic. This changes the scaling applied before segmentation — do not enable SelfNet with a model that was not trained on isotropic input.
