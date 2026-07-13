# RESPAN Pipeline — Step-by-Step Guide

This document covers the complete workflow from raw microscopy images to spine analysis results, ROI export, and model training.

---

## Environments

| Environment | Python | Used for |
|---|---|---|
| `respandev` | 3.10 | RESPAN analysis, ROI export |
| `respaninternal` | 3.9 | nnU-Net segmentation only |

On **Windows**, `run_respan.py` handles both automatically (calls `respaninternal` as a subprocess for nnU-Net, then continues in `respandev`).  
On **Mahti**, run them separately — `run_nnunet_only.py` under `respaninternal` first, then `run_respan.py` under `respandev`.

---

## Part 1 — Data Preparation

### Step 1 — Convert images to TIF (if not already TIF)

Run `batch-convertor/Tif_Batch_2024.ijm` in Fiji:
- Select the folder containing your raw files (LIF, CZI, ND2, OIF, etc.)
- Background subtraction: **0** (none)
- Export MIPs: **No**
- Output lands in a `TIF/` subfolder with the same filenames

If you already have multi-channel TIF files, skip this step.

---

### Step 2 — Extract the structural channel

Run `batch-convertor/extract_channel.ijm` in Fiji:
- Select the folder containing your (multi-channel) TIF files
- Enter the channel index: **0** = first channel, **1** = second, **2** = third
- Output lands in `extracted_ch{N}/` subfolder with identical filenames

The structural/morphology channel (the one you image the cell fill with, not the functional marker) is what RESPAN needs. This is typically channel 0.

---

### Step 3 — Place images in the analysis folder

Copy or move the single-channel TIFs from `extracted_ch{N}/` into:
```
F:\ResPan\images_analysis\images\
```

No renaming required. Any filename works. The `Analysis_Settings.yaml` already in that folder will be used for all images.

---

## Part 2 — Running the Analysis

### Step 4 — Run segmentation and spine analysis

**Windows:**
```powershell
# All images
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\run_respan.py

# Specific images only (by folder name, not filename)
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\run_respan.py crop14_... crop6_...
```

**Mahti:**
```bash
# Step A — nnU-Net segmentation (respaninternal)
/projappl/project_2019059/respan/envs/respaninternal/bin/python \
    /scratch/project_2019059/respan/code/ResPan/run_nnunet_only.py

# Step B — RESPAN analysis (respandev)
/projappl/project_2019059/respan/envs/respandev/bin/python \
    /scratch/project_2019059/respan/code/ResPan/run_respan.py
```

The pipeline creates one output folder per image under `images_analysis/`:
```
images_analysis/<image_name>/
    images/           <- hardlink to original TIF
    labels/           <- nnU-Net segmentation (Z,Y,X) values 1-6
    Tables/           <- detected_spines.csv, dendrite_summary.csv
    Spine_Arrays/     <- per-spine sub-volumes
    Validation_Data/
        Validation_MIPs/   <- 8-channel MIP composites for QC
        Spine_Meshes/      <- per-spine .tif and .obj (head/, neck/, spine/)
    SWC_files/        <- dendrite skeleton
```

If labels already exist for an image, nnU-Net is skipped automatically — only the analysis re-runs.

---

### Step 5 — Export Fiji ROIs

```powershell
# All images
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\generate_fiji_rois.py

# Single image
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\generate_fiji_rois.py --crop <folder_name>
```

Output per image:
- `ROIs/<name>_rois_3d.zip` — one polygon ROI per structure per Z-slice (load on the 3D stack)
- `ROIs/<name>_rois_mip.zip` — one polygon ROI per structure, Z-projected (load on a MIP image)

ROIs are named `dendrite-Z{n}`, `spine_{id}-Z{n}`, `neck_{id}-Z{n}`. The `{id}` matches `spine_id` in the CSV.

**ROI source:** the script reads RESPAN's own instance-separated volume from `Validation_Data/Validation_Vols/<name>.tif` (channel 2 = spines with their IDs, channel 3 = necks). These IDs match the CSV exactly, so ROIs are the same spines RESPAN counted and adjacent spines are never merged. If that file is missing (i.e. `save_val_volumes` was off), it falls back to connected components on the label TIF + centroid matching to the CSV.

**To load in Fiji:** open the image → Analyze > Tools > ROI Manager > More > Open → select zip → check Show All.

**Why ROIs:** to measure mean fluorescence intensity of a second or third channel (e.g. Lifeact-GFP, AMPAR) on the exact same per-spine/neck regions without re-running the pipeline.

---

## Part 3 — QC and Manual Label Correction

### Step 6 — Check results in Fiji

Open `images_analysis/<name>/images/<name>.tif` and load `ROIs/<name>_rois_3d.zip`. Check that:
- Dendrite ROI follows the shaft correctly on each Z slice
- Spine ROIs cover real spines (not artefacts or fragments)
- Neck ROIs connect to the dendrite

For visual reference, also look at `Validation_Data/Validation_MIPs/` — RESPAN generates 8-channel MIP composites showing raw + labels + spine IDs.

---

### Step 7 — Correct labels manually in Fiji (if needed)

The label TIF is at `images_analysis/<name>/labels/<name>.tif`. Values:
```
1 = dendrite
2 = spine core
3 = spine membrane
4 = neck
5 = soma
6 = axon
```

To edit:
1. Open the label TIF in Fiji
2. Use **Image > Color > Split Channels** or make a composite to separate classes
3. Edit with pencil/brush tools (erase false positives, add missed structures)
4. Merge back to a single-channel integer TIF with the same value encoding
5. Save over the original `labels/<name>.tif`

---

### Step 8 — Re-run analysis on corrected images

```powershell
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\run_respan.py <folder_name>
```

nnU-Net is skipped (labels exist). Analysis re-runs and overwrites Tables, Spine_Arrays, Validation_Data, SWC_files. Then re-export ROIs:

```powershell
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\generate_fiji_rois.py --crop <folder_name>
```

---

## Part 4 — Reading the Results

### Spine counts (corrected for multi-head spines)

`detect_multi_head_spines: True` is enabled in all crop YAMLs. This flags cases where nnU-Net split one physical spine head into two segments. To get the correct count:

```python
import pandas as pd

df = pd.read_csv('images_analysis/<name>/Tables/<name>_detected_spines.csv')

# Corrected count — one row per physical spine
n_spines = (df['head_index'] == 0).sum()

# Or if the column is absent (detect_multi_head_spines was False when run):
n_spines = len(df)
```

Check `Tables/<name>_Multi_Head_Spine_Groups.csv` to see which spines were grouped and how many secondary heads were found.

---

## Part 5 — Model Training (when needed)

Only needed if the pretrained model (Dataset002_Spines) performs poorly on new data.

### Step 9 — Collect training images

Training images should be single-channel structural TIFs (Ch0), same format as analysis images. Minimum requirements for nnU-Net:
- Z planes: ≥ 32 (ideally 50+)
- XY frame: ≥ 256 × 256 px
- Bit depth: uint16
- No saturated pixels (max value should be well below 65535)

---

### Step 10 — Generate starting labels

Run `run_respan.py` on your training images using the current best model. The output `labels/<name>.tif` gives you a starting point to correct — you are editing, not annotating from scratch.

---

### Step 11 — Correct labels for training

For each training image, set up a folder under `manual_fixing_labels/`:

```
manual_fixing_labels/
└── case001/
    ├── imagesTr/
    │   └── case001.tif     <- raw structural TIF
    └── labelsTr/
        └── case001.tif     <- label TIF from Step 10 (renamed to match)
```

Run `manual_fixing_labels/imageJ_plugins/1_Labels_to_Channels_for_Annotation.ijm` in Fiji — converts the integer label TIF into a multi-channel composite for editing. Edit each channel (spine, dendrite). Run `2_Channels_to_Labels_for_Training.ijm` to convert back to an integer label TIF.

---

### Step 12 — Build the training dataset

Copy corrected pairs into:
```
F:\ResPan\training_data\
    imagesTr\     <- raw structural TIFs (any filename for now)
    labelsTr\     <- corrected integer label TIFs (same filenames as imagesTr)
```

Then run `batch-convertor/rename_for_nnunet.ijm` in Fiji:
- Select `imagesTr/` as the images folder
- Select `labelsTr/` as the labels folder
- The macro renames both in place to nnU-Net format:
  - Images: `case{NNN}_0000.tif`
  - Labels: `case{NNN}.tif`
- Files already named `case{NNN}...` are skipped — safe to re-run when adding new data
- New case numbers continue from the highest existing one automatically

Both files for the same case must have identical ZYX shape.

---

### Step 13 — Train

```powershell
# Setup dataset (run once)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe f:\ResPan\train_nnunet.py setup

# Plan and preprocess (run once)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe f:\ResPan\train_nnunet.py plan

# Train (resumable — safe to stop and restart)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe f:\ResPan\train_nnunet.py train
```

Expected time: 12–24 hours on a GPU. Monitor `nnunet_training/results/Dataset001_Spines/.../training_log_*.txt` — look for `pseudo_dice` increasing and both train/val loss decreasing.

---

### Step 14 — Point the pipeline to the new model

Place the trained model folder inside:
```
F:\ResPan\images_analysis\model\
```

`run_respan.py` automatically picks the first `Dataset*` folder it finds there alphabetically. No other changes needed.

---

## Quick Reference

| Task | Command / Tool |
|---|---|
| Convert format to TIF | `batch-convertor/Tif_Batch_2024.ijm` (Fiji) |
| Extract single channel | `batch-convertor/extract_channel.ijm` (Fiji) |
| Run full analysis | `respandev python run_respan.py` |
| Run analysis on specific crops | `respandev python run_respan.py crop3_... crop14_...` |
| Export Fiji ROIs | `respandev python generate_fiji_rois.py` |
| Train nnU-Net | `respaninternal python train_nnunet.py train` |
| Train CARE | `respandev python train_care.py` |
| Train SelfNet | `respaninternal python train_selfnet.py` |
