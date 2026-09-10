# How to Run the Pipeline — 0 → 100 (complete runbook)

A linear, copy-paste guide for someone using this **blindly**. It covers the whole loop:

```
raw microscope files
      │  ── PART 1 (Fiji, on your workstation) ───────────────
      ▼
single-channel structural TIFs  →  images_analysis/images/ + YAML
      │  ── PART 2 upload · PART 3 connect ──────────────────
      ▼
nnU-Net SEGMENTATION      (run_nnunet_only.py)   → labels/         ── PART 4
      ▼
RESPAN ANALYSIS           (run_respan.py)        → Tables/, Validation_Data/  ── PART 5
      ▼
Fiji ROI EXPORT           (generate_fiji_rois.py)→ ROIs/           ── PART 5
      ▼
DOWNLOAD results                                                   ── PART 6
      │
      ▼  ── want a better model? correct labels and fine-tune ──
manual LABEL CORRECTION   (Fiji macros 1 → edit → 2)              ── PART 7
      ▼
build training_data/      (naming!)                               ── PART 8
      ▼
FINE-TUNE                 (train_nnunet.py finetune)              ── PART 9
      ▼
USE the new model         (point analysis at it)                 ── PART 10
      └────────────────  iterate: correct → fine-tune → re-run  ─┘
```

### Where each command runs
| Tag | Where | What it's for |
|---|---|---|
| 💻 **WORKSTATION (Fiji)** | your own computer running Fiji (`F:\ResPan`) | data prep, label correction, staging files — the GUI steps that can't run on the cluster |
| 🌐 **DASHBOARD** | browser: `https://www.roihu.csc.fi` (or SFTP) | moving files to/from Roihu |
| 🔑 **GPU-LOGIN** | `roihu-gpu-loginN` (after `ssh`) | light commands, submit jobs — **never** the CPU login |
| 🖥️ **GPU-NODE** | the `rg####` compute node from `srun`/`sbatch` | the actual GPU work |

### Fixed facts for this project (already set up on Roihu)
- User: `kharash` · Project: `project_2019059`
- Code: `/scratch/project_2019059/respan/`
- Envs: `/projappl/project_2019059/respan/envs/respandev` (analysis) and `.../respaninternal` (nnU-Net)
- Runtime modules: `gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12`
- Two Python interpreters, always called by **full path** (no `conda activate`):
  - **respandev** → `/projappl/project_2019059/respan/envs/respandev/bin/python`
  - **respaninternal** → `/projappl/project_2019059/respan/envs/respaninternal/bin/python`

### Label class scheme (what the pixel values mean)
| 0 background | 1 dendrite | 2 spine core | 3 spine shell (membrane) | 4 spine neck | 5 soma (ignored) | 6 axon (ignored) |
|---|---|---|---|---|---|---|

---

# PART 1 — Prepare the data 💻 WORKSTATION (Fiji)

All macros are in `batch‑convertor/`. Run them in Fiji: **Plugins ▸ Macros ▸ Run…**

### 1.1 Raw microscope files → TIF  (only if your data isn't TIF yet)
Macro **`Tif_Batch_2024.ijm`** → select the folder of raw files (CZI / LIF / ND2 / OIF…).
Background subtraction `0`, MIPs off. → multi-channel TIFs in a `TIF/` subfolder.
*(For LIF specifically you can instead use `lif_to_tif.py`.)*

### 1.2 Extract the structural (fill) channel
Macro **`extract_channel.ijm`** → select the TIF folder, enter the **0-based** channel index of
your fill marker (`0` = first channel, `1` = second, …). For this project's Zeiss data the fill
(mScarlet) is **C2 → index `1`**. → single-channel TIFs in `extracted_ch{N}/`.

### 1.3 Keep filenames space-free
Underscores / dashes only. RESPAN dislikes spaces.

### 1.4 Build the dataset folder
Put the single-channel TIFs in an `images/` subfolder and add one `Analysis_Settings.yaml`.
The nnU-Net model goes in `model/`:
```
images_analysis/
    images/
        crop01.tif
        crop02.tif …
        Analysis_Settings.yaml
    model/
        Dataset226_…_V6/           (the nnU-Net model to segment with)
```
> One `images/` folder = **one YAML = one resolution = one model.** Data at a different resolution
> needs its own dataset folder (e.g. `images_analysis2/`) with its own YAML.

### 1.5 Set the resolution in the YAML (µm)
```yaml
Parameters:
    input_resXY: 0.043   # YOUR data's XY pixel size
    input_resZ:  0.150   # YOUR data's Z step
    model_resXY: 0.065   # the MODEL's native XY  (base Dataset226 = 0.065)
    model_resZ:  0.150   # the MODEL's native Z    (base Dataset226 = 0.150)
```
- `input_res` = your voxel size. `model_res` = the resolution the model was trained at.
- If they differ, the scripts rescale the image → model scale for nnU-Net, then rescale the labels
  back to your grid. If equal (e.g. a model fine-tuned at your native res), no rescaling happens.
- All spine-filter/analysis parameters are documented inline in the YAML.

---

# PART 2 — Upload to Roihu 🌐 DASHBOARD

Open `https://www.roihu.csc.fi` (MFA login) → **Files** → go to
`/scratch/project_2019059/respan/` and drag-drop:
- your dataset folder (e.g. `images_analysis/`, containing `images/` + `model/`)
- **if you changed any script**, re-upload it (`run_respan.py`, `run_nnunet_only.py`,
  `generate_fiji_rois.py`, `train_nnunet.py`, …). Code changes just overwrite the file —
  **no env rebuild.** (Tip: after re-uploading `.py` files, you may clear stale bytecode:
  `find /scratch/project_2019059/respan -type d -name __pycache__ -exec rm -rf {} +`)

---

# PART 3 — Connect to the GPU login node 🔑 GPU-LOGIN

💻 From your machine:
```bash
ssh kharash@roihu-gpu.csc.fi
```
**Check the prompt says `roihu-gpu-loginN`.** If it says `roihu-cpu-…` you're on the wrong side —
the GPU modules and the ARM containers only exist on the **-gpu** login.

---

# PART 4 — Segmentation (nnU-Net) 🔑 GPU-LOGIN

`run_nnunet_only.py` = **segmentation only**. It auto-scans every dataset folder, segments each
image **without** a label, skips images that already have one. **No arguments needed.** → writes
`labels/` into each crop folder.

### Option A · Batch job (recommended — survives disconnects)
```bash
cat > ~/run_seg.sh <<'EOF'
#!/bin/bash
#SBATCH --account=project_2019059
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=seg_%j.out
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
/projappl/project_2019059/respan/envs/respaninternal/bin/python /scratch/project_2019059/respan/run_nnunet_only.py
EOF
sbatch ~/run_seg.sh
```
Watch it (🔑 GPU-LOGIN):
```bash
squeue --me            # RUNNING → row disappears when done
cat seg_*.out          # log + any errors
```

### Option B · Interactive session 🔑 GPU-LOGIN → 🖥️ GPU-NODE
```bash
# 🔑 request a GPU (the trailing 'bash' is required — without it the screen hangs)
srun --account=project_2019059 --partition=gputest --gres=gpu:gh200:1 --time=00:15:00 --pty bash
```
```bash
# 🖥️ GPU-NODE (prompt is now rg####)
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
/projappl/project_2019059/respan/envs/respaninternal/bin/python /scratch/project_2019059/respan/run_nnunet_only.py
```

> **Just want labels to hand-correct?** Stop after Part 4. You do **not** need Parts 5.

---

# PART 5 — Analysis + ROI export 🖥️ GPU-NODE (or add to the sbatch script)

### 5.1 RESPAN analysis (measurements)
`run_respan.py` = analysis → `Tables/` (detected_spines + dendrite_summary CSVs),
`Validation_Data/` (MIPs, 3D volumes, per-spine meshes), `SWC_files/`. On Roihu it **skips**
nnU-Net (that's why Part 4 ran first — a Tykky container can't launch another container).
```bash
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
/projappl/project_2019059/respan/envs/respandev/bin/python /scratch/project_2019059/respan/run_respan.py
```
- Limit to one dataset: append its folder name, e.g. `… run_respan.py images_analysis`
- Use a specific model: `… run_respan.py --model <path-to-Dataset*_folder> images_analysis`

### 5.2 Fiji ROI export (optional)
`generate_fiji_rois.py` reads the analysis output and writes, per crop:
`ROIs/<name>_rois_3d.zip` (per-Z-slice polygons) + `ROIs/<name>_rois_mip.zip` (flat MIP polygons).
ROIs are named `dendrite`, `spine_{id}`, `neck_{id}` with IDs matching the CSV.
```bash
/projappl/project_2019059/respan/envs/respandev/bin/python /scratch/project_2019059/respan/generate_fiji_rois.py --root /scratch/project_2019059/respan/images_analysis
```
> ⚠️ **Always pass an absolute `--root`** (or `cd` into the repo first). Its default is a *relative*
> path, so running it from your home directory gives `FileNotFoundError: 'images_analysis'`.
> Also: it needs the **analysis** to have run (Part 5.1) — it reads `Validation_Data/` /`Tables/`.
> One crop only: add `--crop <name>`.

Load the zips in Fiji: **Analyze ▸ Tools ▸ ROI Manager ▸ More ▸ Open**. Use them to measure a
second/third channel's intensity on the exact spine/neck regions.

---

# PART 6 — Download the results 🌐 DASHBOARD

`https://www.roihu.csc.fi` → Files → `/scratch/project_2019059/respan/images_analysis/` → download
the crop folders (each has `labels/`, and if you ran Part 5: `Tables/`, `Validation_Data/`, `ROIs/`).

---

**If the base model is good enough, you're done.** The rest (Parts 7–10) is the fine-tuning loop:
correct the labels, train a better model, and re-run.

---

# PART 7 — Manual label correction 💻 WORKSTATION (Fiji)

Goal: turn the model's rough `labels/` into **corrected ground truth** — necks are the priority
(see the neck rules at the end). Macros are in `manual_fixing_labels/imageJ_plugins/`; a written
guide is `correction_manual.pdf` in the same folder.

### 7.1 Lay out a correction folder
Make a folder (e.g. `training_data2/`) with two subfolders — the **image** and its **label** must
share the **same base name**, with the image carrying the `_0000` suffix:
```
training_data2/
    imagesTr/   Experiment-35-…-01-1_0000.tif     (the structural fill TIF)
    labelsTr/   Experiment-35-…-01-1.tif          (the model's label to correct)
```
(The model wrote `labels/<name>.tif`; the fill image is your Part 1 `extracted_ch1` TIF. Copy the
image in with `_0000` added, the label with the bare name.)

### 7.2 Macro 1 — split label into editable channels
Run **`1_Labels_to_Channels_for_Annotation-v2.ijm`**:
- **folder** = the `training_data2/` folder (the one containing `imagesTr` + `labelsTr`)
- **Maximum label intensity** = the highest class present. For core/shell/neck crops that's **4**;
  use **6** if soma/axon are present.
- → writes multi-channel stacks to `training_data2/labels_as_channels/` — channel 1 = Raw image,
  then one channel per class (c2 = dendrite, c3 = core, c4 = shell, **c5 = neck**, …).

### 7.3 Correct in Fiji
Open each `labels_as_channels/` stack. On each class channel, **paint** (add) or **erase** (remove)
voxels so they match the truth — mostly the **neck** channel. Save the stack back in place when done.
See the neck rules below.

### 7.4 Macro 2 — merge channels back into a label
Run **`2_Channels_to_Labels_for_Training.ijm`** with the **same folder** and the **same Maximum
label intensity**. → writes the corrected single-image labels back out (into a dated `labelsTr_*`
folder). These corrected labels are your ground truth for training.

### Neck correction rules (the important part)
- **Draw necks continuous.** A spine is by definition joined to the dendrite; a broken neck is an
  imaging artifact (thin, dim, and it falls between the 150 nm Z-planes), not real. Bridge the gaps
  with a **thin** continuous neck from head to dendrite.
- **Turn up brightness/contrast first** — most "gaps" are faint signal you can label truthfully.
- **Check adjacent Z-planes** before calling a gap — the neck is often present one slice up/down.
- **Don't fabricate.** No trace *and* no nearby dendrite → leave it disconnected, or drop the spine.
- **Ambiguous regions:** don't guess. Leave the model's pixels, or pick a cleaner crop. Confidently
  wrong labels (erasing a real spine, painting noise as spine) are what hurt the model — not gaps.

---

# PART 8 — Build `training_data/` for the trainer 💻 WORKSTATION (Fiji)

`train_nnunet.py` reads `training_data/imagesTr/` + `training_data/labelsTr/`, pairing files by
**identical name**. So in `training_data/`, the image and its label must have the **exact same
filename** — meaning **drop the `_0000`** from the image side:
```
training_data/
    imagesTr/   crop01.tif        ← fill image, _0000 REMOVED
    labelsTr/   crop01.tif        ← corrected label, same name
```
```powershell
# 💻 WORKSTATION (Fiji) PowerShell — strip _0000 from images while copying; labels copied as-is
Get-ChildItem <corrected>\imagesTr\*_0000.tif | ForEach-Object {
    Copy-Item $_ ("F:\ResPan\training_data\imagesTr\" + ($_.Name -replace '_0000\.tif$','.tif')) }
Copy-Item <corrected>\labelsTr\*.tif F:\ResPan\training_data\labelsTr\
```
> ⚠️ If the names don't match, `setup` errors with `no matching label for …`. The trainer re-adds
> `_0000` and `caseNNN` numbering itself, so `training_data/` uses **bare matching names**.

Then upload `training_data/` to `/scratch/project_2019059/respan/` (🌐 DASHBOARD). Also make sure
the **base model** is on Roihu at
`cloned_respan/Dataset226_Global1Bv2_XY65nm_Z150nm_CARE-Aug_spinecore_neck_dend_soma_axon_V6/`
— fine-tuning initializes from its `checkpoint_final.pth`.

---

# PART 9 — Fine-tune the model 🔑 GPU-LOGIN

`train_nnunet.py finetune` does it all in one command: build the nnU-Net dataset from
`training_data/` → transfer plans from Dataset226 → train with the pretrained weights. It's
**resumable** — nnU-Net checkpoints periodically and continues if you re-submit. Fine-tuning takes
**hours**, so use `gpumedium`, not `gputest`.
```bash
cat > ~/run_finetune.sh <<'EOF'
#!/bin/bash
#SBATCH --account=project_2019059
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=24:00:00
#SBATCH --output=finetune_%j.out
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
cd /scratch/project_2019059/respan
/projappl/project_2019059/respan/envs/respaninternal/bin/python train_nnunet.py finetune
EOF
sbatch ~/run_finetune.sh
```
Watch (🔑 GPU-LOGIN): `squeue --me` and `cat finetune_*.out`.

> Training **from scratch** instead of fine-tuning? Use `train_nnunet.py all`
> (= `setup` → `plan` → `train`). Fine-tuning is the recommended path here.

---

# PART 10 — Use the new model 🖥️ GPU-NODE

The trained model lands in:
```
nnunet_training/results/Dataset001_Spines/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/
```
1. If there's no `checkpoint_final.pth`, copy `checkpoint_latest.pth` (or `checkpoint_best.pth`) to
   `checkpoint_final.pth` — the analysis scripts look for `best` then `final`.
2. Put the `Dataset001_Spines` folder into your dataset's `model/` (as the only `Dataset*`), **or**
   pass `--model /scratch/project_2019059/respan/nnunet_training/results/Dataset001_Spines`.
3. In the dataset's YAML set **`model_res == input_res`** (it was fine-tuned at your native
   resolution → no rescaling): e.g. `model_resXY: 0.043`, `model_resZ: 0.150`.
4. Re-run **Parts 4–5** with the new model → better labels (especially necks).

**Iterate:** correct → fine-tune → segment → inspect necks → correct a few more → fine-tune again.

---

# PART 11 — Troubleshooting (things that actually bite)

| Symptom | Cause → Fix |
|---|---|
| `module … cuda/12.6.3 unknown` | You're on the **CPU** login. `ssh kharash@roihu-gpu.csc.fi`. |
| `FileNotFoundError: 'images_analysis'` (ROIs) | `generate_fiji_rois.py` run from `~`. Pass `--root /scratch/project_2019059/respan/images_analysis` or `cd` there first. |
| ROI script runs but finds nothing | You skipped Part 5.1 — it needs the **analysis** output (`Validation_Data/`/`Tables/`). |
| `no matching label for …` (train setup) | `training_data/imagesTr` and `labelsTr` names differ. Drop `_0000` so names match (Part 8). |
| Black screen after `srun`, frozen | `srun` missing trailing `bash`. Use `--pty bash`. |
| Still black / frozen | Stuck job holding the node: `squeue --me`, then `scancel -u kharash`, retry. |
| Interactive keeps dying | Use **sbatch** (Options A) — it needs no live terminal. |
| Labels look wrong / at wrong scale | YAML `input_res`/`model_res` not set for this data, or wrong model in `model/`. |
| nnU-Net says "no GPU" | Forgot `module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12`. |
| Fine-tune can't find pretrained weights | Base `Dataset226_…_V6/` missing on Roihu (Part 8). |

**Cancel everything:** 🔑 GPU-LOGIN → `scancel -u kharash`

---

# Quick reference — the minimal happy path

```bash
# 💻 WORKSTATION (Fiji): Fiji macros Tif_Batch → extract_channel; build images_analysis/{images,model}; set YAML
#           upload images_analysis/ to /scratch/project_2019059/respan/  (dashboard)

ssh kharash@roihu-gpu.csc.fi                       # 🔑 connect (GPU login!)

cat > ~/run_seg.sh <<'EOF'                          # 🔑 segment
#!/bin/bash
#SBATCH --account=project_2019059
#SBATCH --partition=gputest
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --output=seg_%j.out
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
/projappl/project_2019059/respan/envs/respaninternal/bin/python /scratch/project_2019059/respan/run_nnunet_only.py
EOF
sbatch ~/run_seg.sh ; squeue --me ; cat seg_*.out

# 🖥️ (optional) analysis + ROIs — add after the segment line, or run interactively:
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
/projappl/project_2019059/respan/envs/respandev/bin/python /scratch/project_2019059/respan/run_respan.py
/projappl/project_2019059/respan/envs/respandev/bin/python /scratch/project_2019059/respan/generate_fiji_rois.py --root /scratch/project_2019059/respan/images_analysis

# 🌐 download /scratch/project_2019059/respan/images_analysis/ (dashboard)
```
