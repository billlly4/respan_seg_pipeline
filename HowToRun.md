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
| 🌐 **DASHBOARD** | browser: `https://www.roihu.csc.fi` → **Files** (or SFTP) | moving files to/from Roihu |
| 🔑 **LOGIN SHELL** | the **GPU** login-node shell in the browser (`www.roihu.csc.fi`, prompt `roihu-gpu-loginN`) | light commands, submit + monitor jobs |
| 🖥️ **GPU-NODE** | the `rg####` compute node from `srun`/`sbatch` | the actual GPU work |

### Fixed facts for this project (already set up on Roihu)
- Project: `project_2019059` (use **your own** CSC username where a command needs it)
- Code: `/scratch/project_2019059/respan/`
- Envs: `/projappl/project_2019059/respan/envs/respandev` (analysis) and `.../respaninternal` (nnU-Net)
- Runtime modules: `gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12`
- Two Python interpreters, always called by **full path** (no `conda activate`):
  - **respandev** → `/projappl/project_2019059/respan/envs/respandev/bin/python`
  - **respaninternal** → `/projappl/project_2019059/respan/envs/respaninternal/bin/python`

> **Shared project — anyone on `project_2019059` can use this.** The code, envs, and `--account`
> are all project-level, so it works as-is; just use **your own** CSC username where one is needed
> (`scancel -u $USER` cancels your own jobs). Your `~/run_*.sh` scripts and `*.out` logs live in
> *your* home dir — no conflict. **But** input folders on `/scratch` (`images_analysis/`,
> `training_data/`) are shared: if two people run against the same folder at once they overwrite
> each other, so **work in separate subfolders** or coordinate. Trained models won't clash —
> `--name` auto-assigns a fresh `Dataset###`.

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

# PART 3 — Open a login shell 🔑 LOGIN SHELL

At `https://www.roihu.csc.fi`, open a **login-node shell** in the browser (same place as the
**Files** tab, so the whole workflow stays in the browser).

**Use the GPU login node.** Make sure the prompt says **`roihu-gpu-loginN`** (not `roihu-cpu-…`).
Do all your `sbatch` / `squeue` / `scancel` from here.

---

# PART 4 — Segmentation (nnU-Net) 🔑 LOGIN SHELL

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
Watch it (🔑 LOGIN SHELL):
```bash
squeue --me            # RUNNING → row disappears when done
cat seg_*.out          # log + any errors
```

### Option B · Interactive session 🔑 LOGIN SHELL (must be GPU login) → 🖥️ GPU-NODE
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

`train_nnunet.py` reads `training_data/imagesTr/` + `training_data/labelsTr/`, pairing each image
with its label by **identical filename**. So in `training_data/`, an image and its label must have
the **exact same name**. The trainer then re-adds `_0000` and `caseNNN` numbering itself when it
builds the real nnU-Net dataset (image → `case001_0000.tif`, label → `case001.tif`).

Your corrected labels come out of the macro **bare** (`crop01.tif`) and the images carry **`_0000`**
(`crop01_0000.tif`) — those don't match. Make them match. Either convention works as long as the
two names are identical; the recipe below **adds `_0000` to the labels** so both sides are `_0000`:
```
training_data/
    imagesTr/   crop01_0000.tif        ← fill image (keeps _0000)
    labelsTr/   crop01_0000.tif        ← corrected label (given _0000 to match)
```
```powershell
# 💻 WORKSTATION PowerShell — driven by the labels; adds _0000 so each label matches its image
$src = "<corrected>"                         # folder with imagesTr\ (…_0000.tif) + labelsTr\ (bare)
$dst = "F:\ResPan\training_data"
New-Item -ItemType Directory -Force "$dst\imagesTr","$dst\labelsTr" | Out-Null
Get-ChildItem "$src\labelsTr\*.tif" | ForEach-Object {
    $stem = $_.BaseName -replace '_0000$',''            # normalise, whether label is bare or _0000
    $img  = Join-Path "$src\imagesTr" ($stem + "_0000.tif")
    if (Test-Path $img) {
        Copy-Item $img (Join-Path "$dst\imagesTr" ($stem + "_0000.tif"))
        Copy-Item $_   (Join-Path "$dst\labelsTr" ($stem + "_0000.tif"))
    } else { Write-Warning "No matching image for $stem" }
}
"images: " + (Get-ChildItem "$dst\imagesTr\*.tif").Count
"labels: " + (Get-ChildItem "$dst\labelsTr\*.tif").Count
```
> ⚠️ If the names don't match, `setup` errors with `no matching label for …`. Put **only the crops
> you want to train on** in the labels source first, so the counts come out right.

Then upload `training_data/` to `/scratch/project_2019059/respan/` (🌐 DASHBOARD). Also make sure
the **base model** is on Roihu at
`cloned_respan/Dataset226_Global1Bv2_XY65nm_Z150nm_CARE-Aug_spinecore_neck_dend_soma_axon_V6/`
— fine-tuning initializes from its `checkpoint_final.pth` (or point `--from` at another model, PART 9).

---

# PART 9 — Fine-tune the model 🔑 LOGIN SHELL

`train_nnunet.py finetune` does it all in one command: build the nnU-Net dataset from
`training_data/` → transfer plans from the base model → train with its pretrained weights.
Fine-tuning takes **hours**, so use `gpumedium`, not `gputest`.

**The sbatch script must set three things** (learned the hard way — see PART 11):
- `--cpus-per-task=16` — without it Slurm gives **1 CPU** and the dataloader starves (training hangs).
- `export nnUNet_n_proc_DA=8` — parallel data augmentation (needs the `spawn` fix already in the code).
- `export PYTHONUNBUFFERED=1` — so the `.out` log updates live instead of in big buffered chunks.

```bash
cat > ~/run_finetune.sh <<'EOF'
#!/bin/bash
#SBATCH --account=project_2019059
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=finetune_%j.out
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12
cd /scratch/project_2019059/respan
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=8
/projappl/project_2019059/respan/envs/respaninternal/bin/python train_nnunet.py finetune --name ZeissV1
EOF
sbatch ~/run_finetune.sh
```

**Flags:**
- `--name <suffix>` → result is `Dataset###_<suffix>`; the ID number is auto-assigned (next free), so
  you never overwrite an earlier model and never type an ID.
- `--from <model folder>` → fine-tune from a specific model instead of the default Dataset226, e.g.
  `--from nnunet_training/results/Dataset001_ZeissV1` to continue from your last fine-tune. Chains
  freely: `226 → V1 → V2 …`.

**Watch the training — use the per-epoch log, not the buffered `.out`:**
```bash
LOG=$(ls -t /scratch/project_2019059/respan/nnunet_training/results/Dataset*_ZeissV1/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/training_log_*.txt | head -1)
tail -f "$LOG"
```
Each epoch prints `Pseudo dice [dendrite, core, shell, neck, soma, axon]` and `Epoch time`. With the
settings above expect **~120 s/epoch**. Watch the **neck** (4th value) climb; `soma`/`axon` stay 0.0
if those classes aren't in your crops (normal).

**How long / when to stop:** these are *training* scores (`fold_all` = no held-out validation), so
they can't show overfitting — don't judge quality from them. ~**300–500 epochs** is plenty for a
small fine-tune (lets the learning rate anneal). nnU-Net checkpoints continuously, so the wall
stopping it is fine. Judge the model by **running it on crops it did NOT train on** (PART 10) and
eyeballing the necks — not by the pseudo-dice.

> **Re-running:** plain `finetune` rebuilds a fresh dataset each time (auto-ID picks the next free
> number). To resume/reuse an already-built dataset without redoing setup, run
> `train_nnunet.py finetune_train --id <N> --name <suffix>` targeting that dataset's number.
>
> Training **from scratch** instead? `train_nnunet.py all` (= `setup` → `plan` → `train`).
> Fine-tuning is the recommended path here.

---

# PART 10 — Use the new model 🖥️ GPU-NODE

The trained model lands in the folder named after your `--name` (the log's `Output dataset:` line
tells you the exact `Dataset###_<name>`):
```
nnunet_training/results/Dataset001_ZeissV1/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/
```
1. If there's no `checkpoint_final.pth` (you stopped early / wall hit), copy `checkpoint_best.pth`
   (or `checkpoint_latest.pth`) to `checkpoint_final.pth` — the analysis scripts look for `best`
   then `final`.
2. Put the `Dataset001_ZeissV1` folder into your dataset's `model/` (as the only `Dataset*`), **or**
   pass `--model /scratch/project_2019059/respan/nnunet_training/results/Dataset001_ZeissV1`.
3. In the dataset's YAML set **`model_res == input_res`** (it was fine-tuned at your native
   resolution → no rescaling): e.g. `model_resXY: 0.043`, `model_resZ: 0.150`.
4. Re-run **Parts 4–5** with the new model → better labels (especially necks).

**Iterate:** correct → fine-tune → segment → inspect necks → correct a few more → fine-tune again.

---

# PART 11 — Troubleshooting (things that actually bite)

| Symptom | Cause → Fix |
|---|---|
| `module … cuda/12.6.3 unknown` (interactive only) | You ran `module load` on the **CPU** login. Use a **GPU** login shell (`roihu-gpu`) for interactive work. Doesn't affect `sbatch` — its modules load on the compute node. |
| `FileNotFoundError: 'images_analysis'` (ROIs) | `generate_fiji_rois.py` run from `~`. Pass `--root /scratch/project_2019059/respan/images_analysis` or `cd` there first. |
| ROI script runs but finds nothing | You skipped Part 5.1 — it needs the **analysis** output (`Validation_Data/`/`Tables/`). |
| `no matching label for …` (train setup) | `training_data/imagesTr` and `labelsTr` names differ. Make them identical (Part 8). |
| Fine-tune **hangs at `Epoch 0`**, GPU idle | Job got 1 CPU and/or dataloader `fork` deadlock. Ensure `--cpus-per-task=16` **and** `export nnUNet_n_proc_DA=8` (the `spawn` fix is already in `train_nnunet.py`). Last resort: `nnUNet_n_proc_DA=0` (single-thread, slow ~665 s/epoch). |
| Training log looks **frozen** but GPU is busy | The `.out` is buffered. Add `export PYTHONUNBUFFERED=1`, and watch `results/<ds>/…/fold_all/training_log_*.txt` (flushes per epoch). |
| Epoch time ~600 s+ | Augmentation single-threaded. Set `--cpus-per-task=16` + `nnUNet_n_proc_DA=8` → ~120 s. |
| `LRScheduler … 4 positional args` / `Weights only load failed` | Newer PyTorch on Roihu. Already patched (`polylr.py` + `train_nnunet.py`); **re-apply if you re-clone nnUNet** — see WhatHappened Day 22. |
| `Could not find a dataset with the ID N` (finetune_train) | That dataset isn't built/was deleted. Run full `finetune` (rebuilds it), or target an existing one's number. |
| `Batch script is empty` / `line must start with #!` | Paste dropped the `cat >` prefix or added blank lines. First line of the file must be `#!/bin/bash`; re-`cat >` the whole block or use `nano`. |
| Black screen after `srun`, frozen | `srun` missing trailing `bash`. Use `--pty bash`. |
| Still black / frozen | Stuck job holding the node: `squeue --me`, then `scancel -u $USER`, retry. |
| Interactive keeps dying | Use **sbatch** (Options A) — it needs no live terminal. |
| Labels look wrong / at wrong scale | YAML `input_res`/`model_res` not set for this data, or wrong model in `model/`. |
| nnU-Net says "no GPU" | Forgot `module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12`. |
| Fine-tune can't find pretrained weights | Base `Dataset226_…_V6/` missing on Roihu (Part 8), or bad `--from` path. |

**Cancel everything:** 🔑 LOGIN SHELL → `scancel -u $USER`

---

# Quick reference — the minimal happy path

```bash
# 💻 WORKSTATION (Fiji): Fiji macros Tif_Batch → extract_channel; build images_analysis/{images,model}; set YAML
#           upload images_analysis/ to /scratch/project_2019059/respan/  (dashboard)

# 🔑 open the GPU login shell in the browser at www.roihu.csc.fi (prompt: roihu-gpu-loginN)

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
