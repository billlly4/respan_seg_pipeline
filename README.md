# ResPan — RESPAN Spine-Analysis Pipeline (scripted + HPC)

Scripts to run and train the [RESPAN](https://github.com/lukejhammond/RESPAN) dendritic-spine
pipeline **headless** (no GUI), on a local workstation or on CSC's **Roihu** supercomputer.
It automates: data prep → nnU-Net segmentation → RESPAN spine analysis → Fiji ROI export, plus
model fine-tuning.

> **Want step-by-step commands?** See **[`HowToRun.md`](HowToRun.md)** — a copy-paste 0→100 runbook
> (local + Roihu). This README explains *what's here and how the pieces fit together*.

---

## What's in this repo

| Path | What it is |
|---|---|
| `run_respan.py` | Main driver — nnU-Net segmentation **+** RESPAN analysis (local: one command) |
| `run_nnunet_only.py` | Segmentation only (used on HPC, where the two stages run separately) |
| `generate_fiji_rois.py` | Convert RESPAN's output into Fiji-ready ROI zips |
| `train_nnunet.py` | Train / fine-tune the nnU-Net segmentation model |
| `train_care.py`, `train_selfnet.py` | Optional image-restoration model training (not in the default run) |
| `batch‑convertor/*.ijm` | Fiji macros for data prep (raw format → TIF, channel extraction) |
| `manual_fixing_labels/` | Label-correction macros + guide, for building fine-tuning data |
| `cloned_respan/` | Vendored **RESPAN** (third-party) — the spine-analysis engine |
| `nnUNet/` | Vendored **nnU-Net** (third-party) — the segmentation engine |
| `images_analysis/` | A dataset folder: your `images/`, the `model/`, and per-image outputs |
| `HowToRun.md` | Step-by-step runbook (local + Roihu HPC) |

The two big folders (`cloned_respan/`, `nnUNet/`) are external frameworks used as-is; everything at
the top level is the thin layer that drives them headless, reproducibly, and on HPC.

---

## Contents
1. [Environments](#environments)
2. [Portability — clone and go](#portability)
3. [The pipeline at a glance](#pipeline)
4. [Data preparation](#data-prep)
5. [Folder layout for analysis](#layout)
6. [Analysis settings (resolution!)](#settings)
7. [Run the analysis — local](#run-local)
8. [Run the analysis — Roihu (HPC)](#run-hpc)
9. [Export Fiji ROIs](#rois)
10. [Fine-tune a model on your own data](#finetune)
11. [Optional: CARE / SelfNet restoration](#restore)
12. [Label class scheme](#classes)

---

<a name="environments"></a>
## 1. Environments

Two conda/Tykky environments, kept separate because nnU-Net's PyTorch and RESPAN's
CuPy/analysis stack don't coexist cleanly:

| Environment | Purpose | Key stack |
|---|---|---|
| `respandev` | **RESPAN analysis** + ROI export (`run_respan.py`, `generate_fiji_rois.py`) | CuPy, scikit-image, RESPAN, tifffile, roifile |
| `respaninternal` | **nnU-Net** segmentation & training (`run_nnunet_only.py`, `train_nnunet.py`) | PyTorch, nnU-Net v2 |

You **launch scripts with the environment's Python by full path** (no `conda activate`):
- local (Windows conda): `…\envs\respandev\python.exe run_respan.py`
- Roihu (Tykky): `/projappl/<project>/respan/envs/respandev/bin/python run_respan.py`

`run_respan.py` finds the `respaninternal` interpreter automatically (sibling of the env you
launched with), so nothing is hardcoded — see below.

---

<a name="portability"></a>
## 2. Portability — clone and go

The scripts use `__file__`-relative paths and auto-detect the second environment, so a fresh
clone works with **no path editing**. To use the repo on a new machine:

1. Clone it.
2. Create the two envs named exactly **`respandev`** and **`respaninternal`** (as siblings).
3. Launch with the `respandev` Python.

The only convention that must hold is the **env names** and the folder layout below. Interpreter
resolution order: `$RESPAN_RESPANINTERNAL_PYTHON` (override) → sibling `respaninternal` env →
built-in fallback.

---

<a name="pipeline"></a>
## 3. The pipeline at a glance

```
raw microscope file (.czi/.lif/.nd2)
   │  Fiji: Tif_Batch_2024.ijm  →  extract_channel.ijm      (data prep)
   ▼
single-channel structural TIFs  (the "fill" channel)
   │
   ▼
nnU-Net segmentation   →  labels/   (dendrite/core/shell/neck…)
   │
   ▼
RESPAN analysis        →  Tables/ (CSVs), Validation_Data/, Spine_Meshes/, SWC_files/
   │
   ▼
generate_fiji_rois.py  →  ROIs/  (Fiji-ready ROI zips)
```

**Local:** `run_respan.py` does *both* nnU-Net and analysis in one command.
**Roihu:** nnU-Net and analysis are two commands (`run_nnunet_only.py` then `run_respan.py`),
because a Tykky container can't launch another container.

---

<a name="data-prep"></a>
## 4. Data preparation (Fiji macros in `batch-convertor/`)

1. **Any format → TIF** — `Tif_Batch_2024.ijm`: select the folder of raw files (CZI/LIF/ND2/…),
   background subtraction 0, MIPs off → multi-channel TIFs in a `TIF/` subfolder.
2. **Extract the structural channel** — `extract_channel.ijm`: select the TIF folder, enter the
   **0-based** channel index of your fill marker (0=first, 1=second, …) → single-channel TIFs in
   `extracted_ch{N}/`.
3. Put the single-channel TIFs into a dataset's `images/` folder (below). Keep filenames
   **space-free** (RESPAN dislikes spaces).

(`lif_to_tif.py` is a Python alternative for LIF files.)

---

<a name="layout"></a>
## 5. Folder layout for analysis

`run_respan.py` scans its **working directory** for **dataset folders**, each containing an
`images/` subfolder with the TIFs + one `Analysis_Settings.yaml`. The nnU-Net model lives in
`images_analysis/model/`. The working directory defaults to the scripts' own folder, but you can
point it elsewhere with the `RESPAN_WORKDIR` environment variable — set it to a per-user path so
several people can share one code checkout without their data/results colliding:

```
<repo root>/
├── images_analysis/
│   ├── images/
│   │   ├── crop01.tif …            (single-channel structural crops)
│   │   └── Analysis_Settings.yaml  (resolution + analysis parameters)
│   ├── model/
│   │   └── Dataset###_Spines/…     (nnU-Net model — first Dataset* is used)
│   └── crop01/ …                   (per-image outputs, created by the run)
└── run_respan.py, run_nnunet_only.py, generate_fiji_rois.py, train_*.py
```

- One `images/` folder = **one YAML = one resolution + one model.** Different-resolution data
  needs its own dataset folder with its own YAML.
- The model used is the first `Dataset*` folder in `images_analysis/model/` (alphabetical), or
  override with `--model <path>`.

---

<a name="settings"></a>
## 6. Analysis settings — resolution matters

In `Analysis_Settings.yaml`:

```yaml
Parameters:
    input_resXY: 0.043   # your DATA's XY pixel size (µm)
    input_resZ:  0.150   # your DATA's Z step (µm)
    model_resXY: 0.065   # the MODEL's native XY resolution (µm)
    model_resZ:  0.150   # the MODEL's native Z resolution (µm)
```

- `input_res` = the actual voxel size of your images.
- `model_res` = the resolution the nnU-Net model was trained at.
- If they differ, `run_respan.py`/`run_nnunet_only.py` **rescale the image to the model's scale**
  for nnU-Net, then rescale the labels back to your grid — so the model always sees structures at
  the size it learned. When they're equal (e.g. a model fine-tuned at your native resolution),
  no rescaling happens.
- Model native resolutions: base **Dataset226** = `0.065 / 0.150`; a model you fine-tune at
  native res → set `model_res == input_res`.

All spine-filter/analysis parameters are documented inline in the YAML.

---

<a name="run-local"></a>
## 7. Run the analysis — local

```powershell
# both segmentation + analysis, all datasets found
<respandev-python> run_respan.py

# only specific dataset folders
<respandev-python> run_respan.py images_analysis

# use a specific model
<respandev-python> run_respan.py --model "cloned_respan/Dataset226_…" images_analysis
```

Outputs per image: `labels/`, `Tables/` (detected_spines + dendrite_summary CSVs),
`Validation_Data/` (MIPs, 3D volumes, per-spine meshes), `SWC_files/`.

---

<a name="run-hpc"></a>
## 8. Run the analysis — Roihu (CSC HPC)

Roihu's **GPU partition is ARM (GH200)** with its own login node and modules — always work from
the **GPU** side.

```bash
# 1. connect to the GPU login node (NOT roihu-cpu)
ssh <user>@roihu-gpu.csc.fi

# 2. get a GPU  (interactive — note the trailing 'bash')
srun --account=<project> --partition=gputest --gres=gpu:gh200:1 --time=00:15:00 --pty bash
#    …or submit a batch job (sbatch) to avoid interactive terminals entirely.

# 3. load the runtime modules
module load gcc/13.4.0 cuda/12.6.3 cudnn/9.17.0.29-12

# 4. segment, then analyze (two steps — containers can't nest on Roihu)
/projappl/<project>/respan/envs/respaninternal/bin/python run_nnunet_only.py
/projappl/<project>/respan/envs/respandev/bin/python     run_respan.py
```

- `run_nnunet_only.py` auto-scans for datasets and segments every image **without** a label
  (already-labelled ones are skipped). No folder argument needed.
- If the interactive `srun` terminal misbehaves, use a **batch job** (`sbatch`) instead — it runs
  on the GPU node in the background and survives a dropped connection.
- Containers were built with Tykky (`conda-containerize`); code changes need only re-uploading
  the `.py` files, **no rebuild**.

---

<a name="rois"></a>
## 9. Export Fiji ROIs

```bash
<respandev-python> generate_fiji_rois.py                      # all crops
<respandev-python> generate_fiji_rois.py --crop <name>        # one crop
<respandev-python> generate_fiji_rois.py --source fallback    # force labels+CSV, skip instance volume
```

ROI source: by default it uses RESPAN's instance-segmentation volume when present (cleanest
per-spine separation) and falls back to the semantic label + CSV otherwise; `--source fallback`
forces that fallback path.

Writes per crop: `ROIs/<name>_rois_3d.zip` (per-Z-slice polygons for the stack) and
`ROIs/<name>_rois_mip.zip` (flat polygons for a MIP). ROIs are named `dendrite`, `spine_{id}`,
`neck_{id}` with IDs matching the CSV. Load in Fiji: **Analyze ▸ Tools ▸ ROI Manager ▸ More ▸
Open**. Use them to measure a second/third channel's intensity on the exact spine/neck regions.

---

<a name="finetune"></a>
## 10. Fine-tune a model on your own data

When the base model underperforms on a new microscope/marker, fine-tune:

1. **Segment** your crops (above) to get starting labels.
2. **Correct the labels by hand** — see `manual_fixing_labels/` (`correction_manual.pdf` + the two
   ImageJ macros): `1_Labels_to_Channels…` turns labels into editable channels → fix in Fiji
   (necks are the priority) → `2_Channels_to_Labels…` converts back.
3. **Assemble `training_data/`** — put the fill image and its corrected label under
   `training_data/imagesTr/` and `training_data/labelsTr/` with the **same filename**
   (`<name>.tif` in both). No renaming — the trainer adds nnU-Net's `_0000`/`caseNNN` itself.
4. **Fine-tune** from the base model — one command does it all (build dataset → transfer plans →
   train with pretrained weights):
   ```bash
   <respaninternal-python> train_nnunet.py finetune --name MyModel   # ~hours on GPU
   ```
   `--name` sets the result name (ID auto-assigned); add `--from <model folder>` to fine-tune from a
   model other than the default Dataset226 (e.g. a previous fine-tune). Training-from-scratch is
   `train_nnunet.py all` instead.
5. Point analysis at the new model (`--model`, or place it in `images_analysis/model/`) and set
   `model_res == input_res` in the YAML (fine-tuned at native resolution → no rescaling).

~15–20 corrected single-dendrite crops is a reasonable first set.

---

<a name="restore"></a>
## 11. Optional: CARE / SelfNet restoration

RESPAN can denoise (CARE) and restore axial resolution (SelfNet), but both need **trained
models** (CARE: paired low/high-SNR data; SelfNet: ≥27-plane stacks) and are **not wired into the
scripted run by default** — the pipeline runs nnU-Net on the raw fill channel. Training scripts
exist for when the data is available:

```bash
<respandev-python>      train_care.py     --epochs 150 --batch_size 16
<respaninternal-python> train_selfnet.py  --images a.tif b.tif --scale 0.32 --epochs 40
```

SelfNet Z-plane rule: `original_Z ≥ ceil(64 × scale)`, where `scale = XY / Z step`.

---

<a name="classes"></a>
## 12. Label class scheme

| value | class | | value | class |
|---|---|---|---|---|
| 0 | background | | 4 | spine neck |
| 1 | dendrite   | | 5 | soma (ignored) |
| 2 | spine core | | 6 | axon (ignored) |
| 3 | spine shell (membrane) | | | |

---

*Interpreter paths, project numbers, and data locations are auto-detected or set in the YAML — the
`<respandev-python>` / `<respaninternal-python>` placeholders above are the full-path Python of
each environment. For the exact copy-paste commands (local and Roihu HPC), see
[`HowToRun.md`](HowToRun.md).*
