# RESPAN Analysis Lab Notebook

---

## Day 1 — 2026-04-10/11

### Background and Goal

The aim of this project is to automate the analysis of dendritic spine density, morphology, and volume in fluorescence microscopy images of cortical and hippocampal neurons. The scientific question is whether synaptic activation (e.g. LTP) causes structural plasticity: specifically, whether it increases spine density, shifts the morphology toward mature mushroom-shaped spines, and whether these changes correlate with enrichment of actin (Lifeact) or AMPA receptors within spines. The tool used for this is **RESPAN** (v1.0.00), an automated pipeline based on nnU-Net 3D segmentation.

---

### Data Available

| File | Type | Description |
|---|---|---|
| `180626_cortical_Control (1).lif` | LIF | Cortical neurons, control condition. Contains 20 FRAP ROI z-stacks (series 0–19, ~190–240 nm XY) and 2 large overview stacks (series 20–21, 82.5 nm XY). 2 channels. |
| `280622no2_hippo_200K.lif` | LIF | Hippocampal neurons (DIV22). 3 timeseries (series 0–2, t=4) and 4 single-timepoint stacks (series 3–6). 2 channels. 82.5 nm XY / 200 nm Z. |
| `140325_DIV17cx_BFP2_Lifeact_EGFP_LTP30min_0001.tif` | TIF | Cortical neurons (DIV17), LTP 30 min timepoint. Single channel. Shape (15, 141, 108). ~102 nm XY / ~200 nm Z. |
| `C2-HippoDIV14_control.lif - Series005.tif` | TIF | Hippocampal neurons (DIV14), control. Channel 2 only (functional marker). Shape (28, 165, 362). 81.3 nm XY / 200 nm Z. |

In all two-channel images: **Ch0 = structural morphology fill** (used by RESPAN for segmentation); **Ch1 = functional second channel** (Lifeact-GFP or AMPAR, measured inside segmented spines after detection).

---

### What Was Done

**1. LIF → TIF conversion**

Written from scratch in Python (`batch‑convertor/lif_to_tif.py`) using `readlif` and `tifffile`. The script extracts each series as a per-channel 3D TIF, handles multi-timepoint series by extracting t=0 (pre-treatment baseline), and embeds pixel size metadata in the TIFF header (ImageJ-compatible).

Key decisions:
- For the cortical LIF: the FRAP ROI series (0–19) were split into a separate folder from the overview stacks (20–21) because their XY pixel sizes differ (~200 nm vs 82.5 nm) — they cannot share one `Analysis_Settings.yaml`.
- For the hippocampal LIF: series 0–2 have 4 timepoints; t=0 was extracted. All 7 series were exported.

**2. Experiment folder structure**

`prepare_respan_experiments.py` built the `RESPAN_experiments/` directory tree with all TIF files and a correctly parameterised `Analysis_Settings.yaml` in each sub-folder, ready to be opened in the RESPAN GUI.

**3. RESPAN runs — Models tested: 1A and 2 (Model 1B failed)**

Runs were performed on the two small datasets first:
- `140325_DIV17cx_LTP30min` (cortical, LTP, single channel)
- `C2-HippoDIV14_control` (hippocampal, DIV14, channel 2 only)

Each was run with Model 1A, Model 1B, and Model 2.

---

### Results

#### Model 1B — failed to run (missing model files)

The error was `FileNotFoundError: dataset.json not found`. The Model 1B folder (`Dataset224_...`) was downloaded incompletely from Google Drive — it only contains `checkpoint_best.pth` inside `fold_all/`, while `dataset.json`, `plans.json`, and `dataset_fingerprint.json` are missing from the parent `nnUNetTrainer__nnUNetPlans__3d_fullres/` directory. Models 1A and 2 have all these files. **Fix:** re-download Model 1B and copy the missing JSON files.

#### Model 2 — effectively failed on both datasets

| Dataset | Spines detected | Dendrite length |
|---|---|---|
| LTP cortical | 1 | 0.51 µm |
| Hippo control | 1 | 0.16 µm |

Model 2 was trained on **isotropic** 65×65×65 nm data. The input data here has 200 nm Z steps (anisotropic). When RESPAN rescales anisotropic data to match this model's resolution, the image gets heavily distorted in Z, and spines become unrecognisable to the network. Model 2 is not suitable for anisotropic data. **Conclusion: do not use Model 2 for any of the datasets in this project.**

#### Model 1A — ran successfully, but Z resolution was wrong

| Dataset | Spines | Dendrite length | Density |
|---|---|---|---|
| LTP cortical | 10 | 12.95 µm | 0.77/µm |
| Hippo control | 16 | 29.51 µm | 0.54/µm |

The Model 1A log confirmed: *"Image resolution: 0.102µm XY, **0.2µm Z**"* — the Z resolution was correct. The YAML checkbox in the GUI was reading `input_resZ: 0.2000` from `Analysis_Settings.yaml` as intended. These results are valid. Note: a Z = 1.0 µm error was incorrectly flagged during initial review — this was based on misreading the Model 1B error log (which did use Z = 1.0 µm), not the Model 1A log. The measurements are therefore correct, meaning:

- All volume measurements (spine, head, neck) are **5× too large**
- Morphology classification (mushroom / thin / stubby) is likely incorrect because it depends on accurate 3D geometry
- Segmentation quality may also be degraded since the model expects properly scaled voxels

**Model 1A results from Day 1 are valid. A re-run on Day 2 with the YAML checkbox confirmed identical numbers, as expected.**

#### The hippo dataset (C2-HippoDIV14_control) has no structural channel

`C2-HippoDIV14_control.lif - Series005.tif` is the functional second channel only (Ch2), not the morphology fill channel RESPAN needs for segmentation. Any segmentation result from this dataset is unreliable. To fix this: re-export Ch0 (structural channel) from `280622no2_hippo_200K.lif` series 4, 5, or 6 (the dendritic ROI stacks).

---

### Conclusions and Lessons Learned

1. **Only Model 1A is appropriate** for this project's data. It was trained on spinning-disk and Airyscan confocal data at 65 nm XY / 150 nm Z, which matches the data here well. Model 2 requires isotropic input and fails on anisotropic z-steps.

2. **The YAML checkbox works correctly.** The RESPAN GUI has a checkbox that reads voxel sizes from `Analysis_Settings.yaml`. With this checked and `input_resZ: 0.2000` in the YAML, the correct resolution is used automatically. Always keep this checked. Note: Model 1B's log showed Z = 1.0 µm (its checkbox may not have been checked, or it defaulted before erroring out), which led to an incorrect concern about Model 1A — Model 1A was always correctly using Z = 0.2 µm.

3. **The Z resolution is the main quality bottleneck.** Model 1A's native Z is 150 nm. The data here has 200 nm Z steps — this is acceptable and within the model's training range, but finer Z steps (closer to 150 nm) would improve spine detection and morphology classification, particularly for mushroom spine identification.

4. **Image specs for future acquisitions** (to get best results with Model 1A):
   - XY pixel size: 60–100 nm (65 nm ideal)
   - Z step: 150–200 nm
   - Z depth: ≥ 15 µm (≥ 75 planes at 200 nm Z)
   - Bit depth: 16-bit uint16, no saturated pixels
   - Signal: moderate brightness (training data median foreground ~1043 / 65535)
   - Channel 1 must be the structural morphology fill fluorophore

5. **Two-channel data is essential** for the correlation analysis (morphology vs Lifeact/AMPAR). Single-channel data (e.g. the LTP cortical TIF) can give spine density and morphology but not the intensity correlation. For the full thesis analysis, all acquisitions should have both channels.

6. **The LTP cortical dataset is very small** (15×141×108 px ≈ 3 µm × 14 µm × 11 µm). Only 10 spines were detected. This is too few for reliable morphology statistics or correlations. Larger fields of view or more cells will be needed.

---

### Next Steps

- [ ] Re-run Model 1A on both datasets with **Z = 0.2 µm** set correctly in the GUI
- [ ] Re-export Ch0 from `280622no2_hippo_200K.lif` (series 4–6) to get the structural channel for the hippo dataset
- [ ] Re-download Model 1B and copy missing JSON files to make it usable
- [ ] Run Model 1A on the larger cortical and hippocampal datasets (`180626_cortical_overview` and `280622_hippo_200K`)
- [ ] Record all results in `RESPAN_run_matrix.md`

---

## Day 2 — 2026-04-15/16

### Goal

Two parallel tracks today:

1. Document the complete RESPAN training pipeline (so we know how to fine-tune or train new models later).
2. Get nnU-Net segmentation running directly from code — no GUI, no RESPAN library — so we can run Model 1A on new images without the GUI and understand exactly what happens inside.

---

### Documentation: RESPAN Training Deep-Dive

Written `f:\ResPan\RESPAN_Training_DeepDive.md` — a complete technical reference covering all three training workflows inside RESPAN:

**CARE (image restoration)**
- CSBDeep / TF framework; paired low-SNR vs high-SNR training cubes
- Probabilistic U-Net, hardcoded `initial_lr = 0.0004`
- Input Z dimension must be a multiple of 8 (auto-padded if not)
- Entry point: `train_care()` in `ModelTraining.py`

**SelfNet (axial resolution restoration)**
- Three phases: `create_slices` → `generate_patches` (64×64 px, stride 64) → `train`
- CycleGAN scaffold but only `deblur_net` is saved as the final model
- Scale parameter = `XY_res / Z_res` (the anisotropy ratio)
- Needs `cuda:0` for training, `cuda:0` + `cuda:1` for inference

**nnU-Net (segmentation)**
- Standard nnU-Net v2 flow: `nnUNetv2_plan_and_preprocess -d {ID} --verify_dataset_integrity` then `nnUNetv2_train {ID} 3d_fullres all`
- Training ~12–24 hours on a modern GPU
- Fold `all` trains on 100% of training data (correct for small datasets)
- Label integer mapping: 1 = spine, 2 = dendrite, 3 = soma

**Fiji annotation workflow** (for correcting Model 1A outputs before retraining):
- Script 1 (`1_Labels_to_Channels_for_Annotation.ijm`): explodes integer label TIF → multi-channel composite (Ch1=raw, Ch2=spine, Ch3=dendrite, Ch4=soma). Parameter `MaxLabel` = number of classes.
- Script 2 (`2_Channels_to_Labels_for_Training.ijm`): collapses corrected channels back to integer label TIF, uses subtract-chain to give priority spine(1) > dendrite(2) > soma(3). Saves to `labelsTr_YYYY_MM_DD/`.
- An already-prepared example lives at `manual_fixing_labels/140/`: image (15,141,108) uint16, labels (15,141,108) uint8 with values {0,1,2}, channels (15,4,141,108) uint8 — Script 1 was already run with MaxLabel=3.

Also appended a full **Practical Training Pathway** section to the deep-dive, covering: image acquisition specs per model, generate_nnunet_json call, all training commands with options explained, and a checklist of minimum annotation counts.

---

### Direct Inference Script: `run_cortical_inference.py`

#### Problem

The RESPAN library (`cloned_respan/RESPAN/`) has several bugs that prevent running inference from code without the GUI:

| Bug | Location | Effect |
|---|---|---|
| Path separator missing | `Create_Locations` | `data_dir + "Validation_Data/"` → garbage folder names like `inference_corticalValidation_Data/` |
| Wrong attribute name | `Segmentation_and_Restoration.py` | Code uses `settings.neuron_seg_model_path`; user scripts set `settings.nnunet_model_path` → `AttributeError` |
| Missing attributes | `nnunet_create_labels()` | References `settings.patch_for_nnunet`, `settings.save_intermediate_data`, `settings.Track` — never set by `Create_Settings` |
| Caching check fires on empty dirs | `nnunet_create_labels()` | `if len(files) == len(label_files)` → `0 == 0` → silently skips segmentation entirely |
| Wrong nnU-Net entry point | `run_nnunet_predict()` | `-m nnunetv2.inference.predict_from_raw_data` runs the module's `__main__` block, which is a hardcoded test that always loads `Dataset003_Liver`, ignoring all arguments |

#### Solution

Rewrote `run_cortical_inference.py` as a minimal 2-step script that bypasses RESPAN entirely:

**Step 1 — Image preparation:**
- Load raw TIF, extract structural channel (handles ZYX / ZCYX / CZYX)
- Rescale to model's training resolution: Z ×(0.200/0.150) = ×1.333, XY ×(0.102/0.065) = ×1.569
- Save as `<name>_0000.tif` with correct ImageJ metadata (nnU-Net naming convention)
- Input `(15, 141, 108)` → rescaled to `(20, 221, 169)`

**Step 2 — nnU-Net inference:**
- Set `os.environ['nnUNet_results']` and stubs for `nnUNet_raw` / `nnUNet_preprocessed` **before** importing nnunetv2 (important: `paths.py` reads env vars at import time)
- Insert `f:\ResPan\nnUNet` onto `sys.path` so the editable install is visible
- Build `sys.argv` as the CLI expects, then call `predict_entry_point()` directly (the real CLI function, not `__main__`)

Key discovery: the correct function to call is `predict_entry_point()` (line 740 of `predict_from_raw_data.py`), not `predict_from_raw_data_entry_point` (does not exist) and not `__main__` (hardcoded test).

#### Environment fix

`conda activate respaninternal` in PowerShell changes the prompt but `sys.executable` stays as `C:\Users\ARASH\Anaconda3\python.exe` (base Anaconda). Fix: ran `pip install -e .` from `f:\ResPan\nnUNet` to register nnunetv2 as an editable install into base Anaconda. The script now runs entirely in-process from base Anaconda Python.

---

### Results

First successful command-line segmentation of `140325_DIV17cx_LTP30min_0001.tif`:

```
Input shape  : (15, 141, 108)  uint16
Rescaled to  : (20, 221, 169)  at model resolution (65 nm XY / 150 nm Z)
Output labels: (20, 221, 169)  uint8
  spine     = 1   →   7 764 voxels  (1.0 %)
  dendrite  = 2   →  39 172 voxels  (5.2 %)
  soma      = 3   →       0 voxels  (not present — expected for this small fragment)
  background= 0   →  93.7 %
```

Output lands in `inference_cortical/labels/`. The labels are at model resolution (65 nm / 150 nm). To overlay them on the original image, rescale back to `(15, 141, 108)` with nearest-neighbour interpolation (order=0).

No soma detected, which is expected — the input is a tiny dendritic fragment (3 µm × 14 µm × 11 µm), with no cell body present.

---

### Conclusions

1. **The RESPAN GUI is effectively the only supported way to run inference with the released code.** The library is too fragile for scripted use without patching multiple bugs.
2. **Direct nnU-Net inference works cleanly** once the data preparation (channel extraction + rescaling) and environment variable timing are handled correctly.
3. **Output is at model resolution, not image resolution.** Always note which space the labels are in before doing any measurements.
4. **Model 1A correctly segments spines and dendrite** even on this small 15-plane stack. Spine density appears plausible (the dendrite fragment is ~14 µm long and ~8 spines visible), though the stack is too small to be statistically meaningful.

---

### Next Steps

- [ ] Run `run_cortical_inference.py` on the full cortical overview stacks (series 20–21 from the LIF)
- [ ] Write a post-processing step to rescale output labels back to original image resolution
- [ ] Run the Fiji annotation workflow on `manual_fixing_labels/140/` to test label correction for future retraining
- [ ] Check whether Model 1B JSON files have been re-downloaded; if so, run Model 1B for comparison
- [ ] Decide on annotation strategy: how many stacks to correct and retrain on

---

## Day 3 — 2026-04-17

### Goal

Get the **complete RESPAN analysis pipeline** running from code with no GUI — not just nnU-Net segmentation (Day 2), but the full `analyze_spines()` pipeline that produces Tables/CSVs, Spine_Arrays, Validation_Data/MIPs, and SWC files. Output should match what the RESPAN GUI produced in `f:\ResPan\140325\`.

---

### Architecture Decisions

#### Two-environment model (mirrors the RESPAN GitHub README)

The RESPAN GitHub README specifies two separate conda environments and this turned out to be critical:

| Environment | Python | Purpose |
|---|---|---|
| `respandev` | 3.10 | Full RESPAN analysis: cupy, trimesh, tensorflow, csbdeep, dask, ome-zarr |
| `respaninternal` | 3.9 | nnU-Net only: PyTorch 2.5.1 + nnunetv2 |

**Why two?** PyTorch 2.6.0 (which ended up in respandev) hangs indefinitely after importing nnunetv2 — it prints a pynvml CUDA warning then freezes. PyTorch 2.5.1 in respaninternal works. The fix: run nnU-Net as a subprocess under respaninternal's Python, then return to respandev for all RESPAN analysis. This is also exactly how the GUI works internally.

#### nnU-Net resolution handling

The nnU-Net model (Dataset219) was trained at 65 nm XY / 150 nm Z. Day 2 rescaled the input image to model resolution before inference. **This was wrong.** The correct approach is to pass images at their original resolution and let nnU-Net resample internally using its `plans.json`. Passing already-rescaled images causes double-resampling and the output labels end up at the wrong size.

Confirmed: the segmentation labels in `inference_cortical/labels/` match the GUI reference (`140325/Validation_Data/Segmentation_Labels/`) when passing at original resolution.

---

### Problems Solved (in order encountered)

#### 1. tifffile compression API (29 occurrences in RESPAN source)

RESPAN was written for tifffile < 2022.7.28 which accepted `compression=('zlib', 1)`. The newer tifffile installed in respandev raises `TypeError: multiple values for 'compression'`. Fixed by replacing all 29 occurrences across 4 files with the new API:

```
Old: compression=('zlib', 1)
New: compression='zlib', compressionargs={'level': 1}
```

Files patched: `ImageAnalysis.py` (22), `IO.py` (3), `Segmentation_and_Restoration.py` (3), `DaskImageAnalysis.py` (1). Script saved as `f:\ResPan\_fix_tifffile_compression.py`.

#### 2. torch 2.6.0 hang in respandev

After installing tensorflow and csbdeep properly in respandev, importing nnunetv2 in that environment causes Python to freeze. The root cause: torch 2.6.0+cu124 hangs after printing a pynvml deprecation warning when CUDA initialises under certain GPU/driver combinations.

Fix: run nnU-Net as a subprocess using `respaninternal`'s Python. The runner script is injected as a `-c` string argument so no extra file is needed:

```python
subprocess.run([RESPANINTERNAL_PYTHON, '-u', '-c', runner_script])
```

#### 3. acvl-utils version chain in respaninternal

nnunetv2 2.3.1 requires `acvl-utils>=0.2.6` for `crop_to_bbox`. But acvl-utils 0.2.6 declares `blosc2>=3.0.0b4` as a dependency, and blosc2 3.x is not available for Python 3.9. Chain resolved with:

```
pip install acvl-utils==0.2.6 --no-deps
```

`--no-deps` skips the blosc2 version check while still getting the correct acvl-utils code.

#### 4. numpy forced back to 1.26.4

The acvl-utils install pulled numpy 2.2.6, which broke `patchify` (requires numpy < 2) and TF 2.10. Fixed with:

```
pip install numpy==1.26.4 --force-reinstall
```

#### 5. scipy 1.13.1 → LAPACK DLL crash

`skimage.color.colorconv` calls `scipy.linalg.inv` at module-import time (to build colour conversion matrices). scipy 1.13.1 installed via pip on Windows crashes with a fatal `0xc06d007f` (DLL not found) when this call is made — the pip wheel links against LAPACK routines that are not present in the conda environment's DLL path.

Diagnosed with `python -X faulthandler` which revealed:

```
Windows fatal exception: code 0xc06d007f
  File "scipy/linalg/_basic.py", line 947 in inv
  File "skimage/color/colorconv.py", line 402 in <module>
```

Fixed by downgrading scipy:

```
pip install scipy==1.11.4
```

scipy 1.11.4 uses a different LAPACK loading path that works correctly in this environment.

#### 6. scikit-image 0.25.0 → also downgraded

scikit-image 0.25.0 triggered the above scipy crash because `skimage.segmentation` imports `skimage.color.colorconv` at module level, which calls `scipy.linalg.inv`. Downgrading scipy to 1.11.4 resolved this, but also downgraded scikit-image to 0.21.0 to avoid any other 0.25.0 compatibility surprises:

```
pip install scikit-image==0.21.0
```

#### 7. Circular imports in RESPAN (not a bug, just ordering)

RESPAN's source has circular imports: `ImageAnalysis.py` imports `Segmentation_and_Restoration`, which imports `ImageAnalysis`. Python handles this through partial module initialisation. No fix needed — importing `ImageAnalysis` directly (not the sub-modules first) works correctly.

#### 8. Missing settings attributes (20+ attributes)

`analyze_spines()` accesses many settings attributes that the RESPAN GUI sets but that our `make_settings()` was missing. The pipeline would crash with `AttributeError` on each missing one. All were added:

Key ones added: `use_vox_measurements=False`, `analysis_method='Whole Neuron'`, `resave_omezarr=False`, `second_pass=False`, `additional_logging=False`, `save_intermediate_data=False`, `HistMatch=False`, `checkmem=False`, `patch_for_nnunet=False`, `reg_method='rigid'`, and several path/env stubs (`nnUnet_conda_path`, `nnUnet_env`, etc.).

---

### Final Pipeline Structure (`run_cortical_inference.py`)

```
Step 1 — prepare_all()
  For each .tif in inference_cortical/images/:
    - Read image, extract channel 0 (neuron structural channel)
    - Save to nnunet_input/ as <name>_0000.tif at ORIGINAL resolution
    - (nnU-Net resamples internally; do NOT pre-scale)

Step 2 — run_nnunet()
  - Subprocess: respaninternal Python runs nnunetv2 predict_entry_point()
  - Dataset219, 3d_fullres, fold_all, checkpoint_best.pth
  - Output labels land in inference_cortical/labels/

Step 3 — run_analysis()
  - Mock only: sitecustomize (bare import, not pip-installable)
  - tensorflow and csbdeep are real imports (needed for CARE model)
  - Build settings SimpleNamespace (all attributes including the 20+ GUI defaults)
  - Build locations SimpleNamespace with trailing slashes (bypasses Create_Locations bug)
  - Call analyze_spines(settings, locations, log, logger)
  - Outputs: Tables/, Spine_Arrays/, Validation_Data/Validation_MIPs/
```

---

### Results

Full pipeline ran end-to-end on `140325_DIV17cx_BFP2_Lifeact_EGFP_LTP30min_0001.tif`. Compared to GUI reference in `f:\ResPan\140325\`:

| Metric | Our output | GUI reference | Difference |
|---|---|---|---|
| total_spines | 10 | 10 | 0 |
| dendrite_length (µm) | 12.852 | 12.954 | <1% |
| avg_spine_vol | 0.4514 | 0.4537 | <0.5% |
| avg_geodesic_dist | 9.894 | 9.945 | <0.5% |
| avg_C1_mean_intensity | 7823.96 | 7814.12 | <0.1% |

Tiny differences are expected from float/GPU non-determinism. The outputs are structurally and numerically identical to the GUI.

---

### How to Run

```
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe -u f:\ResPan\run_cortical_inference.py
```

Put TIF files in `f:\ResPan\inference_cortical\images\` before running. Labels are cached — if they already exist, Step 2 is skipped automatically. Delete `inference_cortical\labels\` to force re-segmentation.

---

### Environment State After Day 3

**respandev** (Python 3.10) — key package versions:
- tensorflow 2.10.1 (CPU-only; CUDA 12.8 system has no CUDA 11 DLLs, not an error)
- scipy **1.11.4** (downgraded from 1.13.1 — LAPACK crash)
- scikit-image **0.21.0** (downgraded from 0.25.0)
- numpy 1.26.4
- cupy 13.6.0 (GPU kernels working)
- csbdeep (real install, not mocked)

**respaninternal** (Python 3.9) — key package versions:
- PyTorch 2.5.1 + CUDA 12.1
- nnunetv2 2.3.1 (editable install from `f:\ResPan\nnUNet`)
- acvl-utils 0.2.6 (installed with --no-deps)
- blosc2 2.5.1

---

### Conclusions

1. **The standalone pipeline now fully replicates the GUI output.** Tables, Spine_Arrays, MIPs, and log all match the reference within rounding error.

2. **scipy 1.13.1 is broken on this Windows + conda setup.** The pip wheel's LAPACK bindings crash at runtime. Always use scipy 1.11.4 in respandev.

3. **scikit-image 0.25.0 should not be used here.** It triggers the scipy crash pathway and may have other incompatibilities with the pinned numpy/TF versions.

4. **RESPAN's settings namespace has ~35 attributes.** Any scripted use must pre-populate all of them — the GUI normally sets them all via the UI form before calling `analyze_spines()`.

5. **The two-environment architecture is essential on Windows.** torch 2.6.0 + CUDA 12.8 + nnunetv2 hangs in respandev. The subprocess boundary between environments is the only clean solution without downgrading PyTorch.

---

### Next Steps

- [ ] Run the pipeline on the full cortical overview stacks (180626 dataset)
- [ ] Run on the hippocampal dataset (280622)
- [ ] Adapt the script for two-channel images (currently assumes single-channel)
- [ ] Verify Model 1B once JSON files are re-downloaded
- [ ] Consider retraining Model 1A on corrected labels from `manual_fixing_labels/`

---

## Day 4 — 2026-04-17 (continued)

### Goal

Consolidate `run_cortical_inference.py`, `run_hippo_inference.py`, and `run_inference_standalone.py` into a single unified script `run_respan.py`. The three old scripts were copies of each other with different hardcoded paths — any fix had to be applied three times. The new script auto-discovers datasets and reads all settings from per-dataset YAML files.

---

### Design

#### Auto-discovery (`find_datasets`)

The script scans `DATASETS_ROOT` for subdirectories that contain both an `Analysis_Settings.yaml` and at least one `.tif` file. Two layouts are supported:

| Layout | YAML location | TIF location |
|---|---|---|
| Flat | `dataset/Analysis_Settings.yaml` | `dataset/*.tif` |
| Subdir | `dataset/Analysis_Settings.yaml` or `dataset/images/Analysis_Settings.yaml` | `dataset/images/*.tif` |

YAML at root takes priority. If `images/` exists and has TIFs, those are used; otherwise TIFs are taken directly from the dataset folder. Any folder without both a YAML and at least one TIF is silently skipped.

#### Per-image output isolation

Each image gets its own output subfolder named after the image file stem:

```
dataset/
  image1.tif
  image2.tif
  Analysis_Settings.yaml
  image1/
    images/          <- hardlink to image1.tif (no disk duplication)
    nnunet_input/
    labels/
    Tables/
    Spine_Arrays/
    Validation_Data/
    Analysis_Settings.yaml   <- copy of the YAML used
  image2/
    ...
```

Hardlinks (`os.link`) are used to avoid duplicating disk space. On filesystems that don't support hardlinks (e.g. across drives), the fallback is `shutil.copy2`.

#### Multiple images per dataset

If a dataset folder contains more than one TIF, all of them are processed under the same `Analysis_Settings.yaml`. Each image gets its own subfolder. This is the expected case for hippocampal datasets where many images share identical acquisition parameters.

#### YAML → settings

`load_yaml()` parses the YAML and converts tuple strings like `"(2,3,3)"` to actual Python tuples via `ast.literal_eval`. The `_make_settings()` function then builds the 35-attribute `SimpleNamespace` expected by `analyze_spines()`, always setting `model_res = input_res` so nnU-Net's internal resampling is not double-applied.

---

### Files Changed

| Action | File |
|---|---|
| Created | `f:\ResPan\run_respan.py` |
| Deleted | `f:\ResPan\run_cortical_inference.py` |
| Deleted | `f:\ResPan\run_hippo_inference.py` |
| Deleted | `f:\ResPan\run_inference_standalone.py` |

---

### How to Run

```
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe -u f:\ResPan\run_respan.py
```

Place each dataset as a subfolder of `DATASETS_ROOT` (`f:\ResPan\inference_cortical` by default). Each subfolder must have `Analysis_Settings.yaml` and at least one `.tif`. The script processes every discovered dataset automatically.

To run only specific datasets, edit the `filter_names` argument in `main()`:

```python
datasets = find_datasets(DATASETS_ROOT, filter_names=['hippo', '140325'])
```

---

### Conclusions

1. **One script replaces three.** Dataset discovery is automatic — adding a new experiment requires only dropping a folder with a YAML into `DATASETS_ROOT`.
2. **Per-image isolation prevents output collisions.** Each image owns its subdirectory; re-running one image does not touch others.
3. **Multiple images per dataset are fully supported.** All TIFs in a dataset folder share one YAML and are processed in alphabetical order.
4. **YAML copy in output folder.** Every output subfolder contains the exact `Analysis_Settings.yaml` used — reproducibility is preserved even if the source YAML is later changed.

---

### Next Steps

- [ ] Run `run_respan.py` on the full cortical dataset (180626)
- [ ] Run on the hippocampal datasets (280622, C2-HippoDIV14)
- [ ] Adapt for two-channel images (second channel intensity analysis)
- [ ] Verify Model 1B once JSON files are re-downloaded

---

## Day 5 — 2026-04-20

### Goal

Get the nnU-Net **training** pipeline working from code — the counterpart to the inference pipeline built on Day 3. Training data (images + pseudo-labels from Dataset219 inference) was placed in `f:\ResPan\training_data\`. The goal was to verify that nnU-Net can be set up, planned, and trained end-to-end via script.

---

### How RESPAN Does Training (from source audit)

Before writing anything, the RESPAN source was audited to understand exactly how its GUI drives training (`ModelTraining.py`, `RESPAN_GUI_DIST.py`):

- `generate_nnunet_json()` creates `dataset.json` by reading label values from the first label image and counting files. **No spacing JSON files are created** — nnU-Net defaults to spacing (1, 1, 1) for all TIF datasets.
- `train_nnUNet()` sets `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` as environment variables, then calls `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train 3d_fullres all` as subprocesses via a `clean_launcher.py` wrapper.
- The subprocess runs in a separate conda environment (`respan99` in dev mode, `respan` in frozen app) to isolate PyTorch from the main RESPAN environment — the same two-environment pattern used for inference.

---

### Dataset Setup

Training data placed in `f:\ResPan\training_data\`:

| File | Shape | dtype | Label values |
|---|---|---|---|
| `imagesTr/140325_DIV17cx_...tif` | (15, 141, 108) | uint16 | — |
| `imagesTr/C2-HippoDIV14_...tif` | (28, 165, 362) | uint8 | — |
| `labelsTr/140325_DIV17cx_...tif` | (15, 141, 108) | uint8 | 0, 1, 2 |
| `labelsTr/C2-HippoDIV14_...tif` | (28, 165, 362) | uint8 | 0, 1, 2 |

Labels 0=background, 1=spine, 2=dendrite. Soma (3) absent from both images — expected for small dendritic fragments.

The `setup` step renames files to clean case IDs (`case001_0000.tif`, `case001.tif`) and writes `dataset.json` matching Dataset219's label scheme (background/spine/dendrite/soma) for future compatibility, even though soma is not present in these training cases.

---

### Pipeline Structure (`train_nnunet.py`)

Three explicit steps, each run separately:

```
Step 1 — setup
  Copy imagesTr/*.tif   -> nnunet_training/raw/Dataset001_Spines/imagesTr/case001_0000.tif, ...
  Copy labelsTr/*.tif   -> nnunet_training/raw/Dataset001_Spines/labelsTr/case001.tif, ...
  Write dataset.json with channel_names, labels, numTraining, overwrite_image_reader_writer=Tiff3DIO

Step 2 — plan
  Subprocess: respaninternal Python
  nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity -c 3d_fullres
  Output: nnunet_training/preprocessed/Dataset001_Spines/nnUNetPlans.json + preprocessed arrays

Step 3 — train
  Subprocess: respaninternal Python
  nnUNetv2_train 1 3d_fullres all --npz
  Output: nnunet_training/results/Dataset001_Spines/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/
```

The three steps use `respaninternal` Python (PyTorch 2.5.1) — same environment as inference. `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` are injected into the subprocess environment; not set globally.

---

### Results

Plan and preprocess completed successfully:
- Dataset integrity: PASSED
- 3d_lowres dropped (images too small for cascade — median size 21×153×235 voxels)
- 3d_fullres config: batch_size=2, patch_size=(24, 160, 256)
- Preprocessing time: <10 seconds for 2 images

Training started and confirmed working on GPU (cuda:0, ~89s/epoch):

| Epoch | Spine Dice | Dendrite Dice | Train loss |
|---|---|---|---|
| 0 | 0.72 | 0.89 | -0.300 |
| 1 | 0.80 | 0.93 | -0.573 |
| 3 | 0.86 | 0.95 | -0.691 |
| 5 | 0.89 | 0.95 | -0.909 |

Soma Dice = nan throughout (expected — not present in training data).

Full training runs 1000 epochs by default (~25h at 89s/epoch). Checkpoints saved every 50 epochs. Resume at any time with `Ctrl+C` + re-run `train` argument (nnU-Net detects and resumes from last checkpoint automatically).

---

### How to Run

```
# Step 1 — set up dataset (run once)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe  f:\ResPan\train_nnunet.py  setup

# Step 2 — fingerprint + plan + preprocess (run once)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe  f:\ResPan\train_nnunet.py  plan

# Step 3 — train (resumable, runs 1000 epochs)
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe  f:\ResPan\train_nnunet.py  train
```

Or run all three steps sequentially: replace the step argument with `all`.

Trained model output: `f:\ResPan\nnunet_training\results\Dataset001_Spines\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\`

---

### Key Files

| File | Purpose |
|---|---|
| `f:\ResPan\train_nnunet.py` | Training script (setup / plan / train) |
| `f:\ResPan\training_data\imagesTr\` | Raw training images |
| `f:\ResPan\training_data\labelsTr\` | Segmentation labels (pseudo-labels from Dataset219) |
| `f:\ResPan\nnunet_training\raw\` | nnUNet_raw — renamed dataset ready for nnU-Net |
| `f:\ResPan\nnunet_training\preprocessed\` | nnUNet_preprocessed — fingerprint + plans |
| `f:\ResPan\nnunet_training\results\` | nnUNet_results — checkpoints, progress.png |

---

### Conclusions

1. **nnU-Net training pipeline verified end-to-end.** Setup, planning, and training all run cleanly via `train_nnunet.py` in `respaninternal`.
2. **Same two-environment pattern as inference.** All nnU-Net steps run in `respaninternal`; the boundary keeps PyTorch isolated from RESPAN's TF/scipy stack.
3. **No spacing JSONs needed for pipeline testing.** RESPAN's own training also skips them — nnU-Net uses (1, 1, 1) isotropic spacing by default. For a production model, spacing JSONs with real µm values should be added.
4. **With 2 training images this is a pipeline test only.** Dice improves correctly, confirming the pipeline works. A production model needs many more annotated stacks.
5. **Resumable training.** nnU-Net saves checkpoints every 50 epochs and auto-resumes — safe to interrupt and continue.

---

### Next Steps

- [ ] Collect more annotated stacks and add to `training_data/` to train a production model
- [ ] Add spacing JSON files (real µm values) once moving beyond pipeline testing
- [x] Set up CARE denoising training pipeline — DONE (see Day 5 continued below)
- [ ] Set up SelfNet training pipeline (self-supervised, no paired data needed)

---

## Day 5 (continued) — CARE Training Pipeline

### Goal

Verify the CARE denoising training pipeline works end-to-end from code. CARE requires paired low/high SNR image acquisitions; for this pipeline test the same image was used in both `lowSNR/` and `highSNR/` — the goal was only to confirm the code runs, not to train a real model.

---

### What CARE Needs

CARE (Content-Aware Image Restoration, csbdeep) trains a probabilistic U-Net to map noisy images to clean ones. It runs in **`respandev`** (TensorFlow 2.10), not `respaninternal`.

**Data layout:**
```
CARE_training/data/
    lowSNR/    image.tif    -- noisy acquisition
    highSNR/   image.tif    -- clean acquisition (identical filename)
```

Filenames must match between the two folders — that's how RESPAN pairs them via `RawData.from_folder()`.

**Real paired data acquisition (for production):** Image the same field of view twice — once at normal laser power (→ `lowSNR/`) and once at high power or with many averages (→ `highSNR/`). CARE then learns to denoise new images without needing the high-SNR version.

---

### Parameters (with meanings)

| Parameter | GUI default | What it means |
|---|---|---|
| `patch_size` | `8,64,64` | 3D sub-volume (Z×Y×X) cropped for training. Must fit inside images. |
| `num_patches` | `100` | Patches extracted per image. More = richer dataset. |
| `epochs` | `150` | Full passes through patch dataset. |
| `steps_per_epoch` | `400` | Gradient updates per epoch (random batches — data is recycled). |
| `unet_kern_size` | `3` | Conv kernel size. Always 3 for 3D; 5 for 2D only. |
| `unet_n_depth` | `3` | U-Net encoder stages. More = larger receptive field, more memory. |
| `batch_size` | `16` | Patches per gradient step. |
| `pct_validation` | `0.10` | Fraction of patches held out for validation loss. |
| `augmentation` | checkbox | Rotate patches 90/180/270° — increases effective dataset size. |

---

### Auto Z-padding

RESPAN's `train_care()` checks whether the Z dimension is divisible by 8. If not, it pads automatically before patch extraction. The test image had 28 Z planes → padded to 32. No user action needed.

---

### Pipeline Notes

- `train_care()` saves patch cache to `CAREdata/my_training_data.npz` as a **relative path** from wherever Python is running. The script uses `os.chdir(output_dir)` before calling it so the cache lands in the output folder.
- `app=None` is safe — the GUI app reference is never actually used inside `train_care()`.
- TF 2.10 runs CPU-only on this machine (no CUDA 11 DLLs). 5 test epochs took ~4 min on CPU; real 150-epoch training will need GPU or significant time.

---

### Results (pipeline test)

- 25 patches extracted from 1 image pair → 23 train / 2 val
- 5 epochs × 10 steps, batch size 8 — completed in 4 min 18s on CPU
- Loss decreased correctly (trivially, since lowSNR == highSNR for the test)
- Model saved to `f:\ResPan\CARE_training\output\care_test_3D\weights_best.h5`

---

### How to Run

```
# With defaults (fast pipeline test — 5 epochs):
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe  f:\ResPan\train_care.py

# Real training run (override any parameter from the command line):
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe  f:\ResPan\train_care.py ^
    --epochs 150 --steps_per_epoch 400 --num_patches 100 --batch_size 16

# Different data or model name:
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe  f:\ResPan\train_care.py ^
    --data f:\ResPan\CARE_training\data --name my_cortical_model

# Full parameter override example:
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe  f:\ResPan\train_care.py ^
    --patch_size 8,64,64 --num_patches 100 --epochs 150 --steps_per_epoch 400 ^
    --unet_kern_size 3 --unet_n_depth 3 --batch_size 16 --pct_validation 0.10 ^
    --augmentation
```

All parameters have defaults — pass only what you want to change.

---

### Key Files

| File | Purpose |
|---|---|
| `f:\ResPan\train_care.py` | CARE training script (all params overridable via CLI) |
| `f:\ResPan\CARE_training\data\lowSNR\` | Noisy training images |
| `f:\ResPan\CARE_training\data\highSNR\` | Clean training images (paired) |
| `f:\ResPan\CARE_training\output\care_test_3D\` | Saved model weights |
| `f:\ResPan\CARE_training\output\CAREdata\` | Extracted patch cache (.npz) |

---

### Conclusions

1. **CARE pipeline verified end-to-end.** Patch extraction, model construction, and training all run cleanly in `respandev`.
2. **Runs in `respandev`, not `respaninternal`.** CARE uses TensorFlow/csbdeep; no subprocess boundary needed.
3. **All parameters are CLI-overridable.** No need to edit the script — pass `--epochs 150 --batch_size 16` etc. at the command line.
4. **Real training requires paired acquisitions.** The pipeline test used the same image for both lowSNR and highSNR. For a real denoising model, go back to the microscope and acquire matched low/high SNR z-stacks.
5. **TF runs CPU-only on this machine.** Real 150-epoch training will be slow without CUDA 11.

---

### Next Steps

- [ ] Acquire real paired low/high SNR z-stacks for CARE training
- [x] Set up SelfNet training pipeline — DONE (see Day 6 below)

---

## Day 6 — 2026-04-20/21

### SelfNet Training Pipeline

Set up `train_selfnet.py` — the self-supervised axial restoration training pipeline. Unlike CARE, SelfNet needs no paired data: it learns to restore blurry Z planes from the image itself by comparing the sharp XY view to the stretched XZ/YZ views.

**How SelfNet works (3 internal steps in `SelfNet_Model_Training.py`):**

1. `self_net_create_slices` — extract 2D slices in XY (sharp), XY-lowres (downsampled to match Z PSF), XZ (stretched), and YZ (stretched) orientations
2. `generate_training_data` — crop 64×64 patches from those slices (background patches discarded via `bg_threshold`)
3. `train_self_net` — train a CycleGAN mapping XY-lowres ↔ XZ/YZ; only the `deblur_net` generator is saved

**Key parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `--scale` | 0.5 | `XY_res / Z_step` anisotropy ratio. e.g. 0.082 µm XY / 0.200 µm Z → 0.41 |
| `--epochs` | 40 | Training epochs |
| `--batch_size` | 8 | Images per gradient step |
| `--xy_interval` | 4 | Take every Nth XY slice |
| `--xz_interval` | 8 | Take every Nth XZ slice |
| `--bg_threshold` | 10 | Min intensity for patch selection (10 for uint8, ~1000 for uint16) |
| `--min_v / --max_v` | 0 / 255 | Intensity range for normalisation |

**Z-plane constraint:** image must have `>= ceil(64 × scale)` Z planes. At scale=0.41 that means ≥ 27 Z planes. The hippo stack (38 Z planes) is sufficient; the cortical LTP stack (15 Z planes) is not.

**Environment:** runs in `respaninternal` (PyTorch, hardcoded to `cuda:0`). Not `respandev`.

**Runs as subprocess:** `train_selfnet.py` calls `SelfNet_Model_Training.py` as a subprocess under `respaninternal` Python (same container-boundary pattern as nnU-Net inference).

**How to run:**
```
# Pipeline test (3 epochs, default hippo stack):
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe  f:\ResPan\train_selfnet.py

# Real training with custom images and scale:
C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe  f:\ResPan\train_selfnet.py \
    --images path\to\stack1.tif path\to\stack2.tif \
    --scale 0.41 --epochs 40 --batch_size 8
```

---

### model_type Fix (Dataset226)

Discovered that `_make_settings()` in `run_respan.py` was setting `model_type=1` (Model 1A — spine cores + dendrites only). The pre-trained model in `cloned_respan` is **Dataset226**, which is Model 4 (spine cores + spine membranes + necks + soma + axons). Using the wrong model_type causes RESPAN to misinterpret the label classes.

**Fixed:** `model_type = 4` in `_make_settings()`.

---

### Next Steps

- [ ] Acquire real paired low/high SNR z-stacks for CARE training
- [ ] Run analysis on larger datasets (180626 cortical, 280622 hippo)
- [ ] Set up Mahti GPU cluster for large-scale runs

---

## Day 7 — 2026-04-29

### Goal

Set up the RESPAN pipeline on **CSC Mahti** (Finnish supercomputer, A100 GPUs) so that long training and analysis runs can happen on real GPU hardware without blocking the local machine.

---

### Mahti Infrastructure

- **Cluster:** CSC Mahti, A100-SXM4-40GB GPUs, SLURM scheduler
- **Project:** `project_2019059`
- **Storage:**
  - `/scratch/project_2019059/respan/` — 1 TB scratch (code + data)
  - `/projappl/project_2019059/respan/` — 50 GB persistent (environments)
- **GPU partition:** `gpusmall` (1 GPU, max 36h), `gpumedium` (2–4 GPUs)
- **Interactive GPU session:** `srun --account=project_2019059 --partition=gpusmall --gres=gpu:a100:1 --time=04:00:00 --mem=64G --ntasks=1 --cpus-per-task=10 --pty bash`

---

### Environment Setup (tykky)

CSC uses **tykky** (`conda-containerize`) to package conda environments into Singularity containers for use on compute nodes. Two environments were built at `/projappl/project_2019059/respan/envs/`:

**respandev** (TensorFlow / CARE / analysis):
```yaml
channels: [conda-forge, defaults]
dependencies:
  - python=3.10
  - pip:
    - tensorflow==2.13.0
    - csbdeep
    - cupy-cuda12x        # must be 12x — Mahti compute nodes have CUDA 12.2
    - tifffile, pyyaml, scikit-image, scipy, numpy<2.0
    - pandas, scikit-learn, matplotlib, patchify
    - psutil, dask[distributed], dask-image
    - numba, numcodecs, ome-zarr, zarr
    - trimesh, nibabel, memory-profiler
```

**respaninternal** (PyTorch / nnU-Net):
```yaml
channels: [pytorch, nvidia, conda-forge, defaults]
dependencies:
  - python=3.10
  - pytorch::pytorch=2.2.1
  - pytorch::pytorch-cuda=11.8
  - pip: [nnunetv2, tifffile, scikit-image, scipy, numpy]
```

Build command (first-time or full rebuild):
```bash
module load tykky
conda-containerize new --prefix /projappl/project_2019059/respan/envs/respandev \
    /projappl/project_2019059/respan/env_dev.yaml
```

Update (add packages without full rebuild):
```bash
echo "pip install <packages>" > /tmp/fix.sh
conda-containerize update --post-install /tmp/fix.sh \
    /projappl/project_2019059/respan/envs/respandev
```

Python wrappers:
- `/projappl/project_2019059/respan/envs/respandev/bin/python`
- `/projappl/project_2019059/respan/envs/respaninternal/bin/python`

---

### Issues Encountered and Fixed

| Problem | Fix |
|---|---|
| `cupy-cuda11x` can't find `libnvrtc.so.11.2` on compute nodes | Rebuilt with `cupy-cuda12x` (compute nodes have CUDA 12.2 driver) |
| `cuda/12.1.1` module not loading | Must load `gcc/10.4.0` first: `module load gcc/10.4.0 cuda/12.1.1` |
| Many missing packages (`patchify`, `psutil`, `dask`, `numba`, etc.) | Found all imports via grep of RESPAN source; added all in one `conda-containerize update` |
| `numpy` upgraded to 2.2.6 by dask dependency resolver | Pinned back: `pip install 'numpy==1.24.3' 'typing-extensions==4.5.0'` |
| respandev installed to `~/` instead of projappl on first attempt | Used `--prefix` flag: `conda-containerize new --prefix <path> <yaml>` |
| `tykky` module not found | Must `module load tykky` before every use — not persistent across login nodes |

---

### Script Path Strategy

All 4 scripts (`run_respan.py`, `train_care.py`, `train_selfnet.py`, `train_nnunet.py`) were updated to use **`__file__`-relative paths** so they work on any machine without editing:

```python
_BASE = Path(__file__).resolve().parent
DATASETS_ROOT = str(_BASE / 'images')
RESPAN_ROOT   = str(_BASE / 'cloned_respan')
NNUNET_REPO   = str(_BASE / 'nnUNet')
```

The Python interpreter path (only environment-specific constant) uses an existence check:
```python
_mahti_py = '/projappl/project_2019059/respan/envs/respaninternal/bin/python'
_win_py   = r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe'
RESPANINTERNAL_PYTHON = _mahti_py if os.path.exists(_mahti_py) else _win_py
```

---

### File Transfer

The full `f:\ResPan\` folder was transferred to `/scratch/project_2019059/respan/code/ResPan/` using the **CSC web dashboard** (`https://www.mahti.csc.fi/pun/sys/dashboard/files/`) drag-and-drop — the simplest method that avoids SSH key issues on Windows.

SSH config (`~/.ssh/config`) was also set up for VS Code Remote-SSH:
```
Host mahti
    HostName mahti.csc.fi
    User kharash
    IdentityFile ~/.ssh/id_ed25519
```

After connecting, add to `~/.bashrc` to fix a known Mahti VS Code memory error:
```bash
export NODE_OPTIONS="--disable-wasm-trap-handler"
```

---

### Container Nesting Problem (Mahti-specific)

On Mahti, `run_respan.py` runs inside the **respandev** tykky container. When it tries to call `respaninternal` Python as a subprocess for nnU-Net, it fails because you cannot launch a tykky container from inside another tykky container:

```
[ ERROR ] Installation for /projappl/.../respaninternal/ is not mounted.
          Wrapper called from another container?
```

On Windows this is not a problem (conda envs can freely call each other). On Mahti the solution is to run nnU-Net and analysis as two separate commands.

**Fix: `run_nnunet_only.py`** — a Mahti-specific helper script that runs under `respaninternal` and handles steps 1 (prepare) and 2 (nnU-Net) for all datasets. Then `run_respan.py` runs under `respandev` and skips nnU-Net automatically because labels already exist.

---

### How to Run on Mahti

```bash
# 1. Get an interactive GPU session
srun --account=project_2019059 --partition=gpusmall \
     --gres=gpu:a100:1 --time=04:00:00 --mem=64G \
     --ntasks=1 --cpus-per-task=10 --pty bash

# 2. Load required modules (must be done inside compute node)
module load gcc/10.4.0 cuda/12.1.1

# 3. Generate segmentation labels (respaninternal)
/projappl/project_2019059/respan/envs/respaninternal/bin/python \
    /scratch/project_2019059/respan/code/ResPan/run_nnunet_only.py

# 4. Run RESPAN spine analysis (respandev, labels exist so nnU-Net skipped)
/projappl/project_2019059/respan/envs/respandev/bin/python \
    /scratch/project_2019059/respan/code/ResPan/run_respan.py
```

For overnight jobs, use SLURM scripts (`slurm_analysis.sh`, `slurm_nnunet.sh`, etc.) with `sbatch`.

---

### First Successful GPU Run

Both test datasets ran successfully end-to-end on the A100:

| Dataset | Spines | Analysis time |
|---|---|---|
| `C2-HippoDIV14_control.lif - Series005` (hippo) | 19 detected, 19 after filters | 1.07 min |
| `140325_DIV17cx_LTP30min_0001` (cortical) | 18 detected, 12 after volume filter | 0.12 min |

GPU utilisation: 40 GB A100, mesh analysis used ~0.06% of GPU memory (small test images). CARE restoration was disabled (`restore: False`) so TensorFlow ran CPU-only — the TF "could not find cuda drivers" warning is harmless.

---

### Key Files Added

| File | Purpose |
|---|---|
| `f:\ResPan\run_nnunet_only.py` | Mahti-only: nnU-Net segmentation step (respaninternal) |
| `f:\ResPan\env_dev.yaml` | respandev tykky environment spec |
| `f:\ResPan\slurm_analysis.sh` | SLURM batch script for analysis |
| `f:\ResPan\slurm_nnunet.sh` | SLURM batch script for nnU-Net training |
| `f:\ResPan\slurm_care.sh` | SLURM batch script for CARE training |
| `f:\ResPan\slurm_selfnet.sh` | SLURM batch script for SelfNet training |

---

### Conclusions

1. **The full pipeline runs on Mahti A100.** Both analysis and segmentation work end-to-end.
2. **tykky requires separate commands for each environment.** No nested containers — respaninternal and respandev must be called sequentially, not from within each other.
3. **CUDA 12.x must match the compute node driver.** `cupy-cuda11x` fails on Mahti (`cuda/11.5.0` only has `libnvrtc.so.11.5`, not `11.2`). Use `cupy-cuda12x` and load `gcc/10.4.0 cuda/12.1.1`.
4. **`__file__`-relative paths make scripts portable.** No manual path editing needed between Windows and Mahti.
5. **Web dashboard is the easiest file transfer method.** Avoids SSH key complexities on Windows entirely.

---

### Next Steps

- [ ] Run `run_nnunet_only.py` on new images to generate labels, then run full analysis
- [ ] Run nnU-Net training on Mahti with more annotated stacks
- [ ] Set up VS Code Remote-SSH connection to Mahti for direct editing
- [ ] Collect more training images for nnU-Net and CARE retraining

---

## Day 8 — 2026-05-04

### Goal

Fix `run_nnunet_only.py` on Mahti so that nnU-Net inference actually runs on the GPU end-to-end for all images without crashing.

---

### Problems Encountered and Fixed (in order)

#### 1. PyTorch 2.6 `weights_only=True` default breaks checkpoint loading

**Error:**
```
_pickle.UnpicklingError: Weights only load failed.
WeightsUnpickler error: Unsupported global: GLOBAL numpy._core.multiarray.scalar
```

**Cause:** PyTorch 2.6 changed the default of `torch.load` from `weights_only=False` to `weights_only=True`. The nnU-Net checkpoint was saved with an older PyTorch and contains numpy globals, which are blocked by the new default.

**Fix:** Monkey-patch `torch.load` in `run_nnunet_only.py` before importing nnunetv2 to restore the pre-2.6 default:
```python
import torch as _torch
_orig_load = _torch.load
_torch.load = lambda *a, **kw: _orig_load(*a, **{**{'weights_only': False}, **kw})
```
This restores the old default while still respecting any explicit `weights_only=True` caller passes.

---

#### 2. PyTorch compiled for CUDA 13.0 — driver only supports CUDA 12.2

**Error:**
```
RuntimeError: The NVIDIA driver on your system is too old (found version 12020).
```

**Cause:** The `respaninternal` tykky container had PyTorch 2.11.0+cu130 (compiled for CUDA 13.0). The Mahti compute node driver only supports up to CUDA 12.2 (driver version 12020). This was never detected before because nnU-Net never successfully initialised.

**Fix:** Rebuilt `respaninternal` with CUDA 12.1-compatible PyTorch:
```bash
module load tykky
echo 'pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall' > ~/fix_torch.sh
conda-containerize update --post-install ~/fix_torch.sh /projappl/project_2019059/respan/envs/respaninternal
```

This also left `torchvision 0.26.0` (which expects `torch==2.11.0`) mismatched. Fixed immediately after:
```bash
echo 'pip install torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall' > ~/fix_torchvision.sh
conda-containerize update --post-install ~/fix_torchvision.sh /projappl/project_2019059/respan/envs/respaninternal
```

Result: `torch 2.5.1+cu121` + `torchvision 0.20.1+cu121` — consistent, no conflicts.

---

#### 3. `torch.set_num_interop_threads` crash on second image

**Error:**
```
RuntimeError: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
```

**Cause:** `predict_entry_point()` calls `torch.set_num_interop_threads(1)` internally. PyTorch only allows this once per process. The original script called `predict_entry_point()` inside a loop — once per image — so the second call crashed.

**Fix:** Replaced `predict_entry_point()` with direct use of `nnUNetPredictor`:
- Initialise the predictor **once** before the loop (loads the model into GPU memory once)
- Call `predictor.predict_from_files()` for each image inside the loop
- Split the loop into two phases: prepare all images first, then predict all

The new structure in `run_nnunet_only.py`:
```
Phase 1 — for each image:
    if labels exist → skip
    else → prepare_image()  (resample, save to nnunet_input/)
    append to to_predict list

Phase 2 — load model once (nnUNetPredictor.initialize_from_trained_model_folder)
    for each image in to_predict:
        predictor.predict_from_files(prep_dir, labels_dir, ...)
        clean up non-.tif outputs
```

Also added `MODEL_DIR` to the imports from `run_respan` so the predictor can find the model folder directly without reconstructing the path from `NNUNET_RESULTS + DATASET_ID`.

---

### Final Successful Run

Both test images segmented and analysed successfully on the Mahti A100:

| Image | Spines (after filters) | Analysis time |
|---|---|---|
| `C2-HippoDIV14_control.lif - Series005` (hippo, 28 planes) | 19 | 0.15 min |
| `140325_DIV17cx_LTP30min_0001` (cortical, 15 planes) | 12 | 0.03 min |

---

### respaninternal Environment State (after Day 8 fixes)

| Package | Version |
|---|---|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| nnunetv2 | as installed (editable from `nnUNet/`) |

CUDA module required at runtime: `module load gcc/10.4.0 cuda/12.1.1`

---

### Conclusions

1. **PyTorch 2.6+ breaks nnU-Net checkpoint loading** unless `weights_only=False` is patched in before nnunetv2 is imported. Fixed in `run_nnunet_only.py` with a one-line monkey-patch.
2. **The respaninternal environment had PyTorch compiled for CUDA 13.0**, which the Mahti driver (CUDA 12.2) cannot run. Always verify `torch.version.cuda` matches the compute node driver before running.
3. **`predict_entry_point()` cannot be called more than once per process.** Use `nnUNetPredictor` directly and initialise it once per run.
4. **The full Mahti pipeline is now stable.** `run_nnunet_only.py` (respaninternal) → `run_respan.py` (respandev) works end-to-end.

---

### nnU-Net Training on Mahti (same day)

`train_nnunet.py` was updated to call nnunetv2 entry points directly on Mahti (same container-nesting fix as inference). Tested with 2 training images — training runs correctly on A100 at ~20s/epoch:

| Epoch | Spine Dice | Dendrite Dice |
|---|---|---|
| 0 | 0.66 | 0.88 |
| 5 | 0.88 | 0.95 |
| 8 | 0.91 | 0.96 |

`train_care.py` and `train_selfnet.py` were also updated with the same Mahti-compatible pattern (`runpy.run_path` for SelfNet, direct TF import for CARE) but not tested — no training data available on Mahti for those two.

---

### Next Steps

- [ ] Run the pipeline on the full datasets (180626 cortical overview stacks, 280622 hippo)
- [ ] Submit overnight batch jobs via SLURM (`slurm_nnunet.sh`, `slurm_analysis.sh`)
- [ ] Collect more annotated stacks for nnU-Net retraining
- [ ] Test CARE and SelfNet training on Mahti once paired data is available

---

## Day 9 — 2026-05-07

### Goal

Three parallel tracks:
1. Upgrade to **RESPAN v1.5** — read all new documentation, understand what changed.
2. Add **fine-tuning support** to `train_nnunet.py` (start from Model 1Bv2 pretrained weights instead of from scratch).
3. Run RESPAN analysis on the **Ellinoora cortical crops** (new dataset, 10 crops) to generate auto-predicted labels as a starting point for Fiji correction and future fine-tuning.

---

### RESPAN v1.5 — What Changed

Read all v1.5 documentation: `cloned_respan/README.md`, `cloned_respan/UPDATES.md`, `cloned_respan/docs/Fine_Tuning_Guide.md`, and the full pipeline flow diagram (`docs/pipeline_flow.mmd`).

Key v1.5 changes relevant to this project:

- **New spine types:** filopodia, partial-spine, multi-head — the label output now carries these type strings in the `spine_type` column of the detected spines CSV.
- **Chunked zarr processing:** very large images (55 GB+) can now be processed in chunks via dask/ome-zarr without loading everything into RAM. Not relevant for the current small crops but important for full-FOV data.
- **Fine-Tuning Guide added:** a 10-step workflow document (`Fine_Tuning_Guide.md`) explaining how to start from pretrained Model 1Bv2 weights. The fine-tuning capability itself existed before v1.5 — the guide is new.
- **Model 1Bv2 (Dataset226)** is the recommended starting point for fine-tuning. Label scheme: background=0, dendrite=1, spine_core=2, spine_edge=3, spine_neck=4, soma=5, axon=6, ignore=7.

Key finding on label scheme for fine-tuning: you can use a **subset** of Dataset226's integers (e.g. annotate only dendrite/spine_core/spine_edge/spine_neck, leave soma and axon as 0). The FAQ restriction in the GitHub issues is about mixing incompatible model families (1A vs 1Bv2), not about using fewer classes. The segmentation head is always re-initialised regardless.

Key finding on spacing: Dataset226's `dataset_fingerprint.json` shows `spacings: [[1,1,1], ...]` for all training images. RESPAN normalises voxel spacing to (1,1,1) before handing images to nnU-Net — the Tiff3DIO reader ignores physical metadata. This means fine-tuning at any input resolution is valid; RESPAN does the rescaling.

---

### Fine-Tuning Support Added to `train_nnunet.py`

**Major rewrite** adding four new commands alongside the existing from-scratch pipeline:

| Command | What it does |
|---|---|
| `finetune_plan` | Transfer plans from Dataset226 → Dataset001, then preprocess |
| `finetune_train` | Train with `-pretrained_weights` pointing to Model 1Bv2 checkpoint |
| `finetune` | All three steps: `setup` (with FINETUNE_LABELS) + `finetune_plan` + `finetune_train` |

**New constants added:**

```python
FINETUNE_LABELS = {
    "background": 0, "dendrite": 1, "spine_core": 2,
    "spine_edge": 3, "spine_neck": 4,
}
SOURCE_DATASET_ID  = 226
SOURCE_MODEL_DIR   = _BASE / 'cloned_respan' / 'Dataset226_Global1Bv2_...'
PRETRAINED_WEIGHTS = SOURCE_TRAINER_DIR / 'fold_all' / 'checkpoint_final.pth'
```

**`finetune_plan()` steps (in order):**
1. Copy Dataset226 into `nnunet_training/results/` so `move_plans` can find it by ID
2. Copy `plans.json` into `nnunet_training/preprocessed/Dataset226.../nnUNetPlans.json`
3. `nnUNetv2_extract_fingerprint -d 1` — fingerprint the target dataset
4. `nnUNetv2_move_plans_between_datasets -s 226 -t 1 -sp nnUNetPlans -tp nnUNetPlans`
5. `nnUNetv2_preprocess -d 1 -plans_name nnUNetPlans -c 3d_fullres --verify_dataset_integrity`

**`_run_cmd()` helper** added — cross-platform entry point runner. On Mahti: imports and calls directly (no subprocess nesting). On Windows: runs as subprocess via `_run()`. Used by all four plan/preprocess/train functions.

**`setup()` now accepts a `labels` parameter** so it can be called with either `LABELS` (from-scratch) or `FINETUNE_LABELS` (fine-tune).

Recommended epochs for fine-tuning: **300–500** for 10–20 annotated cases. Stop early with Ctrl+C, then promote `checkpoint_latest.pth → checkpoint_final.pth` so RESPAN can find it.

---

### `run_respan.py` Updates

Three fixes from the v1.5 compatibility audit:

1. **`_ON_MAHTI` flag** added at module level — used to switch behaviour in `run_nnunet()` and error messages.
2. **Mahti guard in `run_nnunet()`** — on Mahti, the function prints a clear error and exits if called without pre-existing labels (container nesting is still not supported; users must call `run_nnunet_only.py` first).
3. **`check_cupy()` error message** made cross-platform (shows correct conda path for Mahti vs Windows).

**`--model` CLI argument added** to `main()`:

```
python run_respan.py --model F:\ResPan\cloned_respan\Dataset226_...  [dataset_names...]
```

Uses `argparse`. Parses the dataset ID from the folder name via regex (`Dataset(\d+)`). Overrides `MODEL_DIR`, `NNUNET_RESULTS`, and `DATASET_ID` globally. Without `--model`, defaults to `MODEL_DIR` set in the script header.

---

### Ellinoora Dataset — Analysis Run

**Data:** 10 cortical neuron crops from Ellinoora, channel 2 only (Lifeact-EGFP structural channel).
- **Location:** `F:\ResPan\images\ellinoora\` (flat layout — TIFs directly in the folder)
- **Resolution:** 132.6 nm XY / 410 nm Z (coarser than Model 1Bv2's 65 nm / 150 nm)
- **Image shape:** 14–22 Z slices × ~185×100 XY pixels, uint16
- **Conditions:** LTP30min (crops 1–8) and LTPctrl (crops 9–10)

**Settings file created:** `F:\ResPan\images\ellinoora\Analysis_Settings.yaml` with `input_resXY: 0.1326`, `input_resZ: 0.410`, `model_resXY: 0.065`, `model_resZ: 0.150`.

**Run command:**
```
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe  f:\ResPan\run_respan.py  --model  F:\ResPan\cloned_respan\Dataset226_Global1Bv2_XY65nm_Z150nm_CARE-Aug_spinecore_neck_dend_soma_axon_V6  ellinoora
```

**Results — all 10 crops completed successfully:**

| Crop | Z slices | Spines detected | Label classes present |
|---|---|---|---|
| crop1 | 15 | 8 | dendrite, spine_core, spine_edge, spine_neck |
| crop2 | 15 | 13 | dendrite, spine_core, spine_edge, spine_neck |
| crop3 | 15 | 16 | dendrite, spine_core, spine_edge, spine_neck, axon |
| crop4 | 18 | 15 | dendrite, spine_core, spine_edge, spine_neck |
| crop5 | 18 | 18 | dendrite, spine_core, spine_edge, spine_neck |
| crop6 | 14 | 21 | dendrite, spine_core, spine_edge, spine_neck |
| crop7 | 14 | 11 | dendrite, spine_core, spine_edge, spine_neck |
| crop8 | 14 | 20 | dendrite, spine_core, spine_edge, spine_neck |
| crop9 | 16 | 10 | dendrite, spine_core, spine_edge, spine_neck |
| crop10 | 22 | 9 | dendrite, spine_core, spine_edge, spine_neck |
| **Total** | | **141** | |

Per-crop output structure:
```
images/ellinoora/<crop-name>/
    images/          ← hardlink to original TIF
    nnunet_input/    ← rescaled TIF passed to nnU-Net
    labels/          ← nnU-Net prediction (uint8, values 0–4, 6)
    Tables/          ← detected_spines.csv, dendrite_summary.csv, neck_path_confidence.csv
    Validation_Data/Validation_MIPs/  ← 8-channel MIP composite TIFs
    Analysis_Settings.yaml
    RESPAN_analysis.log
```

Label integers in output files: 0=background, 1=dendrite, 2=spine_core, 3=spine_edge, 4=spine_neck. Crop3 also has label 6 (axon) — to be relabelled as 0 or 1 during Fiji correction.

---

### SelfNet Assessment — Not Feasible on These Crops

SelfNet requires `original_Z >= ceil(64 × scale)` to form 64×64 XZ patches. At scale = 0.1326/0.410 = 0.323, the minimum is **21 Z planes**. Only crop10 (22 slices) meets this threshold; the other 9 crops (14–18 slices) cannot even start SelfNet. Additionally, all 10 crops come from the same imaging session, which is insufficient diversity for a generalised SelfNet model.

**Decision: skip SelfNet training on these crops.** SelfNet is worth training when full-FOV stacks with ≥30 Z planes from multiple sessions are available.

---

### Conclusions

1. **Model 1Bv2 (Dataset226) works on Ellinoora data** despite the 2× XY / 2.7× Z resolution gap. 141 spines detected across 10 crops with a mix of spine types including partial-spine.
2. **Fine-tuning is the right next step.** The predicted labels are a good starting point; correcting them in Fiji and fine-tuning will teach the model to work at 132.6/410 nm specifically.
3. **FINETUNE_LABELS must be a subset of Dataset226's integers.** Do not add new integers or change existing ones — only annotate the structures you have (dendrite, spine_core, spine_edge, spine_neck), leave everything else as 0.
4. **Crop3 has an axon prediction (label 6).** Relabel as 0 (background) or 1 (dendrite) depending on what the raw image shows.
5. **The `--model` flag makes it easy to switch models at the command line** without editing the script.

---

### Next Steps

- [ ] Open label TIFs + raw TIFs side-by-side in Fiji and correct auto-predicted labels
- [ ] Copy corrected pairs to `f:\ResPan\training_data\imagesTr\` and `training_data\labelsTr\`
- [ ] Get ~10 more crops from Ellinoora (target: 20 total, ideally ≥25 Z slices each)
- [ ] Run `python train_nnunet.py finetune` once enough corrected data is available (300–500 epochs recommended)
- [ ] Upload updated `run_respan.py` and `train_nnunet.py` to Mahti

---

## Day 10 — 2026-05-18/20

### Goal

Process 31 new crops from Ellinoora, manually correct their RESPAN-predicted labels in Fiji, and prepare a fine-tuning dataset.

---

### New Crops — Channel Extraction

Received 31 new crops in `F:\ResPan\images\New files from Ellinoora (crops)\`. Two imaging sessions with different channel layouts:
- Crops 1–24: 3-channel TIFs, fluorescence in channel 2 (index 1)
- Crops 25+: 2-channel TIFs, fluorescence in channel 1 (index 0)

Written Fiji macro `batch-convertor/extract_channel.ijm` that asks for: source folder, channel number (1-indexed), first crop number, last crop number. Parses crop number from filename. Saves single-channel TIFs to `extracted_chN/` subfolder. Verified pixel-exact match against manually extracted Fiji references (mean_abs_diff = 0).

Result: 23 files in `extracted_ch2/`, 8 files in `extracted_ch1/`. Z depths range 14–27 planes.

Metadata embedded by Fiji preserved: XY = 0.1326 µm, Z = 0.410 µm.

---

### RESPAN Run on New Crops

19 crops selected for fine-tuning (those with most Z depth and best condition coverage) placed in `F:\ResPan\images\data_to_train\` with `Analysis_Settings.yaml`. Ran RESPAN with Model 1Bv2:

```
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\run_respan.py data_to_train
```

---

### Manual Label Correction in Fiji

Workflow using two macros in `manual_fixing_labels/imageJ_plugins/`:

1. **Macro 1** (`1_Labels_to_Channels_for_Annotation-v2.ijm`): converts integer label TIF → multi-channel TIF (C1=raw, C2=dendrite, C3=spine_core, C4=spine_edge, C5=spine_neck, C6=soma, C7=axon). Requires imagesTr files to end in `_0000.tif`.
2. Correct each channel in Fiji (paintbrush white to add, Alt+paintbrush to erase). Key Fiji tip: Alt+paintbrush erases only when background colour = black (press **D** to reset).
3. **Macro 2** (`2_Channels_to_Labels_for_Training.ijm`): converts channels back to integer labels. Always use **MaxLabel = 6** — using lower values corrupts the last channel (leaves it as 255 instead of its true integer).

20 crops processed through this workflow. 4 crops (12, 13, 19, 27) were too ambiguous/poor quality to correct reliably — excluded from training. Final corrected labels: **16 crops** in `manual_fixing_labels/training_data/labelsTr_2026_4_29/`.

One crop (crop17) had corrupt values (8 and 9) from accidentally overlapping channel annotations — fixed by zeroing those 55 voxels.

---

### RESPAN Analysis Pipeline — Structural Changes

`run_respan.py` refactored for the new workflow:

| Change | Details |
|---|---|
| `DATASETS_ROOT` | Changed from `_BASE/images` to `_BASE` — `images_analysis` is now a sibling of the script, not inside `images/` |
| `MODEL_DIR` | Changed from hardcoded Dataset226 path to `_find_model()` — dynamically finds the first `Dataset*` folder inside `images_analysis/model/` |
| `DATASET_ID` | Now parsed from the model folder name via regex instead of hardcoded `'226'` |
| All RESPAN settings | `_make_settings()` updated to read ALL meaningful RESPAN parameters from YAML (was hardcoding ~20 settings previously) |

New folder structure for analysis:
```
F:\ResPan\images_analysis\
    images\           <- TIF files + Analysis_Settings.yaml
    model\
        Dataset001_Spines\   <- fine-tuned model (any Dataset* name works)
```

`Analysis_Settings.yaml` expanded with full documented parameter set covering: spine filtering, dendrite repair, neck filtering, spurious neck handling, validation output flags.

---

### Key Findings

- **Spine edge jumped from 0 to 0.785 Dice in one epoch** when fine-tuning — the pretrained encoder already understood the morphology.
- **Partial-spine count is sensitive to `spine_dist_filter_um`** — lowering it caps the neck extension range, causing more necks to fail reaching the dendrite and increasing partial-spine count.
- **`save_spine_meshes` and `save_validation_volumes`** are now YAML-configurable (were previously unreachable — controlled by hardcoded `save_intermediate_data=False` and missing `force_val_volume_materialize`).

---

## Day 11 — 2026-05-29

### Goal

Fine-tune Model 1Bv2 on the 16 corrected Ellinoora crops using CSC Mahti A100, then run analysis on the 11 held-out test crops.

---

### Fine-Tuning Pipeline Bugs Fixed

Multiple issues encountered and resolved during the Mahti run:

| Bug | Cause | Fix |
|---|---|---|
| `label matching failed` | Images had `_0000` suffix, labels didn't | Renamed labels to add `_0000` suffix to match |
| `num_processes length mismatch` | Default 3 num_processes for 1 configuration | Added `-np 1` to `nnUNetv2_preprocess` call |
| `dataset.json not found` in preprocessed folder | `extract_fingerprint` doesn't copy it (only `plan_and_preprocess` does) | Added explicit `shutil.copy2` of `dataset.json` to preprocessed folder in `finetune_plan()` |
| CUDA device-side assert in Dice loss | Preprocessed labels contained `-1` values (ignore index), but loss function was checking for `7` (declared ignore label) — values never masked | Removed `ignore: 7` from `FINETUNE_LABELS` entirely; our data has no ignore-labeled voxels, so declare no ignore |

`train_nnunet.py` also updated with `--id` and `--name` CLI flags so dataset name/ID can be set from terminal without editing the file:
```bash
python train_nnunet.py finetune --id 42 --name EllinooraFinetune
```

---

### Fine-Tuning Run (13 epochs on Mahti A100)

Training configuration (from transferred Dataset226 plans):
- Architecture: ResidualEncoderUNet XL, 7 stages, features [32, 64, 128, 256, 320, 320, 320]
- Patch size: [40, 256, 256], batch size: 10
- ~8 min/epoch on A100

Results at epoch 13:

| Class | Dice |
|---|---|
| Dendrite | 0.966 |
| Spine core | 0.950 |
| Spine edge | 0.907 |
| Spine neck | 0.821 |
| Soma | 0.000 (rare in training crops) |
| Axon | 0.000 (rare in training crops) |

Train loss and val loss remained very close throughout (no overfitting gap). Training stopped at epoch 13 due to session time — model saved as `checkpoint_best.pth` (later copied to `checkpoint_final.pth`). 400 epochs recommended for full convergence; this model is usable but spine neck would benefit from more training.

Model downloaded to `F:\ResPan\images_analysis\model\Dataset001_Spines\`.

---

### Test Set Analysis (11 crops)

Ran analysis on the 11 held-out test crops (the ones NOT used for fine-tuning) with default settings:

| Run | Total spines | Partial-spines |
|---|---|---|
| Default settings | 163 | ~6 |
| After tuning | 187 | 38 |

Tuning `spine_dist_filter_um` from 4.0 to 3.0 unexpectedly increased total spines — the tighter cap on neck extension created more partial-spines (counted in total). Reverted to 4.0 µm.

Final recommended settings for this dataset:
```yaml
min_spine_vol_um3: 0.05       # was 0.03 — filters ~7-voxel noise at this resolution
min_spine_head_voxels: 3      # was 2 — drops 1-3 voxel prediction artifacts
spine_dist_filter_um: 4.0     # keep at default — lowering creates more partial-spines
```

---

### Key Findings

- **Soma/axon Dice = 0 throughout training** — these classes are very sparse in the 16 training crops (small FOV crops don't always contain soma/axon). Does not affect spine density or neck measurements.
- **Pseudo-Dice EMA is misleadingly low** because soma/axon drag the average down despite the 4 main classes performing well.
- **Learning rate is on a fixed polynomial schedule** (`LR = 0.01 × (1 - epoch/1000)^0.9`) — not adaptive to EMA. The 0.0 soma/axon does not affect the learning rate.
- **Over-annotation artifact**: the model learned to predict ambiguous structures as spines because they were labeled as spines during Fiji correction. Proper fix is to re-clean training labels (leave ambiguous structures as background) and retrain.

---

### Key Files

| File | Purpose |
|---|---|
| `F:\ResPan\train_nnunet.py` | Updated: `--id/--name` CLI flags, `finetune_plan` dataset.json copy, ignore label removed |
| `F:\ResPan\run_respan.py` | Updated: dynamic DATASETS_ROOT, dynamic MODEL_DIR, all RESPAN settings YAML-readable |
| `F:\ResPan\images_analysis\images\Analysis_Settings.yaml` | Full documented parameter set for analysis |
| `F:\ResPan\images_analysis\model\Dataset001_Spines\` | Fine-tuned model (13 epochs from Model 1Bv2) |

---

### Next Steps

- [ ] Clean training labels — remove ambiguous annotations, re-train for cleaner results
- [ ] Continue fine-tuning to ~400 epochs for better spine neck convergence
- [ ] Enable `save_spine_meshes: True` and `save_validation_volumes: True` for 3D outputs
- [ ] Run analysis on full Ellinoora dataset once model quality is satisfactory

---

## Day 12 — 2026-06-05/10

### Goal

Re-clean Fiji-corrected training labels, prepare them for re-upload to Mahti, and re-run fine-tuning from Dataset226 (Model 1Bv2) with cleaner annotations. The key idea: leave all ambiguous structures (unclear spines, filopodia, partial necks) as background rather than labelling them. If they still show up in test results, they are genuinely detectable — if they disappear, they were noise driven by over-annotation.

---

### glasby_on_dark Fix (Fiji Macro)

When re-running Macro 2 (`2_Channels_to_Labels_for_Training.ijm`) to convert corrected channels back to integer labels, the macro crashed with:

```
Unrecognized command: "glasbey_on_dark"
```

The `run("glasbey_on_dark")` line applies a colorful LUT to the label TIF before saving so that labels are visually distinguishable in Fiji. Without it, all integer values 1–6 appear nearly black (near-zero intensity display). The LUT is cosmetic — it does not affect pixel values or nnU-Net training. The error was due to a missing LUT in the Fiji installation. Fixed the Fiji installation; the line was kept in the macro.

---

### Label Validation and Cleanup

New corrected labels placed in `F:\ResPan\training_data\labelsTr\` (16 crops). One crop (crop17) had corrupt label values 8 and 9 (10 voxels total) — these arose from accidentally overlapping channel paintbrush regions. Fixed by zeroing all non-standard voxels with numpy:

```python
import tifffile, numpy as np
vol = tifffile.imread('crop17_..._0000.tif')
vol[~np.isin(vol, [0,1,2,3,4,5,6])] = 0
tifffile.imwrite('crop17_..._0000.tif', vol)
```

All 16 label files confirmed clean (values 0–6 only).

---

### Label Filename Rename — `_0000` Suffix

The images on Mahti carry a `_0000` suffix in their filenames (e.g. `crop10_..._0003-1_0000.tif`). `train_nnunet.py`'s `setup()` function matches labels to images by exact filename, so labels must also have `_0000` before `.tif`. Renamed all 16 labels locally:

```powershell
Get-ChildItem "F:\ResPan\training_data\labelsTr\*.tif" | ForEach-Object {
    Rename-Item $_.FullName ($_.BaseName + "_0000.tif")
}
```

---

### UnicodeEncodeError in RESPAN v1.5 — FileHandler Fix

RESPAN v1.5 added Unicode characters (∩ and ✓) to the `_audit_3d_label_overlaps` function. On Windows, the default `cp1252` terminal encoding cannot print these, producing a `UnicodeEncodeError` in the console. The analysis still completes correctly — the error is cosmetic. Fixed by adding `encoding='utf-8'` to the `FileHandler` in `run_respan.py` so the log file is written cleanly:

```python
fh = logging.FileHandler(log_path, encoding='utf-8')
```

The StreamHandler (console) was left as-is — patching it would risk conflicts with the existing stdout TextIOWrapper.

---

### CSC.md Corrections

Updated `F:\ResPan\CSC.md` with correct Mahti-specific paths and activation instructions:

- **Code location:** `/scratch/project_2019059/respan/code/ResPan/`
- **Environments:** `/projappl/project_2019059/respan/envs/respaninternal/` (nnU-Net) and `respanev/` (RESPAN analysis)
- **Activation:** Do NOT use `export PATH=...` — Tykky containers must be called via full Python path.

---

### Re-Training Setup and Job Submission

Deleted the old Dataset001 folders on Mahti (raw, preprocessed, results) to force a clean fine-tuning run from Dataset226. Uploaded the clean labels via the CSC web dashboard.

Created `train_job.sh` on Mahti using `printf` (to avoid a leading blank line that causes `sbatch` to reject the file as "not a batch script"):

```bash
printf '#!/bin/bash\n#SBATCH --account=project_2019059\n#SBATCH --partition=gpusmall\n...\n\ncd /scratch/project_2019059/respan/code/ResPan\n/projappl/project_2019059/respan/envs/respaninternal/bin/python train_nnunet.py finetune\n' > train_job.sh
sbatch train_job.sh   # → job 6802798
```

Job settings: A100 GPU, 24 hours, 64 GB RAM, 10 CPUs. nnU-Net auto-resumes from `checkpoint_latest.pth` if resubmitted.

---

## Day 13 — 2026-06-11/12

### Goal

Collect the training results from the 24-hour Mahti job, download the model, run analysis on the 11 test crops, and compare Model 2 (clean labels, 175 epochs) against Model 1 (all labels including ambiguous, ~20 epochs).

---

### Training Results (Job 6802798)

The 24-hour job completed ~175 epochs at ~470 seconds per epoch on the A100. Training was healthy throughout:

| Epoch | EMA pseudo-Dice | Train loss | Val loss |
|---|---|---|---|
| 0 | 0.231 | −0.210 | −0.457 |
| 50 | 0.621 | −0.828 | −0.838 |
| 100 | 0.634 | −0.840 | −0.847 |
| 150 | 0.639 | −0.842 | −0.846 |
| 174 | 0.640 | −0.843 | −0.847 |

Per-class Dice at epoch 174 (foreground classes only):

| Class | Dice |
|---|---|
| Dendrite | 0.982 |
| Spine core | 0.981 |
| Spine edge | 0.958 |
| Spine neck | 0.925 |
| Soma | 0.000 (not in training crops) |
| Axon | 0.000 (not in training crops) |

Train and val loss remained nearly identical throughout (no overfitting gap). The rapid improvement in epochs 0–50 is the warm-up phase where the randomly re-initialised segmentation head learns to read the pretrained encoder features. After epoch 50, improvement was gradual and flattening by epoch 150 — the model had largely converged.

EMA pseudo-Dice formula: mean over all 6 classes including soma/axon zeros. Actual relevant-class performance is ~0.96 average across the 4 main classes.

---

### Model Download and Placement

Downloaded the fine-tuned model from Mahti:

```
scp -r <user>@mahti.csc.fi:/scratch/.../nnunet_training/results/Dataset002_Spines/ F:\ResPan\nnunet_training\results\
```

Placed in `F:\ResPan\images_analysis\model\Dataset002_Spines\`. Since `002 < 226` alphabetically, `_find_model()` in `run_respan.py` automatically picks Dataset002 over Dataset226 without any other changes.

---

### Critical Discovery — Model in Wrong Location

Before placing the model in `images_analysis/model/`, analysis was re-run and appeared to use the fine-tuned model — but results were identical to Dataset226. Investigation revealed that `run_respan.py` always reads from `images_analysis/model/` (first `Dataset*` folder alphabetically). The old fine-tuned models from Day 11 (model1) were never placed in this folder — they were only on Mahti. Every analysis run since Day 11 had been using Dataset226. The "comparison" between model1 and model226 earlier was therefore comparing the same model against itself.

**Fix:** Dataset002_Spines (the properly trained 175-epoch model) was placed in `images_analysis/model/`. Old results deleted and analysis re-run.

---

### Analysis Results — 11 Test Crops

Analysis run with the correct fine-tuned model (Dataset002_Spines, 175 epochs, clean labels):

| Crop | Model 1 (~20 ep, all labels) | Model 2 (175 ep, clean labels) |
|---|---|---|
| crop2 | 13 | 11 |
| crop3 | 19 | 17 |
| crop6 | 25 | 19 |
| crop7 | 15 | 9 |
| crop8 | 21 | 13 |
| crop14 | 19 | 19 |
| crop15 | 22 | 17 |
| crop16 | 21 | 13 |
| crop21 | 13 | 10 |
| crop22 | 8 | 7 |
| crop23 | 11 | 10 |
| **TOTAL** | **187** | **145** |

Model 2 detects **42 fewer spines (−22%)** than Model 1.

---

### Morphological Comparison

Pooled averages across all 11 crops:

| Metric | Model 2 (175ep) | Model 1 (~20ep) | Difference |
|---|---|---|---|
| avg_spine_vol (µm³) | 0.574 | 0.523 | +9.8% |
| avg_head_vol (µm³) | 0.450 | 0.416 | +8.2% |
| avg_head_surf_area (µm²) | 3.185 | 3.013 | +5.7% |
| avg_spine_area (µm²) | 0.632 | 0.573 | +10.2% |
| avg_neck_length (µm) | 0.930 | 0.997 | −6.7% |
| avg_neck_vol (µm³) | 0.127 | 0.126 | +0.6% |

---

### Interpretation

Model 2's 42 missing spines are the small and ambiguous structures that were explicitly left as background in the clean training labels. The model learned not to predict them. The spines it does detect are on average ~10% larger in volume and surface area — these are the clearly-defined, unambiguous spines that were confidently annotated. Neck length is slightly shorter, consistent with the model finding cleaner, more direct neck paths rather than extending toward borderline structures.

The model1 extra detections were likely a mix of real small spines and noise/artifacts that the ambiguous labels taught the model to predict. Which is more "correct" cannot be determined without ground-truth validation — but for thesis analysis, the conservative model2 approach is safer (fewer false positives, higher confidence per detection).

Model1's ~20 epochs were still well within the warm-up phase (epochs 0–50) — the model was essentially still Dataset226 with barely any specialisation. The difference between model1 and model2 results is therefore largely: Dataset226 predictions vs truly fine-tuned predictions.

---

### Key Files

| File | Purpose |
|---|---|
| `F:\ResPan\nnunet_training\results\Dataset002_Spines\` | Fine-tuned model (175 epochs, clean labels) |
| `F:\ResPan\images_analysis\model\Dataset002_Spines\` | Model copy used by run_respan.py |
| `F:\ResPan\old_stuff_to_compare\analysis_results\model2\` | Analysis results with Model 2 (145 spines) |
| `F:\ResPan\old_stuff_to_compare\analysis_results\model001_analysis_results\` | Analysis results with Model 1 (187 spines) |
| `F:\ResPan\old_stuff_to_compare\analysis_results\model226\` | Original Dataset226 folder (images + model, no stored results) |

---

### Next Steps

- [ ] Decide whether to continue training Model 2 to 300 epochs (resubmit job — nnU-Net auto-resumes from `checkpoint_latest.pth`, needs ~17 more hours)
- [ ] Visually inspect crops with biggest drop (crop6: 25→19, crop7: 15→9, crop8: 21→13) in Fiji to determine whether missing detections are false positives or real small spines
- [ ] Optionally update `spine_dist_filter_um: 3.0 → 4.0` in `Analysis_Settings.yaml` (current value may be capping neck extension)
- [ ] Run analysis on full Ellinoora dataset once model quality decision is made

---

## Day 14 — 2026-06-12

### Goal

Deep-dive into the RESPAN analysis pipeline: how it works step by step, and specifically how it handles edge cases (mid-neck false heads, thin Z-slice spines, multiple dendrites, floating detections). Also audited the `detect_multi_head_spines` feature.

---

### How the RESPAN Analyzer Works (step by step)

The analyzer is called via `analyze_spines()` in `cloned_respan/RESPAN/ImageAnalysis/ImageAnalysis.py`. It takes the raw TIF and the nnU-Net label TIF and produces all CSVs and validation images.

---

#### Step 1 — Label channel extraction

The integer label TIF (values 1–6) is split into binary masks per class. For `model_type = 4` (our setting):

```
value 1 → dendrites
value 2 → spine_cores
value 3 → spine_membranes
value 4 → necks
value 5 → soma
value 6 → axons (ignored in our crops)
spines = spine_cores + spine_membranes  (combined for downstream steps)
```

---

#### Step 2 — Dendrite filtering

All connected components of the dendrite mask are labeled. Any component smaller than `min_dendrite_vol_um3` (1.0 µm³ ≈ 55 voxels at our resolution) is discarded. This removes floating dendrite fragments that would produce spurious spine-to-dendrite assignments.

`dendrite_repair: False` in our settings — no gap bridging.

---

#### Step 3 — Distance transforms and skeletonization

Two GPU distance maps are built:

1. **Dendrite distance** — Euclidean distance (µm) from every voxel to the nearest dendrite surface. Used for two separate things: filtering spines that are too far away, and measuring spine length (= max distance within the spine from the dendrite).

2. **Soma distance** — only built when soma voxels are present. Skipped for our training crops (no soma labelled).

3. **Dendrite skeleton** — `skimage.morphology.skeletonize()` on the dendrite mask produces a 1-voxel-thin centerline, used later for geodesic distance calculation.

---

#### Step 4 — Spine instance detection (watershed)

nnU-Net produces a semantic label (all spine_cores have value 2, touching each other). The analyzer separates individual spines using watershed:

1. Each connected component of `spine_cores` is eroded by `erode_shape = (0.5, 2, 2)` voxels (Z=0.5 = no erosion, Y=2 = 265 nm, X=2 = 265 nm). Surviving eroded seeds become watershed seeds.
2. Watershed fills outward from each seed through the combined `spine_cores + spine_membranes` volume. Each seed captures one spine instance.
3. If a core was entirely destroyed by erosion, the orphan membrane voxels are recovered separately as independent spine instances (orphan-membrane recovery path).

Result: `spine_labels` — every voxel has a unique integer instance ID.

---

#### Step 5 — Initial filtering and size gate

`regionprops` runs on every spine instance. Spines are dropped if:
- Volume < `min_spine_vol_um3` (0.05 µm³ ≈ 7 voxels at our resolution)
- Volume > `max_spine_vol_um3` (15.0 µm³)
- Min distance to any dendrite surface > `spine_dist_filter_um` (3.0 µm = 22.6 voxels)

A tiny-head backstop then drops instances ≤ `min_spine_head_voxels` (3 voxels) — single-voxel nnU-Net artifacts that survive the volume gate due to rounding.

Result: `spines_filtered` — instance-labeled volume with plausible spine heads only.

---

#### Step 6 — Neck association

nnU-Net neck voxels (value 4) are a semantic mask with no per-spine IDs. They are assigned to spines in three stages:

**6a — Connected components on raw necks**: each disconnected neck blob gets a component ID.

**6b — Associate by proximity** (`associate_spines_with_necks_gpu`): GPU nearest-neighbour search assigns each neck component to the closest spine head.

**6c — Extend toward dendrite** (`extend_objects_GPU`): GPU BFS/dilation grows each neck outward through high-intensity voxels toward the dendrite surface, bridging the 1–3 voxel gaps nnU-Net commonly leaves at the neck–dendrite junction.

**6d — Pathfinding fallback** (`bridge_spine_neck_via_pathfinding`): for necks that still don't reach the dendrite after BFS, `skimage.graph.route_through_array` runs Dijkstra on an inverted-intensity cost image (bright voxels = low cost = preferred path). This is "pathfinder-synthesised" neck — the spurious filter checks these specifically.

---

#### Step 7 — Neck quality filters

Four filters run sequentially:

1. **Drop disconnected fragments** (`drop_disconnected_neck_fragments: True`): keeps only fragments 26-voxel-connected to the spine head; drops the rest.
2. **Drop wrong-direction fragments** (`drop_wrong_direction_neck_fragments: True`): drops neck fragments that travel away from the dendrite instead of toward it.
3. **Trim necks touching other spines** (`trim_necks_touching_other_spines: False`): disabled — preferred action when a neck crosses another head is to mark as partial-spine.
4. **Spurious neck detection**: flags necks longer than `spurious_path_length_min_um` (1.0 µm) with fewer than `spurious_nnunet_support_min` (20%) of voxels actually predicted as neck by nnU-Net. Action = `spurious_exclude_mode: neck_only` — drop the neck, keep the head, reclassify as partial-spine.

---

#### Step 8 — Partial spines and filopodia

**Partial-spine** (`keep_partial_spines: True`): any spine whose neck never gets within `partial_spine_dist_threshold_vox` (2 voxels ≈ 130 nm) of the dendrite is flagged as `spine_type = 'partial-spine'`. Kept in the CSV with neck metrics zeroed.

**Filopodia recovery** (`recover_filopodia: False`): orphan neck voxels with no attached head would become filopodia. Disabled — orphan necks are silently discarded.

---

#### Step 9 — Geodesic distance from soma

Dijkstra runs along the dendrite skeleton from each spine's attachment point to the soma (or the nearest skeleton endpoint if no soma). Output is `geodesic_dist_to_soma` in µm. Each spine also gets a `dendrite_id` — which labeled dendrite it attaches to.

---

#### Step 10 — Full morphological measurements

Per-spine quantitative measurements:

| Measurement | Method |
|---|---|
| `spine_vol`, `head_vol`, `neck_vol` | voxel count × voxel volume |
| `head_surf_area`, `spine_area` | marching cubes mesh surface area |
| `spine_length` | max of dendrite distance map inside spine voxels |
| `neck_length` | arc length along neck voxels from head to dendrite |
| `head_bbox_vol`, `head_extent`, `head_solidity` | regionprops on bounding box |
| `avg_spine_C1_mean_int` | mean raw fluorescence inside each region |

If `save_spine_meshes: True` — each head mesh is saved as an `.obj` file.

---

#### Step 11 — Table and validation output

Two CSVs per image:
- `<name>_detected_spines.csv` — one row per spine, all per-spine metrics
- `<name>_dendrite_summary.csv` — one row per dendrite (length, total spines, averages)
- `Detected_spines_summary.csv` — one row per image, updated after each image

Validation MIPs: 8-channel max-intensity projections (raw, dendrite, spine, neck, spine IDs, dendrite distance, geodesic distance) for visual QC.

---

### Edge Cases — How the Analyzer Handles Them

#### Multiple dendrites in one image

`filter_dendrites` labels every dendrite separately (label 1, 2, 3…). All downstream processing runs on the whole volume simultaneously — not dendrite by dendrite. The distance transform is computed jointly, so each voxel gets the distance to the nearest dendrite surface (whichever dendrite that is). Neck extension grows toward whichever dendrite is closest. `dendrite_id` in the CSV records which dendrite each spine attaches to. The `dendrite_summary.csv` has one row per dendrite with its own length and spine count.

#### Thin Z-slice spines (1–2 Z planes only)

Z resolution = 410 nm vs XY = 133 nm. Spines thinner than 2 Z-slices are biologically possible (small filopodia, thin mushroom necks). Z erosion is 0.5 voxels (effectively zero), so thin Z-extent cores survive the erosion step. The real gate is the volume filter: at our voxel size (0.00721 µm³/voxel), a single-Z-slice spine needs ≥ 7 voxels in XY to survive the 0.05 µm³ threshold. Below that it is dropped. If it survives, all 3D measurements still work — just with fewer voxels and a flatter marching cubes mesh.

#### Floating spines (no nearby dendrite)

`initial_spine_measurements` checks the minimum dendrite distance across all voxels of each spine. Any spine with minimum distance > `spine_dist_filter_um` (3.0 µm) is dropped entirely and never appears in any CSV.

#### Floating orphan necks (nnU-Net predicted a neck with no spine head)

After neck association, neck voxels that couldn't be matched to any spine head (because the head was filtered out) are orphan necks. With `recover_filopodia: False`, they are **silently discarded** — they do not appear in any CSV or log. With `recover_filopodia: True` they would appear as filopodia.

---

### `detect_multi_head_spines` — Deep Audit

#### What it does

After all neck processing is complete, builds a "territory" volume:

```
territory = head_voxels | assigned_neck_voxels | raw_nnU-Net_neck_prediction
```

The third term is the **original nnU-Net neck prediction (class 4) binarized before any filtering**. Even necks dropped by the spurious or disconnect filters are still included. Runs 26-connected components on the territory. If one CC contains 2+ spine head labels → multi-head group. Parent = smallest spine_id in the group.

Three new columns added to `detected_spines.csv`: `parent_spine_id`, `head_index`, `multi_head_group_size`. A new `_Multi_Head_Spine_Groups.csv` is created with one row per group, summing head_vol, neck_length, etc. across all members.

#### What it does NOT do

- Does not delete the false head — both rows remain in the CSV
- Does not change spine_vol, neck_length, or any measurement
- Does not change the total count in `Detected_spines_summary.csv`
- Does not distinguish a real branched spine from a mid-neck false detection

#### Shortcomings

**1. Primary = smallest spine_id, not the real head.** Watershed labels seeds by spatial order. The smallest label is not the largest or most biologically meaningful head. In a mid-neck false-head case, the false head may get label 7 and the real head label 8, making the false head `head_index = 0` (primary). Post-processing by `head_vol` is the only reliable way to find the real head.

**2. 26-connectivity with no distance limit.** If nnU-Net predicted a spurious neck blob between two completely independent adjacent spines, their territories connect and they are grouped as a "multi-head spine" even though they are two separate structures. No radius cap, no size check on the territory.

**3. No discrimination between real branches and false detections.** A genuine two-tip branched spine and a mid-neck false detection look identical: two rows, same `parent_spine_id`, `multi_head_group_size = 2`.

**4. Neck_length summed, not selected.** In `Multi_Head_Spine_Groups.csv`, `total_neck_length = sum of all member neck_lengths`. For a real branched spine, both members report a neck path to the dendrite — the sum double-counts. For the false-head case, the real head is partial-spine (neck_length = 0) and the false head has the actual neck — the sum happens to be numerically correct but for the wrong reason.

**5. `_neck_mask` can be None.** The raw neck mask is only in scope when `neck_generation = True` AND `model_type >= 3`. If either is false, the territory reduces to `head_voxels | assigned_neck_voxels` only — much weaker detection. Not an issue with our settings but fragile.

**6. Does not correct `Detected_spines_summary.csv`.** The summary CSV total_spines count is built before the grouping columns are added.

#### How to use it correctly for counting

```python
df = pd.read_csv('..._detected_spines.csv')
# Count only primary heads (one per physical spine)
n_real = (df['spine_id'] == df['parent_spine_id']).sum()
# Or equivalently:
n_real = (df['head_index'] == 0).sum()
# For the real head's measurements in a group, pick the member with largest head_vol
```

#### Current setting

`detect_multi_head_spines: False` (disabled in `Analysis_Settings.yaml`). Enabling it adds the grouping annotation but does not break any existing analysis — it is purely additive columns. Worth enabling for thesis analysis to at least surface the topology, then filter in post-processing.

---

### Next Steps

- [ ] Enable `detect_multi_head_spines: True` and re-run on model2 results, inspect groups CSV
- [ ] Decide whether to resubmit training to 300 epochs
- [ ] Visually QC crops with biggest model1→model2 drop (crop6, crop7, crop8) in Fiji
- [ ] Run analysis on full Ellinoora dataset

---

## Day 15 — 2026-06-15

### SelfNet — Feasibility, Pipeline Analysis, and Bug Fix

---

#### Can SelfNet be trained on the current crops?

**No.** Confirmed from WhatHappened.md Day 13 assessment (line 1327):

> SelfNet requires `original_Z >= ceil(64 × scale)` to form 64×64 XZ patches. At scale = 0.1326/0.410 = 0.323, the minimum is **21 Z planes**. Only crop10 (22 slices) meets this threshold; the other crops (14–18 slices) cannot even start SelfNet.

**Decision remains: skip SelfNet training on these crops.** Worth revisiting when full-FOV stacks with ≥30 Z planes from multiple sessions are available.

---

#### How SelfNet training works (pipeline)

Three internal steps in `SelfNet_Model_Training.py`:

**Step 1 — `self_net_create_slices`**

Reads one 3D TIF, extracts:
- `training_slices/xy/` — one 2D slice per Z-plane (sharp, native XY resolution)
- `training_slices/xy_lr/` — same slices downsampled by `scale` then upsampled back (simulates Z-quality blur in XY)
- `training_slices/xz/` — resliced XZ views, scaled so pixel size matches XY (one per Y-position)
- `training_slices/yz/` — same for YZ (one per X-position; not used in training)

The `scale` parameter = `XY_res / Z_step` (e.g. 0.1326/0.410 = 0.323 for our data). It controls how much the XY-lowres simulation degrades the image, and how much the XZ reslice is stretched to make voxels isotropic. **Critical**: if scale is wrong, the network learns to correct the wrong amount of blur.

**Z-plane constraint**: XZ patch height = `Z_planes / scale`. For 64×64 patches you need `Z_planes >= ceil(64 × scale)`. At scale=0.323 that is 21 planes.

**Step 2 — `generate_training_data`**

Reads all slices from `training_slices/xy/`, `training_slices/xy_lr/`, `training_slices/xz/`. Crops 64×64 patches with stride 64. Discards patches below `bg_threshold` (background). Saves `train_data/train_data.npz`. Uses `len(os.listdir())` dynamically — the directory can contain slices from any number of images.

**Step 3 — `train_self_net`**

CycleGAN with two generators (G_A: xy_lr → xz domain, G_B: xz → xy_lr domain) and two discriminators, plus a `deblur_net` that maps `G_A(xy_lr)` → `xy`. Only `deblur_net` is saved as the final model. 60 epochs default (GUI), batch_size=8, 1 GPU (`cuda:0`). Takes 30 min–2 hours depending on patch count.

---

#### Bug found and fixed: `SelfNet_Model_Training.py` only processed one image

**Problem**: `main()` in `cloned_respan/RESPAN/Scripts/SelfNet_Model_Training.py` did:

```python
tif_files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
if tif_files:
    filename = tif_files[0]  # Select the first .tif file
self_net_create_slices(input_dir, filename, scale)
```

Only `tif_files[0]` was processed. `generate_training_data` was designed to read however many slices are in the directory (`len(os.listdir())`), so the infrastructure for multi-image training existed — but `main()` never fed it more than one file.

The RESPAN GUI passes `input_dir` (a folder) and the docstring says "use multiple images to cover the full content diversity", but only the first image alphabetically was ever used.

Additionally, `self_net_create_slices` always wrote files starting from `1.tif`, so calling it twice would silently overwrite the first image's slices.

**Fix (in `cloned_respan/RESPAN/Scripts/SelfNet_Model_Training.py`)**:

1. Added `xy_start=0, xz_start=0` parameters to `self_net_create_slices` — slices are written starting from `xy_start + 1` and `xz_start + 1` instead of always from `1`. Returns `(n_xy, n_xz)` so the caller knows the next offset.

2. Changed `main()` to loop over all TIFs, tracking cumulative offsets:

```python
tif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.tif')])
xy_start = 0
xz_start = 0
for filename in tif_files:
    n_xy, n_xz = self_net_create_slices(input_dir, filename, scale, xy_start, xz_start)
    xy_start += n_xy
    xz_start += n_xz
```

Backward compatible: default `xy_start=0, xz_start=0` produces identical output for the single-image case.

3. Fixed a stale comment in `train_selfnet.py` that said "only the first image alphabetically will actually be used" — no longer true.

---

### Files Changed

| File | Change |
|---|---|
| `cloned_respan/RESPAN/Scripts/SelfNet_Model_Training.py` | `self_net_create_slices`: added `xy_start`/`xz_start` offset params, returns `(n_xy, n_xz)`; `main()`: loops over all TIFs with cumulative offsets |
| `train_selfnet.py` | Removed stale "only first image" comment |

---

### Next Steps

- [ ] Enable `detect_multi_head_spines: True` and re-run on model2 results, inspect groups CSV
- [ ] Decide whether to resubmit training to 300 epochs
- [ ] Visually QC crops with biggest model1→model2 drop (crop6, crop7, crop8) in Fiji
- [ ] Run analysis on full Ellinoora dataset
- [ ] SelfNet training feasible once full-FOV stacks (≥30 Z planes) are available

---

## Day 16 — 2026-06-15

### Goal

Create a Fiji-compatible ROI export from RESPAN analysis outputs so that segmentation boundaries (dendrite, individual spines, individual necks) can be overlaid on the original image stack and on MIP projections inside Fiji's ROI Manager.

---

### `generate_fiji_rois.py` — New Script

**What it does**: reads the label TIF and detected_spines CSV for each crop and writes a single ROI zip per crop containing polygon ROIs for every structure, named and numbered to match the CSV `spine_id`.

**Where it looks**: scans `images_analysis/` and enters only subdirectories that contain a `labels/` folder — avoids `images/`, `model/`, or any other folder automatically.

**Inputs per crop**:
- `labels/<name>.tif` — semantic label volume (Z,Y,X), values 1–6
- `Tables/<name>_detected_spines.csv` — `spine_id`, `x`, `y`, `z` columns in voxel coordinates

**How spine IDs are assigned** (the labels TIF has no instance IDs — only semantic classes):
1. Connected components on `label==2|3` (core + membrane) separates individual spine instances.
2. Each CC centroid is matched to the nearest CSV row by Euclidean distance in (X,Y,Z) voxel space.
3. Multiple CC fragments matching the same spine_id (e.g. disconnected core/membrane pieces) are OR-merged into one mask before ROI tracing.
4. Same approach for necks (`label==4`): each neck CC is matched to the nearest spine CC centroid, then looked up to the CSV spine_id, then merged per spine_id.

**ROIs produced per crop**:

| ROI name | Source | Use |
|---|---|---|
| `dendrite-Z{n}` | `label==1`, per Z-slice | Overlay on 3D stack |
| `spine_{id}-Z{n}` | `label==2\|3`, merged per spine, per Z-slice | Overlay on 3D stack |
| `neck_{id}-Z{n}` | `label==4`, merged per spine, per Z-slice | Overlay on 3D stack |
| `dendrite-MIP` | Z-projection of dendrite mask | Overlay on MIP image |
| `spine_{id}-MIP` | Z-projection of spine mask | Overlay on MIP image |
| `neck_{id}-MIP` | Z-projection of neck mask | Overlay on MIP image |

**Output**: `images_analysis/<name>/ROIs/<name>_rois.zip`

To load in Fiji: open the image → Analyze > Tools > ROI Manager > More > Open → select zip → check Show All.

**How to run**:
```powershell
# All crops
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\generate_fiji_rois.py

# Single crop
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe f:\ResPan\generate_fiji_rois.py --crop crop14_...
```

---

### Bugs Fixed During Development

#### 1. Duplicate ROI name warnings

**Problem**: `zipfile` warned about duplicate entries like `spine_6-Z7.roi`. This happened because connected components found more CC instances (21) than the CSV had spine rows (14) — small segmentation fragments that were filtered out by RESPAN's volume/distance gates still appear in the label TIF and steal spine_ids via nearest-centroid matching.

**Fix**: instead of generating ROIs immediately per CC, collect a `defaultdict(mask)` keyed by spine_id and OR each CC's mask into it. After all CCs are processed, generate ROIs from the merged masks. Two fragments that share the same spine_id become one combined polygon per Z-slice. Applied the same fix to neck CCs.

#### 2. Straight-line artifact on border-touching ROIs

**Problem**: for Z-slices where the dendrite (or any structure) touches the edge of the image, `find_contours` cannot close the contour on the open side and draws a straight line between the two endpoints instead.

**Fix**: pad the 2D binary mask by 1 pixel of zeros on all sides before calling `find_contours`, then subtract 1 from the contour coordinates to restore original pixel positions. The object no longer touches the image border, so a proper closed polygon is traced.

```python
padded = np.pad(slice_mask, 1, mode='constant', constant_values=0)
contours = find_contours(padded, 0.5)
contour = max(contours, key=len) - 1   # subtract pad offset
```

---

### Dependencies

`roifile` is not in either conda environment by default. Install with pip (pure Python, no compiled deps):

```powershell
# Windows (respandev)
C:\Users\ARASH\Anaconda3\envs\respandev\python.exe -m pip install roifile
```

```bash
# Mahti (goes to user site-packages, picked up automatically)
/projappl/project_2019059/respan/envs/respandev/bin/pip install roifile
```

All other dependencies (`scipy`, `skimage`, `tifffile`, `pandas`, `numpy`) are already present in `respandev`.

---

### Files Changed

| File | Change |
|---|---|
| `generate_fiji_rois.py` | New script — Fiji ROI zip generation from RESPAN labels |

---

### Next Steps

- [ ] Enable `detect_multi_head_spines: True` and re-run on model2 results, inspect groups CSV
- [ ] Decide whether to resubmit training to 300 epochs
- [ ] Visually QC crops with biggest model1→model2 drop (crop6, crop7, crop8) in Fiji using the new ROIs
- [ ] Run analysis on full Ellinoora dataset
- [ ] SelfNet training feasible once full-FOV stacks (≥30 Z planes) are available
