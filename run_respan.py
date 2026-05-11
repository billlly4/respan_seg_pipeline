"""
RESPAN Unified Pipeline
=======================
Discovers every subfolder under DATASETS_ROOT that contains:
    images/                  -- TIF stacks to analyse
    Analysis_Settings.yaml   -- resolution and analysis parameters

For each image found it runs the complete pipeline and writes all outputs
into a per-image subfolder:

    <dataset_folder>/
        <image_stem>/
            images/                   <- hardlink to original TIF
            nnunet_input/             <- nnU-Net-ready single-channel TIF
            labels/                   <- nnU-Net segmentation output
            Tables/                   <- CSV spine measurements
            Spine_Arrays/             <- sub-volume MIPs and volumes
            Validation_Data/
                Validation_MIPs/      <- 8-channel MIP composites
                Validation_Vols/
                Spine_Meshes/
            SWC_files/
            RESPAN_analysis.log
            Analysis_Settings.yaml    <- copy of the settings used

HOW TO RUN:
    C:/Users/ARASH/Anaconda3/envs/respandev/python.exe  f:/ResPan/run_respan.py

    Optional: process only specific dataset folders by passing their names:
    ... run_respan.py  inference_cortical  inference_hippo
"""

import os
import sys
import ast
import types
import shutil
import logging
import subprocess
import warnings
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite

warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------------------------
# PATHS  — auto-detected by hostname; edit the two blocks below if needed
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent

DATASETS_ROOT = str(_BASE / 'images')
RESPAN_ROOT   = str(_BASE / 'cloned_respan')
NNUNET_REPO   = str(_BASE / 'nnUNet')

MODEL_DIR = str(
    _BASE / 'cloned_respan'
    / 'Dataset226_Global1Bv2_XY65nm_Z150nm_CARE-Aug_spinecore_neck_dend_soma_axon_V6'
)

_mahti_py = '/projappl/project_2019059/respan/envs/respaninternal/bin/python'
_win_py   = r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe'
RESPANINTERNAL_PYTHON = _mahti_py if os.path.exists(_mahti_py) else _win_py
_ON_MAHTI = os.path.exists(_mahti_py)

NNUNET_RESULTS = str(Path(MODEL_DIR).parent)
DATASET_ID     = '226'

# 0-indexed channel that contains the structural morphology fill.
# For single-channel TIFs this must be 0.
NEURON_CHANNEL = 0


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml(yaml_path):
    """
    Parse Analysis_Settings.yaml and return a flat dict of settings.
    Tuple strings like '(2,3,3)' are converted to actual tuples.
    """
    import yaml

    def _parse(v):
        if isinstance(v, str) and v.strip().startswith('('):
            try:
                return ast.literal_eval(v.strip())
            except Exception:
                pass
        return v

    with open(yaml_path, encoding='utf-8') as fh:
        raw = yaml.safe_load(fh)

    flat = {}
    for section_data in raw.values():
        if isinstance(section_data, dict):
            for k, v in section_data.items():
                flat[k] = _parse(v)
    return flat


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def find_datasets(root, filter_names=None):
    """
    Return [(dataset_dir, yaml_path, tif_paths)] for every subfolder of
    *root* that contains Analysis_Settings.yaml and at least one TIF.

    Two layouts are supported:
      Flat   -- TIFs sit directly in the dataset folder alongside the YAML
      images/ -- TIFs are in an 'images/' subfolder; YAML at dataset root
                 OR inside 'images/' itself

    filter_names: optional list of folder names to restrict to.
    """
    hits = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        if filter_names and name not in filter_names:
            continue

        # Locate the YAML -- check folder root first, then images/ subdir
        yaml_p = os.path.join(d, 'Analysis_Settings.yaml')
        if not os.path.isfile(yaml_p):
            yaml_p = os.path.join(d, 'images', 'Analysis_Settings.yaml')
            if not os.path.isfile(yaml_p):
                continue   # no YAML anywhere -- skip

        # Collect TIFs -- prefer images/ subfolder, fall back to flat
        imgs_dir = os.path.join(d, 'images')
        if os.path.isdir(imgs_dir):
            tifs = [os.path.join(imgs_dir, f)
                    for f in os.listdir(imgs_dir) if f.lower().endswith('.tif')]
        else:
            tifs = []

        if not tifs:  # flat mode: TIFs directly in dataset folder
            tifs = [os.path.join(d, f)
                    for f in os.listdir(d) if f.lower().endswith('.tif')]

        if tifs:
            hits.append((d, yaml_p, sorted(tifs)))
    return hits


# ---------------------------------------------------------------------------
# Step 1 – prepare image for nnU-Net
# ---------------------------------------------------------------------------

def extract_neuron_channel(img, ch):
    """Accept ZYX, ZCYX, or CZYX and return ZYX float32."""
    if img.ndim == 3:
        return img.astype(np.float32)
    if img.ndim == 4:
        if img.shape[1] <= 4:          # ZCYX
            return img[:, ch, :, :].astype(np.float32)
        if img.shape[0] <= 4:          # CZYX
            return img[ch, :, :, :].astype(np.float32)
        return img[:, ch, :, :].astype(np.float32)
    raise ValueError(f'Unexpected image shape {img.shape}')


def prepare_image(src_path, prep_dir, in_xy, in_z):
    """
    Extract the structural channel and save as <stem>_0000.tif with
    ImageJ spacing metadata so nnU-Net can read the voxel size.
    Images are passed at ORIGINAL resolution -- nnU-Net resamples internally.
    """
    stem   = Path(src_path).stem
    out_path = os.path.join(prep_dir, f'{stem}_0000.tif')
    os.makedirs(prep_dir, exist_ok=True)

    img    = imread(src_path)
    neuron = extract_neuron_channel(img, NEURON_CHANNEL)

    # nnU-Net wants at least 8 Z planes; pad with zeros if needed
    if neuron.shape[0] < 8:
        pad = np.zeros((8 - neuron.shape[0], *neuron.shape[1:]), dtype=neuron.dtype)
        neuron = np.concatenate([neuron, pad], axis=0)
        print(f'    padded Z to {neuron.shape[0]} slices (nnU-Net minimum)')

    imwrite(
        out_path,
        neuron.astype(np.float32),
        imagej=True,
        metadata={'spacing': in_z, 'unit': 'um', 'axes': 'ZYX'},
        resolution=(1 / in_xy, 1 / in_xy),
    )
    print(f'    shape={neuron.shape}  -> {out_path}')
    return out_path


# ---------------------------------------------------------------------------
# Step 2 – nnU-Net segmentation (subprocess in respaninternal)
# ---------------------------------------------------------------------------

def run_nnunet(prep_dir, labels_dir, checkpoint_name='checkpoint_best.pth'):
    """
    Run nnU-Net predict_entry_point() as a subprocess under respaninternal.
    This avoids the torch 2.6.0 hang that occurs when importing nnunetv2
    inside respandev.
    On Mahti, tykky containers cannot be nested — run run_nnunet_only.py first.
    """
    if _ON_MAHTI:
        print('\nERROR: On Mahti, run_respan.py cannot call nnU-Net directly.')
        print('  Run run_nnunet_only.py with respaninternal first to generate labels,')
        print('  then re-run run_respan.py — it will skip the nnU-Net step automatically.')
        sys.exit(1)
    os.makedirs(labels_dir, exist_ok=True)

    runner_script = (
        "import os, sys\n"
        f"os.environ['nnUNet_results'] = r'{NNUNET_RESULTS}'\n"
        "os.environ.setdefault('nnUNet_raw', 'not_used_for_inference')\n"
        "os.environ.setdefault('nnUNet_preprocessed', 'not_used_for_inference')\n"
        f"sys.path.insert(0, r'{NNUNET_REPO}')\n"
        "sys.argv = [\n"
        "    'nnUNetv2_predict',\n"
        f"    '-i', r'{prep_dir}',\n"
        f"    '-o', r'{labels_dir}',\n"
        f"    '-d', '{DATASET_ID}',\n"
        "    '-c', '3d_fullres',\n"
        "    '-f', 'all',\n"
        f"    '-chk', '{checkpoint_name}',\n"
        "]\n"
        "from nnunetv2.inference.predict_from_raw_data import predict_entry_point\n"
        "predict_entry_point()\n"
    )

    result = subprocess.run(
        [RESPANINTERNAL_PYTHON, '-u', '-c', runner_script],
        check=False,
    )
    if result.returncode != 0:
        print(f'\nERROR: nnU-Net subprocess exited with code {result.returncode}')
        sys.exit(1)

    # Remove nnU-Net's json/pkl artifact files, keep only TIFs
    removed = sum(
        1 for f in os.listdir(labels_dir)
        if not f.endswith('.tif')
        and os.path.isfile(os.path.join(labels_dir, f))
        and (os.remove(os.path.join(labels_dir, f)) or True)
    )
    if removed:
        print(f'    removed {removed} nnU-Net artifact file(s)')


# ---------------------------------------------------------------------------
# Step 3 – RESPAN analyze_spines
# ---------------------------------------------------------------------------

def _make_settings(cfg, out_dir):
    """Build a settings SimpleNamespace from YAML cfg + fixed defaults."""
    s = types.SimpleNamespace()

    in_xy = float(cfg.get('input_resXY', 0.102))
    in_z  = float(cfg.get('input_resZ',  0.200))

    s.input_resXY = in_xy
    s.input_resZ  = in_z
    # Set model_res == input_res so RESPAN skips label rescaling
    # (nnU-Net already handled resampling internally)
    s.model_resXY = in_xy
    s.model_resZ  = in_z

    s.model_type     = 4   # "Dendrites, Spine Cores, Spine Membranes, Necks, Soma, Axons"
    s.neuron_channel = 1   # 1-indexed in RESPAN

    voxel_vol = in_xy ** 2 * in_z
    min_dend_um3   = float(cfg.get('min_dendrite_vol_um3',  1.0))
    min_spine_um3  = float(cfg.get('min_spine_vol_um3',     0.03))
    max_spine_um3  = float(cfg.get('max_spine_vol_um3',    15.0))
    spine_dist_um  = float(cfg.get('spine_dist_filter_um',  4.0))
    s.min_dendrite_vol  = int(min_dend_um3 / voxel_vol)
    s.neuron_spine_size = (int(min_spine_um3 / voxel_vol), int(max_spine_um3 / voxel_vol))
    s.neuron_spine_dist = spine_dist_um / in_xy

    # Analysis parameters from YAML (with safe defaults)
    s.tiles_for_prediction = cfg.get('tiles_for_prediction', (2, 3, 3))
    s.roi_volume_size      = cfg.get('roi_volume_size', 4)
    s.GPU_block_size       = cfg.get('GPU_block_size', (150, 500, 500))
    s.erode_shape          = cfg.get('erode_shape', (0.5, 2, 2))
    s.remove_touching_boarders = cfg.get('remove_touching_boarders', False)
    s.validation_format    = cfg.get('validation_format', 'tif')
    s.validation_scale     = cfg.get('validation_scale', (1, 1))
    s.validation_jpeg_comp = cfg.get('validation_jpeg_comp', 100)

    # Feature flags
    s.neck_generation        = True
    s.second_pass            = False
    s.save_val_data          = True
    s.save_intermediate_data = False
    s.mesh_analysis          = True
    s.additional_logging     = False
    s.additional_logging_dev = False
    s.dask_enabled           = False
    s.resave_omezarr         = False
    s.image_restore          = False
    s.axial_restore          = False
    s.Track                  = False
    s.shape_error            = False
    s.Vaa3d                  = False
    s.Vaa3Dpath              = ''
    s.use_vox_measurements   = False
    s.analysis_method        = 'Whole Neuron'
    s.HistMatch              = False
    s.checkmem               = False
    s.patch_for_nnunet       = False
    s.nnunet_patch_size      = (64, 512, 512)
    s.nnunet_stride          = (57, 460, 460)
    s.reg_method             = 'rigid'
    s.nnUnet_conda_path      = ''
    s.nnUnet_env             = ''
    s.nnUnet_env_path        = ''
    s.nnUnet_type            = '3d_fullres'
    s.neuron_seg_model_path  = MODEL_DIR
    s.basepath               = out_dir
    s.internal_py_path       = ''
    s.selfnet_inference_script = ''
    s.selfnet_training_script  = ''
    s.nnunet_predict_bat     = ''
    s.clean_launcher         = ''
    s.refinement_model_path  = None
    s.prev_labels            = True
    s.original_shape         = [None]
    s.filename               = ''

    return s


def _make_locations(out_dir):
    """Build a locations SimpleNamespace with trailing-slash paths."""
    root = out_dir.replace('\\', '/')
    if not root.endswith('/'):
        root += '/'

    loc = types.SimpleNamespace()
    loc.input_dir = root + 'images/'
    loc.labels    = root + 'labels/'
    loc.tables    = root + 'Tables/'
    loc.MIPs      = root + 'Validation_Data/Validation_MIPs/'
    loc.Vols      = root + 'Validation_Data/Validation_Vols/'
    loc.swcs      = root + 'SWC_files/'
    loc.arrays    = root + 'Spine_Arrays/'
    loc.Meshes    = root + 'Validation_Data/Spine_Meshes/'
    return loc


def run_analysis(out_dir, cfg, logger):
    """Import and run RESPAN analyze_spines() for images in out_dir/images/."""
    from unittest.mock import MagicMock
    sys.modules.setdefault('sitecustomize', MagicMock())

    for repo in [RESPAN_ROOT, NNUNET_REPO]:
        if repo not in sys.path:
            sys.path.insert(0, repo)

    settings  = _make_settings(cfg, out_dir)
    locations = _make_locations(out_dir)

    for path in [locations.tables, locations.MIPs, locations.Vols,
                 locations.swcs, locations.arrays, locations.Meshes]:
        os.makedirs(path, exist_ok=True)

    import traceback
    try:
        from RESPAN.ImageAnalysis.ImageAnalysis import analyze_spines
    except Exception:
        print('\nERROR: RESPAN import failed:')
        traceback.print_exc()
        sys.exit(1)

    log = ''
    try:
        analyze_spines(settings, locations, log, logger)
    except Exception:
        print('\nERROR: analyze_spines() failed:')
        traceback.print_exc()
        sys.exit(1)


# ---------------------------------------------------------------------------
# CuPy check
# ---------------------------------------------------------------------------

def check_cupy():
    try:
        import cupy as cp
        cp.unique(cp.array([1, 2, 3]))
    except Exception as e:
        print('\n' + '!' * 65)
        print('ERROR: CuPy GPU check failed.')
        print(f'  {e}')
        print('\nFIX: run this script with the respandev Python:')
        if _ON_MAHTI:
            _respandev = '/projappl/project_2019059/respan/envs/respandev/bin/python'
            print(f'  {_respandev} {__file__}')
        else:
            print(f'  C:\\Users\\ARASH\\Anaconda3\\envs\\respandev\\python.exe {__file__}')
        print('!' * 65 + '\n')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(img_path, out_dir, yaml_src, cfg, checkpoint_name='checkpoint_best.pth'):
    """
    Run the full pipeline for one image.

    out_dir layout:
        images/         <- hardlink to original TIF
        nnunet_input/   <- prepared single-channel TIF
        labels/         <- nnU-Net output
        Tables/         <- CSVs
        Spine_Arrays/   <- sub-volume MIPs/vols
        Validation_Data/Validation_MIPs/
        SWC_files/
        RESPAN_analysis.log
        Analysis_Settings.yaml
    """
    stem = Path(img_path).stem
    print(f'\n{"="*65}')
    print(f'  Image : {Path(img_path).name}')
    print(f'  Output: {out_dir}')
    print(f'{"="*65}')

    img_link_dir = os.path.join(out_dir, 'images')
    img_link     = os.path.join(img_link_dir, Path(img_path).name)
    prep_dir     = os.path.join(out_dir, 'nnunet_input')
    labels_dir   = os.path.join(out_dir, 'labels')

    os.makedirs(img_link_dir, exist_ok=True)
    os.makedirs(prep_dir,     exist_ok=True)
    os.makedirs(labels_dir,   exist_ok=True)

    # Link original image into per-image images/ so analyze_spines sees only this file
    if not os.path.exists(img_link):
        try:
            os.link(img_path, img_link)      # hardlink (same volume, no extra space)
        except OSError:
            shutil.copy2(img_path, img_link) # fallback: copy

    # --- Step 1: prepare ---
    print('\n[1/3]  Preparing image for nnU-Net...')
    in_xy = float(cfg.get('input_resXY', 0.102))
    in_z  = float(cfg.get('input_resZ',  0.200))
    prepare_image(img_link, prep_dir, in_xy, in_z)

    # --- Step 2: nnU-Net ---
    label_file = os.path.join(labels_dir, f'{stem}.tif')
    if os.path.isfile(label_file):
        print(f'\n[2/3]  Labels exist -- skipping nnU-Net.')
        print(f'       (delete {labels_dir} to re-run segmentation)')
    else:
        print('\n[2/3]  Running nnU-Net segmentation...')
        run_nnunet(prep_dir, labels_dir, checkpoint_name)

    # --- Step 3: RESPAN analysis ---
    print('\n[3/3]  Running RESPAN spine analysis...')
    log_path = os.path.join(out_dir, 'RESPAN_analysis.log')
    logger   = logging.getLogger(f'RESPAN.{stem}')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    run_analysis(out_dir, cfg, logger)

    # --- Copy YAML ---
    shutil.copy2(yaml_src, os.path.join(out_dir, 'Analysis_Settings.yaml'))

    print(f'\n  Done.  Results in: {out_dir}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    import re

    parser = argparse.ArgumentParser(
        description='RESPAN Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--model', default=None, metavar='PATH',
        help='Path to nnU-Net model folder (e.g. Dataset226_... or Dataset001_Spines). '
             'Overrides the default MODEL_DIR. Dataset ID is parsed from the folder name.',
    )
    parser.add_argument(
        'datasets', nargs='*',
        help='Dataset folder names to process (default: all found under images/).',
    )
    args = parser.parse_args()

    filter_names = args.datasets or None

    # Allow overriding the model from the command line
    global MODEL_DIR, NNUNET_RESULTS, DATASET_ID
    if args.model:
        MODEL_DIR      = str(Path(args.model).resolve())
        NNUNET_RESULTS = str(Path(MODEL_DIR).parent)
        m = re.match(r'Dataset(\d+)', Path(MODEL_DIR).name)
        if not m:
            print(f'ERROR: --model folder name must start with Dataset<ID>_...')
            print(f'  Got: {Path(MODEL_DIR).name}')
            sys.exit(1)
        DATASET_ID = m.group(1)

    print('=' * 65)
    print('RESPAN Unified Pipeline')
    print(f'Scanning: {DATASETS_ROOT}')
    print(f'Model   : {MODEL_DIR}')
    print('=' * 65)

    check_cupy()

    if not os.path.isdir(MODEL_DIR):
        print(f'ERROR: Model not found:\n  {MODEL_DIR}')
        sys.exit(1)

    for _ckpt in ('checkpoint_best.pth', 'checkpoint_final.pth'):
        checkpoint = os.path.join(
            MODEL_DIR, 'nnUNetTrainer__nnUNetPlans__3d_fullres', 'fold_all', _ckpt)
        if os.path.isfile(checkpoint):
            break
    else:
        print(f'ERROR: no checkpoint found in:\n  {MODEL_DIR}')
        sys.exit(1)

    datasets = find_datasets(DATASETS_ROOT, filter_names)
    if not datasets:
        print('\nNo dataset folders found.')
        print('Each dataset folder needs:')
        print('  Analysis_Settings.yaml   <- resolution config')
        print('  TIF files (directly in folder OR inside an images/ subfolder)')
        sys.exit(0)

    print(f'\nFound {len(datasets)} dataset folder(s):')
    for d, _, tifs in datasets:
        print(f'  {os.path.basename(d)}  ({len(tifs)} image(s))')

    total = sum(len(tifs) for _, _, tifs in datasets)
    print(f'\n{total} image(s) to process.\n')

    for dataset_dir, yaml_path, tif_paths in datasets:
        cfg = load_yaml(yaml_path)

        print(f'\n{"#"*65}')
        print(f'  Dataset: {os.path.basename(dataset_dir)}')
        print(f'  YAML   : {yaml_path}')
        print(f'  Images : {len(tif_paths)}')
        print(f'{"#"*65}')

        for img_path in tif_paths:
            stem    = Path(img_path).stem
            out_dir = os.path.join(dataset_dir, stem)
            process_image(img_path, out_dir, yaml_path, cfg, _ckpt)

    print('=' * 65)
    print('All datasets complete.')
    print('=' * 65)


if __name__ == '__main__':
    main()
