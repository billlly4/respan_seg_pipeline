"""
nnU-Net Training Pipeline (via respaninternal)
==============================================
Sets up an nnU-Net dataset from f:/ResPan/training_data/ and runs
plan_and_preprocess + training, exactly as RESPAN does internally.

Dataset layout expected in TRAINING_DATA:
    imagesTr/   <image_name>.tif   -- raw fluorescence images (single channel)
    labelsTr/   <image_name>.tif   -- matching segmentation labels (uint8)

Output layout (created automatically):
    NNUNET_BASE/
        raw/           Dataset001_Spines/imagesTr, labelsTr, dataset.json
        preprocessed/  (created by plan_and_preprocess)
        results/       (created by training)

HOW TO RUN (three steps, one at a time):
    Step 1 - set up raw dataset:
        C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_nnunet.py  setup

    Step 2 - fingerprint + preprocess:
        C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_nnunet.py  plan

    Step 3 - train (3d_fullres, fold all):
        C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_nnunet.py  train

Use 'all' to run all three steps in sequence:
        C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_nnunet.py  all
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import tifffile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent

TRAINING_DATA = str(_BASE / 'training_data')
NNUNET_BASE   = str(_BASE / 'nnunet_training')

DATASET_ID    = 1
DATASET_NAME  = f'Dataset{DATASET_ID:03d}_Spines'

_mahti_py             = '/projappl/project_2019059/respan/envs/respaninternal/bin/python'
_win_py               = r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe'
RESPANINTERNAL_PYTHON = _mahti_py if os.path.exists(_mahti_py) else _win_py
NNUNET_REPO           = str(_BASE / 'nnUNet')

# Label scheme — matches Dataset219 (Model 1A) for future compatibility
LABELS = {
    "background": 0,
    "spine":      1,
    "dendrite":   2,
    "soma":       3,
}

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------
RAW_DIR          = os.path.join(NNUNET_BASE, 'raw')
PREPROCESSED_DIR = os.path.join(NNUNET_BASE, 'preprocessed')
RESULTS_DIR      = os.path.join(NNUNET_BASE, 'results')
DATASET_DIR      = os.path.join(RAW_DIR, DATASET_NAME)
IMAGES_TR        = os.path.join(DATASET_DIR, 'imagesTr')
LABELS_TR        = os.path.join(DATASET_DIR, 'labelsTr')


_ON_MAHTI = os.path.exists(_mahti_py)


def _nnunet_env():
    env = os.environ.copy()
    env['nnUNet_raw']          = RAW_DIR
    env['nnUNet_preprocessed'] = PREPROCESSED_DIR
    env['nnUNet_results']      = RESULTS_DIR
    env['PYTHONNOUSERSITE']    = '1'
    return env


def _set_nnunet_env():
    """Apply nnUNet env vars to the current process (Mahti direct-call path)."""
    for k, v in _nnunet_env().items():
        os.environ[k] = v


def _run(runner_code, label):
    """Run a Python code block in respaninternal with nnU-Net env vars set."""
    env = _nnunet_env()
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    result = subprocess.run(
        [RESPANINTERNAL_PYTHON, '-u', '-c', runner_code],
        env=env,
    )
    if result.returncode != 0:
        print(f'\nERROR: {label} exited with code {result.returncode}')
        sys.exit(result.returncode)
    print(f'  Done.')


# ---------------------------------------------------------------------------
# Step 1: setup
# ---------------------------------------------------------------------------
def setup():
    """Copy+rename training data into proper nnU-Net raw dataset structure."""
    os.makedirs(IMAGES_TR, exist_ok=True)
    os.makedirs(LABELS_TR, exist_ok=True)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    src_images = Path(TRAINING_DATA) / 'imagesTr'
    src_labels = Path(TRAINING_DATA) / 'labelsTr'

    tifs = sorted(f for f in src_images.iterdir() if f.suffix.lower() == '.tif')
    if not tifs:
        print(f'ERROR: no .tif files found in {src_images}')
        sys.exit(1)

    print(f'\nFound {len(tifs)} image(s):')
    case_map = {}  # original name -> case ID
    for i, img_path in enumerate(tifs):
        case_id   = f'case{i+1:03d}'
        dst_image = os.path.join(IMAGES_TR, f'{case_id}_0000.tif')
        dst_label = os.path.join(LABELS_TR, f'{case_id}.tif')

        lbl_path  = src_labels / img_path.name
        if not lbl_path.exists():
            print(f'ERROR: no matching label for {img_path.name}')
            print(f'  Expected: {lbl_path}')
            sys.exit(1)

        shutil.copy2(img_path, dst_image)
        shutil.copy2(lbl_path, dst_label)

        img = tifffile.imread(dst_image)
        lbl = tifffile.imread(dst_label)
        label_vals = sorted(np.unique(lbl).tolist())
        print(f'  {img_path.name}')
        print(f'    -> {case_id}_0000.tif  shape={img.shape}  dtype={img.dtype}')
        print(f'    -> {case_id}.tif       shape={lbl.shape}  dtype={lbl.dtype}  labels={label_vals}')
        case_map[img_path.name] = case_id

    # Write dataset.json
    dataset_json = {
        "channel_names":              {"0": "fluor"},
        "file_ending":                ".tif",
        "labels":                     LABELS,
        "numTraining":                len(tifs),
        "overwrite_image_reader_writer": "Tiff3DIO",
    }
    dst_json = os.path.join(DATASET_DIR, 'dataset.json')
    with open(dst_json, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f'\nDataset written to: {DATASET_DIR}')
    print(f'  dataset.json labels: {list(LABELS.keys())}')
    print(f'\nNext: run with argument "plan"')


# ---------------------------------------------------------------------------
# Step 2: plan and preprocess
# ---------------------------------------------------------------------------
def plan():
    """Run nnUNetv2_plan_and_preprocess for the dataset."""
    label = f'plan_and_preprocess  Dataset{DATASET_ID:03d}'
    print(f'\n{"="*60}\n  {label}\n{"="*60}')
    if _ON_MAHTI:
        _set_nnunet_env()
        sys.path.insert(0, NNUNET_REPO)
        sys.argv = [
            'nnUNetv2_plan_and_preprocess',
            '-d', str(DATASET_ID),
            '--verify_dataset_integrity',
            '-c', '3d_fullres',
        ]
        from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
        plan_and_preprocess_entry()
    else:
        runner = f"""
import sys
sys.path.insert(0, r'{NNUNET_REPO}')
sys.argv = [
    'nnUNetv2_plan_and_preprocess',
    '-d', '{DATASET_ID}',
    '--verify_dataset_integrity',
    '-c', '3d_fullres',
]
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
plan_and_preprocess_entry()
"""
        _run(runner, label)
    print(f'\nNext: run with argument "train"')


# ---------------------------------------------------------------------------
# Step 3: train
# ---------------------------------------------------------------------------
def train():
    """Run nnUNetv2_train  3d_fullres  fold=all."""
    label = f'train  Dataset{DATASET_ID:03d}  3d_fullres  fold=all'
    print(f'\n{"="*60}\n  {label}\n{"="*60}')
    if _ON_MAHTI:
        _set_nnunet_env()
        sys.path.insert(0, NNUNET_REPO)
        sys.argv = [
            'nnUNetv2_train',
            str(DATASET_ID),
            '3d_fullres',
            'all',
            '--npz',
        ]
        from nnunetv2.run.run_training import run_training_entry
        run_training_entry()
    else:
        runner = f"""
import sys
sys.path.insert(0, r'{NNUNET_REPO}')
sys.argv = [
    'nnUNetv2_train',
    '{DATASET_ID}',
    '3d_fullres',
    'all',
    '--npz',
]
from nnunetv2.run.run_training import run_training_entry
run_training_entry()
"""
        _run(runner, label)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if step == 'setup':
        setup()
    elif step == 'plan':
        plan()
    elif step == 'train':
        train()
    elif step == 'all':
        setup()
        plan()
        train()
    else:
        print(__doc__)
