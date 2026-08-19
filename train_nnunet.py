"""
nnU-Net Training Pipeline (via respaninternal)
==============================================
Sets up an nnU-Net dataset from the training_data/ folder (next to this script)
and runs plan_and_preprocess + training, exactly as RESPAN does internally.

Dataset layout expected in TRAINING_DATA:
    imagesTr/   <image_name>.tif   -- raw fluorescence images (single channel)
    labelsTr/   <image_name>.tif   -- matching segmentation labels (uint8)

Output layout (created automatically):
    NNUNET_BASE/
        raw/           Dataset001_Spines/imagesTr, labelsTr, dataset.json
        preprocessed/  (created by plan/finetune_plan)
        results/       (created by training)

──────────────────────────────────────────────────────────────────
FROM-SCRATCH TRAINING  (new model, your own label scheme)
──────────────────────────────────────────────────────────────────
    python train_nnunet.py setup          # copy + rename training data
    python train_nnunet.py plan           # fingerprint + plan + preprocess
    python train_nnunet.py train          # train 3d_fullres fold=all
    python train_nnunet.py all            # all three steps in sequence

──────────────────────────────────────────────────────────────────
FINE-TUNING  (start from RESPAN Model 1Bv2 pretrained weights)
──────────────────────────────────────────────────────────────────
    python train_nnunet.py setup          # same setup step (uses FINETUNE_LABELS)
    python train_nnunet.py finetune_plan  # transfer plans from Dataset226 + preprocess
    python train_nnunet.py finetune_train # train with -pretrained_weights
    python train_nnunet.py finetune       # all three fine-tune steps in sequence

Fine-tuning label scheme (FINETUNE_LABELS below):
    0=background  1=dendrite  2=spine_core  3=spine_edge  4=spine_neck

Fine-tuning epochs: nnU-Net defaults to 1000 (calibrated for from-scratch).
For fine-tuning, stop early with Ctrl+C (recommended 300-500 epochs for
10-20 cases), then promote the checkpoint so RESPAN can find it:
    copy nnunet_training/results/.../checkpoint_latest.pth
         nnunet_training/results/.../checkpoint_final.pth
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

DATASET_ID   = 1
DATASET_NAME = f'Dataset{DATASET_ID:03d}_Spines'

_mahti_py             = '/projappl/project_2019059/respan/envs/respaninternal/bin/python'
_win_py               = r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe'
RESPANINTERNAL_PYTHON = _mahti_py if os.path.exists(_mahti_py) else _win_py
NNUNET_REPO           = str(_BASE / 'nnUNet')

# Label scheme for from-scratch training
LABELS = {
    "background": 0,
    "spine":      1,
    "dendrite":   2,
    "soma":       3,
}

# Label scheme for fine-tuning — must match Dataset226 integers exactly,
# only annotate the structures you care about, leave the rest as 0.
FINETUNE_LABELS = {
    "background":  0,
    "dendrite":    1,
    "spine_core":  2,
    "spine_edge":  3,
    "spine_neck":  4,
    "soma":        5,
    "axon":        6,
}

# ---------------------------------------------------------------------------
# Fine-tuning source — RESPAN Model 1Bv2 (Dataset226)
# ---------------------------------------------------------------------------
SOURCE_DATASET_ID  = 226
SOURCE_MODEL_DIR   = str(_BASE / 'cloned_respan' /
    'Dataset226_Global1Bv2_XY65nm_Z150nm_CARE-Aug_spinecore_neck_dend_soma_axon_V6')
SOURCE_TRAINER_DIR = os.path.join(SOURCE_MODEL_DIR, 'nnUNetTrainer__nnUNetPlans__3d_fullres')
PRETRAINED_WEIGHTS = os.path.join(SOURCE_TRAINER_DIR, 'fold_all', 'checkpoint_final.pth')

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


def _run_cmd(argv, entry_module, entry_func, label):
    """
    Run any nnUNet entry point.
    Mahti: import and call directly (no subprocess nesting).
    Windows: run as subprocess via _run().
    """
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    if _ON_MAHTI:
        _set_nnunet_env()
        sys.path.insert(0, NNUNET_REPO)
        sys.argv = argv
        import importlib
        mod = importlib.import_module(entry_module)
        getattr(mod, entry_func)()
    else:
        runner = (
            f"import sys\n"
            f"sys.path.insert(0, r'{NNUNET_REPO}')\n"
            f"sys.argv = {argv!r}\n"
            f"from {entry_module} import {entry_func}\n"
            f"{entry_func}()\n"
        )
        _run(runner, label)
    print(f'  Done.')


# ---------------------------------------------------------------------------
# Step 1: setup
# ---------------------------------------------------------------------------
def setup(labels=None):
    """Copy+rename training data into proper nnU-Net raw dataset structure."""
    if labels is None:
        labels = LABELS

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
    for i, img_path in enumerate(tifs):
        case_id   = f'case{i+1:03d}'
        dst_image = os.path.join(IMAGES_TR, f'{case_id}_0000.tif')
        dst_label = os.path.join(LABELS_TR, f'{case_id}.tif')

        lbl_path = src_labels / img_path.name
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

    dataset_json = {
        "channel_names":                 {"0": "fluor"},
        "file_ending":                   ".tif",
        "labels":                        labels,
        "numTraining":                   len(tifs),
        "overwrite_image_reader_writer": "Tiff3DIO",
    }
    dst_json = os.path.join(DATASET_DIR, 'dataset.json')
    with open(dst_json, 'w') as f:
        json.dump(dataset_json, f, indent=4)

    print(f'\nDataset written to: {DATASET_DIR}')
    print(f'  dataset.json labels: {list(labels.keys())}')
    print(f'\nNext: run with argument "plan" (from-scratch) or "finetune_plan" (fine-tune)')


# ---------------------------------------------------------------------------
# Step 2a: plan and preprocess  (from-scratch)
# ---------------------------------------------------------------------------
def plan():
    """Run nnUNetv2_plan_and_preprocess for the dataset."""
    _run_cmd(
        argv=[
            'nnUNetv2_plan_and_preprocess',
            '-d', str(DATASET_ID),
            '--verify_dataset_integrity',
            '-c', '3d_fullres',
        ],
        entry_module='nnunetv2.experiment_planning.plan_and_preprocess_entrypoints',
        entry_func='plan_and_preprocess_entry',
        label=f'plan_and_preprocess  Dataset{DATASET_ID:03d}',
    )
    print(f'\nNext: run with argument "train"')


# ---------------------------------------------------------------------------
# Step 2b: transfer plans + preprocess  (fine-tune)
# ---------------------------------------------------------------------------
def finetune_plan():
    """
    Transfer plans from Dataset226 (Model 1Bv2) and preprocess the dataset.
    Replaces the 'plan' step when fine-tuning.
    """
    # Validate pretrained model exists
    if not os.path.isfile(PRETRAINED_WEIGHTS):
        print(f'ERROR: pretrained weights not found:\n  {PRETRAINED_WEIGHTS}')
        sys.exit(1)

    src_name = Path(SOURCE_MODEL_DIR).name

    # A. Copy Dataset226 into nnUNet_results so move_plans can find it by ID
    dst_results_src = os.path.join(RESULTS_DIR, src_name)
    if not os.path.exists(dst_results_src):
        print(f'\n  Copying Model 1Bv2 into nnUNet_results...')
        shutil.copytree(SOURCE_MODEL_DIR, dst_results_src)
        print(f'  -> {dst_results_src}')
    else:
        print(f'\n  Source model already in nnUNet_results, skipping copy.')

    # B. Copy plans.json into nnUNet_preprocessed/Dataset226.../nnUNetPlans.json
    #    so move_plans can read it from there
    src_plans         = os.path.join(SOURCE_TRAINER_DIR, 'plans.json')
    dst_preproc_src   = os.path.join(PREPROCESSED_DIR, src_name)
    dst_plans_json    = os.path.join(dst_preproc_src, 'nnUNetPlans.json')
    os.makedirs(dst_preproc_src, exist_ok=True)
    shutil.copy2(src_plans, dst_plans_json)
    print(f'  Copied plans.json -> {dst_plans_json}')

    # C. Extract fingerprint of our target dataset
    _run_cmd(
        argv=['nnUNetv2_extract_fingerprint', '-d', str(DATASET_ID)],
        entry_module='nnunetv2.experiment_planning.plan_and_preprocess_entrypoints',
        entry_func='extract_fingerprint_entry',
        label=f'extract_fingerprint  Dataset{DATASET_ID:03d}',
    )

    # D. Transfer plans from Dataset226 to our dataset
    _run_cmd(
        argv=[
            'nnUNetv2_move_plans_between_datasets',
            '-s', str(SOURCE_DATASET_ID),
            '-t', str(DATASET_ID),
            '-sp', 'nnUNetPlans',
            '-tp', 'nnUNetPlans',
        ],
        entry_module='nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets',
        entry_func='entry_point_move_plans_between_datasets',
        label=f'move_plans  Dataset{SOURCE_DATASET_ID:03d} -> Dataset{DATASET_ID:03d}',
    )

    # E. Copy dataset.json to preprocessed folder.
    #    In a normal plan_and_preprocess run nnU-Net does this automatically.
    #    In our split workflow (extract_fingerprint → move_plans → preprocess)
    #    the copy is missing, so we do it explicitly here.
    dst_preproc_dataset = os.path.join(PREPROCESSED_DIR, DATASET_NAME, 'dataset.json')
    shutil.copy2(os.path.join(DATASET_DIR, 'dataset.json'), dst_preproc_dataset)
    print(f'  Copied dataset.json -> {dst_preproc_dataset}')

    # F. Preprocess using the transferred plans (not a fresh fingerprint-based plan)
    _run_cmd(
        argv=[
            'nnUNetv2_preprocess',
            '-d', str(DATASET_ID),
            '-plans_name', 'nnUNetPlans',
            '-c', '3d_fullres',
            '-np', '1',
            '--verify_dataset_integrity',
        ],
        entry_module='nnunetv2.experiment_planning.plan_and_preprocess_entrypoints',
        entry_func='preprocess_entry',
        label=f'preprocess  Dataset{DATASET_ID:03d}  (transferred plans)',
    )

    print(f'\nNext: run with argument "finetune_train"')
    print(f'  Recommended: 300-500 epochs. Stop with Ctrl+C, then promote:')
    print(f'    checkpoint_latest.pth -> checkpoint_final.pth')


# ---------------------------------------------------------------------------
# Step 3a: train  (from-scratch)
# ---------------------------------------------------------------------------
def train():
    """Run nnUNetv2_train  3d_fullres  fold=all."""
    _run_cmd(
        argv=[
            'nnUNetv2_train',
            str(DATASET_ID),
            '3d_fullres',
            'all',
            '--npz',
        ],
        entry_module='nnunetv2.run.run_training',
        entry_func='run_training_entry',
        label=f'train  Dataset{DATASET_ID:03d}  3d_fullres  fold=all',
    )


# ---------------------------------------------------------------------------
# Step 3b: train with pretrained weights  (fine-tune)
# ---------------------------------------------------------------------------
def finetune_train():
    """Run nnUNetv2_train with -pretrained_weights pointing to Model 1Bv2."""
    if not os.path.isfile(PRETRAINED_WEIGHTS):
        print(f'ERROR: pretrained weights not found:\n  {PRETRAINED_WEIGHTS}')
        sys.exit(1)

    print(f'\n  Pretrained weights: {PRETRAINED_WEIGHTS}')
    print(f'  Stop early with Ctrl+C (recommended: 300-500 epochs for 10-20 cases).')
    print(f'  Progress curve saved to: nnunet_training/results/{DATASET_NAME}/'
          f'nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/progress.png')

    _run_cmd(
        argv=[
            'nnUNetv2_train',
            str(DATASET_ID),
            '3d_fullres',
            'all',
            '--npz',
            '-pretrained_weights', PRETRAINED_WEIGHTS,
        ],
        entry_module='nnunetv2.run.run_training',
        entry_func='run_training_entry',
        label=f'finetune  Dataset{DATASET_ID:03d}  3d_fullres  fold=all  (from Model 1Bv2)',
    )

    fold_dir = os.path.join(
        RESULTS_DIR, DATASET_NAME,
        'nnUNetTrainer__nnUNetPlans__3d_fullres', 'fold_all',
    )
    latest = os.path.join(fold_dir, 'checkpoint_latest.pth')
    final  = os.path.join(fold_dir, 'checkpoint_final.pth')
    print(f'\n  If you stopped early, promote the checkpoint so RESPAN can find it:')
    print(f'    copy  {latest}')
    print(f'      ->  {final}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='nnU-Net Training Pipeline')
    parser.add_argument('step', nargs='?', default='help',
                        choices=['setup', 'plan', 'train', 'all',
                                 'finetune_plan', 'finetune_train', 'finetune', 'help'])
    parser.add_argument('--id',   type=int, default=None,
                        help='Dataset ID (e.g. 42). Default: 1')
    parser.add_argument('--name', type=str, default=None,
                        help='Dataset name suffix (e.g. MySpines). Default: Spines')
    args = parser.parse_args()

    if args.id is not None:
        DATASET_ID   = args.id
        DATASET_NAME = f'Dataset{DATASET_ID:03d}_{args.name or "Spines"}'
        DATASET_DIR  = os.path.join(RAW_DIR, DATASET_NAME)
        IMAGES_TR    = os.path.join(DATASET_DIR, 'imagesTr')
        LABELS_TR    = os.path.join(DATASET_DIR, 'labelsTr')
    elif args.name is not None:
        DATASET_NAME = f'Dataset{DATASET_ID:03d}_{args.name}'
        DATASET_DIR  = os.path.join(RAW_DIR, DATASET_NAME)
        IMAGES_TR    = os.path.join(DATASET_DIR, 'imagesTr')
        LABELS_TR    = os.path.join(DATASET_DIR, 'labelsTr')

    step = args.step
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
    elif step == 'finetune_plan':
        finetune_plan()
    elif step == 'finetune_train':
        finetune_train()
    elif step == 'finetune':
        setup(labels=FINETUNE_LABELS)
        finetune_plan()
        finetune_train()
    else:
        print(__doc__)
