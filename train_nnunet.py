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
FINE-TUNING  (start from an existing model's weights)
──────────────────────────────────────────────────────────────────
    python train_nnunet.py finetune                       # from bundled Dataset226
    python train_nnunet.py finetune --name ZeissV1        # name the result; ID auto-assigned
    python train_nnunet.py finetune --from nnunet_training/results/Dataset002_Spines --name ZeissV2

    --from  = model folder to fine-tune FROM (any Dataset###_ folder with plans.json
              + fold_all/checkpoint). Default: bundled Dataset226 (Model 1Bv2).
    --name  = output dataset name suffix (result becomes Dataset###_<name>).
    --id    = force the output ID number; omit it and the next free number is used.

    (finetune = setup[FINETUNE_LABELS] -> finetune_plan -> finetune_train; the
     individual steps still exist as finetune_plan / finetune_train.)

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
_BASE = Path(__file__).resolve().parent          # shared CODE location (read-only)

# Per-user working directory for DATA + OUTPUTS. Set RESPAN_WORKDIR so several
# people share this one code checkout but keep their training_data and trained
# models completely separate (no interference). Defaults to the code dir, so
# single-user behaviour is unchanged.
_WORKDIR = Path(os.environ.get('RESPAN_WORKDIR', _BASE)).resolve()

TRAINING_DATA = str(_WORKDIR / 'training_data')
NNUNET_BASE   = str(_WORKDIR / 'nnunet_training')

DATASET_ID   = 1
DATASET_NAME = f'Dataset{DATASET_ID:03d}_Spines'

# Locate the 'respaninternal' env's Python portably (no hardcoded project/user):
#   1. $RESPAN_RESPANINTERNAL_PYTHON override, else
#   2. a sibling env 'respaninternal' next to the launching interpreter
#      (works on Windows conda AND Roihu Tykky), else
#   3. author-machine fallbacks.
def _sibling_env_python(env_name):
    for parent in Path(sys.executable).parents:
        if parent.name == 'envs':
            for cand in (parent / env_name / 'bin' / 'python',
                         parent / env_name / 'python.exe'):
                if cand.exists():
                    return str(cand)
            break
    return None

_FALLBACK_INTERNAL = [
    '/projappl/project_2019059/respan/envs/respaninternal/bin/python',
    r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe',
]
RESPANINTERNAL_PYTHON = (
    os.environ.get('RESPAN_RESPANINTERNAL_PYTHON')
    or _sibling_env_python('respaninternal')
    or next((f for f in _FALLBACK_INTERNAL if os.path.exists(f)), _FALLBACK_INTERNAL[-1])
)
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

_ON_MAHTI = os.path.isdir('/projappl')   # CSC Tykky: run nnU-Net directly, not nested


def _claim_free_dataset_id(suffix):
    """Atomically pick + claim the next free Dataset ID.

    Scans raw/preprocessed/results for used numbers, then *claims* a number by
    creating its raw dataset dir with exist_ok=False (an atomic filesystem op).
    If two runs race for the same number, only one mkdir wins; the loser bumps
    to the next number and retries. Returns (id, name, dataset_dir).
    """
    import re
    os.makedirs(RAW_DIR, exist_ok=True)
    used = set()
    for base in (RAW_DIR, PREPROCESSED_DIR, RESULTS_DIR):
        if os.path.isdir(base):
            for name in os.listdir(base):
                m = re.match(r'Dataset(\d+)_', name)
                if m:
                    used.add(int(m.group(1)))
    n = 1
    while True:
        while n in used:
            n += 1
        name = f'Dataset{n:03d}_{suffix}'
        dataset_dir = os.path.join(RAW_DIR, name)
        try:
            os.makedirs(dataset_dir, exist_ok=False)   # atomic claim
            return n, name, dataset_dir
        except FileExistsError:
            used.add(n)                                 # someone else grabbed it
            n += 1


def _nnunet_env():
    env = os.environ.copy()
    env['nnUNet_raw']          = RAW_DIR
    env['nnUNet_preprocessed'] = PREPROCESSED_DIR
    env['nnUNet_results']      = RESULTS_DIR
    env['PYTHONNOUSERSITE']    = '1'
    # Put the nnU-Net repo on PYTHONPATH so 'spawn'ed worker processes (fresh
    # interpreters that don't inherit the runtime sys.path.insert) can import it.
    _pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = NNUNET_REPO + (os.pathsep + _pp if _pp else '')
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


def _patch_torch_load():
    """
    PyTorch >=2.6 changed torch.load's default to weights_only=True, which
    rejects the numpy objects stored in nnU-Net / RESPAN checkpoints. Those
    checkpoints are trusted (our own / RESPAN's models), so default
    weights_only=False for every nnU-Net load. No-op on older PyTorch.
    """
    try:
        import torch
        import functools
    except ImportError:
        return
    if getattr(torch.load, '_respan_patched', False):
        return
    _orig_load = torch.load

    @functools.wraps(_orig_load)
    def _load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_load(*args, **kwargs)

    _load._respan_patched = True
    torch.load = _load


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
        _patch_torch_load()
        # Use 'spawn' for worker processes. The default 'fork' deadlocks once
        # CUDA is initialized in the parent (nnU-Net's dataloader workers hang
        # at epoch 0), so spawn is required to run data augmentation in parallel.
        import multiprocessing as _mp
        try:
            _mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
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
            f"import multiprocessing as _mp\n"
            f"_mp.set_start_method('spawn', force=True)\n"
            # PyTorch >=2.6: default weights_only=False so trusted nnU-Net /
            # RESPAN checkpoints (with numpy objects) still load.
            f"import torch, functools\n"
            f"_orig_load = torch.load\n"
            f"@functools.wraps(_orig_load)\n"
            f"def _load(*a, **k):\n"
            f"    k.setdefault('weights_only', False)\n"
            f"    return _orig_load(*a, **k)\n"
            f"torch.load = _load\n"
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
    Transfer plans from the SOURCE model (default Dataset226 / Model 1Bv2, or
    whatever --from points at) and preprocess the dataset.
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
    import re

    parser = argparse.ArgumentParser(description='nnU-Net Training Pipeline')
    parser.add_argument('step', nargs='?', default='help',
                        choices=['setup', 'plan', 'train', 'all',
                                 'finetune_plan', 'finetune_train', 'finetune', 'help'])
    parser.add_argument('--name', type=str, default=None,
                        help='Output dataset name suffix (e.g. ZeissV1). '
                             'Result becomes Dataset###_<name>. Default: Spines.')
    parser.add_argument('--id',   type=int, default=None,
                        help='Force the output dataset ID number. Omit it and the next '
                             'free number is used (when --name is given).')
    parser.add_argument('--from', dest='from_model', type=str, default=None,
                        help='Model folder to fine-tune FROM (any Dataset###_ folder with '
                             'plans.json + fold_all/checkpoint). Default: bundled Dataset226.')
    args = parser.parse_args()

    # --- OUTPUT dataset: pick name + id ----------------------------------------
    suffix = args.name or 'Spines'
    if args.id is not None:
        DATASET_ID   = args.id                     # explicit number wins
        DATASET_NAME = f'Dataset{DATASET_ID:03d}_{suffix}'
        DATASET_DIR  = os.path.join(RAW_DIR, DATASET_NAME)
    elif args.name is not None:
        # name given, no id -> atomically claim a fresh number (concurrency-safe)
        DATASET_ID, DATASET_NAME, DATASET_DIR = _claim_free_dataset_id(suffix)
    else:
        # keep the module default (1) unchanged
        DATASET_NAME = f'Dataset{DATASET_ID:03d}_{suffix}'
        DATASET_DIR  = os.path.join(RAW_DIR, DATASET_NAME)
    IMAGES_TR    = os.path.join(DATASET_DIR, 'imagesTr')
    LABELS_TR    = os.path.join(DATASET_DIR, 'labelsTr')

    # --- SOURCE model to fine-tune FROM (only relevant for finetune steps) ------
    if args.from_model is not None:
        SOURCE_MODEL_DIR = str(Path(args.from_model).resolve())
        m = re.match(r'Dataset(\d+)', Path(SOURCE_MODEL_DIR).name)
        if not m:
            print(f"ERROR: --from folder name must start with 'Dataset<ID>_': "
                  f"{Path(SOURCE_MODEL_DIR).name}")
            sys.exit(1)
        SOURCE_DATASET_ID  = int(m.group(1))
        SOURCE_TRAINER_DIR = os.path.join(SOURCE_MODEL_DIR,
                                          'nnUNetTrainer__nnUNetPlans__3d_fullres')
        for _ck in ('checkpoint_final.pth', 'checkpoint_best.pth', 'checkpoint_latest.pth'):
            _cand = os.path.join(SOURCE_TRAINER_DIR, 'fold_all', _ck)
            if os.path.isfile(_cand):
                PRETRAINED_WEIGHTS = _cand
                break
        else:
            print(f'ERROR: no checkpoint found in '
                  f'{os.path.join(SOURCE_TRAINER_DIR, "fold_all")}')
            sys.exit(1)

    if args.step in ('finetune', 'finetune_plan', 'finetune_train'):
        print(f'Fine-tune FROM : {Path(SOURCE_MODEL_DIR).name}')
        print(f'  weights      : {PRETRAINED_WEIGHTS}')
    if args.step in ('setup', 'plan', 'train', 'all',
                     'finetune', 'finetune_plan', 'finetune_train'):
        print(f'Output dataset : {DATASET_NAME}')

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
