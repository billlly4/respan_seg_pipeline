"""
nnU-Net segmentation step for RESPAN — Mahti only.

RESPAN's nnU-Net code is subprocess-based (designed for the GUI two-environment
architecture). On Mahti, tykky containers cannot be nested, so we call
nnUNetPredictor directly instead (once for all images, avoids the
torch.set_num_interop_threads re-call error when looping).

Run this with respaninternal BEFORE run_respan.py:

    /projappl/project_2019059/respan/envs/respaninternal/bin/python \
        /scratch/project_2019059/respan/code/ResPan/run_nnunet_only.py

Then run the analysis as normal with respandev:

    /projappl/project_2019059/respan/envs/respandev/bin/python \
        /scratch/project_2019059/respan/code/ResPan/run_respan.py

On Windows, this script is not needed — run_respan.py handles everything.
"""

import os
import sys
from pathlib import Path

# Add ResPan root to path so we can import from run_respan.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import shared functions and constants from run_respan
from run_respan import (
    DATASETS_ROOT, NNUNET_REPO, NNUNET_RESULTS, MODEL_DIR,
    load_yaml, find_datasets, prepare_image,
)

# Set nnUNet env vars BEFORE importing nnunetv2 (paths.py reads them at import time)
os.environ['nnUNet_results']       = NNUNET_RESULTS
os.environ['nnUNet_raw']           = 'not_used_for_inference'
os.environ['nnUNet_preprocessed']  = 'not_used_for_inference'
sys.path.insert(0, NNUNET_REPO)

# PyTorch 2.6 changed weights_only default to True, breaking checkpoints saved
# with older versions. Restore pre-2.6 behaviour for this inference-only process.
import torch as _torch
_orig_load = _torch.load
_torch.load = lambda *a, **kw: _orig_load(*a, **{**{'weights_only': False}, **kw})

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def labels_exist(labels_dir):
    return os.path.isdir(labels_dir) and any(
        f.endswith('.tif') for f in os.listdir(labels_dir)
    )


def make_predictor():
    model_folder = os.path.join(MODEL_DIR, 'nnUNetTrainer__nnUNetPlans__3d_fullres')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=_torch.device('cuda', 0),
        verbose=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=('all',),
        checkpoint_name='checkpoint_final.pth',
    )
    return predictor


def main():
    print('=' * 65)
    print('RESPAN — nnU-Net segmentation (respaninternal)')
    print(f'Scanning: {DATASETS_ROOT}')
    print('=' * 65)

    datasets = find_datasets(DATASETS_ROOT)
    if not datasets:
        print('No datasets found.')
        return

    # Phase 1: prepare all images that still need labels
    to_predict = []
    for dataset_dir, yaml_path, tif_paths in datasets:
        cfg   = load_yaml(yaml_path)
        in_xy = float(cfg.get('input_resXY', 0.102))
        in_z  = float(cfg.get('input_resZ',  0.200))

        for tif in tif_paths:
            stem       = Path(tif).stem
            out_dir    = os.path.join(dataset_dir, stem)
            prep_dir   = os.path.join(out_dir, 'nnunet_input')
            labels_dir = os.path.join(out_dir, 'labels')

            print(f'\n  {Path(tif).name}')

            if labels_exist(labels_dir):
                print('  Labels exist — skipping.')
                continue

            print('  Preparing input...')
            prepare_image(tif, prep_dir, in_xy, in_z)
            to_predict.append((tif, prep_dir, labels_dir))

    if not to_predict:
        print('\nAll images already have labels.')
        return

    # Phase 2: load model once, predict all images
    print('\n  Loading nnU-Net model...')
    predictor = make_predictor()

    for tif, prep_dir, labels_dir in to_predict:
        print(f'\n  Running nnU-Net: {Path(tif).name}')
        os.makedirs(labels_dir, exist_ok=True)
        predictor.predict_from_files(
            prep_dir,
            labels_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        for f in os.listdir(labels_dir):
            if not f.endswith('.tif'):
                os.remove(os.path.join(labels_dir, f))
        print(f'  Done -> {labels_dir}')

    print('\n' + '=' * 65)
    print('Segmentation complete.')
    print('Now run run_respan.py with respandev for the analysis.')


if __name__ == '__main__':
    main()
