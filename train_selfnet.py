"""
SelfNet Training Pipeline (via respaninternal / PyTorch)
========================================================
Trains a SelfNet (CycleGAN-based axial restoration) model from one or more 3D
z-stacks. No paired data needed — SelfNet is self-supervised: it learns from
the internal difference between the XY view (sharp, isotropic PSF) and the
XZ/YZ views (blurry/stretched due to anisotropic Z resolution).

How it works (3 internal steps inside SelfNet_Model_Training.py):
  Step 1  self_net_create_slices     Extract XY, XY-lowres, XZ, YZ 2D slices
  Step 2  generate_training_data     Crop 64x64 patches from those slices
  Step 3  train_self_net             Train the CycleGAN on those patches

HOW TO RUN (default: 280622 hippo stack, 3 test epochs):
    C:/Users/ARASH/Anaconda3/envs/respaninternal/python.exe  f:/ResPan/train_selfnet.py

OVERRIDE ANY PARAMETER FROM THE COMMAND LINE:
    python train_selfnet.py --epochs 40 --batch_size 8
    python train_selfnet.py --images f:/path/to/a.tif f:/path/to/b.tif --scale 0.41
    python train_selfnet.py --min_v 0 --max_v 255 --bg_threshold 10

PARAMETER MEANINGS:
    --images        One or more 3D .tif stacks to train on (space-separated)
                    Use multiple images to cover the full content diversity
                    (e.g. some with cell bodies, some with dendrites only).
                    All images must share the same scale / voxel size.
    --output        Folder where model checkpoints are saved
    --name          Model subfolder name inside --output
    --min_v         Minimum intensity for normalisation  (0 for uint8/uint16)
    --max_v         Maximum intensity for normalisation  (255 uint8 / 65535 uint16)
    --bg_threshold  Min intensity for patch selection — patches below this are
                    discarded as background.  Use ~10 for uint8, ~1000 for uint16
    --scale         XY pixel size / Z step size.
                    e.g. 0.082 um XY / 0.200 um Z  ->  0.082/0.200 = 0.41
                    e.g. 0.133 um XY / 0.410 um Z  ->  0.133/0.410 = 0.32
                    GUI label: "XY sampling (um) / Z step (um)".  Default 0.5
                    Z-plane requirement: original_Z >= ceil(64 * scale).
                    For scale=0.41 that means >= 27 Z planes.
    --xy_interval   Take every Nth XY (Z) slice for training  (default 4)
    --xz_interval   Take every Nth XZ slice (Y-position)      (default 8)
    --batch_size    Images per gradient step                   (default 8)
    --epochs        Training epochs                            (default 40)
    --log_interval  Print loss every N batches                 (default 1)
    --imshow_interval  Save sample image every N batches       (default 100)

NOTE: SelfNet is hardcoded to cuda:0 (PyTorch GPU). Runs in respaninternal.
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import runpy
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration  — defaults for fast pipeline test
# ---------------------------------------------------------------------------
# Best single stack: 38 Z planes (rescaled to 92px XZ >> 64px patch), 17/38 signal slices, scale=0.41
_BASE = Path(__file__).resolve().parent

DEFAULT_IMAGES = [str(_BASE / 'selfnet_training' / 'data' / 'extracted_ch2' / 'HippoDIV16_control_Series074_ch2.tif')]
DEFAULT_OUTPUT = str(_BASE / 'selfnet_training' / 'output')
DEFAULT_NAME   = 'selfnet_test'

SELFNET_SCRIPT = str(_BASE / 'cloned_respan' / 'RESPAN' / 'Scripts' / 'SelfNet_Model_Training.py')
NNUNET_REPO    = str(_BASE / 'nnUNet')

_mahti_py         = '/projappl/project_2019059/respan/envs/respaninternal/bin/python'
_win_py           = r'C:\Users\ARASH\Anaconda3\envs\respaninternal\python.exe'
RESPANINTERNAL_PY = _mahti_py if os.path.exists(_mahti_py) else _win_py
_ON_MAHTI         = os.path.exists(_mahti_py)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train a SelfNet axial-restoration model via RESPAN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--images',         default=DEFAULT_IMAGES,  nargs='+', help='One or more 3D .tif stacks to train on (space-separated paths)')
    p.add_argument('--output',         default=DEFAULT_OUTPUT,  help='Output folder for model checkpoints')
    p.add_argument('--name',           default=DEFAULT_NAME,    help='Model subfolder name inside --output')

    # Intensity normalisation
    p.add_argument('--min_v',          default=0,    type=int,   help='Minimum intensity for normalisation (0)')
    p.add_argument('--max_v',          default=255,  type=int,   help='Maximum intensity: 255 (uint8) or 65535 (uint16)')

    # Patch selection
    p.add_argument('--bg_threshold',   default=10,   type=int,   help='Min patch intensity — discard background patches (10 for uint8, 1000 for uint16)')

    # Slice sampling  — critical: sets Z/XY pixel ratio
    p.add_argument('--scale',          default=0.41, type=float, help='XY pixel size / Z step (e.g. 0.082/0.200 = 0.41)')
    p.add_argument('--xy_interval',    default=4,    type=int,   help='Use every Nth Z-plane as XY training slice')
    p.add_argument('--xz_interval',    default=8,    type=int,   help='Use every Nth Y-position as XZ training slice')

    # Training schedule
    p.add_argument('--batch_size',     default=8,    type=int,   help='Patches per gradient step (default 8)')
    p.add_argument('--epochs',         default=3,    type=int,   help='Training epochs (GUI default 40; use 3 for pipeline test)')
    p.add_argument('--log_interval',   default=1,    type=int,   help='Print loss every N batches')
    p.add_argument('--imshow_interval',default=100,  type=int,   help='Save sample image every N batches')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _setup_logger():
    log = logging.getLogger('selfnet_train')
    log.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(h)
    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    logger = _setup_logger()

    image_paths = [Path(p) for p in args.images]
    for ip in image_paths:
        if not ip.exists():
            logger.error(f'Image not found: {ip}')
            sys.exit(1)

    # SelfNet scans input_dir for all .tif files — put all selected images in
    # a clean working folder so only these images are processed.
    work_dir   = os.path.join(args.output, args.name + '_workdir')
    output_dir = os.path.join(args.output, args.name)
    os.makedirs(work_dir,   exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        dst = os.path.join(work_dir, image_path.name)
        if not os.path.exists(dst):
            try:
                os.link(image_path, dst)
            except OSError:
                shutil.copy2(image_path, dst)
            logger.info(f'Linked {image_path.name} -> {work_dir}')
        else:
            logger.info(f'Image already present in workdir: {dst}')

    logger.info('')
    logger.info('=== SelfNet Training Configuration ===')
    for ip in image_paths:
        logger.info(f'  image          = {ip}')
    logger.info(f'  output         = {output_dir}')
    logger.info(f'  min_v / max_v  = {args.min_v} / {args.max_v}')
    logger.info(f'  bg_threshold   = {args.bg_threshold}  (patch selection floor)')
    logger.info(f'  scale          = {args.scale}  (XY pixel / Z step)')
    logger.info(f'  xy_interval    = {args.xy_interval}  (every Nth Z-plane)')
    logger.info(f'  xz_interval    = {args.xz_interval}  (every Nth Y-position)')
    logger.info(f'  batch_size     = {args.batch_size}')
    logger.info(f'  epochs         = {args.epochs}')
    logger.info(f'  log_interval   = {args.log_interval}')
    logger.info(f'  imshow_interval= {args.imshow_interval}')
    logger.info('')
    logger.info('Launching SelfNet_Model_Training.py via respaninternal...')

    selfnet_argv = [
        SELFNET_SCRIPT,
        '--input_dir',    work_dir + os.sep,
        '--model_path',   args.output,
        '--model_name',   args.name,
        '--min_v',        str(args.min_v),
        '--max_v',        str(args.max_v),
        '--bg_threshold', str(args.bg_threshold),
        '--scale',        str(args.scale),
        '--xy_int',       str(args.xy_interval),
        '--xz_int',       str(args.xz_interval),
        '--batch_size',   str(args.batch_size),
        '--epochs',       str(args.epochs),
        '--log_interval', str(args.log_interval),
        '--imshow_interval', str(args.imshow_interval),
    ]

    if _ON_MAHTI:
        logger.info('Mahti: running SelfNet directly (no subprocess)')
        logger.info('')
        sys.argv = selfnet_argv
        runpy.run_path(SELFNET_SCRIPT, run_name='__main__')
    else:
        cmd = [RESPANINTERNAL_PY, '-u'] + selfnet_argv
        logger.info('Command: ' + ' '.join(cmd))
        logger.info('')
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f'SelfNet training failed (exit code {result.returncode})')
            sys.exit(result.returncode)

    logger.info('')
    logger.info(f'=== Done. Model saved to: {output_dir} ===')


if __name__ == '__main__':
    main()
