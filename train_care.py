"""
CARE Training Pipeline (via respandev / TensorFlow)
====================================================
Trains a CARE denoising model using paired low/high SNR z-stacks.

Expected input layout:
    CARE_TRAINING_DATA/
        lowSNR/    <image.tif>   -- noisy acquisitions
        highSNR/   <image.tif>   -- clean acquisitions (same filename)

Output:
    CARE_OUTPUT/
        <MODEL_NAME>/            -- saved Keras model + weights
        CAREdata/                -- extracted patch cache (.npz)

HOW TO RUN (defaults from config block below):
    C:/Users/ARASH/Anaconda3/envs/respandev/python.exe  f:/ResPan/train_care.py

OVERRIDE ANY PARAMETER FROM THE COMMAND LINE:
    python train_care.py --epochs 150 --steps_per_epoch 400 --num_patches 100
    python train_care.py --patch_size 8,64,64 --unet_n_depth 3 --batch_size 16
    python train_care.py --data f:/path/to/data --output f:/path/to/out --name my_model
    python train_care.py --augmentation         # enable rotation augmentation

PARAMETER MEANINGS:
    --data             folder containing lowSNR/ and highSNR/ subfolders
    --output           folder where model and patch cache are saved
    --name             model name (subfolder created inside --output)
    --patch_size       Z,Y,X patch size  e.g. 8,64,64   (must fit inside images)
    --num_patches      patches extracted per image       (more = richer dataset)
    --epochs           training epochs                   (150 for real training)
    --steps_per_epoch  gradient updates per epoch        (400 for real training)
    --unet_kern_size   conv kernel size: always 3 for 3D, 5 for 2D
    --unet_n_depth     U-Net encoder depth               (3 = good default)
    --batch_size       patches per gradient step         (16 for real training)
    --pct_validation   fraction held out for validation  (0.10 = 10%)
    --augmentation     flag: enable 90/180/270 rotation augmentation
    --image_type       3D or 2D                          (default: 3D)
"""

import os
import sys
import argparse
import logging

# ---------------------------------------------------------------------------
# Defaults  (RESPAN GUI values; use small values here for fast pipeline test)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    data          = r'f:\ResPan\CARE_training\data',
    output        = r'f:\ResPan\CARE_training\output',
    name          = 'care_test_3D',

    patch_size    = '8,64,64',   # GUI default: 8,64,64
    num_patches   = 25,          # GUI default: 100   (reduced for test)
    epochs        = 5,           # GUI default: 150   (reduced for test)
    steps_per_epoch = 10,        # GUI default: 400   (reduced for test)
    unet_kern_size  = 3,         # GUI default: 3
    unet_n_depth    = 3,         # GUI default: 3
    batch_size      = 8,         # GUI default: 16    (reduced for test)
    pct_validation  = 0.10,      # GUI default: 0.10
    augmentation    = False,
    image_type      = '3D',
)

RESPAN_ROOT = r'f:\ResPan\cloned_respan\RESPAN'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='Train a CARE denoising model via RESPAN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--data',            default=DEFAULTS['data'],            help='Folder with lowSNR/ and highSNR/ subfolders')
    p.add_argument('--output',          default=DEFAULTS['output'],          help='Output folder for model and patch cache')
    p.add_argument('--name',            default=DEFAULTS['name'],            help='Model name (subfolder inside --output)')
    p.add_argument('--patch_size',      default=DEFAULTS['patch_size'],      help='Z,Y,X patch size e.g. 8,64,64')
    p.add_argument('--num_patches',     default=DEFAULTS['num_patches'],     type=int,   help='Patches per image')
    p.add_argument('--epochs',          default=DEFAULTS['epochs'],          type=int,   help='Training epochs')
    p.add_argument('--steps_per_epoch', default=DEFAULTS['steps_per_epoch'], type=int,   help='Gradient steps per epoch')
    p.add_argument('--unet_kern_size',  default=DEFAULTS['unet_kern_size'],  type=int,   help='Conv kernel size (3 for 3D)')
    p.add_argument('--unet_n_depth',    default=DEFAULTS['unet_n_depth'],    type=int,   help='U-Net encoder depth')
    p.add_argument('--batch_size',      default=DEFAULTS['batch_size'],      type=int,   help='Patches per gradient step')
    p.add_argument('--pct_validation',  default=DEFAULTS['pct_validation'],  type=float, help='Validation fraction (0.0-1.0)')
    p.add_argument('--augmentation',    action='store_true',                             help='Enable rotation augmentation')
    p.add_argument('--image_type',      default=DEFAULTS['image_type'],      choices=['3D', '2D'], help='3D or 2D')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
def _setup_logger():
    log = logging.getLogger('care_train')
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

    patch_size = [int(x) for x in args.patch_size.split(',')]
    if len(patch_size) != 3:
        logger.error('--patch_size must be three comma-separated integers, e.g. 8,64,64')
        sys.exit(1)

    # Add RESPAN source
    sys.path.insert(0, RESPAN_ROOT)
    sys.path.insert(0, os.path.join(RESPAN_ROOT, 'ImageAnalysis'))
    sys.path.insert(0, os.path.join(RESPAN_ROOT, 'Scripts'))

    from unittest.mock import MagicMock
    sys.modules.setdefault('sitecustomize', MagicMock())

    logger.info('Importing RESPAN ModelTraining...')
    from ModelTraining import train_care
    logger.info('Import OK.')

    # train_care saves CAREdata/my_training_data.npz as a relative path
    os.makedirs(args.output, exist_ok=True)
    os.chdir(args.output)
    logger.info(f'Working directory -> {args.output}')

    logger.info('')
    logger.info('=== CARE Training Configuration ===')
    logger.info(f'  data            = {args.data}')
    logger.info(f'  output          = {args.output}')
    logger.info(f'  name            = {args.name}')
    logger.info(f'  patch_size      = {patch_size}')
    logger.info(f'  num_patches     = {args.num_patches}')
    logger.info(f'  epochs          = {args.epochs}')
    logger.info(f'  steps_per_epoch = {args.steps_per_epoch}')
    logger.info(f'  unet_kern_size  = {args.unet_kern_size}')
    logger.info(f'  unet_n_depth    = {args.unet_n_depth}')
    logger.info(f'  batch_size      = {args.batch_size}')
    logger.info(f'  pct_validation  = {args.pct_validation}')
    logger.info(f'  augmentation    = {args.augmentation}')
    logger.info(f'  image_type      = {args.image_type}')
    logger.info('')

    train_care(
        inputdir            = args.data,
        model_output        = args.output,
        model_name          = args.name,
        augmentation        = args.augmentation,
        image_type          = args.image_type,
        patch_size          = patch_size,
        num_patches_per_img = args.num_patches,
        epochs              = args.epochs,
        unet_kern_size      = args.unet_kern_size,
        unet_n_depth        = args.unet_n_depth,
        batch_size          = args.batch_size,
        steps_per_epoch     = args.steps_per_epoch,
        pct_validation      = args.pct_validation,
        app                 = None,
        logger              = logger,
    )

    logger.info('')
    logger.info(f'=== Done. Model saved to: {os.path.join(args.output, args.name)} ===')


if __name__ == '__main__':
    main()
