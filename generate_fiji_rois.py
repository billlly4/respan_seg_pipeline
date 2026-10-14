"""
Generate Fiji-compatible ROI zip files from RESPAN analysis output.

For each analysed image, produces a ZIP of ImageJ polygon ROIs that can be
loaded directly into Fiji's ROI Manager (Analyze > Tools > ROI Manager > More > Open).

Two source modes (chosen automatically per image):

  PREFERRED — RESPAN's own instance-separated volume:
      Validation_Data/Validation_Vols/<name>.tif is a multi-channel volume
      (ZCYX). Its channels carry RESPAN's internal instance labels:
          channel 2  = spines_filtered  (each spine already has its own ID)
          channel 3  = connected_necks  (each neck labelled with its spine ID)
          channel -2 = labeled_dendrites
      These IDs match the CSV spine_id exactly. Using them avoids re-deriving
      instances and never collapses two adjacent spines into one ROI.

  FALLBACK — semantic label TIF + CSV (used only if the Validation_Vols file
      is absent, i.e. save_val_volumes was off):
      Connected components on labels/<name>.tif (class 2|3 spine, 4 neck),
      matched to detected_spines.csv centroids by nearest distance.

ROIs produced per image:
  dendrite          — dendrite mask (one polygon per Z-slice)
  spine_{id}        — head + membrane combined (one polygon per Z-slice)
  neck_{id}         — neck mask (one polygon per Z-slice)
  ...plus a -MIP variant of each (Z-projected, for use on a MIP image)

IDs match the spine_id column in detected_spines.csv.

Usage:
    python generate_fiji_rois.py
    python generate_fiji_rois.py --root images_analysis --crop crop14_...
"""

import os
import io
import sys
import argparse
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread
from scipy import ndimage as ndi
from skimage.measure import find_contours, label as sk_label
import roifile

_BASE = Path(__file__).resolve().parent          # shared CODE location
# Per-user working directory (set RESPAN_WORKDIR to match run_respan.py so each
# person's crops/ROIs stay separate). Defaults to the code dir.
_WORKDIR = Path(os.environ.get('RESPAN_WORKDIR', _BASE)).resolve()


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _slice_contours(slice_mask):
    """Return all polygon contours (>=3 points) of a 2D binary mask, in (X,Y) order.

    Pads by 1 first so masks touching the image border still close properly,
    then subtracts the pad offset. Every disconnected piece yields its own
    contour (e.g. a branching dendrite, or a structure split within one slice).
    """
    padded = np.pad(slice_mask, 1, mode='constant', constant_values=0)
    out = []
    for contour in find_contours(padded, 0.5):
        if len(contour) < 3:
            continue
        contour = contour - 1                              # undo pad offset
        out.append(contour[:, ::-1].astype(np.float32))    # (row,col)->(X,Y)
    return out


def _mask_to_rois(mask_3d, name_prefix, z_offset=0):
    """
    Convert a 3D binary mask to a list of polygon ImagejRoi objects.
    One ROI per disconnected piece per Z-slice. When a slice holds more than
    one piece, names are suffixed -1, -2, ... so they stay unique in Fiji.
    """
    rois = []
    for z in range(mask_3d.shape[0]):
        slice_mask = mask_3d[z].astype(bool)
        if not slice_mask.any():
            continue
        contours = _slice_contours(slice_mask)
        for k, xy in enumerate(contours):
            zname = f'{name_prefix}-Z{z + 1 + z_offset}'
            if len(contours) > 1:
                zname += f'-{k + 1}'
            roi = roifile.ImagejRoi.frompoints(xy, name=zname, z=z + z_offset)
            roi.position = z + 1 + z_offset   # 1-indexed Z for Fiji
            rois.append(roi)
    return rois


def _mask_to_mip_rois(mask_3d, name):
    """Project 3D mask along Z and return flat polygon ROIs (one per piece) for a MIP image."""
    mip = np.any(mask_3d, axis=0)
    if not mip.any():
        return []
    contours = _slice_contours(mip)
    rois = []
    for k, xy in enumerate(contours):
        rname = name if len(contours) == 1 else f'{name}-{k + 1}'
        roi = roifile.ImagejRoi.frompoints(xy, name=rname)
        roi.position = 0   # 0 = applies to all slices / flat image
        rois.append(roi)
    return rois


def _save_roi_zip(rois, out_path):
    """Write a list of ImagejRoi objects to a Fiji-compatible zip file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for roi in rois:
            zf.writestr(roi.name + '.roi', roi.tobytes())
    print(f'  Saved {len(rois)} ROIs -> {out_path}')


# ---------------------------------------------------------------------------
# Mask extraction — PREFERRED source: RESPAN instance volume
# ---------------------------------------------------------------------------

def _masks_from_instance_volume(vol_path):
    """
    Read RESPAN's multi-channel Validation_Vols file and return
    (dend_mask, spine_id_to_mask, neck_id_to_mask) using RESPAN's own
    instance IDs. Returns None if the file can't be used.

    Channel layout of _vol_arrays (from RESPAN ImageAnalysis.py):
        [neuron, spines, spines_filtered, connected_necks,
         (_grouped_spines), labeled_dendrites, skeleton]
    _grouped_spines is present only when detect_multi_head_spines is on, so
    spines_filtered is always ch2, connected_necks ch3, dendrites ch[-2].
    """
    vol = imread(str(vol_path))
    if vol.ndim != 4 or vol.shape[1] < 6:
        return None   # not the expected ZCYX multi-channel volume

    spines_filtered  = vol[:, 2, :, :]
    connected_necks  = vol[:, 3, :, :]
    labeled_dendrite = vol[:, -2, :, :]

    dend_mask = labeled_dendrite > 0

    spine_id_to_mask = {}
    for sid in np.unique(spines_filtered):
        if sid == 0:
            continue
        spine_id_to_mask[int(sid)] = (spines_filtered == sid)

    neck_id_to_mask = {}
    for sid in np.unique(connected_necks):
        if sid == 0:
            continue
        neck_id_to_mask[int(sid)] = (connected_necks == sid)

    return dend_mask, spine_id_to_mask, neck_id_to_mask


# ---------------------------------------------------------------------------
# Mask extraction — FALLBACK source: semantic label TIF + CSV centroids
# ---------------------------------------------------------------------------

def _masks_from_labels_csv(label_path, csv_path):
    """
    Re-derive instances from the semantic label TIF by connected components,
    then match each component to the nearest CSV spine_id centroid.
    Returns (dend_mask, spine_id_to_mask, neck_id_to_mask).
    """
    label_vol = imread(str(label_path))     # (Z, Y, X) uint8, values 1–6
    df = pd.read_csv(str(csv_path))
    csv_xyz = df[['x', 'y', 'z']].values     # (n_spines, 3) in (X, Y, Z)

    dend_mask = (label_vol == 1)

    # --- spines: class 2 (core) | 3 (membrane) ---
    spine_mask = (label_vol == 2) | (label_vol == 3)
    spine_cc, n_spine = sk_label(spine_mask, connectivity=3, return_num=True)
    cc_centroids = ndi.center_of_mass(spine_mask, spine_cc, range(1, n_spine + 1))

    spine_id_to_mask = defaultdict(lambda: np.zeros(label_vol.shape, dtype=bool))
    for cc_idx, (cz, cy, cx) in enumerate(cc_centroids):
        dists = np.sqrt((csv_xyz[:, 0] - cx) ** 2 +
                        (csv_xyz[:, 1] - cy) ** 2 +
                        (csv_xyz[:, 2] - cz) ** 2)
        spine_id = int(df.iloc[int(np.argmin(dists))]['spine_id'])
        spine_id_to_mask[spine_id] |= (spine_cc == cc_idx + 1)

    # --- necks: class 4, assigned to nearest spine CC then to that spine_id ---
    neck_mask = (label_vol == 4)
    neck_cc, n_neck = sk_label(neck_mask, connectivity=3, return_num=True)

    neck_id_to_mask = defaultdict(lambda: np.zeros(label_vol.shape, dtype=bool))
    for ncc_idx in range(1, n_neck + 1):
        this_neck = (neck_cc == ncc_idx)
        nz, ny, nx = ndi.center_of_mass(this_neck)
        dists_to_spine = np.array([(sz - nz) ** 2 + (sy - ny) ** 2 + (sx - nx) ** 2
                                   for sz, sy, sx in cc_centroids])
        sz, sy, sx = cc_centroids[int(np.argmin(dists_to_spine))]
        dists_csv = np.sqrt((csv_xyz[:, 0] - sx) ** 2 +
                            (csv_xyz[:, 1] - sy) ** 2 +
                            (csv_xyz[:, 2] - sz) ** 2)
        spine_id = int(df.iloc[int(np.argmin(dists_csv))]['spine_id'])
        neck_id_to_mask[spine_id] |= this_neck

    return dend_mask, dict(spine_id_to_mask), dict(neck_id_to_mask)


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(crop_dir, source='auto'):
    crop_dir = Path(crop_dir)
    name = crop_dir.name

    label_path   = crop_dir / 'labels' / (name + '.tif')
    csv_path     = crop_dir / 'Tables' / (name + '_detected_spines.csv')
    vol_path     = crop_dir / 'Validation_Data' / 'Validation_Vols' / (name + '.tif')
    out_3d_path  = crop_dir / 'ROIs' / (name + '_rois_3d.zip')
    out_mip_path = crop_dir / 'ROIs' / (name + '_rois_mip.zip')

    print(f'Processing {name}')

    masks = None
    # source='fallback' forces the semantic-label + CSV path (skip instance volume).
    if source != 'fallback' and vol_path.exists():
        try:
            masks = _masks_from_instance_volume(vol_path)
            if masks is not None:
                print('  source: RESPAN instance volume (Validation_Vols)')
        except Exception as e:
            print(f'  [warn] could not read instance volume ({e}); falling back')
            masks = None

    if masks is None:
        if not label_path.exists() or not csv_path.exists():
            print(f'  [skip] no instance volume and no label TIF + CSV')
            return
        masks = _masks_from_labels_csv(label_path, csv_path)
        print('  source: semantic labels + CSV centroid matching (fallback)')

    dend_mask, spine_id_to_mask, neck_id_to_mask = masks

    rois_3d, rois_mip = [], []

    # Dendrite
    rois_3d  += _mask_to_rois(dend_mask, 'dendrite')
    rois_mip += _mask_to_mip_rois(dend_mask, 'dendrite-MIP')
    print(f'  dendrite: {len(rois_3d)} Z-slice ROIs')

    # Spines
    for spine_id, mask in sorted(spine_id_to_mask.items()):
        rois_3d  += _mask_to_rois(mask, f'spine_{spine_id}')
        rois_mip += _mask_to_mip_rois(mask, f'spine_{spine_id}-MIP')
    print(f'  spines: {len(spine_id_to_mask)}')

    # Necks
    for spine_id, mask in sorted(neck_id_to_mask.items()):
        rois_3d  += _mask_to_rois(mask, f'neck_{spine_id}')
        rois_mip += _mask_to_mip_rois(mask, f'neck_{spine_id}-MIP')
    print(f'  necks: {len(neck_id_to_mask)}')

    _save_roi_zip(rois_3d,  str(out_3d_path))
    _save_roi_zip(rois_mip, str(out_mip_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',  default=str(_WORKDIR / 'images_analysis'),
                   help='Root directory containing per-crop folders '
                        '(default: $RESPAN_WORKDIR/images_analysis)')
    p.add_argument('--crop',  default=None,
                   help='Process only this crop subfolder name (default: all)')
    p.add_argument('--source', choices=['auto', 'fallback'], default='auto',
                   help="ROI source: 'auto' = RESPAN instance volume if present else "
                        "labels+CSV; 'fallback' = always use labels + CSV (skip the "
                        "instance volume).")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    if args.crop:
        crop_dirs = [root / args.crop]
    else:
        crop_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / 'labels').exists()
        )

    if not crop_dirs:
        print(f'No crop directories found under {root}')
        sys.exit(1)

    for crop_dir in crop_dirs:
        process_image(crop_dir, source=args.source)


if __name__ == '__main__':
    main()
