"""
Generate Fiji-compatible ROI zip files from RESPAN analysis output.

For each analysed image, reads the nnU-Net label TIF and the detected_spines CSV
and produces a ZIP of ImageJ polygon ROIs that can be loaded directly into Fiji's
ROI Manager (Analyze > Tools > ROI Manager > More > Open).

ROIs produced per image:
  dendrite          — full dendrite mask (one polygon per Z-slice)
  spine_{id}        — head + membrane combined (one polygon per Z-slice)
  neck_{id}         — neck mask (one polygon per Z-slice)
  dendrite-MIP      — Z-projection of dendrite mask (for use on MIP image)
  spine_{id}-MIP    — Z-projection of spine mask (for use on MIP image)
  neck_{id}-MIP     — Z-projection of neck mask (for use on MIP image)

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


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _mask_to_rois(mask_3d, name_prefix, z_offset=0):
    """
    Convert a 3D binary mask to a list of polygon ImagejRoi objects.
    One ROI is created per Z-slice where the mask is non-empty.
    z_offset: add to z-index when setting .position (for 1-indexed Fiji).
    """
    rois = []
    for z in range(mask_3d.shape[0]):
        slice_mask = mask_3d[z].astype(bool)
        if not slice_mask.any():
            continue
        # Pad by 1 so masks touching the image border get a proper closed contour
        padded = np.pad(slice_mask, 1, mode='constant', constant_values=0)
        contours = find_contours(padded, 0.5)
        if not contours:
            continue
        # Use the longest contour (largest perimeter = outer boundary)
        contour = max(contours, key=len)
        # Subtract pad offset, then swap (row,col)->(col,row) = (Y,X)->(X,Y) for Fiji
        contour = contour - 1
        xy = contour[:, ::-1].astype(np.float32)
        roi = roifile.ImagejRoi.frompoints(xy, name=f'{name_prefix}-Z{z + 1 + z_offset}',
                                           z=z + z_offset)
        roi.position = z + 1 + z_offset   # 1-indexed Z for Fiji
        rois.append(roi)
    return rois


def _mask_to_mip_roi(mask_3d, name):
    """Project 3D mask along Z and return a single flat polygon ROI for use on a MIP image."""
    mip = np.any(mask_3d, axis=0)
    if not mip.any():
        return None
    padded = np.pad(mip, 1, mode='constant', constant_values=0)
    contours = find_contours(padded, 0.5)
    if not contours:
        return None
    contour = max(contours, key=len) - 1   # subtract pad offset
    xy = contour[:, ::-1].astype(np.float32)
    roi = roifile.ImagejRoi.frompoints(xy, name=name)
    roi.position = 0   # 0 = applies to all slices / flat image
    return roi


def _save_roi_zip(rois, out_path):
    """Write a list of ImagejRoi objects to a Fiji-compatible zip file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for roi in rois:
            zf.writestr(roi.name + '.roi', roi.tobytes())
    print(f'  Saved {len(rois)} ROIs -> {out_path}')


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image(crop_dir):
    crop_dir = Path(crop_dir)
    name = crop_dir.name

    label_path   = crop_dir / 'labels' / (name + '.tif')
    csv_path     = crop_dir / 'Tables' / (name + '_detected_spines.csv')
    out_3d_path  = crop_dir / 'ROIs' / (name + '_rois_3d.zip')
    out_mip_path = crop_dir / 'ROIs' / (name + '_rois_mip.zip')

    if not label_path.exists():
        print(f'  [skip] no label TIF: {label_path}')
        return
    if not csv_path.exists():
        print(f'  [skip] no CSV: {csv_path}')
        return

    print(f'Processing {name}')

    label_vol = imread(str(label_path))   # (Z, Y, X) uint8, values 1–6
    df = pd.read_csv(str(csv_path))

    rois_3d  = []
    rois_mip = []

    # ------------------------------------------------------------------
    # 1. Dendrite
    # ------------------------------------------------------------------
    dend_mask = (label_vol == 1)
    rois_3d += _mask_to_rois(dend_mask, 'dendrite')
    mip_roi = _mask_to_mip_roi(dend_mask, 'dendrite-MIP')
    if mip_roi:
        rois_mip.append(mip_roi)
    print(f'  dendrite: {len(rois_3d)} Z-slice ROIs')

    # ------------------------------------------------------------------
    # 2. Spine heads (label 2 = core, label 3 = membrane, combined)
    # ------------------------------------------------------------------
    spine_mask = (label_vol == 2) | (label_vol == 3)
    spine_cc, n_spine = sk_label(spine_mask, connectivity=3, return_num=True)

    # Compute centroid for each CC (in voxels, ZYX order)
    cc_centroids = ndi.center_of_mass(spine_mask, spine_cc, range(1, n_spine + 1))
    # cc_centroids[i] = (z, y, x) for CC label i+1

    # Match each CC to the nearest CSV row by (x, y, z) centroid.
    # Multiple disconnected CC fragments (e.g. separated core/membrane pieces)
    # may map to the same spine_id — merge their masks before generating ROIs.
    csv_xyz = df[['x', 'y', 'z']].values   # shape (n_spines, 3) in (X, Y, Z)

    spine_id_to_mask = defaultdict(lambda: np.zeros(label_vol.shape, dtype=bool))

    for cc_idx, (cz, cy, cx) in enumerate(cc_centroids):
        cc_label = cc_idx + 1
        dists = np.sqrt(
            (csv_xyz[:, 0] - cx) ** 2 +
            (csv_xyz[:, 1] - cy) ** 2 +
            (csv_xyz[:, 2] - cz) ** 2
        )
        spine_id = int(df.iloc[int(np.argmin(dists))]['spine_id'])
        spine_id_to_mask[spine_id] |= (spine_cc == cc_label)

    for spine_id, merged_mask in sorted(spine_id_to_mask.items()):
        rois_3d += _mask_to_rois(merged_mask, f'spine_{spine_id}')
        mip_roi = _mask_to_mip_roi(merged_mask, f'spine_{spine_id}-MIP')
        if mip_roi:
            rois_mip.append(mip_roi)

    print(f'  spines: {len(spine_id_to_mask)} matched')

    # ------------------------------------------------------------------
    # 3. Necks (label 4) — assign each neck CC to nearest spine CC, then merge
    # ------------------------------------------------------------------
    neck_mask = (label_vol == 4)
    neck_cc, n_neck = sk_label(neck_mask, connectivity=3, return_num=True)

    neck_id_to_mask = defaultdict(lambda: np.zeros(label_vol.shape, dtype=bool))

    for ncc_idx in range(1, n_neck + 1):
        this_neck = (neck_cc == ncc_idx)
        nz, ny, nx = ndi.center_of_mass(this_neck)

        # Nearest spine CC centroid -> look up that CC's spine_id
        dists_to_spine = np.array([
            (sz - nz) ** 2 + (sy - ny) ** 2 + (sx - nx) ** 2
            for sz, sy, sx in cc_centroids
        ])
        nearest_scc = int(np.argmin(dists_to_spine))   # 0-indexed
        sz, sy, sx = cc_centroids[nearest_scc]
        dists_csv = np.sqrt(
            (csv_xyz[:, 0] - sx) ** 2 +
            (csv_xyz[:, 1] - sy) ** 2 +
            (csv_xyz[:, 2] - sz) ** 2
        )
        spine_id = int(df.iloc[int(np.argmin(dists_csv))]['spine_id'])
        neck_id_to_mask[spine_id] |= this_neck

    for spine_id, merged_mask in sorted(neck_id_to_mask.items()):
        rois_3d += _mask_to_rois(merged_mask, f'neck_{spine_id}')
        mip_roi = _mask_to_mip_roi(merged_mask, f'neck_{spine_id}-MIP')
        if mip_roi:
            rois_mip.append(mip_roi)

    print(f'  necks: {len(neck_id_to_mask)} matched')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    _save_roi_zip(rois_3d,  str(out_3d_path))
    _save_roi_zip(rois_mip, str(out_mip_path))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root',  default='images_analysis',
                   help='Root directory containing per-crop folders')
    p.add_argument('--crop',  default=None,
                   help='Process only this crop subfolder name (default: all)')
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
        process_image(crop_dir)


if __name__ == '__main__':
    main()
