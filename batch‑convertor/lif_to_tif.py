"""
lif_to_tif.py
-------------
Converts Leica LIF files to individual per-series, per-channel 3-D TIFF files
suitable for RESPAN analysis.

Usage
-----
    python lif_to_tif.py <input.lif> <output_dir> [options]

Options
-------
    --series         Comma-separated list of 0-based series indices to export
                     (default: all single-timepoint 3-D series)
    --channels       Comma-separated list of channel indices to export
                     (default: all channels)
    --timepoint      Which timepoint to extract for multi-timepoint series (default: 0)
    --prefix         Output file name prefix (default: derived from LIF filename)
    --list           Just print series metadata and exit (no conversion)

Output naming
-------------
    <prefix>_series<NN>_ch<C>.tif

Pixel resolution is embedded in the TIFF metadata (ImageJ-compatible).

Requirements
------------
    pip install readlif tifffile
"""

import argparse
import os
import sys
import numpy as np
import tifffile
from readlif.reader import LifFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_resolution(lif_image):
    """
    Return (res_xy_um, res_z_um) from a readlif image object.
    readlif stores scale as pixels-per-micrometre; we invert to get µm/px.
    Z scale is stored as a negative value (ascending Z convention).
    """
    scale = lif_image.scale          # (x_px_per_um, y_px_per_um, z_px_per_um, t_px_per_s)
    res_xy = 1.0 / abs(scale[0]) if scale[0] else None
    res_z  = 1.0 / abs(scale[2]) if scale[2] else None
    return res_xy, res_z


def is_useful_series(lif_image, min_z=3, min_xy=64):
    """
    Return True if the series looks like a 3-D z-stack worth analysing.
    Skips 2-D images, series with too few z-planes, and very small XY fields.
    """
    d = lif_image.dims
    if d.z < min_z:
        return False
    if d.x < min_xy or d.y < min_xy:
        return False
    return True


def extract_stack(lif_image, channel, timepoint=0):
    """
    Extract a single channel / timepoint as a (Z, Y, X) numpy uint16 array.
    """
    frames = []
    for z_img in lif_image.get_iter_z(t=timepoint, c=channel):
        arr = np.array(z_img)
        if arr.dtype != np.uint16:
            # Normalise to 16-bit to keep downstream tools happy
            if arr.max() > 0:
                arr = (arr.astype(np.float32) / arr.max() * 65535).astype(np.uint16)
            else:
                arr = arr.astype(np.uint16)
        frames.append(arr)
    return np.stack(frames, axis=0)   # (Z, Y, X)


def save_tif(array, out_path, res_xy_um, res_z_um):
    """
    Save a (Z, Y, X) uint16 array as an ImageJ-compatible TIFF with
    pixel-size metadata embedded.
    """
    # ImageJ resolution units are pixels-per-micron
    res_xy = 1.0 / res_xy_um if res_xy_um else 1.0
    res_z  = res_z_um if res_z_um else 1.0       # stored in µm in ImageJ metadata

    tifffile.imwrite(
        out_path,
        array,
        imagej=True,
        resolution=(res_xy, res_xy),             # (x, y) in px/µm
        metadata={"spacing": res_z, "unit": "um"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def list_series(lif_path):
    lf = LifFile(lif_path)
    print(f"\nSeries in: {os.path.basename(lif_path)}\n")
    print(f"{'Idx':>4}  {'Name':<45} {'X':>5} {'Y':>5} {'Z':>5} {'T':>4} {'Ch':>3}  {'XY res (µm)':>12}  {'Z res (µm)':>10}")
    print("-" * 110)
    for i, img in enumerate(lf.get_iter_image()):
        d = img.dims
        res_xy, res_z = get_resolution(img)
        res_xy_s = f"{res_xy:.4f}" if res_xy else "?"
        res_z_s  = f"{res_z:.4f}"  if res_z  else "?"
        useful = "*" if is_useful_series(img) else " "
        print(f"{i:>4}{useful} {img.name:<45} {d.x:>5} {d.y:>5} {d.z:>5} {d.t:>4} {img.channels:>3}  {res_xy_s:>12}  {res_z_s:>10}")
    print("\n* = candidate series for RESPAN (3-D, sufficient size)")


def convert(lif_path, out_dir, series_indices=None, channels=None, timepoint=0, prefix=None):
    os.makedirs(out_dir, exist_ok=True)
    lf = LifFile(lif_path)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(lif_path))[0]
        # Sanitise prefix for file names
        prefix = "".join(c if c.isalnum() or c in "_-" else "_" for c in prefix)

    all_images = list(lf.get_iter_image())

    if series_indices is None:
        series_indices = [i for i, img in enumerate(all_images) if is_useful_series(img)]
        if not series_indices:
            print("WARNING: No candidate series found with default filters. "
                  "Use --list to inspect and --series to specify manually.")
            return

    exported = []
    for idx in series_indices:
        if idx >= len(all_images):
            print(f"WARNING: Series index {idx} out of range – skipping.")
            continue
        img = all_images[idx]
        d   = img.dims
        res_xy, res_z = get_resolution(img)

        chans = channels if channels is not None else list(range(img.channels))
        tp    = min(timepoint, d.t - 1)   # guard against out-of-range timepoint

        print(f"\nSeries {idx:02d}: {img.name}")
        print(f"  Dims    : Z={d.z}  Y={d.y}  X={d.x}  T={d.t}  Channels={img.channels}")
        print(f"  Res     : XY={res_xy:.4f} µm/px  Z={res_z:.4f} µm/px" if res_xy and res_z
              else "  Res     : unknown")
        if d.t > 1:
            print(f"  Note    : {d.t} timepoints found – extracting timepoint {tp}")

        for ch in chans:
            if ch >= img.channels:
                print(f"  WARNING : channel {ch} does not exist – skipping.")
                continue
            fname = f"{prefix}_series{idx:03d}_ch{ch}.tif"
            out_path = os.path.join(out_dir, fname)
            stack = extract_stack(img, channel=ch, timepoint=tp)
            save_tif(stack, out_path, res_xy, res_z)
            print(f"  Saved   : {fname}  shape={stack.shape}  dtype={stack.dtype}")
            exported.append({
                "series_idx": idx,
                "series_name": img.name,
                "channel": ch,
                "timepoint": tp,
                "shape": stack.shape,
                "res_xy_um": res_xy,
                "res_z_um": res_z,
                "file": out_path,
            })

    print(f"\nDone. {len(exported)} TIF file(s) written to: {out_dir}")
    return exported


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Convert Leica LIF series to per-channel 3-D TIFF files for RESPAN."
    )
    parser.add_argument("lif_file",  help="Path to the .lif file")
    parser.add_argument("output_dir", nargs="?", default=None,
                        help="Output directory (default: <lif_file_stem>_tifs/ next to LIF)")
    parser.add_argument("--series",    type=parse_int_list, default=None,
                        help="Comma-separated series indices to export (default: all 3-D candidates)")
    parser.add_argument("--channels",  type=parse_int_list, default=None,
                        help="Comma-separated channel indices to export (default: all)")
    parser.add_argument("--timepoint", type=int, default=0,
                        help="Timepoint index to extract for multi-timepoint series (default: 0)")
    parser.add_argument("--prefix",    type=str, default=None,
                        help="Output filename prefix (default: derived from LIF filename)")
    parser.add_argument("--list",      action="store_true",
                        help="List series metadata only, do not convert")
    args = parser.parse_args()

    if not os.path.isfile(args.lif_file):
        print(f"ERROR: File not found: {args.lif_file}")
        sys.exit(1)

    if args.list:
        list_series(args.lif_file)
        return

    if args.output_dir is None:
        stem = os.path.splitext(os.path.basename(args.lif_file))[0]
        stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)
        args.output_dir = os.path.join(os.path.dirname(args.lif_file), stem + "_tifs")

    convert(
        lif_path      = args.lif_file,
        out_dir       = args.output_dir,
        series_indices= args.series,
        channels      = args.channels,
        timepoint     = args.timepoint,
        prefix        = args.prefix,
    )


if __name__ == "__main__":
    main()
