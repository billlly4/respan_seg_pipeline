"""
prepare_respan_experiments.py
------------------------------
One-shot script that:
  1. Converts LIF files to per-channel TIF stacks (t=0 for any timeseries).
  2. Copies already-TIF files into the right experiment sub-folders.
  3. Writes a tailored Analysis_Settings.yaml into each sub-folder.

Rule: every z-stack in every LIF series is extracted. For multi-timepoint
series, t=0 is used. All channels are saved as separate TIF files.

The cortical LIF is split into two sub-folders because the FRAP ROI series
(~190-240 nm XY) and the full-field overview series (82.5 nm XY) have
different pixel sizes, so they need separate Analysis_Settings.yaml files.

Run from the ResPan root directory:
    python prepare_respan_experiments.py
"""

import os
import sys
import shutil
import textwrap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "batch\u2011convertor"))  # non-breaking hyphen
from lif_to_tif import convert as lif_convert

ROOT     = SCRIPT_DIR
DATA_DIR = os.path.join(ROOT, "data")
OUT_BASE = os.path.join(ROOT, "RESPAN_experiments")

# ---------------------------------------------------------------------------
# Analysis_Settings.yaml template
# ---------------------------------------------------------------------------
YAML_TEMPLATE = textwrap.dedent("""\
Parameters:
    input_resXY: {res_xy}  #raw data XY pixel size in microns
    input_resZ: {res_z}    #raw data Z step size in microns
    model_resXY: {res_xy}
    model_resZ: {res_z}

Analysis:
    tiles_for_prediction: (2,3,3)
    roi_volume_size: 4
    GPU_block_size: (150,500,500)
    validation_format: tif
    validation_scale: (1,1)
    validation_jpeg_comp: 100
    erode_shape: (0.5,2,2)
    remove_touching_boarders: False

Vaa3D:
    path: C:/Program Files/Vaa3D_1_1_4 windows/Vaa3D-x.exe

Channel1:
    restore: False
    rest_model_path: None
    rest_type: ("care", 100)
Channel2:
    restore: False
    rest_model_path: None
    rest_type: ("care", 100)
Channel3:
    restore: False
    rest_model_path: None
    rest_type: ("care", 100)
Channel4:
    restore: False
    rest_model_path: None
    rest_type: ("care", 100)

SelfNet:
    model_path: None
""")

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# series: list of 0-based indices to extract, or None = all candidate series
# res_xy_um / res_z_um: representative pixel size for Analysis_Settings.yaml
# (each individual TIF also carries its own metadata from the LIF file)

EXPERIMENTS = [
    # -- Cortical LIF: FRAP ROI acquisitions (series 0-19, ~190-240 nm XY) --
    {
        "name"      : "180626_cortical_FRAP_ROIs",
        "lif"       : "180626_cortical_Control (1).lif",
        "series"    : list(range(20)),   # series 0-19
        "channels"  : None,              # all channels
        "timepoint" : 0,                 # t=0 for any multi-timepoint series
        "tif_files" : [],
        "res_xy_um" : 0.200,             # representative value; individual series vary 0.181-0.242
        "res_z_um"  : 0.200,
        "notes"     : (
            "Cortical neurons, Control. FRAP ROI z-stacks (series 0-19 from the LIF).\n"
            "XY resolution varies per series (~181-242 nm); Z step ~200 nm.\n"
            "Multi-timepoint Pre-FRAP series: t=0 (pre-bleach baseline) extracted.\n"
            "All channels exported as separate TIF files (_ch0 = structural, _ch1 = functional)."
        ),
    },
    # -- Cortical LIF: large-field overview stacks (series 20-21, 82.5 nm XY) --
    {
        "name"      : "180626_cortical_overview",
        "lif"       : "180626_cortical_Control (1).lif",
        "series"    : [20, 21],
        "channels"  : None,
        "timepoint" : 0,
        "tif_files" : [],
        "res_xy_um" : 0.0825,
        "res_z_um"  : 0.100,             # Series 20 = 100 nm Z; Series 21 = 200 nm Z
        "notes"     : (
            "Cortical neurons, Control. Full-field overview z-stacks (series 20-21 from the LIF).\n"
            "Series 20: 2984x2984x33 z-planes, XY=82.5 nm, Z=100 nm.\n"
            "Series 21: 2480x2480x25 z-planes, XY=82.5 nm, Z=200 nm.\n"
            "Note: Series 21 has Z=200 nm; if running RESPAN separately, update resZ in GUI."
        ),
    },
    # -- Hippocampal LIF: all series, all at ~82.5 nm XY / 200 nm Z ----------
    {
        "name"      : "280622_hippo_200K",
        "lif"       : "280622no2_hippo_200K.lif",
        "series"    : None,              # all series (0-6)
        "channels"  : None,
        "timepoint" : 0,                 # t=0 for series 0-2 (4-timepoint timeseries)
        "tif_files" : [],
        "res_xy_um" : 0.0825,
        "res_z_um"  : 0.200,
        "notes"     : (
            "Hippocampal neurons (DIV22), 200K seeding, spinning-disk confocal.\n"
            "Series 0-2: 2984x2984x28, 4 timepoints — t=0 extracted.\n"
            "Series 3:   2984x2984x28, 1 timepoint (single overview).\n"
            "Series 4-6: smaller dendritic ROIs (488-888 px, 11-19 z-planes).\n"
            "All channels: ch0 = structural, ch1 = functional (Lifeact/AMPAR)."
        ),
    },
    # -- Already-TIF files ----------------------------------------------------
    {
        "name"      : "140325_DIV17cx_LTP30min",
        "lif"       : None,
        "series"    : None,
        "channels"  : None,
        "timepoint" : 0,
        "tif_files" : ["140325_DIV17cx_BFP2_Lifeact_EGFP_LTP30min_0001.tif"],
        "res_xy_um" : 0.102,
        "res_z_um"  : 0.200,
        "notes"     : (
            "Cortical neurons (DIV17), LTP 30 min. Single-channel TIF, shape (15, 141, 108).\n"
            "Resolution is a best-guess (0.102 µm XY); verify from original acquisition metadata."
        ),
    },
    {
        "name"      : "C2-HippoDIV14_control",
        "lif"       : None,
        "series"    : None,
        "channels"  : None,
        "timepoint" : 0,
        "tif_files" : ["C2-HippoDIV14_control.lif - Series005.tif"],
        "res_xy_um" : 0.0813,
        "res_z_um"  : 0.200,
        "notes"     : (
            "Hippocampal neurons (DIV14), Control. Channel 2 only (shape 28x165x362).\n"
            "WARNING: structural channel (ch0) is missing. RESPAN needs the morphology\n"
            "channel to segment spines. Re-export ch0 from the original LIF if available."
        ),
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(path, res_xy, res_z):
    with open(path, "w") as f:
        f.write(YAML_TEMPLATE.format(
            res_xy=f"{res_xy:.4f}",
            res_z=f"{res_z:.4f}",
        ))


def write_notes(path, notes, exp_name):
    with open(path, "w") as f:
        f.write(f"# RESPAN Experiment: {exp_name}\n\n{notes}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare():
    os.makedirs(OUT_BASE, exist_ok=True)

    for exp in EXPERIMENTS:
        name    = exp["name"]
        exp_dir = os.path.join(OUT_BASE, name)
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Preparing: {name}")

        # LIF conversion
        if exp["lif"]:
            lif_path = os.path.join(DATA_DIR, exp["lif"])
            if not os.path.isfile(lif_path):
                print(f"  WARNING: LIF not found: {lif_path}")
            else:
                lif_convert(
                    lif_path       = lif_path,
                    out_dir        = exp_dir,
                    series_indices = exp["series"],   # None = all candidate series
                    channels       = exp["channels"],
                    timepoint      = exp["timepoint"],
                    prefix         = name,
                )

        # Copy existing TIF files
        for rel in exp["tif_files"]:
            src = os.path.join(DATA_DIR, rel)
            dst = os.path.join(exp_dir, os.path.basename(rel))
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied : {os.path.basename(rel)}")
            else:
                print(f"  WARNING: TIF not found: {src}")

        write_yaml(os.path.join(exp_dir, "Analysis_Settings.yaml"),
                   exp["res_xy_um"], exp["res_z_um"])
        print(f"  YAML   : resXY={exp['res_xy_um']}, resZ={exp['res_z_um']}")

        write_notes(os.path.join(exp_dir, "experiment_notes.txt"),
                    exp["notes"], name)

    print(f"\n{'='*60}")
    print(f"Done. Experiment folders created at:\n  {OUT_BASE}")
    print("""
Next steps:
  1. Launch RESPAN_v1_0_00/RESPAN.exe
  2. For each run, select one experiment sub-folder in the GUI
  3. Select the segmentation model (fold_all folder):
       Model 1A  pretrained_models/Dataset219_.../fold_all
       Model 1B  pretrained_models/Dataset224_.../fold_all
       Model 2   pretrained_models/Dataset817_.../fold_all
  4. Verify XY/Z resolution matches Analysis_Settings.yaml
  5. Run and record results in RESPAN_run_matrix.md
""")


if __name__ == "__main__":
    prepare()
