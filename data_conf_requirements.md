# Training Data Configuration Requirements

## 1. CARE

CARE is the content-aware denoising/restoration model used before segmentation.
Restoration models are modality- and resolution-specific.

**Group images together by:**
- Same microscope/modality (spinning disk vs laser scanning vs two-photon should not be mixed)
- Same objective / PSF family (same NA, similar PSF behavior)
- Same voxel size, or very close
- Same fluorophore/channel type (structural dendrite-spine channel stays with structural)
- Same raw-intensity/noise regime (same detector, same exposure style, same gain range)
- Same preprocessing convention (do not mix raw data with heavily filtered/CLAHE'd data)
- Paired low-SNR and high-SNR images of the exact same field of view

**What can vary:**
- Branch shape, dendrite orientation, number of spines, local density

**Do not group by:**
- Biological treatment condition alone, if the imaging physics are the same

The pairing requirement (same FOV, low/high SNR) is the binding practical constraint —
it determines what you *can* use before any question of what would be ideal.

The question for CARE is not "are the cells biologically similar enough?"
It is "do these images come from the same imaging system and noise world?"

---

## 2. SelfNet

SelfNet is an axial-resolution restoration model, not a general denoiser.
It learns from the relationship between sharp XY structure and degraded axial appearance.

**Group images together by:**
- Same structural channel type
- Same XY pixel size
- Same Z step size (therefore same XY/Z ratio / scale parameter)
- Same modality / PSF family
- Same preprocessing convention(without noise, getting from the CARE model)

**What the data needs:**
- A 3D structural stack where XY is clearly better than Z
- Enough Z-slices to learn axial structure (see Z-plane constraint below)
- Continuous dendrite-like signal, not just sparse puncta

**Z-plane constraint:**

SelfNet cuts 64×64 patches from the XZ view. Before cutting, it stretches the Z
axis to match XY scale (so the XZ view represents real physical proportions).
The stretch factor is `1/scale`. The XZ image height after stretching must be
≥ 64 or the patch loop produces zero patches and training crashes — with no
error message, just an empty array.

The rule:
```
scale = input_resXY / input_resZ 
=> example:
 0.10 / 0.50 = 0.2

rescaled_Z     = int(original_Z × (1 / scale))   must be ≥ 64    
=> example: 
rescaled_Z = int(20 × (1 / 0.2))
           = int(20 × 5)
           = 100   ✓  (≥ 64, passes)

minimum_Z      = ceil(64 × scale)

the difference between z and xy is more than 3, we might get bad results.
```

Where the rule lives in the code — it is not an explicit check, it is an
accidental consequence of two lines in two different functions:

- `reslice_and_resize_torch` line 93: `new_x = int(x * scale_factor)`
  — produces the rescaled Z dimension
- `generate_training_data` line 236: `for m in range(0, xz_img.shape[0] - 64 + 1, 64)`
  — if rescaled Z < 64, this range is empty, zero XZ patches, silent crash

There is no guard or error message. The constraint is a silent assumption
baked into the loop arithmetic.

**What can vary:**
- Exact dendrite shape, spine number, local branching complexity

**Do not mix:**
- 0.08×0.08×0.20 µm family with 0.13×0.13×0.41 µm family
- Near-isotropic data with strongly anisotropic data
- Structural dendrite channel with an unrelated organelle channel

Mixing different voxel size families means the model is seeing two different physical
imaging systems — two different axial degradation problems. That makes the learned
restoration misleading rather than useful.

---

## 3. nnU-Net

nnU-Net is the segmentation model. Its most important requirement is annotation
consistency, not perfect optical identity.

**Group images together by:**
- Same label ontology: background / spine / dendrite / soma (whatever classes are chosen)
- Same annotation protocol: where the spine starts, how neck vs dendrite border is
  defined, whether faint protrusions count, whether touching spines are split or merged
- Same biological target (structural dendrite-spine channel stays dendrite-spine)
- Same or resampled-compatible voxel family

**What can tolerate more variation than CARE/SelfNet:**
- Branch density, number of dendrites in field, field size, morphology diversity
- Some moderate imaging variation

**What should not vary:**
- Wildly different voxel sizes without resampling — nnU-Net's fingerprinting computes
  a median voxel size across all training cases and designs the network architecture
  (kernel sizes, pooling geometry) around it. Mixing e.g. 65 nm isotropic with
  130×410 nm anisotropic produces a compromise architecture that serves neither well.
  Either keep a tight voxel family, or train separate models per family.
- Incompatible channel biology
- Annotation rules that change mid-dataset

**Minimum case count:**
nnU-Net strongly recommends at least 10-20 annotated volumes for 3d_fullres to
generalize. "Many dendrites, properly cropped into subvolumes" is the realistic
path to reaching that count without needing 20 separate imaging sessions.

---

## 4. Training unit: single dendrite vs many dendrites

The right framing is source image vs training unit — these are different things.

**Best training unit:** a clean 3D subvolume with clearly annotatable dendrite/spines.

That subvolume can come from:
- An image with one dendrite (already a clean crop)
- A larger image with many dendrites (crop and annotate each segment separately)

A single-dendrite crop is excellent because:
- Labels are less ambiguous
- Easier to define shaft vs spine
- Easier to validate errors
- Easier to compare models fairly

A larger field with many dendrites is also useful because it adds:
- Diversity, more spines, more backgrounds, more morphology examples

A 512×512×80 field with 6 dendrites becomes 6 subvolumes — each independently
annotatable, each a clean training case.

The rule is not "one dendrite only." The rule is:
**each training sample should contain structures that can be annotated unambiguously.**

---

## 5. Summary table

| Property                  | CARE           | SelfNet        | nnU-Net                  |
|---------------------------|----------------|----------------|--------------------------|
| Microscope/modality       | must match     | must match     | tolerate some variation  |
| Voxel size                | must match     | must match     | keep tight family        |
| XY/Z ratio                | close          | must match     | resampling can help      |
| Channel type              | must match     | must match     | must match               |
| Noise/preprocessing       | must match     | must match     | less critical            |
| Paired same-FOV images    | required       | not needed     | not needed               |
| Annotation consistency    | n/a            | n/a            | most important factor    |
| Biological condition      | not a factor   | not a factor   | not a factor by itself   |

---

## 6. alignment

The models are already support this split directly:
- CARE-3D, SelfNet, and nnU-Net are trained separately with their own pipelines.
- Restoration models are modality- and resolution-specific and should be retrained
  for the microscope/objective/resolution used.
- Pretrained segmentation models are organized by imaging family
  (65×65×150 nm, 65×65×65 nm, 102×102×1000 nm), confirming that even segmentation
  models are grouped by voxel geometry, not mixed freely.

---

## 7. Why train CARE at all if you can just take clean images?

Because you almost never have the luxury of simply choosing "take clean images."
Getting a clean image requires more photons, and more photons means:

- Longer exposure → phototoxicity stresses or kills the cell
- Higher laser power → photobleaching burns out the fluorophore
- More averaging frames → more time, the cell moves, you drift

So the real-world choice is not "noisy vs clean." It is:
- Fast, low-dose acquisition → noisy, but the cell stays alive and healthy
- Slow, high-dose acquisition → clean, but you've damaged or killed the sample

CARE lets you have both. You image a small paired dataset once — the same field,
first fast/noisy, then slow/clean. You train CARE on those pairs. Then for your
actual experiment you only ever image fast and noisy (safe for the cell), and
CARE restores it to look clean.

For spine imaging this matters directly: dendrites are fragile. A long high-power
Z-stack to get a clean image causes spines to retract or the cell to die. You want
acquisition to be as fast and gentle as possible, especially for time-lapse or
condition comparisons.

The trade-off: a small upfront investment (one paired acquisition session) buys
you the ability to image gently across your entire experiment and still get
clean-looking data.

If your images are already clean enough without it, you skip CARE entirely —
RESPAN supports running without a restoration model.

---

## 8. The chicken-and-egg problem for SelfNet training data

The correct pipeline order is: CARE first, then SelfNet uses the CARE output.
But without paired data to train CARE, there is no path to clean input for SelfNet either.

Current data situation:

- lifeact: bad signal density, mostly empty stacks → bad SelfNet training data regardless
- extracted_ch2: too noisy → bad SelfNet training data, needs CARE first
- CARE: can't train, no paired data


