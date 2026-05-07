Actual bugs / silent failures:

The Z-plane constraint in SelfNet has no guard — it crashes silently with an empty array. A pre-flight check that tells you exactly which images fail and why would save a lot of debugging.
The multi-image SelfNet support we added was missing entirely — that's already a real contribution.
Pipeline gaps:

There's no automated way to preprocess training images for nnU-Net with CARE/SelfNet before training. You have to do it manually. A proper prepare_nnunet_training.py that applies your trained restoration models to imagesTr/ automatically would close the loop.
No validation that your training images and analysis images came from the same "imaging family" before you run — you could accidentally train nnU-Net on raw images and analyze restored ones, or mix voxel size families, with no warning.
Usability:

The prepare_respan_experiments.py you have — batch experiment preparation is something RESPAN doesn't do natively. Making that robust and well-documented is a real contribution.
No metadata-driven automatic parameter selection — your .txt spec files next to each image already contain voxel size, Z step, channel info. A script that reads those and auto-sets scale, max_v, bg_threshold etc. would remove a big source of human error.

validation and guidance — a script that looks at your data, reads the spec files, and tells you:

which images are valid for SelfNet (pass Z constraint, same voxel family)
whether your nnU-Net training images match what you're feeding at inference
whether you're accidentally mixing voxel families