# Thalamus Current Python Implementation (`segment_subregions` / `ThalamicNuclei`)

## Scope
This document captures the current Python implementation in this repo:

- `samseg/cli/segment_subregions.py`
- `samseg/subregions/process.py`
- `samseg/subregions/thalamus.py`
- supporting base class behavior in `samseg/subregions/core.py`

## Entrypoints and interfaces

### CLI interface
`segment_subregions` exposes thalamus segmentation through:

- `segment_subregions thalamus --cross <subject>`
- `segment_subregions thalamus --long-base <base>`

Evidence:
- `samseg/cli/segment_subregions.py:17`
- `samseg/cli/segment_subregions.py:21`

### Structure wiring
Thalamus model class is wired through:

- `model_lookup['thalamus'] = ThalamicNuclei`

Evidence:
- `samseg/subregions/process.py:13`

## Input and modality behavior

### Current CLI defaults
- Input segmentation: `aseg.mgz`
- Input image list hard-coded to `['norm.mgz']`

Evidence:
- `samseg/cli/segment_subregions.py:65`
- `samseg/cli/segment_subregions.py:66`

### Multi-image support status
- `ThalamicNuclei.preprocess_images()` loops over `self.inputImages` and stacks them, so the model path can accept multiple loaded images.
- However, CLI only passes one image today.
- Hyperparameter estimation is still single-image-only (`DATA = self.inputImages[0]`) with explicit TODO.
- Second-component hyperparameter logic has a TODO for non-T1 contrasts and currently hard-codes the T1-like branch.
- Base `prepare_for_image_fitting()` still has “multi-image cases down the road” TODO comment and squeezes working image data.

Evidence:
- image stacking loop: `samseg/subregions/thalamus.py:114`
- single-image TODO: `samseg/subregions/thalamus.py:255`
- hard-coded second-component branch TODO: `samseg/subregions/thalamus.py:335`
- base-class multi-image TODO: `samseg/subregions/core.py:415`

## Pipeline summary

Cross-sectional path (`run_cross_sectional`):
1. Initialize model and preprocess.
2. Align atlas to segmentation mask.
3. Fit mesh to segmentation (cheating stage).
4. Fit mesh to image.
5. Extract and postprocess segmentation and volumes.

Evidence:
- `samseg/subregions/process.py:32`
- `samseg/subregions/process.py:43`
- `samseg/subregions/process.py:50`
- `samseg/subregions/process.py:57`
- `samseg/subregions/process.py:65`

Longitudinal path (`run_longitudinal`) exists and includes base-space alignment and global iterations.

Evidence:
- `samseg/subregions/process.py:79`

## Thalamus-specific behavior

### Atlas and core settings
- Atlas path: `${FREESURFER_HOME}/average/ThalamicNuclei/atlas`
- Two-component modeling enabled.
- Cheating schedule: `[3.0, 2.0]`
- Image fit schedule: `meshSmoothingSigmas=[1.5, 1.125, 0.75, 0]`, `maxIterations=[7,5,5,3]`

Evidence:
- `samseg/subregions/thalamus.py:14`
- `samseg/subregions/thalamus.py:18`
- `samseg/subregions/thalamus.py:21`
- `samseg/subregions/thalamus.py:25`

### Preprocessing and postprocessing
- Uses ASEG-derived TH/ventral-DC target mask for alignment.
- Builds synthetic ASEG (`TH` merged with `VentralDC`) for cheating fit.
- Crops around thalamus and masks image before fitting.
- Postprocessing removes reticular labels, keeps largest components per side, writes thalamic outputs and volume summaries.

Evidence:
- TH/DE constants and mask: `samseg/subregions/thalamus.py:43`
- synthetic merge: `samseg/subregions/thalamus.py:95`
- crop/mask: `samseg/subregions/thalamus.py:103`
- reticular and CC filtering: `samseg/subregions/thalamus.py:140`
- outputs: `samseg/subregions/thalamus.py:152`
- whole-thalamus totals: `samseg/subregions/thalamus.py:165`

## Output artifacts

- `ThalamicNuclei<suffix>.mgz`
- `ThalamicNuclei<suffix>.FSvoxelSpace.mgz`
- `ThalamicNuclei<suffix>.volumes.txt`

Evidence:
- `samseg/subregions/thalamus.py:152`
- `samseg/subregions/thalamus.py:154`
- `samseg/subregions/thalamus.py:169`

## Migration-relevant notes

- Current Python preserves core structural thalamus pipeline behavior from base MATLAB.
- Major updated-MATLAB capabilities (broad dynamic modality interface, DTI likelihood families, rich output diagnostics) are not exposed in this Python thalamus path yet.
