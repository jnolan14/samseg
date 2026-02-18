# Thalamus Base MATLAB Implementation (`SegmentThalamicNuclei.m`)

## Scope
This document captures the behavior of the original MATLAB thalamic segmentation entrypoint at:

- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m`

It is used as implementation ground truth for migration comparison.

## Entry point and API surface

### Declared interface (commented signature)
The file documents this function interface:

- `SegmentThalamicNuclei(subjectName, subjectDir, resolution, atlasMeshFileName, atlasDumpFileName, compressionLUTfileName, K, optimizerType, suffix, FSpath, useTwoComponents, MRFconstant, additionalVol, analysisID, doBFcorrection, BBregisterMode)`

Evidence:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:7`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:32`

### Runtime shape in this repo copy
In this copy, the function declaration is commented and the file contains script-style setup blocks (`clear`, hard-coded test values) before pipeline execution.

Evidence:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:32`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:56`

### Validation and option checks
The script enforces:

- minimum argument count (`nargin >= 11`)
- optimizer type in `{FixedStepGradientDescent, GradientDescent, ConjugateGradient, L-BFGS}`
- numeric checks on key scalar parameters
- optional additional-volume contract (`additionalVol`, `analysisID`, `doBFcorrection`, `BBregisterMode`)
- `BBregisterMode` in `{none, t1, t2}`

Evidence:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:139`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:153`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:175`

## Pipeline summary

1. Configure flags, parse optional environment toggles, and clear KVL state.
2. Build temp workspace path (optionally redirected to `/scratch` via `USE_SCRATCH`).
3. Optionally register an additional image into subject space (`bbregister` or `mri_convert`) and create registration QC GIF.
4. Build TH+ventral-DC mask from ASEG and affine-align atlas image dump (`imageDump.mgz`) to this target.
5. Prepare modified ASEG/synthetic target and perform “cheating” mesh fit to segmentation.
6. Prepare intensity image (`norm.mgz` or additional volume with optional bias correction), crop around thalamus, resample to working resolution, and mask.
7. Perform main mesh-to-image optimization (with one- or two-component model).
8. Build MAP labels and posterior-derived volumes, remove reticular labels, keep largest connected components, and write outputs.
9. Resample labels back to FreeSurfer voxel space and move results into subject `mri` directory.
10. Optionally write posteriors and mesh/debug artifacts; cleanup temp directory.

Evidence:
- temp and KVL setup: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:208`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:215`
- additional volume registration: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:230`
- TH/DE registration target: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:282`
- cheating fit schedule: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:535`
- intensity selection and BF correction path: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:724`
- crop/resample/mask: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:838`
- MAP and volume writeout: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1809`
- reticular suppression + CC filtering: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1910`
- final outputs and move: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:2064`
- cleanup: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:2091`

## Input and modality handling

- Structural default input is effectively `norm.mgz`.
- Optional single `additionalVol` can replace T1 analysis image (not fused as a parallel channel).
- Optional bias-field correction exists for the additional volume (`doBFcorrection`).
- Optional bbregister-based alignment exists for additional volume with contrast mode `t1`/`t2`.

Evidence:
- additional volume contract: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:23`
- registration mode handling: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:235`
- image selection logic: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:724`

## Optimization/modeling controls

- `useTwoComponents` is supported and expected to be on for thalamus.
- ASEG fit schedule is hard-coded (`meshSmoothingSigmas=[3.0,2.0]`).
- `MRFconstant` exists, but MRF smoothing path is disabled and raises an error if enabled.

Evidence:
- option description: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:21`
- cheating schedule: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:535`
- MRF disabled: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1928`

## Output artifacts

Primary outputs:

- `ThalamicNuclei.<suffix2>.mgz`
- `ThalamicNuclei.<suffix2>.FSvoxelSpace.mgz`
- `ThalamicNuclei.<suffix2>.volumes.txt`

Optional outputs:

- per-label posterior volumes (`posterior_*.mgz`)
- warped mesh outputs and transform text when `WRITE_MESHES>0`

Evidence:
- main moves: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:2064`
- posterior output toggle: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:2069`
- mesh output toggle: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:2074`

## Migration-relevant notes

- Supports one extra volume workflow but not generalized N-channel fusion.
- Includes modality-adaptive registration and optional BF correction for the extra volume.
- Uses explicit thalamus postprocessing conventions still reflected in Python (reticular removal, largest CCs, whole-thalamus sums).
