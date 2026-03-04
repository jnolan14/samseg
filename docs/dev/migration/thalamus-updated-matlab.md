# Thalamus Updated MATLAB Implementation (`TS_fnc_thalamus_seg_gem_joint.m`)

## Scope
This document captures the updated MATLAB entrypoint behavior at:

- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m`

## Entry point and parameter surface

Entrypoint is function-based with `varargin` and `inputParser`.

Evidence:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:1`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:101`

### Major parameter groups

Data/runtime:
- `data_dir`, `subject`, `out_dir`, `temp_dir`, `remove_temp`, `seg_name`

Initialization/alignment:
- `switch_forceReload`, `switch_useSynthseg`, `switch_useReflection`, `init_param_file`, `reprocessPosteriors`

Modality/model:
- `DTIlikelihood` in `{logFrobenius,wishart,DSWbeta,Structural,FA,Trace}`
- `N_Structural` in `{1,2}`
- `reduced_channel_DTI_initSwitch`
- Wishart priors `alpha`, `beta`

Optimization:
- separate structural/GMM and joint schedules (`meshSmoothingSigmas*`, `max_def_its*`, `N_iter*`)
- `meshStiffnessKbase`, `modifyStiffnessFlag`
- voxel-ratio controls (`correctForVoxelRatio`, `setVoxelRatio`, `hyperVoxelRatio`)

Outputs/debugging:
- `intermediate_out_switch`, `parameter_out_switch`, `post_and_like_out`, `output_smoothed_switch`, `output_detailed_stats`, `use_parfor_switch`

Evidence:
- parser declarations: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:104`
- likelihood options: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:12`
- structural channel count: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:168`

## Dynamic modality handling

### Structural channels
- `N_Structural=1` uses T1 (`norm.mgz`)
- `N_Structural=2` adds T2 (`T2w_hires.norm.mgz`)

Evidence:
- source files: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:468`
- channel load: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:723`

### DTI integration
- Supports multiple DTI likelihood families and likelihood-specific default parameter files.
- Can add reduced DTI channel to structural initialization (`FA` or `log(det)` path).
- Uses full DTI likelihood terms in joint optimization stage.

Evidence:
- likelihood routing defaults: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:314`
- reduced DTI init channel generation: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:705`
- structural+DTI deformation calculator: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3199`

### Coarse segmentation fallback
Initialization segmentation can come from `aseg.mgz`, `synthseg.mgz`, `aseg_from_gif.mgz`, or converted `gif_parcellation*`.

Evidence:
- fallback logic: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:480`

## Pipeline summary

1. Parse options and normalize runtime flags.
2. Build subject-specific directories and source file set.
3. Optionally flip LR for debugging (`fliplr`).
4. Register atlas to coarse segmentation (`TS_fnc_AffineImageDumpRegistration`, `TS_fnc_ASEGAtlasRegistration`), with cache/reload behavior.
5. Optionally derive reduced DTI init channel and load structural + init channels.
6. Merge ASEG/SynthSeg priors with atlas and compute reduced/grouped class maps.
7. Fit initial structural GMM stage (with optional reduced DTI init channel).
8. Fit tensor/WMM distributions and then run full joint GEM loops (structural + DTI terms).
9. Generate final posteriors/likelihoods/segmentations, optional smoothing and grouped outputs.
10. Optionally emit diagnostics and parameter snapshots, then clear temp dir.

Evidence:
- atlas registration/caching: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:574`
- merge stage: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:801`
- group sorting: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:847`
- structural init loop: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:1094`
- joint deformation loop: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3171`
- final outputs: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3782`

## Output behavior

Core outputs include:
- final segmentation (`*_seg_final.nii.gz` or smoothed variant)
- optional per-group posterior volumes
- grouped segmentation (`*_seg_grouped.nii.gz`)
- optional objective figures and `Parameters.mat`

Evidence:
- final segmentation write: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3840`
- smoothed branch write: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3987`
- grouped output: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3993`
- parameter export: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:4049`

## Migration-relevant notes

- This is the primary implementation carrying dynamic modality support and sequence-adaptive behavior.
- The configuration surface is substantially broader than base MATLAB and current Python CLI.
- The code path explicitly separates initialization-space channels from full joint-likelihood fitting.
