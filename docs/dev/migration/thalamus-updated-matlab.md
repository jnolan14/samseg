# Updated MATLAB Thalamus (`TS_fnc_thalamus_seg_gem_joint.m`)

## Entrypoint
- Function-first entrypoint with `inputParser` and broad option surface.
- Source: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m`

## Key Behaviors
- Dynamic modality controls include `N_Structural` (`1` or `2`) for structural channel count.
- Dynamic modality controls include `DTIlikelihood` families (`DSWbeta`, `wishart`, `logFrobenius`, `structural`, `FA`, `Trace`).
- Dynamic modality controls include `reduced_channel_DTI_initSwitch` for reduced DTI init channel.
- Joint optimization path combining structural and diffusion terms.
- Runtime controls for reload/reprocess (`switch_forceReload`, `reprocessPosteriors`).
- Optional reflection and voxel-ratio weighting controls.
- Rich diagnostics/output controls (`post_and_like_out`, parameter snapshots, grouped outputs).

## Evidence Pointers
- Parser surface: lines ~104-229.
- Dynamic dimensionality (`GMM_dimension`): lines ~299-302.
- Registration/reload behavior: lines ~574 onward.
- Reduced DTI init channel: lines ~709 onward.
- Joint calculator setup: lines ~3199-3203.
- Parameter output (`Parameters.mat`): lines ~4049 onward.

## Migration Notes
- This is the effective parity target for sequence-adaptive structural+DTI segmentation.
- Python parity should be measured against this entrypoint, not only the 2018 base script.
