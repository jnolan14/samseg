# Thalamus Feature Matrix (Base MATLAB vs Updated MATLAB vs Python)

Status vocabulary:
- `Present`: implemented and directly evidenced
- `Partial`: some support exists, but limited vs target behavior
- `Missing`: not found in current implementation
- `Unknown`: not resolved from inspected artifacts

Paper tags:
- `Implemented (code)` = observed in code
- `Intent (paper)` = design objective from papers

## Matrix (stable Python path baseline)

| ID | Feature | Base MATLAB | Updated MATLAB | Python current | Evidence | Paper intent |
|---|---|---|---|---|---|---|
| F01 | Canonical function-style entrypoint | Partial (documented signature but repo copy is script-shaped) | Present (`function ... varargin`) | Present (CLI+class path) | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:32`; updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:1`; py: `samseg/cli/segment_subregions.py:42` | Intent: N/A |
| F02 | Coarse alignment from structural segmentation | Present | Present (ASEG/SynthSeg/GIF fallback) | Present (aseg-based) | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:282`; updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:480`; py: `samseg/subregions/thalamus.py:49` | 2018 `Implemented (code)` |
| F03 | Optional additional structural image | Present (single `additionalVol`) | Present (T1/T2 via `N_Structural`) | Partial (model can stack, CLI fixed to `norm.mgz`) | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:139`; updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:168`; py CLI: `samseg/cli/segment_subregions.py:65`, py model: `samseg/subregions/thalamus.py:114` | 2019 `Intent (paper)` |
| F04 | Explicit multi-contrast structural channel count control | Missing (single active analysis image at a time) | Present (`N_Structural` 1 or 2) | Missing (no user-facing control) | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:168`; py: `samseg/cli/segment_subregions.py:65` | 2019 `Intent (paper)` |
| F05 | DTI likelihood family support | Missing | Present (`DSWbeta`, `wishart`, `logFrobenius`, etc.) | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:12` | 2019/2023 `Intent (paper)` |
| F06 | Reduced DTI channel for initialization | Missing | Present (`reduced_channel_DTI_initSwitch`) | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:245`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:705` | 2019/2023 `Intent (paper)` |
| F07 | Full joint structural + DTI deformation objective | Missing | Present | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3199` | 2019/2023 `Intent (paper)` |
| F08 | Additional-volume registration mode control (`none/t1/t2`) | Present | Partial (different architecture; not base-style bbregister toggle) | Missing (no thalamus exposure) | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:175`; py core has `bbregisterMode` type only: `samseg/subregions/core.py:118` | 2019 `Intent (paper)` |
| F09 | Optional bias-field correction path for extra image | Present | Partial (different DTI/struct preprocessing path) | Missing | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:732` | Intent: N/A |
| F10 | Two-component thalamus model | Present | Present (broader mixture grouping framework) | Present | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1105`; py: `samseg/subregions/thalamus.py:18` | 2018 `Implemented (code)` |
| F11 | Reticular label removal and CC cleanup | Present | Partial (different grouped output postprocessing) | Present | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1910`; py: `samseg/subregions/thalamus.py:140` | 2018 `Implemented (code)` |
| F12 | Whole-thalamus hemisphere totals | Present | Present (through output grouping pipeline) | Present | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:1891`; py: `samseg/subregions/thalamus.py:165` | 2018 `Implemented (code)` |
| F13 | Rich output controls (posteriors, grouped, likelihood, params, figures) | Partial (`WRITE_POSTERIORS`, `WRITE_MESHES`) | Present (many output switches) | Missing (minimal outputs + debug intermediates) | base: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m:109`; updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:126`; py outputs: `samseg/subregions/thalamus.py:152` | 2023 `Intent (paper)` |
| F14 | Reprocess-only mode (skip optimization, regenerate posteriors) | Missing | Present (`reprocessPosteriors`) | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:229` | Intent: N/A |
| F15 | Initialization cache/reload control | Missing | Present (`switch_forceReload`, cached init files) | Partial (debug intermediates only, no equivalent switch) | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:574` | Intent: N/A |
| F16 | Reflection symmetry modeling controls | Missing | Present (`switch_useReflection`) | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:110` | 2023 `Intent (paper)` |
| F17 | Voxel-ratio scaling controls for DTI likelihood | Missing | Present | Missing | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:224` | 2019/2023 `Intent (paper)` |
| F18 | Longitudinal thalamus workflow | Missing | Unknown (not in this entrypoint) | Present (`run_longitudinal`) | py: `samseg/subregions/process.py:79` | Intent: N/A |
| F19 | Multi-image hyperparameter estimation | Missing | Present (GMM dimension built from channels) | Partial (explicit TODO; first image only in hyps) | updated: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:299`; py: `samseg/subregions/thalamus.py:255` | 2019/2023 `Intent (paper)` |
| F20 | Non-T1 second-component behavior | Present (via additionalVol path + settings) | Present (likelihood-dependent framework) | Partial (TODO, hard-coded T1 branch active) | py: `samseg/subregions/thalamus.py:335` | 2019/2023 `Intent (paper)` |
| F21 | Public tool packaging for command execution | Unknown | Unknown | Present (`segment_subregions` console script) | `setup.cfg:37` | Intent: N/A |

## DTI prototype overlay (`dti_integration` branch)

This overlay captures branch-only status that is not yet reflected in released/stable Python behavior:

| Area | `thalamusDTI` status | Evidence |
|---|---|---|
| Class presence and model lookup | Present | `samseg/subregions/process.py:16`, `samseg/subregions/thalamusDTI.py:22` |
| CLI execution readiness | Missing (constructor args not wired) | `samseg/cli/segment_subregions.py:90`, `samseg/subregions/thalamusDTI.py:22` |
| DTI likelihood option plumbing | Partial | `samseg/subregions/thalamusDTI.py:217` |
| Reduced DTI init channel parity | Partial | `samseg/subregions/thalamusDTI.py` |
| Production readiness | Missing (debug/breakpoint/hard-coded path) | `samseg/subregions/thalamusDTI.py:324`, `samseg/subregions/thalamusDTI.py:516` |

## Paper-intent notes (high level)

- 2018 paper frames a probabilistic atlas + Bayesian inference approach for thalamic nuclei segmentation from structural MRI with contrast robustness claims.
- 2019 paper abstract explicitly frames joint sMRI/dMRI Bayesian segmentation with sequence-adaptive behavior across acquisition protocols.
- 2023 paper title/abstract frame improved histological atlas + diffusion MRI integration for improved thalamic nuclei segmentation.

Paper files:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2018_NeuroImage_Iglessias_thalamicAtlas.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2019_IPMI_Iglesias_diffusionMRBayesianSeg.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/Tregidgo et al. - 2023 - Accurate Bayesian segmentation of thalamic nuclei using diffusion MRI and an improved histological a.pdf`
