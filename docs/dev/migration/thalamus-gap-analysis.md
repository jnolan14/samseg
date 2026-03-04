# Thalamus Migration Gap Analysis

This file prioritizes migration gaps from current Python to updated MATLAB capability, using the matrix in `docs/dev/migration/thalamus-feature-matrix.md`.

## Priority legend
- `P0`: blocks parity on dynamic modality / joint sMRI+dMRI capability, or blocks branch integration
- `P1`: major quality or robustness gap
- `P2`: tooling, ergonomics, diagnostics, or secondary behavior

## Immediate branch integration blockers (`dti_integration`)

### GAP-BR-01: `thalamusDTI` constructor not wired from CLI parameters
- Current state: `model_lookup` includes `thalamusDTI`, but CLI parameter assembly does not provide required args (`atlasDir`, `inputDTIDirName`, `dtiLikelihood`).
- Why it matters: selecting `thalamusDTI` from CLI cannot run cleanly.
- Evidence:
  - `samseg/subregions/process.py:16`
  - `samseg/cli/segment_subregions.py:90`
  - `samseg/subregions/thalamusDTI.py:22`
- Verification definition:
  - CLI/API can invoke `thalamusDTI` end-to-end with explicit DTI options and no constructor errors.

### GAP-BR-02: Hard-coded external JSON path
- Current state: code reads `/autofs/.../means_groupings.json` directly.
- Why it matters: non-portable runtime dependency.
- Evidence:
  - `samseg/subregions/thalamusDTI.py:324`
- Verification definition:
  - Grouping JSON is configurable (argument or atlas-relative default) and validated at runtime.

### GAP-BR-03: Shared utility bug in hyperparameter hook path
- Current state: `bimodal_thal_hack()` calls `find_hyps_idx()` with missing argument.
- Why it matters: runtime error on hook usage.
- Evidence:
  - `samseg/subregions/utils.py:70`
  - `samseg/subregions/utils.py:114`
- Verification definition:
  - Hook path executes without argument/shape errors in test or controlled run.

### GAP-BR-04: Debug instrumentation in production paths
- Current state: heavy `print()` output and `breakpoint()` in `thalamusDTI` and shared `core.py`.
- Why it matters: unstable/fragile runtime and noisy logs.
- Evidence:
  - `samseg/subregions/thalamusDTI.py:516`
  - `samseg/subregions/core.py:608`
- Verification definition:
  - Production runs contain only intentional logging and no interactive breakpoints.

## P0 gaps

### GAP-P0-01: Dynamic structural channel configuration
- Current (Python): CLI hard-codes `['norm.mgz']`.
- Target (updated MATLAB): explicit `N_Structural` control and T1/T2 structural channel loading.
- Why it matters: baseline requirement for sequence-adaptive path and multi-contrast support.
- Evidence:
  - Python: `samseg/cli/segment_subregions.py:65`
  - Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:168`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:726`
- Verification definition:
  - CLI/API accepts configurable structural channel list and executes with 1 and 2 channels.
  - Hyperparameter estimation consumes configured channels.

### GAP-P0-02: DTI likelihood family integration
- Current (Python): no DTI likelihood interface in stable thalamus path.
- Target (updated MATLAB): `DTIlikelihood` family with DTI shared-parameter routing.
- Why it matters: core 2019/2023 capability.
- Evidence:
  - Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:12`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:314`
- Verification definition:
  - Python pathway supports at least one joint DTI likelihood mode behind explicit options.
  - Likelihood selection changes parameter-file routing and objective assembly.

### GAP-P0-03: Reduced DTI initialization channel
- Current (Python): absent in stable path, partial prototype work in branch.
- Target (updated MATLAB): optional reduced DTI init channel (`FA`/`log(det)` path).
- Why it matters: critical bridge from coarse initialization to joint model.
- Evidence:
  - Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:705`
- Verification definition:
  - Python can generate/load reduced DTI init channel and include it in initialization dimensions.

### GAP-P0-04: Joint structural + DTI deformation objective
- Current (Python): stable objective is structural intensity-only in thalamus module.
- Target (updated MATLAB): combined GMM + WMM/tensor objective in deformation calculator.
- Why it matters: defining behavior of updated method.
- Evidence:
  - Python image fitting flow: `samseg/subregions/process.py:65`
  - Updated MATLAB calculator construction: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:3199`
- Verification definition:
  - Python objective includes both structural and diffusion terms and runs deformation iterations with both contributions.

## P1 gaps

### GAP-P1-01: Hyperparameter estimation still single-image centric
- Current (Python): `get_gaussian_hyps` uses only first input image with TODO.
- Target: channel-aware hyperparameter estimation.
- Evidence:
  - Python: `samseg/subregions/thalamus.py:255`
  - Updated MATLAB dimensional handling: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:299`
- Verification definition:
  - Hyperparameter estimation scales with configured channel dimension.

### GAP-P1-02: Non-T1 second-component handling not generalized
- Current (Python): hard-coded T1 branch active; non-T1 branch disabled by constant `if True`.
- Target: modality-aware handling as in updated framework.
- Evidence:
  - Python: `samseg/subregions/thalamus.py:335`
- Verification definition:
  - Component priors adapt to modality configuration and are no longer hard-coded to T1-only behavior.

### GAP-P1-03: Coarse segmentation fallback ecosystem
- Current (Python): assumes `aseg.mgz` input from CLI path.
- Target: fallback to synthseg/GIF-derived coarse labels.
- Evidence:
  - Python: `samseg/cli/segment_subregions.py:66`
  - Updated MATLAB fallback: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:492`
- Verification definition:
  - Python accepts configurable coarse segmentation source with documented fallback order.

## P2 gaps

### GAP-P2-01: Output controls and diagnostics
- Current (Python): fixed core outputs; no configurable likelihood/posterior export set.
- Target: richer output switches (grouped, posterior, likelihood, objective plots, parameter dumps).
- Evidence:
  - Python output path: `samseg/subregions/thalamus.py:152`
  - Updated MATLAB controls: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:126`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:4049`
- Verification definition:
  - Python adds opt-in output controls with deterministic naming and docs.

### GAP-P2-02: Reprocess-only and reload/caching controls
- Current (Python): no equivalent `reprocessPosteriors`/`switch_forceReload` semantics.
- Target: support cached initialization reload and posterior-only regeneration modes.
- Evidence:
  - Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:104`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:229`
- Verification definition:
  - Python can skip/refit stages based on explicit run-mode flags with clear precondition checks.

### GAP-P2-03: Reflection and voxel-ratio controls
- Current (Python): absent in thalamus public options.
- Target: parity for reflection modeling and voxel-ratio weighting controls where relevant.
- Evidence:
  - Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:110`, `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:224`
- Verification definition:
  - Python exposes and tests optional controls, or explicitly documents intentional non-parity.

## Suggested migration order
1. GAP-BR-01
2. GAP-BR-02
3. GAP-BR-03
4. GAP-BR-04
5. GAP-P0-01
6. GAP-P0-03
7. GAP-P0-02
8. GAP-P0-04
9. GAP-P1-01
10. GAP-P1-02
11. GAP-P1-03
12. GAP-P2-01
13. GAP-P2-02
14. GAP-P2-03

## Open questions
- Whether Python parity target includes full diffusion/tensor model in this repo, or only structural dynamic-channel parity first.
- Whether updated-MATLAB longitudinal behavior should also be mapped (not evident from this entrypoint alone).
