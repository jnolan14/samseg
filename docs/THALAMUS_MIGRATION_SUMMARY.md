# Thalamus Migration Summary

## Scope
This summary documents migration status across three implementation lines:

1. Original MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m`
2. Updated MATLAB: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m`
3. Python in this repo: `samseg/subregions/thalamus.py`, `samseg/subregions/thalamusDTI.py`, `samseg/subregions/core.py`, `samseg/subregions/process.py`

Audit baseline (2026-02-24):
- `origin/dev`: `4ed1da39`
- `origin/dti_integration`: `c82b0b9f`
- `origin/docs/dti-migration-checklists`: `9e222f57`

Detailed docs:
- `docs/dev/migration/branch-audit.md`
- `docs/dev/migration/thalamus-base-matlab.md`
- `docs/dev/migration/thalamus-updated-matlab.md`
- `docs/dev/migration/thalamus-python-current.md`
- `docs/dev/migration/thalamus-feature-matrix.md`
- `docs/dev/migration/thalamus-gap-analysis.md`

## Current Status
- Base structural thalamus workflow is implemented in Python (`thalamus`).
- Updated-MATLAB-style dynamic structural + diffusion framework is partially started in Python (`thalamusDTI`) but not production-wired.
- `dti_integration` currently has merge blockers that should be fixed before landing on `origin/dev`.

## Branch Merge Checklist

### Docs branch (`docs/dti-migration-checklists`)
- [ ] Cherry-pick docs commits into integration branch.
- [ ] Decide whether to keep or drop `setup.cfg` surfa change (`30a5669`).
- [ ] Keep doc-only changes isolated from runtime code changes.

### DTI code branch (`dti_integration`)
- [ ] Fix CLI/constructor wiring mismatch for `thalamusDTI`.
- [ ] Remove hard-coded external JSON path in `thalamusDTI.initialize()`.
- [ ] Remove debug prints and `breakpoint()` calls from production path.
- [ ] Fix `utils.bimodal_thal_hack()` call to `find_hyps_idx()`.
- [ ] Verify `core.py` multi-channel changes do not regress non-DTI structures.

### Merge sequencing (recommended)
- [ ] Create integration branch from `origin/dev`.
- [ ] Apply docs-only commits first.
- [ ] Port `dti_integration` code changes in focused commits (`process.py`, `thalamusDTI.py`, `core.py`, `utils.py`).
- [ ] Run targeted regression tests for `thalamus`, `brainstem`, `hippo-amygdala`.

## MATLAB Parity Checklist

### P0 (parity-critical)
- [ ] Dynamic structural channel config (updated MATLAB `N_Structural`) in public CLI/API.
- [ ] DTI likelihood family selection and parameter routing.
- [ ] Reduced DTI init channel support.
- [ ] Joint structural + DTI deformation objective.

### P1 (quality/robustness)
- [ ] Channel-aware hyperparameter estimation.
- [ ] Modality-aware non-T1 second-component behavior.
- [ ] Coarse-seg fallback beyond `aseg.mgz`.

### P2 (controls/diagnostics)
- [ ] Reprocess/reload run modes.
- [ ] Output/diagnostic switches (posteriors, likelihoods, parameter snapshots).
- [ ] Reflection and voxel-ratio controls if parity target includes full diffusion path.

## Paper Context (Intent)
- 2018 NeuroImage: *A probabilistic atlas of the human thalamic nuclei combining ex vivo MRI and histology*.
- 2019 IPMI: *Joint inference on structural and diffusion MRI for sequence-adaptive Bayesian segmentation of thalamic nuclei with probabilistic atlases*.
- 2023: *Accurate Bayesian segmentation of thalamic nuclei using diffusion MRI and an improved histological atlas*.

Paper files:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2018_NeuroImage_Iglessias_thalamicAtlas.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2019_IPMI_Iglesias_diffusionMRBayesianSeg.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/Tregidgo et al. - 2023 - Accurate Bayesian segmentation of thalamic nuclei using diffusion MRI and an improved histological a.pdf`
