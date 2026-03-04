# Thalamus Migration Summary

## Scope
This summary tracks migration status across three implementation lines:

1. Base MATLAB implementation (`SegmentThalamicNuclei.m`)
2. Updated MATLAB implementation (`TS_fnc_thalamus_seg_gem_joint.m`)
3. Current Python implementation (`segment_subregions` + `ThalamicNuclei`) plus `dti_integration` prototype work (`thalamusDTI`)

Audit baseline used for branch comparisons:
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

## Current Status (High Level)

- Structural thalamus segmentation pipeline parity is partially achieved in Python.
- Dynamic modality support and full joint structural+diffusion modeling from updated MATLAB are not yet parity-complete.
- Core thalamus postprocessing conventions (reticular handling, largest-CC filtering, whole-thalamus totals) are present in Python.
- `dti_integration` contains meaningful prototype work, but has immediate integration blockers before merge to `origin/dev`.

## Scorecard by Capability Area

| Area | Status vs updated MATLAB | Notes |
|---|---|---|
| Structural-only thalamus pipeline | Partial | Present in Python, but with narrower input interface |
| Dynamic structural channels | Missing | Updated MATLAB supports explicit `N_Structural`; Python CLI fixed to `norm.mgz` |
| DTI likelihood integration | Missing/Partial | DTI prototype exists in branch, not production-wired |
| Joint sMRI+dMRI optimization | Missing/Partial | Updated MATLAB has joint objective and tensor/GMM coupling |
| Postprocessing conventions | Present | Core label cleanup logic is retained |
| Output diagnostics/configurability | Missing/Partial | Updated MATLAB is much richer |
| Longitudinal support | Partial | Present in Python subregions framework; not mapped for updated MATLAB entrypoint |

## Branch Merge Checklist

### Docs branch (`docs/dti-migration-checklists`)
- [ ] Keep migration docs commits isolated from runtime code changes.
- [ ] Decide whether to keep or drop `setup.cfg` surfa pointer change from earlier docs branch history (`30a5669`).
- [ ] Land merged docs first so code PRs can reference stable checklist IDs.

### DTI code branch (`dti_integration`)
- [ ] Fix CLI/constructor wiring mismatch for `thalamusDTI`.
- [ ] Remove hard-coded external JSON path in `thalamusDTI.initialize()`.
- [ ] Remove debug prints and `breakpoint()` calls from production path.
- [ ] Fix `utils.bimodal_thal_hack()` call to `find_hyps_idx()`.
- [ ] Verify `core.py` multi-channel changes do not regress non-DTI structures.

### Merge sequencing (recommended)
- [ ] Create integration branch from `origin/dev`.
- [ ] Apply docs-only commits first.
- [ ] Port `dti_integration` changes in focused commits (`process.py`, `thalamusDTI.py`, `core.py`, `utils.py`).
- [ ] Run targeted regressions for `thalamus`, `brainstem`, `hippo-amygdala` because `core.py` is shared.

## Master Migration Checklist

### Inputs and channels
- [ ] Add configurable structural channel inputs (close GAP-P0-01, matrix F03/F04).
- [ ] Add reduced DTI initialization channel path (close GAP-P0-03, matrix F06).
- [ ] Add robust coarse segmentation fallback options (close GAP-P1-03, matrix F02/F03).

### Likelihood/modeling
- [ ] Implement DTI likelihood family selection and parameter routing (close GAP-P0-02, matrix F05).
- [ ] Implement joint structural + DTI deformation objective (close GAP-P0-04, matrix F07).
- [ ] Make Gaussian hyperparameter estimation channel-aware (close GAP-P1-01, matrix F19).
- [ ] Replace hard-coded T1-only second-component behavior with modality-aware logic (close GAP-P1-02, matrix F20).

### Optimization/runtime controls
- [ ] Add run-mode controls for reload/reprocess behavior (close GAP-P2-02, matrix F14/F15).
- [ ] Add optional reflection and voxel-ratio controls if parity target includes diffusion path (close GAP-P2-03, matrix F16/F17).

### Outputs and diagnostics
- [ ] Add richer, opt-in posterior/likelihood/grouped output controls (close GAP-P2-01, matrix F13).
- [ ] Add parameter snapshot export for reproducibility and re-entry workflows (close GAP-P2-01, matrix F13).

## Evidence model used

- `Implemented (code)`: claims grounded in source code line references.
- `Intent (paper)`: claims grounded in paper abstracts/titles and treated as design goals, not implementation proof.

Paper references used for intent mapping:
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2018_NeuroImage_Iglessias_thalamicAtlas.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/2019_IPMI_Iglesias_diffusionMRBayesianSeg.pdf`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/papers/Tregidgo et al. - 2023 - Accurate Bayesian segmentation of thalamic nuclei using diffusion MRI and an improved histological a.pdf`

## Suggested Execution Sequence

1. Clear branch integration blockers from `docs/dev/migration/branch-audit.md`.
2. Implement dynamic structural input surface (P0).
3. Implement reduced DTI init channel and interface plumbing (P0).
4. Implement DTI likelihood families and shared-parameter routing (P0).
5. Implement joint sMRI+dMRI objective and optimization loop integration (P0).
6. Bring up parity-quality improvements (P1), then diagnostics/control completeness (P2).
