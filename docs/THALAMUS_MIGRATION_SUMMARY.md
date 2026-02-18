# Thalamus Migration Summary

## What this is
This summary tracks migration status across:

1. Base MATLAB implementation (`SegmentThalamicNuclei.m`)
2. Updated MATLAB implementation (`TS_fnc_thalamus_seg_gem_joint.m`)
3. Current Python implementation (`segment_subregions` + `ThalamicNuclei`)

Detailed docs:
- `docs/dev/migration/thalamus-base-matlab.md`
- `docs/dev/migration/thalamus-updated-matlab.md`
- `docs/dev/migration/thalamus-python-current.md`
- `docs/dev/migration/thalamus-feature-matrix.md`
- `docs/dev/migration/thalamus-gap-analysis.md`

## Current status (high level)

- Structural thalamus segmentation pipeline parity is **partially achieved** in Python.
- Dynamic modality support and joint structural+diffusion modeling from updated MATLAB are **not yet migrated**.
- Core thalamus postprocessing conventions (reticular handling, largest-CC filtering, whole-thalamus totals) are present in Python.

## Scorecard by capability area

| Area | Status vs updated MATLAB | Notes |
|---|---|---|
| Structural-only thalamus pipeline | Partial | Present in Python, but with narrower input interface |
| Dynamic structural channels | Missing | Updated MATLAB supports explicit `N_Structural`; Python CLI fixed to `norm.mgz` |
| DTI likelihood integration | Missing | No DTI likelihood families in Python thalamus path |
| Joint sMRI+dMRI optimization | Missing | Updated MATLAB has joint objective and tensor/GMM coupling |
| Postprocessing conventions | Present | Core label cleanup logic is retained |
| Output diagnostics/configurability | Missing/Partial | Updated MATLAB is much richer |
| Longitudinal support | Partial | Present in Python subregions framework; not mapped for updated MATLAB entrypoint |

## Master migration checklist

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

## Suggested execution sequence

1. Implement dynamic structural input surface (P0).
2. Implement reduced DTI init channel and interface plumbing (P0).
3. Implement DTI likelihood families and shared-parameter routing (P0).
4. Implement joint sMRI+dMRI objective and optimization loop integration (P0).
5. Bring up parity-quality improvements (P1), then diagnostics/control completeness (P2).
