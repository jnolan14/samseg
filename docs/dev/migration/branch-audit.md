# Branch Audit for Thalamus Migration

## Compared Branches
- Target base: `origin/dev` (`4ed1da39`)
- Candidate code branch: `origin/dti_integration` (`c82b0b9f`)
- Candidate docs branch: `origin/docs/dti-migration-checklists` (`9e222f57`)

## Change Summary

### `origin/dev...dti_integration`
- `M samseg/subregions/core.py`
- `M samseg/subregions/process.py`
- `A samseg/subregions/thalamusDTI.py`
- `M samseg/subregions/utils.py`
- Diff size: 1381 insertions, 25 deletions

### `origin/dev...docs/dti-migration-checklists`
- Adds migration docs under `docs/dev/migration/`
- Adds `docs/THALAMUS_MIGRATION_SUMMARY.md`
- Adds `AGENTS.md`
- Modifies `setup.cfg` (surfa source pointer)

## Findings (Ordered by Severity)

### P0 blockers
1. `thalamusDTI` constructor is not wired from CLI parameters.
- `process.py` registers `thalamusDTI` in `model_lookup`, but CLI parameter builder only supplies baseline args.
- Evidence: `samseg/subregions/process.py:14`, `samseg/cli/segment_subregions.py:90`, `samseg/subregions/thalamusDTI.py:22`

2. Hard-coded external dependency path in runtime code.
- `thalamusDTI.initialize()` reads a JSON from `/autofs/.../means_groupings.json`.
- Evidence: `samseg/subregions/thalamusDTI.py:324`

3. Immediate bug in helper hook path.
- `bimodal_thal_hack()` calls `find_hyps_idx()` without required `label_groupings` argument.
- Evidence: `samseg/subregions/utils.py:70`, `samseg/subregions/utils.py:114`

### P1 high-risk items
1. Large debug surface in core training/inference loops.
- Many unconditional prints in `core.py` hot paths.
- Evidence: `samseg/subregions/core.py:459`, `samseg/subregions/core.py:608`, `samseg/subregions/core.py:736`

2. Debug breakpoints remain in model code.
- `breakpoint()` still present in `thalamusDTI.py`.
- Evidence: `samseg/subregions/thalamusDTI.py:516`, `samseg/subregions/thalamusDTI.py:960`, `samseg/subregions/thalamusDTI.py:985`

3. Multi-channel refactor changes shared `MeshModel` behavior for all subregion structures.
- Needs structure-wide regression verification before merge.
- Evidence: `samseg/subregions/core.py`

## Merge Plan
1. Create `integration/thalamus-migration` from `origin/dev`.
2. Bring docs commits first (prefer commit-level cherry-pick).
3. Port DTI code with safety gates:
- Add CLI args for `thalamusDTI` or temporarily remove `thalamusDTI` from `model_lookup` until wired.
- Replace hard-coded JSON path with configurable atlas-relative default.
- Remove debug print/breakpoint instrumentation.
- Fix `utils.bimodal_thal_hack()` signature usage.
4. Run non-DTI regressions (`thalamus`, `brainstem`, `hippo-amygdala`) due `core.py` touch points.

## Suggested PR split
- PR1: docs and migration matrices only.
- PR2: interface and wiring (`segment_subregions.py`, `process.py`).
- PR3: `thalamusDTI.py` stabilization.
- PR4: `core.py` multi-channel hardening + regression tests.
