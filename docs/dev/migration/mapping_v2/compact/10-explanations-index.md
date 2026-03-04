# Compact Mapping Explanations Index

This explanation layer augments the compact mapping tables with decision-ready
context for merge and migration work.

## How to use
1. Start with compact tables for scan speed.
2. Use the explanation docs for mismatch rationale and missing behavior.
3. Use `20-gap-action-summary.md` as the implementation checklist.

## Canonical row-key contract
- Row keys are copied verbatim from compact tables.
- Explanation docs focus on `Partial` and `None` entries.
- `Priority` and `Confidence` are provided per row key.

## Relevant files and purposes
- `core.py`: shared mesh/EM/deformation engine used by subregion models.
- `thalamus.py`: stable structural thalamus implementation.
- `thalamusDTI.py`: DTI prototype implementation in `dti_integration`.
- `process.py`: model dispatch and high-level run orchestration.
- `utils.py`: shared compression/grouping and branch-specific helper functions.
- `TS_fnc_thalamus_seg_gem_joint.m`: updated MATLAB entrypoint and control flow.
- `SegmentThalamicNuclei.m`: original MATLAB entrypoint.

## Companion docs
- `11-table1-explanations-by-subsystem.md`
- `12-table2-explanations-by-subsystem.md`
- `13-table3-explanations-by-subsystem.md`
- `20-gap-action-summary.md`
