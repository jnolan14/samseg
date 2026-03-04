# Thalamus Mapping V2 Index

This directory contains new mapping artifacts (separate from prior migration docs) for branch merge and parity tracking.

Pinned SHAs:
- `origin/dev`: `4ed1da39e1e6150de8b3f0a812a5c87908cd3636`
- `origin/dti_integration`: `c82b0b9f48656b2a30070e40d8ae65dc08c299f3`
- docs branch at creation: `17e751762697ca70e7ac76c1fd2bbc7eb6f3f9d4`

Artifacts:
- `ledgers/matlab_called_project_functions.tsv`
- `ledgers/dti_integration_changed_symbols.tsv`
- `ledgers/mapping_conflict_queue.tsv`
- `01-matlab-called-functions-to-python-branches.md`
- `02-dti-integration-symbols-to-matlab.md`
- `03-dti-integration-symbols-to-dev.md`

Coverage counts:
- MATLAB called project functions mapped in Table 1: `47`
- dti_integration changed symbols mapped in Table 2: `31`
- dti_integration changed symbols mapped in Table 3: `31`
- Unresolved conflicts in conflict queue: `0` (header-only file)

Confidence rubric:
- `High`: direct semantic + structural correspondence with clear evidence anchors.
- `Medium`: same intent with structural/API differences.
- `Low`: nearest plausible equivalent but incomplete or indirect alignment.
- `None`: no meaningful equivalent located.

Match rubric:
- `Exact`: equivalent behavior and role.
- `Partial`: overlapping role but missing behavior/coverage.
- `None`: no equivalent implementation block.
