# Python Thalamus Status in This Repo

## Structural (`thalamus`)
- Entrypoint wiring: `segment_subregions thalamus` via `model_lookup`.
- Preserves core structural behavior from base MATLAB.
- ASEG-driven atlas alignment target.
- Cheating fit then image fit.
- Reticular suppression + largest-CC filtering.
- Whole-thalamus volume totals and standard output files.

Evidence:
- `samseg/cli/segment_subregions.py:65`
- `samseg/subregions/process.py:14`
- `samseg/subregions/thalamus.py:43`
- `samseg/subregions/thalamus.py:140`
- `samseg/subregions/thalamus.py:165`

## DTI Prototype (`thalamusDTI`)
- New class exists and is registered in `model_lookup`.
- Adds parameters for `inputDTIDirName` and `dtiLikelihood`.
- Includes DTI directory parsing and FA loading.
- Contains substantial TODO/HACK/debug scaffolding and breakpoints.

Evidence:
- `samseg/subregions/process.py:16`
- `samseg/subregions/thalamusDTI.py:22`
- `samseg/subregions/thalamusDTI.py:106`
- `samseg/subregions/thalamusDTI.py:217`

## Integration Gaps (Current Branch)
1. CLI does not provide required constructor args for `thalamusDTI`.
- `samseg/cli/segment_subregions.py:90`
- `samseg/subregions/thalamusDTI.py:22`

2. Runtime hard-coded JSON path in `initialize()` is not portable.
- `samseg/subregions/thalamusDTI.py:324`

3. Utility hook bug in `bimodal_thal_hack()`.
- `samseg/subregions/utils.py:70`
- `samseg/subregions/utils.py:114`

4. Shared `MeshModel` in `core.py` has broad multi-channel modifications with heavy debug output; regression risk spans all structures.
- `samseg/subregions/core.py:431`
- `samseg/subregions/core.py:608`
- `samseg/subregions/core.py:736`
