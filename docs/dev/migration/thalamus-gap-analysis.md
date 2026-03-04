# Thalamus Migration Gap Analysis

## Priority legend
- `P0`: merge-blocking or parity-critical
- `P1`: major robustness/quality gap
- `P2`: tooling/completeness gap

## P0

### GAP-P0-01: `thalamusDTI` interface is not executable from current CLI
- `thalamusDTI` is discoverable from `model_lookup` but required constructor args are not provided by CLI parameter assembly.
- Evidence:
- `samseg/subregions/process.py:16`
- `samseg/cli/segment_subregions.py:90`
- `samseg/subregions/thalamusDTI.py:22`

### GAP-P0-02: Hard-coded external JSON path in runtime
- Non-portable dependency on `/autofs/.../means_groupings.json`.
- Evidence: `samseg/subregions/thalamusDTI.py:324`

### GAP-P0-03: Dynamic structural channel control not exposed
- Updated MATLAB has `N_Structural`; Python CLI still hard-codes `norm.mgz`.
- Evidence:
- `samseg/cli/segment_subregions.py:65`
- `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m` (`N_Structural` parser)

### GAP-P0-04: Joint structural+DTI objective parity incomplete
- Updated MATLAB has explicit joint calculator flow.
- Python DTI class has partial scaffolding but not parity-grade end-to-end flow.
- Evidence:
- `samseg/subregions/thalamusDTI.py`
- Updated MATLAB joint calculator: lines ~3199 onward.

## P1

### GAP-P1-01: Utility hook bug
- `find_hyps_idx()` requires `label_groupings`, but call sites omit it.
- Evidence:
- `samseg/subregions/utils.py:70`
- `samseg/subregions/utils.py:114`

### GAP-P1-02: Debug instrumentation in production paths
- Extensive `print()` and `breakpoint()` usage in DTI and shared core paths.
- Evidence:
- `samseg/subregions/thalamusDTI.py:516`
- `samseg/subregions/core.py:608`

### GAP-P1-03: Shared-core refactor risk
- Multi-channel changes in `core.py` affect all subregions and require regression coverage.
- Evidence: `samseg/subregions/core.py`

## P2

### GAP-P2-01: Missing runtime controls from updated MATLAB
- No Python analog for `reprocessPosteriors`, `switch_forceReload`, reflection/voxel-ratio controls.

### GAP-P2-02: Output/diagnostic parity gap
- Updated MATLAB supports broad diagnostic outputs and parameter snapshots; Python does not provide equivalent polished controls.

## Recommended implementation order
1. P0-01
2. P0-02
3. P0-03
4. P0-04
5. P1-01
6. P1-02
7. P1-03
8. P2-01
9. P2-02
