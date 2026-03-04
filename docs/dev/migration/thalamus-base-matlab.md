# Base MATLAB Thalamus (`SegmentThalamicNuclei.m`)

## Entrypoint
- Script/function hybrid in this checkout; documented function signature is present in comments.
- Source: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m`

## Key Behaviors
- Structural pipeline centered on `norm.mgz` with optional single `additionalVol` replacement path.
- Additional-volume registration supports `BBregisterMode` values `none`, `t1`, `t2`.
- Optional additional-volume bias-field correction (`doBFcorrection`).
- Initial atlas alignment to thalamus/ventral-DC mask from ASEG.
- Two-stage fitting: segmentation-guided mesh fit, then intensity-based fit.
- Postprocessing removes reticular labels and keeps largest connected components.
- Outputs include segmentation in cropped and FS voxel spaces + volume text file.

## Evidence Pointers
- Input contracts and arg checks: lines ~9-27, 139-176.
- Additional-volume registration path: lines ~230-244.
- Cheating mesh schedule: line ~535.
- Intensity image selection / BF correction: lines ~724-732.
- Reticular and CC postprocessing: lines ~1910 onward.
- Whole-thalamus totals: lines ~1891-1892.

## Migration Notes
- Good baseline for structural behavior and cleanup conventions.
- Not a dynamic multi-channel structural+DTI framework.
