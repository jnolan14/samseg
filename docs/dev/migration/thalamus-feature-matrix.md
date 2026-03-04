# Thalamus Feature Matrix

Status legend:
- `Present`: implemented and wired for use.
- `Partial`: implemented in code but not fully wired/robust.
- `Missing`: not implemented.

| ID | Capability | Base MATLAB | Updated MATLAB | Python (`thalamus`) | Python (`thalamusDTI`) |
|---|---|---|---|---|---|
| F01 | Functional entrypoint | Partial | Present | Present | Partial |
| F02 | ASEG-driven coarse alignment | Present | Present | Present | Present |
| F03 | Additional structural image support | Present (single) | Present (dynamic) | Partial (internal stack, fixed CLI) | Partial |
| F04 | Explicit structural channel count control | Missing | Present (`N_Structural`) | Missing | Missing (no public CLI wiring) |
| F05 | DTI likelihood family selection | Missing | Present | Missing | Partial (arg exists, not production-wired) |
| F06 | Reduced DTI init channel | Missing | Present | Missing | Partial |
| F07 | Joint structural + DTI objective | Missing | Present | Missing | Partial |
| F08 | Reprocess/reload controls | Missing | Present | Missing | Missing |
| F09 | Reflection controls | Missing | Present | Missing | Missing |
| F10 | Voxel-ratio weighting controls | Missing | Present | Missing | Missing |
| F11 | Reticular + largest-CC cleanup | Present | Present/variant | Present | Missing/variant |
| F12 | Whole-thalamus totals | Present | Present | Present | Partial |
| F13 | Rich diagnostics / parameter snapshots | Partial | Present | Missing | Partial (debug-heavy, not polished) |
| F14 | Coarse segmentation fallback ecosystem | Limited | Present (ASEG/SynthSeg/GIF) | Limited (`aseg.mgz`) | Limited |
| F15 | Channel-aware hyperparameter estimation | Missing | Present | Partial | Partial |

## Paper intent tags
- 2018: structural Bayesian atlas segmentation baseline.
- 2019: sequence-adaptive joint structural+diffusion Bayesian inference.
- 2023: improved atlas + diffusion-driven accuracy gains.
