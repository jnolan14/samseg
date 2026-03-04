# Table 2 Explanations By Subsystem

Scope: explanations for all non-`Exact` rows in
`02-dti-integration-symbols-to-matlab-compact.md`.

## Core architecture and orchestration

| Row key | Purpose | Current coverage vs MATLAB | Missing behavior | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `core.py` | Shared mesh fitting engine for all subregions. | Partial overlap with updated MATLAB main loop. | MATLAB-specific modality routing and objective branching are not fully encoded. | Shared file changes can cause cross-structure regressions. | Isolate thalamus-DTI logic behind explicit strategy hooks in `MeshModel`. | `P0` | `High` |
| `MeshModel` | Encapsulates lifecycle for alignment, fitting, extraction. | Partial semantic match to MATLAB entrypoint orchestration. | MATLAB uses more helper-level explicit stages and outputs. | Hard to validate parity when responsibilities are merged implicitly. | Add stage-level interfaces and artifact contracts in class methods. | `P1` | `High` |
| `MeshModel.fit_mesh_to_seg` | Segmentation-guided mesh fitting stage. | Partial match to ASEG cheating phases. | Missing configurable variants present in MATLAB modes. | Early-stage mesh pose influences entire run. | Add mode-specific settings object (structural, reduced-DTI-init, joint). | `P1` | `High` |
| `MeshModel.prepare_for_image_fitting` | Prepares cropped/masked data for objective fitting. | Partial match to MATLAB preprocessing stages. | Multi-channel setup is less explicit than MATLAB parameterization. | Channel setup errors propagate silently into EM/deformation. | Return structured channel metadata and validated dimensions. | `P0` | `Medium` |
| `MeshModel.fit_mesh_to_image` | Main EM and deformation optimization loop. | Partial match to MATLAB joint optimization loop. | Full tensor/WMM objective variants are not parity-complete. | This is the core parity target for updated method. | Split objective assembly from optimizer loop; add diffusion terms explicitly. | `P0` | `High` |
| `MeshModel.extract_segmentation` | Produces posterior-derived labels and outputs. | Partial stage correspondence. | MATLAB grouped output and posterior-space operations are richer. | Output differences hinder parity verification. | Add grouped/posterior export modes and deterministic naming. | `P1` | `Medium` |
| `process.py` | High-level run orchestration and model invocation. | Partial equivalent to MATLAB top-level flow. | MATLAB mode switches and option matrix are broader. | Orchestration gaps block reproducible mode parity. | Introduce explicit run modes mapped to MATLAB likelihood families. | `P0` | `High` |
| `model_lookup (module dictionary)` | Dispatches model key to class. | Partial equivalent to MATLAB parser mode routing. | No full validation of DTI-specific option bundle before dispatch. | Invalid dispatch states fail late and opaquely. | Add pre-dispatch validation for thalamusDTI required args/options. | `P0` | `Medium` |
| `thalamusDTI.py` | Prototype DTI-capable thalamus model surface. | Partial file-level equivalent to MATLAB entrypoint+helpers. | Prototype still contains hard-coded paths, debug scaffolding, and incomplete parity. | Main integration target for updated method. | Promote to production-ready module with configurable inputs and tests. | `P0` | `High` |
| `ThalamicNucleiDTI` | Class-level encapsulation of joint model behavior. | Partial match to MATLAB procedural core. | Some MATLAB helper behaviors are unimplemented or condensed. | Without explicit parity ledger, behavior drift is likely. | Define method-by-method parity checklist in class docstring/tests. | `P1` | `High` |
| `ThalamicNucleiDTI.__init__` | Collects runtime options and defaults. | Partial match to MATLAB `inputParser` role. | Missing strict schema/validation for modality-dependent settings. | Bad config states surface deep in runtime. | Add typed config validation and required/optional option groups. | `P0` | `High` |
| `ThalamicNucleiDTI.initialize` | Runtime setup and dependency loading. | Partial equivalence to MATLAB initialization setup. | Hard-coded external JSON and testing hooks remain. | Non-portable initialization blocks collaboration and CI. | Move all external paths to args or atlas-relative defaults; remove debug hooks. | `P0` | `High` |

## Data ingress and preprocessing

| Row key | Purpose | Current coverage vs MATLAB | Missing behavior | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `ThalamicNucleiDTI.parse_dti_dir` | Parses and validates DTI input directory layout. | Partial nearest to MATLAB DTI setup calls. | Ingestion contract is lighter than MATLAB expectations. | Dataset variability will break brittle parsers. | Define explicit required files and validation errors for each likelihood mode. | `P0` | `Medium` |
| `ThalamicNucleiDTI.preprocess_image` | Loads/resamples structural and DTI channels. | Partial overlap with MATLAB preprocessing path. | Full channel matrix assembly and reduced-channel initialization are incomplete. | Preprocessing parity is prerequisite for objective parity. | Implement channel-stack builder with modality tags and affine checks. | `P0` | `High` |
| `ThalamicNucleiDTI.synthseg_kmeans` | Prototype helper for SynthSeg-based seed derivation. | Partial conceptual relation to MATLAB SynthSeg merge logic. | Method is prototype-specific and not mapped to stable runtime contract. | Useful fallback path is currently fragile. | Replace with production `merge_synthseg_with_atlas` utility and tests. | `P1` | `Low` |
| `ThalamicNucleiDTI.postprocess_segmentation` | Writes final labels and summaries. | Partial match to MATLAB output stage. | Rich probability/group outputs are missing. | Output parity needed for QC and result comparison. | Expand postprocess API with optional grouped/posterior artifacts. | `P1` | `High` |

## Grouping and hyperparameter modeling

| Row key | Purpose | Current coverage vs MATLAB | Missing behavior | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `ThalamicNucleiDTI.parse_grouping_json` | Reads grouping definitions for means/hyperparameters. | Partial match to MATLAB grouping helpers. | JSON schema and transformation richness are narrower. | Group definitions control priors and class reductions. | Formalize JSON schema and include transform matrix semantics. | `P0` | `Medium` |
| `ThalamicNucleiDTI.get_cheating_label_groups` | Builds initial grouped labels for cheating stage. | Partial mapping to MATLAB group setup helpers. | Coverage of edge-case groups differs from MATLAB. | Weak group initialization degrades mesh fitting startup. | Consolidate group construction through shared `utils.py` schema. | `P1` | `Medium` |
| `ThalamicNucleiDTI.label_group_names_to_indices` | Converts group names to index lists. | Partial mapping to MATLAB name-index logic. | Error handling/validation for unknown names is limited. | Silent mis-indexing can corrupt priors. | Add strict validation and deterministic ordering checks. | `P1` | `Medium` |
| `ThalamicNucleiDTI.get_cheating_gaussians` | Produces initial Gaussian stats for cheating fit. | Partial inferred mapping to MATLAB behavior. | No one-to-one helper parity and limited configurability. | Initialization quality influences downstream convergence. | Add explicit config for source statistics and fallback strategy. | `P1` | `Low` |
| `ThalamicNucleiDTI.get_label_groups` | Primary group definitions for main fit. | Partial match to MATLAB grouping stages. | Missing some MATLAB transform/group metadata. | Group mismatch changes posterior aggregation. | Implement full grouped metadata object and parity tests. | `P0` | `Medium` |
| `ThalamicNucleiDTI.get_gaussian_hyps` | Computes group/class Gaussian hyperparameters. | Partial match to MATLAB hyperparameter fitting. | Full multi-channel parity and robust fallback behavior are incomplete. | Core statistical parity requirement. | Expand to channel-aware estimation and include golden comparisons. | `P0` | `High` |
| `ThalamicNucleiDTI.get_second_label_groups` | Secondary grouping stage for two-component model. | Partial match to MATLAB reduced grouping logic. | Selection logic and edge cases differ. | Affects bimodal handling and robustness. | Align second-stage grouping rules with MATLAB lookup tables. | `P1` | `Medium` |
| `ThalamicNucleiDTI.get_second_gaussian_hyps` | Second-component hyperparameter setup. | Partial stage correspondence. | Hard-coded behavior still present in prototype path. | Biases secondary component estimation. | Replace hard-coded branches with modality-aware config logic. | `P1` | `Medium` |
| `find_hyps_idx` | Utility to locate hyperparameter entries by label/group. | Partial nearest to MATLAB grouping loops. | Contract and error semantics are under-specified. | Utility bugs propagate to multiple hyperparameter hooks. | Define typed inputs/outputs and add unit tests for missing keys. | `P1` | `Low` |
| `bimodal_thal_hack` | Prototype override for bimodal thalamus priors. | Partial conceptual mapping to MATLAB bimodal behavior. | Contains fragile assumptions and depends on hack utilities. | High regression risk and low reproducibility. | Replace with explicit bimodal policy module and remove ad hoc hacks. | `P1` | `Low` |

## Utilities and branch-only hooks

| Row key | Purpose | Current coverage vs MATLAB | Missing behavior | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `utils.py` | Shared utility surface for grouping and helper hooks. | Partial equivalent to distributed MATLAB helper layer. | Mixed production utilities and prototype hacks in one file. | Entangles stable code with experimental behavior. | Split into `utils_core.py` and `utils_experimental.py` with explicit imports. | `P1` | `Medium` |
| `import_hyps_hack` | Dynamic import helper for prototype hyperparameter injection. | No MATLAB equivalent. | Non-deterministic behavior surface and no stable contract. | Hard to reason about reproducibility and safety. | Remove or replace with explicit config-file based override path. | `P2` | `None` |
| `test_hack` | Branch-local test/debug helper. | No MATLAB equivalent. | Runtime path pollution for production module imports. | Increases accidental behavior drift. | Remove from production path; keep only in test fixtures. | `P2` | `None` |
| `vdc_hack` | Branch-local helper without stable algorithm role. | No MATLAB equivalent. | Undefined parity objective and no guardrails. | Raises maintenance cost with unclear benefit. | Document intent or remove pending explicit use case. | `P2` | `None` |
