# Table 3 Explanations By Subsystem

Scope: explanations for all non-`Exact` rows in
`03-dti-integration-symbols-to-dev-compact.md`.

## Dispatch and model surface deltas

| Row key | Purpose | Current coverage vs `dev` | Missing behavior in merge target | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `model_lookup (module dictionary)` | Routes CLI model key to implementation class. | Partial: `dti_integration` adds `thalamusDTI` route. | `dev` path lacks guarded option checks for DTI model selection. | Unvalidated route introduces runtime constructor failures. | Add model-specific option validation before instantiation in `process.py`. | `P0` | `High` |
| `thalamusDTI.py` | New DTI-oriented thalamus implementation file. | Partial: no direct file equivalent in `dev`. | Merge target lacks file-level integration plan and tests. | Largest merge payload and highest drift risk. | Land file behind feature flag and incremental test gates. | `P0` | `Medium` |
| `ThalamicNucleiDTI` | Class wrapper for DTI workflow. | Partial nearest to `ThalamicNuclei`. | `dev` does not have a DTI-specific class contract. | Missing contract blocks safe extension points. | Define minimal public interface aligned with `ThalamicNuclei` lifecycle. | `P0` | `High` |
| `ThalamicNucleiDTI.__init__` | Constructor for DTI options and defaults. | Partial nearest to `ThalamicNuclei.__init__`. | Required DTI args/options are not represented in `dev` CLI config flow. | Constructor mismatch causes immediate runtime failure. | Add typed config object and CLI plumbing for DTI args. | `P0` | `High` |

## DTI ingress and preprocessing additions

| Row key | Purpose | Current coverage vs `dev` | Missing behavior in merge target | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `ThalamicNucleiDTI.parse_dti_dir` | Validates and resolves DTI input files. | `None` in `dev`. | No DTI directory ingestion path exists. | Blocks any DTI-enabled execution. | Add parser utility with strict schema and clear errors. | `P0` | `None` |
| `ThalamicNucleiDTI.parse_grouping_json` | Reads JSON grouping definitions for DTI-aware stats. | Partial nearest to `get_label_groups`. | `dev` lacks JSON-driven grouping contract. | Cannot reproduce branch grouping behavior deterministically. | Introduce versioned grouping schema and loader in `utils.py`. | `P1` | `Low` |
| `ThalamicNucleiDTI.initialize` | Performs DTI runtime setup and preparation. | Partial split across `MeshModel.initialize` and `preprocess_images`. | Integrated DTI initialization lifecycle is absent. | Merge requires deterministic init order and resource setup. | Add `initialize()` override with explicit stage sequence and checks. | `P0` | `Medium` |
| `ThalamicNucleiDTI.preprocess_image` | Loads/resamples structural and DTI channels. | Partial nearest to structural `preprocess_images`. | Multi-modal channel stack logic absent in `dev`. | Central blocker for joint likelihood work. | Implement multi-channel preprocessing contract and tests. | `P0` | `High` |
| `ThalamicNucleiDTI.get_cheating_label_groups` | DTI-aware grouped labels for cheating fit. | Partial nearest to structural method. | Additional grouping behavior not carried in `dev`. | Cheating stage quality affects all later fits. | Extend structural grouping method via optional DTI policy hooks. | `P1` | `High` |
| `ThalamicNucleiDTI.synthseg_kmeans` | Prototype fallback seeding using SynthSeg labels. | `None` in `dev`. | No equivalent helper or fallback policy. | Limits robustness when ASEG quality is poor. | Replace with stable fallback utility or defer behind feature flag. | `P2` | `None` |

## Statistical and output behavior deltas

| Row key | Purpose | Current coverage vs `dev` | Missing behavior in merge target | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `ThalamicNucleiDTI.get_gaussian_hyps` | Hyperparameter estimation with DTI/grouping hooks. | Partial nearest to structural implementation. | `dev` lacks DTI-specific estimation extensions. | Hyperparameter differences can change segmentation boundaries. | Add optional DTI-aware branch in shared hyperparameter helper. | `P1` | `High` |
| `ThalamicNucleiDTI.get_second_gaussian_hyps` | Second-component hyperparameter logic. | Partial nearest to structural method. | Additional branch heuristics are not present in `dev`. | Impacts bimodal thalamus handling and convergence. | Refactor to shared second-component policy with modality switches. | `P1` | `High` |
| `ThalamicNucleiDTI.postprocess_segmentation` | Writes final artifacts and summaries. | Partial nearest to structural postprocess method. | DTI branch output options are not represented in `dev`. | Output incompatibility blocks regression comparison. | Add optional DTI artifacts while preserving structural defaults. | `P1` | `High` |
| `bimodal_thal_hack` | Ad hoc override for bimodal prior behavior. | Partial conceptual equivalent only. | No structured counterpart in `dev`. | Hacky behavior is hard to merge safely. | Convert to explicit, testable policy or remove before merge. | `P1` | `Low` |

## Branch-local utility additions

| Row key | Purpose | Current coverage vs `dev` | Missing behavior in merge target | Why it matters | Recommended action | Priority | Confidence |
|---|---|---|---|---|---|---|---|
| `find_hyps_idx` | Hyperparameter index helper. | `None` in `dev`. | Utility contract absent. | Needed by DTI-specific hyperparameter pathways. | Add tested helper with strict input validation. | `P1` | `None` |
| `import_hyps_hack` | Dynamic import override helper. | `None` in `dev`. | No sanctioned equivalent; behavior is experimental. | Unsafe merge target without deterministic config path. | Do not merge as-is; replace with config-driven import mechanism. | `P2` | `None` |
| `test_hack` | Debug/test helper function. | `None` in `dev`. | No production equivalent required. | Not appropriate for production runtime surface. | Keep out of merge target; move to test-only fixtures if needed. | `P2` | `None` |
| `vdc_hack` | Experimental helper with unclear stable role. | `None` in `dev`. | No merge-ready contract. | Adds maintenance burden and ambiguity. | Defer until specific algorithmic requirement is documented. | `P2` | `None` |
