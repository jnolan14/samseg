# Gap Action Summary

This checklist consolidates the highest-value actions from:
- `11-table1-explanations-by-subsystem.md`
- `12-table2-explanations-by-subsystem.md`
- `13-table3-explanations-by-subsystem.md`

## P0

- [ ] `GAP-C01` Wire DTI model configuration end-to-end.
  - Source keys: `model_lookup (module dictionary)`, `ThalamicNucleiDTI.__init__`, `ThalamicNucleiDTI`.
  - Target files: `process.py`, `segment_subregions.py`, `thalamusDTI.py`.
  - Acceptance: `thalamusDTI` can be selected from CLI with validated required args and no constructor/runtime option errors.

- [ ] `GAP-C02` Implement robust DTI ingest and resampling contract.
  - Source keys: `TS_fnc_readDTIintoGEMS`, `TS_fnc_resampleDTIfiles`, `ThalamicNucleiDTI.parse_dti_dir`, `ThalamicNucleiDTI.preprocess_image`.
  - Target files: `thalamusDTI.py`, `core.py`.
  - Acceptance: validated channel stack (structural + DTI) is produced with affine checks and deterministic resampling outputs.

- [ ] `GAP-C03` Complete joint objective assembly for deformation.
  - Source keys: `TS_fnc_constructMeshDeformationCalculator`, `TS_fnc_evaluateTensorLikelihood`, `MeshModel.fit_mesh_to_image`.
  - Target files: `core.py`, `thalamusDTI.py`.
  - Acceptance: objective includes explicit structural and diffusion terms with switchable likelihood mode.

- [ ] `GAP-C04` Formalize grouping schema for DTI path.
  - Source keys: `TS_fnc_groupSorting`, `TS_fnc_group_compression_read`, `ThalamicNucleiDTI.parse_grouping_json`, `ThalamicNucleiDTI.get_label_groups`.
  - Target files: `utils.py`, `thalamusDTI.py`.
  - Acceptance: grouping schema is versioned, validated, and shared across initialization and fit stages.

- [ ] `GAP-C05` Add EM safety for disappearing classes.
  - Source keys: `TS_fnc_checkForDisappearingClass`.
  - Target files: `core.py`.
  - Acceptance: class-collapse detection triggers controlled fallback behavior and emits deterministic diagnostics.

## P1

- [ ] `GAP-C06` Align hyperparameter fitting with multi-channel behavior.
  - Source keys: `TS_fnc_fitGaussianHyperParams`, `TS_fnc_fitGaussian_withprior`, `TS_fnc_gaussprior_NIW`, `ThalamicNucleiDTI.get_gaussian_hyps`.
  - Target files: `thalamus.py`, `thalamusDTI.py`, `core.py`.
  - Acceptance: hyperparameter estimation supports configured channel dimensions and consistent grouped priors.

- [ ] `GAP-C07` Stabilize initialization and atlas-registration parity.
  - Source keys: `TS_fnc_ASEGAtlasRegistration`, `TS_fnc_AffineImageDumpRegistration`, `TS_fnc_mergeASEGwithAtlas`, `ThalamicNucleiDTI.initialize`.
  - Target files: `core.py`, `thalamus.py`, `thalamusDTI.py`.
  - Acceptance: initialization stages are explicit, configurable, and free of hard-coded external paths.

- [ ] `GAP-C08` Add fallback coarse-label flows.
  - Source keys: `TS_fnc_mergeSYNTHSEGwithAtlas`, `ThalamicNucleiDTI.synthseg_kmeans`.
  - Target files: `thalamus.py`, `thalamusDTI.py`.
  - Acceptance: pipeline supports documented fallback order (ASEG then SynthSeg) with deterministic mapping outputs.

- [ ] `GAP-C09` Expand output parity for QA and comparison.
  - Source keys: `TS_fnc_OutputVolumeMeasurements`, `TS_fnc_outputProbVolumes`, `ThalamicNucleiDTI.postprocess_segmentation`.
  - Target files: `thalamus.py`, `thalamusDTI.py`, `core.py`.
  - Acceptance: grouped volume and optional posterior exports are available behind explicit flags.

- [ ] `GAP-C10` Remove hack coupling from group/hyperparameter utilities.
  - Source keys: `find_hyps_idx`, `bimodal_thal_hack`, `utils.py`.
  - Target files: `utils.py`, `thalamusDTI.py`.
  - Acceptance: helper contracts are typed and tested; ad hoc hooks are replaced by explicit policy modules.

## P2

- [ ] `GAP-C11` Clean debug-only code from production path.
  - Source keys: `import_hyps_hack`, `test_hack`, `vdc_hack`, `kvlClear`, `kvlSetImageBuffer`.
  - Target files: `utils.py`, `core.py`, `thalamusDTI.py`.
  - Acceptance: production runtime has no debug-only helper dependencies and uses explicit lifecycle/buffer wrappers.

- [ ] `GAP-C12` Decide on optional low-priority parity helpers.
  - Source keys: `TS_fnc_convertGIF2ASEG`, `TS_fnc_flipAseg`, `TS_fnc_erfireplacement`, `TS_fnc_fitTensorReflection`, `TS_fnc_fitTensorReflection_initial`.
  - Target files: `segment_subregions.py`, `utils.py`, `thalamusDTI.py`.
  - Acceptance: each helper is either implemented with tests or explicitly marked out of scope in docs.
