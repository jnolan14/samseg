# Compact Table 3: dti_integration Symbols -> Dev

Simplifications: source/equivalent shown by filename only, no line ranges.

| dti_integration symbol | Kind | Source file | Change type | Dev nearest equivalent | Dev file(s) | Match | Confidence | Notes |
|---|---|---|---|---|---|---|---|---|
| `core.py` | `file` | `core.py` | `MODIFIED_FILE` | `core.py (same file)` | `core.py` | `Exact` | `High` | File exists in both branches; dti_integration modifies internals. |
| `MeshModel` | `class` | `core.py` | `MODIFIED_SYMBOL` | `MeshModel` | `core.py` | `Exact` | `High` | Class retained and modified. |
| `MeshModel.fit_mesh_to_seg` | `method` | `core.py` | `MODIFIED_SYMBOL` | `MeshModel.fit_mesh_to_seg` | `core.py` | `Exact` | `High` | Method modified in-place from same base method. |
| `MeshModel.prepare_for_image_fitting` | `method` | `core.py` | `MODIFIED_SYMBOL` | `MeshModel.prepare_for_image_fitting` | `core.py` | `Exact` | `High` | Method modified in-place from same base method. |
| `MeshModel.fit_mesh_to_image` | `method` | `core.py` | `MODIFIED_SYMBOL` | `MeshModel.fit_mesh_to_image` | `core.py` | `Exact` | `High` | Method modified in-place from same base method. |
| `MeshModel.extract_segmentation` | `method` | `core.py` | `MODIFIED_SYMBOL` | `MeshModel.extract_segmentation` | `core.py` | `Exact` | `High` | Method modified in-place from same base method. |
| `process.py` | `file` | `process.py` | `MODIFIED_FILE` | `process.py (same file)` | `process.py` | `Exact` | `High` | File exists in both branches with lookup change. |
| `model_lookup (module dictionary)` | `code_block` | `process.py` | `MODIFIED_SYMBOL` | `model_lookup dictionary without thalamusDTI` | `process.py` | `Partial` | `High` | dti_integration adds `thalamusDTI` entry. |
| `thalamusDTI.py` | `file` | `thalamusDTI.py` | `ADDED_FILE` | `Closest: thalamus.py + core.py` | `thalamus.py, core.py` | `Partial` | `Medium` | No file-level equivalent in dev. |
| `ThalamicNucleiDTI` | `class` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei` | `thalamus.py` | `Partial` | `High` | Class-level nearest counterpart is structural thalamus model. |
| `ThalamicNucleiDTI.__init__` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.__init__` | `thalamus.py` | `Partial` | `High` | Same constructor role with extra DTI parameters in integration branch. |
| `ThalamicNucleiDTI.parse_dti_dir` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | No DTI directory parser in dev branch. |
| `ThalamicNucleiDTI.parse_grouping_json` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `Closest: ThalamicNuclei.get_label_groups` | `thalamus.py` | `Partial` | `Low` | No JSON-driven grouping parser in dev. |
| `ThalamicNucleiDTI.initialize` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `MeshModel.initialize + ThalamicNuclei.preprocess_images` | `core.py, thalamus.py` | `Partial` | `Medium` | Equivalent lifecycle split across two methods in dev. |
| `ThalamicNucleiDTI.preprocess_image` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.preprocess_images` | `thalamus.py` | `Partial` | `High` | Closest direct preprocessing counterpart. |
| `ThalamicNucleiDTI.get_cheating_label_groups` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_cheating_label_groups` | `thalamus.py` | `Partial` | `High` | Same stage with broader logic in DTI class. |
| `ThalamicNucleiDTI.label_group_names_to_indices` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `MeshModel.label_group_names_to_indices` | `core.py` | `Exact` | `High` | Direct method-level equivalent in base class. |
| `ThalamicNucleiDTI.get_cheating_gaussians` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_cheating_gaussians` | `thalamus.py` | `Exact` | `High` | Direct method-level equivalent. |
| `ThalamicNucleiDTI.get_label_groups` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_label_groups` | `thalamus.py` | `Exact` | `High` | Direct method-level equivalent. |
| `ThalamicNucleiDTI.synthseg_kmeans` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | No SynthSeg KMeans helper in dev. |
| `ThalamicNucleiDTI.get_gaussian_hyps` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_gaussian_hyps` | `thalamus.py` | `Partial` | `High` | Same purpose with extra grouping hooks in DTI class. |
| `ThalamicNucleiDTI.get_second_label_groups` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_second_label_groups` | `thalamus.py` | `Exact` | `High` | Direct method-level equivalent. |
| `ThalamicNucleiDTI.get_second_gaussian_hyps` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.get_second_gaussian_hyps` | `thalamus.py` | `Partial` | `High` | Same purpose with extended hook logic in DTI class. |
| `ThalamicNucleiDTI.postprocess_segmentation` | `method` | `thalamusDTI.py` | `ADDED_SYMBOL` | `ThalamicNuclei.postprocess_segmentation` | `thalamus.py` | `Partial` | `High` | Same output stage, reduced behavior in prototype. |
| `utils.py` | `file` | `utils.py` | `MODIFIED_FILE` | `utils.py (same file)` | `utils.py` | `Exact` | `High` | File exists in both branches; added helper functions in integration branch. |
| `read_compression_lookup_table` | `function` | `utils.py` | `MODIFIED_SYMBOL` | `read_compression_lookup_table` | `utils.py` | `Exact` | `High` | Same function role with backend call change. |
| `find_hyps_idx` | `function` | `utils.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | Added function has no equivalent in dev. |
| `import_hyps_hack` | `function` | `utils.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | Added function has no equivalent in dev. |
| `test_hack` | `function` | `utils.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | Added debug helper has no equivalent in dev. |
| `vdc_hack` | `function` | `utils.py` | `ADDED_SYMBOL` | `None` | `N/A` | `None` | `None` | Added helper has no equivalent in dev. |
| `bimodal_thal_hack` | `function` | `utils.py` | `ADDED_SYMBOL` | `Closest: ThalamicNuclei.get_second_gaussian_hyps` | `thalamus.py` | `Partial` | `Low` | Conceptually related to bimodal thalamus adjustments. |
