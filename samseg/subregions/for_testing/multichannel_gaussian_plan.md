# Structural Subfields++ GMM Integration Plan

Date: 2026-08-06

Status: superseded historical implementation plan. Do not use the settled
design or staged implementation details below as current requirements.

The 2026-08-13 investigation-closure checkpoint restored `samseg/GMM.py` to
its established implementation and retired the temporary parallel Gaussian
helper. Current evidence and design context are recorded in:

- `gaussian_gmm_audit.md` for provenance, callers, mathematics, and the
  original architectural audit;
- `gmm_subfields_replacement_review.md` for the narrower no-GMM-extension
  structural replacement boundary; and
- `subfields_plus_architecture.md` for the current successor context and
  unresolved questions.

In particular, the historical body below overstates settled low-mass,
convergence, covariance-mode, jitter, log-domain, and public NIW-mapping
decisions. It is retained unchanged as provenance for how the investigation
was staged.

Implementation base: `HT-subregions-integration` at
`2412fbb6ad5c16e142b551072fa7973ea6a03ec3`, based on `origin/rectify` at
`53936b2625356bda0de04445c9032f9df54a3bf0`.

## Purpose

Deliver a structurally clean, configurable, multichannel structural
subfields++ path without changing the original thalamus, hippocampus,
brainstem, or other legacy subregions implementations.

The first release target is cross-sectional structural thalamus. The current
`ThalamicNucleiDTI` class is treated as a transitional implementation name,
not as evidence that diffusion fitting belongs in this branch endpoint.

The evidence and rationale for this plan are recorded in
`gaussian_gmm_audit.md`. In particular:

- `samseg.GMM.GMM` is the established owner of multicontrast Gaussian-mixture
  mathematics in SAMSEG;
- the original subregions formulas are a direct-port legacy implementation,
  not a required backend for subfields++;
- the standalone `samseg/subregions/gaussian.py` helper is a temporary
  mathematical oracle and is not the intended permanent owner;
- full covariance, multiple configured components, and fitted mixture weights
  are useful general capabilities rather than incompatibilities; and
- anatomical grouping, hyperparameter initialization, scheduling, and narrow
  region-specific operations remain subfields policy outside the GMM engine.

## Compatibility Boundary

- Do not refactor or change the original `ThalamicNuclei`, hippocampal,
  brainstem, or other established model classes.
- Do not route the existing `segment_subregions` command through subfields++.
- Do not change existing standard, lesion, tumour, or longitudinal SAMSEG GMM
  defaults or array contracts.
- Do not change the existing main-GMM prior-cost evaluator in place.
- Work only from `origin/rectify` during this branch. Do not merge or rebase
  `upstream/dev` yet.
- Do not implement the final DTI structural/diffusion component-incidence
  model in this branch.
- Do not implement hippocampal missing-channel ECM in this branch.
- Do not make longitudinal subfields++ support a release blocker.

Existing behavior is preserved where it is demonstrably part of a legacy or
active interface. Prototype behavior is not preserved merely because it is
present in `thalamusDTI.py` or the local test harness.

## Settled Design

### Statistical ownership

- `GMM` becomes the single owner of Gaussian likelihoods, responsibilities,
  sufficient-statistic updates, covariance constraints, component weights,
  and the parameter arrays handed to GEMS.
- Subfields++ uses a narrow opt-in GMM seam. Existing GMM callers retain their
  current defaults and behavior.
- `samseg/subregions/gaussian.py` remains only long enough to compare the new
  GMM behavior against independently tested formulas. Its focused tests are
  migrated to GMM and the helper is removed before branch completion.
- Do not create another permanent Gaussian kernel module in this branch.

### Covariance modes

- Structural subfields++ defaults to full covariance.
- Structural subfields++ also supports explicit diagonal covariance.
- Diagonal covariance is the diagonal projection of the same coherent update,
  not the legacy subregions `0.01`-perturbed formula.
- Existing whole-brain GMM defaults remain unchanged.
- GEMS receives full `(gaussians, channels, channels)` covariance matrices in
  both modes; diagonal mode supplies matrices with zero off-diagonal entries.
- The same GMM likelihood implementation is used during fitting and final
  posterior reconstruction.

### NIW parameterization and objective

- Use the final MATLAB/2023 structural covariance update as the subfields++
  reference.
- Map its covariance prior into existing GMM notation with
  `h = nu + number_of_channels + 2` and `H = Psi / h`.
- Structural subfields++ initially opts into `nu = 0` and `Psi = 0`, matching
  the documented non-informative structural setting.
- Add a separate opt-in coherent NIW cost only if subfields++ needs it for
  stopping or reporting.
- Do not replace or silently alter `evaluateMinLogPriorOfGMMParameters`,
  because its existing callers and convergence semantics must be assessed in
  a separate change.

### Low mass and numerical stabilization

- If a component has insufficient effective mass, retain its previous mean,
  covariance, and weight state rather than introducing NaNs or the legacy
  variance-100 fallback.
- Low-mass retention is an explicit opt-in policy for subfields++; existing
  GMM defaults remain unchanged.
- Preserve valid mixture-weight normalization within each class. If an entire
  class has insufficient mass, retain its previous within-class weights.
- Construct symmetric covariance estimates directly.
- Attempt Cholesky factorization on covariance use.
- Add adaptive diagonal jitter only after a failed factorization, and only for
  the failing matrix.
- Do not run an eigendecomposition in the normal fitting path.
- Numerical stabilization must not act as a hidden informative covariance
  prior.

### Components and mixture weights

- Parse `sharedGMMparameters.txt` with the established
  `kvlReadSharedGMMParameters` machinery and retain each configured component
  count.
- Represent configured components and fitted mixture weights through GMM.
- Current structural thalamus configurations use one component per configured
  group, so no new clustering policy is required for the first path.
- If a supplied configuration requests multiple components, reuse or narrowly
  extend established GMM initialization rather than inventing a
  subfields-specific component initializer.
- Do not implement the final DTI many-to-many component-incidence model here.

### Group hyperparameters

- `sharedGMMparameters.txt` is the source of anatomical parameter-sharing
  groups and component counts.
- Rasterize the configured reduced atlas classes in the working image grid.
- Assign support by hard prior argmax, then apply the established erosion.
- Use a common mask of voxels finite in every supplied structural channel.
- Compute one robust median per group and channel.
- Store `meanHyper` as `(classes, channels)` and `nHyper` as `(classes,)`.
- Set each supported group's strength to
  `max(resolution_adjusted_complete_case_support, 10)`.
- The value 10 is a minimum/fallback, not an additive pseudo-count.
- If eroded support is empty, retry the same group without erosion.
- If support is still empty, use global complete-case channel medians, emit a
  clear warning, and set `nHyper` to 10.
- Channel-wise missing-data ECM is explicitly deferred; complete-case fitting
  is the documented first-release behavior.

### Existing JSON-selected operations

- Keep Jackson's existing JSON-selected dotted-callable mechanism in its
  familiar form.
- Do not formalize, rename, generalize, or replace it with a new registry or
  strategy framework in this branch.
- No flag and no JSON file selects the clean structural path.
- `--preset original-thalamus` loads a packaged JSON configuration through the
  same existing mechanism.
- `--hacks-config FILE` loads a custom configuration in the existing format.
- `--preset` and `--hacks-config` are mutually exclusive.
- The original-thalamus preset is T1-only and rejects multichannel input.
- Region-specific behavior remains in narrowly scoped operations; Gaussian
  mathematics does not move into those operations.
- Remove developer-specific hard-coded paths and repair only operations needed
  by the supported preset or custom configuration path.
- Do not create a separate hard-coded legacy regrouping pipeline for the
  preset.

### State ownership

- `model.gmm` is the canonical mutable Gaussian state for subfields++.
- Do not mirror mutable means, covariance arrays, component counts, or weights
  onto `MeshModel`.
- Add read-only compatibility aliases only if a demonstrated local tool needs
  them.
- Legacy model state remains untouched because legacy classes do not use the
  new backend.

### Command-line interface

- Add a separate `segment_subregions_plus` console command.
- Initially expose a `thalamus` subcommand only.
- Keep `segment_subregions` and the established region commands unchanged.
- Use direct cross-sectional file arguments rather than requiring a
  FreeSurfer subject-directory layout.
- The intended interface is:

```text
segment_subregions_plus thalamus \
  --input T1.mgz [T2.mgz ...] \
  --seg aseg.mgz \
  --output output-directory \
  [--atlas atlas-directory] \
  [--gmm sharedGMMparameters.txt] \
  [--covariance-mode full|diagonal] \
  [--preset original-thalamus | --hacks-config operations.json] \
  [--temp-dir directory] [--suffix text] [--debug] [--threads count]
```

- Full covariance is the command's default.
- Structural invocation must not require DTI directories, transforms, FA
  images, or diffusion-specific options.
- Longitudinal support may be included only if it is a trivial, low-risk reuse
  after the cross-sectional path is complete. Otherwise record it as deferred.

### Documentation standard

Every new or materially modified function in this work requires a NumPy-style
docstring. In addition to the conventional sections, each docstring must
contain a section headed exactly:

```text
Subfields Useage
----------------
```

That section briefly states why the function exists and how it fits into the
SAMSEG/subfields++ execution path.

## Review Discipline

Each stage below is a separate human-in-the-loop review unit.

Before every review stop:

1. Run the stage's focused tests and syntax checks.
2. Run `git diff --check`.
3. Inspect `git status --short --branch` and the complete stage diff.
4. Write a commit message under `/tmp` with a subject no longer than 50
   characters and a body wrapped at 78 characters.
5. Give Henry the exact test and commit commands.
6. Stop. Do not begin the next stage until Henry reviews the result.

Codex does not commit, merge, rebase, push, or otherwise mutate refs as part of
a review stop.

## Reviewable Execution Plan

### Stage 1: Reset the planning record

Scope:

- Add `gaussian_gmm_audit.md` as the durable evidence record.
- Replace this previously helper-centric plan in place with the approved
  GMM-owned structural subfields++ plan.
- Record that `subregions/gaussian.py` is temporary and must be retired after
  its formulas have been transferred and independently covered in GMM tests.
- Record the branch endpoint, compatibility boundary, deferred DTI/ECM work,
  CLI boundary, JSON-operation policy, and review cadence.
- Do not edit production code or tests.

Verification:

```bash
.venv/bin/python -m pytest -q samseg/tests/test_subregions_gaussian.py
git diff --check
```

Commit message:

```text
/tmp/subfields-plus-replan-commit-msg.txt
```

Review boundary: documentation and branch intent only.

### Stage 2: Add opt-in subfields capabilities to GMM

Scope:

- Add opt-in, numerically stable log-likelihood and log-responsibility paths
  without changing existing density-valued methods.
- Add the coherent NIW update/cost mapping needed by structural subfields++.
- Add opt-in low-mass retention and failure-only Cholesky jitter.
- Implement diagonal covariance as projection of the same update.
- Preserve existing constructor behavior and all established caller defaults.
- Preserve multiple-component fitting and normalized within-class mixture
  weights.
- Compare all new GMM kernels against the temporary helper and independent
  formulas, then move the focused tests to GMM.
- Remove `samseg/subregions/gaussian.py` and helper-only tests once equivalent
  GMM coverage exists.
- Add no subregions orchestration in this stage.

Required tests:

- one- and two-channel full covariance likelihoods and updates;
- explicit diagonal projection from the common update;
- log-space responsibilities and normalization;
- exact NIW update and opt-in cost under `h = nu + d + 2`;
- zero-scale structural prior behavior;
- low-mass retention for one component and an entire class;
- failure-only jitter with no eigendecomposition in the hot path;
- multiple components and within-class mixture weights; and
- regression checks showing current GMM defaults and density APIs are
  unchanged.

Commit message:

```text
/tmp/gmm-opt-in-subfields-commit-msg.txt
```

Review boundary: GMM statistical behavior and focused tests only.

### Stage 3: Integrate the GMM backend into subfields++

Scope:

- Add a narrow backend seam selected only by the transitional
  `ThalamicNucleiDTI`/subfields++ path.
- Construct GMM from configured reduced classes and component counts.
- Make `model.gmm` the sole mutable Gaussian state.
- Default the new path to full covariance and support explicit diagonal mode.
- Use the same GMM likelihood for image EM and final posterior extraction.
- Pass GMM means, full covariance matrices, weights, and actual component
  counts to GEMS.
- Retain existing mesh/deformation orchestration outside GMM.
- Do not change legacy `MeshModel` behavior used by established models.
- Do not add grouping/hyperparameter policy changes in this stage.

Required tests:

- two-channel full and diagonal backend construction;
- one complete subfields++ image-EM update;
- canonical `model.gmm` state without mutable duplicates;
- fitting/extraction likelihood agreement;
- non-zero off-diagonal covariance reaching the GEMS boundary;
- configured counts and fitted weights reaching GEMS; and
- unchanged construction and core behavior for a representative legacy model.

Commit message:

```text
/tmp/subfields-plus-gmm-integration-commit-msg.txt
```

Review boundary: subfields++ statistical integration only.

### Stage 4: Implement group initialization and operations policy

Scope:

- Replace ad hoc parsing with `kvlReadSharedGMMParameters`, preserving
  component counts and search strings.
- Rasterize configured reduced-class priors, select hard argmax support, and
  erode it.
- Compute complete-case channel medians and the resolution-adjusted `nHyper`
  floor.
- Implement the un-eroded and global-median fallback sequence with warnings.
- Keep the clean path independent of JSON-selected operations.
- Remove the developer-specific hard-coded JSON path.
- Package a corrected T1-only `original-thalamus` JSON preset using the same
  existing dotted-callable mechanism.
- Support custom existing-format JSON through an explicit path.
- Repair only the narrow existing operations required by the supported preset
  and add no new registry abstraction.

Required tests:

- parsing group names, search strings, and component counts;
- deterministic hard-argmax and erosion support;
- two-channel group medians and `(classes, channels)` `meanHyper`;
- `nHyper = max(adjusted_support, 10)` rather than `10 + support`;
- complete-case masking;
- empty eroded support, empty group support, warning, and fallback behavior;
- clean no-operations path;
- packaged preset loading through the existing mechanism;
- multichannel rejection for `original-thalamus`; and
- custom JSON-selected operation execution.

Commit message:

```text
/tmp/subfields-plus-policy-commit-msg.txt
```

Review boundary: group initialization and existing operation policy only.

### Stage 5: Add the separate structural CLI

Scope:

- Add `segment_subregions_plus` without modifying `segment_subregions`.
- Add the `thalamus` subcommand and direct structural input/segmentation/output
  arguments.
- Wire atlas, GMM parameter file, covariance mode, preset/custom operations,
  temporary directory, suffix, debug, and thread options.
- Enforce preset/config exclusivity and T1-only preset validation before the
  run starts.
- Remove structural dependence on DTI-only constructor requirements.
- Keep cross-sectional execution as the required complete path.

Required tests:

- parser defaults and full-covariance default;
- repeated structural inputs;
- required argument and mutual-exclusion validation;
- clean, preset, and custom-config routing;
- no DTI argument requirement;
- separate console-entry routing; and
- proof that the legacy CLI still resolves to its original implementation.

Commit message:

```text
/tmp/segment-subregions-plus-cli-commit-msg.txt
```

Review boundary: user-facing structural invocation only.

### Stage 6: Smoke validation and branch closure

Scope:

- Run all focused automated tests introduced in Stages 2-5.
- Run a clean two-channel full-covariance structural thalamus smoke test.
- Run an explicit two-channel diagonal smoke test.
- Run a one-channel `original-thalamus` preset smoke test.
- Run a custom JSON-selected operation smoke test.
- Confirm that fitted off-diagonal covariance reaches GEMS and that final
  extraction uses the fitted model.
- Run a representative unchanged legacy thalamus smoke test.
- Confirm that no developer-specific path, duplicate mutable Gaussian state,
  permanent subregions Gaussian helper, or accidental DTI requirement remains.
- Inspect longitudinal reuse only after cross-sectional acceptance; include it
  only if the change is trivial and low risk, otherwise document deferral.
- Refresh the final diff and branch ledger against `origin/rectify` without
  integrating `upstream/dev`.

Commit message:

```text
/tmp/subfields-plus-branch-complete-commit-msg.txt
```

Review boundary: acceptance evidence and branch hygiene. Stop before any
upstream integration or pull-request preparation.

## Automated Test Strategy

Use the repository venv explicitly:

```bash
.venv/bin/python -m pytest -q <focused-test-modules>
```

Keep new tests in focused modules that import only the functionality under
test. The unrelated TensorFlow collection failure in `test_samseg.py` must not
be hidden or reported as a subfields failure. Where broader collection is
useful, exclude the lesion import path rather than changing TensorFlow
dependencies in this branch.

Long FreeSurfer data runs are manual acceptance checks, not mandatory unit
tests. Record their exact inputs, options, environment, and output location so
the result is reproducible.

## Branch Acceptance Criteria

- The completed audit and this plan are committed as durable records.
- Structural subfields++ uses GMM as the sole Gaussian-mixture mathematics
  owner.
- The temporary subregions Gaussian helper has been removed after equivalent
  GMM coverage is established.
- Full covariance is the structural subfields++ default; explicit diagonal
  mode uses the same coherent update with diagonal projection.
- Existing GMM callers and legacy subregions models retain their defaults and
  behavior.
- Fitting and posterior extraction use the same GMM likelihood.
- GEMS receives full covariance matrices, configured component counts, and
  fitted mixture weights.
- Dynamic groups and component counts come from `sharedGMMparameters.txt`.
- Multichannel hypermeans and `nHyper` follow the documented complete-case,
  hard-support, erosion, floor, and fallback rules.
- The existing JSON-selected operation mechanism remains recognizable and is
  selectable only through explicit clean/preset/custom policy.
- `segment_subregions_plus thalamus` is a functional cross-sectional
  structural command with no DTI requirement.
- Focused tests and required smoke runs pass.
- Final DTI incidence, missing-channel ECM, nontrivial longitudinal work,
  `upstream/dev` integration, and broad prior-evaluator changes remain
  explicitly deferred.

## Deferred Work

- Generalized many-to-many class/component incidence and class-specific
  weights for the final structural/diffusion model.
- Diffusion likelihoods, WMM/Wishart behavior, DTI initialization, and joint
  structural/diffusion scheduling.
- Hippocampal channel-wise missing-data ECM.
- Any nontrivial longitudinal subfields++ implementation.
- A global correction or migration of the current main-GMM prior-cost
  evaluator.
- Broad redesign of Jackson's JSON-selected operations.
- Merge, rebase, or compatibility work against `upstream/dev`.
