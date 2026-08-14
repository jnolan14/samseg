# Structural Subfields++ GMM Replacement Review

Date: 2026-08-11

Revision: 2026-08-13 investigation-closure checkpoint. The exploratory
`GMM.py` APIs described in the method ledger have been removed from the live
worktree and retained only in an ignored local recovery patch. This revision
supersedes the initial Stage 2 retention recommendations for log-domain APIs,
public NIW parameter mapping, and low-mass retention. The mathematical and
numerical evidence remains relevant; the prototype API is not current
production architecture.

Status: durable audit and decision record. This is not an implementation plan
for `MeshModelPlus` and does not settle the unresolved statistical policies.

## Executive determination

The architectural hypothesis is supported for the first structural-thalamus
target:

1. The current `MeshModel` path manually reimplements a restricted instance of
   the statistical model already represented by `samseg.GMM.GMM`.
2. Existing `GMM` state and established methods already cover multiple image
   channels, full or diagonal covariance matrices, multiple components per
   class, fitted mixture weights, NIW-compatible mean/covariance updates, GEMS
   parameter arrays, and final structure reconstruction.
3. Structural subfields++ can therefore make one `GMM` object authoritative
   without adopting the whole-brain `Samseg` orchestration. A thin adapter is
   still needed around class configuration, atlas-derived initialization,
   model-specific operations, mesh scheduling, and output handling.
4. The manual subregions likelihood, E-step, M-step, diagonal covariance
   packaging, unit component counts/weights, and extraction likelihood should
   not be preserved in the successor merely because they exist in the direct
   MATLAB port. The original thalamus, hippocampus, and other legacy classes
   remain outside this replacement boundary and must stay untouched.
5. The repository establishes direct-source correspondence between the old
   Python path and `SegmentThalamicNuclei.m`. It does **not** establish that the
   porter misunderstood either the mathematics or SAMSEG's OOP design. That is
   a plausible historical explanation, not an evidence-backed finding.

Mathematical validity does not by itself place a capability inside the first
GMM-reuse contribution. Under the revised evidence-backed contribution
boundary:

- reuse the established GMM density E-step, M-step, component/mixture state,
  covariance representation, GEMS handoff, and reconstruction unchanged;
- perform the exact thalamus `(nu, Psi)` to GMM `(h, H)` conversion privately at
  the subfields++ adapter boundary;
- defer all new log-domain GMM APIs from the first integration contribution;
  an ignored local recovery patch preserves the implementation and adversarial
  tests as non-authoritative prototype material for a possible focused
  numerical-stability follow-up;
- retain deletion of the temporary `subregions.gaussian` implementation;
- defer/remove the prototype low-mass retention method because it encodes an
  unvalidated hybrid policy, even though the underlying zero-mass problem is
  real;
- continue to defer adaptive covariance jitter and the unused alternative NIW
  prior-cost evaluator.

No production change to `GMM.py` has therefore been demonstrated as necessary
for the first structural GMM-reuse contribution.

This narrower contribution boundary permits construction and comparison of an
isolated successor without claiming statistical completion. The realistic
probe shows that zero-support and stopping policy must be resolved before the
successor is accepted for production activation, but they do not prevent the
architecture from being constructed and exercised.

Two statistical-policy questions remain open before statistical acceptance or
production activation:

- how to handle components with demonstrated exact zero support, and whether
  any positive low-mass threshold is justified;
- what coherent objective or stopping statistic to use with the intended
  zero-scale covariance prior.

Initialization is also an adapter responsibility: the established generic GMM
initializer is not valid for zero-support thalamus classes under this proposed
prior, but the constructor already accepts explicit initialized state.

## Scope and boundaries

This review answers two questions only:

1. What would be required to replace the structural subfields++ manual
   Gaussian/EM machinery with `GMM`?
2. Which current uncommitted Stage 2 additions are long-term GMM capabilities,
   and which appear to preserve temporary implementation choices?

The first target is the two-channel structural thalamus configuration. The
following remain constraints on later designs rather than subjects of this
replacement decision:

- hippocampal fitting with partially missing channels and ECM-style sufficient
  statistics;
- broader brainstem/other-subfields generalization;
- the final DTI model's many-to-many class/component incidence and
  class-specific component weights.

Standard, lesion, tumour, and longitudinal SAMSEG were inspected only to
establish the existing `GMM` contract and compatibility boundary. This was not
a model audit of those paths.

The investigation itself did not change legacy subfields, CLI behavior, or
successor architecture. At the closure checkpoint, production `GMM.py` was
restored exactly to `HEAD`, the unintegrated helper was retired, and the broad
prototype tests were reduced to focused evidence for existing GMM capability.

## Evidence snapshot

### SAMSEG state

| Item | Value |
|---|---|
| Repository | `/Users/henrytregidgo/PycharmProjects/Samseg/samseg` |
| Branch | `HT-subregions-integration` |
| `HEAD` | `fb77540ebd21fea49c7ff9cd19051bcba58e515d` |
| Local `origin/rectify` used | `53936b2625356bda0de04445c9032f9df54a3bf0` |
| Helper introduction | `2412fbb6ad5c16e142b551072fa7973ea6a03ec3` |
| Pre-cleanup prototype `GMM.py` SHA-1 | `b4d22e0ec84c3655d23937a583e06bba4bd9df0c` |
| Restored `HEAD:GMM.py` Git blob | `9d45e3aa191f21e65215044eb4113d9b6d0a6a76` |
| Ignored prototype patch SHA-256 | `88a990767c2142387ef2f640101d5fb37e9e9c8684452137563e2bcbb8590458` |
| `subregions/core.py` SHA-1 | `8ccfd685648c3a0caae83302fadb78f3357acc34` |
| Existing architectural audit SHA-1 | `7061eef9668c243995512229d8cf35c5e07b2cb2` |
| Existing integration plan SHA-1 | `6291d64a3275564fa935b9389d0400f34878f9cf` |

The pre-cleanup worktree contained the exploratory Stage 2 changes:

```text
 M samseg/GMM.py
 D samseg/subregions/gaussian.py
 D samseg/tests/test_subregions_gaussian.py
?? samseg/tests/test_gmm_subfields.py
```

No remote fetch was required for this bounded review. The table records the
exact local revisions used; it does not claim that un-fetched remote tips were
revalidated on 2026-08-11.

The prototype recovery patch is stored under the ignored
`for_testing_outputs/agent_discussions/` area. It is a convenience artifact,
not tracked evidence, project policy, or a prerequisite for later work.

### Reference implementations and artifacts

- MATLAB repository:
  `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg`
- MATLAB `HEAD`: `bdc58ad47633f10e360b999ddfd6af0a80339958`
- Direct-port reference:
  `SegmentThalamicNuclei.m`
- Final multichannel update reference:
  `HTtestfunctions/TS_fnc_fitGaussian_withprior.m:65-101`
- Final low-support condition:
  `HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m:1207` and `:3036`
- Installed structural/DTI atlas configuration:
  `/Applications/freesurfer/8.2.0/average/ThalamicNuclei/atlas_DTI/sharedGMMparameters.txt`
- Local two-channel smoke artifact:
  `tmp_mul_ch/processedImageMasked.mgz`,
  `tmp_mul_ch/processedImageMask.mgz`, and
  `tmp_mul_ch/finalWarpedMesh.txt.gz`

The local smoke artifact is useful feasibility evidence, but it is not a
validated scientific result or a substitute for the later manual smoke run.
The existing architectural audit established that identical FreeSurfer 8.0
beta, 8.1, and 8.2 configuration artifacts should be treated as stability
evidence rather than re-audited independently.

### Runtime

| Dependency | Version/source |
|---|---|
| Python | `.venv/bin/python`, 3.12.0 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| pytest | 9.0.3 |
| GEMS | repository-local `samseg/gems` |

## Evidence rules

This review uses five separate fields rather than forcing each behavior into
one mutually exclusive category:

- provenance or implementation context;
- demonstrated usage and compatibility status;
- mathematical or algorithmic consequence;
- preservation recommendation;
- confidence and unresolved evidence.

Observed source/runtime facts and architectural inference are stated
separately. In particular, absence of usage means no usage was found in the
audited repository, history already covered by the durable audit, tests,
scripts, shipped artifacts, or available workflows. Unknown private or
external workflows cannot be categorically excluded, but unsupported
hypothetical use does not create a strong preservation requirement.

## Existing ownership boundary

`GMM` is a statistical model object. It accepts an `(N, d)` observation matrix
and class priors, owns component parameters and hyperparameters, evaluates
component/structure likelihoods, performs parameter updates, and exposes the
arrays consumed by GEMS. It does not own image preprocessing, mesh rasterizing,
anatomical hyperparameter selection, deformation schedules, registered
operations, or output postprocessing.

Main SAMSEG already follows that boundary:

- it preserves `numberOfComponents` from `sharedGMMparameters.txt` in
  `samseg/io.py:5-24`;
- it creates the `GMM` with those counts in `samseg/Samseg.py:840-860`;
- it obtains responsibilities and performs the M-step in
  `samseg/Samseg.py:934-998`;
- it hands `gmm.means`, full `gmm.variances`, real mixture weights, and real
  component counts to GEMS in `samseg/Samseg.py:1024-1030`;
- it uses the same `GMM` object for final reconstruction in
  `samseg/Samseg.py:1107-1152`.

The structural subregions path instead owns a second, narrower statistical
implementation in `samseg/subregions/core.py:578-693` and duplicates its
likelihood again in `samseg/subregions/core.py:739-774`.

## Fitting lifecycle replacement map

| Lifecycle concern | Current structural subfields | Existing GMM/SAMSEG capability | Ownership and determination |
|---|---|---|---|
| Image and mesh preparation | `ThalamicNucleiDTI` resamples/stacks channels (`thalamusDTI.py:428`); `MeshModel` crops, masks, loads/rasterizes the mesh (`core.py:403`). | `GMM` consumes numerical data and priors only. | Keep in subfields orchestration. Do not adopt whole-brain preprocessing merely to use `GMM`. |
| Shared configuration | `get_label_groups()` parses only search strings and discards merged names/component counts (`thalamusDTI.py:670`). | `kvlReadSharedGMMParameters()` retains names, counts, and searches (`io.py:5-24`). | Reuse the established parser. Add only a narrow atlas adapter if a real matching difference is demonstrated. |
| Current structural grouping | Transitional code rebuilds groups and lookup tables. | The shipped configuration already defines separate one-component `LateralThal`, `MedialThal`, and `CornerThal` classes. | Direct reuse is feasible. A read-only mapping probe found 16 classes, 16 components, 132 structures, and no unmapped structure. |
| Class-prior reduction | `reduce_alphas()` manually sums alpha columns and creates one integer lookup (`core.py:174`). | `kvlGetMergingFractionsTable()` and `kvlMergeAlphas()` represent and apply class-to-structure fractions (`merge_alphas.py:14-57`). | Reuse established fractions. They support fitting reduction and mixture-aware reconstruction. |
| Hyperparameter estimation | Thalamus-specific masks, medians, and strengths are built in `thalamusDTI.py:884-922`; the current implementation reads only `inputImages[0]`. | `GMM` accepts per-component, per-channel hypermeans and matrix covariance priors but intentionally does not choose anatomy-specific summaries. | Keep policy/registered operations in subfields; translate their results to GMM arrays. Channel-aware estimation is unfinished orchestration, not a GMM math gap. |
| Initial state | Current code uses the legacy `0.01`-perturbed update and falls back to hypermean/variance 100 (`core.py:578-600`). | `GMM` supports generic initialization and explicit constructor-injected means, covariances, and weights through its constructor and `initializeGMMParameters()`. | Make GMM state authoritative. Atlas-derived initialization remains adapter policy. Do not preserve epsilon/variance-100 rules in the successor without scientific evidence. |
| Mid-run two-component hack | At one resolution, the grouping and hyperparameters can be replaced and Gaussian state reset (`core.py:504-520`). This is a class split, not a within-class mixture. | The structural config already expresses lateral, medial, and corner as classes; GMM separately supports genuine within-class mixtures. | Do not preserve or remove the reset automatically. Review its original role method by method; leave legacy classes unchanged. |
| E-step | One diagonal Gaussian per reduced class is evaluated and normalized manually (`core.py:603-631`). | Established `getGaussianPosteriors()` supports arbitrary components and full matrices; the archived Stage 2 prototype explored an opt-in log version. | Replace the manual E-step with the established density GMM method. Defer log-domain normalization to an independent numerical-stability review. |
| Mean/covariance M-step | Channelwise variance update with `0.01` perturbations (`core.py:645-661`); no cross-channel covariance or mixtures. | `fitGMMParameters()` updates means, weighted full covariance, optional diagonal projection, and class-normalized mixture weights. | Delegate the duplicated update equations to GMM. Review any surrounding historical fitting policy separately rather than preserving or discarding it by association. |
| NIW parameterization | Final thalamus MATLAB uses conventional `(nu, Psi)`. | Existing GMM consumes `(h, H)` in the same M-step. The investigation proves the exact map `h = nu + d + 2`, `H = Psi/h`. | Keep GMM's established native parameterization and perform this model-specific conversion privately in the subfields++ adapter. No second M-step or public GMM mapping API is needed. |
| Low posterior mass | Classes at/below mass `0.01` reset to hypermean/variance 100 (`core.py:648-660`). | Default GMM always updates. The archived Stage 2 prototype explored component-mass retention. | Exact zero support creates a demonstrated policy gap. Do not preserve either historical fallback automatically; review the legacy reset in context, and do not infer that a positive threshold is required. |
| Convergence objective | Manual density normalizer plus a partial mean-prior term controls inner stopping (`core.py:607-639`). | Main SAMSEG combines GMM likelihood and `evaluateMinLogPriorOfGMMParameters()` in its orchestration. | Loop ownership stays outside GMM. The existing prior evaluator cannot be blindly used with zero-scale `H=0`; stopping semantics remain unresolved. |
| Mesh deformation | `MeshModel` owns smoothing, alternating EM/deformation, optimizer construction, and stopping (`core.py:497-710`). | Main SAMSEG similarly orchestrates GMM around deformation; `ProbabilisticAtlas` has a different whole-brain policy. | Keep the subfields schedule. Reusing GMM does not imply replacing it with whole-brain deformation orchestration. |
| GEMS handoff | Variance vectors are converted to diagonal matrices and counts/weights are hardcoded to one (`core.py:671-693`). | GMM already stores the required `(G,d)`, `(G,d,d)`, `(G,)`, and `(C,)` arrays. Main SAMSEG passes them directly (`Samseg.py:1024-1030`). | Hand off authoritative GMM state. Remove successor hardcoding and diagonal reconstruction. |
| Final extraction | Original alphas are restored, structures are mapped to one class, and the diagonal likelihood is reimplemented (`core.py:739-774`). | Established `getLikelihoods()`/`getPosteriors()` reconstruct structures through components, weights, and fractions. | Reuse the same GMM object and established density reconstruction used in fitting. Keep rasterization, volumes, label selection, and postprocessing in subfields. |
| Mutable Gaussian state | `MeshModel` owns vector-shaped `self.variances`; prototype `synthseg_kmeans()` is the only non-core reader found (`thalamusDTI.py:797`). | GMM owns matrix covariances and all component state. | Make `self.gmm` authoritative in subfields++. No demonstrated active dependency requires preserving vector storage there. |
| Registered operations | JSON resolves groups and dotted functions during hyperparameter construction (`thalamusDTI.py:136`, `:910`; `utils.py:79`). | GMM has no anatomical registry. | Preserve this familiar subfields policy boundary, not inside GMM. Current hardcoded JSON location and broken/unused hooks are prototype evidence, not contracts. |

## Configuration and mixture feasibility

The installed structural configuration maps naturally to established SAMSEG
objects:

- `kvlReadSharedGMMParameters()` retains every configured class name,
  component count, and matching pattern.
- `kvlGetMergingFractionsTable()` produces the class-to-anatomical-structure
  matrix used both to reduce priors and reconstruct structure likelihoods.
- The installed structural/DTI file currently has 16 classes and one component
  per class, including separate lateral, medial, and corner thalamus classes.
- The current one-component configuration does not require a mixture extension,
  but using `GMM` stops the successor from discarding component counts when a
  configuration requests more than one.
- A `HEAD`-only synthetic probe with one class and two components fitted weights
  `[0.5, 0.5]` summing to one. Multiple-Gaussian state and weight updates are
  therefore established GMM capability, not a Stage 2 addition.

This supports making configured components and mixture weights first-class in
subfields++ without inventing a second component model.

## Numerical feasibility probe

### Method

The read-only probe used:

- the FreeSurfer 8.2 `atlas_DTI/sharedGMMparameters.txt`;
- the final warped mesh and two-channel masked image from `tmp_mul_ch`;
- mesh-derived class priors at the 18,811 masked voxels;
- channel-wise hard-class medians as provisional hypermeans;
- a minimum provisional mean strength of 10 solely to make the fitting probe
  well-defined;
- the intended structural covariance prior `nu = 0`, `Psi = 0`;
- explicit constructor injection of initial means, global full covariance, and
  equal within-class weights.

The median/strength choices are probe scaffolding, not an accepted
hyperparameter policy. The ephemeral reproducer is
`/tmp/gmm_replacement_probe.py`; it did not change repository files.

### Results

| Question | Result | Interpretation |
|---|---|---|
| Can GMM represent the configured model? | Means `(16,2)`, covariances `(16,2,2)`, weights `(16,)`, counts 16 x 1. | Yes. No storage conversion is required for GEMS. |
| Does the NIW mapping match? | `h = 4` in both calculations and maximum `H` difference `0`. | Exact mapping for `d=2`, `nu=0`, `Psi=0`. |
| Does log E-step match the established density E-step here? | Maximum responsibility difference `9.126033262418787e-14`; costs `77186.58060545148` and `77186.58060545177`; zero unsupported rows in both. | Numerically equivalent on this realistic artifact. Log space is not needed to rescue this particular dataset. |
| Does the full M-step match the final NIW equation? | For the weakest active class (mass `0.017448417339601657`), maximum mean and covariance differences were both `0`. | Exact independent numerical confirmation of the established parameter mapping/update equivalence. |
| Are active full covariances positive definite? | Minimum active covariance eigenvalue `9.180522962105635e-05`. | No realistic active-class jitter need was demonstrated. |
| Does diagonal mode require separate math? | Diagonal entries matched full-fit diagonals exactly; maximum off-diagonal was `0`. | Existing GMM diagonal projection implements the intended coherent diagonal mode. |
| Does existing GMM support mixtures? | `HEAD` fitted a two-component one-class example with weights `[0.5, 0.5]`. | Mixtures and weights predate Stage 2. |
| Does final reconstruction agree? | Density/log posterior maximum difference `9.126033262418787e-14`; no zero rows. | The same GMM state can drive extraction. |

The class posterior masses were:

```text
[79.16321369296682,
 8852.047132891508,
 367.24361916419497,
 0.0,
 0.0,
 0.0,
 4505.623601191206,
 313.7724070892901,
 223.3602069045198,
 0.017448417339601657,
 0.0,
 4469.772370648973,
 0.0,
 0.0,
 0.0,
 0.0]
```

Nine classes had no hard-assignment support and eight had posterior mass at or
below `0.01`. This is a realistic reason to resolve degenerate-component
behavior before statistical acceptance or production activation. It does not
prevent controlled successor execution, and it is not evidence that every
class below a positive threshold needs special handling.

### Initialization and zero-mass findings

Calling the established `initializeGMMParameters()` directly on these priors
produced 16 non-finite mean entries and 32 non-finite covariance entries for
zero-support classes. This does not require a new Gaussian engine: GMM already
accepts explicit initial state, and anatomical/group initialization belongs at
the adapter boundary. It does mean that a naive call to the generic initializer
is not a valid structural-thalamus integration.

After the ordinary M-step with the intended zero-scale covariance prior, eight
zero-mass components had non-positive (zero) covariance matrices. Applying the
prototype Stage 2 retention method kept the initial global covariance and left
no non-positive matrix. Therefore:

- an explicit exact-zero-support decision is necessary somewhere;
- the default zero-scale update is not positive definite by construction for a
  component with no data support;
- this result does not validate `minimumPosteriorMass=0.01`, retaining mixture
  weights, or the method's active-weight redistribution;
- no active component in this artifact required adaptive jitter.

The probe does not establish a general positive low-mass problem or threshold.

## Zero-support and low-mass policy review

Four materially different behaviors are present in the evidence:

| Source | Trigger | Behavior |
|---|---|---|
| Established main GMM | No special trigger | Always update. Zero mass plus zero-scale prior can yield singular covariance. |
| Legacy Python subregions | Total class mass `<= 0.01` | Reset mean to hypermean and variance to 100. |
| Deleted temporary helper | Copied broad low-mass fallback | Reset/fallback with helper stabilization. |
| Final MATLAB joint code | Update only if more than one responsibility exceeds machine epsilon | Otherwise retain previous fitted state. |
| Archived Stage 2 prototype | Total component mass at/below a caller threshold | Retain mean, covariance, and weight; redistribute remaining within-class weight over active components. |

The archived `fitGMMParametersWithLowMassRetention()` prototype was opt-in and
did not change the legacy default path, but it was not a direct implementation
of any one established policy. It combined:

- a mass-sum threshold derived from legacy/helper thinking;
- parameter retention resembling the final MATLAB behavior;
- newly invented mixture-weight retention and redistribution;
- a new rejection of tied Gaussians;
- a duplicated copy of the M-step.

The focused tests prove that the method implements those chosen semantics. They
do not establish scientific validity, a caller-selected threshold, equivalence
to MATLAB, or correct mixed frozen/active bookkeeping under nonzero mixture
hypercounts. Its survival through the documentation cleanup did not constitute
acceptance.

**Preservation recommendation:** defer/remove this method from Stage 2. Keep
the exact-zero-support policy unresolved until its structural context and
ownership boundary are reviewed; separately, do not assume that a positive
low-mass threshold is required. Confidence is high that the prototype method
is not justified; confidence is not yet available for the eventual policy.

## Convergence objective gap

Reusing GMM's E- and M-steps does not settle inner-loop stopping:

- current subregions uses the density normalizer plus a partial prior-like term;
- main SAMSEG uses likelihood plus
  `evaluateMinLogPriorOfGMMParameters()`;
- the existing durable audit found a parameter-dependent discrepancy in that
  prior evaluator;
- the intended thalamus mapping uses zero scale `H=0`, for which the established
  evaluator cannot simply be assumed valid;
- the alternative NIW prior-cost method was correctly removed from Stage 2
  because no concrete caller required it.

The earlier paper review lowers the urgency of the prior-normalization issue:
the matching M-step means it has not been shown to alter fitted Gaussian
parameters or responsibilities. It can still affect reported objective values
or a stopping decision if that evaluator is used. This is a policy/orchestration
gap to resolve before statistical acceptance or production activation, not
evidence for separate Gaussian math and not a blocker to controlled successor
construction or execution.

## Stage 2 method ledger

This ledger supersedes the earlier conclusion that mathematically sound,
isolated additions should remain in the first structural integration. The
revised test is whether a change is necessary at the same contribution
boundary as replacing duplicated subregions machinery.

All methods in this ledger describe the archived prototype, not functions in
the restored live `samseg/GMM.py`. Historical line numbers have deliberately
been removed because they no longer identify production code.

### `_log_power`

- **Provenance/context:** New private helper for log-domain powers, including
  deliberate zero-exponent semantics. It was not copied from the temporary
  Gaussian module.
- **Usage/compatibility:** Used only by new opt-in log methods. It has no effect
  on legacy density paths.
- **Consequence:** Prevents `0 * log(0)` from becoming `NaN` and preserves
  `dataWeight`/`priorWeight` behavior.
- **Recommendation:** **Defer from the first GMM-reuse contribution.** Preserve
  it with the log-domain prototype for the separate numerical-stability review.
- **Confidence/unresolved:** High. The boundary-level finite/nonnegative scan is
  proportionate and is outside per-Gaussian data loops.

### `_gaussian_log_likelihoods`

- **Provenance/context:** Cholesky-based log Gaussian kernel, algebraically
  equivalent to the density kernel and deleted helper.
- **Usage/compatibility:** Used only by additive log APIs. The historical
  density implementation remains independent.
- **Consequence:** Avoids converting already-underflowed densities to logs and
  naturally rejects singular/non-PD covariance through Cholesky failure.
- **Recommendation:** **Defer from the first GMM-reuse contribution.** Preserve
  the tested kernel as prototype evidence rather than production surface in the
  structural ownership change.
- **Confidence/unresolved:** High. No adaptive repair or repeated full-data
  validation occurs inside the component loop.

### `mapNormalInverseWishartPrior`

- **Provenance/context:** Named conversion from conventional `(nu, Psi)` to the
  established GMM `(h, H)` parameterization.
- **Usage/compatibility:** No production caller exists. The only demonstrated
  consumer starts from `(nu, Psi)` because that is how thalamus model policy
  expresses its prior; established GMM callers already use `(h, H)`.
- **Consequence:** `h = nu + d + 2`, `H = Psi/h` makes the existing M-step exact
  for the final thalamus NIW update.
- **Recommendation:** **Move the exact conversion to a private subfields++
  adapter helper.** Do not add a second public prior-parameterization interface
  to GMM for one model-specific caller.
- **Confidence/unresolved:** High for the mapping and adapter ownership. A
  future independent second consumer could justify promoting a general
  conversion later.

### `getGaussianLogLikelihoods`

- **Provenance/context:** Public wrapper around the private kernel. It most
  closely resembles API surface inherited from the temporary helper.
- **Usage/compatibility:** No high-level method or production code calls it;
  only focused tests do. Since `GMM` is exported from `samseg.__init__`, it would
  become public surface if committed.
- **Consequence:** Adds repeated shape validation and API surface without
  enabling fitting or extraction.
- **Recommendation:** **Defer/remove from the first contribution.** Preserve it
  only with the separate log-domain prototype if direct kernel testing remains
  useful there.
- **Confidence/unresolved:** High.

### `getGaussianPosteriorsLogSpace`

- **Provenance/context:** Additive, component-level log-domain E-step using
  log-sum-exp and established class/component semantics.
- **Usage/compatibility:** No production caller exists. Established
  `getGaussianPosteriors()` already supplies the immediate structural
  replacement, and the realistic two-channel artifact did not underflow.
- **Consequence:** Supports multiple components, mixture weights, weighted
  priors/data with stable normalization and avoids density underflow. The
  established density method already supports the same model features. The two
  paths agree on ordinary and realistic data; adversarial focused tests
  establish the numerical-stability difference.
- **Recommendation:** **Defer from the first GMM-reuse contribution.** Preserve
  the implementation and adversarial tests for an immediate, independently
  reviewed numerical-stability follow-up.
- **Confidence/unresolved:** High for mathematics and isolation. The realistic
  two-channel artifact did not itself underflow, so this is principled stable
  capability rather than a demonstrated dataset rescue.

### `getLogLikelihoods`

- **Provenance/context:** Log-domain equivalent of per-structure mixture
  likelihood aggregation.
- **Usage/compatibility:** Called by `getPosteriorsLogSpace`; no independent
  production or planned caller for raw log likelihoods was found.
- **Consequence:** Correctly performs component and class-to-structure
  log-sum-exp and preserves the density method's fractions cutoff.
- **Recommendation:** **Defer with the rest of the log-domain family.** A future
  stability review should decide whether this computation remains private or
  whether canonical existing methods adopt stable internals.
- **Confidence/unresolved:** High for the math. Long-term API visibility is a
  question for the independent numerical-stability review, not the structural
  adapter.

### `getPosteriorsLogSpace`

- **Provenance/context:** Additive anatomical posterior reconstruction using the
  same GMM component state and class fractions as fitting.
- **Usage/compatibility:** No production caller exists. Established
  `getPosteriors()` already supplies the immediate structural reconstruction,
  and it agreed with this prototype to approximately `1e-13` on the realistic
  artifact.
- **Consequence:** Preserves mixtures/weights and remains stable when density
  products underflow. The established density reconstruction is already
  sufficient to remove the duplicated subregions extraction likelihood in the
  first contribution.
- **Recommendation:** **Defer from the first GMM-reuse contribution.** Do not
  establish parallel stable/unstable posterior APIs as permanent architecture
  without a focused compatibility and API review.
- **Confidence/unresolved:** High.

### `fitGMMParametersWithLowMassRetention`

- **Provenance/context:** New hybrid policy, not a generic restatement of the
  established M-step or one historical thalamus implementation.
- **Usage/compatibility:** No production caller. It is opt-in and the default
  M-step remains untouched, but committing it would create public API surface.
- **Consequence:** Solves singular zero-mass covariance in the probe but
  duplicates M-step math and introduces unvalidated threshold, tied-Gaussian,
  and mixture-weight policies.
- **Recommendation:** **Defer/remove in its current form.** The need is real;
  this API is not accepted.
- **Confidence/unresolved:** High on non-acceptance. Eventual trigger, fallback,
  mixture behavior, and owner remain unresolved.

### Temporary helper and test retirement

- **Provenance/context:** `samseg/subregions/gaussian.py` and its tests were
  introduced entirely by `2412fbb` as an unintegrated first-stage helper.
- **Usage/compatibility:** No production caller or demonstrated external
  workflow was found. Unknown private use cannot be excluded, but there is no
  evidence for a preservation contract.
- **Consequence:** Keeping it would duplicate full likelihood and NIW update
  mathematics. Its validator, GEMS converter, jitter repair, and prior-cost
  evaluator become unnecessary when GMM owns matrix covariance state and the
  speculative policies remain deferred.
- **Recommendation:** **Retain deletion.** Preserve first-contribution tests
  only where they demonstrate existing GMM reuse, adapter mapping, or legacy
  isolation. Keep deferred log-domain tests with the separate prototype rather
  than using them to justify production API in this contribution.
- **Confidence/unresolved:** High.

## Stage 2 compatibility and isolation

Before cleanup, repository caller search found every direct use of the Stage 2
APIs in the prototype `samseg/tests/test_gmm_subfields.py`; no production
Python, CLI, shell, JSON, documentation workflow, or shipped workflow activated
them. The retained focused test file now uses only established GMM methods.
Consequently, removing the additions has no demonstrated compatibility cost.

The structural ownership evidence therefore supports restoring `GMM.py` to its
established production surface for the first contribution. The log-domain
implementation and tests remain useful ignored prototype evidence, but their
correctness is not evidence that they must land with subfields++ integration.

The two mature methods affected by the first Stage 2 attempt have already been
restored:

- `getGaussianLikelihoods()` is AST-identical to `HEAD` and retains its
  historical density execution path;
- default `fitGMMParameters()` is AST-identical to `HEAD` and retains its
  historical reduction order and semantics.

The previously measured representative single-channel ratios of approximately
`0.98-1.01x` are benchmark noise rather than evidence of regression. The
prototype validation and Cholesky operations are isolated from the established
density path, but the revised boundary defers them rather than relying on that
isolation as a reason to expand GMM in the first contribution.

## Temporary-requirement drift

The following assumptions in
`samseg/subregions/for_testing/multichannel_gaussian_plan.md` are stale and must
not be treated as accepted Stage 2 requirements:

- exact `Subfields Useage` docstring-heading enforcement: this was a temporary
  helper requirement; the enforcement test has been removed, the typo has been
  corrected in useful docstrings, and no repository-wide policy exists;
- failure-only adaptive jitter: realistic active components in the available
  artifact were positive definite, so speculative repair remains deferred;
- settled low-mass retention: an exact-zero-support policy is needed, but no
  positive low-mass threshold is established and the prototype method is
  explicitly unaccepted by this review;
- retention of log-domain APIs merely because they are mathematically sound and
  isolated: realistic evidence does not make them a prerequisite for replacing
  duplicated subregions machinery;
- retention of a public NIW mapping merely because the conversion is exact:
  the only demonstrated `(nu, Psi)` caller belongs to thalamus model policy.

The log-domain implementation and NIW mapping are not incorrect. Their revised
disposition follows contribution scope and ownership evidence, not a reversal
of the algebraic or numerical findings. Natural singular-covariance failure at
exact zero support remains unresolved; a broader positive low-mass problem has
not been established.

## Deferred constraints

### Missing structural channels

Neither current `MeshModel` nor current `GMM.fitGMMParameters()` computes
conditional sufficient statistics for observations with different missing
channels. Hippocampal ECM-like fitting is a later general-GMM question and does
not block the present structural-thalamus target, whose two channels are jointly
observed in the probe.

### Final DTI incidence model

Current GMM assigns a contiguous private component slice to each class. The
final DTI formulation requires components shared across classes with
class-specific weights. That is a later ownership/design question. Success for
the current 16-class structural model must not be reported as final DTI
completeness.

### Existing operation registry

Registered JSON operations remain a subfields policy surface, not a GMM
requirement. The current code demonstrates the intended familiar mechanism but
not a stable contract: the JSON path is hardcoded (`thalamusDTI.py:323`),
`second_hyps_hack` is not consumed, and `bimodal_thal_hack` calls
`find_hyps_idx` incorrectly (`utils.py:107`). These observations constrain the
future adapter but do not justify moving hooks into GMM or broadening this
review into hook cleanup.

## Decision record

### Established conclusively

- Structural subfields++ does not need a separate Gaussian mathematics module.
- Existing GMM can represent and fit the current multichannel structural model.
- Full NIW fitting is exactly equivalent under the documented parameter map.
- Coherent diagonal mode is the diagonal projection of the same GMM update;
  the legacy vector formula is not a required second model.
- Existing GMM already owns multiple components and mixture weights.
- The same GMM state can be passed directly to GEMS and used for extraction.
- Established density E-step, M-step, and reconstruction behavior are adequate
  for the demonstrated first structural replacement.
- No production extension to `GMM.py` is currently demonstrated as necessary
  for that contribution.
- The exact thalamus NIW conversion can be owned privately by the adapter while
  GMM retains its established `(h, H)` contract.
- The temporary helper has no demonstrated compatibility contract.
- Zero-support classes are realistic and require an explicit decision before
  statistical acceptance; they do not block controlled successor construction
  or execution.
- The prototype low-mass method is not sufficiently evidence-backed to retain.

### Remains inference or unresolved

- Why the original porter did not reuse GMM. Source correspondence is proven;
  personal understanding and delivery context are not.
- The scientifically preferred exact-zero-support fallback, and whether any
  positive low-mass trigger is needed.
- Mixture-weight behavior when only some components of a class are frozen.
- The convergence/stopping objective under the zero-scale covariance prior.
- The long-term stable-likelihood design, including whether canonical density
  APIs should adopt log-domain internals or any parallel APIs should exist.
- Validated output parity on a complete structural-thalamus smoke run.
- Requirements imposed later by hippocampal missing channels or final DTI
  shared-component incidence.

## Checkpoint decision

Human review accepted the narrow no-`GMM.py`-extension boundary for the first
structural replacement. The closure checkpoint therefore:

1. restores production `GMM.py` exactly to its established implementation;
2. retires the unintegrated `subregions.gaussian` helper and its helper-specific
   tests;
3. retains focused tests of the existing GMM capabilities that justify that
   retirement;
4. keeps the broad prototype only as an ignored recovery patch; and
5. stops before introducing `MeshModelPlus` or changing any legacy lifecycle.

The exact zero-support policy, convergence statistic, covariance-mode
configuration, initialization edge cases, resolution-level reset provenance,
and output acceptance criteria remain unresolved. They are statistical
acceptance or successor-design questions, not prerequisites to creating and
exercising an isolated successor architecture.

See `subfields_plus_architecture.md` for the current design context. That note
separates repository facts, reported team direction, present preferences, and
unresolved questions rather than converting this audit into a specification.

## Verification record

The checkpoint is verified with:

```bash
git diff --exit-code HEAD -- samseg/GMM.py
.venv/bin/python -m py_compile samseg/GMM.py samseg/tests/test_gmm_subfields.py
.venv/bin/python -m pytest -q samseg/tests/test_gmm_subfields.py
.venv/bin/python -m pytest -q samseg/tests
git diff --check
```

The focused tests cover established full-covariance likelihood evaluation,
one- and two-channel `(h, H)` updates, diagonal projection, multiple-component
responsibilities and fitted mixture weights, and structure reconstruction.
They deliberately do not establish a zero-support policy or retain any
prototype API.

Final command results for the closure checkpoint are recorded below.

Results on 2026-08-13:

```text
Focused existing-GMM evidence: 6 passed in 1.31s
Maintained samseg/tests lane: 13 passed, 5 warnings in 12.17s
Production GMM diff: empty
Tracked whitespace check: passed
```

The maintained warnings were the existing physical-core detection fallback and
four divide-by-zero runtime warnings from segmentation and longitudinal tests.
