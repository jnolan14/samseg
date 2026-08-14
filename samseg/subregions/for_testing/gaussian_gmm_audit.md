# SAMSEG Gaussian/GMM Architectural Audit

Date: 2026-08-06

Status: complete, durable local audit record. Revised after primary-source
review of the 2015 hippocampal subfields paper, the 2016 SAMSEG paper, the
2018 structural thalamus paper, the 2019 preliminary DTI paper, the 2023 final
DTI paper and supplement, and `SegmentThalamicNuclei.m`. This report stops
before code changes or migration planning.

Later decision note (2026-08-13): subsequent replacement review narrowed the
first structural contribution further than this audit's option-B wording.
Existing `GMM` density APIs are sufficient to begin the isolated successor, so
the exploratory log-domain APIs, public NIW conversion, and low-mass policy
were not retained in production `GMM.py`. The temporary `subregions.gaussian`
helper was retired. See `gmm_subfields_replacement_review.md` for that decision
and `subfields_plus_architecture.md` for current successor context. References
to the helper below remain historical evidence, not current architecture.

## Executive conclusion

The permanent duplication represented by `samseg/subregions/gaussian.py` is
not justified by the evidence collected.

Main SAMSEG already owns the more general statistical model: multiple image
contrasts, full covariance matrices, a diagonal constraint, multiple Gaussian
components per class, fitted mixture weights, and full parameter transfer to
GEMS. Its full-covariance M-step and Gaussian likelihood are mathematically
equivalent to the new subregions helper under an exact NIW parameter mapping.
The differences that remain are chiefly policy, incomplete prototype behavior,
or defects, rather than an irreducible subregions-specific Gaussian model.

The 2016 SAMSEG paper confirms that this is the intended abstraction of the
SAMSEG likelihood model, not merely an accidental capability of `GMM.py`. It
defines multicontrast observations, full covariance Gaussian mixtures,
per-component weights, and sharing of one mixture model across related
anatomical structures. This makes an evolved `GMM` the leading ownership
hypothesis.

The 2015 hippocampal subfields paper independently establishes joint T1/T2
full-covariance fitting as published and exercised subfields behavior. It also
implements expectation conditional maximization (ECM) for voxels where a
channel is missing because the T2 field of view is incomplete. This does not
establish a current Python API contract, but it means that multichannel
subfields fitting and missing-channel support cannot be dismissed as new or
DTI-only prototype capabilities.

Using `samseg.GMM.GMM` entirely unchanged is not yet supported. The current
class has an inconsistent prior-cost evaluator, no safe default zero-mass
policy, and a density-space responsibility path that can underflow. Its public
arrays and density-valued methods are actively consumed by standard, lesion,
tumour, bias-field, GEMS, and longitudinal code. These are constraints on how
`GMM` is evolved, not evidence that another permanent owner is required.

For the structural subfields++ stage, the best-supported target is therefore
option B: reuse and modestly evolve `GMM` so subfields++ can consume the
established general model without inheriting whole-brain orchestration. The
final 2023 DTI model, however, demonstrates a many-to-many mapping between
global structural or diffusion components and anatomical classes, with
class-specific mixture weights. Current `GMM` does not represent that mapping
unchanged. Option B must eventually generalize component incidence if the
final DTI model remains the target; otherwise option C, extraction of neutral
shared kernels behind policy wrappers, becomes a concrete alternative. The
evidence establishes the need for one owner, but does not make a neutral
module a mathematical necessity.

The established single-channel subregions output remains a shipped legacy
behavior, but the original thalamus, hippocampus, brainstem, and related model
classes are outside the subfields++ upgrade path and will remain untouched.
Subfields++ therefore does not need to reproduce the legacy `0.01`-perturbed
update as a compatibility policy. That update is not a one-dimensional NIW
update, is not translation equivariant, and is not the optimizer of its
reported prior-augmented cost. It should remain confined to the legacy paths
rather than define the new multichannel backend.

For the thalamus, the evidence supports keeping anatomical grouping and
hyperparameter initialization outside the Gaussian engine. The 2018 method
preassigns nuclei to structural parameter-sharing groups, and the 2023 method
adds a third structural distribution motivated by medial PuM. The actual
MATLAB configurations define separate one-component lateral, medial, and
sometimes corner classes from atlas groups. The final DTI initialization first
groups structural labels from that configured model and runs several
structural-only GEM iterations; k-means is subsequently used to initialize
configured diffusion mixtures, not to create the structural medial/lateral
split. The separate generic MATLAB structural k-means helper is likewise
conditional on a configured component count greater than one. The Python
`second_hyps_hack` configuration is an incomplete test prototype and should
not be treated as an existing contract.

The main-GMM prior-cost discrepancy remains mathematically demonstrated, but
its operational priority is lower than the ownership decision. It does not
enter responsibility calculation and does not change the algebraic M-step for
a fixed set of responsibilities. It is used for inner and outer stopping, so
it can change iteration counts and potentially outputs indirectly; no material
segmentation difference has yet been demonstrated.

## Project intent and compatibility boundary

The following is project intent supplied by the maintainer, rather than an
inference from discovered callers:

- "subfields" denotes the family containing thalamic nuclei, hippocampal
  subfields, brainstem, and related region-specific models;
- the existing region-specific implementations remain available and are not
  to be changed as part of subfields++;
- subfields++ is the structurally clean successor path, with dynamic grouping
  from shared-parameter files, multichannel structural inputs, and explicit
  JSON-selected operations for unavoidable region-specific behavior;
- those operations are policy hooks around the common fitting machinery, not
  alternative owners of Gaussian mathematics;
- the class currently named `ThalamicNucleiDTI` is a transitional home for
  subfields++ structural work; actual DTI/WMM likelihood processing is a later
  stage.

This boundary removes a previously assumed requirement to preserve legacy
subregions numerical behavior inside subfields++. Established outputs remain
available through the untouched original model classes. Subfields++ can adopt
the principled general SAMSEG model while retaining only scientifically or
operationally necessary region policy.

## Scope and evidence rules

This audit covered:

- historical provenance of main GMM, Python subregions, MATLAB structural
  fitting, grouping files, covariance modes, priors, low-mass handling,
  mixture support, GEMS handoff, and extraction;
- the published hippocampal subfields, SAMSEG, structural thalamus,
  preliminary DTI, and final DTI likelihood and optimization descriptions from
  2015, 2016, 2018, 2019, and 2023, including the 2023 supplementary methods,
  plus direct source correspondence with `SegmentThalamicNuclei.m`;
- callers and activation paths in production Python, CLIs, tests, scripts,
  configurations, local workflow artifacts, installed FreeSurfer releases,
  and reachable branch history;
- full fitting-lifecycle comparisons for main SAMSEG, Python subregions, and
  the relevant MATLAB structural path;
- one-channel and multichannel mathematics, density versus log-density,
  prior normalization, and the six mandatory deterministic probes;
- architecture options A through D.

Standard, lesion, tumour, and longitudinal SAMSEG were inspected only far
enough to establish the GMM contract and compatibility risk. They were not
subjected to a complete scientific model audit.

Observed facts, mathematical deductions, and historical inferences are
identified separately. Existing code is treated as evidence, not automatically
as a contract. Absence claims are bounded to the audited evidence sources.

## Frozen evidence snapshot

### SAMSEG repository

- Repository: `/Users/henrytregidgo/PycharmProjects/Samseg/samseg`
- Branch: `HT-subregions-integration`
- `HEAD`: `2412fbb6ad5c16e142b551072fa7973ea6a03ec3`
- Upstream relation: two commits ahead of `origin/rectify`, zero behind
- `origin`: `git@github.com:jnolan14/samseg.git`
- `upstream`: `https://github.com/freesurfer/samseg.git`
- Tracked worktree before and after evidence collection: clean. This durable
  report is the sole subsequent untracked repository file.

Remote refs were fetched without changing a branch or tracked file. The refs
used after the fetch were:

| Ref | SHA |
|---|---|
| `origin/rectify` | `53936b2625356bda0de04445c9032f9df54a3bf0` |
| `origin/dti_integration` | `c4c66febbe3a22718c16759d5f66a9cee4ef1988` |
| `origin/HT-local-subregions-testing` | `3be107e344e46ed8e4d4dd579f316d8379734ca8` |
| `origin/dev` | `a516ee9586a4174522db826b5ea4831c8121e895` |
| `upstream/dev` | `cb4eff23f7c1b36c704d42acac7fb135a818fb25` |
| local `docs/dti-migration-checklists` | `bd0e97f3311948740aec7db2f6c1f584b0d87bec` |

The audited `GMM.py` has Git blob
`9d45e3aa191f21e65215044eb4113d9b6d0a6a76` on all primary refs above. Its
filesystem SHA-1 is `820a6cea77a68ecfee38e10fd4f6ef14f330403f`.

### MATLAB repository

- Repository: `/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg`
- Branch: `master`
- `HEAD` and `origin/master`:
  `bdc58ad47633f10e360b999ddfd6af0a80339958`
- Remote: `git@github.com:htregidgo/ThalamusSeg.git`
- Worktree before and after audit: clean
- Relevant remote refs included
  `origin/29-mixture-assignments=5a9e63e5d74e533f94e55960e6dc875a36754987`,
  `origin/35-variable-likelihoods=73de14b21fd3e5dba56bed431bb29a148f436b6a`,
  and
  `origin/fix-initial-objective=eeb0fc59b417b87e996221456db530ae4e558b95`.

### Runtime and shipped artifacts

- Python: `.venv/bin/python`, version 3.12.0
- NumPy: 2.4.6
- SciPy: 1.17.1
- pytest: 9.0.3
- GEMS imported from the repository-local
  `/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/gems/__init__.py`.
- No MATLAB or Octave executable was available; MATLAB evidence is source,
  history, algebra, and independent NumPy evaluation rather than executing
  MATLAB itself.

FreeSurfer 8.1 and 8.2 copies of `GMM.py` are byte-identical to the current
file. Their Python subregions cores are also byte-identical to each other, with
filesystem SHA-1 `6561c273d7bfc1d0edbf34ad992e559e99866e03`.
The current multichannel core differs, with SHA-1
`8ccfd685648c3a0caae83302fadb78f3357acc34`.

The shipped DTI `sharedGMMparameters.txt` files in available FreeSurfer
8.0-beta, 8.1, and 8.2 installations are identical, with SHA-1
`32a3ff08c5c405c8ebd1eb99d64cda536c3ca09e`. Identical releases were treated
as stability evidence and not re-audited separately.

FreeSurfer 8.1 and 8.2 ship the single-channel Python subregions CLI, but no
Python `thalamusDTI.py`. They separately ship a compiled MATLAB Runtime DTI
workflow through `segmentThalamicNuclei_DTI.sh` and
`SegmentThalamicNuclei_DTI.app`.

## Decision ledger

The fields below are deliberately not mutually exclusive. A behavior can be an
active dependency and a probable defect at the same time.

| Behavior | Provenance or implementation context | Demonstrated usage and compatibility status | Mathematical or algorithmic consequence | Preservation recommendation | Confidence and unresolved evidence |
|---|---|---|---|---|---|
| Published SAMSEG GMM abstraction | Puonti et al. (2016) defines multicontrast full-covariance mixtures, per-component weights, and sharing one mixture across related structures | Main `GMM` implements and active callers depend on this general model | The required subfields++ statistics already fit the intended GMM abstraction | Treat an evolved `GMM` as the presumptive single owner; extract kernels only for a demonstrated constraint | Published abstraction conclusive; pre-2023 code-design history unavailable |
| Published multichannel subfields | Iglesias et al. (2015) jointly fits T1/T2 hippocampal data with full covariance and ECM for incomplete T2 coverage | Published experiments exercise simultaneous T1/T2 fitting; no equivalent current Python activation was found | Establishes cross-channel covariance and missing-channel estimation as prior subfields capabilities | Preserve these as architectural capabilities where the successor scope requires them; do not infer a current Python storage/API contract | Scientific use conclusive; current external workflow dependence and first-release ECM scope unresolved |
| Subfields++ compatibility boundary | Maintainer-defined successor path; original region classes remain untouched | Legacy models retain established outputs; subfields++ itself has no released compatibility contract | New fitting can be principled without reproducing legacy scalar quirks | Do not add a legacy-update compatibility mode to subfields++; preserve behavior by leaving original paths unchanged | Project requirement explicit; future external subfields++ use does not yet exist |
| Full Gaussian likelihood | Present in imported main GMM, 2020 MATLAB structural code, and the new helper | Main GMM and MATLAB are active; helper is test-only at `HEAD` | Algebraically equivalent when parameters match | Reuse the `GMM` owner; preserve current density-returning methods where active callers require them | Conclusive algebra and numerical evidence |
| Final full covariance M-step | Main GMM predates local history; the current MATLAB helper was added in May 2020; the 2023 supplement publishes the same update; the new helper follows it | Main GMM actively used; MATLAB ships as compiled DTI workflow; helper not integrated | Exact equivalence under `h=nu+d+2`, `H=Psi/h` | Reuse one `GMM`-owned coherent NIW update with explicit parameter mapping | Conclusive for the final 2023/MATLAB formulation; original pre-2023 GMM implementation history is unavailable |
| Published covariance-update lineage | The 2015 hippocampal paper reports denominator `N`; the 2019 preliminary DTI paper reports `N+1`; the 2023 supplement reports `N+nu+d+2` | Each formula documents a distinct generation; the final formula corresponds to the shipped DTI MATLAB family | These are different covariance modes, not notation-only variants; only the final form maps exactly to current GMM NIW hyperparameters | Treat the 2023/final formulation as the DTI reference and record older formulas as historical model lineage, not successor compatibility requirements | Equations conclusive; rationale for each transition is not fully documented |
| Legacy diagonal subregions update | Directly ported from `SegmentThalamicNuclei.m`; mechanically vectorized in 2026 | Shipped legacy single-channel behavior; original model classes will remain untouched | Not NIW, not translation equivariant, and differs from diagonal projection of full update | Leave it in untouched legacy paths; do not reproduce it in subfields++ | Formula, source correspondence, and project boundary conclusive; original rationale for `0.01` unresolved |
| Main GMM diagonal mode | Imported with main GMM | Active configurable main-GMM behavior | Exactly diagonal projection of the full update under the same priors | Preserve as the principled diagonal constraint in shared statistics | Conclusive |
| Main GMM prior cost | Imported unchanged; the 2016 paper assumes a flat likelihood-parameter prior and does not document this later NIW evaluator | Used in standard and tumour EM stopping, outer stopping, summaries, and longitudinal costs; not used in responsibilities | Contains an extra parameter-dependent `0.5 log det(Sigma)`; fixed-responsibility M-step is correct, but reported objective is inconsistent | Track as a separate convergence/reporting defect; do not make it a blocker to GMM ownership without an output-impact probe | Mathematical discrepancy high confidence; no demonstrated segmentation difference; external cost consumers unknown |
| MATLAB proper-IW normalization | Added June 2020 | Structural calls use `nu=0`, `Psi=0`, so the affected normalization is inactive there | Sign of the `nu/2 log det(Psi)` term differs from normalized IW; constant in fitted parameters for fixed hyperparameters | Follow the normalized mathematical form, document MATLAB structural-kernel parity separately | Conclusive equation/numerical evidence; original intent unresolved |
| Linear-density GMM API | Already present at 2023 import | Active internal and exported API used by standard, lesion, tumour, and extraction paths | Can underflow to zero and replace true NLL with an epsilon floor | Preserve density-valued public methods; use log space internally or add log methods | Active compatibility plus probable numerical defect; origin rationale unresolved |
| Multiple components and weights | Core published SAMSEG model; present in imported GMM; MATLAB generalized component machinery in 2021; final DTI paper fits structural and diffusion mixture weights | Main atlas configures `Soft 3 Soft`; tumour constructs three components; GEMS consumes weights/counts; final DTI uses mixtures scientifically | Represents useful and eventually required mixture behavior absent from legacy subregions | Reuse and expose established mixture capability rather than omit or reimplement it | Main capability and published DTI requirement conclusive; automated production-atlas coverage is missing |
| Global class/component incidence | The 2023 DTI model defines global structural and diffusion components with class-specific weights and permits many-to-many sharing | Published final DTI model, including a CSF component-sharing example; current structural subfields++ stage need not activate it immediately | More general than current GMM's contiguous components owned by one class | Do not let the structural-stage design preclude this mapping; generalize GMM incidence or use shared kernels before final DTI integration | Published requirement conclusive; ownership design and first-release scope unresolved |
| Low-mass handling | Four independently evolved policies | MATLAB policy has explicit history; legacy fallback remains in untouched models; main default can produce NaNs | Results differ materially and main default zero mass is unsafe | Give subfields++ an explicit safe policy through the GMM boundary; do not carry forward the legacy variance-100 fallback by default | Consequences conclusive; scientifically preferred policy needs validation |
| Covariance stabilization | MATLAB added determinant repair in Oct 2021; helper now uses failure-only Cholesky/jitter | Helper only; no current production cost | Zero-scale/improper NIW can be singular, so SPD is not guaranteed by construction | Retain a failure-only safeguard as policy; do not use unconditional eigendecomposition | Need for safeguard conclusive; optimal threshold requires data evidence |
| Missing-channel estimation | Iglesias et al. (2015) uses ECM and a closed-form two-channel special case for incomplete T2 field of view; GEMS likelihood code can omit missing channels | Published T1/T2 experiments exercise it; no current Python subregions fitting path implementing ECM was found | Complete-case fitting or dropping cross-covariance would change the established multichannel model | Include missingness in the architectural contract if hippocampal T1/T2 parity is in successor scope; do not bolt it into region hooks | Published behavior conclusive; current Python demand and initial subfields++ release scope unresolved |
| Subregions `self.variances` vector shape | Scalar at import, changed to `(classes, channels)` in 2026 | No reader outside subregions core found in audited sources | Representation only; GEMS already requires full matrices | Do not treat vector shape as a strong contract; prefer full matrices at shared boundary | High confidence within audited sources; private direct-Python callers cannot be excluded |
| `sharedGMMparameters.txt` component count | MATLAB and main SAMSEG consume component structure; Python DTI discards count | Shipped DTI files currently all use count one | Python subregions silently cannot activate mixtures even when configured | Preserve the configuration semantics, including counts, in the general engine | Conclusive code/config evidence |
| Atlas lateral/medial/corner split | The 2018 paper preassigns MD/pulvinar versus other nuclei to shared structural distributions; the 2023 model adds a third distribution motivated by medial PuM; Python and MATLAB configs use lateral/medial/corner names | Shipped and published model-specific behavior | Changes class priors and fitted parameters; not the same as a mixture sharing one class prior | Keep anatomical grouping in shared-parameter/model configuration outside the Gaussian engine | Configured grouping conclusive; `CornerThal` is implementation terminology and the exact validated config remains unresolved |
| K-means component initialization | Generic structural MATLAB support added February 2021; the 2023 final method separately uses clustering for configured diffusion mixtures after structural initialization | No tracked current structural thalamus config has component count greater than one; final DTI diffusion initialization is published | Initializes components inside an already configured group; it does not define the medial/lateral structural split | Make available when component configuration requests it; do not make it the default structural thalamus split | Structural configuration and final DTI staging conclusive; private structural configs cannot be excluded |
| Hyperparameter initialization policy | Published methods use modality-wise or channel-wise robust medians, anatomical masks or eroded groups, class volume/count strength, and specialized hippocampal partial-volume simulation | Exercised by hippocampal, structural thalamus, and DTI methods; no paper support found for `+5/-5` offsets or an `nHyper` floor | Initialization can materially alter early responsibilities while remaining separate from Gaussian update mathematics | Keep these operations in model configuration and narrow region-policy hooks; use channel-aware group medians as the general baseline | Published positive evidence strong; preferred low-mass floor and exact successor hook contract unresolved |
| JSON hyperparameter hooks | Python DTI/subfields++ prototype from February/March 2026 | `post_em_update` is partially called; `second_hyps_hack` is never consumed and its target is broken | Intended to isolate unavoidable region policy, not define alternative Gaussian statistics | Keep the existing familiar mechanism narrow; repair only the hooks required by subfields++ and keep Gaussian math in `GMM` | Current implementation status conclusive; final hook contract unresolved |
| Fitting/extraction/GEMS consistency | Main GMM reuses its object; current subregions duplicates diagonal formula; prototype connected helpers | Main paths active; helper-only restart not active | Current diagonal fitting and extraction agree, but full mode reaches neither at `HEAD` | The single GMM owner must serve fitting and reconstruction; always pass full matrices to GEMS | Conclusive source evidence |

## Historical provenance

### Published SAMSEG abstraction

The primary published reference is
[Puonti et al. (2016)](/Users/henrytregidgo/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/ReadingMaterials/2016_NI_Puonti_samseg.pdf),
"Fast and sequence-adaptive whole-brain segmentation using parametric Bayesian
modeling."

Printed page 238 (PDF page 4) defines each observation as a multicontrast
vector and each anatomical label likelihood as a Gaussian mixture with full
covariance matrices and component weights. Printed page 239 gives the GEM
responsibility, mean, covariance, weight, and bias-field updates. Its
implementation section explicitly modifies those updates so a set of related
anatomical structures can share one mixture model.

This evidence establishes that `GMM` is intended to represent a reusable
multicontrast Gaussian-mixture likelihood model, including shared anatomical
classes. It is not merely whole-brain orchestration that happens to contain
useful Gaussian functions. Dynamic subfields++ groupings and multiple
components fit this abstraction naturally.

The paper assumes a uniform likelihood-parameter prior, `p(theta) proportional
to 1`. It therefore does not document or validate the later NIW hyperprior
evaluator in current `GMM.py`. It does state that each GEM iteration increases
its objective, that the objective controls inner likelihood-parameter
stopping, and that cost decrease controls outer GEM/deformation interleaving.
Those statements are relevant to the current prior-cost discrepancy only
through convergence monitoring, not as evidence that the published M-step used
the current NIW prior.

### Published subfields and thalamus lineage

The hippocampal subfields reference is
[Iglesias et al. (2015)](/Users/henrytregidgo/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/ReadingMaterials/2015_NeuroImage_Iglesias_HippocampalSubfields.pdf),
"A computational atlas of the hippocampal formation using ex vivo,
ultra-high resolution MRI: Application to adaptive segmentation of in vivo
MRI."

PDF pages 16-20 define and exercise a global-class Gaussian model in which
anatomical labels share tissue parameters. The observations and means are
vectors and the covariances are full matrices when T1 and T2 are fitted
simultaneously. The Winterburn experiment explicitly compares T1, T2, and
joint T1/T2 fitting. PDF page 25 then applies simultaneous T1/T2 fitting to the
ADNI data, where T2 has a restricted field of view. Appendix 1 on PDF pages
33-34 uses ECM to estimate Gaussian parameters when individual voxel channels
are missing, including a closed-form specialization when one of two channels
is always observed. This is demonstrated published multichannel subfields
usage, not merely permissive array shape.

The same paper initializes each channel's hypermean with the modality-wise
median from an `aseg` anatomical class and its mean-prior strength with the
number of contributing voxels. It restricts some supporting classes to local
anatomical regions to reduce bias-field drift and uses subject-specific
partial-volume simulation for the alveus and molecular layer. These are
scientifically motivated initialization policies around a general Gaussian
model, not distinct Gaussian mathematics.

The 2015 paper states a normal-inverse-Wishart prior with zero covariance
hyperparameters, but its displayed covariance M-step on PDF page 18 divides
by effective mass `N`. A normalized conditional Gaussian mean prior contributes
an additional covariance determinant power, for which the corresponding mode
would instead contain an extra denominator term. The published prior statement
and displayed update are therefore not algebraically coherent as one
normalized MAP objective. The `N` update has real published provenance, but
that provenance does not make it a scientific or successor compatibility
contract.

The structural thalamus reference is
[Iglesias et al. (2018)](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2018_NeuroImage_Iglessias_thalamicAtlas.pdf),
"A probabilistic atlas of the human thalamic nuclei combining ex vivo MRI and
histology." PDF page 7 assigns reticular nucleus to a white-matter group,
MDm/MDl and PuA/PuM/PuL/PuI to a second shared structural distribution, and
the remaining nuclei to a third. The paper explains that the boundary between
the latter two groups is visible in in vivo MRI and stabilizes atlas fitting.
These groups are preassigned anatomical parameter-sharing sets rather than a
per-subject clustering result.

The 2018 contrast-robustness experiment applies the same adaptive algorithm
separately to each available MRI contrast and compares the resulting
segmentations. It does not describe a joint structural multichannel thalamus
fit. It also does not define a `CornerThal` group or publish the legacy
`+5/-5` hypermean offsets.

The preliminary DTI reference is
[Iglesias et al. (2019)](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2019_IPMI_Iglesias_diffusionMRBayesianSeg.pdf),
"Joint inference on structural and diffusion MRI for sequence-adaptive
Bayesian segmentation of thalamic nuclei with probabilistic atlases." PDF
pages 4-7 specify a possibly multispectral structural Gaussian with full
covariance and a mean/covariance prior, but report the covariance update

```text
Sigma = [n (mu-M)(mu-M)^T + Cmu] / [N + 1].
```

The experiments on PDF page 8 use one T1 structural image plus dMRI, with the
T1 hypermean set to an `aseg` class median and its strength to class volume in
mm3. Structural distributions are shared across predefined anatomical groups;
the paper does not fit structural Gaussian mixtures. It therefore establishes
a generic multichannel formulation but not an evaluated two-structural-channel
thalamus workflow.

The final DTI reference is
[Tregidgo et al. (2023)](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2023_NeuroImage_Tergidgo_thalamusBayesian.pdf),
"Accurate Bayesian segmentation of thalamic nuclei using diffusion MRI and an
improved histological atlas." Equations 6 and 12 on PDF page 4 define global
structural components `i`, global diffusion components `j`, and class-specific
weights `g[c,i]` and `w[c,j]`. PDF page 6 explicitly permits many-to-many
class/component relationships, including a CSF class containing a clean-CSF
Gaussian shared with ventricles and a messy-CSF Gaussian shared with choroid
plexus. This is more general than merely assigning a private contiguous block
of Gaussian components to each class.

The 2023 atlas splits PuM spatially into lateral and medial labels, fits them
separately, and merges them for output. The paper explicitly describes the
previous structural model as two Gaussian distributions for medial/lateral
contrast, adds a third Gaussian for medial PuM, and evaluates 33 candidate
three-distribution structural groupings. This provides scientific context for
a third structural distribution; `CornerThal` remains an
implementation/configuration name, not terminology from the paper.

The official
[2023 supplementary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10636587/)
resolve two further ambiguities. Section S.1.1 reports the final structural
update

```text
mu = [n M + sum_v q_v x_v] / [n + N]

Sigma = [Psi + n (mu-M)(mu-M)^T + Cmu]
        / [nu + d + N + 2].
```

The main experiments use one T1 structural channel; adding T2 was explored
with limited benefit. During initialization, however, the method temporarily
uses two scalar channels: T1 plus FA for DSW-beta, or T1 plus tensor
log-determinant for Wishart and log-Gaussian. Section S.3 first groups labels
according to configured structural components, erodes the corresponding
`aseg` groups, computes robust prior statistics, and runs several
structural-only GEM iterations. K-means is then used to initialize already
configured multi-component diffusion mixtures. It does not discover the
medial/lateral structural grouping.

The supplement generally sets `nu=0` and `Psi=0`, describing this as
non-informative covariance prior information. Algebraically this is an
improper zero-scale kernel and leaves the final denominator `N+d+2`; it does
not add a positive-definite scale matrix. Rank-deficient or nearly empty
weighted scatter can therefore still produce a singular covariance, which
supports failure-triggered stabilization rather than an unconditional
eigendecomposition.

### Main GMM and original Python subregions

`2c353bfd1f0b198ce827c8f4406dfbaaf7837066`, 2023-03-29,
`ste93ste`, "add latest FS samseg code, reshuffle files", is the first local
appearance of both [GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):8
and the Python subregions implementation. The parent has neither path.

The imported GMM already had full matrix storage, diagonal projection,
multicontrast Cholesky likelihoods, multiple components, mixture weights,
weighted updates, and parameter priors. The current file is unchanged from
that import across the primary audited refs and FreeSurfer 8.1/8.2.

`1da1451b0c4260247ab4683ce8c8235b6a045af6`, 2023-03-30, Oula,
"Fixed setup, added atlas, moved cli/ inside samseg/", added the main atlas.
Its [sharedGMMParameters.txt](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/atlas/20Subjects_smoothing2_down2_smoothingForAffine2/sharedGMMParameters.txt):10
contains `Soft 3 Soft`, demonstrating configured mixture use.

The original [subregions core](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/core.py):44
describes itself as an "almost perfect port of the subfield matlab code". The
maintainer identifies
[SegmentThalamicNuclei.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m):1
as the direct source of the original Python thalamus/subfields path. Source
correspondence independently supports that provenance:

- MATLAB lines 1315-1338 and 1419-1440 contain the same `EPS=1e-2`
  initialization/M-step denominators, exterior variance addition, low-mass
  hypermean/variance-100 fallback, and exact-zero repair;
- lines 1347-1365 calculate scalar Gaussian responsibilities and the same
  conditional-mean-like prior cost;
- lines 1497-1500 pass scalar variances, unit weights, and one Gaussian per
  reduced class to GEMS;
- lines 1626-1636 recompute the scalar likelihood during extraction.

At the importing commit the Python code reproduces each of these structures.
This is direct-port compatibility behavior, not an independently designed
alternative to the general SAMSEG GMM model.

The original [thalamus policy](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamus.py):218
already used a two-stage reduction. The second stage separates atlas-listed
lateral and medial anatomical classes and applies T1-specific `+5/-5`
hypermeans at
[thalamus.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamus.py):326.
This predates Python multichannel work and is not a within-class Gaussian
mixture. The MATLAB source uses the `useTwoComponents` name but likewise
rebuilds the reduced atlas into separate lateral and medial fitted classes at
lines 1134-1213, with T1-specific `+5/-5` hypermeans. "Component" in this
legacy path should not be confused with two mixture components sharing one
class prior in the SAMSEG GMM sense.

The repository does not contain pre-import FreeSurfer history. Original
authorship and rationale for the main density API, later prior
parameterization, and GMM defaults remain unresolved. The direct source family
of the Python port is established, although the exact source revision used by
the original porter is not recorded in this repository.

### Python multichannel and covariance work

| Commit | Author/date | Relevant change |
|---|---|---|
| `c82b0b9f48656b2a30070e40d8ae65dc08c299f3` | Jackson Nolan, 2026-02-23 | Added the transitional subfields++ class currently named `thalamusDTI.py`; vectorized means/variances over channels while retaining the scalar formula and fallback |
| `37ceb28241fea79e1d1a167da196608c3d27c096` | Jackson Nolan, 2026-03-31 | Added the hard-coded local JSON/test harness |
| `c4c66febbe3a22718c16759d5f66a9cee4ef1988` | Jackson Nolan, 2026-05-01 | Corrected final extraction for N channels |
| `ff9a401aff6a041dbf20d33b07e2485aa466c4e7` | Henry Tregidgo, 2026-05-06 | Branch-only local smoke configuration |
| `6bf638e4cf4ca80747b1c42e495802561ce92375` | Henry Tregidgo, 2026-05-14 | Branch-only local test profiles |
| `3be107e344e46ed8e4d4dd579f316d8379734ca8` | Henry Tregidgo, 2026-05-27 | First integrated diagonal/full covariance prototype |
| `53936b2625356bda0de04445c9032f9df54a3bf0` | Jackson Nolan, 2026-07-28 | Corrected crop/resample handling for image lists, not Gaussian math |
| `2412fbb6ad5c16e142b551072fa7973ea6a03ec3` | Henry Tregidgo, 2026-08-06 | Restarted with standalone Gaussian helpers/tests, without production integration |

The February 2026 change is best described as an independent-channel
generalization of the legacy scalar code. It kept the old threshold,
denominators, and variance-100 fallback. No commit message or nearby code
documents a scientific decision to prefer this diagonal model. Its filename
reflects the intended later DTI expansion; the current first release target is
the configurable multichannel structural subfields++ backend.

The May covariance prototype explicitly documented covariance modes,
channel-aware means, shared fitting/extraction helpers, and full GEMS handoff,
but remained on `origin/HT-local-subregions-testing`. Its full mode used an
NIW-style M-step while reporting only a mean-prior cost, so the prototype did
not yet have a coherent full objective.

The current helper commit deliberately restarts at a reviewable mathematical
boundary. At `HEAD`, no production module imports
[gaussian.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/gaussian.py):356;
only [test_subregions_gaussian.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/tests/test_subregions_gaussian.py):1
uses it.

### MATLAB evolution

| Commit | Author/date | Relevant change |
|---|---|---|
| `879021316b73b138ee176504f93e6eb6d9262a26` | Henry Tregidgo, 2020-04-29 | Added separate shared GMM/WMM grouping files |
| `7657083eb0b07089e1d5c1411a2c0107b220ba17` | Henry Tregidgo, 2020-05-04 | Added multichannel structural Gaussian fitting |
| `3ebf90e40d48c192a99b2a246106bb1ebf06f4f5` | Henry Tregidgo, 2020-05-05 | Corrected Gaussian fitting bugs |
| `6a51ee11bac87a8a38d3b51d7ac3d9a433385d31` | Henry Tregidgo, 2020-05-05 | Passed structural channels to GEM |
| `439dc54e782e463be8c3dbbc3afc12bb4c03ff19` | Henry Tregidgo, 2020-06-05 | Added NIW prior evaluation |
| `112da4ab31fa806a91caae737c9bd3369f57161d` | Henry Tregidgo, 2020-11-25 | Optimized Gaussian fitting |
| `fd333916228d9b5d06d765da439c083ead403539` | Henry Tregidgo, 2021-02-10 | Added k-means hyperparameter fitting for split components |
| `6db11b296e06a6ded8828cd84ec649687111ea17` | Henry Tregidgo, 2021-09-06 | Fixed the initial objective; message notes convergence impact |
| `613264832cf9bb88923bf41c99e067e2f263ff71` | Henry Tregidgo, 2021-10-20 | Retained parameters for disappearing GMM classes |
| `eeb0fc59b417b87e996221456db530ae4e558b95` | Henry Tregidgo, 2021-10-25 | Added almost-zero posterior and covariance determinant handling |
| `cdfd953ee3b43a7f332fd7f71718c2c82552f456` | Henry Tregidgo, 2021-11-16 | Added zero-posterior normalization handling |
| `7fa44c8d816908a4fa36eeaf69c8ccfc501ac7a0` | Henry Tregidgo, 2022-01-05 | Allowed medial/corner priors to be calculated jointly |

The MATLAB full-covariance structural implementation therefore predates both
Python subregions multichannel implementations. Generic k-means component
initialization came later and is conditioned on a configured component count
greater than one.

The current generic helper uses channel-wise medians for one component and
city-block k-means plus atlas weight-mask assignment for multiple components:
[TS_fnc_fitGaussianHyperParams.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_fitGaussianHyperParams.m):108.
The tracked current thalamus parameter files all request one component. They
define separate lateral/medial groups; selected model configurations also
define a separate corner group.

## Caller and usage map

### Evidence levels

| Evidence level | Main SAMSEG GMM | Python subregions |
|---|---|---|
| Technically permits | Multiple contrasts, full/diagonal covariance, multiple components and weights | Cross-sectional core accepts image lists, but fits one independent diagonal Gaussian per reduced class |
| Covered by tests | Standard, lesion, longitudinal, and longitudinal-lesion test definitions, all one contrast and one component per test class | Helper unit tests include genuine two-channel off-diagonal covariance; no integrated fit test |
| Exercised by scripts | `run_samseg` accepts repeated inputs; `run_samseg_long` accepts modality lists per timepoint | Released CLI hardcodes `norm.mgz`; manual DTI harness supplies T1+FA |
| Demonstrated workflow | Installed wrappers support multiple imported modalities; production atlas uses a three-component class | Ignored local artifacts demonstrate an apparent diagonal T1+FA thalamus run from before the full-mode commit |
| Documented/external interface | Public `GMM` export and documented T1+FLAIR, T1+FLAIR+PD, and longitudinal inputs | Official and executable interface is T1-only; no secondary-image CLI argument |

### Main GMM dependencies

- [Samseg.initializeGMM](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):840
  derives component counts from the shared configuration, contrast count from
  images, and covariance mode from model specifications.
- [Samseg EM](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):971
  calls responsibilities, prior cost, and the M-step.
- [Samseg mesh handoff](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):1024
  passes means, full matrices, weights, and component counts.
- [Samseg extraction](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):1144
  calls the same GMM object's posterior path.
- [BiasField](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/BiasField.py):137
  directly inverts each full covariance matrix.
- [ProbabilisticAtlas](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/ProbabilisticAtlas.py):119
  constructs one GEMS image per contrast and transfers the complete GMM state.
- [SamsegLesion](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/SamsegLesion.py):103
  changes hyperparameters, ties lesion and white-matter Gaussians, and directly
  reads component counts, means, weights, and contrast count.
- [SamsegTumor](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/SamsegTumor.py):375
  directly invokes likelihood, responsibility, prior, M-step, bias-field,
  deformation, and posterior methods. It also constructs a three-component
  model at line 747.
- [SamsegLongitudinal](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/SamsegLongitudinal.py):397
  copies means, full covariance matrices, and weights between SST/timepoint
  models, and its latent updates invert those matrices.
- [samseg.__init__](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/__init__.py):8
  exports `GMM` as a direct Python API.

The main covariance shape, component ordering, per-class weights, mutable
parameter arrays, and density-valued methods are therefore active compatibility
surfaces. This does not make their internal equations correct, but it constrains
how they can be changed.

### Subregions dependencies

The production route is the registry in
[process.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/process.py):14,
the lifecycle beginning at line 35, and the console entry point in
[setup.cfg](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/setup.cfg):31.

The released
[segment_subregions CLI](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/cli/segment_subregions.py):64
explicitly leaves variable image lists as future work and hardcodes
`['norm.mgz']`. Longitudinal subregions resamples only the first image and has a
multi-image TODO at
[process.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/process.py):142.

Current `process.py` advertises `thalamusDTI`, but the constructor requires
`atlasDir`, `inputDTIDirName`, and `dtiLikelihood`, which the generic CLI does
not supply. That registry entry is not a functional CLI activation path.

Brainstem, standard thalamus, and hippocampus can technically receive image
lists through direct construction, but their hyperparameter estimators derive
contrast information only from `inputImages[0]`:
[brainstem.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/brainstem.py):168,
[thalamus.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamus.py):256,
and
[hippocampus.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/hippocampus.py):381.
This is technical permissiveness, not demonstrated multichannel support.

The only discovered Python multichannel subregions activation is the local
thalamic DTI harness:
[dti_args_FA.json](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/for_testing/dti_args_FA.json):4
and
[shell_test.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/for_testing/shell_test.py):36.
Ignored artifacts dated 2026-05-14 contain a final mesh and segmentation, but
predate the 2026-05-27 covariance-mode commit. They support a completed
diagonal T1+FA run, not a full-covariance run.

No reader of subregions `self.variances` outside
[core.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/core.py):482
was found across audited refs, tests, scripts, configs, installed packages, or
available workflows. This is much weaker compatibility evidence than for main
`gmm.variances`.

No current Python multichannel subregions activation outside
`ThalamicNucleiDTI`, and no completed current Python full-covariance
subregions workflow, was found in the audited repositories, reachable history,
tests, scripts, shipped artifacts, official interfaces, or available
workflows. Unknown private or external direct-Python workflows cannot be
categorically excluded, but no strong Python API-preservation requirement
follows from hypothetical use alone.

This current-caller result is distinct from scientific and historical usage.
Iglesias et al. (2015) reports and evaluates joint T1/T2 hippocampal subfields
segmentation with full covariance, including missing T2 voxels. The published
workflow is strong evidence for model capability and provenance, but the paper
does not by itself demonstrate dependence on the current Python
`self.variances` representation or prototype `ThalamicNucleiDTI` interface.

## Full fitting-lifecycle comparison

| Lifecycle stage | Main `GMM` | Current Python subregions | MATLAB structural path |
|---|---|---|---|
| Anatomy to class mapping | Shared config maps structures to classes | Subclasses build reduction tables; DTI parses labels but discards component count | Separate structural, diffusion, reduced-class, component, and weight maps |
| Components per class | Configurable contiguous component counts | Exactly one | Configurable; components can participate in richer mapped spaces |
| Mixture weights | Equal initialization, fitted with optional Dirichlet-like hypercounts | Fixed to one | Explicit weight-space responsibilities and updates |
| Mean initialization | Weighted class mean, split intervals for multiple components | Hypermean or legacy weighted update | Median for one component; k-means/atlas assignment for multiple components |
| Covariance initialization | Full class covariance, optionally projected diagonal | Per-channel vectors | Full matrices |
| Likelihood | Cholesky Gaussian density | Computes diagonal log density, then exponentiates | Full log density |
| Missing channels | Python fitter assumes a common observed channel vector; GEMS filtering can omit unavailable channels | No missing-channel fitting path | Published 2015 hippocampal path uses ECM; current DTI structural helper assumes supplied channels are observed |
| Responsibilities | Density times weight/class prior, normalized with float64 epsilon | Density times atlas prior, normalized with float32 epsilon | Log terms, voxelwise max subtraction, then normalization |
| Sufficient statistics | Per-component weighted masses/scatter | Per-class mass and channelwise squared deviations | Per-component weighted mass/scatter |
| Mean M-step | Coherent weighted mean with `N+kappa` | Adds `0.01` to denominator only | Coherent weighted mean with `N+kappa` |
| Covariance M-step | Full NIW-style matrix update; optional diagonal projection | Legacy channelwise variance plus exterior `0.01` | Full NIW-style matrix update |
| Prior cost | Mean, IW-like, mixture terms; extra log determinant | Conditional-mean-like term inconsistent with M-step | NIW kernel; proper-scale normalization has a sign discrepancy |
| Low mass | No skip; proper priors work, defaults produce NaNs at zero mass | Reset to hypermean and variance 100 below `0.01` | Retain previous values unless more than one responsibility exceeds epsilon |
| Regularization | Hypervariance floor, no explicit failed-component branch | Replace exact zero variance with 100; helper uses failure-only jitter | Determinant threshold followed by isotropic replacement |
| EM objective | Density normalizer plus prior cost | Density normalizer plus mean-prior-like cost | Log-space likelihood plus Gaussian prior and other terms |
| Mesh interaction | Passes full GMM state to GEMS | Converts vectors to diagonal matrices; unit weights/counts | Passes full matrices, weights, counts, and channels |
| Final extraction | Same GMM object and mixture model | Duplicated diagonal formula matching fit likelihood | Reuses parameters/weights in log-sum-exp-style reconstruction |

The MATLAB class/component representation is more general than current main
GMM in one respect: a likelihood component can be reused across reduced spaces
with class-specific weights after structural/diffusion intersections. Main GMM
assigns each contiguous component to exactly one class. The 2023 final DTI
paper explicitly requires the more general many-to-many mapping; this is no
longer a hypothetical capability question. It does not block the current
one-component structural thalamus configuration, but the structural-stage
architecture must not make final DTI incidence prohibitively difficult.

The transitional subfields++ class currently named `ThalamicNucleiDTI` loads
and validates DTI-specific files and a `dtiLikelihood` selector, but it does
not override `fit_mesh_to_image`; the base Gaussian path remains the structural
image-stack fitter. No production use of the loaded `FAImage` or selector to
implement the MATLAB WMM likelihood was found. The current first-stage target
is therefore a configurable multichannel structural backend, not a complete
port of the joint structural/diffusion MATLAB likelihood.

## Equations and parameter mapping

Let samples be `x_i` in `R^d`, responsibilities be `r_i`, and define

```text
N   = sum_i r_i
s   = sum_i r_i x_i
Cmu = sum_i r_i (x_i - mu)(x_i - mu)^T
```

Let `mu0` be the prior mean, `kappa` its strength, `Psi` the inverse-Wishart
scale, and `nu` its degrees of freedom.

### Gaussian likelihood

All full implementations evaluate the same log density:

```text
log p(x) = -1/2 [d log(2 pi) + log|Sigma|
                 + (x-mu)^T Sigma^-1 (x-mu)]
```

Main [GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):112
returns its exponential. The new
[helper](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/gaussian.py):466
returns the log value. MATLAB
[TS_fnc_gaussianPDF.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_gaussianPDF.m):65
can return either. A diagonal matrix reduces exactly to the helper's
channelwise diagonal likelihood.

### Published covariance-update lineage

The supplied publications contain three materially different covariance
denominators under the notation above:

```text
2015 hippocampal subfields:
    Sigma = [Cmu + kappa (mu-mu0)(mu-mu0)^T] / N

2019 preliminary thalamus DTI:
    Sigma = [Cmu + kappa (mu-mu0)(mu-mu0)^T] / [N + 1]

2023 final thalamus DTI:
    Sigma = [Psi + Cmu + kappa (mu-mu0)(mu-mu0)^T]
            / [N + nu + d + 2]
```

These are different objectives or approximations, not alternative symbols for
one update. The 2015 formula is on PDF page 18 of Iglesias et al. (2015), the
2019 formula is Equation 13 on PDF page 7 of Iglesias et al. (2019), and the
final formula is Equation S.4 in Section S.1.1 of the 2023 supplement. The
current generic MATLAB helper belongs to the final 2020-2023 lineage. Calling
all three simply "the MATLAB NIW update" would obscure a real model evolution.

### Final 2023/MATLAB full covariance update

The final 2023 formulation, current MATLAB helper, and new Python helper use:

```text
mu = (s + kappa mu0) / (N + kappa)

Sigma = [Psi + Cmu + kappa (mu-mu0)(mu-mu0)^T]
        / [N + nu + d + 2]
```

See
[TS_fnc_fitGaussian_withprior.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_fitGaussian_withprior.m):61
and
[gaussian.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/gaussian.py):634.

Main [GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):244
uses covariance hyperparameters `H` and `h`:

```text
Sigma = [Cmu + kappa (mu-mu0)(mu-mu0)^T + h H] / [N + h]
```

The exact mapping is:

```text
h = nu + d + 2
H = Psi / h
```

Under this mapping the updates are identical, before any failure-only
stabilization. Defaults are not equivalent: main GMM defaults approximately to
`h=d-1, H=I`, whereas the structural MATLAB call uses `nu=0, Psi=0`, hence
`h=d+2, H=0`.

### Legacy subregions diagonal update

With `tau=0.01`, current subregions uses:

```text
mu_s = (s + kappa mu0) / (N + kappa + tau)

v_s[j] = [sum_i r_i (x_ij-mu_s[j])^2
          + kappa (mu_s[j]-mu0[j])^2] / (N + tau) + tau
```

There is no fixed NIW/GMM mapping for this formula. The unmatched `tau` in the
mean denominator has no pseudo-observation in the numerator, and the exterior
variance floor corresponds to a mass-dependent numerator. Shifting all data
and `mu0` by a constant does not shift `mu_s` by that constant.

Main GMM diagonal mode instead performs the full update and sets off-diagonal
elements to zero. It is exactly the diagonal projection of the full result.

## Mandatory numerical evidence

All probes used NumPy float64 at the frozen SHAs. They directly instantiated
`GMM`, called the helper functions, and evaluated the written equations. The
focused helper tests also passed. Inputs and expected values below are enough
to reproduce each comparison without a broad audit harness.

### 1. One-channel update equivalence and divergence

Input:

```text
X = [[0], [2], [5], [8]]
r = [0.2, 0.4, 0.8, 0.3]
mu0 = [1.5]
kappa = 2
N = 1.7
nu = 2
Psi = [[0.7]]
```

Mapped main GMM and full helper:

```text
mean       = 2.756756756757
variance   = 2.668818071803
max delta  = 0
```

Closest main-GMM representation of the legacy setup versus the legacy helper:

```text
main mean       = 2.756756756757
main variance   = 10.106518282989
legacy mean     = 2.749326145553
legacy variance = 10.057535305741
legacy - main   = -0.007430611204, -0.048982977248
```

Maximum one-channel log-likelihood differences were `8.88e-16` for main GMM
versus full helper and `1.78e-15` for diagonal versus full helper.

### 2. Two-channel full-covariance equivalence

Using a deterministic weighted two-channel sample and the mapping above:

```text
max log-likelihood difference, GMM vs helper = 4.44e-16
max mean update difference                   = 0
max covariance update difference             = 0
```

This reconfirms the previous equivalence to floating-point precision.
Because Equation S.4 of the 2023 supplement is the same mapped update, this
probe is also numerical evidence for equivalence to the final published
structural covariance formula. It is not evidence that the distinct 2015 or
2019 denominators are equivalent.

### 3. Diagonal-update divergence

For the same two-channel data and structural MATLAB defaults `h=4`, `H=0`:

```text
main/full mean              = [1.614583333333, 1.937500000000]
main diagonal variances     = [1.128885582011, 0.743055555556]
diag(full helper)           = [1.128885582011, 0.743055555556]
maximum projection delta    = 0
unconstrained off diagonal  = 0.695932539683

legacy mean                 = [1.611226611227, 1.933471933472]
legacy variances            = [3.088802273189, 2.036548866466]
legacy - main mean          = [-0.003356722107, -0.004028066528]
legacy - main variances     = [1.959916691178, 1.293493310910]
```

### 4. Multiple Gaussians and mixture weights

For one class, two one-dimensional components with means `[0,5]`, variances
`[1,1]`, starting weights `[0.8,0.2]`, and data `[0,1,4,5]`:

```text
responsibilities =
[[0.999999068338, 0.000000931662],
 [0.999861748024, 0.000138251976],
 [0.002207453848, 0.997792546152],
 [0.000014906390, 0.999985093610]]

component masses = [2.002083176599, 1.997916823401]
weights without prior = [0.500520794150, 0.499479205850]
```

With prior weights `[0.25,0.75]` and strength 4:

```text
fitted weights = [0.375260397075, 0.624739602925]
expected       = (mass + [1,3]) / 8
```

This is established main-GMM behavior. The helper has no component or mixture
abstraction. The probe covers ordinary private components within one GMM
class; it does not establish support for the final DTI model's globally shared
components with class-specific incidence weights.

### 5. Low-mass behavior

At total posterior mass `0.005`:

```text
legacy/helper diagonal: mean=mu0, variance=[100,100]
helper full:            mean=mu0, covariance=100 I

main GMM with proper priors:
mean = [1.012468827930, 2.024937655860]
covariance = [[1.059847260749, 0.122191400400],
              [0.122191400400, 1.243134361350]]
```

At exactly zero mass:

```text
main GMM with proper priors: mean=mu0, covariance=I
main GMM with defaults:      mean=NaN, covariance=NaN
```

MATLAB instead skips fitting unless more than one responsibility exceeds
machine epsilon and retains prior iteration parameters.

### 6. Prior-cost discrepancy

For two positive-definite covariance matrices:

```text
GMM prior delta                       = 1.827248486506
coherent NIW prior delta              = 1.584188861183
delta of deltas                       = 0.243059625323
0.5 * change in log determinant       = 0.243059625323
```

A central finite difference with step `1e-6` along
`Sigma(s)=exp(s) Sigma_Mstep` gave:

```text
d(data NLL + GMM prior)/ds       = 1.000000000140
d(data NLL + coherent prior)/ds  = 0.000000000000
expected discrepancy d/2, d=2   = 1
```

For the MATLAB structural kernel (`nu=0`, `Psi=0`), MATLAB and the helper prior
values differed by `8.88e-16`. With a proper nonzero scale:

```text
MATLAB prior - helper prior = -0.471011246429
-nu * logdet(Psi)           = -0.471011246429
```

### Probe method

The full-update comparisons can be reproduced by constructing one main GMM
component with:

```python
h = nu + X.shape[1] + 2
gmm = GMM(
    [1], X.shape[1], useDiagonalCovarianceMatrices=False,
    initialMeans=np.zeros((1, X.shape[1])),
    initialVariances=np.eye(X.shape[1])[None, ...],
    initialMixtureWeights=np.ones(1),
    initialHyperMeans=mu0[None, :],
    initialHyperMeansNumberOfMeasurements=np.array([kappa]),
    initialHyperVariances=(Psi / h)[None, ...],
    initialHyperVariancesNumberOfMeasurements=np.array([h]),
)
gmm.fitGMMParameters(X, r[:, None])
helper_mean, helper_covariance = full_covariance_posterior_update(
    X, r, mu0, kappa,
    degrees_of_freedom=nu,
    scale_matrix=Psi,
)
```

The executed evidence additionally compared direct log-density transforms,
the legacy helper, main diagonal projection, main mixture responsibilities,
zero/tiny-mass branches, written prior equations, and a central difference.
No extraction or live GEMS probe was needed because the exact array shape and
parameter transfer are explicit at the Python/C++ boundary.

## Prior and normalization audit

The 2016 SAMSEG paper is important context but not a specification of this
current prior code. Its published likelihood model sets the parameter prior to
be uniform and gives maximum-likelihood GEM updates. The NIW hyperparameters
and `evaluateMinLogPriorOfGMMParameters()` analyzed here are later
implementation behavior. The paper therefore supports the general GMM
ownership boundary and the importance of a coherent stopping objective, but it
does not establish that the current NIW evaluator is part of the published
statistical model.

The subfields papers do specify informative mean priors, but they do not form
one unchanging NIW lineage. Iglesias et al. (2015) reports denominator `N`,
Iglesias et al. (2019) reports `N+1`, and the final 2023 supplement reports
`N+nu+d+2`. The analysis below applies to the final formulation and current
MATLAB helper unless an older paper is named explicitly.

For the NIW model

```text
mu | Sigma ~ N(mu0, Sigma/kappa)
Sigma      ~ IW(Psi, nu)
```

the parameter-dependent joint log prior is:

```text
log p(mu,Sigma) =
    -kappa/2 * (mu-mu0)^T Sigma^-1 (mu-mu0)
    -(nu+d+2)/2 * log|Sigma|
    -1/2 * trace(Psi Sigma^-1)
    + normalization constants
```

Under `h=nu+d+2`, `Psi=hH`, the coherent negative log prior is:

```text
J = kappa/2 * q
    + h/2 * log|Sigma|
    + h/2 * trace(H Sigma^-1)
    + constants
```

The 2019 `N+1` update is the covariance stationary point obtained from a
normalized conditional Gaussian mean prior with no additional covariance
prior determinant power. The 2015 `N` update corresponds to dropping even the
conditional Gaussian's parameter-dependent determinant normalization while
retaining its quadratic mean penalty. The final 2023 update includes the
conditional mean determinant and the stated inverse-Wishart kernel. These
differences explain the denominator sequence; they do not cancel elsewhere in
a fixed-responsibility M-step.

Current
[GMM.evaluateMinLogPriorOfGMMParameters](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):282
instead evaluates:

```text
J_GMM = kappa/2 * q
        + (h+1)/2 * log|Sigma|
        + h/2 * trace(H Sigma^-1)
        + constants
```

The difference is `0.5 log|Sigma|` plus parameter-independent terms. No
alternative interpretation of `h` makes both this evaluator and the M-step
denominator coherent. The finite-difference probe confirms that the M-step is
stationary for the coherent objective, not the reported one.

The prior is added to each main EM stopping cost at
[Samseg.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):971,
used for EM convergence at line 983, added to outer coordinate-descent cost at
line 1050, and written to summaries/cost files through
[run_samseg.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/cli/run_samseg.py):228.
Tumour and longitudinal paths also consume it or aggregated final costs.

GEMS receives Gaussian parameters but not the Python prior value. The
discrepancy therefore does not directly alter a deformation gradient. It can
change EM and outer stopping, iteration counts, recorded costs, and indirectly
the final result. No internal model-selection caller comparing alternative GMM
models by this cost was found. Unknown external consumers of recorded costs
cannot be excluded.

For a fixed set of responsibilities, the discrepancy does not change the
parameter update. It is also absent from responsibility calculation itself;
responsibilities depend on the current means, covariances, weights, and class
priors. Consequently, there is no present evidence that the discrepancy has
materially changed fitted parameters, posterior responsibilities, or final
segmentations in a real run. Its demonstrated status is a theoretical
objective inconsistency and a possible convergence/reporting defect. It should
be investigated independently, but it is lower priority than establishing the
single GMM ownership boundary for subfields++.

Main GMM omits other normalizing constants, so even a corrected parameter-
dependent term would not turn its reported value into absolute model evidence.

MATLAB
[TS_fnc_gaussprior_NIW.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_gaussprior_NIW.m):85
uses the opposite sign from normalized inverse-Wishart mathematics for
`nu/2 log|Psi|`. That difference is constant in fitted parameters when
hyperparameters are fixed. It does not change the structural M-step and is
inactive in actual structural calls because they use `nu=0`, `Psi=0`, but it
would change absolute costs across differing proper scale priors.

The legacy subregions prior cost also fails to match its M-step. For more than
one channel it includes `-0.5 log(kappa)` rather than
`-d/2 log(kappa)`, although that missing part is constant during a fixed run.
More importantly, the variance optimum of its normalized likelihood/mean prior
has denominator `N+1`, while the implementation uses approximately `N` plus
threshold perturbations. The unperturbed `N` denominator has antecedent in the
2015 published hippocampal formula, so it is not merely an accidental Python
port change; the paper and implementation nevertheless share the same
objective inconsistency. Existing subregions EM is not coordinate descent on
one coherent reported MAP objective.

## Density versus log-density

Main GMM's density-valued methods are an active API convention:

- lesion sampling consumes Gaussian densities;
- lesion and tumour paths consume per-structure likelihoods;
- standard final segmentation consumes normalized posteriors;
- `GMM` is directly exported.

The EM mathematics does not require density-space operation. Main GMM
immediately multiplies densities by weights and priors and normalizes them at
[GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):123.
No available history documents a scientific or GEMS requirement for this
representation.

The deterministic stress input

```text
data        = [[1000,1000]]
means       = [[0,0], [1,1]]
covariances = [I,I]
weights     = [0.5,0.5]
```

produced:

```text
raw main-GMM densities = [0,0]
main-GMM posterior     = [[0,0]]
reported min NLL       = 36.043653389117
true log component terms = [-1000002.5310242471, -998003.5310242471]
log-sum-exp min NLL       = 998003.5310242471
```

The reported main value is `-log(float64 epsilon)`, not an approximation to
the likelihood. Current subregions computes a log likelihood but exponentiates
before normalization and uses float32 epsilon, giving an even lower ceiling of
`-log(float32 epsilon)=15.942385`.

MATLAB retains log terms and uses voxelwise maximum subtraction before
normalization and final reconstruction. The evidence therefore supports
log-space internals while retaining density-valued compatibility methods for
existing callers.

## Mixtures, grouping, and thalamus policy

Multiple components and weights are not an incompatibility with subregions;
they are useful established capabilities that current subregions omits. The
2023 final DTI paper further makes them part of the eventual scientific model,
not an optional convenience.

Main [io.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/io.py):5
preserves component counts, the production atlas requests three components for
`Soft`, and [GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):267
fits and normalizes per-class weights. The C++ GEMS boundary accepts full
matrices, weights, and counts.

That established main-GMM behavior is necessary but not sufficient to express
the entire final DTI model unchanged. Tregidgo et al. (2023) defines global
structural and diffusion components that may each contribute to multiple
classes, while each class has its own weights over those components. Current
GMM assigns a contiguous private component slice to each class. A structural
subfields++ configuration with one distribution per shared anatomical group
maps naturally to current GMM; the eventual many-to-many DTI mapping requires
either generalized incidence in GMM or a shared engine capable of representing
it.

MATLAB has separate structural, diffusion, reduced-class, component, and
weight mappings in
[TS_fnc_groupSorting.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_groupSorting.m):411.
Its generic multiple-component hyperparameter path uses k-means, but actual
tracked thalamus configs request one component for each atlas-defined group.
This generic structural helper should not be conflated with Section S.3 of the
2023 supplement. The final method performs robust configured structural-group
initialization and several structural-only GEM iterations first, then uses
k-means or deterministic k-MLE++ to initialize configured multi-component
diffusion distributions.

Python DTI reads `sharedGMMparameters.txt` with `split()[2:]` at
[thalamusDTI.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamusDTI.py):1002,
discarding the merged name and component-count columns. The base core then
passes unit weights and one Gaussian per class at
[core.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/core.py):684.
Current shipped files happen to request one component, so discarding the count
does not alter those configurations, but the parser silently disables the
general capability.

The `synthseg_kmeans` code in Python DTI is for missing/choroid-related initial
segmentation handling, not evidence of the thalamic medial/lateral Gaussian
split. The base thalamus split and MATLAB configurations are atlas anchored.

The publications support modality-wise or channel-wise robust medians and
anatomical group volume/counts as the general structural hyperparameters.
They also support region-specific operations such as mask erosion and
hippocampal partial-volume simulation. None of the supplied papers establishes
the T1-specific `+5/-5` offsets or a minimum `nHyper` floor as part of the
general statistical model. Such behavior belongs, if retained, in explicit
model configuration or a narrow region-policy operation rather than in GMM
mathematics.

The JSON dotted-function mechanism has limited documented intent as a way to
configure model-specific operations without copying the mesh framework. Its
current implementation is incomplete:

- the production path hardcodes a developer `/autofs/.../means_groupings.json`
  path at
  [thalamusDTI.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamusDTI.py):323;
- `post_em_update` is partially consumed;
- `second_hyps_hack` appears only in the test JSON and planning material;
- its configured module does not export the target function;
- `bimodal_thal_hack` calls `find_hyps_idx` without the required grouping
  argument at
  [utils.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/utils.py):107.

This supports retaining a familiar model-policy hook concept if later needed,
but not treating the current artifact as an interface or scientific contract.

## Low mass and covariance repair

Four policies are present:

| Path | Low-mass action |
|---|---|
| Main GMM | Always update; proper priors yield prior-dominated values, default zero mass yields NaNs |
| Legacy subregions | Below total mass `0.01`, reset to hypermean and variance 100 |
| New helper | Preserves the legacy reset, using `100 I` for full mode |
| MATLAB | Unless more than one responsibility exceeds epsilon, retain previous parameters |

These are statistically and behaviorally distinct. The MATLAB behavior has
explicit commits for disappearing and almost-zero classes. The legacy Python
fallback is old and shipped, but no evidence explains 100 as a scientifically
chosen covariance. Main GMM's default NaNs are a probable defect.

An NIW update is positive definite only when its scale or weighted scatter
provides full rank. The structural MATLAB defaults use an improper zero scale,
so positive definiteness is not guaranteed by construction for a degenerate
class. A safeguard is therefore required.

The helper's
[stabilize_covariance](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/gaussian.py):565
symmetrizes, tries Cholesky once, and adds escalating diagonal jitter only on
failure. It no longer uses eigendecomposition. The full update currently pays
for one Cholesky validation per fitted matrix even when already SPD; this code
is not integrated at `HEAD`, and channel counts in the target workflow are
small. MATLAB instead checks a determinant threshold and replaces a failing
matrix with an isotropic one. The existence of a safeguard is justified; its
trigger and fallback value are model/numerical policy, not settled scientific
contracts.

## GEMS handoff and posterior extraction

The C++ bridge at
[pyKvlCalculator.h](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/cxx/pyKvlCalculator.h):26
expects means `(G,d)`, covariance matrices `(G,d,d)`, weights `(G,)`, and
component counts `(C,)`. The GEMS likelihood filter evaluates full Gaussian
mixtures and can handle missing channels. No C++ change is implied by adding
subregions covariance support. This likelihood capability does not itself
implement the ECM parameter-estimation step used by the 2015 hippocampal
method; restoring incomplete-field-of-view fitting would still require an
explicit statistical-engine capability.

Main SAMSEG stores and passes the required full state directly. Current
subregions converts each variance vector to a diagonal matrix and supplies
unit weights/counts. Final subregions extraction recomputes the same diagonal
formula separately at
[core.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/core.py):739.

The current diagonal fit/extraction formulas are internally consistent as
likelihoods. The new full helper reaches neither path at `HEAD`. The abandoned
prototype did pass full matrices consistently, demonstrating no GEMS shape
barrier, but no completed full-mode workflow was found.

## Architectural options

### A. Subregions directly uses current `samseg.GMM.GMM`

Required change: adapt reduced classes/hyperparameters to component arrays and
store a GMM instance rather than vectors.

Advantages:

- removes duplicate likelihood/update code;
- immediately supports multiple components, weights, full matrices, diagonal
  projection, and established GEMS shapes;
- maps naturally to the component-count field in shared configuration.

Constraints requiring an adapter or targeted extension:

- current GMM defaults are not the MATLAB structural prior mapping;
- subfields++ needs an explicit safe low-mass policy;
- current GMM has no ECM update for channel-wise missing structural data;
- the existing prior-cost evaluator and density underflow are real but
  separable defects, not evidence for a second statistical owner;
- the final published DTI component-to-weight incidence model is not
  representable unchanged;
- changing existing public behavior in place affects many active callers.

Assessment: do not instantiate it entirely unchanged and assume every default
is suitable. It is nevertheless the strongest implementation base and intended
abstraction.

### B. Modestly extend `GMM` for subregions

Required capability for structural subfields++: a construction/adapter
boundary for dynamic group and component counts, explicit NIW parameter
mapping, safe low-mass behavior, and consistent use during fitting and
reconstruction. Missing-channel ECM is additionally required if incomplete
T1/T2 coverage is included in successor scope. Existing density methods and
mutable array semantics remain available to active callers. The prior-cost and
log-space issues can be corrected through separately reviewed additions rather
than requiring a broad main-SAMSEG rewrite first.

Required capability before final DTI integration: generalize the relationship
between classes and components so global components can be shared with
class-specific weights. This is a demonstrated target-model requirement, not
an optional extension inferred from the MATLAB code.

Advantages:

- `GMM` itself is already standalone; subregions need not inherit Samseg;
- the published SAMSEG model identifies this as the intended reusable
  multicontrast mixture abstraction;
- retains one user-facing general model with mixture support;
- shared parameter counts and dynamic anatomical groupings map naturally;
- avoids adding a new ownership layer without a demonstrated need.

Risks:

- its public mutable arrays and methods have many active dependencies;
- new policy branches must not become hard-coded thalamus behavior inside the
  general class;
- richer component incidence may exceed a modest extension.

Assessment: recommended leading hypothesis. Keep Gaussian/mixture mathematics
in `GMM`; place dynamic grouping, hyperparameter initialization, staged model
operations, and JSON-selected hacks in the subfields++ policy layer. Reassess
option C when designing the demonstrated global component-incidence or
missing-channel requirements, if implementing them cleanly would make `GMM`
an inappropriate policy and compatibility container.

### C. Neutral shared statistical engine with policy wrappers

Required capability: one general component engine for matrix Gaussian
likelihoods, log responsibilities, NIW updates/costs, mixture weights,
low-mass decisions, covariance stabilization, generalized class/component
incidence, and, if in scope, missing-channel sufficient statistics. Main GMM
and subregions keep their orchestration, anatomical grouping, hyperparameter
estimation, and compatibility APIs.

Advantages:

- eliminates duplicated mathematics without declaring current GMM defects to
  be contracts;
- supports mixtures/weights and can accommodate richer incidence mappings;
- preserves the main density API while allowing independent wrapper policy;
- keeps thalamus/hippocampus/brainstem decisions out of the statistical core;
- permits one full-matrix representation at the engine/GEMS boundary.

Risks:

- both wrappers require strong parity tests before delegating;
- established main behavior must be characterized before delegation;
- broader than simply wiring the current helper into core.

Assessment: valid alternative, but not uniquely required by the mathematics.
The component-mapping requirement is demonstrated; what remains undecided is
whether generalized `GMM` ownership or a neutral engine is the cleaner way to
meet it while preserving active callers. Do not introduce a neutral layer
merely to avoid designing and evaluating a narrow `GMM` extension.

### D. Permanently separate subregions Gaussian implementation

Advantages:

- smallest immediate integration surface;
- provides focused mathematical test oracles.

Costs:

- duplicates likelihood, covariance update, prior, normalization,
  stabilization, GEMS conversion, and extraction math;
- currently lacks mixtures and component weights;
- creates two places to fix the same underflow/low-mass/prior issues;
- no irreducible scientific difference was established.

Assessment: not justified as the permanent architecture. The helper can remain
a temporary test oracle while the `GMM` boundary is resolved, but code age,
API shape, and legacy storage are insufficient reasons for permanent
duplication. Legacy compatibility is already provided by the untouched
original model paths.

## Architectural recommendation

Adopt option B as the leading architectural direction for the structural
subfields++ stage: make the established `GMM` abstraction the single owner of
Gaussian-mixture mathematics and add the narrow construction or policy
boundary needed by subfields++. Do not directly integrate the helper-only
implementation into subregions EM as the permanent owner.

Before final DTI integration, explicitly compare a generalized option B with
option C against the now-demonstrated many-to-many component incidence. Option
C remains an implementation choice rather than a mathematically forced
result, but its trigger is no longer lack of evidence that richer mapping may
be needed. Use a neutral kernel layer if detailed design shows that `GMM`
cannot cleanly own that mapping, or missing-channel statistics where required,
while preserving its active interfaces.

The evidence supports these policy boundaries:

- `GMM` owns full matrix Gaussian math, its diagonal constraint, components,
  weights, parameter updates, and fitting/reconstruction likelihood semantics;
- subfields++ owns anatomical class grouping, atlas-derived or data-derived
  hyperparameter initialization, staged scheduling, and explicit
  region-specific hook operations;
- original thalamus, hippocampus, brainstem, and other legacy model paths
  remain untouched; subfields++ does not reproduce their scalar update as a
  compatibility mode;
- main SAMSEG retains its existing exported arrays and density-valued methods
  unless callers are deliberately migrated;
- GEMS always receives full covariance matrices, actual component counts, and
  actual weights;
- thalamus default grouping follows configured atlas classes; k-means is a
  component initializer only after configuration requests multiple components,
  and final DTI diffusion clustering occurs after structural-only GEM;
- global class/component incidence is an eventual DTI engine requirement,
  while shared anatomical grouping remains model configuration;
- channel-wise missing structural data requires an explicit ECM-capable fitting
  path if hippocampal T1/T2 parity is included in successor scope;
- the prior-cost discrepancy is investigated as a separate lower-priority
  convergence/reporting issue, with an output-impact probe before broad
  changes;
- the current `thalamusDTI` name is treated as transitional structural
  subfields++ work, not proof that the DTI/WMM stage is already implemented.

This is an architectural finding, not a migration plan. No implementation
sequence, branch mutation, or Stage 2 integration is proposed here.

## Coverage gaps and unresolved evidence

### Conclusive findings

- The published SAMSEG abstraction is a reusable multicontrast,
  full-covariance Gaussian mixture model with component weights and shared
  models across related anatomical structures.
- The 2015 hippocampal method exercised joint T1/T2 full-covariance subfields
  fitting and ECM for channel-wise missing T2 data.
- The published covariance updates form three distinct generations: 2015
  denominator `N`, 2019 denominator `N+1`, and final 2023 denominator
  `N+nu+d+2`.
- The final 2023 covariance update maps exactly to main GMM under
  `h=nu+d+2`, `H=Psi/h`; this equivalence does not extend to the older
  denominators.
- The final DTI model requires globally shared structural and diffusion
  components with class-specific weights and many-to-many incidence.
- Published thalamus structural grouping is anatomically configured; final DTI
  clustering initializes configured diffusion mixtures after structural-only
  GEM rather than creating the medial/lateral structural split.
- `SegmentThalamicNuclei.m` is the direct source family of the original Python
  subregions thalamus path; the scalar updates, fallbacks, class split, GEMS
  handoff, and extraction correspond directly.
- The maintainer-defined subfields++ boundary leaves original region models
  untouched, so the new backend has no requirement to preserve the legacy
  scalar update.
- Full Gaussian likelihood equivalence.
- Exact full-update parameter mapping and numerical equality.
- Legacy one-channel/diagonal divergence from NIW.
- Main diagonal mode as projection of the full update.
- Established main-GMM mixture and weight behavior.
- Four distinct low-mass policies and main default zero-mass NaNs.
- Extra parameter-dependent determinant term in main prior cost, with no
  demonstrated real-run segmentation impact.
- Prior-cost participation in inner/outer convergence and reporting, but not
  responsibilities, fixed-responsibility M-step algebra, or direct GEMS
  gradients.
- Density-space underflow and epsilon-capped false objectives.
- MATLAB proper-scale normalization sign discrepancy and its inactivity for
  the structural zero-scale call.
- Main GMM active dependency surface and subregions helper production non-use.
- No GEMS shape barrier to full covariance or mixtures.
- Actual MATLAB thalamus configs use one-component atlas groups, not automatic
  k-means splitting.
- Current `second_hyps_hack` is not an active functioning path.

### Historical inference or unresolved

- Pre-2023 FreeSurfer provenance and original statistical intent of `GMM.py`.
- Whether the legacy `0.01` perturbations were intended statistics or numerical
  guards.
- The exact `SegmentThalamicNuclei.m` revision used by the original Python
  porter.
- Which MATLAB model-selection configuration produced each validated or
  published output.
- Whether external tools interpret saved SAMSEG costs as comparable evidence.
- Whether the prior-cost discrepancy measurably changes iteration counts or
  segmentation outputs in a representative run.
- Whether generalized incidence belongs directly in an evolved `GMM` or in a
  neutral shared statistical engine.
- Whether missing-channel ECM is required in the first subfields++ release or
  only when restoring the published hippocampal T1/T2 workflow.
- Why the 2015, 2019, and 2023 covariance-prior formulations changed between
  publications; the equations are established but the transition rationale is
  not fully documented.
- The scientifically preferred low-mass policy and regularization threshold.
- Unknown private or external direct-Python workflows.

No automated test currently exercises nonzero cross-channel covariance through
main GMM or integrated subregions EM, bias-field fitting, extraction, and GEMS
handoff. Main tests use a one-contrast cube atlas with one component per class,
so they do not cover production mixture fitting. No integrated subregions fit
test exists. Published multichannel experiments establish scientific use but
do not replace current implementation tests for covariance, component
incidence, ECM, extraction, or GEMS transfer.

The local `test_samseg.py` suite could not be collected because TensorFlow is
absent and `SamsegLesion` imports `samseg/VAE.py`. This is a test-environment
gap, not a GEMS failure. The focused helper tests and source-level GEMS boundary
were available.

## Evidence index

Primary Python files:

- [GMM.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/GMM.py):8
- [Samseg.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/Samseg.py):840
- [BiasField.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/BiasField.py):137
- [ProbabilisticAtlas.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/ProbabilisticAtlas.py):119
- [subregions/core.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/core.py):468
- [subregions/gaussian.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/gaussian.py):356
- [subregions/thalamus.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamus.py):218
- [subregions/thalamusDTI.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/subregions/thalamusDTI.py):21
- [test_subregions_gaussian.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/tests/test_subregions_gaussian.py):1
- [test_samseg.py](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/tests/test_samseg.py):205
- [pyKvlCalculator.h](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/samseg/cxx/pyKvlCalculator.h):26
- [kvlGMMLikelihoodImageFilter.hxx](/Users/henrytregidgo/PycharmProjects/Samseg/samseg/gems/kvlGMMLikelihoodImageFilter.hxx):33

Primary MATLAB files:

- [SegmentThalamicNuclei.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/SegmentThalamicNuclei.m):1
- [TS_fnc_fitGaussian_withprior.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_fitGaussian_withprior.m):61
- [TS_fnc_gaussprior_NIW.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_gaussprior_NIW.m):66
- [TS_fnc_gaussianPDF.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_gaussianPDF.m):65
- [TS_fnc_fitGaussianHyperParams.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_fitGaussianHyperParams.m):108
- [TS_fnc_groupSorting.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_groupSorting.m):411
- [TS_fnc_getWeightSpacePosteriors.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_getWeightSpacePosteriors.m):112
- [TS_fnc_thalamus_seg_gem_joint.m](/Users/henrytregidgo/PycharmProjects/ThalamusMatlab/ThalamusSeg/HTtestfunctions/TS_fnc_thalamus_seg_gem_joint.m):1207

Primary papers:

- [Puonti et al., NeuroImage 2016](/Users/henrytregidgo/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/ReadingMaterials/2016_NI_Puonti_samseg.pdf),
  especially printed pages 238-239 (PDF pages 4-5)
- [Iglesias et al., NeuroImage 2015](/Users/henrytregidgo/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Documents/ReadingMaterials/2015_NeuroImage_Iglesias_HippocampalSubfields.pdf),
  especially PDF pages 16-20, 25, and 33-34
- [Iglesias et al., NeuroImage 2018](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2018_NeuroImage_Iglessias_thalamicAtlas.pdf),
  especially PDF pages 7 and 9-10
- [Iglesias et al., IPMI 2019](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2019_IPMI_Iglesias_diffusionMRBayesianSeg.pdf),
  especially PDF pages 4 and 6-8
- [Tregidgo et al., NeuroImage 2023](/Users/henrytregidgo/Documents/UCLDocuments/pdfsForSorting/ReadingMaterials/relatedToProject/2023_NeuroImage_Tergidgo_thalamusBayesian.pdf),
  especially PDF pages 4 and 6-8
- [Tregidgo et al. 2023 supplementary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10636587/),
  especially Sections S.1.1 and S.3

Provenance and callers were gathered with bounded uses of:

```bash
git log --all --follow -- <path>
git log --all -S'<term>' -- <path>
git log --all -G'<regex>' -- <path>
git blame -L <start>,<end> <path>
git show <sha>:<path>
git branch --all --contains <sha>
git diff --no-index <artifact-a> <artifact-b>
git rev-parse <ref>:<path>
rg -n '<caller-or-config-pattern>' <audited-roots>
shasum <installed-artifacts>
```

No relevant deleted GMM, Gaussian helper, grouping, hook, or MATLAB
implementation was found with `git log --all --diff-filter=D`. That absence is
bounded to reachable audited history.

## Verification record

Executed successfully:

```bash
.venv/bin/python -m py_compile \
  samseg/subregions/gaussian.py samseg/subregions/core.py samseg/GMM.py

.venv/bin/python -m pytest -q samseg/tests/test_subregions_gaussian.py
# 20 passed
```

Attempted:

```bash
.venv/bin/python -m pytest --collect-only -q samseg/tests/test_samseg.py
# collection error: ModuleNotFoundError: No module named 'tensorflow'
```

Final repository checks:

```text
git status --short --branch
## HT-subregions-integration...origin/rectify [ahead 2]
?? samseg/subregions/for_testing/gaussian_gmm_audit.md

git diff --stat
<empty>
```

The audit created only this durable report. It did not modify production code,
tests, commits, or refs. The primary-source revision was documentation-only, so
the previously recorded code tests were not rerun.
