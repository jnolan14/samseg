# Structural Subfields++ Successor Context

Date: 2026-08-13

Status: durable design context at the Gaussian-investigation closure
checkpoint. This note records evidence and current direction before
MeshModelPlus implementation. It is not an API specification or an accepted
statistical-policy document.

Related records:

- gaussian_gmm_audit.md contains the architectural, historical, usage, and
  mathematical audit.
- gmm_subfields_replacement_review.md records the narrower structural
  replacement boundary and the retired GMM.py prototype.
- multichannel_gaussian_plan.md is a superseded historical implementation
  plan and must not be read as current requirements.

## Epistemic Labels

The sections below deliberately distinguish four kinds of statement:

- **Repository-established fact:** directly supported by inspected source,
  history, tests, artifacts, or numerical probes.
- **Reported team direction:** context reported from a discussion with Jackson;
  relevant evidence of intent, but not proof of current behavior.
- **Current design preference:** the leading interpretation after review, still
  subject to implementation evidence.
- **Unresolved question:** a decision or validation result not established by
  current evidence.

## Why A Successor Is Being Considered

**Repository-established fact:** legacy MeshModel combines the ROI lifecycle
with a manually coded Gaussian fitting path. Main SAMSEG instead gives an
authoritative GMM object responsibility for multicontrast Gaussian mixtures,
including components, covariance matrices, mixture weights, fitting, and
structure reconstruction.

**Repository-established fact:** the subregions path corresponds closely to
the original structural thalamus MATLAB implementation. The audit establishes
duplication and simplification of available GMM mathematics, but it does not
establish the original developer's motivation or level of understanding.

**Current design preference:** structural subfields++ should correct that
ownership boundary by delegating structural mixture mathematics to the
established GMM. This does not require changing the legacy thalamus,
hippocampus, brainstem, or other existing subregions implementations.

## Successor Relationship

**Reported team direction:** retain the existing MeshModel hierarchy as an
isolated behavioral reference while creating a separate MeshModelPlus
successor. Begin from a controlled copy of a known MeshModel revision because
the ownership changes affect the lifecycle broadly rather than adding one leaf
behavior.

**Current design preference:** prefer an independent successor over
MeshModelPlus(MeshModel). Inheriting while overriding most stateful lifecycle
methods risks retaining an implicit dependency on legacy self.means and
vector-shaped self.variances.

**Current design preference:** keep the initial copy recognizably attributable
to its source revision. Divergences should be narrow and reviewable; the copy is
not an opportunity for unrelated cleanup.

**Reported team direction:** the duplication is intended to have a migration
lifecycle, not necessarily permanent ownership:

    legacy MeshModel      -> isolated behavioral reference
    MeshModelPlus         -> replacement candidate
    compare and validate  -> migrate suitable region children
    proven successor      -> possible future canonical MeshModel
    legacy path           -> potentially removable

**Unresolved question:** source evidence does not uniquely determine whether a
literal copy, selective extraction, or another independent construction will
best preserve attribution. That should be decided at the implementation
boundary, without modifying legacy classes.

## Intended Ownership Boundaries

### MeshModelPlus

**Current design preference:** the successor base owns the shared structural
ROI lifecycle and the authoritative structural self.gmm. It orchestrates
preparation, fitting, mesh deformation interaction, GEMS handoff, and final
reconstruction without owning duplicate Gaussian equations.

**Reported team direction:** major lifetime state and model objects should be
declared deliberately rather than appearing opportunistically across methods.
This does not imply that all data-dependent state must be calculated in
__init__.

### Region Children

**Repository-established fact:** current region classes contain substantial
differences in preprocessing, atlas handling, hyperparameter preparation,
postprocessing, and output behavior.

**Current design preference:** substantial anatomical behavior remains in
region children. A thalamus successor can validate the shared structural path
without forcing hippocampal, brainstem, or DTI-specific algorithms into the
base.

### Structural GMM

**Repository-established fact:** established GMM already supports multiple
contrasts, full and diagonal covariance matrices, multiple components per
class, estimated mixture weights, GEMS-compatible parameter arrays, and
structure reconstruction.

**Current design preference:** use that established interface unchanged for the
first structural replacement. Convert thalamus conventional (nu, Psi) prior
parameters privately at the adapter boundary into native GMM (h, H) values.

**Reported team direction:** shared GMM parameter files specify the classes,
component counts, and structure-to-class mapping. Mixture weights are estimated
by GMM during fitting; configured initial or fixed weight values are not a
requirement of this work.

### Configuration And Registered Operations

**Repository-established fact:** the current JSON mechanism contains both
ordinary policy data and dotted callable selections. It is partly wired and
does not currently form a general callback framework.

**Reported team direction:** preserve its familiar shape. It was intended both
to externalize constants/model choices inherited from MATLAB-derived code and
to select narrow executable adjustments where values alone are insufficient.

**Current design preference:** configuration and registered operations remain
outside GMM. Moving Gaussian state into self.gmm must not bypass a currently
used operation, but statistical kernels should not be expressed as hooks.

**Unresolved question:** some current hooks appear unused or non-executable.
Their exact preservation status must be based on demonstrated use and
method-level review, not on either their presence or their awkwardness.

### Future Likelihood Models

**Reported team direction:** a future thalamus DTI successor may own an
additional joint structural/diffusion mixture model and explicit additional
initialization/fitting phases.

**Repository-established fact:** the current ThalamicNucleiDTI class does not
yet implement that authoritative joint model, and current GMM does not
represent the final many-to-many DTI component-incidence model unchanged.

**Current design preference:** future genuinely different likelihoods should be
additional model objects rather than reasons to widen structural GMM
indefinitely. Names and APIs remain schematic.

## Lifecycle Evidence To Preserve

A GMM-backed successor is incomplete if it changes only the E-step or M-step.
The implementation comparison must trace:

- GMM construction and class/component configuration;
- atlas-derived hyperparameters and initial state;
- mixture weights and covariance-mode selection;
- responsibilities and parameter updates;
- state consumed during mesh deformation;
- transfer of means, matrix covariances, and weights to GEMS;
- final structure likelihoods and posteriors from the same GMM state;
- volumes, labels, rasterization, and region-specific postprocessing;
- registered configuration/operation phases; and
- any direct readers of legacy self.means or self.variances.

**Repository-established fact:** no demonstrated active dependency requires
vector-shaped variance storage in the successor. Legacy paths retain their
existing storage unchanged.

## Comparison And Acceptance

**Current design preference:** compare successor behavior with the legacy
implementation phase by phase.

For phases intended to remain unchanged, exact or very close equivalence may be
appropriate. For fitted parameters, posteriors, segmentations, and volumes,
differences remain important evidence but are investigated rather than treated
automatically as failures. The legacy implementation is an empirical reference,
not an unquestioned statistical contract.

Three checkpoints should remain distinct:

1. **Architectural construction:** the isolated successor exists with clear
   ownership and no legacy-path mutation.
2. **Executable integration:** a structural thalamus path runs end to end and
   exposes phase-level comparisons.
3. **Statistical acceptance:** unresolved policies and output differences have
   been reviewed and acceptance criteria met.

## Unresolved Questions

The following are intentionally not settled by this checkpoint:

- exact zero-support behavior under the intended zero-scale covariance prior;
- whether any positive low-mass threshold is scientifically justified;
- mixture-weight policy if only some components are unsupported;
- inner EM convergence and objective semantics;
- covariance-mode configuration and the structural default;
- channel-aware initialization and unsupported-class initialization;
- provenance and intended role of the resolution-level Gaussian-state reset;
- which current JSON operations are supported, prototype-only, or defective;
- final structural-thalamus smoke artifacts and acceptance tolerances;
- hippocampal missing-channel ECM requirements;
- final DTI shared-component incidence and joint-model ownership; and
- whether stable log-domain normalization should later replace canonical GMM
  internals in a separate focused contribution.

The exact zero-support and convergence questions affect eventual statistical
acceptance. They do not prevent construction and controlled exercise of the
isolated successor, provided no production-readiness claim is made.

## Checkpoint Boundary

This note supports a clean pre-architecture checkpoint only. That checkpoint:

- leaves samseg/GMM.py at its established implementation;
- retires the temporary parallel Gaussian helper;
- preserves focused evidence that existing GMM covers the required structural
  mathematics;
- keeps exploratory prototype code outside tracked project policy; and
- introduces no MeshModelPlus, CLI, JSON, or legacy lifecycle change.
