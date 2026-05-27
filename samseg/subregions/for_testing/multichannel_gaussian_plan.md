# Subregions Multi-Channel Gaussian Plan

This note records the current state of the DTI/multi-channel Gaussian work and
the expected shape of the fix. It is intentionally a planning document, not a
production design.

## Current state

The current Python subregions path supports multi-channel image data, but it
models the channels as conditionally independent given the class label.

In `samseg/subregions/core.py`, the fitted Gaussian parameters are stored as:

- `self.means`: one mean vector per class, shape `(n_classes, n_channels)`.
- `self.variances`: one variance vector per class, shape
  `(n_classes, n_channels)`.

The E-step likelihood is currently computed as a sum of independent univariate
Gaussian log likelihoods:

```text
log p(x_i | c) =
  -0.5 * sum_d [ (x_id - mu_cd)^2 / sigma_cd^2
                 + log(2*pi*sigma_cd^2) ]
```

That happens in both the fitting loop and final segmentation extraction:

- `core.py`: EM likelihood inside `fit_mesh_to_image`.
- `core.py`: posterior reconstruction inside `extract_segmentation`.

The M-step also updates only per-channel variances:

```text
mu_c =
  (n_c0 * mu_c0 + sum_i gamma_ic * x_i)
  / (n_c0 + sum_i gamma_ic + eps)

sigma_cd^2 =
  (sum_i gamma_ic * (x_id - mu_cd)^2
   + n_c0 * (mu_cd - mu_c0d)^2)
  / (sum_i gamma_ic + eps)
```

For mesh deformation, the Python path expands the per-channel variance vector
into a diagonal covariance matrix before calling the C++ calculator:

```python
full_variance[i] = np.diag(self.variances[i])
```

That means the current implementation is mathematically consistent only for a
diagonal covariance model. It is not a full multivariate Gaussian model.

## Existing support to keep

The diagonal/independent-channel behavior is still useful and should remain
available behind an explicit option, for example:

```text
covariance_mode = "diagonal"
```

or:

```text
independent_channels = True
```

This mode gives a simpler model, easier diagnostics, and a useful regression
target while implementing full covariance.

## Required target

The full multi-channel model should allow one covariance matrix per class:

```text
Sigma_c in R^(D x D)
```

where `D` is the number of channels. The log likelihood should become:

```text
log p(x_i | c) =
  -0.5 * [ (x_i - mu_c)^T Sigma_c^-1 (x_i - mu_c)
           + log det(Sigma_c)
           + D * log(2*pi) ]
```

The M-step needs to estimate the full weighted covariance:

```text
Sigma_c =
  [ sum_i gamma_ic * (x_i - mu_c)(x_i - mu_c)^T + prior_terms ]
  / [ sum_i gamma_ic + prior_weight ]
```

The existing `meanHyper`/`nHyper` prior logic is vector/diagonal-oriented. A
full covariance implementation therefore needs an explicit decision about the
covariance prior or shrinkage strategy. A normal-inverse-Wishart style prior is
the conventional full-covariance equivalent, but a simpler shrinkage-to-diagonal
regularizer may be enough for the local DTI migration if it is documented and
tested.

## Implementation shape

1. Add an explicit covariance-mode configuration path.

   The default should preserve current behavior until full covariance has been
   tested. A likely interface is:

   ```text
   covariance_mode = "diagonal" | "full"
   ```

2. Introduce shared Gaussian helpers in the Python subregions code.

   The fitting loop and `extract_segmentation` should call the same likelihood
   helper instead of duplicating the formula. The helper should support both
   diagonal vectors and full covariance matrices.

3. Normalize parameter storage.

   Keep diagonal mode simple, but define clear shapes:

   ```text
   diagonal: self.variances shape = (classes, channels)
   full:     self.variances shape = (classes, channels, channels)
   ```

   Longer term, renaming `variances` to `covariances` would be clearer, but that
   should be handled carefully because the C++/GEMS interfaces also use the name
   `variances` for covariance matrices.

4. Update initialization and the M-step.

   Diagonal mode should reproduce the current formulas. Full mode should compute
   weighted covariance matrices and apply a positive-definite regularizer. The
   implementation must handle disappearing classes without producing singular
   matrices.

5. Pass full covariance matrices into mesh deformation.

   The C++ GEMS likelihood/filter path already accepts covariance matrices in
   several interfaces, and `kvlGMMLikelihoodImageFilter` computes inverses and
   determinants from those matrices. The Python binding and the active
   calculator path still need to be verified with a small full-covariance test.

6. Extend validation.

   Minimum useful checks:

   - Single-channel behavior is unchanged.
   - Diagonal mode reproduces the current local smoke-test outputs within a
     small tolerance.
   - Full mode matches a manual or SciPy multivariate Gaussian likelihood on a
     small synthetic two-channel example with non-zero off-diagonal covariance.
   - Full mode rejects or regularizes singular covariance estimates.
   - A local DTI smoke test completes with `covariance_mode = "full"`.

## Open questions

- What should the full covariance prior be?
- Should full covariance be enabled only for DTI/multi-channel runs, or exposed
  generally through the subregions model configuration?
- How much output parity with MATLAB is needed before merging?
- Does the MATLAB DTI path use a full covariance prior, a fixed covariance, or
  an empirical covariance update that should be copied directly?
- Should covariance estimation happen for all classes, or only for classes with
  DTI-specific groupings?

## Related audit material

The broader MATLAB-to-Python migration audit is currently on the
`docs/dti-migration-checklists` branch, not this branch. The most relevant
starting points there are:

- `docs/dev/migration/thalamus-gap-analysis.md`
- `docs/dev/migration/mapping_v2/00-index.md`
- `docs/dev/migration/mapping_v2/compact/20-gap-action-summary.md`

Those documents were created against older branch tips, so they should be
refreshed against the current `dti_integration` tip before being treated as the
final migration state.
