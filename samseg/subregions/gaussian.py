import numpy as np


def validate_covariance_mode(covariance_mode):
    """
    Validate the covariance parameterization used by the subregions model.

    Parameters
    ----------
    covariance_mode : str
        Requested covariance representation. Must be either ``"diagonal"``
        for per-channel variances or ``"full"`` for multivariate covariance
        matrices.

    Returns
    -------
    str
        The validated covariance mode string.

    Raises
    ------
    ValueError
        If ``covariance_mode`` is not one of the supported values.

    Subfields Useage
    ----------------
    This keeps the covariance-mode choice consistent before the EM loop,
    posterior extraction, and GEMS bridge all consume it.
    """
    if covariance_mode not in ("diagonal", "full"):
        raise ValueError(
            "covariance_mode must be one of: 'diagonal', 'full'"
        )
    return covariance_mode


def repair_covariance_eigh(covariance, min_eigenvalue=1e-6):
    """
    Repair a covariance matrix with an eigenvalue floor.

    Parameters
    ----------
    covariance : array_like
        Input covariance matrix to repair.
    min_eigenvalue : float, optional
        Absolute lower bound applied to each eigenvalue after symmetrization.

    Returns
    -------
    ndarray
        Symmetric positive-definite covariance matrix.

    Subfields Useage
    ----------------
    This is an expensive last-resort repair for external or diagnostic
    covariance inputs. The normal subfields EM path should create SPD
    covariances by construction and should not call this in the fitting loop.
    """
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    scale = np.max(np.abs(eigvals)) if eigvals.size else 1.0
    floor = max(min_eigenvalue, np.finfo(float).eps * scale)
    eigvals = np.maximum(eigvals, floor)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return 0.5 * (repaired + repaired.T)


def covariance_for_gems(covariance, covariance_mode):
    """
    Convert a covariance representation into the matrix form expected by GEMS.

    Parameters
    ----------
    covariance : array_like
        Diagonal vector or full covariance matrix for a single class.
    covariance_mode : {"diagonal", "full"}
        Covariance representation used by the Python EM path.

    Returns
    -------
    ndarray
        A covariance matrix suitable for the GEMS cost/gradient calculator.

    Raises
    ------
    ValueError
        If the input shape is incompatible with the selected mode.

    Subfields Useage
    ----------------
    The Python fitting loop stores class covariances in the mode-specific
    internal form, but the mesh deformation bridge always needs matrices.
    """
    validate_covariance_mode(covariance_mode)
    covariance = np.asarray(covariance, dtype=float)
    if covariance_mode == "diagonal":
        if covariance.ndim == 1:
            return np.diag(covariance)
        if covariance.ndim == 2:
            if covariance.shape[0] != covariance.shape[1]:
                raise ValueError("Diagonal covariance matrix must be square")
            return np.diag(np.diag(covariance))
        raise ValueError("Diagonal covariance expects a vector or matrix")
    if covariance.ndim == 1:
        return np.diag(covariance)
    if covariance.ndim != 2:
        raise ValueError("Full covariance expects a vector or matrix")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Full covariance matrix must be square")
    return covariance


def diagonal_gaussian_log_likelihood(data, mean, variances):
    """
    Evaluate a diagonal multivariate Gaussian log likelihood.

    Parameters
    ----------
    data : array_like
        Sample matrix with shape ``(n_samples, n_channels)``.
    mean : array_like
        Mean vector with shape ``(n_channels,)``.
    variances : array_like
        Per-channel variances with shape ``(n_channels,)``.

    Returns
    -------
    ndarray
        Log likelihood for each sample.

    Subfields Useage
    ----------------
    This is the current subregions likelihood used by both the EM update and
    final posterior reconstruction when the model stays in diagonal mode.
    """
    data = np.asarray(data, dtype=float)
    mean = np.asarray(mean, dtype=float)
    variances = np.asarray(variances, dtype=float)
    return -0.5 * np.sum(
        ((data - mean) ** 2) / variances + np.log(2 * np.pi * variances),
        axis=-1,
    )


def full_gaussian_log_likelihood(data, mean, covariance):
    """
    Evaluate a full-covariance multivariate Gaussian log likelihood.

    Parameters
    ----------
    data : array_like
        Sample matrix with shape ``(n_samples, n_channels)``.
    mean : array_like
        Mean vector with shape ``(n_channels,)``.
    covariance : array_like
        Symmetric covariance matrix with shape ``(n_channels, n_channels)``.

    Returns
    -------
    ndarray
        Log likelihood for each sample.

    Subfields Useage
    ----------------
    Full covariance mode uses this in the same places as the diagonal helper
    so the EM path and the segmentation posterior use one likelihood formula.
    """
    data = np.asarray(data, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    delta = data - mean
    chol = np.linalg.cholesky(covariance)
    logdet = 2 * np.sum(np.log(np.diag(chol)))
    solved = np.linalg.solve(chol, delta.T).T
    quadratic = np.sum(solved * solved, axis=-1)
    dim = covariance.shape[0]
    return -0.5 * (quadratic + logdet + dim * np.log(2 * np.pi))


def diagonal_posterior_update(data, posterior, mean_hyper, n_hyper, thresh=1e-2):
    """
    Update diagonal Gaussian parameters from weighted posteriors.

    Parameters
    ----------
    data : array_like
        Sample matrix with shape ``(n_samples, n_channels)``.
    posterior : array_like
        Responsibility weights for the class.
    mean_hyper : array_like
        Prior mean for the class.
    n_hyper : float
        Prior strength associated with the mean hyperparameter.
    thresh : float, optional
        Minimum posterior mass before falling back to a default covariance.

    Returns
    -------
    mean : ndarray
        Updated mean vector.
    variances : ndarray
        Updated per-channel variances.

    Subfields Useage
    ----------------
    The diagonal M-step in `core.py` uses this to preserve the existing
    per-channel update rule while keeping the code in one shared helper.
    """
    data = np.asarray(data, dtype=float)
    posterior = np.asarray(posterior, dtype=float)
    mean_hyper = np.asarray(mean_hyper, dtype=float)
    n_hyper = float(n_hyper)
    total = float(np.sum(posterior))
    if total <= thresh:
        return mean_hyper.copy(), np.full(data.shape[1], 100.0)
    mu = (mean_hyper * n_hyper + data.T @ posterior) / (n_hyper + total + thresh)
    variance = (((data - mu) ** 2).T @ posterior + n_hyper * (mu - mean_hyper) ** 2) / (total + thresh)
    variance = np.maximum(variance + thresh, thresh)
    return mu, variance


def full_mean_prior_cost(mean, covariance, mean_hyper, n_hyper):
    """
    Evaluate the full-covariance mean-prior cost term.

    Parameters
    ----------
    mean : array_like
        Current class mean vector.
    covariance : array_like
        Current class covariance matrix.
    mean_hyper : array_like
        Prior mean vector.
    n_hyper : float
        Prior strength associated with the mean.

    Returns
    -------
    float
        Scalar negative log-prior contribution for the current class mean.

    Subfields Useage
    ----------------
    This provides the full-covariance analogue of the existing diagonal
    mean-prior cost used in the subregions EM objective.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    mean_hyper = np.asarray(mean_hyper, dtype=float)
    n_hyper = float(n_hyper)
    dim = mean.shape[0]
    chol = np.linalg.cholesky(covariance)
    logdet = 2 * np.sum(np.log(np.diag(chol)))
    delta = mean - mean_hyper
    solved = np.linalg.solve(chol, delta)
    return 0.5 * (
        n_hyper * solved @ solved
        + logdet
        + dim * np.log(2 * np.pi)
        - dim * np.log(n_hyper)
    )


def full_covariance_posterior_update(data, posterior, mean_hyper, n_hyper, thresh=1e-2):
    """
    Update full covariance Gaussian parameters with a weighted ridge step.

    Parameters
    ----------
    data : array_like
        Sample matrix with shape ``(n_samples, n_channels)``.
    posterior : array_like
        Responsibility weights for the class.
    mean_hyper : array_like
        Prior mean vector.
    n_hyper : float
        Prior strength associated with the mean hyperparameter.
    thresh : float, optional
        Minimum posterior mass before falling back to a broad covariance.

    Returns
    -------
    mean : ndarray
        Updated mean vector.
    covariance : ndarray
        Updated full covariance matrix.

    Subfields Useage
    ----------------
    Full covariance mode uses this M-step helper so `core.py` can estimate
    multichannel class covariances without duplicating the weighted scatter
    and mean-prior terms.
    """
    data = np.asarray(data, dtype=float)
    posterior = np.asarray(posterior, dtype=float)
    mean_hyper = np.asarray(mean_hyper, dtype=float)
    n_hyper = float(n_hyper)
    total = float(np.sum(posterior))
    dim = data.shape[1]
    if total <= thresh:
        return mean_hyper.copy(), np.eye(dim) * 100.0
    mu = (mean_hyper * n_hyper + data.T @ posterior) / (n_hyper + total + thresh)
    centered = data - mu
    covariance = (centered.T * posterior) @ centered
    covariance += n_hyper * np.outer(mu - mean_hyper, mu - mean_hyper)
    covariance /= (total + thresh)
    covariance = covariance + thresh * np.eye(dim)
    covariance = 0.5 * (covariance + covariance.T)
    return mu, covariance
