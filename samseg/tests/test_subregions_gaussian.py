import importlib.util
from pathlib import Path

import numpy as np
import pytest


GAUSSIAN_PATH = Path(__file__).resolve().parents[1] / "subregions" / "gaussian.py"
SPEC = importlib.util.spec_from_file_location("subregions_gaussian", GAUSSIAN_PATH)
gaussian = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gaussian)


def test_validate_covariance_mode():
    assert gaussian.validate_covariance_mode("diagonal") == "diagonal"
    assert gaussian.validate_covariance_mode("full") == "full"
    with pytest.raises(ValueError):
        gaussian.validate_covariance_mode("bogus")


def test_diagonal_log_likelihood_matches_manual():
    data = np.array([[0.0, 1.0], [1.0, 2.0]])
    mean = np.array([0.0, 1.0])
    variances = np.array([1.0, 4.0])
    got = gaussian.diagonal_gaussian_log_likelihood(data, mean, variances)
    expected = np.array([
        -0.5 * (0.0 + np.log(2 * np.pi * 1.0) + 0.0 + np.log(2 * np.pi * 4.0)),
        -0.5 * (1.0 + np.log(2 * np.pi * 1.0) + 0.25 + np.log(2 * np.pi * 4.0)),
    ])
    np.testing.assert_allclose(got, expected)


def test_full_log_likelihood_matches_manual():
    data = np.array([[1.0, 2.0]])
    mean = np.array([0.5, 1.5])
    covariance = np.array([[2.0, 0.25], [0.25, 1.5]])
    got = gaussian.full_gaussian_log_likelihood(data, mean, covariance)
    delta = data - mean
    expected = -0.5 * (
        delta @ np.linalg.solve(covariance, delta.T)
        + np.linalg.slogdet(covariance)[1]
        + 2 * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(got, expected.ravel())


def test_diagonal_posterior_update_matches_current_formula():
    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    posterior = np.array([0.25, 0.75])
    mean_hyper = np.array([1.0, 2.0])
    mu, variance = gaussian.diagonal_posterior_update(data, posterior, mean_hyper, 10.0)
    np.testing.assert_allclose(mu, np.array([1.044505, 2.04359673]), rtol=1e-6)
    assert np.all(variance > 0)


def test_full_covariance_posterior_update_and_mean_prior_cost_are_pd_safe():
    data = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 3.0]])
    posterior = np.array([0.2, 0.3, 0.5])
    mean_hyper = np.array([0.5, 1.0])
    mu, covariance = gaussian.full_covariance_posterior_update(data, posterior, mean_hyper, 10.0)
    assert covariance.shape == (2, 2)
    np.testing.assert_allclose(covariance, covariance.T)
    np.linalg.cholesky(covariance)
    prior_cost = gaussian.full_mean_prior_cost(mu, covariance, mean_hyper, 10.0)
    assert np.isfinite(prior_cost)


def test_full_covariance_posterior_update_low_mass_returns_broad_identity():
    data = np.array([[0.0, 0.0], [2.0, 1.0]])
    posterior = np.array([0.0, 0.0])
    mean_hyper = np.array([0.5, 1.0])
    mu, covariance = gaussian.full_covariance_posterior_update(data, posterior, mean_hyper, 10.0)
    np.testing.assert_allclose(mu, mean_hyper)
    np.testing.assert_allclose(covariance, np.eye(2) * 100.0)


def test_full_mean_prior_cost_matches_diagonal_one_dimensional_formula():
    mean = np.array([2.0])
    mean_hyper = np.array([1.5])
    covariance = np.array([[4.0]])
    n_hyper = 10.0
    got = gaussian.full_mean_prior_cost(mean, covariance, mean_hyper, n_hyper)
    expected = (
        0.5 * np.log(2 * np.pi * covariance[0, 0])
        - 0.5 * np.log(n_hyper)
        + 0.5 * (n_hyper / covariance[0, 0]) * (mean[0] - mean_hyper[0]) ** 2
    )
    np.testing.assert_allclose(got, expected)


def test_repair_covariance_eigh_repairs_singular_matrix():
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    repaired = gaussian.repair_covariance_eigh(covariance)
    np.linalg.cholesky(repaired)


def test_full_log_likelihood_rejects_singular_covariance():
    data = np.array([[1.0, 2.0]])
    mean = np.array([0.5, 1.5])
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        gaussian.full_gaussian_log_likelihood(data, mean, covariance)


def test_covariance_for_gems_shapes():
    diag = np.array([1.0, 2.0])
    full = np.array([[1.0, 0.1], [0.1, 2.0]])
    np.testing.assert_allclose(gaussian.covariance_for_gems(diag, "diagonal"), np.diag(diag))
    np.testing.assert_allclose(
        gaussian.covariance_for_gems(diag, "full"),
        np.diag(diag),
    )
    np.testing.assert_allclose(
        gaussian.covariance_for_gems(full, "full"),
        full,
    )


def test_covariance_for_gems_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        gaussian.covariance_for_gems(np.zeros((2, 3)), "full")
    with pytest.raises(ValueError):
        gaussian.covariance_for_gems(np.zeros((2, 2, 2)), "full")
    with pytest.raises(ValueError):
        gaussian.covariance_for_gems(np.zeros((2, 3)), "diagonal")
