import numpy as np

from samseg.GMM import GMM, eps


def _normal_density(data, mean, covariance):
    difference = data - mean
    exponent = np.sum(
        difference * np.linalg.solve(covariance, difference.T).T, axis=1)
    determinant = np.linalg.det(covariance)
    scale = np.sqrt((2 * np.pi) ** data.shape[1] * determinant)
    return np.exp(-0.5 * exponent) / scale


def _make_single_gaussian_gmm(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h,
        diagonal=False):
    gmm = GMM(
        [1],
        data.shape[1],
        useDiagonalCovarianceMatrices=diagonal,
        initialMeans=np.zeros((1, data.shape[1])),
        initialVariances=np.eye(data.shape[1])[None, ...],
        initialMixtureWeights=np.ones(1),
        initialHyperMeans=hyper_mean[None, :],
        initialHyperMeansNumberOfMeasurements=np.array([mean_strength]),
        initialHyperVariances=hyper_variance[None, ...],
        initialHyperVariancesNumberOfMeasurements=np.array([h]),
    )
    gmm.fitGMMParameters(data, posterior[:, None])
    return gmm


def _expected_update(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h):
    mass = np.sum(posterior)
    mean = (
        data.T @ posterior + mean_strength * hyper_mean
    ) / (mass + mean_strength)
    difference = data - mean
    hyper_difference = mean - hyper_mean
    covariance = (
        difference.T @ (difference * posterior[:, None])
        + mean_strength * np.outer(hyper_difference, hyper_difference)
        + h * hyper_variance
    ) / (mass + h)
    return mean, covariance


def test_full_covariance_likelihood_matches_independent_reference():
    data = np.array([[1.0, 2.0], [-0.5, 1.0], [2.5, -1.0]])
    mean = np.array([0.5, 1.5])
    covariance = np.array([[2.0, 0.25], [0.25, 1.5]])
    gmm = GMM([1], 2, useDiagonalCovarianceMatrices=False)

    result = gmm.getGaussianLikelihoods(
        data, mean[:, None], covariance)

    np.testing.assert_allclose(
        result, _normal_density(data, mean, covariance), rtol=1e-14)


def test_one_channel_h_H_prior_update_matches_direct_equations():
    data = np.array([[0.0], [2.0], [5.0], [8.0]])
    posterior = np.array([0.2, 0.4, 0.8, 0.3])
    hyper_mean = np.array([1.5])
    mean_strength = 2.0
    h = 5.0
    hyper_variance = np.array([[0.14]])

    expected_mean, expected_covariance = _expected_update(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h)
    gmm = _make_single_gaussian_gmm(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h)

    np.testing.assert_allclose(gmm.means[0], expected_mean)
    np.testing.assert_allclose(gmm.variances[0], expected_covariance)


def test_two_channel_h_H_prior_update_matches_direct_equations():
    data = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    posterior = np.ones(3)
    hyper_mean = np.array([1.0, 1.0])
    mean_strength = 1.0
    h = 7.0
    hyper_variance = np.array([[0.1, 1.0 / 70.0], [1.0 / 70.0, 0.5 / 7.0]])

    expected_mean, expected_covariance = _expected_update(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h)
    gmm = _make_single_gaussian_gmm(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h)

    np.testing.assert_allclose(gmm.means[0], expected_mean)
    np.testing.assert_allclose(gmm.variances[0], expected_covariance)
    assert not np.isclose(gmm.variances[0, 0, 1], 0.0)


def test_diagonal_mode_projects_the_same_covariance_update():
    data = np.array(
        [[0.0, 1.0], [2.0, 2.0], [4.0, 5.0], [3.0, 1.0]])
    posterior = np.array([0.2, 0.6, 0.7, 0.4])
    hyper_mean = np.array([1.0, 1.5])
    mean_strength = 2.0
    h = 4.0
    hyper_variance = np.zeros((2, 2))

    full = _make_single_gaussian_gmm(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h)
    diagonal = _make_single_gaussian_gmm(
        data, posterior, hyper_mean, mean_strength, hyper_variance, h,
        diagonal=True)

    np.testing.assert_allclose(diagonal.means, full.means)
    np.testing.assert_allclose(
        np.diag(diagonal.variances[0]), np.diag(full.variances[0]))
    np.testing.assert_array_equal(
        diagonal.variances[0], np.diag(np.diag(diagonal.variances[0])))
    assert not np.isclose(full.variances[0, 0, 1], 0.0)


def test_multiple_components_produce_responsibilities_and_fitted_weights():
    data = np.array([[0.0], [1.0], [4.0], [5.0]])
    class_priors = np.ones((4, 1))
    gmm = GMM(
        [2],
        1,
        initialMeans=np.array([[0.0], [5.0]]),
        initialVariances=np.array([[[1.0]], [[1.0]]]),
        initialMixtureWeights=np.array([0.8, 0.2]),
        initialHyperMixtureWeights=np.array([0.25, 0.75]),
        initialHyperMixtureWeightsNumberOfMeasurements=np.array([4.0]),
    )

    posteriors, _ = gmm.getGaussianPosteriors(data, class_priors)
    posterior_mass = np.sum(posteriors + eps, axis=0)
    expected_weights = posterior_mass + np.array([1.0, 3.0])
    expected_weights /= np.sum(expected_weights)

    np.testing.assert_allclose(np.sum(posteriors, axis=1), 1.0)
    assert np.all(posteriors[:2, 0] > posteriors[:2, 1])
    assert np.all(posteriors[2:, 1] > posteriors[2:, 0])

    gmm.fitGMMParameters(data, posteriors)

    np.testing.assert_allclose(gmm.mixtureWeights, expected_weights)
    np.testing.assert_allclose(np.sum(gmm.mixtureWeights), 1.0)


def test_structure_reconstruction_uses_components_weights_and_fractions():
    data = np.array([[0.0], [2.0], [8.0]])
    gmm = GMM(
        [2, 1],
        1,
        initialMeans=np.array([[0.0], [2.0], [8.0]]),
        initialVariances=np.array([[[1.0]], [[4.0]], [[1.5]]]),
        initialMixtureWeights=np.array([0.25, 0.75, 1.0]),
    )
    fractions = np.array([[1.0, 0.25], [0.0, 0.75]])
    priors = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])

    first_class = (
        0.25 * _normal_density(data, np.array([0.0]), np.array([[1.0]]))
        + 0.75 * _normal_density(data, np.array([2.0]), np.array([[4.0]]))
    )
    second_class = _normal_density(
        data, np.array([8.0]), np.array([[1.5]]))
    expected_likelihoods = np.column_stack(
        [first_class, 0.25 * first_class + 0.75 * second_class])
    expected_posteriors = expected_likelihoods * priors
    expected_posteriors /= (
        np.sum(expected_posteriors, axis=1, keepdims=True) + eps)

    np.testing.assert_allclose(
        gmm.getLikelihoods(data, fractions), expected_likelihoods)
    np.testing.assert_allclose(
        gmm.getPosteriors(data, priors, fractions), expected_posteriors)
