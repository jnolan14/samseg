import json
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage

# -----------------------------------------------------------------------------
# Policy decisions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MeanHyperparameterFallbackConfiguration:
    """Configure fallback mean hyperparameters for an unsupported class."""

    strategy: str = 'subject_non_background_median'
    strength: float = 10.0
    mean: object = None


def _subject_non_background_median_mean_hyperparameter_fallback(
        configuration, aggregateObservations, numberOfChannels):
    """Return subject-median mean hyperparameters for one unsupported class."""
    observations = np.asarray(aggregateObservations)
    if (observations.ndim != 2
            or observations.shape[1] != numberOfChannels):
        raise ValueError(
            'Aggregate mean-hyperparameter fallback observations must be a '
            'sample-by-channel array')
    if observations.shape[0] == 0:
        raise RuntimeError(
            'Cannot initialize an unsupported class because the subject has '
            'no usable non-background observations')
    return np.median(observations, axis=0), configuration.strength


def _fixed_mean_hyperparameter_fallback(
        configuration, _aggregateObservations, numberOfChannels):
    """Return configured fixed mean hyperparameters for one unsupported class."""
    try:
        mean = np.broadcast_to(
            np.asarray(configuration.mean, dtype='float64'),
            (numberOfChannels,)).copy()
    except ValueError as error:
        raise ValueError(
            'Fixed mean-hyperparameter fallback is not broadcast-compatible '
            f'with {numberOfChannels} channels') from error
    return mean, configuration.strength


def _leave_affine_target_unchanged(support, _structure):
    """Return the affine-alignment target without morphology."""
    return support


def _open_affine_target(support, structure):
    """Open the affine-alignment target with the supplied structure."""
    support = scipy.ndimage.binary_erosion(
        support, structure=structure, border_value=1)
    return scipy.ndimage.binary_dilation(support, structure=structure)


def _close_affine_target(support, structure):
    """Close the affine-alignment target with the supplied structure."""
    support = scipy.ndimage.binary_dilation(support, structure=structure)
    return scipy.ndimage.binary_erosion(
        support, structure=structure, border_value=1)


def _no_initial_gmm_covariance_fallback(_gmm, _observations):
    """Decline to replace an unusable first-state covariance."""
    return None


def _regional_fitting_covariance(gmm, regionalFittingObservations):
    """Pool the observations already selected by the regional fitting mask."""
    observations = np.asarray(
        regionalFittingObservations, dtype='float64')
    if (observations.ndim != 2
            or observations.shape[1] != gmm.numberOfContrasts
            or len(observations) < 2):
        return None
    covariance = np.atleast_2d(
        np.cov(observations, rowvar=False, ddof=1))
    if gmm.useDiagonalCovarianceMatrices:
        covariance = np.diag(np.diag(covariance))
    return covariance


# Policy artifacts may contain only identifiers from the keys below. Each key
# selects a local implementation, never a Python path or dynamic callback.
_MEAN_HYPERPARAMETER_FALLBACKS = {
    'subject_non_background_median': (
        _subject_non_background_median_mean_hyperparameter_fallback),
    'fixed': _fixed_mean_hyperparameter_fallback,
}

_AFFINE_TARGET_MORPHOLOGIES = {
    'none': _leave_affine_target_unchanged,
    'opening': _open_affine_target,
    'closing': _close_affine_target,
}

_INITIAL_GMM_COVARIANCE_FALLBACKS = {
    'none': _no_initial_gmm_covariance_fallback,
    'regional_fitting_covariance': _regional_fitting_covariance,
}


@dataclass(frozen=True)
class SubregionModelPolicy:
    """Typed model-specific decisions used by the shared subregion lifecycle.

    ``MeshModelPlus`` owns lifecycle mechanics, while configured model artifacts
    own ordinary class structure. This policy supplies the sparse physical and
    strategy decisions needed at the explicit seams represented below.

    Parameters
    ----------
    preliminaryLocalizerLabelMembershipsByProfile : dict
        Exact memberships that shared-parameter matching cannot infer, keyed
        first by preliminary profile and then by preliminary class name.
    affineTargetMorphology : {'opening', 'closing', 'none'}
        Morphology applied to the anatomical affine-alignment target.
    localizerAnatomicalSupportMarginInMm : float
        Physical margin around valid non-background localizer anatomy. The
        same anatomical validity concept serves regional and whole-field
        initialization consumers.
    preliminaryAtlasDomainInteriorMarginInMm : float
        Physical inward margin from the preliminary atlas cuboid boundary.
    regionalAtlasDomainInteriorMarginInMm : float
        Physical inward margin from the regional atlas cuboid boundary.
    meanHyperparameterFallback : MeanHyperparameterFallbackConfiguration
        Configured fallback used only when a class has no usable
        class-specific observations for estimating its mean hyperparameters.
    initialGMMCovarianceFallback : {'none', 'regional_fitting_covariance'}
        Strategy used when the first K=1 GMM M-step does not produce a usable
        covariance.
    maximumGMMIterations : int
        Maximum number of completed GMM M-steps per outer mesh iteration.
        Reaching this work ceiling is distinct from convergence.

    Notes
    -----
    The JSON artifact serializes the policy but does not define its ownership.
    Preliminary memberships fill labels with no shared-parameter match and
    cannot override inferred or ambiguous owners. Profile-conditioned
    memberships coexist with lifecycle-wide physical and strategy decisions.
    """

    preliminaryLocalizerLabelMembershipsByProfile: dict = field(
        default_factory=dict)
    affineTargetMorphology: str = 'none'
    localizerAnatomicalSupportMarginInMm: float = 0.0
    preliminaryAtlasDomainInteriorMarginInMm: float = 0.0
    regionalAtlasDomainInteriorMarginInMm: float = 0.0
    meanHyperparameterFallback: MeanHyperparameterFallbackConfiguration = (
        field(default_factory=MeanHyperparameterFallbackConfiguration))
    initialGMMCovarianceFallback: str = 'none'
    maximumGMMIterations: int = 100

    def get_preliminary_localizer_label_memberships(self, profileName):
        """Return sparse preliminary memberships for one selected profile."""
        return self.preliminaryLocalizerLabelMembershipsByProfile.get(
            profileName, {})

    def get_fallback_mean_hyperparameters(
            self, aggregateObservations, numberOfChannels):
        """Return configured mean hyperparameters for an unsupported class."""
        configuration = self.meanHyperparameterFallback
        try:
            fallback = _MEAN_HYPERPARAMETER_FALLBACKS[configuration.strategy]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                'Unsupported mean-hyperparameter fallback '
                f'{configuration.strategy!r}') from error
        return fallback(
            configuration, aggregateObservations, numberOfChannels)

    def apply_affine_target_morphology(self, support, structure):
        """Apply the configured morphology to the affine-alignment target."""
        try:
            morphology = _AFFINE_TARGET_MORPHOLOGIES[
                self.affineTargetMorphology]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                'Unsupported affine-target morphology '
                f'{self.affineTargetMorphology!r}') from error
        return morphology(support, structure)

    def update_gmm_parameters(self, gmm, data, gaussianPosteriors):
        """Update a GMM while retaining components with weak support."""
        if gmm.tied:
            raise NotImplementedError(
                'Plus low-support retention does not support tied Gaussians')

        masses = np.sum(gaussianPosteriors, axis=0)
        weak = masses <= 1e-2
        previousMeans = gmm.means.copy()
        previousVariances = gmm.variances.copy()
        previousWeights = gmm.mixtureWeights.copy()

        gmm.fitGMMParameters(data, gaussianPosteriors)
        unusable = np.array([
            not np.all(np.isfinite(gmm.means[gaussianNumber]))
            or not _covariance_is_usable(covariance)
            for gaussianNumber, covariance in enumerate(gmm.variances)
        ])
        retained = weak | unusable
        gmm.means[retained] = previousMeans[retained]
        gmm.variances[retained] = previousVariances[retained]
        if np.any(unusable & ~weak):
            warnings.warn(
                'Retaining last valid GMM state after an unusable numerical '
                'update', RuntimeWarning)

        gaussianOffset = 0
        for numberOfComponents in gmm.numberOfGaussiansPerClass:
            gaussianSlice = slice(
                gaussianOffset, gaussianOffset + numberOfComponents)
            gaussianOffset += numberOfComponents
            classRetained = retained[gaussianSlice]
            if not np.any(classRetained):
                continue
            if np.all(classRetained):
                gmm.mixtureWeights[gaussianSlice] = (
                    previousWeights[gaussianSlice])
                continue

            updatedWeights = gmm.mixtureWeights[gaussianSlice].copy()
            retainedWeights = previousWeights[gaussianSlice][classRetained]
            residual = 1.0 - np.sum(retainedWeights)
            activeTotal = np.sum(updatedWeights[~classRetained])
            if (residual <= 0
                    or activeTotal <= 0
                    or not np.isfinite(activeTotal)):
                raise RuntimeError(
                    'Cannot normalize active mixture weights around retained '
                    'low-support components')
            updatedWeights[classRetained] = retainedWeights
            updatedWeights[~classRetained] *= residual / activeTotal
            gmm.mixtureWeights[gaussianSlice] = updatedWeights

    def get_initial_gmm_fallback_covariance(
            self, gmm, regionalFittingObservations):
        """Delegate the configured first-state K=1 covariance behavior."""
        try:
            fallback = _INITIAL_GMM_COVARIANCE_FALLBACKS[
                self.initialGMMCovarianceFallback]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                'Unsupported initial GMM covariance fallback '
                f'{self.initialGMMCovarianceFallback!r}') from error
        return fallback(gmm, regionalFittingObservations)

    def has_gmm_converged(
            self, previousObjective, currentObjective, completedIterations):
        """Return whether the configured structural GMM has converged."""
        if (not np.isfinite(previousObjective)
                or not np.isfinite(currentObjective)
                or currentObjective == 0):
            raise RuntimeError(
                'Structural GMM relative objective improvement is not finite')
        if completedIterations < 2:
            return False
        relativeImprovement = (
            (previousObjective - currentObjective) / abs(currentObjective))
        if not np.isfinite(relativeImprovement):
            raise RuntimeError(
                'Structural GMM relative objective improvement is not finite')
        return relativeImprovement <= 1e-5

    @classmethod
    def read(cls, fileName):
        """Read a sparse subregion model-policy artifact.

        Parameters
        ----------
        fileName : path-like
            JSON policy artifact to read.

        Returns
        -------
        SubregionModelPolicy
            Validated lifecycle policy.
        """
        with open(fileName) as file:
            specification = json.load(file)

        # Validate the artifact envelope before interpreting policy fields.
        if not isinstance(specification, dict):
            raise ValueError('Subregion model policy must be a JSON object')

        supportedFields = {
            'preliminary_localizer_label_memberships_by_profile',
            'affine_target_morphology',
            'localizer_anatomical_support_margin_mm',
            'preliminary_atlas_domain_interior_margin_mm',
            'regional_atlas_domain_interior_margin_mm',
            'mean_hyperparameter_fallback',
            'initial_gmm_covariance_fallback',
            'maximum_gmm_iterations',
        }
        unsupportedFields = sorted(set(specification) - supportedFields)
        if unsupportedFields:
            raise ValueError(
                'Unsupported subregion model-policy fields: '
                + ', '.join(unsupportedFields))

        # Validate profile-scoped memberships and enforce one policy owner per
        # localizer label within each profile.
        membershipsByProfile = specification.get(
            'preliminary_localizer_label_memberships_by_profile', {})
        if not isinstance(membershipsByProfile, dict):
            raise ValueError(
                'preliminary_localizer_label_memberships_by_profile must be '
                'an object')

        validatedMembershipsByProfile = {}
        for profileName, memberships in membershipsByProfile.items():
            if not isinstance(profileName, str) or not profileName:
                raise ValueError(
                    'Preliminary localizer membership profiles require '
                    'nonempty names')
            if not isinstance(memberships, dict):
                raise ValueError(
                    'Preliminary localizer memberships for profile '
                    f'{profileName!r} must be an object')

            validatedMemberships = {}
            labelOwners = {}
            for className, labels in memberships.items():
                if not isinstance(className, str) or not className:
                    raise ValueError(
                        'Preliminary localizer memberships require nonempty '
                        'class names')
                validatedLabels = _read_nonnegative_integers(
                    labels,
                    'preliminary_localizer_label_memberships_by_profile'
                    f'[{profileName!r}][{className!r}]')
                for label in validatedLabels:
                    previousOwner = labelOwners.get(label)
                    if previousOwner is not None:
                        raise ValueError(
                            f'Localizer label {label} belongs to both '
                            f'{previousOwner!r} and {className!r} in profile '
                            f'{profileName!r}')
                    labelOwners[label] = className
                validatedMemberships[className] = validatedLabels
            validatedMembershipsByProfile[profileName] = validatedMemberships

        # Validate lifecycle-wide choices after the membership map is complete.
        affineTargetMorphology = specification.get(
            'affine_target_morphology', 'none')
        if (not isinstance(affineTargetMorphology, str)
                or affineTargetMorphology not in
                _AFFINE_TARGET_MORPHOLOGIES):
            raise ValueError(
                'affine_target_morphology must be one of: closing, none, '
                'opening')

        initialGMMCovarianceFallback = specification.get(
            'initial_gmm_covariance_fallback', 'none')
        if (not isinstance(initialGMMCovarianceFallback, str)
                or initialGMMCovarianceFallback not in
                _INITIAL_GMM_COVARIANCE_FALLBACKS):
            raise ValueError(
                'initial_gmm_covariance_fallback must be one of: '
                + ', '.join(sorted(
                    _INITIAL_GMM_COVARIANCE_FALLBACKS)))

        return cls(
            preliminaryLocalizerLabelMembershipsByProfile=(
                validatedMembershipsByProfile),
            affineTargetMorphology=affineTargetMorphology,
            localizerAnatomicalSupportMarginInMm=(
                _read_nonnegative_finite_number(
                    specification,
                    'localizer_anatomical_support_margin_mm')),
            preliminaryAtlasDomainInteriorMarginInMm=(
                _read_nonnegative_finite_number(
                    specification,
                    'preliminary_atlas_domain_interior_margin_mm')),
            regionalAtlasDomainInteriorMarginInMm=(
                _read_nonnegative_finite_number(
                    specification,
                    'regional_atlas_domain_interior_margin_mm')),
            meanHyperparameterFallback=(
                _read_mean_hyperparameter_fallback(specification)),
            initialGMMCovarianceFallback=initialGMMCovarianceFallback,
            maximumGMMIterations=_read_integer_at_least(
                specification, 'maximum_gmm_iterations', 2, default=100))


# -----------------------------------------------------------------------------
# JSON parsing and validation
# -----------------------------------------------------------------------------


def _read_nonnegative_integers(values, fieldName):
    """Return unique nonnegative integer labels in configured order."""
    if not isinstance(values, list):
        raise ValueError(f'{fieldName} must be a list')
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value < 0 for value in values):
        raise ValueError(
            f'{fieldName} must contain nonnegative integer labels')
    if len(values) != len(set(values)):
        raise ValueError(f'{fieldName} contains duplicate labels')
    return tuple(values)


def _covariance_is_usable(covariance):
    """Return whether a covariance is finite and positive definite."""
    if not np.all(np.isfinite(covariance)):
        return False
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return False
    return True


def _read_nonnegative_finite_number(specification, fieldName, default=0.0):
    """Return one finite nonnegative numeric policy value."""
    value = specification.get(fieldName, default)
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0):
        raise ValueError(f'{fieldName} must be a finite nonnegative number')
    return float(value)


def _read_integer_at_least(specification, fieldName, minimum, default):
    """Return one integer policy value no smaller than ``minimum``."""
    value = specification.get(fieldName, default)
    if (isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum):
        raise ValueError(
            f'{fieldName} must be an integer greater than or equal to '
            f'{minimum}')
    return value


def _read_mean_hyperparameter_fallback(specification):
    """Read the configured fallback strategy and its parameters."""
    fieldName = 'mean_hyperparameter_fallback'
    value = specification.get(fieldName, {})
    if not isinstance(value, dict):
        raise ValueError(f'{fieldName} must be an object')

    strategy = value.get('strategy', 'subject_non_background_median')
    supportedFieldsByStrategy = {
        'subject_non_background_median': {'strategy', 'strength'},
        'fixed': {'strategy', 'mean', 'strength'},
    }
    if (not isinstance(strategy, str)
            or strategy not in _MEAN_HYPERPARAMETER_FALLBACKS):
        raise ValueError(
            f'{fieldName}.strategy must be one of: '
            + ', '.join(sorted(_MEAN_HYPERPARAMETER_FALLBACKS)))
    unsupportedFields = sorted(
        set(value) - supportedFieldsByStrategy[strategy])
    if unsupportedFields:
        raise ValueError(
            f'Unsupported {fieldName} fields for {strategy!r}: '
            + ', '.join(unsupportedFields))

    strength = _read_nonnegative_finite_number(
        value, 'strength', default=10.0)
    mean = value.get('mean')
    if strategy == 'fixed':
        if mean is None:
            raise ValueError(f'{fieldName}.mean is required for fixed strategy')
        meanArray = np.asarray(mean)
        if (meanArray.ndim > 1
                or not np.issubdtype(meanArray.dtype, np.number)
                or not np.all(np.isfinite(meanArray))):
            raise ValueError(
                f'{fieldName}.mean must be a finite scalar or vector')

    return MeanHyperparameterFallbackConfiguration(
        strategy=strategy, strength=strength, mean=mean)
