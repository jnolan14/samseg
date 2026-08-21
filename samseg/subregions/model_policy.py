import json
import math
from dataclasses import dataclass, field

import numpy as np

# -----------------------------------------------------------------------------
# Policy decisions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroEvidenceInitializationPolicy:
    """Choose hyperparameters when class-specific evidence is unavailable."""

    strategy: str = 'subject_non_background_median'
    strength: float = 10.0
    mean: object = None

    def initialize(self, aggregateObservations, numberOfChannels):
        """Return fallback means and strength for one unsupported class.

        ``MeshModelPlus`` supplies aggregate observations that already satisfy
        its multichannel validity and non-background support rules.
        """
        if self.strategy == 'subject_non_background_median':
            observations = np.asarray(aggregateObservations)
            if observations.ndim != 2 or observations.shape[1] != numberOfChannels:
                raise ValueError(
                    'Aggregate zero-evidence observations must be a '
                    'sample-by-channel array')
            if observations.shape[0] == 0:
                raise RuntimeError(
                    'Cannot initialize an unsupported class because the '
                    'subject has no usable non-background observations')
            return np.median(observations, axis=0), self.strength

        if self.strategy == 'fixed':
            try:
                mean = np.broadcast_to(
                    np.asarray(self.mean, dtype='float64'),
                    (numberOfChannels,)).copy()
            except ValueError as error:
                raise ValueError(
                    'Fixed zero-evidence mean is not broadcast-compatible '
                    f'with {numberOfChannels} channels') from error
            return mean, self.strength

        raise RuntimeError(
            f'Unsupported zero-evidence strategy {self.strategy!r}')


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
    zeroEvidenceInitialization : ZeroEvidenceInitializationPolicy
        Strategy used only when a class has no usable class-specific evidence.

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
    zeroEvidenceInitialization: ZeroEvidenceInitializationPolicy = field(
        default_factory=ZeroEvidenceInitializationPolicy)

    def get_preliminary_localizer_label_memberships(self, profileName):
        """Return sparse preliminary memberships for one selected profile."""
        return self.preliminaryLocalizerLabelMembershipsByProfile.get(
            profileName, {})

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
            'zero_evidence_initialization',
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
        if affineTargetMorphology not in {'opening', 'closing', 'none'}:
            raise ValueError(
                'affine_target_morphology must be one of: closing, none, '
                'opening')

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
            zeroEvidenceInitialization=(
                _read_zero_evidence_initialization(specification)))


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


def _read_nonnegative_finite_number(specification, fieldName, default=0.0):
    """Return one finite nonnegative numeric policy value."""
    value = specification.get(fieldName, default)
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0):
        raise ValueError(f'{fieldName} must be a finite nonnegative number')
    return float(value)


def _read_zero_evidence_initialization(specification):
    """Read the configured zero-evidence strategy and its parameters."""
    fieldName = 'zero_evidence_initialization'
    value = specification.get(fieldName, {})
    if not isinstance(value, dict):
        raise ValueError(f'{fieldName} must be an object')

    strategy = value.get('strategy', 'subject_non_background_median')
    supportedStrategies = {
        'subject_non_background_median': {'strategy', 'strength'},
        'fixed': {'strategy', 'mean', 'strength'},
    }
    if strategy not in supportedStrategies:
        raise ValueError(
            f'{fieldName}.strategy must be one of: '
            + ', '.join(sorted(supportedStrategies)))
    unsupportedFields = sorted(set(value) - supportedStrategies[strategy])
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

    return ZeroEvidenceInitializationPolicy(
        strategy=strategy, strength=strength, mean=mean)
