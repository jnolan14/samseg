import json
from dataclasses import dataclass, field


def _read_nonnegative_integers(values, fieldName):
    """Read and validate a sequence of nonnegative integer labels.

    Parameters
    ----------
    values : sequence
        Values loaded from the policy artifact.
    fieldName : str
        Field description used in validation errors.

    Returns
    -------
    tuple of int
        Validated labels in their configured order.
    """
    if not isinstance(values, list):
        raise ValueError(f'{fieldName} must be a list')
    if any(isinstance(value, bool) or not isinstance(value, int)
           or value < 0 for value in values):
        raise ValueError(
            f'{fieldName} must contain nonnegative integer labels')
    if len(values) != len(set(values)):
        raise ValueError(f'{fieldName} contains duplicate labels')
    return tuple(values)


@dataclass(frozen=True)
class SubregionModelPolicy:
    """Sparse policy data for a subregion model lifecycle.

    Parameters
    ----------
    preliminaryLocalizerLabelMemberships : dict
        Exact memberships for localizer labels that shared-parameter matching
        cannot infer, keyed by preliminary class name.

    Notes
    -----
    Shared-GMM parameters define class names and matching semantics. This
    policy only fills labels with no shared-parameter match. It cannot override
    a unique inferred owner or resolve ambiguous matching semantics.
    """

    preliminaryLocalizerLabelMemberships: dict = field(default_factory=dict)

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
            Validated policy data.
        """
        with open(fileName) as file:
            specification = json.load(file)
        if not isinstance(specification, dict):
            raise ValueError('Subregion model policy must be a JSON object')

        supportedFields = {'preliminary_localizer_label_memberships'}
        unsupportedFields = sorted(set(specification) - supportedFields)
        if unsupportedFields:
            raise ValueError(
                'Unsupported subregion model-policy fields: '
                + ', '.join(unsupportedFields))

        memberships = specification.get(
            'preliminary_localizer_label_memberships', {})
        if not isinstance(memberships, dict):
            raise ValueError(
                'preliminary_localizer_label_memberships must be an object')

        validatedMemberships = {}
        labelOwners = {}
        for className, labels in memberships.items():
            if not isinstance(className, str) or not className:
                raise ValueError(
                    'Preliminary localizer memberships require nonempty '
                    'class names')
            validatedLabels = _read_nonnegative_integers(
                labels,
                'preliminary_localizer_label_memberships'
                f'[{className!r}]')
            for label in validatedLabels:
                previousOwner = labelOwners.get(label)
                if previousOwner is not None:
                    raise ValueError(
                        f'Localizer label {label} belongs to both '
                        f'{previousOwner!r} and {className!r}')
                labelOwners[label] = className
            validatedMemberships[className] = validatedLabels

        return cls(
            preliminaryLocalizerLabelMemberships=validatedMemberships)
