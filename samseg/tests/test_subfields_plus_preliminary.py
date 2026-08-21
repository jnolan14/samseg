import json

import numpy as np
import pytest
import surfa as sf

from samseg.io import kvlReadSharedGMMParameters
from samseg.subregions import core_plus
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.model_policy import SubregionModelPolicy


class _ConfiguredPreliminaryModel(MeshModelPlus):

    def get_cheating_gaussians(self, sameGaussianParameters):
        assert self.preliminaryClassNames == ['Background', 'Tissue']
        return self.artificialMeans, self.artificialVariances


class _CopiedPreliminaryModel(MeshModelPlus):

    def get_cheating_label_groups(self):
        return [['Background'], ['TissueA', 'TissueB']]

    def get_cheating_gaussians(self, sameGaussianParameters):
        assert sameGaussianParameters == [[0], [1, 2]]
        return np.array([1.0, 10.0]), np.array([0.01, 0.01])


class _LabelMapping:

    def __init__(self, values):
        self.values = values

    def search(self, name, exact=False):
        return self.values.get(name)


class _WorkingImage:

    def __init__(self):
        self.data = np.ones((1, 1, 1), dtype='float32')

    @property
    def shape(self):
        return self.data.shape

    def __setitem__(self, key, value):
        self.data[key] = value


class _Segmentation:

    def __init__(self, labels):
        self.data = np.asarray(labels)


class _Mesh:

    def __init__(self, alphas):
        self.points = np.zeros((alphas.shape[0], 3), dtype='float32')
        self.alphas = alphas

    def rasterize(self, shape):
        numberOfClasses = self.alphas.shape[1]
        priors = np.zeros(tuple(shape) + (numberOfClasses,), dtype='uint16')
        priors[..., 0] = 65535
        return priors


class _PriorMesh:

    def __init__(self, priors, numberOfNodes=1):
        self.priors = np.asarray(priors, dtype='uint16')
        self.alphas = np.zeros(
            (numberOfNodes, self.priors.shape[-1]), dtype='float32')

    def rasterize(self, shape):
        assert tuple(shape) == self.priors.shape[:-1]
        return self.priors.copy()


class _MeshCollection:

    def __init__(self, alphas):
        self.reference_mesh = _Mesh(alphas)
        self.k = None

    def read(self, fileName):
        pass

    def transform(self, transform):
        pass


def _write_parameters(path, tissue_components=1, include_tissue_b=True):
    tissueSearchStrings = (
        'TissueA TissueB' if include_tissue_b else 'TissueA')
    path.write_text(
        '# mergedName numberOfComponents searchStrings\n'
        'Background 1 Background\n'
        f'Tissue {tissue_components} {tissueSearchStrings}\n')


def _write_overlapping_parameters(path):
    path.write_text(
        '# mergedName numberOfComponents searchStrings\n'
        'Background 1 Background\n'
        'Tissue 1 Tissue\n'
        'TissueA 1 TissueA\n')


def _write_localizer_lut(path, entries):
    path.write_text(''.join(
        f'{label} {name} 0 0 0 1\n' for label, name in entries))


def _write_policy(
        path, memberships=None, profileName='test', supportMarginInMm=0.0,
        **extraFields):
    specification = {
        'preliminary_localizer_label_memberships_by_profile': {
            profileName: memberships or {},
        },
        'localizer_anatomical_support_margin_mm': supportMarginInMm,
    }
    specification.update(extraFields)
    path.write_text(json.dumps(specification))


def _configured_model(
        parameterFile, withOriginalAlphas=True, policyFile=None,
        localizerLutFile=None):
    model = _ConfiguredPreliminaryModel(
        atlasDir=str(parameterFile.parent),
        outDir=str(parameterFile.parent / 'output'),
        inputImageFileNames=['image.mgz'],
        inputSegFileName='segmentation.mgz',
        preliminarySharedGMMParametersFileName=str(parameterFile),
    )
    model.modelPolicyFileName = (
        str(policyFile) if policyFile is not None else None)
    if policyFile is not None:
        model.preliminaryModelProfileName = 'test'
    if localizerLutFile is not None:
        model.preliminaryLocalizerLookupTableFileName = str(localizerLutFile)
    model.names = ['Background', 'TissueA', 'TissueB']
    model.FreeSurferLabels = np.array([0, 1, 2])
    if withOriginalAlphas:
        model.originalAlphas = np.array([
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.5],
        ])
    model.artificialMeans = np.array([1.0, 10.0])
    model.artificialVariances = np.array([0.01, 0.01])
    return model


def _run_preliminary_preparation(model, monkeypatch):
    originalAlphas = np.array([
        [0.2, 0.3, 0.5],
        [0.1, 0.4, 0.5],
    ], dtype='float32')
    meshCollection = _MeshCollection(originalAlphas)
    monkeypatch.setattr(
        core_plus.gems, 'KvlMeshCollection', lambda: meshCollection)
    model.synthImage = object()
    model.crop_image_by_atlas = lambda image: (_WorkingImage(), object())
    model.debug = False
    model.prepare_for_seg_fitting()


def _profile(parameterFile, localizerLutFile):
    return {
        'sharedGMMParametersFileName': str(parameterFile),
        'localizerLookupTableFileName': str(localizerLutFile),
    }


def _build_localizer_groups(parameters, lookupTable, policy=None):
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy or SubregionModelPolicy()
    model.modelPolicyFileName = None
    model.preliminaryModelProfileName = 'test'
    return model._build_preliminary_localizer_label_groups(
        parameters, lookupTable)


def _profiles(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    asegLutFile = tmp_path / 'ASEGlocalizerLUT.txt'
    synthsegLutFile = tmp_path / 'SYNTHSEGlocalizerLUT.txt'
    _write_parameters(parameterFile)
    # Labels 0/2 are shared by both synthetic profiles. Labels 31/1001 are
    # deliberate profile discriminators for ASEG and SynthSeg respectively.
    _write_localizer_lut(asegLutFile, [
        (0, 'Background'),
        (2, 'TissueA'),
        (31, 'TissueB'),
    ])
    _write_localizer_lut(synthsegLutFile, [
        (0, 'Background'),
        (2, 'TissueA'),
        (1001, 'TissueB'),
    ])
    return {
        'aseg': _profile(parameterFile, asegLutFile),
        'synthseg': _profile(parameterFile, synthsegLutFile),
    }


def test_explicit_preliminary_profile_sets_its_shared_parameters_and_localizer_lut(
        tmp_path):
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation([0, 2])

    selected = model._configure_preliminary_model_profile(
        profiles, requestedProfileName='synthseg')

    assert selected == 'synthseg'
    assert model.preliminaryModelProfileName == 'synthseg'
    assert model.preliminarySharedGMMParametersFileName == profiles[
        'synthseg']['sharedGMMParametersFileName']
    assert model.preliminaryLocalizerLookupTableFileName == profiles[
        'synthseg']['localizerLookupTableFileName']
    assert model.modelPolicyFileName is None


@pytest.mark.parametrize(
    ('fileName', 'expected'),
    [
        ('aseg.mgz', 'aseg'),
        ('aparc+aseg.nii.gz', 'aseg'),
        ('synthseg.mgh', 'synthseg'),
    ],
)
def test_preliminary_profile_selection_uses_filename_provenance(
        tmp_path, fileName, expected):
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSegFileName = str(tmp_path / fileName)
    model.inputSeg = _Segmentation([0, 2])

    assert model._configure_preliminary_model_profile(profiles) == expected


@pytest.mark.parametrize(
    ('labels', 'expected'),
    [
        ([0, 2, 31], 'aseg'),
        ([0, 2, 1001], 'synthseg'),
    ],
)
def test_preliminary_profile_is_inferred_when_labels_match_only_one_profile(
        tmp_path, labels, expected):
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation(labels)

    assert model._configure_preliminary_model_profile(profiles) == expected


@pytest.mark.parametrize(
    ('labels', 'error'),
    [
        ([0, 2], 'distinguish'),
        ([0, 136], 'unsupported'),
    ],
)
def test_preliminary_profile_selection_rejects_ambiguous_or_unsupported(
        tmp_path, labels, error):
    # Labels 0/2 fit both profile LUTs, whereas label 136 fits neither.
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation(labels)

    with pytest.raises(ValueError, match=error):
        model._configure_preliminary_model_profile(profiles)


def test_explicit_preliminary_profile_still_validates_vocabulary(tmp_path):
    # Label 31 belongs to the ASEG fixture; explicitly selecting SynthSeg must
    # still reject it as outside that profile's localizer LUT.
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation([0, 31])

    with pytest.raises(ValueError, match="outside.*synthseg.*31"):
        model._configure_preliminary_model_profile(
            profiles, requestedProfileName='synthseg')


def test_profile_inference_reports_missing_localizer_lut(tmp_path):
    profiles = _profiles(tmp_path)
    profiles['synthseg']['localizerLookupTableFileName'] = str(
        tmp_path / 'missingLUT.txt')
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation([0, 2, 31])

    with pytest.raises(ValueError, match="synthseg.*does not exist"):
        model._configure_preliminary_model_profile(profiles)


def test_preliminary_state_materializes_configured_classes_and_alphas(
        tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile)

    # TissueA/TissueB merge into the single Tissue row. Class fractions and
    # merged alphas therefore follow the shared-parameter row order.
    model._ensure_preliminary_model_state()

    assert [parameter.mergedName
            for parameter in model.preliminarySharedGMMParameters] == [
                'Background', 'Tissue']
    assert model.preliminaryClassNames == ['Background', 'Tissue']
    assert model.sameGaussianParameters == [[0], [1, 2]]
    np.testing.assert_array_equal(
        model.preliminaryClassFractions,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]))
    np.testing.assert_allclose(
        model.preliminaryAlphas,
        np.array([[0.2, 0.8], [0.1, 0.9]]))
    assert model.preliminaryLocalizerLabelGroups is None


def test_preliminary_state_completes_when_atlas_alphas_become_available(
        tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile, withOriginalAlphas=False)

    model._ensure_preliminary_model_state()

    assert model.sameGaussianParameters == [[0], [1, 2]]
    assert model.preliminaryAlphas is None

    model.originalAlphas = np.array([
        [0.2, 0.3, 0.5],
        [0.1, 0.4, 0.5],
    ])
    model._ensure_preliminary_model_state()
    np.testing.assert_allclose(
        model.preliminaryAlphas,
        np.array([[0.2, 0.8], [0.1, 0.9]]))


def test_preliminary_preparation_sets_configured_class_means_and_variances(
        monkeypatch, tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile, withOriginalAlphas=False)

    _run_preliminary_preparation(model, monkeypatch)

    np.testing.assert_array_equal(model.cheatingMeans, [1.0, 10.0])
    np.testing.assert_array_equal(model.cheatingVariances, [0.01, 0.01])


def test_localizer_groups_use_only_labels_from_the_selected_localizer_lut(
        tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    policyFile = tmp_path / 'modelPolicy.json'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    # Atlas labels 0/1/2 and localizer labels 100/200/300 are deliberately
    # disjoint. Unmatched localizer label 300 is supplied only by sparse policy.
    _write_parameters(parameterFile)
    _write_policy(policyFile, {'Tissue': [300]})
    _write_localizer_lut(localizerLutFile, [
        (100, 'Background'),
        (200, 'TissueA'),
        (300, 'Uninferable'),
    ])
    model = _configured_model(
        parameterFile, policyFile=policyFile,
        localizerLutFile=localizerLutFile)

    model._ensure_preliminary_model_state()

    assert model.sameGaussianParameters == [[0], [1, 2]]
    assert model.preliminaryLocalizerLabelGroups == [[100], [200, 300]]
    assert not ({0, 1, 2} & {100, 200, 300})


def test_policy_only_fills_labels_unmatched_by_shared_parameters(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    _write_parameters(parameterFile)
    _write_localizer_lut(localizerLutFile, [
        (0, 'Background'),
        (1, 'TissueA'),
    ])
    parameters = kvlReadSharedGMMParameters(parameterFile)
    lookupTable = sf.load_label_lookup(localizerLutFile)
    policy = SubregionModelPolicy({
        'test': {'Tissue': (1,)},
    })

    # Localizer label 1 is already owned through its TissueA name; policy may
    # fill only labels that shared-parameter matching could not infer.
    with pytest.raises(ValueError, match='only assign labels unmatched'):
        _build_localizer_groups(parameters, lookupTable, policy)


def test_policy_cannot_resolve_ambiguous_shared_parameter_matching(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    _write_overlapping_parameters(parameterFile)
    _write_localizer_lut(localizerLutFile, [
        (0, 'Background'),
        (1, 'TissueA'),
    ])
    parameters = kvlReadSharedGMMParameters(parameterFile)
    lookupTable = sf.load_label_lookup(localizerLutFile)
    policy = SubregionModelPolicy({
        'test': {'Tissue': (1,)},
    })

    # Overlapping Tissue and TissueA search strings give localizer label 1 two
    # inferred owners. Sparse policy cannot arbitrate that model-definition
    # error.
    with pytest.raises(ValueError, match='multiple preliminary classes'):
        _build_localizer_groups(parameters, lookupTable, policy)


def test_numeric_collision_uses_localizer_name_not_atlas_membership(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    policyFile = tmp_path / 'modelPolicy.json'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    # Numeric label 77 deliberately denotes an atlas tract but localizer
    # background. Ownership must follow names within each namespace.
    parameterFile.write_text(
        'Background 1 Background\n'
        'WhiteMatter 1 Tract Cerebral-White-Matter\n')
    _write_policy(policyFile)
    _write_localizer_lut(localizerLutFile, [
        (77, 'Localizer-Background'),
        (2, 'Left-Cerebral-White-Matter'),
    ])
    model = MeshModelPlus(
        atlasDir=str(tmp_path),
        outDir=str(tmp_path / 'output'),
        inputImageFileNames=['image.mgz'],
        inputSegFileName='segmentation.mgz',
        preliminarySharedGMMParametersFileName=str(parameterFile),
    )
    model.modelPolicyFileName = str(policyFile)
    model.preliminaryLocalizerLookupTableFileName = str(localizerLutFile)
    model.names = ['Atlas-Background', 'Atlas-Tract']
    model.FreeSurferLabels = np.array([0, 77])

    model._ensure_preliminary_model_state()

    assert model.sameGaussianParameters == [[0], [77]]
    assert model.preliminaryLocalizerLabelGroups == [[77], [2]]


def test_default_preliminary_gaussians_follow_localizer_label_groups(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    _write_parameters(parameterFile)
    _write_localizer_lut(localizerLutFile, [
        (0, 'Background'),
        (10, 'TissueA'),
    ])
    model = MeshModelPlus(
        atlasDir=str(tmp_path),
        outDir=str(tmp_path / 'output'),
        inputImageFileNames=['image.mgz'],
        inputSegFileName='segmentation.mgz',
        preliminarySharedGMMParametersFileName=str(parameterFile),
    )
    model.preliminaryLocalizerLookupTableFileName = str(localizerLutFile)
    model.names = ['Background', 'TissueA', 'TissueB']
    model.FreeSurferLabels = np.array([0, 1, 2])

    # Localizer tissue label 10 differs from atlas labels 1/2, proving that
    # artificial means come from localizer groups; background 0 maps to 1.
    model._ensure_preliminary_model_state()
    means, variances = model.get_cheating_gaussians(
        model.preliminaryLocalizerLabelGroups)

    np.testing.assert_array_equal(means, [1.0, 10.0])
    np.testing.assert_array_equal(variances, [0.01, 0.01])


@pytest.mark.parametrize(
    ('memberships', 'error'),
    [
        ({'Missing': [3]}, 'unknown preliminary classes'),
        ({'Background': [3], 'Tissue': [3]}, 'belongs to both'),
    ],
)
def test_policy_rejects_unknown_classes_and_overlapping_labels(
        tmp_path, memberships, error):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(policyFile, memberships)

    if error == 'belongs to both':
        with pytest.raises(ValueError, match=error):
            SubregionModelPolicy.read(policyFile)
        return

    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    _write_parameters(parameterFile)
    _write_localizer_lut(localizerLutFile, [
        (0, 'Background'),
        (3, 'TissueA'),
    ])
    model = _configured_model(
        parameterFile, policyFile=policyFile,
        localizerLutFile=localizerLutFile)
    with pytest.raises(ValueError, match=error):
        model._ensure_preliminary_model_state()


def test_policy_rejects_removed_schema_field(tmp_path):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(policyFile, schema='aseg')

    with pytest.raises(ValueError, match='Unsupported.*schema'):
        SubregionModelPolicy.read(policyFile)


@pytest.mark.parametrize('margin', [-1, float('nan'), True])
def test_policy_rejects_invalid_physical_margin(tmp_path, margin):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(policyFile, supportMarginInMm=margin)

    with pytest.raises(ValueError, match='finite nonnegative'):
        SubregionModelPolicy.read(policyFile)


def test_policy_reads_morphology_margins_and_zero_evidence_settings(tmp_path):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(
        policyFile,
        affine_target_morphology='opening',
        localizer_anatomical_support_margin_mm=1.5,
        preliminary_atlas_domain_interior_margin_mm=3.0,
        regional_atlas_domain_interior_margin_mm=2.0,
        zero_evidence_initialization={
            'strategy': 'fixed',
            'mean': [55.0, 65.0],
            'strength': 7.0,
        })

    # Fixed means 55/65 and strength 7 are arbitrary two-channel parser
    # sentinels; 55 is not a historical or default 55/10 policy.
    policy = SubregionModelPolicy.read(policyFile)

    assert policy.affineTargetMorphology == 'opening'
    assert policy.localizerAnatomicalSupportMarginInMm == 1.5
    assert policy.preliminaryAtlasDomainInteriorMarginInMm == 3.0
    assert policy.regionalAtlasDomainInteriorMarginInMm == 2.0
    assert policy.zeroEvidenceInitialization.strategy == 'fixed'
    np.testing.assert_array_equal(
        policy.zeroEvidenceInitialization.mean, [55.0, 65.0])
    assert policy.zeroEvidenceInitialization.strength == 7.0


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        ('affine_target_morphology', 'blur'),
        ('preliminary_atlas_domain_interior_margin_mm', -1),
        ('regional_atlas_domain_interior_margin_mm', True),
        ('zero_evidence_initialization', {'strategy': 'callback'}),
    ],
)
def test_policy_rejects_invalid_morphology_margins_and_zero_evidence_strategy(
        tmp_path, field, value):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(policyFile, **{field: value})

    with pytest.raises(ValueError):
        SubregionModelPolicy.read(policyFile)


def test_empty_policy_uses_neutral_geometry_and_subject_median_fallback():
    policy = SubregionModelPolicy()

    assert policy.preliminaryLocalizerLabelMembershipsByProfile == {}
    assert policy.affineTargetMorphology == 'none'
    assert policy.localizerAnatomicalSupportMarginInMm == 0.0
    assert policy.preliminaryAtlasDomainInteriorMarginInMm == 0.0
    assert policy.regionalAtlasDomainInteriorMarginInMm == 0.0
    assert policy.zeroEvidenceInitialization.strategy == (
        'subject_non_background_median')
    assert policy.get_preliminary_localizer_label_memberships('aseg') == {}


@pytest.mark.parametrize(
    ('entries', 'error'),
    [
        ([(0, 'Background'), (1, 'Other')], 'unmatched'),
        ([(0, 'Background'), (1, 'TissueA')], 'multiple'),
    ],
)
def test_localizer_vocabulary_requires_exactly_one_class(
        tmp_path, entries, error):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    policyFile = tmp_path / 'modelPolicy.json'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
    if error == 'multiple':
        _write_overlapping_parameters(parameterFile)
    else:
        _write_parameters(parameterFile)
    _write_policy(policyFile)
    _write_localizer_lut(localizerLutFile, entries)
    policy = SubregionModelPolicy.read(policyFile)
    parameters = kvlReadSharedGMMParameters(parameterFile)
    lookupTable = sf.load_label_lookup(localizerLutFile)

    # Other matches no shared class; the overlapping parameter fixture makes
    # TissueA match two classes.
    with pytest.raises(ValueError, match=error):
        _build_localizer_groups(parameters, lookupTable, policy)


def test_preliminary_state_requires_complete_atlas_coverage(tmp_path):
    parameterFile = tmp_path / 'incompleteSharedGMMParameters.txt'
    _write_parameters(parameterFile, include_tissue_b=False)
    model = _configured_model(parameterFile)

    with pytest.raises(
            ValueError,
            match=('some structures are not associated with any '
                   'super-structures')):
        model._ensure_preliminary_model_state()


def test_preliminary_state_requires_one_gaussian_per_class(tmp_path):
    parameterFile = tmp_path / 'multiComponentSharedGMMParameters.txt'
    _write_parameters(parameterFile, tissue_components=2)
    model = _configured_model(parameterFile)

    with pytest.raises(
            ValueError,
            match='requires exactly one Gaussian per class'):
        model._ensure_preliminary_model_state()


def test_atlas_structure_cannot_match_multiple_preliminary_classes(tmp_path):
    parameterFile = tmp_path / 'overlappingSharedGMMParameters.txt'
    _write_overlapping_parameters(parameterFile)
    model = _configured_model(parameterFile)

    with pytest.raises(
            ValueError,
            match='must match exactly one class'):
        model._ensure_preliminary_model_state()


@pytest.mark.parametrize(
    ('means', 'variances'),
    [
        (np.array([1.0]), np.array([0.01, 0.01])),
        (np.array([1.0, 10.0]), np.array([[0.01], [0.01]])),
    ],
)
def test_preliminary_gaussians_must_be_class_aligned(
        monkeypatch, tmp_path, means, variances):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile, withOriginalAlphas=False)
    model.artificialMeans = means
    model.artificialVariances = variances

    with pytest.raises(
            ValueError,
            match='one scalar per shared-GMM class'):
        _run_preliminary_preparation(model, monkeypatch)


def test_reconstruction_uses_fitted_full_priors_with_coarse_class_evidence():
    """Reconstruct fine labels from fitted priors and coarse class evidence.

    Prior columns are Background, VDC 28, and nucleus 8101. Rows exercise the
    opposing fitted winners, background coarse evidence, an unmatched coarse
    intensity, and zero atlas support.
    """
    priors = np.array([
        [[[0, 10000, 55535]]],
        [[[0, 55535, 10000]]],
        [[[10000, 0, 55535]]],
        [[[0, 10000, 55535]]],
        [[[0, 0, 0]]],
    ], dtype='uint16')
    model = object.__new__(MeshModelPlus)
    model.mesh = _PriorMesh(priors)
    model.workingImage = sf.Volume(np.array(
        [[[10]], [[10]], [[1]], [[99]], [[1]]], dtype='float32'))
    model.workingImageShape = model.workingImage.shape
    model.originalAlphas = np.zeros((1, 3), dtype='float32')
    model.preliminaryClassFractions = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ])
    model.cheatingMeans = np.array([1.0, 10.0])
    model.FreeSurferLabels = np.array([0, 28, 8101])
    model.inputImages = [sf.Volume(np.ones((5, 1, 1), dtype='float32'))]
    model.inputSeg = sf.Volume(np.array(
        [[[28]], [[10]], [[0]], [[0]], [[0]]], dtype='int32'))
    originalLocalizer = model.inputSeg.data.copy()
    model.labelMapping = sf.LabelLookup()

    segmentation, support = (
        model._reconstruct_initialization_state())

    # Source VDC can become thalamus and source thalamus can become VDC.
    np.testing.assert_array_equal(
        segmentation.data[:, 0, 0], [8101, 28, 0, 8101, 0])
    np.testing.assert_array_equal(
        support.data[:, 0, 0], [True, True, True, True, False])
    np.testing.assert_array_equal(model.inputSeg.data, originalLocalizer)


def test_models_without_shared_parameter_artifact_use_legacy_label_groups(
        tmp_path):
    """Retain legacy groups for subclasses without shared-parameter artifacts."""
    model = _CopiedPreliminaryModel(
        atlasDir=str(tmp_path),
        outDir=str(tmp_path / 'output'),
        inputImageFileNames=['image.mgz'],
        inputSegFileName='segmentation.mgz',
    )
    model.labelMapping = _LabelMapping({
        'Background': 0,
        'TissueA': 1,
        'TissueB': 2,
    })
    model.FreeSurferLabels = np.array([0, 1, 2])
    model.originalAlphas = np.array([
        [0.2, 0.3, 0.5],
        [0.1, 0.4, 0.5],
    ], dtype='float32')

    model._ensure_preliminary_model_state()
    model.cheatingMeans, model.cheatingVariances = (
        model.get_cheating_gaussians(model.sameGaussianParameters))

    assert model.preliminarySharedGMMParameters is None
    assert model.preliminaryClassFractions is None
    assert model.preliminaryClassNames is None
    assert model.sameGaussianParameters == [[0], [1, 2]]
    np.testing.assert_array_equal(
        model.preliminaryAlphas,
        np.array([[0.2, 0.8], [0.1, 0.9]], dtype='float32'))
    np.testing.assert_array_equal(model.cheatingMeans, [1.0, 10.0])
    np.testing.assert_array_equal(model.cheatingVariances, [0.01, 0.01])
