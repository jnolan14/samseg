import inspect
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
        self.gaussianPolicyCalls += 1
        self.gaussianPolicyLabels = sameGaussianParameters
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


def _write_policy(path, memberships=None, **extraFields):
    specification = {
        'preliminary_localizer_label_memberships': memberships or {},
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
    model.gaussianPolicyCalls = 0
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
    model.cheatingAlphaMaskStrel = 0
    model.debug = False
    model.prepare_for_seg_fitting()


def _profile(parameterFile, localizerLutFile, policyFile=None):
    return {
        'sharedGMMParametersFileName': str(parameterFile),
        'localizerLookupTableFileName': str(localizerLutFile),
        'modelPolicyFileName': (
            str(policyFile) if policyFile is not None else None),
    }


def _build_localizer_groups(parameters, lookupTable, policy=None):
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy or SubregionModelPolicy()
    return model._build_preliminary_localizer_label_groups(
        parameters, lookupTable)


def _profiles(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    asegLutFile = tmp_path / 'ASEGlocalizerLUT.txt'
    synthsegLutFile = tmp_path / 'SYNTHSEGlocalizerLUT.txt'
    _write_parameters(parameterFile)
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


def test_preliminary_profile_selection_is_generic_and_atomic(tmp_path):
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
def test_preliminary_profile_selection_uses_unique_bounded_vocabulary(
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
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation(labels)

    with pytest.raises(ValueError, match=error):
        model._configure_preliminary_model_profile(profiles)


def test_explicit_preliminary_profile_still_validates_vocabulary(tmp_path):
    profiles = _profiles(tmp_path)
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation([0, 31])

    with pytest.raises(ValueError, match="outside.*synthseg.*31"):
        model._configure_preliminary_model_profile(
            profiles, requestedProfileName='synthseg')


def test_missing_candidate_vocabulary_is_a_profile_error(tmp_path):
    profiles = _profiles(tmp_path)
    profiles['synthseg']['localizerLookupTableFileName'] = str(
        tmp_path / 'missingLUT.txt')
    model = _configured_model(
        tmp_path / 'preliminarySharedGMMParameters.txt')
    model.inputSeg = _Segmentation([0, 2, 31])

    with pytest.raises(ValueError, match="synthseg.*does not exist"):
        model._configure_preliminary_model_profile(profiles)


def test_preliminary_state_uses_standard_samseg_merging(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile)

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
    assert model.gaussianPolicyCalls == 0


def test_preliminary_state_is_lazy_and_idempotent(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile, withOriginalAlphas=False)

    model._ensure_preliminary_model_state()

    sharedParameters = model.preliminarySharedGMMParameters
    classFractions = model.preliminaryClassFractions
    sameGaussianParameters = model.sameGaussianParameters
    assert sameGaussianParameters == [[0], [1, 2]]
    assert model.preliminaryAlphas is None

    model.originalAlphas = np.array([
        [0.2, 0.3, 0.5],
        [0.1, 0.4, 0.5],
    ])
    model._ensure_preliminary_model_state()
    preliminaryAlphas = model.preliminaryAlphas
    model._ensure_preliminary_model_state()

    assert model.preliminarySharedGMMParameters is sharedParameters
    assert model.preliminaryClassFractions is classFractions
    assert model.sameGaussianParameters is sameGaussianParameters
    assert model.preliminaryAlphas is preliminaryAlphas


def test_prepare_keeps_explicit_gaussian_policy_call(monkeypatch, tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameterFile)
    model = _configured_model(parameterFile, withOriginalAlphas=False)

    _run_preliminary_preparation(model, monkeypatch)

    assert model.gaussianPolicyCalls == 1
    assert model.gaussianPolicyLabels == [[0], [1, 2]]
    np.testing.assert_array_equal(model.cheatingMeans, [1.0, 10.0])
    np.testing.assert_array_equal(model.cheatingVariances, [0.01, 0.01])

    source = inspect.getsource(MeshModelPlus.prepare_for_seg_fitting)
    assert source.index('_ensure_preliminary_model_state') < source.index(
        'rasterize') < source.index('get_cheating_gaussians')
    assert 'self.get_cheating_gaussians(gaussianLabelGroups)' in source


def test_localizer_membership_is_derived_without_atlas_numeric_union(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    policyFile = tmp_path / 'modelPolicy.json'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
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
        'Tissue': (1,),
    })

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
        'Tissue': (1,),
    })

    with pytest.raises(ValueError, match='multiple preliminary classes'):
        _build_localizer_groups(parameters, lookupTable, policy)


def test_numeric_collision_uses_localizer_name_not_atlas_membership(tmp_path):
    parameterFile = tmp_path / 'preliminarySharedGMMParameters.txt'
    policyFile = tmp_path / 'modelPolicy.json'
    localizerLutFile = tmp_path / 'localizerLUT.txt'
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


def test_generic_preliminary_gaussians_follow_localizer_vocabulary(tmp_path):
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


def test_policy_rejects_obsolete_stage_specific_fields(tmp_path):
    policyFile = tmp_path / 'modelPolicy.json'
    _write_policy(policyFile, schema='aseg')

    with pytest.raises(ValueError, match='Unsupported.*schema'):
        SubregionModelPolicy.read(policyFile)


def test_default_policy_contains_no_model_construction_behavior():
    policy = SubregionModelPolicy()

    assert policy.preliminaryLocalizerLabelMemberships == {}
    assert not hasattr(policy, 'build_preliminary_localizer_label_groups')


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


def test_preliminary_state_rejects_fractional_atlas_membership(tmp_path):
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


def test_reconstruction_runs_before_fitted_mesh_leaves_subject_space():
    source = inspect.getsource(MeshModelPlus.fit_mesh_to_seg)

    assert source.index('self.mesh.alphas = self.originalAlphas') < source.index(
        '_reconstruct_initialization_state') < source.index(
            'set_positions') < source.index('inverseTransform')


def test_checkpoint_compatibility_branch_preserves_copied_behavior(tmp_path):
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
    preliminaryAlphas = model.preliminaryAlphas
    model._ensure_preliminary_model_state()
    model.cheatingMeans, model.cheatingVariances = (
        model.get_cheating_gaussians(model.sameGaussianParameters))

    assert model.preliminarySharedGMMParameters is None
    assert model.preliminaryClassFractions is None
    assert model.preliminaryClassNames is None
    assert model.sameGaussianParameters == [[0], [1, 2]]
    assert model.preliminaryAlphas is preliminaryAlphas
    np.testing.assert_array_equal(
        model.preliminaryAlphas,
        np.array([[0.2, 0.8], [0.1, 0.9]], dtype='float32'))
    np.testing.assert_array_equal(model.cheatingMeans, [1.0, 10.0])
    np.testing.assert_array_equal(model.cheatingVariances, [0.01, 0.01])


def test_preliminary_successor_api_adds_no_new_region_hooks():
    assert hasattr(MeshModelPlus, 'get_cheating_gaussians')
    assert hasattr(MeshModelPlus, 'get_cheating_label_groups')
    assert not hasattr(
        MeshModelPlus, 'get_preliminary_shared_gmm_parameters_file')
    assert not hasattr(MeshModelPlus, 'get_preliminary_gaussians')
