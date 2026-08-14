import inspect

import numpy as np
import pytest

from samseg.subregions import core_plus
from samseg.subregions.core_plus import MeshModelPlus


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


class _Mesh:

    def __init__(self, alphas):
        self.points = np.zeros((alphas.shape[0], 3), dtype='float32')
        self.alphas = alphas

    def rasterize(self, shape):
        numberOfClasses = self.alphas.shape[1]
        priors = np.zeros(tuple(shape) + (numberOfClasses,), dtype='uint16')
        priors[..., 0] = 65535
        return priors


class _MeshCollection:

    def __init__(self, alphas):
        self.reference_mesh = _Mesh(alphas)
        self.k = None

    def read(self, fileName):
        pass

    def transform(self, transform):
        pass


def _write_parameters(path, tissue_components=1, include_tissue_b=True):
    tissue_search_strings = (
        'TissueA TissueB' if include_tissue_b else 'TissueA')
    path.write_text(
        '# mergedName numberOfComponents searchStrings\n'
        'Background 1 Background\n'
        f'Tissue {tissue_components} {tissue_search_strings}\n')


def _write_overlapping_parameters(path):
    path.write_text(
        '# mergedName numberOfComponents searchStrings\n'
        'Background 1 Background\n'
        'Tissue 1 Tissue\n'
        'TissueA 1 TissueA\n')


def _configured_model(parameter_file, with_original_alphas=True):
    model = _ConfiguredPreliminaryModel(
        atlasDir=str(parameter_file.parent),
        outDir=str(parameter_file.parent / 'output'),
        inputImageFileNames=['image.mgz'],
        inputSegFileName='segmentation.mgz',
        preliminarySharedGMMParametersFileName=str(parameter_file),
    )
    model.names = ['Background', 'TissueA', 'TissueB']
    model.FreeSurferLabels = np.array([0, 1, 2])
    if with_original_alphas:
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


def test_preliminary_state_uses_standard_samseg_merging(tmp_path):
    parameter_file = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameter_file)
    model = _configured_model(parameter_file)

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
    assert model.gaussianPolicyCalls == 0
    assert model.cheatingMeans is None
    assert model.cheatingVariances is None


def test_preliminary_state_is_lazy_and_idempotent(tmp_path):
    parameter_file = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameter_file)
    model = _configured_model(parameter_file, with_original_alphas=False)

    model._ensure_preliminary_model_state()

    sharedParameters = model.preliminarySharedGMMParameters
    classFractions = model.preliminaryClassFractions
    sameGaussianParameters = model.sameGaussianParameters
    assert sameGaussianParameters == [[0], [1, 2]]
    assert model.preliminaryAlphas is None
    assert model.gaussianPolicyCalls == 0

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
    assert model.gaussianPolicyCalls == 0


def test_prepare_keeps_explicit_gaussian_policy_call(monkeypatch, tmp_path):
    parameter_file = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameter_file)
    model = _configured_model(parameter_file, with_original_alphas=False)

    _run_preliminary_preparation(model, monkeypatch)

    assert model.gaussianPolicyCalls == 1
    assert model.gaussianPolicyLabels == [[0], [1, 2]]
    np.testing.assert_array_equal(model.cheatingMeans, [1.0, 10.0])
    np.testing.assert_array_equal(model.cheatingVariances, [0.01, 0.01])

    source = inspect.getsource(MeshModelPlus.prepare_for_seg_fitting)
    assert source.index('_ensure_preliminary_model_state') < source.index(
        'rasterize') < source.index('get_cheating_gaussians')
    assert 'get_cheating_gaussians(self.sameGaussianParameters)' in source


def test_preliminary_state_requires_complete_atlas_coverage(tmp_path):
    parameter_file = tmp_path / 'incompleteSharedGMMParameters.txt'
    _write_parameters(parameter_file, include_tissue_b=False)
    model = _configured_model(parameter_file)

    with pytest.raises(
            ValueError,
            match=('some structures are not associated with any '
                   'super-structures')):
        model._ensure_preliminary_model_state()


def test_preliminary_state_requires_one_gaussian_per_class(tmp_path):
    parameter_file = tmp_path / 'multiComponentSharedGMMParameters.txt'
    _write_parameters(parameter_file, tissue_components=2)
    model = _configured_model(parameter_file)

    with pytest.raises(
            ValueError,
            match='requires exactly one Gaussian per class'):
        model._ensure_preliminary_model_state()


def test_preliminary_state_rejects_fractional_class_membership(tmp_path):
    parameter_file = tmp_path / 'overlappingSharedGMMParameters.txt'
    _write_overlapping_parameters(parameter_file)
    model = _configured_model(parameter_file)

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
    parameter_file = tmp_path / 'preliminarySharedGMMParameters.txt'
    _write_parameters(parameter_file)
    model = _configured_model(parameter_file, with_original_alphas=False)
    model.artificialMeans = means
    model.artificialVariances = variances

    with pytest.raises(
            ValueError,
            match='one scalar per shared-GMM class'):
        _run_preliminary_preparation(model, monkeypatch)


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
