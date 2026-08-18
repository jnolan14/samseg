import ast
import inspect
import textwrap

from samseg.subregions.core import MeshModel
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.thalamus import ThalamicNuclei
from samseg.subregions.thalamus_plus import ThalamicNucleiPlus


def _self_attributes(method):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == 'self'
    }


def test_thalamic_nuclei_plus_preserves_legacy_constructor_state(
        tmp_path, monkeypatch):
    freesurfer_home = tmp_path / 'freesurfer'
    monkeypatch.setenv('FREESURFER_HOME', str(freesurfer_home))
    arguments = {
        'outDir': str(tmp_path / 'output'),
        'inputImageFileNames': [str(tmp_path / 't1.mgz')],
        'inputSegFileName': str(tmp_path / 'segmentation.mgz'),
    }

    legacy = ThalamicNuclei(**arguments)
    successor = ThalamicNucleiPlus(**arguments)

    assert ThalamicNuclei.__bases__ == (MeshModel,)
    assert ThalamicNucleiPlus.__bases__ == (MeshModelPlus,)
    assert not issubclass(MeshModelPlus, MeshModel)

    legacy_state = legacy.__dict__
    successor_state = successor.__dict__
    assert {name: successor_state[name] for name in legacy_state} == legacy_state
    assert set(successor_state) - set(legacy_state) == {
        'bootstrapGMMState',
        'cheatingMeans',
        'cheatingVariances',
        'gmm',
        'inputSegmentationSchemaOverride',
        'intensityPriorImage',
        'intensityPriorReferenceImage',
        'lastValidFittedGMMState',
        'modelPolicy',
        'modelPolicyFileName',
        'optimizationHistory',
        'preliminaryAlphas',
        'preliminaryClassFractions',
        'preliminaryClassNames',
        'preliminaryLocalizerLabelGroups',
        'preliminaryLocalizerLookupTableFileName',
        'preliminaryModelDirectory',
        'preliminaryModelProfileName',
        'preliminaryModelProfiles',
        'preliminarySharedGMMParameters',
        'preliminarySharedGMMParametersFileName',
        'initializationMask',
        'initializationSegmentation',
        'intensityStage',
    }


def test_preliminary_gaussians_are_isolated_from_intensity_state():
    preliminary_attributes = (
        _self_attributes(MeshModelPlus.prepare_for_seg_fitting)
        | _self_attributes(MeshModelPlus.fit_mesh_to_seg)
    )

    assert {'cheatingMeans', 'cheatingVariances'} <= preliminary_attributes
    assert not {'means', 'variances'} & preliminary_attributes


def test_successor_declares_intensity_lifecycle_state(tmp_path, monkeypatch):
    monkeypatch.setenv('FREESURFER_HOME', str(tmp_path / 'freesurfer'))
    model = ThalamicNucleiPlus(
        outDir=str(tmp_path / 'output'),
        inputImageFileNames=[str(tmp_path / 't1.mgz')],
        inputSegFileName=str(tmp_path / 'segmentation.mgz'),
    )

    assert model.cheatingMeans is None
    assert model.cheatingVariances is None
    assert model.inputSegmentationSchemaOverride is None
    assert model.preliminaryModelProfileName is None
    assert model.preliminaryModelDirectory == model.atlasDir
    assert set(model.preliminaryModelProfiles) == {'aseg', 'synthseg'}
    assert model.preliminarySharedGMMParametersFileName is None
    assert model.preliminarySharedGMMParameters is None
    assert model.preliminaryClassFractions is None
    assert model.preliminaryClassNames is None
    assert model.modelPolicyFileName is None
    assert model.modelPolicy is None
    assert model.preliminaryLocalizerLookupTableFileName is None
    assert model.preliminaryLocalizerLabelGroups is None
    assert model.preliminaryAlphas is None
    assert model.initializationSegmentation is None
    assert model.initializationMask is None
    assert model.intensityPriorReferenceImage is None
    assert model.intensityPriorImage is None
    assert model.intensityStage is None
    assert model.gmm is None
    assert model.bootstrapGMMState is None
    assert model.lastValidFittedGMMState is None
    assert model.optimizationHistory == []
