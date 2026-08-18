import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import surfa as sf

from samseg.io import kvlReadSharedGMMParameters
from samseg.merge_alphas import kvlGetMergingFractionsTable
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.model_policy import SubregionModelPolicy
from samseg.subregions import thalamus_plus
from samseg.subregions.thalamus_plus import ThalamicNucleiPlus


TEST_DATA_DIR = Path(__file__).parent / 'data' / 'subfields_plus'
MODEL_ARTIFACT_DIR = (
    Path(__file__).parents[1] / 'subregions' / 'for_testing'
    / 'model_artifacts' / 'thalamus_plus')

PARAMETER_FILES = {
    'aseg': MODEL_ARTIFACT_DIR / 'ASEGsharedGMMparameters.txt',
    'synthseg': MODEL_ARTIFACT_DIR / 'SYNTHSEGsharedGMMparameters.txt',
}
LOCALIZER_LUT_FILES = {
    'aseg': MODEL_ARTIFACT_DIR / 'ASEGlocalizerLookupTable.txt',
    'synthseg': MODEL_ARTIFACT_DIR / 'SYNTHSEGlocalizerLookupTable.txt',
}
POLICY_FILES = {
    'aseg': MODEL_ARTIFACT_DIR / 'ASEGmodelPolicy.json',
}

ATLAS_LUT_VOCABULARIES = (
    TEST_DATA_DIR / 'historical_thalamus_lut_names.txt',
    TEST_DATA_DIR / 'installed_dti_thalamus_lut_names.txt',
    TEST_DATA_DIR / 'matlab_dti_thalamus_lut_names.txt',
)

ASEG_LABELS = {
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51,
    52, 53, 54, 58, 60, 62, 63, 72, 77, 80, 85,
    251, 252, 253, 254, 255,
}
SYNTHSEG_BASE_LABELS = {
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54,
    58, 60,
}
SYNTHSEG_PARCELLATION_LABELS = (
    {1001, 1002, 1003, *range(1005, 1036)}
    | {2001, 2002, 2003, *range(2005, 2036)})

EXPECTED_TARGETS = {
    'aseg': {
        'Unknown': 1,
        'CrbrlWM': 2,
        'CrbrlCrtx': 3,
        'CrblmCrtx': 8,
        'CrblmWM': 7,
        'BrainStem': 16,
        'Ventricle': 4,
        'Chrd': 31,
        'Putamen': 12,
        'Pallidum': 13,
        'Accmbns': 26,
        'Caudate': 11,
        'LeftThalamus': 10,
        'RightThalamus': 49,
    },
    'synthseg': {
        'Unknown': 1,
        'CrbrlWM': 2,
        'CrbrlCrtx': 3,
        'CrblmCrtx': 8,
        'CrblmWM': 7,
        'BrainStem': 16,
        'Ventricle': 4,
        'Putamen': 12,
        'Pallidum': 13,
        'Accmbns': 26,
        'Caudate': 11,
        'LeftThalamus': 10,
        'RightThalamus': 49,
    },
}


class _Segmentation:

    def __init__(self, labels):
        self.data = np.asarray(labels)

    def new(self, data):
        return _Segmentation(data)


def _model(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv('FREESURFER_HOME', str(tmp_path / 'freesurfer'))
    arguments = {
        'outDir': str(tmp_path / 'output'),
        'inputImageFileNames': [str(tmp_path / 't1.mgz')],
        'inputSegFileName': str(tmp_path / 'segmentation.mgz'),
    }
    arguments.update(overrides)
    return ThalamicNucleiPlus(**arguments)


def _read_names(path):
    return [
        line for line in path.read_text().splitlines()
        if line and not line.startswith('#')
    ]


def _load_groups(schema):
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES[schema])
    policy = (
        SubregionModelPolicy.read(POLICY_FILES[schema])
        if schema in POLICY_FILES else SubregionModelPolicy())
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES[schema])
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy
    return parameters, model._build_preliminary_localizer_label_groups(
        parameters, lookupTable)


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
@pytest.mark.parametrize('lutFile', ATLAS_LUT_VOCABULARIES)
def test_universal_preliminary_artifacts_cover_audited_atlas_luts_once(
        schema, lutFile):
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES[schema])
    assert [parameter.mergedName for parameter in parameters] == list(
        EXPECTED_TARGETS[schema])
    with contextlib.redirect_stdout(io.StringIO()):
        fractions, _ = kvlGetMergingFractionsTable(
            _read_names(lutFile), parameters)

    np.testing.assert_array_equal(
        np.count_nonzero(fractions, axis=0),
        np.ones(fractions.shape[1], dtype=int))


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_new_thalamus_families_use_compact_nonreticular_prefixes(schema):
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES[schema])
    byClass = {
        parameter.mergedName: parameter.searchStrings
        for parameter in parameters
    }

    for side in ('Left', 'Right'):
        familyTokens = {
            f'{side}-ThalNuc-{initial}'
            for initial in 'ACLMPV'
        }
        configuredTokens = {
            token for token in byClass[f'{side}Thalamus']
            if token.startswith(f'{side}-ThalNuc-')
        }
        assert configuredTokens == familyTokens
        assert f'{side}-ThalNuc-R' not in configuredTokens


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
@pytest.mark.parametrize('lutFile', ATLAS_LUT_VOCABULARIES[1:])
def test_new_reticular_labels_remain_exclusively_white_matter(
        schema, lutFile):
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES[schema])
    names = _read_names(lutFile)
    with contextlib.redirect_stdout(io.StringIO()):
        fractions, classNames = kvlGetMergingFractionsTable(
            names, parameters)
    classNumbers = {
        className: classNumber
        for classNumber, className in enumerate(classNames)
    }

    for side in ('Left', 'Right'):
        structureNumber = names.index(f'{side}-ThalNuc-R')
        assert fractions[
            classNumbers['CrbrlWM'], structureNumber] == 1
        assert fractions[
            classNumbers[f'{side}Thalamus'], structureNumber] == 0


@pytest.mark.parametrize(
    ('schema', 'expectedLabels'),
    [
        ('aseg', ASEG_LABELS),
        ('synthseg',
         SYNTHSEG_BASE_LABELS | SYNTHSEG_PARCELLATION_LABELS),
    ],
)
def test_localizer_artifact_is_the_bounded_model_vocabulary(
        schema, expectedLabels):
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES[schema])

    assert set(lookupTable) == expectedLabels


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_localizer_vocabulary_derives_established_targets(schema):
    parameters, groups = _load_groups(schema)
    classNames = [parameter.mergedName for parameter in parameters]
    allLabels = [label for group in groups for label in group]

    assert len(allLabels) == len(set(allLabels))
    assert set(allLabels) == set(
        sf.load_label_lookup(LOCALIZER_LUT_FILES[schema]))
    assert [max(1, min(labels)) for labels in groups] == [
        EXPECTED_TARGETS[schema][name] for name in classNames]


def test_policy_contains_only_irreducible_exact_memberships():
    aseg = SubregionModelPolicy.read(POLICY_FILES['aseg'])

    assert aseg.preliminaryLocalizerLabelMemberships == {
        'CrbrlWM': (77,),
    }
    assert not (MODEL_ARTIFACT_DIR / 'SYNTHSEGmodelPolicy.json').exists()


def test_thalamus_plus_accepts_atlas_override_without_changing_default(
        tmp_path, monkeypatch):
    defaultModel = _model(tmp_path, monkeypatch)
    explicitAtlas = tmp_path / 'selected-atlas'
    selectedModel = _model(
        tmp_path, monkeypatch,
        atlasDir=str(explicitAtlas),
        preliminaryModelDirectory=str(MODEL_ARTIFACT_DIR))

    assert defaultModel.atlasDir == str(
        tmp_path / 'freesurfer' / 'average' / 'ThalamicNuclei' / 'atlas')
    assert defaultModel.preliminaryModelDirectory == defaultModel.atlasDir
    assert selectedModel.atlasDir == str(explicitAtlas)
    assert selectedModel.preliminaryModelDirectory == str(MODEL_ARTIFACT_DIR)
    assert set(selectedModel.preliminaryModelProfiles) == {
        'aseg', 'synthseg'}


def test_thalamus_plus_validates_schema_override(tmp_path, monkeypatch):
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(MODEL_ARTIFACT_DIR),
        inputSegmentationSchema='unsupported')
    model.inputSeg = _Segmentation([0, 2, 3])

    with pytest.raises(
            ValueError,
            match='Unknown preliminary model profile.*unsupported'):
        model._configure_preliminary_model_profile(
            model.preliminaryModelProfiles,
            requestedProfileName=model.inputSegmentationSchemaOverride)


def test_thalamus_profile_selection_configures_atomic_artifact_set(
        tmp_path, monkeypatch):
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(MODEL_ARTIFACT_DIR),
        inputSegmentationSchema='synthseg')
    model.inputSeg = _Segmentation([0, 2, 3])

    selected = model._configure_preliminary_model_profile(
        model.preliminaryModelProfiles,
        requestedProfileName=model.inputSegmentationSchemaOverride)

    assert selected == 'synthseg'
    assert model.preliminaryModelProfileName == 'synthseg'
    assert model.preliminarySharedGMMParametersFileName == str(
        PARAMETER_FILES['synthseg'])
    assert model.preliminaryLocalizerLookupTableFileName == str(
        LOCALIZER_LUT_FILES['synthseg'])
    assert model.modelPolicyFileName is None


@pytest.mark.parametrize(
    ('missingName', 'expected'),
    [
        ('ASEGsharedGMMparameters.txt', 'shared-GMM parameter'),
        ('ASEGlocalizerLookupTable.txt', 'localizer lookup table'),
        ('ASEGmodelPolicy.json', 'model policy'),
    ],
)
def test_missing_preliminary_artifact_fails_with_selected_path(
        tmp_path, monkeypatch, missingName, expected):
    modelDirectory = tmp_path / 'model'
    modelDirectory.mkdir()
    sources = (
        *PARAMETER_FILES.values(),
        *LOCALIZER_LUT_FILES.values(),
        *POLICY_FILES.values(),
    )
    for source in sources:
        if source.name != missingName:
            (modelDirectory / source.name).write_text(source.read_text())
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(modelDirectory),
        inputSegmentationSchema='aseg')
    model.inputSeg = _Segmentation([0, 2, 3])

    with pytest.raises(ValueError, match=expected):
        model._configure_preliminary_model_profile(
            model.preliminaryModelProfiles,
            requestedProfileName=model.inputSegmentationSchemaOverride)


def test_cross_schema_policy_fails_against_selected_vocabulary():
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES['synthseg'])
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES['synthseg'])
    asegPolicy = SubregionModelPolicy.read(POLICY_FILES['aseg'])
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = asegPolicy

    with pytest.raises(ValueError, match='absent.*77'):
        model._build_preliminary_localizer_label_groups(
            parameters, lookupTable)


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_generic_builder_uses_fixed_groups_without_mutating_localizer(
        tmp_path, monkeypatch, schema):
    model = _model(tmp_path, monkeypatch)
    parameters, groups = _load_groups(schema)
    model.preliminarySharedGMMParameters = parameters
    model.preliminaryLocalizerLabelGroups = groups
    monkeypatch.setattr(model, '_ensure_preliminary_model_state', lambda: None)

    representatives = [min(labels) for labels in groups]
    segmentation = _Segmentation(representatives)
    original = segmentation.data.copy()

    synthetic = model._build_preliminary_synthetic_image(segmentation)

    np.testing.assert_array_equal(segmentation.data, original)
    np.testing.assert_array_equal(
        synthetic.data,
        [EXPECTED_TARGETS[schema][parameter.mergedName]
         for parameter in parameters])


def test_builder_rejects_label_outside_selected_vocabulary(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    parameters, groups = _load_groups('aseg')
    model.preliminarySharedGMMParameters = parameters
    model.preliminaryLocalizerLabelGroups = groups
    monkeypatch.setattr(model, '_ensure_preliminary_model_state', lambda: None)

    with pytest.raises(ValueError, match='outside.*136'):
        model._build_preliminary_synthetic_image(
            _Segmentation([0, 2, 136]))


def test_cheating_means_do_not_depend_on_subject_observed_labels(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    _, groups = _load_groups('aseg')
    model.preliminaryLocalizerLabelGroups = groups

    completeMeans, _ = model.get_cheating_gaussians(groups)
    observedSubset = _Segmentation([0, 2, 4, 10, 49])
    monkeypatch.setattr(model, '_ensure_preliminary_model_state', lambda: None)
    model._build_preliminary_synthetic_image(observedSubset)
    subsetMeans, _ = model.get_cheating_gaussians(groups)

    np.testing.assert_array_equal(subsetMeans, completeMeans)


def test_global_lut_entries_cannot_change_selected_vocabulary_means(
        tmp_path):
    parameters, groups = _load_groups('aseg')
    policy = SubregionModelPolicy.read(POLICY_FILES['aseg'])
    extendedLut = tmp_path / 'globalFreeSurferLUT.txt'
    extendedLut.write_text(
        LOCALIZER_LUT_FILES['aseg'].read_text()
        + '1 Left-Cerebral-White-Matter 0 0 0 1\n')

    selectedMeans = [max(1, min(labels)) for labels in groups]
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy
    globalGroups = model._build_preliminary_localizer_label_groups(
        parameters, sf.load_label_lookup(extendedLut))

    assert [max(1, min(labels)) for labels in globalGroups] != selectedMeans
    assert 1 not in {label for group in groups for label in group}


def test_aparc_aseg_provenance_does_not_expand_aseg_model_vocabulary(
        tmp_path, monkeypatch):
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(MODEL_ARTIFACT_DIR),
        inputSegFileName=str(tmp_path / 'aparc+aseg.mgz'))
    model.inputSeg = _Segmentation([0, 2, 1001])

    with pytest.raises(ValueError, match="outside.*aseg.*1001"):
        model._configure_preliminary_model_profile(
            model.preliminaryModelProfiles)


def test_aseg_vocabulary_reproduces_legacy_supported_targets():
    parameters, groups = _load_groups('aseg')
    targets = {
        label: EXPECTED_TARGETS['aseg'][parameter.mergedName]
        for parameter, labels in zip(parameters, groups)
        for label in labels
    }

    assert set(targets) == ASEG_LABELS
    assert targets[0] == 1
    assert all(targets[label] == 2 for label in (30, 62, 77, 251, 255))
    assert all(targets[label] == 4 for label in (5, 14, 15, 24, 44, 72))
    assert all(targets[label] == 3 for label in (17, 18, 42, 53, 54))
    assert targets[28] == 10
    assert targets[60] == 49
    assert targets[63] == 31
    assert targets[80] == 1
    assert targets[85] == 1


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_thalamus_labels_are_inferred_from_shared_search_strings(schema):
    parameters, groups = _load_groups(schema)
    memberships = {
        parameter.mergedName: labels
        for parameter, labels in zip(parameters, groups)
    }

    assert memberships['LeftThalamus'] == [10, 28]
    assert memberships['RightThalamus'] == [49, 60]


def test_thalamus_affine_support_is_derived_from_profile_groups(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    parameters, groups = _load_groups('aseg')
    model.preliminaryClassNames = [
        parameter.mergedName for parameter in parameters]
    model.preliminaryLocalizerLabelGroups = groups

    assert model._get_preliminary_affine_support_labels() == [10, 28, 49, 60]
    assert not hasattr(model, 'DElabelLeft')
    assert not hasattr(model, 'DElabelRight')


def test_thalamus_plus_delegates_static_preliminary_policy_to_base():
    assert 'get_cheating_label_groups' not in ThalamicNucleiPlus.__dict__
    assert 'get_cheating_gaussians' not in ThalamicNucleiPlus.__dict__
    assert '_recode_preliminary_segmentation' not in (
        ThalamicNucleiPlus.__dict__)


def test_thalamus_hyperparameters_use_fitted_atlas_reconstruction(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    model.resolution = 1.0
    model.inputImages = [sf.Volume(np.array(
        [[[20.0]], [[100.0]]], dtype='float32'))]
    model.inputSeg = sf.Volume(np.array(
        [[[8101]], [[28]]], dtype='int32'))
    model.structuralInitializationSegmentation = sf.Volume(np.array(
        [[[28]], [[8101]]], dtype='int32'))
    model.structuralInitializationMask = sf.Volume(np.ones(
        (2, 1, 1), dtype='bool'))
    monkeypatch.setattr(
        thalamus_plus.scipy.ndimage.morphology,
        'binary_erosion',
        lambda mask, *args, **kwargs: mask)

    means, strengths = model.get_gaussian_hyps(
        [[28, 60], [8101, 8201]], mesh=None)

    np.testing.assert_array_equal(means, [20.0, 100.0])
    np.testing.assert_array_equal(strengths, [10.0, 11.0])


def test_thalamus_hyperparameters_require_post_fit_reconstruction(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match='fit_mesh_to_seg'):
        model.get_gaussian_hyps([[28, 60]], mesh=None)
