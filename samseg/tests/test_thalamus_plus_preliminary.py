import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
import surfa as sf

from samseg.io import GMMparameter, kvlReadSharedGMMParameters
from samseg.merge_alphas import (
    kvlGetMergingFractionsTable,
    kvlResolveSharedGMMParameters,
)
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.model_policy import SubregionModelPolicy
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
POLICY_FILE = MODEL_ARTIFACT_DIR / 'modelPolicy.json'
STRUCTURAL_PARAMETER_FILE = (
    MODEL_ARTIFACT_DIR / 'atlas' / 'multiResolutionLevel1'
    / 'sharedGMMparameters.txt')

# Captured label-name inventories from the historical structural atlas, the
# installed DTI atlas, and the mature MATLAB DTI atlas. Shared parameters must
# assign every listed structure to exactly one preliminary class.
ATLAS_LUT_VOCABULARIES = (
    TEST_DATA_DIR / 'historical_thalamus_lut_names.txt',
    TEST_DATA_DIR / 'installed_dti_thalamus_lut_names.txt',
    TEST_DATA_DIR / 'matlab_dti_thalamus_lut_names.txt',
)

# Exact input-label vocabularies declared by the profile-local lookup tables.
# Keeping them explicit catches unintended expansion from a global LUT.
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

# Synthetic output label for each coarse shared-parameter class. These are
# model-artifact targets, not labels inferred from a particular subject.
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
    policy = SubregionModelPolicy.read(POLICY_FILE)
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES[schema])
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy
    model.modelPolicyFileName = None
    model.preliminaryModelProfileName = schema
    return parameters, model._build_preliminary_localizer_label_groups(
        parameters, lookupTable)


def test_shared_gmm_resolution_preserves_sparse_lut_membership_semantics():
    """Resolve suffixes and singletons without normalizing row overlap."""
    names = ['Left-VA', 'Left-VAmc', 'Right-VA', 'Other', 'Bridge']
    parameters = [
        GMMparameter('ExactVA', 2, ["VA'"]),
        GMMparameter('LeftFamily', 1, ['Left-']),
        GMMparameter('Unused', 3, ['Missing']),
    ]

    resolved, memberships = kvlResolveSharedGMMParameters(
        names, parameters)

    assert [parameter.mergedName for parameter in resolved] == [
        'ExactVA', 'LeftFamily', 'Other', 'Bridge']
    assert [parameter.numberOfComponents for parameter in resolved] == [
        2, 1, 1, 1]
    np.testing.assert_array_equal(memberships, [
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ])


def test_merging_fractions_treat_trailing_apostrophe_as_literal_text():
    """Preserve maintained SAMSEG's substring-only matching contract."""
    names = ['Left-VA', "Literal-VA'"]
    parameters = [
        GMMparameter('Ordinary', 1, ['Left-VA']),
        GMMparameter('Literal', 1, ["VA'"]),
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        fractions, _ = kvlGetMergingFractionsTable(names, parameters)

    np.testing.assert_array_equal(fractions, [
        [1.0, 0.0],
        [0.0, 1.0],
    ])


def test_level_one_structural_parameters_reproduce_the_established_partition():
    """Protect the first-pass model while replacing handwritten grouping."""
    names = _read_names(ATLAS_LUT_VOCABULARIES[0])
    model = object.__new__(MeshModelPlus)
    model.atlasDir = str(STRUCTURAL_PARAMETER_FILE.parent)
    model.gmmFileName = str(STRUCTURAL_PARAMETER_FILE)
    model.names = names

    model._configure_shared_gmm_parameters()
    resolved = model.sharedGMMParameters
    memberships = model.classFractions.astype(bool)

    expectedSingletons = [
        'Unknown',
        'Left-Cerebral-Cortex',
        'Left-Cerebellum-Cortex',
        'Left-Cerebellum-White-Matter',
        'Brain-Stem',
        'Left-Lateral-Ventricle',
        'Left-choroid-plexus',
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Accumbens-area',
        'Left-Caudate',
    ]
    assert [parameter.mergedName for parameter in resolved] == [
        'CerebralWM', 'VentralDC', 'Thalamus', *expectedSingletons]
    np.testing.assert_array_equal(
        np.count_nonzero(memberships, axis=0),
        np.ones(len(names), dtype=int))

    structuresByClass = {
        parameter.mergedName: {
            names[structureNumber]
            for structureNumber in np.flatnonzero(memberships[classNumber])
        }
        for classNumber, parameter in enumerate(resolved)
    }
    assert structuresByClass['CerebralWM'] == {
        'Left-Cerebral-White-Matter', 'Left-R', 'Right-R'}
    assert structuresByClass['VentralDC'] == {
        'Left-VentralDC', 'Right-VentralDC'}
    thalamicStructureNames = [
        'L-Sg', 'LGN', 'MGN', 'PuI', 'PuM', 'H', 'PuL', 'VPI', 'PuA',
        'MV(Re)', 'Pf', 'CM', 'LP', 'VLa', 'VPL', 'VLp', 'MDm', 'VM',
        'CeM', 'MDl', 'Pc', 'MDv', 'Pv', 'CL', 'VA', 'VPM', 'AV',
        'VAmc', 'Pt', 'AD', 'LD',
    ]
    expectedThalamicStructures = {
        f'{side}-{structureName}'
        for side in ('Left', 'Right')
        for structureName in thalamicStructureNames
    }
    assert structuresByClass['Thalamus'] == (
        expectedThalamicStructures.intersection(names))
    for name in expectedSingletons:
        assert structuresByClass[name] == {name}


def test_plus_rejects_structural_models_it_cannot_yet_interpret(
        tmp_path):
    """Cross-row overlap must fail before unsupported Plus activation."""
    parameterFile = tmp_path / 'sharedGMMparameters.txt'
    parameterFile.write_text('First 1 Tissue\nSecond 1 Tissue\n')
    model = object.__new__(MeshModelPlus)
    model.atlasDir = str(tmp_path)
    model.gmmFileName = str(parameterFile)
    model.names = ['Tissue']

    with pytest.raises(NotImplementedError, match='multiple parameter rows'):
        model._configure_shared_gmm_parameters()


def test_plus_accepts_disjoint_multicomponent_structural_class(tmp_path):
    parameterFile = tmp_path / 'sharedGMMparameters.txt'
    parameterFile.write_text('Tissue 2 Tissue\n')
    model = object.__new__(MeshModelPlus)
    model.atlasDir = str(tmp_path)
    model.gmmFileName = str(parameterFile)
    model.names = ['Tissue']

    model._configure_shared_gmm_parameters()

    assert model.sharedGMMParameters[0].numberOfComponents == 2
    np.testing.assert_array_equal(model.classFractions, [[1.0]])


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
@pytest.mark.parametrize('lutFile', ATLAS_LUT_VOCABULARIES)
def test_preliminary_profiles_define_expected_classes_and_map_each_atlas_label_once(
        schema, lutFile):
    """Check class order and unique coverage across supported atlas names.

    A failure indicates missing or overlapping shared-parameter search strings,
    or an unintended change to the preliminary class definitions.
    """
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
def test_thalamus_classes_match_aclmpv_nuclei_but_exclude_reticular_nuclei(
        schema):
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES[schema])
    byClass = {
        parameter.mergedName: parameter.searchStrings
        for parameter in parameters
    }

    for side in ('Left', 'Right'):
        # A/C/L/M/P/V are the compact prefixes present in the supported DTI
        # thalamic-nucleus vocabularies. R is reticular and belongs to white
        # matter.
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
def test_reticular_nucleus_labels_map_to_white_matter_not_thalamus(
        schema, lutFile):
    """Reticular nuclei belong to cerebral white matter in both DTI vocabularies.

    A failure means a reticular label would enter the coarse thalamus class.
    """
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
def test_localizer_lookup_tables_contain_exact_profile_label_sets(
        schema, expectedLabels):
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES[schema])

    assert set(lookupTable) == expectedLabels


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_localizer_groups_partition_profile_labels_and_use_expected_targets(
        schema):
    parameters, groups = _load_groups(schema)
    classNames = [parameter.mergedName for parameter in parameters]
    allLabels = [label for group in groups for label in group]

    assert len(allLabels) == len(set(allLabels))
    assert set(allLabels) == set(
        sf.load_label_lookup(LOCALIZER_LUT_FILES[schema]))
    # Synthetic recoding uses the minimum class member as its target, except
    # that an Unknown class containing label 0 is represented by nonzero 1.
    assert [max(1, min(labels)) for labels in groups] == [
        EXPECTED_TARGETS[schema][name] for name in classNames]


def test_default_thalamus_policy_defines_profile_memberships_and_initialization_values():
    """Record model-policy values consumed before the first GMM is constructed."""
    policy = SubregionModelPolicy.read(POLICY_FILE)

    # Label 77 is ASEG WM-hypointensities; it has no SynthSeg membership.
    assert policy.get_preliminary_localizer_label_memberships('aseg') == {
        'CrbrlWM': (77,),
    }
    assert policy.get_preliminary_localizer_label_memberships('synthseg') == {}
    assert policy.affineTargetMorphology == 'opening'
    assert policy.localizerAnatomicalSupportMarginInMm == 1.5
    assert policy.preliminaryAtlasDomainInteriorMarginInMm == 3.0
    assert policy.regionalAtlasDomainInteriorMarginInMm == 2.0
    assert policy.initialGMMCovarianceFallback == (
        'regional_fitting_covariance')
    assert policy.maximumGMMIterations == 100
    assert policy.zeroEvidenceInitialization.strategy == (
        'subject_non_background_median')
    # Strength 10 belongs to the zero-evidence subject-median fallback, not the
    # historical fixed VDC strength.
    assert policy.zeroEvidenceInitialization.strength == 10.0


def test_thalamus_plus_accepts_atlas_override_without_changing_default(
        tmp_path, monkeypatch):
    defaultModel = _model(tmp_path, monkeypatch)
    # Deliberately separate the mesh-atlas and preliminary-model directories to
    # show that atlas selection does not redirect the localizer artifacts.
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


def test_thalamus_profile_selection_selects_artifacts_and_shared_policy(
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
    assert model.modelPolicyFileName == str(POLICY_FILE)


@pytest.mark.parametrize(
    ('missingName', 'expected'),
    [
        ('ASEGsharedGMMparameters.txt', 'shared-GMM parameter'),
        ('ASEGlocalizerLookupTable.txt', 'localizer lookup table'),
    ],
)
def test_missing_preliminary_artifact_fails_with_selected_path(
        tmp_path, monkeypatch, missingName, expected):
    modelDirectory = tmp_path / 'model'
    modelDirectory.mkdir()
    sources = (
        *PARAMETER_FILES.values(),
        *LOCALIZER_LUT_FILES.values(),
        POLICY_FILE,
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


def test_missing_model_policy_file_is_reported_when_policy_is_loaded(
        tmp_path, monkeypatch):
    modelDirectory = tmp_path / 'model'
    modelDirectory.mkdir()
    for source in (*PARAMETER_FILES.values(), *LOCALIZER_LUT_FILES.values()):
        (modelDirectory / source.name).write_text(source.read_text())
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(modelDirectory))

    with pytest.raises(ValueError, match='model policy file does not exist'):
        model._ensure_model_policy()


def test_aseg_only_label_77_is_not_added_to_synthseg_groups():
    # Label 77 exists only in the ASEG localizer LUT, where policy assigns
    # WM-hypointensities to the cerebral-white-matter preliminary class.
    parameters = kvlReadSharedGMMParameters(PARAMETER_FILES['synthseg'])
    lookupTable = sf.load_label_lookup(LOCALIZER_LUT_FILES['synthseg'])
    policy = SubregionModelPolicy.read(POLICY_FILE)
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy
    model.modelPolicyFileName = None
    model.preliminaryModelProfileName = 'synthseg'

    groups = model._build_preliminary_localizer_label_groups(
        parameters, lookupTable)

    assert 77 not in {label for group in groups for label in group}


@pytest.mark.parametrize('schema', ['aseg', 'synthseg'])
def test_synthetic_preliminary_image_uses_profile_targets_without_changing_input(
        tmp_path, monkeypatch, schema):
    model = _model(tmp_path, monkeypatch)
    parameters, groups = _load_groups(schema)
    model.preliminarySharedGMMParameters = parameters
    model.preliminaryLocalizerLabelGroups = groups
    monkeypatch.setattr(model, '_ensure_preliminary_model_state', lambda: None)

    # One representative from every configured class makes each recoding
    # target observable without depending on labels present in a real subject.
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

    # Label 136 is deliberately absent from the selected ASEG localizer LUT.
    with pytest.raises(ValueError, match='outside.*136'):
        model._build_preliminary_synthetic_image(
            _Segmentation([0, 2, 136]))


def test_preliminary_gaussian_means_do_not_depend_on_subject_labels(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    _, groups = _load_groups('aseg')
    model.preliminaryLocalizerLabelGroups = groups

    completeMeans, _ = model.get_cheating_gaussians(groups)
    # This subject subset omits most configured classes. Model-defined means
    # must nevertheless remain available for the complete profile.
    observedSubset = _Segmentation([0, 2, 4, 10, 49])
    monkeypatch.setattr(model, '_ensure_preliminary_model_state', lambda: None)
    model._build_preliminary_synthetic_image(observedSubset)
    subsetMeans, _ = model.get_cheating_gaussians(groups)

    np.testing.assert_array_equal(subsetMeans, completeMeans)


def test_global_lut_entries_cannot_change_selected_vocabulary_means(
        tmp_path):
    parameters, groups = _load_groups('aseg')
    policy = SubregionModelPolicy.read(POLICY_FILE)
    extendedLut = tmp_path / 'globalFreeSurferLUT.txt'
    # Label 1 matches cerebral WM and would lower that class's synthetic target
    # from 2 to 1, but it is absent from the profile-local LUT.
    extendedLut.write_text(
        LOCALIZER_LUT_FILES['aseg'].read_text()
        + '1 Left-Cerebral-White-Matter 0 0 0 1\n')

    selectedMeans = [max(1, min(labels)) for labels in groups]
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = policy
    model.modelPolicyFileName = None
    model.preliminaryModelProfileName = 'aseg'
    globalGroups = model._build_preliminary_localizer_label_groups(
        parameters, sf.load_label_lookup(extendedLut))

    assert [max(1, min(labels)) for labels in globalGroups] != selectedMeans
    assert 1 not in {label for group in groups for label in group}


def test_aparc_aseg_provenance_does_not_expand_aseg_model_vocabulary(
        tmp_path, monkeypatch):
    # Label 1001 is a cortical parcellation label accepted by the SynthSeg
    # profile but absent from the ASEG localizer LUT. The filename must not
    # broaden the selected ASEG profile.
    model = _model(
        tmp_path, monkeypatch,
        preliminaryModelDirectory=str(MODEL_ARTIFACT_DIR),
        inputSegFileName=str(tmp_path / 'aparc+aseg.mgz'))
    model.inputSeg = _Segmentation([0, 2, 1001])

    with pytest.raises(ValueError, match="outside.*aseg.*1001"):
        model._configure_preliminary_model_profile(
            model.preliminaryModelProfiles)


def test_aseg_vocabulary_reproduces_legacy_supported_targets():
    """Preserve the structural localizer's coarse recoding across label families.

    The assertions sample white-matter, ventricular, cortical, bilateral
    VDC/thalamus, choroid, and Unknown groups. A failure points to changed
    shared search-string or policy-membership semantics.
    """
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

    # Labels 10/49 are left/right thalamus and 28/60 are left/right VDC; both
    # pairs belong to the corresponding coarse thalamus class.
    assert memberships['LeftThalamus'] == [10, 28]
    assert memberships['RightThalamus'] == [49, 60]


def test_thalamus_affine_support_is_derived_from_profile_groups(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)
    parameters, groups = _load_groups('aseg')
    model.preliminaryClassNames = [
        parameter.mergedName for parameter in parameters]
    model.preliminaryLocalizerLabelGroups = groups

    # Affine support includes both anatomical label families assigned to the
    # coarse classes: thalamus 10/49 and VDC 28/60.
    assert model._get_preliminary_affine_support_labels() == [10, 28, 49, 60]


def test_default_thalamus_vdc_hyperparameters_use_fitted_whole_field_evidence(
        tmp_path, monkeypatch):
    """Use fitted whole-field evidence rather than raw localizer labels.

    Raw and fitted labels are deliberately swapped: 20/200 must initialize VDC
    through fitted label 28, while 100/1000 initialize the [8101, 8201] nucleus
    class. A failure implicates evidence selection or first-stage statistics.
    """
    model = _model(tmp_path, monkeypatch)
    model.modelPolicy = SubregionModelPolicy()
    model.resolution = 1.0
    model.intensityPriorImage = sf.Volume(np.array(
        [[[[20.0, 200.0]]], [[[100.0, 1000.0]]]], dtype='float32'))
    model.workingImage = sf.Volume(np.ones(
        (2, 1, 1, 2), dtype='float32'))
    model.inputSeg = sf.Volume(np.array(
        [[[8101]], [[28]]], dtype='int32'))
    model.intensityPriorInitializationSegmentation = sf.Volume(np.array(
        [[[28]], [[8101]]], dtype='int32'))
    model.intensityPriorInitializationMask = sf.Volume(np.ones(
        (2, 1, 1), dtype='bool'))

    means, strengths = model.get_gaussian_hyps(
        [[28, 60], [8101, 8201]], mesh=None)

    # Labels 28/60 are VDC. Strength 11 comes from its one fitted-support
    # voxel, rather than the historical fixed strength 10.
    np.testing.assert_array_equal(
        means, [[20.0, 200.0], [100.0, 1000.0]])
    np.testing.assert_array_equal(strengths, [11.0, 11.0])


def test_thalamus_hyperparameters_require_post_fit_reconstruction(
        tmp_path, monkeypatch):
    model = _model(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match='prepare_for_image_fitting'):
        model.get_gaussian_hyps([[28, 60]], mesh=None)
