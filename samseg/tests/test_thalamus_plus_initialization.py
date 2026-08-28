import numpy as np
import pytest
import surfa as sf

from samseg.io import GMMparameter
from samseg.subregions import core_plus
from samseg.subregions.core_plus import MeshModelPlus
from samseg.subregions.model_policy import SubregionModelPolicy
from samseg.subregions.model_policy import ZeroEvidenceInitializationPolicy
from samseg.subregions.thalamus_plus import ThalamicNucleiPlus


def _volume(data, voxsize=(1.0, 1.0, 1.0)):
    data = np.asarray(data)
    geometry = sf.ImageGeometry(
        data.shape[:3], voxsize=voxsize, center=(0.0, 0.0, 0.0))
    return sf.Volume(data, geometry=geometry)


def _merge_model(shape=(5, 1, 1)):
    # Labels 10/28 share a coarse thalamus class, while 8101 represents a
    # fitted nucleus. Label 77 deliberately has conflicting atlas/localizer
    # owners, so numeric equality cannot masquerade as semantic agreement.
    model = object.__new__(MeshModelPlus)
    model.intensityPriorImage = _volume(
        np.ones(shape + (2,), dtype='float32'))
    model.inputSeg = _volume(np.array(
        [[[2]], [[77]], [[28]], [[10]], [[0]]], dtype='int32'))
    model.synthImage = _volume(np.array(
        [[[2]], [[3]], [[10]], [[10]], [[1]]], dtype='float32'))
    model.FreeSurferLabels = np.array([0, 2, 77, 3, 10, 28, 8101])
    model.preliminaryClassFractions = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
    ], dtype='float32')
    model.preliminaryLocalizerLabelGroups = [
        [0], [2], [3, 77], [10, 28],
    ]
    model.cheatingMeans = np.array([1, 2, 3, 10], dtype='float32')
    model.modelPolicy = SubregionModelPolicy()
    model.modelPolicyFileName = None
    model.labelMapping = sf.LabelLookup()
    return model


def test_regional_initialization_excludes_fitted_labels_outside_em_support():
    # Bilateral VDC labels 28/60 flank fitted nucleus label 8101, which is
    # deliberately outside the EM mask and must disappear only from the
    # materialized regional state.
    model = object.__new__(MeshModelPlus)
    model.initializationSegmentation = _volume(np.array(
        [[[28]], [[8101]], [[60]]], dtype='int32'))
    model.initializationMask = _volume(np.ones((3, 1, 1), dtype='uint8'))
    model.workingImage = _volume(np.ones((3, 1, 1), dtype='float32'))
    model.workingMask = _volume(np.array(
        [[[1]], [[0]], [[1]]], dtype='uint8'))
    model.labelMapping = sf.LabelLookup()

    segmentation, support = (
        model._materialize_working_initialization_state())

    np.testing.assert_array_equal(
        segmentation.data[:, 0, 0], [28, 0, 60])
    np.testing.assert_array_equal(
        support.data[:, 0, 0], [1, 0, 1])
    np.testing.assert_array_equal(
        model.initializationSegmentation.data[:, 0, 0], [28, 8101, 60])


def test_whole_field_initialization_preserves_matching_localizer_labels_and_prioritizes_fitted_labels():
    """Combine compatible localizer evidence with higher-priority fitted labels.

    A failure points to whole-field merge precedence or atlas/localizer label
    ownership.
    """
    model = _merge_model()
    fittedSegmentation = _volume(np.array(
        [[[0]], [[0]], [[0]], [[8101]], [[0]]], dtype='int32'))
    fittedMask = _volume(np.array(
        [[[0]], [[0]], [[0]], [[1]], [[0]]], dtype='uint8'))
    originalLocalizer = model.inputSeg.data.copy()

    segmentation, support = (
        model._materialize_intensity_prior_initialization_state(
            fittedSegmentation, fittedMask))

    # Label 77 has conflicting atlas/localizer ownership, so collapsed class
    # evidence wins. Fitted anatomy wins inside fitted support.
    np.testing.assert_array_equal(
        segmentation.data[:, 0, 0], [2, 3, 28, 8101, 0])
    np.testing.assert_array_equal(
        support.data[:, 0, 0], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(model.inputSeg.data, originalLocalizer)


def test_localizer_label_without_atlas_identity_remains_a_collapsed_class():
    model = _merge_model()
    model.FreeSurferLabels = np.array([0, 2, 3, 10, 28, 8101])
    model.preliminaryClassFractions = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype='float32')
    fittedSegmentation = _volume(np.zeros((5, 1, 1), dtype='int32'))
    fittedMask = _volume(np.zeros((5, 1, 1), dtype='uint8'))

    segmentation, _ = (
        model._materialize_intensity_prior_initialization_state(
            fittedSegmentation, fittedMask))

    # Policy gives localizer label 77 semantic ownership of class 3, but no
    # atlas label 77 exists. The whole-field state therefore keeps class 3.
    assert segmentation.data[1, 0, 0] == 3


def test_whole_field_support_uses_localizer_halo_and_complete_cases():
    """Check the localizer margin and multichannel observation-validity rule.

    A failure points to whole-field support/background handling, not to
    class-specific hyperparameter erosion.
    """
    model = _merge_model()
    model.modelPolicy = SubregionModelPolicy(
        localizerAnatomicalSupportMarginInMm=1.5)
    # Label 10 at index 2 is anatomy; 80/85 belong to the profile's
    # Unknown/background class. Index 1 is a finite-negative observation
    # inside the 1.5 mm margin; index 0 is too far away, while indices 3/4
    # contain zero/NaN.
    model.preliminaryLocalizerLabelGroups[0] = [0, 80, 85]
    model.inputSeg = _volume(np.array(
        [[[80]], [[85]], [[10]], [[80]], [[85]]], dtype='int32'))
    model.intensityPriorImage.data[1, 0, 0, 1] = -2
    model.intensityPriorImage.data[3, 0, 0, 1] = 0
    model.intensityPriorImage.data[4, 0, 0, 1] = np.nan
    fittedSegmentation = _volume(np.zeros((5, 1, 1), dtype='int32'))
    fittedMask = _volume(np.zeros((5, 1, 1), dtype='uint8'))

    _, support = model._materialize_intensity_prior_initialization_state(
        fittedSegmentation, fittedMask)

    np.testing.assert_array_equal(
        support.data[:, 0, 0], [0, 1, 1, 0, 0])


@pytest.mark.parametrize(
    'morphology', ['opening', 'closing', 'none'],
)
def test_affine_target_opening_closing_and_none_produce_expected_masks(
        morphology):
    """Check mature opening, compatibility closing, and no-op mask results."""
    model = object.__new__(MeshModelPlus)
    model.modelPolicy = SubregionModelPolicy(
        affineTargetMorphology=morphology)
    if morphology == 'opening':
        support = np.zeros((5, 5, 5), dtype=bool)
        support[2, 2, 2] = True
        expected = np.zeros_like(support)
    elif morphology == 'closing':
        support = np.ones((5, 5, 5), dtype=bool)
        support[2, 2, 2] = False
        expected = np.ones_like(support)
    else:
        support = np.zeros((5, 5, 5), dtype=bool)
        support[1:4, 2, 2] = True
        expected = support.copy()
    original = support.copy()

    result = model._apply_affine_target_morphology(support)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(support, original)


def test_atlas_domain_margin_erodes_by_physical_distance_on_anisotropic_grid():
    model = object.__new__(MeshModelPlus)
    image = _volume(
        np.ones((13, 7, 5), dtype='float32'),
        voxsize=(0.5, 1.0, 2.0))
    support = np.zeros(image.shape, dtype=bool)
    support[1:12, 1:6, 1:4] = True

    # A 2 mm margin removes four 0.5-mm x voxels, two 1-mm y voxels, and one
    # 2-mm z voxel from each face, leaving only three central x positions.
    result = model._apply_atlas_domain_interior_margin(
        support, image, 2.0)

    np.testing.assert_array_equal(
        np.argwhere(result),
        [[5, 3, 2], [6, 3, 2], [7, 3, 2]])


def _refinement_model():
    # The first two labels are inside fitted support; the final background
    # voxel is deliberately outside it.
    model = object.__new__(MeshModelPlus)
    model.workingImage = _volume(np.ones((3, 1, 1), dtype='float32'))
    model.initializationSegmentation = _volume(np.array(
        [[[10]], [[28]], [[0]]], dtype='int32'))
    model.initializationMask = _volume(np.array(
        [[[1]], [[1]], [[0]]], dtype='uint8'))
    model.workingInitializationSegmentation = (
        model.initializationSegmentation.copy())
    model.workingInitializationMask = model.initializationMask.copy()
    model.labelMapping = sf.LabelLookup()
    return model


def _synthseg_refinement_model(fittedLabels, intensities, bilateral=True):
    """Build fitted regional evidence with nonstandard semantic label values."""
    if bilateral:
        atlasLabels = [0, 401, 402, 403, 501, 502]
        atlasNames = [
            'Unknown',
            'Left-Lateral-Ventricle',
            'Right-Lateral-Ventricle',
            '3rd-Ventricle',
            'Left-choroid-plexus',
            'Right-choroid-plexus',
        ]
        priorWeights = [0, 16000, 16000, 13535, 10000, 10000]
    else:
        # The tracked historical atlas has a unilateral left C/V family; its
        # single side-named choroid target is not a bilateral fallback.
        atlasLabels = [0, 401, 501]
        atlasNames = [
            'Unknown',
            'Left-Lateral-Ventricle',
            'Left-choroid-plexus',
        ]
        priorWeights = [0, 45535, 20000]

    fittedLabels = np.asarray(fittedLabels, dtype='int32')
    shape = (len(fittedLabels), 1, 1)
    model = object.__new__(ThalamicNucleiPlus)
    model.preliminaryModelProfileName = 'synthseg'
    model.FreeSurferLabels = np.asarray(atlasLabels)
    model.names = atlasNames
    model.preliminaryClassNames = ['Unknown', 'Ventricle']
    model.preliminaryClassFractions = np.zeros(
        (2, len(atlasLabels)), dtype='float32')
    model.preliminaryClassFractions[0, 0] = 1
    model.preliminaryClassFractions[1, 1:] = 1
    model.workingImage = _volume(
        np.asarray(intensities, dtype='float32').reshape(shape))
    model.workingInitializationSegmentation = _volume(
        fittedLabels.reshape(shape))
    model.workingInitializationMask = _volume(
        np.ones(shape, dtype='uint8'))
    model.initializationSegmentation = (
        model.workingInitializationSegmentation.copy())
    model.initializationMask = model.workingInitializationMask.copy()
    model.labelMapping = sf.LabelLookup()

    # Every candidate has both anatomical alternatives. Their common scale is
    # irrelevant to the likelihood-times-prior comparison.
    priorWeights = np.asarray(priorWeights, dtype='float64')
    fullPriors = np.broadcast_to(
        priorWeights, shape + (len(atlasLabels),)).copy()
    return model, fullPriors


def test_no_refinement_preserves_initialization_values_and_geometry():
    model = _refinement_model()
    originalSegmentation = model.initializationSegmentation.data.copy()
    originalSupport = model.initializationMask.data.copy()

    segmentation, support = (
        model._apply_working_initialization_refinement(None))

    np.testing.assert_array_equal(segmentation.data, originalSegmentation)
    np.testing.assert_array_equal(support.data, originalSupport)
    assert sf.transform.image_geometry_equal(
        segmentation, model.initializationSegmentation, tol=1e-5)
    assert sf.transform.image_geometry_equal(
        support, model.initializationMask, tol=1e-5)
    np.testing.assert_array_equal(
        model.initializationSegmentation.data, originalSegmentation)
    np.testing.assert_array_equal(
        model.initializationMask.data, originalSupport)


def test_regional_refinement_projects_only_supported_changes():
    model = _refinement_model()
    refined = model.workingInitializationSegmentation.copy()
    refined.data[1, 0, 0] = 8101

    segmentation, support = (
        model._apply_working_initialization_refinement(refined))

    np.testing.assert_array_equal(
        segmentation.data[:, 0, 0], [10, 8101, 0])
    np.testing.assert_array_equal(
        support.data[:, 0, 0], [1, 1, 0])
    assert sf.transform.image_geometry_equal(
        support, model.initializationMask, tol=1e-5)
    np.testing.assert_array_equal(
        model.initializationSegmentation.data[:, 0, 0], [10, 28, 0])


def test_regional_refinement_rejects_changes_outside_fitted_support():
    model = _refinement_model()
    refined = model.workingInitializationSegmentation.copy()
    refined.data[2, 0, 0] = 8101

    with pytest.raises(ValueError, match='outside fitted regional support'):
        model._apply_working_initialization_refinement(refined)


@pytest.mark.parametrize(
    'bilateral',
    [True, False],
    ids=['bilateral-choroid', 'unilateral-historical-choroid'],
)
def test_synthseg_refinement_overlays_choroid_from_oriented_joint_evidence(
        bilateral):
    """Use rough fitted choroid to orient modes, then overlay choroid.

    The bilateral case uses deliberately nonstandard left/right and midline
    labels; the historical case is genuinely unilateral. In both, provisional
    choroid spans the two intensity modes but favors the high mode, without
    assuming whether choroid should be bright or dark. Low-mode provisional
    choroid remains unchanged because a ventricular score does not write back.
    """
    if bilateral:
        fittedLabels = [
            401, 402, 403, 401, 501,
            501, 502, 501, 401, 402, 403, 402,
        ]
        intensities = [10, 11, 9, 12, 13, 90, 91, 89, 92, 93, 94, 88]
        expected = np.array([
            401, 402, 403, 401, 501,
            501, 502, 501, 501, 502, 403, 402,
        ])
    else:
        fittedLabels = [401, 501, 401, 501, 501, 401]
        intensities = [10, 11, 12, 90, 91, 92]
        expected = np.array([401, 501, 401, 501, 501, 501])
    model, fullPriors = _synthseg_refinement_model(
        fittedLabels, intensities, bilateral=bilateral)
    if bilateral:
        # The final high-intensity right-ventricle voxel has zero fitted
        # choroid prior, proving that image-mode membership alone cannot
        # promote it.
        fullPriors[11, 0, 0, 4:] = 0
        fullPriors[11, 0, 0, 1] += 20000
    original = model.workingInitializationSegmentation.data.copy()

    first = model._refine_initialization_state(fullPriors)
    second = model._refine_initialization_state(fullPriors)

    np.testing.assert_array_equal(first.data[:, 0, 0], expected)
    np.testing.assert_array_equal(second.data, first.data)
    np.testing.assert_array_equal(
        model.workingInitializationSegmentation.data, original)


@pytest.mark.parametrize(
    ('fittedLabels', 'intensities', 'warning'),
    [
        pytest.param(
            [501, 401, 402, 502, 401, 402],
            [10, 11, 12, 90, 91, 92],
            'fitted choroid support does not resolve',
            id='choroid-orientation-support-ties'),
        pytest.param(
            [501, 401, 402, 501],
            [10, 10, 10, 10],
            'two distinct subject-intensity modes',
            id='intensity-modes-are-degenerate'),
    ],
)
def test_synthseg_refinement_warns_and_preserves_fitted_labels_when_evidence_is_unusable(
        fittedLabels, intensities, warning):
    """Ambiguous orientation or unusable modes must preserve fitted anatomy.

    The first case places one provisional choroid voxel in each mode, tying the
    mature orientation vote. The second has no intensity separation.
    """
    model, fullPriors = _synthseg_refinement_model(
        fittedLabels, intensities)
    original = model.initializationSegmentation.data.copy()

    with pytest.warns(RuntimeWarning, match=warning):
        refinement = model._refine_initialization_state(fullPriors)
    segmentation, support = model._apply_working_initialization_refinement(
        refinement)

    np.testing.assert_array_equal(segmentation.data, original)
    np.testing.assert_array_equal(
        support.data, model.initializationMask.data)


def test_aseg_initialization_requires_no_regional_refinement():
    model = object.__new__(ThalamicNucleiPlus)
    model.preliminaryModelProfileName = 'aseg'
    assert model._refine_initialization_state(np.empty((0,))) is None


def test_multichannel_hyperparameters_use_common_support_and_voxel_volume():
    """Check first-stage means, volume-scaled strength, and zero evidence."""
    model = object.__new__(ThalamicNucleiPlus)
    model.intensityPriorImage = _volume(
        np.array([
            [[[10.0, -100.0]]],
            [[[30.0, -300.0]]],
            [[[50.0, -500.0]]],
            [[[70.0, 700.0]]],
        ], dtype='float32'),
        voxsize=(2.0, 2.0, 2.0))
    model.workingImage = _volume(
        np.ones((4, 1, 1, 2), dtype='float32'))
    model.intensityPriorInitializationSegmentation = _volume(
        np.array([[[2]], [[2]], [[28]], [[0]]], dtype='int32'),
        voxsize=(2.0, 2.0, 2.0))
    model.intensityPriorInitializationMask = _volume(
        np.ones((4, 1, 1), dtype='uint8'),
        voxsize=(2.0, 2.0, 2.0))

    with pytest.warns(RuntimeWarning, match='class 2'):
        means, strengths = model.get_gaussian_hyps(
            [[2], [28, 60], [999]], mesh=None)

    # Two class-2 voxels and one VDC voxel each occupy 8 mm3. Negative second-
    # channel values remain valid, VDC labels 28/60 use ordinary support-
    # derived strength 18, and unsupported label 999 uses the subject median
    # and weak configured strength.
    np.testing.assert_array_equal(
        means, [[20, -200], [50, -500], [30, -300]])
    np.testing.assert_array_equal(strengths, [26, 18, 10])


def test_zero_evidence_fixed_strategy_broadcasts_and_checks_channel_shape():
    # The value 55 is an arbitrary configured fixed-policy fixture, not the
    # default successor rule or a claim about the historical 55/10 fallback.
    policy = ZeroEvidenceInitializationPolicy(
        strategy='fixed', mean=55.0, strength=10.0)

    means, strength = policy.initialize(np.empty((0, 2)), 2)

    np.testing.assert_array_equal(means, [55.0, 55.0])
    assert strength == 10.0

    incompatible = ZeroEvidenceInitializationPolicy(
        strategy='fixed', mean=[1.0, 2.0, 3.0], strength=10.0)
    with pytest.raises(ValueError, match='broadcast-compatible'):
        incompatible.initialize(np.empty((0, 2)), 2)


def test_subject_median_zero_evidence_strategy_fails_without_valid_observations():
    policy = ZeroEvidenceInitializationPolicy()

    with pytest.raises(RuntimeError, match='no usable non-background'):
        policy.initialize(np.empty((0, 2)), 2)


def test_hyperparameter_support_uses_one_mm_erosion_on_anisotropic_grid():
    """Check that class statistics use approximately 1 mm physical erosion."""
    model = object.__new__(MeshModelPlus)
    shape = (9, 5, 6)
    voxelSize = (0.5, 1.0, 2.0)
    classSupport = np.zeros(shape, dtype=bool)
    classSupport[1:8, 1:4, 1:5] = True
    physicalSupport = np.zeros(shape, dtype=bool)
    physicalSupport[3:6, 2, 1:5] = True
    voxelRadiusSupport = np.zeros(shape, dtype=bool)
    voxelRadiusSupport[2:7, 2, 2:4] = True
    overlap = physicalSupport & voxelRadiusSupport
    # On this 0.5 x 1 x 2 mm grid, physical 1-mm erosion and naive one-voxel
    # erosion select different supports. Values 100 in physical-only voxels,
    # 10 in their overlap, and 1 in voxel-radius-only voxels make median 55
    # specific to the physical-distance interpretation.
    intensities = np.zeros(shape, dtype='float32')
    intensities[classSupport] = 50.0
    intensities[voxelRadiusSupport & ~physicalSupport] = 1.0
    intensities[physicalSupport & ~voxelRadiusSupport] = 100.0
    intensities[overlap] = 10.0
    model.intensityPriorImage = _volume(
        intensities[..., np.newaxis], voxsize=voxelSize)
    model.workingImage = _volume(
        np.ones(shape, dtype='float32'), voxsize=voxelSize)
    model.intensityPriorInitializationSegmentation = _volume(
        np.where(classSupport, 2, 0).astype('int32'), voxsize=voxelSize)
    model.intensityPriorInitializationMask = _volume(
        np.ones(shape, dtype='uint8'), voxsize=voxelSize)

    means, strengths = model._estimate_intensity_hyperparameters([[2]])

    assert means[0, 0] == 55.0
    assert strengths[0] == 22.0


@pytest.mark.parametrize('supportCase', ['relaxed', 'whole-label'])
def test_small_class_erosion_relaxes_or_uses_whole_label_support(supportCase):
    """Relax erosion before falling back to whole-label class support."""
    model = object.__new__(MeshModelPlus)
    if supportCase == 'relaxed':
        # A 1-mm erosion of this 5^3 block leaves one sample; relaxation leaves
        # a 3^3 core whose value 30 identifies the selected support.
        shape = (7, 7, 7)
        classSupport = np.zeros(shape, dtype=bool)
        classSupport[1:6, 1:6, 1:6] = True
        intensities = np.zeros(shape, dtype='float32')
        intensities[classSupport] = 100.0
        intensities[2:5, 2:5, 2:5] = 30.0
        expectedMean = 30.0
    else:
        # Every erosion of this 3^3 block has fewer than 10 samples. Central
        # value 1 distinguishes accidental eroded statistics from the
        # whole-label median 70.
        shape = (5, 5, 5)
        classSupport = np.zeros(shape, dtype=bool)
        classSupport[1:4, 1:4, 1:4] = True
        intensities = np.zeros(shape, dtype='float32')
        intensities[classSupport] = 70.0
        intensities[2, 2, 2] = 1.0
        expectedMean = 70.0
    voxelSize = (0.5, 0.5, 0.5)
    model.intensityPriorImage = _volume(
        intensities[..., np.newaxis], voxsize=voxelSize)
    model.workingImage = _volume(
        np.ones(shape, dtype='float32'), voxsize=voxelSize)
    model.intensityPriorInitializationSegmentation = _volume(
        np.where(classSupport, 2, 0).astype('int32'), voxsize=voxelSize)
    model.intensityPriorInitializationMask = _volume(
        np.ones(shape, dtype='uint8'), voxsize=voxelSize)

    means, strengths = model._estimate_intensity_hyperparameters([[2]])

    # Prior and EM voxels have equal volume, so 27 samples give strength 10+27.
    assert means[0, 0] == expectedMean
    assert strengths[0] == 37.0


def test_configured_structural_model_populates_alphas_and_gmm_mean_priors(
        monkeypatch):
    """Map configured classes and multichannel mean priors into maintained GMM."""
    class Mesh:
        def __init__(self):
            self.alphas = None

        def rasterize(self, shape):
            priors = np.empty(tuple(shape) + (3,), dtype='uint16')
            priors[..., 0] = 20000
            priors[..., 1] = 20000
            priors[..., 2] = 25535
            return priors

    class MeshCollection:
        def __init__(self):
            self.mesh = Mesh()
            self.k = None

        def read(self, fileName):
            pass

        def transform(self, transform):
            pass

        def get_mesh(self, meshNumber):
            assert meshNumber == 0
            return self.mesh

    expectedMeans = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
    ])
    expectedStrengths = np.array([12.0, 34.0])
    model = object.__new__(MeshModelPlus)
    model.processedImage = _volume(
        np.ones((2, 2, 2, 2), dtype='float32'))
    model.crop_image_by_atlas = lambda image: (image.copy(), object())
    model.warpedMeshFileName = 'unused'
    model.meshStiffness = 0.05
    model.modelPolicy = SubregionModelPolicy()
    model.debug = False
    model._prepare_intensity_initialization_evidence = lambda priors: None
    model.FreeSurferLabels = np.array([2, 3, 4])
    model.classFractions = np.array([
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    model.sharedGMMParameters = [
        GMMparameter('First', 1, ['First']),
        GMMparameter('Second', 1, ['Second']),
    ]
    model.originalAlphas = np.array([[0.2, 0.3, 0.5]])
    model.get_gaussian_hyps = lambda groups, mesh: (
        expectedMeans.copy(), expectedStrengths.copy())
    monkeypatch.setattr(
        core_plus.gems, 'KvlMeshCollection', MeshCollection)

    model.prepare_for_image_fitting()

    assert model.sameGaussianParameters == [[2, 3], [4]]
    np.testing.assert_allclose(model.reducedAlphas, [[0.5, 0.5]])
    np.testing.assert_array_equal(model.reducingLookupTable, [0, 0, 1])
    np.testing.assert_array_equal(model.gmm.hyperMeans, expectedMeans)
    np.testing.assert_array_equal(
        model.gmm.fullHyperMeansNumberOfMeasurements,
        expectedStrengths)
    assert model.gmm.numberOfGaussiansPerClass == [1, 1]
    assert model.gmm.numberOfContrasts == 2
    assert model.gmm.useDiagonalCovarianceMatrices
    assert model.gmm.means is None
    assert model.gmm.variances is None
    assert model.gmm.mixtureWeights is None
    assert model.means is None
    assert model.variances is None


def test_regional_em_mask_accepts_finite_negative_and_rejects_missing_channels(
        monkeypatch):
    class Mesh:
        def __init__(self):
            self.alphas = np.ones((1, 1), dtype='float32')

        def rasterize(self, shape):
            return np.full(tuple(shape) + (1,), 65535, dtype='uint16')

    class MeshCollection:
        def __init__(self):
            self.mesh = Mesh()
            self.k = None

        def read(self, fileName):
            pass

        def transform(self, transform):
            pass

        def get_mesh(self, meshNumber):
            assert meshNumber == 0
            return self.mesh

    # The voxels differ only in channel 2: finite -2 is valid, exact zero is
    # the missing/background encoding, and NaN is invalid.
    data = np.ones((3, 2, 2, 2), dtype='float32')
    data[0, 0, 0, 1] = -2
    data[1, 0, 0, 1] = 0
    data[2, 0, 0, 1] = np.nan
    model = object.__new__(MeshModelPlus)
    model.processedImage = _volume(data)
    model.crop_image_by_atlas = lambda image: (image.copy(), object())
    model.warpedMeshFileName = 'unused'
    model.meshStiffness = 0.05
    model.modelPolicy = SubregionModelPolicy()
    model.debug = False
    model._prepare_intensity_initialization_evidence = lambda priors: None
    model.FreeSurferLabels = np.array([0])
    model.classFractions = np.ones((1, 1), dtype='float64')
    model.sharedGMMParameters = [
        GMMparameter('Tissue', 1, ['Tissue'])]
    model.originalAlphas = np.ones((1, 1), dtype='float32')
    monkeypatch.setattr(
        core_plus.gems, 'KvlMeshCollection', MeshCollection)

    model.prepare_for_image_fitting(compute_hyps=False)

    assert model.workingMask.data[0, 0, 0]
    assert not model.workingMask.data[1, 0, 0]
    assert not model.workingMask.data[2, 0, 0]
    assert model.workingImage.data[0, 0, 0, 1] == -2


def test_coarse_to_refined_transition_fails_without_changing_initial_hyperparameters():
    """The deferred transition must fail before altering first-stage values."""
    model = object.__new__(ThalamicNucleiPlus)
    # Nonuniform channel means and strengths make any partial mutation visible.
    means = np.array([[20.0, 200.0], [80.0, 800.0]])
    strengths = np.array([10.0, 12.0])
    originalMeans = means.copy()
    originalStrengths = strengths.copy()

    with pytest.raises(NotImplementedError, match='configured source and target'):
        model.get_second_label_groups()
    with pytest.raises(NotImplementedError, match='configured source and target'):
        model.get_second_gaussian_hyps(
            [[], [], []], means, strengths)

    np.testing.assert_array_equal(means, originalMeans)
    np.testing.assert_array_equal(strengths, originalStrengths)
