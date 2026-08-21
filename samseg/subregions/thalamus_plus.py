import os
import numpy as np

from samseg.subregions import utils
from samseg.subregions.core_plus import MeshModelPlus


_PRELIMINARY_MODEL_PROFILE_FILES = {
    'aseg': {
        'sharedGMMParametersFileName': 'ASEGsharedGMMparameters.txt',
        'localizerLookupTableFileName': 'ASEGlocalizerLookupTable.txt',
    },
    'synthseg': {
        'sharedGMMParametersFileName': 'SYNTHSEGsharedGMMparameters.txt',
        'localizerLookupTableFileName': 'SYNTHSEGlocalizerLookupTable.txt',
    },
}


class ThalamicNucleiPlus(MeshModelPlus):

    """Provide thalamus model selection and regional lifecycle behavior.

    The region selects preliminary artifacts, constructs thalamic alignment and
    crop targets, supplies first-stage label grouping and refinement decisions,
    gates unsupported target-stage transitions, and postprocesses thalamic
    output. ``MeshModelPlus`` owns shared geometry, initialization evidence, and
    hyperparameter mechanics.
    """

    # -------------------------------------------------------------------------
    # Model defaults and lifecycle configuration
    # -------------------------------------------------------------------------

    def __init__(self, **kwargs):
        atlasDir = kwargs.pop('atlasDir', None)
        if atlasDir is None:
            atlasDir = os.path.join(
                os.environ.get('FREESURFER_HOME'), 'average',
                'ThalamicNuclei', 'atlas')

        preliminaryModelDirectory = kwargs.pop(
            'preliminaryModelDirectory', atlasDir)
        inputSegmentationSchema = kwargs.pop(
            'inputSegmentationSchema', None)

        super().__init__(atlasDir=atlasDir, **kwargs)

        # Resolve the region-owned preliminary model and policy artifacts.
        self.preliminaryModelDirectory = preliminaryModelDirectory
        self.modelPolicyFileName = os.path.join(
            preliminaryModelDirectory, 'modelPolicy.json')
        self.preliminaryModelProfiles = {
            profileName: {
                fieldName: (
                    os.path.join(preliminaryModelDirectory, fileName)
                    if fileName is not None else None)
                for fieldName, fileName in profile.items()
            }
            for profileName, profile
            in _PRELIMINARY_MODEL_PROFILE_FILES.items()
        }
        self.inputSegmentationSchemaOverride = inputSegmentationSchema

        # Preserve the two-stage boundary; the unsupported target transition
        # fails closed through the explicit extension methods below.
        self.useTwoComponents = True

        # Segmentation mesh-fitting parameters
        self.cheatingMeshSmoothingSigmas = [3.0, 2.0]
        self.cheatingMaxIterations = [300, 150]

        # Image mesh-fitting parameters
        self.meshSmoothingSigmas = [1.5, 1.125, 0.75, 0]
        self.imageSmoothingSigmas = [0, 0, 0, 0]
        self.maxIterations = [7, 5, 5, 3]

        # Longitudinal mesh-fitting parameters
        self.longMeshSmoothingSigmas = [[1.5, 1.125, 0.75], [1.125, 0.75, 0]]
        self.longImageSmoothingSigmas = [[0, 0, 0], [0, 0, 0]]
        self.longMaxIterations = [[7, 5, 3], [3, 2, 1]]

    # -------------------------------------------------------------------------
    # Preliminary profile and regional input preparation
    # -------------------------------------------------------------------------

    def preprocess_images(self):
        """Prepare preliminary targets and ordinary intensity grids.

        Preregistered intensity channels are sampled independently from their
        supplied sources. The unmasked intensity-prior stack preserves the
        complete first-channel field of view for later hyperparameter support;
        the masked ``processedImage`` is a separate regional EM representation
        at ``self.resolution``.
        """

        # Output/postprocessing uses the canonical thalamus segmentation labels.
        self.THlabelLeft = 10
        self.THlabelRight = 49

        self._configure_preliminary_model_profile(
            self.preliminaryModelProfiles,
            requestedProfileName=self.inputSegmentationSchemaOverride)
        self._ensure_preliminary_model_state()

        # Build the affine target from the selected profile's thalamus classes.
        match_labels = self._get_preliminary_affine_support_labels()
        mask = np.isin(self.inputSeg.data, match_labels).astype('float32') * 255
        self.atlasAlignmentTarget = self.inputSeg.new(mask)

        # Build the preliminary target without changing the source localizer.
        # VDC and thalamus intentionally share each hemispheric class because
        # their coarse-localizer boundary is not trusted for mesh deformation;
        # fitted full atlas priors reconstruct that distinction afterwards.
        self.synthImage = self._build_preliminary_synthetic_image(
            self.inputSeg)

        # The regional field of view extends approximately 15 mm beyond the
        # thalamus on the localizer grid.
        thalamicMask = ((self.synthImage == self.THlabelLeft)
                        | (self.synthImage == self.THlabelRight))
        cropMarginInVoxels = int(
            np.round(15 / np.mean(self.inputSeg.geom.voxsize)))
        imageCropping = self.synthImage.new(thalamicMask).bbox(
            margin=cropMarginInVoxels)

        # Preserve the complete first-channel field of view so later
        # hyperparameter support is not constrained by the regional EM crop.
        # Fitted anatomical labels are materialized after preliminary fitting.
        self.intensityPriorImage = (
            self._resample_and_stack_intensity_channels(
                self.inputImageFileNames,
                self.intensityPriorReferenceImage,
                'intensityPrior'))

        # The regional EM grid keeps the historical isotropic resolution. Crop
        # indices never cross voxel grids: every channel is resampled directly
        # from its preregistered source onto this geometry.
        regionalReference = self.synthImage[imageCropping].resize(
            self.resolution, method='nearest')
        # Materialize the shared policy-defined localizer-anatomical validity
        # support directly on this grid. Later regional preparation intersects
        # it with channel validity and the regional atlas-domain restriction.
        self.longMask = regionalReference.new(
            self._localizer_anatomical_support(
                regionalReference).astype('uint8'))
        self.processedImage = self._resample_and_stack_intensity_channels(
            self.inputImageFileNames,
            regionalReference,
            'regionalIntensity',
            mask=self.longMask)

    def _get_preliminary_affine_support_labels(self):
        """Return selected-profile labels owned by either thalamus class."""
        classNumbers = {
            className: classNumber
            for classNumber, className
            in enumerate(self.preliminaryClassNames)
        }
        try:
            thalamusClassNumbers = [
                classNumbers['LeftThalamus'],
                classNumbers['RightThalamus'],
            ]
        except KeyError as error:
            raise ValueError(
                'Thalamus preliminary model requires LeftThalamus and '
                'RightThalamus classes') from error
        return sorted({
            label
            for classNumber in thalamusClassNumbers
            for label in self.preliminaryLocalizerLabelGroups[classNumber]
        })

    # -------------------------------------------------------------------------
    # First-stage intensity initialization
    # -------------------------------------------------------------------------

    def _refine_initialization_state(self, fullPriors):
        """Apply the profile-specific regional initialization refinement.

        ASEG needs no additional correction after generic fitted-prior
        reconstruction. SynthSeg fails closed because its image-informed
        choroid correction is not yet supported.

        Parameters
        ----------
        fullPriors : numpy.ndarray
            Full fitted-atlas priors on the regional EM grid.

        Returns
        -------
        None
            ASEG uses the generic reconstruction unchanged.

        Raises
        ------
        NotImplementedError
            If the selected profile is SynthSeg.
        """
        if self.preliminaryModelProfileName == 'synthseg':
            raise NotImplementedError(
                'SynthSeg initialization requires supported choroid refinement '
                'before intensity hyperparameters can be estimated')
        return super()._refine_initialization_state(fullPriors)

    def get_label_groups(self):
        """Return the current coarse grouping for first-stage intensity fitting."""
        labelGroups = [
            ['Unknown'],
            ['Left-Cerebral-White-Matter', 'Left-R', 'Right-R'],
            ['Left-Cerebral-Cortex'],
            ['Left-Cerebellum-Cortex'],
            ['Left-Cerebellum-White-Matter'],
            ['Brain-Stem'],
            ['Left-Lateral-Ventricle'],
            ['Left-choroid-plexus'],
            ['Left-Putamen'],
            ['Left-Pallidum'],
            ['Left-Accumbens-area'],
            ['Left-Caudate'],
            ['Left-VentralDC', 'Right-VentralDC'],
        ]

        # The first stage models all bilateral thalamic nuclei together. This
        # grouping is distinct from the output-reporting whitelist below.
        thalamicLabels = [
            'L-Sg', 'LGN', 'MGN', 'PuI', 'PuM', 'H', 'PuL', 'VPI',
            'PuA', 'MV(Re)', 'Pf', 'CM', 'LP', 'VLa', 'VPL', 'VLp',
            'MDm', 'VM', 'CeM', 'MDl', 'Pc', 'MDv', 'Pv', 'CL', 'VA',
            'VPM', 'AV', 'VAmc', 'Pt', 'AD', 'LD',
        ]
        labelGroups.append([
            f'{side}-{label}'
            for side in ('Left', 'Right')
            for label in thalamicLabels
        ])
        return labelGroups

    def get_gaussian_hyps(self, sameGaussianParameters, mesh):
        """Return first-stage hyperparameters from whole-field evidence.

        ``MeshModelPlus`` owns the generic estimator; thalamus contributes the
        active label grouping and model policy.

        Parameters
        ----------
        sameGaussianParameters : sequence of sequence of int
            Full atlas labels sharing each intensity Gaussian.
        mesh : object
            Retained by the inherited region hook. The fitted mesh has already
            contributed through initialization reconstruction.

        Returns
        -------
        meanHyper : numpy.ndarray
            Class-by-channel prior means.
        nHyper : numpy.ndarray
            Effective prior sample count for each class.
        """
        return self._estimate_intensity_hyperparameters(
            sameGaussianParameters)

    # -------------------------------------------------------------------------
    # Deferred intensity-stage transition gates
    # -------------------------------------------------------------------------

    def get_second_label_groups(self):
        """Fail until a configured target-stage grouping is available."""
        raise NotImplementedError(
            'ThalamicNucleiPlus refinement requires configured source and '
            'target intensity stages with atlas-membership correspondence')

    def get_second_gaussian_hyps(self, sameGaussianParameters, meanHyper, nHyper):
        """Fail until target-stage hyperparameters and transfer are available."""
        raise NotImplementedError(
            'ThalamicNucleiPlus refinement hyperparameters require configured '
            'source and target intensity stages')

    # -------------------------------------------------------------------------
    # Segmentation output
    # -------------------------------------------------------------------------

    def postprocess_segmentation(self):
        """Filter connected thalamic output and write reported nucleus volumes."""

        # Retain nucleus labels and the canonical whole-thalamus labels.
        segmentation = self.discreteLabels.copy()
        segmentation[
            (segmentation < 100)
            & (segmentation != 10)
            & (segmentation != 49)
        ] = 0

        # Reticular labels participate in fitting but are not reported.
        leftReticular = self.labelMapping.search('Left-R', exact=True)
        rightReticular = self.labelMapping.search('Right-R', exact=True)
        segmentation[segmentation == leftReticular] = 0
        segmentation[segmentation == rightReticular] = 0

        # Nucleus LUT labels use the 81xx namespace on the left and 82xx on the
        # right; whole-thalamus labels 10 and 49 are handled explicitly.
        leftComponent = utils.get_largest_cc(
            (segmentation < 8200)
            & ((segmentation > 100)
               | (segmentation == self.THlabelLeft)))
        rightComponent = utils.get_largest_cc(
            (segmentation > 8200)
            | (segmentation == self.THlabelRight))
        connectedThalami = leftComponent | rightComponent
        segmentation[connectedThalami == 0] = 0

        segFilePrefix = os.path.join(
            self.outDir, f'ThalamicNuclei{self.fileSuffix}')
        segmentation.save(segFilePrefix + '.mgz')
        segmentation.resample_like(
            self.inputSeg, method='nearest').save(
                segFilePrefix + '.FSvoxelSpace.mgz')

        # The current set of nuclei reported in the volumes file intentionally
        # excludes reticular labels.
        reportedNucleusNames = [
            'L-Sg', 'LGN', 'MGN', 'PuI', 'PuM', 'H', 'PuL', 'VPI',
            'PuA', 'MV(Re)', 'Pf', 'CM', 'LP', 'VLa', 'VPL', 'VLp',
            'MDm', 'VM', 'CeM', 'MDl', 'Pc', 'MDv', 'Pv', 'CL', 'VA',
            'VPM', 'AV', 'VAmc', 'Pt', 'AD', 'LD',
        ]
        isReportedNucleus = lambda name: (
            name.replace('Left-', '') in reportedNucleusNames
            or name.replace('Right-', '') in reportedNucleusNames)
        self.volumes = {
            name: volume
            for name, volume in self.volumes.items()
            if isReportedNucleus(name)
        }

        # Add whole-thalamus totals for the retained nuclei in each hemisphere.
        self.volumes['Left-Whole_thalamus'] = np.sum([
            volume for name, volume in self.volumes.items()
            if name.startswith('Left')
        ])
        self.volumes['Right-Whole_thalamus'] = np.sum([
            volume for name, volume in self.volumes.items()
            if name.startswith('Right')
        ])

        # Write the volumes
        self.write_volumes(segFilePrefix + '.volumes.txt')
