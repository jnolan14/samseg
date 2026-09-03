import os
import shlex
import shutil
import tempfile
import warnings
import numpy as np
import surfa as sf
import scipy.ndimage
from sklearn.cluster import KMeans
from samseg import gems
from samseg.GMM import GMM
from samseg.io import kvlReadSharedGMMParameters
from samseg.merge_alphas import (
    kvlGetMergingFractionsTable,
    kvlMergeAlphas,
    kvlResolveSharedGMMParameters,
)
from samseg.utilities import requireNumpyArray
from samseg.subregions import utils
from samseg.subregions.model_policy import SubregionModelPolicy


class MeshModelPlus:

    """Provide the shared lifecycle for successor subregion models.

    The base owns preliminary-model configuration, multichannel geometry,
    fitted-state reconstruction, post-preliminary initialization evidence, and
    ordinary intensity fitting. Region subclasses supply model defaults and
    anatomical preprocessing, optional regional refinement, label grouping,
    and output behavior.

    Notes
    -----
    The source is arranged in lifecycle order: regional preprocessing, atlas
    alignment and preliminary fitting, initialization-evidence preparation,
    regional intensity fitting, and output. Configured intensity-stage
    transitions remain explicit fail-closed extension seams.
    """

    # -------------------------------------------------------------------------
    # Shared lifecycle state
    # -------------------------------------------------------------------------

    def __init__(
        self,
        atlasDir,
        outDir,
        inputImageFileNames,
        inputSegFileName,
        meshStiffness=0.05,
        optimizerType='L-BFGS',
        bbregisterMode=None,
        resolution=0.5,
        useTwoComponents=False,
        tempDir=None,
        fileSuffix='',
        debug=False,
        preliminarySharedGMMParametersFileName=None,
        gmmFileName=None,
        useDiagonalCovarianceMatrices=False,
        ):
        """Initialize shared lifecycle state and region-configurable defaults."""

        # Set some paths
        self.outDir = outDir
        self.atlasDir = atlasDir
        self.atlasMeshFileName = os.path.join(atlasDir, 'AtlasMesh.gz')
        self.atlasDumpFileName = os.path.join(atlasDir, 'AtlasDump.mgz')
        self.compressionLookupTableFileName = os.path.join(atlasDir, 'compressionLookupTable.txt')
        self.inputImageFileNames = inputImageFileNames
        self.inputSegFileName = inputSegFileName
        self.preliminarySharedGMMParametersFileName = (
            preliminarySharedGMMParametersFileName)
        self.gmmFileName = gmmFileName
        self.useDiagonalCovarianceMatrices = useDiagonalCovarianceMatrices
        self.preliminaryModelProfileName = None
        self.preliminaryLocalizerLookupTableFileName = None
        self.modelPolicyFileName = None

        # Some settings
        self.meshStiffness = meshStiffness
        self.optimizerType = optimizerType
        self.bbregisterMode = bbregisterMode
        self.resolution = resolution
        self.useTwoComponents = useTwoComponents
        self.tempDir = tempDir
        self.fileSuffix = fileSuffix
        self.debug = debug

        # Preliminary state belongs only to the coarse, localizer-driven mesh
        # fit and remains separate from the ordinary intensity model.
        self.cheatingMeans = None
        self.cheatingVariances = None
        self.preliminarySharedGMMParameters = None
        self.preliminaryClassFractions = None
        self.preliminaryClassNames = None
        self.modelPolicy = None
        # These localizer labels never contain numeric atlas memberships.
        self.preliminaryLocalizerLabelGroups = None
        self.preliminaryAlphas = None

        # The fitted preliminary mesh supplies full labels on the prior-reference
        # geometry, with validity restricted to fitted regional support.
        self.initializationSegmentation = None
        self.initializationMask = None
        # Regional EM consumers use an independently materialized form of that
        # evidence, not the whole-field hyperparameter label map.
        self.workingInitializationSegmentation = None
        self.workingInitializationMask = None
        # Whole-field hyperparameter consumers use a separate merge of fitted
        # labels and semantically compatible source-localizer anatomy.
        self.intensityPriorInitializationSegmentation = None
        self.intensityPriorInitializationMask = None
        # The first intensity supplies the whole-field prior geometry; the
        # aligned stack is reserved for intensity-prior and hyperparameter use.
        self.intensityPriorReferenceImage = None
        self.intensityPriorImage = None
        # The structural shared-parameter model remains separate from the
        # preliminary localizer model above.
        self.sharedGMMParameters = None
        self.classFractions = None
        self.gmm = None
        self.optimizationHistory = []

        # Some optimization defaults that should be overwritten by each subclass
        self.cheatingMeshSmoothingSigmas = [3.0, 2.0]
        self.cheatingMaxIterations = [300, 150]

        self.meshSmoothingSigmas = [1.5, 0.75, 0]
        self.imageSmoothingSigmas = [0, 0, 0]
        self.maxIterations = [7, 5, 3]

        self.isLong = False
        self.longMeshSmoothingSigmas = [[1.5, 0.75], [0.75, 0]]
        self.longImageSmoothingSigmas = [[0, 0], [0, 0]]
        self.longMaxIterations = [[6, 3], [2, 1]]
        self.maxGlobalLongIterations = 2
        self.longMask = None

    @property
    def means(self):
        """Expose the Gaussian-component means owned by the configured GMM."""
        return self.gmm.means

    @means.setter
    def means(self, value):
        self.gmm.means = value

    @property
    def variances(self):
        """Expose the covariance matrices owned by the configured GMM."""
        return self.gmm.variances

    @variances.setter
    def variances(self, value):
        self.gmm.variances = value

    def cleanup(self):
        """
        Essentially the only thing to do during cleanup is (potentially)
        remove the temporary directory.
        """
        if not self.debug:
            shutil.rmtree(self.tempDir)
        else:
            print(f'Not removing temporary directory: {self.tempDir}')

    def initialize(self):
        """
        Initialize the mesh model by running sanity checks on the input options,
        loading input volumes, creating the temporary directory, and doing image preprocessing.
        """

        # First thing: set up the temporary directory. IO to this space should be relatively
        # limited unless debug mode is enabled
        if self.tempDir is None:
            self.tempDir = tempfile.mkdtemp()
        else:
            os.makedirs(self.tempDir, exist_ok=True)

        # Make sure the output directory exists as well
        os.makedirs(self.outDir, exist_ok=True)

        # Sanity check on the optimizer type
        optimizerTypes = ['FixedStepGradientDescent', 'GradientDescent', 'ConjugateGradient', 'L-BFGS']
        if self.optimizerType not in optimizerTypes:
            sf.system.fatal('Optimizer type must be one of: ' + ', '.join(optimizerTypes))

        # Sanity check on the registration mode for alternative images
        bbregisterModes = [None, 't1', 't2']
        if self.bbregisterMode not in bbregisterModes:
            sf.system.fatal('BBregister mode must be one of: ' + ', '.join(bbregisterModes))

        # Make sure all the atlas files are there
        if not os.path.isfile(self.atlasMeshFileName):
            sf.system.fatal(f'Provided atlas mesh file `{self.atlasMeshFileName}` does not exist.')
        if not os.path.isfile(self.atlasDumpFileName):
            sf.system.fatal(f'Provided atlas image `{self.atlasDumpFileName}` does not exist.')
        if not os.path.isfile(self.compressionLookupTableFileName):
            sf.system.fatal(f'Provided compression LUT `{self.compressionLookupTableFileName}` does not exist.')

        # Load compressed and FreeSurfer label mapping information
        self.labelMapping, self.names, self.FreeSurferLabels = utils.read_compression_lookup_table(self.compressionLookupTableFileName)
        self._configure_shared_gmm_parameters()

        # Set the target mesh file paths
        self.warpedMeshFileName = os.path.join(self.tempDir, 'warpedOriginalMesh.txt')
        self.warpedMeshNoAffineFileName = os.path.join(self.tempDir, 'warpedOriginalMeshNoAffine.txt')

        # The localizer remains immutable. Intensity sources are assumed to be
        # anatomically preregistered but may use different sampling grids; each
        # lifecycle representation is derived independently from these sources.
        self.inputSeg = sf.load_volume(self.inputSegFileName)
        self.inputImages = [sf.load_volume(path) for path in self.inputImageFileNames]
        self.correctedImages = [img.copy() for img in self.inputImages]
        self.intensityPriorReferenceImage = self.inputImages[0]
        self.highResImage = np.mean(self.inputImages[0].geom.voxsize) < 0.99

        # Region preprocessing supplies the anatomical targets and stage-specific
        # intensity representations described by preprocess_images().
        self.preprocess_images()

    def _configure_shared_gmm_parameters(self):
        """Resolve the configured structural GMM against the selected LUT."""
        if self.gmmFileName is None:
            raise ValueError(
                'A structural GMM parameter file must be supplied through '
                'gmmFileName')

        gmmFileName = os.fspath(self.gmmFileName)
        if not os.path.isfile(gmmFileName):
            gmmFileName = os.path.join(self.atlasDir, gmmFileName)
        if not os.path.isfile(gmmFileName):
            raise ValueError(
                f'GMM parameter file does not exist: {gmmFileName}')

        configuredParameters = kvlReadSharedGMMParameters(gmmFileName)
        sharedGMMParameters, memberships = (
            kvlResolveSharedGMMParameters(
                self.names, configuredParameters))

        componentCounts = np.asarray([
            parameter.numberOfComponents
            for parameter in sharedGMMParameters
        ])
        if np.any(componentCounts <= 0):
            raise ValueError(
                'Structural shared-GMM component counts must be positive')
        if np.any(np.count_nonzero(memberships, axis=0) > 1):
            raise NotImplementedError(
                'Plus structural GMM initialization does not yet support '
                'structures shared across multiple parameter rows')

        self.gmmFileName = gmmFileName
        self.sharedGMMParameters = sharedGMMParameters
        # In the supported disjoint case, mature Boolean membership and
        # maintained SAMSEG class fractions are exactly equivalent.
        self.classFractions = memberships.astype(float)

    # -------------------------------------------------------------------------
    # Regional preprocessing and shared input geometry
    # -------------------------------------------------------------------------

    def preprocess_images(self):
        """Construct region-selected targets and intensity representations.

        Descendants orchestrate anatomical preprocessing and must provide the
        affine ``atlasAlignmentTarget``, collapsed preliminary ``synthImage``,
        whole-field ``intensityPriorImage``, and independently sampled regional
        ``processedImage``. Generic base primitives handle common geometry and
        model mechanics without deciding regional crop or support policy.
        """
        raise NotImplementedError('All subclasses of MeshModel must implement the preprocess_images() function!')

    def _resample_and_stack_intensity_channels(
            self, sourceFileNames, referenceImage, outputPrefix, mask=None):
        """Resample preregistered intensity channels onto a supplied grid.

        Parameters
        ----------
        sourceFileNames : sequence of str
            Intensity image files in channel order. Their world coordinates
            must already describe the same anatomy.
        referenceImage : surfa.Volume
            Three-dimensional target geometry. Its voxel values are ignored.
        outputPrefix : str
            Prefix used for temporary reference and channel files.
        mask : surfa.Volume, optional
            Three-dimensional mask already expressed on ``referenceImage``.

        Returns
        -------
        surfa.Volume
            Resampled channels stacked along the frame axis.

        Notes
        -----
        This reconciles sampling grids; it does not estimate registration.
        Callers choose the target geometry and semantic support, then supply
        the authoritative stage sources directly to avoid chained
        interpolation.
        """
        if not sourceFileNames:
            raise ValueError('At least one intensity image is required')
        if referenceImage.data.ndim != 3:
            raise ValueError('Intensity reference image must be three-dimensional')
        if mask is not None:
            if mask.data.ndim != 3:
                raise ValueError('Intensity mask must be three-dimensional')
            if not sf.transform.image_geometry_equal(
                    mask, referenceImage, tol=1e-5):
                raise ValueError(
                    'Intensity mask must match the supplied reference geometry')

        referenceFileName = os.path.join(
            self.tempDir, f'{outputPrefix}Reference.mgz')
        referenceImage.save(referenceFileName)

        channels = []
        for channelNumber, sourceFileName in enumerate(sourceFileNames):
            resampledFileName = os.path.join(
                self.tempDir, f'{outputPrefix}_{channelNumber}.mgz')
            utils.run(
                f'mri_convert {shlex.quote(sourceFileName)} '
                f'{shlex.quote(resampledFileName)} -odt float -rt cubic '
                f'-rl {shlex.quote(referenceFileName)}')
            image = sf.load_volume(resampledFileName)

            if (image.data.ndim != 3
                    or not sf.transform.image_geometry_equal(
                        image, referenceImage, tol=1e-5)):
                raise RuntimeError(
                    'Intensity channel '
                    f'{channelNumber} ({sourceFileName}) did not match the '
                    'requested reference geometry')

            if mask is not None:
                image[mask == 0] = 0
            channels.append(image.data)

        return referenceImage.new(np.stack(channels, axis=-1))

    # TODO: This duplicates the intended physical-radius behavior of
    # utils.spherical_strel(), whose current pixel-size handling is not
    # equivalent for the anisotropic physical spacing required here. Replace
    # this only after correcting the shared implementation and
    # compatibility-auditing its existing callers.
    # -------------------------------------------------------------------------

    @staticmethod
    def _physical_spherical_structure(image, radiusInMm):
        """Construct a voxel-grid structuring element for a physical radius."""
        if radiusInMm < 0:
            raise ValueError('Physical morphology radius must be nonnegative')
        voxelSize = np.asarray(image.geom.voxsize, dtype='float64')
        extents = np.ceil(radiusInMm / voxelSize).astype(int)
        coordinates = np.ogrid[
            tuple(slice(-extent, extent + 1) for extent in extents)]
        squaredDistance = sum(
            (coordinate * spacing) ** 2
            for coordinate, spacing in zip(coordinates, voxelSize))
        return squaredDistance <= radiusInMm ** 2 + 1e-12

    # -------------------------------------------------------------------------

    def _localizer_anatomical_support(self, referenceImage):
        """Return policy-expanded non-background localizer support on a grid."""
        if self.preliminaryClassFractions is None:
            raise RuntimeError(
                'Preliminary class ownership is required for localizer '
                'anatomical support')
        atlasLabels = np.asarray(self.FreeSurferLabels)
        backgroundStructures = np.flatnonzero(atlasLabels == 0)
        if len(backgroundStructures) != 1:
            raise RuntimeError(
                'Preliminary model must contain exactly one atlas background '
                'label')
        backgroundClasses = np.flatnonzero(
            np.asarray(self.preliminaryClassFractions)[
                :, backgroundStructures[0]] > 0)
        if len(backgroundClasses) != 1:
            raise RuntimeError(
                'Atlas background must have exactly one preliminary class '
                'owner')
        backgroundLabels = self.preliminaryLocalizerLabelGroups[
            backgroundClasses[0]]
        localizer = self.inputSeg.resample_like(
            referenceImage, method='nearest')
        support = ~np.isin(localizer.data, backgroundLabels)

        marginInMm = (
            self._ensure_model_policy()
            .localizerAnatomicalSupportMarginInMm)
        if marginInMm > 0:
            support = scipy.ndimage.binary_dilation(
                support,
                structure=self._physical_spherical_structure(
                    referenceImage, marginInMm))
        return support

    def _apply_atlas_domain_interior_margin(self, support, image, marginInMm):
        """Apply a physical inward margin to atlas cuboid support."""
        if marginInMm <= 0:
            return support
        return scipy.ndimage.binary_erosion(
            support,
            structure=self._physical_spherical_structure(image, marginInMm),
            border_value=1)

    def _apply_affine_target_morphology(self, support):
        """Apply the policy-selected one-voxel affine-target morphology."""
        structure = utils.spherical_strel(1)
        return self._ensure_model_policy().apply_affine_target_morphology(
            support, structure)

    # -------------------------------------------------------------------------
    # Shared label and class reduction
    # -------------------------------------------------------------------------

    def label_group_names_to_indices(self, labelNames):
        """
        Clean and convert a group of label names (list of lists) to a grouping of label indices.
        """
        labelIndices = [[self.labelMapping.search(name, exact=True) for name in group] for group in labelNames]
        labelIndices = [[i for i in group if i is not None] for group in labelIndices]
        labelIndices = [g for g in labelIndices if g]
        return labelIndices

    def reduce_alphas(self, sameGaussianParameters, alphas=None):
        """
        Compute a set of reduced alpha values given groups of labels. Will use the original
        alpha values if alphas is None.
        """
        if alphas is None:
            alphas = self.originalAlphas

        numberOfReducedLabels = len(sameGaussianParameters)
        # TODO: Confirm whether GEMS alpha buffers are always float32.
        reducedAlphas = np.zeros((alphas.shape[0], numberOfReducedLabels), dtype='float32')
        reducingLookupTable = np.zeros(alphas.shape[1], dtype='int32')

        # Convert to list so we can use index
        fslabels = list(self.FreeSurferLabels)

        # Reduce the labels
        for reducedLabel in range(numberOfReducedLabels):
            sameGaussians = sameGaussianParameters[reducedLabel]
            for label in sameGaussians:
                compressedLabel = fslabels.index(label)
                reducedAlphas[:, reducedLabel] += alphas[:, compressedLabel]
                reducingLookupTable[compressedLabel] = reducedLabel

        # Make sure classes sum to one
        if np.max(np.abs(np.sum(reducedAlphas, -1) - 1)) > 1e-5:
            sf.system.fatal('The vector of prior probabilities in the mesh nodes must always sum to one over all classes')

        return (reducedAlphas, reducingLookupTable)

    # -------------------------------------------------------------------------
    # Preliminary-model configuration
    # -------------------------------------------------------------------------

    def _configure_preliminary_model_profile(
            self, profiles, requestedProfileName=None):
        """Select and configure one compatible preliminary model profile.

        Parameters
        ----------
        profiles : dict
            Region-provided mapping from profile names to resolved shared-GMM
            and localizer-LUT paths.
        requestedProfileName : str, optional
            Explicit profile selection. When omitted, canonical input naming
            and bounded localizer vocabularies are used for inference.

        Returns
        -------
        str
            Selected profile name.
        """
        # Validate each profile definition and record its bounded vocabulary.
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(
                'At least one preliminary model profile is required')

        requiredFields = {
            'sharedGMMParametersFileName',
            'localizerLookupTableFileName',
        }
        supportedFields = requiredFields
        normalizedProfiles = {}
        profileVocabularies = {}
        for profileName, profile in profiles.items():
            if not isinstance(profileName, str) or not profileName:
                raise ValueError(
                    'Preliminary model profile names must be nonempty strings')
            if not isinstance(profile, dict):
                raise ValueError(
                    f'Preliminary model profile {profileName!r} must be a '
                    'mapping')
            missingFields = sorted(requiredFields - set(profile))
            unsupportedFields = sorted(set(profile) - supportedFields)
            if missingFields or unsupportedFields:
                details = []
                if missingFields:
                    details.append('missing ' + ', '.join(missingFields))
                if unsupportedFields:
                    details.append(
                        'unsupported ' + ', '.join(unsupportedFields))
                raise ValueError(
                    f'Invalid preliminary model profile {profileName!r}: '
                    + '; '.join(details))

            normalizedProfile = {
                'sharedGMMParametersFileName':
                    profile['sharedGMMParametersFileName'],
                'localizerLookupTableFileName':
                    profile['localizerLookupTableFileName'],
            }
            localizerLookupTableFileName = normalizedProfile[
                'localizerLookupTableFileName']
            if not os.path.isfile(localizerLookupTableFileName):
                raise ValueError(
                    f'Preliminary model profile {profileName!r} localizer '
                    'lookup table does not exist: '
                    f'{localizerLookupTableFileName}')
            localizerLookupTable = sf.load_label_lookup(
                localizerLookupTableFileName)
            normalizedProfiles[profileName] = normalizedProfile
            profileVocabularies[profileName] = {
                int(label) for label in localizerLookupTable.keys()}

        # Prefer an explicit selection, then canonical filename provenance.
        if requestedProfileName is not None:
            if requestedProfileName not in normalizedProfiles:
                availableProfiles = ', '.join(sorted(normalizedProfiles))
                raise ValueError(
                    f'Unknown preliminary model profile '
                    f'{requestedProfileName!r}; available profiles: '
                    f'{availableProfiles}')
            selectedProfileName = requestedProfileName
        else:
            fileName = os.path.basename(self.inputSegFileName).lower()
            for suffix in ('.nii.gz', '.mgz', '.mgh', '.nii'):
                if fileName.endswith(suffix):
                    fileName = fileName[:-len(suffix)]
                    break
            provenanceMatches = [
                profileName for profileName in normalizedProfiles
                if (fileName == profileName.lower()
                    or fileName.endswith('+' + profileName.lower()))
            ]
            if len(provenanceMatches) > 1:
                raise ValueError(
                    'Input segmentation name matches multiple preliminary '
                    'model profiles')
            selectedProfileName = (
                provenanceMatches[0] if provenanceMatches else None)

        # If provenance did not select a profile, use observed-label coverage.
        if not hasattr(self, 'inputSeg'):
            raise ValueError(
                'Input segmentation must be loaded before selecting a '
                'preliminary model profile')
        observedLabels = set(
            np.unique(self.inputSeg.data).astype(int).tolist())
        if selectedProfileName is None:
            compatibleProfiles = [
                profileName
                for profileName, vocabulary in profileVocabularies.items()
                if observedLabels <= vocabulary
            ]
            if len(compatibleProfiles) == 1:
                selectedProfileName = compatibleProfiles[0]
            elif not compatibleProfiles:
                raise ValueError(
                    'Input segmentation labels are unsupported by the '
                    'available preliminary model profiles')
            else:
                raise ValueError(
                    'Unable to distinguish compatible preliminary model '
                    'profiles; select one explicitly')

        unsupportedLabels = sorted(
            observedLabels - profileVocabularies[selectedProfileName])
        if unsupportedLabels:
            raise ValueError(
                f'Input segmentation contains labels outside preliminary '
                f'model profile {selectedProfileName!r}: '
                + ', '.join(str(label) for label in unsupportedLabels))

        # Validate the selected model artifact before publishing profile state.
        selectedProfile = normalizedProfiles[selectedProfileName]
        for fieldName, description in (
                ('sharedGMMParametersFileName',
                 'shared-GMM parameter file'),):
            fileName = selectedProfile[fieldName]
            if fileName is not None and not os.path.isfile(fileName):
                raise ValueError(
                    f'Preliminary model profile {selectedProfileName!r} '
                    f'{description} does not exist: {fileName}')

        self.preliminaryModelProfileName = selectedProfileName
        self.preliminarySharedGMMParametersFileName = selectedProfile[
            'sharedGMMParametersFileName']
        self.preliminaryLocalizerLookupTableFileName = selectedProfile[
            'localizerLookupTableFileName']
        return selectedProfileName

    def _ensure_model_policy(self):
        """Load the one sparse policy object for the full model lifecycle."""
        if getattr(self, 'modelPolicy', None) is not None:
            return self.modelPolicy
        if getattr(self, 'modelPolicyFileName', None) is None:
            self.modelPolicy = SubregionModelPolicy()
            return self.modelPolicy
        if not os.path.isfile(self.modelPolicyFileName):
            raise ValueError(
                'Subregion model policy file does not exist: '
                f'{self.modelPolicyFileName}')
        self.modelPolicy = SubregionModelPolicy.read(self.modelPolicyFileName)
        return self.modelPolicy

    def _ensure_preliminary_model_state(self):
        """Materialize the shared preliminary grouping state when possible.

        Parsing and class construction do not depend on a loaded mesh. Merged
        alphas are added later, once ``originalAlphas`` is available. Repeated
        calls only fill missing state.

        When no shared-parameter artifact is configured, a region subclass may
        still supply its preliminary grouping through the inherited label-list
        extension seam.
        """
        parameterFileName = self.preliminarySharedGMMParametersFileName
        if parameterFileName is None:
            if not hasattr(self, 'sameGaussianParameters'):
                labelGroups = self.get_cheating_label_groups()
                self.sameGaussianParameters = (
                    self.label_group_names_to_indices(labelGroups))

            if (self.preliminaryAlphas is None
                    and getattr(self, 'originalAlphas', None) is not None):
                self.preliminaryAlphas, _ = self.reduce_alphas(
                    self.sameGaussianParameters)
            return

        if self.preliminarySharedGMMParameters is None:
            sharedGMMParameters = kvlReadSharedGMMParameters(parameterFileName)
            if not sharedGMMParameters:
                raise ValueError(
                    'Preliminary shared-GMM parameter file defines no classes')
            if any(parameter.numberOfComponents != 1
                   for parameter in sharedGMMParameters):
                raise ValueError(
                    'Preliminary segmentation fitting requires exactly one '
                    'Gaussian per class')
            self.preliminarySharedGMMParameters = sharedGMMParameters

        if self.preliminaryClassFractions is None:
            if not hasattr(self, 'names'):
                return
            classFractions, classNames = kvlGetMergingFractionsTable(
                self.names, self.preliminarySharedGMMParameters)
            if np.any(np.count_nonzero(classFractions, axis=0) != 1):
                raise ValueError(
                    'Each atlas structure in a preliminary shared-GMM file '
                    'must match exactly one class')
            FreeSurferLabels = np.asarray(self.FreeSurferLabels)
            self.preliminaryClassFractions = classFractions
            self.preliminaryClassNames = classNames
            # Preserve atlas/compression-LUT labels only for alpha reduction.
            self.sameGaussianParameters = [
                FreeSurferLabels[fractions > 0].tolist()
                for fractions in classFractions
            ]

        if (self.preliminaryLocalizerLabelGroups is None
                and self.preliminaryLocalizerLookupTableFileName is not None):
            self._ensure_model_policy()
            preliminaryLocalizerLookupTable = sf.load_label_lookup(
                self.preliminaryLocalizerLookupTableFileName)
            self.preliminaryLocalizerLabelGroups = (
                self._build_preliminary_localizer_label_groups(
                    self.preliminarySharedGMMParameters,
                    preliminaryLocalizerLookupTable))

        if (self.preliminaryAlphas is None
                and getattr(self, 'originalAlphas', None) is not None):
            self.preliminaryAlphas = kvlMergeAlphas(
                self.originalAlphas, self.preliminaryClassFractions)

    def _build_preliminary_localizer_label_groups(
            self, sharedGMMParameters, localizerLookupTable):
        """Build class-aligned groups from a bounded localizer vocabulary.

        Parameters
        ----------
        sharedGMMParameters : sequence
            Parsed shared-GMM rows defining class names and search strings.
        localizerLookupTable : surfa.LabelLookup
            Selected model's bounded localizer label vocabulary.

        Returns
        -------
        list of list of int
            Localizer labels assigned to each shared-GMM class in row order.

        Raises
        ------
        ValueError
            If policy references are invalid or a vocabulary label cannot be
            assigned to exactly one preliminary class.
        """
        # Validate shared class identities and sparse policy references.
        classNames = [
            parameter.mergedName for parameter in sharedGMMParameters]
        if not classNames:
            raise ValueError(
                'Preliminary shared-GMM parameters define no classes')
        if len(classNames) != len(set(classNames)):
            raise ValueError('Preliminary class names must be unique')

        self._ensure_model_policy()
        policyMemberships = (
            self.modelPolicy.get_preliminary_localizer_label_memberships(
                self.preliminaryModelProfileName))
        unknownClasses = sorted(set(policyMemberships) - set(classNames))
        if unknownClasses:
            raise ValueError(
                'Subregion model policy references unknown preliminary '
                'classes: ' + ', '.join(unknownClasses))

        classNumbers = {
            className: classNumber
            for classNumber, className in enumerate(classNames)
        }
        exactOwners = {
            label: classNumbers[className]
            for className, labels in policyMemberships.items()
            for label in labels
        }
        vocabularyLabels = {
            int(label) for label in localizerLookupTable.keys()}
        missingPolicyLabels = sorted(set(exactOwners) - vocabularyLabels)
        if missingPolicyLabels:
            raise ValueError(
                'Subregion model policy references labels absent from the '
                'selected localizer vocabulary: '
                + ', '.join(str(label) for label in missingPolicyLabels))

        # Infer vocabulary ownership from shared-GMM search strings. Policy
        # memberships may fill only labels that inference leaves unmatched.
        groups = [[] for _ in sharedGMMParameters]
        unmatched = []
        ambiguous = []
        inferredPolicyLabels = []
        for label, element in sorted(localizerLookupTable.items()):
            label = int(label)
            matchingClasses = [
                classNumber
                for classNumber, parameter
                in enumerate(sharedGMMParameters)
                if any(searchString in element.name
                       for searchString in parameter.searchStrings)
            ]
            if len(matchingClasses) == 1:
                groups[matchingClasses[0]].append(label)
                if label in exactOwners:
                    inferredPolicyLabels.append((
                        label,
                        element.name,
                        classNames[matchingClasses[0]],
                    ))
            elif not matchingClasses:
                exactOwner = exactOwners.get(label)
                if exactOwner is None:
                    unmatched.append((label, element.name))
                else:
                    groups[exactOwner].append(label)
            else:
                ambiguous.append((
                    label,
                    element.name,
                    [classNames[classNumber]
                     for classNumber in matchingClasses],
                ))

        # Report every ownership failure before accepting the aligned groups.
        if inferredPolicyLabels:
            details = ', '.join(
                f'{label} ({name}: {className})'
                for label, name, className in inferredPolicyLabels)
            raise ValueError(
                'Subregion model policy may only assign labels unmatched by '
                f'shared-parameter inference; already inferred: {details}')
        if unmatched:
            details = ', '.join(
                f'{label} ({name})' for label, name in unmatched)
            raise ValueError(
                'Selected localizer vocabulary contains labels unmatched by '
                f'the preliminary model: {details}')
        if ambiguous:
            details = ', '.join(
                f'{label} ({name}: {"/".join(matches)})'
                for label, name, matches in ambiguous)
            raise ValueError(
                'Selected localizer vocabulary contains labels assigned to '
                f'multiple preliminary classes: {details}')

        # Every shared class must be represented in the selected vocabulary.
        emptyClasses = [
            className for className, labels in zip(classNames, groups)
            if not labels
        ]
        if emptyClasses:
            raise ValueError(
                'Preliminary classes have no labels in the selected '
                'localizer vocabulary: ' + ', '.join(emptyClasses))
        return groups

    def _build_preliminary_synthetic_image(self, segmentation):
        """Construct the artificial segmentation used for preliminary fitting.

        Parameters
        ----------
        segmentation : surfa.Volume
            Original input localizer. Its data are not modified.

        Returns
        -------
        surfa.Volume
            Class-reduced artificial target in the input geometry.

        Raises
        ------
        ValueError
            If model state is incomplete or observed labels are unsupported.
        """
        self._ensure_preliminary_model_state()
        labelGroups = self.preliminaryLocalizerLabelGroups
        if labelGroups is None:
            raise ValueError(
                'A localizer vocabulary is required to construct the '
                'preliminary synthetic segmentation')

        means, _ = self.get_cheating_gaussians(labelGroups)
        means = np.asarray(means)
        expectedShape = (len(labelGroups),)
        if means.shape != expectedShape:
            raise ValueError(
                'Preliminary means must contain one scalar per class')

        labelToClass = {
            label: classNumber
            for classNumber, labels in enumerate(labelGroups)
            for label in labels
        }
        observedLabels = set(
            np.unique(segmentation.data).astype(int).tolist())
        unsupportedLabels = sorted(observedLabels - set(labelToClass))
        if unsupportedLabels:
            raise ValueError(
                'Input segmentation contains labels outside the selected '
                'preliminary localizer vocabulary: '
                + ', '.join(str(label) for label in unsupportedLabels))

        source = segmentation.data
        target = np.zeros(source.shape, dtype='float32')
        for label, classNumber in labelToClass.items():
            target[source == label] = means[classNumber]
        return segmentation.new(target)

    # -------------------------------------------------------------------------
    # Atlas alignment and preliminary fitting
    # -------------------------------------------------------------------------

    def crop_image_by_atlas(self, image):
        """
        Crop image to the aligned atlas image. Also construct a 3-D affine transformation that will later be used
        to transform the location of the atlas mesh's nodes into the coordinate system of the image.
        """
        trf = image.geom.world2vox @ self.alignedAtlas.geom.vox2world
        template_corners = np.mgrid[:2, :2, :2].T.reshape(-1, 3) * (np.array(self.alignedAtlas.shape[:3]) - 1)
        corners = trf.transform(template_corners)
        lower = corners.min(0).astype(int)
        upper = (corners.max(0) + 1).astype(int)

        image_limit = np.array(image.shape[:3]) - 1
        lower = np.clip(lower, (0, 0, 0), image_limit)
        upper = np.clip(upper, (0, 0, 0), image_limit)

        trf.matrix[:3, -1] -= lower
        cropping = tuple([slice(l, u + 1) for l, u in zip(lower, upper)])

        transform = gems.KvlTransform(np.asfortranarray(trf.matrix))
        return (image[cropping].copy(), transform)

    def align_atlas_to_seg(self):
        """
        The initial stage before mesh fitting involves aligning the atlas coordinates to
        the image coordinates by registering an atlas image to the subject's segmentation.
        This step requires that the atlasAlignmentTarget has been properly configured
        during preprocessing.
        """

        # Make sure the subclass has computed the target mask
        mask = self.atlasAlignmentTarget.copy()
        if mask is None:
            sf.system.fatal('All MeshModel subclasses must compute atlasAlignmentTarget during preprocessing!')

        # No need for a high-resolution alignment here
        mask = mask > 0
        if np.mean(mask.geom.voxsize) < 0.99:
            mask = mask.resize(1, method='nearest')

        # Crop mask to the label bounding box
        mask = mask.crop_to_bbox(margin=6)

        # Apply the policy-selected morphology to the affine target.
        mask.data = self._apply_affine_target_morphology(mask.data)

        # We're going to use mri_robust_register for this registration, so let's ensure the mask
        # value is 255 and we'll write to disk
        mask.data = mask.data.astype('float32') * 255
        targetMaskFile = os.path.join(self.tempDir, 'targetMask.mgz')
        mask.save(targetMaskFile)

        # Write the atlas as well
        alignedAtlasFile = os.path.join(self.tempDir, 'alignedAtlasImage.mgz')
        # Copying the atlas dump preserves the current registration input; an
        # equivalent in-memory write has not been established.
        shutil.copyfile(self.atlasDumpFileName, alignedAtlasFile)

        # Run the actual registration and load the result
        utils.run(f'mri_robust_register --mov {alignedAtlasFile} --dst {targetMaskFile} --lta {self.tempDir}/trash.lta --mapmovhdr {alignedAtlasFile} --sat 50 -verbose 0')
        utils.run(f'mri_robust_register --mov {alignedAtlasFile} --dst {targetMaskFile} --lta {self.tempDir}/trash.lta --mapmovhdr {alignedAtlasFile} --affine --sat 50 -verbose 0')
        self.alignedAtlas = sf.load_volume(alignedAtlasFile)

    def prepare_for_seg_fitting(self):
        """
        Prepare the mesh collection, preprocessed image, reduced alphas, and Gaussians parameters.
        """

        # Make sure the subclass has computed the synthed target
        if self.synthImage is None:
            sf.system.fatal('All MeshModel subclasses must compute synthImage during preprocessing!')

        # Crop the synthesized image by the aligned atlas and compute the new mesh alignment
        self.workingImage, self.transform = self.crop_image_by_atlas(self.synthImage)
        self.workingImageShape = self.workingImage.shape[:3]

        # Read in collection, set stiffness, and apply transform
        self.meshCollection = gems.KvlMeshCollection()
        self.meshCollection.read(self.atlasMeshFileName)
        self.meshCollection.transform(self.transform)
        self.meshCollection.k = self.meshStiffness

        # Retrieve the reference mesh, i.e. the mesh representing the average shape
        self.mesh = self.meshCollection.reference_mesh
        self.originalNodePositions = self.mesh.points.copy(order='K')
        self.originalAlphas = self.mesh.alphas.copy(order='K')

        # Materialize the shared preliminary grouping and its merged alphas.
        self._ensure_preliminary_model_state()
        self.mesh.alphas = self.preliminaryAlphas
        mask = (self.mesh.rasterize(self.workingImageShape).sum(-1) / 65535) > 0.99
        mask = self._apply_atlas_domain_interior_margin(
            mask,
            self.workingImage,
            self._ensure_model_policy()
            .preliminaryAtlasDomainInteriorMarginInMm)
        self.workingImage[mask == 0] = 0

        # Get the region-specific artificial Gaussian parameters.
        gaussianLabelGroups = (
            self.preliminaryLocalizerLabelGroups
            if self.preliminaryLocalizerLabelGroups is not None
            else self.sameGaussianParameters)
        self.cheatingMeans, self.cheatingVariances = (
            self.get_cheating_gaussians(gaussianLabelGroups))

        if self.preliminarySharedGMMParameters is not None:
            self.cheatingMeans = np.asarray(self.cheatingMeans)
            self.cheatingVariances = np.asarray(self.cheatingVariances)
            expectedShape = (len(self.preliminarySharedGMMParameters),)
            if (self.cheatingMeans.shape != expectedShape
                    or self.cheatingVariances.shape != expectedShape):
                raise ValueError(
                    'Preliminary means and variances must contain one scalar '
                    'per shared-GMM class')

        # Write the initial and cropped/masked images for debugging.
        if self.debug:
            self.synthImage.save(os.path.join(self.tempDir, 'synthImage.mgz'))
            self.workingImage.save(os.path.join(self.tempDir, 'synthImageMasked.mgz'))

    def fit_mesh_to_seg(self):
        """Fit the coarse preliminary mesh to the collapsed localizer target.

        The fitted mesh is restored to full anatomical alphas before full-label
        initialization evidence is reconstructed on the intensity-prior grid.
        """

        # Just get the image buffer (array) and convert to a Kvl image object
        imageBuffer = self.workingImage.data.copy(order='K')
        image = gems.KvlImage(requireNumpyArray(imageBuffer))

        # Use a multi-resolution approach
        for multiResolutionLevel, meshSmoothingSigma in enumerate(self.cheatingMeshSmoothingSigmas):

            # Set mesh alphas
            self.mesh.alphas = self.preliminaryAlphas

            # Smooth the mesh to limit boundary compression during deformation.
            if meshSmoothingSigma > 0:
                print(f'Smoothing mesh collection with kernel size {meshSmoothingSigma}')
                self.meshCollection.smooth(meshSmoothingSigma)

            # Note that it uses variances instead of precisions
            calculator = gems.KvlCostAndGradientCalculator(
                typeName='AtlasMeshToIntensityImage',
                images=[image],
                boundaryCondition='Sliding',
                transform=self.transform,
                means=self.cheatingMeans.reshape((-1, 1)),
                variances=self.cheatingVariances.reshape((-1, 1, 1)),
                mixtureWeights=np.ones(len(self.cheatingMeans), dtype='float32'),
                numberOfGaussiansPerClass=np.ones(
                    len(self.cheatingMeans), dtype='int32'))

            # Step some optimizer stop criteria
            maximalDeformationStopCriterion = 1e-10
            relativeChangeInCostStopCriterion = 1e-10

            # Get optimizer and plug calculator into it
            optimizationParams = {
                'Verbose': False,
                'MaximalDeformationStopCriterion': maximalDeformationStopCriterion,
                'LineSearchMaximalDeformationIntervalStopCriterion': 1e-10,
                'MaximumNumberOfIterations': 1000,
                'BFGS-MaximumMemoryLength': 12
            }
            optimizer = gems.KvlOptimizer(self.optimizerType, self.mesh, calculator, optimizationParams)

            # Run the optimizations
            history = []
            for iteration in range(self.cheatingMaxIterations[multiResolutionLevel]):

                # Step optimizer
                minLogLikelihoodTimesPrior, maximalDeformation = optimizer.step_optimizer_samseg()

                # Log step information
                iterationInfo = [
                    f'Res: {multiResolutionLevel + 1:03d}',
                    f'Iter: {iteration + 1:03d}',
                    f'MaxDef: {maximalDeformation:.4f}',
                    f'MinLLxP: {minLogLikelihoodTimesPrior:.4f}',
                ]
                print('  '.join(iterationInfo))

                # Track optimization history
                previous = history[-1] if history else np.finfo(np.float32).max
                history.append(minLogLikelihoodTimesPrior)

                # Check for stop criteria
                relativeChange = np.abs((previous - minLogLikelihoodTimesPrior) / minLogLikelihoodTimesPrior)
                if maximalDeformation <= maximalDeformationStopCriterion or relativeChange < relativeChangeInCostStopCriterion:
                    break

        # Restore full anatomy while the mesh still occupies its fitted subject
        # coordinates, then materialize ordinary-model initialization evidence.
        self.mesh.alphas = self.originalAlphas
        if self.preliminarySharedGMMParameters is not None:
            (self.initializationSegmentation,
             self.initializationMask) = self._reconstruct_initialization_state()

        # Assign fitted positions to the first training-subject warp before
        # returning the collection to native atlas space.
        self.meshCollection.set_positions(self.originalNodePositions, [self.mesh.points])

        # Return the fitted collection to native atlas space for later reuse.
        inverseTransform = gems.KvlTransform(np.asfortranarray(np.linalg.inv(self.transform.as_numpy_array)))
        self.meshCollection.transform(inverseTransform)
        self.meshCollection.write(self.warpedMeshFileName)

    def _reconstruct_initialization_state(self):
        """Reconstruct full atlas labels after preliminary mesh fitting.

        The preliminary target deliberately suppresses anatomical distinctions
        that the coarse localizer cannot support reliably. This method restores
        full labels from the subject-fitted atlas priors while retaining only
        the coarse class evidence used for that fit. Optional region-specific
        refinements, such as SynthSeg choroid handling, operate on the resulting
        state later.

        Returns
        -------
        tuple of surfa.Volume
            Full-label initialization segmentation and its valid fitted-atlas
            support mask, both on the intensity-prior reference geometry. This
            state initializes the subsequent ordinary intensity model.

        Raises
        ------
        RuntimeError
            If the configured preliminary model or fitted mesh state is
            incomplete.
        """
        # Validate the fitted mesh, coarse-class mapping, and target geometry.
        requiredState = {
            'mesh': getattr(self, 'mesh', None),
            'workingImage': getattr(self, 'workingImage', None),
            'originalAlphas': getattr(self, 'originalAlphas', None),
            'preliminaryClassFractions': self.preliminaryClassFractions,
            'cheatingMeans': self.cheatingMeans,
        }
        missingState = [
            name for name, value in requiredState.items() if value is None]
        if missingState:
            raise RuntimeError(
                'Cannot reconstruct initialization state before the '
                'preliminary fit has materialized: '
                + ', '.join(missingState))

        fullPriors = self.mesh.rasterize(self.workingImageShape)
        numberOfStructures = self.originalAlphas.shape[1]
        expectedPriorShape = tuple(self.workingImageShape) + (
            numberOfStructures,)
        if fullPriors.shape != expectedPriorShape:
            raise RuntimeError(
                'Fitted full-prior rasterization has shape '
                f'{fullPriors.shape}, expected {expectedPriorShape}')
        if len(self.FreeSurferLabels) != numberOfStructures:
            raise RuntimeError(
                'Compression-LUT labels do not align with fitted full priors')

        classFractions = np.asarray(self.preliminaryClassFractions)
        if classFractions.shape[1] != numberOfStructures:
            raise RuntimeError(
                'Preliminary class fractions do not align with fitted full '
                'priors')
        if np.any(np.count_nonzero(classFractions, axis=0) != 1):
            raise RuntimeError(
                'Every full atlas structure must belong to exactly one '
                'preliminary class')
        structureClassNumbers = np.argmax(classFractions, axis=0)

        # Score only structures compatible with the observed coarse class.
        cheatingMeans = np.asarray(self.cheatingMeans)
        if cheatingMeans.shape != (classFractions.shape[0],):
            raise RuntimeError(
                'Preliminary means do not align with preliminary classes')
        structureMeans = cheatingMeans[structureClassNumbers]
        classEvidence = (
            np.asarray(self.workingImage.data)[..., np.newaxis]
            == structureMeans)
        scores = np.where(classEvidence, fullPriors, 0)

        # Fall back to the fitted full priors where quantized coarse evidence is
        # absent, but only inside the fitted atlas domain.
        priorMass = np.sum(fullPriors, axis=-1, dtype=np.uint64)
        validSupport = priorMass > (0.99 * 65535)
        missingEvidence = validSupport & ~np.any(scores > 0, axis=-1)
        scores[missingEvidence] = fullPriors[missingEvidence]

        winningStructures = np.argmax(scores, axis=-1)
        labels = np.asarray(self.FreeSurferLabels)[winningStructures]
        labels = labels.copy()
        labels[~validSupport] = 0

        # Project the fitted labels and support onto the prior-reference grid.
        croppedSegmentation = self.workingImage.new(labels)
        croppedMask = self.workingImage.new(
            validSupport.astype('uint8'))
        targetImage = self.inputImages[0]
        segmentation = croppedSegmentation.resample_like(
            targetImage, method='nearest')
        supportMask = croppedMask.resample_like(
            targetImage, method='nearest')
        supportMask.data = supportMask.data > 0
        segmentation.data[~supportMask.data] = 0
        segmentation.labels = self.labelMapping
        return segmentation, supportMask

    # -------------------------------------------------------------------------
    # Post-preliminary initialization evidence
    # -------------------------------------------------------------------------

    def _prepare_intensity_initialization_evidence(self, fullPriors):
        """Complete the post-preliminary evidence lifecycle for intensities.

        Regional fitted evidence is materialized first so a region may refine
        it. Supported corrections are then projected back before the distinct
        whole-field state used for intensity hyperparameters is constructed.
        """
        self._materialize_working_initialization_state()
        refinedInitialization = self._refine_initialization_state(fullPriors)
        fittedSegmentation, fittedMask = (
            self._apply_working_initialization_refinement(
                refinedInitialization))
        return self._materialize_intensity_prior_initialization_state(
            fittedSegmentation, fittedMask)

    def _materialize_working_initialization_state(self):
        """Materialize fitted initialization evidence on the regional EM grid.

        ``initializationSegmentation`` remains the fitted-prior reconstruction
        on the intensity-prior geometry. This method creates the distinct
        regional form consumed by ordinary image fitting and restricts it to
        both fitted-atlas support and the regional intensity mask.

        Returns
        -------
        tuple of surfa.Volume
            Regional full-label initialization segmentation and support mask.
        """
        if (self.initializationSegmentation is None
                or self.initializationMask is None):
            raise RuntimeError(
                'Preliminary fitting must reconstruct initialization state '
                'before regional initialization is materialized')

        segmentation = self.initializationSegmentation.resample_like(
            self.workingImage, method='nearest')
        fittedSupport = self.initializationMask.resample_like(
            self.workingImage, method='nearest').data > 0
        support = fittedSupport & (self.workingMask.data > 0)
        segmentation.data[~support] = 0
        segmentation.labels = self.labelMapping
        supportMask = self.workingImage.new(support.astype('uint8'))

        self.workingInitializationSegmentation = segmentation
        self.workingInitializationMask = supportMask
        return segmentation, supportMask

    def _refine_initialization_state(self, fullPriors):
        """Optionally refine regional initialization labels before statistics.

        The base intensity lifecycle has no anatomical correction to apply.
        Region descendants may return a corrected regional segmentation while
        preserving the established support; returning ``None`` is a no-op.

        Parameters
        ----------
        fullPriors : numpy.ndarray
            Full fitted-atlas priors rasterized on the regional EM grid.

        Returns
        -------
        surfa.Volume or None
            Corrected regional initialization labels, or ``None``.
        """
        return None

    def _apply_working_initialization_refinement(self, refinedSegmentation):
        """Project an optional regional correction back to fitted evidence.

        Only labels deliberately changed inside fitted regional support are
        projected. The no-op path returns the original prior-grid state without
        a resampling round trip.

        Parameters
        ----------
        refinedSegmentation : surfa.Volume or None
            Optional corrected labels on the regional EM grid.

        Returns
        -------
        tuple of surfa.Volume
            Effective fitted segmentation and support on the prior geometry.

        Raises
        ------
        ValueError
            If a correction has the wrong geometry or changes labels outside
            fitted regional support.
        """
        if refinedSegmentation is None:
            return self.initializationSegmentation, self.initializationMask
        if not sf.transform.image_geometry_equal(
                refinedSegmentation, self.workingImage, tol=1e-5):
            raise ValueError(
                'Refined initialization segmentation must match the regional '
                'EM geometry')

        original = np.asarray(self.workingInitializationSegmentation.data)
        refined = np.asarray(refinedSegmentation.data)
        if refined.shape != original.shape:
            raise ValueError(
                'Refined initialization segmentation must be three-dimensional')
        changed = refined != original
        regionalSupport = self.workingInitializationMask.data > 0
        if np.any(changed & ~regionalSupport):
            raise ValueError(
                'Initialization refinement cannot change labels outside fitted '
                'regional support')
        if not np.any(changed):
            return self.initializationSegmentation, self.initializationMask

        changedVolume = self.workingImage.new(changed.astype('uint8'))
        changedOnPriorGrid = changedVolume.resample_like(
            self.initializationSegmentation, method='nearest').data > 0
        labelsOnPriorGrid = refinedSegmentation.resample_like(
            self.initializationSegmentation, method='nearest').data
        fittedSupport = self.initializationMask.data > 0
        changedOnPriorGrid &= fittedSupport

        effectiveSegmentation = self.initializationSegmentation.copy()
        effectiveSegmentation.data[changedOnPriorGrid] = (
            labelsOnPriorGrid[changedOnPriorGrid])
        effectiveSegmentation.labels = self.labelMapping
        return effectiveSegmentation, self.initializationMask

    def _materialize_intensity_prior_initialization_state(
            self, fittedSegmentation, fittedMask):
        """Build whole-field labels and support for intensity hyperparameters.

        The fitted reconstruction is regional evidence, whereas intensity
        statistics can require anatomy across the complete prior-reference
        field. Outside fitted support, immutable source-localizer labels are
        retained only when their preliminary-class ownership agrees in the
        independent localizer and atlas namespaces; otherwise the deliberately
        collapsed preliminary target supplies the class-compatible fallback.

        Parameters
        ----------
        fittedSegmentation : surfa.Volume
            Effective fitted full-label evidence on the prior geometry.
        fittedMask : surfa.Volume
            Support of ``fittedSegmentation``.

        Returns
        -------
        tuple of surfa.Volume
            Whole-field initialization labels and statistical support on the
            intensity-prior reference geometry.
        """
        # Validate the three source representations and their common geometry.
        requiredState = {
            'intensityPriorImage': self.intensityPriorImage,
            'inputSeg': self.inputSeg,
            'synthImage': self.synthImage,
            'preliminaryClassFractions': self.preliminaryClassFractions,
            'preliminaryLocalizerLabelGroups': (
                self.preliminaryLocalizerLabelGroups),
            'cheatingMeans': self.cheatingMeans,
        }
        missing = [
            name for name, value in requiredState.items() if value is None]
        if missing:
            raise RuntimeError(
                'Cannot build intensity-prior initialization state before: '
                + ', '.join(missing))
        for name, volume in (
                ('fitted segmentation', fittedSegmentation),
                ('fitted mask', fittedMask)):
            if not sf.transform.image_geometry_equal(
                    volume, self.intensityPriorImage, tol=1e-5):
                raise RuntimeError(
                    f'{name.capitalize()} must match the intensity-prior '
                    'geometry')

        # Materialize immutable localizer and collapsed-class evidence on the
        # whole-field intensity-prior grid.
        source = self.inputSeg.resample_like(
            self.intensityPriorImage, method='nearest')
        collapsed = self.synthImage.resample_like(
            self.intensityPriorImage, method='nearest')
        sourceLabels = np.asarray(source.data).astype(int, copy=False)
        collapsedLabels = np.asarray(collapsed.data).astype(int, copy=True)

        # Retain source labels only when the atlas and localizer namespaces give
        # them the same preliminary-class owner.
        atlasOwners, localizerOwners = self._preliminary_class_ownership()
        compatibleSource = np.zeros(sourceLabels.shape, dtype=bool)
        for label in np.unique(sourceLabels):
            atlasOwner = atlasOwners.get(int(label))
            localizerOwner = localizerOwners.get(int(label))
            if atlasOwner is not None and atlasOwner == localizerOwner:
                compatibleSource[sourceLabels == label] = True

        atlasLabels = np.asarray(self.FreeSurferLabels)
        backgroundStructures = np.flatnonzero(atlasLabels == 0)
        if len(backgroundStructures) != 1:
            raise RuntimeError(
                'Preliminary model must contain exactly one atlas background '
                'label')
        backgroundClass = np.flatnonzero(
            np.asarray(self.preliminaryClassFractions)[
                :, backgroundStructures[0]] > 0)
        if len(backgroundClass) != 1:
            raise RuntimeError(
                'Atlas background must have exactly one preliminary class '
                'owner')
        backgroundValue = np.asarray(self.cheatingMeans)[backgroundClass[0]]
        collapsedLabels[collapsedLabels == backgroundValue] = 0

        # Fitted reconstruction has highest precedence inside fitted support.
        merged = collapsedLabels
        merged[compatibleSource] = sourceLabels[compatibleSource]
        fittedSupport = fittedMask.data > 0
        merged[fittedSupport] = fittedSegmentation.data[fittedSupport]

        # Statistical support combines complete multichannel observations with
        # policy-expanded non-background localizer anatomy.
        intensityData = np.asarray(self.intensityPriorImage.framed_data)
        completeCase = np.all(
            np.isfinite(intensityData) & (intensityData != 0), axis=-1)
        anatomicalSupport = self._localizer_anatomical_support(
            self.intensityPriorImage)
        support = completeCase & anatomicalSupport

        segmentation = self.intensityPriorImage.new(
            merged.astype('int32', copy=False))
        segmentation.labels = self.labelMapping
        supportMask = self.intensityPriorImage.new(
            support.astype('uint8'))
        self.intensityPriorInitializationSegmentation = segmentation
        self.intensityPriorInitializationMask = supportMask
        return segmentation, supportMask

    def _preliminary_class_ownership(self):
        """Return independent atlas and localizer preliminary-class maps."""
        classFractions = np.asarray(self.preliminaryClassFractions)
        atlasMembership = classFractions > 0
        if atlasMembership.shape[1] != len(self.FreeSurferLabels):
            raise RuntimeError(
                'Preliminary atlas memberships do not align with the '
                'compression LUT')
        if np.any(np.count_nonzero(atlasMembership, axis=0) != 1):
            raise RuntimeError(
                'Every atlas structure must have exactly one preliminary '
                'class owner')

        atlasOwners = {
            int(label): int(classNumber)
            for structureNumber, label in enumerate(self.FreeSurferLabels)
            for classNumber in np.flatnonzero(
                atlasMembership[:, structureNumber])
        }
        localizerOwners = {}
        for classNumber, labels in enumerate(
                self.preliminaryLocalizerLabelGroups):
            for label in labels:
                label = int(label)
                if label in localizerOwners:
                    raise RuntimeError(
                        f'Localizer label {label} has multiple preliminary '
                        'class owners')
                localizerOwners[label] = classNumber
        return atlasOwners, localizerOwners

    def _estimate_intensity_hyperparameters(self, sameGaussianParameters):
        """Estimate current multichannel intensity hyperparameters.

        The current implementation takes per-channel medians from complete-case
        whole-field class support, prefers an eroded support when sufficient
        samples remain, and expresses support strength in regional-EM-equivalent
        voxels. Classes without usable class-specific observations invoke the
        model policy's configured mean-hyperparameter fallback.
        """
        if (self.intensityPriorInitializationSegmentation is None
                or self.intensityPriorInitializationMask is None
                or self.intensityPriorImage is None):
            raise RuntimeError(
                'prepare_for_image_fitting() must materialize whole-field '
                'initialization evidence before hyperparameters are estimated')
        if not sf.transform.image_geometry_equal(
                self.intensityPriorInitializationSegmentation,
                self.intensityPriorImage,
                tol=1e-5):
            raise RuntimeError(
                'Initialization labels must match the intensity-prior geometry')
        if not sf.transform.image_geometry_equal(
                self.intensityPriorInitializationMask,
                self.intensityPriorImage,
                tol=1e-5):
            raise RuntimeError(
                'Initialization support must match the intensity-prior geometry')
        if self.workingImage is None:
            raise RuntimeError(
                'Regional EM geometry is required to scale hyperparameter '
                'strengths')

        # Establish common observation support and voxel-volume scaling.
        data = np.asarray(self.intensityPriorImage.framed_data)
        numberOfChannels = data.shape[-1]
        componentCounts = (
            [parameter.numberOfComponents
             for parameter in self.sharedGMMParameters]
            if getattr(self, 'sharedGMMParameters', None) is not None
            else [1] * len(sameGaussianParameters))
        if len(componentCounts) != len(sameGaussianParameters):
            raise RuntimeError(
                'Configured component counts do not align with the resolved '
                'structural classes')
        numberOfGaussians = int(np.sum(componentCounts))
        meanHyper = np.empty(
            (numberOfGaussians, numberOfChannels), dtype='float64')
        nHyper = np.empty(numberOfGaussians, dtype='float64')
        initializationVariances = np.full(
            (numberOfGaussians, numberOfChannels, numberOfChannels),
            np.nan,
            dtype='float64')
        initializationMeans = np.full(
            (numberOfGaussians, numberOfChannels), np.nan, dtype='float64')
        labelsImage = self.intensityPriorInitializationSegmentation.data
        validSupport = self.intensityPriorInitializationMask.data > 0
        validSupport &= np.all(np.isfinite(data) & (data != 0), axis=-1)
        priorVoxelVolume = np.prod(
            self.intensityPriorImage.geom.voxsize)
        emVoxelVolume = np.prod(self.workingImage.geom.voxsize)
        aggregateSupport = validSupport & (labelsImage != 0)
        aggregateObservations = data[aggregateSupport, :]
        modelPolicy = self._ensure_model_policy()

        # Prefer support eroded by approximately 1 mm in physical space. Small
        # classes progressively relax that erosion before falling back to their
        # full support, preserving evidence rather than changing its strength.
        voxelSize = np.asarray(
            self.intensityPriorImage.geom.voxsize, dtype='float64')
        targetPhysicalRadius = 1.0
        extents = np.ceil(targetPhysicalRadius / voxelSize).astype(int)
        coordinates = np.ogrid[
            tuple(slice(-extent, extent + 1) for extent in extents)]
        squaredDistance = sum(
            (coordinate * spacing) ** 2
            for coordinate, spacing in zip(coordinates, voxelSize))
        voxelCenterDistances = np.sqrt(squaredDistance)
        representedRadii = np.unique(voxelCenterDistances[
            (voxelCenterDistances > 0)
            & (voxelCenterDistances < targetPhysicalRadius)])
        candidateRadii = [
            targetPhysicalRadius, *representedRadii[::-1].tolist()]
        erosionStructures = []
        for radius in candidateRadii:
            structure = voxelCenterDistances <= radius + 1e-12
            if not any(np.array_equal(structure, candidate)
                       for candidate in erosionStructures):
                erosionStructures.append(structure)

        # Select class support, invoking the mean-hyperparameter fallback only
        # when no usable class-specific observations exist.
        gaussianOffset = 0
        for classNumber, classLabels in enumerate(sameGaussianParameters):
            numberOfComponents = componentCounts[classNumber]
            gaussianNumbers = np.arange(
                gaussianOffset, gaussianOffset + numberOfComponents)
            gaussianOffset += numberOfComponents
            labels = np.asarray(classLabels)
            unErodedSupport = np.isin(labelsImage, labels) & validSupport
            if not np.any(unErodedSupport):
                if numberOfComponents > 1:
                    raise RuntimeError(
                        'No usable initialization evidence is available to '
                        f'identify the {numberOfComponents} Gaussian '
                        f'components of class {classNumber}')
                means, strength = modelPolicy.get_fallback_mean_hyperparameters(
                    aggregateObservations, numberOfChannels)
                meanHyper[gaussianNumbers[0], :] = means
                nHyper[gaussianNumbers[0]] = strength
                warnings.warn(
                    'No usable class-specific initialization evidence for '
                    f'class {classNumber} with labels {labels.tolist()}; '
                    'using configured mean-hyperparameter fallback',
                    RuntimeWarning)
                continue

            statisticsSupport = None
            for structure in erosionStructures:
                eroded = scipy.ndimage.binary_erosion(
                    unErodedSupport,
                    structure=structure,
                    border_value=1)
                if np.count_nonzero(eroded) >= 10:
                    statisticsSupport = eroded
                    break
            if statisticsSupport is None:
                statisticsSupport = unErodedSupport

            observations = data[statisticsSupport, :]
            strengthScale = priorVoxelVolume / emVoxelVolume
            if numberOfComponents == 1:
                meanHyper[gaussianNumbers[0], :] = np.median(
                    observations, axis=0)
                nHyper[gaussianNumbers[0]] = (
                    10 + len(observations) * strengthScale)
                continue

            if (len(observations) < numberOfComponents
                    or len(np.unique(observations, axis=0))
                    < numberOfComponents):
                raise RuntimeError(
                    'Initialization observations do not identify all '
                    f'{numberOfComponents} Gaussian components of class '
                    f'{classNumber}')

            # The configured component slots are exchangeable for the
            # currently supported disjoint classes. Use the same maintained,
            # deterministic Euclidean clustering as the SynthSeg refinement;
            # mature L1 and overlap-informed assignment remain later work.
            clustering = KMeans(
                n_clusters=numberOfComponents,
                random_state=0,
                n_init=10)
            assignments = clustering.fit_predict(observations)
            centers = clustering.cluster_centers_
            if (not np.all(np.isfinite(centers))
                    or np.any(np.bincount(
                        assignments, minlength=numberOfComponents) == 0)):
                raise RuntimeError(
                    'Initialization observations do not identify all '
                    f'{numberOfComponents} Gaussian components of class '
                    f'{classNumber}')

            # Give the otherwise anonymous components a deterministic order.
            order = np.lexsort(tuple(
                centers[:, axis]
                for axis in reversed(range(centers.shape[1]))))
            inverseOrder = np.empty_like(order)
            inverseOrder[order] = np.arange(numberOfComponents)
            centers = centers[order]
            assignments = inverseOrder[assignments]
            for componentNumber, gaussianNumber in enumerate(gaussianNumbers):
                componentObservations = observations[
                    assignments == componentNumber]
                meanHyper[gaussianNumber, :] = centers[componentNumber]
                nHyper[gaussianNumber] = (
                    len(componentObservations) * strengthScale)

                # This finite state is used only to subdivide the already
                # rasterized class prior before the first authoritative M-step.
                componentWeight = 1 / numberOfComponents
                effectiveMass = len(componentObservations) * componentWeight
                currentMean = (
                    componentWeight * np.sum(componentObservations, axis=0)
                    + nHyper[gaussianNumber] * centers[componentNumber]
                ) / (effectiveMass + nHyper[gaussianNumber])
                differences = componentObservations - currentMean
                covariance = (
                    componentWeight * differences.T @ differences
                    + nHyper[gaussianNumber]
                    * np.outer(
                        currentMean - centers[componentNumber],
                        currentMean - centers[componentNumber])
                ) / (effectiveMass + numberOfChannels + 2)
                if self.useDiagonalCovarianceMatrices:
                    covariance = np.diag(np.diag(covariance))
                try:
                    np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError as error:
                    raise RuntimeError(
                        'Initialization evidence did not produce a finite '
                        f'positive-definite covariance for class {classNumber}, '
                        f'component {componentNumber}') from error
                initializationMeans[gaussianNumber] = currentMean
                initializationVariances[gaussianNumber] = covariance

        self._initializationGaussianMeans = initializationMeans
        self._initializationGaussianVariances = initializationVariances

        return meanHyper, nHyper

    # -------------------------------------------------------------------------
    # Regional intensity fitting and deformation
    # -------------------------------------------------------------------------

    def prepare_for_image_fitting(self, compute_hyps=True):
        """Prepare regional intensity fitting and initialization evidence.

        The method establishes regional image and mesh state, rasterizes fitted
        full priors, runs the post-preliminary evidence lifecycle, and only then
        reduces classes and estimates ordinary intensity hyperparameters.
        """

        # Make sure the subclass has computed the target image
        if self.processedImage is None:
            sf.system.fatal('All MeshModel subclasses must compute processedImage during preprocessing!')

        # Crop the image by the aligned atlas and compute the new mesh alignment
        self.workingImage, self.transform = self.crop_image_by_atlas(self.processedImage)

        # GEMS accepts a scalar buffer for one channel and a framed buffer for
        # multiple channels.
        self.workingImage.data = np.asfortranarray(self.workingImage.data.squeeze())
        self.workingImageShape = self.workingImage.data.shape[:3]

        # Load the fitted mesh and transform it onto the regional EM grid.
        self.meshCollection = gems.KvlMeshCollection()
        self.meshCollection.read(self.warpedMeshFileName)
        self.meshCollection.transform(self.transform)
        self.meshCollection.k = self.meshStiffness

        # The first stored deformation is the fitted subject mesh.
        self.mesh = self.meshCollection.get_mesh(0)

        # Rasterize full priors once while the mesh still has full anatomical
        # alphas. They define both fitted support and the optional anatomical
        # refinement seam before structural class reduction.
        fullPriors = self.mesh.rasterize(self.workingImageShape)
        mask = (fullPriors.sum(-1) / 65535) > 0.99
        mask = self._apply_atlas_domain_interior_margin(
            mask,
            self.workingImage,
            self._ensure_model_policy().regionalAtlasDomainInteriorMarginInMm)

        # Regional fitting requires a three-dimensional complete-case mask;
        # every channel must be finite and nonzero at a retained voxel.
        regionalData = self.workingImage.data
        validVoxels = (
            np.isfinite(regionalData) & (regionalData != 0)
            if regionalData.ndim == 3
            else np.all(
                np.isfinite(regionalData) & (regionalData != 0), axis=-1))
        mask = np.asfortranarray(mask & validVoxels)

        # GEMS skips zero-valued voxels, so materialize the semantic mask in both
        # an explicit volume and the working image buffer.
        self.workingMask = self.workingImage.new(mask)
        self.workingImage[~mask, ...] = 0

        # Preserve GEMS' Fortran-order voxel indexing convention.
        self.maskIndices = np.unravel_index(
            np.where(mask.flatten(order='F'))[0],
            self.workingImageShape,
            order='F')

        # Write the initial and cropped/masked images for debugging purposes
        if self.debug:
            self.processedImage.save(os.path.join(self.tempDir, 'processedImage.mgz'))
            self.workingImage.save(os.path.join(self.tempDir, 'processedImageMasked.mgz'))
            self.workingMask.save(os.path.join(self.tempDir, 'processedImageMask.mgz'))

        self._prepare_intensity_initialization_evidence(fullPriors)

        # Materialize the supported one-hot structural model. The hard label
        # groups remain the input to the fitted initialization evidence, while
        # classFractions is the maintained-SAMSEG map used for alpha merging and
        # final structure posteriors.
        FreeSurferLabels = np.asarray(self.FreeSurferLabels)
        self.sameGaussianParameters = [
            FreeSurferLabels[fractions > 0].tolist()
            for fractions in self.classFractions
        ]
        self.reducedAlphas = kvlMergeAlphas(
            self.originalAlphas, self.classFractions)
        self.mesh.alphas = self.reducedAlphas

        self.gmm = None
        if compute_hyps:
            # Compute the hyperparameters
            self.meanHyper, self.nHyper = self.get_gaussian_hyps(
                self.sameGaussianParameters,
                self.mesh)
            componentCounts = [
                parameter.numberOfComponents
                for parameter in self.sharedGMMParameters]
            numberOfGaussians = int(np.sum(componentCounts))
            numberOfChannels = self.meanHyper.shape[1]
            hyperMixtureWeights = np.empty(
                numberOfGaussians, dtype='float64')
            gaussianOffset = 0
            for numberOfComponents in componentCounts:
                hyperMixtureWeights[
                    gaussianOffset:gaussianOffset + numberOfComponents
                ] = 1 / numberOfComponents
                gaussianOffset += numberOfComponents
            self.gmm = GMM(
                componentCounts,
                numberOfContrasts=numberOfChannels,
                useDiagonalCovarianceMatrices=(
                    getattr(self, 'useDiagonalCovarianceMatrices', False)),
                initialHyperMeans=self.meanHyper.copy(),
                initialHyperMeansNumberOfMeasurements=self.nHyper.copy(),
                initialHyperVariances=np.zeros(
                    (numberOfGaussians, numberOfChannels, numberOfChannels),
                    dtype='float64'),
                initialHyperVariancesNumberOfMeasurements=np.full(
                    numberOfGaussians, numberOfChannels + 2, dtype='float64'),
                initialHyperMixtureWeights=hyperMixtureWeights,
                initialHyperMixtureWeightsNumberOfMeasurements=np.zeros(
                    len(componentCounts), dtype='float64'))

    @staticmethod
    def _class_gaussian_slices(numberOfGaussiansPerClass):
        """Return contiguous Gaussian slices in configured class order."""
        offset = 0
        slices = []
        for count in numberOfGaussiansPerClass:
            slices.append(slice(offset, offset + count))
            offset += count
        return slices

    @staticmethod
    def _covariance_is_usable(covariance):
        """Return whether a covariance is finite and positive definite."""
        if not np.all(np.isfinite(covariance)):
            return False
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return False
        return True

    def _initialize_gmm_parameters(self, data, classPriors):
        """Create finite current GMM state from the rasterized class priors."""
        if self.gmm is None:
            raise RuntimeError(
                'Configured GMM hyperparameters are required before fitting')
        if self.gmm.tied:
            raise NotImplementedError(
                'Plus initialization does not support tied Gaussians')

        classSlices = self._class_gaussian_slices(
            self.gmm.numberOfGaussiansPerClass)
        means = self.gmm.hyperMeans.copy()
        variances = np.zeros(
            (self.gmm.numberOfGaussians,
             self.gmm.numberOfContrasts,
             self.gmm.numberOfContrasts),
            dtype='float64')
        mixtureWeights = self.gmm.hyperMixtureWeights.copy()

        initializationResponsibilities = np.zeros(
            (len(data), self.gmm.numberOfGaussians),
            dtype='float64', order='F')
        seedVariances = getattr(
            self, '_initializationGaussianVariances', None)

        for classNumber, gaussianSlice in enumerate(classSlices):
            classPrior = classPriors[:, classNumber]
            numberOfComponents = (
                gaussianSlice.stop - gaussianSlice.start)
            if numberOfComponents == 1:
                initializationResponsibilities[:, gaussianSlice.start] = (
                    classPrior)
                continue

            if seedVariances is None:
                raise RuntimeError(
                    f'No initialization covariance state is available for '
                    f'the {numberOfComponents} components of class '
                    f'{classNumber}')
            seedMeans = getattr(self, '_initializationGaussianMeans', None)
            if seedMeans is None:
                raise RuntimeError(
                    f'No initialization mean state is available for the '
                    f'{numberOfComponents} components of class {classNumber}')
            means[gaussianSlice] = seedMeans[gaussianSlice]
            variances[gaussianSlice] = seedVariances[gaussianSlice]
            for gaussianNumber in range(
                    gaussianSlice.start, gaussianSlice.stop):
                if not self._covariance_is_usable(
                        variances[gaussianNumber]):
                    raise RuntimeError(
                        'Initialization evidence did not produce finite '
                        f'covariance state for Gaussian {gaussianNumber}')

            weightedLikelihoods = np.zeros(
                (len(data), numberOfComponents), dtype='float64')
            for componentNumber, gaussianNumber in enumerate(range(
                    gaussianSlice.start, gaussianSlice.stop)):
                weightedLikelihoods[:, componentNumber] = (
                    self.gmm.getGaussianLikelihoods(
                        data,
                        np.expand_dims(means[gaussianNumber], 1),
                        variances[gaussianNumber])
                    * mixtureWeights[gaussianNumber])
            normalizer = np.sum(weightedLikelihoods, axis=1)
            supported = classPrior > 0
            if (not np.any(supported)
                    or np.any(~np.isfinite(normalizer[supported]))
                    or np.any(normalizer[supported] <= 0)):
                raise RuntimeError(
                    'Rasterized class prior and initialization evidence '
                    f'cannot identify all components of class {classNumber}')
            initializationResponsibilities[
                supported, gaussianSlice] = (
                    weightedLikelihoods[supported]
                    / normalizer[supported, np.newaxis]
                    * classPrior[supported, np.newaxis])

        # fitGMMParameters owns the established mean/covariance update. These
        # arrays merely provide writable current state for that first M-step.
        self.gmm.means = means
        self.gmm.variances = variances
        self.gmm.mixtureWeights = mixtureWeights
        self.gmm.fitGMMParameters(data, initializationResponsibilities)

        # Mature initialization carries the configured uniform weights into the
        # first ordinary likelihood E-step. Posterior-mass weights are learned
        # by the next ordinary M-step.
        self.gmm.mixtureWeights = mixtureWeights

        fallbackCovariance = None
        for gaussianNumber, covariance in enumerate(self.gmm.variances):
            if (np.all(np.isfinite(self.gmm.means[gaussianNumber]))
                    and self._covariance_is_usable(covariance)):
                continue
            classNumber = next(
                classIndex for classIndex, gaussianSlice in enumerate(classSlices)
                if gaussianSlice.start <= gaussianNumber < gaussianSlice.stop)
            if self.gmm.numberOfGaussiansPerClass[classNumber] != 1:
                raise RuntimeError(
                    'The first GMM M-step did not produce finite state for '
                    f'Gaussian {gaussianNumber}; multi-component state cannot '
                    'be inferred without usable initialization evidence')
            if fallbackCovariance is None:
                fallbackCovariance = (
                    self._ensure_model_policy()
                    .get_initial_gmm_fallback_covariance(self.gmm, data))
            if (fallbackCovariance is None
                    or not self._covariance_is_usable(fallbackCovariance)):
                raise RuntimeError(
                    'The first GMM M-step did not produce finite state for '
                    f'class {classNumber}, and no usable model-specific '
                    'fallback covariance is available')
            self.gmm.means[gaussianNumber] = self.gmm.hyperMeans[gaussianNumber]
            self.gmm.variances[gaussianNumber] = fallbackCovariance
            warnings.warn(
                'Using model-policy regional fitting covariance fallback for '
                f'class {classNumber}', RuntimeWarning)

        for classNumber, gaussianSlice in enumerate(classSlices):
            weights = self.gmm.mixtureWeights[gaussianSlice]
            if (not np.all(np.isfinite(weights))
                    or np.any(weights <= 0)
                    or not np.isclose(np.sum(weights), 1.0)):
                raise RuntimeError(
                    f'Initial mixture weights are invalid for class '
                    f'{classNumber}')

    def fit_mesh_to_image(self):
        """Fit one configured structural GMM through all resolution levels."""

        if self.gmm is None:
            raise RuntimeError(
                'prepare_for_image_fitting(compute_hyps=True) is required '
                'before authoritative Plus GMM fitting')

        modelPolicy = self._ensure_model_policy()
        imageBuffer = self.workingImage.data.copy(order='K')
        numMaskIndices = self.maskIndices[0].shape[-1]
        numberOfClasses = self.gmm.numberOfClasses

        numberOfMultiResolutionLevels = len(self.meshSmoothingSigmas)
        for multiResolutionLevel in range(numberOfMultiResolutionLevels):
            if self.isLong:
                self.mesh = self.meshCollection.get_mesh(0)

            if self.useTwoComponents and multiResolutionLevel == 1:
                raise NotImplementedError(
                    'A topology change requires a separately configured '
                    'target shared-GMM model; fixed-topology fitting keeps '
                    'the existing GMM across resolution levels')

            self.mesh.alphas = self.reducedAlphas

            meshSmoothingSigma = self.meshSmoothingSigmas[multiResolutionLevel]
            if meshSmoothingSigma > 0:
                print(f'Smoothing mesh collection with kernel size {meshSmoothingSigma:.4f}')
                self.meshCollection.smooth(meshSmoothingSigma)

            imageSigma = self.imageSmoothingSigmas[multiResolutionLevel]
            if imageSigma > 0:
                raise NotImplementedError('Image smoothing not implemented yet!')

            positionUpdatingMaximumNumberOfIterations = 30
            maximumNumberOfIterations = self.maxIterations[multiResolutionLevel]

            historyOfCost = []
            for iterationNumber in range(maximumNumberOfIterations):
                print(f'Iteration {iterationNumber + 1} of {maximumNumberOfIterations}')

                if imageBuffer.ndim == 3:
                    data = imageBuffer[self.maskIndices].reshape(-1, 1)
                else:
                    data = imageBuffer[self.maskIndices]
                    if data.ndim != 2:
                        data = data.reshape(-1, imageBuffer.shape[-1])

                classPriors = np.empty(
                    (numMaskIndices, numberOfClasses), dtype='float64')
                for classNumber in range(numberOfClasses):
                    classPriors[:, classNumber] = (
                        self.mesh.rasterize(
                            self.workingImageShape, classNumber)[self.maskIndices]
                        / 65535)

                historyOfEMCost = []
                completedEMIterations = 0
                if self.gmm.means is None:
                    self._initialize_gmm_parameters(data, classPriors)
                    completedEMIterations = 1
                gaussianPosteriors, minLogLikelihood = (
                    self.gmm.getGaussianPosteriors(data, classPriors))
                # TODO: The maintained historical prior score contributes one
                # additional 0.5 * log|Sigma| per Gaussian relative to the
                # covariance stationary point used by fitGMMParameters(). This
                # affects convergence monitoring, not parameter updates;
                # revisit only if realistic fitting shows a material effect.
                currentCost = (
                    minLogLikelihood
                    + self.gmm.evaluateMinLogPriorOfGMMParameters())
                if not np.isfinite(currentCost):
                    raise RuntimeError(
                        'Structural GMM objective is not finite')
                historyOfEMCost.append(currentCost)

                maximumEMIterations = modelPolicy.maximumGMMIterations
                while completedEMIterations < maximumEMIterations:
                    modelPolicy.update_gmm_parameters(
                        self.gmm, data, gaussianPosteriors)
                    completedEMIterations += 1
                    gaussianPosteriors, minLogLikelihood = (
                        self.gmm.getGaussianPosteriors(data, classPriors))
                    currentCost = (
                        minLogLikelihood
                        + self.gmm.evaluateMinLogPriorOfGMMParameters())
                    if not np.isfinite(currentCost):
                        raise RuntimeError(
                            'Structural GMM objective is not finite')

                    print('  '.join([
                        f'Res: {multiResolutionLevel + 1:03d}',
                        f'Iter: {iterationNumber + 1:03d} | '
                        f'{completedEMIterations:03d}',
                        f'MinLL: {currentCost:.4f}',
                    ]))

                    previousCost = historyOfEMCost[-1]
                    historyOfEMCost.append(currentCost)
                    if modelPolicy.has_gmm_converged(
                            previousCost,
                            currentCost,
                            completedEMIterations):
                        if currentCost > previousCost:
                            print('EM objective did not improve - stopping')
                        else:
                            print('EM converged!')
                        break

                if self.isLong:
                    self.mesh = self.meshCollection.get_mesh(0)

                haveMoved = False

                if imageBuffer.ndim == 3:
                    image_list = [gems.KvlImage(requireNumpyArray(imageBuffer))]
                else:
                    image_list = [
                        gems.KvlImage(requireNumpyArray(imageBuffer[..., idx]))
                        for idx in range(imageBuffer.shape[-1])]

                calculator = gems.KvlCostAndGradientCalculator(
                    typeName='AtlasMeshToIntensityImage',
                    images=image_list,
                    boundaryCondition='Sliding',
                    transform=self.transform,
                    means=self.gmm.means,
                    variances=self.gmm.variances,
                    mixtureWeights=self.gmm.mixtureWeights,
                    numberOfGaussiansPerClass=np.asarray(
                        self.gmm.numberOfGaussiansPerClass, dtype='int32'))

                maximalDeformationStopCriterion = 1e-10
                optimizationParameters = {
                    'Verbose': 0,
                    'MaximalDeformationStopCriterion': maximalDeformationStopCriterion,
                    'LineSearchMaximalDeformationIntervalStopCriterion': 1e-10,
                    'MaximumNumberOfIterations': 1000,
                    'BFGS-MaximumMemoryLength': 12
                }
                optimizer = gems.KvlOptimizer(self.optimizerType, self.mesh, calculator, optimizationParameters)

                for positionUpdatingIterationNumber in range(positionUpdatingMaximumNumberOfIterations):

                    # Calculate a good step
                    minLogLikelihoodTimesPrior, maximalDeformation = optimizer.step_optimizer_samseg()

                    # Log optimization information
                    iterationInfo = [
                        f'Res: {multiResolutionLevel + 1:03d}',
                        f'Iter: {iterationNumber + 1:03d} | {positionUpdatingIterationNumber + 1:03d}',
                        f'MaxDef: {maximalDeformation:.4f}',
                        f'MinLLxP: {minLogLikelihoodTimesPrior:.4f}',
                    ]
                    print('  '.join(iterationInfo))

                    if np.isnan(minLogLikelihoodTimesPrior):
                        print('error: minLogLikelihoodTimesPrior is NaN')

                    if maximalDeformation > 0:
                        haveMoved = True

                    # Check if we need to stop
                    if maximalDeformation <= maximalDeformationStopCriterion:
                        print('maximalDeformation is too small - stopping')
                        break

                previous = historyOfCost[-1] if historyOfCost else np.finfo(np.float32).max
                historyOfCost.append(minLogLikelihoodTimesPrior)

                if not haveMoved or (((previous - minLogLikelihoodTimesPrior) / minLogLikelihoodTimesPrior) < 1e-6):
                    break

    # -------------------------------------------------------------------------
    # Segmentation and outputs
    # -------------------------------------------------------------------------

    def extract_segmentation(self):
        """
        Extract discrete labels and volumes from the fit mesh.
        """

        # First, undo the collapsing of several structures into super-structures
        self.mesh.alphas = self.originalAlphas
        numberOfClasses = self.originalAlphas.shape[-1]
        numMaskIndices = self.maskIndices[0].shape[0]

        # Compute normalized posteriors
        imgdata = self.workingImage[self.maskIndices]
        if self.workingImage.data.ndim == 3:
            imgdata = imgdata.reshape(-1,1)
        else:
            imgdata = imgdata.reshape(-1, self.workingImage.data.shape[-1])

        priors = np.zeros(
            (numMaskIndices, numberOfClasses), dtype='float64')
        for classNumber in range(numberOfClasses):
            prior = self.mesh.rasterize(self.workingImageShape, classNumber)
            priors[:, classNumber] = prior[self.maskIndices] / 65535

        posteriors = self.gmm.getPosteriors(
            imgdata, priors, self.classFractions)
        posteriors = np.round(posteriors * 65535).astype('uint16')

        if self.debug:
            # Write the resulting atlas mesh to file for future reference
            self.meshCollection.write(os.path.join(self.tempDir, 'finalWarpedMesh.txt'))
            # Also write the warped mesh in atlas space
            inverseTransform = gems.KvlTransform(np.asfortranarray(np.linalg.inv(self.transform.as_numpy_array)))
            self.meshCollection.transform(inverseTransform)
            self.meshCollection.write(os.path.join(self.tempDir, 'finalWarpedMeshNoAffine.txt'))

        # Here we do a memory efficient computation of discrete labels and volumes
        self.volumes = {}
        inds = np.zeros(self.workingImageShape, dtype='int32')
        for i in range(numberOfClasses):

            if i == 0:
                sillyAlphas = np.zeros((len(self.originalAlphas), 2), dtype='float32')
                sillyAlphas[:, 0] = self.originalAlphas[:, 0]
                sillyAlphas[:, 1] = 1 - sillyAlphas[:, 0]
                self.mesh.alphas = sillyAlphas
                post = self.mesh.rasterize(self.workingImageShape)[..., 0]
                self.mesh.alphas = self.originalAlphas
            else:
                post = self.mesh.rasterize(self.workingImageShape, i)
            
            post[self.maskIndices] = posteriors[:, i]

            if i == 0:
                max_post = post
            else:
                M = post > max_post
                inds[M] = i
                max_post[M] = post[M]
            
            # Compute volume
            self.volumes[self.names[i]] = (self.resolution ** 3) * (post.sum() / 65535)

        # Compute all discrete labels and mask
        self.discreteLabels = self.workingImage.new(self.FreeSurferLabels[inds])
        self.discreteLabels[self.workingMask == 0] = 0
        lut_filename = os.path.join(os.environ.get('FREESURFER_HOME'), 'FreeSurferColorLUT.txt')
        self.discreteLabels.labels = sf.load_label_lookup(lut_filename)

        if self.debug:
            self.discreteLabels.save(os.path.join(self.tempDir, 'discreteLabelsAll.mgz'))

    def write_volumes(self, filename, volumes=None):
        """
        Write the cached volume dictionary to a text file.
        """
        if volumes is None:
            volumes = self.volumes
        with open(filename, 'w') as file:
            for name, volume in volumes.items():
                file.write(f'{name} {volume:.6f}\n')

    # -------------------------------------------------------------------------
    # Region extension seams
    # -------------------------------------------------------------------------

    def postprocess_segmentation(self):
        """Apply region-specific output filtering and write final results."""
        raise NotImplementedError('A MeshModel subclass must implement the postprocess_segmentation() function!')

    def get_cheating_label_groups(self):
        """Return preliminary groups when no shared-parameter artifact is set."""
        raise NotImplementedError('A MeshModel subclass must implement the get_cheating_label_groups() function!')

    def get_cheating_gaussians(self, sameGaussianParameters):
        """Return artificial Gaussians for preliminary segmentation fitting.

        The current configured-profile default uses the nonzero minimum label
        per localizer class and variance 0.01. Region subclasses may override
        this method when that convention does not represent their model.
        """
        if self.preliminaryLocalizerLabelGroups is None:
            raise NotImplementedError(
                'A MeshModel subclass without configured localizer groups '
                'must implement get_cheating_gaussians()')
        if any(not labels for labels in sameGaussianParameters):
            raise ValueError(
                'Every preliminary Gaussian requires at least one localizer '
                'label')
        means = np.asarray([
            max(1, min(labels)) for labels in sameGaussianParameters
        ], dtype=float)
        variances = np.full(
            len(sameGaussianParameters),
            0.01,
            dtype=float)
        return means, variances

    def get_gaussian_hyps(self, sameGaussianParameters, mesh):
        """Return mean and strength hyperparameters for the active classes."""
        raise NotImplementedError('A MeshModel subclass must implement the get_gaussian_hyps() function!')

    def get_second_label_groups(self):
        """Return target-stage grouping for a supported two-stage model."""
        raise NotImplementedError('A two-component MeshModel must implement the get_second_label_groups() function!')

    def get_second_gaussian_hyps(self, sameGaussianParameters, meanHyper, nHyper):
        """Return target-stage hyperparameters for a supported two-stage model."""
        raise NotImplementedError('A two-component MeshModel must implement the get_second_gaussian_hyps() function!')
