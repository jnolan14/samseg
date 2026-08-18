import os
import shutil
import tempfile
import numpy as np
import surfa as sf
import scipy.ndimage
import samseg
from samseg import gems
from samseg.io import kvlReadSharedGMMParameters
from samseg.merge_alphas import kvlGetMergingFractionsTable, kvlMergeAlphas
from samseg.utilities import requireNumpyArray
from samseg.subregions import utils
from samseg.subregions.model_policy import SubregionModelPolicy


class MeshModelPlus:

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
        ):
        """
        MeshModel is a generic base class to facilitate GEMS mesh deformation for given ROIs.
        To implement a mesh model for a particular set of structures, this class must be subclassed
        and the following functions MUST be implemented:

        self.preprocess_images()         : Precompute the mask, segmentation, and image volumes
        self.get_cheating_gaussians()    : Return the artificial preliminary Gaussian parameters
        self.get_label_groups()          : Return the reduced labels group used to fit the mesh to the image
        self.get_gaussian_hyps()         : Return the hyperparameters used to estimate the Gaussian parameters during image-fitting
        self.postprocess_segmentation()  : Update and write the label volumes and discrete segmentation(s)

        Copied region classes that have not yet migrated to shared preliminary
        parameters must also implement self.get_cheating_label_groups().

        Further information is documented in each function definition.

        This is a framework that was meant to facilitate an (almost perfect) port of the subfield matlab code.
        """

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

        # Keep the preliminary segmentation-fit model separate from the
        # structural image model that later successor stages will construct.
        self.cheatingMeans = None
        self.cheatingVariances = None
        self.preliminarySharedGMMParameters = None
        self.preliminaryClassFractions = None
        self.preliminaryClassNames = None
        self.modelPolicy = None
        # These localizer labels never contain numeric atlas memberships.
        self.preliminaryLocalizerLabelGroups = None
        self.preliminaryAlphas = None

        # Successor-owned structural lifecycle state. Later commits populate
        # these fields without reusing the preliminary Gaussian state.
        self.structuralInitializationSegmentation = None
        self.structuralInitializationMask = None
        self.structuralStage = None
        self.gmm = None
        self.bootstrapGMMState = None
        self.lastValidFittedGMMState = None
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

        # Here are some options that control how to much to dilate masks throughout different
        # stages. Might be necessary to tune depending on the geometry of the ROI (like brainstem).
        self.atlasTargetSmoothing = 'forward'
        self.cheatingAlphaMaskStrel = 3
        self.alphaMaskStrel = 5

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

        # Set the target mesh file paths
        self.warpedMeshFileName = os.path.join(self.tempDir, 'warpedOriginalMesh.txt')
        self.warpedMeshNoAffineFileName = os.path.join(self.tempDir, 'warpedOriginalMeshNoAffine.txt')

        # The input segmentation remains the immutable source localizer. Derived
        # preliminary and structural-initialization states are stored separately.
        self.inputSeg = sf.load_volume(self.inputSegFileName)
        self.inputImages = [sf.load_volume(path) for path in self.inputImageFileNames]
        self.correctedImages = [img.copy() for img in self.inputImages]
        self.highResImage = np.mean(self.inputImages[0].geom.voxsize) < 0.99

        # Now we define a set of volume members that must be properly computed during
        # the `preprocess_images` stage of all MeshModel subclasses. Further documentation below.
        self.preprocess_images()

    def preprocess_images(self):
        """
        Preprocess the input images for later processing. This function must be redefined in a subclass,
        and the following volumes (at the minimum) must be set during this stage:

            1. self.atlasAlignmentTarget : A binary tissue mask that acts as the target for the initial
                                          affine atlas registration.
            2. self.synthImage : A synthetic image generated from the input segmentation, used for the initial
                                fitting of the mesh to the subject (the `cheating` step).
            3. self.processedImage: An image (or set of images represented by each frame) used for the primary
                                    mesh fitting. It is expected that this image has been properly resampled
                                    to the working target resolution.

        It is expected that these are surfa.Volume objects with proper geometry information.
        """
        raise NotImplementedError('All subclasses of MeshModel must implement the preprocess_images() function!')

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
        # ATH: are alphas always 32-bit floats?
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

    def _ensure_preliminary_model_state(self):
        """Materialize the shared preliminary grouping state when possible.

        Parsing and class construction do not depend on a loaded mesh. Merged
        alphas are added later, once ``originalAlphas`` is available. Repeated
        calls only fill missing state.

        The label-list branch temporarily keeps copied region classes runnable
        until they are migrated to shared-parameter specifications.
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
            if self.modelPolicy is None:
                if self.modelPolicyFileName is None:
                    self.modelPolicy = SubregionModelPolicy()
                else:
                    self.modelPolicy = SubregionModelPolicy.read(
                        self.modelPolicyFileName)
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
        classNames = [
            parameter.mergedName for parameter in sharedGMMParameters]
        if not classNames:
            raise ValueError(
                'Preliminary shared-GMM parameters define no classes')
        if len(classNames) != len(set(classNames)):
            raise ValueError('Preliminary class names must be unique')

        policyMemberships = (
            self.modelPolicy.preliminaryLocalizerLabelMemberships)
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

        emptyClasses = [
            className for className, labels in zip(classNames, groups)
            if not labels
        ]
        if emptyClasses:
            raise ValueError(
                'Preliminary classes have no labels in the selected '
                'localizer vocabulary: ' + ', '.join(emptyClasses))
        return groups

    def _configure_preliminary_model_profile(
            self, profiles, requestedProfileName=None):
        """Select and configure one compatible preliminary model profile.

        Parameters
        ----------
        profiles : dict
            Region-provided mapping from profile names to resolved shared-GMM,
            localizer-LUT, and optional model-policy paths.
        requestedProfileName : str, optional
            Explicit profile selection. When omitted, canonical input naming
            and bounded localizer vocabularies are used for inference.

        Returns
        -------
        str
            Selected profile name.
        """
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError(
                'At least one preliminary model profile is required')

        requiredFields = {
            'sharedGMMParametersFileName',
            'localizerLookupTableFileName',
        }
        optionalFields = {'modelPolicyFileName'}
        supportedFields = requiredFields | optionalFields
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
                'modelPolicyFileName': profile.get('modelPolicyFileName'),
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

        selectedProfile = normalizedProfiles[selectedProfileName]
        for fieldName, description in (
                ('sharedGMMParametersFileName',
                 'shared-GMM parameter file'),
                ('modelPolicyFileName', 'model policy file')):
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
        self.modelPolicyFileName = selectedProfile['modelPolicyFileName']
        return selectedProfileName

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

    def _reconstruct_structural_initialization_state(self):
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
            Full-label structural-initialization segmentation and its valid
            fitted-atlas support mask, both in the first structural image grid.

        Raises
        ------
        RuntimeError
            If the configured preliminary model or fitted mesh state is
            incomplete.
        """
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
                'Cannot reconstruct structural initialization before the '
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

        cheatingMeans = np.asarray(self.cheatingMeans)
        if cheatingMeans.shape != (classFractions.shape[0],):
            raise RuntimeError(
                'Preliminary means do not align with preliminary classes')
        structureMeans = cheatingMeans[structureClassNumbers]
        classEvidence = (
            np.asarray(self.workingImage.data)[..., np.newaxis]
            == structureMeans)
        scores = np.where(classEvidence, fullPriors, 0)

        priorMass = np.sum(fullPriors, axis=-1, dtype=np.uint64)
        validSupport = priorMass > (0.99 * 65535)
        missingEvidence = validSupport & ~np.any(scores > 0, axis=-1)
        scores[missingEvidence] = fullPriors[missingEvidence]

        winningStructures = np.argmax(scores, axis=-1)
        labels = np.asarray(self.FreeSurferLabels)[winningStructures]
        labels = labels.copy()
        labels[~validSupport] = 0

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

        # Let's smooth the mask a bit (maybe) with one dilation and erosion pass
        if self.atlasTargetSmoothing == 'forward':
            strel = utils.spherical_strel(1)
            mask.data = scipy.ndimage.morphology.binary_dilation(mask.data, strel)
            mask.data = scipy.ndimage.morphology.binary_erosion(mask.data, strel, border_value=1)
        elif self.atlasTargetSmoothing == 'backward':
            strel = utils.spherical_strel(1)
            mask.data = scipy.ndimage.morphology.binary_erosion(mask.data, strel, border_value=1)
            mask.data = scipy.ndimage.morphology.binary_dilation(mask.data, strel)
        elif self.atlasTargetSmoothing is not None:
            sf.system.fatal(f'Unknown atlasTargetSmoothing option `{self.atlasTargetSmoothing}`.')

        # We're going to use mri_robust_register for this registration, so let's ensure the mask
        # value is 255 and we'll write to disk
        mask.data = mask.data.astype('float32') * 255
        targetMaskFile = os.path.join(self.tempDir, 'targetMask.mgz')
        mask.save(targetMaskFile)

        # Write the atlas as well
        alignedAtlasFile = os.path.join(self.tempDir, 'alignedAtlasImage.mgz')
        # ATH skipping this for now... we'll just copy instead
        # self.atlasImage.save(alignedAtlasFile)
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
        if self.cheatingAlphaMaskStrel > 0:
            mask = scipy.ndimage.morphology.binary_erosion(mask, utils.spherical_strel(self.cheatingAlphaMaskStrel), border_value=1)
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

        # Write the inital and cropped/masked images for debugging purposes
        if self.debug:
            self.synthImage.save(os.path.join(self.tempDir, 'synthImage.mgz'))
            self.workingImage.save(os.path.join(self.tempDir, 'synthImageMasked.mgz'))

    def fit_mesh_to_seg(self):
        """
        The second processing step involves deforming the roughly-aligned mesh to the subject segmentation.
        This is the initial 'cheating' step and requires that the synthImage volume has been properly configured
        during preprocessing.
        """

        # Just get the image buffer (array) and convert to a Kvl image object
        imageBuffer = self.workingImage.data.copy(order='K')
        image = gems.KvlImage(requireNumpyArray(imageBuffer))

        # Use a multi-resolution approach
        for multiResolutionLevel, meshSmoothingSigma in enumerate(self.cheatingMeshSmoothingSigmas):

            # Set mesh alphas
            self.mesh.alphas = self.preliminaryAlphas

            # It's good to smooth the mesh, otherwise we get weird compressions of the mesh along the boundaries
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
        # coordinates, then materialize the structural-initialization state.
        self.mesh.alphas = self.originalAlphas 
        if self.preliminarySharedGMMParameters is not None:
            (self.structuralInitializationSegmentation,
             self.structuralInitializationMask) = (
                self._reconstruct_structural_initialization_state())

        # Assign fitted positions to the first training-subject warp before
        # returning the collection to native atlas space.
        self.meshCollection.set_positions(self.originalNodePositions, [self.mesh.points])

        # Write the resulting atlas mesh to file in native atlas space.
        # This is nice because all we need to do is to modify imageDump_coregistered
        # with the T1-to-T2 transform to have the warped mesh in T2 space
        inverseTransform = gems.KvlTransform(np.asfortranarray(np.linalg.inv(self.transform.as_numpy_array)))
        self.meshCollection.transform(inverseTransform)
        self.meshCollection.write(self.warpedMeshFileName)

    def prepare_for_image_fitting(self, compute_hyps=True):
        """
        Prepare the mesh collection, preprocessed image, reduced alphas, and estimated hyperparameters.
        """

        # Make sure the subclass has computed the target image
        if self.processedImage is None:
            sf.system.fatal('All MeshModel subclasses must compute processedImage during preprocessing!')

        # Crop the image by the aligned atlas and compute the new mesh alignment
        self.workingImage, self.transform = self.crop_image_by_atlas(self.processedImage)

        # ATH for now let's squeeze the data, but will need to adapt something better
        # for multi-image cases down the road
        self.workingImage.data = np.asfortranarray(self.workingImage.data.squeeze())
        self.workingImageShape = self.workingImage.data.shape[:3]

        # Read the atlas mesh from file, and apply the previously determined transform to the location of its nodes
        # ATH does this have to be re-read?
        self.meshCollection = gems.KvlMeshCollection()
        self.meshCollection.read(self.warpedMeshFileName)
        self.meshCollection.transform(self.transform)
        self.meshCollection.k = self.meshStiffness

        # Retrieve the correct mesh to use from the meshCollection
        self.mesh = self.meshCollection.get_mesh(0)

        # We're not interested in image areas that fall outside our cuboid ROI where our atlas is defined. Therefore,
        # generate a mask of what's inside the ROI. Also, by convention we're skipping all voxels with zero intensity.
        mask = (self.mesh.rasterize(self.workingImageShape).sum(-1) / 65535) > 0.99
        if self.alphaMaskStrel > 0:
            mask = scipy.ndimage.morphology.binary_erosion(mask, utils.spherical_strel(self.alphaMaskStrel), border_value=1)

        # mask must be 3D to properly index the priors; test for non-0 voxels along stacked dim
        # and force it to be 3D in the case of multi channel before creating the final mask 
        validVoxels = self.workingImage.data > 0 if self.workingImage.data.ndim == 3 else np.all(self.workingImage.data > 0, axis=-1)
        mask = np.asfortranarray(mask & validVoxels)

        # Apply the mask to the image we're analyzing by setting the intensity of all voxels not belonging
        # to the brain mask to zero. This will automatically discard those voxels in subsequent C++ routines, as
        # voxels with intensity zero are simply skipped in the computations.
        self.workingMask = self.workingImage.new(mask)
        self.workingImage[~mask, ...] = 0

        ### DEBUG
        print(f'mask_shape: {mask.shape}')
        print(f'working_mask: {self.workingMask.shape}')


        # Let's do this to make results more similar to the matlab version
        #self.maskIndices = np.unravel_index(np.where(mask.flatten(order='F')), self.workingImageShape, order='F')
        self.maskIndices = np.unravel_index(np.where(mask.flatten(order='F'))[0], self.workingImageShape, order='F')

        #$# debug 
        print(f"maskIndices type: {type(self.maskIndices)}")
        print(f"maskIndices length: {len(self.maskIndices)}")
        for i, m in enumerate(self.maskIndices):
            print(f"maskIndices[{i}].shape: {m.shape}")

        # Write the initial and cropped/masked images for debugging purposes
        if self.debug:
            self.processedImage.save(os.path.join(self.tempDir, 'processedImage.mgz'))
            self.workingImage.save(os.path.join(self.tempDir, 'processedImageMasked.mgz'))
            self.workingMask.save(os.path.join(self.tempDir, 'processedImageMask.mgz'))

        # Compute the Gaussian label groups
        labelGroups = self.get_label_groups()
        self.sameGaussianParameters = self.label_group_names_to_indices(labelGroups)

        # Compute the reduced alphas
        self.reducedAlphas, self.reducingLookupTable = self.reduce_alphas(self.sameGaussianParameters)
        self.mesh.alphas = self.reducedAlphas

        if compute_hyps:
            # Compute the hyperparameters
            self.meanHyper, self.nHyper = self.get_gaussian_hyps(self.sameGaussianParameters, self.mesh)

        # Init empty means and variances
        self.means = None
        self.variances = None

    def fit_mesh_to_image(self):
        """
        Fit mesh to the image data.
        """

        # Just get the original image buffer (array) and convert to a Kvl image object
        imageBuffer = self.workingImage.data.copy(order='K')
        image = gems.KvlImage(requireNumpyArray(imageBuffer))

        # Useful to have cached
        numMaskIndices = self.maskIndices[0].shape[-1]
        numberOfClasses = len(self.sameGaussianParameters)

        # Multi-resolution loop
        numberOfMultiResolutionLevels = len(self.meshSmoothingSigmas)
        for multiResolutionLevel in range(numberOfMultiResolutionLevels):

            if self.isLong:
                self.mesh = self.meshCollection.get_mesh(0)

            # Special case when we want to recompute reduced alphas for a second-component
            # Note: how should we deal with more than one component during longitudinal global iterations?
            if self.useTwoComponents and multiResolutionLevel == 1:
                # Get second component label groups
                labelGroups = self.get_second_label_groups()
                self.sameGaussianParameters = self.label_group_names_to_indices(labelGroups)
                numberOfClasses = len(self.sameGaussianParameters)
                self.reducedAlphas, self.reducingLookupTable = self.reduce_alphas(self.sameGaussianParameters)
                self.mesh.alphas = self.reducedAlphas
                # Compute new Gaussian hyperparameters
                self.meanHyper, self.nHyper = self.get_second_gaussian_hyps(self.sameGaussianParameters, self.meanHyper, self.nHyper)
                # Reset means and variances to be computed
                self.means = None
                self.variances = None

            # Set the mesh alphas back
            # ATH is this necessary though?
            self.mesh.alphas = self.reducedAlphas

            # Smooth the mesh using a Gaussian kernel
            meshSmoothingSigma = self.meshSmoothingSigmas[multiResolutionLevel]
            if meshSmoothingSigma > 0:
                print(f'Smoothing mesh collection with kernel size {meshSmoothingSigma:.4f}')
                self.meshCollection.smooth(meshSmoothingSigma)

            # Smooth the image using a Gaussian kernel
            imageSigma = self.imageSmoothingSigmas[multiResolutionLevel]
            if imageSigma > 0:
                raise NotImplementedError('Image smoothing not implemented yet!')

            # ATH this is in case the above smoothing only sets the buffer, but this should be removed
            # really since it's not necessary if things are correctly implemented
            image = gems.KvlImage(requireNumpyArray(imageBuffer))
            
            # Now with this smoothed atlas, we're ready for the real work. There are essentially two sets of parameters
            # to estimate in our generative model: (1) the mesh node locations (parameters of the prior), and (2) the
            # means and variances of the Gaussian intensity models (parameters of the
            # likelihood function, which is really a hugely simplistic model of the MR imaging process). Let's optimize
            # these two sets alternately until convergence.

            positionUpdatingMaximumNumberOfIterations = 30
            maximumNumberOfIterations = self.maxIterations[multiResolutionLevel]
            
            historyOfCost = []
            for iterationNumber in range(maximumNumberOfIterations):
                print(f'Iteration {iterationNumber + 1} of {maximumNumberOfIterations}')

                # Part I: estimate Gaussian mean and variances using EM

                # Get the priors as dictated by the current mesh position as well as the image intensities
                ##$## This is where the 3D <-> 4D mask indices matter, need to build the 4D here
                #data = imageBuffer[self.maskIndices]

                if imageBuffer.ndim == 3:
                    data = imageBuffer[self.maskIndices].reshape(-1, 1)

                else:
                    data = imageBuffer[self.maskIndices]
                    if data.ndim != 2:
                        data = data.reshape(-1, imageBuffer.shape[-1])

                # Avoid spike in memory during the posterior computation
                priors = np.zeros((numMaskIndices, numberOfClasses), dtype='uint16')
                for l in range(numberOfClasses):
                    """
                        Rasterization will always return a 3D data shape
                        Stack of input images will always be 4D for intensity volumes

                        Add axis to the rasterization of the mask and then broadcast to be the proper shape in the 4th dim to match the input stack, this should allow us to index it based on the list of true indices 
                    """
                    priors[:, l] = self.mesh.rasterize(self.workingImageShape, l)[self.maskIndices]

                posteriors = priors / 65535

                # Start EM iterations. Initialize the parameters if this is the first time ever you run this
                if (self.means is None) or (self.variances is None):

                    n_channels = 1 if imageBuffer.ndim == 3 else imageBuffer.shape[-1]
                    self.means = np.zeros((numberOfClasses, n_channels))
                    self.variances = np.zeros((numberOfClasses, n_channels))

                    thresh = 1e-2
                    for classNumber in range(numberOfClasses):
                        posterior = posteriors[:, classNumber]
                        if np.sum(posterior) > thresh:

                            mu = (self.meanHyper[classNumber] * self.nHyper[classNumber] + data.T @ posterior) / (self.nHyper[classNumber] + np.sum(posterior) + thresh)
                            variance = (((data - mu) ** 2).T @ posterior + self.nHyper[classNumber] * (mu - self.meanHyper[classNumber]) ** 2) / (np.sum(posterior) + thresh)
                            self.means[classNumber] = mu
                            self.variances[classNumber] = variance + thresh
                        else:
                            self.means[classNumber] = self.meanHyper[classNumber]
                            self.variances[classNumber] = 100

                    # Prevents NaNs during the optimization
                    self.variances[self.variances == 0] = 100

                stopCriterionEM = 1e-5
                historyOfEMCost = []
                for EMIterationNumber in range(100):

                    # E-step: compute the posteriors based on the current parameters

                    minLogLikelihood = 0
                    for classNumber in range(numberOfClasses):
                        

                        mu = self.means[classNumber]
                        variance = self.variances[classNumber]
                        prior = priors[:, classNumber] / 65535

                        log_likelihood = -0.5 * np.sum(((data - mu) ** 2) / variance + np.log(2 * np.pi * variance), axis=1)

                        posteriors[:, classNumber] = np.exp(log_likelihood) * prior

                        minLogLikelihood = minLogLikelihood + 0.5 * np.sum(np.log(2 * np.pi * variance)) - 0.5 * np.log(self.nHyper[classNumber]) + 0.5 * np.sum((self.nHyper[classNumber] / variance) * (mu - self.meanHyper[classNumber]) ** 2)
                        
                    normalizer = np.sum(posteriors, -1) + np.finfo(np.float32).eps
                    posteriors /= normalizer[..., np.newaxis]
                    minLogLikelihood = minLogLikelihood - np.sum(np.log(normalizer))  # This is what we're optimizing with EM
                    if np.isnan(minLogLikelihood):
                        sf.system.fatal('minLogLikelihood is NaN')

                    # Log some iteration information
                    iterationInfo = [
                        f'Res: {multiResolutionLevel + 1:03d}',
                        f'Iter: {iterationNumber + 1:03d} | {EMIterationNumber + 1:03d}',
                        f'MinLL: {minLogLikelihood:.4f}',
                    ]
                    print('  '.join(iterationInfo))

                    # Track EM history
                    previous = historyOfEMCost[-1] if historyOfEMCost else np.finfo(np.float32).max
                    historyOfEMCost.append(minLogLikelihood)

                    # Check for convergence
                    relativeChangeCost = (previous - minLogLikelihood) / minLogLikelihood
                    if relativeChangeCost < stopCriterionEM:
                        print('EM converged!')
                        break

                    # M-step: derive parameters from the posteriors

                    # Update parameters of Gaussian mixture model
                    thresh = 1e-2
                    for classNumber in range(numberOfClasses):
                        posterior = posteriors[:, classNumber]
                        if np.sum(posterior) > thresh:
                            mu = (self.meanHyper[classNumber] * self.nHyper[classNumber] + data.T @ posterior) / (self.nHyper[classNumber] + np.sum(posterior) + thresh)
                            variance = (((data - mu) ** 2).T @ posterior + self.nHyper[classNumber] * (mu - self.meanHyper[classNumber]) ** 2) / (np.sum(posterior) + thresh)
                            self.means[classNumber] = mu
                            self.variances[classNumber] = variance + thresh
                        else:
                            self.means[classNumber] = self.meanHyper[classNumber]
                            self.variances[classNumber] = 100

                    # Prevents NaNs during the optimization
                    self.variances[self.variances == 0] = 100

                # Part II: update the position of the mesh nodes for the current set of Gaussian parameters

                if self.isLong:
                    self.mesh = self.meshCollection.get_mesh(0)

                # Keep track if the mesh has moved or not
                haveMoved = False

                ##$ reshape the variances
                full_variance = np.zeros((self.means.shape[0], self.means.shape[1], self.means.shape[1]))
                for i in range(self.means.shape[0]):
                    full_variance[i] = np.diag(self.variances[i])

                ##$ handle building the image list for single and multi channel
                if imageBuffer.ndim == 3:
                    n_channels = 1
                    image_list = [gems.KvlImage(requireNumpyArray(imageBuffer))]
                else:
                    n_channels = imageBuffer.shape[-1]
                    image_list = [gems.KvlImage(requireNumpyArray(imageBuffer[..., idx])) for idx in range(n_channels)]
                
                # Note that it uses variances instead of precisions
                calculator = gems.KvlCostAndGradientCalculator(
                    typeName='AtlasMeshToIntensityImage',
                    images=image_list,
                    boundaryCondition='Sliding',
                    transform=self.transform,
                    means=self.means,
                    variances=full_variance,
                    mixtureWeights=np.ones(len(self.means), dtype='float32'),
                    numberOfGaussiansPerClass=np.ones(len(self.means), dtype='int32'))

                # Get optimizer and plug calculator into it
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

                # Keep track of the cost function we're optimizing
                previous = historyOfCost[-1] if historyOfCost else np.finfo(np.float32).max
                historyOfCost.append(minLogLikelihoodTimesPrior)
                
                # Determine if we should stop the overall iterations over the two set of parameters
                if not haveMoved or (((previous - minLogLikelihoodTimesPrior) / minLogLikelihoodTimesPrior) < 1e-6):
                    break

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

        posteriors = np.zeros((numMaskIndices, numberOfClasses), dtype='float32')

        for classNumber in range(numberOfClasses):
            prior = self.mesh.rasterize(self.workingImageShape, classNumber)
            mu = self.means[self.reducingLookupTable[classNumber]]
            variance = self.variances[self.reducingLookupTable[classNumber]]

            # changed to handle multiple channels
            log_likelihood = -0.5 * np.sum(((imgdata - mu) ** 2) / variance + np.log(2 * np.pi * variance), axis=-1)
            posteriors[:, classNumber] = np.exp(log_likelihood) * (prior[self.maskIndices] / 65535)
            #posteriors[:, classNumber] = (np.exp(-(imgdata - mu) ** 2 / 2 / variance) * (prior[self.maskIndices[:3]] / 65535)) / np.sqrt(2 * np.pi * variance)

        normalizer = np.sum(posteriors, -1) + np.finfo(np.float32).eps
        posteriors /= normalizer[..., np.newaxis]
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

    def postprocess_segmentation(self):
        """
        This function should perform any necessary modifications to and write the discreteLabels segmentation and labelVolumes.
        """
        raise NotImplementedError('A MeshModel subclass must implement the postprocess_segmentation() function!')

    def get_cheating_label_groups(self):
        """
        This function should return a group (list of lists) of label names that determine the class
        reductions for the initial segmentation-fitting stage.
        """
        raise NotImplementedError('A MeshModel subclass must implement the get_cheating_label_groups() function!')

    def get_cheating_gaussians(self, sameGaussianParameters):
        """Return artificial Gaussians for preliminary segmentation fitting.

        Configured profiles default to the established nonzero minimum label
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

    def get_label_groups(self):
        """
        This function should return a group (list of lists) of label names that determine the class
        reductions for the primary image-fitting stage.
        """
        raise NotImplementedError('A MeshModel subclass must implement the get_label_groups() function!')

    def get_gaussian_hyps(self, sameGaussianParameters, mesh):
        """
        This function should return a tuple of (meanHyps, nHyps) for Gaussian parameter estimation.
        """
        raise NotImplementedError('A MeshModel subclass must implement the get_gaussian_hyps() function!')

    def get_second_label_groups(self):
        """
        This optional function should return a group (list of lists) of label names that determine the class
        reductions for the second-component of the primary image-fitting stage.
        """
        raise NotImplementedError('A two-component MeshModel must implement the get_second_label_groups() function!')

    def get_second_gaussian_hyps(self, sameGaussianParameters, meanHyper, nHyper):
        """
        This optional function should return a tuple of (meanHyps, nHyps) for Gaussian parameter estimation
        in the second-component of the primary image-fitting stage.
        """
        raise NotImplementedError('A two-component MeshModel must implement the get_second_gaussian_hyps() function!')
