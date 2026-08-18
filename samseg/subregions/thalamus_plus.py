import os
import shutil
import numpy as np
import scipy.ndimage
import surfa as sf

from samseg.subregions import utils
from samseg.subregions.core_plus import MeshModelPlus


_PRELIMINARY_MODEL_PROFILE_FILES = {
    'aseg': {
        'sharedGMMParametersFileName': 'ASEGsharedGMMparameters.txt',
        'localizerLookupTableFileName': 'ASEGlocalizerLookupTable.txt',
        'modelPolicyFileName': 'ASEGmodelPolicy.json',
    },
    'synthseg': {
        'sharedGMMParametersFileName': 'SYNTHSEGsharedGMMparameters.txt',
        'localizerLookupTableFileName': 'SYNTHSEGlocalizerLookupTable.txt',
        'modelPolicyFileName': None,
    },
}


class ThalamicNucleiPlus(MeshModelPlus):

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

        self.preliminaryModelDirectory = preliminaryModelDirectory
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

        # Model thalamus with two components
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

        # When creating the smooth atlas alignment target, dilate before eroding
        self.atlasTargetSmoothing = 'forward'

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

        # Atlas alignment target is a masked segmentation
        match_labels = self._get_preliminary_affine_support_labels()
        mask = np.isin(self.inputSeg.data, match_labels).astype('float32') * 255
        self.atlasAlignmentTarget = self.inputSeg.new(mask)

        # Build the preliminary target without changing the source localizer.
        # VDC and thalamus intentionally share each hemispheric class because
        # their coarse-localizer boundary is not trusted for mesh deformation;
        # fitted full atlas priors reconstruct that distinction afterwards.
        self.synthImage = self._build_preliminary_synthetic_image(
            self.inputSeg)

        # And also used for image cropping around the thalamus
        thalamicMask = ((self.synthImage == self.THlabelLeft)
                        | (self.synthImage == self.THlabelRight))
        fixedMargin = int(np.round(15 / np.mean(self.inputSeg.geom.voxsize)))
        imageCropping = self.synthImage.new(thalamicMask).bbox(
            margin=fixedMargin)

        # Lastly, use it to make the image mask
        struct = np.ones((3, 3, 3))
        mask = scipy.ndimage.morphology.binary_dilation(self.synthImage > 1, structure=struct, iterations=2)
        imageMask = self.synthImage.new(mask)

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
        # Materialize one shared regional mask so channel processing cannot
        # accumulate channel-dependent mask interpolation.
        self.longMask = imageMask.resample_like(
            regionalReference, method='nearest')
        self.processedImage = self._resample_and_stack_intensity_channels(
            self.inputImageFileNames,
            regionalReference,
            'regionalIntensity',
            mask=self.longMask)

    def _get_preliminary_affine_support_labels(self):
        """Return localizer labels supporting thalamus affine alignment."""
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

    def postprocess_segmentation(self):
        """
        Post-process the segmentation and computed volumes.
        """

        # Recode segmentation
        A = self.discreteLabels.copy()
        A[(A < 100) & (A != 10) & (A != 49) ] = 0

        # Kill reticular labels
        leftReticular = self.labelMapping.search('Left-R', exact=True)
        rightReticular = self.labelMapping.search('Right-R', exact=True)
        A[A == leftReticular] = 0
        A[A == rightReticular] = 0

        # Get only connected components (sometimes the two thalami are not connected)
        left = utils.get_largest_cc((A < 8200) & ((A > 100) | (A == self.THlabelLeft)))
        right = utils.get_largest_cc((A > 8200) | (A == self.THlabelRight))
        cc_mask = left | right
        A[cc_mask == 0] = 0

        segFilePrefix = os.path.join(self.outDir, f'ThalamicNuclei{self.fileSuffix}')
        A.save(segFilePrefix + '.mgz')
        A.resample_like(self.inputSeg, method='nearest').save(segFilePrefix + '.FSvoxelSpace.mgz')

        # Prune the volumes to what we care about (also let's leave reticular 'R' out)
        validLabels = ['L-Sg', 'LGN', 'MGN', 'PuI', 'PuM', 'H', 'PuL',
                       'VPI', 'PuA', 'MV(Re)', 'Pf', 'CM', 'LP', 'VLa', 'VPL', 'VLp',
                       'MDm', 'VM', 'CeM', 'MDl', 'Pc', 'MDv', 'Pv', 'CL', 'VA', 'VPM',
                       'AV', 'VAmc', 'Pt', 'AD', 'LD']
        isValid = lambda name: (name.replace('Left-', '') in validLabels) or (name.replace('Right-', '') in validLabels)
        self.volumes = {name: vol for name, vol in self.volumes.items() if isValid(name)}

        # Sum up the total volumes per hemisphere 
        self.volumes['Left-Whole_thalamus'] = np.sum([vol for name, vol in self.volumes.items() if name.startswith('Left')])
        self.volumes['Right-Whole_thalamus'] = np.sum([vol for name, vol in self.volumes.items() if name.startswith('Right')])

        # Write the volumes
        self.write_volumes(segFilePrefix + '.volumes.txt')

    def get_label_groups(self):
        """
        Return a group (list of lists) of label names that determine the class reductions for
        the primary image-fitting stage.
        """
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

        # Configure left/right thalamic labels
        thalamicLabels = [
            'L-Sg', 'LGN', 'MGN', 'PuI', 'PuM', 'H', 'PuL', 'VPI', 'PuA', 'MV(Re)', 'Pf',
            'CM', 'LP', 'VLa', 'VPL', 'VLp', 'MDm', 'VM', 'CeM', 'MDl', 'Pc', 'MDv', 'Pv',
            'CL', 'VA', 'VPM', 'AV', 'VAmc', 'Pt', 'AD', 'LD',
        ]
        labelGroups.append([f'{side}-{label}' for side in ('Left', 'Right') for label in thalamicLabels])
        return labelGroups

    def get_gaussian_hyps(self, sameGaussianParameters, mesh):
        """Estimate intensity Gaussian hyperparameters after atlas fitting.

        Masks come from the full-label reconstruction produced by the fitted
        preliminary mesh, not from source-localizer boundaries that were
        deliberately collapsed during preliminary deformation.
        """
        if (self.initializationSegmentation is None
                or self.initializationMask is None):
            raise RuntimeError(
                'fit_mesh_to_seg() must reconstruct initialization state '
                'before Gaussian hyperparameters are estimated')

        nHyper = np.zeros(len(sameGaussianParameters))
        meanHyper = np.zeros(len(sameGaussianParameters))

        # TODO this needs to be adapted for multi-image cases (with masking)
        DATA = self.inputImages[0]

        for g in range(len(sameGaussianParameters)):
            
            labels = np.array(sameGaussianParameters[g])

            if len(labels) > 0:
                MASK = np.isin(
                    self.initializationSegmentation.data,
                    labels)
                MASK &= self.initializationMask.data
                radius = np.round(1 / np.mean(DATA.geom.voxsize))
                MASK = scipy.ndimage.morphology.binary_erosion(MASK, utils.spherical_strel(radius), border_value=1)
                total_mask = MASK & (DATA.data > 0)
                data = DATA.data[total_mask]
                meanHyper[g] = np.median(data)
                if any(labels == 28):
                    # Special case: VDE is kind of bimodal in FreeSurfer
                    nHyper[g] = 10
                else:
                    nHyper[g] = 10 + len(data) * np.prod(DATA.geom.voxsize) / (self.resolution ** 3)

        # If any NaN, replace by background
        # ATH: I don't there would ever be NaNs here?
        nans = np.isnan(meanHyper)
        meanHyper[nans] = 55
        nHyper[nans] = 10

        return (meanHyper, nHyper)

    def get_second_label_groups(self):
        """
        Return a group (list of lists) of label names that determine the class reductions for the
        second-component of the primary image-fitting stage.
        """
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
            ['Left-L-Sg', 'Left-LGN', 'Left-MGN', 'Left-H',
            'Left-VPI', 'Left-MV(Re)', 'Left-Pf', 'Left-CM', 'Left-LP', 'Left-VLa', 'Left-VPL', 'Left-VLp',
            'Left-VM', 'Left-CeM', 'Left-Pc', 'Left-MDv', 'Left-Pv', 'Left-CL', 'Left-VA', 'Left-VPM',
            'Left-AV', 'Left-VAmc', 'Left-Pt', 'Left-AD', 'Left-LD', 'Right-L-Sg', 'Right-LGN', 'Right-MGN', 'Right-H',
            'Right-VPI', 'Right-MV(Re)', 'Right-Pf', 'Right-CM', 'Right-LP', 'Right-VLa', 'Right-VPL', 'Right-VLp',
            'Right-VM', 'Right-CeM', 'Right-Pc', 'Right-MDv', 'Right-Pv', 'Right-CL', 'Right-VA', 'Right-VPM',
            'Right-AV', 'Right-VAmc', 'Right-Pt', 'Right-AD', 'Right-LD'],
            ['Left-PuA', 'Left-PuI', 'Left-PuL', 'Left-PuM', 'Left-MDl', 'Left-MDm',
            'Right-PuA', 'Right-PuI', 'Right-PuL', 'Right-PuM', 'Right-MDl', 'Right-MDm']
        ]
        return labelGroups

    def get_second_gaussian_hyps(self, sameGaussianParameters, meanHyper, nHyper):
        """
        Return a tuple of (meanHyps, nHyps) for Gaussian parameter estimation in the second-component
        of the primary image-fitting stage.
        """
        WMind = 1
        GMind = 2
        ThInt = meanHyper[-1]

        # TODO this needs to be enabled with non-T1s are used
        if True:
            # Lateral, brighter
            nHyper[-1] = 25
            meanHyper[-1] = ThInt + 5
            # Medial, darker
            nHyper = np.append(nHyper, 25)
            meanHyper = np.append(meanHyper, ThInt - 5)
        else:
            nHyper[-1] = 25
            nHyper = np.append(nHyper, 25)
            # Lateral, more WM-ish (e.g., darker, in FGATIR)
            meanHyper[-1] = ThInt * (0.95 + 0.1 * (meanHyper[WMind] >= meanHyper[GMind]))
            # Medial, more GM-ish (e.g., brighter, in FGATIR)
            meanHyper = np.append(meanHyper, ThInt * (0.95 + 0.1 * (meanHyper[WMind] < meanHyper[GMind])))
        return (meanHyper, nHyper)
