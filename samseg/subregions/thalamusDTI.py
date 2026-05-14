import os
import glob
import shutil
import json
import tempfile
import numpy as np
import scipy.ndimage
import surfa as sf

# for k-means clustering
from sklearn.cluster import KMeans

from samseg.subregions import utils
from samseg.subregions.core import MeshModel

# define the thalamusDTI class as an extension of MeshModel
# We have some additional args and maybe slightly different preprocessing we'll
# try and integrate/extend the thalamus class later


class ThalamicNucleiDTI(MeshModel):
    def __init__(
        self,
        atlasDir,  # =os.path.join(os.environ.get('FREESURFER_HOME'), 'average/ThalamicNuclei/atlas_DTI'),
        outDir,
        inputImageFileNames,  # HACK: see initialize note
        inputSegFileName,
        inputDTIDirName,  # NEW
        dtiLikelihood,  # NEW
        meshStiffness=0.05,
        optimizerType="L-BFGS",  # is this where we should set the gems calculator type? Nope
        bbregisterMode=None,
        resolution=0.5,
        useTwoComponents=True,  # maybe we just hard code this? nothing depends on this in the super.__init__, so we could always hard code post call
        tempDir=None,
        fileSuffix="_thalamus_joint",
        debug=True,
    ):
        # call to the parent constructor
        super().__init__(
            atlasDir=atlasDir,
            outDir=outDir,
            inputImageFileNames=inputImageFileNames,
            inputSegFileName=inputSegFileName,
            meshStiffness=meshStiffness,
            optimizerType=optimizerType,
            bbregisterMode=bbregisterMode,
            resolution=resolution,
            useTwoComponents=useTwoComponents,
            tempDir=tempDir,
            fileSuffix=fileSuffix,
            debug=debug,
        )

        # Just like the 'expert options' are just hard coded, we can hard code
        # the paths to the default atlas/shared params files in __init__

        # set the paths
        # self.outDir = outDir                # set in super
        # self.atlasDir = atlasDir            # set in super

        # overwrite these fields hard coded in super
        self.atlasMeshFileName = os.path.join(
            atlasDir, "tractAtlasCorrected.txt.gz"
        )  # overwrite
        self.atlasDumpFileName = os.path.join(atlasDir, "AtlasDump.mgz")  # overwrite
        self.compressionLookupTableFileName = os.path.join(
            atlasDir, "tractAtlasCorrectedCompressionLUT.txt"
        )  # overwrite

        self.inputDTIDirName = inputDTIDirName  # unique
        self.dtiLikelihood = dtiLikelihood  # unique
        self.atlasTargetSmoothing = (
            "forward"  # we might want an arg for this, same as thalamus
        )

        # mesh fitting params
        # cheating stuff is all for the aseg registration step,
        # 2 channel is anything named gmm
        # full thing is 'thing'...
        self.cheatingMeshSmoothingSigmas = [
            3.0,
            2.0,
        ]  # ??? stole this from thalamus.py, what is this in henry's? is it calculated on the fly?
        # above should be in the aseg reg function

        self.cheatingMaxIterations = [300, 150]  # ASEG_max_def_its_default

        self.meshSmoothingSigmas = [
            1.5,
            1.125,
            0.75,
            0.375,
            0,
        ]  # meshSmoothingSigmas_default
        self.imageSmoothingSigmas = [0, 0, 0, 0, 0]  # ??? what is this in henry's code?
        self.maxIterations = [7, 5, 5, 3, 3]  # max_def_itsGMM_default

        # REMOVE THIS, FOR DEBUGGING ONLY
        self.bp = False

        # TODO: figure out where the missing smoothing sigmas come from, test in python shell

    # def cleanup() DON'T NEED TO OVERWRITE, CALL FROM SUPER

    def parse_dti_dir(self):
        """
        Parse the dti directory, set references to necessary files/inputs
        We make the gross assumption here that we have the T1 in the first spot
        of the inputImageFileNames var
        NOTE: is it better to set the name for the struct2diff reg'd files, or
        just update the value of the structural vol?
        """
        # Find the struct2diff lta file
        lta_search_path = os.path.join(
            self.inputDTIDirName, "xfms", "anatorig2diff.*lta"
        )
        matches = glob.glob(lta_search_path)
        if not matches:
            print(
                "WARNING: Could not locate the structural to DTI space LTA file, assuming structural scans are aligned to DTI space"
            )
            self.structToDiffLTAPath = None
        else:
            self.structToDiffLTAPath = matches[0]

        # find the FA volume
        # NOTE: do we need different ones depending on the calculator?
        FAvol_search_path = os.path.join(self.inputDTIDirName, "dtifit*FA.nii.gz")
        matches = glob.glob(FAvol_search_path)
        if not matches:
            sf.system.fatal("Could not locate the FA volume in the DTI dir")
        # may need to make this more robust depending on whats in the DTI dir
        self.FAImagePath = matches[0]

    def parse_grouping_json(self, grouping_json, GMMfile):
        # TODO: we need to either/or set json_path in __init__ based on atlas dir, allow user to specify, for now, we'll hard code a path for testing

        # grouping_json = json.load(open(grouping_json,'r'))

        # $# TODO: Need to clean this up, no way to pass the GMM to the inner calls, but we need it here to get the first col to set the json keys
        labelGroups = self.get_label_groups()
        labelIdx = self.label_group_names_to_indices(labelGroups)

        # read the GMM file
        with open(GMMfile, "r") as f:
            GMM_lines = f.readlines()

        merged_names = [x for x in GMM_lines if "#" not in x and x.strip() != ""]
        merged_names = [x.strip().split()[0] for x in merged_names]

        # HERE: need to test that merged_names  is same len as label* and if not, we need to find the missing labels and add those names to the list of merged_names

        # zip up the merged_names, labelGroups, labelIdx and assign to a class field
        self.gmmGroupings = list(zip(merged_names, labelGroups, labelIdx))
        print(*self.gmmGroupings)

        print(grouping_json)
        # You fool you're operating on this object and adding fields during this lool and it's a class atribure
        # make a copy of the grouping_json before the loop, then update the class atribure at the end of the method, or maybe better return the new json and reassign to the atribure in the methods that call this
        grouping_copy = grouping_json.copy()
        for k, v in grouping_copy.items():
            print(k)
            labelGroups_idx = []
            group_no = []
            for g_key in v["group_keys"]:
                print(f"G_KEY: {g_key}")
                # find idx in labelGroups
                """
                #idx = [i for i,sub_lst in enumerate#(labelGroups) if g_key in sub_lst]

                #labelGroups_idx += idx
                """
                # test if we have a merged name that matches
                print("BEGIN gmmGrouping Loop")
                for i, merged_name in enumerate(self.gmmGroupings):
                    print(f"{merged_name[0]}::{g_key}")
                    if g_key == merged_name[0]:
                        labelGroups_idx += merged_name[2]
                        group_no += [i]
                        break

                    print(f"GNO: {group_no}")
                    print(f"GID: {labelGroups_idx}")

            print(labelGroups_idx)
            # set group_no in dict
            grouping_json[k]["group_no"] = labelGroups_idx
            if grouping_json[k]["group_vals"] is None:
                # add a list to hold the list of labels
                grouping_json[k]["group_vals"] = []
                for group in labelGroups_idx:
                    # concat the lists of labels for all group_idx
                    grouping_json[k]["group_vals"] += [self.labelMapping[group].name]

        print(grouping_json.items())

    def initialize(self):
        """
        Sanity check args, load input volumes, create temp dirs
        """
        # set up the temp dir
        if self.tempDir is None:
            self.tempDir = tempfile.mkdtemp()
        else:
            os.makedirs(self.tempDir, exist_ok=True)

        # to keep things similar to the current python implementation, always
        # create the expected output structure under the outDIR
        self.outDir = os.path.join(
            self.outDir, "results", "EM", (self.dtiLikelihood + self.fileSuffix)
        )
        # create the output file tree if it doesn't already exits
        os.makedirs(self.outDir, exist_ok=True)

        # check for valid likelihood option
        likelihood_options = [
            "logFrob",
            "wishart",
            "DSWbeta",
            "structural",
            "fa",
            "trace",
        ]
        if self.dtiLikelihood not in likelihood_options:
            sf.system.fatal(
                "Invalid dtiLikelihood calculator selected, must be"
                f"one of:{', '.join(likelihood_options)}"
            )

        # set WMMSharedParameters based on dtiLikelihood
        if self.dtiLikelihood in ["DSWbeta", "structural", "fa"]:
            self.WMMSharedParameters = os.path.join(
                self.atlasDir, "sharedWMMparameters_DSWbeta.txt"
            )
        elif self.dtiLikelihood in ["logFrob", "trace"]:
            self.WMMSharedParameters = os.path.join(
                self.atlasDir, "sharedWMMparameters_LogFrob.txt"
            )
        else:
            self.WMMSharedParameters = os.path.join(
                self.atlasDir, "sharedWMMparameters_Wishart.txt"
            )

        # sanity check optimizer type
        # do we need to change this for the DTI, or replace with likelihood?
        optimizerTypes = [
            "FixedStepGradientDescent",
            "GradientDescent",
            "ConjugateGradient",
            "L-BFGS",
        ]
        if self.optimizerType not in optimizerTypes:
            sf.system.fatal(
                "Optimizer type must be one of: " + ", ".join(optimizerTypes)
            )

        # skip the bbreg stuff, not called in the thalamus subregions code

        # check for all the atlas files
        # NOTE: Do we need additional checks for the combinded atlases and other files for DTI?
        if not os.path.isfile(self.atlasMeshFileName):
            sf.system.fatal(
                f"Provided atlas mesh file `{self.atlasMeshFileName}` does not exist."
            )
        if not os.path.isfile(self.atlasDumpFileName):
            sf.system.fatal(
                f"Provided atlas image `{self.atlasDumpFileName}` does not exist."
            )
        if not os.path.isfile(self.compressionLookupTableFileName):
            sf.system.fatal(
                f"Provided compression LUT `{self.compressionLookupTableFileName}` does not exist."
            )

        # set target mesh file paths
        self.warpedMeshFileName = os.path.join(self.tempDir, "warpedOriginalMesh.txt")
        self.warpedMeshNoAffineFileName = os.path.join(
            self.tempDir, "warpedOriginalMeshNoAffine.txt"
        )

        # now that sanity checks have passed, and temp/out dirs are created, start loading things

        # load the cLUT and label info
        self.labelMapping, self.names, self.FreeSurferLabels = (
            utils.read_compression_lookup_table(self.compressionLookupTableFileName)
        )

        # NOTE: are there labels in samseg not in synth or vise versa? so we can pick the lookup tables from that?
        # load the input volumes
        self.inputSeg = sf.load_volume(self.inputSegFileName)
        # NOTE: we really only have one, should we rename? change from list comp?
        self.inputImages = [sf.load_volume(path) for path in self.inputImageFileNames]
        self.correctedImages = [img.copy() for img in self.inputImages]
        # NOTE: do we need this? I don't see it referenced anywhere?
        self.highResImage = np.mean(self.inputImages[0].geom.voxsize) < 0.99

        # parse the dti_dir for the required files
        self.parse_dti_dir()

        # load the DTI FA file
        self.FAImage = sf.load_volume(self.FAImagePath)

        # check if we have a synthseg or aseg inputSeg, synth max val = 60
        # set the sharedGaussianParameters accordingly
        if np.any(self.inputSeg.data > 60):
            self.sameGaussianParameters = os.path.join(
                self.atlasDir, "ASEGsharedGMMparameters.txt"
            )
            self.inputSegType = "aseg"
        else:
            self.sameGaussianParameters = os.path.join(
                self.atlasDir, "SYNTHSEGsharedGMMparameters.txt"
            )
            self.inputSegType = "synthseg"
        # HACK: Maybe rather than taking the path to the structural, segVol, lta
        # we can just take the path to the structural file to segment and derive
        # the path to the segmentation and lta from that

        # call self.preprocess_image
        self.preprocess_image()

        # NEW parse the grouping json
        print("LOADING JSON GROUPINGS")
        self.grouping_dict = json.load(
            open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "for_testing",
                    "means_groupings.json",
                ),
                "r",
            )
        )
        self.parse_grouping_json(
            self.grouping_dict, os.path.join(self.atlasDir, "sharedGMMparameters.txt")
        )

    def preprocess_image(self):
        """
        Preprocess the input seg and images

        Must populate fields:
            - self.atlasAlignmentTarget: masked segmentation, affineImageDumpReg
            - self.synthImage: image used for initial mesh fitting
            - self.processedImage:

        """

        # Define a few hardcoded label constants
        self.THlabelLeft = 10
        self.THlabelRight = 49
        self.DElabelLeft = 28
        self.DElabelRight = 60

        # Atlas alignment target is a masked segmentation
        match_labels = [
            self.THlabelLeft,
            self.THlabelRight,
            self.DElabelLeft,
            self.DElabelRight,
        ]
        mask = np.isin(self.inputSeg.data, match_labels).astype("float32") * 255
        self.atlasAlignmentTarget = self.inputSeg.new(mask)

        # this section is taken directly from the 'thalamus.py' script.
        # do we need to change these values to match those that are in the
        # tract cLUT?

        # the data and self.inputSeg.data will share the same 'id' this is for convenience
        data = self.inputSeg.data

        # There's a bunch of labels in the SEG that we don't have in our atlas
        # So we'll have to get rid of those
        data[data == 5] = 4  # left-inf-lat-vent -> left-lat-vent
        data[data == 44] = 4  # right-inf-lat-vent -> left-lat-vent
        data[data == 14] = 4  # 3rd vent -> left-lat-vent
        data[data == 15] = 4  # 4th vent -> LV (we're killing brainstem anyway)
        data[data == 17] = 3  # left HP -> left cortex
        data[data == 53] = 3  # right HP -> left cortex
        data[data == 18] = 3  # left amygdala -> left cortex
        data[data == 54] = 3  # right amygdala -> left cortex
        data[data == 24] = 4  # CSF -> left-lat-vent
        data[data == 30] = 2  # left-vessel -> left WM
        data[data == 62] = 2  # right-vessel -> left WM
        data[data == 72] = 4  # 5th ventricle -> left-lat-vent
        data[data == 77] = 2  # WM hippoint -> left WM
        data[data == 80] = 0  # non-WM hippo -> background
        data[data == 85] = 0  # optic chiasm -> background
        data[data > 250] = 2  # CC labels -> left WM

        # Next we want to remove hemi-specific lables, so we convert right labels to left
        data[data == 41] = 2  # WM
        data[data == 42] = 3  # CT
        data[data == 43] = 4  # LV
        data[data == 46] = 7  # cerebellum WM
        data[data == 47] = 8  # cerebellum CT
        data[data == 50] = 11  # CA
        data[data == 51] = 12  # PU
        data[data == 52] = 13  # PA
        data[data == 58] = 26  # AA
        data[data == 63] = 31  # CP

        # Remove a few remainders
        removal_mask = np.isin(data, [44, 62, 63, 41, 42, 43, 50, 51, 52, 53, 54, 58])
        data[removal_mask] = 0

        # And convert background to 1
        data[data == 0] = 1

        # Now, create a mask with DE merged into thalamus. This will be the
        # synthetic image used for initial mesh fitting
        segMerged = self.inputSeg.copy()
        segMerged[segMerged == self.DElabelLeft] = self.THlabelLeft
        segMerged[segMerged == self.DElabelRight] = self.THlabelRight
        self.synthImage = segMerged

        # And also used for image cropping around the thalamus
        thalamicMask = (segMerged == self.THlabelLeft) | (
            segMerged == self.THlabelRight
        )
        fixedMargin = int(np.round(15 / np.mean(self.inputSeg.geom.voxsize)))
        imageCropping = segMerged.new(thalamicMask).bbox(margin=fixedMargin)

        # Lastly, use it to make the image mask
        struct = np.ones((3, 3, 3))
        mask = scipy.ndimage.morphology.binary_dilation(
            self.synthImage > 1, structure=struct, iterations=2
        )
        imageMask = self.synthImage.new(mask)

        # Mask and convert to the target resolution
        images = []
        refGeom = None
        for i, image in enumerate(self.inputImages):
            # FS python library does not have cubic interpolation yet, so we'll use mri_convert
            tempFile = os.path.join(self.tempDir, f"tempImage_{i}.mgz")
            image[imageCropping].save(tempFile)
            utils.run(
                f"mri_convert {tempFile} {tempFile} -odt float -rt cubic -vs {self.resolution} {self.resolution} {self.resolution}"
            )
            image = sf.load_volume(tempFile)

            # handle cases where images are not on the same grid
            if refGeom is None:
                refGeom = image.geom
            else:
                image = image.resample_like(refGeom, method='linear')

            # Resample and apply the image mask in high-resolution target space
            imageMask = imageMask.resample_like(image, method="nearest")
            image[imageMask == 0] = 0
            images.append(image.data)
            self.longMask = imageMask

        # Define the pre-processed target image
        self.processedImage = image.new(np.stack(images, axis=-1))

    def get_cheating_label_groups(self):
        """
        The label groups are determined by the sameGaussianParameters file
        That means we need to branch depending on if we have aseg or synthseg
        for our initial segmentation

        Col 1: merged name
        Col 2: number of components
        Col 3: search strings

        TODO: Handle labels in the atlas that aren't in the sameGaussianParameters fileargs['tempDir'] = 'tmp_thal'
        """

        # load the shared GMM params file
        with open(self.sameGaussianParameters, "r") as f:
            labels = f.readlines()

        # process the file
        # Remove header and blank lines, split line on white space, slice
        label_groups = [x.split()[2:] for x in labels if "#" not in x and x != "\n"]

        # get the mergedName list, to build self.grouping_dict
        merged_names = [x.split()[0] for x in labels if "#" not in x and x != "\n"]

        # flatten the list for convenience
        # label_names = sum(label_groups, [])

        ## handle labels in self.FreeSurferLabels not in the shared params
        # get the label indices
        label_group_indices = self.label_group_names_to_indices(label_groups)
        # flatten
        flat_indices = set(sum(label_group_indices, []))
        # fs label set
        fs_labels = set(self.FreeSurferLabels)
        # unhandled_labels indices
        unhandled_labels = fs_labels - flat_indices

        for label in unhandled_labels:
            label_groups.append([self.labelMapping[label].name])

        # iterate over the grouped labels, and build the json
        """
        for i, lg in enumerate(label_groups):
            # test if merged_names[i] in json already, if so can skip
            merged_name = merged_names[i] if i < len(merged_names) else label_groups[i][0]
            print(merged_name)
            if merged_name in self.grouping_dict.keys():
                continue
            # if not - add to dict, w/ 'defaults'
            else:
                

                self.grouping_dict[merged_name] = {
                    "group_keys": lg,
                    "group_vals": None,
                    "post_em_update": False
                    }
            """

        # recursive? call to parse_grouping_json to clean it up
        # self.parse_grouping_json(self.grouping_dict)
        if self.bp:
            breakpoint()
        return label_groups

        # test if we have aseg or synthseg sharred GMM parameters
        if self.inputSegType == "aseg":
            labelGroups = [
                ["Unknown"],
                ["Cerebral-White-Matter", "ThalNuc-R", "Tract"],
                ["Cerebral-Cortex", "Hippocampus", "Amygdala"],
                ["Cerebellum-Cortex"],
                ["Cerebellum-White-Matter"],
                ["Ventricle", "Lat-Vent", "CSF"],
                ["choroid-plexus"],
                ["Putamen"],
                ["Pallidum"],
                ["Accumbens-area"],
                ["Caudate"],
                [
                    "Left-VentralDC",
                    "Left-ThalNuc-L-Sg",
                    "Left-ThalNuc-LGN",
                    "Left-ThalNuc-MGN",
                    "Left-ThalNuc-H",
                    "Left-ThalNuc-VPI",
                    "Left-ThalNuc-MV(Re)",
                    "Left-ThalNuc-Pf",
                    "Left-ThalNuc-CM",
                    "Left-ThalNuc-LP",
                    "Left-ThalNuc-VLa",
                    "Left-ThalNuc-VPL",
                    "Left-ThalNuc-VLp",
                    "Left-ThalNuc-VM",
                    "Left-ThalNuc-CeM",
                    "Left-ThalNuc-Pc",
                    "Left-ThalNuc-MDv",
                    "Left-ThalNuc-Pv",
                    "Left-ThalNuc-CL",
                    "Left-ThalNuc-VA",
                    "Left-ThalNuc-VPM",
                    "Left-ThalNuc-AV",
                    "Left-ThalNuc-VAmc",
                    "Left-ThalNuc-Pt",
                    "Left-ThalNuc-AD",
                    "Left-ThalNuc-LD",
                    "Left-ThalNuc-PuA",
                    "Left-ThalNuc-PuI",
                    "Left-ThalNuc-PuL",
                    "Left-ThalNuc-PuMm",
                    "Left-ThalNuc-PuMl",
                    "Left-ThalNuc-MDl",
                    "Left-ThalNuc-MDm",
                ],
                [
                    "Right-VentralDC",
                    "Right-ThalNuc-L-Sg",
                    "Right-ThalNuc-LGN",
                    "Right-ThalNuc-MGN",
                    "Right-ThalNuc-H",
                    "Right-ThalNuc-VPI",
                    "Right-ThalNuc-MV(Re)",
                    "Right-ThalNuc-Pf",
                    "Right-ThalNuc-CM",
                    "Right-ThalNuc-LP",
                    "Right-ThalNuc-VLa",
                    "Right-ThalNuc-VPL",
                    "Right-ThalNuc-VLp",
                    "Right-ThalNuc-VM",
                    "Right-ThalNuc-CeM",
                    "Right-ThalNuc-Pc",
                    "Right-ThalNuc-MDv",
                    "Right-ThalNuc-Pv",
                    "Right-ThalNuc-CL",
                    "Right-ThalNuc-VA",
                    "Right-ThalNuc-VPM",
                    "Right-ThalNuc-AV",
                    "Right-ThalNuc-VAmc",
                    "Right-ThalNuc-Pt",
                    "Right-ThalNuc-AD",
                    "Right-ThalNuc-LD",
                    "Right-ThalNuc-PuA",
                    "Right-ThalNuc-PuI",
                    "Right-ThalNuc-PuL",
                    "Right-ThalNuc-PuMm",
                    "Right-ThalNuc-PuMl",
                    "Right-ThalNuc-MDl",
                    "Right-ThalNuc-MDm",
                ],
            ]
        elif self.inputSegType == "synthseg":
            return None

    def label_group_names_to_indices(self, labelGroups):
        """
        Needs to return a list of lists, internal list is of type int (label index)

        """
        # loop through the label groups, find all instances of labels that match
        # the name (will have to tune this maybe regex?)
        # Then, add the index of the matching labels to a list
        # return the list of lists of label indices
        combinedLabels = []
        for group in labelGroups:
            # loop through the list of search strings to combine
            combined = []
            for search_str in group:
                # concat a list of all labelGroup search string matches
                combined += self.labelMapping.search(search_str)
            # append list of label indices matching the list of search strings
            # HACK CITY RIGHT HERE - there's one repeated value in the lists, and idk why
            combinedLabels.append(list(set(combined)))
        # sort all the label lists to make assigning the means easier in the next step
        [x.sort() for x in combinedLabels]
        return combinedLabels

    def get_cheating_gaussians(self, sameGaussianParameters):
        """
        Return a tuple (means, variances)
        means are the label index #
        variances are all 0.01, and will be adjusted later
        """
        # create an array of zeros of len(sameGaussianParameters)
        means = np.zeros(len(sameGaussianParameters))
        variances = 0.01 * np.ones(len(sameGaussianParameters))

        for i, _ in enumerate(sameGaussianParameters):
            # only consider the first label index in the list
            label = sameGaussianParameters[i][0]
            # check the label index, and assign the 'mean' class index
            # LEFT THALAIC NUCLEI (8100-8199)
            # min val from our shared params file is 28
            if label == 28 or (label >= 8100 and label < 8200):
                means[i] = self.THlabelLeft
            # RIGHT THALAMIC NUCLEI (8200-8299)
            elif label == 60 or label >= 8200:
                means[i] = self.THlabelRight
            # SET BACKGROUND TO 1 SO GEMS DOESN'T COMPLAIN
            elif label == 0:
                means[i] = 1
            else:
                # Henry says that we want the minimum value here, so sort all the lists before we assign 'label'
                means[i] = label

        return (means, variances)

    def get_label_groups(self):
        """
        return a list of lists, of label names that determine the class reductions
        list for each group, containing label names belonging to that group
        THIS SHOULD READ THIS FILE: sharedGMMparameters.txt
        """
        # read in the sharredFMMparameters_* file
        sharedGMMpath = os.path.join(self.atlasDir, "sharedGMMparameters.txt")
        with open(sharedGMMpath, "r") as f:
            labels = f.readlines()

        # read the file, drop comment and blank lines
        label_groups = [x.split()[2:] for x in labels if "#" not in x and x != "\n"]

        # convert the label names to indices, handle cases with trailing "'"
        combined_label_indices = []
        for group in label_groups:
            combined = []
            for search_str in group:
                if "'" in search_str:
                    # if there is a trailing "'" there might also be left/right not in group name that needs to match
                    # combined+= [self.labelMapping.search(search_str[:-1],exact=True)]
                    matches = self.labelMapping.search(search_str[:-1])
                    combined = [
                        x
                        for x in matches
                        if self.labelMapping[x].name.endswith(search_str[:-1])
                    ]
                else:
                    combined += self.labelMapping.search(search_str)
            combined_label_indices.append(list(set(combined)))

        # handle any additional labels in the atlas not in the sharredWMMparams
        flat_indices = set(sum(combined_label_indices, []))

        fs_indices = set(self.FreeSurferLabels)

        unhandled_labels = fs_indices - flat_indices

        for label in unhandled_labels:
            combined_label_indices.append([label])

        # convert the lists of lists of label indices to label names
        combined_label_names = []
        for group in combined_label_indices:
            names = []
            for label in group:
                names.append(self.labelMapping[label].name)

            combined_label_names.append(names)

        return combined_label_names

    def synthseg_kmeans(self):
        """
        If our input segmentation is a synthseg, then we'll be missing some
        of the labels in the initial segmentation that are present in the mesh.
        We need to add these missing labels into the workingImage
        """
        ### QUESTION: SHOULD THIS OPERATE ON THE FULL IMAGE OR CROPPING?
        im_dims = list(self.workingImageShape)
        num_labels = [len(self.FreeSurferLabels)]
        # place holder for initial priors (image_x, image_y, image_z, len(fslabels)
        init_priors = np.zeros(im_dims + num_labels, dtype="uint16")

        # Loop over the labels, get the alphas
        for i in range(
            len(self.FreeSurferLabels)
        ):  # would it be better if this was origAlphas.shape[-1]?
            # silly alphas stuff
            if i == 0:
                sillyAlphas = np.zeros((len(self.originalAlphas), 2), dtype="float32")
                sillyAlphas[:, 0] = self.originalAlphas[:, 0]
                sillyAlphas[:, 1] = 1 - self.originalAlphas[:, 0]
                self.mesh.alphas = sillyAlphas

                # Get the prior
                prior = self.mesh.rasterize(self.workingImageShape)[..., 0]

                # reset the mesh alphas
                self.mesh.alphas = self.originalAlphas
            else:
                # Get the prior
                prior = self.mesh.rasterize(self.workingImageShape, i)

            # update the prior in the init_priors
            init_priors[:, :, :, i] = prior

        # SHOULD OPTIONALLY SAVE THE PRIORS HERE

        # Get max priors, ie initial class assignments
        init_seg_class = np.argmax(init_priors, axis=3)

        # Make the THDE mask
        init_seg_label = self.FreeSurferLabels[init_seg_class]

        prior_seg_outpath = os.path.join(self.tempDir, "initialSegFromPriors.mgz")

        # make a copy of the working vol so we don't muck it
        tmp_vol = self.workingImage.copy()

        tmp_vol.data = init_seg_label

        print(f"Saving initial segmentation from priors to: {prior_seg_outpath}")
        tmp_vol.save(prior_seg_outpath)

        aseg_boxed_path = os.path.join(self.tempDir, "boxedASEGTHDE.mgz")
        synth_image_path = os.path.join(self.tempDir, "synthImage.mgz")

        utils.run(
            f"mri_convert {synth_image_path} {aseg_boxed_path} -rl {prior_seg_outpath} -rt nearest -odt float"
        )

        # $# we need a version of the input intensities in the same space and
        #   with the same geom as the initialSegFromPriors, this is what the
        #   boxed_aseg is resampled into the space of WAIT, JUST USE self.workingImage

        aseg_boxed = sf.load_volume(aseg_boxed_path)

        # LIKELIHOODS
        init_likelihoods = np.zeros(im_dims + num_labels, dtype="uint16")

        for i in range(len(self.FreeSurferLabels)):
            for j in range(len(self.sameGaussianParameters)):
                if np.any(self.sameGaussianParameters[j] == self.FreeSurferLabels[i]):
                    aseg_idx = j
                    break
            aseg_mean = self.means[aseg_idx]
            init_likelihoods[..., i] = aseg_boxed.data == aseg_mean

        init_posterior = init_likelihoods * init_priors

        # handle the missing voxels (ones with posterior = 0, and prior > 0)
        missing_voxels = np.logical_not(np.any(init_posterior > 0, axis=3)) & np.any(
            init_priors > 0, axis=3
        )
        # missing_voxels = np.repeat(missing_voxels[...,np.newaxis], len(self.FreeSurferLabels), axis=3)

        # init_posterior[missing_voxels] = init_priors[missing_voxels]

        # Take the argmax of the posteriors to get get the segmentation
        init_posterior_seg_class = np.argmax(init_posterior, axis=3)
        # TODO: Is this just for debugging? this should be a the initial priors? uncomment line539?
        init_posterior_seg_class[missing_voxels] = 1500

        ## Handle the choroid plexus if we have a synthseg input
        if self.inputSegType == "synthseg":
            # find the indices of the choroid labels
            choroid_indices = set(
                [x[0] for x in self.labelMapping.items() if "choroid" in x[1].name]
            )
            # find the indices of the ventricle labels
            vent_indices = set(
                [
                    x[0]
                    for x in self.labelMapping.items()
                    if "Ventricle" in x[1].name or "Lat-Vent" in x[1].name
                ]
            )
            # find left and right indices
            left_indices = set(
                [x[0] for x in self.labelMapping.items() if "Left" in x[1].name]
            )
            right_indices = set(
                [x[0] for x in self.labelMapping.items() if "Right" in x[1].name]
            )

            left_choroid_idx = left_indices.intersection(choroid_indices)
            right_choroid_idx = right_indices.intersection(choroid_indices)

            # create masks of vent+choroid and choroid
            # $# THIS SHOULDN'T BE ON THE SYNTHIMAGE, LABElS HAVE BEEN MERGED
            #   USE THE BOXED ASEG
            # vent_choroid_mask = np.isin(self.synthImage.data, [*choroid_indices, *vent_indices])
            # choroid_mask = np.isin(self.synthImage.data, [*choroid_indices])

            # $# Changed to test for the values in the the init_posteriors vol
            vent_choroid_mask = np.isin(
                init_posterior_seg_class, [*choroid_indices, *vent_indices]
            )
            choroid_mask = np.isin(init_posterior_seg_class, [*choroid_indices])

            # vector of intensities found in the vent+choroid mask
            vent_intensities = np.zeros(
                [np.sum(vent_choroid_mask), len(self.inputImages)]
            )
            # TODO: why do we permute in the matlab?
            # vent_intensities[:, 0] = self.inputImages[0].data[vent_choroid_mask]

            # $# Changed to read the intensity data from gems
            vent_intensities[:, 0] = self.workingImage.data[vent_choroid_mask]

            # TODO: we need to loop over all the inputImages here
            # $# THIS IS WRONG FIX IT (I think)
            vect_choroid_mask = choroid_mask[vent_choroid_mask]

            # k-means clustering
            kmeans = KMeans(n_clusters=2, init="k-means++")
            kmeans.fit(vent_intensities)
            # idx is a list of length sum(vent_choroid_mask); num non-0 voxels in the mask
            # the value corresponds to the group that the voxel belongs to
            idx = kmeans.labels_

            part1_mean = np.mean(vent_intensities[idx == 0, :])
            part2_mean = np.mean(vent_intensities[idx == 1, :])

            # THIS IS GIVING ALL NANS
            part1_cov = np.cov(vent_intensities[idx == 0, :])
            part2_cov = np.cov(vent_intensities[idx == 1, :])

        # update the data matrix in the tmp vol, save it
        tmp_vol.data = init_posterior_seg_class
        tmp_vol.save(os.path.join(self.tempDir, "initialSegFromPosteriors.mgz"))

    def get_gaussian_hyps(self, sameGaussianParameters, mesh):
        """
        TAKEN FROM ORIGINAL THALAMUS CLASS
        Return a tuple of (meanHyps, nHyps) for Gaussian parameter estimation.
        """
        # should have already called get_label_groups, so sgp should be the sharedGMM version
        nHyper = np.zeros(len(sameGaussianParameters))
        meanHyper = np.zeros(len(sameGaussianParameters))

        # TODO this needs to be adapted for multi-image cases (with masking)
        DATA = self.inputImages[0]

        # placeholder for the json of gaussian groupings
        # this field is set in initialize - self.grouping_dict
        """
        hypsDict = {
            # Need to tie this to the sharred parameter file
            # Check in the init step that they match, we lose the file name after we read in and group the sarredParametersFile
            "Thal": {               # human readable, not checked
                "group_keys": [],   # groups from gmm
                "group_vals": [],   # labels to use to mask intensity vol
                "post_em_update": False
            }
        }
        """

        for g in range(len(sameGaussianParameters)):
            labels = np.array(sameGaussianParameters[g])
            # HENRY: should we strip L-R here?
            labelNames = [self.labelMapping[idx].name for idx in labels]
            hypsGrouping = None
            post_em_update = None
            # test to see if any of the label names are in the json
            for name in labelNames:
                print(name)
                # grouping list nested in grouping_dict, loop over
                for _, v in self.grouping_dict.items():
                    print(name, v["group_vals"])
                    if name in v["group_vals"]:
                        hypsGrouping = v["group_no"]
                        post_em_update = v["post_em_update"]
                        break

                if post_em_update:
                    print("IMPORTING POST EM UPDATE")
                    post_em_update = utils.import_hyps_hack(post_em_update)
                    # post_em_update(self)
                    break

            if hypsGrouping is None:
                listMask = labels
            elif len(hypsGrouping) > 0:
                listMask = hypsGrouping
            else:
                listMask = labels

            """
            if any(labels > 8225):  # thalamus
                listMask = [10, 49]
            elif any(labels == 28): # VDE
                listMask = [28, 60]
            elif any(labels == 0):  # background
                listMask = [1]
            else:
                listMask = labels
            """
            if len(listMask) > 0:
                MASK = np.zeros(DATA.shape, dtype="bool")
                for l in range(len(listMask)):
                    # Ensure that this uses a modified segmentation
                    MASK = MASK | (self.inputSeg == listMask[l])
                radius = np.round(1 / np.mean(DATA.geom.voxsize))
                MASK = scipy.ndimage.morphology.binary_erosion(
                    MASK, utils.spherical_strel(radius), border_value=1
                )
                total_mask = MASK & (DATA > 0)
                data = DATA[total_mask]
                meanHyper[g] = np.median(data)
                """WE NEED TO  DECIDE HOW TO STORE THE LAMBDAS"""
                # PESUDO CODE:
                # if post_em_update is not None:
                # APPLY THE LAMBDA
                if post_em_update:
                    print("applying lambda")
                    print(post_em_update)
                    if self.bp:
                        breakpoint()
                    M, H = post_em_update(self)
                    # optionally update the meanHyper and nHyper if new values returned
                    if M is not None:
                        meanHyper[g] = M
                    if H is not None:
                        nHyper[g] = H
                    if self.bp:
                        breakpoint()

                # if any(labels == 28):
                # Special case: VDE is kind of bimodal in FreeSurfer
                #    nHyper[g] = 10
                else:
                    nHyper[g] = 10 + len(data) * np.prod(DATA.geom.voxsize) / (
                        self.resolution**3
                    )

        # If any NaN, replace by background
        # ATH: I don't there would ever be NaNs here?
        nans = np.isnan(meanHyper)
        meanHyper[nans] = 55
        nHyper[nans] = 10
        print("get_g_hyps end")
        if self.bp:
            breakpoint()
        return (meanHyper, nHyper)

    def get_second_label_groups(self):
        """
        return a list of lists, of label names that determine the class reductions
        list for each group, containing label names belonging to that group
        THIS SHOULD READ THIS FILE: sharedGMMparameters.txt
        """
        # read in the sharredGMMparameters_* file
        sharedGMMpath = os.path.join(self.atlasDir, "sharedGMMparameters.txt")
        with open(sharedGMMpath, "r") as f:
            labels = f.readlines()

        # read the file, drop comment and blank lines
        label_groups = [x.split()[2:] for x in labels if "#" not in x and x != "\n"]

        # convert the label names to indices, handle cases with trailing "'"
        combined_label_indices = []
        for group in label_groups:
            combined = []
            for search_str in group:
                if "'" in search_str:
                    # if there is a trailing "'" there might also be left/right not in group name that needs to match
                    # combined+= [self.labelMapping.search(search_str[:-1],exact=True)]
                    matches = self.labelMapping.search(search_str[:-1])
                    combined = [
                        x
                        for x in matches
                        if self.labelMapping[x].name.endswith(search_str[:-1])
                    ]
                else:
                    combined += self.labelMapping.search(search_str)
            combined_label_indices.append(list(set(combined)))

        # handle any additional labels in the atlas not in the sharredWMMparams
        flat_indices = set(sum(combined_label_indices, []))

        fs_indices = set(self.FreeSurferLabels)

        unhandled_labels = fs_indices - flat_indices

        for label in unhandled_labels:
            combined_label_indices.append([label])

        # convert the lists of lists of label indices to label names
        combined_label_names = []
        for group in combined_label_indices:
            names = []
            for label in group:
                names.append(self.labelMapping[label].name)

            combined_label_names.append(names)

        return combined_label_names

    def get_second_gaussian_hyps(self, sameGaussianParameters, meanHyper, nHyper):
        """
        Return a tuple of (meanHyps, nHyps) for Gaussian parameter estimation in the second-component
        of the primary image-fitting stage.

        This exists because in T-1 images, the medial thal is lighter than lateral.
        """

        """
        This is where we need to handle the second resolution grouping. We will do the same thing in this function as in get_gaussian_hyps, where we iterate over the list of groups provided by sameGaussianParameters, and test if any of those groups are in the json

        If they are, then we need to apply things accordingly:
            - What things could we possibly specify here:
                - how to adjust hyper parameters
                - new mask to calculate them?
            - Do/can any of these options depend on what was done in the first hyps call? or will everything be specified as a distinct key in the json?

        We don't want to allow arbitrairy code execution so we need to define the functions in here, and get the name and args from the json to call 
        """
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
        """
        for g in range(len(sameGaussianParameters)):
            labels = np.array(sameGaussianParameters[g])
            # HENRY: should we strip L-R here?
            labelNames = [self.labelMapping[idx].name for idx in labels]
            hypsGrouping = None
            post_em_update = None
            # test to see if any of the label names are in the json
            for name in labelNames:
                print(name)
                # grouping list nested in grouping_dict, loop over
                for _, v in self.grouping_dict.items():
                    print(name, v["group_vals"])
                    if name in v["group_vals"]:
                        hypsGrouping = v["group_no"]
                        post_em_update = v["post_em_update"]
                        break

                if post_em_update:
                    print("IMPORTING POST EM UPDATE")
                    post_em_update = utils.import_hyps_hack(post_em_update)
                    print("applying lambda")
                    print(post_em_update)
                    M, H = post_em_update(self)
                    # optionally update the meanHyper and nHyper if new values returned
                    if M is not None:
                        meanHyper[g] = M
                    if H is not None:
                        nHyper[g] = H

        return (meanHyper, nHyper)

    def postprocess_segmentation(self):
        """
        Post-process the segmentation and computed volumes.
        """
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
        """
        # extracted from original code above, just so we can generate the output
        # volumes and visualize them in FV
        A = self.discreteLabels.copy()
        segFilePrefix = os.path.join(self.outDir, f"ThalamicNuclei{self.fileSuffix}")
        A.save(segFilePrefix + ".mgz")
        A.resample_like(self.inputSeg, method="nearest").save(
            segFilePrefix + ".FSvoxelSpace.mgz"
        )

        segFilePrefix = os.path.join(self.outDir, f"ThalamicNuclei{self.fileSuffix}")
        # Write the volumes
        self.write_volumes(segFilePrefix + ".volumes.txt")
