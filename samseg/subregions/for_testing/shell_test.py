#!/usr/bin/env python3

"""
Python shell calls to run the subfields++

In current format, will run the standard thalamus subregions code, one channel DTI, multi channel DTI.

Lines can be commented out and the script can be run as is, or lines/blocks can be copy/pasted into a python shell for testing.

Steps follow the steps for processing a crossectional subregions run. Taken from process.py
"""

### import required modules
import json
from pathlib import Path
from samseg.subregions import thalamus as thalamus
from samseg.subregions import thalamusDTI as DTI

TESTING_DIR = Path(__file__).resolve().parent

ARG_PROFILES = [
    {
        "name": "Henry",
        "dti": TESTING_DIR / "dti_args_henry.json",
        "multi": TESTING_DIR / "dti_args_FA_henry.json",
    },
    {
        "name": "Jackson",
        "dti": TESTING_DIR / "dti_args.json",
        "multi": TESTING_DIR / "dti_args_FA.json",
    },
]


def load_args(path):
    with open(path, "r") as f:
        return json.load(f)


def required_paths(args):
    paths = {
        "atlasDir": Path(args["atlasDir"]),
        "inputSegFileName": Path(args["inputSegFileName"]),
        "inputDTIDirName": Path(args["inputDTIDirName"]),
        "inputImageFileNames[0]": Path(args["inputImageFileNames"][0]),
    }
    return paths


def select_args_profile():
    missing_by_profile = {}
    for profile in ARG_PROFILES:
        dti_args = load_args(profile["dti"])
        multi_args = load_args(profile["multi"])
        missing = [
            f"{args_name}.{path_name}: {path}"
            for args_name, args in (("dti", dti_args), ("multi", multi_args))
            for path_name, path in required_paths(args).items()
            if not path.exists()
        ]
        if not missing:
            return profile["name"], dti_args, multi_args
        missing_by_profile[profile["name"]] = missing

    lines = ["No valid subregions testing profile found.", "Missing paths:"]
    for profile_name, missing in missing_by_profile.items():
        lines.append(f"- {profile_name}")
        lines.extend(f"  - {path}" for path in missing)
    raise SystemExit("\n".join(lines))


profile_name, args, multi_args = select_args_profile()
print(f"Using subregions testing profile: {profile_name}")

### init the DTI class
dti = DTI.ThalamicNucleiDTI(**args)

### pop DTI specific args, update temp dir name for standard subregions init
args.pop("atlasDir")
args.pop("inputDTIDirName")
args.pop("dtiLikelihood")
args["tempDir"] = str(Path(args["tempDir"]).parent / "tmp_thal")
args["outDir"] = str(Path(args["outDir"]).parent / "thal")

### init standard thalamus subregions class
thal = thalamus.ThalamicNuclei(**args)

### init the multi channel MeshModel
multi = DTI.ThalamicNucleiDTI(**multi_args)

### BEGIN PROCESS.PY CALLS
## initialize
dti.initialize()
thal.initialize()
multi.initialize()

## align atlas to input seg
dti.align_atlas_to_seg()
thal.align_atlas_to_seg()
multi.align_atlas_to_seg()

## prep for seg fitting
dti.prepare_for_seg_fitting()
thal.prepare_for_seg_fitting()
multi.prepare_for_seg_fitting()

## fit mesh to seg
dti.fit_mesh_to_seg()
thal.fit_mesh_to_seg()
multi.fit_mesh_to_seg()

## additional k-means clustering step for DTI
# Not performed on standard thalamus pipeline
dti.synthseg_kmeans()
multi.synthseg_kmeans()

## prepare for image fitting
dti.prepare_for_image_fitting()
thal.prepare_for_image_fitting()
multi.prepare_for_image_fitting()

## fit mesh to image
dti.fit_mesh_to_image()
thal.fit_mesh_to_image()
multi.fit_mesh_to_image()

## extract segmentation
# This will need some work for the DTI classes
dti.extract_segmentation()
thal.extract_segmentation()
multi.extract_segmentation()

## postprocess segmentation
# This will also need a bit of work for the DTI calsses
dti.postprocess_segmentation()
thal.postprocess_segmentation()
multi.postprocess_segmentation()

## cleanup
# not really needed for tests, should just remove temp files
dti.cleanup()
thal.cleanup()
multi.cleanup()
