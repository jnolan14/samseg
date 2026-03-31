#!/usr/bin/env python3

"""
Python shell calls to run the subfields++

In current format, will run the standard thalamus subregions code, one channel DTI, multi channel DTI.

Lines can be commented out and the script can be run as is, or lines/blocks can be copy/pasted into a python shell for testing.

Steps follow the steps for processing a crossectional subregions run. Taken from process.py
"""

### import required modules
import json
from samseg.subregions import thalamus as thalamus
from samseg.subregions import thalamusDTI as DTI

### load the json file with the MeshModel args
f = '/autofs/space/anubis_001/users/jackson/samsegDTI/port/samseg/samseg/subregions/dti_args.json' # update this path
f = open(f,'r')
args =json.load(f)
f.close()

### init the DTI class
dti = DTI.ThalamicNucleiDTI(**args)

### pop DTI specific args, update temp dir name for standard subregions init
args.pop('atlasDir')
args.pop('inputDTIDirName')
args.pop('dtiLikelihood')
args['tempDir'] = 'tmp_thal'

### init standard thalamus subregions class
thal = thalamus.ThalamicNuclei(**args)

### load the json for the multi channel MeshModel class
f = '/autofs/space/anubis_001/users/jackson/samsegDTI/port/samseg/samseg/subregions/dti_args_FA.json' # NOTE: this file differs from previous json, also includes path to FA image
f = open(f,'r')
args =json.load(f)
f.close()

### init the multi channel MeshModel
multi = DTI.ThalamicNucleiDTI(**args)

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