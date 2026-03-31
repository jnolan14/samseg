# Setup instructions for testing subregions code

This directory contains the required json config files, as well as a python script that will instantiate the `MeshModel` classes for each of the three cases: standard thalamus subregions, single channel subfields++, multi channel subfields++.

The `MeshModel` pipelines will make some system calls to FreeSurfer tools, i.e. `mri_convert`, if FreeSurfer is not sourced, the pipelines will fail. Anything newer than 7.4.1 should be fine.

Directory Contents:
- `README.md` - you're looking at it
- `dti_args*.json` - these two files contain the args for initializing the `MeshModel` objects, paths for the first 5 fields in each will need to be updated to match your system path
- `means_geoupings.json` - json file holding the information for how to group classes and apply the 'hacks'. Path to this file is hard coded in line 326 of `thalamusDTI.py` and will need to be updated (ultimately needs to be an arg passed to the `MeshModel` class)
- `shell_test.py` - python script containing functionality to instantiate the proper `MeshModel` class for each of the three test cases listed above. In the current implementation, the script can be run as is and will run all 3. Lines can be commented out for testing, or they can be copy/pasted into a python shell to make it easier to interrogate some of the objects

## Setup Steps:
1. Install `samseg` from source, following [Installing from source (on \*nix)](https://github.com/jnolan14/samseg/blob/dti_integration/README.md#installing-from-source-on-nix)
	1. Would highly recommend creating a clean conda env (or a new venv with your manager of choice). Disagreement between the version of python here and `fspython` installed with the version of FS you'll be sourcing won't be an issue here. Only compiled C/C++ executables from FS get called so the `fspython` environment is irrelevant.
2. Update file paths in the `dti_args*.json` files. If you only want to run the multi channel, then you only need to do this in the `dti_args_FA.json`, the other is for the single channel cases. Keys in the json correspond to the `MeshModel.__init__` args
	1. `atlasDir` - path to the directory containing the atlas file(s) you want to use
		1. NOTE: For the standard thalamus case, this arg is popped from the dict of args used to initialize the class and will use the default location for the atlas file in the FS install. If you want to use another atlas, change line 28 of `shell_test.py` to update the field to be the path to that atlas file, rather than just pop that value.
	2. `outDir` - path to output directory
	3. `inputImageFileNames` - input structural images, this should be specified as a list in both the single and multi channel cases
	4. `inputSegFileName` - path to the input segmentation volume (an `aseg.mgz` or `synthseg.mgz`)
	5. `inputDTIDirName` - path to the directory containing the preprocessed DTI files (not needed for the standard thalamus)
3. Update the path to the `means_grouping.json` file in line 326 of the `thalamusDTI.py` file
	1. This is the 'hacks.json', which will be added to the `MeshModel` class args. If you'd like to play around with the 'hacks', they are defined in `samseg/subregions/utils.py`