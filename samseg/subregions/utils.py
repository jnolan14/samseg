import os
import importlib
import scipy.ndimage
import numpy as np
import surfa as sf
from samseg.io import kvlReadCompressionLookupTable


def run(cmd):
    """
    Run a command in a bash shell. Output is silenced, but is printed if an error occurs.
    """
    print(f'Running command: {cmd}')
    output, ret = sf.system.collect_output(cmd)
    if ret != 0:
        print(output)
        sf.system.fatal('Command failed')


def spherical_strel(radius, pixsize=1.0):
    """
    Compute a 3D spherical binary structure for mask manipulation.
    """
    pixsize = np.array([pixsize] * 3)
    shape = np.ceil(2 * radius / pixsize + 1).astype(int)
    shape += np.mod(shape + 1, 2)
    center = (shape - 1) / 2
    coords = np.array(np.ogrid[:shape[0], :shape[1], :shape[2]], dtype=object)
    return np.sum((coords - center) ** 2, axis=0) <= (radius ** 2)


def read_compression_lookup_table(filename):
    """
    Read a compressed label lookup table file into an ordered dictionary
    to labels and names. This also returns corresponding label indices and names
    in a tuple, although we can probably re-extract this info from the labelMapping
    object down the road.
    """
    labelMapping = sf.LabelLookup()
    labels, names, colors = kvlReadCompressionLookupTable(filename)
    labels = np.array(labels)
    for label, name, color in zip(labels, names, colors):
        labelMapping[label] = (name, color)
    return (labelMapping, names, labels)


def get_largest_cc(mask):
    """
    Find the largest connected component of a binary mask. All over components are
    masked away in the returned array.
    ATH TODO: This should be implemented as a function of the Volume object.
    """
    labels = scipy.ndimage.label(mask)[0]
    return labels == np.argmax(np.bincount(labels.flatten())[1:]) + 1


def geometries_differ(a, b):
    """
    Compare the similarity of two volume geometries.
    """
    if not np.array_equal(a.shape[:3], b.shape[:3]):
        return True
    if not np.max(np.abs(a.voxsize - b.voxsize)) > 1e-5:
        return True
    if not np.max(np.abs(a.matrix - b.matrix)) > 1e-5:
        return True
    return False

def find_hyps_idx(label_name, label_groupings):
    """
    Return the index of a label in the list of hyps
    label_name = name of label in the gmm file
    label_grouping = DTI.gmmGroupings (list of tuples of groups)
    """
    for idx, group in enumerate(label_groupings):
        if group[0] == label_name:
            return idx

def import_hyps_hack(hack):
    """
    Function to allow the user to specify a custom 'hack' to apply to the hyperparameters.

    Specify the function as a string like: 'samseg.subregions.utils.<func>'

    All custom functions must accept a MeshModel class as an argument, and all 
    functions must return meanHypr, nHypr
    If either are None, the default value will be used
    """
    module, _, func = hack.rpartition('.')
    module = importlib.import_module(module)

    assert hasattr(module, func), f'The hyps hack "{func}" is not defined in "{module}"'

    return getattr(module, func)

def test_hack(val):
    print(val.grouping_dict)
    return None, None
    #exit()

def vdc_hack(val):
    """
    Set the nHyper to 10 for the VDC
    """
    return None, 10

def bimodal_thal_hack(val):
    """
    Adjust the hyps for medial and lateral thal
    Eugenio's hack
    """
    
    lat_thal_idx = find_hyps_idx('LateralThal')
    med_thal_idx = find_hyps_idx('MedialThal')

    # cache the original meanHyper from thalamus
    thal_mean = val.meanHyper[lat_thal_idx]

    # set lateral new val
    val.nHyper[lat_thal_idx] = 25
    val.meanHyper[lat_thal_idx] = thal_mean + 5
    # test if med_thal in hyps
    if med_thal_idx is not None:
        val.nHyper[med_thal_idx] = 25
        val.meanHyper[med_thal_idx] = thal_mean -5
    # if not already there, add it in
    else:
        val.nHyper = np.append(val.nHyper, 25)
        val.meanHyper = np.append(val.meanHyper, thal_mean - 5)

    return None, None
"""
                # Lateral, brighter
            nHyper[-1] = 25
            meanHyper[-1] = ThInt + 5
            # Medial, darker
            nHyper = np.append(nHyper, 25)
            meanHyper = np.append(meanHyper, ThInt - 5)
"""
