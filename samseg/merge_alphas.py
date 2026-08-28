import logging
import math
import numpy as np
import colorsys

from samseg.io import GMMparameter


def meshValidityTest(alphas, name):
    probability_discrepancy = np.max(np.abs(np.sum(alphas, axis=1) - 1))
    if probability_discrepancy > 1e-5:
        message = '%s invalid: class probabilities in the mesh should sum to one in all nodes' % name
        raise ValueError(message)


def _matches_literal_substring(name, searchString):
    return searchString in name


def _matches_resolved_shared_parameter(name, searchString):
    if searchString.endswith("'"):
        return name.endswith(searchString[:-1])
    return _matches_literal_substring(name, searchString)


def _match_shared_gmm_parameters(names, mergeOptions, matchesSearchString):
    """Return Boolean row-to-structure matches using one named contract."""
    memberships = np.zeros((len(mergeOptions), len(names)), dtype=bool)
    for classNumber, mergeOption in enumerate(mergeOptions):
        for searchString in mergeOption.searchStrings:
            matches = [
                matchesSearchString(name, searchString)
                for name in names
            ]
            memberships[classNumber, :] |= matches
    return memberships


def kvlResolveSharedGMMParameters(names, sharedGMMParameters):
    """Resolve sparse shared parameters against one compression LUT.

    This additive resolver implements the mature subregions shared-parameter
    convention beside SAMSEG's existing alpha/group machinery so both can
    reuse the LUT iteration without redefining
    ``kvlGetMergingFractionsTable()``. It can move behind the subregions model
    boundary if maintainers prefer without changing established SAMSEG
    behavior.

    Empty configured rows are removed, and uncovered LUT structures become
    exact-name one-component singleton rows. Cross-row overlap is preserved in
    the returned Boolean membership matrix for the consuming model to
    interpret or reject.
    """
    memberships = _match_shared_gmm_parameters(
        names,
        sharedGMMParameters,
        _matches_resolved_shared_parameter)
    nonemptyRows = np.any(memberships, axis=1)
    resolvedParameters = [
        parameter
        for parameter, nonempty in zip(sharedGMMParameters, nonemptyRows)
        if nonempty
    ]
    memberships = memberships[nonemptyRows, :]

    uncoveredStructures = np.flatnonzero(~np.any(memberships, axis=0))
    if len(uncoveredStructures):
        singletonMemberships = np.zeros(
            (len(uncoveredStructures), len(names)), dtype=bool)
        # The membership added here is exact for this compression lookup table
        # entry. The generated parameter row keeps the structure name as a
        # search string, which ordinary shared-parameter matching treats as a
        # substring. Do not re-resolve generated singleton rows expecting
        # exact-name matching.
        for singletonNumber, structureNumber in enumerate(
                uncoveredStructures):
            name = names[structureNumber]
            resolvedParameters.append(GMMparameter(name, 1, [name]))
            singletonMemberships[singletonNumber, structureNumber] = True
        memberships = np.vstack((memberships, singletonMemberships))

    return resolvedParameters, memberships


def kvlGetMergingFractionsTable( names, mergeOptions ):
    '''Computes a numerOfClasses x numberOfStructures matrix where each column indicates the fractions 
    of the various classes (super-structures) in the corresponding structure. So each column sums to 1.'''

    #
    numberOfClasses = len( mergeOptions )

    fractionsTable = _match_shared_gmm_parameters(
        names,
        mergeOptions,
        _matches_literal_substring).astype(float)
    mergedNames = [];
    for classNumber, mergeOption in enumerate( mergeOptions ):
        mergedNames.append( mergeOption.mergedName.strip() )

    if not fractionsTable.any( axis=0 ).all():
        raise ValueError( 'some structures are not associated with any super-structures' )

    fractionsTable = fractionsTable / np.sum(fractionsTable, 0)

    # Print out merge info
    for classNumber in range( numberOfClasses ):
        print( mergedNames[ classNumber ] )
        for structureNumber in range( len( names ) ):
            percentage = int( fractionsTable[ classNumber, structureNumber ] * 100 )
            if percentage > 0:
                print( '    %s (%d%%)' % ( names[ structureNumber ].ljust( len( max( names, key=len ) ) ), percentage ) )

    return fractionsTable, mergedNames


def kvlMergeAlphas( alphas, fractionsTable ):
    '''Creates a 'mergedAlphas' matrix where one or more columns of the 'alphas'
    matrix have been "merged" (i.e., added together).'''

    # Make sure we're dealing with a valid mesh
    meshValidityTest( alphas, 'alphas' )

    # Do the actual merging
    mergedAlphas = np.dot( alphas, fractionsTable.T )

    # Make sure we're dealing with a valid mesh
    meshValidityTest( mergedAlphas, 'mergedAlphas' )

    return mergedAlphas
