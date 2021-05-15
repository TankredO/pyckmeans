''' distance

    Module for distance calculations.
'''

import numpy
from scipy.linalg import eigh

from ckmeans.io import NucleotideAlignment


class UnknownDistanceTypeError(Exception):
    '''UnknownDistanceTypeError
    '''

def alignment_distance(
    alignment: NucleotideAlignment,
    distance_type: str = 'p'
) -> numpy.ndarray:
    '''genetic_distance

    Calculate genetic distance based on a nucleotide alignment.

    Parameters
    ----------
    alignment : NucleotideAlignment
        Nucleotide alignment.
    distance_type : str, optional
        Type of genetic distance to calculate, by default 'p'.

    Returns
    -------
    numpy.ndarray
        n*n distance matrix.

    Raises
    ------
    UnknownDistanceTypeError
        Raised if invalid distance_type is passed.
    '''

    if distance_type == 'p':
        return p_distance(alignment.sequences)
    else:
        msg = f'Unknown distance type "{distance_type}".'
        raise UnknownDistanceTypeError(msg)

def p_distance(alignment: numpy.ndarray) -> numpy.ndarray:
    dist_mat = numpy.zeros((alignment.shape[0], alignment.shape[0]))

    for i in range(alignment.shape[0]):
        for j in range(i, alignment.shape[0]):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                p_dist = numpy.sum(alignment[i,] != alignment[j,]) / alignment.shape[1]
                dist_mat[i, j] = p_dist
                dist_mat[j, i] = p_dist

    return dist_mat
