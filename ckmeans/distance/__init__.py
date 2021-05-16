''' distance

    Module for distance calculations.
'''

from typing import Iterable, Optional

import numpy

from ckmeans.io import NucleotideAlignment

class IncompatibleNamesError(Exception):
    '''IncompatibleNamesError'''

class DistanceMatrix:
    '''__init__

    Distance Matrix, optionally named.

    Parameters
    ----------
    dist_mat : numpy.ndarray
        n*n distance matrix.
    names : Optional[Iterable[str]]
        Names, by default None.

    Raises
    ------
    IncompatibleNamesError
        Raised if dimension of names and dist_mat are incompatible.
    '''
    def __init__(self, dist_mat: numpy.ndarray, names: Optional[Iterable[str]] = None):
        self.dist_mat = dist_mat
        self.names = None

        if not names is None:
            n = dist_mat.shape[0]
            if len(names) != n:
                msg = f'Expected {n} names for {n}x{n} distance matrix ' +\
                    f'but {len(names)} were passed.'
                raise IncompatibleNamesError(msg)

            self.names = list(names)

    def __repr__(self) -> str:
        '''__repr__

        Returns
        -------
        str
            String representation.
        '''
        return f'{repr(self.names)}\n{repr(self.dist_mat)}'

class UnknownDistanceTypeError(Exception):
    '''UnknownDistanceTypeError'''

def alignment_distance(
    alignment: NucleotideAlignment,
    distance_type: str = 'p'
) -> DistanceMatrix:
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
    DistanceMatrix
        n*n distance matrix.

    Raises
    ------
    UnknownDistanceTypeError
        Raised if invalid distance_type is passed.
    '''

    if distance_type == 'p':
        return DistanceMatrix(
            p_distance(alignment.sequences),
            alignment.names,
        )
    else:
        msg = f'Unknown distance type "{distance_type}".'
        raise UnknownDistanceTypeError(msg)

def p_distance(alignment: numpy.ndarray) -> numpy.ndarray:
    # TODO
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
