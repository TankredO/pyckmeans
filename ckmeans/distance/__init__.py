''' distance

    Module for distance calculations.
'''

from typing import Iterable, Optional, Tuple

import numpy

import ckmeans.io

from .c_interop import p_distance, jc_distance, k2p_distance

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

    @property
    def shape(self) -> Tuple[int]:
        '''shape

        Get matrix shape.

        Returns
        -------
        Tuple[int]
            Matrix shape.
        '''
        return self.dist_mat.shape

class InvalidDistanceTypeError(Exception):
    '''UnknownDistanceTypeError'''

def alignment_distance(
    alignment: "ckmeans.io.NucleotideAlignment",
    distance_type: str = 'p',
    pairwise_deletion: bool = True,
) -> DistanceMatrix:
    '''genetic_distance

    Calculate genetic distance based on a nucleotide alignment.

    Parameters
    ----------
    alignment : ckmeans.io.NucleotideAlignment
        Nucleotide alignment.
    distance_type : str, optional
        Type of genetic distance to calculate, by default 'p'.
        Available distance types are p-distances ('p'),
        Jukes-Cantor distances ('jc'), and Kimura 2-paramater distances
        ('k2p').
    pairwise_deletion : bool
        Use pairwise deletion as action to deal with missing data.
        If False, complete deletion is applied.
        Gaps ("-", "~", " "), "?", and ambiguous bases are treated as
        missing data.
    Returns
    -------
    DistanceMatrix
        n*n distance matrix.

    Raises
    ------
    InvalidDistanceTypeError
        Raised if invalid distance_type is passed.
    '''
    distance_type = distance_type.lower()
    if distance_type in ['p', 'raw']:
        return DistanceMatrix(
            p_distance(alignment.sequences, pairwise_deletion),
            alignment.names,
        )
    elif distance_type in ['jc', 'jc69']:
        return DistanceMatrix(
            jc_distance(alignment.sequences, pairwise_deletion),
            alignment.names,
        )
    elif distance_type in ['k2p', 'k80']:
        return DistanceMatrix(
            k2p_distance(alignment.sequences, pairwise_deletion),
            alignment.names,
        )
    else:
        msg = f'Unknown distance type "{distance_type}".'
        raise InvalidDistanceTypeError(msg)
