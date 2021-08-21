''' distance

    Module for distance calculations.
'''

from typing import Iterable, Optional, Tuple

import numpy

import pyckmeans.io

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

            self.names = numpy.array(names)

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

    @staticmethod
    def from_phylip(file_path: str) -> 'DistanceMatrix':
        '''from_phylip

        Read PHYLIP distance matrix.

        Returns
        -------
        DistanceMatrix
            DistanceMatrix object.
        '''
        return pyckmeans.io.phylip.read_phylip_distmat(file_path)

    @staticmethod
    def from_csv( # pylint: disable=missing-param-doc
        file_path: str,
        header: Optional[int] = 0,
        index_col: Optional[int] = 0,
        sep: str = ',',
        **kwargs,
    ) -> 'DistanceMatrix':
        '''read_csv_distmat

        Read distance matrix from CSV file.

        Parameters
        ----------
        file_path : str
            Path to CSV file.
        header : Optional[int]
            Determines the row in the CSV file containing
            sample names. Is passed to pandas.read_csv(). By default 0, meaning
            the first row.
        index_col : Optional[int]
            Determines the index column. By default, the first column is expected
            to contain sample names. Passed to pandas.read_csv().
        sep : str
            Column separator, be default ','. Passed to Passed to pandas.read_csv().
        **kwargs
            Additional keyword arguments passed to pandas.read_csv().
        Returns
        -------
        pyckmeans.distance.DistanceMatrix
            DistanceMatrix object.
        '''
        return pyckmeans.io.csv.read_csv_distmat(
            file_path=file_path,
            header=header,
            index_col=index_col,
            sep=sep,
            **kwargs,
        )

    def to_phylip(
        self,
        file_path: str,
        force: bool = False,
    ):
        '''to_phylip

        Write distance matrix to file in PHYLIP matrix format.

        Parameters
        ----------
        file_path : str
            Output file path.
        force : bool, optional
            Force overwrite if file exists, by default False
        '''
        pyckmeans.io.phylip.write_phylip_distmat(
            dist=self,
            file_path=file_path,
            force=force,
        )

    def to_csv(
        self,
        file_path: str,
        force: bool = False,
    ):
        '''to_csv

        Write DistanceMatrix object to CSV.

        Parameters
        ----------
        file_path : str
            CSV file path.
        force : bool, optional
            Force overwrite if file_path already exists, by default False
        '''
        pyckmeans.io.csv.write_csv_distmat(
            dist=self,
            file_path=file_path,
            force=force,
        )

class InvalidDistanceTypeError(Exception):
    '''UnknownDistanceTypeError'''

def alignment_distance(
    alignment: "pyckmeans.io.NucleotideAlignment",
    distance_type: str = 'p',
    pairwise_deletion: bool = True,
) -> DistanceMatrix:
    '''genetic_distance

    Calculate genetic distance based on a nucleotide alignment.

    Parameters
    ----------
    alignment : pyckmeans.io.NucleotideAlignment
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
