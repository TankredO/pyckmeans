''' csv

    Comma Separated Value (CSV) input and output.
'''
import os
from typing import Optional

import pandas

import pyckmeans.distance

class InvalidMatrixShapeError(Exception):
    '''InvalidMatrixShapeError'''

class IncompatibleNamesError(Exception):
    '''IncompatibleNamesError'''

def read_csv_distmat( # pylint: disable=missing-param-doc
    file_path: str,
    header: Optional[int] = 0,
    index_col: Optional[int] = 0,
    sep: str = ',',
    **kwargs,
) -> 'pyckmeans.distance.DistanceMatrix':
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

    Raises
    ------
    InvalidMatrixShapeError
        Raised if matrix is not square.
    IncompatibleNamesError
        Raised if column and row names do not match.
    '''
    dist_df = pandas.read_csv(
        file_path,
        header=header,
        index_col=index_col,
        sep=sep,
        **kwargs
    )

    dist_mat = dist_df.values

    # distance matrix must be a square matrix
    if dist_mat.shape[0] != dist_mat.shape[1]:
        msg = 'Expected a square matrix but matrix has dimensions '+ \
            f'{dist_mat.shape[0]}x{dist_mat.shape[1]}.'
        raise InvalidMatrixShapeError(msg)

    names = None
    # names are present in file
    if (not header is None) or (not index_col is None):
        # row and column names are present
        if (not header is None) and (not index_col is None):
            names_a = [nm.strip() for nm in dist_df.index.astype(str)]
            names_b = [nm.strip() for nm in dist_df.columns.astype(str)]

            # if row names and column names do not match, something
            # is probably wrong
            if not all([a == b for a, b in zip(names_a, names_b)]):
                raise IncompatibleNamesError('Column and row names do not match.')

            names = names_a
        # column names are present
        elif not header is None:
            names = [nm.strip() for nm in dist_df.columns.astype(str)]
        # row names are present
        elif not index_col is None:
            names = [nm.strip() for nm in dist_df.index.astype(str)]


    return pyckmeans.distance.DistanceMatrix(
        dist_mat,
        names,
    )

def write_csv_distmat(
    dist: 'pyckmeans.distance.DistanceMatrix',
    file_path: str,
    force: bool = False,
) -> None:
    '''write_csv_distmat

    Write DistanceMatrix object to CSV.

    Parameters
    ----------
    dist : pyckmeans.distance.DistanceMatrix
        DistanceMatrix object.
    file_path : str
        CSV file path.
    force : bool, optional
        Force overwrite if file_path already exists, by default False

    Raises
    ------
    FileExistsError
        Raised if file at file_path already exists and force is False.
    FileExistsError
        Raised if file_path points to an existing directory.
    '''
    if os.path.exists(file_path):
        if os.path.isfile(file_path) and not force:
            msg = f'File {file_path} already exists. If you want to overwrite ' +\
                'it run the function with force=True.'
            raise FileExistsError(msg)
        else:
            msg = f'A directory exists at path {file_path}.'
            raise FileExistsError(msg)

    dist_df = pandas.DataFrame(
        dist.dist_mat,
        columns=dist.names,
        index=dist.names,
    )

    dist_df.to_csv(file_path, index_label='sample')
