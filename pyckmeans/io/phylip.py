''' fasta

    Module for reading and writing PHYLIP files.
'''

import os
import re
from typing import Tuple, Union

import numpy

import pyckmeans.distance

WHITESPACE_RE = re.compile(r'\s+')

class InvalidPhylipAlignmentError(Exception):
    '''InvalidPhylipAlignmentError
    '''

def read_phylip_alignment(
    phylip_file: str,
    dtype: Union[str, numpy.dtype] = 'U',
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''read_phylip_alignment

    Read phylip alignment file. This function expects the phylip to be a valid alignment,
    meaning that it should contain at least 2 sequences of the same length, including
    gaps.

    WARNING: whitespace characters in entry names are NOT supported.

    Parameters
    ----------
    phylip_file : str
        Path to a phylip file.
    dtype: Union[str, numpy.dtype]
        Data type to use for the sequence array.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Tuple of sequences and names, each as numpy array.

    Raises
    ------
    InvalidPhylipAlignmentError
        Raised if header is malformed.
    InvalidPhylipAlignmentError
        Raised if less than 2 entries are present in phylip_file.
    InvalidPhylipAlignmentError
        Raised if number of entries does not match header.
    '''

    names = []
    seqs = []
    with open(phylip_file) as phylip_f:
        # header
        header_str = next(phylip_f)
        try:
            n_entries, n_sites = [int(s) for s in header_str.split()]
        except:
            raise InvalidPhylipAlignmentError('Malformed header.')

        for line in phylip_f:
            _line = re.sub(WHITESPACE_RE, '', line)
            if not _line:
                continue
            l_len = len(_line)
            start = l_len-n_sites
            name = _line[:start]
            seq = _line[start:].upper()

            names.append(name)
            seqs.append(list(seq))

    # check alignment validity
    n_seq = len(seqs)
    if len(seqs) < 2:
        msg = f'Expected at least 2 entries but found only {n_seq}.'
        raise InvalidPhylipAlignmentError(msg)

    if n_seq != n_entries:
        msg = f'Expected {n_entries} entries but found {n_seq} instead.'
        raise InvalidPhylipAlignmentError(msg)

    # construct output
    seqs = numpy.array(seqs, dtype=dtype)
    names = numpy.array(names)

    return seqs, names


class InvalidPhylipMatrixError(Exception):
    '''InvalidPhylipMatrixTypeError
    '''

def read_phylip_distmat(phylip_file: str) -> 'pyckmeans.distance.DistanceMatrix':
    '''read_phylip_distmat

    Read distance matrix in PHYLIP format.
    Supports full and lower-triangle matrices.

    Parameters
    ----------
    phylip_file : str
        Path to distance file in phylip format.

    Returns
    -------
    pyckmeans.distance.DistanceMatrix
        Distance matrix as pyckmeans.distance DistanceMatrix object.

    Raises
    ------
    InvalidPhylipMatrixError
        Raised if the header is malformed.
    InvalidPhylipMatrixError
        Raised if an empty line is encountered as second line.
    InvalidPhylipMatrixError
        Raised if file format can neither be inferred as full nor
        as lower-triangle matrix.
    InvalidPhylipMatrixError
        Raised if an empty line is encountered.
    InvalidPhylipMatrixError
        Raised if expecting a full matrix but number of values
        does not match the header.
    InvalidPhylipMatrixError
        Raised if an empty line is encountered.
    InvalidPhylipMatrixError
        Raised if expecting lower-triangle matrix but number of values
        does not match the expected number of values for that entry.
    InvalidPhylipMatrixError
        Raised if number of names does not match number of entries
        stated in the header.
    '''
    with open(phylip_file) as phylip_f:
        # == header
        header_str = next(phylip_f)
        try:
            n_entries = int(header_str.strip())
        except:
            raise InvalidPhylipMatrixError('Malformed header.')

        dist_mat = numpy.zeros((n_entries, n_entries))
        names = []

        # == detect matrix type (full, lower-triangle)
        line = next(phylip_f)
        _line = line.strip()
        if not _line:
            msg = 'Line 2: Empty lines are not allowed.'
            raise InvalidPhylipMatrixError(msg)
        name, *mat_entries = _line.split()
        names.append(name)

        # lower-triangle matrix
        if len(mat_entries) == 0:
            mat_type = 'lower-triangle'
        # full matrix
        elif len(mat_entries) == n_entries:
            mat_type = 'full'
            dist_mat[0,] = numpy.array(mat_entries, dtype=float)
        # error
        else:
            msg = 'Line 2: Expected either 0 values for a lower-triangle ' +\
                f'matrix or {n_entries} values for a full matrix; found ' +\
                f'{len(mat_entries)} values instead.'
            raise InvalidPhylipMatrixError(msg)

        # == full matrix
        if mat_type == 'full':
            for i, line in enumerate(phylip_f):
                l_num = i + 3 # 1-based line number: header + first line already read

                _line = line.strip()
                if not _line:
                    # last line can be empty
                    if i + 2 == n_entries:
                        continue
                    msg = f'Line {l_num}: Empty lines are not allowed.'
                    raise InvalidPhylipMatrixError(msg)
                name, *mat_entries = _line.split()
                names.append(name)

                # error
                if len(mat_entries) != n_entries:
                    msg = f'Line {l_num}: Expected {n_entries} values for a full matrix but ' +\
                        f'found {len(mat_entries)} values instead.'
                    raise InvalidPhylipMatrixError(msg)

                dist_mat[i+1,] = numpy.array(mat_entries, dtype=float)

        # == lower-triangle matrix
        elif mat_type == 'lower-triangle':
            for i, line in enumerate(phylip_f):
                l_num = i + 3 # 1-based line number: header + first line already read

                _line = line.strip()
                if not _line:
                    # last line can be empty
                    if i + 2 == n_entries:
                        continue
                    msg = f'Line {l_num}: Empty lines are not allowed.'
                    raise InvalidPhylipMatrixError(msg)
                name, *mat_entries = _line.split()
                names.append(name)

                # error
                if len(mat_entries) != i+1:
                    msg = f'Line {l_num}: Expected {i+1} values for a lower-triangle ' +\
                        f'matrix but found {len(mat_entries)} values instead.'
                    raise InvalidPhylipMatrixError(msg)

                dist_mat[i+1, :i+1] = numpy.array(mat_entries, dtype=float)

            # fill upper triangle
            dist_mat = dist_mat + dist_mat.T

    # check validity
    if len(names) != n_entries:
        msg = f'Expected {n_entries} entries but found {len(names)}.'
        raise InvalidPhylipMatrixError(msg)

    return pyckmeans.distance.DistanceMatrix(dist_mat, names)

class IncompatibleNamesError(Exception):
    '''IncompatibleNamesError'''

NAME_PADDING = 64

def write_phylip_distmat(
    dist: 'pyckmeans.distance.DistanceMatrix',
    file_path: str,
    force: bool = False,
) -> None:
    '''write_phylip_distmat

    Write distance matrix to file in PHYLIP matrix format.

    Parameters
    ----------
    dist : pyckmeans.distance.DistanceMatrix
        Distance matrix as pyckmeans.distance DistanceMatrix object.
    file_path : str
        Output file path.
    force : bool, optional
        Force overwrite if file exists, by default False

    Raises
    ------
    FileExistsError
        Raised if file at file_path already exists and force is False.
    FileExistsError
        Raised if file_path points to an existing directory.
    IncompatibleNamesError
        Raised if names are incompatible with dist_mat.
    '''
    if os.path.exists(file_path):
        if os.path.isfile(file_path) and not force:
            msg = f'File {file_path} already exists. If you want to overwrite ' +\
                'it run the function with force=True.'
            raise FileExistsError(msg)
        else:
            msg = f'A directory exists at path {file_path}.'
            raise FileExistsError(msg)

    dist_mat = dist.dist_mat
    names = dist.names

    n_entries = dist_mat.shape[0]
    if len(names) != n_entries:
        msg = f'Expected {n_entries} names but got {len(names)} instead.'
        raise IncompatibleNamesError(msg)

    with open(file_path, 'w') as phylip_f:
        # header
        phylip_f.write(f'{n_entries}\n')

        # body
        for name, dists in zip(names, dist_mat):
            nm_str = f'{name: <{NAME_PADDING}}'
            dst_str = '\t'.join(dists.astype(str))
            phylip_f.write(f'{nm_str} {dst_str}\n')
