import ctypes
import pathlib

import numpy

# load the shared library
libfile = pathlib.Path(__file__).parent / 'lib' / 'distance.so'
lib = ctypes.CDLL(str(libfile))

# == p distance
lib.pDistance.restype = None
lib.pDistance.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
    ctypes.c_int,               # pairwiseDeletion
    numpy.ctypeslib.ndpointer(  # (output) distMat: n * n distance matrixmatrix
        dtype=numpy.double,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
]

def p_distance(
    alignment: numpy.ndarray,
    pairwise_deletion: bool = True,
) -> numpy.ndarray:
    '''p_distance

    Calculate p-distance for a nucleotide alignment.

    Parameters
    ----------
    alignment : numpy.ndarray
        n*m numpy alignment, where n is the number of entries and m is
        the number of sites. Bases must be encoded in the format of
        pyckmeans.io.NucleotideAlignment.
    pairwise_deletion : bool, optional
        Calculate distances with pairwise-deletion in case of missing
        data, by default True

    Returns
    -------
    numpy.ndarray
        n*n distance matrix.
    '''
    if not alignment.flags['C_CONTIGUOUS']:
        alignment = numpy.ascontiguousarray(alignment)

    n, m = alignment.shape

    dist_mat = numpy.zeros((n, n), dtype=numpy.double)

    lib.pDistance(alignment, n, m, pairwise_deletion, dist_mat)

    return dist_mat

# == Jukes-Cantor distance
lib.jcDistance.restype = None
lib.jcDistance.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
    ctypes.c_int,               # pairwiseDeletion
    numpy.ctypeslib.ndpointer(  # (output) distMat: n * n distance matrixmatrix
        dtype=numpy.double,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
]

def jc_distance(
    alignment: numpy.ndarray,
    pairwise_deletion: bool = True,
) -> numpy.ndarray:
    '''jc_distance

    Calculate Jukes-Cantor distance for a nucleotide alignment.

    Parameters
    ----------
    alignment : numpy.ndarray
        n*m numpy alignment, where n is the number of entries and m is
        the number of sites. Bases must be encoded in the format of
        pyckmeans.io.NucleotideAlignment.
    pairwise_deletion : bool, optional
        Calculate distances with pairwise-deletion in case of missing
        data, by default True

    Returns
    -------
    numpy.ndarray
        n*n distance matrix.
    '''
    if not alignment.flags['C_CONTIGUOUS']:
        alignment = numpy.ascontiguousarray(alignment)

    n, m = alignment.shape

    dist_mat = numpy.zeros((n, n), dtype=numpy.double)

    lib.jcDistance(alignment, n, m, pairwise_deletion, dist_mat)

    return dist_mat

# == Kimura 2-parameter distance
lib.k2pDistance.restype = None
lib.k2pDistance.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
    ctypes.c_int,               # pairwiseDeletion
    numpy.ctypeslib.ndpointer(  # (output) distMat: n * n distance matrixmatrix
        dtype=numpy.double,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
]

def k2p_distance(
    alignment: numpy.ndarray,
    pairwise_deletion: bool = True,
) -> numpy.ndarray:
    '''jc_distance

    Calculate Kimura 2-parameter distance for a nucleotide alignment.

    Parameters
    ----------
    alignment : numpy.ndarray
        n*m numpy alignment, where n is the number of entries and m is
        the number of sites. Bases must be encoded in the format of
        pyckmeans.io.NucleotideAlignment.
    pairwise_deletion : bool, optional
        Calculate distances with pairwise-deletion in case of missing
        data, by default True

    Returns
    -------
    numpy.ndarray
        n*n distance matrix.
    '''
    if not alignment.flags['C_CONTIGUOUS']:
        alignment = numpy.ascontiguousarray(alignment)

    n, m = alignment.shape

    dist_mat = numpy.zeros((n, n), dtype=numpy.double)

    lib.k2pDistance(alignment, n, m, pairwise_deletion, dist_mat)

    return dist_mat
