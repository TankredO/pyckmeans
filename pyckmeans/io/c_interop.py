import ctypes
import pathlib

import numpy

# load the shared library
libfile = pathlib.Path(__file__).parent / 'lib' / 'nucencode.so'
lib = ctypes.CDLL(str(libfile))

# == p distance
lib.encodeNucleotides.restype = None
lib.encodeNucleotides.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
]

def encode_nucleotides(
    alignment: numpy.ndarray,
) -> numpy.ndarray:
    '''encode_nucleotides

    Encode nucleotide alignment INPLACE.

    Parameters
    ----------
    alignment : numpy.ndarray
        n*m numpy alignment, where n is the number of entries and m is
        the number of sites. Dtype must be 'U1' or 'S'.

    Returns
    -------
    numpy.ndarray
        The encoded alignment.
    '''
    if not alignment.flags['C_CONTIGUOUS']:
        alignment = numpy.ascontiguousarray(alignment)

    n, m = alignment.shape

    lib.encodeNucleotides(alignment.view(numpy.uint8), n, m)

    return alignment.view(numpy.uint8)
