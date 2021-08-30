import ctypes
from os import error
import pathlib

import numpy

# load the shared library
libfile = pathlib.Path(__file__).parent / 'lib' / 'nucencode.so'
lib = ctypes.CDLL(str(libfile))

# == nucleotide encoding
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

lib.encodeNucleotides_uint32.restype = None
lib.encodeNucleotides_uint32.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint32,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
    numpy.ctypeslib.ndpointer(  # encodedAlignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
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

    Raises
    ------
    Exception
        Raised if alignment has invalid dtype.
    '''
    if not alignment.flags['C_CONTIGUOUS']:
        alignment = numpy.ascontiguousarray(alignment)

    n, m = alignment.shape

    # ASCII encoding? 1 byte per character
    if alignment.dtype.type == numpy.dtype('S'):
        lib.encodeNucleotides(alignment.view(numpy.uint8), n, m)
        return alignment.view(numpy.uint8)
    # Unicode encoding. Expecting 4 bytes per character
    elif alignment.dtype.type == numpy.dtype('U'):
        alignment_encoded = numpy.zeros_like(alignment, dtype=numpy.uint8)
        lib.encodeNucleotides_uint32(alignment.view(numpy.uint32), n, m, alignment_encoded)
        return alignment_encoded
    else:
        msg = f'Can not encode sequences with dtype {alignment.dtype}.'
        raise Exception(msg)
