import ctypes
import pathlib

import numpy

# path of the shared library
libfile = pathlib.Path(__file__).parent / 'lib' / 'distance.so'
print(libfile)

lib = ctypes.CDLL(str(libfile))

# test
lib.helloWorld.restype = None
lib.helloWorld.argtypes = None

def hello_world():
    '''hello_world
    '''
    lib.helloWorld()

# p distance
lib.pDistance.restype = None
lib.pDistance.argtypes = [
    numpy.ctypeslib.ndpointer(  # alignment: n * m matrix
        dtype=numpy.uint8,
        ndim=2,
        flags='C_CONTIGUOUS',
    ),
    ctypes.c_int,               # n: number of entries
    ctypes.c_int,               # m: number of sites
    ctypes.c_int,               # gapAction: how to handle gaps
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
    n, m = alignment.shape

    dist_mat = numpy.zeros((n, n), dtype=numpy.double)

    lib.pDistance(alignment, n, m, pairwise_deletion, dist_mat)

    return dist_mat
