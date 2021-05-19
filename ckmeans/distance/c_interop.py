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


# globals
GAP_PAIRWISE_DELETION = ctypes.c_int.in_dll(lib, 'GAP_PAIRWISE_DELETION')
GAP_COMPLETE_DELETION = ctypes.c_int.in_dll(lib, 'GAP_COMPLETE_DELETION')

AMBIGUITY_STRICT = ctypes.c_int.in_dll(lib, 'AMBIGUITY_STRICT')
AMBIGUITY_RELAXED = ctypes.c_int.in_dll(lib, 'AMBIGUITY_STRICT')

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

class InvalidGapActionError(Exception):
    '''InvalidGapActionError'''

def p_distance(
    alignment: numpy.ndarray,
    gap_action: str = 'pairwise-deletion',
) -> numpy.ndarray:
    n, m = alignment.shape

    dist_mat = numpy.zeros((n, n), dtype=numpy.double)

    if gap_action in ['pairwise', 'pairwise-deletion']:
        gap_action = GAP_PAIRWISE_DELETION
    elif gap_action in ['complete', 'complete-deletion']:
        gap_action = GAP_COMPLETE_DELETION
    else:
        msg = f'Unknown gap action "{gap_action}". Available gap actions are ' +\
            '"pairwise-deletion" and "complete-deletion".'
        raise InvalidGapActionError(msg)

    lib.pDistance(alignment, n, m, gap_action, dist_mat)

    return dist_mat
