from numpy.testing._private.utils import assert_raises
import pytest

import numpy as np
from pyckmeans.distance import DistanceMatrix
from pyckmeans.pcoa import pcoa

@pytest.fixture()
def prepare_distmats():
    nm_0 = ['a', 'b', 'c', 'd']
    d_0 = np.array([
        [0.00, 0.90, 0.80, 0.30],
        [0.90, 0.00, 0.40, 0.70],
        [0.80, 0.40, 0.00, 0.50],
        [0.30, 0.70, 0.50, 0.00],
    ])

    return (
        (d_0, nm_0),
        (DistanceMatrix(d_0, nm_0), nm_0),
    )

def test_pcoa_simple(prepare_distmats):
    pcoares_0 = pcoa(prepare_distmats[0][0])
    assert pcoares_0.names is None
    assert pcoares_0.vectors.shape[0] == prepare_distmats[0][0].shape[0]
    print('pcoares_0:', pcoares_0)
    print('pcoares_0.vectors:', pcoares_0.vectors)
    print('pcoares_0.values:', pcoares_0.values)
    print('pcoares_0.names:', pcoares_0.names)


    pcoares_1 = pcoa(prepare_distmats[1][0])
    assert all([nm_a == nm_b for nm_a, nm_b in zip(pcoares_1.names, prepare_distmats[1][1])])
    assert pcoares_1.vectors.shape[0] == prepare_distmats[1][0].shape[0]
    print('pcoares_0:', pcoares_1)
    print('pcoares_0.vectors:', pcoares_1.vectors)
    print('pcoares_0.values:', pcoares_1.values)
    print('pcoares_0.names:', pcoares_1.names)


