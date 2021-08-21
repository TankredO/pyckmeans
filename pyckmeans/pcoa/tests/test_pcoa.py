from numpy.testing._private.utils import assert_raises
import pytest

import numpy as np
import pandas as pd
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

    vectors_0_np = pcoares_0.get_vectors()
    assert isinstance(vectors_0_np, np.ndarray)
    assert vectors_0_np.shape == pcoares_0.vectors.shape
    vectors_0_pd = pcoares_0.get_vectors(out_format='pd')
    assert isinstance(vectors_0_pd, pd.DataFrame)
    assert vectors_0_pd.shape == pcoares_0.vectors.shape

    vectors_1_np = pcoares_0.get_vectors(filter_by='eigvals_rel_cum', filter_th=0.6, out_format='np')
    vectors_1_np_expected = pcoares_0.vectors[
        :,
        (pcoares_0.values['eigvals_rel_cum'] < 0.6)[:pcoares_0.vectors.shape[1]]
    ]
    assert vectors_1_np.shape == vectors_1_np_expected.shape
    assert abs(vectors_1_np - vectors_1_np_expected).sum() < 0.0001

    vectors_1_pd = pcoares_0.get_vectors(filter_by='eigvals_rel_cum', filter_th=0.6, out_format='pd')
    assert vectors_1_pd.shape == vectors_1_np_expected.shape
    assert abs(vectors_1_pd.values - vectors_1_np_expected).sum() < 0.0001


    pcoares_1 = pcoa(prepare_distmats[1][0])
    assert all([nm_a == nm_b for nm_a, nm_b in zip(pcoares_1.names, prepare_distmats[1][1])])
    assert pcoares_1.vectors.shape[0] == prepare_distmats[1][0].shape[0]
    print('pcoares_0:', pcoares_1)
    print('pcoares_0.vectors:', pcoares_1.vectors)
    print('pcoares_0.values:', pcoares_1.values)
    print('pcoares_0.names:', pcoares_1.names)

    vectors_1_pd = pcoares_1.get_vectors(filter_by='eigvals_rel_cum', filter_th=0.6, out_format='pd')
    vectors_1_np_expected = pcoares_1.vectors[
        :,
        (pcoares_1.values['eigvals_rel_cum'] < 0.6)[:pcoares_1.vectors.shape[1]]
    ]
    assert vectors_1_pd.shape == vectors_1_np_expected.shape
    assert abs(vectors_1_pd.values - vectors_1_np_expected).sum() < 0.0001
    print(vectors_1_pd.index.values)
    assert np.all(vectors_1_pd.index.values == pcoares_1.names)
