import pytest
import tempfile
import os

import numpy as np
import pandas as pd
from pyckmeans.distance import DistanceMatrix
from pyckmeans.ordination import pcoa, PCOAResult

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

@pytest.fixture(scope='session')
def test_dir():
    with tempfile.TemporaryDirectory() as tempdir:

        yield tempdir

        print(f'Deleted temporary directory {tempdir}.')

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

def assert_pcoa_res_are_equal(
    a: PCOAResult,
    b: PCOAResult,
    eps: float=1e-8,
):
    assert (np.abs(a.vectors - b.vectors) < eps).all()
    assert (np.abs(a.values - b.values) < eps).values.all()
    assert (a.trace - b.trace) < eps or a.trace is b.trace
    assert a.trace_corr is b.trace_corr or (a.trace_corr - b.trace_corr) < eps
    assert a.correction == b.correction or a.correction is b.correction
    assert a.negative_eigvals == b.negative_eigvals

@pytest.mark.parametrize('correction', [None, 'lingoes', 'cailliez'])
def test_save_load(prepare_distmats, test_dir, correction):
    pcoa_res_0 = pcoa(prepare_distmats[0][0],  correction=correction)
    print('correction:', correction)

    pcoa_res_0_json_file = os.path.join(test_dir, f'{correction}_pcoa_res_0.json')
    pcoa_res_0.to_json(pcoa_res_0_json_file)
    pcoa_res_0_l = PCOAResult.from_json(pcoa_res_0_json_file)
    assert_pcoa_res_are_equal(pcoa_res_0, pcoa_res_0_l)

    pcoa_res_0_dir = os.path.join(test_dir, f'{correction}_pcoa_res_0')
    pcoa_res_0.to_dir(pcoa_res_0_dir)
    pcoa_res_0_l = PCOAResult.from_dir(pcoa_res_0_dir)
    assert_pcoa_res_are_equal(pcoa_res_0, pcoa_res_0_l)

    pcoa_res_1 = pcoa(prepare_distmats[1][0],  correction=correction)
    print('correction:', correction)

    pcoa_res_1_json_file = os.path.join(test_dir, f'{correction}_pcoa_res_1.json')
    pcoa_res_1.to_json(pcoa_res_1_json_file)
    pcoa_res_1_l = PCOAResult.from_json(pcoa_res_1_json_file)
    assert_pcoa_res_are_equal(pcoa_res_1, pcoa_res_1_l)

    pcoa_res_1_dir = os.path.join(test_dir, f'{correction}_pcoa_res_1')
    pcoa_res_1.to_dir(pcoa_res_1_dir)
    pcoa_res_1_l = PCOAResult.from_dir(pcoa_res_1_dir)
    assert_pcoa_res_are_equal(pcoa_res_1, pcoa_res_1_l)
