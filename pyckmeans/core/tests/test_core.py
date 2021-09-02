import pytest
import tempfile
import os

import numpy as np

from sklearn.datasets import make_blobs
import pyckmeans
from pyckmeans.core import CKmeansResult

@pytest.fixture(scope='session')
def test_dir():
    with tempfile.TemporaryDirectory() as tempdir:

        yield tempdir

        print(f'Deleted temporary directory {tempdir}.')

def test_simple():
    ckm_0 = pyckmeans.CKmeans(2)
    ckm_1 = pyckmeans.CKmeans(np.array(3, dtype=int))
    ckm_2 = pyckmeans.CKmeans(np.array(3, dtype=np.int64))

def test_multi():
    mckm_0 = pyckmeans.MultiCKMeans(np.arange(10))

def assert_ckm_res_equal(a: CKmeansResult, b: CKmeansResult, eps=1e-8):
    assert (np.abs(a.cmatrix - b.cmatrix) < eps).all()
    assert (a.cl == b.cl).all()
    assert (a.bic is b.bic) or (a.bic - b.bic) < eps
    assert (a.db is b.db) or (a.db - b.db) < eps
    assert (a.sil is b.sil) or (a.sil - b.sil) < eps
    assert (a.ch is b.ch) or (a.ch - b.ch) < eps
    assert (a.names == b.names).all()
    assert a.km_cls is b.km_cls or \
        (np.abs(a.km_cls - b.km_cls) < eps).all()

@pytest.mark.parametrize('return_cls', [True, False])
def test_save_load(test_dir, return_cls):
    x, _ = make_blobs(100, 5, centers=3)
    ckm = pyckmeans.CKmeans(3, 20)
    ckm.fit(x)
    ckm_res = ckm.predict(x, return_cls=return_cls)

    ckm_res_json_file = os.path.join(test_dir, f'{return_cls}_ckm_res.json')
    ckm_res.to_json(ckm_res_json_file)
    ckm_res_l = CKmeansResult.from_json(ckm_res_json_file)
    assert_ckm_res_equal(ckm_res, ckm_res_l)

    ckm_res_dir = os.path.join(test_dir, f'{return_cls}_ckm_res')
    ckm_res.to_dir(ckm_res_dir)
    ckm_res_l = CKmeansResult.from_dir(ckm_res_dir)
    assert_ckm_res_equal(ckm_res, ckm_res_l)
