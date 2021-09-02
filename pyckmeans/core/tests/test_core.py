import pytest
import tempfile
import os

import numpy as np
from sklearn.datasets import make_blobs

from pyckmeans.core.multickmeans import MultiCKMeans
from pyckmeans.core.ckmeans import CKmeans, CKmeansResult
from pyckmeans.core.wecr import WECR, WECRResult

@pytest.fixture(scope='session')
def test_dir():
    with tempfile.TemporaryDirectory() as tempdir:

        yield tempdir

        print(f'Deleted temporary directory {tempdir}.')

def test_simple():
    ckm_0 = CKmeans(2)
    ckm_1 = CKmeans(np.array(3, dtype=int))
    ckm_2 = CKmeans(np.array(3, dtype=np.int64))

def test_multi():
    mckm_0 = MultiCKMeans(np.arange(10))

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
def test_save_load_ckm_res(test_dir, return_cls):
    x, _ = make_blobs(100, 5, centers=3)
    ckm = CKmeans(3, 20)
    ckm.fit(x)
    ckm_res = ckm.predict(x, return_cls=return_cls)

    if return_cls:
        ckm_res_km_cls_file = os.path.join(test_dir, f'{return_cls}_ckm_res_km_cls.txt')
        ckm_res.save_km_cls(ckm_res_km_cls_file, one_hot=False)
        ckm_res.save_km_cls(ckm_res_km_cls_file, one_hot=True)

    assert_ckm_res_equal(
        ckm_res, CKmeansResult.from_json_str(ckm_res.to_json())
    )

    ckm_res_json_file = os.path.join(test_dir, f'{return_cls}_ckm_res.json')
    ckm_res.to_json(ckm_res_json_file)
    ckm_res_l = CKmeansResult.from_json(ckm_res_json_file)

    assert_ckm_res_equal(ckm_res, ckm_res_l)

    ckm_res_dir = os.path.join(test_dir, f'{return_cls}_ckm_res')
    ckm_res.to_dir(ckm_res_dir)
    ckm_res.to_dir(ckm_res_dir, force=True)
    with pytest.raises(Exception):
        ckm_res.to_dir(ckm_res_dir, force=False)
    ckm_res_l = CKmeansResult.from_dir(ckm_res_dir)
    assert_ckm_res_equal(ckm_res, ckm_res_l)

def assert_wecr_res_equal(a: WECRResult, b: WECRResult, eps=1e-8):
    assert (np.abs(a.cmatrix - b.cmatrix) < eps).all()
    assert (a.cl == b.cl).all()
    assert (a.bic is b.bic) or ((a.bic - b.bic) < eps).all()
    assert (a.db is b.db) or ((a.db - b.db) < eps).all()
    assert (a.sil is b.sil) or ((a.sil - b.sil) < eps).all()
    assert (a.ch is b.ch) or ((a.ch - b.ch) < eps).all()
    assert (a.names == b.names).all()
    assert a.km_cls is b.km_cls or \
        (np.abs(a.km_cls - b.km_cls) < eps).all()

@pytest.mark.parametrize('return_cls', [True, False])
def test_save_load_wecr_res(test_dir, return_cls):
    x, _ = make_blobs(100, 5, centers=3, center_box=[-15, 15])
    wecr = WECR([2,3,4,5], 100)
    wecr.fit(x)
    wecr_res = wecr.predict(x, return_cls=return_cls)

    if return_cls:
        wecr_res_km_cls_file = os.path.join(test_dir, f'{return_cls}_wecr_res_km_cls.txt')
        wecr_res.save_km_cls(wecr_res_km_cls_file, one_hot=False)
        wecr_res.save_km_cls(wecr_res_km_cls_file, one_hot=True)

    assert_wecr_res_equal(
        wecr_res, WECRResult.from_json_str(wecr_res.to_json())
    )

    wecr_res_json_file = os.path.join(test_dir, f'{return_cls}_wecr_res.json')
    wecr_res.to_json(wecr_res_json_file)
    wecr_res_l = WECRResult.from_json(wecr_res_json_file)
    assert_wecr_res_equal(wecr_res, wecr_res_l)

    wecr_res_dir = os.path.join(test_dir, f'{return_cls}_wecr_res')
    wecr_res.to_dir(wecr_res_dir)
    wecr_res.to_dir(wecr_res_dir, force=True)
    with pytest.raises(Exception):
        wecr_res.to_dir(wecr_res_dir, force=False)
    wecr_res_l = WECRResult.from_dir(wecr_res_dir)
    assert_wecr_res_equal(wecr_res, wecr_res_l)
