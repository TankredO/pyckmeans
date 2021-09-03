import pandas
import pytest
import tempfile
import os

import numpy as np
from sklearn.datasets import make_blobs

from pyckmeans.core.multickmeans import MultiCKMeans
from pyckmeans.core.ckmeans import CKmeans, CKmeansResult, InvalidClusteringMetric
from pyckmeans.core.wecr import WECR, WECRResult, InvalidConstraintsError, InvalidKError

@pytest.fixture(scope='session')
def test_dir():
    with tempfile.TemporaryDirectory() as tempdir:

        yield tempdir

        print(f'Deleted temporary directory {tempdir}.')

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

def test_simple():
    ckm_0 = CKmeans(2)
    ckm_1 = CKmeans(np.array(3, dtype=int))
    ckm_2 = CKmeans(np.array(3, dtype=np.int64))

def test_ckmeans():
    x_0, _ = make_blobs(100, 5, centers=3, center_box=[-15, 15], shuffle=False)
    ckm_0 = CKmeans(3, metrics=['sil', 'bic', 'db', 'ch'])
    ckm_0.fit(x_0)
    ckm_res_0 = ckm_0.predict(x_0)

    with pytest.raises(InvalidClusteringMetric):
        CKmeans(3, metrics=['NONEXISTENT_METRIC'])

def test_wecr():
    x_0, _ = make_blobs(100, 5, centers=3, center_box=[-15, 15], shuffle=False)
    wecr_0 = WECR([2,3,4,5], 100)
    wecr_0.fit(x_0)
    
    wecr_res_0 = wecr_0.predict(x_0, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51], [5, 100]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[[0, 1], [101, 2]], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[['a', 'b'], ['c', 'd']], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[[0, 1], [0, 2]], must_not_link=[['a', 'b'], ['c', 'd']])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[[0, 1, 0, 2]], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_0, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51, 5, 99]])

    x_1 = pandas.DataFrame(x_0)
    wecr_0.fit(x_1)
    wecr_res_1 = wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51], [5, 99]])
    wecr_res_1 = wecr_0.predict(x_1, must_link=[['0', '1'], ['0', '2']], must_not_link=[[0, 51], [5, 99]])
    wecr_res_1 = wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[['0', '51'], ['5', '99']])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51], [5, 100]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[['a', 'b'], ['c', 'd']], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 'b'], ['c', 'd']], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], ['c', 'd']], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], [0, 'd']], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[['a', 'b'], ['c', 'd']])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51], [5, 'd']])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1, 0, 2]], must_not_link=[[0, 51], [5, 99]])
    with pytest.raises(InvalidConstraintsError):
        wecr_0.predict(x_1, must_link=[[0, 1], [0, 2]], must_not_link=[[0, 51, 5, 99]])

    wecr_res_1_ro = wecr_res_1.reorder(wecr_res_1.order(linkage_type='single'))
    wecr_res_1_sort = wecr_res_1.sort(linkage_type='single')
    assert_wecr_res_equal(wecr_res_1_ro, wecr_res_1_sort)

    wecr_res_1.plot(2)
    wecr_res_1.plot_metrics()

    wecr_res_1_rcm = wecr_res_1.recalculate_cluster_memberships(x_1, 'average', in_place=True)
    assert wecr_res_1_rcm is wecr_res_1

    wecr_res_1_rcm = wecr_res_1.recalculate_cluster_memberships(x_1, 'average', in_place=False)
    assert not wecr_res_1_rcm is wecr_res_1

    cl = wecr_res_1.get_cl(2, with_names=False)
    cl = wecr_res_1.get_cl(2, with_names=True)
    with pytest.raises(InvalidKError):
        wecr_res_1.get_cl(12500)
    with pytest.raises(InvalidKError):
        wecr_res_1.get_cl(-1)

def test_multickmeans():
    x_0, _ = make_blobs(100, 5, centers=3, center_box=[-15, 15], shuffle=False)
    mckm_0 = MultiCKMeans([2,3,4,5], n_rep=25)
    mckm_0.fit(x_0)
    mckm_res_0 = mckm_0.predict(x_0)
    mckm_res_0_ro = mckm_res_0.reorder(mckm_res_0.order(2), in_place=True)
    assert mckm_res_0 is mckm_res_0_ro

    mckm_res_0_ro = mckm_res_0.reorder(mckm_res_0.order(2), in_place=False)
    assert not mckm_res_0 is mckm_res_0_ro

    with pytest.raises(InvalidClusteringMetric):
        MultiCKMeans([2,3,4,5], n_rep=25, metrics=['NONEXISTENT_METRIC'])

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

    with pytest.raises(Exception):
        CKmeansResult.from_dir('SOME_NONEXISTENT_DIR')

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

    with pytest.raises(Exception):
        WECRResult.from_dir('SOME_NONEXISTENT_DIR')
