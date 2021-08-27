import numpy
import pandas as pd
import pytest
import tempfile
import os

from pyckmeans.io import read_alignment
from pyckmeans.distance import alignment_distance
from pyckmeans.ordination import PCOAResult, pcoa
from pyckmeans.core import CKmeans, MultiCKMeans
from pyckmeans.utils import plot_ckmeans_result, plot_multickmeans_metrics

PHYLIP_STR_0 = \
'''10 14
Sample0 ACTGTCATGAAGGA
Sample1 ACT--CATCAAGGA
Sample2 ACTCTCATGAAGGA
Sample3 AGTCTCTTGAAGGA
Sample4 AGT--CATGAACTG
Sample5 ACTGTCATGAACTG
Sample6 ACTC-CATCAACTG
Sample7 AGGCTCCTGAACTG
Sample8 ACTCTCTTTAACTG
Sample9 TTTCTCACGAACTG
'''

@pytest.fixture(scope='session')
def prep_phylip_files():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        phylip_file_0 = os.path.join(tempdir, 'phylip_0.phy')
        with open(phylip_file_0, 'w') as f:
            f.write(PHYLIP_STR_0)

        yield (
            phylip_file_0,
        )

        print(f'Destroyed temporary directory {tempdir}.')

@pytest.fixture(scope='session')
def prep_pcoa_results(prep_phylip_files):
    na_0 = read_alignment(prep_phylip_files[0])
    d_0_p = alignment_distance(na_0, 'p')
    pcoares_0 = pcoa(d_0_p, 'lingoes')

    return (
        pcoares_0,
    )

def test_simple_workflow(prep_phylip_files):
    na_0 = read_alignment(prep_phylip_files[0])
    d_0_p = alignment_distance(na_0, 'p')
    pcoares_0 = pcoa(d_0_p, 'lingoes')
    ckm_0 = CKmeans(k=2, n_rep=50, n_init=2)
    ckm_0.fit(pcoares_0.vectors)
    ckm_0_res = ckm_0.predict(pcoares_0.vectors)
    ckm_0_res.sort(in_place=True)

    print('pcoares_0.vectors', pcoares_0.vectors)
    print('ckm_0_res.cl:', ckm_0_res.cl)

    ckm_1 = CKmeans(k=2, n_rep=50, n_init=2)
    ckm_1.fit(pcoares_0)
    ckm_1_res = ckm_1.predict(pcoares_0)
    ckm_1_res.sort(in_place=True)

    print('ckm_1_res.cl:', ckm_1_res.cl)
    print('ckm_1_res.names:', ckm_1_res.names)

    ckm_2 = CKmeans(k=2, n_rep=50, n_init=2)
    df = pd.DataFrame(pcoares_0.vectors, pcoares_0.names)
    ckm_2.fit(df)
    ckm_2_res = ckm_2.predict(df)
    ckm_2_res.sort(in_place=True)
    print('ckm_2_res.cl:', ckm_2_res.cl)
    print('ckm_2_res.names:', ckm_2_res.names)

    ckm_2_res_cls = ckm_2.predict(df, return_cls=True)
    assert ckm_2_res_cls.km_cls.shape == (ckm_2.n_rep, df.shape[0])
    ckm_2_res_cls.sort()

    # test copy
    ckm_2_res_cls_cp = ckm_2_res_cls.copy()
    ckm_2_res_cls_cp.cl[0] = -1000
    ckm_2_res_cls_cp.km_cls[0,0] = -1000
    assert ckm_2_res_cls_cp.cl[0] != ckm_2_res_cls.cl[0]
    assert ckm_2_res_cls_cp.km_cls[0,0] != ckm_2_res_cls.km_cls[0,0]
    assert not ckm_2_res_cls_cp is ckm_2_res_cls

    # test recalculate cluster memberships
    ckm_2_res_cls_cp_rcm_1 = ckm_2_res_cls_cp.recalculate_cluster_memberships(df, 'average', in_place=False)
    ckm_2_res_cls_cp_rcm_2 = ckm_2_res_cls_cp.recalculate_cluster_memberships(df, 'average', in_place=True)
    assert ckm_2_res_cls_cp_rcm_2 is ckm_2_res_cls_cp
    assert not ckm_2_res_cls_cp_rcm_1 is ckm_2_res_cls_cp

    # test distance
    d_0_p_0 = alignment_distance(na_0, 'p')
    d_0_p_1 = na_0.distance('p')
    assert numpy.abs(d_0_p_0.dist_mat - d_0_p_1.dist_mat).sum() < 0.001

def test_multi_workflow(prep_pcoa_results):
    pcoares_0: PCOAResult = prep_pcoa_results[0]
    mckm_0 = MultiCKMeans([2,3,3])
    mckm_0.fit(pcoares_0)
    mckm_0_res = mckm_0.predict(pcoares_0)

    plot_multickmeans_metrics(mckm_0_res)
    mckm_0_res.plot_metrics()

    mckm_1 = MultiCKMeans([2,3,3])
    mckm_1.fit(pcoares_0.vectors)
    mckm_1_res = mckm_1.predict(pcoares_0.vectors)
    plot_multickmeans_metrics(mckm_1_res)
    mckm_1_res.plot_metrics()

    mckm_2 = MultiCKMeans([2,3,3], n_rep=100)
    df = pd.DataFrame(pcoares_0.vectors, pcoares_0.names)
    mckm_2.fit(df)
    mckm_2_res = mckm_2.predict(df)
    plot_multickmeans_metrics(mckm_2_res)
    mckm_2_res.plot_metrics()

    mckm_2_res_cls = mckm_2.predict(df, return_cls=True)
    assert mckm_2_res_cls.ckmeans_results[0].km_cls.shape == (mckm_2.n_rep, df.shape[0])
    mckm_2_res_cls.sort(0)

def test_plotting(prep_pcoa_results):
    pcoares_0 = prep_pcoa_results[0]
    ckm_0 = CKmeans(k=2, n_rep=10)
    ckm_0.fit(pcoares_0)
    ckm_0_res = ckm_0.predict(pcoares_0)

    ckm_0_res.sort()
    ord = ckm_0_res.order()
    ckm_0_res.reorder(ord)

    plot_ckmeans_result(ckm_0_res)
    plot_ckmeans_result(ckm_0_res, order=None)
    plot_ckmeans_result(ckm_0_res, order=ord)

    ckm_0_res.plot()
    ckm_0_res.plot(order=None)
    ckm_0_res.plot(order=ord)
