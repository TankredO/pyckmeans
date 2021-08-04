from numpy.core.numeric import allclose
import pytest
import tempfile
import os

from ckmeans.io import read_alignment
from ckmeans.distance import alignment_distance
from ckmeans.pcoa import pcoa
from ckmeans.core import CKmeans
from ckmeans.plotting import plot_ckmeans_result

PHYLIP_STR_0 = \
'''10 9
Sample0 ACTGTCATG
Sample1 ACT--CATC
Sample2 ACTCTCATG
Sample3 AGTCTCTTG
Sample4 AGT--CATG
Sample5 ACTGTCATG
Sample6 ACTC-CATC
Sample7 AGGCTCCTG
Sample8 ACTCTCTTT
Sample9 TTTCTCACG
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
    ckm_0 = CKmeans(k=2, n_rep=10)
    ckm_0.fit(pcoares_0.vectors)
    ckm_0_res = ckm_0.predict(pcoares_0.vectors)
    ckm_0_res.sort(in_place=True)

    print('pcoares_0.vectors', pcoares_0.vectors)
    print('ckm_0_res.cl:', ckm_0_res.cl)

    ckm_1 = CKmeans(k=2, n_rep=10)
    ckm_1.fit(pcoares_0)
    ckm_1_res = ckm_1.predict(pcoares_0)
    ckm_1_res.sort(in_place=True)

    print('ckm_1_res.cl:', ckm_1_res.cl)
    print('ckm_1_res.names:', ckm_1_res.names)

def test_plotting(prep_pcoa_results):
    pcoares_0 = prep_pcoa_results[0]
    ckm_0 = CKmeans(k=2, n_rep=10)
    ckm_0.fit(pcoares_0)
    ckm_0_res = ckm_0.predict(pcoares_0)

    plot_ckmeans_result(ckm_0_res)
