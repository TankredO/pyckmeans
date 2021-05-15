from numpy.core.numeric import allclose
import pytest
import tempfile
import os

from ckmeans.io import read_alignment
from ckmeans.distance import alignment_distance
from ckmeans.pcoa import pcoa
from ckmeans.core import CKmeans

PHYLIP_STR_0 = \
'''10 10
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

def test_simple_worflow(prep_phylip_files):
    na_0 = read_alignment(prep_phylip_files[0])
    d_0 = alignment_distance(na_0, 'p')
    pcoares_0 = pcoa(d_0, 'lingoes')
    ckm_0 = CKmeans(k=2, n_rep=10)
    ckm_0.fit(pcoares_0.vectors)
    cm_0 = ckm_0.predict(pcoares_0.vectors)

    print(cm_0)
