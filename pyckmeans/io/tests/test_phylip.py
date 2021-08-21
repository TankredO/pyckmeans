import pytest
import tempfile
import os

import numpy as np

from pyckmeans.io import phylip
from pyckmeans.io.phylip import InvalidPhylipAlignmentError, InvalidPhylipMatrixError, read_phylip_distmat, write_phylip_distmat


# ==== alignment

PHYLIP_STR_0 = \
'''2 9
Sample0 ACTGTCATG
Sample1 ACT--CATC
'''

PHYLIP_STR_1 = \
'''2 9
Sample0 ACTGT CATG
Sample1 ACT-- CATC
'''

PHYLIP_STR_2 = \
'''2 9
Sample0 ACTGT CATG

Sample1 ACT-- CATC

'''

PHYLIP_STR_3 = \
'''2 9 3
Sample0 ACTGTCATG
Sample1 ACT--CATC
'''

PHYLIP_STR_4 = \
'''2 8
Sample0 ACTGTCATG
Sample1 ACT--CATC
Sample2 ACTTGCATC
'''

@pytest.fixture(scope='session')
def prep_phylip_files():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        phylip_file_0 = os.path.join(tempdir, 'phylip_0.phy')
        with open(phylip_file_0, 'w') as f:
            f.write(PHYLIP_STR_0)

        phylip_file_1 = os.path.join(tempdir, 'phylip_1.phy')
        with open(phylip_file_1, 'w') as f:
            f.write(PHYLIP_STR_1)

        phylip_file_2 = os.path.join(tempdir, 'phylip_2.phy')
        with open(phylip_file_2, 'w') as f:
            f.write(PHYLIP_STR_2)
        
        phylip_file_3 = os.path.join(tempdir, 'phylip_3.phy')
        with open(phylip_file_3, 'w') as f:
            f.write(PHYLIP_STR_3)
        
        phylip_file_4 = os.path.join(tempdir, 'phylip_4.phy')
        with open(phylip_file_4, 'w') as f:
            f.write(PHYLIP_STR_4)

        yield (
            # should work
            phylip_file_0,
            phylip_file_1,
            phylip_file_2,

            # shouldn't work
            phylip_file_3,
            phylip_file_4,
        )

        print(f'Deleted temporary directory {tempdir}.')

def test_read_fasta_alignment(prep_phylip_files):
    r_0 = phylip.read_phylip_alignment(prep_phylip_files[0])
    r_1 = phylip.read_phylip_alignment(prep_phylip_files[1])
    r_2 = phylip.read_phylip_alignment(prep_phylip_files[2])

    print('r_0', r_0)
    print('r_1', r_1)
    print('r_2', r_2)

    with pytest.raises(InvalidPhylipAlignmentError):
        r_3 = phylip.read_phylip_alignment(prep_phylip_files[3])
    with pytest.raises(InvalidPhylipAlignmentError):
        r_4 = phylip.read_phylip_alignment(prep_phylip_files[4])



# ==== distance

PHYLIP_DIST_STR_0 = \
'''4
Sample0 0.00 0.90 0.80 0.30
Sample1 0.90 0.00 0.40 0.70
Sample2 0.80 0.40 0.00 0.50
Sample3 0.30 0.70 0.50 0.00
'''

PHYLIP_DIST_STR_1 = \
'''4
Sample0
Sample1 0.90
Sample2 0.80 0.40
Sample3 0.30 0.70 0.50
'''

PHYLIP_DIST_STR_2 = \
'''5
Sample0
Sample1 0.90
Sample2 0.80 0.40
Sample3 0.30 0.70 0.50
'''

PHYLIP_DIST_STR_3 = \
'''4
Sample0
Sample1 0.90
Sample2 0.80 
Sample3 0.30 0.70 0.50
'''

PHYLIP_DIST_STR_4 = \
'''5
Sample0 0.00 0.90 0.80 0.30
Sample1 0.90 0.00 0.40 0.70
Sample2 0.80 0.40 0.00 0.50
Sample3 0.30 0.70 0.50 0.00
'''

PHYLIP_DIST_STR_5 = \
'''5
Sample0 0.00 0.90 0.80 0.30
Sample1 0.90  0.40 0.70
Sample2 0.80 0.40 0.00 0.50
Sample3 0.30 0.70 0.50 0.00
'''

PHYLIP_DIST_STR_6 = \
'''5
Sample0 0.00 0.90 0.80 0.30

Sample1 0.90 0.00 0.40 0.70
Sample2 0.80 0.40 0.00 0.50
Sample3 0.30 0.70 0.50 0.00
'''

@pytest.fixture(scope='session')
def prep_phylip_dist_files():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        phylip_dist_file_0 = os.path.join(tempdir, 'phylip_dist_0.dist')
        with open(phylip_dist_file_0, 'w') as f:
            f.write(PHYLIP_DIST_STR_0)

        phylip_dist_file_1 = os.path.join(tempdir, 'phylip_dist_1.dist')
        with open(phylip_dist_file_1, 'w') as f:
            f.write(PHYLIP_DIST_STR_1)

        phylip_dist_file_2 = os.path.join(tempdir, 'phylip_dist_2.dist')
        with open(phylip_dist_file_2, 'w') as f:
            f.write(PHYLIP_DIST_STR_2)

        phylip_dist_file_3 = os.path.join(tempdir, 'phylip_dist_3.dist')
        with open(phylip_dist_file_3, 'w') as f:
            f.write(PHYLIP_DIST_STR_3)

        phylip_dist_file_4 = os.path.join(tempdir, 'phylip_dist_4.dist')
        with open(phylip_dist_file_4, 'w') as f:
            f.write(PHYLIP_DIST_STR_4)

        phylip_dist_file_5 = os.path.join(tempdir, 'phylip_dist_5.dist')
        with open(phylip_dist_file_5, 'w') as f:
            f.write(PHYLIP_DIST_STR_5)

        phylip_dist_file_6 = os.path.join(tempdir, 'phylip_dist_6.dist')
        with open(phylip_dist_file_6, 'w') as f:
            f.write(PHYLIP_DIST_STR_6)

        yield (
            # should work
            phylip_dist_file_0,
            phylip_dist_file_1,

            # shouldn't work
            phylip_dist_file_2,
            phylip_dist_file_3,
            phylip_dist_file_4,
            phylip_dist_file_5,
            phylip_dist_file_6,
        )

        print(f'Deleted temporary directory {tempdir}.')

def test_phylip_distance(prep_phylip_dist_files):
    eps = 0.0001
    
    # == reading
    d_0 = read_phylip_distmat(prep_phylip_dist_files[0])
    nm_0 = d_0.names
    print('d_0:', d_0)

    d_1 = read_phylip_distmat(prep_phylip_dist_files[1])
    nm_1 = d_1.names
    print('d_1:', d_1)

    assert np.sum(np.abs(d_0.dist_mat - d_1.dist_mat)) < eps


    with pytest.raises(InvalidPhylipMatrixError):
        phylip.read_phylip_distmat(prep_phylip_dist_files[2])
    with pytest.raises(InvalidPhylipMatrixError):
        phylip.read_phylip_distmat(prep_phylip_dist_files[3])
    with pytest.raises(InvalidPhylipMatrixError):
        phylip.read_phylip_distmat(prep_phylip_dist_files[4])
    with pytest.raises(InvalidPhylipMatrixError):
        phylip.read_phylip_distmat(prep_phylip_dist_files[5])
    with pytest.raises(InvalidPhylipMatrixError):
        phylip.read_phylip_distmat(prep_phylip_dist_files[6])

    # == writing
    with tempfile.TemporaryDirectory() as tempdir:
        d_file_0 = os.path.join(tempdir, 'd_file_0.dist')
        write_phylip_distmat(d_0, d_file_0)

        d_0_r = read_phylip_distmat(d_file_0)
        nm_0_r = d_0_r.names
        assert all([a == b for a, b in zip(nm_0, nm_0_r)])
        assert np.sum(np.abs(d_0.dist_mat - d_0_r.dist_mat)) < eps

        d_file_1 = os.path.join(tempdir, 'd_file_1.dist')
        write_phylip_distmat(d_1, d_file_1)
        d_1_r = read_phylip_distmat(d_file_1)
        nm_1_r = d_1_r.names
        assert all([a == b for a, b in zip(nm_1, nm_1_r)])
        assert np.sum(np.abs(d_1.dist_mat - d_1_r.dist_mat)) < eps
