import pytest
import tempfile
import os

from ckmeans.io import phylip
from ckmeans.io.phylip import InvalidPhylipAlignmentError

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
