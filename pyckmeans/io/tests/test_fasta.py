import pytest
import tempfile
import os

from pyckmeans.io import fasta
from pyckmeans.io.fasta import InvalidFastaAlignmentError

FASTA_STR_0 = \
'''
>Sample 0
ACTGTCATG
>Sample 1
ACT--CATC
'''

FASTA_STR_1 = \
'''
>Sample 0
ACT GTC ATG
>Sample 1
ACT --C ATC
'''

FASTA_STR_2 = \
'''
>Sample 0
ACT
GTC
ATG
>Sample 1
ACT
--C
ATC
'''

FASTA_STR_3 = \
'''
>Sample 0
ACTGTCAT
>Sample 1
ACT--CATC
'''

FASTA_STR_4 = \
'''
>Sample 0
ACTGTCATA
>Sample 1
ACT--CAT
'''

@pytest.fixture(scope='session')
def prep_fasta_files():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        fasta_file_0 = os.path.join(tempdir, 'fasta_0.fasta')
        with open(fasta_file_0, 'w') as f:
            f.write(FASTA_STR_0)

        fasta_file_1 = os.path.join(tempdir, 'fasta_1.fasta')
        with open(fasta_file_1, 'w') as f:
            f.write(FASTA_STR_1)

        fasta_file_2 = os.path.join(tempdir, 'fasta_2.fasta')
        with open(fasta_file_2, 'w') as f:
            f.write(FASTA_STR_2)
        
        fasta_file_3 = os.path.join(tempdir, 'fasta_3.fasta')
        with open(fasta_file_3, 'w') as f:
            f.write(FASTA_STR_3)
        
        fasta_file_4 = os.path.join(tempdir, 'fasta_4.fasta')
        with open(fasta_file_4, 'w') as f:
            f.write(FASTA_STR_4)

        yield (
            # should work
            fasta_file_0,
            fasta_file_1,
            fasta_file_2,

            # shouldn't work
            fasta_file_3,
            fasta_file_4,
        )

        print(f'Deleted temporary directory {tempdir}.')

def test_read_fasta_alignment(prep_fasta_files):
    r_0 = fasta.read_fasta_alignment(prep_fasta_files[0])
    r_1 = fasta.read_fasta_alignment(prep_fasta_files[1])
    r_2 = fasta.read_fasta_alignment(prep_fasta_files[2])

    print('r_0', r_0)
    print('r_1', r_1)
    print('r_2', r_2)

    with pytest.raises(InvalidFastaAlignmentError):
        r_3 = fasta.read_fasta_alignment(prep_fasta_files[3])
    with pytest.raises(InvalidFastaAlignmentError):
        r_4 = fasta.read_fasta_alignment(prep_fasta_files[4])
