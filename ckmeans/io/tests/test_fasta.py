import pytest
import tempfile

from ckmeans.io import fasta

FASTA_0 = '''
>Sample 0
ACTGTCATG
>Sample 1
ACT--CATC
'''

def test_read_fasta_alignment():
    fasta.read_fasta_alignment('')
