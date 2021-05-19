import pytest

import numpy as np

from ckmeans.io import NucleotideAlignment
from ckmeans.distance.c_interop import \
    p_distance,\
    jc_distance,\
    k2p_distance

@pytest.fixture(scope='session')
def prepare_alignments():
    aln_0 = NucleotideAlignment(
        ['s0', 's1', 's2', 's3'],
        np.array([
            ['A', 'C', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['T', 'C', '-', 'G', 'C', 'C', 'T', 'T', 'G', 'A'],
            ['A', 'G', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['A', 'C', 'T', 'A', 'A', 'A', 'T', 'A', 'G', 'A'],
        ])
    )

    return (
        aln_0,
    )

def test_p_distance(prepare_alignments):
    aln_0 = prepare_alignments[0]

    print(aln_0.sequences)

    p_d_0_pd = p_distance(aln_0.sequences, True)
    print('p_d_0_pd:', p_d_0_pd)
    p_d_0_cd = p_distance(aln_0.sequences, False)
    print('p_d_0_cd:', p_d_0_cd)

def test_jc_distance(prepare_alignments):
    aln_0 = prepare_alignments[0]

    print(aln_0.sequences)

    jc_d_0_pd = jc_distance(aln_0.sequences, True)
    print('jc_d_0_pd:', jc_d_0_pd)
    jc_d_0_cd = jc_distance(aln_0.sequences, False)
    print('jc_d_0_cd:', jc_d_0_cd)

def test_k2p_distance(prepare_alignments):
    aln_0 = prepare_alignments[0]

    print(aln_0.sequences)

    k2p_d_0_pd = k2p_distance(aln_0.sequences, True)
    print('k2p_d_0_pd:', k2p_d_0_pd)
    k2p_d_0_cd = k2p_distance(aln_0.sequences, False)
    print('k2p_d_0_cd:', k2p_d_0_cd)
