import pytest

import numpy as np

from pyckmeans.io import NucleotideAlignment
from pyckmeans.distance.c_interop import \
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
    p_d_0_pd = np.array([
        [0.0000000, 0.2222222, 0.1000000, 0.3000000],
        [0.2222222, 0.0000000, 0.3333333, 0.5555556],
        [0.1000000, 0.3333333, 0.0000000, 0.4000000],
        [0.3000000, 0.5555556, 0.4000000, 0.0000000]
    ])
    p_d_0_cd = np.array([
        [0.0000000, 0.2222222, 0.1111111, 0.3333333],
        [0.2222222, 0.0000000, 0.3333333, 0.5555556],
        [0.1111111, 0.3333333, 0.0000000, 0.4444444],
        [0.3333333, 0.5555556, 0.4444444, 0.0000000]
    ])
    jc_d_0_pd = np.array([
        [0.0000000, 0.2635484, 0.1073256, 0.3831192],
        [0.2635484, 0.0000000, 0.4408400, 1.0124450],
        [0.1073256, 0.4408400, 0.0000000, 0.5716050],
        [0.3831192, 1.0124450, 0.5716050, 0.0000000]
    ])
    jc_d_0_cd = np.array([
        [0.0000000, 0.2635484, 0.1202570, 0.4408400],
        [0.2635484, 0.0000000, 0.4408400, 1.0124450],
        [0.1202570, 0.4408400, 0.0000000, 0.6734562],
        [0.4408400, 1.0124450, 0.6734562, 0.0000000]
    ])
    k2p_d_0_pd = np.array([
        [0.0000000, 0.2726039, 0.1084661, 0.3831192],
        [0.2726039, 0.0000000, 0.4773856, 1.0986123],
        [0.1084661, 0.4773856, 0.0000000, 0.5756463],
        [0.3831192, 1.0986123, 0.5756463, 0.0000000]
    ])
    k2p_d_0_cd = np.array([
        [0.0000000,0.2726039,0.1217201,0.4408400],
        [0.2726039,0.0000000,0.4773856,1.0986123],
        [0.1217201,0.4773856,0.0000000,0.6801182],
        [0.4408400,1.0986123,0.6801182,0.0000000],
    ])


    return (
        (
            aln_0,
            {
                'p_pd': p_d_0_pd, 'p_cd': p_d_0_cd,
                'jc_pd': jc_d_0_pd, 'jc_cd': jc_d_0_cd,
                'k2p_pd': k2p_d_0_pd, 'k2p_cd': k2p_d_0_cd,
            }
        ),
    )

def test_p_distance(prepare_alignments):
    eps = 0.001

    aln_0, d_expected_0 = prepare_alignments[0]
    p_d_0_pd_expected = d_expected_0['p_pd']
    p_d_0_cd_expected = d_expected_0['p_cd']

    print(aln_0.sequences)

    p_d_0_pd = p_distance(aln_0.sequences, True)
    print('p_d_0_pd:', p_d_0_pd)
    assert np.max(np.abs(p_d_0_pd - p_d_0_pd_expected)) < eps
    p_d_0_cd = p_distance(aln_0.sequences, False)
    print('p_d_0_cd:', p_d_0_cd)
    assert np.max(np.abs(p_d_0_cd - p_d_0_cd_expected)) < eps

def test_jc_distance(prepare_alignments):
    eps = 0.001

    aln_0, d_expected_0 = prepare_alignments[0]
    jc_d_0_pd_expected = d_expected_0['jc_pd']
    jc_d_0_cd_expected = d_expected_0['jc_cd']

    print(aln_0.sequences)

    jc_d_0_pd = jc_distance(aln_0.sequences, True)
    print('jc_d_0_pd:', jc_d_0_pd)
    assert np.max(np.abs(jc_d_0_pd - jc_d_0_pd_expected)) < eps
    jc_d_0_cd = jc_distance(aln_0.sequences, False)
    print('jc_d_0_cd:', jc_d_0_cd)
    assert np.max(np.abs(jc_d_0_cd - jc_d_0_cd_expected)) < eps

def test_k2p_distance(prepare_alignments):
    eps = 0.001

    aln_0, d_expected_0 = prepare_alignments[0]
    k2p_d_0_pd_expected = d_expected_0['k2p_pd']
    k2p_d_0_cd_expected = d_expected_0['k2p_cd']

    print(aln_0.sequences)

    k2p_d_0_pd = k2p_distance(aln_0.sequences, True)
    print('k2p_d_0_pd:', k2p_d_0_pd)
    assert np.max(np.abs(k2p_d_0_pd - k2p_d_0_pd_expected)) < eps
    k2p_d_0_cd = k2p_distance(aln_0.sequences, False)
    print('k2p_d_0_cd:', k2p_d_0_cd)
    assert np.max(np.abs(k2p_d_0_cd - k2p_d_0_cd_expected)) < eps
