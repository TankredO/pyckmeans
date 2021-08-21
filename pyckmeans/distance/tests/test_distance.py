import pytest

import numpy as np

from pyckmeans.io import NucleotideAlignment
from pyckmeans.distance import alignment_distance, p_distance, InvalidDistanceTypeError

@pytest.fixture()
def prepare_alignments():
    aln_0 = NucleotideAlignment(
        ['s0', 's1', 's2', 's3'],
        np.array([
            ['A', 'C', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['T', 'C', 'T', 'G', 'C', 'C', 'T', 'T', 'G', 'A'],
            ['A', 'G', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['A', 'C', 'T', 'A', 'A', 'A', 'T', 'A', 'G', 'A'],
        ])
    )
    d_0_p = np.array([
        [0.0, 0.2, 0.1, 0.3],
        [0.2, 0.0, 0.3, 0.5],
        [0.1, 0.3, 0.0, 0.4],
        [0.3, 0.5, 0.4, 0.0],
    ])

    return (
        (aln_0, d_0_p),
    )

def test_p_distance(prepare_alignments):
    eps = 0.0001

    d_0 = alignment_distance(prepare_alignments[0][0], 'p')
    d_0_expected = prepare_alignments[0][1]
    assert np.all(np.abs(d_0.dist_mat - d_0_expected) < eps)

    print('d_0', d_0)

    d_0_p = p_distance(prepare_alignments[0][0].sequences)
    assert np.all(np.abs(d_0_p - d_0_expected) < eps)

def test_distances_simple(prepare_alignments):
    alignment_distance(prepare_alignments[0][0], 'p', True)
    alignment_distance(prepare_alignments[0][0], 'p', False)
    alignment_distance(prepare_alignments[0][0], 'jc', True)
    alignment_distance(prepare_alignments[0][0], 'jc', False)
    alignment_distance(prepare_alignments[0][0], 'k2p', True)
    alignment_distance(prepare_alignments[0][0], 'k2p', False)

    with pytest.raises(InvalidDistanceTypeError):
        alignment_distance(prepare_alignments[0][0], 'xyz', True)
    with pytest.raises(InvalidDistanceTypeError):
        alignment_distance(prepare_alignments[0][0], 'xyz', False)
