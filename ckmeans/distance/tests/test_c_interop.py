import pytest

import numpy as np

from ckmeans.io import NucleotideAlignment
from ckmeans.distance.c_interop import \
    p_distance

def test_p_distance():
    aln_0 = NucleotideAlignment(
        ['s0', 's1', 's2', 's3'],
        np.array([
            ['A', 'C', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['T', 'C', '-', 'G', 'C', 'C', 'T', 'T', 'G', 'A'],
            ['A', 'G', 'T', 'G', 'C', 'C', 'T', 'A', 'G', 'A'],
            ['A', 'C', 'T', 'A', 'A', 'A', 'T', 'A', 'G', 'A'],
        ])
    )

    print(aln_0.sequences)

    p_d_0 = p_distance(aln_0.sequences)
    print('p_d_0:', p_d_0)
