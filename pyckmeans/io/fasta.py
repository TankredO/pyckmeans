''' fasta

    Module for reading and writing FASTA files.
'''


import itertools
import re
from typing import Tuple, Union

import numpy

class InvalidFastaAlignmentError(Exception):
    '''InvalidFastaAlignmentError
    '''

WHITESPACE_RE = re.compile(r'\s+')

def read_fasta_alignment(
        fasta_file: str,
        dtype: Union[str, numpy.dtype] = 'U',
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''read_fasta_alignment

    Read fasta alignment file. This function expects the fasta to be a valid alignment,
    meaning that it should contain at least 2 sequences of the same length, including
    gaps.

    Parameters
    ----------
    fasta_file : str
        Path to a fasta file.
    dtype: Union[str, numpy.dtype]
        Data type to use for the sequence array.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Tuple of sequences and names, each as numpy array.

    Raises
    ------
    InvalidFastaAlignmentError
        Raised if less than 2 sequences are present in fasta_file.
    InvalidFastaAlignmentError
        Raised if the sequences have different lengths.
    '''

    names = []
    seqs = []
    first = True
    with open(fasta_file) as fasta_f:
        seq_buffer = []
        for line in fasta_f:
            _line = line.strip()

            # empty line
            if not _line:
                continue

            # name line
            if _line[0] == '>':
                names.append(_line[1:])
                if not first:
                    seqs.append(list(itertools.chain(*seq_buffer)))
                    seq_buffer = []
                else:
                    first = False
            # sequence line
            else:
                seq_buffer.append(re.sub(WHITESPACE_RE, '', _line).upper())

        seqs.append(list(itertools.chain(*seq_buffer)))

    # check alignment validity
    n_seq = len(seqs)
    if len(seqs) < 2:
        msg = f'Expected at least 2 entries but found only {n_seq}.'
        raise InvalidFastaAlignmentError(msg)

    seq_len = len(seqs[0])
    for i, seq in enumerate(seqs[1:]):
        cur_seq_len = len(seq)
        if cur_seq_len != seq_len:
            msg = f'Expected all sequences to have length {seq_len}' +\
                f'(length of sequence #0) but sequence #{i+1} has length {cur_seq_len}.'
            raise InvalidFastaAlignmentError(msg)

    seqs = numpy.array(seqs, dtype=dtype)
    names = numpy.array(names)

    return seqs, names
