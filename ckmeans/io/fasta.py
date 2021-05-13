''' fasta

    Module for reading and writing FASTA files.
'''


import itertools
import re

import numpy as np

WHITESPACE_RE = re.compile(r'\s+')

def read_fasta_alignment(fasta_file: str):
    names = []
    seqs = []
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
                seqs.append(itertools.chain(*seq_buffer))
                seq_buffer = []
            # sequence line
            else:
                seq_buffer.append(re.sub(WHITESPACE_RE, '', _line))

    return names, seqs
