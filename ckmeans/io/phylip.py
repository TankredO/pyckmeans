''' fasta

    Module for reading and writing PHYLIP files.
'''

import re
from typing import Tuple, List

import numpy

from .nucleotide_alignment import NucleotideAlignment

WHITESPACE_RE = re.compile(r'\s+')

class InvalidPhylipAlignmentError(Exception):
    '''InvalidPhylipAlignmentError
    '''

def read_phylip_alignment(phylip_file: str) -> Tuple[List[str], numpy.ndarray]:
    '''read_phylip_alignment

    Read phylip alignment file. This function expects the phylip to be a valid alignment,
    meaning that it should contain at least 2 sequences of the same length, including
    gaps.

    Parameters
    ----------
    phylip_file : str
        Path to a phylip file.

    Returns
    -------
    Tuple[List[str], numpy.ndarray]
        Tuple, where the first element is a list of entry names and the second entry
        is a n*m numpy ndarray, where n is the number of entries and m the number of
        sites in the alignment.

    Raises
    ------
    InvalidPhylipAlignmentError
        Raised if header is malformed.
    InvalidPhylipAlignmentError
        Raised if less than 2 entries are present in phylip_file.
    InvalidPhylipAlignmentError
        Raised if number of entries does not match header.
    '''

    names = []
    seqs = []
    with open(phylip_file) as phylip_f:
        # header
        header_str = next(phylip_f)
        try:
            n_entries, n_sites = [int(s) for s in header_str.split()]
        except:
            raise InvalidPhylipAlignmentError('Malformed header.')

        for line in phylip_f:
            _line = re.sub(WHITESPACE_RE, '', line)
            if not _line:
                continue
            l_len = len(_line)
            start = l_len-n_sites
            name = _line[:start]
            seq = _line[start:].upper()

            names.append(name)
            seqs.append(list(seq))

    # check alignment validity
    n_seq = len(seqs)
    if len(seqs) < 2:
        msg = f'Expected at least 2 entries but found only {n_seq}.'
        raise InvalidPhylipAlignmentError(msg)

    if n_seq != n_entries:
        msg = f'Expected {n_entries} entries but found {n_seq} instead.'
        raise InvalidPhylipAlignmentError(msg)

    seqs = numpy.array(seqs)

    return NucleotideAlignment(names, seqs)


class InvalidPhylipMatrixError(Exception):
    '''InvalidPhylipMatrixTypeError
    '''

def read_phylip_distmat(phylip_file: str) -> numpy.ndarray:
    with open(phylip_file) as phylip_f:
        # header
        header_str = next(phylip_f)
        try:
            n_entries = int(header_str.strip())
        except:
            raise InvalidPhylipMatrixError('Malformed header.')

        dist_mat = numpy.zeros((n_entries, n_entries))
        names = []

        raise NotImplementedError('')