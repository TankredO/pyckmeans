''' nucleotide_alignment

    Module for the representation of nucleotide alignments.
'''

import os
from typing import Iterable, Tuple

import numpy
import pyckmeans.distance

# Base encoding as used by R package ape.
# See http://ape-package.ird.fr/misc/BitLevelCodingScheme.html
#
# Summary:
# Most significant four bits are base information (A, G, C, T)
# 0b00001000 -> base is known
# 0b00000100 -> gap
# 0b00000010 -> unknown base
BASE_ENCODING = {
    # bases
    'A': 0b10001000, 'a': 0b10001000,
    'G': 0b01001000, 'g': 0b01001000,
    'C': 0b00101000, 't': 0b00101000,
    'T': 0b00011000, 'c': 0b00011000,
    # wobbles
    'R': 0b11000000, 'r': 0b11000000, # A|G
    'M': 0b10100000, 'm': 0b10100000, # A|C
    'W': 0b10010000, 'w': 0b10010000, # A|T
    'S': 0b01100000, 's': 0b01100000, # G|C
    'K': 0b01010000, 'k': 0b01010000, # G|T
    'Y': 0b00110000, 'y': 0b00110000, # C|T
    'V': 0b11100000, 'v': 0b11100000, # A|G|C
    'H': 0b10110000, 'h': 0b10110000, # A|C|T
    'D': 0b11010000, 'd': 0b11010000, # A|G|T
    'B': 0b01110000, 'b': 0b01110000, # G|C|T
    'N': 0b11110000, 'n': 0b11110000, # A|G|C|T
    # gaps
    '-': 0b00000100,
    '~': 0b00000100,
    ' ': 0b00000100,
    # unknown/missing state
    '?': 0b00000010
}
BASE_ENCODING_INVERSE = {
    v:k for k, v in BASE_ENCODING.items() if k.isupper() or k in ('-', '?')
}

class InvalidAlignmentFileExtensionError(Exception):
    '''InvalidAlignmentFileExtensionError'''

class InvalidAlignmentFileFormatError(Exception):
    '''InvalidAlignmentFileFormatError'''

class InvalidAlignmentCharacterError(Exception):
    '''InvalidAlignmentCharacterError'''

class InvalidSeqIORecordsError(Exception):
    '''InvalidSeqIORecordsError'''

class NucleotideAlignment:
    '''NucleotideAlignment

    Class for nucleotide alignments.

    Parameters
    ----------
    names : List[str]
        Sequence identifiers/names.
    sequences : numpy.ndarray
        n*m alignment matrix, where n is the number of entries and m
        is the number of sites.
    '''

    def __init__(self, names: Iterable[str], sequences: numpy.ndarray):
        # check validity
        n_names = len(names)
        n_seqs = sequences.shape[0]
        if n_names != n_seqs:
            msg = f'Number of names ({n_names}) does not match number of sequences ({n_seqs}).'
            raise Exception(msg)
        self.names = numpy.array(names)

        # encode strings as uint8, see BASE_ENCODING
        if sequences.dtype != numpy.uint8:
            try:
                self.sequences = numpy.array(
                    [[BASE_ENCODING[n] for n in row] for row in sequences],
                    dtype=numpy.uint8,
                )
            except KeyError as k_err:
                msg = f'Encountered unknown character in alignment: {str(k_err)}'
                raise InvalidAlignmentCharacterError(msg) from k_err
        else:
            self.sequences = sequences

    def drop_invariant_sites(self, in_place: bool = False) -> 'NucleotideAlignment':
        '''drop_invariant_sites

        Remove invariant sites from alignment. Invariant sites
        are sites, where each entry has the same symbol.

        Parameters
        ----------
        in_place : bool, optional
            Modify self in place, by default False

        Returns
        -------
        NucleotideAlignment
            NucleotideAlignment without invariant sites.
            If in_place is set to True, self is returned.
        '''
        if in_place:
            self.sequences = self.sequences[
                :,
                ~numpy.all((self.sequences == self.sequences[0,]), axis=0)
            ]
            return self
        else:
            return NucleotideAlignment(
                self.names.copy(),
                self.sequences[
                    :, ~numpy.all((self.sequences == self.sequences[0,]), axis=0)
                ].copy(),
            )

    def copy(self) -> 'NucleotideAlignment':
        '''copy

        Return a copy of the NucleotideAligment object.

        Returns
        -------
        NucleotideAlignment
            Copy of self.
        '''
        return NucleotideAlignment(self.names.copy(), self.sequences.copy())

    def distance(
        self,
        distance_type: str = 'p',
        pairwise_deletion: bool = True,
    ) -> 'pyckmeans.distance.DistanceMatrix':
        '''distance

        Calculate genetic distance.

        Parameters
        ----------
        distance_type : str, optional
            Type of genetic distance to calculate, by default 'p'.
            Available distance types are p-distances ('p'),
            Jukes-Cantor distances ('jc'), and Kimura 2-paramater distances
            ('k2p').
        pairwise_deletion : bool
            Use pairwise deletion as action to deal with missing data.
            If False, complete deletion is applied.
            Gaps ("-", "~", " "), "?", and ambiguous bases are treated as
            missing data.
        Returns
        -------
        pyckmeans.distance.DistanceMatrix
            n*n distance matrix.
        '''

        return pyckmeans.distance.alignment_distance(
            alignment=self,
            distance_type=distance_type,
            pairwise_deletion=pairwise_deletion,
        )

    @property
    def shape(self) -> Tuple[int, int]:
        '''shape

        Get alignment dimensions/shapes.

        Returns
        -------
        Tuple[int, int]
            Number of samples n, number of sites m
        '''
        return self.sequences.shape

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return NucleotideAlignment(self.names[idx[0]], self.sequences[idx])
        else:
            return NucleotideAlignment(self.names[idx], self.sequences[idx])

    def __repr__(self) -> str:
        '''__repr__

        Returns
        -------
        str
            String representation
        '''
        shape = self.shape
        return f'<NucleotideAlignment; #samples: {shape[0]}, #sites: {shape[1]}>'

    @classmethod
    def from_bp_seqio_records(
        cls,
        records: Iterable['Bio.SeqRecord.SeqRecord'],
    ) -> 'NucleotideAlignment':
        '''from_bp_seqio_records

        Build NucleotideAlignment from iterable of Bio.SeqRecord.SeqRecord.
        Such an iterable is, for example, returned by Bio.SeqIO.parse() or
        can be constructed using Bio.Align.MultipleSequenceAlignment().

        Returns
        -------
        NucleotideAlignment
            NucleotideAlignment object.

        Raises
        ------
        InvalidSeqIORecordsError
            Raised of sequences have different lengths.
        '''
        names = []
        seqs = []

        for record in records:
            names.append(record.id)
            seqs.append(list(record.seq))

        # check if all sequences have same length
        seq_len = len(seqs[0])
        for i, seq in enumerate(seqs[1:]):
            cur_seq_len = len(seq)
            if cur_seq_len != seq_len:
                msg = f'Expected all sequences to have length {seq_len}' +\
                    f'(length of sequence #0) but sequence #{i+1} has length {cur_seq_len}.'
                raise InvalidSeqIORecordsError(msg)

        seqs = numpy.array(seqs)
        names = numpy.array(names)

        return cls(names, seqs)

    @staticmethod
    def from_file(file_path: str, file_format='auto') -> 'NucleotideAlignment':
        '''from_file

        Read nucleotide alignment from file.

        Parameters
        ----------
        file_path: str
            Path to alignment file.
        file_format: str
            Alignment file format. Either "auto", "fasta" or "phylip".
            When "auto" the file format will be inferred based on the file extension.

        Returns
        -------
        NucleotideAlignment
            NucleotideAlignment instance.

        Raises
        ------
        InvalidAlignmentFileExtensionError
            Raised if file_format is "auto" and the file extension is not understood.
        InvalidAlignmentFileFormatError
            Raised if an invalid file_format is passed.
        '''
        if file_format == 'auto':
            ext = os.path.splitext(file_path)[1].lower()

            if ext in ['.fasta', '.fas', '.fa']:
                file_format = 'fasta'
            elif ext in ['.phylip', '.phy']:
                file_format = 'phylip'
            else:
                msg = f'Unknown alignment file extension "{ext}". Please set file_format manually.'
                raise InvalidAlignmentFileExtensionError(msg)

        if file_format in ['fasta', 'FASTA']:
            from .fasta import read_fasta_alignment

            return read_fasta_alignment(file_path)

        elif file_format in ['phylip', 'PHYLIP']:
            from .phylip import read_phylip_alignment

            return read_phylip_alignment(file_path)

        else:
            msg = f'Unknown aligment file format "{file_format}". ' +\
                'Supported formats are "fasta" and "phylip".'
            raise InvalidAlignmentFileFormatError(msg)

def read_alignment(file_path: str, file_format: str = 'auto') -> NucleotideAlignment:
    '''read_alignment

    Read nucleotide alignment from file.
    Alias for NucleotideAlignment.from_file.

    Parameters
    ----------
    file_path: str
        Path to alignment file.
    file_format: str
        Alignment file format. Either "auto", "fasta" or "phylip".
        When "auto" the file format will be inferred based on the file extension.

    Returns
    -------
    NucleotideAlignment
        NucleotideAlignment instance.

    Raises
    ------
    InvalidAlignmentFileExtensionError
        Raised if file_format is "auto" and the file extension is not understood.
    InvalidAlignmentFileFormatError
        Raised if an invalid file_format is passed.
    '''

    return NucleotideAlignment.from_file(file_path, file_format)
