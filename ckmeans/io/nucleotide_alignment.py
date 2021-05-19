''' nucleotide_alignment

    Module for the representation of nucleotide alignments.
'''

import os
from typing import List, Tuple

import numpy


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
    'A': 0b10001000,
    'G': 0b01001000,
    'C': 0b00101000,
    'T': 0b00011000,
    # wobbles
    'R': 0b11000000, # A|G
    'M': 0b10100000, # A|C
    'W': 0b10010000, # A|T
    'S': 0b01100000, # G|C
    'K': 0b01010000, # G|T
    'Y': 0b00110000, # C|T
    'V': 0b11100000, # A|G|C
    'H': 0b10110000, # A|C|T
    'D': 0b11010000, # A|G|T
    'B': 0b01110000, # G|C|T
    'N': 0b11110000, # A|G|C|T
    # gaps
    '-': 0b00000100,
    '~': 0b00000100,
    ' ': 0b00000100,
    # unknown/missing state
    '?': 0b00000010
}
BASE_ENCODING_INVERSE = {
    v:k for k, v in BASE_ENCODING.items() if k.isupper() or k == '-'
}

class InvalidAlignmentFileExtensionError(Exception):
    '''InvalidAlignmentFileExtensionError'''

class InvalidAlignmentFileFormatError(Exception):
    '''InvalidAlignmentFileFormatError'''

class InvalidAlignmentCharacterError(Exception):
    '''InvalidAlignmentCharacterError'''

class NucleotideAlignment:
    '''NucleotideAlignment

    Class for representing nucleotide alignments

    Parameters
    ----------
    names : List[str]
        Sequence identifiers/names.
    sequences : numpy.ndarray
        n*m alignment matrix, where n is the number of entries and m
        is the number of sites.
    '''

    def __init__(self, names: List[str], sequences: numpy.ndarray):
        self.names = names

        # check validity
        n_names = len(names)
        n_seqs = sequences.shape[0]
        if n_names != n_seqs:
            msg = f'Number of names ({n_names}) does not match number of sequences ({n_seqs}).'
            raise Exception(msg)

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

    def __repr__(self) -> str:
        '''__repr__

        Returns
        -------
        str
            String representation
        '''
        shape = self.shape
        return f'<NucleotideAlignment; #samples: {shape[0]}, #sites: {shape[1]}>'

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
