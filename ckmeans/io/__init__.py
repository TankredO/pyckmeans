''' io

    Module containing input and output functionality.
'''

from .nucleotide_alignment import \
    NucleotideAlignment, \
    read_alignment, \
    InvalidAlignmentFileExtensionError, \
    InvalidAlignmentFileFormatError
from .phylip import \
    read_phylip_alignment, \
    InvalidPhylipAlignmentError, \
    read_phylip_distmat, \
    InvalidPhylipMatrixError
from .fasta import \
    read_fasta_alignment, \
    InvalidFastaAlignmentError
from .csv import \
    read_csv_distmat
