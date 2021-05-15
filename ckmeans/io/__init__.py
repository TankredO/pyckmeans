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
    InvalidPhylipAlignmentError
from .fasta import \
    read_fasta_alignment, \
    InvalidFastaAlignmentError
