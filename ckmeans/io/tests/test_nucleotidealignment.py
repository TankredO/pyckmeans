import pytest

from ckmeans.io import \
    NucleotideAlignment, \
    read_alignment, \
    InvalidAlignmentFileExtensionError, \
    InvalidAlignmentFileFormatError

from test_fasta import prep_fasta_files
from test_phylip import prep_phylip_files

def test_simple(prep_fasta_files, prep_phylip_files):
    na_fa_0 = NucleotideAlignment.from_file(prep_fasta_files[0])
    na_fa_1 = NucleotideAlignment.from_file(prep_fasta_files[1])
    na_fa_2 = NucleotideAlignment.from_file(prep_fasta_files[2], 'fasta')

    print('na_fa_0:', na_fa_0)
    print('na_fa_1:', na_fa_1)
    print('na_fa_2:', na_fa_2)

    na_phy_0 = NucleotideAlignment.from_file(prep_phylip_files[0])
    na_phy_1 = NucleotideAlignment.from_file(prep_phylip_files[1])
    na_phy_2 = NucleotideAlignment.from_file(prep_phylip_files[2], 'phylip')

    print('na_phy_0:', na_phy_0)
    print('na_phy_1:', na_phy_1)
    print('na_phy_2:', na_phy_2)

    with pytest.raises(InvalidAlignmentFileFormatError):
        NucleotideAlignment.from_file(prep_fasta_files[0], 'xyz')
    with pytest.raises(InvalidAlignmentFileExtensionError):
        NucleotideAlignment.from_file('test.png', 'auto')


def test_read_alignment(prep_fasta_files, prep_phylip_files):
    na_fa_0 = read_alignment(prep_fasta_files[0])
    na_fa_1 = read_alignment(prep_fasta_files[1])
    na_fa_2 = read_alignment(prep_fasta_files[2], 'fasta')

    print('na_fa_0:', na_fa_0)
    print('na_fa_1:', na_fa_1)
    print('na_fa_2:', na_fa_2)

    na_phy_0 = read_alignment(prep_phylip_files[0])
    na_phy_1 = read_alignment(prep_phylip_files[1])
    na_phy_2 = read_alignment(prep_phylip_files[2], 'phylip')

    print('na_phy_0:', na_phy_0)
    print('na_phy_1:', na_phy_1)
    print('na_phy_2:', na_phy_2)

    with pytest.raises(InvalidAlignmentFileFormatError):
        read_alignment(prep_fasta_files[0], 'xyz')
    with pytest.raises(InvalidAlignmentFileExtensionError):
        read_alignment('test.png', 'auto')
