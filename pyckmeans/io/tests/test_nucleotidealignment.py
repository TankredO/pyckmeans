import warnings

from pyckmeans.io.nucleotide_alignment import InvalidAlignmentCharacterError
import numpy
import pytest

from pyckmeans.io import \
    NucleotideAlignment, \
    read_alignment, \
    InvalidAlignmentFileExtensionError, \
    InvalidAlignmentFileFormatError

from test_fasta import prep_fasta_files
from test_phylip import prep_phylip_files

Bio = None
try:
    import Bio
    from Bio import SeqIO, AlignIO
except:
    warnings.warn('Could not test Biopython since it is not installed.')


def test_simple(prep_fasta_files, prep_phylip_files):
    na_fa_0 = NucleotideAlignment.from_file(prep_fasta_files[0])
    na_fa_1 = NucleotideAlignment.from_file(prep_fasta_files[1])
    na_fa_2 = NucleotideAlignment.from_file(prep_fasta_files[2], 'fasta')
    na_fa_3 = NucleotideAlignment.from_file(prep_fasta_files[2], 'fasta', fast_encoding=True)

    assert (na_fa_2.sequences == na_fa_3.sequences).all()

    print('na_fa_0:', na_fa_0)
    print('na_fa_1:', na_fa_1)
    print('na_fa_2:', na_fa_2)

    na_phy_0 = NucleotideAlignment.from_file(prep_phylip_files[0])
    na_phy_1 = NucleotideAlignment.from_file(prep_phylip_files[1])
    na_phy_2 = NucleotideAlignment.from_file(prep_phylip_files[2], 'phylip')
    na_phy_3 = NucleotideAlignment.from_file(prep_phylip_files[2], 'phylip', fast_encoding=True)

    assert (na_phy_2.sequences == na_phy_3.sequences).all()

    print('na_phy_0:', na_phy_0)
    print('na_phy_1:', na_phy_1)
    print('na_phy_2:', na_phy_2)

    with pytest.raises(InvalidAlignmentFileFormatError):
        NucleotideAlignment.from_file(prep_fasta_files[0], 'xyz')
    with pytest.raises(InvalidAlignmentFileExtensionError):
        NucleotideAlignment.from_file('test.png', 'auto')

    na_phy_0_di = na_phy_0.drop_invariant_sites(in_place=False)
    assert not na_phy_0_di is na_phy_0
    na_phy_0_di = na_phy_0.drop_invariant_sites(in_place=True)
    assert na_phy_0_di is na_phy_0

    na_phy_0_cp = na_phy_0.copy()
    assert (na_phy_0_cp.names == na_phy_0.names).all()
    assert (na_phy_0_cp.sequences == na_phy_0.sequences).all()

    with pytest.raises(Exception):
        NucleotideAlignment(
            ['a', 'b'],
            numpy.array([
                ['A', 'C'],
                ['A', 'T'],
                ['A', 'T']
            ]),
            fast_encoding=False
        )
    
    with pytest.raises(InvalidAlignmentCharacterError):
        NucleotideAlignment(
            ['a', 'b', 'C'],
            numpy.array([
                ['A', '3'],
                ['A', 'T'],
                ['A', 'T']
            ]),
            fast_encoding=False,
        )

    if not Bio is None:
        bio_aln = AlignIO.read(prep_fasta_files[0], format='fasta')
        aln_b = NucleotideAlignment.from_bp_seqio_records(bio_aln)
        print('aln_b:', aln_b)


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

def test_utils():
    na_0 = NucleotideAlignment(
        ['a', 'b', 'c', 'd', 'e'],
        numpy.array([
            ['a', 't', 'a', 't', 't', 'g', 'c'],
            ['a', 'a', '-', 't', 't', 'g', 'c'],
            ['a', 'a', '-', 't', 't', 'g', 'c'],
            ['a', 't', 'a', 't', 'g', 'g', 'c'],
            ['a', 't', 'a', 't', 'g', 'g', 'c'],
        ]),
    )
    na_0_0 = na_0[:2]
    assert na_0_0.shape == (2, na_0.shape[1])
    assert (na_0_0.names == na_0.names[:2]).all()
    assert (na_0_0.sequences == na_0.sequences[:2]).all()

    na_0_1 = na_0[::2]
    assert na_0_1.shape == (3, na_0.shape[1])
    assert (na_0_1.names == na_0.names[::2]).all()
    assert (na_0_1.sequences == na_0.sequences[::2]).all()

    na_0_2 = na_0[:4, :3]
    assert na_0_2.shape == (4, 3)
    assert (na_0_2.names == na_0.names[:4]).all()
    assert (na_0_2.sequences == na_0.sequences[:4, :3]).all()

    assert na_0.drop_invariant_sites().shape == (5, 3)
    