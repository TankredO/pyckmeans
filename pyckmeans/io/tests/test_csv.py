from numpy.testing._private.utils import assert_raises
import pytest
import tempfile
import os

import numpy as np
import pandas as pd

from pyckmeans.io.csv import read_csv_distmat, write_csv_distmat

d_0 = np.array([
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 3.0],
    [2.0, 3.0, 0.0],
])
nm_0 = ['a', 'b', 'c']
df_0 = pd.DataFrame(d_0, columns=nm_0, index=nm_0)

d_1 = np.array([
    [0.0, 1.0, 2.0, 0.5],
    [1.0, 0.0, 3.0, 2.4],
    [2.0, 3.0, 0.0, 1.5],
    [0.5, 2.4, 1.5, 0.0]
])
nm_1 = ['a', 'b', 'c', 'd']
df_1 = pd.DataFrame(d_1, columns=nm_1, index=nm_1)

@pytest.fixture(scope='session')
def prep_csv_files():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        csv_file_0 = os.path.join(tempdir, 'dist_0.csv')
        df_0.to_csv(csv_file_0)

        csv_file_1 = os.path.join(tempdir, 'dist_1.csv')
        df_1.to_csv(csv_file_1, index=None)

        csv_file_2 = os.path.join(tempdir, 'dist_2.csv')
        df_0.to_csv(csv_file_2, header=None)

        csv_file_3 = os.path.join(tempdir, 'dist_3.csv')
        df_0.to_csv(csv_file_3, index=None, header=None)

        yield (
            # should work
            (csv_file_0, d_0, nm_0),
            (csv_file_1, d_1, nm_1),
            (csv_file_2, d_0, nm_0),
            (csv_file_3, d_0, None),
        )

        print(f'Deleted temporary directory {tempdir}.')

@pytest.fixture(scope='session')
def prep_outdir():
    with tempfile.TemporaryDirectory() as tempdir:
        print(f'Created temporary directory {tempdir}.')

        yield tempdir

        print(f'Deleted temporary directory {tempdir}.')

def test_csv(prep_csv_files, prep_outdir):
    eps = 0.00001
    
    csv_f_0, d_0_expected, nm_0_expected = prep_csv_files[0]
    dm_0 = read_csv_distmat(csv_f_0, 0, 0, ',')
    assert np.max(np.abs(dm_0.dist_mat - d_0_expected)) < eps
    assert all([a == b for a, b in zip(dm_0.names, nm_0_expected)])
    csv_of_0 = os.path.join(prep_outdir, 'dist_0.csv')
    write_csv_distmat(dm_0, csv_of_0)
    dm_0_r = read_csv_distmat(csv_of_0, 0, 0, ',')
    assert np.max(np.abs(dm_0.dist_mat - dm_0_r.dist_mat)) < eps
    assert all([a == b for a, b in zip(dm_0.names, dm_0_r.names)])

    csv_f_1, d_1_expected, nm_1_expected = prep_csv_files[1]
    dm_1 = read_csv_distmat(csv_f_1, 0, None, ',')
    assert np.max(np.abs(dm_1.dist_mat - d_1_expected)) < eps
    assert all([a == b for a, b in zip(dm_1.names, nm_1_expected)])
    csv_of_1 = os.path.join(prep_outdir, 'dist_1.csv')
    write_csv_distmat(dm_1, csv_of_1)
    dm_1_r = read_csv_distmat(csv_of_1, 0, 0, ',')
    assert np.max(np.abs(dm_1.dist_mat - dm_1_r.dist_mat)) < eps
    assert all([a == b for a, b in zip(dm_1.names, dm_1_r.names)])

    csv_f_2, d_2_expected, nm_2_expected = prep_csv_files[2]
    dm_2 = read_csv_distmat(csv_f_2, None, 0, ',')
    assert np.max(np.abs(dm_2.dist_mat - d_2_expected)) < eps
    assert all([a == b for a, b in zip(dm_2.names, nm_2_expected)])

    csv_f_3, d_3_expected, nm_3_expected = prep_csv_files[3]
    dm_3 = read_csv_distmat(csv_f_3, None, None, ',')
    assert np.max(np.abs(dm_3.dist_mat - d_3_expected)) < eps
    assert dm_3.names == nm_3_expected
    csv_of_3 = os.path.join(prep_outdir, 'dist_3.csv')
    write_csv_distmat(dm_3, csv_of_3)
    dm_3_r = read_csv_distmat(csv_of_3)
    assert np.max(np.abs(dm_3.dist_mat - dm_3_r.dist_mat)) < eps
