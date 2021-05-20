''' csv

    Comma Separated Value (CSV) input and output.
'''

import pandas

import ckmeans.distance

def read_csv_distmat(
    file_path: str,
    header: bool = True,
    row_names: bool = True,
    sep: str = ',',
) -> 'ckmeans.distance.DistanceMatrix':
    pass

def write_csv_distmat(
    dist: 'ckmeans.distance.DistanceMatrix',
    file_path: str,
    force: bool = False,
) -> None:
    pass
