import pytest
import numpy as np
import pyckmeans

def test_simple():
    ckm_0 = pyckmeans.CKmeans(2)
    ckm_1 = pyckmeans.CKmeans(np.array(3, dtype=int))
    ckm_2 = pyckmeans.CKmeans(np.array(3, dtype=np.int64))

def test_multi():
    mckm_0 = pyckmeans.MultiCKMeans(np.arange(10))
