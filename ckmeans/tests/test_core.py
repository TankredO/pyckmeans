import pytest
import numpy as np
import ckmeans

def test_simple():
    ckm_0 = ckmeans.CKmeans(2)
    ckm_1 = ckmeans.CKmeans(np.array(3, dtype=int))
    ckm_2 = ckmeans.CKmeans(np.array(3, dtype=np.int64))

