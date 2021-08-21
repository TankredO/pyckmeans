import pytest

from sklearn.datasets import make_blobs
from scipy.spatial.distance import squareform, pdist

from pyckmeans.ordering import distance_order, reorder_distance
from pyckmeans.distance import DistanceMatrix

@pytest.fixture(scope='session')
def prepare_distances():
    x0, _ = make_blobs(n_samples=10, n_features=2, centers=2)
    d0_np = squareform(pdist(x0))
    d0_dm = DistanceMatrix(d0_np)

    x1, _ = make_blobs(n_samples=50, n_features=3, centers=3)
    d1_np = squareform(pdist(x1))
    d1_dm = DistanceMatrix(d1_np)

    return (
        d0_np,
        d0_dm,
        d1_np,
        d1_dm,
    )

def test_reorder(prepare_distances):
    d0, d1, d2, d3 = prepare_distances

    d0_o = reorder_distance(d0)
    print('d0_o:', d0_o)
    d1_o = reorder_distance(d1)
    print('d1_o:', d1_o)
    d2_o = reorder_distance(d2)
    print('d2_o:', d2_o)
    d3_o = reorder_distance(d3)
    print('d3_o:', d3_o)

def test_order(prepare_distances):
    d0, d1, d2, d3 = prepare_distances

    o0 = distance_order(d0)
    print('o0:', o0)
    o1 = distance_order(d1)
    print('o1:', o1)
    o2 = distance_order(d2)
    print('o2:', o2)
    o3 = distance_order(d3)
    print('o3:', o3)
