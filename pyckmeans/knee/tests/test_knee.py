import pytest
import numpy

from pyckmeans.knee import KneeLocator

def test_simple():
    x = numpy.array([1.0, 2.0, 3.0 ,4.0, 5.0, 6.0,  7.0,  8.0,  9.0 ])
    y = numpy.array([1.0, 2.2, 3.4, 4.5, 7.0, 10.0, 15.0, 22.0, 30.0])

    kl = KneeLocator(x, y, curve='convex', direction='increasing')
    print('kl.norm_knee:', kl.knee)
