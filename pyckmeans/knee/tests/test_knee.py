import pytest
import numpy

from pyckmeans.knee import KneeLocator

@pytest.mark.parametrize('direction', ['increasing', 'decreasing'])
@pytest.mark.parametrize('curve', ['convex', 'concave'])
def test_simple(direction, curve):
    x = numpy.array([1.0, 2.0, 3.0 ,4.0, 5.0, 6.0,  7.0,  8.0,  9.0 ])
    y = numpy.array([1.0, 2.2, 3.4, 4.5, 7.0, 10.0, 15.0, 22.0, 30.0])

    kl_0 = KneeLocator(x, y, curve=curve, direction=direction, interp_method='interp1d')
    print('kl_0.knee:', kl_0.knee)
    print('kl_0.elbow:', kl_0.elbow)
    print('kl_0.norm_elbow:', kl_0.norm_elbow)
    print('kl_0.elbow_y:', kl_0.elbow_y)
    print('kl_0.norm_elbow_y:', kl_0.norm_elbow_y)
    print('kl_0.all_elbows:', kl_0.all_elbows)
    print('kl_0.all_norm_elbows:', kl_0.all_norm_elbows)
    print('kl_0.all_elbows_y:', kl_0.all_elbows_y)
    print('kl_0.all_norm_elbows_y:', kl_0.all_norm_elbows_y)

    kl_1 = KneeLocator(x, y, curve=curve, direction=direction, interp_method='polynomial')

    with pytest.raises(ValueError):
        KneeLocator(x, y, curve=curve, direction=direction, interp_method='XYZ')
