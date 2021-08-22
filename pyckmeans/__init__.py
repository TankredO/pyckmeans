''' pyckmeans

    pyckmeans, a Python package for Consensus K-Means clustering.
'''

__version__ = '0.0.4'

__all__ = [
    'CKmeans',
    'MultiCKMeans',
    'plot_ckmeans_result',
    'plot_multickmeans_metrics',
]

from .core import CKmeans, MultiCKMeans
from .utils import plot_ckmeans_result, plot_multickmeans_metrics
