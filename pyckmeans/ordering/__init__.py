''' Module for distance matrix ordering.
'''

from typing import Union
import numpy
from scipy.cluster import hierarchy

import pyckmeans.distance

class InvalidReorderMethod(Exception):
    '''InvalidReorderMethod'''
class InvalidLinkageType(Exception):
    '''InvalidLinkageType'''

REORDER_METHODS = (
    'GW',
    'OLO',
)
LINKAGE_TYPES = (
    'average',
    'complete',
    'single',
    'weighted',
    'centroid',
)


def distance_order(
    dist: Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix'],
    method: str = 'GW',
    linkage_type: str = 'average',
) -> numpy.ndarray:
    '''distance_order

    Get optimal distance matrix order.

    Parameters
    ----------
    dist : Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix']
        A n * n distance matrix as either numpy.ndarray or as
        pyckmeans.distance.DistanceMatrix object.
    method : str
        Reordering method. Either 'GW' (Gruvaeus & Wainer, 1972) or 'OLO' for
        scipy.hierarchy.optimal_leaf_ordering.

        Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
        The British Psychological Society 25.
    linkage_type : str
        Linkage type for the hierarchical clustering. One of

        * 'average'
        * 'complete'
        * 'single'
        * 'weighted'
        * 'centroid'

        See scipy.cluster.hierarchy.linkage for details.

    Returns
    -------
    numpy.ndarray
        Optimal order as vector.

    Raises
    ------
    InvalidReorderMethod
        Raised if an unknown reordering method is passed.
    InvalidLinkageType
        Raised if an unknown linakage type is passed.
    '''

    method = method.upper()
    if method not in REORDER_METHODS:
        msg = f'"{method}" is not a valid reordering method. Available ' +\
            f'methods are {REORDER_METHODS}.'
        raise InvalidReorderMethod(msg)

    linkage_type = linkage_type.lower()
    if linkage_type not in LINKAGE_TYPES:
        msg = f'"{linkage_type}" is not a valid linkage type. Available ' +\
            f'types are {LINKAGE_TYPES}.'
        raise InvalidLinkageType(msg)

    is_ndarray = isinstance(dist, numpy.ndarray)
    if is_ndarray:
        dist_mat = dist
    else:
        dist_mat = dist.dist_mat

    dist_mat_cond = condensed_form(dist_mat)
    linkage_mat = hierarchy.linkage(dist_mat_cond, method=linkage_type)
    if method == 'OLO':
        linkage_mat = hierarchy.optimal_leaf_ordering(linkage_mat, dist_mat_cond)
    elif method == 'GW':
        linkage_mat = reorder_linkage_gw(linkage_mat, dist_mat)

    order = hierarchy.leaves_list(linkage_mat)
    dist_mat = dist_mat[order, :][:, order]

    return order

# This function duplicates code from distance_order, but I
# want to keep this duplication for now for more flexibility
def reorder_distance(
    dist: Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix'],
    method: str = 'GW',
    linkage_type: str = 'average',
) -> Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix']:
    '''reorder_distance

    Reorder distance matrix using hierarchical clustering.

    Parameters
    ----------
    dist : Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix']
        A n * n distance matrix as either numpy.ndarray or as
        pyckmeans.distance.DistanceMatrix object.
    method : str
        Reordering method. Either 'GW' (Gruvaeus & Wainer, 1972) or 'OLO' for
        scipy.hierarchy.optimal_leaf_ordering.

        Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
        The British Psychological Society 25.
    linkage_type : str
        Linkage type for the hierarchical clustering. One of

        * 'average'
        * 'complete'
        * 'single'
        * 'weighted'
        * 'centroid'

        See scipy.cluster.hierarchy.linkage for details.

    Returns
    -------
    Union[numpy.ndarray, 'pyckmeans.distance.DistanceMatrix']
        The sorted distance matrix as either numpy.ndarray or
        pyckmeans.distance.DistanceMatrix, depending on the input.

    Raises
    ------
    InvalidReorderMethod
        Raised if an unknown reordering method is passed.
    InvalidLinkageType
        Raised if an unknown linakage type is passed.
    '''
    method = method.upper()
    if method not in REORDER_METHODS:
        msg = f'"{method}" is not a valid reordering method. Available ' +\
            f'methods are {REORDER_METHODS}.'
        raise InvalidReorderMethod(msg)

    linkage_type = linkage_type.lower()
    if linkage_type not in LINKAGE_TYPES:
        msg = f'"{linkage_type}" is not a valid linkage type. Available ' +\
            f'types are {LINKAGE_TYPES}.'
        raise InvalidLinkageType(msg)

    is_ndarray = isinstance(dist, numpy.ndarray)
    if is_ndarray:
        dist_mat = dist
    else:
        dist_mat = dist.dist_mat

    dist_mat_cond = condensed_form(dist_mat)
    linkage_mat = hierarchy.linkage(dist_mat_cond, method=linkage_type)
    if method == 'OLO':
        linkage_mat = hierarchy.optimal_leaf_ordering(linkage_mat, dist_mat_cond)
    elif method == 'GW':
        linkage_mat = reorder_linkage_gw(linkage_mat, dist_mat)

    order = hierarchy.leaves_list(linkage_mat)
    dist_mat = dist_mat[order, :][:, order]

    if is_ndarray:
        return dist_mat
    else:
        return pyckmeans.distance.DistanceMatrix(
            dist_mat,
            dist.names[order] if not dist.names is None else None,
        )

def condensed_form(dist: numpy.ndarray) -> numpy.ndarray:
    '''condensed_form

    Convert n*n distance matrix to condensed vector form.

    Parameters
    ----------
    dist : numpy.ndarray
        n * n distance matrix.

    Returns
    -------
    numpy.ndarray
        Distance matrix in condensed vector form as expected by
        scipy.cluster.hierarchy.linkage.
    '''

    return dist[numpy.triu_indices_from(dist, k=1)]

def reorder_linkage_gw(
    linkage: numpy.ndarray,
    dist: numpy.ndarray,
) -> numpy.ndarray:
    '''reorder_linkage_gw

    Reorder linkage matrix using the algorithm described by Gruvaeus & Wainer (1972) [1]_.

    Parameters
    ----------
    linkage : numpy.ndarray
        Linkage matrix as returned from scipy.cluster.hierarchy.linkage.
    dist : numpy.ndarray
        n * n distance matrix.

    Returns
    -------
    numpy.ndarray
        Reordered linkage matrix.

    References
    ----------
    .. [1]  Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
            The British Psychological Society 25.
    '''
    linkage = linkage.copy()

    n = linkage.shape[0]

    # left and right leaves of a cluster
    l_r = numpy.zeros((n, 2))
    # matrix determining, whether a cluster (subtree) should be flipped
    flip = numpy.full((n, 2), False)

    # find left and right leaves of clusters
    # and determine, whether cluster should
    # be flipped
    for i in range(n):
        l, r = linkage[i, [0, 1]].astype(int)

        # l and r are singletons
        if l <= n and r <= n:
            l_r[i] = (l, r)
        # only l is a singleton
        elif l <= n:
            l_r[i, 0] = l

            # left and right leaves of cluster r
            rl, rr = l_r[r - (n + 1)].astype(int)

            if dist[l, rl] < dist[l, rr]:
                l_r[i, 1] = rr
            else:
                l_r[i, 1] = rl
                flip[i, 1] = True
        # only r is singleton
        elif r <= n:
            l_r[i, 1] = r

            # left and right leaves of cluster l
            ll, lr = l_r[l - (n + 1)].astype(int)

            if dist[r, ll] < dist[r, lr]:
                l_r[i, 0] = lr
                flip[i, 0] = True
            else:
                l_r[i, 0] = ll
        # none of l and r are singletons
        else:
            # left and right leaves
            ll, lr = l_r[l - (n + 1)].astype(int)
            rl, rr = l_r[r - (n + 1)].astype(int)

            d_ll_rl = dist[ll, rl] # 0
            d_ll_rr = dist[ll, rr] # 1
            d_lr_rl = dist[lr, rl] # 2
            d_lr_rr = dist[lr, rr] # 3

            mn_idx = numpy.argmin([d_ll_rl, d_ll_rr, d_lr_rl, d_lr_rr])
            if mn_idx == 0: # d_ll_rl
                l_r[i] = (lr, rr)
                flip[i, 0] = True
            elif mn_idx == 1: # d_ll_rr
                l_r[i] = (lr, rl)
                flip[i] = (True, True)
            elif mn_idx == 2: # d_lr_rl
                l_r[i] = (ll, rr)
            else: # d_lr_rr
                l_r[i] = (ll, rl)
                flip[i, 1] = True

    # apply flip
    for i in range((n-1), 0, -1):
        if flip[i, 0]:
            c = linkage[i, 0].astype(int)
            # non-singleton cluster
            if c > n:
                c = c - (n + 1)
                linkage[c, [0, 1]] = linkage[c, [1, 0]]
                if flip[c, 0] == flip[c, 1]:
                    flip[c] = ~flip[c]
        if flip[i, 1]:
            c = linkage[i, 1].astype(int)
            if c > n:
                c = c - (n + 1)
                linkage[c, [0, 1]] = linkage[c, [1, 0]]
                if flip[c, 0] == flip[c, 1]:
                    flip[c] = ~flip[c]

    return linkage
