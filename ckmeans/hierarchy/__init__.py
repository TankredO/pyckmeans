''' Module for hierarchical clustering.
'''

import numpy
from scipy.cluster import hierarchy

def reorder_linkage_gw(
    linkage: numpy.ndarray,
    dist: numpy.ndarray,
) -> numpy.ndarray:
    '''reorder_linkage_gw

    Reorder linkage matrix using the algorithm described by Gruvaeus & Wainer (1972).

    Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
    The British Psychological Society 25.

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
