''' Weighted Ensemble Consensus of Random K-Means (WECR K-Means)
'''

import numpy
import pandas
from sklearn.cluster import KMeans
from typing import Union, Optional, Iterable, Callable, Tuple, Dict, Any
from sklearn.metrics import silhouette_samples

# adapted from:
# https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
# This is equivalent to the loop:
# sim_mat_norm = numpy.zeros_like(sim_mat)
# for i in range(sim_mat.shape[0]):
#     for j in range(sim_mat.shape[1]):
#         sim_mat_norm[i,j] = sim_mat[i,j] / numpy.sqrt(sim_mat[i,i] * sim_mat[j,j])
def _normalize_similarity_matrix(
    sim_mat: numpy.ndarray,
) -> numpy.ndarray:
    self_sim = numpy.sqrt(numpy.diag(sim_mat))
    outer_self_sim = numpy.outer(self_sim, self_sim)
    sim_mat_norm = sim_mat / outer_self_sim
    sim_mat_norm[sim_mat == 0] = 0

    return sim_mat_norm

def _prepare_constraints(
    must_link: numpy.ndarray,
    must_not_link: numpy.ndarray,
    names: Union[None, numpy.ndarray],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    if must_link.dtype.type == numpy.dtype(str) or must_not_link.dtype.type == numpy.dtype(str):
        if names is None:
            msg = 'Constraints (must_link, must_not_link) can only contain character strings, ' +\
                'if names can be inferred from x (pandas.DataFrame, pyckmeans.ordination.PCOAResult).'
            raise Exception(msg)

    if must_link.dtype.type == numpy.dtype(str):
        ml = numpy.zeros_like(must_link, dtype=int)
        for i, (a, b) in enumerate(must_link):
            a_idcs = numpy.argwhere(a == names)
            b_idcs = numpy.argwhere(b == names)

            if (len(a_idcs)) < 1:
                msg = f'must_link: Could not find row with name "{a}" in x.'
                raise Exception(msg)
            if (len(b_idcs)) < 1:
                msg = f'must_link: Could not find row with name "{b}" in x.'
                raise Exception(msg)
            ml[i] = numpy.array([a_idcs[0][0], b_idcs[0][0]])
    else:
        ml = must_link

    if must_not_link.dtype.type == numpy.dtype(str):
        mnl = numpy.zeros_like(must_not_link, dtype=int)
        for i, (a, b) in enumerate(must_not_link):
            a_idcs = numpy.argwhere(a == names)
            b_idcs = numpy.argwhere(b == names)

            if (len(a_idcs)) < 1:
                msg = f'must_not_link: Could not find row with name "{a}" in x.'
                raise Exception(msg)
            if (len(b_idcs)) < 1:
                msg = f'must_not_link: Could not find row with name "{b}" in x.'
                raise Exception(msg)
            mnl[i] = numpy.array([a_idcs[0][0], b_idcs[0][0]])
    else:
        mnl = must_not_link

    return ml, mnl

class WECR:
    def __init__(
        self,
        k: Union[int, Iterable[int]],
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
        **kwargs: Dict[str, Any],
    ):
        self.k = numpy.array(k).reshape(-1)
        self.n_rep = n_rep
        self.p_samp = p_samp
        self.p_feat = p_feat
        self._kmeans_kwargs = dict(
            n_init=2,
        )
        self._kmeans_kwargs.update(kwargs)

        self.kmeans = None
        self.sel_feat = None
        self.ks = None

        self.must_link = None
        self.ml = None
        self.must_not_link = None
        self.mnl = None

    def fit(
        self,
        x: Union[numpy.ndarray, pandas.DataFrame],
        progress_callback: Optional[Callable] = None,
    ):
        self._reset()

        if isinstance(x, pandas.DataFrame):
            x = x.values

        # == Fit K-Means
        n_samp = numpy.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = numpy.ceil(self.p_feat * x.shape[1]).astype(int)

        self.sel_feat = numpy.zeros((self.n_rep, n_feat), dtype=int)

        self.ks = numpy.zeros(self.n_rep, dtype=int)
        self.kmeans = []
        for i in range(self.n_rep):
            samp_idcs = numpy.random.choice(x.shape[0], size=n_samp)
            feat_idcs = numpy.random.choice(x.shape[1], size=n_feat)
            self.sel_feat[i] = feat_idcs
            x_subset = x[samp_idcs][:, feat_idcs]

            k = numpy.random.choice(self.k, 1)[0]
            km = KMeans(
                n_clusters=k,
                **self._kmeans_kwargs
            )
            km.fit(x_subset)
            self.kmeans.append(km)
            self.ks[i] = k

            if progress_callback:
                progress_callback()

    def predict(
        self,
        x: Union[numpy.ndarray, pandas.DataFrame],
        must_link: Optional[Iterable] = None,
        must_not_link: Optional[Iterable] = None,
        gamma: float = 0.5,
        progress_callback: Optional[Callable] = None,
    ) -> numpy.ndarray:
        names = None
        if isinstance(x, pandas.DataFrame):
            names = x.index
            x = x.values

        # == prepare constraints
        must_link = numpy.array(must_link) if must_link is not None else numpy.zeros((0,2))
        must_not_link = numpy.array(must_not_link) if must_not_link is not None else numpy.zeros((0,2))

        # In the following, the comments will try to map the code as close as possible
        # to the notations given in Lai et al. (2019, "An Adaptive Robust Semi-supervised
        # Clustering Framework Using Weighted Consensus of Random k-Means Ensemble").
        # In contrast to the paper, the consensus matrix (S) is updated after each
        # K-Means, instead of forming the two big matrices H and W, and calculting
        # the membership matrix in one go. This was done to save memory.

        ml, mnl = _prepare_constraints(must_link, must_not_link, names)
        # Mu: ml, must link as numpy array of sample indices [[a0, b0], [a1, b1]]
        # C: mnl, must not link as numpy array of sample indices [[a0, b0], [a1, b1]]

        self.must_link = must_link
        self.ml = ml
        self.must_not_link = must_not_link
        self.mnl = mnl

        # N: number of samples (total number of instances)
        n = x.shape[0]
        # |M| + |C|: total number of constraints
        n_constraints = ml.shape[0] + mnl.shape[0]

        # == prepare output consensus matrix (co-association matrix)
        cmat = numpy.zeros((x.shape[0], x.shape[0]))

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            k = self.ks[i]
            # H_{i}: cluster membership one-hot encoded (binary membership matrix)
            cl_oh = numpy.eye(k)[cl.reshape(-1)]

            # simga_{i}: scaled (0-1) silhouette scores (internal validation measure)
            sil_scores = (silhouette_samples(x, cl) + 1) / 2

            # If there are no constraints, use internal validation only (silhoutte)
            if n_constraints == 0:
                # pi_{ij}: cluster sizes
                cluster_size = numpy.zeros(k, dtype=int)
                # sigma_{ij}: scaled (0-1) mean cluster silhouette score
                sil = numpy.zeros(k)
                for j in range(k):
                    cluster_size[j] = (cl == j).sum()
                    sil[j] = sil_scores[cl == j].sum() / cluster_size[j]

                # S = HWH^T: weighted consensus matrix (weighted co-association matrix)
                co_assoc = cl_oh.dot(numpy.diag(sil)).dot(cl_oh.T) 

            # If there are constraints, apply standard algorithm
            else:
                # mu_{i}: clustering-level consistency
                # number of fulfilled constraints divided by total number of constraints
                clustering_cons = \
                    (cl[ml[:,0]] == cl[ml[:,1]]).sum() if ml.shape[0] != 0 else 0 +\
                    (cl[mnl[:,0]] != cl[mnl[:,1]]).sum() if mnl.shape[0] != 0 else 0
                clustering_cons /= n_constraints

                # w{ij}: final cluster weight
                weight = numpy.zeros(k)
                for j in range(k):
                    # Mu_{ij}, C_{ij}: associated constraint sets
                    ml_assoc = ml[(cl[ml[:, 0]] == j) | (cl[ml[:, 1]] == j)] if ml.shape[0] > 0 else numpy.zeros((0,2))
                    mnl_assoc = mnl[(cl[mnl[:, 0]] == j) | (cl[mnl[:, 1]] == j)] if mnl.shape[0] > 0 else numpy.zeros((0,2))
                    # p{ij} = |Mu_{ij}| + |C_{ij}|: number of associated constraints
                    n_constraints_assoc = ml_assoc.shape[0] + mnl_assoc.shape[0]

                    # nu_{ij}: cluster-level consistency 
                    if n_constraints_assoc == 0:
                        cluster_cons = 1.0
                    else:
                        cluster_cons = (
                            (cl[ml_assoc[:,0]] == cl[ml_assoc[:,1]]).sum() if ml_assoc.shape[0] > 0 else 0 +\
                            (cl[mnl_assoc[:,0]] != cl[mnl_assoc[:,1]]).sum() if mnl_assoc.shape[0] > 0 else 0
                        ) / n_constraints_assoc

                    # pi_{ij}: cluster sizes
                    cluster_size = (cl == j).sum()

                    # E(p_{ij}): expected size of the associated constraints set
                    n_exp_constraints_assoc = cluster_size / n * n_constraints

                    # g_{gamma}(pi_{ij}): cluster weight
                    # 0 < p{ij} < E(p{ij})
                    if n_constraints_assoc > 0 and n_constraints_assoc < n_exp_constraints_assoc:
                        weight_cons = n_constraints_assoc / n_exp_constraints_assoc * gamma
                    else:
                        weight_cons = \
                            (n_constraints_assoc - n_exp_constraints_assoc) /\
                            (n_constraints - n_exp_constraints_assoc) *\
                            (1 - gamma) + gamma

                    # phi(pi{ij}): linear combination function value
                    consistency = (1-weight_cons) * clustering_cons + weight_cons * cluster_cons

                    # sigma_{ij}: scaled (0-1) mean silhouette score
                    sil = sil_scores[cl == j].sum() / cluster_size

                    # w{ij}: final cluster weight
                    weight[j] = sil * consistency

                # S_{i} = H_{i}W_{i}H_{i}^T: weighted consensus matrix (weighted co-association matrix)
                co_assoc =  cl_oh.dot(numpy.diag(weight)).dot(cl_oh.T)

            cmat += co_assoc

            if progress_callback:
                progress_callback()

        cmat /= self.n_rep

        return cmat

    def _reset(self):
        self.kmeans = None
        self.sel_feat = None
        self.ks = None
        self.must_link = None
        self.ml = None
        self.must_not_link = None
        self.mnl = None
