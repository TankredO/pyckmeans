''' Weighted Ensemble of Random K-Means

    !!! EXPERIMENTAL !!!
'''

import numpy
from sklearn.cluster import KMeans
from typing import Union
from sklearn.metrics import silhouette_samples

class WECR:
    def __init__(
        self,
        k: Union[int, numpy.ndarray],
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
        gamma: float = 0.5,
    ):
        if isinstance(k, int):
            k = numpy.array(k)

        self.k = k
        self.n_rep = n_rep
        self.p_samp = p_samp
        self.p_feat = p_feat
        self.gamma = gamma

        self.kmeans = None
        self.ks = None
        self.qualities = None
        self.sel_feat = None

    def fit(
        self,
        x: numpy.ndarray,
        must_link: numpy.ndarray,
        must_not_link: numpy.ndarray,
    ):
        self.kmeans = []
        self.ks = []
        self.qualities = []

        n_samp = numpy.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = numpy.ceil(self.p_feat * x.shape[1]).astype(int)

        n_constraints = must_link.shape[0] + must_not_link.shape[0]

        self.sel_feat = numpy.zeros((self.n_rep, n_feat), dtype=int)

        for i in range(self.n_rep):
            samp_idcs = numpy.random.choice(x.shape[0], size=n_samp)
            feat_idcs = numpy.random.choice(x.shape[1], size=n_feat)
            self.sel_feat[i] = feat_idcs

            x_subset = x[samp_idcs][:, feat_idcs]

            k = numpy.random.choice(self.k)
            self.ks.append(k)

            km = KMeans(k)
            km.fit(x_subset)
            self.kmeans.append(km)

            cl = km.predict(x[:, feat_idcs])

            sils = silhouette_samples(x, cl)

            # == clustering-level consistency
            if n_constraints == 0:
                clustering_consistency = 1
            else:
                n_ml_matches = (cl[must_link[:, 0]] == cl[must_link[:, 1]]).sum()
                n_mnl_matches = (cl[must_not_link[:, 0]] != cl[must_not_link[:, 1]]).sum()
                clustering_consistency = (n_ml_matches + n_mnl_matches) / n_constraints
            # print('clustering-level consistency', clustering_consistency)

            # == cluster-level consistencies
            cluster_consistencies = numpy.zeros(k)
            internal_qualities = numpy.zeros(k)
            weights = numpy.zeros(k)
            # print('k:', k)
            for j in range(k):
                cl_idcs = numpy.nonzero(cl == j)[0]
                # print(cl_idcs, type(cl_idcs))

                # == cluster consistencies
                if n_constraints == 0:
                    cluster_consistencies[j] = 1
                    n_cl_contraints = 0
                else:
                    cl_ml = must_link[numpy.isin(must_link, cl_idcs).any(1), :]
                    cl_mnl = must_not_link[numpy.isin(must_not_link, cl_idcs).any(1), :]
                    n_cl_contraints = cl_ml.shape[0] + cl_mnl.shape[0]

                    if n_cl_contraints == 0:
                        cluster_consistencies[j] = 1
                    else:
                        n_cl_ml_matches = (cl[cl_ml[:, 0]] == cl[cl_ml[:, 1]]).sum()
                        n_cl_mnl_matches = (cl[cl_mnl[:, 0]] != cl[cl_mnl[:, 1]]).sum()
                        cluster_consistencies[j] = (
                            n_cl_ml_matches + n_cl_mnl_matches
                        ) / n_cl_contraints

                # == weight by associated constraint set
                expected_n_constraints = (cl_idcs.shape[0] / x.shape[0]) * n_constraints
                if (n_constraints > 0) and (n_cl_contraints < expected_n_constraints):
                    weights[j] = (n_cl_contraints / expected_n_constraints) * self.gamma
                else:
                    weights[j] = (
                        n_cl_contraints - expected_n_constraints
                    ) / (
                        n_constraints - expected_n_constraints
                    )

                # == internal qualities
                internal_qualities[j] = sils[cl_idcs].mean()

            # == cluster qualities
            cluster_qualities = internal_qualities * (
                (1-weights) * clustering_consistency + weights * cluster_consistencies
            )
            self.qualities.append(cluster_qualities)

    def predict(self, x: numpy.ndarray):
        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        for i, (km, w) in enumerate(zip(self.kmeans, self.qualities)):
            cl = km.predict(x[:, self.sel_feat[i]])
            a, b = numpy.meshgrid(cl, cl)
            for j in range(km.n_clusters):
                cmatrix += ((a == j) & (b == j)) * w[j]

        return cmatrix
