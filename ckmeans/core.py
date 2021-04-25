''' core

    Module for ckmeans core functionality.
'''

import multiprocessing

import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CKmeans:
    def __init__(
        self,
        k: int,
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
    ):
        self.k = k
        self.n_rep = n_rep
        self.p_samp = p_samp
        self.p_feat = p_feat

        self.centers = None
        self.kmeans = None
        self.sel_feat = None
        self.sils = None

    def fit(self, x: numpy.ndarray, n_jobs: int = 1):
        if n_jobs < 2:
            self._fit(x)
        else:
            self._fit_parallel(x, n_jobs)

    def predict(self, x: numpy.ndarray):
        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            a, b = numpy.meshgrid(cl, cl)
            cmatrix += a == b

        return cmatrix / self.n_rep

    def _fit(self, x: numpy.ndarray):
        self.kmeans = []
        self.sils = numpy.zeros(self.n_rep)

        n_samp = numpy.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = numpy.ceil(self.p_feat * x.shape[1]).astype(int)

        self.sel_feat = numpy.zeros((self.n_rep, n_feat), dtype=int)
        self.centers = numpy.zeros((self.n_rep, self.k, n_feat))

        for i in range(self.n_rep):
            samp_idcs = numpy.random.choice(x.shape[0], size = n_samp)
            feat_idcs = numpy.random.choice(x.shape[1], size = n_feat)
            self.sel_feat[i] = feat_idcs

            x_subset = x[samp_idcs][:, feat_idcs]

            km = KMeans(self.k)
            km.fit(x_subset)
            self.kmeans.append(km)
            self.centers[i] = km.cluster_centers_

            self.sils[i] = silhouette_score(x_subset, km.predict(x_subset))

    def _fit_parallel(self, x: numpy.ndarray, n_jobs: int):
        n_samp = numpy.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = numpy.ceil(self.p_feat * x.shape[1]).astype(int)

        raise NotImplementedError()
