''' core

    Module for ckmeans core functionality.
'''

from typing import Callable, Iterable, Optional, Union

import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster import hierarchy

from .ordering import distance_order, condensed_form

# Could get this directly from sklearn.cluster.KMeans.inertia_,
# but will keep this for flexibility. wss results and inertia_ values
# match perfectly.
def wss(
    x: numpy.ndarray,
    centers: numpy.ndarray,
    cl: Iterable[int]
) -> float:
    '''wss

    Calculate within cluster sum of squares.

    Parameters
    ----------
    x : numpy.ndarray
        n * m matrix, where n is the number of samples (observations) and m is
        the number of features (predictors).
    centers : numpy.ndarray
        k * m matrix of cluster centers (centroids), where k is the number of
        clusters and m is the number of features (predictors).
    cl : Iterable[int]
        Iterable of length n, containing cluster membership as coded as integer.

    Returns
    -------
    float
        Within cluster sum of squares.
    '''
    res = 0
    for cur_cl in numpy.unique(cl):
        cur_x = x[cl == cur_cl, ]
        res += ((cur_x - centers[cur_cl, :]) ** 2).sum()

    return res

def bic_kmeans(
    x: numpy.ndarray,
    cl: numpy.ndarray,
    centers: Optional[numpy.ndarray] = None,
) -> float:
    '''bic_kmeans

    Calculate the Bayesian Information Criterion (BIC) for a KMeans result.
    The formula is using the BIC calculation for the Gaussian special case.

    Parameters
    ----------
    x : numpy.ndarray
        n * m matrix, where n is the number of samples (observations) and m is
        the number of features (predictors).
    cl : Iterable[int]
        Iterable of length n, containing cluster membership coded as integer.
    centers : Optional[numpy.ndarray]
        k * m matrix of cluster centers (centroids), where k is the number of
        clusters and m is the number of features (predictors). If None, centers
        will be calculated from cl and x.

    Returns
    -------
    float
        BIC
    '''
    k = len(numpy.unique(cl))
    n = x.shape[0]

    if centers is None:
        centers = numpy.zeros((k, x.shape[1]))
        for i, c in enumerate(numpy.unique(cl)):
            centers[i] = x[cl == c].mean(axis=0)

    rss = wss(x, centers, cl)

    return n * numpy.log(rss/n) + numpy.log(n) * k

class CKmeansResult:
    '''CKmeansResult

    Result of CKmeans.predict.

    Has the members

    * cmatrix: n * n consensus matrix
    * cl: n-length vector of cluster memberships

    Parameters
    ----------
    consensus_matrix : numpy.ndarray
        n * n consensus matrix.
    cluster_membership : numpy.ndarray
        n-length vector of cluster memberships
    '''
    def __init__(
        self,
        consensus_matrix: numpy.ndarray,
        cluster_membership: numpy.ndarray,
    ):

        self.cmatrix = consensus_matrix
        self.cl = cluster_membership

    def order(
        self,
        method: str = 'GW',
        linkage_type: str = 'average',
    ) -> numpy.ndarray:
        '''order

        Get optimal sample order according to hierarchical clustering.

        Parameters
        ----------
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
            Optimal sample order.
        '''

        return distance_order(1 - self.cmatrix, method=method, linkage_type=linkage_type)

    def sort(
        self,
        method: str = 'GW',
        linkage_type: str = 'average',
        in_place: bool = False,
    ) -> 'CKmeansResult':
        '''sort

        Sort CKmeansResult using hierarchical clustering.

        Parameters
        ----------
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
        in_place : bool
            If False, a new, sorted CKmeansResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        CKmeansResult
            Sorted CKmeansResult
        '''

        order = self.order(method=method, linkage_type=linkage_type)

        if in_place:
            ckmres = self
        else:
            ckmres = CKmeansResult(self.cmatrix, self.cl)

        ckmres.cmatrix = ckmres.cmatrix[order, :][:, order]
        ckmres.cl = ckmres.cl[order]

        return ckmres


class InvalidClusteringMetric(Exception):
    '''InvalidClusteringMetric'''

class CKmeans:
    '''CKmeans

    Consensus K-Means.

    Parameters
    ----------
    k : int
        Number of clusters.
    n_rep : int, optional
        Number of K-Means to fit, by default 100
    p_samp : float, optional
        Proportion of samples (observations) to randomly draw per K-Means run, by default 0.8.
        The resulting number of samples will be rounded up. I.e. if number of samples is 10 and
        p_samp is 0.75, each K-Means will use 8 randomly drawn samples (0.72 * 10 = 7.2, 7.2 -> 8).
    p_feat : float, optional
        Proportion of features (predictors) to randomly draw per K-Means run, by default 0.8.
        The resulting number of features will be rounded up. I.e. if number of features is 10 and
        p_feat is 0.72, each K-Means will use 8 randomly drawn features (0.72 * 10 = 7.5, 7.2 -> 8).
    metrics : Iterable[str]
        Clustering quality metrics to calculate. Available metrics are
        * "sil" (Silhouette Index)
        * "bic" (Bayesian Information Criterion)
        * "db" (Davies-Bouldin Index)
        * "ch" (Calinski-Harabasz).
    '''

    AVAILABLE_METRICS = ('sil', 'bic', 'db', 'ch')

    def __init__(
        self,
        k: int,
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
        metrics: Iterable[str] = ('sil', 'bic'),
    ):
        self.k = k
        self.n_rep = n_rep
        self.p_samp = p_samp
        self.p_feat = p_feat

        for metric in metrics:
            if not metric in self.AVAILABLE_METRICS:
                am_str = ", ".join(self.AVAILABLE_METRICS)
                msg = f'Unknown metric "{metric}". Available metrics are {am_str}.'
                raise InvalidClusteringMetric(msg)

        self._metrics = metrics

        self.kmeans = None
        self.centers = None
        self.sel_feat = None
        self.sils = None
        self.bics = None
        self.dbs = None
        self.chs = None

    def fit(
        self,
        x: numpy.ndarray,
        progress_callback: Optional[Callable] = None,
    ):
        '''fit

        Fit CKmeans.

        Parameters
        ----------
        x : numpy.ndarray
            n * m matrix, where n is the number of samples (observations) and m is
            the number of features (predictors).
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''

        # _fit is called here to be able to extend later on.
        # the plan is to add a parallel fitting function later on
        # e.g. _fit_parallel(x, progress_callback, n_cores)
        self._fit(x, progress_callback=progress_callback)

    def predict(
        self,
        x: numpy.ndarray,
        linkage_type: str = 'average',
        progress_callback: Optional[Callable] = None,
    ) -> CKmeansResult:
        '''predict

        Predict cluster membership of new data from fitted CKmeans.

        Parameters
        ----------
        x : numpy.ndarray
            n * m matrix, where n is the number of samples (observations) and m is
            the number of features (predictors).
        linkage_type : str
            Linkage type of the hierarchical clustering that is used for consensus cluster
            calculation. One of

            * 'average'
            * 'complete'
            * 'single'
            * 'weighted'
            * 'centroid'

            See scipy.cluster.hierarchy.linkage for details.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.

        Returns
        -------
        CKmeansResult
            Object comprising a  n * n consensus matrix, and a n-length vector of
            precited cluster memberships.
        '''
        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            a, b = numpy.meshgrid(cl, cl)
            cmatrix += a == b

            if progress_callback:
                progress_callback()

        cmatrix /= self.n_rep

        linkage = hierarchy.linkage(
            condensed_form(1 - cmatrix),
            method=linkage_type,
        )
        cl = hierarchy.fcluster(linkage, self.k, criterion='maxclust') - 1

        return CKmeansResult(cmatrix, cl)

    def _fit(
        self,
        x: numpy.ndarray,
        progress_callback: Optional[Callable] = None,
    ):
        '''_fit

        Internal sequential fitting function.

        Parameters
        ----------
        x : numpy.ndarray
            n * m matrix, where n is the number of samples (observations) and m is
            the number of features (predictors).
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''
        self._reset()

        self.kmeans = []

        if len(self._metrics) > 0:
            if 'sil' in self._metrics:
                self.sils = numpy.zeros(self.n_rep)
            if 'bic' in self._metrics:
                self.bics = numpy.zeros(self.n_rep)
            if 'db' in self._metrics:
                self.dbs = numpy.zeros(self.n_rep)
            if 'ch' in self._metrics:
                self.chs = numpy.zeros(self.n_rep)

        n_samp = numpy.ceil(self.p_samp * x.shape[0]).astype(int)
        n_feat = numpy.ceil(self.p_feat * x.shape[1]).astype(int)

        self.sel_feat = numpy.zeros((self.n_rep, n_feat), dtype=int)
        self.centers = numpy.zeros((self.n_rep, self.k, n_feat))

        for i in range(self.n_rep):
            samp_idcs = numpy.random.choice(x.shape[0], size=n_samp)
            feat_idcs = numpy.random.choice(x.shape[1], size=n_feat)
            self.sel_feat[i] = feat_idcs

            x_subset = x[samp_idcs][:, feat_idcs]

            km = KMeans(self.k)
            km.fit(x_subset)
            self.kmeans.append(km)
            self.centers[i] = km.cluster_centers_

            if len(self._metrics) > 0:
                cl = km.predict(x_subset)

                if 'sil' in self._metrics:
                    self.sils[i] = silhouette_score(x_subset, cl)
                if 'bic' in self._metrics:
                    self.bics[i] = bic_kmeans(x_subset, cl, km.cluster_centers_)
                if 'db' in self._metrics:
                    self.dbs[i] = davies_bouldin_score(x_subset, cl)
                if 'ch' in self._metrics:
                    self.chs[i] = calinski_harabasz_score(x_subset, cl)

            if progress_callback:
                progress_callback()

    def _reset(self):
        '''_reset

        Reset CKmeans object.
        '''
        self.centers = None
        self.kmeans = None
        self.sel_feat = None
        self.sils = None
        self.bics = None
        self.dbs = None

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
