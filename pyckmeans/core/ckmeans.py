'''ckmeans module'''

from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster import hierarchy

import pyckmeans.pcoa
import pyckmeans.ordering

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

    Contains the members
    * cmatrix: n * n consensus matrix
    * cl: n-length vector of cluster memberships

    Parameters
    ----------
    consensus_matrix : numpy.ndarray
        n * n consensus matrix.
    cluster_membership : numpy.ndarray
        n-length vector of cluster memberships
    k : float
        number of clusters
    bic: Optional[float]
        BIC score of the consensus clustering.
    sil: Optional[float]
        Silhouette score of the consensus clustering.
    db: Optional[float]
        Davies-Bouldin score of the consensus clustering.
    ch: Optional[float]
        Calinski-Harabasz score of the consensus clustering.
    names: Optional[Iterable(str)]
        Sample names.

    Attributes
    ----------
    cmatrix : numpy.ndarray
        Consensus matrix.
    cl : numpy.ndarray
        Cluster membership.
    names : Optional[numpy.ndarray]
        Sample names.
    k : int
        Number of clusters.
    bic : Optional[float]
        Bayesian Information Criterion score of the clustering.
    sil : Optional[float]
        Silhouette scor of the clustering.
    db : Optional[float]
        Davies-Bouldin score of the clustering.
    ch : Optional[float]
        Calinski-Harabasz score of the clustering.
    '''
    def __init__(
        self,
        consensus_matrix: numpy.ndarray,
        cluster_membership: numpy.ndarray,
        k: int,
        bic: Optional[float] = None,
        sil: Optional[float] = None,
        db: Optional[float] = None,
        ch: Optional[float] = None,
        names: Optional[Iterable[str]] = None,
    ):
        self.cmatrix = consensus_matrix
        self.cl = cluster_membership
        self.k = k

        self.bic = bic
        self.sil = sil
        self.db = db
        self.ch = ch

        self.names: Optional[numpy.ndarray] = None if names is None else numpy.array(names)

    def order(
        self,
        method: str = 'GW',
        linkage_type: str = 'average',
    ) -> numpy.ndarray:
        '''order

        Get optimal order according to hierarchical clustering.

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

        return pyckmeans.ordering.distance_order(
            1 - self.cmatrix,
            method=method,
            linkage_type=linkage_type
        )

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

        return self.reorder(order, in_place=in_place)

    def reorder(
        self,
        order: numpy.ndarray,
        in_place: bool = False,
    ) -> 'CKmeansResult':
        '''reorder

        Reorder samples according to provided order.

        Parameters
        ----------
        order : numpy.ndarray
            New sample order.
        in_place : bool
            If False, a new, sorted CKmeansResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        CKmeansResult
            Reordered CKmeansResult
        '''

        if in_place:
            ckmres = self
        else:
            ckmres = CKmeansResult(
                consensus_matrix=self.cmatrix.copy(),
                cluster_membership=self.cl.copy(),
                k=self.k,
                bic=self.bic,
                sil=self.sil,
                db=self.db,
                ch=self.ch,
                names=None if self.names is None else self.names.copy(),
            )

        ckmres.cmatrix = ckmres.cmatrix[order, :][:, order]
        ckmres.cl = ckmres.cl[order]
        ckmres.names = None if ckmres.names is None else ckmres.names[order]

        return ckmres


class InvalidClusteringMetric(Exception):
    '''InvalidClusteringMetric

    Error signalling that an invalid clustering metric was provided.
    '''

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
        Clustering quality metrics to calculate while training. Available metrics are
        * "sil" (Silhouette Index)
        * "bic" (Bayesian Information Criterion)
        * "db" (Davies-Bouldin Index)
        * "ch" (Calinski-Harabasz).
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to sklearn.cluster.KMeans.
    '''

    AVAILABLE_METRICS = ('sil', 'bic', 'db', 'ch')

    def __init__(
        self,
        k: int,
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
        metrics: Iterable[str] = ('sil', 'bic'),
        **kwargs: Dict[str, Any],
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

        # KMeans options
        self._kmeans_kwargs = {
            'n_init': 2,
        }
        self._kmeans_kwargs.update(kwargs)

    def fit(
        self,
        x: Union[numpy.ndarray, pyckmeans.pcoa.PCOAResult],
        progress_callback: Optional[Callable] = None,
    ):
        '''fit

        Fit CKmeans.

        Parameters
        ----------
        x : numpy.ndarray
            n * m matrix, where n is the number of samples (observations) and m is
            the number of features (predictors).
            Alternatively a pyckmeans.pcoa.PCOAResult as returned from pyckmeans.pcoa.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''

        if isinstance(x, pyckmeans.pcoa.PCOAResult):
            x = x.vectors

        # _fit is called here to be able to extend later on.
        # The plan is to add a parallel fitting function later on
        # e.g. _fit_parallel(x, progress_callback, n_cores)
        self._fit(x, progress_callback=progress_callback)

    def predict(
        self,
        x: Union[numpy.ndarray, pyckmeans.pcoa.PCOAResult],
        linkage_type: str = 'average',
        progress_callback: Optional[Callable] = None,
    ) -> CKmeansResult:
        '''predict

        Predict cluster membership of new data from fitted CKmeans.

        Parameters
        ----------
        x : Union[numpy.ndarray, PCOAResult]
            n * m matrix, where n is the number of samples (observations) and m is
            the number of features (predictors).
            Alternatively a pyckmeans.pcoa.PCOAResult as returned from pyckmeans.pcoa.
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
        names = None
        if isinstance(x, pyckmeans.pcoa.PCOAResult):
            names = x.names
            x = x.vectors

        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            a, b = numpy.meshgrid(cl, cl)
            cmatrix += a == b

            if progress_callback:
                progress_callback()

        cmatrix /= self.n_rep

        # = prepare output
        linkage = hierarchy.linkage(
            pyckmeans.ordering.condensed_form(1 - cmatrix),
            method=linkage_type,
        )
        # fcluster clusters start at one
        cl = hierarchy.fcluster(linkage, self.k, criterion='maxclust') - 1

        bic = bic_kmeans(x, cl)
        sil = silhouette_score(x, cl)
        db = davies_bouldin_score(x, cl)
        ch = calinski_harabasz_score(x, cl)

        return CKmeansResult(
            cmatrix,
            cl,
            k=self.k,
            bic=bic,
            sil=sil,
            db=db,
            ch=ch,
            names=names,
        )

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

            km = KMeans(self.k, **self._kmeans_kwargs)
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
        self.chs = None
