'''multickmeans module'''

from typing import List, Optional, Iterable, Dict, Any, Tuple, Union, Callable
import numpy
import pandas

from pyckmeans.ordination import PCOAResult
from .ckmeans import CKmeansResult, CKmeans, InvalidClusteringMetric

class MultiCKmeansResult:
    '''MultiCKmeansResult

    Result of MultiCKmeansResult.predict.

    Parameters
    ----------
    ckmeans_results: List[CKmeansResult]
        List of CKmeansResults.
    names: Optional[Iterable(str)]
        Sample names.
    '''
    def __init__(
        self,
        ckmeans_results: List[CKmeansResult],
        names: Optional[Iterable[str]] = None,
    ):
        self.ckmeans_results = ckmeans_results
        self.names: Optional[numpy.ndarray] = None if names is None else numpy.array(names)

        self.ks = numpy.array([ckm_res.k for ckm_res in ckmeans_results])

        self.sils = numpy.array([ckm_res.sil for ckm_res in ckmeans_results])
        self.bics = numpy.array([ckm_res.bic for ckm_res in ckmeans_results])
        self.dbs = numpy.array([ckm_res.db for ckm_res in ckmeans_results])
        self.chs = numpy.array([ckm_res.ch for ckm_res in ckmeans_results])

        self.metrics = pandas.DataFrame({
            'k': self.ks,
            'sil': self.sils,
            'bic': self.bics,
            'db': self.dbs,
            'ch': self.chs,
        })

    def order(
        self,
        by: int,
        method: str = 'GW',
        linkage_type: str = 'average',
    ) -> numpy.ndarray:
        '''order

        Get optimal sample order according to hierarchical clustering of the
        CKmeansResult at index "by".

        Parameters
        ----------
        by : int
            Index of the CKMeansResult to order by.
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
        ckm_res = self.ckmeans_results[by]

        return ckm_res.order(method=method, linkage_type=linkage_type)

    def sort(
        self,
        by: int,
        method: str = 'GW',
        linkage_type: str = 'average',
        in_place: bool = False,
    ) -> 'MultiCKmeansResult':
        '''sort

        Sort samples according to hierarchical clustering of the
        CKmeansResult at index "by".

        Parameters
        ----------
        by : int
            Index of the CKMeansResult to sort by.
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
            If False, a new, sorted MultiCKmeansResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        MultiCKmeansResult
            Sorted MultiCKmeansResult
        '''

        order = self.order(by=by, method=method, linkage_type=linkage_type)

        return self.reorder(order, in_place=in_place)

    def reorder(
        self,
        order: numpy.ndarray,
        in_place: bool = False,
    ) -> 'MultiCKmeansResult':
        '''reorder

        Reorder samples in all CKmeansResults according to provided order.

        Parameters
        ----------
        order : numpy.ndarray
            New sample order.
        in_place : bool
            If False, a new, sorted MultiCKmeansResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        MultiCKmeansResult
            Reordered MultiCKmeansResult
        '''

        if in_place:
            mckmres = self
            for ckmres in self.ckmeans_results:
                ckmres.reorder(order, in_place=True)
        else:
            ckm_results = [
                ckmres.reorder(order, in_place=False) for ckmres in self.ckmeans_results
            ]
            names = None if self.names is None else self.names.copy()
            mckmres = MultiCKmeansResult(ckm_results, names=names)

        return mckmres

    def plot_metrics(
        self,
        figsize: Tuple[float, float] = (7, 7),
    ) -> 'matplotlib.figure.Figure':
        '''plot_metrics

        Plot MultiCKMeansResult metrics.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size for the matplotlib figure, by default (7, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib Figure of the metrics plot.
        '''

        from pyckmeans.utils import plot_multickmeans_metrics

        fig = plot_multickmeans_metrics(
            mckm_res=self,
            figsize=figsize,
        )
        fig.tight_layout()
        return fig

class MultiCKMeans:
    '''MultiCKMeans

    Convenience class wrapping Consensus K-Means runs for multiple different numbers of clusters.

    Parameters
    ----------
    k : Iterable[int]
        List of cluster counts for CKmeans.
    n_rep : int, optional
        Number of K-Means to fit for each single CKmeans, by default 100
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
    def __init__(
        self,
        k: Iterable[int],
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
            if not metric in CKmeans.AVAILABLE_METRICS:
                am_str = ", ".join(CKmeans.AVAILABLE_METRICS)
                msg = f'Unknown metric "{metric}". Available metrics are {am_str}.'
                raise InvalidClusteringMetric(msg)

        self._metrics = metrics
        self._kmeans_kwargs = kwargs

        self.ckmeans: Optional[List[CKmeans]] = None

    def fit(
        self,
        x: Union[numpy.ndarray, PCOAResult, pandas.DataFrame],
        progress_callback: Optional[Callable] = None,
    ):
        '''fit

        Fit MultiCKmeans.

        Parameters
        ----------
        x : Union[numpy.ndarray, PCOAResult]
            a n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors).
            Alternatively a pyckmeans.ordination.PCOAResult as returned from pyckmeans.pcoa.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''

        if isinstance(x, PCOAResult):
            x = x.vectors
        elif isinstance(x, pandas.DataFrame):
            x = x.values

        # _fit is called here to be able to extend later on.
        # The plan is to add a parallel fitting function later on
        # e.g. _fit_parallel(x, progress_callback, n_cores)
        self._fit(x, progress_callback=progress_callback)

    def predict(
        self,
        x: Union[numpy.ndarray, PCOAResult, pandas.DataFrame],
        linkage_type: str = 'average',
        return_cls: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> MultiCKmeansResult:
        '''predict

        Predict cluster membership of new data from all fitted CKmeans.

        Parameters
        ----------
        x : Union[numpy.ndarray, PCOAResult]
            a n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors). If x is a
            dataframe, the index will be used a sample names.
            Alternatively a pyckmeans.ordination.PCOAResult as returned from pyckmeans.pcoa.
        linkage_type : str
            Linkage type of the hierarchical clustering that is used for consensus cluster
            calculation. One of

            * 'average'
            * 'complete'
            * 'single'
            * 'weighted'
            * 'centroid'

            See scipy.cluster.hierarchy.linkage for details.
        return_cls : bool
            If True, the cluster memberships of the single K-Means runs will be present
            in the output.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.

        Returns
        -------
        CKmeansResult
            Object comprising a  n * n consensus matrix, and a n-length vector of
            precited cluster memberships.
        '''
        names = None
        if isinstance(x, PCOAResult):
            names = x.names
        elif isinstance(x, pandas.DataFrame):
            names = x.index

        ckmeans_results: List[CKmeansResult] = []
        for ckm in self.ckmeans:
            ckm_res = ckm.predict(
                x=x,
                linkage_type=linkage_type,
                return_cls=return_cls,
                progress_callback=progress_callback,
            )
            ckmeans_results.append(ckm_res)

        return MultiCKmeansResult(
            ckmeans_results=ckmeans_results,
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

        self.ckmeans = []

        for k in self.k:
            ckm = CKmeans(
                k=k,
                n_rep=self.n_rep,
                p_samp=self.p_samp,
                p_feat=self.p_feat,
                metrics=self._metrics,
                **self._kmeans_kwargs,
            )
            ckm.fit(x=x, progress_callback=progress_callback)
            self.ckmeans.append(ckm)

    def _reset(self):
        '''_reset

        Reset MultiCKmeans object.
        '''
        self.ckmeans = None
