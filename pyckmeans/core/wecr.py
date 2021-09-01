''' Weighted Ensemble Consensus of Random K-Means (WECR K-Means)
'''

import os
from typing import Union, Optional, Iterable, Callable, Tuple, Dict, Any, TYPE_CHECKING

import numpy
import pandas
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster import hierarchy

import pyckmeans.ordination
import pyckmeans.ordering
from pyckmeans.core.ckmeans import bic_kmeans

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure

class InvalidKError(Exception):
    '''InvalidKError'''

class WECRResult:
    '''WECRResult

    Result of WECR.predict.

    Parameters
    ----------
    consensus_matrix : numpy.ndarray
        n * n weighted consensus (co-association) matrix.
    cluster_membership : numpy.ndarray
        n * m matrix cluster memberships, where m in the number of different k values.
    k : Iterable[int]
        Vector of cluster numbers.
    bic : Optional[numpy.ndarray]
        m-length vector of BIC scores of the consensus clustering for each k.
    sil : Optional[numpy.ndarray]
        m-length vector of Silhouette scores of the consensus clustering for each k.
    db : Optional[numpy.ndarray]
        m-length vector of Davies-Bouldin score of the consensus clustering for each k.
    ch : Optional[numpy.ndarray]
        m-length vector of Calinski-Harabasz score of the consensus clustering for each k.
    names : Optional[Iterable(str)]
        Sample names.
    km_cls : Optional[numpy.ndarray]
        m*n matrix of predicted cluster memberships for each single K-Means run,
        where m is the number of single K-Means runs and n is the number samples.

    Attributes
    ----------
    cmatrix : numpy.ndarray
        Consensus matrix.
    cl : numpy.ndarray
        Cluster memberships for each k.
    names : Optional[numpy.ndarray]
        Sample names.
    k : numpy.ndarray
        Number of clusters.
    bic : Optional[numpy.ndarray]
        Bayesian Information Criterion score of the clustering.
    sil : Optional[numpy.ndarray]
        Silhouette scor of the clustering.
    db : Optional[numpy.ndarray]
        Davies-Bouldin score of the clustering.
    ch : Optional[numpy.ndarray]
        Calinski-Harabasz score of the clustering.
    km_cls : Optional[numpy.ndarray]
        m*n matrix of predicted cluster memberships for each single K-Means run,
        where m is the number of single K-Means runs and n is the number samples.
    '''
    def __init__(
        self,
        consensus_matrix: numpy.ndarray,
        cluster_membership: numpy.ndarray,
        k: numpy.ndarray,
        bic: Optional[numpy.ndarray] = None,
        sil: Optional[numpy.ndarray] = None,
        db: Optional[numpy.ndarray] = None,
        ch: Optional[numpy.ndarray] = None,
        names: Optional[Iterable[str]] = None,
        km_cls: Optional[numpy.ndarray] = None,
    ):
        self.cmatrix = consensus_matrix
        self.cl = cluster_membership
        self.k = k

        self.bic = bic
        self.sil = sil
        self.db = db
        self.ch = ch

        self.names: Optional[numpy.ndarray] = None if names is None else numpy.array(names)

        self.km_cls = km_cls

    def copy(self) -> 'WECRResult':
        '''copy

        Get a deep copied WECRResult.

        Returns
        -------
        WECRResult
            A deep copy of self.
        '''

        return WECRResult(
            consensus_matrix=self.cmatrix.copy(),
            cluster_membership=self.cl.copy(),
            k=self.k.copy(),
            bic=None if self.bic is None else self.bic.copy(),
            sil=None if self.sil is None else self.sil.copy(),
            db=None if self.db is None else self.db.copy(),
            ch=None if self.ch is None else self.ch.copy(),
            names=None if self.names is None else self.names.copy(),
            km_cls=None if self.km_cls is None else self.km_cls.copy(),
        )

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
            Reordering method. Either 'GW' (Gruvaeus & Wainer, 1972) [1]_ or 'OLO' for
            scipy.hierarchy.optimal_leaf_ordering.

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

        References
        ----------
        .. [1]  Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
                The British Psychological Society 25.
        '''
        return pyckmeans.ordering.distance_order(
            1 - self.cmatrix,
            method=method,
            linkage_type=linkage_type
        )

    def reorder(
        self,
        order: numpy.ndarray,
        in_place: bool = False,
    ) -> 'WECRResult':
        '''reorder

        Reorder samples according to provided order.

        Parameters
        ----------
        order : numpy.ndarray
            New sample order.
        in_place : bool
            If False, a new, sorted WECRResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        WECRResult
            Reordered WECRResult
        '''

        wecr_res = self if in_place else self.copy()

        wecr_res.cmatrix = wecr_res.cmatrix[order, :][:, order]
        wecr_res.cl = wecr_res.cl[:, order]
        wecr_res.names = None if wecr_res.names is None else wecr_res.names[order]
        wecr_res.km_cls = None if wecr_res.km_cls is None else wecr_res.km_cls[:, order]

        return wecr_res

    def sort(
        self,
        method: str = 'GW',
        linkage_type: str = 'average',
        in_place: bool = False,
    ) -> 'WECRResult':
        '''sort

        Sort WECRResult using hierarchical clustering.

        Parameters
        ----------
        method : str
            Reordering method. Either 'GW' (Gruvaeus & Wainer, 1972) [1]_ or 'OLO' for
            scipy.hierarchy.optimal_leaf_ordering.

        linkage_type : str
            Linkage type for the hierarchical clustering. One of

            * 'average'
            * 'complete'
            * 'single'
            * 'weighted'
            * 'centroid'

            See scipy.cluster.hierarchy.linkage for details.
        in_place : bool
            If False, a new, sorted WECRResult object will be returned.
            If True, the object will be sorted in place and self will be returned.

        Returns
        -------
        WECRResult
            Sorted WECRResult

        References
        ----------
        .. [1]  Gruvaeus, G., H., Wainer. 1972. Two Additions to Hierarchical Cluster Analysis.
                The British Psychological Society 25.
        '''

        order = self.order(method=method, linkage_type=linkage_type)

        return self.reorder(order, in_place=in_place)

    def plot(
        self,
        k: int,
        names: Optional[Iterable[str]] = None,
        order: Optional[Union[str, numpy.ndarray]] = 'GW',
        cmap_cm: Union[str, 'matplotlib.colors.Colormap'] = 'Blues',
        cmap_clbar: Union[str, 'matplotlib.colors.Colormap'] = 'tab20',
        figsize: Tuple[float, float] = (7, 7),
    ) -> 'matplotlib.figure.Figure':
        '''plot

        Plot wecr result consensus matrix with consensus clusters.

        Parameters
        ----------
        k: int
            The number of clusters k to use for plotting.
        names : Optional[Iterable[str]]
            Sample names to be plotted.
        order : Optional[Union[str, numpy.ndarray]]
            Sample Plotting order. Either a string, determining the oder method to use
            (see CKmeansResult.order), or a numpy.ndarray giving the sample order,
            or None to apply no reordering.
        cmap_cm : Union[str, matplotlib.colors.Colormap], optional
            Colormap for the consensus matrix, by default 'Blues'
        cmap_clbar : Union[str, matplotlib.colors.Colormap], optional
            Colormap for the cluster bar, by default 'tab20'
        figsize : Tuple[float, float], optional
            Figure size for the matplotlib figure, by default (7, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure.
        '''
        from pyckmeans.utils.plotting import plot_wecr_result

        return plot_wecr_result(
            wecr_res=self,
            k=k,
            names=names,
            order=order,
            cmap_cm=cmap_cm,
            cmap_clbar=cmap_clbar,
            figsize=figsize,
        )

    def plot_metrics(
        self,
        figsize: Tuple[float, float] = (7, 7),
    ) -> 'matplotlib.figure.Figure':
        '''plot_metrics

        Plot WECRResult metrics.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size for the matplotlib figure, by default (7, 7).

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib Figure of the metrics plot.
        '''

        from pyckmeans.utils import plot_wecr_result_metrics

        fig = plot_wecr_result_metrics(
            wecr_res=self,
            figsize=figsize,
        )
        fig.tight_layout()
        return fig

    def save_km_cls(
        self,
        out_file: str,
        one_hot: bool = False,
        row_names: bool = False,
        col_names: bool = False,
    ):
        '''save_km_cls

        Save predicted cluster membership for the single K-Means runs to
        a file. The file format depends on the one_hot parameter.

        Parameters
        ----------
        out_file : str
            Output file path.
        one_hot : bool
            If False, a tab-delimited text file will be written containing a n*m cluster
            membership matrix, where n is the number of K-Means runs and m is the number
            of samples.

            If True, a file comprising n one-hot encoded m*k cluster membership matrices
            in tab-delimited text format, separated by an empty line, will be written,
            where k is the number of clusters.
        row_names : bool
            If True, row names will be written.
        col_names : bool
            If True, column names will be written.
        '''

        if one_hot:
            with open(out_file, 'wb') as out_f:
                for cl in self.km_cls:
                    mapping = numpy.eye(len(numpy.unique(cl))).astype(int)
                    cl_oh_df = pandas.DataFrame(mapping[cl], index=self.names)
                    cl_oh_df.to_csv(
                        out_f,
                        sep='\t',
                        index=row_names,
                        header=col_names,
                        mode='binary',
                    )
                    out_f.writelines([bytes(os.linesep, encoding='UTF-8')])
        else:
            km_cls_df = pandas.DataFrame(self.km_cls, columns=self.names)
            km_cls_df.to_csv(out_file, sep='\t', index=row_names, header=col_names)

    def recalculate_cluster_memberships(
        self,
        x: Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame],
        linkage_type: str,
        in_place: bool = False,
    ) -> 'WECRResult':
        '''recalculate_cluster_memberships

        ATTENTION: This method may only be used if the WECRResult was not reordered,
        or if x was reordered the same way as the WECRResult.

        Recalculate cluster memberships using hierarchical clustering based on the given
        linkage type.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
            The data that was used to predict the present WECRResult.
            A n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors).
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
        in_place : bool
            If False, a new, CKmeansResult object will be returned.
            If True, the object will modified in place and self will be returned.

        Returns
        -------
        WECRResult
            WECRResult with recalculated cluster memberships.
        '''
        wecr_res = self if in_place else self.copy()

        if isinstance(x, pyckmeans.ordination.PCOAResult):
            x = x.vectors
        elif isinstance(x, pandas.DataFrame):
            x = x.values

        bic = []
        sil = []
        db = []
        ch = []
        cluster_membership = []
        for k in wecr_res.k:
            linkage = hierarchy.linkage(
                pyckmeans.ordering.condensed_form(1 - wecr_res.cmatrix),
                method=linkage_type,
            )
            # fcluster clusters start at one
            cl = hierarchy.fcluster(linkage, k, criterion='maxclust') - 1
            cluster_membership.append(cl)

            bic.append(bic_kmeans(x, cl))
            sil.append(silhouette_score(x, cl))
            db.append(davies_bouldin_score(x, cl))
            ch.append(calinski_harabasz_score(x, cl))

        wecr_res.bic = numpy.array(bic)
        wecr_res.sil = numpy.array(sil)
        wecr_res.db = numpy.array(db)
        wecr_res.ch = numpy.array(ch)
        wecr_res.cl = numpy.array(cluster_membership, dtype=int)

        return wecr_res

    def get_cl(
        self,
        k: int,
        with_names: bool = False,
    ) -> Union[numpy.ndarray, pandas.Series]:
        '''get_cl

        Return cluster memberships at a specified k.

        Parameters
        ----------
        k : int
            Number of clusters to return the cluster memberships for.
        with_names : bool, optional
            Return cluster memberships including sample names.
            If True, a pandas.Series will be returned.

        Returns
        -------
        Union[numpy.ndarray, pandas.Series]
            Cluster memberships

        Raises
        ------
        wecr.InvalidKError
            Raised if an invalid k argument is provided.
        '''

        if not k in self.k:
            msg = f'Result for k={k} not found. Available k are {self.k}.'
            raise InvalidKError(msg)

        cl = self.cl[numpy.argmax(self.k == k)]
        if with_names:
            return pandas.Series(cl, self.names)
        else:
            return cl

    # TODO:
    # - additional cluster membership calculations

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

class InvalidConstraintsError(Exception):
    '''InvalidConstraintsError'''

def _prepare_constraints(
    must_link: numpy.ndarray,
    must_not_link: numpy.ndarray,
    names: Union[None, numpy.ndarray],
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    '''_prepare_constraints

    Check and prepare must_link and must_not_link constraints
    for WECR.

    Parameters
    ----------
    must_link : numpy.ndarray
        Must link constraints.
    must_not_link : numpy.ndarray
        Must not link constraints.
    names : Union[None, numpy.ndarray]
        Names or None.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        must link and must not link constraints as n*2 numpy arrays,
        where rows are constraints and columns a sample indices.

    Raises
    ------
    InvalidConstraintsError
        Raised if an invalid constraints argument is provided.
    '''
    if must_link.dtype.type == numpy.dtype(str) or must_not_link.dtype.type == numpy.dtype(str):
        if names is None:
            msg = 'Constraints (must_link, must_not_link) can only contain character strings, if' +\
                ' names can be inferred from x (pandas.DataFrame, pyckmeans.ordination.PCOAResult).'
            raise InvalidConstraintsError(msg)

    if must_link.dtype.type == numpy.dtype(str):
        ml = numpy.zeros_like(must_link, dtype=int)
        for i, (a, b) in enumerate(must_link):
            a_idcs = numpy.argwhere(a == names)
            b_idcs = numpy.argwhere(b == names)

            if (len(a_idcs)) < 1:
                msg = f'must_link: Could not find row with name "{a}" in x.'
                raise InvalidConstraintsError(msg)
            if (len(b_idcs)) < 1:
                msg = f'must_link: Could not find row with name "{b}" in x.'
                raise InvalidConstraintsError(msg)
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
                raise InvalidConstraintsError(msg)
            if (len(b_idcs)) < 1:
                msg = f'must_not_link: Could not find row with name "{b}" in x.'
                raise InvalidConstraintsError(msg)
            mnl[i] = numpy.array([a_idcs[0][0], b_idcs[0][0]])
    else:
        mnl = must_not_link

    return ml, mnl

class WECR:
    '''WECR K-Means

    A class representing a Weighted Ensemble Consensus of Random K-Means [1]_.

    Parameters
    ----------
    k : Union[int, Iterable[int]]
        Number of clusters to drawn from for each K-Means run.
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
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to sklearn.cluster.KMeans.

    References
    ----------
    .. [1]  Lai, Y., S., He, Z., Lin, F., Yang, Q., Zhou, X., Zhou. 2019.
            "An Adaptive Robust Semi-Supervised Clustering Framework Using Weighted Consensus of Random K-Means Ensemble".
            IEEE Transactions on Knowledge and Data Engineering, vol. 33, no. 5, pp. 1877-1890. doi: 10.1109/TKDE.2019.2952596.
    '''
    def __init__(
        self,
        k: Union[int, Iterable[int]],
        n_rep: int = 100,
        p_samp: float = 0.8,
        p_feat: float = 0.8,
        **kwargs: Dict[str, Any],
    ):
        self.k: numpy.ndarray = numpy.array(k).reshape(-1)
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
        x: Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame],
        progress_callback: Optional[Callable] = None,
    ):
        '''fit

        Fit the WECR K-Means.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
            a n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors).
            Alternatively a pyckmeans.ordination.PCOAResult as returned from pyckmeans.pcoa.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''
        self._reset()

        if isinstance(x, pandas.DataFrame):
            x = x.values
        elif isinstance(x, pyckmeans.ordination.PCOAResult):
            x = x.vectors

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
        x: Union[numpy.ndarray, pandas.DataFrame, pyckmeans.ordination.PCOAResult],
        must_link: Optional[Iterable] = None,
        must_not_link: Optional[Iterable] = None,
        gamma: float = 0.5,
        scale_consensus_matrix: bool = True,
        linkage_type: str = 'average',
        return_cls: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> WECRResult:
        '''predict

        Predict from WECR.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
            a n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors). If x is a
            dataframe, the index will be used a sample names.
            Alternatively a pyckmeans.ordination.PCOAResult as returned from pyckmeans.pcoa.
        must_link : Optional[Iterable], optional
            Must-link constraints. Any 2-dimensional iterable object with constraints as first
            dimension and sample indices (or names) as second dimension.
            For example: [[1, 2], [3, 4]], np.array([['A', 'B'], ['A', 'D']])
            Can be None for no constraints.
        must_not_link : Optional[Iterable], optional
            Must-not-link constraints. Any 2-dimensional iterable object with constraints as first
            dimension and sample indices (or names) as second dimension.
            For example: [[1, 2], [3, 4]], np.array([['A', 'B'], ['A', 'D']])
            Can be None for no constraints.
        gamma : float, optional
            Weight parameter for the constraints. Must be between 0.0 and 1.0, by default 0.5.
            Higher values increase the weight of the constraints on the final result.
        scale_consensus_matrix : bool
            If true, the consensus matrix will be scaled in such a way that the diagonal entries
            are all 1.
        linkage_type : str
            Linkage type of the hierarchical clustering that is used for final consensus cluster
            calculation.

            One of

            * 'average'
            * 'complete'
            * 'single'
            * 'weighted'
            * 'centroid'

            See scipy.cluster.hierarchy.linkage for details.
        return_cls : bool
            If True, the cluster memberships of the single K-Means runs will be present
            in the output.
        progress_callback : Optional[Callable], optional
            Optional callback function for progress reporting.

        Returns
        -------
        WECRResult
            WECRResult object.
        '''
        names = None
        if isinstance(x, pandas.DataFrame):
            names = numpy.array(x.index)
            x = x.values
        elif isinstance(x, pyckmeans.ordination.PCOAResult):
            names = numpy.array(x.names)
            x = x.vectors

        # == prepare constraints
        must_link = numpy.array(must_link) if must_link is not None else numpy.zeros((0, 2))
        must_not_link = numpy.array(must_not_link) if must_not_link is not None else numpy.zeros((0, 2))

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

        # output consensus matrix (co-association matrix)
        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        km_cls: Optional[numpy.ndarray] = None
        if return_cls:
            km_cls = numpy.zeros((self.n_rep, x.shape[0]), dtype=int)

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            k = self.ks[i]

            if return_cls:
                km_cls[i] = cl

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

                # S_{i} = H_{i}W_{i}H_{i}^T: weighted consensus matrix (weighted co-association matrix)
                co_assoc = cl_oh.dot(numpy.diag(sil)).dot(cl_oh.T)

            # If there are constraints, apply standard algorithm
            else:
                # mu_{i}: clustering-level consistency
                # number of fulfilled constraints divided by total number of constraints
                clustering_cons = \
                    (cl[ml[:, 0]] == cl[ml[:,1]]).sum() if ml.shape[0] != 0 else 0 +\
                    (cl[mnl[:, 0]] != cl[mnl[:,1]]).sum() if mnl.shape[0] != 0 else 0
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
                co_assoc = cl_oh.dot(numpy.diag(weight)).dot(cl_oh.T)

            cmatrix += co_assoc

            if progress_callback:
                progress_callback()

        cmatrix /= self.n_rep

        # == prepare output
        if scale_consensus_matrix:
            cmatrix = _normalize_similarity_matrix(cmatrix)
        bic = []
        sil = []
        db = []
        ch = []
        cluster_membership = []

        ks: numpy.ndarray = numpy.unique(self.k)
        for k in ks:
            linkage = hierarchy.linkage(
                pyckmeans.ordering.condensed_form(1 - cmatrix),
                method=linkage_type,
            )
            # fcluster clusters start at one
            cl = hierarchy.fcluster(linkage, k, criterion='maxclust') - 1
            cluster_membership.append(cl)

            bic.append(bic_kmeans(x, cl))
            sil.append(silhouette_score(x, cl))
            db.append(davies_bouldin_score(x, cl))
            ch.append(calinski_harabasz_score(x, cl))

        return WECRResult(
            consensus_matrix=cmatrix,
            cluster_membership=numpy.array(cluster_membership, dtype=int),
            k=ks,
            bic=numpy.array(bic),
            sil=numpy.array(sil),
            db=numpy.array(db),
            ch=numpy.array(ch),
            names=names,
            km_cls=km_cls,
        )

    def _reset(self):
        '''_reset

        Reset the internal state of the WECR object.
        '''
        self.kmeans = None
        self.sel_feat = None
        self.ks = None
        self.must_link = None
        self.ml = None
        self.must_not_link = None
        self.mnl = None
