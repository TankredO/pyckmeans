'''ckmeans module'''

from typing import Any, Callable, Dict, Iterable, Optional, Union, Tuple, TYPE_CHECKING
import os
import json
from _pytest.config import cmdline

import numpy
import pandas
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster import hierarchy

import pyckmeans.ordination
import pyckmeans.ordering

from .utils import NumpyEncoder

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.figure

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

    Parameters
    ----------
    consensus_matrix : numpy.ndarray
        n * n consensus matrix.
    cluster_membership : numpy.ndarray
        n-length vector of cluster memberships.
    k : int
        number of clusters.
    bic : Optional[float]
        BIC score of the consensus clustering.
    sil : Optional[float]
        Silhouette score of the consensus clustering.
    db : Optional[float]
        Davies-Bouldin score of the consensus clustering.
    ch : Optional[float]
        Calinski-Harabasz score of the consensus clustering.
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
    km_cls : Optional[numpy.ndarray]
        m*n matrix of predicted cluster memberships for each single K-Means run,
        where m is the number of single K-Means runs and n is the number samples.
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
        km_cls: Optional[numpy.ndarray] = None,
    ):
        self.cmatrix = consensus_matrix
        self.cl = cluster_membership
        self.k = k

        self.bic = bic
        self.sil = sil
        self.db = db
        self.ch = ch

        self.names = numpy.arange(consensus_matrix.shape[0]) if names is None else numpy.array(names)

        self.km_cls = km_cls

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

        ckmres = self if in_place else self.copy()

        ckmres.cmatrix = ckmres.cmatrix[order, :][:, order]
        ckmres.cl = ckmres.cl[order]
        ckmres.names = None if ckmres.names is None else ckmres.names[order]
        ckmres.km_cls = None if ckmres.km_cls is None else ckmres.km_cls[:, order]

        return ckmres

    def copy(self) -> 'CKmeansResult':
        '''copy

        Get a deep copied CKmeansResult.

        Returns
        -------
        CKmeansResult
            A deep copy of self.
        '''

        return CKmeansResult(
            consensus_matrix=self.cmatrix.copy(),
            cluster_membership=self.cl.copy(),
            k=self.k,
            bic=self.bic,
            sil=self.sil,
            db=self.db,
            ch=self.ch,
            names=None if self.names is None else self.names.copy(),
            km_cls=None if self.km_cls is None else self.km_cls.copy(),
        )

    def plot(
        self,
        names: Optional[Iterable[str]] = None,
        order: Optional[Union[str, numpy.ndarray]] = 'GW',
        cmap_cm: Union[str, 'matplotlib.colors.Colormap'] = 'Blues',
        cmap_clbar: Union[str, 'matplotlib.colors.Colormap'] = 'tab20',
        figsize: Tuple[float, float] = (7, 7),
    ) -> 'matplotlib.figure.Figure':
        '''plot

        Plot pyckmeans result consensus matrix with consensus clusters.

        Parameters
        ----------
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
        from pyckmeans.utils import plot_ckmeans_result

        return plot_ckmeans_result(
            ckm_res=self,
            names=names,
            order=order,
            cmap_cm=cmap_cm,
            cmap_clbar=cmap_clbar,
            figsize=figsize,
        )

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
            mapping = numpy.eye(self.k).astype(int)
            with open(out_file, 'wb') as out_f:
                for cl in self.km_cls:
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
    ) -> 'CKmeansResult':
        '''recalculate_cluster_memberships

        ATTENTION: This method may only be used if the WECRResult was not reordered,
        or if x was reordered the same way as the WECRResult. 

        Recalculate cluster memberships using hierarchical clustering based on the given
        linkage type.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
            The data that was used to predict the present CKmeansResult.
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
        CKmeansResult
            CKmeansResult with recalculated cluster memberships.
        '''
        ckmres = self if in_place else self.copy()

        if isinstance(x, pyckmeans.ordination.PCOAResult):
            x = x.vectors
        elif isinstance(x, pandas.DataFrame):
            x = x.values


        linkage = hierarchy.linkage(
            pyckmeans.ordering.condensed_form(1 - ckmres.cmatrix),
            method=linkage_type,
        )
        # fcluster clusters start at one
        ckmres.cl = hierarchy.fcluster(linkage, ckmres.k, criterion='maxclust') - 1

        ckmres.bic = bic_kmeans(x, ckmres.cl)
        ckmres.sil = silhouette_score(x, ckmres.cl)
        ckmres.db = davies_bouldin_score(x, ckmres.cl)
        ckmres.ch = calinski_harabasz_score(x, ckmres.cl)

        return ckmres

    def to_dict(
        self,
    ) -> Dict:
        '''to_dict

        Convert CKmeansResult to dictionary.

        Returns
        -------
        Dict
            CKmeansResult as dictionary.
        '''

        return {
            'cmatrix': self.cmatrix,
            'cl': self.cl,
            'k': self.k,
            'bic': self.bic,
            'sil': self.sil,
            'db': self.db,
            'ch': self.ch,
            'names': self.names,
            'km_cls': self.km_cls,
        }

    @classmethod
    def from_dict(
        cls,
        ckm_res_dict: Dict,
    ) -> 'CKmeansResult':
        '''from_dict

        Construct CKmeansResult from dictionary.

        Parameters
        ----------
        ckm_res_dict : Dict
            CKmeansResult as dictionary.

        Returns
        -------
        CKmeansResult
            CKmeansResult
        '''
        return cls(
            consensus_matrix=ckm_res_dict['cmatrix'],
            cluster_membership=ckm_res_dict['cl'],
            k=ckm_res_dict['k'],
            bic=ckm_res_dict['bic'],
            sil=ckm_res_dict['sil'],
            db=ckm_res_dict['db'],
            ch=ckm_res_dict['ch'],
            names=ckm_res_dict['names'],
            km_cls=ckm_res_dict['km_cls'],
        )

    def to_json(
        self,
        file: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> Optional[str]:
        '''to_json

        Convert CKmeansResult to JSON string or file.

        Parameters
        ----------
        file : Optional[str], optional
            File path to write the CKmeansResult to or None.
            If None, the JSON string will be returned.
        kwargs : Dict[str, Any]
            Additional keyword arguments passed to json.dump or json.dumps.

        Returns
        -------
        Optional[str]
            None or JSON string.
        '''
        ckm_res_dict = self.to_dict()
        if file is None:
            return json.dumps(ckm_res_dict, cls=NumpyEncoder, **kwargs)
        else:
            with open(file, 'w') as json_f:
                json.dump(ckm_res_dict, json_f, cls=NumpyEncoder, **kwargs)

        return None


    @classmethod
    def from_json_str(
        cls,
        json_str: str,
        **kwargs: Dict[str, Any],
    ) -> 'CKmeansResult':
        '''from_json_str

        Construct CKmeansResult from JSON string.

        Parameters
        ----------
        json_str: str
            JSON string.
        kwargs : Dict[str, Any]
            Additional keyword arguments passed to json.loads.

        Returns
        -------
        CKmeansResult
            CKmeansResult
        '''
        json_dict = json.loads(json_str, **kwargs)

        json_dict['cmatrix'] = numpy.array(json_dict['cmatrix'])
        json_dict['cl'] = numpy.array(json_dict['cl'], dtype=int)
        json_dict['names'] = \
            numpy.array(json_dict['names']) if not json_dict['names'] is None else None
        json_dict['km_cls'] = \
             numpy.array(json_dict['km_cls'], dtype=int) if not json_dict['km_cls'] is None else None

        return cls.from_dict(json_dict)

    @classmethod
    def from_json(
        cls,
        file: str,
        **kwargs: Dict[str, Any],
    ) -> 'CKmeansResult':
        '''from_json

        Construct CKmeansResult from JSON file.

        Parameters
        ----------
        file : str
            JSON file
        kwargs : Dict[str, Any]
            Additional keyword arguments passed to json.loads.
        Returns
        -------
        CKmeansResult
            CKmeansResult
        '''
        with open(file, 'r') as json_f:
            json_string = json_f.read()

        return cls.from_json_str(json_string, **kwargs)

    def to_dir(
        self,
        out_dir: str,
        force: bool = False,
    ):
        '''to_dir

        Save CKmeansResult to directory.
        The directory will contain the three files 'cmatrix.csv', comprising the consensus matrix,
        'clusters.csv', comprising the consensus cluster membership, and 'metrics.csv', comprising
        the clustering metrics.
        If the CKmeansResult contains clustering information considering the single K-Means runs,
        those will be written to 'km_clusters.csv'.

        Parameters
        ----------
        out_dir : str
            Output directory. Will be created if it does not exist.
        force : bool, optional
            Write into out_dir even if it does already exist, by default False.

        Raises
        ------
        Exception
            Raised if there is a problem with out_dir.
        '''
        if os.path.exists(out_dir):
            if not force:
                msg = f'Output directory "{out_dir}" already exists.'
                raise Exception(msg)
        else:
            os.mkdir(out_dir)

        cmatrix_file = os.path.join(out_dir, 'cmatrix.csv')
        cmatrix_df = pandas.DataFrame(
            self.cmatrix,
            index=self.names,
            columns=self.names,
        )
        cmatrix_df.index.name = 'sample'
        cmatrix_df.to_csv(cmatrix_file, index=True, header=True)

        cl_file = os.path.join(out_dir, 'clusters.csv')
        cl_df = pandas.DataFrame(
            {'cl': self.cl},
            index=self.names,
        )
        cl_df.to_csv(cl_file, index=True, header=True)

        metrics_file = os.path.join(out_dir, 'metrics.csv')
        metrics_df = pandas.DataFrame(
            {
                'bic': [self.bic],
                'db': [self.db],
                'sil': [self.sil],
                'ch': [self.ch],
            },
        )
        metrics_df.to_csv(metrics_file, index=False, header=True)

        if not self.km_cls is None:
            km_cls_file = os.path.join(out_dir, 'km_clusters.csv')
            km_cls_df = pandas.DataFrame(
                self.km_cls,
                columns=self.names,
                index=[f'KM{n}' for n in range(self.km_cls.shape[0])],
            )
            km_cls_df.to_csv(km_cls_file, index=True, header=True)

    @classmethod
    def from_dir(
        cls,
        directory: str,
    ) -> 'CKmeansResult':
        '''from_dir

        Construct CKmeansResult from a directory contraining the three files 'cmatrix.csv',
        'clusters.csv', 'metrics.csv', and optionally 'km_clusters.csv'.
        See :func:`<pyckmeans.core.ckmeans.CKmeansResult.to_dir>`.

        Parameters
        ----------
        directory : str
            CKmeansResult directory.

        Returns
        -------
        CKmeansResult
            CKmeansResult

        Raises
        ------
        Exception
            Raised if there is a problem with directory.
        '''
        if not os.path.exists(directory) or not os.path.isdir(directory):
            msg = f'Could not find directory at "{directory}".'
            raise Exception(msg)

        cmatrix_file = os.path.join(directory, 'cmatrix.csv')
        cmatrix = pandas.read_csv(cmatrix_file, header=0, index_col=0).values

        cl_file = os.path.join(directory, 'clusters.csv')
        cl_df = pandas.read_csv(cl_file, header=0, index_col=0)['cl']
        cl = cl_df.values
        names = numpy.array(cl_df.index)

        metrics_file = os.path.join(directory, 'metrics.csv')
        metrics_df = pandas.read_csv(metrics_file, header=0, index_col=None)
        bic = metrics_df['bic'].values[0]
        bic = bic if not numpy.isnan(bic) else None
        db = metrics_df['db'].values[0]
        db = db if not numpy.isnan(db) else None
        sil = metrics_df['sil'].values[0]
        sil = sil if not numpy.isnan(sil) else None
        ch = metrics_df['ch'].values[0]
        ch = ch if not numpy.isnan(ch) else None

        km_cls_file = os.path.join(directory, 'km_clusters.csv')
        km_cls = None
        if os.path.exists(km_cls_file):
            km_cls = pandas.read_csv(km_cls_file, index_col=0, header=0).values

        return cls(
            consensus_matrix=cmatrix,
            cluster_membership=cl,
            k=len(numpy.unique(cl)),
            names=names,
            bic=bic,
            db=db,
            sil=sil,
            ch=ch,
            km_cls=km_cls,
        )

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
        x: Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame],
        progress_callback: Optional[Callable] = None,
    ):
        '''fit

        Fit CKmeans.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
            a n * m matrix (numpy.ndarray) or dataframe (pandas.DataFrame), where n is the number
            of samples (observations) and m is the number of features (predictors).
            Alternatively a pyckmeans.ordination.PCOAResult as returned from pyckmeans.pcoa.
        progress_callback : Optional[Callable]
            Optional callback function for progress reporting.
        '''

        if isinstance(x, pyckmeans.ordination.PCOAResult):
            x = x.vectors
        elif isinstance(x, pandas.DataFrame):
            x = x.values

        # _fit is called here to be able to extend later on.
        # The plan is to add a parallel fitting function later on
        # e.g. _fit_parallel(x, progress_callback, n_cores)
        self._fit(x, progress_callback=progress_callback)

    def predict(
        self,
        x: Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame],
        linkage_type: str = 'average',
        return_cls: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> CKmeansResult:
        '''predict

        Predict cluster membership of new data from fitted CKmeans.

        Parameters
        ----------
        x : Union[numpy.ndarray, pyckmeans.ordination.PCOAResult, pandas.DataFrame]
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
        if isinstance(x, pyckmeans.ordination.PCOAResult):
            names = x.names
            x = x.vectors
        elif isinstance(x, pandas.DataFrame):
            names = numpy.array(x.index)
            x = x.values

        cmatrix = numpy.zeros((x.shape[0], x.shape[0]))

        km_cls = None
        if return_cls:
            km_cls = numpy.zeros((self.n_rep, x.shape[0]), dtype=int)

        for i, km in enumerate(self.kmeans):
            cl = km.predict(x[:, self.sel_feat[i]])
            if return_cls:
                km_cls[i] = cl
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
            km_cls=km_cls,
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
