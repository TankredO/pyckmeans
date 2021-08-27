''' ordination

    Module for Principle Coordinate Analysis (PCoA) calculation.
'''

# Inspired by https://github.com/cran/ape/blob/master/R/pcoa.R and
# https://github.com/biocore/scikit-bio/blob/0.5.4/skbio/stats/ordination/_principal_coordinate_analysis.py#L23

from warnings import warn
from typing import Iterable, Optional, Union

import numpy
import pandas

import pyckmeans.distance

class InvalidPCOAResultError(Exception):
    '''InvalidPCOAResultError

    Error, signalling an invalid PCOAResult.
    '''

class IncompatibleNamesError(Exception):
    '''IncompatibleNamesError

    Error, signalling that provided names are incompatible with provided data.
    '''

class InvalidFilterError(Exception):
    '''InvalidFilterError'''

class InvalidOutFormatError(Exception):
    '''InvalidOutFormatError'''

class PCOAResult:
    '''PCOAResult

    Object containing the results of a Principle Coordinate Analysis.

    Parameters
    ----------
    vectors : numpy.ndarray
        n * m matrix of potentially corrected eigenvectors, where
        n is the number of samples and m is the number of retained axes.
    eigvals : numpy.ndarray
        Vector of eigenvalues.
    eigvals_rel : numpy.ndarray
        Vector of relative eigenvalues.
    trace : float
        Trace.
    negative_eigvals : bool
        Bool, determining whether negative eigenvalues are present.
    correction : Optional[str], optional
        Type of correction, by default None
    eigvals_corr_rel : Optional[numpy.ndarray], optional
        Vector of corrected relative eigenvalues, by default None
    trace_corr : Optional[float], optional
        Corrected trace, by default None
    names : Optional[Iterable[str]], optional
        Vector of names, by default None

    Raises
    ------
    InvalidPCOAResultError
        Raised if invalid parameter combinations are provided.
    IncompatibleNamesError
        Raised if provided names are incompatible with provided data.
    '''
    def __init__(
        self,
        vectors: numpy.ndarray,
        eigvals: numpy.ndarray,
        eigvals_rel: numpy.ndarray,
        trace: float,
        negative_eigvals: bool,
        correction: Optional[str] = None,
        eigvals_corr_rel: Optional[numpy.ndarray] = None,
        trace_corr: Optional[float] = None,
        names: Optional[Iterable[str]] = None,
    ):
        self.vectors = vectors
        self.eigvals = eigvals
        self.eigvals_rel = eigvals_rel
        self.eigvals_rel_cum = numpy.cumsum(eigvals_rel)
        self.trace = trace

        self.negative_eigvals = negative_eigvals
        self.correction = correction

        self.eigvals_corr_rel = None
        self.eigvals_corr_rel_cum = None
        self.trace_corr = None

        self.names = None

        if negative_eigvals:
            if eigvals_corr_rel is None:
                msg = 'negative_eigvals is True, expecting eigvals_corr_rel.'
                raise InvalidPCOAResultError(msg)

            self.eigvals_corr_rel = eigvals_corr_rel
            self.eigvals_corr_rel_cum = numpy.cumsum(eigvals_corr_rel)

            if correction:
                if trace_corr is None:
                    msg = 'negative eigenvalue correction is set, ' +\
                        'expecting trace_corr.'
                self.trace_corr = trace_corr

            self.values = pandas.DataFrame({
                'eigvals': self.eigvals,
                'eigvals_rel': self.eigvals_rel,
                'eigvals_rel_cum': self.eigvals_corr_rel_cum,
                'eigvals_rel_corrected': self.eigvals_corr_rel,
                'eigvals_rel_corrected_cum': self.eigvals_corr_rel_cum,
            })
        else:
            self.values = pandas.DataFrame({
                'eigvals': self.eigvals,
                'eigvals_rel': self.eigvals_rel,
                'eigvals_rel_cum': self.eigvals_rel_cum,
            })

        if not names is None:
            n = vectors.shape[0]
            if len(names) != n:
                msg = f'Expected {n} names for {n}x{n} distance matrix ' +\
                    f'but {len(names)} were passed.'
                raise IncompatibleNamesError(msg)

            self.names = numpy.array(names)

    def __repr__(self) -> str:
        str_repr = f'<PCOAResult; neg. eigvals: {self.negative_eigvals}, ' +\
            f'correction: {self.correction}>'

        return str_repr

    def get_vectors(
        self,
        filter_by: Optional[str] = None,
        filter_th: Optional[float] = None,
        out_format: str = 'numpy',
    ) -> Union[numpy.ndarray, pandas.DataFrame]:
        '''getVectors

        Get eigenvectors, potentially filtered by cumulative eigenvalue scores.

        Parameters
        ----------
        filter_by : Optional[str], optional
            Cumulative eigenvalue score to filter by or None if no filtering should be applied,
            by default None.
            Can be one of:

            * 'eigvals_cum'
            * 'eigvals_rel_cum'
            * 'eigvals_rel_corrected_cum'

            If filter_by is provided filter_th is required.

        filter_th : Optional[float], optional
            Filtering treshold, by default None.
            Eigenvectors will be retained until cumulative eigenvalue score
            is greater than or equal to filter_th.
        out_format : str, optional
            Output format, by default 'numpy'.
            Must be 'numpy' or 'np' for output as numpy.ndarray (Note: name information
            will be lost), or 'pandas' or 'pd' for output as pandas.DataFrame.

        Returns
        -------
        Union[numpy.ndarray, pandas.DataFrame]
            Eigenvectors.

        Raises
        ------
        InvalidFilterError
            Raised if filter arguments are invalid.
        InvalidOutFormatError
            Raised if an invalid out_format is provided.
        '''
        available_filters = ('eigvals_cum', 'eigvals_rel_cum', 'eigvals_rel_corrected_cum')
        output_formats = ('numpy', 'np', 'pandas', 'pd')

        x = self.vectors

        if (not filter_by is None) or (not filter_th is None):
            if filter_th is None:
                raise InvalidFilterError('filter_th is required since filter_by is provided')
            if filter_by is None:
                raise InvalidFilterError('filter_by is required since filter_th is provided')

            if filter_by not in available_filters:
                msg = f'Invalid filter_by argument "{filter_by}". ' +\
                    f'filter_by must be one of {available_filters}.'
                raise InvalidFilterError(msg)

            idcs = (self.values[filter_by] < filter_th)[:x.shape[1]]
            x = x[:, idcs]

        if out_format not in output_formats:
            msg = f'Invalid out_format argument "{out_format}". ' +\
                f'out_format must be one of {output_formats}.'
            raise InvalidOutFormatError(msg)

        if out_format in ('pandas', 'pd'):
            x = pandas.DataFrame(x, index=self.names)

        return x

def _center_mat(dmat: numpy.ndarray) -> numpy.ndarray:
    '''_center_mat

    Center n*n matrix.

    Parameters
    ----------
    dmat : numpy.ndarray
        n*n matrix.

    Returns
    -------
    numpy.ndarray
        Centered matrix.
    '''

    n = dmat.shape[0]
    mat = numpy.full((n, n), -1/n)
    mat[numpy.diag_indices(n)] += 1

    return mat.dot(dmat).dot(mat)

class InvalidCorrectionTypeError(Exception):
    '''InvalidCorrectionTypeError'''

class NegativeEigenvaluesCorrectionError(Exception):
    '''FailedCorrectionError

    Error, signalling that the correction of negative eigenvalues failed.
    '''

class NegativeEigenvaluesWarning(Warning):
    '''NegativeEigenvaluesWarning

    Warning, signalling that negative eigenvalues were encountered.
    '''

def pcoa(
    dist: Union[numpy.ndarray, pyckmeans.distance.DistanceMatrix],
    correction: Optional[str] = None,
    eps: float = 1e-8,
) -> PCOAResult:
    '''pcoa

    Principle Coordinate Analysis.

    Parameters
    ----------
    dist : Union[numpy.ndarray, pyckmeans.distance.DistanceMatrix]
        n*n distance matrix either as numpy ndarray or as pyckmeans DistanceMatrix.
    correction: Optional[str]
        Correction for negative eigenvalues, by default None.
        Available corrections are:
            - None: negative eigenvalues are set to 0
            - lingoes: Lingoes correction
            - cailliez: Cailliet correction
    eps : float, optional
        Eigenvalues smaller than eps will be dropped. By default 0.0001

    Returns
    -------
    PCOAResult
        PCoA result object.

    Raises
    ------
    InvalidCorrectionTypeError
        Raised if an unknown correction type is passed.
    NegativeEigenvaluesCorrectionError
        Raised if correction parameter is set and correction of negative
        eigenvalues is not successful.
    '''
    names = None
    if isinstance(dist, pyckmeans.distance.DistanceMatrix):
        names = dist.names
        x = dist.dist_mat
    else:
        x = dist

    n = x.shape[0]

    # center matrix
    dmat_centered = _center_mat((x * x) / -2)
    trace = numpy.diag(dmat_centered).sum()

    # eigen decomposition
    eigvals, eigvecs = numpy.linalg.eigh(dmat_centered, 'U')

    # order descending
    ord_idcs = numpy.argsort(eigvals)[::-1]
    eigvals = eigvals[ord_idcs]
    eigvecs = eigvecs[:, ord_idcs]

    # get min eigenvalue
    min_eigval = numpy.min(eigvals)

    # set small eigenvalues to 0
    zero_eigval_idcs = numpy.nonzero(numpy.abs(eigvals) < eps)[0]
    eigvals[zero_eigval_idcs] = 0

    # no negative eigenvalues
    if min_eigval > -eps:

        eigvals_rel = eigvals / trace

        fze_idx = len(numpy.nonzero(eigvals > eps)[0]) # index of first zero in eigvals
        vectors = eigvecs[:, :fze_idx] * numpy.sqrt(eigvals[:fze_idx])

        return PCOAResult(
            vectors=vectors,
            eigvals=eigvals,
            eigvals_rel=eigvals_rel,
            trace=trace,
            negative_eigvals=False,
            names=names,
        )

    # negative eigenvalues
    else:
        eigvals_rel = eigvals / trace
        eigvals_rel_corrected = (eigvals - min_eigval) / (trace - (n - 1) * min_eigval)
        if len(zero_eigval_idcs) > 0:
            eigvals_rel_corrected = numpy.r_[
                numpy.delete(eigvals_rel_corrected, zero_eigval_idcs[0]), 0
            ]

        fze_idx = len(numpy.nonzero(eigvals > eps)[0]) # index of first zero in eigvals
        vectors = eigvecs[:, :fze_idx] * numpy.sqrt(eigvals[:fze_idx])

        # negative eigenvalues, no correction
        if not correction:
            warn(
                'Negative eigenvalues encountered but no correction applied. '
                'Negative eigenvalues will be treated as 0.',
                NegativeEigenvaluesWarning,
            )

            return PCOAResult(
                vectors=vectors,
                eigvals=eigvals,
                eigvals_rel=eigvals_rel,
                trace=trace,
                negative_eigvals=True,
                correction=None,
                eigvals_corr_rel=eigvals_rel_corrected,
                names=names,
            )

        # negative eigenvalues, correction
        if not correction in ['lingoes', 'cailliez']:
            msg = f'Unknown correction type "{correction}". ' +\
                'Available correction types are: "lingoes", "cailliez"'
            raise InvalidCorrectionTypeError(msg)

        # -- correct distance matrix
        # lingoes correction
        if correction == 'lingoes':
            corr_1 = -min_eigval

            # corrected distance matrix
            x_ncorr = -0.5 * ((x * x) + 2 * corr_1)
        elif correction == 'cailliez':
            dmat_centered_2 = _center_mat(-0.5 * x)

            # prepare matrix for correction
            upper = numpy.c_[numpy.zeros((x.shape[0], x.shape[0])), 2 * dmat_centered]
            lower = numpy.c_[numpy.diag(numpy.full(x.shape[0], -1)), -4 * dmat_centered_2]
            sp_mat = numpy.r_[upper, lower]

            corr_2 = numpy.max(numpy.real(numpy.linalg.eigvals(sp_mat)))

            # corrected distance matrix
            x_ncorr = -0.5 * (x + corr_2)**2

        # -- apply PCoA to corrected distance matrix
        x_ncorr[numpy.diag_indices(x_ncorr.shape[0])] = 0
        x_ncorr = _center_mat(x_ncorr)
        trace_ncorr = numpy.diag(x_ncorr).sum()

        eigvals_ncorr, eigvecs_ncorr = numpy.linalg.eigh(x_ncorr, 'U')

        # order descending
        ord_idcs_ncorr = numpy.argsort(eigvals_ncorr)[::-1]
        eigvals_ncorr = eigvals_ncorr[ord_idcs_ncorr]
        eigvecs_ncorr = eigvecs_ncorr[:, ord_idcs_ncorr]

        # get min eigenvalue
        min_eigval_ncorr = numpy.min(eigvals_ncorr)

        # set small eigenvalues to 0
        zero_eigval_idcs_ncorr = numpy.nonzero(numpy.abs(eigvals_ncorr) < eps)[0]
        eigvals_ncorr[zero_eigval_idcs_ncorr] = 0

        if min_eigval_ncorr < -eps:
            msg = 'Correction failed. There are still negative eigenvalues after applying ' +\
                f'{correction.capitalize()} correction.'
            raise NegativeEigenvaluesCorrectionError(msg)

        eigvals_ncorr_rel = eigvals_ncorr / trace_ncorr

        fze_idx_ncorr = len(numpy.nonzero(eigvals_ncorr > eps)[0]) # index of first zero in eigvals
        vectors_ncorr = eigvecs_ncorr[:, :fze_idx_ncorr] * numpy.sqrt(eigvals_ncorr[:fze_idx_ncorr])

        return PCOAResult(
            vectors=vectors_ncorr,
            eigvals=eigvals,
            eigvals_rel=eigvals_rel,
            trace=trace,
            negative_eigvals=True,
            correction=correction,
            eigvals_corr_rel=eigvals_ncorr_rel,
            trace_corr=trace_ncorr,
            names=names,
        )
