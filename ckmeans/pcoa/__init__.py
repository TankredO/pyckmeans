''' pcoa

    Module for Principle Coordinate Analysis (PCoA) calculation.
'''

# Inspired by https://github.com/cran/ape/blob/master/R/pcoa.R and
# https://github.com/biocore/scikit-bio/blob/0.5.4/skbio/stats/ordination/_principal_coordinate_analysis.py#L23

from warnings import warn
from typing import Iterable, Optional, Tuple, Dict, Union

import numpy
import pandas

import ckmeans.distance

class InvalidPCOAResultError(Exception):
    '''InvalidPCOAResultError'''

class IncompatibleNamesError(Exception):
    '''IncompatibleNamesError'''

class PCOAResult:
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

            self.names = list(names)

    def __repr__(self) -> str:
        str_repr = f'<PCOAResult; neg. eigvals: {self.negative_eigvals}, ' +\
            f'correction: {self.correction}>'

        return str_repr


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
    '''FailedCorrectionError'''

class NegativeEigenvaluesWarning(Warning):
    '''NegativeEigenvaluesWarning'''

def pcoa(
    dist: Union[numpy.ndarray, ckmeans.distance.DistanceMatrix],
    correction: Optional[str] = None,
    eps: float = 0.0001
) -> Tuple[numpy.ndarray, Dict]:
    '''pcoa

    Principle Component Analysis.

    Parameters
    ----------
    x : Union[numpy.ndarray, ckmeans.distance.DistanceMatrix]
        n*n distance matrix either as numpy ndarray or as ckmeans DistanceMatrix.
    correction: Optional[str]
        Correction for negative eigenvalues, by default None.
        Available corrections are:
            - None: negative eigenvalues are set to 0
            - lingoes: Lingoes correction
            - cailliez: Cailliet correction
    eps : float, optional
        Epsilon, by default 0.0001

    Returns
    -------
    PCOAResult
        PCOA result object.

    Raises
    ------
    InvalidCorrectionTypeError
        Raised if an unknown correction type is passed.
    NegativeEigenvaluesCorrectionError
        Raised if correction parameter is set and correction of negative
        eigenvalues is not successful.
    '''
    names = None
    if isinstance(dist, ckmeans.distance.DistanceMatrix):
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
                'Negative eigenvalues encountered but no correction applied. ' +\
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
                names = names,
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
            names = names,
        )
