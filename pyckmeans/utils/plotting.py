''' Plotting utitlies
'''

from pyckmeans.core import wecr
from typing import Iterable, Optional, Tuple, Union
import numpy
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.colors
import matplotlib.axes
import pyckmeans.core

def plot_ckmeans_result(
    ckm_res: pyckmeans.core.CKmeansResult,
    names: Optional[Iterable[str]] = None,
    order: Optional[Union[str, numpy.ndarray]] = 'GW',
    cmap_cm: Union[str, matplotlib.colors.Colormap] = 'Blues',
    cmap_clbar: Union[str, matplotlib.colors.Colormap] = 'tab20',
    figsize: Tuple[float, float] = (7, 7),
) -> matplotlib.figure.Figure:
    '''plot_ckmeans_result

    Plot pyckmeans result consensus matrix with consensus clusters.

    Parameters
    ----------
    ckm_res : CKmeansResult
        CKmeansResult as returned from CKmeans.predict.
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
    # if order is None do not reorder
    if order is None:
        order = numpy.arange(ckm_res.cmatrix.shape[0])
    # if order is str use CKMeansResult order
    elif isinstance(order, str):
        order = ckm_res.order(method=order)
    # else order must be numpy.ndarray giving the sample order

    ckm_res = ckm_res.reorder(order=order, in_place=False)
    cl = ckm_res.cl

    # if names is passed use names, else try to get names
    # from ckm_res, else just use samples indices
    if names is None:
        if ckm_res.names is not None:
            nms = ckm_res.names
        else:
            nms = order.astype('str')
    else:
        nms = numpy.array(names)[order]

    # build figure layout
    fig = plt.figure(figsize=figsize)
    ax_cmat = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_clbar = fig.add_axes([0.05, 0.1, 0.05, 0.8])
    ax_cbar = fig.add_axes([0.925, 0.1, 0.025, 0.8])

    # = consensus matrix
    ax_cmat.imshow(ckm_res.cmatrix, cmap=cmap_cm)
    ax_cmat.set_xticks(numpy.arange(len(nms)))
    ax_cmat.set_xticklabels(nms)
    for tick in ax_cmat.get_xticklabels():
        tick.set_rotation(90)
    ax_cmat.set_yticks([])
    ax_cmat.tick_params(left=False)

    # cluster lines
    cl_01 = []
    cl_start = 0
    for i in range(1, len(cl)):
        if cl[i] != cl[cl_start]:
            cl_01.append((cl_start, i))
            cl_start = i
    cl_01.append((cl_start, len(cl)))
    cl_01 = numpy.array(cl_01)

    ax_cmat.hlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(nms) - 0.5, color='white', linewidth=2)
    ax_cmat.vlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(nms) - 0.5, color='white', linewidth=2)

    # = cluster membership bar
    ax_clbar.imshow(ckm_res.cl.reshape(-1, 1), cmap=cmap_clbar)
    ax_clbar.set_xticks([])
    ax_clbar.set_yticks(numpy.arange(len(nms)))
    ax_clbar.set_yticklabels(nms)

    # = color bar
    ax_cbar.set_xticks([])
    ax_cbar.yaxis.tick_right()
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_cm), cax=ax_cbar)

    return fig

def plot_multickmeans_metrics(
    mckm_res: pyckmeans.core.MultiCKmeansResult,
    figsize: Tuple[float, float] = (7, 7),
) -> matplotlib.figure.Figure:
    '''plot_multickmeans_metrics

    Plot MultiCKMeansResult metrics.

    Parameters
    ----------
    mckm_res : MultiCKmeansResult
        MultiCKmeansResult object
    figsize : Tuple[float, float], optional
        Figure size for the matplotlib figure, by default (7, 7).

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure of the metrics plot.
    '''

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    axs = axs.flatten()
    for ax in axs:
        ax.grid(axis='x')
        ax.set_xticks(mckm_res.ks)
        ax.set_xlabel('k')

    axs[0].plot(mckm_res.ks, mckm_res.bics)
    axs[0].set_title('BIC')
    axs[0].set_ylabel('BIC')

    axs[1].plot(mckm_res.ks, mckm_res.dbs)
    axs[1].set_title('DB')
    axs[1].set_ylabel('DB')

    axs[2].plot(mckm_res.ks, mckm_res.sils)
    axs[2].set_title('SIL')
    axs[2].set_ylabel('SIL')

    axs[3].plot(mckm_res.ks, mckm_res.chs)
    axs[3].set_title('CH')
    axs[3].set_ylabel('CH')

    return fig

def plot_wecr_result(
    wecr_res: pyckmeans.core.WECRResult,
    k: int,
    names: Optional[Iterable[str]] = None,
    order: Optional[Union[str, numpy.ndarray]] = 'GW',
    cmap_cm: Union[str, matplotlib.colors.Colormap] = 'Blues',
    cmap_clbar: Union[str, matplotlib.colors.Colormap] = 'tab20',
    figsize: Tuple[float, float] = (7, 7),
) -> matplotlib.figure.Figure:
    '''plot_wecr_result

    Plot wecr result consensus matrix with consensus clusters.

    Parameters
    ----------
    wecr_res : pyckmeans.core.WECRResult
        WECRResult as returned from pyckmeans.core.WECR.predict.
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

    Raises
    ------
    wecr.InvalidKError
        Raised if an invalid k argument is provided.
    '''
    # if order is None do not reorder
    if order is None:
        order = numpy.arange(wecr_res.cmatrix.shape[0])
    # if order is str use WECRResult order
    elif isinstance(order, str):
        order = wecr_res.order(method=order)
    # else order must be numpy.ndarray giving the sample order

    wecr_res = wecr_res.reorder(order=order, in_place=False)
    if not k in wecr_res.k:
        msg = f'Result for k={k} not found. Available k are {wecr_res.k}.'
        raise wecr.InvalidKError(msg)
    cl = wecr_res.cl[numpy.argmax(wecr_res.k == k)]

    # if names is passed use names, else try to get names
    # from wecr_res, else just use samples indices
    if names is None:
        if wecr_res.names is not None:
            nms = wecr_res.names
        else:
            nms = order.astype('str')
    else:
        nms = numpy.array(names)[order]

    # build figure layout
    fig = plt.figure(figsize=figsize)
    ax_cmat = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_clbar = fig.add_axes([0.05, 0.1, 0.05, 0.8])
    ax_cbar = fig.add_axes([0.925, 0.1, 0.025, 0.8])

    # = consensus matrix
    ax_cmat.imshow(wecr_res.cmatrix, cmap=cmap_cm)
    ax_cmat.set_xticks(numpy.arange(len(nms)))
    ax_cmat.set_xticklabels(nms)
    for tick in ax_cmat.get_xticklabels():
        tick.set_rotation(90)
    ax_cmat.set_yticks([])
    ax_cmat.tick_params(left=False)

    # cluster lines
    cl_01 = []
    cl_start = 0
    for i in range(1, len(cl)):
        if cl[i] != cl[cl_start]:
            cl_01.append((cl_start, i))
            cl_start = i
    cl_01.append((cl_start, len(cl)))
    cl_01 = numpy.array(cl_01)

    ax_cmat.hlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(nms) - 0.5, color='white', linewidth=2)
    ax_cmat.vlines(cl_01[:, 0] + 0.5 - 1, -0.5, len(nms) - 0.5, color='white', linewidth=2)

    # = cluster membership bar
    ax_clbar.imshow(cl.reshape(-1, 1), cmap=cmap_clbar)
    ax_clbar.set_xticks([])
    ax_clbar.set_yticks(numpy.arange(len(nms)))
    ax_clbar.set_yticklabels(nms)

    # = color bar
    ax_cbar.set_xticks([])
    ax_cbar.yaxis.tick_right()
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_cm), cax=ax_cbar)

    return fig

def plot_wecr_result_metrics(
    wecr_res: pyckmeans.core.WECRResult,
    figsize: Tuple[float, float] = (7, 7),
) -> matplotlib.figure.Figure:
    '''plot_wecr_result_metrics

    Plot WECRResult metrics.

    Parameters
    ----------
    wecr_res : WECRResult
        WECRResult object
    figsize : Tuple[float, float], optional
        Figure size for the matplotlib figure, by default (7, 7).

    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib Figure of the metrics plot.
    '''

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)

    order = numpy.argsort(wecr_res.k)

    axs = axs.flatten()
    for ax in axs:
        ax.grid(axis='x')
        ax.set_xticks(wecr_res.k[order])
        ax.set_xlabel('k')

    axs[0].plot(wecr_res.k[order], wecr_res.bic[order])
    axs[0].set_title('BIC')
    axs[0].set_ylabel('BIC')

    axs[1].plot(wecr_res.k[order], wecr_res.db[order])
    axs[1].set_title('DB')
    axs[1].set_ylabel('DB')

    axs[2].plot(wecr_res.k[order], wecr_res.sil[order])
    axs[2].set_title('SIL')
    axs[2].set_ylabel('SIL')

    axs[3].plot(wecr_res.k[order], wecr_res.ch[order])
    axs[3].set_title('CH')
    axs[3].set_ylabel('CH')

    return fig
