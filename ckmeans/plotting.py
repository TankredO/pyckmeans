''' Plotting module
'''

from typing import Iterable, Optional, Tuple, Union
import numpy
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.colors
from ckmeans.core import CKmeansResult

def plot_ckmeans_result(
    ckm_res: 'CKmeansResult',
    names: Optional[Iterable[str]] = None,
    cmap_cm: Union[str, matplotlib.colors.Colormap] = 'Blues',
    cmap_clbar: Union[str, matplotlib.colors.Colormap] = 'tab20',
    figsize: Tuple[float, float] = (7, 7),
) -> matplotlib.figure.Figure:
    '''plot_ckmeans_result

    Plot ckmeans result consensus matrix with consensus clusters.

    Parameters
    ----------
    ckm_res : CKmeansResult
        CKmeansResult as returned from CKmeans.predict.
    names : Optional[Iterable[str]]
        Sample names to be plotted.
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
    order = ckm_res.order() # need to store order for name ordering
    ckm_res = ckm_res.sort(in_place=False)
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
