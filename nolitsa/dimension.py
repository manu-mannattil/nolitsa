# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from nolitsa import utils


def _afn_parallel(d, x, tau=1, metric='euclidean', window=10, maxnum=-1):
    """Return E(d) and E*(d) for a single d.

    Returns E(d) and E*(d) for the AFN method for a single d.  This
    function is meant to be called from the main `afn` function.  See
    the docstring of `afn` for more.
    """
    # We need to reduce the number of points in dimension `d` by `tau`
    # so that after reconstruction, there'll be equal number of points
    # in both dimension `d` as well as dimension `d + 1`.
    y1 = utils.reconstruct(x[:-tau], d, tau)
    y2 = utils.reconstruct(x, d + 1, tau)

    # Find near neighbors in dimension `d`.
    index, dist = utils.neighbors(y1, metric=metric, num=1, window=window,
                                  maxnum=maxnum)

    # Compute the magnification and the increase in near-neighbor
    # distances and return the averages.
    E = utils.dist(y2, y2[index], metric=metric) / dist
    Es = np.abs(y2[:, -1] - y2[index, -1])

    return np.mean(E), np.mean(Es)


def afn(x, dim=[1], tau=1, metric='euclidean', window=10, maxnum=-1,
        parallel=True):
    """Averaged false neighbors algorithm.

    This function implements the averaged false neighbors method
    described by Cao (1997) to calculate the minimum embedding dimension
    required to embed a scalar time series.

    Parameters
    ----------
    x : array
        1D scalar time series.
    dim : int, array (default = [1])
        Embedding dimensions for which E(d) and E*(d) should be
        computed.
    tau : int, optional (default = 1)
        Time delay.
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.
    maxnum : int, optional (default = -1 (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors which have a
        non-zero distance, this will have to be increased (i.e., beyond
        (num + 2 * window + 2)).
    parallel : bool, optional (default = True)
        Calculate E(d) and E*(d) for each d in parallel.

    Returns
    -------
    E : array
        E(d) for each of the d's.
    Es : array
        E*(d) for each of the d's.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    return utils.parallel_map(_afn_parallel, dim, (x,), {
                              'tau': tau,
                              'metric': metric,
                              'window': window,
                              'maxnum': maxnum
                              }, processes).T
