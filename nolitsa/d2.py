# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from scipy.spatial import distance
from nolitsa import utils


def c2(y, r=1000, metric='chebyshev', window=10):
    """Compute the correlation sum for the given distances.

    Computes the correlation sum of the given time series for the
    specified distances (Grassberger & Procaccia, 1983).

    We could've used scipy.spatial.KDTree.count_neighbors() for this.
    But then one cannot specify a minimum temporal separation which is
    crucial for overcoming the autocorrelation error.

    Parameters
    ----------
    y : ndarray
        Time series containing points in the phase space.
    r : int or array (default = 1000)
        Distances for which the correlation sum should be calculated.
        If `r` is an int, then the distances are taken to be a geometric
        progression between a minimum and maximum length scale
        (estimated according to the metric and the input series).
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "chebyshev" (aka the maximum norm metric), "cityblock" (aka the
        Manhattan metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between pairs.

    Returns
    -------
    r : array
        Distances for which correlation sums have been calculated.  Note
        that this might be different from the supplied `r` as only the
        ones with a nonzero ``C(r)`` is included.
    c : array
        Correlation sums for the given distances.
    """
    # Estimate the extent of the reconstructed phase space.
    if isinstance(r, int):
        if metric == 'chebyshev':
            extent = np.max(np.max(y, axis=0) - np.min(y, axis=0))
        elif metric == 'cityblock':
            extent = np.sum(np.max(y, axis=0) - np.min(y, axis=0))
        elif metric == 'euclidean':
            extent = np.sqrt(np.sum((np.max(y, axis=0) -
                                     np.min(y, axis=0)) ** 2))
        else:
            raise ValueError('Unknown metric.  Should be one of "chebyshev", '
                             '"cityblock", or "euclidean".')

        r = utils.gprange(extent / 10000, extent, r)
    else:
        r = np.asarray(r)
        r = np.sort(r[r > 0])

    bins = np.insert(r, 0, -1)
    c = np.zeros(len(r))
    n = len(y)

    for i in xrange(n - window - 1):
        dists = distance.cdist([y[i]], y[i + window + 1:], metric=metric)[0]
        c += np.histogram(dists, bins=bins)[0]

    pairs = 0.5 * (n - window - 1) * (n - window)
    c = np.cumsum(c) / pairs

    return r[c > 0], c[c > 0]


def c2_embed(x, dim=[1], tau=1, r=1000, metric='chebyshev', window=10,
             parallel=True):
    """Compute the correlation sum using time delayed vectors.

    Computes the correlation sum using time delayed vectors constructed
    from the time series.

    Parameters
    ----------
    x : array
        1D scalar time series.
    dim : int array (default = [1])
        Embedding dimensions for which the correlation sum should be
        computed.
    tau : int, optional (default = 1)
        Time delay.
    r : int or array (default = 1000)
        Distances for which the correlation sum should be calculated.
        If `r` is an int, then the distances are taken to be a geometric
        progression between a minimum and maximum length scale
        (estimated according to the metric and the input series).
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "chebyshev" (aka the maximum norm metric), "cityblock" (aka the
        Manhattan metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between pairs.
    parallel : bool, optional (default = True)
        Calculate the correlation sum for each embedding dimension in
        parallel.

    Returns
    -------
    rc : ndarray
        The output is an array with shape ``(len(dim), 2, len(r))`` of
        ``(r, C(r))`` pairs for each dimension.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    yy = [utils.reconstruct(x, dim=d, tau=tau) for d in dim]

    return utils.parallel_map(c2, yy, kwargs={
                              'r': r,
                              'metric': metric,
                              'window': window,
                              }, processes=processes)


def d2(r, c, hwin=3):
    """Compute D2 using a local least square fit.

    Computes D2 using a local least square fit of the equation C(r) ~
    r^D2.  D2 at each point is computed by doing a least square fit
    inside a window of size 2*hwin + 1 around it.  The idea is analogous
    to that of a simple moving average.

    Parameters
    ----------
    r : array
        Distances for which correlation sums have been calculated.
    c : array
        Correlation sums for the given distances.
    hwin : int, optional (default = 3)
        Half-window length.  Actual window size is 2*hwin + 1.

    Returns
    -------
    d : array
        Average D2 at each r in `r[hwin:-hwin]`
    """
    N = len(r) - 2 * hwin

    d = np.empty(N)
    x, y = np.log(r), np.log(c)

    for i in xrange(N):
        p, q = x[i:i + 2 * hwin + 1], y[i:i + 2 * hwin + 1]
        A = np.vstack([p, np.ones(2 * hwin + 1)]).T
        d[i] = np.linalg.lstsq(A, q)[0][0]

    return d
