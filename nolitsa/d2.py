# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from scipy.spatial import distance
from nolitsa import utils


def d2(y, r=100, metric='euclidean', window=10):
    """Calculate the correlation sum for the given distances.

    Calculates the correlation sum C(r) for the given time series at the
    specified distances r (Grassberger & Procaccia, 1983).

    Parameters
    ----------
    y : ndarray
        Time series containing points in the phase space.
    r : int or array (default = 100)
        r values at which C(r) should be computed.  If `r` is an int,
        the r's are taken to be a geometric progression between a
        minimum and maximum length scale (estimated according to the
        metric and the input series).
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "chebyshev" (aka the maximum norm metric), "cityblock" (aka the
        Manhattan metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between pairs.

    Returns
    -------
    r : array
        Distances for which correlation sums have been calculated.
    c : array
        Correlation sums for the given r's.
    """
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

        r = utils.gprange(extent / 1000, extent, r)
    else:
        r = np.asarray(r)
        r = np.sort(r[r > 0])

    bins = np.insert(r, 0, -1)
    c = np.zeros(len(r))
    n = len(y)

    for i in xrange(n - window - 1):
        dists = distance.cdist([y[i]], y[i + window + 1:], metric=metric)[0]
        c += np.histogram(dists, bins=bins)[0]

    pairs = 0.5 * (len(y) - window - 1) * (len(y) - window)
    return r, np.cumsum(c) / pairs
