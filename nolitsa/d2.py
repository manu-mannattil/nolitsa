# -*- coding: utf-8 -*-

"""Functions to estimate correlation sums and dimensions.

This module provides functions to estimate the correlation sum and the
correlation dimension from both scalar and vector time series.

Correlation Sum
---------------

  * c2 -- estimates the correlation sum from a vector time series.
  * c2_embed -- estimates the correlation sum from a scalar time series
    after embedding.

Correlation Dimension
---------------------

  * d2 -- estimates the "local" correlation dimension from correlation
    sums and distances using a local least squares fit.
  * ttmle -- estimates the correlation dimension from correlation sums
    and distances using a maximum likelihood estimator.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from scipy.spatial import distance
from . import utils


def c2(y, r=100, metric='chebyshev', window=10):
    """Compute the correlation sum for the given distances.

    Computes the correlation sum of the given time series for the
    specified distances (Grassberger & Procaccia 1983).

    Parameters
    ----------
    y : ndarray
        N-dimensional real input array containing points in the phase
        space.
    r : int or array, optional (default = 100)
        Distances for which the correlation sum should be calculated.
        If r is an int, then the distances are taken to be a geometric
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
        that this might be different from the supplied r as only the
        distances with a nonzero C(r) are included.
    c : array
        Correlation sums for the given distances.

    Notes
    -----
    This function is meant to be used to calculate the correlation sum
    from an array of points in the phase space.  If you want to
    calculate it after embedding a time series, see d2_embed().
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

        r = utils.gprange(extent / 1000, extent, r)
    else:
        r = np.asarray(r)
        r = np.sort(r[r > 0])

    bins = np.insert(r, 0, -1)
    c = np.zeros(len(r))
    n = len(y)

    for i in range(n - window - 1):
        dists = distance.cdist([y[i]], y[i + window + 1:], metric=metric)[0]
        c += np.histogram(dists, bins=bins)[0]

    pairs = 0.5 * (n - window - 1) * (n - window)
    c = np.cumsum(c) / pairs

    return r[c > 0], c[c > 0]


def c2_embed(x, dim=[1], tau=1, r=100, metric='chebyshev', window=10,
             parallel=True):
    """Compute the correlation sum using time-delayed vectors.

    Computes the correlation sum using time-delayed vectors constructed
    from a time series.

    Parameters
    ----------
    x : array
        1-D real input array containing the time series.
    dim : int array, optional (default = [1])
        Embedding dimensions for which the correlation sums ought to be
        computed.
    tau : int, optional (default = 1)
        Time delay.
    r : int or array, optional (default = 100)
        Distances for which the correlation sum should be calculated.
        If r is an int, then the distances are taken to be a geometric
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
        Calculate the correlation sums for each embedding dimension in
        parallel.

    Returns
    -------
    rc : ndarray
        The output is an array with shape (len(dim), 2, len(r)) of
        (r, C(r)) pairs for each dimension.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    yy = [utils.reconstruct(x, dim=d, tau=tau) for d in dim]

    return utils.parallel_map(c2, yy, kwargs={
                              'r': r,
                              'metric': metric,
                              'window': window
                              }, processes=processes)


def d2(r, c, hwin=3):
    """Compute D2 using a local least squares fit.

    Computes D2 using a local least squares fit of the equation
    C(r) ~ r^D2.  D2 at each point is computed by doing a least
    squares fit inside a window of size 2*hwin + 1 around it (Galka
    2000).

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
        Average D2 at each distance in r[hwin:-hwin]
    """
    N = len(r) - 2 * hwin

    d = np.empty(N)
    x, y = np.log(r), np.log(c)

    for i in range(N):
        p, q = x[i:i + 2 * hwin + 1], y[i:i + 2 * hwin + 1]
        A = np.vstack([p, np.ones(2 * hwin + 1)]).T
        d[i] = np.linalg.lstsq(A, q)[0][0]

    return d


def ttmle(r, c, zero=True):
    """Compute the Takens-Theiler maximum likelihood estimator.

    Computes the Takens-Theiler maximum likelihood estimator (MLE) for
    a given set of distances and the corresponding correlation sums
    (Theiler 1990).  The MLE is calculated by assuming that C(r) obeys
    a true power law between adjacent r's.

    Parameters
    ----------
    r : array
        Distances for which the correlation sums have been calculated.
    c : array
        Correlation sums for the given distances.
    zero : bool, optional (default = True)
        Integrate the MLE starting from zero (see Notes).

    Returns
    -------
    r : array
        Distances at which the Takens-Theiler MLE has been computed.
    d : array
        Takens-Theiler MLE for the given distances.

    Notes
    -----
    Integrating the expression for MLE from zero has the advantage that
    for a true power law of the from C(r) ~ r^D, the MLE gives D as the
    estimate for all values of r.  Some implementations (e.g., TISEAN,
    Hegger et al. 1999) starts the integration only from the minimum
    distance supplied.  In any case, this does not make much difference
    as the only real use of a "dimension" estimator is as a statistic
    for surrogate testing.
    """
    # Prune the arrays so that only unique correlation sums remain.
    c, i = np.unique(c, return_index=True)
    r = r[i]

    x1, y1 = np.log(r[:-1]), np.log(c[:-1])
    x2, y2 = np.log(r[1:]), np.log(c[1:])

    a = (y2 - y1) / (x2 - x1)
    b = (y1 * x2 - y2 * x1) / (x2 - x1)

    # To integrate, we use the discrete expression (Eq. 24) given in
    # the TISEAN paper (Hegger et al. 1999).
    denom = np.cumsum(np.exp(b) / a * (r[1:] ** a - r[:-1] ** a))

    if zero:
        # Assume that the power law between r[0] and r[1] holds
        # between 0 and r[0].
        denom = np.insert(denom, 0, np.exp(b[0]) / a[0] * r[0] ** a[0])
        denom[1:] = denom[1:] + denom[0]
        return r, c / denom
    else:
        return r[1:], c[1:] / denom
