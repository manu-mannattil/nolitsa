# -*- coding: utf-8 -*-

"""Functions to estimate the maximum Lyapunov exponent.

This module provides two functions to estimate the maximum Lyapunov
exponent (MLE) from a scalar and vector time series.

  * mle -- estimate the MLE from a vector time series
  * mle_embed -- estimate the MLE from a scalar time series after
    reconstruction.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from . import utils


def mle(y, maxt=500, window=10, metric='euclidean', maxnum=None):
    """Estimate the maximum Lyapunov exponent.

    Estimates the maximum Lyapunov exponent (MLE) from a
    multi-dimensional series using the algorithm described by
    Rosenstein et al. (1993).

    Parameters
    ----------
    y : ndarray
        Multi-dimensional real input array containing points in the
        phase space.
    maxt : int, optional (default = 500)
        Maximum time (iterations) up to which the average divergence
        should be computed.
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors (see Notes).
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

    Returns
    -------
    d : array
        Average divergence for each time up to maxt.

    Notes
    -----
    This function does not directly estimate the MLE.  The MLE should be
    estimated by linearly fitting the average divergence (i.e., the
    average of the logarithms of near-neighbor distances) with time.
    It is also important to choose an appropriate Theiler window so that
    the near neighbors do not lie on the same trajectory, in which case
    the estimated MLE will always be close to zero.
    """
    index, dist = utils.neighbors(y, metric=metric, window=window,
                                  maxnum=maxnum)
    m = len(y)
    maxt = min(m - window - 1, maxt)

    d = np.empty(maxt)
    d[0] = np.mean(np.log(dist))

    for t in range(1, maxt):
        t1 = np.arange(t, m)
        t2 = index[:-t] + t

        # Sometimes the nearest point would be farther than (m - maxt)
        # in time.  Such trajectories needs to be omitted.
        valid = t2 < m
        t1, t2 = t1[valid], t2[valid]

        d[t] = np.mean(np.log(utils.dist(y[t1], y[t2], metric=metric)))

    return d


def mle_embed(x, dim=[1], tau=1, window=10, maxt=500,
              metric='euclidean', maxnum=None, parallel=True):
    """Estimate the maximum Lyapunov exponent from a scalar time series.

    Estimates the maximum Lyapunov exponent (MLE) using time-delayed
    vectors created from a scalar time series (Rosenstein et al. 1993).

    Parameters
    ----------
    x : ndarray
        1-D real input array containing the time series.
    dim : int array, optional (default = [1])
        Embedding dimensions for which the average divergence should be
        computed.
    tau : int, optional (default = 1)
        Time delay.
    maxt : int, optional (default = 500)
        Maximum time (iterations) up to which the average divergence
        should be computed.
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors (see Notes).
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).
    parallel : bool, optional (default = True)
        Compute the average divergence for each embedding dimension in
        parallel.

    Returns
    -------
    d : array
        Average divergence for each time up to maxt, for each embedding
        dimension.

    Notes
    -----
    This function does not directly estimate the MLE.  The MLE should be
    estimated by linearly fitting the average divergence (i.e., the
    average of the logarithms of near-neighbor distances) with time.
    It is also important to choose an appropriate Theiler window so that
    the near neighbors do not lie on the same trajectory, in which case
    the estimated MLE will always be close to zero.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    yy = [utils.reconstruct(x, dim=d, tau=tau) for d in dim]

    return utils.parallel_map(mle, yy, kwargs={
                              'maxt': maxt,
                              'window': window,
                              'metric': metric,
                              'maxnum': maxnum
                              }, processes=processes)
