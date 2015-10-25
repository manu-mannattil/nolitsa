# -*- coding: utf-8 -*-

from __future__ import division
from scipy.spatial import cKDTree as KDTree
import numpy as np


def gprange(start, end, num=100):
    """Return a geometric progression between start and end.

    Returns a geometric progression between start and end (inclusive).

    Parameters
    ----------
    start : float
        Starting point of the progression.
    end : float
        Ending point of the progression.
    num : int, optional (default = 100)
        Number of points between start and end (inclusive).

    Returns
    -------
    gp : array
        Required geometric progression.
    """
    if end / start > 0:
        ratio = (end / start) ** (1.0 / (num - 1))
    elif end / start < 0 and num % 2 == 0:
        ratio = -abs(end / start) ** (1.0 / (num - 1))
    else:
        raise ValueError('If start and end have different signs, '
                         'a real ratio is possible iff num is even.')

    return start * ratio ** np.arange(num)


def neighbors(y, metric='euclidean', num=1, window=0, maxnum=-1):
    """Find nearest neighbors to all points in the given array.

    Finds nearest neighbors to all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time delayed vectors.
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "manhattan", "euclidean", or "chebyshev".
    num : int, optional (default = 1)
        Number of near neighbors that should be found for each point.
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents.
    maxnum : int, optional (default = -1 (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors which have a
        nonzero distance, this will have to be increased (i.e., beyond
        (num + 2 * window + 2)).

    Returns
    -------
    index : array
        1D array containing indices of near neighbors.
    dist : array
        1D array containing near neighbor distances.
    """
    if metric == 'manhattan':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf
    else:
        raise ValueError('Unknown metric.  Should be one of "manhattan", '
                         '"euclidean", or "chebyshev".')

    tree = KDTree(y)

    dists = list()
    indices = list()

    # In most cases a nonzero neighbor will be found when maxnum = 10.
    maxnum = max(10, maxnum, num + 2 * window + 2)

    if maxnum >= len(y):
        raise ValueError('maxnum is bigger than array length.')

    dists = list()
    indices = list()

    for i, x in enumerate(y):
        for k in xrange(num + 1, maxnum + 1):
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid) == num:
                indices.append(index[valid])
                dists.append(dist[valid])
                break
            if k == maxnum:
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)


def rescale(x, interval=(0, 1)):
    """Rescale the given scalar time series into a desired interval.

    Rescales the given scalar time series into a desired interval using
    a simple linear transformation.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    interval: tuple, optional (default = (0, 1))
        Extent of the interval specified as a tuple.

    Returns
    -------
    y : array
        Rescaled scalar time series.
    """
    x = np.asarray(x)
    if interval[1] == interval[0]:
        raise ValueError('Interval must have a nonzero length.')

    return (interval[0] + (x - np.min(x)) * (interval[1] - interval[0]) /
            (np.max(x) - np.min(x)))
