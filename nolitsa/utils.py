# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance


def dist(x, y, metric='euclidean'):
    """Compute the distance between all sequential pairs of points.

    Computes the distance between all sequential pairs of points from
    two arrays using scipy.spatial.distance.

    Paramters
    ---------
    x : ndarray
        Input array.
    y : ndarray
        Input array.

    Returns
    -------
    d : ndarray
        Array containing distances.
    """
    func = getattr(distance, metric)
    return np.asarray([func(i, j) for i, j in zip(x, y)])


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
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    num : int, optional (default = 1)
        Number of near neighbors that should be found for each point.
    window : int, optional (default = 0)
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
        Array containing indices of near neighbors.
    dist : array
        Array containing near neighbor distances.
    """
    if metric == 'cityblock':
        p = 1
    elif metric == 'euclidean':
        p = 2
    elif metric == 'chebyshev':
        p = np.inf
    else:
        raise ValueError('Unknown metric.  Should be one of "cityblock", '
                         '"euclidean", or "chebyshev".')

    tree = KDTree(y)
    n = len(y)

    # In most cases a nonzero neighbor will be found when maxnum = 10.
    maxnum = max(10, maxnum, num + 2 * window + 2)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        for k in xrange(num + 1, maxnum + 1):
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid) == num:
                dists[i] = dist[valid]
                indices[i] = index[valid]
                break

            if k == maxnum:
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)


def parallel_map(func, values, args=tuple(), kwargs=dict(),
                 processes=None):
    """Use `Pool.apply_async` to get a parallel map().

    Uses `Pool.apply_async` to provide a parallel version of map().
    Unlike Pool's map() which does not let you accept arguments and/or
    keyword arguments, this one does.

    Parameters
    ----------
    func : function
        This function will be applied on every element in `values` in
        parallel.
    values : array
        Input array.
    args : tuple, optional (default: ())
        Additional arguments for `func`.
    kwargs : dictionary, optional (default: {})
        Additional keyword arguments for `func`.
    processes : int, optional (default: None)
        Number of processes to run in parallel.  By default, the output
        of `cpu_count()` is used.

    Returns
    -------
    results : array
        Output after applying `func` on each element in `values`.
    """
    from multiprocessing import Pool

    pool = Pool(processes=processes)
    results = [pool.apply_async(func, (value,) + args, kwargs)
               for value in values]

    pool.close()
    pool.join()

    return np.asarray([result.get() for result in results])


def reconstruct(x, dim=1, tau=1):
    """Construct time delayed vectors from a time series.

    Constructs n-dimensional time delayed vectors from a scalar time
    series.

    Parameters
    ----------
    x : array
        1D scalar time series.
    dim : int, optional (default = 1)
        Embedding dimension (n).
    tau : int, optional (default = 1)
        Time delay

    Returns
    -------
    ps : ndarray
        The reconstructed n-dimensional phase space.
    """
    m = len(x) - (dim - 1) * tau
    if m <= 0:
        raise ValueError('Length of the time series is <= (dim - 1) * tau.')

    return np.asarray([x[i:i + (dim - 1) * tau + 1:tau] for i in range(m)])


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
