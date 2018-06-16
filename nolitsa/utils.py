# -*- coding: utf-8 -*-

"""Miscellaneous utility functions.

A module for common utility functions used elsewhere.

  * corrupt -- corrupts a time series with noise.
  * dist -- computes the distance between points from two arrays.
  * gprange -- generates a geometric progression between two points.
  * neighbors -- finds the nearest neighbors of all points in an array.
  * parallel_map -- a parallel version of map().
  * reconstruct -- constructs time-delayed vectors from a scalar time
    series.
  * rescale -- rescales a scalar time series into a desired interval.
  * spectrum -- returns the power spectrum of a scalar time series.
  * statcheck -- checks if a time series is stationary.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from scipy import stats
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import distance


def corrupt(x, y, snr=100):
    """Corrupt time series with noise.

    Corrupts input time series with supplied noise to obtain a series
    with the specified signal-to-noise ratio.

    Parameters
    ----------
    x : array
        1-D array with scalar time series (the 'signal').
    y : ndarray
        1-D array with noise (the 'noise').
    snr : float, optional (default = 100).
        Signal-to-noise ratio (SNR) (see Notes).

    Returns
    -------
    x : array
        1-D array with corrupted series.

    Notes
    -----
    Contrary to the convention used in engineering sciences, here SNR is
    defined as the ratio of the variance of the signal to the variance
    of the noise.  The noise is also assumed to have zero mean.
    """
    if len(x) != len(y):
        raise ValueError('Signal and noise arrays should be of equal length.)')

    y = y - np.mean(y)
    return x + (np.std(x) / np.sqrt(snr)) * (y / np.std(y))


def dist(x, y, metric='chebyshev'):
    """Compute the distance between all sequential pairs of points.

    Computes the distance between all sequential pairs of points from
    two arrays using scipy.spatial.distance.

    Paramters
    ---------
    x : ndarray
        Input array.
    y : ndarray
        Input array.
    metric : string, optional (default = 'chebyshev')
        Metric to use while computing distances.

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


def neighbors(y, metric='chebyshev', window=0, maxnum=None):
    """Find nearest neighbors of all points in the given array.

    Finds the nearest neighbors of all points in the given array using
    SciPy's KDTree search.

    Parameters
    ----------
    y : ndarray
        N-dimensional array containing time-delayed vectors.
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    window : int, optional (default = 0)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.  This is crucial while computing
        Lyapunov exponents and the correlation dimension.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).

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

    if not maxnum:
        maxnum = (window + 1) + 1 + (window + 1)
    else:
        maxnum = max(1, maxnum)

    if maxnum >= n:
        raise ValueError('maxnum is bigger than array length.')

    dists = np.empty(n)
    indices = np.empty(n, dtype=int)

    for i, x in enumerate(y):
        for k in range(2, maxnum + 2):
            dist, index = tree.query(x, k=k, p=p)
            valid = (np.abs(index - i) > window) & (dist > 0)

            if np.count_nonzero(valid):
                dists[i] = dist[valid][0]
                indices[i] = index[valid][0]
                break

            if k == (maxnum + 1):
                raise Exception('Could not find any near neighbor with a '
                                'nonzero distance.  Try increasing the '
                                'value of maxnum.')

    return np.squeeze(indices), np.squeeze(dists)


def parallel_map(func, values, args=tuple(), kwargs=dict(),
                 processes=None):
    """Use Pool.apply_async() to get a parallel map().

    Uses Pool.apply_async() to provide a parallel version of map().
    Unlike Pool's map() which does not let you accept arguments and/or
    keyword arguments, this one does.

    Parameters
    ----------
    func : function
        This function will be applied on every element of values in
        parallel.
    values : array
        Input array.
    args : tuple, optional (default: ())
        Additional arguments for func.
    kwargs : dictionary, optional (default: {})
        Additional keyword arguments for func.
    processes : int, optional (default: None)
        Number of processes to run in parallel.  By default, the output
        of cpu_count() is used.

    Returns
    -------
    results : array
        Output after applying func on each element in values.
    """
    # True single core processing, in order to allow the func to be executed in
    # a Pool in a calling script.
    if processes == 1:
        return np.asarray([func(value, *args, **kwargs) for value in values])

    from multiprocessing import Pool

    pool = Pool(processes=processes)
    results = [pool.apply_async(func, (value,) + args, kwargs)
               for value in values]

    pool.close()
    pool.join()

    return np.asarray([result.get() for result in results])


def reconstruct(x, dim=1, tau=1):
    """Construct time-delayed vectors from a time series.

    Constructs time-delayed vectors from a scalar time series.

    Parameters
    ----------
    x : array
        1-D scalar time series.
    dim : int, optional (default = 1)
        Embedding dimension.
    tau : int, optional (default = 1)
        Time delay

    Returns
    -------
    ps : ndarray
        Array with time-delayed vectors.
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


def spectrum(x, dt=1.0, detrend=False):
    """Return the power spectrum of the given time series.

    Returns the power spectrum of the given time series.  This function
    is a very simple implementation that does not involve any averaging
    or windowing and assumes that the input series is periodic.  For
    real-world data, use scipy.signal.welch() for accurate estimation of
    the power spectrum.

    Parameters
    ----------
    x : array
        1-D real input array of length N containing the time series.
    dt : float, optional (default = 1.0)
        Sampling time (= 1/(sampling rate)).
    detrend : bool, optional (default=False)
        Subtract the mean from the series (i.e., a constant detrend).

    Returns
    -------
    freqs : array
        Array containing frequencies k/(N*dt) for k = 1, ..., N/2.
    power : array
        Array containing P(f).

    Example
    -------
    >>> signal = np.random.random(1024)
    >>> power = spectrum(signal)[1]
    >>> np.allclose(np.mean(signal ** 2), np.sum(power))
    True

    The above example is just the Parseval's theorem which states that
    the mean squared amplitude of the input signal is equal to the sum
    of P(f).
    """
    N = len(x)

    if detrend:
        x = x - np.mean(x)

    # See Section 13.4 of Press et al. (2007) for the convention.
    power = 2.0 * np.abs(np.fft.rfft(x)) ** 2 / N ** 2
    power[0] = power[0] / 2.0
    if N % 2 == 0:
        power[-1] = power[-1] / 2.0

    freqs = np.fft.rfftfreq(N, d=dt)
    return freqs, power


def statcheck(x, bins=100):
    """Check for stationarity using a chi-squared test.

    Checks for stationarity in a time series using the stationarity
    test discussed by Isliker & Kurths (1993).

    Parameters
    ----------
    x : array
        Input time series
    bins : int, optional (default = 100)
        Number of equiprobable bins used to compute the histograms.

    Returns
    -------
    chisq : float
        Chi-squared test statistic.
    p : float
        p-value of the test computed according to the number of bins
        used and chisq, using the chi-squared distribution.  If it is
        smaller than the significance level (say, 0.05), the series is
        nonstationary.  (One should actually say we can reject the
        null hypothesis of stationarity at 0.05 significance level.)

    Notes
    -----
    The value of bins should be selected such that there is at least 5
    points in each bin.
    """
    if len(x) / bins <= 5:
        raise ValueError('Using %d bins will result in bins with '
                         'less than 5 points each.' % bins)

    # Use the m-quantile function to compute equiprobable bins.
    prob = np.arange(1.0 / bins, 1.0, 1.0 / bins)
    bins = np.append(stats.mstats.mquantiles(x, prob=prob), np.max(x))

    p_full = np.histogram(x, bins)[0]
    p_full = p_full / np.sum(p_full)

    y = x[:int(len(x) / 2)]
    observed = np.histogram(y, bins)[0]
    expected = len(y) * p_full

    return stats.chisquare(observed, expected)
