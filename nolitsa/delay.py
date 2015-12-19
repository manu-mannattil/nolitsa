# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from nolitsa import utils


def acorr(x, maxlag=None, norm=True, detrend=True):
    """Return the autocorrelation of the given scalar time series.

    Calculates the autocorrelation r(t) of the given scalar time series
    using the Wiener-Khinchin theorem.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    maxlag : int, optional (default = N)
        Return the autocorrelation only upto this lag.
    norm : bool, optional (default = True)
        Normalize the autocorrelation such that r(0) = 1.
    detrend: bool, optional (default = True)
        Subtract the mean from the time series.  This is done so that
        for uncorrelated data, r(0) = 0.

    Returns
    -------
    r : array
        Array with the autocorrelation upto maxlag.
    """
    x = np.asarray(x)
    N = len(x)

    if not maxlag:
        maxlag = N
    else:
        maxlag = min(N, maxlag)

    if detrend:
        x = x - np.mean(x)

    # We have to zero pad the data to give it a length 2N - 1.
    # See: http://dsp.stackexchange.com/q/1919
    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))

    if norm:
        return r[:maxlag] / r[0]
    else:
        return r[:maxlag]


def mi(x, y, bins=64):
    """Calculate the mutual information between two random variables.

    Calculates mutual information, I = S(x) + S(y) - S(x,y), between two
    random variables x and y.

    Parameters
    ----------
    x : array
        First random variable.
    y : array
        Second random variable.
    bins : int
        Number of bins to use while creating the histogram.

    Returns
    -------
    i : float
        Mutual information.
    """
    p_x = np.histogram(x, bins)[0]
    p_y = np.histogram(y, bins)[0]
    p_xy = np.histogram2d(x, y, bins)[0].flatten()

    # Convert frequencies into probabilities.  Also, in the limit
    # p -> 0, p*log(p) is 0.  We need to take out those.
    p_x = p_x[p_x > 0] / np.sum(p_x)
    p_y = p_y[p_y > 0] / np.sum(p_y)
    p_xy = p_xy[p_xy > 0] / np.sum(p_xy)

    # Calculate the corresponding Shannon entropies.
    h_x = np.sum(p_x * np.log2(p_x))
    h_y = np.sum(p_y * np.log2(p_y))
    h_xy = np.sum(p_xy * np.log2(p_xy))

    return h_xy - h_x - h_y


def dmi(x, maxlag=1024, bins=64):
    """Return the time delayed mutual information of ``x_i``.

    Returns the mutual information between ``x_i`` and ``x_{i + t}``
    upto a `t` equal to `maxlag` (i.e., the delayed mutual information).

    Parameters
    ----------
    x : array
        1D scalar time series.
    maxlag : int, optional (default = min(N, 1024))
        Return the mutual information only upto this lag.  Since the
        mutual information calculation is computationally expensive,
        it is always advisable to use a small number.
    bins : int
        Number of bins to use while calculating the histogram.

    Returns
    -------
    ii : array
        Array with the mutual information upto maxlag.
    """
    N = len(x)
    maxlag = min(N, maxlag)

    ii = np.empty(maxlag)
    ii[0] = mi(x, x, bins)

    for lag in range(1, maxlag):
        ii[lag] = mi(x[:-lag], x[lag:], bins)

    return ii
