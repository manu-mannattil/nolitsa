# -*- coding: utf-8 -*-

"""Functions to estimate the embedding delay.

This module provides a set of functions that can be used to estimate the
time delay required to embed a scalar time series.

  * acorr -- computes the autocorrelation of a scalar time series.
  * mi -- computes the mutual information between two scalar time
    series.
  * dmi -- computes the mutual information between a scalar time series
    and its time-delayed counterpart.
  * adfd -- computes the average displacement of the time-delayed
    vectors from the phase space diagonal as a function of the time
    delay.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from . import utils


def acorr(x, maxtau=None, norm=True, detrend=True):
    """Return the autocorrelation of the given scalar time series.

    Calculates the autocorrelation of the given scalar time series
    using the Wiener-Khinchin theorem.

    Parameters
    ----------
    x : array_like
        1-D real time series of length N.
    maxtau : int, optional (default = N)
        Return the autocorrelation only up to this time delay.
    norm : bool, optional (default = True)
        Normalize the autocorrelation so that it is equal to 1 for
        zero time delay.
    detrend: bool, optional (default = True)
        Subtract the mean from the time series (i.e., a constant
        detrend).  This is done so that for uncorrelated data, the
        autocorrelation vanishes for all nonzero time delays.

    Returns
    -------
    r : array
        Array with the autocorrelation up to maxtau.
    """
    x = np.asarray(x)
    N = len(x)

    if not maxtau:
        maxtau = N
    else:
        maxtau = min(N, maxtau)

    if detrend:
        x = x - np.mean(x)

    # We have to zero pad the data to give it a length 2N - 1.
    # See: http://dsp.stackexchange.com/q/1919
    y = np.fft.fft(x, 2 * N - 1)
    r = np.real(np.fft.ifft(y * y.conj(), 2 * N - 1))

    if norm:
        return r[:maxtau] / r[0]
    else:
        return r[:maxtau]


def mi(x, y, bins=64):
    """Calculate the mutual information between two random variables.

    Calculates mutual information, I = S(x) + S(y) - S(x,y), between two
    random variables x and y, where S(x) is the Shannon entropy.

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


def dmi(x, maxtau=1000, bins=64):
    """Return the time-delayed mutual information of x_i.

    Returns the mutual information between x_i and x_{i + t} (i.e., the
    time-delayed mutual information), up to a t equal to maxtau.  Based
    on the paper by Fraser & Swinney (1986), but uses a much simpler,
    albeit, time-consuming algorithm.

    Parameters
    ----------
    x : array
        1-D real time series of length N.
    maxtau : int, optional (default = min(N, 1000))
        Return the mutual information only up to this time delay.
    bins : int
        Number of bins to use while calculating the histogram.

    Returns
    -------
    ii : array
        Array with the time-delayed mutual information up to maxtau.

    Notes
    -----
    For the purpose of finding the time delay of minimum delayed mutual
    information, the exact number of bins is not very important.
    """
    N = len(x)
    maxtau = min(N, maxtau)

    ii = np.empty(maxtau)
    ii[0] = mi(x, x, bins)

    for tau in range(1, maxtau):
        ii[tau] = mi(x[:-tau], x[tau:], bins)

    return ii


def adfd(x, dim=1, maxtau=100):
    """Compute average displacement from the diagonal (ADFD).

    Computes the average displacement of the time-delayed vectors from
    the phase space diagonal which helps in picking a suitable time
    delay (Rosenstein et al. 1994).

    Parameters
    ----------
    x : array
        1-D real time series of length N.
    dim : int, optional (default = 1)
        Embedding dimension.
    maxtau : int, optional (default = 100)
        Calculate the ADFD only up to this delay.

    Returns
    -------
    disp : array
        ADFD for all time delays up to maxtau.
    """
    disp = np.zeros(maxtau)
    N = len(x)

    maxtau = min(maxtau, int(N / dim))

    for tau in range(1, maxtau):
        y1 = utils.reconstruct(x, dim=dim, tau=tau)

        # Reconstruct with zero time delay.
        y2 = x[:N - (dim - 1) * tau]
        y2 = y2.repeat(dim).reshape(len(y2), dim)

        disp[tau] = np.mean(utils.dist(y1, y2, metric='euclidean'))

    return disp
