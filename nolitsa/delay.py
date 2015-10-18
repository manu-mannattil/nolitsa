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
