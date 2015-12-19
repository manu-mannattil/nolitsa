#!/usr/bin/env python

from __future__ import division

import numpy as np

from nolitsa import surrogates
from numpy.testing import assert_allclose, run_module_suite


def spectrum(x, d=1.0):
    """Return the power spetrum for the input series.

    Returns the power spectrum P(f) of the given input array.

    Parameters
    ----------
    x : array
        1D input array of length N.
    d : float
        Sample spacing = 1/(sampling rate).

    Returns
    -------
    freqs : array
        Array containing frequencies k/(N*d) for k = 1, ..., N/2.
    power : array
        Array containing P(f).

    Example
    -------
    >>> signal = np.random.random(111)
    >>> signal = signal - np.mean(signal)
    >>> freqs, power = spectrum(signal)
    >>> np.allclose(np.mean(signal ** 2), np.sum(power))
    True

    The above example is just the Parseval's theorem which states that
    the mean squared amplitude of the input signal is equal to the sum
    of P(f).
    """
    N = len(x)

    # See Section 13.4 of Press et al. (2007) for the convention.
    power = 2.0 * np.abs(np.fft.rfft(x)) ** 2 / N ** 2
    power[0] /= 2.0
    if N % 2 == 0:
        power[-1] /= 2.0

    freqs = np.fft.rfftfreq(N, d=1.0)
    return freqs, power


def test_ft():
    # Test surrogates.ft()
    # Always test for both odd and even number of points.
    for n in (2 ** 12 - 1, 2 ** 12):
        # NOTE that zero mean series almost always causes an assertion
        # error since the relative tolerance between different "zeros"
        # can be quite large.  This is not a bug!
        x = 1.0 + np.random.random(n)
        y = surrogates.ft(x)

        assert_allclose(spectrum(x)[1], spectrum(y)[1])


def test_aaft():
    # Test surrogates.aaft()
    # Always test for both odd and even number of points.
    for n in (2 ** 16 - 1, 2 ** 16):
        # Input is a Gaussian transformed using f(x) = exp(1.0 + x).
        x = np.exp(1.0 + np.random.normal(size=n, scale=0.5))
        y = surrogates.aaft(x)

        assert_allclose(spectrum(x)[1], spectrum(y)[1], atol=1e-3)


if __name__ == '__main__':
    run_module_suite()
