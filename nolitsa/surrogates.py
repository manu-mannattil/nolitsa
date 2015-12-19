# -*- coding: utf-8 -*-

import numpy as np


def ft(x):
    """Return simple Fourier transform surrogates.

    Returns phase randomized surrogates which preserve the power
    spectrum (or equivalently the linear correlations) but *completely
    destroy* the probability distribution.

    Parameters
    ----------
    x : array
        Real input array containg the time series.

    Returns
    -------
    y : array
        Surrogates with the same power spectrum as x.
    """
    y = np.fft.rfft(x)
    n = len(y)

    phi = 2 * np.pi * np.random.random(n)
    phi[0] = 0.0

    # Even number of *input* points.
    if n % 2 == 1:
        phi[-1] = 0.0

    y = y * np.exp(1j * phi)
    return np.fft.irfft(y, n=len(x))


def aaft(x):
    """Return amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted surrogates with crudely
    the same power spectrum and distribution as the original data
    (Theiler et al., 1992).  AAFT surrogates are used in testing the
    null hypothesis that the input series is correlated Gaussian noise
    transformed by a monotonic time-independent measuring function.

    Parameters
    ----------
    x : array
        1D input array containg the time series.

    Returns
    -------
    y : array
        Surrogate series with (crudely) the same power spectrum and
        distribution.
    """
    # Generate uncorrelated Gaussian random numbers.
    y = np.random.normal(size=len(x), scale=np.std(x))

    # Introduce correlations in the random numbers by rank ordering.
    y = np.sort(y)[np.argsort(np.argsort(x))]
    y = ft(y)

    return np.sort(x)[np.argsort(np.argsort(y))]
