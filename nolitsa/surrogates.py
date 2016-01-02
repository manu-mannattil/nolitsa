# -*- coding: utf-8 -*-

import numpy as np
from nolitsa import noise


def ft(x):
    """Return simple Fourier transform surrogates.

    Returns phase randomized (FT) surrogates which preserve the power
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

    phi = 2 * np.pi * np.random.random(len(y))

    phi[0] = 0.0
    if len(x) % 2 == 0:
        phi[-1] = 0.0

    y = y * np.exp(1j * phi)
    return np.fft.irfft(y, n=len(x))


def aaft(x):
    """Return amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (AAFT) surrogates with
    crudely the same power spectrum and distribution as the original
    data (Theiler et al., 1992).  AAFT surrogates are used in testing
    the null hypothesis that the input series is correlated Gaussian
    noise transformed by a monotonic time-independent measuring
    function.

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
    y = np.random.normal(size=len(x))

    # Introduce correlations in the random numbers by rank ordering.
    y = np.sort(y)[np.argsort(np.argsort(x))]
    y = ft(y)

    return np.sort(x)[np.argsort(np.argsort(y))]


def iaaft(x, maxiter=1000, atol=1e-8, rtol=1e-10, smooth=7):
    """Return iterative amplitude adjusted Fourier transform surrogates.

    Returns phase randomized, amplitude adjusted (IAAFT) surrogates with
    the same power spectrum (to a very high accuracy) and distribution
    as the original data using an iterative scheme (Schreiber & Schmitz,
    1996).

    Parameters
    ----------
    x : array
        1D real input array of length N containing the time series.
    maxiter : int, optional (default = 1000)
        Maximum iterations to be performed while checking for
        convergence.  The scheme may converge before this number as
        well (see Notes).
    atol : float, optional (default = 1e-8)
        Absolute tolerance for checking convergence (see Notes).
    rtol : float, optional (default = 1e-10)
        Relative tolerance for checking convergence (see Notes).
    smooth : int, optional (default = 7)
        Use a simple moving average (SMA) to smoothen the Fourier
        amplitudes over (2*smooth + 1) frequencies while checking
        convergence.

    Returns
    -------
    y : array
        Surrogate series with (almost) the same power spectrum and
        distribution.
    i : int
        Number of iterations that have been performed.
    e : float
        Root-mean-square-deviation (RMSD) between the power spectrum of
        the surrogate series and that of the original series.

    Notes
    -----
    While computing Fourier transforms, the full time series is used
    without any smoothening.  Only while testing for convergence is the
    spectrum smoothened.

    To check if the power spectrum has converged, we see if the absolute
    difference between the current (cerr) and previous (perr) RMSD
    errors is within the limits set by the tolerance levels, i.e., if
    ``abs(cerr - perr) <= atol + rtol*perr``.  This follows the
    convention used in the NumPy function `numpy.allclose()`.

    Additionally, `atol` and `rtol` can be both set to zero in which
    case the iterations end only when the RMSD error stops changing or
    when `maxiter` is reached.
    """
    # Calculate "true" Fourier amplitudes and sort the series.
    ampl = np.abs(np.fft.rfft(x))
    power = noise.sma(ampl, hwin=smooth)
    sort = np.sort(x)

    # Previous and current error.
    perr, cerr = (-1, 1)

    # Start with a random permutation.
    t = np.fft.rfft(np.random.permutation(x))

    for i in range(maxiter):
        # Match power spectrum.
        s = np.real(np.fft.irfft(ampl * t / np.abs(t), n=len(x)))

        # Match distribution by rank ordering.
        y = sort[np.argsort(np.argsort(s))]

        t = np.fft.rfft(y)
        cerr = np.sum((power - noise.sma(np.abs(t), hwin=smooth)) ** 2)

        # Check convergence.
        if abs(cerr - perr) <= atol + rtol * abs(perr)
            break
        else:
            perr = cerr

    return y, i, np.sqrt(cerr / len(ampl))
