# -*- coding: utf-8 -*-

"""Functions to generate time series of some popular chaotic systems.

The parameters, initial conditions, etc. have been taken from Appendix A
of Sprott (2003).
"""


from __future__ import division

import numpy as np

from numpy import fft
from scipy.integrate import odeint


def henon(length=10000, x0=None, a=1.4, b=0.3, discard=500):
    """Generate time series from the Henon map.

    Generates time series from the Henon map.

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the map.
    a : float, optional (default = 1.4)
        Constant a in the Henon map.
    b : float, optional (default = 0.3)
        Constant b in the Henon map.
    discard : int, optional (default = 500)
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : ndarray, shape (length, 2)
        Array containing points in phase space.
    """
    if not x0:
        x = [(0.0, 0.9) + 0.1 * (-1 + 2 * np.random.random(2))]
    else:
        x = [x0]

    for i in range(length + discard - 1):
        x.append([1 - a * x[-1][0] ** 2 + b * x[-1][1], x[-1][0]])

    return np.asarray(x[discard:])


def lorenz(length=10000, x0=None, sigma=10.0, beta=8.0/3.0, rho=28.0,
           step=0.001, sample=30, discard=1000):
    """Generate time series for the Lorenz system.

    Generates time series for the Lorenz system.

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the flow.
    sigma : float, optional (default = 10.0)
        Constant sigma in the Lorenz system.
    beta : float, optional (default = 8.0/3.0)
        Constant beta in the Lorenz system.
    rho : float, optional (default = 28.0)
        Constant rho in the Lorenz system.
    step : float, optional (default = 0.001)
        Approximate step size of integration.
    sample : int, optional (default = 100)
        Sampling rate of the time series.
    discard : int, optional (default = 1000)
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    t : array
        The time values at which the points have been sampled.
    x : ndarray, shape (length, 3)
        Array containing points in phase space.
    """
    def _lorenz(x, t):
        return [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2]]

    if not x0:
        x0 = (0.0, -0.01, 9.0) + 0.25 * (-1 + 2 * np.random.random(3))

    t = np.linspace(0, (sample * (length + discard)) * step,
                    sample * (length + discard))

    return (t[discard * sample::sample],
            odeint(_lorenz, x0, t)[discard * sample::sample])


def roessler(length=10000, x0=None, a=0.15, b=0.20, c=10.0, step=0.001,
             sample=100, discard=1000):
    """Generate time series for the Rössler system.

    Generates time series for the Rössler system.

    Parameters
    ----------
    length : int, optional (default = 10000)
        Length of the time series to be generated.
    x0 : array, optional (default = random)
        Initial condition for the flow.
    a : float, optional (default = 0.15)
        Constant a in the Röessler system.
    b : float, optional (default = 0.20)
        Constant b in the Röessler system.
    c : float, optional (default = 10.0)
        Constant c in the Röessler system.
    step : float, optional (default = 0.001)
        Approximate step size of integration.
    sample : int, optional (default = 100)
        Sampling rate of the time series
    discard : int, optional (default = 1000)
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    t : array
        The time values at which the points have been sampled.
    x : ndarray, shape (length, 3)
        Array containing points in phase space.
    """
    def _roessler(x, t):
        return [-(x[1] + x[2]), x[0] + a * x[1], b + x[2] * (x[0] - c)]

    t = np.linspace(0, (sample * (length + discard)) * step,
                    sample * (length + discard))

    if not x0:
        x0 = (-9.0, 0.0, 0.0) + 0.25 * (-1 + 2 * np.random.random(3))

    return (t[discard * sample::sample],
            odeint(_roessler, x0, t)[discard * sample::sample])
