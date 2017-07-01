#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Skew statistic fails for time series from the Lorenz attractor.

Time series from the Lorenz attractor is quite symmetric in time.
So the skew statistic fails to detect the very strong nonlinearity
present (cf. "skewnoise.py").
"""

import matplotlib.pyplot as plt
import numpy as np
from nolitsa import data, surrogates


def skew(x, t=1):
    """Skew statistic to measure asymmetry w.r.t. time reversal.

    Skew statistic measures the asymmetry in the time series w.r.t. time
    reversal.  This asymmetry is often considered to be an indicator of
    nonlinearity (see Notes).

    Parameters
    ----------
    x : array
        1D real input array containing the time series.
    t : int, optional (default = 1)
        Skew stastic measures the skewness in the distribution of
        t-increments of the time series.  By default the skewness in
        the distribution of its first-increments is returned.

    Returns
    -------
    s : float
        Coefficient of skewness of the distribution of t-increments.

    Notes
    -----
    The skew statistic is often touted to have good distinguishing power
    between nonlinearity and linearity.  But it is known to fail
    miserably in both cases (i.e., it often judges nonlinear series as
    linear and vice-versa) and should be avoided for serious analysis.
    """
    dx = x[t:] - x[:-t]
    dx = dx - np.mean(dx)
    return np.mean(dx ** 3) / np.mean(dx ** 2) ** 1.5


x = data.lorenz(length=(2 ** 12))[1][:, 0]

plt.figure(1)

plt.subplot(121)
plt.title('Actual')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-5], x[5:])

plt.subplot(122)
plt.title('Reversed')
plt.xlabel(r'$\hat{x}(t)$')
plt.ylabel(r'$\hat{x}(t + \tau)$')
x_rev = x[::-1]
plt.plot(x_rev[:-5], x_rev[5:])

s0 = np.empty(39)
for i in range(39):
    y = surrogates.aaft(x)
    s0[i] = skew(y)

plt.figure(2)
plt.title('Skew statistic for time series from the Lorenz attractor')
plt.vlines(s0, 0.0, 0.5)
plt.vlines(skew(x), 0.0, 1.0)
plt.yticks([])
plt.ylim(0, 3.0)

plt.show()
