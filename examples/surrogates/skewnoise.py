#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Skew statistic fails for linear stochastic data.

The skew statistic quantifies asymmetry under time reversal by measuring
the skewness in the distribution of the increments of a time series.
This makes sense as this distribution is flipped left-to-right (along
with a sign change) for all kinds of series under time reversal.  So if
this distribution is asymmetric, the time series must exhibit asymmetry
under time reversal.  But asymmetry in the distribution of increments
isn't a very good measure of nonlinearity as we'll show here.

Time series from the Henon map shows very strong asymmetry under time
reversal.  So if we start with time series from the Henon map, take the
first difference, shuffle the increments, and calculate the cumulative
sum of the shuffled increments, we end up with a time series which would
come out as nonlinear according to the above rule of thumb.  But since
this new series is the cumulative sum of uncorrelated random numbers,
it's a purely linear one.  Obviously, this doesn't make any sense.

Of course, the distribution of increments would slowly become symmetric
as we take larger and larger increments, and pretty soon the series
would fail to reject the null hypothesis of linearity.  But how large is
large?  It should also be noted that many nonlinear series also fail to
reject the null hypothesis of linearity with this statistic when larger
increments are considered.  Thus, this statistic is almost useless in
practical situations.
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


# Start with time series from the Henon map, take the first-difference,
# shuffle the increments, and calculate the cumulative sum of the
# shuffled increments.
x = data.henon(length=(2 ** 12))[:, 0]
dx = x[1:] - x[:-1]
np.random.shuffle(dx)
x = np.cumsum(dx)

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
plt.title('Skew statistic fails for stochastic data')
plt.vlines(s0, 0.0, 0.5)
plt.vlines(skew(x), 0.0, 1.0)
plt.yticks([])
plt.ylim(0, 3.0)
plt.show()
