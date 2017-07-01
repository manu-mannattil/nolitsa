#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transforming a time series into a uniform deviate is harmful.

Uniform deviate transformation is a nonlinear transformation, and
thus, it does not preserve the linear properties of a time series.
In the example below, we see that the power spectra of the surrogates
don't match the power spectrum of the original time series if they're
both converted into a uniform deviate.

Some authors (e.g., Harikrishnan et al. [Physica D 215 (2006) 137-145])
perform a uniform deviate transformation just before surrogate analysis.
This can lead to incorrect results.
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from nolitsa import surrogates
from scipy.signal import welch


def uniform(x):
    """Convert series into a uniform deviate.

    Converts a time series into a uniform deviate using a probability
    integral transform.
    """
    y = np.empty(len(x))

    for i in range(len(x)):
        y[i] = np.count_nonzero(x <= x[i])

    return y / len(x)


x = np.loadtxt('../series/br1.dat')[:, 1]

for i in range(19):
    y = surrogates.iaaft(x)[0]
    plt.figure(1)
    f, p = welch(y, nperseg=256, detrend='constant',
                 window='boxcar', scaling='spectrum', fs=2.0)
    plt.semilogy(f[1:], p[1:], color='#BBBBBB')

    plt.figure(2)
    f, p = welch(uniform(y), nperseg=256, detrend='constant',
                 window='boxcar', scaling='spectrum', fs=2.0)
    plt.semilogy(f[1:], p[1:], color='#BBBBBB')

plt.figure(1)
plt.title('Normal PSD')
plt.xlabel(r'Frequency $f$')
plt.ylabel(r'Power $P(f)$')
f, p = welch(x, nperseg=256, detrend='constant',
             window='boxcar', scaling='spectrum', fs=2.0)
plt.semilogy(f[1:], p[1:], color='#000000')

plt.figure(2)
plt.title('PSD of uniform deviate')
plt.xlabel(r'Frequency $f$')
plt.ylabel(r'Power $P(f)$')
f, p = welch(uniform(x), nperseg=256, detrend='constant',
             window='boxcar', scaling='spectrum', fs=2.0)
plt.semilogy(f[1:], p[1:], color='#000000')

plt.show()
