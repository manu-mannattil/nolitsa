#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Illustration of Theiler window using an AR(1) series.

An AR(1) time series is temporally correlated.  Thus, if a judicious
(nonzero) value of the Theiler window is not used, the estimated
dimension converges to the fractal dimension of the trajectory formed by
the time series in the phase space.  This, however, has nothing to do
with any low-dimensional nature of the underlying process.
"""

from nolitsa import d2
import numpy as np
import matplotlib.pyplot as plt

N = 5000
x = np.empty(N)
np.random.seed(882)
n = np.random.normal(size=(N), loc=0, scale=1.0)
a = 0.998

x[0] = n[0]
for i in range(1, N):
    x[i] = a * x[i - 1] + n[i]

# Delay is the autocorrelation time.
tau = 400

dim = np.arange(1, 10 + 1)

plt.figure(1)
plt.title(r'Local $D_2$ vs $r$ for AR(1) time series with $W = 0$')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=0):
    plt.semilogx(r[3:-3], d2.d2(r, c))

plt.figure(2)
plt.title(r'Local $D_2$ vs $r$ for AR(1) time series with $W = 400$')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=400):
    plt.semilogx(r[3:-3], d2.d2(r, c))

plt.show()
