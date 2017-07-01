#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AFN on correlated random time series.

The AFN algorithm is not impervious to correlated stochastic data.
Here we use a time series coming from an AR(1) process and see that the
E2(d) curve has a nontrivial appearance, falsely giving the appearance
of determinism.

Two points are to be noted:

1. Although the E2(d) curves have a nontrivial appearance, they are
   different from the curves seen for deterministic data.  Here the
   value of E2(d) first decreases with d and then increases, whereas
   for deterministic data, E2(d) is seen to increase right from the
   beginning.

2. Imposing a minimum temporal separation equal to the autocorrelation
   time of the series while searching for near neighbors solves the
   problem.
"""

from nolitsa import delay, dimension, utils
import matplotlib.pyplot as plt
import numpy as np

# Generate stochastic data.
N = 5 * 1000
x = np.empty(N)

np.random.seed(999)
eta = np.random.normal(size=(N), loc=0, scale=1.0)
a = 0.99

x[0] = eta[0]
for i in range(1, N):
    x[i] = a * x[i - 1] + eta[i]

x = utils.rescale(x)

# Calculate the autocorrelation time.
tau = np.argmax(delay.acorr(x) < 1.0 / np.e)

# AFN without any minimum temporal separation.
dim = np.arange(1, 10 + 2)
F, Fs = dimension.afn(x, tau=tau, dim=dim, window=0)
F1, F2 = F[1:] / F[:-1], Fs[1:] / Fs[:-1]

# AFN with a minimum temporal separation (equal to the autocorrelation
# time) between near-neighbors.
dim = np.arange(1, 10 + 2)
E, Es = dimension.afn(x, tau=tau, dim=dim, window=tau)
E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

plt.figure(1)
plt.title(r'AR(1) process with $a = 0.99$')
plt.xlabel(r'i')
plt.ylabel(r'$x_i$')
plt.plot(x)

plt.figure(2)
plt.title(r'AFN without any minimum temporal separation')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], F1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], F2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.figure(3)
plt.title(r'AFN with a minimum temporal separation of $%d$' % tau)
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], E1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], E2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.show()
