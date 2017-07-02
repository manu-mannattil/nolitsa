#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AFN for 8-bit time series from the Rössler oscillator.

Since each point in the time series is an 8-bit integer (i.e., it's in
the range [-127, 127]), the reconstructed phase space is essentially a
grid with zero dimension.  To actually measure the dimension of this
data set, we have to "kick" points off the grid a little bit by adding
an insignificant amount of noise.  See Example 6.4 in Kantz & Schreiber
(2004).

But the quality of reconstruction depends on the noise level.  Adding
an insignificant amount of noise does not help at all!  This is
probably one of the rare case where a higher level of additive noise
improves the results.
"""

from nolitsa import data, dimension, utils
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = data.roessler(length=5000)[1][:, 0]

# Convert to 8-bit.
x = np.int8(utils.rescale(x, (-127, 127)))

# Add uniform noise of two different noise levels.
y1 = x + (-0.001 + 0.002 * np.random.random(len(x)))
y2 = x + (-0.5 + 1.0 * np.random.random(len(x)))

# AFN algorithm.
dim = np.arange(1, 10 + 2)
F, Fs = dimension.afn(y1, tau=14, dim=dim, window=40)
F1, F2 = F[1:] / F[:-1], Fs[1:] / Fs[:-1]

E, Es = dimension.afn(y2, tau=14, dim=dim, window=40)
E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

plt.figure(1)
plt.title(r'AFN after corrupting with uniform noise in $[-0.001, 0.001]$')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], F1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], F2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.figure(2)
plt.title(r'AFN after corrupting with uniform noise in $[-0.5, 0.5]$')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], E1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], E2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.show()
