#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN for correlated random numbers.

Correlated random numbers are created by running a simple moving average
(with 61 bins) over uncorrelated random numbers in [0, 1].  Without
a Theiler window, the FNN fraction drops to zero soon after
d ~ log(3000) / log(10) ~ 4.0.  Ordinarily the second test would have
helped here and an increase in FNN should occur.  But here, the strong
temporal correlations between the points in the series prevent it from
working.

Of course, once we impose a Theiler window equal to the autocorrelation
time of the series, the second test reports large amounts of FNNs.
"""

from nolitsa import dimension, noise, delay
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
np.random.seed(17)
x = noise.sma(np.random.random(3000), hwin=30)

tau = np.argmax(delay.acorr(x) < 1 / np.e)

# FNN without Theiler window.
dim = np.arange(1, 10 + 1)
f1, f2, f3 = dimension.fnn(x, tau=tau, dim=dim, window=0, metric='cityblock')

plt.figure(1)
plt.title(r'FNN for correlated random numbers (no Theiler window)')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

# FNN with Theiler window equal to autocorrelation time.
dim = np.arange(1, 10 + 1)
f1, f2, f3 = dimension.fnn(x, tau=tau, dim=dim, window=tau, metric='cityblock')

plt.figure(2)
plt.title(r'FNN for correlated random numbers (Theiler window = %d)' % tau)
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I or II')
plt.legend()

plt.show()
