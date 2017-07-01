#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AFN for time series from the Rössler oscillator.

E1 saturates near an embedding dimension of 4.  E2 != 1 at many values
of d.  Thus, the series is definitely deterministic.
"""

from nolitsa import data, dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = data.roessler()[1][:, 0]

# AFN algorithm.
dim = np.arange(1, 10 + 2)
E, Es = dimension.afn(x, tau=14, dim=dim, window=45, metric='cityblock')
E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

plt.title(r'AFN for time series from the Rössler oscillator')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], E1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], E2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.show()
