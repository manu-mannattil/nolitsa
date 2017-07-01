#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AFN for data from the Mackey-Glass delay differential equation.

The minimum embedding dimension comes out to be 5-7 (depending on the
initial condition) with both E1 and E2 curves giving very strong hints
of determinism.  According to Grassberger & Procaccia (1983) the
correlation dimension of the Mackey-Glass system with a delay of 23 is
~ 2.5.  Thus, the results are definitely comparable.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, dimension

x = data.mackey_glass(tau=23.0, sample=0.46, n=1000)

# Since we're resampling the time series using a sampling step of
# 0.46, the time delay required is 23.0/0.46 = 50.
tau = 50
dim = np.arange(1, 16 + 2)

# AFN algorithm.
E, Es = dimension.afn(x, tau=tau, dim=dim, window=100)
E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

plt.title(r'AFN for time series from the Mackey-Glass system')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], E1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], E2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.show()
