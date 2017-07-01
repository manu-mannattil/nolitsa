#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN for time series from the Henon map.

As expected, the FNN fraction goes to zero at an embedding dimension
equal to 2.
"""

from nolitsa import data, dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = data.henon(length=5000)[:, 0]

dim = np.arange(1, 10 + 1)
f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=10, metric='cityblock')

plt.title(r'FNN for Henon map')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.show()
