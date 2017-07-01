#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN for time series from the Ikeda map.

This is a purely deterministic time series, yet we see the second test
reporting FNNs at large embedding dimensions.  The whole problem is that
the fraction of FNN strongly depends on the threshold parameters
used (apart from the metric).
"""

from nolitsa import data, dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = data.ikeda(length=5000)[:, 0]
dim = np.arange(1, 15 + 1)

f1 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='chebyshev')[2]
f2 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='euclidean')[2]
f3 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='cityblock')[2]

plt.title(r'FNN for the Ikeda map')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo-', label=r'Chebyshev')
plt.plot(dim, 100 * f2, 'g^-', label=r'Euclidean')
plt.plot(dim, 100 * f3, 'rs-', label=r'Cityblock')
plt.legend()

plt.show()
