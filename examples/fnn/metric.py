#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN is metric depended for noisy data.

The FNN fraction is metric depended for noisy time series.  In
particular, the second FNN test, which measures the "boundedness" of the
reconstructed attractor depends heavily on the metric used.  E.g., if
the Chebyshev metric is used, the near-neighbor distances in the
reconstructed attractor are always bounded and therefore the reported
FNN fraction becomes a constant (approximately) instead of increasing
with the embedding dimension.
"""

from nolitsa import dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = np.random.normal(size=5000)
dim = np.arange(1, 10 + 1)

plt.figure(1)
f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='chebyshev')
plt.title(r'FNN with Chebyshev metric')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.figure(2)
f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='euclidean')
plt.title(r'FNN with Euclidean metric')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.figure(3)
f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='cityblock')
plt.title(r'FNN with cityblock (Manhattan) metric')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.show()
