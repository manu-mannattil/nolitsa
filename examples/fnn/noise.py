#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN for uncorrelated random numbers.

Note how the fraction of FNN reported by Test I goes to zero soon after
d = log(10^4) / log(10) = 4.0.  Of course, Test II shows that something
is awry.
"""

from nolitsa import dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = np.random.random(5000)

dim = np.arange(1, 10 + 1)
f1, f2, f3 = dimension.fnn(x, tau=1, dim=dim, window=0, metric='euclidean')

plt.title(r'FNN for uncorrelated random numbers in $[0, 1]$')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I + II')
plt.legend()

plt.show()
