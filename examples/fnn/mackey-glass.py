#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FNN for time series from the Mackey-Glass equation.

The embedding dimension reported is around 4-5.  But the second test
reports FNNs at larger d's.  Should we trust our results in such a case?
"""

from nolitsa import data, dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = data.mackey_glass(tau=23.0, sample=0.46, n=1000)

# Since we're resampling the time series using a sampling step of
# 0.46, the time delay required is 23.0/0.46 = 50.
tau = 50
dim = np.arange(1, 15 + 1)
f1, f2, f3 = dimension.fnn(x, tau=50, dim=dim, window=100, metric='euclidean')

plt.title(r'FNN for the Mackey-Glass delay differential equation')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'FNN (%)')
plt.plot(dim, 100 * f1, 'bo--', label=r'Test I')
plt.plot(dim, 100 * f2, 'g^--', label=r'Test II')
plt.plot(dim, 100 * f3, 'rs-', label=r'Test I or II')
plt.legend()

plt.show()
