#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AFN for time series from a far-infrared laser.

The time series is from Data Set A of the Sata Fe Time Series
Competition.  This is a map-like data from a far-infrared laser.

Since each value of the time series is an 8-bit integer (i.e., it's in
the range [0, 255]), the reconstructed phase space is essentially a grid
with zero dimension.  To actually measure the dimension of this data
set, we have to "kick" points off the grid a little bit by adding an
insignificant amount of noise.  See Example 6.4 in Kantz & Schreiber
(2004).

From the E1(d) curve, one concludes that the minimum embedding dimension
should be close to 8 [Cao (1997) reports 7 as the minimum embedding
dimension].  This is somewhat surprising since this series has a very
low correlation dimension (near 2.0).
"""

from nolitsa import dimension
import matplotlib.pyplot as plt
import numpy as np

# Generate data.
x = np.loadtxt('../series/laser.dat')

# Add uniform noise in [-0.025, 0.025] to "shake" the grid.
x = x + (-0.025 + 0.050 * np.random.random(len(x)))

# AFN algorithm.
dim = np.arange(1, 15 + 2)
E, Es = dimension.afn(x, tau=1, dim=dim, window=50)
E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

plt.title(r'AFN for time series from a far-infrared laser')
plt.xlabel(r'Embedding dimension $d$')
plt.ylabel(r'$E_1(d)$ and $E_2(d)$')
plt.plot(dim[:-1], E1, 'bo-', label=r'$E_1(d)$')
plt.plot(dim[:-1], E2, 'go-', label=r'$E_2(d)$')
plt.legend()

plt.show()
