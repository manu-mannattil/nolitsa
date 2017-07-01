#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filtering data from a far-infrared laser.

The data is from Data Set A [1] of the Sata Fe Time Series competition.
This is a map-like data [see Example 4.5 of Kantz & Schreiber (2004)].

The "structure" in the data (arguably) becomes more prominent after
filtering.  Also note that the discreteness of the data disappears after
filtering.  Thus, nonlinear filtering can be used as an alternative to
the method of adding a small amount of noise to "undiscretize" such data
sets.

[1]: http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe.html
"""

from nolitsa import noise
import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('../series/laser.dat')

plt.figure(1)
plt.title('Noisy series from a far-infrared laser')
plt.xlim(20, 140)
plt.ylim(20, 140)
plt.xlabel('$x(t)$')
plt.ylabel('$x(t + 1)$')
plt.plot(x[:-1], x[1:], '.')

y = noise.nored(x, dim=7, tau=1, r=2.0, repeat=5)

plt.figure(2)
plt.title('Cleaned series from a far-infrared laser')
plt.xlim(20, 140)
plt.ylim(20, 140)
plt.xlabel('$x(t)$')
plt.ylabel('$x(t + 1)$')
plt.plot(y[:-1], y[1:], '.')

plt.show()
