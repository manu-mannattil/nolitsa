#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filtering human breath data.

The data was obtained from 2048 continuous samples of dataset B1
(starting from the 12750th) of the Santa Fe time series contest [1].
This data is low dimensional, and is thought to be a limit cycle [See
Example 10.7 of Kantz & Schreiber (2004).]  As can be seen, the
structure of the limit cycle is much more prominent when the filtered
time series is used.

[1]: http://www.physionet.org/physiobank/database/santa-fe/
"""

import numpy as np
from nolitsa import noise, utils
import matplotlib.pyplot as plt

x = utils.rescale(np.loadtxt('../series/br2.dat')[:, 1])
y = noise.nored(x, dim=7, r=0.23, repeat=5, tau=1)

plt.figure(1)
plt.title('Noisy human breath data')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-1], x[1:], '.')

plt.figure(2)
plt.title('Filtered human breath data')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(y[:-1], y[1:], '.')

plt.show()
