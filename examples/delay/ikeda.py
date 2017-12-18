#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Time delay estimation for time series from the Ikeda map.

For map like data, the redundancy between components of the time delayed
vectors decrease drastically (or equivalently, the irrelevance increases
rapidly).  Best results are often obtained with a time delay of 1.

Here, we see that for data coming out of the Ikeda map, the delayed
mutual information curve (which does not have any local minima) gives
us a very bad estimate of the time delay.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, delay

x = data.ikeda()[:, 0]

# Compute autocorrelation and delayed mutual information.
lag = np.arange(50)
r = delay.acorr(x, maxtau=50)
i = delay.dmi(x, maxtau=50)

r_delay = np.argmax(r < 1.0 / np.e)
print(r'Autocorrelation time = %d' % r_delay)

plt.figure(1)

plt.subplot(211)
plt.title(r'Delay estimation for Ikeda map')
plt.ylabel(r'Delayed mutual information')
plt.plot(lag, i)

plt.subplot(212)
plt.xlabel(r'Time delay $\tau$')
plt.ylabel(r'Autocorrelation')
plt.plot(lag, r, r_delay, r[r_delay], 'o')

plt.figure(2)
plt.subplot(121)
plt.title(r'Time delay = 10')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-10], x[10:], '.')

plt.subplot(122)
plt.title(r'Time delay = %d' % r_delay)
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-r_delay], x[r_delay:], '.')

plt.show()
