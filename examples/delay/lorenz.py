#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Time delay estimation for time series from the Lorenz attractor

The first minimum of the delayed mutual information occurs at 5 and the
autocorrelation time is 11.  Note that both these values depend on the
sampling time used and should not be taken as "universal" time delays
for reconstructing the Lorenz attractor.

If one plots the 2D phase portrait of the attractor with a delay of 11,
one can see that it's overfolded.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, delay, noise


def localmin(x):
    """Return all local minimums from the given data set.

    Returns all local minimums from the given data set.  Note that even
    "kinky" minimums (which are probably not minimums) will also be
    returned.

    Parameters
    ----------
    x : array
        1D scalar data set.

    Returns
    -------
    i : array
        Array containing location of all local minimums.
    """
    return (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1


x = data.lorenz()[1][:, 0]

# Compute autocorrelation and delayed mutual information.
lag = np.arange(100)
r = delay.acorr(x, maxtau=100)
i = delay.dmi(x, maxtau=100)

# While looking for local minimas in the DMI curve, it's useful to do an
# SMA to remove "kinky" minimas.
i_delay = localmin(noise.sma(i, hwin=1)) + 1
r_delay = np.argmax(r < 1.0 / np.e)

print(r'Minimas of delayed mutual information = %s' % i_delay)
print(r'Autocorrelation time = %d' % r_delay)

plt.figure(1)

plt.subplot(211)
plt.title(r'Delay estimation for Lorenz attractor')
plt.ylabel(r'Delayed mutual information')
plt.plot(lag, i, i_delay, i[i_delay], 'o')

plt.subplot(212)
plt.xlabel(r'Time delay $\tau$')
plt.ylabel(r'Autocorrelation')
plt.plot(lag, r, r_delay, r[r_delay], 'o')

plt.figure(2)
plt.subplot(121)
plt.title(r'Time delay = %d' % i_delay[0])
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-i_delay[0]], x[i_delay[0]:])

plt.subplot(122)
plt.title(r'Time delay = %d' % r_delay)
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + \tau)$')
plt.plot(x[:-r_delay], x[r_delay:])

plt.show()
