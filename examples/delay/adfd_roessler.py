#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ADFD algorithm using time series from the Rössler oscillator.

The time delay is taken to be the delay at which the derivative of the
ADFD falls to 40% of its initial value.  The estimated time delay is 5.
Compare with Fig. 6 of Rosenstein et al. (1994).
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, delay

sample = 0.10
x = data.roessler(a=0.20, b=0.40, c=5.7, sample=sample, length=2500,
                  discard=5000)[1][:, 0]

dim = 7
maxtau = 50
tau = np.arange(maxtau)

disp = delay.adfd(x, dim=dim, maxtau=maxtau)
ddisp = np.diff(disp)
forty = np.argmax(ddisp < 0.4 * ddisp[1])

print('Time delay = %d' % forty)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time ($\tau\Delta t$)')
ax1.set_ylabel(r'$\mathrm{ADFD}$')
ax1.plot(tau[1:] * sample, disp[1:])

ax2 = ax1.twinx()
ax2.plot(tau[1:] * sample, ddisp, 'g--')
ax2.plot(tau[forty + 1] * sample, ddisp[forty], 'o')
ax2.set_ylabel(r'$\frac{d}{d\tau}(\mathrm{ADFD}$)')

plt.show()
