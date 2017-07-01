#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ADFD algorithm using time series from the Mackey-Glass system.

The time delay is taken to be the delay at which the derivative of the
ADFD falls to 40% of its initial value.  The actual time delay used for
generating the time series is 17, and the estimated time delay is 15.
Compare with Fig. 8 of Rosenstein et al. (1994).
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, delay

sample = 0.25
x = data.mackey_glass(length=2500, a=0.2, b=0.1, c=10.0, tau=17.0,
                      discard=500, sample=sample)

dim = 7
maxtau = 50
tau = np.arange(maxtau)

disp = delay.adfd(x, dim=dim, maxtau=maxtau)
ddisp = np.diff(disp)
forty = np.argmax(ddisp < 0.4 * ddisp[1])

print(r'Time delay %d' % forty)

fig, ax1 = plt.subplots()

ax1.set_xlabel(r'Time ($\tau\Delta t$)')
ax1.set_ylabel(r'$\mathrm{ADFD}$')
ax1.plot(tau[1:] * sample, disp[1:])

ax2 = ax1.twinx()
ax2.plot(tau[1:] * sample, ddisp, 'g--')
ax2.plot(tau[forty + 1] * sample, ddisp[forty], 'o')
ax2.set_ylabel(r'$\frac{d}{d\tau}(\mathrm{ADFD}$)')

plt.show()
