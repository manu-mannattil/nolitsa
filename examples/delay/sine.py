#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Autocorrelation function of a *finite* sine wave.

Autocorrelation function of a finite sine wave over n cycles is:

  r(tau) = [(2*n*pi - tau)*cos(tau) + sin(tau)] / 2*n*pi

As n -> infinity, r(tau) = cos(tau) as expected.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import delay

n = 2 ** 3
t = np.linspace(0, n * 2 * np.pi, n * 2 ** 10)
x = np.sin(t)

r = delay.acorr(x)
r_exp = (((2 * n * np.pi - t) * np.cos(t) + np.sin(t)) / (2 * n * np.pi))

plt.title(r'Autocorrelation of a finite sine wave')
plt.xlabel(r'$t$')
plt.ylabel(r'$r(t)$')
plt.plot(t[::25], r[::25], 'o', label='Numerical')
plt.plot(t, r_exp, label='Theoretical')

plt.legend()
plt.show()
