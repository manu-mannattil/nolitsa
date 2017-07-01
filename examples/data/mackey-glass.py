#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate time series using the Mackey-Glass equation.

Generates time series using the discrete approximation of the
Mackey-Glass delay differential equation described by Grassberger &
Procaccia (1983).

Typical values of the parameters in the Mackey-Glass delay differential
equation are: a = 0.2, b = 0.1, c = 10.0, and tau = 23.0 with the grid
size n usually taken larger than 1000.
"""

import matplotlib.pyplot as plt
from nolitsa import data

x = data.mackey_glass(tau=23.0, sample=0.46, n=1000)

# Since we're resampling the time series using a sampling step of
# 0.46, the time delay of the resampled series is 23.0/0.46 = 50.
plt.title('Mackey-Glass delay differential equation')
plt.plot(x[50:], x[:-50])
plt.xlabel(r'$x(t - \tau)$')
plt.ylabel(r'$x(t)$')
plt.show()
