#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ILD algorithm using time series from the Rössler oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
from lib.nolitsa.nolitsa import data, delay

sample = 0.01

x = data.lorenz(length=10000, x0=None, sigma=10.0, beta=8.0/3.0, rho=28.0,
                step=0.001, sample=sample, discard=1000)[1][:, 0]

dim = np.arange(2, 10, 2)
maxtau = 30

ilds = delay.ild(x, dim=dim, qmax=10, maxtau=maxtau, rp=0.1, nrefp=None, k=20)

plt.title('ILD for Rossler oscillator')
plt.xlabel('Time delay')
plt.ylabel('ILD')

for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()

plt.show()
