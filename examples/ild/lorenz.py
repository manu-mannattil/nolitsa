#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ILD algorithm using time series from the Lorenz oscillator.
The correct time delay is 10, and the averaged local deformation has a clear
local minimum near that value. Moreover, the resulting plot resembles Fig. 9
from Buzug & Pfister, 1992, as expected.
"""

import matplotlib.pyplot as plt
import numpy as np

from nolitsa import data, dimension

sample = 0.01

x = data.lorenz(length=10000, x0=None, sigma=10.0, beta=8.0/3.0, rho=28.0,
                step=0.001, sample=sample, discard=1000)[1][:, 0]

dim = np.arange(2, 7, 1)
maxtau = 60

ilds = dimension.ild(x, dim=dim, qmax=10, maxtau=maxtau, rp=0.04, frefp=0.02,
                     k=None)

plt.title('ILD for Lorenz attractor')
plt.xlabel('Time delay')
plt.ylabel('ILD')

for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()

plt.show()
