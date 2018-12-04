#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ILD algorithm using time series from the Rössler oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
from lib.nolitsa.nolitsa import data, delay

sample = 0.001
x = data.roessler(a=0.20, b=0.40, c=5.7, sample=sample, length=10000,
                  discard=5000)[1][:, 0]

dim = np.arange(2, 10, 2)
maxtau = 100

ilds = delay.ild(x, dim=dim, qmax=4, maxtau=maxtau)

plt.title('ILD for Rossler oscillator')
plt.xlabel('Time delay')
plt.ylabel('ILD')

for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()

plt.show()
