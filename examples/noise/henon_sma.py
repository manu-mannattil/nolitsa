#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple moving average vs. nonlinear noise reduction.

We will compare the effectiveness of a linear filter like the simple
moving average (SMA) and nonlinear noise reduction in filtering a noisy
deterministic time series (from the Henon map).

As we can see, SMA performs quite badly and distorts the structure in
the time series considerably (even with a very small averaging window).
However, nonlinear reduction works well (within limits).
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, noise, utils

x = data.henon()[:, 0]
x = utils.corrupt(x, np.random.normal(size=(10 * 1000)), snr=500)

y1 = noise.nored(x, dim=7, tau=1, r=0.10, repeat=5)
y2 = noise.sma(x, hwin=1)

plt.figure(1)
plt.title('Time series from the Henon map with an SNR of 500')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.plot(x[:-1], x[1:], '.')

plt.figure(2)
plt.title('After doing an SMA over 3 bins')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.plot(y2[:-1], y2[1:], '.')

plt.figure(3)
plt.title('After using the simple nonlinear filter')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.plot(y1[:-1], y1[1:], '.')

plt.show()
