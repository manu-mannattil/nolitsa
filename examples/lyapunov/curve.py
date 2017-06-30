#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Maximum Lyapunov exponent of a closed noisy curve.

A trajectory in the form of a closed curve should have a Lyapunov
exponent equal to zero (or the average divergence should not vary with
time).  But our curves for the average divergence appear to be
oscillatory and don't look very flat.  What's wrong?
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import lyapunov, utils

t = np.linspace(0, 100 * np.pi, 5000)
x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.sin(5 * t)
x = utils.corrupt(x, np.random.normal(size=5000), snr=1000)

# Time delay.
tau = 25

window = 100

# Embedding dimension.
dim = [10]

d = lyapunov.mle_embed(x, dim=dim, tau=tau, maxt=300, window=window)[0]

plt.title('Maximum Lyapunov exponent for a closed curve')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
plt.plot(t[:300], d)

plt.show()
