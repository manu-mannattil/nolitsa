#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Maximum Lyapunov exponent of the Rössler oscillator.

The "accepted" value is 0.0714, which is quite close to what we get.
"""
from nolitsa import data, lyapunov, utils
import numpy as np
import matplotlib.pyplot as plt

sample = 0.2
x0 = [-3.2916983, -1.42162302, 0.02197593]
x = data.roessler(length=3000, x0=x0, sample=sample)[1][:, 0]

# Choose appropriate Theiler window.
# Since Rössler is an aperiodic oscillator, the average time period is
# a good choice.
f, p = utils.spectrum(x)
window = int(1 / f[np.argmax(p)])

# Time delay.
tau = 7

# Embedding dimension.
dim = [3]

d = lyapunov.mle_embed(x, dim=dim, tau=tau, maxt=200, window=window)[0]
t = np.arange(200)

plt.title(u'Maximum Lyapunov exponent for the Rössler oscillator')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
plt.plot(sample * t, d)
plt.plot(sample * t, d[0] + sample * t * 0.0714, '--')

plt.show()
