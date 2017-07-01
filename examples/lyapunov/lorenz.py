#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Maximum Lyapunov exponent for the Lorenz system.

Our estimate is quite close to the "accepted" value of 1.50.
Cf. Fig. 2 of Rosenstein et al. (1993).
"""

from nolitsa import data, lyapunov
import numpy as np
import matplotlib.pyplot as plt

sample = 0.01
x0 = [0.62225717, -0.08232857, 30.60845379]
x = data.lorenz(length=5000, sample=sample, x0=x0,
                sigma=16.0, beta=4.0, rho=45.92)[1][:, 0]

# Choose appropriate Theiler window.
window = 60

# Time delay.
tau = 13

# Embedding dimension.
dim = [5]

d = lyapunov.mle_embed(x, dim=dim, tau=tau, maxt=300, window=window)[0]
t = np.arange(300)

plt.title('Maximum Lyapunov exponent for the Lorenz system')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
plt.plot(sample * t, d)
plt.plot(sample * t, sample * t * 1.50, '--')

plt.show()
