#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Maximum Lyapunov exponent exponent of the Henon map.

The "accepted" value is ~ 0.419, which is quite close to the estimates
we get here.  See Fig. 3(b) of Rosenstein et al. (1993).
"""

from nolitsa import data, lyapunov
import numpy as np
import matplotlib.pyplot as plt

x = data.henon(length=5000)[:, 0]

# Time delay.
tau = 1

# Embedding dimension.
dim = [2]

d = lyapunov.mle_embed(x, dim=dim, tau=tau, maxt=25)[0]
t = np.arange(25)

plt.title('Maximum Lyapunov exponent for the Henon system')
plt.xlabel(r'Time $t$')
plt.ylabel(r'Average divergence $\langle d_i(t) \rangle$')
plt.plot(t, d)
plt.plot(t, t * 0.419 + d[0], '--')

plt.show()
