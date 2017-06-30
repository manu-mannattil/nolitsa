#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 of the Lorenz system.

The estimates here match the "accepted" value of 2.068 quite closely.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, data, utils

x = utils.rescale(data.lorenz(length=5000)[1][:, 0])

dim = np.arange(1, 10 + 1)
tau = 5

plt.title('Local $D_2$ vs $r$ for Lorenz attractor')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=50):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 2.068 * np.ones(100),
             color='#000000')
plt.show()
