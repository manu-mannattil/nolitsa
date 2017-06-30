#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 of the Rössler oscillator.

The estimates here match the "accepted" value of 1.991 quite closely.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, data, utils

x0 = [-3.2916983, -1.42162302, 0.02197593]
x = utils.rescale(data.roessler(length=5000, x0=x0)[1][:, 0])

dim = np.arange(1, 10 + 1)
tau = 14

plt.title(u'Local $D_2$ vs $r$ for Rössler oscillator')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=50,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 1.991 * np.ones(100),
             color='#000000')
plt.show()
