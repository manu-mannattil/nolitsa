#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 of the Henon map.

The estimates here match the "accepted" value of 1.220 quite closely.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, data, utils

x = utils.rescale(data.henon(length=5000)[:, 0])

dim = np.arange(1, 10 + 1)
tau = 1

plt.title('Local $D_2$ vs $r$ for Henon map')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=2,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 1.220 * np.ones(100),
             color='#000000')
plt.show()
