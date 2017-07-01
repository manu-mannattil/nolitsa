#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 for white noise.

D2 is (theoretically) equal to the embedding dimension for white noise.
"""

import numpy as np
import matplotlib.pyplot as plt

from nolitsa import d2, utils

x = np.random.random(5 * 1000)

dim = np.arange(1, 10 + 1)
tau = 1

plt.title('Local $D_2$ vs $r$ for white noise')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=2,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[1:-1], d2.d2(r, c, hwin=1), color='#4682B4')

plt.show()
