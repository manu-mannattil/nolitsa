#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 for Brown noise.

Expected: D2 = 2 / (alpha - 1) = 2.0

Of course, this value is not due to the existence of any invariant
measure.  What is being measured here is the fractal dimension of the
Brownian trail.  The scaling region would vanish if we impose a nonzero
Theiler window, telling us that the underlying system is not
low dimensional.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, data, utils

np.random.seed(101)
x = utils.rescale(data.falpha(alpha=2.0, length=(2 ** 14))[:10 * 1000])

dim = np.arange(1, 10 + 1)
tau = 500

plt.title('Local $D_2$ vs $r$ for Brown noise')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=0,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[2:-2], d2.d2(r, c, hwin=2), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 2.0 * np.ones(100),
             color='#000000')
plt.show()
