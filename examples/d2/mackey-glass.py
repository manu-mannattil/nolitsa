#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 for the Mackey-Glass system.

The estimates here are, depending on the initial condition, sometimes
lower than the value (D2 ~ 2.4) calculated by Grassberger & Procaccia
(1983).  One should use average over an ensemble of initial conditions
in such a case.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, data, utils

x = utils.rescale(data.mackey_glass(tau=23.0, sample=0.46, n=1000))

# Since we're resampling the time series using a sampling step of
# 0.46, the time delay required is 23.0/0.46 = 50.
tau = 50
dim = np.arange(1, 10 + 1)

plt.title('Local $D_2$ vs $r$ for Mackey-Glass system')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=100,
                        r=utils.gprange(0.001, 1.0, 100)):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.semilogx(utils.gprange(0.001, 1.0, 100), 2.4 * np.ones(100),
             color='#000000')
plt.show()
