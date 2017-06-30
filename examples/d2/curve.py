#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""D2 for a closed noisy curve.

Although there is a proper scaling region with D2 between 1.2 and 1.5,
it is higher than the expected value of 1.0, perhaps due to the additive
noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, utils

t = np.linspace(0, 100 * np.pi, 5000)
x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.sin(5 * t)
x = utils.corrupt(x, np.random.normal(size=5000), snr=1000)

# Time delay.
tau = 25

window = 100

# Embedding dimension.
dim = np.arange(1, 10)

plt.title('Local $D_2$ vs $r$ for a noisy closed curve')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Local $D_2$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=window):
    plt.semilogx(r[3:-3], d2.d2(r, c), color='#4682B4')

plt.plot(r[3:-3], np.ones(len(r) - 6), color='#000000')
plt.show()
