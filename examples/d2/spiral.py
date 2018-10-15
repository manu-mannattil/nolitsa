#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Correlation sum/D2 for a spiral.

A spiral, though a one-dimensional curve, is a nonstationary object.
Thus, the estimated correlation dimension would heavily depend on the
Theiler window used.  However, the values of C(r) at large r's would
roughly be the same.
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import d2, utils

phi = np.linspace(2 * np.pi, 52 * np.pi, 1000)
x = phi * np.cos(phi)
x = utils.rescale(x)

dim = np.arange(1, 10 + 1)
tau = 10
r = utils.gprange(0.01, 1.0, 100)

plt.figure(1)
plt.title('Correlation sum $C(r)$ without any Theiler window')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Correlation sum $C(r)$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=0, r=r):
    plt.loglog(r, c)

plt.figure(2)
plt.title('Correlation sum $C(r)$ with a Theiler window of 100')
plt.xlabel(r'Distance $r$')
plt.ylabel(r'Correlation sum $C(r)$')

for r, c in d2.c2_embed(x, tau=tau, dim=dim, window=100, r=r):
    plt.loglog(r, c)

plt.show()
