#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Surrogate analysis of time series from the Lorenz attractor.

In this script, we perform surrogate analysis of a time series from the
Lorenz attractor with Takens's maximum likelihood estimator (MLE) of the
correlation dimension as the test statistic.  As expected, we can reject
the null hypothesis of a linear correlated stochastic process.
"""

import matplotlib.pyplot as plt
import numpy as np
from nolitsa import surrogates, d2, data

x = data.lorenz(x0=[-13.5, -16.0, 31.0], length=(2 ** 12))[1][:, 0]
x = x[422:3547]

mle = np.empty(19)

# Compute 19 IAAFT surrogates and compute the correlation sum.
for k in range(19):
    y = surrogates.iaaft(x)[0]
    r, c = d2.c2_embed(y, dim=[5], tau=5, window=100)[0]

    # Compute the Takens MLE.
    r_mle, mle_surr = d2.ttmle(r, c, zero=False)
    i = np.argmax(r_mle > 0.5 * np.std(y))
    mle[k] = mle_surr[i]

    plt.loglog(r, c, color='#BC8F8F')

r, c = d2.c2_embed(x, dim=[5], tau=5, window=100)[0]

# Compute the Takens MLE.
r_mle, true_mle = d2.ttmle(r, c, zero=False)
i = np.argmax(r_mle > 0.5 * np.std(x))
true_mle = true_mle[i]

plt.title('IAAFT surrogates for Lorenz')
plt.xlabel('Distance $r$')
plt.ylabel('Correlation sum $C(r)$')
plt.loglog(r, c, color='#000000')

plt.figure(2)
plt.title('Takens\'s MLE for Lorenz')
plt.xlabel(r'$D_\mathrm{MLE}$')
plt.vlines(mle, 0.0, 0.5)
plt.vlines(true_mle, 0.0, 1.0)
plt.yticks([])
plt.ylim(0, 3.0)

plt.show()
