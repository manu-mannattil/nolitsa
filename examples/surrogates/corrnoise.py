#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IAAFT surrogates for correlated noise.

The properties of linearly correlated noise can be captured quite
accurately by IAAFT surrogates.  Thus, they cannot easily fool
a dimension estimator (here we use Takens's maximum likelihood estimator
for the correlation dimension) if surrogate analysis is performed
additionally.
"""

import matplotlib.pyplot as plt
import numpy as np
from nolitsa import surrogates, d2, noise, delay

x = noise.sma(np.random.normal(size=(2 ** 12)), hwin=100)
ends = surrogates.mismatch(x)[0]
x = x[ends[0]:ends[1]]
act = np.argmax(delay.acorr(x) < 1 / np.e)

mle = np.empty(19)

# Compute 19 IAAFT surrogates and compute the correlation sum.
for k in range(19):
    y = surrogates.iaaft(x)[0]
    r, c = d2.c2_embed(y, dim=[7], tau=act, window=act)[0]

    # Compute the Takens MLE.
    r_mle, mle_surr = d2.ttmle(r, c)
    i = np.argmax(r_mle > 0.5 * np.std(y))
    mle[k] = mle_surr[i]

    plt.loglog(r, c, color='#BC8F8F')

r, c = d2.c2_embed(x, dim=[7], tau=act, window=act)[0]

# Compute the Takens MLE.
r_mle, true_mle = d2.ttmle(r, c)
i = np.argmax(r_mle > 0.5 * np.std(x))
true_mle = true_mle[i]

plt.title('IAAFT surrogates for correlated noise')
plt.xlabel('Distance $r$')
plt.ylabel('Correlation sum $C(r)$')
plt.loglog(r, c, color='#000000')

plt.figure(2)
plt.title('Takens\'s MLE for correlated noise')
plt.xlabel(r'$D_\mathrm{MLE}$')
plt.vlines(mle, 0.0, 0.5)
plt.vlines(true_mle, 0.0, 1.0)
plt.yticks([])
plt.ylim(0, 3.0)

plt.show()
