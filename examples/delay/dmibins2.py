#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Delayed mutual information calculation and number of bins.

The time delay is often picked to be the location of the first minimum
of the delayed mutual information (DMI) of the series.  The
probabilities required for its computation are estimated by binning the
time series.

For many examples, the DMI at a lag of zero computed with 2^m bins is
approximately m bits.  This is because the distribution is nearly flat
when the number of bins is small, making the probability of being in a
bin ~ 2^-m.

Surprisingly, using a small number of bins doesn't seem to affect the
estimation of the delay.  Even with two bins, the extremas of the DMI
are clearly visible.  (Why?)
"""

import numpy as np
import matplotlib.pyplot as plt
from nolitsa import data, delay

x = data.mackey_glass()

plt.title(r'Delayed mutual information for the Mackey-Glass system')
plt.xlabel(r'$\tau$')
plt.ylabel(r'$I(\tau)$')

for bins in (2 ** np.arange(1, 8 + 1)):
    ii = delay.dmi(x, maxtau=500, bins=bins)
    plt.plot(ii, label=(r'Bins = $%d$' % bins))

plt.legend()
plt.show()
