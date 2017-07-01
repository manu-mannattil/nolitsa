#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Illustration of IAAFT surrogates.

This script illustrates IAAFT surrogates for human breath rate data.
Note that compared to AAFT surrogates, the power spectra of the
surrogates are closer to the true power spectrum (cf. the plot in
"aaft.py").
"""

from scipy.signal import welch
from nolitsa import surrogates

import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('../series/br1.dat', usecols=[1], unpack=True)

plt.title(r'Power spectrum of human breath rate (IAAFT surrogates)')
plt.xlabel(r'Frequency $f$')
plt.ylabel(r'Power $P(f)$')

# Compute 19 IAAFT surrogates and plot the spectrum.
for k in range(19):
    y, i, e = surrogates.iaaft(x)
    f, p = welch(y, nperseg=128, detrend='constant',
                 window='boxcar', scaling='spectrum', fs=2.0)

    plt.semilogy(f, p, color='#CA5B7C')

# Calculate true power spectrum.
f0, p0 = welch(x, nperseg=128, detrend='constant',
               window='boxcar', scaling='spectrum', fs=2.0)

plt.semilogy(f0, p0, color='#000000')
plt.show()
