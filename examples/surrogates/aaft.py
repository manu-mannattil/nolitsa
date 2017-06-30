#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Illustration of AAFT surrogates.

This script illustrates AAFT surrogates for human breath rate data.  The
plot corresponds to Fig. 1 of Schreiber & Schmitz (1996).  As we can
see, the power spectra of the AAFT surrogates deviate considerably from
the actual power spectrum.  Better results can be obtained if IAAFT
surrogates are used instead.
"""

from scipy.signal import welch
from nolitsa import surrogates

import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('../series/br1.dat', usecols=[1], unpack=True)

plt.title(r'Power spectrum of human breath rate')
plt.xlabel(r'Frequency $f$')
plt.ylabel(r'Power $P(f)$')

# Compute 19 AAFT surrogates and plot the spectrum.
for i in range(19):
    y = surrogates.aaft(x)
    f, p = welch(y, nperseg=128, detrend='constant',
                 window='boxcar', scaling='spectrum', fs=2.0)

    plt.semilogy(f, p, color='#CA5B7C')

# Calculate true power spectrum.
f0, p0 = welch(x, nperseg=128, detrend='constant',
               window='boxcar', scaling='spectrum', fs=2.0)

plt.semilogy(f0, p0, color='#000000')
plt.show()
