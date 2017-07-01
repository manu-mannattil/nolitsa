#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Illustration of end-point mismatch.

As one can clearly see, there are spurious high-frequency oscillations
in the surrogate series generated with the second data set (whose
end-points don't match).  These high-frequency oscillations appear as a
sort of "crinkliness" spread throughout the time series.
"""

from nolitsa import data, surrogates

import matplotlib.pyplot as plt
import numpy as np

x = data.lorenz(x0=[-13.5, -16.0, 31.0], length=(2 ** 12))[1][:, 0]

# Maximum mismatch occurs for the segment (537, 3662).
# Minimum mismatch occurs for the segment (422, 3547).
# end, d = surrogates.mismatch(x, length=1024)

plt.subplot(211)
plt.title(r'Original time series')
plt.ylabel(r'Measurement $x(t)$')

plt.plot(np.arange(3800), x[100:3900], '--')
plt.plot(np.arange(437, 3562), x[537:3662])

plt.subplot(212)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Measurement $x(t)$')
plt.plot(np.arange(3800), x[100:3900], '--')
plt.plot(np.arange(322, 3447), x[422:3547])

y1 = surrogates.iaaft(x[537:3663])[0]
y2 = surrogates.iaaft(x[422:3547])[0]

plt.figure(2)

plt.subplot(211)
plt.title(r'Surrogate time series')
plt.ylabel(r'Measurement $x(t)$')
plt.plot(y1[:500])

plt.subplot(212)
plt.xlabel(r'Time $t$')
plt.ylabel(r'Measurement $x(t)$')
plt.plot(y2[:500])

plt.show()
