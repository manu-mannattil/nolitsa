# -*- coding: utf-8 -*-

from __future__ import division

import itertools
import numpy as np

from nolitsa import d2
from numpy.testing import assert_, assert_allclose, run_module_suite


def test_d2():
    # Test d2.d2()
    # Particle moving uniformly in 5d: y(t) = a + b*t
    a = np.random.random(5)
    b = np.random.random(5)

    n = 250
    window = 15
    t = np.arange(n)
    y = a + b * t[:, np.newaxis]

    for metric in ('chebyshev', 'cityblock', 'euclidean'):
        if metric == 'chebyshev':
            modb = np.max(np.abs(b))
        elif metric == 'cityblock':
            modb = np.sum(np.abs(b))
        elif metric == 'euclidean':
            modb = np.sqrt(np.sum(b ** 2))

        minr = (window + 1) * modb
        maxr = (n - 1) * modb

        # We need to offset the r values a bit so that the the half-open
        # bins used in np.histogram get closed.
        r = np.arange(window + 1, n) * modb + 1e-10

        _, c = d2.d2(y, r=r, window=window, metric=metric)
        desired = (np.cumsum(np.arange(n - window - 1, 0, -1)) /
                   (0.5 * (n - window - 1) * (n - window)))
        assert_allclose(c, desired)

if __name__ == '__main__':
    run_module_suite()
