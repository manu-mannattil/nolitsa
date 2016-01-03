# -*- coding: utf-8 -*-

from __future__ import division
import itertools
import numpy as np

from nolitsa import dimension
from numpy.testing import assert_, assert_allclose, run_module_suite


class TestAFN:
    # Tests for dimension.afn()

    def test_noise(self):
        # Test dimension.afn() using uncorrelated random numbers.
        x = np.random.random(1000)
        dim = np.arange(1, 5 + 2)
        window = 10
        metric = 'chebyshev'
        E, Es = dimension.afn(x, dim=dim, metric=metric, window=window)
        E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

        # The standard deviation of E2 should be ~ 0 for uncorrelated random
        # numbers (Ramdani et al., 2006).  Additionally, the mean of E2
        # should be ~ 1.0.
        assert_allclose(np.std(E2), 0, atol=0.1)
        assert_allclose(np.mean(E2), 1, atol=0.1)

    def test_line(self):
        # Test dimension.afn() by embedding a line.
        # Particle moving uniformly in 1D.
        a, b = np.random.random(2)
        t = np.arange(100)
        x = a + b * t
        dim = np.arange(1, 10 + 2)
        window = 10

        # Chebyshev distances remain bounded.  This gives "cleaner"
        # results when embedding known objects like a line.  For a line,
        # E = 1.0 for all dimensions as expected, whereas it is
        # (d + 1) / d (for citblock) and sqrt(d + 1) / sqrt(d) for
        # Euclidean.
        for metric in ('chebyshev', 'cityblock', 'euclidean'):
            Es_des = (window + 1) * b

            if metric == 'chebyshev':
                E_des = 1.0
            elif metric == 'cityblock':
                E_des = (dim + 1) / dim
            elif metric == 'euclidean':
                E_des = np.sqrt((dim + 1) / dim)

            E, Es = dimension.afn(x, dim=dim, metric=metric)

            assert_allclose(E_des, E)
            assert_allclose(Es_des, Es)

if __name__ == '__main__':
    run_module_suite()
