# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from nolitsa import dimension
from numpy.testing import assert_allclose, run_module_suite


class TestAFN(object):
    # Tests for dimension.afn()

    def test_noise(self):
        # Test dimension.afn() using uncorrelated random numbers.
        x = np.random.random(1000)
        dim = np.arange(1, 5 + 2)
        window = 10
        metric = 'chebyshev'
        E, Es = dimension.afn(x, dim=dim, metric=metric, window=window)
        _, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

        # The standard deviation of E2 should be ~ 0 for uncorrelated
        # random numbers [Ramdani et al., Physica D 223, 229 (2006)].
        # Additionally, the mean of E2 should be ~ 1.0.
        assert_allclose(np.std(E2), 0, atol=0.1)
        assert_allclose(np.mean(E2), 1, atol=0.1)

    def test_line(self):
        # Test dimension.afn() by embedding a line.
        # Particle moving uniformly in 1-D.
        a, b = np.random.random(2)
        t = np.arange(100)
        x = a + b * t
        dim = np.arange(1, 10 + 2)
        window = 10

        # Chebyshev distances between near-neighbors remain bounded.
        # This gives "cleaner" results when embedding known objects like
        # a line.  For a line, E = 1.0 for all dimensions as expected,
        # whereas it is (d + 1) / d (for cityblock) and sqrt(d + 1) /
        # sqrt(d) for Euclidean.  In both cases, E -> 1.0 at large d,
        # but E = 1.0 is definitely preferred.
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


class TestFNN(object):
    # Because of the binary magnifaction function used in the FNN test,
    # it's not easy to create unit-tests like AFN.  So we make-do with
    # silly tests.
    def test_line(self):
        x = np.linspace(0, 10, 1000)
        dim = np.arange(1, 10 + 1)

        # A line has zero FNN at all embedding dimensions.
        f1, f2, f3 = dimension.fnn(x, dim=dim, tau=1, window=0)
        np.allclose(f1, 0)
        np.allclose(f2, 0)
        np.allclose(f3, 0)

    def test_circle(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t)
        dim = np.arange(1, 10 + 1)

        # A circle has no FNN after d = 2.
        desired = np.zeros(10)
        desired[0] = 1.0

        f1 = dimension.fnn(x, dim=dim, tau=25)[0]
        np.allclose(f1, desired)

    def test_curve(self):
        t = np.linspace(0, 100 * np.pi, 5000)
        x = np.sin(t) + np.sin(2 * t) + np.sin(3 * t) + np.sin(5 * t)
        dim = np.arange(1, 10 + 1)

        # Though this curve is a deformation of a circle, it has zero
        # FNN only after d = 3.
        desired = np.zeros(10)
        desired[:2] = 1.0

        f1 = dimension.fnn(x, dim=dim, tau=25)[0]
        np.allclose(f1, desired)


if __name__ == '__main__':
    run_module_suite()
