# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import combinations

import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
                           run_module_suite)

from lib.nolitsa.nolitsa import dimension


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


class TestILD(object):

    def test_constant(self):
        # ILD of a constant time-series is a zero function.
        x_ones = np.ones(500)
        dim = np.arange(2, 6 + 1)
        maxtau = 40
        with assert_warns(Warning):
            assert_equal(dimension.ild(x_ones, dim=dim, maxtau=maxtau),
                         [np.zeros(maxtau)] * len(dim))

    def test_line(self):
        # The emb. dim. of a line is 1, so all ILD's should be approximately
        # equal.
        a, b = np.random.random(2)
        t = np.arange(500)
        x_line = a + b * t
        dim = np.arange(2, 6 + 1)
        maxtau = 40
        assert_allclose(dimension.ild(x_line, dim=dim, maxtau=maxtau),
                        [np.zeros(maxtau)] * len(dim), atol=1e-5, rtol=0)

    def test_random(self):
        # A random time series should have infinite emb. dim. and no optimal
        # time dimension so the ILD's should be approx. constant and not converge
        x_random = np.random.random(500)
        dim = np.arange(2, 6 + 1)
        maxtau = 40
        ilds = dimension.ild(x_random, dim=dim, maxtau=maxtau)
        for ild in ilds:
            demeaned_ild = np.abs(ild - np.mean(ild))
            assert_allclose(demeaned_ild, np.zeros(maxtau), atol=0.5, rtol=0)

        # We will just check that not all ILDs are almost equal
        if all([np.allclose(ild1, ild2, atol=1e-8, rtol=1e-8) for ild1, ild2 in
                combinations(ilds, 2)]):
            raise AssertionError('All ILDs for randomly generated time series '
                                 'are equal.')


if __name__ == '__main__':
    run_module_suite()
