# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np
from nolitsa import delay
from numpy.testing import assert_allclose, run_module_suite


class TestAcorr(object):
    # Test delay.acorr()

    def test_random(self):
        # Test by calculating autocorrelation by brute force.
        n = 32
        x = np.random.random(n)
        x = x - x.mean()

        desired = np.empty(n)
        desired[0] = np.sum(x ** 2)

        for i in range(1, n):
            desired[i] = np.sum(x[:-i] * x[i:])

        desired = desired / desired[0]
        assert_allclose(delay.acorr(x), desired)

    def test_sin(self):
        # Test using a finite sine wave.
        #
        # Autocorrelation function of a /finite/ sine wave over n
        # cycles is:
        #
        #   r(tau) = [(2*n*pi - tau)*cos(tau) + sin(tau)] / 2*n*pi
        #
        # As n -> infty, r(tau) = cos(tau) as expected.
        n = 2 ** 5
        t = np.linspace(0, n * 2 * np.pi, n * 2 ** 10)
        x = np.sin(t)

        desired = ((np.cos(t) * (2 * n * np.pi - t) + np.sin(t)) /
                   (2 * n * np.pi))
        assert_allclose(delay.acorr(x), desired, atol=1E-5)


def test_mi():
    # Test delay.mi()
    # Silly tests will have to do for now.
    x = np.random.normal(loc=5.0, size=100)
    y = np.random.normal(loc=-5.0, size=100)
    assert_allclose(delay.mi(x, y), delay.mi(y, x))

    bins = 128
    x = np.arange(50 * bins)
    assert_allclose(delay.mi(x, x, bins=bins), np.log2(bins))


def test_adfd():
    # Test delay.adfd()
    # Embed a straight line.
    a, b = 1.0 + np.random.random(2)
    t = np.arange(1000)
    x = a + b * t

    # Sum of squares of first n natural numbers.
    sqsum = lambda n: n * (n + 1) * (2 * n + 1) / 6.0

    dim, maxtau = 7, 25
    desired = np.sqrt(sqsum(dim - 1)) * b * np.arange(maxtau)
    assert_allclose(delay.adfd(x, dim=dim, maxtau=maxtau), desired)


if __name__ == '__main__':
    run_module_suite()
