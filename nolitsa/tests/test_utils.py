# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import itertools
import numpy as np

from time import sleep
from nolitsa import utils
from numpy.testing import assert_, assert_allclose, run_module_suite


def test_corrupt():
    # Test utils.corrupt()
    x = np.random.random(100)
    x = x - np.mean(x)
    assert_allclose(utils.corrupt(x, x, snr=16.0), 1.25 * x)


def test_dist():
    # Test utils.dist()
    x = np.random.random((100, 5))
    y = np.random.random((100, 5))

    desired = np.max(np.abs(x - y), axis=1)
    assert_allclose(utils.dist(x, y), desired)

    desired = np.sum(np.abs(x - y), axis=1)
    assert_allclose(utils.dist(x, y, metric='cityblock'), desired)

    desired = np.sqrt(np.sum((x - y) ** 2, axis=1))
    assert_allclose(utils.dist(x, y, metric='euclidean'), desired)


def test_gprange():
    # Test utils.gprange()
    num = 10
    pi = np.pi

    # Start and end are both positive.
    start, end = pi, pi * pi ** (num - 1)
    desired = pi * pi ** np.arange(num)
    assert_allclose(utils.gprange(start, end, num=num), desired)

    # Start and end have different signs.
    start, end = pi, pi * (-pi) ** (num - 1)
    desired = pi * (-pi) ** np.arange(num)
    assert_allclose(utils.gprange(start, end, num=num), desired)


class TestNeighbors(object):
    # Test utils.neighbors()

    def test_uniform_acceleration(self):
        # As test data, we use the position of a particle under constant
        # acceleration moving in a d-dimensional space.
        d = 5
        t_max = 1000
        t = np.arange(t_max)[:, np.newaxis].repeat(d, 1)
        a = 1.0 + np.random.random(d)
        v0 = 1.0 + np.random.random(d)
        x0 = 1.0 + np.random.random(d)
        x = x0 + v0 * t + 0.5 * a * t ** 2

        # Since it's uniformly accelerated motion, the closest point at
        # each instant of time is the last point visited.  (Not true
        # when t <= window, in which case it is the next point after
        # "window time" in future.)  Since the acceleration and velocity
        # have the same sign, we don't have to worry about the particle
        # reversing its motion either.
        window = 15
        index, dists = utils.neighbors(x, window=window)
        desired = np.hstack((np.arange(window + 1, 2 * window + 2,),
                             np.arange(t_max - window - 1)))
        assert_allclose(index, desired)

    def test_duplicates(self):
        # We want to make sure that the right exceptions are raised if a
        # neighbor with a nonzero distance is not found satisfying the
        # window/maxnum conditions.
        x = np.repeat(np.arange(10) ** 2, 2 * 15 + 1)[:, np.newaxis]

        # It should fail when window < 15.
        for window in range(15):
            try:
                utils.neighbors(x, window=window)
            except:
                assert True
            else:
                assert False

        # Now it should run without any problem.
        window = 15
        utils.neighbors(x, window=window)

    def test_grid(self):
        # A very simple test to find near neighbors in a 3x3x3 grid.
        dx, dy, dz = 1.0 + np.random.random(3)

        # There are probably more elegant ways to do a Cartesian
        # product, but this will have to do for now.
        grid = np.array([(dx * x, dy * y, dz * z) for x, y, z in
                         itertools.product(np.arange(10), repeat=3)])
        np.random.shuffle(grid)

        index, dists = utils.neighbors(grid)
        desired = min(dx, dy, dz)
        assert_allclose(dists, desired)

    def test_random(self):
        # We are creating a random data set whose near neighbor
        # distances are already known for all three metrics.
        d = 5
        n = 500
        x = np.arange(d * n).reshape(n, d) + 100 * np.random.random((n, d))
        desired = np.random.random(n)

        y = np.vstack((x, x + desired[:, np.newaxis]))
        np.random.shuffle(y)

        index, dists = utils.neighbors(y, metric='euclidean')
        assert_allclose(np.sort(dists),
                        np.sqrt(d) * np.sort(desired).repeat(2))

        index, dists = utils.neighbors(y, metric='cityblock')
        assert_allclose(np.sort(dists), d * np.sort(desired).repeat(2))

        index, dists = utils.neighbors(y, metric='chebyshev')
        assert_allclose(np.sort(dists), np.sort(desired).repeat(2))

    def test_maxnum(self):
        # Make sure that appropriate exceptions are raised if no nonzero
        # neighbor is found with the given maxnum.
        x = np.arange(10).repeat(15)[:, np.newaxis]

        # Should raise exceptions.
        for maxnum in range(1, 15):
            try:
                utils.neighbors(x, maxnum=maxnum)
            except:
                assert True
            else:
                assert False

        # Should work now.
        utils.neighbors(x, maxnum=15)


def _func_shm(t, ampl, omega=(0.1 * np.pi), phase=0):
    # Utility function to test utils.parallel_map()
    sleep(0.5 * np.random.random())
    return ampl * np.sin(omega * t + phase)


def test_parallel_map():
    # Test utils.parallel_map()
    tt = np.arange(5)
    ampl, omega, phase = np.random.random(3)

    desired = [_func_shm(t, ampl, omega=omega, phase=phase) for t in tt]
    kwargs = {'omega': omega, 'phase': phase}

    xx = utils.parallel_map(_func_shm, tt, args=(ampl,), kwargs=kwargs)
    assert_allclose(xx, desired)

    xx = utils.parallel_map(_func_shm, tt, args=(ampl,), kwargs=kwargs,
                            processes=1)
    assert_allclose(xx, desired)


def test_reconstruct():
    # Test utils.reconstruct()
    # We're reconstructing a circle.
    t = np.linspace(0, 10 * 2 * np.pi, 10000)
    x = np.sin(t)
    dim = 2
    tau = 250

    x1, x2 = utils.reconstruct(x, dim, tau).T
    desired = np.cos(t[:-tau])

    assert_allclose(x2, desired, atol=1e-3)


def test_rescale():
    # Test utils.rescale()
    x = 1.0 + np.random.random(100)
    y = utils.rescale(x, interval=(-np.pi, np.pi))
    assert_(abs(np.min(y)) == np.max(y) == np.pi)


def test_spectrum():
    # Test utils.spectrum()
    # Parseval's theorem.
    for length in (2 ** 10, 3 ** 7):
        x = np.random.random(length)
        power = utils.spectrum(x)[1]
        assert_allclose(np.mean(x ** 2), np.sum(power))


class TestStatcheck(object):
    def test_stationary(self):
        x = np.arange(500)
        x = np.hstack([x, x])
        assert_allclose(utils.statcheck(x)[1], 1.0)

    def test_non_stationary(self):
        x = np.arange(1000)
        assert_(utils.statcheck(x)[1] < 1E-30)


if __name__ == '__main__':
    run_module_suite()
