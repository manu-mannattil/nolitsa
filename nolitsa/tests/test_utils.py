# -*- coding: utf-8 -*-

import numpy as np
from nolitsa import utils
from numpy.testing import assert_, assert_allclose, run_module_suite, raises


def test_rescale():
    # Test utils.rescale()
    x = np.random.random(100)
    y = utils.rescale(x, interval=(-np.pi, np.pi))
    assert_(abs(np.min(y)) == np.max(y) == np.pi)


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


class TestNeighbors:
    # Test utils.neighbors()

    def test_uniform_acceleration(self):
        # As test data, we use the position of a particle under constant
        # acceleration moving in a d-dimensional space.
        d = 5
        t_max = 1000
        t = np.arange(t_max).reshape(t_max, 1).repeat(d, 1)
        a = np.random.random(d)
        v0 = np.random.random(d)
        x0 = np.random.random(d)
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
        x = np.repeat(np.arange(10) ** 2, 2 * 15 + 1).reshape(310, 1)

        # It should fail when window < 15.
        for window in range(14 + 1):
            try:
                index, dists = utils.neighbors(x, window=window)
                assert False
            except:
                assert True

        # Now it should run without any problem.
        window = 15
        index, dists = utils.neighbors(x, window=window)


if __name__ == '__main__':
    run_module_suite()
