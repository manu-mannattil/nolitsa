# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from nolitsa import lyapunov
from numpy.testing import assert_allclose, run_module_suite


def test_mle():
    # Test lyapunov.mle()
    # Particle moving uniformly in 7d: y(t) = a + b*t
    a = np.random.random(7)
    b = np.random.random(7)

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

        desired = np.log((window + 1) * modb)
        assert_allclose(lyapunov.mle(y, window=window, metric=metric), desired)


def test_mle_embed():
    # Test lyapunov.mle_embed()
    t = np.linspace(0, 10 * 2 * np.pi, 5000)
    y = np.array([np.sin(t), np.cos(t)]).T
    desired = lyapunov.mle(y, maxt=25)

    dim = [2]
    tau = 125
    x = y[:, 0]

    assert_allclose(desired, lyapunov.mle_embed(x, dim=dim, tau=tau,
                                                maxt=25)[0], atol=1e-1)


if __name__ == '__main__':
    run_module_suite()
