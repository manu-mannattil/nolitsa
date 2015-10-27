# -*- coding: utf-8 -*-

import itertools
import numpy as np

from nolitsa import dimension
from numpy.testing import assert_, assert_allclose, run_module_suite


def test_afn():
    # Test dimension.afn() using uncorrelated random numbers.
    x = np.random.random(1000)
    dim = np.arange(1, 5 + 2)
    tau = 1
    window = 10
    metric = 'chebyshev'
    E, Es = dimension.afn(x, dim=dim, tau=tau, metric=metric, window=window)
    E1, E2 = E[1:] / E[:-1], Es[1:] / Es[:-1]

    # The standard deviation of E2 should be ~ 0 for uncorrelated random
    # numbers (Ramdani et al., 2006).  Additionally, the mean of E2
    # should be ~ 1.0.
    assert_allclose(np.std(E2), 0, atol=0.1)
    assert_allclose(np.mean(E2), 1, atol=0.1)

if __name__ == '__main__':
    run_module_suite()
