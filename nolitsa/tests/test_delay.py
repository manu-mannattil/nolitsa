# -*- coding: utf-8 -*-

import numpy as np
from nolitsa import delay
from numpy.testing import assert_, assert_allclose, run_module_suite


def test_acorr():
    # Test delay.acorr()
    #
    # Autocorrelation function of a /finite/ sine wave over n
    # cycles is:
    #
    #   r(tau) = [cos(tau)(2*n*pi - tau) + sin(tau)] / 2*n*pi
    #
    # As n -> infty, r(tau) = cos(tau) as expected.
    n = 2 ** 5
    t = np.linspace(0, n * 2 * np.pi, n * 2 ** 10)
    x = np.sin(t)

    desired = ((np.cos(t) * (2 * n * np.pi - t) + np.sin(t)) / (2 * n * np.pi))
    assert_allclose(delay.acorr(x), desired, atol=1E-5)


if __name__ == '__main__':
    run_module_suite()
