# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from nolitsa import surrogates, utils
from numpy.testing import assert_allclose, run_module_suite

def test_ft():
    # Test surrogates.ft()
    # Always test for both odd and even number of points.
    for n in (1021, 1024):
        # NOTE that zero mean series almost always causes an assertion
        # error since the relative tolerance between different "zeros"
        # can be quite large.  This is not a bug!
        x = 1.0 + np.random.random(n)
        y = surrogates.ft(x)

        assert_allclose(utils.spectrum(x)[1], utils.spectrum(y)[1])


def test_aaft():
    # Test surrogates.aaft()
    # Always test for both odd and even number of points.
    for n in (2 ** 16 - 1, 2 ** 16):
        # Input is a Gaussian transformed using f(x) = exp(1.0 + x).
        x = np.exp(1.0 + np.random.normal(size=n, scale=0.5))
        y = surrogates.aaft(x)

        assert_allclose(utils.spectrum(x)[1], utils.spectrum(y)[1], atol=1e-3)


if __name__ == '__main__':
    run_module_suite()
