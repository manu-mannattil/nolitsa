# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from nolitsa import data, utils
from numpy.testing import assert_allclose, run_module_suite


def test_falpha():
    # Tests data.falpha()
    x = data.falpha(length=(2 ** 10), mean=np.pi, var=np.e)
    assert_allclose(np.mean(x), np.pi)
    assert_allclose(np.std(x) ** 2, np.e)

    for length in (2 ** 10, 3 ** 7):
        for alpha in (1.0, 2.0, 3.0):
            mean, var = 1.0 + np.random.random(2)
            x = data.falpha(alpha=alpha, length=length,
                            mean=mean, var=var)

            # Estimate slope of power spectrum.
            freq, power = utils.spectrum(x)
            desired = np.mean(np.diff(np.log(power[1:])) /
                              np.diff(np.log(freq[1:])))

            assert_allclose(-alpha, desired)


if __name__ == '__main__':
    run_module_suite()
