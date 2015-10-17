import numpy as np
from nolitsa import utils
from numpy.testing import assert_, assert_allclose, run_module_suite


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


if __name__ == '__main__':
    run_module_suite()
