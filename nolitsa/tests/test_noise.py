# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import numpy as np

from nolitsa import noise
from numpy.testing import assert_allclose, run_module_suite


class TestNoRed(object):
    # Test noise.nored()
    def test_zero_radius(self):
        # With zero radius the function should simply return the
        # original series.
        for n in (200, 201):
            for tau in (1, 2, 3, 12, 13):
                for dim in (1, 2, 3, 12, 13):
                    x = np.random.random(n)
                    assert_allclose(noise.nored(x, r=0, tau=tau), x)

    def test_line(self):
        # We embed a line of the form x = a + i*b and do noise reduction
        # with a radius of 1.5*b.  Apart from the end points (of the
        # reconstructed series) each of which have only one neighbor,
        # all the other points have two neighbors -- the points before
        # and after in time.  It's not difficult to come up with an
        # expression for the new time series from this information.  Of
        # course the results depend on whether the embedding dimension
        # is even or odd.  So we test for all odd/even combinations of
        # length, delay, and dimension.  (TISEAN's `lazy` fails this
        # test with a moderate imprecision.)
        for n in (200, 201):
            for tau in (1, 2, 3, 12, 13):
                for dim in (1, 2, 3, 12, 13):
                    m = n - (dim - 1) * tau

                    i = np.arange(1, n + 1)
                    a, b = 1.0 + np.random.random(2)
                    x = a + i * b

                    y = noise.nored(x, dim=dim, r=(1.5 * b), tau=tau)
                    z = np.empty(n)

                    # I'll be damned if I have to derive this again.
                    if dim == 1:
                        z[0] = 0.5 * (x[0] + x[1])
                        z[-1] = 0.5 * (x[-2] + x[-1])
                        z[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
                    elif dim % 2 == 0:
                        c = tau * dim // 2

                        # Start points.
                        z[:c] = x[:c]
                        z[c] = a + (1 + c + 0.5) * b

                        # Points in the middle.
                        z[c + 1:-(c - tau + 1)] = a + (np.arange(2, m) + c) * b

                        # End points.
                        z[-(c - tau + 1)] = a + (m + c - 0.5) * b
                        if c > tau:
                            # If c <= tau, then there is only one end point.
                            z[-(c - tau):] = x[-(c - tau):]
                    else:
                        c = tau * (dim - 1) // 2

                        # Start points.
                        z[:c] = x[:c]
                        z[c] = a + (1 + c + 0.5) * b

                        # Points in the middle.
                        z[c + 1:-(c + 1)] = a + (np.arange(2, m) + c) * b

                        # End points.
                        z[-(c + 1)] = a + (m + c - 0.5) * b
                        z[-c:] = x[-c:]

                    assert_allclose(y, z)


if __name__ == '__main__':
    run_module_suite()
