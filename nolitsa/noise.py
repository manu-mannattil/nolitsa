# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree as KDTree
from nolitsa import utils

def sma(x, hwin=5):
    """Compute simple moving average.

    Computes the simple moving average (SMA) of a given time series.

    Parameters
    ----------
    x : array
        1D real input array of length N containing the time series.
    hwin : int, optional (default = 5)
        Half-window length.  Actual window size is 2*hwin + 1.

    Returns
    -------
    y : array
        Averaged array of length N - 2*hwin

    Notes
    -----
    An SMA is a linear filter and is known to distort nonlinear
    structures in the time series considerably.
    """
    if hwin > 0:
        win = 2 * hwin + 1
        y = np.cumsum(x)
        y[win:] = y[win:] - y[:-win]

        return y[win - 1:] / win
    else:
        return x



def nored(x, dim=1, r=0, repeat=1):
    """Simple noise reduction based on local phase space averaging.

    Simple noise reduction based on local phase space averaging (Kantz &
    Schreiber, 2003).

    Parameters
    ----------
    x : array
        1D real input array of length N containing the time series.
    dim : int, optional (default = 1)
        Embedding dimension.
    r : float, optional (default = 0)
        Absolute radius of neighborhood (see Notes).
    repeat: int, optional (default = 1)
        Number of iterations.

    Return
    ------
    y : ndarray
        1D real input array of length N containing the time series after
        noise reduction.

    Notes
    -----
    Choosing the right neighborhood radius is crucial for proper noise
    reduction.  A large radius will result in too much filtering.  A
    very large radius may also result in a memory overflow.  By default
    a radius of zero is used, which means that no noise reduction is
    done.  (This function is equivalent to the TISEAN program `lazy`.)
    """
    # Choose middle coordinate appropriately.
    if dim % 2 == 0:
        mid = dim / 2
    else:
        mid = (dim - 1) / 2

    y = np.copy(x)

    for rep in xrange(repeat):
        z = np.copy(y)
        ps = utils.reconstruct(y, dim=dim)

        tree = KDTree(ps)
        neighbors = tree.query_ball_tree(tree, r=r, p=np.inf)

        # State-space averaging.
        for i in xrange(len(ps)):
            y[i + mid] = np.mean(ps[neighbors[i]][:, mid])

        # Choose the average correction as the new radius.
        r = np.sqrt(np.mean((y - z) ** 2))

        # Stop as soon as the series stops changing.
        if r == 0:
            break

    return y
