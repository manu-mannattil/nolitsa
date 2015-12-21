# -*- coding: utf-8 -*-

import numpy as np


def sma(x, hwin=5):
    """Compute simple moving average.

    Computes the simple moving average of a given time series.

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
    """
    if hwin > 0:
        win = 2 * hwin + 1
        y = np.cumsum(x)
        y[win:] = y[win:] - y[:-win]

        return y[win - 1:] / win
    else:
        return x
