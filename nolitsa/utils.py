import numpy as np

def gprange(start, end, num=100):
    """Return a geometric progression between start and end.

    Returns a geometric progression between start and end (inclusive).

    Parameters
    ----------
    start : float
        Starting point of the progression.
    end : float
        Ending point of the progression.
    num : int, optional (default = 100)
        Number of points between start and end (inclusive).

    Returns
    -------
    gp : array
        Required geometric progression.
    """
    if end / start > 0:
        ratio = (1.0 * end / start) ** (1.0 / (num - 1))
    elif end / start < 0 and num % 2 == 0:
        ratio = -abs(1.0 * end / start) ** (1.0 / (num - 1))
    else:
        raise ValueError('If start and end have different signs, '
                         'a real ratio is possible iff num is even.')

    return start * ratio ** np.arange(num)


def rescale(x, interval=(0, 1)):
    """Rescale the given scalar time series into a desired interval.

    Rescales the given scalar time series into a desired interval using
    a simple linear transformation.

    Parameters
    ----------
    x : array_like
        Scalar time series.
    interval: tuple, optional (default = (0, 1))
        Extent of the interval specified as a tuple.

    Returns
    -------
    y : array
        Rescaled scalar time series.
    """
    x = np.asarray(x)
    if interval[1] == interval[0]:
        raise ValueError('Interval must have a nonzero length.')

    return (interval[0] + (x - np.min(x)) * (interval[1] - interval[0]) /
            (np.max(x) - np.min(x)))


