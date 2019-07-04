# -*- coding: utf-8 -*-

"""Functions to estimate embedding dimension.

This module provides a set of functions to estimate the minimum
embedding dimension required to embed a scalar time series.

  * afn -- use the averaged false neighbors method to estimate the
    minimum embedding dimension.
  * fnn -- use the false nearest neighbors method to estimate the
    minimum embedding dimension.
"""

from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np
from scipy.spatial import distance

from . import utils


def _afn(d, x, tau=1, metric='chebyshev', window=10, maxnum=None):
    """Return E(d) and E^*(d) for a single d.

    Returns E(d) and E^*(d) for the AFN method for a single d.  This
    function is meant to be called from the main afn() function.  See
    the docstring of afn( for more.)
    """
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = utils.reconstruct(x[:-tau], d, tau)
    y2 = utils.reconstruct(x, d + 1, tau)

    # Find near neighbors in dimension d.
    index, dist = utils.neighbors(y1, metric=metric, window=window,
                                  maxnum=maxnum)

    # Compute the magnification and the increase in the near-neighbor
    # distances and return the averages.
    E = utils.dist(y2, y2[index], metric=metric) / dist
    Es = np.abs(y2[:, -1] - y2[index, -1])

    return np.mean(E), np.mean(Es)


def afn(x, dim=[1], tau=1, metric='chebyshev', window=10, maxnum=None,
        parallel=True):
    """Averaged false neighbors algorithm.

    This function implements the averaged false neighbors method
    described by Cao (1997) to estimate the minimum embedding dimension
    required to embed a scalar time series.

    Parameters
    ----------
    x : array
        1-D scalar time series.
    dim : int array (default = [1])
        Embedding dimensions for which E(d) and E^*(d) should be
        computed.
    tau : int, optional (default = 1)
        Time delay.
    metric : string, optional (default = 'chebyshev')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).
    parallel : bool, optional (default = True)
        Calculate E(d) and E^*(d) for each d in parallel.

    Returns
    -------
    E : array
        E(d) for each of the d's.
    Es : array
        E^*(d) for each of the d's.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    return utils.parallel_map(_afn, dim, (x,), {
                              'tau': tau,
                              'metric': metric,
                              'window': window,
                              'maxnum': maxnum
                              }, processes).T


def _fnn(d, x, tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
         maxnum=None):
    """Return fraction of false nearest neighbors for a single d.

    Returns the fraction of false nearest neighbors for a single d.
    This function is meant to be called from the main fnn() function.
    See the docstring of fnn() for more.
    """
    # We need to reduce the number of points in dimension d by tau
    # so that after reconstruction, there'll be equal number of points
    # at both dimension d as well as dimension d + 1.
    y1 = utils.reconstruct(x[:-tau], d, tau)
    y2 = utils.reconstruct(x, d + 1, tau)

    # Find near neighbors in dimension d.
    index, dist = utils.neighbors(y1, metric=metric, window=window,
                                  maxnum=maxnum)

    # Find all potential false neighbors using Kennel et al.'s tests.
    f1 = np.abs(y2[:, -1] - y2[index, -1]) / dist > R
    f2 = utils.dist(y2, y2[index], metric=metric) / np.std(x) > A
    f3 = f1 | f2

    return np.mean(f1), np.mean(f2), np.mean(f3)


def fnn(x, dim=[1], tau=1, R=10.0, A=2.0, metric='euclidean', window=10,
        maxnum=None, parallel=True):
    """Compute the fraction of false nearest neighbors.

    Implements the false nearest neighbors (FNN) method described by
    Kennel et al. (1992) to calculate the minimum embedding dimension
    required to embed a scalar time series.

    Parameters
    ----------
    x : array
        1-D real input array containing the time series.
    dim : int array (default = [1])
        Embedding dimensions for which the fraction of false nearest
        neighbors should be computed.
    tau : int, optional (default = 1)
        Time delay.
    R : float, optional (default = 10.0)
        Tolerance parameter for FNN Test I.
    A : float, optional (default = 2.0)
        Tolerance parameter for FNN Test II.
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".  Also see Notes.
    window : int, optional (default = 10)
        Minimum temporal separation (Theiler window) that should exist
        between near neighbors.
    maxnum : int, optional (default = None (optimum))
        Maximum number of near neighbors that should be found for each
        point.  In rare cases, when there are no neighbors that are at a
        nonzero distance, this will have to be increased (i.e., beyond
        2 * window + 3).
    parallel : bool, optional (default = True)
        Calculate the fraction of false nearest neighbors for each d
        in parallel.

    Returns
    -------
    f1 : array
        Fraction of neighbors classified as false by Test I.
    f2 : array
        Fraction of neighbors classified as false by Test II.
    f3 : array
        Fraction of neighbors classified as false by either Test I
        or Test II.

    Notes
    -----
    The FNN fraction is metric depended for noisy time series.  In
    particular, the second FNN test, which measures the boundedness of
    the reconstructed attractor depends heavily on the metric used.
    E.g., if the Chebyshev metric is used, the near-neighbor distances
    in the reconstructed attractor are always bounded and therefore the
    reported FNN fraction becomes a nonzero constant (approximately)
    instead of increasing with the embedding dimension.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    return utils.parallel_map(_fnn, dim, (x,), {
                              'tau': tau,
                              'R': R,
                              'A': A,
                              'metric': metric,
                              'window': window,
                              'maxnum': maxnum
                              }, processes).T


def _ild(dim, x, maxtau=100, qmax=4, nrefp=None, k=None, rp=0.04, window=10,
         metric='euclidean'):
    def dx(i, q, y, index):
        # Helper function which computes 'absolute growth of distance' -
        # how much has the distance between point at index i and the centroid
        # of its neighbors changed after q steps?
        x0_com = np.mean(y[index[i]], axis=0)
        xq_com = np.mean(y[np.clip(index[i]+q, 0, y.shape[0]-1)], axis=0)
        return d(xq_com, y[i+q]) - d(x0_com, y[i])

    ild = np.empty(maxtau)
    d = getattr(distance, metric)
    yy = [utils.reconstruct(x, dim=dim, tau=tau) for tau in
          np.arange(1, maxtau+1)]

    for idx, y in enumerate(yy):
        n = y.shape[0]
        refp = np.arange(n-qmax) if nrefp is None else \
            np.random.choice(
                n-qmax, min(np.int(np.ceil(nrefp*n)), n-qmax), replace=False)
        if k is None:
            r = rp * utils.extent(y, metric=metric)
            index = utils.neighbors_r(y, r, metric=metric)
        else:
            index, _ = utils.neighbors(y, metric=metric, minnum=k,
                                       window=window)
        # For time delay idx+1, ILD is the average of discrete integrals of
        # local deformations around the reference points.
        ild[idx] = \
            np.average([
                np.sum(
                    [dx(i, q-1, y, index) + dx(i, q, y, index)
                     for q in np.arange(1, qmax+1)])
                for i in refp])
    return ild


def ild(x, dim=[1], maxtau=100, qmax=4, nrefp=None, k=None, rp=1.0,
        window=10, metric='euclidean', parallel=True):
    """Computes Integral Local Deformation (ILD) (Buzug & Pfister 1992).

    Parameters
    ----------
    x : array
        1-D real time series of length N.
    dim : int, optional (default = 1)
        List of embedding dimensions to find ILDs for.
    maxtau : int, optional (default = 100)
        Calculate the ILD only up to this delay.
    qmax : int, optional (default = 4)
        The number of steps each point is evolved.
    nrefp : float, optional (default = None)
        The number of randomly selected reference points in the percentage of
        the number of points in the embedding. By default, all points in the
        attractor are selected. The more points are selected, the more precise
        the results, but the higher the computational cost.
    k : int, optional (default = 20)
        Number of neighbors in radius.
    rp : float, optional (default = 1.0)
        The ratio of the attractor diameter to sample to pick neighbors from.
    window : int, optional (default = 10)
        The Theiler window (minimal temporal separation between neighbors in
        number of steps).
    metric : string, optional (default = 'euclidean')
        Metric to use for distance computation.  Must be one of
        "cityblock" (aka the Manhattan metric), "chebyshev" (aka the
        maximum norm metric), or "euclidean".

    Returns
    -------
    disp : array
        Array where each element represents ILD (as a function of time delay)
        for all time delays up to maxtau.
    """
    if parallel:
        processes = None
    else:
        processes = 1

    minv, maxv = np.min(x), np.max(x)

    if minv == maxv:
        warn('A constant time series provided.')
        return [np.zeros(maxtau)] * len(dim)

    if len(x) - (max(dim) - 1) * maxtau - qmax < 0:
        raise ValueError('In some attractors the number of evolution steps is '
                         'greater than the number of points - try increasing '
                         'the minimum embedding dimension.')

    return utils.parallel_map(_ild, dim, args=(x,), kwargs={
                              'qmax': qmax,
                              'nrefp': nrefp,
                              'maxtau': maxtau,
                              'k': k,
                              'rp': rp,
                              'window': window,
                              'metric': metric}, processes=processes) \
        / (2*(maxv - minv))
