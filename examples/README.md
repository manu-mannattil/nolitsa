General workflow
================

1. Check stationarity.
1. Estimate time delay (e.g., autcorrelation, mutual information, 2D/3D
   phase portraits, etc.).
1. Estimate embedding dimension (AFN, IFNN, FNN, etc.).
1. Noise reduction (if required).
1. Estimate an invariant (e.g., Lyapunov exponent, correlation
   dimension/entropy, etc.).
1. Surrogate analysis (IAAFT, FT, cycle shuffled, etc.)
1. Additional tests for nonlinearity (e.g., prediction error, etc.)
1. Conclusion.

Examples
========

* Generating test data sets
    + [Mackey–Glass system](data/mackey-glass.py)

* Estimating the time delay
    + [Autocorrelation function of a *finite* sine wave](delay/sine.py)
    + [How many bins should one take while estimating the delayed mutual
      information?](delay/dmibins.py)
    + [Delayed mutual information for map like data can give bad
      estimates](delay/henon.py)
    + Time delay estimation for
        - [Ikeda map](delay/ikeda.py)
        - [Rössler attractor](delay/roessler.py)
        - [Lorenz attractor](delay/lorenz.py)

* Averaged false neighbors (AFN), aka Cao's test
    + Averaged false neighbors for:
        - [Henon map](afn/henon.py)
        - [Ikeda map](afn/ikeda.py)
        - [Lorenz system](afn/lorenz.py)
        - [Mackey–Glass system](afn/mackey-glass.py)
        - [Rössler system](afn/roessler.py)
        - [Data from a far-infrared laser](afn/laser.py)
    + [AFN is not impervious to every stochastic data](afn/ar1.py)
    + [AFN can cause trouble with discrete data](afn/roessler-8bit.py)

* Surrogate analysis
    + Time reversal asymmetry:
        - [Skew-statistic fails for linear stochastic data](surrogates/skewnoise.py)
        - [Skew-statistic fails for Lorenz](surrogates/skewlorenz.py)

Tips
=====

While there is no dearth of good literature on nonlinear time series
analysis, here are a few things that I found to be useful in practical
situations.

1. Picking a good time delay hinges on balancing redundance and
   irrelevance between the components of the time delayed vectors and
   there are no straightforward theoretical results that help us do
   this.  Therefore, always plot two- and three-dimensional phase
   portraits of the reconstructed attractor before settling on a time
   delay and verify that the attractor (or whatever structure appears)
   looks unfolded.  Don't blindly pick values by just looking at the
   delayed mutual information or the autocorrelation function of the
   time series.

1. If possible, use the Chebyshev metric while computing the correlation
   sum $C(r)$.  The Chebyshev metric has many advantages, especially
   when we are trying to evaluate $C(r)$ after embedding the time
   series.

    * It's computationally faster than the cityblock and the Euclidean
      metric.  For larger data sets, this is an obvious advantage.

    * Distances are independent of the embedding dimension $d$ and
      always remain bounded as opposed to Euclidean and cityblock
      distances which crudely go as $\sqrt{d}$ and $d$ respectively.
      This helps in comparing the correlation sum plots at different
      embedding dimensions.

    * Since the distances always remain bounded, we can evaluate $C(r)$
      at the same $r$'s for all embedding dimensions.  And $C(r)$ at
      $r_{\text{max}} = \max_i x_i - \min_i x_i$ is $1$ regardless of
      the embedding dimension.  Of course, $C(r)$ could be $1$ even for
      $r < r_{\text{max}}$.  Nonetheless, this helps in choosing a range
      of $r$ values.

1. Ensure that temporal correlations between points are removed in all
   cases where it is known to result in spurious estimates of dimension
   and/or determinism.

1. Avoid the skew-statistic which attempts to measures asymmetry w.r.t.
   time reversal for detecting nonlinearity.  It tells very little about
   the origin of nonlinearity and fails miserably in many cases.

1. The second FNN test tests for "boundedness" of the reconstructed
   attractor.

1. When plotting the reconstructed phase space, use same scaling for all
   the axes as all $d$ coordinates are sampled from the same
   distribution.

1. If the series has a strong periodic component, a reasonable time
   delay is the quarter of the time period.

1. Some chaotic time series such as those from Lorenz attractor may
   display long-range correlations (because of its "reversing" nature).
   In such cases a delay may be determined by computing autocorrelation
   function of the *square* of the original time series.
