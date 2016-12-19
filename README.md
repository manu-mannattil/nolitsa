NoLiTSA
=======

NoLiTSA (**No**n**Li**near **T**ime **S**eries **A**nalysis) is
a rudimentary Python module implementing several standard algorithms
used in nonlinear time series analysis.

Features
--------

* Estimation of embedding delay using autocorrelation, delayed mutual
  information, and reconstruction expansion.
* Embedding dimension estimation using false nearest neighbors and
  average false neighbors.
* Computation of correlation sum and correlation dimension from both
  scalar and vector time series.
* Estimation of the maximal Lyapunov exponent from both scalar and
  vector time series.
* Generation of FT, AAFT, and IAAFT surrogates from a scalar time
  series.
* Simple noise reduction scheme for filtering deterministic time series.
* Miscellaneous functions for end point correction, stationarity check,
  fast near neighbor search, etc.

Publications
------------

Various iterations of the code were used in the following
publication(s):

* M. Mannattil, H. Gupta, and S. Chakraborty, “Revisiting Evidence of Chaos in X-ray Light Curves: The Case of GRS 1915+105,” [Astrophys. J. __833__, 208 (2016)](https://dx.doi.org/10.3847/1538-4357/833/2/208).

License
-------

NoLiTSA is licensed under the 3-clause BSD license.  See the file
LICENSE for more details.
