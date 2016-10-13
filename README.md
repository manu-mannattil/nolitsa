NoLiTSA
=======
[![Licence: CC0/Public Domain](https://img.shields.io/badge/license-CC0-blue.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

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

License
-------

Written between 2015 and 2016 by Manu Mannattil.

To the extent possible under law, the author(s) have dedicated all
copyright and related and neighboring rights to this software to the
public domain worldwide.  This software is distributed without any
warranty.

You should have received a copy of the CC0 Public Domain Dedication
along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
