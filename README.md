NoLiTSA
=======

NoLiTSA (**No**n**Li**near **T**ime **S**eries **A**nalysis) is
a rudimentary Python module implementing several standard algorithms
used in nonlinear time series analysis.

[![Build Status](https://travis-ci.org/manu-mannattil/nolitsa.svg?branch=master)](https://travis-ci.org/manu-mannattil/nolitsa)
[![Coverage Status](https://coveralls.io/repos/github/manu-mannattil/nolitsa/badge.svg)](https://coveralls.io/github/manu-mannattil/nolitsa)

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

Installation
------------

NoLiTSA can be installed by running

    $ pip install git+https://https://github.com/manu-mannattil/nolitsa.git

NoLiTSA requires `numpy` and `scipy`, and should (theoretically) work
with both Python 2 and 3.

### Tests

NoLiTSA's unit tests can be run by calling `nosetests` from the
`nolitsa/tests/` directory, or by using

    $ python setup.py test

Publications
------------

Various iterations of the code were used in the following
publication(s):

* M. Mannattil, H. Gupta, and S. Chakraborty, “Revisiting Evidence of Chaos in X-ray Light Curves: The Case of GRS 1915+105,” [Astrophys. J. __833__, 208 (2016)](https://dx.doi.org/10.3847/1538-4357/833/2/208).

License
-------

NoLiTSA is licensed under the 3-clause BSD license.  See the file
LICENSE for more details.
