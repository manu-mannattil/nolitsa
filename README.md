NoLiTSA
=======

NoLiTSA (<b>No</b>n<b>Li</b>near <b>T</b>ime <b>S</b>eries
<b>A</b>nalysis) is a Python module implementing several standard
algorithms used in nonlinear time series analysis.

[![Build Status](https://travis-ci.org/manu-mannattil/nolitsa.svg?branch=master)](https://travis-ci.org/manu-mannattil/nolitsa)
[![Coverage Status](https://coveralls.io/repos/github/manu-mannattil/nolitsa/badge.svg)](https://coveralls.io/github/manu-mannattil/nolitsa)


Features
--------

-   Estimation of embedding delay using autocorrelation, delayed mutual
    information, and reconstruction expansion.
-   Embedding dimension estimation using false nearest neighbors and
    averaged false neighbors.
-   Computation of correlation sum and correlation dimension from both
    scalar and vector time series.
-   Estimation of the maximal Lyapunov exponent from both scalar and
    vector time series.
-   Generation of FT, AAFT, and IAAFT surrogates from a scalar
    time series.
-   Simple noise reduction scheme for filtering deterministic
    time series.
-   Miscellaneous functions for end point correction, stationarity
    check, fast near neighbor search, etc.

Installation
------------

NoLiTSA can be installed using pip by running

    git clone https://github.com/manu-mannattil/nolitsa.git
    pip install ./nolitsa

or, with newer versions of pip:

    pip install git+https://github.com/manu-mannattil/nolitsa.git

NoLiTSA requires NumPy and SciPy, and (theoretically) works with both
Python 2 and 3.

### Tests

NoLiTSA’s unit tests can be executed by running `nosetests`, or by using

    python setup.py test

Publications
------------

Versions of NoLiTSA were used in the following publications:

-   M. Mannattil, H. Gupta, and S. Chakraborty, “Revisiting Evidence of
    Chaos in X-ray Light Curves: The Case of GRS 1915+105,”
    [Astrophys. J. **833**,
    208 (2016)](https://dx.doi.org/10.3847/1538-4357/833/2/208).

-   M. Mannattil, A. Pandey, M. K. Verma, and S. Chakraborty, “On the
    applicability of low-dimensional models for convective flow
    reversals at extreme Prandtl numbers,” [Eur. Phys. J. B **90**, 259
    (2017)](https://dx.doi.org/10.1140/epjb/e2017-80391-1)

Acknowledgments
---------------

Sagar Chakraborty is thanked for several critical discussions.

License
-------

NoLiTSA is licensed under the 3-clause BSD license. See the file LICENSE
for more details.
