NoLiTSA
=======

NoLiTSA (**No**n**Li**near **T**ime **S**eries **A**nalysis) is a
rudimentary Python module implementing several standard algorithms used
in nonlinear time series analysis.

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

    $ git clone https://github.com/manu-mannattil/nolitsa.git
    $ pip install nolitsa

or, with newer versions of pip:

    $ pip install git+https://github.com/manu-mannattil/nolitsa.git

NoLiTSA requires NumPy and SciPy, and should (theoretically) work with
both Python 2 and 3.

### Tests

NoLiTSA’s unit tests can be executed by running `nosetests`, or by using

    $ python setup.py test

Publications
------------

Various iterations of NoLiTSA were used in the following publications:

-   M. Mannattil, H. Gupta, and S. Chakraborty, “Revisiting Evidence of
    Chaos in X-ray Light Curves: The Case of GRS 1915+105,”
    [Astrophys. J. **833**,
    208 (2016)](https://dx.doi.org/10.3847/1538-4357/833/2/208).

-   M. Mannattil, A. Pandey, M. K. Verma, and S. Chakraborty, “On the
    applicability of low-dimensional models for convective flow
    reversals at extreme Prandtl numbers,” [Eur. Phys. J. B **90**, 259
    (2017)](https://doi.org/10.1140/epjb/e2017-80391-1)

License
-------

NoLiTSA is licensed under the 3-clause BSD license. See the file LICENSE
for more details.
