#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='nolitsa',
    version='0.1',
    description='A rudimentary Python module for nonlinear time series analysis',
    long_description="""\
NoLiTSA is a rudimentary Python module that implements some standard
algorithms used for nonlinear time series analysis.""",
    keywords='chaos nonlinear time series analysis',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python'
        'Programming Language :: Python :: 3',
    ],
    url='https://github.com/manu-mannattil/nolitsa',
    author='Manu Mannattil',
    author_email='manu.mannattil@gmail.com',
    license='BSD',
    packages=['nolitsa'],
    install_requires=['numpy>=1.8.1', 'scipy>=0.13.3'],
    test_suite='nose.collector',
    tests_require=['nose>=1.3.1'],
    include_package_data=True,
    zip_safe=False
)
