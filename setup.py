#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='nolitsa',
    version='0.1',
    description='A rudimentary Python module for nonlinear time series analysis',
    long_description="""\
NoLiTSA is a Python module that implements some standard algorithms used
for nonlinear time series analysis.""",
    keywords='chaos nonlinear time series analysis',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python'
        'Programming Language :: Python :: 3',
    ],
    author='Manu Mannattil',
    author_email='manu.mannattil@gmail.com',
    license='BSD',
    packages=['nolitsa'],
    install_requires=['numpy>=1.11.0', 'scipy>=0.17.0'],
    test_suite='nose.collector',
    tests_require=['nose>=1.3.1'],
    include_package_data=True,
    zip_safe=False
)
