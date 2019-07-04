#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from nolitsa import data, delay

sample = 0.01

x_random = np.random.random(10000)

a, b = np.random.random(2)
t = np.arange(10000)
x_line = a + b * t

t = np.linspace(0, 100 * np.pi, 500)
x_circle = np.sin(t)

x_zeros = np.zeros(1000)
x_ones = np.zeros(1000)

dim = np.arange(2, 7, 1)
maxtau = 60

ilds = delay.ild(x_random, dim=dim, qmax=4, maxtau=maxtau, rp=0.04, nrefp=0.02,
                 k=None)
plt.figure(1)
plt.title('ILD for random time series')
plt.xlabel('Time delay')
plt.ylabel('ILD')
for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')
plt.legend()
plt.savefig('/home/miroslav/ild_random.png')

ilds = delay.ild(x_circle, dim=dim, qmax=4, maxtau=maxtau, rp=0.04, nrefp=0.02,
                 k=None)
plt.figure(2)
plt.title('ILD for circle series')
plt.xlabel('Time delay')
plt.ylabel('ILD')
for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')
plt.legend()
plt.savefig('/home/miroslav/ild_circle.png')


ilds = delay.ild(x_line, dim=dim, qmax=4, maxtau=maxtau, rp=0.04, nrefp=0.02,
                 k=None)
plt.figure(3)
plt.title('ILD for line time series')
plt.xlabel('Time delay')
plt.ylabel('ILD')
for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()
plt.savefig('/home/miroslav/ild_line.png')

ilds = delay.ild(x_ones, dim=dim, qmax=4, maxtau=maxtau, rp=0.04, nrefp=0.02,
                 k=None)
plt.figure(4)
plt.title('ILD for ones time series')
plt.xlabel('Time delay')
plt.ylabel('ILD')
for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()
plt.savefig('/home/miroslav/ild_ones.png')


ilds = delay.ild(x_zeros, dim=dim, qmax=4, maxtau=maxtau, rp=0.04, nrefp=0.02,
                 k=None)
plt.figure(5)
plt.title('ILD for ones time series')
plt.xlabel('Time delay')
plt.ylabel('ILD')
for d, ild in zip(dim, ilds):
    plt.plot(np.arange(1, maxtau+1), ild, label=f'm = {d}')

plt.legend()
plt.savefig('/home/miroslav/ild_zeros.png')
