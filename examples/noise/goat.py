#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cleaning the GOAT vowel.

To generate this data set, I recorded a friend saying the "GOAT vowel"
phoneme /əʊ/ (the vowel sound in "goat", "boat", etc. [1]) and took a
2 second section of the recording that looked fairly periodic.
Although the audio was recorded at 44.1 kHz, I downsampled it to
11.025 kHz to create a more manageable data set.  The original log
that Audacity produced during the recording is given below:

  Sample Rate: 44100 Hz. Sample values on linear scale. 1 channel (mono).
  Length processed: 88200 samples, 2.00000 seconds.
  Peak amplitude: 0.89125 (lin) -1.00000 dB.  Unweighted rms: -6.63167 dB.
  DC offset: -0.00015 linear, -76.75034 dB.

A fairly stationary segment of the time series can be found between
samples 9604 and 14572.  This data comes from a limit cycle whose
structure becomes more prominent after filtering.  The time delay of 14
used for embedding is the quarter of the average time period of the
oscillations.

NOTE: An audio recording of this data can be heard in the file
      "goat.mp3" in the "series" directory.

[1]: http://teflpedia.com/IPA_phoneme_/%C9%99%CA%8A/
"""

from nolitsa import noise
import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt('../series/goat.dat', usecols=[1])[9604:14572]

plt.figure(1)
plt.title('Noisy goat vowel')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + 14)$')
plt.plot(x[:-14], x[14:], '.')

y = noise.nored(x, dim=10, tau=14, r=0.2, repeat=5)
y = y[70:-70]

plt.figure(2)
plt.title('Cleaned goat vowel')
plt.xlabel(r'$x(t)$')
plt.ylabel(r'$x(t + 14)$')
plt.plot(y[:-14], y[14:], '.')

plt.show()
