#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:00:20 2023

@author: sflores
"""

import numpy as np
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show


#------------------------------------------------
# Create a signal for demonstration.
#------------------------------------------------

sample_rate = 1000.0
nsamples = 1000
t = arange(nsamples) / sample_rate
x = cos(2*pi*120*t) + 2*sin(2*pi*250*t+0.1) + \
        2*sin(2*pi*100*t) + sin(2*pi*40*t + 0.1) + \
            3*sin(2*pi*23.45*t+.8)
            

            



#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist  rate.  We'll design the filter
# with a 5 Hz transition width.
width = 5/nyq_rate

# The desired attenuation in the stop band, in dB.
ripple_db = 48.0

# Compute the order and Kaiser parameter for the FIR filter.
N, beta = kaiserord(ripple_db, width)
# N=int(sample_rate/2)
if (N % 2) == 0:
    N+=1

# The cutoff frequency of the filter.
cutoff_hz = np.array([50,140])

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta),pass_zero='bandpass' )

# Use lfilter to filter x with the FIR filter.
half_len=round(len(x)/2)
pad=min(N,half_len)
x2=np.lib.pad(x,(pad+10,pad+10),'reflect')
filtered_x = lfilter(taps, 1.0, x2)
filtered_x=filtered_x[pad+10:-(pad+10)]
x2=cos(2*pi*120*t) + 2*sin(2*pi*100*t)
#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
grid(True)

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------

figure(2)
clf()
w, h = freqz(taps, worN=8000)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
# ylim(-0.05, 1.05)
grid(True)

# Upper inset plot.
ax1 = axes([0.42, 0.6, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
# xlim(0,8.0)
# ylim(0.9985, 1.001)
grid(True)

# Lower inset plot
ax2 = axes([0.42, 0.25, .45, .25])
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
# xlim(12.0, 20.0)
# ylim(0.0, 0.0025)
grid(True)

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------

# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(3)
# Plot the original signal.
# plot(t, x, 'k')
plot(t, x2, '--k')
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t, filtered_x, 'r-')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
# plot(t, filtered_x, 'g', linewidth=4)

xlabel('t')
grid(True)

show()