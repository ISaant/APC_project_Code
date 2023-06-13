#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:34:18 2023

@author: isaac
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

#%%
# import numpy as np
from scipy.signal import firls, filtfilt

def eegfilt(data, srate, locutoff, hicutoff, epochframes=None, filtorder=None, revfilt=None):
    if len(data.shape) > 1 and data.shape[1] == 1:
        raise ValueError("Input data should be a row vector.")

    frames = len(data)
    nyq = srate * 0.5  # Nyquist frequency
    MINFREQ = 0
    minfac = 3  # this many (lo)cutoff-freq cycles in filter, filtfilt demands it 
    min_filtorder = 15  # minimum filter length
    trans = 0.15  # fractional width of transition zones

    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError("locutoff > hicutoff")

    if locutoff < 0 or hicutoff < 0:
        raise ValueError("locutoff or hicutoff < 0")

    if locutoff > nyq:
        raise ValueError("Low cutoff frequency cannot be > srate/2")

    if hicutoff > nyq:
        raise ValueError("High cutoff frequency cannot be > srate/2")

    if filtorder is None:
        filtorder = 0

    if revfilt is None:
        revfilt = 0

    if filtorder == 0 or filtorder is None:
        if locutoff > 0:
            filtorder = minfac * int(srate / locutoff)
        elif hicutoff > 0:
            filtorder = minfac * int(srate / hicutoff)

        if filtorder < min_filtorder:
            filtorder = min_filtorder

    if epochframes is None:
        epochframes = 0

    if epochframes == 0:
        epochframes = frames

    epochs = frames // epochframes

    if epochs * epochframes != frames:
        raise ValueError("epochframes does not divide frames.")

    # if filtorder * 3 > epochframes:
    #     raise ValueError("epochframes must be at least 3 times the filtorder.")

    if (1 + trans) * hicutoff / nyq > 1:
        raise ValueError("High cutoff frequency too close to Nyquist frequency")

    if locutoff > 0 and hicutoff > 0:
        # if revfilt:
        #     print("eegfilt() - performing {}-point notch filtering.".format(filtorder))
        # else:
        #     print("eegfilt() - performing {}-point bandpass filtering.".format(filtorder))

        f = [MINFREQ, (1 - trans) * locutoff / nyq, locutoff / nyq, hicutoff / nyq, (1 + trans) * hicutoff / nyq, 1]
        m = [0, 0, 1, 1, 0, 0]
    elif locutoff > 0:
        print("eegfilt() - performing {}-point highpass filtering.".format(filtorder))
        f = [MINFREQ, locutoff * (1 - trans) / nyq, locutoff / nyq, 1]
        m = [0, 0, 1, 1]
    elif hicutoff > 0:
        print("eegfilt() - performing {}-point lowpass filtering.".format(filtorder))
        f = [MINFREQ, hicutoff / nyq, hicutoff * (1 + trans) / nyq, 1]
        m = [1, 1, 0, 0]
    else:
        raise ValueError("You must provide a non-zero low or high cut-off frequency")

    if revfilt:
        m = np.logical_not(m)
    
    print(filtorder,f,m)
    if filtorder%2 == 1:
        filtorder+=1
    filtwts = firls(filtorder+1, f, m)  # get FIR filter coefficients

    # smoothdata = np.zeros((frames))
    # for e in range(epochs):  # filter each epoch, channel
    smoothdata = filtfilt(filtwts, 1, data)

            # if epochs == 1 and c % 20 != 0:
            #     print('.', end='')
            # elif epochs == 1:
            #     print(c, end='')

    return smoothdata, filtwts

#%%


def ModIndex_v2(Phase, Amp, position):
    nbin = len(position)
    winsize = 2 * np.pi / nbin

    MeanAmp = np.zeros(nbin)
    for j in range(nbin):
        I = np.where((Phase < position[j] + winsize) & (Phase >= position[j]))
        MeanAmp[j] = np.mean(Amp[I])

    MI = (np.log(nbin) - (-np.sum((MeanAmp / np.sum(MeanAmp)) * np.log((MeanAmp / np.sum(MeanAmp)))))) / np.log(nbin)

    return MI, MeanAmp

#%%

# Constructing an example
# data_length = 2**15
# srate = 1024
# dt = 1 / srate
# t = dt * np.arange(1, data_length + 1)

fs= 1000
srate= fs
t = np.arange(1/fs,1,1/fs)
data_length=len(t)

nonmodulatedamplitude = 2
Phase_Modulating_Freq = 10
Amp_Modulated_Freq = 80

lfp = (0.2 * (np.sin(2 * np.pi * t * Phase_Modulating_Freq) + 1) + nonmodulatedamplitude * 0.1) * np.sin(
    2 * np.pi * t * Amp_Modulated_Freq) + np.sin(2 * np.pi * t * Phase_Modulating_Freq)
lfp = lfp + 1 * np.random.randn(len(lfp))

# Plotting the signal
plt.figure()
plt.subplot(1, 1, 1)
plt.plot(t, lfp)
plt.xlim([0, 1])
plt.xlabel('time (s)')
plt.ylabel('mV')

# Define the amplitude- and phase-frequencies
PhaseFreqVector = np.arange(2, 51, 2)
AmpFreqVector = np.arange(10, 201, 5)
PhaseFreq_BandWidth = 4
AmpFreq_BandWidth = 20

# Define phase bins
nbin = 18
position = np.linspace(-np.pi, np.pi - 2 * np.pi / nbin, nbin)

# Filtering and Hilbert transform
Comodulogram = np.zeros((len(PhaseFreqVector), len(AmpFreqVector)))
AmpFreqTransformed = np.zeros((len(AmpFreqVector), data_length))
PhaseFreqTransformed = np.zeros((len(PhaseFreqVector), data_length))

for ii, Af1 in enumerate(AmpFreqVector):
    Af2 = Af1 + AmpFreq_BandWidth
    AmpFreq = lfp
    # Perform filtering
    AmpFreq=eegfilt(lfp,srate,Af1,Af2)[0]
    # AmpFreq = filtfilt(b, a, AmpFreq)
    AmpFreqTransformed[ii, :] = np.abs(hilbert(AmpFreq))  # Getting the amplitude envelope

for jj, Pf1 in enumerate(PhaseFreqVector):
    Pf2 = Pf1 + PhaseFreq_BandWidth
    PhaseFreq = lfp
    # Perform filtering
    PhaseFreq=eegfilt(lfp,srate,Pf1,Pf2)[0]
    PhaseFreqTransformed[jj, :] = np.angle(hilbert(PhaseFreq))  # Getting the phase time series

# Compute MI and comodulogram
Comodulogram = np.zeros((len(PhaseFreqVector), len(AmpFreqVector)))

for ii, Pf1 in enumerate(PhaseFreqVector):
    Pf2 = Pf1 + PhaseFreq_BandWidth
    for jj, Af1 in enumerate(AmpFreqVector):
        Af2 = Af1 + AmpFreq_BandWidth
        MI, MeanAmp = ModIndex_v2(PhaseFreqTransformed[ii, :], AmpFreqTransformed[jj, :], position)
        Comodulogram[ii, jj] = MI

# Plot comodulogram
plt.figure()
plt.contourf(PhaseFreqVector + PhaseFreq_BandWidth / 2, AmpFreqVector + AmpFreq_BandWidth / 2, Comodulogram.T, 30, cmap='jet')
plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amplitude Frequency (Hz)')
plt.colorbar()
plt.show()
