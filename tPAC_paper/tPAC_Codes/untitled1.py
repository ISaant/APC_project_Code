#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:20:33 2023

@author: sflores
"""

def FIR(x,fs,freqs,width):
    
    #1) Design filter
    N=int(fs/2)
    if (N % 2) == 0:
        N+=1
    
    t=np.arange(len(x))/fs
    cutoff_hz = np.array(freqs)/nyq_rate
    taps = firwin(N, cutoff_hz,pass_zero='bandpass' )
    x3= 2*sin(2*pi*130*t)
    
    #2) Mirror signal 
    half_len=round(len(x)/2)
    pad=min(N,half_len)
    print(pad)
    x2=np.lib.pad(x,(pad+10,pad+10),'reflect')
    
    #3) Filter Signal
    filtered_x = lfilter(taps, 1.0, x2)
    figure(0)
    plot(filtered_x)
    #4) Crop signal
    filtered_x=filtered_x[pad+10:-(pad+10)]
    
    #Plot
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

    figure(3)
    plot(t, x, 'g')
    plot(t, x3, '--k')
    plot(t, filtered_x, 'r-')


    xlabel('t')
    grid(True)

    show()
    
fs = 1000.0
t = arange(5000) / fs
x = cos(2*pi*120*t) + 2*sin(2*pi*250*t) + \
        2*sin(2*pi*130*t) + sin(2*pi*40*t) + \
            3*sin(2*pi*23.45*t)
            
FIR(x,fs,[129,131],width=10)