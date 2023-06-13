#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:49:28 2023

@author: isaac
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cmath
import mne
from tqdm import tqdm
from scipy import signal
from numpy import cos, sin, pi, absolute, arange
from scipy.signal import kaiserord, lfilter, firwin, freqz, firls, filtfilt

## color gradilent code =======================================================
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex, n):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

# %% FIR filter================================================================
def Kaisser_FIR(x,fs,freqs,width):
    
    #1) Design filter
    nyq_rate = fs / 2.0
    width = 5/nyq_rate #Change this to max(diff,max(fp))
    ripple_db = 48.0
    N, beta = kaiserord(ripple_db, width)
    while N>=int(fs/4):
        # print('in')
        N, beta = kaiserord(ripple_db, width)
        ripple_db -= 3
    if (N % 2) == 0:
        N+=1
    print(N)
    t=np.arange(len(x))/fs
    cutoff_hz = np.array(freqs)/nyq_rate
    taps = firwin(N, cutoff_hz, window=('kaiser', beta),pass_zero='bandpass' )
    # x3=cos(2*pi*120*t) + 2*sin(2*pi*100*t)
    
    #2) Mirror signal 
    half_len=round(len(x)/2)
    pad=min(N,half_len)
    # x2=np.lib.pad(x,(pad+10,pad+10),'reflect')
    
    #3) Filter Signal
    filtered_x = lfilter(taps, 1.0, x)
    
    #4) Crop signal
    delay = int(0.5 * (N-1))
    filtered_x=filtered_x[delay:-(delay)]
    
    #Plot
    plt.figure(1)
    plt.plot(taps, 'bo-', linewidth=2)
    plt.title('Filter Coefficients (%d taps)' % N)
    plt.grid(True)

    #------------------------------------------------
    # Plot the magnitude response of the filter.
    #------------------------------------------------

    plt.figure(2)
    plt.clf()
    w, h = freqz(taps, worN=8000)
    plt.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.title('Frequency Response')
    plt.grid(True)

    plt.figure(3)
    plt.plot(t, x, 'g')
    # plt.plot(t, x3, '--k')
    plt.plot(t[delay:-delay]-delay/fs, filtered_x, 'r-')
    plt.xlabel('t')
    plt.grid(True)

    plt.show()
    return filtered_x

#%% 
def Hamming_FIR(x,fs,freqs, plot):
    
    #1) Design filter
    
    N=int(fs/4)
    if (N % 2) == 0:
        N+=1
    # print(N)
    nyq_rate = fs / 2.0
    t=np.arange(len(x))/fs
    cutoff_hz = np.array(freqs)/nyq_rate
    taps = firwin(N, cutoff_hz,pass_zero='bandpass' )
    # x3=cos(2*pi*120*t) + 2*sin(2*pi*100*t)
    
    #2) Mirror signal 
    # x2=np.lib.pad(x,(pad+10,pad+10),'reflect')
    
    #3) Filter Signal
    filtered_x = lfilter(taps, 1.0, x)
    
    #4) Crop signal
    delay = int(0.5 * (N-1))
    # filtered_x=filtered_x[delay:-(delay)]
    
    if plot:
    #Plot
        plt.figure(1)
        plt.plot(taps, 'bo-', linewidth=2)
        plt.title('Filter Coefficients (%d taps)' % N)
        plt.grid(True)
    
        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------
    
        plt.figure(2)
        plt.clf()
        w, h = freqz(taps, worN=8000)
        plt.plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.grid(True)
    
        plt.figure(3)
        plt.plot(t, x, 'g')
        # plt.plot(t, x3, '--k')
        plt.plot(t[delay:]-delay/fs, filtered_x[delay:], 'r-')
        plt.xlabel('t')
        plt.grid(True)
    
        plt.show()
    return filtered_x[delay:],delay

#%%
def eegfilt_Mod(data,fs,freqs, plot):
    
    #1) Design filter
    nyq = fs * 0.5  # Nyquist frequency
    MINFREQ = 0
    minfac = 3  # this many (lo)cutoff-freq cycles in filter, filtfilt demands it 
    min_filtorder = 15  # minimum filter length
    trans = 0.15  # fractional width of transition zones
    locutoff=freqs[0]
    hicutoff=freqs[1]
    filtorder = 0
    revfilt = 0

    
    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError("locutoff > hicutoff")
       
    if locutoff < 0 or hicutoff < 0:
        raise ValueError("locutoff or hicutoff < 0")
       
    if locutoff > nyq:
        raise ValueError("Low cutoff frequency cannot be > fs/2")
       
    if hicutoff > nyq:
        raise ValueError("High cutoff frequency cannot be > fs/2")
       
   

       
    if filtorder == 0 or filtorder is None:
        if locutoff > 0:
            filtorder = minfac * int(fs / locutoff)
        elif hicutoff > 0:
            filtorder = minfac * int(fs / hicutoff)
       
        if filtorder < min_filtorder:
            filtorder = min_filtorder
      
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
    
    # print(filtorder,f,m)
    if filtorder%2 == 1:
        filtorder+=1
    filtwts = firls(filtorder+1, f, m)  # get FIR filter coefficients
       
    # smoothdata = np.zeros((frames))
    # for e in range(epochs):  # filter each epoch, channel
    #Mirror signal 
    pad=len(data)
    data=np.lib.pad(data,(pad,pad),'reflect')
    data=np.concatenate((np.zeros(fs*2),data,np.zeros(fs*2)))
    filtered_x = filtfilt(filtwts, 1, data)[pad+(fs*2):pad*2+(fs*2)]
    # filtered_x = lfilter(filtwts, 1.0, data)
    # delay = int(0.5 * (filtorder-1))
    # print(len(filtered_x),delay)
       
            # if epochs == 1 and c % 20 != 0:
            #     print('.', end='')
            # elif epochs == 1:
            #     print(c, end='')
       
    # return smoothdata, filtwts
    
    #3) Filter Signal
    # filtered_x = lfilter(taps, 1.0, x)
    
    #4) Crop signal
    # delay = int(0.5 * (N-1))
    # filtered_x=filtered_x[delay:-(delay)]
    
    if plot:
    #Plot
        plt.figure()
        plt.plot(filtwts, 'bo-', linewidth=2)
        plt.title('Filter Coefficients (%d filtwts)' % filtorder)
        plt.grid(True)
    
        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------
    
        plt.figure()
        plt.clf()
        w, h = freqz(filtwts, worN=8000)
        plt.plot((w/pi)*nyq, absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.grid(True)
    
        plt.figure()
        plt.plot(t, x, 'g')
        # plt.plot(t, x3, '--k')
        plt.plot(t, filtered_x, 'r-')
        plt.xlabel('t')
        plt.grid(True)
    
        plt.show()
    # return filtered_x[delay:], filtwts
    return filtered_x, filtwts

#%%
def eegfilt(data,fs,freqs, plot):
    
    #1) Design filter
    nyq = fs * 0.5  # Nyquist frequency
    MINFREQ = 0
    minfac = 3  # this many (lo)cutoff-freq cycles in filter, filtfilt demands it 
    min_filtorder = 15  # minimum filter length
    trans = 0.15  # fractional width of transition zones
    locutoff=freqs[0]
    hicutoff=freqs[1]
    filtorder = 0
    revfilt = 0

    
    if locutoff > 0 and hicutoff > 0 and locutoff > hicutoff:
        raise ValueError("locutoff > hicutoff")
       
    if locutoff < 0 or hicutoff < 0:
        raise ValueError("locutoff or hicutoff < 0")
       
    if locutoff > nyq:
        raise ValueError("Low cutoff frequency cannot be > fs/2")
       
    if hicutoff > nyq:
        raise ValueError("High cutoff frequency cannot be > fs/2")
       
   

       
    if filtorder == 0 or filtorder is None:
        if locutoff > 0:
            filtorder = minfac * int(fs / locutoff)
        elif hicutoff > 0:
            filtorder = minfac * int(fs / hicutoff)
       
        if filtorder < min_filtorder:
            filtorder = min_filtorder
      
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
    
    # print(filtorder,f,m)
    if filtorder%2 == 1:
        filtorder+=1
    filtwts = firls(filtorder+1, f, m)  # get FIR filter coefficients
       
    # smoothdata = np.zeros((frames))
    # for e in range(epochs):  # filter each epoch, channel
    pad=len(data)
    data=np.lib.pad(data,(pad,pad),'reflect')
    data=np.concatenate((np.zeros(fs),data,np.zeros(fs)))
    filtered_x = filtfilt(filtwts, 1, data)[pad+(fs):pad*2+(fs)]
       
            # if epochs == 1 and c % 20 != 0:
            #     print('.', end='')
            # elif epochs == 1:
            #     print(c, end='')
       
    # return smoothdata, filtwts
    
    #3) Filter Signal
    # filtered_x = lfilter(taps, 1.0, x)
    
    #4) Crop signal
    # delay = int(0.5 * (N-1))
    # filtered_x=filtered_x[delay:-(delay)]
    
    if plot:
    #Plot
        plt.figure()
        plt.plot(filtwts, 'bo-', linewidth=2)
        plt.title('Filter Coefficients (%d filtwts)' % filtorder)
        plt.grid(True)
    
        #------------------------------------------------
        # Plot the magnitude response of the filter.
        #------------------------------------------------
    
        plt.figure()
        plt.clf()
        w, h = freqz(filtwts, worN=8000)
        plt.plot((w/pi)*nyq, absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.grid(True)
    
        plt.figure()
        plt.plot(t, x, 'g')
        # plt.plot(t, x3, '--k')
        plt.plot(t, filtered_x, 'r-')
        plt.xlabel('t')
        plt.grid(True)
    
        plt.show()
    return filtered_x, filtwts

#%% FIR Notch
def notchFIR(notchFreq,signal,fs):
    fL = notchFreq-3  # Cutoff frequency.
    fH = notchFreq+3  # Cutoff frequency.
    NL = 261  # Filter length for roll-off at fL, must be odd.
    NH = 261  # Filter length for roll-off at fH, must be odd.
    
    # Compute a low-pass filter with cutoff frequency fL.
    hlpf = np.sinc(2 * fL / fs * (np.arange(NL) - (NL - 1) / 2))
    hlpf *= np.blackman(NL)
    hlpf /= np.sum(hlpf)
    
    # Compute a high-pass filter with cutoff frequency fH.
    hhpf = np.sinc(2 * fH / fs * (np.arange(NH) - (NH - 1) / 2))
    hhpf *= np.blackman(NH)
    hhpf /= np.sum(hhpf)
    hhpf = -hhpf
    hhpf[(NH - 1) // 2] += 1
    
    # Add both filters.
    if NH >= NL:
        h = hhpf
        h[(NH - NL) // 2 : (NH - NL) // 2 + NL] += hlpf
    else:
        h = hlpf
        h[(NL - NH) // 2 : (NL - NH) // 2 + NH] += hhpf
    filtered_x = filtfilt(h, 1, signal)
    return filtered_x
#%%
def relu(x):
    y=[np.max((i,.1)) for i in x]
    return y

#%%
def relu2(x):
    y=[np.max((i,0)) for i in x]
    return y
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
# %% Read data
plt.close('all')
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+'/Simulations/'




#Preprocessing Parameters
fs=1000
high_freq=[50,140]
low_freq=[2,12]
winsize=1024

# read signal
x=pd.read_csv(Path2openSim+'x4.csv',header=None).to_numpy().reshape(6001,)[:2561]
x=notchFIR(60, x, fs)
t=pd.read_csv(Path2openSim+'t.csv',header=None).to_numpy().reshape(6001,)[:2561]
data_length=len(x)
#%% Comodulogram

PhaseFreqVector = np.arange(low_freq[0], low_freq[1], 1)
AmpFreqVector = np.arange(high_freq[0], high_freq[1], 1)
PhaseFreq_BandWidth = 4
AmpFreq_BandWidth = 20

# Define phase bins
nbin = 36
position = np.linspace(-np.pi, np.pi - 2 * np.pi / nbin, nbin)

# Filtering and Hilbert transform
Comodulogram = np.zeros((len(PhaseFreqVector), len(AmpFreqVector)))
AmpFreqTransformed = np.zeros((len(AmpFreqVector), data_length))
PhaseFreqTransformed = np.zeros((len(PhaseFreqVector), data_length))

for ii, Af1 in enumerate(AmpFreqVector):
    Af2 = Af1 + AmpFreq_BandWidth
    # Perform filtering
    AmpFreq=eegfilt_Mod(x,fs,[Af1,Af2],False)[0]
    # AmpFreq = filtfilt(b, a, AmpFreq)
    AmpFreqTransformed[ii, :] = np.abs(signal.hilbert(AmpFreq))  # Getting the amplitude envelope

for jj, Pf1 in enumerate(PhaseFreqVector):
    Pf2 = Pf1 + PhaseFreq_BandWidth
    PhaseFreq = x
    # Perform filtering
    PhaseFreq=eegfilt_Mod(x,fs,[Pf1,Pf2],False)[0]
    PhaseFreqTransformed[jj, :] = np.angle(signal.hilbert(PhaseFreq))  # Getting the phase time series

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
plt.contourf(PhaseFreqVector + PhaseFreq_BandWidth / 2, 
             AmpFreqVector + AmpFreq_BandWidth / 2, 
             Comodulogram.T, 30, cmap='jet')
plt.xlabel('Phase Frequency (Hz)')
plt.ylabel('Amplitude Frequency (Hz)')
plt.colorbar()
plt.show()

#%%PAC, Here there is no sliding window
x_high,_=eegfilt_Mod(x,fs,high_freq,True)
analytic_signal_high=signal.hilbert(x_high)
envelope = np.abs(analytic_signal_high)
f,Pxx_high= signal.welch(envelope,fs=fs,nperseg=winsize,noverlap=winsize/2)
fp=f[np.argmax(Pxx_high)]
x_low,delay=eegfilt_Mod(x, fs,[fp-2,fp+2],True)
analytic_signal_low=signal.hilbert(x_low)
phase_low=np.angle(analytic_signal_low)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.polar(phase_low,envelope, 'g.',alpha=.2)
complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
mean_amp=np.abs(np.mean(complex_vectors))
mean_phase=np.angle(np.mean(complex_vectors))
plt.polar(mean_phase,mean_amp, 'r.')
plt.show()



#%%tPAC (PAC +  sliding window)
Windows=np.arange(0,int(len(x_high)-winsize),128)
tPAC=np.zeros((20,len(Windows)))
tPAC_phase=np.zeros((20,len(Windows)))
fP=[]

freqs,step=np.linspace(high_freq[0], high_freq[1], 20, retstep = True) 
width=max(step,max(low_freq))
sig_low,_=eegfilt_Mod(x,fs,low_freq,False)
for i,freq in enumerate(tqdm(freqs)):
    x_high,_=eegfilt_Mod(x,fs,[freq-width/2,freq+width/2],False)
    analytic_signal_high=signal.hilbert(x_high)
    for j,win in enumerate(Windows):
        envelope = np.abs(analytic_signal_high[win:win+winsize])
        # Pxx_high,f= mne.time_frequency.psd_array_multitaper(envelope, sfreq=fs, fmin=3, adaptive=True, n_jobs=-1, verbose=False)
        f,Pxx_high= signal.welch(envelope,fs=fs)
        f,Pxx_nofilter= signal.welch(sig_low[win:win+winsize],fs=fs)
        # plt.plot(f,Pxx_high/max(Pxx_high),'k')
        # plt.plot(f,Pxx_nofilter/max(Pxx_nofilter),'r')
        # corr = signal.correlate(Pxx_high, Pxx_nofilter,mode='full')
        # lags = signal.correlation_lags(len(Pxx_high), len(Pxx_nofilter), mode="full")
        # lag = lags[np.argmax(corr)]
        fp=f[np.argmax(Pxx_high)]
        fP.append(fp)
        x_low,delay=eegfilt_Mod(x[win:win+winsize],fs,relu2([fp-PhaseFreq_BandWidth/2,fp+PhaseFreq_BandWidth/2]),False)
        analytic_signal_low=signal.hilbert(x_low)
        phase_low=np.angle(analytic_signal_low)
        complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
        mean_amp=np.abs(np.mean(complex_vectors))
        mean_phase=np.angle(np.mean(complex_vectors))
        tPAC[i,j]=mean_amp
        tPAC_phase[i,j]=mean_phase
        
#%%
fig=plt.figure()
plt.imshow(tPAC, aspect='auto', cmap='jet', 
           interpolation='gaussian',
           extent=[(Windows[0]+min(t*fs))+(winsize/2),(Windows[-1]+min(t*fs))+winsize/2,round(freqs[-1]),round(freqs[0])])
plt.title('tPAC, winsize='+str(winsize))
plt.xlabel('Time [s]')
plt.ylabel('fA')
plt.colorbar()
#%%
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# X, Y = np.meshgrid((Windows+winsize/2)/fs, freqs)
# ax.plot_surface(X, Y, tPAC, cmap='jet')
# ax.view_init(azim=-90, elev=90)
# ax.set_xlabel('time')
# ax.set_ylabel('FA')
# ax.set_zlabel('Coupling strength')

#%%

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

Colors = linear_gradient("#22c1c3", "#fdbb2d", 20)
cont=0
for PAC_Phase,PAC_time,c in zip(tPAC_phase,tPAC,Colors['hex']):
    for phase,pac in zip(PAC_Phase,PAC_time):
        ax.plot(phase,pac,color=c,marker='.',alpha=.1)
    polar,=ax.plot(np.mean(PAC_Phase),np.mean(PAC_time),color=c,marker='o',markersize=10, markeredgecolor = 'k')
    polar.set_label(str(round(freqs[cont]))+' [Hz]')
    cont+=1
ax.legend(loc="lower left",bbox_to_anchor=(1.1, -0.05))
plt.title('APC, Polar plot. ')
# plt.annotate(f"75.7°", (1, .007),fontsize=15)
# plt.legend(freqs)

fig = plt.figure()
plt.hist(fP)
plt.title('fP Histogram')
plt.xlabel('fP')
plt.ylabel('Frequency')
plt.show()