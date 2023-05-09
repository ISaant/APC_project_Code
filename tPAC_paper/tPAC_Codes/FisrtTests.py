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
from scipy.signal import kaiserord, lfilter, firwin, freqz

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
def relu(x):
    y=[np.max((i,.1)) for i in x]
    return y

# %% Read data
plt.close('all')
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+'/Simulations/'




#Preprocessing Parameters
fs=1000
high_freq=[50,140]
low_freq=[2,12]
winsize=fs

# read signal
x=pd.read_csv(Path2openSim+'x2.csv',header=None).to_numpy().reshape(6001,)
t=pd.read_csv(Path2openSim+'t.csv',header=None).to_numpy().reshape(6001,)

#PAC, Here there is no sliding window
x_high,_=Hamming_FIR(x,fs,[60,140],True)
analytic_signal_high=signal.hilbert(x_high)
envelope = np.abs(analytic_signal_high)
f,Pxx_high= signal.welch(envelope,fs=fs,nperseg=3000,noverlap=1500)
fp=f[np.argmax(Pxx_high)]
x_low,delay=Hamming_FIR(x, fs,[fp-2,fp+2],True)
analytic_signal_low=signal.hilbert(x_low)
phase_low=np.angle(analytic_signal_low)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.polar(phase_low,envelope, 'g.',alpha=.2)
complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
mean_amp=np.abs(np.mean(complex_vectors))
mean_phase=np.angle(np.mean(complex_vectors))
plt.polar(mean_phase,mean_amp, 'r.')
plt.show()

#%%tPAC, with sliding window
Windows=np.arange(0,int(len(x_high)-winsize),5)
tPAC=np.zeros((20,len(Windows)))
tPAC_phase=np.zeros((20,len(Windows)))
fP=[]

freqs,step=np.linspace(high_freq[0], high_freq[1], 20, retstep = True) 
width=max(step,max(low_freq))
for i,freq in enumerate(tqdm(freqs)):
    for j,win in enumerate(Windows):
        x_high,delay=Hamming_FIR(x[win:win+winsize],fs,[freq-width/2,freq+width/2],False)
        analytic_signal_high=signal.hilbert(x_high)
        envelope = np.abs(analytic_signal_high)
        # Pxx_high,f= mne.time_frequency.psd_array_multitaper(envelope, sfreq=fs, fmin=3, adaptive=True, n_jobs=-1, verbose=False)
        f,Pxx_high= signal.welch(envelope,fs=fs)
        fp=f[np.argmax(Pxx_high)]
        fP.append(fp)
        x_low,delay=Hamming_FIR(x[win:win+winsize],fs,relu([fp-width/2,fp+width/2]),False)
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
           extent=[Windows[0]+500+delay,Windows[-1]+500+delay,round(freqs[-1]),round(freqs[0])])
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
polars=[]
for PAC_Phase,PAC_time,c in zip(tPAC_phase,tPAC,Colors['hex']):
    for phase,pac in zip(PAC_Phase,PAC_time):
        ax.plot(phase,pac,color=c,marker='.',alpha=.1)
    polar,=ax.plot(np.mean(PAC_Phase),np.mean(PAC_time),color=c,marker='o',markersize=10, markeredgecolor = 'k')
    polar.set_label(str(round(freqs[cont]))+' [Hz]')
    cont+=1
ax.legend(loc="lower left",bbox_to_anchor=(1.1, -0.05))
plt.title('tPAC, Polar plot. ')
# plt.annotate(f"75.7°", (1, .007),fontsize=15)
# plt.legend(freqs)

fig = plt.figure()
plt.hist(fP)
plt.title('fP Histogram')
plt.xlabel('fP')
plt.ylabel('Frequency')
plt.show()