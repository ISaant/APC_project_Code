#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:49:28 2023

@author: isaac
"""
import os
import pandas as pd 
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cmath
import mne

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

# %%
plt.close('all')
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+'/Simulations/'

def filt(freqs,sig):
    b,a=signal.butter(3,[freqs[0]/(fs/2), freqs[1]/(fs/2)],btype='bandpass')
    sig=signal.filtfilt(b,a,sig)
    return sig
#Preprocessing Parameters
fs=1000
high_freq=[60,120]
low_freq=[8,12]
winsize=fs

# read signal
x=pd.read_csv(Path2openSim+'x.csv',header=None).to_numpy().reshape(6001,)
t=pd.read_csv(Path2openSim+'t.csv',header=None).to_numpy().reshape(6001,)

#PAC
x_high=filt([60,120],x)
analytic_signal_high=signal.hilbert(x_high)
envelope = np.abs(analytic_signal_high)
f,Pxx_high= signal.welch(envelope,fs=fs,nperseg=3000,noverlap=1500)
fp=f[np.argmax(Pxx_high)]
x_low=filt([fp-2,fp+2],x)
analytic_signal_low=signal.hilbert(x_low)
phase_low=np.angle(analytic_signal_low)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.polar(phase_low,envelope, 'g.',alpha=.2)
complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
mean_amp=np.abs(np.mean(complex_vectors))
mean_phase=np.angle(np.mean(complex_vectors))
plt.polar(mean_phase,mean_amp, 'r.')
plt.show()

#tPAC
tPAC=np.zeros((20,11))
tPAC_phase=np.zeros((20,11))
fP=[]

freqs,step=np.linspace(high_freq[0], high_freq[1], 20, retstep = True)
for i,freq in enumerate(freqs):
    Windows=np.arange(0,int(len(x)-winsize),500)
    for j,win in enumerate(Windows):
        x_high=filt([freq,freq+step],x[win:win+winsize])
        analytic_signal_high=signal.hilbert(x_high)
        envelope = np.abs(analytic_signal_high)
        # Pxx_high,f= mne.time_frequency.psd_array_multitaper(envelope, sfreq=fs, fmin=3, adaptive=True, n_jobs=-1, verbose=False)
        f,Pxx_high= signal.welch(envelope,fs=fs)
        fp=f[np.argmax(Pxx_high)]
        fP.append(fp)
        x_low=filt([fp-2,fp+2],x[win:win+winsize])
        analytic_signal_low=signal.hilbert(x_low)
        phase_low=np.angle(analytic_signal_low)
        complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
        mean_amp=np.abs(np.mean(complex_vectors))
        mean_phase=np.angle(np.mean(complex_vectors))
        tPAC[i,j]=mean_amp
        tPAC_phase[i,j]=mean_phase
        
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid((Windows+500)/fs, freqs)
ax.plot_surface(X, Y, tPAC, cmap='jet')

ax.set_xlabel('time')
ax.set_ylabel('FA')
ax.set_zlabel('Coupling strength')

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

Colors = linear_gradient("#C33764 ", "#1D2671", 20)
for PAC_Phase,PAC_time,c in zip(tPAC_phase,tPAC,Colors['hex']):
    for phase,pac in zip(PAC_Phase,PAC_time):
        plt.polar(phase,pac,color=c,marker='.',alpha=.2)
    plt.polar(np.mean(PAC_Phase),np.mean(PAC_time),color=c,marker='.',markersize=10)
plt.annotate(f"75.7°", (1, .007),fontsize=15)

fig = plt.figure()
plt.hist(fp)

plt.show()