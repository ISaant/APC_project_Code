#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 19:29:30 2023

@author: isaac
"""
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cmath
import mne
import pywt as pw
from copy import copy
from tqdm import tqdm
from scipy import signal
from scipy.signal import periodogram, firls, filtfilt, correlate,correlation_lags

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

#%% ===========================================================================

def eegfilt_Generate(fs,freqs):
    
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
        # print("eegfilt() - performing {}-point highpass filtering.".format(filtorder))
        f = [MINFREQ, locutoff * (1 - trans) / nyq, locutoff / nyq, 1]
        m = [0, 0, 1, 1]
    elif hicutoff > 0:
        # print("eegfilt() - performing {}-point lowpass filtering.".format(filtorder))
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
    # pad=len(signal)
    # signal=np.lib.pad(signal,(pad,pad),'reflect')
    # signal=np.concatenate((np.zeros(fs*3),signal,np.zeros(fs*3)))
    # filtered_x = filtfilt(filtwts, 1, signal)[pad+(fs*3):pad*2+(fs*3)]
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
    
    # if plot:
    # #Plot
    #     plt.figure()
    #     plt.plot(filtwts, 'bo-', linewidth=2)
    #     plt.title('Filter Coefficients (%d filtwts)' % filtorder)
    #     plt.grid(True)
    
    #     #------------------------------------------------
    #     # Plot the magnitude response of the filter.
    #     #------------------------------------------------
    
    #     plt.figure()
    #     plt.clf()
    #     w, h = freqz(filtwts, worN=8000)
    #     plt.plot((w/pi)*nyq, absolute(h), linewidth=2)
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Gain')
    #     plt.title('Frequency Response')
    #     plt.grid(True)
    #     plt.figure()
    #     plt.plot(t, data, 'g')
    #     # plt.plot(t, x3, '--k')
    #     plt.plot(t, filtered_x, 'r-')
    #     plt.xlabel('t')
    #     plt.grid(True)
    
    #     plt.show()
    # return filtered_x[delay:], filtwts
    return filtwts

#%% ===========================================================================
def notchFIR_Generate(fs,notchFreq):
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
    
    return h

#%% ===========================================================================
def FIR_Apply(signal,fs,h):
    pad=len(signal)
    signal=np.lib.pad(signal,(pad,pad),'reflect')
    signal=np.concatenate((np.zeros(fs*2),signal,np.zeros(fs*2)))
    filtered_x = filtfilt(h, 1, signal)[pad+(fs*2):pad*2+(fs*2)]
    return filtered_x

#%% ===========================================================================
def relu(x):
    y=[np.max((i,.1)) for i in x]
    return y

#%% ===========================================================================
def relu2(x):
    y=[np.max((i,0)) for i in x]
    return y
#%% ===========================================================================
def ModIndex_v2(Phase, Amp, position):
    nbin = len(position)
    winsize = 2 * np.pi / nbin

    MeanAmp = np.zeros(nbin)
    for j in range(nbin):
        I = np.where((Phase < position[j] + winsize) & (Phase >= position[j]))
        MeanAmp[j] = np.mean(Amp[I])

    MI = (np.log(nbin) - (-np.sum((MeanAmp / np.sum(MeanAmp)) * np.log((MeanAmp / np.sum(MeanAmp)))))) / np.log(nbin)

    return MI, MeanAmp

#%% ===========================================================================
def freq2scale(MotherWavelet,fs,fmin,fmax):
    cf=pw.central_frequency(MotherWavelet)
    Scale=[]
    freqs=[fmax,fmin]
    for fr in freqs:
        Scale.append(np.round(np.log((cf*fs)/fr))/np.log(2))
    return Scale

#%% ===========================================================================

def meanfreq(signal,fs):
    def find_nearest(Pxx,freq, value):
        array = np.asarray(Pxx)
        idx = (np.abs(array - value)).argmin()
        return Pxx[idx], freq[idx], idx
    freq,Pxx=periodogram(signal,fs=fs)
    NormPwr=Pxx/np.sum(Pxx)
    CumSum=np.cumsum(NormPwr)
    NearesValue,CtrMassFrq, idx=find_nearest(CumSum,freq,.5)
    return CtrMassFrq

# =============================================================================
# %% Read data
plt.close('all')
current_path = os.getcwd()
ParentPath=os.path.abspath(os.path.join(current_path,os.pardir))
Path2openSim=ParentPath+'/Simulations/'

#Preprocessing Parameters
fs=1000
high_freq=[50,140]
high_freqcwt=[50,200]
low_freq=[2,12]
winsize=1024
steps=20
# read signal
x=pd.read_csv(Path2openSim+'x4.csv',header=None).to_numpy().reshape(6001,)[:2048]
notch=notchFIR_Generate(fs, 60)
x=FIR_Apply(x, fs, notch)
t=pd.read_csv(Path2openSim+'t.csv',header=None).to_numpy().reshape(6001,)[:2048]
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
    bandpass_Amp=eegfilt_Generate(fs, [Af1,Af2])
    AmpFreq=FIR_Apply(x, fs, bandpass_Amp)#[0]
    # AmpFreq = filtfilt(b, a, AmpFreq)
    AmpFreqTransformed[ii, :] = np.abs(signal.hilbert(AmpFreq))  # Getting the amplitude envelope

for jj, Pf1 in enumerate(PhaseFreqVector):
    Pf2 = Pf1 + PhaseFreq_BandWidth
    PhaseFreq = x
    # Perform filtering
    bandpass_Phase=eegfilt_Generate(fs,[Pf1,Pf2])
    PhaseFreq=FIR_Apply(x, fs, bandpass_Phase)#[0]
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
bandpass_high=eegfilt_Generate(fs,high_freq)
x_high=FIR_Apply(x, fs, bandpass_high)
analytic_signal_high=signal.hilbert(x_high)
envelope = np.abs(analytic_signal_high)
f,Pxx_high= signal.welch(envelope,fs=fs,nperseg=winsize,noverlap=winsize/2)
fp=f[np.argmax(Pxx_high)]
bandpass_low=eegfilt_Generate(fs,[fp-2,fp+2])
x_low=FIR_Apply(x, fs, bandpass_low)
analytic_signal_low=signal.hilbert(x_low)
phase_low=np.angle(analytic_signal_low)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.polar(phase_low,envelope, 'g.',alpha=.2)
complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
mean_amp=np.abs(np.mean(complex_vectors))
mean_phase=np.angle(np.mean(complex_vectors))
plt.polar(mean_phase,mean_amp, 'r.')
plt.show()

#%% tPAC

Windows=np.arange(0,int(len(x)-winsize),128)
tPAC=np.zeros((steps,len(Windows)))
tPAC_phase=np.zeros((steps,len(Windows)))
fP=[]

freqs,step=np.linspace(high_freq[0], high_freq[1], steps, retstep = True) 
width=max(step,max(low_freq))
fsig,PSDsignal=signal.welch(x,fs,nperseg=winsize)
# width=step
for i,freq in enumerate(freqs):
    bandpass_fA = eegfilt_Generate(fs, [freq-width/2,freq+width/2])
    x_high=FIR_Apply(x,fs,bandpass_fA)
    analytic_signal_high=signal.hilbert(x_high)
    envelope = np.abs(analytic_signal_high)
    for j,win in enumerate(Windows):
        # Pxx_high,f= mne.time_frequency.psd_array_multitaper(envelope, sfreq=fs, fmin=3, adaptive=True, n_jobs=-1, verbose=False)
        f,Pxx_high= signal.welch(envelope[win:win+winsize],fs=fs,nperseg=len(envelope),noverlap=0)
        f,Pxx_noFilter= signal.welch(x[win:win+winsize],fs=fs,nperseg=len(envelope),noverlap=0)
        pks=signal.find_peaks(Pxx_noFilter[np.argmax(Pxx_high)-2:np.argmax(Pxx_high)+3])[0]
        # print(i,j,pks)
        if pks.any():
            fp=f[np.argmax(Pxx_high)]
            fP.append(fp)
            bandpass_fP = eegfilt_Generate(fs, relu2([fp-PhaseFreq_BandWidth/2,fp+PhaseFreq_BandWidth/2]))
            x_low=FIR_Apply(x[win:win+winsize],fs,bandpass_fP)
            analytic_signal_low=signal.hilbert(x_low)
            phase_low=np.angle(analytic_signal_low)
            complex_vectors=[envelope[win:win+winsize][i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope[win:win+winsize])]
            mean_amp=np.abs(np.mean(complex_vectors))
            mean_phase=np.angle(np.mean(complex_vectors))
            tPAC[i,j]=mean_amp
            tPAC_phase[i,j]=mean_phase
        else:
            tPAC[i,j]=0
            tPAC_phase[i,j]=0
        
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

#%% CWT signal

sc=freq2scale('morl', fs, high_freqcwt[0], high_freqcwt[1])
widths=[]
v=np.arange(sc[0],sc[1],1)
M=16
for J in v:   #generate the scales
        a1=[]
        for m in np.arange(1,M+1): 
            a1.append(2**(J+(m/M)))
        
        widths.append(a1)
widths=np.array(widths)
widths=widths.reshape(widths.shape[1]*widths.shape[0],)
# scales=np.squeeze(pd.read_csv('scales.csv',header=None).to_numpy())
cmplxdata=signal.hilbert(x)
data=copy(x)
cwtmatr, freqs = pw.cwt(cmplxdata, widths, 'morl', sampling_period=1/fs)
print(freqs[-1],freqs[0])
idx=np.concatenate(( np.argwhere(freqs >=high_freq[1]),np.argwhere(freqs<=high_freq[0])))
freqs=np.delete(freqs, idx)
cwtmatr=np.delete(cwtmatr, idx,axis=0)
cfs= abs(cwtmatr)
#%% Wavelet minus (1/f)

muC = np.mean(cfs, axis=1)
sigmaC = np.std(cfs, axis=1)
cfs_h = cfs - np.expand_dims(muC, axis=1)
cfs_h = cfs_h / np.expand_dims(sigmaC, axis=1)

#%% Select the fp with the highest frequency
f,c=np.unique(fP,return_counts=True)
c_sort=np.flip(np.sort(c))
#%%
epoch = [-.75, .75] 
# for cc in c_sort[:2]:
cc=c_sort[0]
Fp=f[c==cc][1]
bandpass_fP = eegfilt_Generate(fs, relu2([Fp-PhaseFreq_BandWidth/2,Fp+PhaseFreq_BandWidth/2]))
fp_signal=FIR_Apply(x,fs,bandpass_fP)
locs_fp, _ = signal.find_peaks(-fp_signal)
pks_fp=fp_signal[locs_fp]
plt.figure()
plt.plot(fp_signal)
plt.plot(locs_fp, pks_fp, "x")
# pksDf_fp=pd.DataFrame({'pks_fp':pks_fp,'locs_fp':locs_fp})
# pksDf_fp.sort_values('pks_fp',inplace=True, ascending=False)
tevent = t[locs_fp]

epoch_fp = 1 * np.array([-1/Fp, 1/Fp])  #Window to averge around troughs

nEpochs = 0
epochSignal_fp = None

data = copy(x)
for k in range(len(locs_fp)):
    epochTime_fp = tevent[k] + epoch_fp  # (tevent[k] + 0*np.random.rand(1)) + epoch

    if epochTime_fp[0] < t[0] or epochTime_fp[1] > t[-1]:  # epoch outside time range
        continue
    else:
        nEpochs += 1
        ID = (locs_fp[k] + epoch_fp * fs).astype(int)  # I[k] + epoch_fp * fs

        if nEpochs == 1:
            epochSignal_fp = data[ID[0]:ID[1]]
            wltSignal_fp = cfs_h[:, ID[0]:ID[1]]  # TFD
        else:
            epochSignal_fp += data[ID[0]:ID[1]]
            wltSignal_fp += cfs_h[:, ID[0]:ID[1]]  # TFD
result = f"{nEpochs} fP-cycles registered"
print(result)    

timeEpoch_fp = np.linspace(epoch[0], epoch[1], len(epochSignal_fp))
epochSignal_fp = epochSignal_fp / nEpochs  # Original signal around troughs
wltSignal_fp = wltSignal_fp / nEpochs  # Filtered signal in fA range around troughs
plt.figure()
fepochAVG = plt.plot(timeEpoch_fp, epochSignal_fp, linewidth=1, color='k')
plt.title('Avg signal around troughs')
plt.xlabel('Time (S)')
plt.xlim([min(timeEpoch_fp), max(timeEpoch_fp)])
plt.axhline(0)
# plt.gca().set_yticks([])  # Remove y-axis labels and ticks
plt.show()

epochSignalClean = signal.medfilt(epochSignal_fp,kernel_size=9)
# for _ in range(2): # to get a N order of median filter, it has to be applied N times 
    # epochSignalClean = medfilt(epochSignalClean,kernel_size=9)
epochSignalClean -= np.mean(epochSignalClean)
plt.plot(timeEpoch_fp, epochSignalClean, linewidth=1, color='b')
fP = meanfreq(epochSignalClean, fs)

xCorr=[]
lagCorr=[]


for row in range(wltSignal_fp.shape[0]):
    xCorr.append(correlate(wltSignal_fp[row,:], epochSignalClean)) # cant limit the lag range as in xcorr in matlab "xcorr(x,y,round(0.8* fs/fP))"
    lagCorr.append(correlation_lags(wltSignal_fp[row,:].size, epochSignalClean.size))
   
xCorr=np.array(xCorr)
lagCorr=np.array(lagCorr)
   


MxCorr = np.max(-xCorr, axis=1)  # minus sign (-xCorr) because the lag is measured b/w trough of fP and max of fA amplitude
I_xCorr = np.argmax(MxCorr)
J_xCorr = np.argmax(MxCorr)

# Plot correlation between TF map and low-frequency component

fig = plt.figure()
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0:2])
ax2 = fig.add_subplot(gs[2])
# hp = ax1.pcolor(timeEpoch_fp, freqs, wltSignal_fp,cmap='jet')
hp = ax1.contourf(timeEpoch_fp, freqs, wltSignal_fp, 100,extend='both',cmap='jet')
# hp.set_facecolor('interp')
plt.suptitle('[smoothed: ] z-scored induced signal')
ax1.set_xlabel('Time (S)')
ax1.set_ylabel('Frequency (Hz)')
# plt.colorbar(hp,ax=ax1)
ax2.plot(timeEpoch_fp, epochSignalClean, linewidth=1.25, color='k')
ax2.set_xlabel('Time (S)')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(-0.75, 0.75)

fAfPlags = fP*lagCorr/fs; #find modes in fA/fP cross-correlation (to see if phase and amplitude envelope are related):
xCorrTrace = np.sum(abs(xCorr),axis=1) # Sum the corr at each frequency 

# xCorrTrace(ifreqNoise) = NaN; 
# Detect peaks 
xCorrTrace = np.flip(xCorrTrace)
frq = np.flip(freqs)
locs, dic = signal.find_peaks(xCorrTrace,prominence=1) 
w = signal.peak_widths(xCorrTrace, locs, rel_height=0.5) # widths
p=dic['prominences']
pks=xCorrTrace[locs]
pksDf=pd.DataFrame({'pks':pks,'locs':locs, 'prom': p, 'width': w[0]})
pksDf.sort_values('prom',inplace=True, ascending=False)
locs=pksDf['locs'].to_numpy()
pks=pksDf['pks'].to_numpy()
p=pksDf['prom'].to_numpy()
w=pksDf['width'].to_numpy()
#%%

plt.figure()
plt.plot(frq,xCorrTrace)
plt.plot(frq[locs],xCorrTrace[locs],'x')

p = p/max(p)
ipeaks = np.argwhere(p > 0.1)
fAWidth = w[ipeaks] 
fAFreq = locs[ipeaks];
fAA = frq[fAFreq[0]][0] # Frequency for amplitude
fAAWidth = fAWidth[0][0]


fig = plt.figure()
# hp = ax1.pcolor(timeEpoch_fp, freqs, wltSignal_fp,cmap='jet')
hp = plt.contourf(lagCorr[0,:]/fs, freqs, xCorr, 100,extend='both',cmap='jet')
# hp.set_facecolor('interp')
plt.suptitle('slow cycle ('+str(round(fP,1))+') Hz vs. envelope of fast components (' +str(round(freqs[J_xCorr],1))+') Hz')
# plt.xlim([-.075,.075])
plt.xlabel('Cross-correlation delay (ms)')
plt.ylabel('Frequency for amplitude (Hz)')
plt.colorbar(hp)

plt.figure()
plt.plot(frq,xCorrTrace)
# plt.plot(frq[locs],xCorrTrace[locs],'v')
plt.title('fP=' +str(round(fP,1))+' Hz / fA=' +str(round(fAA,1))+ ' Hz')
plt.xlabel('Frequency (Hz)')
for l in range(len(locs)): 
    plt.annotate(str(l), xy=(frq[locs[l]], xCorrTrace[locs[l]]),
        arrowprops=dict(facecolor='black'))


    
plt.show()