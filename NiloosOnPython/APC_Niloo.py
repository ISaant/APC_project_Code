#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:09:22 2023

@author: isaac
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pywt as pw
from copy import copy
from scipy import signal
from scipy.signal import find_peaks, peak_widths,periodogram, medfilt,correlate, correlation_lags
from scipy.stats import zscore
from vmdpy import VMD as vmd
import cmath
#%%
def freq2scale(MotherWavelet,fs,fmin,fmax):
    cf=pw.central_frequency(MotherWavelet)
    Scale=[]
    freqs=[fmax,fmin]
    for fr in freqs:
        Scale.append(np.round(np.log((cf*fs)/fr))/np.log(2))
    return Scale

#%%
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
#%%
def calc_fP_vmd(epochSignal, timeEpoch, fs, diagm):
    # Inputs:
    # epochSignal: a 1D array containing the signal epoch of interest.
    # timeEpoch: a 1D array containing the time vector for the signal epoch.
    # fs: a scalar value representing the sampling frequency of the epoch signal.
    # diagm: a string argument indicating whether or not to display diagnostic figures.

    #. some sample parameters for VMD  
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
    K = 5              # 5 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7 
    mra,u_hat, omega = vmd(epochSignal,alpha, tau, K, DC, init, tol)
    mp = mra.shape[0]
    n = mra.shape[1] // 2
    mraRMS = np.sqrt(np.mean(np.square(mra[:,np.arange(int(n-n/2), int(n+n/2))]), axis=1))  # focus around centre of epoch, and remove the first component. Find the component with more power using rms
    iMRA = np.argsort(mraRMS)
    fP = meanfreq(mra[iMRA[-1],:], fs)  # Extract the frequency of the component
    
    if diagm:
        plt.figure()
        plt.subplot(mp+1, 1, 1)
        plt.plot(timeEpoch, epochSignal, linewidth=1, color='k')
        plt.title('Averaged signal around bursts')
        plt.ylabel('Signal')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        plt.xlim([-0.3, 0.3])
        plt.axis('tight')
        for k in range(mp):
            plt.subplot(mp+1, 1, k+2)
            plt.plot(timeEpoch, mra[k,:], linewidth=1, color='k')
            plt.ylabel('IMF ' + str(k))
            plt.title('{:.1f} Hz'.format(meanfreq(mra[k,:], fs)))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().xaxis.set_ticks_position('bottom')
            plt.gca().yaxis.set_ticks_position('left')
            plt.xlim([-0.3, 0.3])
            plt.axis('tight')
        plt.xlabel('Time (s)')
        plt.show()

    return fP

#%%
def FIRFilt(signal,fs,freqs):
    from scipy.signal import firls,filtfilt
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
        print("filt() - performing {}-point highpass filtering.".format(filtorder))
        f = [MINFREQ, locutoff * (1 - trans) / nyq, locutoff / nyq, 1]
        m = [0, 0, 1, 1]
    elif hicutoff > 0:
        print("filt() - performing {}-point lowpass filtering.".format(filtorder))
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
    pad=len(signal)
    signal=np.lib.pad(signal,(pad,pad),'reflect')
    filtered_x = filtfilt(filtwts, 1, signal)[pad:pad*2]
    return filtered_x, filtwts
#%% Read data
lfpHG=pd.read_csv('lfpHG.csv',header=None).to_numpy()
lfp=np.squeeze(lfpHG)[:10000]
# lfpHG=pd.read_csv('x4.csv',header=None).to_numpy()
# lfp=np.squeeze(lfpHG)
# plt.plot(lfpHG[:500])

#%% Paramteres and data

i=1
fA = [15,250]
epoch = [-.75, .75] # time limits of epoch of interest around each peak
nbin = 18
lfp_len = len(lfp);
fs=1000
t=np.arange(0,lfp_len/fs,1/fs)
diagm=True

#%% Wavelet decomposition of signal
sc=freq2scale('morl', fs, fA[0], fA[1])
widths=[]
v=np.arange(sc[0],sc[1],1)
M=10
for J in v:   #generate the scales
        a1=[]
        for m in np.arange(1,M+1): 
            a1.append(2**(J+(m/M)))
        
        widths.append(a1)
widths=np.array(widths)
widths=widths.reshape(widths.shape[1]*widths.shape[0],)
# scales=np.squeeze(pd.read_csv('scales.csv',header=None).to_numpy())
cmplxdata=signal.hilbert(lfp)
data=copy(lfp)
cwtmatr, freqs = pw.cwt(cmplxdata, widths, 'morl', sampling_period=1/fs)
idx=np.concatenate(( np.argwhere(freqs >=fA[1]),np.argwhere(freqs<=fA[0])))
freqs=np.delete(freqs, idx)
cwtmatr=np.delete(cwtmatr, idx,axis=0)
plt.imshow(abs(cwtmatr),aspect='auto')
rms=np.sqrt(np.mean(abs(cwtmatr),axis=0)**2)
plt.figure()
plt.plot(rms)

locs_p, _ = find_peaks(rms, distance=.9/(fA[0]*(1/fs)))
pks_p=rms[locs_p]
plt.plot(locs_p, pks_p, "x")
pksDf_p=pd.DataFrame({'pks_p':pks_p,'locs_p':locs_p})
pksDf_p.sort_values('pks_p',inplace=True, ascending=False)
tevent = t[pksDf_p['locs_p']]
meanCycle = np.mean(np.diff(pksDf_p['locs_p']))
locs_p=pksDf_p['locs_p'].to_numpy()
pks_p=pksDf_p['pks_p'].to_numpy()
cfs= abs(cwtmatr)

#%%

nEpochs = 0
for k in range(len(locs_p)):
    # random control event: ctrlID define a random interval 
    ctrlID = np.floor(1+len(t)*np.random.rand(1))[0]
    ctrlID = np.round([ctrlID, ctrlID+np.diff(epoch)[0]*fs])
    while ctrlID[-1] > len(t): # make sure control epoch is within data time range
        ctrlID = np.floor(1+len(t)*np.random.rand(1))[0]
        ctrlID = np.round([ctrlID, ctrlID+np.diff(epoch)[0]*fs]);
    epochTime = tevent[k] + epoch
    ID = locs_p[k] + np.dot(epoch, fs) # Real interval around each peak
    # If peak time +/- window is outside of the total timing (t)
    if epochTime[0] < t[0] or epochTime[1] > t[-1]: # epoch outside time range
        # do nothing
        pass
    else:
        # Finding the similar peak in the original signal without filtering
        nEpochs += 1

        if nEpochs == 1:
            epochSignal = data[int(ID[0]):int(ID[1])]
            epochControl = data[int(ctrlID[0]):int(ctrlID[1])]
            wltSignal = cfs[:, int(ID[0]):int(ID[1])]  # TFD
            wltControl = cfs[:, int(ctrlID[0]):int(ctrlID[1])]
        else:
            epochSignal = epochSignal + data[int(ID[0]):int(ID[1])]
            epochControl = epochControl + data[int(ctrlID[0]):int(ctrlID[1])]
            wltSignal = wltSignal + cfs[:, int(ID[0]):int(ID[1])]  # TFD
            wltControl = wltControl + cfs[:, int(ctrlID[0]):int(ctrlID[1])]

print(f"{nEpochs} fA-bursts registered")

# Average around peaks based on the number of fA registered
epochSignal = epochSignal / nEpochs
timeEpoch = np.linspace(epoch[0], epoch[1], len(epochSignal))

epochControl = epochControl / nEpochs
wltSignal = wltSignal / nEpochs
wltControl = wltControl / nEpochs

#%% Plotting

if diagm:
    fig = plt.figure()
    fepochAVG, = plt.plot(timeEpoch, epochSignal, linewidth=1, color='k')
    # fepochAVG, = plt.plot(Time[0:1000], Value[0:1000], linewidth=1, color='k')
    plt.title('Avg signal around bursts')
    plt.xlabel('Time (S)')
    plt.xlim([-0.75, 0.75])
    plt.axhline(0, color='k')
    # Specify common font to all subplots
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 2
    plt.gca().set_yticklabels([])  # Remove numbers and axis
    # Give common xlabel, ylabel, and title to your figure
    han = fig.add_subplot(111, visible=False)
    han.set_title('Avg signal around bursts')
    han.set_xlabel('Time (S)')
    han.set_ylabel('')
    # fctrlAVG, = plt.plot(timeEpoch, epochControl, linewidth=1, color='m')
    # fepochAVG_filtered, = plt.plot(timeEpoch, wltControl, linewidth=1, color='r')
    # fctrlAVG_filtered, = plt.plot(timeEpoch, wltSignal, linewidth=1, color='y')
    # plt.legend(['EpochSignal-original', 'EpochControl-original', 'EpochControl-Filtered', 'EpochSignal-Filtered'])
    plt.show()

#%% Selecting fP
fP = calc_fP_vmd(epochSignal, timeEpoch, fs, diagm)
#%% Filter signal around fP
fPSignal,_ = FIRFilt(data, fs, [fP-.25,fP+.25])
freq,Pxx=periodogram(fPSignal,fs=fs)
plt.figure()
plt.plot(freq,Pxx)

#%% Find troughs
locs_fp, _ = find_peaks(-fPSignal)
pks_fp=fPSignal[locs_fp]
plt.figure()
plt.plot(fPSignal)
plt.plot(locs_fp, pks_fp, "x")
# pksDf_fp=pd.DataFrame({'pks_fp':pks_fp,'locs_fp':locs_fp})
# pksDf_fp.sort_values('pks_fp',inplace=True, ascending=False)
tevent = t[locs_fp]
# meanCycle = np.mean(np.diff(pksDf_fp['locs_fp']))
# locs_fp=pksDf_fp['locs_fp'].to_numpy()
# pks_fp=pksDf_fp['pks_fp'].to_numpy()

#%%

#%% Wavelet minus (1/f)

muC = np.mean(cfs, axis=1)
sigmaC = np.std(cfs, axis=1)
cfs_h = cfs - np.expand_dims(muC, axis=1)
cfs_h = cfs_h / np.expand_dims(sigmaC, axis=1)

#%%

epoch_fp = 1 * np.array([-1/fP, 1/fP])  #Window to averge around troughs

nEpochs = 0
epochSignal_fp = None
wltSignal_fp = None

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

wltControl_perm = []
epochControl_perm = []
nEpochsCtrl = 0

# Control event (deprecated)
num_perm_ctrl = 10
for j in range(num_perm_ctrl):
    for k in range(len(locs_fp)):
        # random control event
        ctrlID = np.floor(1 + len(t) * np.random.rand(1)).astype(int)
        ctrlID = [int(ctrlID[0]), int(ctrlID[0] + len(epochSignal_fp) - 1)]
        while ctrlID[-1] > len(t):  # make sure control epoch is within data time range
            ctrlID = np.floor(1 + len(t) * np.random.rand(1)).astype(int)
            ctrlID = [int(ctrlID[0]), int(ctrlID[0] + len(epochSignal_fp) - 1)]

        nEpochsCtrl += 1

        if nEpochsCtrl == 1:
            epochControl_fp = data[ctrlID[0]:ctrlID[1]]
            wltControl_fp = cfs_h[:, ctrlID[0]:ctrlID[1]]
        else:
            epochControl_fp += data[ctrlID[0]:ctrlID[1]]
            wltControl_fp += cfs_h[:, ctrlID[0]:ctrlID[1]]

    epochControl_perm.append(epochControl_fp)
    wltControl_perm.append(wltControl_fp)

epochControl_fP = np.mean(epochControl_perm, axis=0)
wltControl_fP = np.mean(wltControl_perm, axis=0)

result = f"{nEpochs} fP-cycles registered"
print(result)

epochSignal_fp = epochSignal_fp / nEpochs  # Original signal around troughs
epochControl_fp = epochControl_fp / nEpochsCtrl
wltSignal_fp = wltSignal_fp / nEpochs  # Filtered signal in fA range around troughs
wltControl_fp = wltControl_fp / nEpochsCtrl

timeEpoch_fp = np.linspace(epoch[0], epoch[1], len(epochSignal_fp))
timeEpoch_fp_ctrl = np.linspace(epoch[0], epoch[1], len(epochControl_fp))
#%%
# Plot original and ctrl signal around troughs
if diagm :
    plt.figure()
    fepochAVG = plt.plot(timeEpoch_fp, epochSignal_fp, linewidth=1, color='k')
    plt.title('Avg signal around troughs')
    plt.xlabel('Time (S)')
    plt.xlim([min(timeEpoch_fp), max(timeEpoch_fp)])
    plt.axhline(0)
    # plt.gca().set_yticks([])  # Remove y-axis labels and ticks
    plt.show()

epochSignalClean = medfilt(epochSignal_fp,kernel_size=9)
# for _ in range(2): # to get a N order of median filter, it has to be applied N times 
    # epochSignalClean = medfilt(epochSignalClean,kernel_size=9)
epochSignalClean -= np.mean(epochSignalClean)
plt.plot(timeEpoch_fp, epochSignalClean, linewidth=1, color='b')
fP = meanfreq(epochSignalClean, fs)

#%%
xCorr=[]
lagCorr=[]
xCorrCtrl=[]
lagCorrCtrl=[]

for row in range(wltSignal_fp.shape[0]):
    xCorr.append(correlate(wltSignal_fp[row,:], epochSignalClean)) # cant limit the lag range as in xcorr in matlab "xcorr(x,y,round(0.8* fs/fP))"
    lagCorr.append(correlation_lags(wltSignal_fp[row,:].size, epochSignalClean.size))
    xCorrCtrl.append(correlate(wltControl_fp[row,:], epochSignalClean)) 
    lagCorrCtrl.append(correlation_lags(wltControl_fp[row,:].size, epochControl_fP.size))

xCorr=np.array(xCorr)
lagCorr=np.array(lagCorr)
xCorrCtrl=np.array(xCorrCtrl)
lagCorrCtrl=np.array(lagCorrCtrl)


MxCorr = np.max(-xCorr, axis=1)  # minus sign (-xCorr) because the lag is measured b/w trough of fP and max of fA amplitude
I_xCorr = np.argmax(MxCorr)
J_xCorr = np.argmax(MxCorr)

# Plot correlation between TF map and low-frequency component
if diagm:
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
    plt.colorbar(hp,ax=ax1)
    ax2.plot(timeEpoch_fp, epochSignalClean, linewidth=1.25, color='k')
    ax2.set_xlabel('Time (S)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(-0.75, 0.75)
    
    fig = plt.figure()
    gs = fig.add_gridspec(3,1)
    ax1 = fig.add_subplot(gs[0:2])
    ax2 = fig.add_subplot(gs[2])
    # hp = ax1.pcolor(timeEpoch_fp, freqs, wltSignal_fp,cmap='jet')
    hp = ax1.contourf(timeEpoch_fp_ctrl, freqs, wltControl_fp, 20,extend='both',cmap='jet')
    # hp.set_facecolor('interp')
    plt.suptitle('[smoothed: ] z-scored control signal')
    ax1.set_xlabel('Time (S)')
    ax1.set_ylabel('Frequency (Hz)')
    plt.colorbar(hp,ax=ax1)
    ax2.plot(timeEpoch_fp_ctrl, epochControl_fp, linewidth=1, color='k')
    ax2.set_xlabel('Time (S)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(-0.75, 0.75)
    
plt.show()

#%%

fAfPlags = fP*lagCorr/fs; #find modes in fA/fP cross-correlation (to see if phase and amplitude envelope are related):
xCorrTrace = np.sum(abs(xCorr),axis=1) # Sum the corr at each frequency 

# xCorrTrace(ifreqNoise) = NaN; 
# Detect peaks 
xCorrTrace = np.flip(xCorrTrace)
frq = np.flip(freqs)
locs, dic = find_peaks(xCorrTrace,prominence=1) 
w = peak_widths(xCorrTrace, locs, rel_height=0.5) # widths
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

if diagm:
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

#%%
data=copy(lfp) # I dont know why, but data was overwriten somewhere... probabably memory location
fASignal,_ = FIRFilt(data, fs, [fAA-fAAWidth/2,fAA+fAAWidth/2])
analytic_signal_high=signal.hilbert(fASignal)
envelope = np.abs(analytic_signal_high)
analytic_signal_high=signal.hilbert(fASignal)
envelope = np.abs(analytic_signal_high)
analytic_signal_low=signal.hilbert(fPSignal)
phase_low=np.angle(analytic_signal_low)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
plt.polar(phase_low,envelope, 'g.',alpha=.2)
complex_vectors=[envelope[i]*cmath.exp(complex(0,1)*phase_low[i]) for i,_ in enumerate(envelope)]
mean_amp=np.abs(np.mean(complex_vectors))
mean_phase=np.angle(np.mean(complex_vectors))
plt.polar(mean_phase,mean_amp, 'r.')
plt.show()
