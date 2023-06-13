% Load different types of dataset
%====================================================================================================
%% Dataset 1: simulated data
% sFiles = {...
%     'tPAC_FoooF/maximum_values/matrix_pac__221117_1751.mat'};

% load(file_fullpath(sFiles{1}))

%% Dataset 2: LFP data
load('LFP_HG_HFO.mat');

%% Dataset 3: MEG resting data
% sFiles = {...
%     'link|PAC_MEG_example/@rawsubj002_spontaneous_20111102_01_AUX/results_MN_MEG_KERNEL_220922_1037.mat|PAC_MEG_example/@rawsubj002_spontaneous_20111102_01_AUX/data_0raw_subj002_spontaneous_20111102_01_AUX.mat'};

%Source of interest
% iSrc = 28; % 1, 3, 5, 647 work great, ramping/chirping features 1002, 1003; 
% Load source results
% sResults = in_bst_results(sFiles{1}, 0);
% Load sensor data
% sData = in_bst(sResults.DataFile, [5,295]); %[5,295]; %in_bst(sResults.DataFile, [timeStart, timeStop]); %time in seconds

% Extract source time series
% data = sResults.ImagingKernel(iSrc,:) * sData.F(sResults.GoodChannel,1:end);
% t = sData.Time; % For MEG data, imported from Brainstorm

%% Paramteres and data
% Band of interest for frequency for amplitude 
i=1; % for detection of fP candidate
fA = [15, 250] 
epoch = [-.75, .75]; % time limits of epoch of interest around each peak
decomposition = 'trend';
diagm = 'yes';
nbin = 18;
% Dataset: time and value
data = lfpHG(1:10000);
% figure
% plot(timeEpoch,epochSignal,timeEpoch,data)
data_length = length(data);
srate = 1000;
dt = 1/srate;
t = (1:data_length)*dt
% t = EEG.times;
% Sampling rate
fs = 1/(t(3)-t(2)); % For simulated data and LFP
%====================== Data length comparison (zero-padding)===========
% margin = max(epoch);
% nMargin = fix(margin*fs);
% data = [zeros(size(data,1),nMargin), data, zeros(size(data,1),nMargin)];
% tstart= linspace(t(1) + min(epoch), t(1)- 1/fs , nMargin);
% tstop = linspace(t(end) + 1/fs, t(end)+max(epoch), nMargin);
% t = [tstart, t, tstop];

%=======================================================================
% Plot signal, fASignal, fPSignal
% figure;
% subplot(4,1,[1 2])
% plot(t(1:1000),data(1:1000), 'k', 'LineWidth', 1)
% set(gca,'XColor', 'none','YColor','none');
% 
% subplot(4,1,[3])
% plot(t(1:1000),fASignal(1:1000), 'k', 'LineWidth',1)
% hold on
% plot(t(1:1000),amp(1:1000), 'k.', 'LineWidth',1.25)
% set(gca,'XColor', 'none','YColor','none');
% 
% subplot(4,1,[4])
% plot(t(1:1000),fPSignal(1:1000), 'k', 'LineWidth',1)
% set(gca,'YColor','none');

% Wavelet decomposition of signal time series
fb = cwtfilterbank('SignalLength',numel(data),'SamplingFrequency',fs,...
     'FrequencyLimits',fA, 'TimeBandwidth',5); % TimeBandwidth controls the frequency resolution

[cfs,fc_h] = cwt(data,'FilterBank',fb); 
sp = rms(abs(cfs)); % Envelop of signal
tp = t; 
% figure;periodogram(LFP, [], [], fs);


% Detect peaks in spectrogram
[pks_p, locs_p, ~, ~] = findpeaks(sp, 'SortStr','descend', 'MinPeakDistance', .9/(fA(1)*(tp(2)-tp(1))) );
tevent = tp(locs_p);

meanCycle = mean(diff(locs_p))
meanCycle = mean(diff(locs_p))
% continuous wavelet transform of entire time series, before chopping it
% off in epochs
cfs = abs(cfs); 
%%
nEpochs = 0;
for k = 1:length(locs_p)
    %random control event
    ctrlID = floor(1+length(t)*rand(1));
    ctrlID = round([ctrlID(1), ctrlID(1)+diff(epoch)*fs]);
    while ctrlID(end)>length(t) % % make sure control epoch is within data time range
        ctrlID = floor(1+length(t)*rand(1));
        ctrlID = round([ctrlID(1), ctrlID(1)+diff(epoch)*fs]);
    end
    
    epochTime = tevent(k)+epoch;
    ID = locs_p(k) + epoch * fs;
    % If peak time +/- window is outside of the total timning (t)
    if epochTime(1)<t(1) || epochTime(2)>t(end) % epoch outside time range
        % do nothing

    else
        % Finding the similar peak in the orignial signal without filtering
        nEpochs = nEpochs+1;
%         ID = dsearchn(t',epochTime'); % find sample IDs for epochTime in original data
             
        if nEpochs==1
            epochSignal = data(ID(1):ID(2));
            %spectrumSignal =    sp(IDspec(1):IDspec(2));
            epochControl = data(ctrlID(1):ctrlID(2));
          
            wltSignal = cfs(:,ID(1):ID(2)); % TFD 
            wltControl = cfs(:,ctrlID(1):ctrlID(2)); 
            
        else
            epochSignal = epochSignal+ data(ID(1):ID(2));
%             spectrumSignal = spectrumSignal + sp(IDspec(1):IDspec(2));
            
            epochControl= epochControl + data(ctrlID(1):ctrlID(2));
            
            wltSignal = wltSignal + cfs(:,ID(1):ID(2)); % TFD 
            wltControl = wltControl + cfs(:,ctrlID(1):ctrlID(2));       
        end
    end
end
sprintf('%d fA-bursts registered', nEpochs)

% Average around peaks based on the number of fA registered 
epochSignal = epochSignal/nEpochs;
timeEpoch = linspace(epoch(1), epoch(2), length(epochSignal));

epochControl = epochControl/nEpochs;
wltSignal = (wltSignal)/nEpochs;
wltControl = (wltControl)/nEpochs;
%spectrumSignal = spectrumSignal/nEpochs;
%% Plotting
if strcmp(diagm, 'yes')
    fig = figure
    fepochAVG = plot(timeEpoch,epochSignal , 'LineWidth', 1, 'Color' , 'k');
    % fepochAVG = plot(Time(1:1000),Value(1:1000) , 'LineWidth', 1, 'Color' , 'k'); 
    title('Avg signal around bursts'); xlabel('Time (S)')
    xlim([-0.75 0.75])
    yline(0)
    % Specify common font to all subplots
    set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 2);
    set(gca,'YColor','none'); %Remove numbers and axis
    % Give common xlabel, ylabel and title to your figure
    han=axes(fig,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    % fctrlAVG = plot(timeEpoch,epochControl, 'LineWidth', 1, 'Color' , 'm');hold on
    % fepochAVG_filtered = plot(timeEpoch,wltControl , 'LineWidth', 1, 'Color' , 'r');
    % fctrlAVG_filtered = plot(timeEpoch, wltSignal, 'LineWidth', 1, 'Color' , 'y');
    % legend('EpochSignal-original', 'EpochControl-original', 'EpochControl-Filtered', 'EpochSignal-Filtered')
end

% mra = ewt(epochSignal,'MaxNumPeaks',10, 'LogSpectrum', 0);
switch decomposition
    case 'vmd'
        fP = calc_fP_vmd(epochSignal, timeEpoch, fs, diagm);
    case 'trend'
        fP = calc_fP_trend(epochSignal, fs, timeEpoch, diagm);
end
  

bpFP_filter = designfilt('bandpassfir', ...
                'SampleRate', fs,...
                'PassbandRipple', 1,...
                'DesignMethod','equiripple',...
    'PassbandFrequency1', fP-.25, 'StopbandFrequency2', fP+1,'StopbandAttenuation1', 20,...
    'StopbandAttenuation2', 20, 'StopbandFrequency1', fP-1, 'PassbandFrequency2', fP+.25);
% fvtool(bpFP_filter)


margin = max(epoch);
nMargin = fix(margin*fs);
data_pad = [zeros(size(data,1),nMargin), data-mean(data), zeros(size(data,1),nMargin)];
% tstart= linspace(t(1) + min(epoch), t(1)- 1/fs , nMargin);
% tstop = linspace(t(end) + 1/fs, t(end)+max(epoch), nMargin);
% t = [tstart, t, tstop];

fPSignal = filtfilt(bpFP_filter,data_pad);
fPSignal = fPSignal(:,nMargin+1:end-nMargin); % Removing Margin

% Detect troughs
[pks_fp,locs_fp,~,~] = findpeaks(-fPSignal, 'SortStr','descend','MinPeakDistance', fs/(fP+.25)); % -fPSignal: detect troughs, fP+.25 is faster cutoff of fPSignal bandpass

% Time of troughs in the original signal 
% I = 1:round(1*numel(locs_fp));
% tevent = t(locs_fp(I)); 
tevent = t(locs_fp); 
nEpochs = 0;

%clear epochSignal epochControl wltSignal wltControl 

fb = cwtfilterbank('SignalLength',numel(data),'SamplingFrequency',fs,...
    'FrequencyLimits',fA, 'TimeBandwidth',60); % Higher frequency resolution than when detecting bursts above.
[cfs_h,fc] = cwt(data,'FilterBank',fb);
cfs_h = abs(cfs_h);

muC = mean(cfs_h,2);
sigmaC = std(cfs_h,[],2); 
cfs_h = bsxfun(@minus, cfs_h, muC);
cfs_h = bsxfun(@rdivide, cfs_h, sigmaC);

%Epoch length adapted to fP cycle length (5 fP cycles on both sides of t=0)
epoch_fp = 1*[-1/fP, 1/fP]; %TODO= 2 used instead of 5

for k = 1:length(locs_fp)%length(I)
    
    epochTime_fp = tevent(k)+epoch_fp;%(tevent(k)+0*rand(1))+epoch;

    if epochTime_fp(1)<t(1) || epochTime_fp(2)>t(end) % epoch outside time range

        % do nothing
    else
        nEpochs = nEpochs+1;
%         ID = dsearchn(t',epochTime_fp'); % find sample IDs for epochTime in original data   
          ID = locs_fp(k) + epoch_fp * fs; %I(k) + epoch_fp * fs

        if nEpochs == 1
            epochSignal_fp = data(ID(1):ID(2));     
            wltSignal_fp = cfs_h(:,ID(1):ID(2)); % TFD 
        else
            epochSignal_fp = epochSignal_fp + data(ID(1):ID(2));
            wltSignal_fp = wltSignal_fp + cfs_h(:,ID(1):ID(2)); % TFD 
        end
    end
end

wltControl_perm = [];
epochControl_perm = [];
nEpochsCtrl = 0;
% Control event (deprecated)
for num_perm_ctrl = 10
for k = 1:1*length(locs_fp)
    
    %random control event
    ctrlID = floor(1+length(t)*rand(1));
    ctrlID = [ctrlID(1), ctrlID(1)+length(epochSignal_fp)-1];
    while ctrlID(end)>length(t) % % make sure control epoch is within data time range
        ctrlID = floor(1+length(t)*rand(1));
        ctrlID = [ctrlID(1), ctrlID(1)+length(epochSignal_fp)-1];
    end
    
    nEpochsCtrl = nEpochsCtrl+1;
    
    if nEpochsCtrl==1
        epochControl_fp = data(ctrlID(1):ctrlID(2));    
        wltControl_fp = cfs_h(:,ctrlID(1):ctrlID(2));
    else
        epochControl_fp = epochControl_fp + data(ctrlID(1):ctrlID(2));
        wltControl_fp = wltControl_fp + cfs_h(:,ctrlID(1):ctrlID(2));
    end

   
end
 epochControl_perm = [epochControl_perm ;epochControl_fp];
 wltControl_perm = [wltControl_perm ;  wltControl_fp];

end
epochControl_fP = mean(epochControl_perm);
wltControl_fP = mean(wltControl_perm);

sprintf('%d fP-cycles registered', nEpochs)

epochSignal_fp = epochSignal_fp/nEpochs; % Original signal around troughs
epochControl_fp = epochControl_fp/nEpochsCtrl;
wltSignal_fp = (wltSignal_fp)/nEpochs; % Filtered signal in fA range around troughs
wltControl_fp = (wltControl_fp)/(nEpochsCtrl);

timeEpoch_fp = linspace(epoch(1), epoch(2), length(epochSignal_fp));
timeEpoch_fp_ctrl = linspace(epoch(1), epoch(2), length(epochControl_fp));
% Plot original and ctrl signal around troughs
if strcmp(diagm, 'yes')
figure;
% yyaxis left
fepochAVG = plot(timeEpoch_fp,epochSignal_fp , 'LineWidth', 1, 'Color' , 'k'); %hold on
title('Avg signal around troughs'); xlabel('Time (S)')
% Specify common font to all subplots
set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 2);
xlim([min(timeEpoch_fp) max(timeEpoch_fp)])
yline(0);
set(gca,'YColor','none'); %Remove numbers and axis
% Give common xlabel, ylabel and title to your figure
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
% fctrlAVG = plot(timeEpoch_fp_ctrl, epochControl_fp, 'LineWidth', 1, 'Color' , 'm');
% legend('EpochSignal', 'ControlSignal')
end
% Filter the sudden drop
epochSignalClean = medfilt1(epochSignal_fp);
epochSignalClean  = epochSignalClean - mean(epochSignalClean); % mra2(:,end);

% figure; plot(timeEpoch_fp, epochSignalClean, 'LineWidth', 2, 'Color' , 'm');

% Plot PSD of original and control signal
% figure
% pspectrum(epochSignal_fp,fs,'FrequencyLimits', [1, 150], 'FrequencyResolution',3);
%, 'FrequencyResolution',3);% 'spectrogram','FrequencyLimits', [0, 150],'OverlapPercent',20,'Reassign',false)
% hold on
% pspectrum(epochControl_fp,fs,'FrequencyLimits', [1, 150]);
%, 'FrequencyResolution',3);% 'spectrogram','FrequencyLimits', [0, 150],'OverlapPercent',20,'Reassign',false)
% legend('signal','control')

fP = medfreq(epochSignalClean, fs); % frequency for phase (updated)


% Standardize amplitude per frequency bin
zwltSignal = wltSignal_fp;
% Control component
zwltControl = wltControl_fp;


clear xCorr lagCorr xCorrControl lagCorrControl

for k = 1:size(zwltSignal,1)
    [xCorr(:,k),lagCorr(:,k)] = xcorr(zwltSignal(k,:),epochSignalClean,...
        round(0.8* fs/fP));
    
    [xCorrControl(:,k),lagCorrControl(:,k)] = xcorr(zwltControl(k,:),epochControl_fp,...
        round(0.8* fs/fP)); 
end

[MxCorr,I_xCorr] = max(-xCorr, [], 1); % minus sign (-xCorr) because the lag is measured b/w trough of fP and max of fA amplitude
[~,J_xCorr] = max(MxCorr);
% Plot correaltion b/w, TF map and low frequncy component 
if strcmp(diagm, 'yes')
figure
%yyaxis left 
subplot(3,1,[1 2]);
hp = pcolor(timeEpoch_fp,fc,zwltSignal);
hp.EdgeColor = 'none'; hp.FaceColor = 'interp'; %set(gca,'YScale', 'log')
title('[smoothed: ] z-scored induced signal');
xlabel('Time (S)'); ylabel('Frequency (Hz)');
colormap('jet')
cb = colorbar;
% cb.Position = cb.Position + 1e-2; % or + [left, bottom, width, height]
% caxis([-.6*max(abs(zwltSignal(:))), .6*max(abs(zwltSignal(:)))])
set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 1, 'color', 'k');
%yyaxis right
subplot(3,1,[3]);
plot(timeEpoch_fp, epochSignalClean, 'LineWidth', 1.25, 'Color' , 'k'); %hold on
set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 1); ylabel('Amplitude');
xlim([-.75 .75])
xlabel('Time (S)');

figure;
subplot(3,1,[1 2]);
% yyaxis left 
hp = pcolor(timeEpoch_fp_ctrl,fc,zwltControl);
hp.EdgeColor = 'none'; hp.FaceColor = 'interp'; %set(gca,'YScale', 'log')
title('[smoothed: ] z-scored control signal');
xlabel('Time (S)'); ylabel('Frequency (Hz)');
colorbar, colormap('jet'), 
% caxis([-.6*max(abs(zwltControl(:))), .6*max(abs(zwltControl(:)))])
% yyaxis right
set(findobj(gcf,'type','axes'),'FontSize',14,'LineWidth', 1, 'color', 'k');
subplot(3,1,[3]);
plot(timeEpoch_fp_ctrl, epochControl_fp, 'LineWidth', 1, 'Color' , 'k');
set(findobj(gcf,'type','axes'),'FontSize',14); ylabel('Amplitude');
xlim([-.75 .75])
xlabel('Time (S)')
end

%% TODO: This calculate PAC strength based on correlation analyis. 
% to decide to use this or instead use the filtered signal 
% % Compute PAC
% phaseAngle = angle(hilbert(epochSignalClean)); % angles for TROUGHS=+/-PI, PEAKS=0
% phaseAngleControl = angle(hilbert(epochControl_fp)); % angles for TROUGHS=+/-PI, PEAKS=0

% clear signalPAC
% signalPAC = zwltSignal(J_xCorr,:) .* exp (1i * phaseAngle);
% controlPAC = zwltControl(J_xCorr,:).* exp (1i * phaseAngleControl);
% % %====================================Comment============================================
% fig = figure;
% polarplot(signalPAC,'.','color','blue') % All of dots showing the value of high-frequency amplitude at specific phase of slow oscillation
% hold on
% % polarplot(controlPAC,'+');
% PAC = mean(controlPAC, 'omitnan'); % Average of dots
% polarplot(PAC, 'o','MarkerSize', 12,'MarkerFaceColor', 'b'); 
% % anglePAC = linspace(-pi,pi);
% PAC_sources = abs(PAC); % Amplitude average of dots
% Preferred_phase_sources = angle(PAC); % Angle of average of dots
% Degree_theta = rad2deg(Preferred_phase_sources); % Convert angle from radians to degrees
% %========================================================================================
% phi = phaseAngle;           % Compute phase of low-freq signal 
% amp = zwltSignal(J_xCorr,:);  % Compute amplitude of high-freq signal 
% p_bins = -pi: 0.1: pi;
% a_mean = zeros(length(p_bins) -1,1); % vector to hold avg amp env results p_mean =
% p_mean = zeros ( length ( p_bins ) -1,1); % vector to hold center of phase bins
% for k = 1: length(p_bins) -1
% pL = p_bins(k); % phase lower limit for this bin
% pR = p_bins(k+1); % phase upper limit for this bin
% indices = find(phi >= pL & phi < pR); % find phase values in this %range
% a_mean(k) = mean(amp(indices)); % compute mean amplitude at these % phases
% p_mean(k) = mean([pL, pR]); % label the phase bin with the center %phase
% end
% h = max(a_mean) - min(a_mean); % diff bw max and min modulation
% % plot the mean envelope vs phase
% figure; plot(p_mean , a_mean , 'k');
% axis tight
% xlabel('Low freq phase '); ylabel('High freq amplitude'); 
% title([ 'Metric h=' num2str(h)]);
% 
% signalPAC_bins = a_mean .* exp (1i * p_mean); % result based on bins
% figure;
% polarplot(signalPAC_bins'.','color','blue') % All of dots showing the value of high-frequency amplitude at specific phase of slow oscillation
% 
% [x, y] = pol2cart(p_mean, a_mean); % Transfer polar to cart
% [xc,yc,Re,a] = circfit(x,y); % Fit a circle
% xe = Re*cos(p_mean)+xc; ye = Re*sin(p_mean)+yc;
% [xc_polar, yc_polar] = cart2pol(xc, yc);
% [thetaCH, rhoCH] = cart2pol(xe, ye);
% polarplot(thetaCH, rhoCH, 'ro-');
% hold on
% center = yc_polar .* exp (1i * xc_polar);
% polarplot(center,'o','MarkerSize', 12,'MarkerFaceColor', 'r')
% % plot the labels
% title(sprintf('Theta=%g ; PAC=%g, Diameter=%g',rad2deg(angle(center)), 2*yc_polar, 2*Re)) 

%--------------------------------Applying control signal---------------------------------------
% Assign non-significant zscore map value to ZERO.
% xCorrDiff = xCorr;
% xCorrDiff(abs(xCorrControl)>abs(xCorr))=NaN; 
% Inan = isnan(xCorrDiff');
% Inan = sum(Inan,2);
% ifreqNoise = find(Inan>size(xCorrDiff,2)/2);  
% %% TODO: for now, just keep all of the values without removing noise
% zwltSignal(ifreqNoise,:) = 0;
% % Zwlt signal after removing control signal 
% figure
% subplot(221)
% yyaxis left 
% hp = pcolor(timeEpoch_fp,fc,zwltSignal);
% hp.EdgeColor = 'none'; hp.FaceColor = 'interp'; set(gca,'YScale', 'log')
% % title('[smoothed: ] z-scored induced signal');
% title('Averaged relative spectrogram and signal')
% xlabel('Time (S)'); ylabel('Frequency (Hz)');
% colorbar, colormap('jet'), 
% caxis([-.6*max(abs(zwltSignal(:))), .6*max(abs(zwltSignal(:)))])
% yyaxis right
% ylim([5 250])
% plot(timeEpoch_fp, epochSignal_fp, 'LineWidth', 1.25, 'Color' , 'k'); hold on
% 
% subplot(222)
% hp = pcolor(lagCorr(:,1)/fs, fc, xCorr');
% hp.EdgeColor = 'none'; hp.FaceColor = 'interp';set(gca,'YScale', 'log')
% colorbar, colormap('jet')
% ylabel('frequency for amplitude (Hz)')
% xlabel('cross-correlation delay (ms)')
% title(sprintf('slow cycle (%3.1f Hz) vs. envelope of fast components/ fA=%3.1f Hz', fP, fc(J_xCorr)))


% subplot(223)
% hp = pcolor(lagCorrControl(:,1)/fs, fc, xCorrControl');
% hp.EdgeColor = 'none'; hp.FaceColor = 'interp'; set(gca,'YScale', 'log')
% colorbar
% ylabel('frequency for amplitude (Hz)')
% xlabel('cross-correlation delay (ms)')
% title(sprintf('slow cycle (%3.1f Hz) vs. envelope of fast components (control)', fP))


% subplot(224)
% hp = pcolor(lagCorr(:,1)/fs, fc, xCorrDiff');
% hp.EdgeColor = 'none'; hp.FaceColor = 'interp';set(gca,'YScale', 'log')
% colorbar, colormap('jet')
% ylabel('frequency for amplitude (Hz)')
% xlabel('cross-correlation delay (ms)')
% title(sprintf('slow cycle (%3.1f Hz) vs. envelope of fast components (xCorrDiff)/ fA=%3.1f Hz', fP, fc(J_xCorr)))
%sprintf('fP=%3.1f Hz / fA=%3.1f Hz / fP-fA time shift = %3.2f x fP cycle', fP, fc(J_xCorr) , fP*lagCorr(I(J_xCorr),J_xCorr)/fs)
%=====================================================================================================
fAfPlags = fP*lagCorr/fs;
%find modes in fA/fP cross-correlation (to see if phase and amplitude envelope are related):
xCorrTrace = sum(abs(xCorr),1); % Sum the corr at each frequency 

% xCorrTrace(ifreqNoise) = NaN; 
% Detect peaks 
xCorrTrace = fliplr(xCorrTrace);
freqs = fliplr(fc');
[pks,locs,w,p] = findpeaks(xCorrTrace,freqs,'WidthReference','halfprom', 'SortStr','descend');

%[B,Ipeak] = sort(p,'descend'); % sort peaks by preminence (see findpeaks)
% hold on,
% text(locs+.02, pks, num2str((1:numel(pks))'))

%scatter(freqs(locs(Ipeak)), B(Ipeak));
%scatter(locs, pks);
% figure
% findpeaks(xCorrTrace,fliplr(fc(ifc)'),'WidthReference','halfprom', 'SortStr','descend','Annotate','extents')

% Pick the most prominent peaks
p = p/max(p);
ipeaks = find(p > 0.1);

fAWidth = w(ipeaks);
fAFreq = locs(ipeaks);

fAA = fAFreq(1); % Frequency for amplitude
fAAWidth = fAWidth(1);
% Plot corr plot b/w TF map and low frequncy 
if strcmp(diagm, 'yes')
figure
hp = pcolor(lagCorr(:,1)/fs, fc, xCorr');
hp.EdgeColor = 'none'; hp.FaceColor = 'interp';set(gca,'YScale', 'log')
colorbar, colormap('jet')
ylabel('Frequency for amplitude (Hz)')
xlabel('Cross-correlation delay (ms)')
title(sprintf('slow cycle (%3.1f Hz) vs. envelope of fast components', fP, fc(J_xCorr)))
set(findobj(gcf,'type','axes'),'FontSize',14,'LineWidth', 1);


figure
[psor,lsor] = findpeaks(xCorrTrace,freqs,'WidthReference','halfprom', 'SortStr','descend');
findpeaks(xCorrTrace,freqs,'WidthReference','halfprom', 'SortStr','descend');
% title(sprintf('fP=%3.1f Hz / fA=%3.1f Hz / fP-fA time shift = %3.2f x fP cycle', fP, fAA , fP*lagCorr(I(J_xCorr),J_xCorr)/fs))
title(sprintf('fP=%3.1f Hz / fA=%3.1f Hz', fP, fAA))
text(lsor+.02,psor,num2str((1:numel(psor))'))
xlabel('Frequency (Hz)')
grid off
set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 1);
end

%% TODO: Refine the filter design
bpFA_filter = designfilt('bandpassfir', ...
              'SampleRate', fs,...
              'PassbandRipple', 1,...
                'DesignMethod','equiripple',...
    'StopbandFrequency1', max([eps, fAA-fAAWidth]),'PassbandFrequency1', fAA-fAAWidth/2, 'StopbandAttenuation1', 20,...
    'StopbandAttenuation2', 20, 'PassbandFrequency2', fAA+fAAWidth/2, 'StopbandFrequency2', min([fAA+fAAWidth, fs/2.001]));

data_pad = [zeros(size(data,1),nMargin), data, zeros(size(data,1),nMargin)];
fASignal = filtfilt(bpFA_filter,data_pad);
fASignal = fASignal(:,nMargin+1:end-nMargin); % Removing Margin


%% Applying filter on control signal (epoched around troughs) 
% bpFP_filter = designfilt('bandpassfir', ...
%                 'SampleRate', fs,...
%                 'PassbandRipple', 1,...
%                 'DesignMethod','equiripple',...
%     'PassbandFrequency1', fP-.25, 'StopbandFrequency2', fP+1,'StopbandAttenuation1', 20,...
%     'StopbandAttenuation2', 20, 'StopbandFrequency1', fP-1, 'PassbandFrequency2', fP+.25);
% 
% fPSignal = filtfilt(bpFP_filter,epochControl_fp-mean(epochControl_fp)); % Zero-phase filtering
% 
% bpFA_filter = designfilt('bandpassfir', ...
%               'SampleRate', fs,...
%               'PassbandRipple', 1,...
%                 'DesignMethod','equiripple',...
%     'StopbandFrequency1', max([eps, fAA-fAAWidth]),'PassbandFrequency1', fAA-fAAWidth/2, 'StopbandAttenuation1', 20,...
%     'StopbandAttenuation2', 20, 'PassbandFrequency2', fAA+fAAWidth/2, 'StopbandFrequency2', min([fAA+fAAWidth, fs/2.001]));
% 
% fASignal = filtfilt(bpFA_filter,epochControl_fp);
%% TODO: fPSignal should be updated based on new fP (fP = medfreq(epochSignalClean, fs))
% fPSignal = fPSignal(:,nMargin+1:end-nMargin); % Removing Margin
% fASignal = fASignal(:,nMargin+1:end-nMargin); % Removing Margin

phi = angle(hilbert(fPSignal)); % Compute phase of low-freq signal 
amp = abs(hilbert(fASignal)); % Compute amplitude of high-freq signal

[APC_circle, Theta] = calc_circle(phi,amp, nbin, diagm) % Ozkurt, canolty, tort and Ellipse 


function fP = calc_fP_vmd(epochSignal, timeEpoch, fs, diagm)
% Inputs:
% epochSignal: a 1D array containing the signal epoch of interest.
% timeEpoch: a 1D array containing the time vector for the signal epoch.
% fs: a scalar value representing the sampling frequency of the epoch signal.
% diagm: a string argument indicating whether or not to display diagnostic figures.

mra = vmd(epochSignal);
mp = size(mra,2);
n = size(mra,1)/2;
mraRMS = rms(mra(round(n-n/2:n+n/2),2:end) ,1); % focus around centre of epoch, and remove the first componenet. Find the component with more power using rms
[~,iMRA] = sort(mraRMS,'descend');
fP = meanfreq(mra(:,iMRA(1)+1), fs); % Extract the frequency of the component 

if strcmp(diagm, 'yes')
figure;
subplot(mp+1,1,1)
plot(timeEpoch,epochSignal,'LineWidth',1,'color','k')
title('Averaged signal around bursts')
ylabel('Signal')
set(gca,'XColor', 'none','YColor','none');
xlim([-.3, .3])
axis tight
for k=1:mp
    subplot(mp+1,1,k+1)
    plot(timeEpoch,mra(:,k),'LineWidth',1,'color','k')
    ylabel(['IMF ',num2str(k)])
    title(sprintf('%3.1f Hz',meanfreq(mra(:,k),fs)))
%     set(gca,'XColor', 'none','YColor','none'); % turn off axis and
%     numbers
    axis tight
    xlim([-.3, .3])
end
xlabel('Time (s)')
end

%% TODO: while fP <= 1 % Filtering around fP
%     i = i+1;
%     try 
%         fP = meanfreq(mra(:,iMRA(i)+1), fs); 
%     catch 
%         sprintf('No fP candidate detected above 1 (min fP value ~%3.1f Hz)', meanfreq(mra(:,end),fs))
%         continue
%     end
% end
% if fP>=fA(1)   
%     sprintf('No fP candidate detected below the minimum of fA range (min fP value ~%3.1f Hz)', meanfreq(mra(:,end),fs))
% continue
% else
% while fP>=fA(1)
%     i = i+1;
%     try 
%         fP = meanfreq(mra(:,iMRA(i)+1), fs); 
%     catch 
%         sprintf('No fP candidate detected below fA range (min fP value ~%3.1f Hz)', meanfreq(mra(:,end),fs))
%         return
%     end
% end

end

function fP = calc_fP_trend(epochSignal, fs, timeEpoch, diagm)
% The function calc_fP_trend performs trend decomposition on the input epochSignal using the function 
% trenddecomp, which decomposes the signal into a long-term trend (LT), a short-term trend (ST), 
% and a remainder (R). It then calculates the frequency of the component with 
% the highest root mean square (RMS) amplitude in the short-term trend using the meanfreq function.

% Inputs:
% epochSignal: a 1D array containing the signal epoch of interest.
% timeEpoch: a 1D array containing the time vector for the signal epoch.
% fs: a scalar value representing the sampling frequency of the epoch signal.
% diagm: a string argument indicating whether or not to display diagnostic figures.

[LT,ST,R] = trenddecomp(epochSignal);
mp = size(ST, 2);
n = size(ST,1)/2;
mraRMS = rms(ST(round(n-n/2:n+n/2),1:end) ,1); % focus around centre of epoch, and remove the first componenet
[~,iMRA] = sort(mraRMS,'descend');
fP = meanfreq(ST(:,iMRA(1)), fs); % Extract the frequency of the component 


if strcmp(diagm, 'yes')
figure;
subplot(mp+1,1,1)
plot(timeEpoch,epochSignal,'LineWidth',1,'color','k')
title('Averaged signal around bursts')
ylabel('Signal')
set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 1);
set(gca,'XColor', 'none','YColor','none');
% xlim([-.3, .3])
axis tight
for k=1:mp
    subplot(mp+1,1,k+1)
    plot(timeEpoch,ST(:,k),'LineWidth',1,'color','k')
    ylabel(['Trend ',num2str(k)])
    title(sprintf('%3.1f Hz',meanfreq(ST(:,k),fs)))
    set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 1);
    set(gca,'XColor', 'none','YColor','none');
    axis tight
end
xlabel('Time (s)')

end
end 



% Circle
    function [MI, Theta] = calc_circle(Phase,Amp, nbin, diagm)
%     Inputs:
%     Phase: A vector containing the values of the PhaseFreq.
%     Amp: A vector containing the values of the AmpFreq.
%     nbin: The number of phase bins used to calculate PAC.
%     diagm: A string specifying whether to plot the results or not. If set to 'yes', the code generates a polar plot showing the circle fit and the values of high-frequency amplitude at specific phase of the slow oscillation.
%     
%     Outputs:
%     MI: The Modulation Index, which is a measure of the strength of PAC.
%     Theta: The preferred phase angle of the AmpFreq relative to the PhaseFreq.


        position=zeros(1,nbin); % this variable will get the beginning (not the center) of each phase bin (in rads)
        winsize = 2*pi/nbin;
        for j=1:nbin 
            position(j) = -pi+(j-1)*winsize; 
        end
        
        % Now we search for a Phase-Amp relation between these frequencies by
        % caclulating the mean amplitude of the AmpFreq in each phase bin of the
        % PhaseFreq
         
        % Computing the mean amplitude in each phase:
        
        nbin=length(position);
        winsize = 2*pi/nbin;
        
        MeanAmp=zeros(1,nbin); 
        MeanPhase = zeros(1,nbin);
        for j=1:nbin   
        I = find(Phase <  position(j)+winsize & Phase >=  position(j));
        MeanAmp(j)=mean(Amp(I)); 
        MeanPhase(j) = mean([position(j), position(j)+winsize]);
        end
         
        signalPAC_bins = MeanAmp .* exp (1i * MeanPhase); % result based on bins
        [x, y] = pol2cart(MeanPhase, MeanAmp); % Transfer polar to cart
        [xc,yc,Re,a] = circfit(x,y); % Fit a circle
        xe = Re*cos(MeanPhase) + xc; ye = Re*sin(MeanPhase) + yc;
        [xc_polar, yc_polar] = cart2pol(xc, yc);
        [thetaCH, rhoCH] = cart2pol(xe, ye);
        center = yc_polar .* exp (1i * xc_polar);
        MI = 2*yc_polar;
        Theta = rad2deg(angle(center))
        if strcmp(diagm, 'yes')
            figure;
            polarplot(signalPAC_bins,'.','color','blue') % All of dots showing the value of high-frequency amplitude at specific phase of slow oscillation
            hold on
            polarplot(thetaCH, rhoCH, 'ko-');
            polarplot(thetaCH, rhoCH, 'ko-');
            polarplot(center,'o','MarkerSize', 12,'MarkerFaceColor', 'k')
            % Specify common font to all subplots
            set(findobj(gcf,'type','axes'),'FontSize',14, 'LineWidth', 2);
            % plot the labels
            title(sprintf('Method: Circle, PAC=%g, Theta=%g',MI, rad2deg(angle(center))));
        end
    end












