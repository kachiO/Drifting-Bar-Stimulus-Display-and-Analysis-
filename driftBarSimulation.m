%below script gives an intuition for running and analyzing periodic
%(Fourier) mapping experiments
%Written by Kachi Odoemene Dec 2015

clear;
figure(9);clf
figure(10);clf
%% Vertical Bar
screen = zeros(600,800);
barThickness = 10;

screenW = size(screen,2);
screenH = size(screen,1);

numcycles = 1;
period = 20;
stimperiod = period*numcycles;
framerate =5;
nframes = (stimperiod*framerate); %number of frames

tHF = floor(linspace(1,screenW-barThickness,nframes));
tHR = fliplr(tHF);
tVF = floor(linspace(1,screenH-barThickness,nframes));
tVR = fliplr(tVF);
T = [tHF;tHR;tVF;tVR];
thisBar = [1 1 0 0];
k = numcycles; %number of cycles presented during the stimulus epoch.
N = stimperiod; %duration of stimulus frames in secs.
n = linspace(0,stimperiod,nframes); %discrete time steps/stamps
angularFreqofFrames = (2*pi*k*n)./N; %angular frequency. belongs in the exponential of the fourier transform equation. see thisFourierComponent below.

phaseMaps = zeros(screenH,screenW,4);
for p = 1:4
    
    t = T(p,:);
    
    for nf = 1:nframes
        thisScreen = zeros(600,800);
        pos = t(nf);
        
        if thisBar(p)
            thisScreen(:,pos:pos+barThickness) = 1; %vertical bar
        else
            thisScreen(pos:pos+barThickness,:) = 1; %horizontal bar
        end
        figure(9);
        subplot(2,2,p);
        imshow(thisScreen);shg
        
        if nf == 1
            fourierTransform  = zeros(size(thisScreen)); %start with zero. the fourier transform of each frame will be added together
        end
        
        thisFourierComponent = thisScreen .* exp(1i*angularFreqofFrames(nf)); % x[n]*exp((-j*2*pi*n*k)/N)
        fourierTransform = fourierTransform + thisFourierComponent; % The fourier transform at k cycles, X[k] = Cummulative sum, from n = 1 to N, of x[n]*exp((-j*2*pi*n*k)/N) take a cummulative sum of the fourier component of each frame (i.e. sample)
    end
    
    
    % f0 = 4096 - double(mean(stimFrames(:,:,1:2),3));
    % fourierTransform = fourierTransform - f0*sum(exp(1i*anglularFreqofFrames)); %subtract spectral (f0) leakage. From the little I understand this helps to reduce smearing of the frequency spectrum caused by computing the Fourier transform.
    fourierTransform = 2*fourierTransform ./nf; %normalize/scale the transform
    
    fourierPhase =angle(fourierTransform); %compute the mean phase (angle) for each peak and sum for all trials in this condition.
    figure(10);
    subplot(2,2,p)
    imagesc(fourierPhase);colormap(jet); axis off;axis image
    
    phaseMaps(:,:,p) = fourierPhase;
end
%%
figure(11);clf
azimuth = phaseMaps(:,:,1) - phaseMaps(:,:,2);
azimuth = 2*pi*((azimuth - min(azimuth(:))) ./(max(azimuth(:)) - min(azimuth(:))));
subplot(1,2,1); imagesc(azimuth);colormap(hsv);axis off; axis image

elevation = phaseMaps(:,:,3) - phaseMaps(:,:,4); 
subplot(1,2,2); imagesc(elevation);colormap(hsv);axis off;axis image
