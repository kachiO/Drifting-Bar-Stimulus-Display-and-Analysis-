function maps = makeFourierPhaseMaps(subject,day,exptnum, rescaleImg,rotateImage,gcampFlag)
% this code is based off of Ian Nauhaus's (Callaway lab) online analysis of the Fourier
% phase map.
% It works by computing a fourier component for each frame, adding them together and taking the mean. This seems reasonable.
% see OnlineAnalysis code in Callaway lab intrinsic imaging package

%subject    :   string representing the animal name e.g. 'dr35'. This parameter is necessary.
%day        :   string representing the day the experiment was done.e.g.'20-June-2014. This parameter is necessary.
%exptnum    :   string representing the experiment number e.g. '001'. This parameter is necessary.
%recaleImg  :   number indicating the factor by which to reduce the image X & Y pixel. e.g
%               if the data were collected at 600x500 pixels (width x height).
%               If the rescaleImg parameter is 2, the resulting data will prodce an image that is 300x250.
%               Why you might want to do this is to save on computational memory/power. Having more pixels takes longer to compute the fourier transform.
%rotateImg  :   angle to rotate image
%gcampFlag  :   binary flag to indicate whether session was calcium imaging
%               or intrinsic. 

%Kachi O. circa 2014. Updated Aug 2016

%example usage:
%   maps = makeFourierPhaseMaps('k32','25-June-2015','000',2,5,0,1);
%
if ~exist('rescaleImg','var')
    rescaleImg = 1; %default value. reduce pixel size by a factor of 2.
end

if ~exist('rotateImage','var')
    rotateImage = 0;
end

if ~exist('gcampFlag','var')
    gcampFlag = 0;
end

day = datestr(day,'dd-mmm-yyyy');

%% choose directories 
%this section needs to modified to point the script to the correct
    %location.
if ispc 
    %ImgDirHeader = 'C:\Users\Kachi\Desktop\IntrinsicMaps\FourierMaps\';
    
%         dataDirHeader = ['Z:\Intrinsic\' fullfile(subject,'intrinsic imaging',day,exptnum)];
%     dataDirHeader = fullfile('C:\Users\Baguette\Documents\Intrinsic Data\',subject,day,exptnum);
    dataDirHeader = fullfile('E:\IntrinsicData\',subject,day,exptnum);
    
    ImgOutputDirHeader = 'C:\Users\ChurchlandLab\Desktop\Widefield\maps\fourier\';
    
elseif ismac
    ImgOutputDirHeader = '~/Desktop/maps/FourierMaps/';
    dataDirHeader = fullfile ('/Volumes/Churchland/Intrinsic',subject,day,exptnum);
    
end
%check file directory
if ~isdir(dataDirHeader)
    error('please check file directory, server connection, subject name,or date and try again')
end

%specify directory for output of image file.
ImageOutputDir = fullfile(ImgOutputDirHeader,subject,day,exptnum);

if ~isdir(ImageOutputDir)
    mkdir(ImageOutputDir) %create one.
end

set(0,'DefaultFigureWindowStyle','docked')

%% go to the data directory and get the data files
getfiles = dir(dataDirHeader);
files = getfiles(~[getfiles.isdir]); %cell array listing files in the data directory

datafiles = files(cellfun(@(x) strcmp(x(end-8:end),'_data.mat'),{files.name})); %find imaging data files within the cell array
datafiles = {datafiles.name};
numFiles = numel(datafiles);

%% grab session stimulus params and trial conditions
%the stimulus parameters are saved in a file called "analyzer.mat"
%load this file and extract the number of conditions, the stimulus
%parameter names and values, and the trial indices corresponding to a given
%trial condition

analyzerfile  = ls(fullfile(dataDirHeader,'*analyzer.mat')); %find the analyzer file
analyzerfile = fullfile(dataDirHeader,analyzerfile); %get the full file name including directory
load(deblank(analyzerfile)); %load the Analyzer file into the workspace.
trialconditions = Analyzer.loops.conds; %find the conditions
numconditions  = numel(trialconditions); %number of conditions
paramVals = nan(numconditions,numel(Analyzer.L.param)); %find parameter values
paramTrialInds = cell(1,numconditions);

%% get trial condition information
for n = 1:numconditions
    thisparamval = cell2mat(trialconditions{1,n}.val);
    paramNames(n,:) = trialconditions{1,n}.symbol;
    
    if ~isempty(thisparamval)
        paramVals(n,:) = thisparamval;
    end
    
    repeats = cell2mat(trialconditions{1,n}.repeats); %this has the trial numbers for each condition
    paramTrialInds{n} = struct2array(repeats);
end

%get frame rate
try
    %the frame rate parameter should be in Analyzer
    fps = Analyzer.M.framerate;
    
%     if isfield(Analyzer.M,'binFrameRate')
%         if ~isnan(Analyzer.M.binFrameRate)
%             fps = Analyzer.M.binFrameRate;
%         end
%     end
    
catch
    %if the frame rate is not there for some reason(old dataset), calculate
    %the frame rate by taking the mean difference of the timestamps of each
    %frame. The frames are stored in the variable "IntrinsicData"
    thisdatafile = fullfile(dataDirHeader,datafiles{1});
    load(thisdatafile);
    fps = str2double (sprintf('%1.2f',(1./mean(diff(IntrinsicData.timestamps)))));
    
end

%Make parameter list and save file
paramfilename = [subject '_' day '_' exptnum '_stimulus parameters.txt']; %parameter text file name

%loop will list the parameters and enter them into the designated text
%file.
for kk = 1:length(Analyzer.P.param)
    thisparamname  = Analyzer.P.param{kk}{1};
    thisparamval = Analyzer.P.param{kk}{3};
    eval(['params.' thisparamname '= thisparamval;']);    
end

%get trial durations
preT =  params.PreStimDelay;
stimT = params.StimDuration;
postT = params.PostStimDelay;

totT = preT+stimT+postT;
nframes = totT*fps;

%get the indices i.e. frames that correspond to a particlar time in the trial.
prestim_ind = 1:floor(preT*fps); %indices(or frames) that correspond to prestimulus period
stim_ind = floor(preT*fps)+1:floor(preT*fps)+ floor(stimT*fps); %indices(or frames) that correspond to stimulus period
post_ind = stim_ind(end)+1:nframes; %indices(or frames) that correspond to poststimulus period

%get stimulus cycle
if isfield(params,'CyclePeriod')
    %this is for the rotating wedge.
    numcycles = floor(params(1).StimDuration/params(1).CyclePeriod); %# cycles in session
    cyclePeriod = params.CyclePeriod;
else
    %this is for the sweeping bar.
    numcycles = params.NumCycles; %# cycles in session
    cyclePeriod = stimT/numcycles;
end

%% compute Fourier maps
%initialize storage variables
meanFourierPhaseMaps = cell(numconditions,1); %empty cell array for storing the phase maps for each condition
meanFourierMagMaps = cell(numconditions,1);

conditions = meanFourierPhaseMaps;
fignum = 0;
close all
for thiscondition = 1 :numconditions
    %Each trial is saved as a sequence of images. One might have 32 trials for the entire
    %imaging session, with 4 conditions. This turns out to be 8 trials per
    %condition. However the sequence of trials and consequently the data files might not be saved sequentially.
    %i.e. trials 1-8 might not correspon to conditon #1.
    %To solve this, need to loop thru each condtion and find the trial numbers that correspond to a
    %particular condition.
    timer = tic;
    trialstolookfor = paramTrialInds{thiscondition};
    
    for tt = 1 :numel(trialstolookfor)
        
        trial = trialstolookfor(tt);
        trialstring = ['_' num2str(trial) '_data.mat'];
        
        try
            %to find filename that matches trial string ("'trial#'_data.mat") and load data
            thisFile = datafiles{cellfun(@(x) strcmp(x(end-numel(trialstring)+1:end),trialstring),datafiles)};
        catch err
            %this hack, avoids trials that were not collected, say for
            %example when the session was interrupted.
            continue
        end
        %load the file into the workspace and extract the frames
        thisFile = fullfile(dataDirHeader,thisFile);
        load(deblank(thisFile)) %load to workspace
        thisTrialFrames = imrotate(imresize(single(squeeze(IntrinsicData.data)),(1/rescaleImg)),rotateImage); %extract data frames.squeeze to get rid of extraneous dimensions i.e. dimensions equal to 1
        
        stimFrames = thisTrialFrames(:,:,stim_ind); %grab only frames during which the stimulus was presented
        timeStamps = IntrinsicData.timestamps; %get the time stamps, i.e. when the frames were taken.
        frameTimes = timeStamps(stim_ind); %take only the times when the stimulus was present.
        frameTimes = frameTimes - frameTimes(1); %start from zero
        clear IntrinsicData

         %% Compute the discrete Fourier transform (i.e. component) for each frame
        %the Fourier transform at k cycles of a sequence x[n] is
        % X[k] = Cummulative sum, from n = 1 to N, of x[n]*exp((-j*2*pi*n*k)/N) OR otherewise written as x[n]*exp(-j*w*n)
        % where,
        % w is the angular frequency  = (2*pi*k)/N
        % N is equal to number of frames (samples) in one cycle period (i.e k = 1) OR the total number of frames (i.e. k = however many cycles of your stimulus was presented within the total number of frames)
        % x[n] is the image frame with w by h pixels
        % n is the frame number (i.e. discrete time point)
        % k is the number of cycles presented
        
        k = numcycles; %number of cycles presented during the stimulus epoch.
        N = stimT; %duration of stimulus frames in secs.
        n = frameTimes; %discrete time steps/stamps
        anglularFreqofFrames = (2*pi*k*n)./N; %angular frequency. belongs in the exponential of the fourier transform equation. see thisFourierComponent below.
        %         anglularFreqofFrames = 2*pi*n/cyclePeriod; % here k = 1,and N = cyclePeriod. It ends up being the same thing as the above equation.
                
        for n = 1:size(stimFrames,3)
            thisframe = stimFrames(:,:,n);
            if ~gcampFlag
                thisframe = 4096 - thisframe; %inverts the pixel response, only for intrinsic data which is a negative dip
            end
            if n == 1
                fourierTransform  = zeros(size(thisframe)); %start with zero. the fourier transform of each frame will be added together
            end
            thisFourierComponent = thisframe .* exp(1i*anglularFreqofFrames(n)); % x[n]*exp((-j*2*pi*n*k)/N)
            fourierTransform = fourierTransform + thisFourierComponent; % The fourier transform at k cycles, X[k] = Cummulative sum, from n = 1 to N, of x[n]*exp((-j*2*pi*n*k)/N) take a cummulative sum of the fourier component of each frame (i.e. sample)
        end
        
        %% remove spectral (f0) leakage.--> optional
        %From the little I understand this helps to reduce smearing of the frequency spectrum caused by computing the Fourier transform.
        %The smearing occurs because the assumption of the FFT is that the
        %signal is periodic and it's duration is infinite, and we only ever observe a fraction
        %of the infinitely long signal. Because of this we might have discontinuities (or imperfect periodic signal) which contribute to spectral leakage.
        %This website:http://bugra.github.io/work/notes/2012-09-15/Spectral-Leakage/ offers a nice explanation with figures.
        f0 = sum(stimFrames(:,:,1:2),3)/2;
        if ~gcampFlag
            f0 = 4096 - f0;
        end
        fourierTransform = fourierTransform - f0*sum(exp(1i*anglularFreqofFrames)); 
        fourierTransform = 2*fourierTransform ./n; %normalize/scale the transform
       
        %% sum phase and magnitude maps
        if tt == 1
            fourierPhase = zeros(size(thisframe));
            fourierMag = zeros(size(thisframe));
        end
        
        fourierPhase = fourierPhase + angle(fourierTransform); %compute the mean phase (angle) for each peak and sum for all trials in this condition.
        fourierMag = fourierMag + abs(fourierTransform);
    end
    thisFourierPhaseMap = fourierPhase/tt;%take the average angle (phase) map for this condition
    thisFourierMagMap = fourierMag/tt;%average magnitude map for this condition
    thisFourierMagMap = (thisFourierMagMap - min(thisFourierMagMap(:)))./range(thisFourierMagMap(:)); %normalize from zero to 1
    
    %make condition title
    conditionTitle = [];
    for i = 1:size(paramNames,2)
        conditionTitle = [conditionTitle paramNames(thiscondition,i) ': ' num2str(paramVals(thiscondition,i)) ' '];
    end
    conditionTitle = cell2mat(conditionTitle);
    conditionTitle = deblank(conditionTitle);
    
    %plot mean phase map
    fignum = fignum+1;
    figure(fignum);
    subplot(2,1,1)
    imshow(thisFourierPhaseMap,[]);colorbar;colormap((jet));
    title(conditionTitle);
    subplot(2,1,2)
    imshow(thisFourierMagMap,stretchlim(thisFourierMagMap)');colorbar;colormap((jet));
    
    %convert to rgb image
    bit = 2^8;
    thisFourierPhaseMapDisplay = (thisFourierPhaseMap - min(thisFourierPhaseMap(:)))./(max(thisFourierPhaseMap(:))-min(thisFourierPhaseMap(:))); %scale the map from 0 to 1 by subtracting the minimum value of the mean phase map from the mean phase map, then dividing by the range (max - min).
    phaseIndxMap = gray2ind(imadjust(thisFourierPhaseMapDisplay),bit);
    phaseRGBMap = ind2rgb(phaseIndxMap,jet(bit));
    figFilename = conditionTitle;
    figFilename(~((figFilename ~= ':') & (figFilename ~= ';'))) = '_';
    
    imwrite(phaseRGBMap,(fullfile(ImageOutputDir,[figFilename '.tiff']))) %write TiFF file.
%     print(fignum,(fullfile(ImageOutputDir,[figFilename '.eps'])),'-depsc2') %create EPS file
    
    conditions{thiscondition} = conditionTitle;
    meanFourierPhaseMaps{thiscondition} = thisFourierPhaseMap; %store phase map in cell array.
    meanFourierMagMaps{thiscondition} = thisFourierMagMap; %store mag map in cell array
    
    toc(timer)
end
maps.phaseMaps = meanFourierPhaseMaps;
maps.magMaps = meanFourierMagMaps;
maps.conditions = conditions;
%% subtract maps generated from stimulus moving in the opposite direction
azimuthWidthDeg = round(Analyzer.M.widthDeg);
elevationHeightDeg = round(Analyzer.M.heightDeg);
monitorAzDeg = round(Analyzer.M.screenAngle);
monitorElDeg = round(Analyzer.M.screenCenterEyeVerticalDeg);
azimuthRange = [monitorAzDeg-round(azimuthWidthDeg/2) monitorAzDeg+round(azimuthWidthDeg/2)];
elevationRange = [0 elevationHeightDeg] - elevationHeightDeg/2 + monitorElDeg;

if isfield(params,'BarOrient')
    %for sweeping bar stimulus
    
    %BarOrient == 1 horizontal bar; BarOrient == 0 vertical bar
    barOrientValcol = find(strcmp('BarOrient',paramNames)); %
    hbar_inds = find(paramVals(barOrientValcol) == 1);
    verticalMap = meanFourierPhaseMaps{hbar_inds(1)} - meanFourierPhaseMaps{hbar_inds(2)}; %to cancel hemodynamic delay, subtract phase map for reverse direction
    verticalMap = (verticalMap - min(verticalMap(:))) ./(max(verticalMap(:)) - min(verticalMap(:))); %scale from 0 to 1
    maps.vertical = verticalMap;
    verticalMapScaled = diff(elevationRange)*(verticalMap) + elevationRange(1); %scale to stimulus units

    %plot vertical map
    figure(5);clf
    subplot(2,1,1)
    imshow(verticalMapScaled,[]);colorbar;colormap((jet));
    title('vertical (elevation) retinotopy')
    subplot(2,1,2)
    magMapV =(meanFourierMagMaps{hbar_inds(1)}+ meanFourierMagMaps{hbar_inds(2)})/2;
    magMapV = (magMapV - min(magMapV(:)))./(range(magMapV(:)));
    imshow(magMapV,stretchlim(magMapV));colorbar;colormap(jet);
%     
    %save tiff and eps versions   
    figFilename = 'retinotopy_elevation map';
    verticalIndxMap = gray2ind(imadjust(verticalMap),bit);
    verticalRGBMap = ind2rgb(verticalIndxMap,jet(bit));
    imwrite(verticalRGBMap,(fullfile(ImageOutputDir,[figFilename '.tiff'])))
%     print(fignum,(fullfile(ImageOutputDir,[figFilename '.eps'])),'-depsc2')
   
    %horizontal map
    vbar_inds = find(paramVals(barOrientValcol) == 0);
    horizontalMap = meanFourierPhaseMaps{vbar_inds(1)} - meanFourierPhaseMaps{vbar_inds(2)};
    horizontalMap = (horizontalMap - min(horizontalMap(:)))./ (max(horizontalMap(:))- min(horizontalMap(:))); %scale from 0 to 1
    horizontalMapScaled  = diff(azimuthRange)*(horizontalMap) + azimuthRange(1); %scale to stimulus units 
    
    maps.horizontal = horizontalMap;
    figure(6);clf
    subplot(2,1,1)
    imshow(horizontalMapScaled,[]); colorbar;colormap((jet));
    title('horizontal (azimuth) retinotopy')
    subplot(2,1,2)
    magMapH =(meanFourierMagMaps{vbar_inds(1)}+ meanFourierMagMaps{vbar_inds(2)})/2;
    magMapH = (magMapH - min(magMapH(:)))./(range(magMapH(:)));
    imshow(magMapH,stretchlim(magMapH));colorbar;colormap(jet);
    
%     %save two (tiff and eps) versions of the horizontal map
    figFilename = 'retinotopy_azimuth map';   
    horizontalIndxMap = gray2ind(imadjust(horizontalMap),bit);
    horizontalRGBMap = ind2rgb(horizontalIndxMap,jet(bit));
    imwrite(horizontalRGBMap,(fullfile(ImageOutputDir,[figFilename '.tiff'])))
    %     print(fignum,(fullfile(ImageOutputDir,[figFilename '.eps'])),'-depsc2')
    
    save(fullfile(ImageOutputDir,'retinotopy_azimuth map.mat'),'horizontalMapScaled')
    save(fullfile(ImageOutputDir,'retinotopy_elevation map.mat'),'verticalMapScaled')
    %% Compute visual field sign map from Callaway lab visual area segmentation paper-->optional
    
    kmap_hor = spatialFilterGaussian(horizontalMapScaled,1);
    kmap_vert = spatialFilterGaussian(verticalMapScaled,1);
     
    pixpermm = 1000/(7.2*rescaleImg*Analyzer.M.spatialBinFactor); %camera: 1 pixel = 7.2 um
        
    mmperpix = 1/pixpermm;
    
    [dhdx, dhdy] = gradient(kmap_hor);
    [dvdx, dvdy] = gradient(kmap_vert);
    
    xdom = (0:size(kmap_hor,2)-1)*mmperpix;
    ydom = (0:size(kmap_hor,1)-1)*mmperpix;

    graddir_hor = atan2(dhdy,dhdx);
    graddir_vert = atan2(dvdy,dvdx);
    
    vdiff = exp(1i*graddir_hor) .* exp(-1i*graddir_vert); %Should be vert-hor, but the gradient in Matlab for y is opposite.
    VFS = sin(angle(vdiff)); %Visual field sign map
    id = find(isnan(VFS));
    VFS(id) = 0;
    VFS = spatialFilterGaussian(VFS,5);  %Important to smooth before thresholding below
    
    figure(111), clf
    imagesc(xdom,ydom,VFS,[-1 1]), axis image; axis off
    colorbar
    title('sin(angle(Hor)-angle(Vert))')
    
    
elseif isfield(params,'Rotation')
    %for rotating wedge stimulus
    rotationDiffMap = maps.phaseMaps{1} - maps.phaseMaps{2};
    rotationDiffMap = (rotationDiffMap - min(min(rotationDiffMap)))./(max(max(rotationDiffMap))- min(min(rotationDiffMap)));
    rotationDiffMap = rotationDiffMap*2*pi;
    maps.rotationDiff = rotationDiffMap;
    
    figure(5);clf
    imshow(rotationDiffMap,[]);colorbar;colormap((jet));
    title('Rotating Wedge difference map')
    figFilename = 'Rotating wedge difference map';
    
    rotationIndxMap = gray2ind(imadjust(rotationDiffMap./(2*pi)),bit);
    rotationRGBMap = ind2rgb(rotationIndxMap,jet(bit));
%     imwrite(rotationRGBMap,fullfile(ImageOutputDir,[figFilename '.tiff']))
%     print(fignum,fullfile(ImageOutputDir,[figFilename '.eps']))
end

set(0,'DefaultFigureWindowStyle','normal');
% horizontalMapScaled = imrotate(horizontalMapScaled,rotateImage);
% verticalMapScaled = imrotate(verticalMapScaled,rotateImage);


function img = spatialFilterGaussian(img, sigma)

% cutoff = ceil(2*sigma);
% h=fspecial('gaussian',[1,2*cutoff+1],sigma);
% 
% imgOut = conv2(h,h,img,'same'); %apply spatial filter
hh = fspecial('gaussian',size(img),sigma); 
hh = hh/sum(hh(:));
img = ifft2(fft2(img).*abs(fft2(hh)));
