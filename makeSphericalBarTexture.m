function makeSphericalBarTexture

%drifting bar stimulus i.e. full field drifting grating
%last modified 30June2016

global  screenPTR Mstate screenNum srcRect destRect 
global SyncStateTxtr barTex syncWX syncWY  SyncLoc SyncPiece

black = BlackIndex(screenPTR);
Screen('FillRect', screenPTR,black);
Screen('Flip', screenPTR);

%% get stimulus parameters
Pstruct = getParamStruct;
barThicknessDeg = Pstruct.BarThickness;	
barChecksize = Pstruct.CheckSize; %in degrees, if 0 make white bar
flickerBar = Pstruct.FlickerRate;
numcycles = Pstruct.NumCycles;
barOrient = Pstruct.BarOrient;
stimdur = Pstruct.StimDuration;
movieDurationSecs = stimdur/numcycles; % Run the movie animation for a fixed period.

screenDistance = Mstate.screenDist; %in cm, needs to be converted to pixels
w = Mstate.screenXcm;% width of monitor
h = Mstate.screenYcm; %height of monitor
cx = Pstruct.eyeXLocation; %eye x location, cm
cy = Pstruct.eyeYLocation; %eye y location, cm

zdistTop = hypot(screenDistance,h-cy); %distance from eye to top of screen
zdistBottom = hypot(screenDistance,cy); %distance from eye to bottom of screen

% Find the color values which correspond to white and black: Usually
% black is always 0 and white 255, but this rule is not true if one of
% the high precision framebuffer modes is enabled via the
% PsychImaging() commmand, so we query the true values via the
% functions WhiteIndex and BlackIndex:
white=WhiteIndex(screenNum);
black=BlackIndex(screenNum);

priorityLevel=MaxPriority(screenPTR); %#ok<NASGU>

screenRes = Screen('Resolution',screenNum);

scale = Pstruct.ScreenScaleFactor;

pxXmax = screenRes.width/scale;
pxYmax = screenRes.height/scale;
pixelspercmX = round(pxXmax/w);
pixelspercmY = round(pxYmax/h);

%internal conversions?
top = h - cy;
bottom = -cy;
right = cx;
left = cx-w;

[xi,yi] = meshgrid(1:pxXmax,1:pxYmax);
cart_pointsX = left + (w/pxXmax).*xi;
cart_pointsY = top - (h/pxYmax).*yi;
cart_pointsZ = zdistTop + ((zdistBottom-zdistTop)/pxYmax).*yi;
[sphr_pointsTh sphr_pointsPh sphr_pointsR] = cart2sph(cart_pointsZ,cart_pointsX,cart_pointsY);

%rescale Cartesian maps into dimensions of radians
xmaxRad = max(sphr_pointsTh(:));
ymaxRad = max(sphr_pointsPh(:));
fx = xmaxRad/max(cart_pointsX(:));
fy = ymaxRad/max(cart_pointsY(:));

%% determine stimulus size in pixels
% barThicknessPixels = round(tan(barThicknessDeg)*screenDistance

texsizeY = ceil(pxXmax/2);
texsizeZ = ceil(pxYmax/2);

% This is the visible size of the texture. It is twice the half-width
% of the texture plus one pixel to make sure it has an odd number of
% pixels and is therefore symmetric around the center of the texture:
visiblesizeX=2*texsizeY+1;
visiblesizeY = 2*texsizeZ+1;

%% define destination of stimuli
frameRate=Screen('FrameRate',screenNum);

if frameRate == 0
    frameRate=60;
end
Mstate.screenFrameRate = frameRate;
numframes = floor(frameRate); %temporal period, i.e. number of frames in one cycle of bar sweep

barThicknesscm = tan(barThicknessDeg*pi/180)*screenDistance;
barXThickness = round(barThicknesscm * pixelspercmX); %vertical bar
barYThickness = round(barThicknesscm * pixelspercmY); %horizontal bar

screenSize = [pxYmax pxXmax];
if barOrient %horizontal bar
    barAngle = 90;
    barThickness = barYThickness;
    thisScreenSize = screenSize(1) + 2* barThickness; %direction for horizontal bar, vertical travel
else %vertical bar
    barAngle = 0;
    barThickness = barXThickness;
    thisScreenSize =screenSize(2) + 2*barThickness; %azimuth travel
end
cyclesPerPixel = 1/(thisScreenSize);
dutyCycle = 1 - (barThickness/thisScreenSize); %determines bar thickness

% grating
[x,y]=meshgrid(-screenSize(2)/2:screenSize(2)/2-1,-screenSize(1)/2:screenSize(1)/2-1);

angle=barAngle*pi/180; % bar orientation.
f=cyclesPerPixel*2*pi; % cycles/pixel

a=cos(angle)*f;
b=sin(angle)*f;

flickerInds = ones(1,numframes); %non-flickering checkerboard
barBackground = ones(pxYmax,pxXmax,flickerInds);

if barChecksize > 0
    barChecksizecm = tan(barChecksize*pi/180)*screenDistance;
    barChecksizepixels = round(barChecksizecm*pixelspercmY);

    checkerPattern = double(checkerboard(barChecksizepixels,1+round((pxYmax/2)/barChecksizepixels),1+round((pxXmax/2)/barChecksizepixels))<0.5);
    checkerPattern = checkerPattern(1:pxYmax,1:pxXmax);
    barBackground(:,:,1) = checkerPattern;  %checkerboard
    barBackground(:,:,2) = abs(1-checkerPattern); %reverse checkerboard
    
    if flickerBar >0
        flipChecksFrames = (frameRate/flickerBar)/2; %convert to frames per cycle
        flickerInds = floor(mod((0:numframes)/flipChecksFrames,2))+1;
    end
    
end

% Compute each frame of the movie and convert the frames, stored in
% MATLAB matices, into Psychtoolbox OpenGL textures using 'MakeTexture';
barTex = nan(1,numframes); 
%%

timestart = tic;

for ff = 1:numframes
    phase=((ff-1)/(numframes-1))*2*pi; %controls movement
    img=cos(a*x+b*y+phase);   %from DriftDemo2.m and Ian Nauhaus' code %makePerGratFrame_insep.m
    img = sign(img - cos(dutyCycle*pi)); %from Ian Nauhaus' code %makePerGratFrame_insep.m
    img = double(img<0);
    
    thisBarBackground = barBackground(:,:,flickerInds(ff));
    
    barTextureImg = img .* thisBarBackground;

    if Pstruct.sphereCorrectON
        barTextureImg = interp2(cart_pointsX.*fx,cart_pointsY.*fy,barTextureImg,sphr_pointsTh,sphr_pointsPh);
    end
    
    barTextureImg = white*(barTextureImg);
    barTex(1,ff)=Screen('MakeTexture', screenPTR, barTextureImg);
    
end

disp('')    
disp('done making texture')
toc(timestart)

srcRect=[0 0 visiblesizeX visiblesizeY]; %texture size
destRect = [0 0 pxXmax*scale pxYmax*scale]; %scale the stimulus to fill the screen


%%
syncWX = round(pixelspercmX*Mstate.syncSize*scale);
syncWY = round(pixelspercmY*Mstate.syncSize*scale);


SyncStateTxtr(1) = Screen(screenPTR, 'MakeTexture', white*ones(syncWY,syncWX));  % "hi"
SyncStateTxtr(2) = Screen(screenPTR, 'MakeTexture', black*ones(syncWY,syncWX)); % "low"
% coordinates for sync marker
SyncLoc = [0 0 syncWX-1 syncWY-1]';
SyncPiece = [0 0 syncWX-1 syncWY-1]';

