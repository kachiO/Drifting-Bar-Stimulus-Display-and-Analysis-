%grating equation obtained from driftDemo.m
%finally figured out how to make moving bar based on drifting grating. 
%Kachi June 2016

clear
numFrames = 24; %movie frames to generate
barorient = 90; % in degrees, 90 (horizontal) or 0 (vertical)
screenSize = [1280 760];
barThickness = 15; %in pixels

screenSize = screenSize + 2*barThickness;
thisScreenSize = screenSize(1); %direction for horizontal bar, vertical travel

if ~barorient %vertical bar
    thisScreenSize =screenSize(2); %azimuth travel
end

cyclesPerPixel = 1/(thisScreenSize + 2*barThickness);
dutyCycle = 1-(barThickness/thisScreenSize); %determines bar thickness

% grating
[x,y]=meshgrid(-screenSize(2)/2:screenSize(2)/2-1,-screenSize(1)/2:screenSize(1)/2-1);

angle=barorient*pi/180; % orientation.
f=cyclesPerPixel*2*pi; % cycles/pixel

%coefficients determine the orientation of the bar 
a=cos(angle)*f; 
b=sin(angle)*f;
%%
for nn = 1:3 %for multiple cycles. need to only generate one cycle for each texture
    for i = 1:numFrames;
        phase=((i-1)/(numFrames-1))*2*pi; %controls movement
        m=cos(a*x+b*y+phase);
        m = sign(m - cos(dutyCycle*pi));
        m = single(m<0);
        figure(1);clf; imshow(m)
        pause(1/60)
    end
end