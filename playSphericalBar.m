function playSphericalBar

global  screenPTR Mstate srcRect destRect 
global SyncStateTxtr barTex  SyncLoc SyncPiece

%%
black = BlackIndex(screenPTR);
Screen('FillRect', screenPTR,black);
Screen('Flip', screenPTR);

Pstruct = getParamStruct;

% flickerBar = Pstruct.FlickerRate * (Pstruct.CheckSize ~=0);
direction = Pstruct.BarDirection;
numcycles = Pstruct.NumCycles;

stimdur = Pstruct.StimDuration;
movieDurationSecs = stimdur/numcycles; % Run the movie animation for a fixed period.

frameRate  = Mstate.screenFrameRate;
numframes = floor(frameRate); %temporal period, i.e. number of frames in one cycle of bar sweep

% Convert movieDuration in seconds to duration in frames to draw:
movieDurationFrames=round(movieDurationSecs * frameRate);
movieFrameIndices=floor(mod((0:movieDurationFrames-1)/(movieDurationFrames/numframes), numframes)) + 1;

if direction ==1 %forward direction
    movieFrameIndices = reshape(movieFrameIndices,1,max(size(movieFrameIndices)));
    movieFrameIndices = fliplr(movieFrameIndices);
   
end

% 
% if flickerBar
%     flipBarframes = (frameRate/flickerBar)/2; %convert to frames per cycle
%     flipBar = floor(mod((0:movieDurationFrames)/flipBarframes,2))+1;    
% else
    flipBar = ones(1,movieDurationFrames);

% end

% Use realtime priority for better timing precision:
priorityLevel=MaxPriority(screenPTR);
Priority(priorityLevel);

 % Query duration of one monitor refresh interval:
ifi=Screen('GetFlipInterval', screenPTR);

% Translate that into the amount of seconds to wait between screen
% redraws/updates:

% waitframes = 1 means: Redraw every monitor refresh. 
waitframes = 1;       

%% prestimulus delay
Screen('DrawTexture', screenPTR, SyncStateTxtr(2),SyncPiece,SyncLoc);
prestimTimeR  = Screen(screenPTR, 'Flip');% Sync us to the vertical retrace
PreStimTime  = prestimTimeR + Pstruct.PreStimDelay;

while prestimTimeR < PreStimTime
    Screen('DrawTexture', screenPTR, SyncStateTxtr(1),SyncPiece,SyncLoc);
    prestimTimeR = Screen('Flip', screenPTR, prestimTimeR + (waitframes - 0.5) * ifi);
end
Screen('DrawTexture', screenPTR, SyncStateTxtr(2),SyncPiece,SyncLoc);
Screen(screenPTR, 'Flip');

tic        
%% present Stimulus animation loop:
vbl = Screen('Flip',screenPTR);

for n = 1:numcycles
    for i=1:movieDurationFrames
        % Draw image:
        Screen('DrawTexture', screenPTR, barTex(flipBar(i),movieFrameIndices(i)),srcRect,destRect);
%         Screen('Flip', screenPTR);
        Screen('DrawingFinished',screenPTR);
        vbl = Screen('Flip',screenPTR,vbl+(waitframes-0.5)*ifi);
    end
end
toc

Screen('FillRect', screenPTR,black);
Screen('Flip', screenPTR);

%% Poststimulus loop--------------------------------%%%%%%%%%%%%:

Screen('DrawTexture', screenPTR, SyncStateTxtr(2),SyncPiece,SyncLoc);
postStimTimeR  = Screen(screenPTR, 'Flip');
PostStimTime  = postStimTimeR + Pstruct.PostStimDelay;

while postStimTimeR < PostStimTime
    Screen('DrawTexture', screenPTR, SyncStateTxtr(1),SyncPiece,SyncLoc);
    postStimTimeR = Screen('Flip', screenPTR, postStimTimeR + (waitframes - 0.5) * ifi);
end

Screen(screenPTR, 'Flip');
Screen('DrawTexture', screenPTR, SyncStateTxtr(2),SyncPiece,SyncLoc);
Screen(screenPTR, 'Flip');

Priority(0);

return
