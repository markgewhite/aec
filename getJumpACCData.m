% ************************************************************************
% Function: getJumpACCData
% Purpose:  Load accelerometer data from the jumps
%
% Parameters:
%
% Output:
%       X: extracted accelerometer time series
%       Y: outcome variable
%
% ************************************************************************


function [ X, Y ] =  getJumpACCData( outcome )

if ismac
    rootpath = '/Users/markgewhite/Google Drive/';
else 
    rootpath = 'C:\Users\m.g.e.white\My Drive\';
end
dataFolder = 'Academia/Postdoc/Datasets/Jumps/';
datapath = [ rootpath dataFolder ];

% load data from file
load( fullfile( datapath, 'jumpACCData.mat' ), ...
      'accSignal', 'withArms', 'outcomes' );

% extract the relevant data
X = accSignal{1,2}; % lower back sensor for jumps with/without arm swing

switch outcome
    case 'JumpType'
        Y = withArms; % jump class (with/without arm swing)
    case 'JumpHeight'
        Y = outcomes.all.jumpHeight;
    case 'PeakPower'
        Y = outcomes.all.peakPower;
    otherwise
        error('Unrecognised outcome variable.');
end

end