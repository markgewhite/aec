% ************************************************************************
% Function: getJumpACCData
% Purpose:  Load accelerometer data from the jumps
%
% Parameters:
%
% Output:
%       curveSet: extracted VGRF data structure
%       typeSet: jump type
%
% ************************************************************************


function [ X, A ] =  getJumpACCData

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
A = withArms; % jump class (with/without arm swing)

end