% ************************************************************************
% Function: getJumpGRFData
% Purpose:  Load and extract ground reaction force data from jumps
%
% Parameters:
%
% Output:
%       curveSet: extracted VGRF data structure
%       typeSet: jump type
%
% ************************************************************************


function [ X, A ] =  getJumpGRFData

if ismac
    rootpath = '/Users/markgewhite/Google Drive/';
else 
    rootpath = 'C:\Users\m.g.e.white\My Drive\';
end
dataFolder = 'Academia/Postdoc/Datasets/Jumps';
datapath = [ rootpath dataFolder ];

% load data from file
load( fullfile( datapath, 'jumpGRFData.mat' ), ...
      'grf', 'bwall', 'sDataID', 'sJumpID', ...
      'jumpOrder', 'nJumpsPerSubject' );

% exclude jumps from subjects in the second data collection
subjectExclusions = find( ismember( sDataID, ...
            [ 14, 39, 68, 86, 87, 11, 22, 28, 40, 43, 82, 88, 95, 97, ...
              100, 121, 156, 163, 196 ] ) );

% exclude specific jumps with excessive double movements
jumpExclusions = [  6104, 6114, 6116, ...
                    9404, 9411, 9413, 9416, ...
                    0101, 0103, 0106, 0111, 0114 ];

% set the options for jump detection
options.tFreq = 1; % time intervals per second
options.initial = 1; % initial padding value
options.threshold1 = 0.08; % fraction of BW for first detection
options.threshold2 = 0.025; % fraction of BW for sustained low threshold
options.prctileLimit = 90; % no outliers are beyond this limit

[ rawDataSet, ~, typeSet ] =  extractVGRFData( ... 
                                    grf, bwall, nJumpsPerSubject, ...
                                    sDataID, sJumpID, jumpOrder, ...
                                    subjectExclusions, jumpExclusions, ...
                                    options );
% extract the relevant data
X = rawDataSet{2}; % jumps with and without arm swing
A = typeSet{2}; % jump class (with/without arm swing)

end