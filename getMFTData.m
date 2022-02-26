% ************************************************************************
% Function: getMFTData
% Purpose:  Load and extract Multistage Fitness Test data
%
% Parameters:
%
% Output:

%
% ************************************************************************


function [ X, S ] =  getMFTData

if ismac
    rootpath = '/Users/markgewhite/Google Drive/';
else 
    rootpath = 'C:\Users\m.g.e.white\My Drive\';
end
dataFolder = 'Academia/Postdoc/Datasets/MFT';
datapath = [ rootpath dataFolder ];

% get the file and folder listing
listing = dir(fullfile( datapath, 'p*' ));
listing = struct2table( listing );
nFiles = size( listing, 1 );

% read raw data
Xr = cell( nFiles, 1 );
for i = 1:nFiles

    fileData = readtable( fullfile(datapath, listing.name{i} ) );
    Xr{i} = table2array(fileData( :, 4:6 ));

end


% setup filter
freqSample = 40;
freqCutoff = 5;
freqNorm = freqCutoff/(freqSample/2);
filterDesign = designfilt( 'HighPassFir', ...
                           'FilterOrder', 70, ...
                           'CutoffFrequency', freqNorm );

d = mean( grpdelay( filterDesign ) );

% pad out time series
Xr = cellfun( @(x) [x; zeros(d,3)], Xr, 'UniformOutput', false );

% apply filter
Xf = cellfun( @(x) filter( filterDesign, x ), Xr, 'UniformOutput', false );

% take the first derivative, squared
dXf = cellfun( @(x) diff(x).^2, Xf , 'UniformOutput', false );

% take the sum across dimensions
dXf = cellfun( @(x) sum(x, 2), dXf , 'UniformOutput', false );

% find the prominent peaks
minPkProm = 5;
[ ~, pkIdx ] = cellfun( @(x) findpeaks( x, 'MinPeakProminence', minPkProm ), ...
                                        dXf , 'UniformOutput', false );

% find the cycle lengths
cycleLen = cellfun( @(x) diff(x), pkIdx, 'UniformOutput', false );

% set cycle limits in relation to the mode
lenMin = fix(0.75*cellfun( @mode, cycleLen ));
lenMax = 2*cellfun( @mode, cycleLen );

% find the peaks again but ignore those too close together
[ ~, pkIdx ] = cellfun( @(x,len) ...
    findpeaks( x, 'MinPeakProminence', minPkProm, ...
                  'MinPeakDistance', len  ), ...
                  dXf, num2cell(lenMin) , 'UniformOutput', false );

% obtain the revised individual cycle lengths
cycleLen = cellfun( @(x) diff(x), pkIdx, 'UniformOutput', false );

isValid = cellfun( @(x,len) x <= len, ...
                     cycleLen, num2cell(lenMax), 'UniformOutput', false );
nValid = sum(cellfun(@sum, isValid));

X = cell( nValid, 1 );
S = zeros( nValid, 1 );

% divide up the time series into cycles
k = 0;
for i = 1:nFiles
    for j = 2:length( pkIdx{i} )
        if isValid{i}(j-1)
            k = k + 1;
            X{k} = Xr{i}( pkIdx{i}(j-1):pkIdx{i}(j), : );
            S(k) = i;
        end
    end
end

end