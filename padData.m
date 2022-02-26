% ************************************************************************
% Function: padData
% Purpose:  Pad time series to a specified length
%
% Parameters:
%       X: cell array of time series 
%       padLen: specified padded length
%       padValue: padding value
%
% Output:
%       Graphs
%
% ************************************************************************


function XP = padData( X, padLen, padValue )

% number of series
nObs = length( X );

% number of dimensions (assuming all the same)
nDim = size( X{1}, 2 );

% padded matrix of series
XP = zeros( padLen, nObs, nDim );

for i = 1:nObs
    % trial length
    trialLen = min( [ size(X{i}, 1), padLen] );
    % insert padding at beginning
    XP( :, i, : ) = [ ones( padLen-trialLen, nDim )*padValue; ...
                        X{i}(end - trialLen+1:end, :) ];
end
    
end