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

if ischar( padValue )
    if strcmpi( padValue, 'same' )
        sameValue = true;
    else
        error( 'Unrecognised padding type.');
    end
else
    sameValue = false;
end

nObs = length( X );
nDim = size( X{1}, 2 );

XP = zeros( padLen, nObs, nDim );

for i = 1:nObs
    % trial length
    trialLen = min( [ size(X{i}, 1), padLen] );
    if sameValue
        padValue = X{i}(1);
    end
    % insert padding at beginning
    XP( :, i, : ) = [ ones( padLen-trialLen, nDim )*padValue; ...
                        X{i}(end - trialLen+1:end, :) ];
end
    
end