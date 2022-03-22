% ************************************************************************
% Function: padData
% Purpose:  Pad time series to a specified length
%
% Parameters:
%       X: cell array of time series 
%       padLen: specified padded length
%       padValue: padding value
%       padLoc: padding location {'Start', 'End', 'Both'}
%
% Output:
%       XP: padded numeric array
%
% ************************************************************************


function XP = padData( X, padLen, padValue, padLoc )

if ischar( padValue )
    if strcmpi( padValue, 'same' )
        sameValue = true;
    else
        error( 'Unrecognised padding type.');
    end
else
    sameValue = false;
end

padLoc = lower( padLoc );

nObs = length( X );
nDim = size( X{1}, 2 );

XP = zeros( padLen, nObs, nDim );

for i = 1:nObs

    trialLen = min( [ size(X{i}, 1), padLen] );

    if sameValue
        xStart = X{i}(1);
        xEnd = X{i}(end);
    else
        xStart = padValue;
        xEnd = padValue;
    end

    switch padLoc
        case 'start' 
            % insert padding at the beginning
            value = [ xStart 0 ];
            lengths = [ padLen-trialLen 0 ];
        case 'end'
            % insert padding at the end
            value = [ 0 xEnd ];
            lengths = [ 0 padLen-trialLen ];
        case 'both'
            % insert padding at both ends, roughly evenly
            value = [ xStart xEnd ];
            startLen = fix( (padLen-trialLen)/2 );
            lengths = [ startLen padLen-trialLen-startLen ];
        otherwise
            error('Unrecognised padding location.');
    end

    XP( :, i, : ) = [ ones( lengths(1), nDim )*value(1); ...
                        X{i}(end - trialLen+1:end, :); ...
                          ones( lengths(2), nDim )*value(2) ];

end
    
end

