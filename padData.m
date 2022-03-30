% ************************************************************************
% Function: padData
% Purpose:  Pad time series to a specified length
%
% Parameters:
%       X: cell array of time series 
%       padLen: specified padded length or 'longest'
%       padValue: padding value
%       padLoc: padding location {'Start', 'End', 'Both', 'Symmetric'}
%
% Output:
%       XP: padded numeric array
%
% ************************************************************************


function XP = padData( X, padLen, padValue, padLoc )

if ischar( padLen )
   if strcmpi( padLen, 'longest' )
        padLen = max( cellfun( @length, X ) );
    else
        error( 'Unrecognised padding length.');
   end 
end

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
        xStart = X{i}(1,:);
        xEnd = X{i}(end,:);
    else
        xStart = padValue;
        xEnd = padValue;
    end

    switch padLoc
        case 'left' 
            % insert padding at the beginning
            padLeft = ones( padLen-trialLen, 1 )*xStart;
            padRight = [];

        case 'right'
            % insert padding at the end
            padLeft = [];
            padRight = ones( padLen-trialLen, 1 )*xEnd;

        case 'both'
            % insert padding at both ends, roughly evenly
            startLen = fix( (padLen-trialLen)/2 );
            padLeft = ones( startLen, 1 )*xStart;
            padRight = ones( padLen-trialLen-startLen, 1 )*xEnd;

        case 'symmetric'
            % insert padding at both ends as mirror image of opposite end
            startLen = fix( (padLen-trialLen)/2 );
            padLeft = X{i}( end-startLen+1:end, : );
            padRight = X{i}( 1:startLen, : );

        otherwise
            error('Unrecognised padding location.');
    end

    XP( :, i, : ) = [ padLeft; ...
                        X{i}(end - trialLen+1:end, :); ...
                          padRight ];

end
    
end

