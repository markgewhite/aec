% ************************************************************************
% Function: timeNormalize
% Purpose:  Time normalise data to list of specified lengths.
%
% Parameters:
%       X           : cell array of time series
%       fixedLength : fixed length for time normalisation
%
% Output:
%       XN          : time normalised time series (array)
%
% ************************************************************************


function XN = timeNormalize( X, fixedLength )

if iscell( X )
    nObs = length( X );
    nDim = size( X{1}, 2 );
else
    [ ~, nObs, nDim ] = size( X );
end

XN = single(zeros( fixedLength, nObs, nDim ));

% define standard time series
tSpan1 = linspace( 0, 1, fixedLength ); 

for i = 1:nObs
    % get next row
    if iscell( X )
        x = X{i};
    else
        x = X( :, i,: );
    end
    % define current time series
    tSpan0 = linspace( 0, 1, size(x,1) );
    % interpolate to fit the standard timescale
    for j = 1:nDim
        XN( :, i, j ) = interp1( tSpan0, x(:,j), tSpan1, 'spline', 'extrap' ); 
    end
end

end