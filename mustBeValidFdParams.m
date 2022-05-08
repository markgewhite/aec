% ************************************************************************
% Function: mustBeValidFdParams
%
% Custom validator function (code from Matlab)
% Check if the structure has all the requisite fields 
% and that all those fields are valid
%
% ************************************************************************

function mustBeValidFdParams( fdParams )
    % Validate if functional data parameters are valid
    arguments
        fdParams   struct
    end

    assert( isempty(setxor( fieldnames( fdParams ), ...
                {'tSpan', 'basisOrder', 'penaltyOrder', 'lambda'})), ...
        'Padding must have fields: tSpan, basisOrder, penaltyOrder and lambda.');

    assert( isvector( fdParams.tSpan ), ...
                'tSpan is not a vector.' );

    assert( fix( fdParams.basisOrder ) == fdParams.basisOrder & ... 
            fdParams.basisOrder >= 2, ...
                'basisOrder is not an integer >= 2.' );

    assert( fix( fdParams.penaltyOrder ) == fdParams.penaltyOrder &  ...
            fdParams.penaltyOrder >= 0 & ...
            fdParams.penaltyOrder <= fdParams.basisOrder-2, ...
                'penaltyOrder is not an integer >= 0 and <= basisOrder-2.' );

    assert( isa( fdParams.lambda, 'double' ), ...
                'lambda is not a double.' );


end