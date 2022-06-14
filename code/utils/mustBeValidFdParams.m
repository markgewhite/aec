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
                {'BasisOrder', 'PenaltyOrder', 'Lambda'})), ...
        'Padding must have fields: BasisOrder, PenaltyOrder and Lambda.');

    assert( fix( fdParams.BasisOrder ) == fdParams.BasisOrder & ... 
            fdParams.BasisOrder >= 2, ...
                'basisOrder is not an integer >= 2.' );

    assert( fix( fdParams.PenaltyOrder ) == fdParams.PenaltyOrder &  ...
            fdParams.PenaltyOrder >= 0 & ...
            fdParams.PenaltyOrder <= fdParams.BasisOrder-2, ...
                'penaltyOrder is not an integer >= 0 and <= basisOrder-2.' );

    assert( isa( fdParams.Lambda, 'double' ), ...
                'lambda is not a double.' );


end