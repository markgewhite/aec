% ************************************************************************
% Function: mustBeValidPadding
%
% Custom validator function (code from Matlab)
% Check if the structure has all the requisite fields 
% and that all those fields are valid
%
% ************************************************************************

function mustBeValidPadding( padding )
    arguments
        padding   struct
    end

    assert( isempty(setxor( fieldnames( padding ), ...
                {'value', 'location', 'length'})), ...
                'Padding must have fields: value, location and length');

    assert( isa( padding.value, 'double' ), ...
                'Padding value not numeric.' );

    assert( isa( padding.length, 'double' ) &  ...
                fix( padding.length ) == padding.length & ...
                padding.length > 0, ...
                'Padding length not an integer.' );

    assert( ismember( padding.location, ...
                        {'left', 'right', 'both', 'symmetric'} ), ...
                'Padding location is not valid.' );

end