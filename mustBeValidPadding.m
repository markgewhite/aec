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
            {'length', 'value', 'location', 'longest', 'same', 'anchoring' })), ...
        'Padding must have fields: length, value, location, longest, same and anchoring');

    assert( isa( padding.value, 'double' ), ...
                'Padding value not numeric.' );

    assert( isa( padding.length, 'double' ) &  ...
                fix( padding.length ) == padding.length & ...
                padding.length > 0, ...
                'Padding length not an integer.' );

    assert( ismember( padding.location, ...
                        {'Left', 'Right', 'Both', 'Symmetric'} ), ...
                'Padding location is not valid.' );

    assert( ismember( padding.anchoring, ...
                        {'None', 'Left', 'Right', 'Both'} ), ...
                'Anchoring location is not valid.' );

end