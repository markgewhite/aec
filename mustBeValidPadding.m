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
            {'Length', 'Value', 'Location', 'Longest', 'Same', 'Anchoring' })), ...
        'Padding must have fields: Length, Value, Location, Longest, Same and Anchoring');

    assert( isa( padding.Value, 'double' ), ...
                'Padding value not numeric.' );

    assert( isa( padding.Length, 'double' ) &  ...
                fix( padding.Length ) == padding.Length & ...
                padding.Length > 0, ...
                'Padding length not an integer.' );

    assert( ismember( padding.Location, ...
                        {'Left', 'Right', 'Both', 'Symmetric'} ), ...
                'Padding location is not valid.' );

    assert( ismember( padding.Anchoring, ...
                        {'None', 'Left', 'Right', 'Both'} ), ...
                'Anchoring location is not valid.' );

end