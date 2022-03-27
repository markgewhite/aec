% ************************************************************************
% Function: preprocMiniBatch
%
% Preprocess a sequence batch for training
%
% ************************************************************************

function [ X, XN, Y ] = preprocMiniBatch( XCell, XNCell, YCell, ...
                                          padValue, padLoc )

X = padData( XCell, 'Longest', padValue, padLoc );
X = permute( X, [ 3 1 2 ] );

if ~isempty( XNCell )
    XN = cat( 2, XNCell{:} );   
else
    XN = [];
end

if ~isempty( YCell )
    Y = cat( 2, YCell{:} );
else
    Y = [];
end


end