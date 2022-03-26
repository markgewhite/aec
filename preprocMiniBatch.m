% ************************************************************************
% Function: preprocMiniBatch
%
% Preprocess a sequence batch for training
%
% ************************************************************************

function [ X, XN, Y ] = preprocMiniBatch( XCell, XNCell, YCell )

X = padsequences( XCell, 2 );

if nargin > 1
    XN = cat( 2, XNCell{:} );   
    Y = cat( 2, YCell{:} );
else
    XN = [];
    Y = [];
end

end