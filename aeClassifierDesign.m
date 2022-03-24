% ************************************************************************
% Function: aeClassifierDesign
%
% Initialise the classifier network
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetCls    : initialised classifier network
%
% ************************************************************************


function dlnetCls = aeClassifierDesign( paramCls )


% create the input layer
layersCls = featureInputLayer( paramCls.input, 'Name', 'in' );

% create the hidden layers
for i = 1:paramCls.nHidden
    layersCls = [ layersCls; ...
        fullyConnectedLayer( paramCls.nFC, 'Name', ['fc' num2str(i)] )
        leakyReluLayer( paramCls.scale, 'Name', ['relu' num2str(i)] )
        dropoutLayer( paramCls.dropout, 'Name', ['drop' num2str(i)] )
        ]; %#ok<AGROW> 
end

% create final layers
layersCls = [ layersCls; ...    
        fullyConnectedLayer( paramCls.output, 'Name', 'fcout' )
        sigmoidLayer( 'Name', 'out' )
        ];

lgraphCls = layerGraph( layersCls );
dlnetCls = dlnetwork( lgraphCls );

end