% ************************************************************************
% Function: aeDiscriminatorDesign
%
% Initialise the discriminator network
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetDis    : initialised discriminator network
%
% ************************************************************************


function dlnetDis = aeDiscriminatorDesign( paramDis )


% create the input layer
layersDis = featureInputLayer( paramDis.input, 'Name', 'in' );

% create the hidden layers
for i = 1:paramDis.nHidden
    layersDis = [ layersDis; ...
        fullyConnectedLayer( paramDis.nFC, 'Name', ['fc' num2str(i)] )
        leakyReluLayer( paramDis.scale, 'Name', ['relu' num2str(i)] )
        dropoutLayer( paramDis.dropout, 'Name', ['drop' num2str(i)] )
        ]; %#ok<AGROW> 
end

% create final layers
layersDis = [ layersDis; ...    
        fullyConnectedLayer( 1, 'Name', 'fcout' )
        sigmoidLayer( 'Name', 'out' )
        ];

lgraphDis = layerGraph( layersDis );
dlnetDis = dlnetwork( lgraphDis );

end