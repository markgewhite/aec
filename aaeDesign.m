% ************************************************************************
% Function: aaeDesign
%
% Initialise the adversarial autoencoder networks
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec ] = aaeDesign( paramEnc, paramDec )


% define the encoder network
layersEnc = [
    featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 )
    %dropoutLayer( 0.1, 'Name', 'drop1' )
    fullyConnectedLayer( 200, 'Name', 'fc1' )
    %leakyReluLayer( paramEnc.scale, 'Name', 'lrelu1' );
    dropoutLayer( 0.2, 'Name', 'drop2' )
    fullyConnectedLayer( paramEnc.outZ, 'Name', 'fc2' )
    ];
    
lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
layersDec = [
    featureInputLayer( paramDec.input, 'Name', 'in' )
    %dropoutLayer( 0.1, 'Name', 'drop1' )
    fullyConnectedLayer( 200, 'Name', 'fc1' )
    dropoutLayer( 0.2, 'Name', 'drop2' )
    %leakyReluLayer( paramEnc.scale, 'Name', 'lrelu1' );
    fullyConnectedLayer( paramDec.outX, 'Name', 'fc2' )
    ];

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


end