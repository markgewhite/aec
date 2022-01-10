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
%           dlnetDis    : initialised discriminator network
%           dlnetCls    : initialised classifier network
%
% ************************************************************************

function [ dlnetEnc, dlnetDec, dlnetDis, dlnetCls ] = ...
                aaeDesign( paramEnc, paramDec, paramDis, paramCls )


% define the encoder network
layersEnc = [
    featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 )
    fullyConnectedLayer( 100, 'Name', 'fc1' )
    sigmoidLayer( 'Name', 'sig1' )
    dropoutLayer( 0.2, 'Name', 'drop2' )
    fullyConnectedLayer( paramEnc.outZ, 'Name', 'fc2' )
    ];
    
lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
layersDec = [
    featureInputLayer( paramDec.input, 'Name', 'in' )
    fullyConnectedLayer( 100, 'Name', 'fc1' )
    sigmoidLayer( 'Name', 'sig1' )
    dropoutLayer( 0.2, 'Name', 'drop2' )
    fullyConnectedLayer( paramDec.outX, 'Name', 'fc2' )
    ];

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


% define the discriminator network
layersDis = [
    featureInputLayer( paramDis.input, 'Name', 'in' )
    fullyConnectedLayer( 21, 'Name', 'fc1' )
    dropoutLayer( 0.2, 'Name', 'drop1' )
    fullyConnectedLayer( 1, 'Name', 'fc2' )
    sigmoidLayer( 'Name', 'out' )
    ];

lgraphDis = layerGraph( layersDis );
dlnetDis = dlnetwork( lgraphDis );

% define the classifier network
layersCls = [
    featureInputLayer( paramCls.input, 'Name', 'in' )
    fullyConnectedLayer( 21, 'Name', 'fc1' )
    dropoutLayer( 0.2, 'Name', 'drop1' )
    fullyConnectedLayer( paramCls.output, 'Name', 'fc2' )
    sigmoidLayer( 'Name', 'out' )
    ];

lgraphCls = layerGraph( layersCls );
dlnetCls = dlnetwork( lgraphCls );


end