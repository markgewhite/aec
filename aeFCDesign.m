% ************************************************************************
% Function: aeFCDesign
%
% Initialise autoencoder networks with a fully connected design
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%
% ************************************************************************


function [ dlnetEnc, dlnetDec ] = aeFCDesign( paramEnc, paramDec )


% define the encoder network
% --------------------------

layersEnc = [ featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 ) 
              dropoutLayer( paramEnc.dropout, 'Name', 'drop0' ) ];

for i = 1:paramEnc.nHidden
    nNodes = fix( paramEnc.nFC*2^(paramEnc.fcFactor*(1-i)) );
    layersEnc = [ layersEnc; ...
        fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
        leakyReluLayer( paramEnc.scale, ...
                        'Name', ['relu' num2str(i)] )
        dropoutLayer( paramEnc.dropout, 'Name', ...
                                         ['drop' num2str(i)] )
        ]; %#ok<*AGROW> 
end

layersEnc = [ layersEnc; ...
      fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' ) ];

lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
% --------------------------

layersDec = featureInputLayer( paramDec.input, 'Name', 'in' );

for i = 1:paramDec.nHidden
    nNodes = fix( paramDec.nFC*2^(paramDec.fcFactor*(-paramDec.nHidden+i)) );
    layersDec = [ layersDec; ...
        fullyConnectedLayer( nNodes, 'Name', ['fc' num2str(i)] )
        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
        leakyReluLayer( paramDec.scale, ...
                        'Name', ['relu' num2str(i)] )           
        ]; %#ok<*AGROW> 
end

layersDec = [ layersDec; ...
              fullyConnectedLayer( prod(paramDec.outX), 'Name', 'fcout' )
              reshapeLayer( paramDec.outX, 'Name', 'out' ) ];

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


end