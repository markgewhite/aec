% ************************************************************************
% Function: aeTCNDesign
%
% Initialise autoencoder networks with a Temporal Convolution design
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%
% ************************************************************************


function [ dlnetEnc, dlnetDec ] = aeTCNDesign( paramEnc, paramDec )


% define the encoder network
% --------------------------

layersEnc = [ featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 )
              dropoutLayer( paramEnc.initialDropout, ...
                            'Name', 'drop0' )
              reshapeLayer( paramEnc.projectionSize, ...
                            'Name', 'proj' ) ];

for i = 1:paramEnc.nHidden
    nDilation = 2^(i-1);
    layersEnc = [ layersEnc; ...
        convolution1dLayer( paramEnc.filterSize, ...
                            paramEnc.nFilters, ...
                            'DilationFactor', nDilation, ...
                            'Padding', 'causal', ...
                            'Name', ['conv' num2str(i)] )
        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
        leakyReluLayer( paramEnc.scale, ...
                        'Name', ['relu' num2str(i)] )
        spatialDropoutLayer( paramEnc.dropout, ...
                             'Name', ['drop' num2str(i)] )
        ]; %#ok<*AGROW> 
end

layersEnc = [ layersEnc; ...
      fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' ) ];

lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );



% define the decoder network
% --------------------------

layersDec = [ featureInputLayer( paramDec.input, 'Name', 'in' )
              dropoutLayer( paramDec.initialDropout, ...
                            'Name', 'drop0' )
              projectAndReshapeLayer( paramDec.projectionSize, ...
                            paramDec.input, 'Name', 'proj' ) ];
for i = 1:paramDec.nHidden
    nDilation = 2^(paramDec.nHidden-i);
    layersDec = [ layersDec; ...
        convolution1dLayer( paramDec.filterSize, ...
                            paramDec.nFilters, ...
                            'DilationFactor', nDilation, ...
                            'Padding', 'causal', ...
                            'Name', ['conv' num2str(i)] )
        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
        leakyReluLayer( paramDec.scale, ...
                        'Name', ['relu' num2str(i)] )
        spatialDropoutLayer( paramDec.dropout, ...
                             'Name', ['drop' num2str(i)] )
        ]; %#ok<*AGROW> 
end



layersDec = [ layersDec; ...
              fullyConnectedLayer( prod(paramDec.outX), 'Name', 'fcout' )
              reshapeLayer( paramDec.outX, 'Name', 'out' ) ];

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


end