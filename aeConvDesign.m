% ************************************************************************
% Function: aeConvDesign
%
% Initialise autoencoder networks with a traditional convolution design
%
% Parameters:
%           setup       : network design parameters
%           
% Outputs:
%           dlnetEnc    : initialised encoder network
%           dlnetDec    : initialised decoder network
%
% ************************************************************************


function [ dlnetEnc, dlnetDec ] = aeConvDesign( paramEnc, paramDec )


% define the encoder network
% --------------------------

projectionSize = [ paramEnc.projectionSize 1 1 ];

layersEnc = [ featureInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 ) 
              dropoutLayer( paramEnc.dropout, 'Name', 'drop0' )
              reshapeLayer( projectionSize, 'Name', 'proj' ) ];

filterSize = [ paramEnc.filterSize 1 ];
stride = [ paramEnc.stride 1 ];

for i = 1:paramEnc.nHidden
    nFilters = 2^(paramEnc.nHidden-i);
    layersEnc = [ layersEnc; ...
                    convolution2dLayer( filterSize, ...
                            nFilters*paramEnc.nFilters, ...
                            'Stride', stride, ...
                            'Name', ['conv' num2str(i)] )
        batchNormalizationLayer( 'Name', ['bnorm' num2str(i)] )
        leakyReluLayer( paramEnc.scale, ...
                        'Name', ['relu' num2str(i)] )
        ]; %#ok<*AGROW> 
end

layersEnc = [ layersEnc; ...
      fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' ) ];

lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
% --------------------------

projectionSize = [ paramDec.projectionSize 1 1 ];
layersDec = [ featureInputLayer( paramDec.input, 'Name', 'in' )
              projectAndReshapeLayer( projectionSize, ...
                            paramDec.input, 'Name', 'proj' ) ];

filterSize = [ paramDec.filterSize 1 ];
stride = [ paramDec.stride 1 ];

for i = 1:paramDec.nHidden
    nFilters = 2^(i-1);
    layersDec = [ layersDec; ...
                transposedConv2dLayer( filterSize, ...
                nFilters*paramDec.nFilters, ...
                'Stride', stride, ...
                'Name', ['tconv' num2str(i)] )
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