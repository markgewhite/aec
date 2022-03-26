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

% define input layers
layersEnc = [ sequenceInputLayer( paramEnc.input, 'Name', 'in', ...
                       'Normalization', 'zscore', ...
                       'Mean', 0, 'StandardDeviation', 1 )
              dropoutLayer( paramEnc.initialDropout, ...
                            'Name', 'drop0' ) ];
lgraphEnc = layerGraph( layersEnc );
lastLayer = 'drop0';

% create hidden layers
for i = 1:paramEnc.nHidden
    dilations = [ 2^(i*2-2) 2^(i*2-1) ];
    [lgraphEnc, lastLayer] = addResidualBlock( lgraphEnc, i, ...
                                        dilations, lastLayer, paramEnc );
end

% add the output layers
switch paramEnc.pooling
    case 'GlobalMax'
        outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
        poolingLayer = 'maxPool';
    case 'GlobalAvg'
        outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
        poolingLayer = 'avgPool';
    case 'None'
        outLayers = [];
        poolingLayer = 'out';
    otherwise
        error( 'Unrecognised pooling type for the decoder.' );
end

outLayers = [ outLayers;
              fullyConnectedLayer( paramEnc.outZ, 'Name', 'out' ) ];

lgraphEnc = addLayers( lgraphEnc, outLayers );
lgraphEnc = connectLayers( lgraphEnc, ...
                           lastLayer, poolingLayer );

dlnetEnc = dlnetwork( lgraphEnc );



% define the decoder network
% --------------------------

layersDec = [ featureInputLayer( paramDec.input, 'Name', 'in' )
              dropoutLayer( paramDec.initialDropout, ...
                            'Name', 'drop0' )
              projectAndReshapeLayer( paramDec.projectionSize, ...
                            paramDec.input, 'Name', 'proj' ) ];

lgraphDec = layerGraph( layersDec );
lastLayer = 'proj';

for i = 1:paramDec.nHidden
    dilations = [ 2^(2*(paramDec.nHidden-i)+1) ...
                    2^(2*(paramDec.nHidden-i)) ];
    [lgraphDec, lastLayer] = addResidualBlock( lgraphDec, i, ...
                                        dilations, lastLayer, paramDec );
end

% add the output layers
switch paramDec.pooling
    case 'GlobalMax'
        outLayers = globalMaxPooling1dLayer( 'Name', 'maxPool' );
        poolingLayer = 'maxPool';
    case 'GlobalAvg'
        outLayers = globalAveragePooling1dLayer( 'Name', 'avgPool' );
        poolingLayer = 'avgPool';
    case 'None'
        outLayers = [];
        poolingLayer = 'fcout';
    otherwise
        error( 'Unrecognised pooling type for the decoder.' );
end

outLayers = [ outLayers; 
              fullyConnectedLayer( prod(paramDec.outX), 'Name', 'fcout' ) ];

if paramDec.outX(2) > 1
    outLayers = [ outLayers; 
              reshapeLayer( paramDec.outX, 'Name', 'reshape' ) ];
end

outLayers = [ outLayers;
              functionLayer( @(X) smoothOutput( X, 'Gaussian' ), ...
                             'Formattable', false, ...
                             'Description', 'Smoothing', ...
                             'Name', 'out' ) ];

lgraphDec = addLayers( lgraphDec, outLayers );
lgraphDec = connectLayers( lgraphDec, ...
                           lastLayer, poolingLayer );

dlnetDec = dlnetwork( lgraphDec );


end



function [ lgraph, lastLayer ] = addResidualBlock( ...
                                  lgraph, i, dilations, lastLayer, params )

    i1 = i*2-1;
    i2 = i1+1;

    % define residual block
    block = [   convolution1dLayer( params.filterSize, ...
                                    params.nFilters, ...
                                    'DilationFactor', dilations(1), ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i1)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i1)] )
                leakyReluLayer( params.scale, ...
                                'Name', ['relu' num2str(i1)] )
                spatialDropoutLayer( params.dropout, ...
                                     'Name', ['drop' num2str(i1)] )
                convolution1dLayer( params.filterSize, ...
                                    params.nFilters, ...
                                    'DilationFactor', dilations(2), ...
                                    'Padding', 'causal', ...
                                    'Name', ['conv' num2str(i2)] )
                batchNormalizationLayer( 'Name', ['bnorm' num2str(i2)] )
                ];

    if params.useSkips
        block = [ block; 
                  additionLayer( 2, 'Name', ['add' num2str(i)] ) ];
    end

    block = [ block;
                leakyReluLayer( params.scale, ...
                                'Name', ['relu' num2str(i2)] )
                spatialDropoutLayer( params.dropout, ...
                                     'Name', ['drop' num2str(i2)] )
                ];

    % connect layers at the front
    lgraph = addLayers( lgraph, block );
    lgraph = connectLayers( lgraph, ...
                            lastLayer, ['conv' num2str(i1)] );
    
    if params.useSkips
        % include a short circuit ('skip')

        if i == 1
            % include convolution in first skip connection
            skipLayer = convolution1dLayer( 1, params.nFilters, ...
                                            'Name', 'convSkip' );
            lgraph = addLayers( lgraph, skipLayer );
            lgraph = connectLayers( lgraph, ...
                                       lastLayer, 'convSkip' );
            lgraph = connectLayers( lgraph, ...
                               'convSkip', ['add' num2str(i) '/in2'] );
        else
            % connect the skip
            lgraph = connectLayers( lgraph, ...
                               lastLayer, ['add' num2str(i) '/in2'] );
        end

    end

    lastLayer = ['drop' num2str(i2)];

end